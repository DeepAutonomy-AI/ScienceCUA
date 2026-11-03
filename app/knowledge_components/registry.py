"""Deterministic audit exports and native Agent entry registration artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..agent_export import _reference_content, export_agent_entries
from ..models import (
    CandidateValidation, CorpusSnapshot, EvidenceCitation, EvidenceExcerpt,
    KnowledgeSnapshot, ReferenceCandidate, SkillCandidate, TargetProfile,
    TrajectoryCandidate, utc_now,
)
from .common import _safe_filename, _safe_skill_markdown_stem
from ..reference_text import reference_text_payload, text_only_reference
from ..storage import RunStore


class SkillRegistry:
    """Deterministic exporter. Model agents never receive a write tool for this directory."""

    def __init__(
        self,
        output_dir: Path,
        *,
        skill_output_language: str = "English",
        artifact_store: RunStore | None = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.skill_output_language = skill_output_language
        self.artifact_store = artifact_store

    def commit(
        self,
        *,
        run_id: str,
        target_profile: TargetProfile,
        corpus: CorpusSnapshot,
        knowledge: KnowledgeSnapshot,
        trajectories: list[TrajectoryCandidate],
        skills: list[SkillCandidate],
        validations: list[CandidateValidation],
        quality_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        software = _safe_filename(target_profile.software_id or target_profile.software_name, "software")
        run_destination = self.output_dir / software / _safe_filename(run_id, "run")
        destination = run_destination / "audit"
        destination.mkdir(parents=True, exist_ok=True)
        skills = [item.model_copy(update={"status": "committed"}) for item in skills]
        evidence_by_id = {item.evidence_id: item for item in corpus.evidence}
        references = [text_only_reference(ref) for ref in knowledge.references]
        payload = {
            "schema_version": "v8",
            "run_id": run_id,
            "created_at": utc_now(),
            "target_profile": target_profile.model_dump(mode="json"),
            "corpus_snapshot_id": corpus.corpus_snapshot_id,
            "knowledge_snapshot_id": knowledge.knowledge_snapshot_id,
            "skill_output_language": self.skill_output_language,
            "tasks": [item.model_dump(mode="json") for item in knowledge.tasks],
            "references": [reference_text_payload(item.model_dump(mode="json")) for item in references],
            "trajectories": [item.model_dump(mode="json") for item in trajectories],
            "skills": [self._skill_payload(item) for item in skills],
            "validation": [item.model_dump(mode="json") for item in validations],
            "evidence_manifest": [
                {
                    "evidence_id": item.evidence_id,
                    "source_id": item.source_id,
                    "source_part_id": item.source_part_id,
                    "title": item.title,
                    "locator": item.locator,
                    "resolved_version": item.resolved_version,
                    "source_origin": item.locator.get("source_origin", "user"),
                    "source_priority": item.locator.get("source_priority"),
                    "target_version": item.locator.get("target_version"),
                    "source_resolved_version": item.locator.get(
                        "source_resolved_version", item.resolved_version
                    ),
                    "version_relation": item.locator.get("version_relation"),
                    "evidence_applicability": item.locator.get("evidence_applicability"),
                    "content_sha256": item.content_sha256,
                }
                for item in corpus.evidence
            ],
        }
        self._write_json(destination / "skills.json", payload)
        self._write_json(destination / "tasks.json", payload["tasks"])
        self._write_json(destination / "references.json", payload["references"])
        self._write_json(destination / "trajectories.json", payload["trajectories"])
        self._write_json(destination / "validation.json", payload["validation"])
        if quality_report is not None:
            self._write_json(destination / "quality_report.json", quality_report)
        for skill in skills:
            self._write_skill_markdown(
                destination / "skills" / f"{skill.skill_id}.md", skill,
                skill_output_language=self.skill_output_language, evidence_by_id=evidence_by_id,
            )
        agent_export = export_agent_entries(
            run_destination, run_id=run_id, target_profile=target_profile,
            skills=skills, references=references,
            skill_output_language=self.skill_output_language, evidence_by_id=evidence_by_id,
        )
        self._write_readme(destination, payload, skill_output_language=self.skill_output_language)
        return {
            "output_directory": str(run_destination),
            "audit_directory": str(destination),
            "skills_file": str(destination / "skills.json"),
            **agent_export,
            # Compatibility path aliases; both kinds now share the entries root.
            "agent_skill_directory": agent_export["agent_entry_directory"],
            "agent_reference_directory": agent_export["agent_entry_directory"],
            "agent_skill_index_file": agent_export["agent_entry_index_file"],
            "agent_reference_index_file": agent_export["agent_entry_index_file"],
            "skill_output_language": self.skill_output_language,
            "skill_count": len(skills),
            "task_count": len(knowledge.tasks),
            "reference_count": len(knowledge.references),
            "trajectory_count": len(trajectories),
            "quality_report_file": str(destination / "quality_report.json")
            if quality_report is not None
            else None,
        }

    @staticmethod
    def _skill_payload(skill: SkillCandidate) -> dict[str, Any]:
        return skill.model_dump(mode="json")

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _citation_label(
        citation: EvidenceCitation,
        evidence_by_id: dict[str, EvidenceExcerpt],
    ) -> str:
        return SkillRegistry._evidence_id_label(citation.evidence_id, evidence_by_id)

    @staticmethod
    def _evidence_id_label(
        evidence_id: str,
        evidence_by_id: dict[str, EvidenceExcerpt],
    ) -> str:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            return f"`{evidence_id}`"
        provenance = SkillRegistry._evidence_provenance_label(evidence)
        return f"`{evidence_id}` ({provenance})" if provenance else f"`{evidence_id}`"

    @staticmethod
    def _evidence_provenance_label(evidence: EvidenceExcerpt) -> str:
        locator = evidence.locator or {}
        pieces = [
            f"source={locator.get('source_origin', 'user')}",
            f"version={locator.get('source_resolved_version', evidence.resolved_version)}",
        ]
        relation = locator.get("version_relation")
        applicability = locator.get("evidence_applicability")
        if relation:
            pieces.append(f"relation={relation}")
        if applicability:
            pieces.append(f"applicability={applicability}")
        priority = locator.get("source_priority")
        if priority:
            pieces.append(f"priority={priority}")
        return "; ".join(str(item) for item in pieces if item)

    @staticmethod
    def _write_skill_markdown(
        path: Path,
        skill: SkillCandidate,
        *,
        skill_output_language: str = "English",
        evidence_by_id: dict[str, EvidenceExcerpt] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = _skill_markdown_labels(skill_output_language)
        evidence_by_id = evidence_by_id or {}
        lines = [
            f"# {skill.title}",
            "",
            f"- {labels['skill_id']}: `{skill.skill_id}`",
            f"- {labels['status']}: `{skill.status}`",
            f"- {labels['trajectories']}: {', '.join(f'`{item}`' for item in skill.trajectory_ids)}",
            "",
            f"## {labels['description']}",
            "",
            skill.description,
            "",
            f"## {labels['applicability']}",
            "",
            skill.applicability,
            "",
        ]
        lines.extend([f"## {labels['preconditions']}", ""])
        if skill.preconditions:
            lines.append(skill.preconditions)
        else:
            lines.append(f"- {labels['no_preconditions']}")
        lines.extend(["", f"## {labels['procedure']}", ""])
        for step in skill.procedure:
            citations = ", ".join(
                SkillRegistry._citation_label(item, evidence_by_id)
                for item in step.citations
            )
            lines.extend(
                [
                    f"{step.ordinal}. {step.instruction}",
                    f"   {labels['evidence']}: {citations}",
                ]
            )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _write_agent_skill_markdown(
        path: Path,
        skill: SkillCandidate,
        *,
        skill_output_language: str = "English",
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = _skill_markdown_labels(skill_output_language)
        lines = [
            f"# {skill.title}",
            "",
            f"description: {SkillRegistry._agent_description(skill.description)}",
            "",
            f"## {labels['applicability']}",
            "",
            skill.applicability,
            "",
            f"## {labels['preconditions']}",
            "",
        ]
        if skill.preconditions:
            lines.append(skill.preconditions)
        else:
            lines.append(f"- {labels['no_preconditions']}")

        lines.extend(["", f"## {labels['procedure']}", ""])
        for step in skill.procedure:
            lines.extend(
                [
                    f"{step.ordinal}. {step.instruction}",
                ]
            )


        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _agent_skill_markdown_entries(skills: list[SkillCandidate]) -> list[tuple[SkillCandidate, str]]:
        used = {"skill.md"}
        entries: list[tuple[SkillCandidate, str]] = []
        for skill in skills:
            stem = _safe_skill_markdown_stem(skill.title, skill.skill_id)
            filename = f"{stem}.md"
            if filename.casefold() in used:
                suffix = f"_{_safe_skill_markdown_stem(skill.skill_id, 'skill')}"
                stem = stem[: max(1, 150 - len(suffix))].rstrip(" .") or "skill"
                filename = f"{stem}{suffix}.md"
            ordinal = 2
            while filename.casefold() in used:
                suffix = f"_{_safe_skill_markdown_stem(skill.skill_id, 'skill')}_{ordinal}"
                stem = _safe_skill_markdown_stem(skill.title, "skill")
                stem = stem[: max(1, 150 - len(suffix))].rstrip(" .") or "skill"
                filename = f"{stem}{suffix}.md"
                ordinal += 1
            used.add(filename.casefold())
            entries.append((skill, filename))
        return entries

    @staticmethod
    def _agent_reference_markdown_filenames(
        references: list[ReferenceCandidate],
    ) -> dict[str, str]:
        used: set[str] = set()
        filenames: dict[str, str] = {}
        for reference in references:
            stem = _safe_skill_markdown_stem(reference.reference_id, "reference")
            filename = f"{stem}.md"
            ordinal = 2
            while filename.casefold() in used:
                filename = f"{stem}_{ordinal}.md"
                ordinal += 1
            used.add(filename.casefold())
            filenames[reference.reference_id] = filename
        return filenames

    @staticmethod
    def _write_agent_reference_markdown(
        path: Path,
        reference: ReferenceCandidate,
        *,
        evidence_by_id: dict[str, EvidenceExcerpt] | None = None,
        image_reader: Callable[[str], bytes] | None = None,
    ) -> None:
        reference = text_only_reference(reference)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(_reference_content(reference).encode("utf-8"))
        temporary.replace(path)

    @staticmethod
    def _write_agent_skill_index(
        path: Path,
        target_profile: TargetProfile,
        entries: list[tuple[SkillCandidate, str]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {target_profile.software_name} Skills", ""]
        if not entries:
            lines.append("No skills were generated.")
        else:
            for skill, _filename in entries:
                lines.extend(
                    [
                        f"## {skill.title}",
                        "",
                        f"description: {SkillRegistry._agent_description(skill.description)}",
                        "",
                    ]
                )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _write_agent_reference_index(
        path: Path,
        target_profile: TargetProfile,
        references: list[ReferenceCandidate],
    ) -> None:
        """Write the deterministic, human-readable index for exported References."""

        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {target_profile.software_name} References", ""]
        if not references:
            lines.append("No references were generated.")
        else:
            for reference in references:
                lines.extend([f"## {reference.title}", ""])
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _agent_description(description: str) -> str:
        text = " ".join(description.split())
        if not text:
            return ""
        sentences = [
            match.group(0).strip()
            for match in re.finditer(r".+?(?:[。！？.!?]+(?:\s+|$)|$)", text)
            if match.group(0).strip()
        ]
        summary = " ".join(sentences[:2]).strip() or text
        if len(summary) <= 360:
            return summary
        return summary[:357].rstrip() + "..."

    @staticmethod
    def _write_readme(
        destination: Path,
        payload: dict[str, Any],
        *,
        skill_output_language: str = "English",
    ) -> None:
        labels = _skill_markdown_labels(skill_output_language)
        if labels["is_chinese"]:
            lines = [
                "# 生成的技能",
                "",
                "这些技能是第一阶段根据资料推断出的产物。每个事实性操作均须引用输入资料中的 Evidence；",
                "技能使用自然语言表达可复用操作，不暴露模板参数，也不把某个脚本、文件、对象或示例值当成技能范围。",
                "具体调用示例只作为示例绑定；这些技能尚未在目标软件中执行或验证。",
                "",
                f"- 运行：`{payload['run_id']}`",
                f"- 语料快照：`{payload['corpus_snapshot_id']}`",
                f"- 知识快照：`{payload['knowledge_snapshot_id']}`",
                f"- 输出语言：`{skill_output_language}`",
                f"- 技能数量：`{len(payload['skills'])}`",
                "",
                "完整机器可读结果见 `skills.json`；单个技能 Markdown 位于 `skills/`；",
                "仅依赖本次资料的溯源、关联和覆盖诊断见 `quality_report.json`。",
            ]
        else:
            lines = [
                "# Generated Skills",
                "",
                "These Skills are first-stage inferred artifacts. Every factual procedure must cite Evidence",
                "from the input material. Skills use natural language to describe generalized operations and do",
                "not expose template parameters. A Skill must not be scoped to one script, file, object, or example",
                "value unless the source-backed capability is truly that exact item. None of these Skills has been",
                "executed or verified in the target software.",
                "",
                f"- Run: `{payload['run_id']}`",
                f"- Corpus snapshot: `{payload['corpus_snapshot_id']}`",
                f"- Knowledge snapshot: `{payload['knowledge_snapshot_id']}`",
                f"- Output language: `{skill_output_language}`",
                f"- Skills: `{len(payload['skills'])}`",
                "",
                "Use `skills.json` for the complete machine-readable output, `skills/` for per-Skill Markdown,",
                "and `quality_report.json` for source-only provenance, linkage, and coverage diagnostics.",
            ]
        text = "\n".join(lines)
        path = destination / "README.md"
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(path)


def _skill_markdown_labels(skill_output_language: str) -> dict[str, str | bool]:
    normalized = " ".join(str(skill_output_language or "").strip().lower().split())
    is_chinese = normalized in {
        "zh",
        "zh-cn",
        "zh_hans",
        "chinese",
        "simplified chinese",
        "chinese (simplified)",
    }
    if is_chinese:
        return {
            "is_chinese": True,
            "skill_id": "技能 ID",
            "status": "状态",
            "tasks": "关联任务",
            "trajectories": "关联轨迹",
            "description": "说明",
            "applicability": "适用范围",
            "preconditions": "前置条件",
            "no_preconditions": "该技能未要求任何由资料支持的前置条件。",
            "procedure": "操作步骤",
            "evidence": "证据",
            "caveats": "注意事项",
        }
    return {
        "is_chinese": False,
        "skill_id": "Skill ID",
        "status": "Status",
        "trajectories": "Trajectories",
        "references": "References",
        "no_references": "none",
        "description": "Description",
        "applicability": "Applicability",
        "preconditions": "Preconditions",
        "no_preconditions": "No source-backed precondition was required by this Skill.",
        "procedure": "Procedure",
        "evidence": "Evidence",
        "caveats": "Caveats",
    }
