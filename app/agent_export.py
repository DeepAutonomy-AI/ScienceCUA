"""Export procedural Skills and lookup References as peer Codex skill entries.

This module only writes portable artifacts. Discovery/installation is left to the
consumer; neither the human index nor the manifest is an injection mechanism.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.models import EvidenceExcerpt, ReferenceCandidate, SkillCandidate, TargetProfile
from app.reference_text import reference_presentation_content, reference_text, text_only_reference


AGENT_ENTRY_SCHEMA_VERSION = "codex-agent-entries.v1"
MAX_METADATA_DESCRIPTION_LENGTH = 1_024


def export_agent_entries(
    destination: Path,
    *,
    run_id: str,
    target_profile: TargetProfile,
    skills: Sequence[SkillCandidate],
    references: Sequence[ReferenceCandidate],
    skill_output_language: str = "English",
    evidence_by_id: Mapping[str, EvidenceExcerpt] | None = None,
    skill_body_renderer: Callable[[SkillCandidate], str] | None = None,
    reference_body_renderer: Callable[[ReferenceCandidate], str] | None = None,
    image_reader: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Write one independently discoverable directory per accepted artifact.

    ``destination`` is the software/run root, not the ``entries`` directory.
    Descriptions must already have been authored upstream; this exporter never
    fills a missing Reference description from its title or source content.
    All input validation and rendering precede writes; individual files are
    replaced atomically as UTF-8 without changing source-content line endings.
    Optional renderers may supply an existing complete Markdown body.
    """
    destination = Path(destination).resolve()
    evidence_by_id = evidence_by_id or {}
    references = [text_only_reference(ref) for ref in references]
    items = [("skill", item) for item in skills] + [
        ("reference", item) for item in references
    ]
    descriptions: list[str] = []
    for kind, item in items:
        description = getattr(item, "description", None)
        if not isinstance(description, str) or not description.strip():
            source_id = getattr(item, f"{kind}_id", "unknown")
            raise ValueError(f"{kind} {source_id} requires an authored description for export")
        descriptions.append(_metadata_description(kind, item, description.strip()))

    names = _entry_names(target_profile, items)
    records: list[dict[str, Any]] = []
    files: list[tuple[Path, str]] = []
    for (kind, item), name, description in zip(items, names, descriptions, strict=True):
        entry_relative = f"entries/{name}/SKILL.md"
        content_relative = f"entries/{name}/content.md" if kind == "reference" else None
        record = {
            "kind": kind,
            "source_id": getattr(item, f"{kind}_id"),
            "title": item.title,
            "name": name,
            "description": description,
            "entry_path": entry_relative,
            "content_path": content_relative,
        }
        records.append(record)
        if kind == "skill":
            body = (
                skill_body_renderer(item)
                if skill_body_renderer is not None
                else _skill_body(item, skill_output_language)
            )
        else:
            body = _reference_entry_body(item, description, skill_output_language)
            content = (_reference_content(item) if reference_presentation_content(item) else
                       reference_text(reference_body_renderer(item)) if reference_body_renderer is not None
                       else _reference_content(item))
            files.append((destination / str(content_relative), content))
        files.append((destination / entry_relative, _frontmatter(name, description) + body))

    manifest = {
        "schema_version": AGENT_ENTRY_SCHEMA_VERSION,
        "run_id": run_id,
        "software_name": target_profile.software_name,
        "software_id": target_profile.software_id,
        "target_version": target_profile.target_version,
        "skill_output_language": skill_output_language,
        "entry_count": len(records),
        "skill_count": len(skills),
        "reference_count": len(references),
        "entries": records,
    }
    index_path = destination / "INDEX.md"
    manifest_path = destination / "manifest.json"
    files.extend(
        [
            (index_path, _index(target_profile, records, skill_output_language)),
            (manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"),
        ]
    )
    _check_existing_entries(destination, records)
    (destination / "entries").mkdir(parents=True, exist_ok=True)
    for path, content in files:
        _write_text_atomic(path, content)

    returned_records = [
        {
            **record,
            "entry_file": str(destination / record["entry_path"]),
            "content_file": str(destination / record["content_path"])
            if record["content_path"]
            else None,
        }
        for record in records
    ]
    return {
        "agent_entry_directory": str(destination / "entries"),
        "agent_entry_index_file": str(index_path),
        "agent_entry_manifest_file": str(manifest_path),
        "agent_entry_count": len(records),
        "agent_entries": returned_records,
        "agent_skill_entries": [item for item in returned_records if item["kind"] == "skill"],
        "agent_reference_entries": [
            item for item in returned_records if item["kind"] == "reference"
        ],
        "agent_skill_count": len(skills),
        "agent_reference_count": len(references),
    }


def _ascii_slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_value.casefold()).strip("-")


def _metadata_description(kind: str, item: Any, description: str) -> str:
    if len(description) <= MAX_METADATA_DESCRIPTION_LENGTH:
        return description
    if kind == "reference":
        raise ValueError(
            f"reference {item.reference_id} description exceeds the native "
            f"{MAX_METADATA_DESCRIPTION_LENGTH}-character metadata limit"
        )
    text = " ".join(description.split())
    if len(text) <= MAX_METADATA_DESCRIPTION_LENGTH:
        return text
    prefix = text[: MAX_METADATA_DESCRIPTION_LENGTH - 3]
    sentence_boundaries = list(re.finditer(r"[。！？]|[.!?](?:\s|$)", prefix))
    if sentence_boundaries:
        prefix = prefix[: sentence_boundaries[-1].end()].rstrip()
    elif " " in prefix:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix.rstrip() + "..."


def _check_existing_entries(destination: Path, records: Sequence[dict[str, Any]]) -> None:
    """Refuse stale discoverable directories rather than deleting old artifacts."""
    expected_names = {record["name"] for record in records}
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_entries = previous["entries"]
            if previous.get("schema_version") != AGENT_ENTRY_SCHEMA_VERSION:
                raise ValueError("unsupported existing manifest schema")
            previous_names = {entry["name"] for entry in previous_entries}
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Cannot verify existing agent-entry manifest: {manifest_path}") from error
        if previous_names != expected_names:
            raise ValueError(
                "Existing agent-entry manifest describes a different set of entries; "
                "export to a fresh run directory to avoid stale discoverable entries"
            )
    entry_directory = destination / "entries"
    if not entry_directory.exists():
        return
    stale = [
        str(path)
        for path in entry_directory.iterdir()
        if path.is_dir() and path.name not in expected_names and (path / "SKILL.md").is_file()
    ]
    if stale:
        raise ValueError(f"Existing discoverable entries are outside this export: {', '.join(sorted(stale))}")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _semantic_fingerprint(kind: str, item: Any) -> str:
    # Do not use run IDs, candidate IDs, citation IDs, or timestamps in names.
    payload = {
        "kind": kind,
        "title": item.title,
        "description": getattr(item, "description", ""),
        "content": getattr(item, "content", ""),
        "applicability": getattr(item, "applicability", ""),
        "preconditions": getattr(item, "preconditions", ""),
        "procedure": [
            [step.ordinal, step.instruction]
            for step in getattr(item, "procedure", [])
        ],
        "entries": [[entry.title, entry.content] for entry in getattr(item, "entries", [])],
    }
    return _digest(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _entry_names(
    target_profile: TargetProfile, items: Sequence[tuple[str, Any]]
) -> list[str]:
    software_text = target_profile.software_id or target_profile.software_name
    software = _ascii_slug(software_text) or f"software-{_digest(software_text)}"
    software = software[:22].rstrip("-")
    bases: list[str] = []
    for kind, item in items:
        title = _ascii_slug(item.title) or f"entry-{_digest(item.title)}"
        prefix = f"{software}-{kind}-"
        bases.append(prefix + title[: 64 - len(prefix)].rstrip("-"))
    counts = Counter(bases)
    names: list[str] = []
    used: set[str] = set()
    for (kind, item), base in zip(items, bases, strict=True):
        name = base
        if counts[base] > 1:
            suffix = f"-{_semantic_fingerprint(kind, item)}"
            name = base[: 64 - len(suffix)].rstrip("-") + suffix
        ordinal = 2
        original = name
        while name in used:
            suffix = f"-{ordinal}"
            name = original[: 64 - len(suffix)].rstrip("-") + suffix
            ordinal += 1
        used.add(name)
        names.append(name)
    return names


def _frontmatter(name: str, description: str) -> str:
    # JSON-quoted strings are YAML scalars, including quotes, colons, Unicode,
    # backslashes, newlines, and strings that YAML otherwise treats as booleans.
    return (
        "---\n"
        f"name: {json.dumps(name, ensure_ascii=False)}\n"
        f"description: {json.dumps(description, ensure_ascii=False)}\n"
        "---\n\n"
    )


def _is_chinese_output(language: str) -> bool:
    return " ".join(str(language or "").strip().casefold().split()) in {
        "zh", "zh-cn", "zh_hans", "chinese", "simplified chinese", "chinese (simplified)"
    }


def _skill_body(skill: SkillCandidate, language: str) -> str:
    is_chinese = _is_chinese_output(language)
    labels = (
        ("说明", "适用范围", "前置条件", "操作步骤", "注意事项",
         "该技能未要求任何由资料支持的前置条件。")
        if is_chinese
        else ("Description", "Applicability", "Preconditions", "Procedure",
              "Caveats", "No source-backed precondition was required by this Skill.")
    )
    description, applicability, preconditions, procedure, caveats_label, none = labels
    lines = [
        f"# {skill.title}", "", f"## {description}", "", skill.description, "",
        f"## {applicability}", "", skill.applicability, "", f"## {preconditions}", "",
        skill.preconditions if skill.preconditions else f"- {none}", "", f"## {procedure}", "",
    ]
    for step in skill.procedure:
        lines.append(f"{step.ordinal}. {step.instruction}")
    # Some saved/custom candidates still carry source caveats. Current cleaned
    # Skill models may omit the field entirely; preserve it whenever supplied.
    caveats = getattr(skill, "caveats", [])
    if caveats:
        lines.extend(["", f"## {caveats_label}", "", *(f"- {item}" for item in caveats)])
    return "\n".join(lines) + "\n"


def _reference_entry_body(reference: ReferenceCandidate, description: str, language: str) -> str:
    if _is_chinese_output(language):
        lines = [
            f"# {reference.title}", "", "## 知识范围", "", description, "",
            "## 按需查阅", "",
            "在任何任务中需要本范围内的信息时，随时查阅本参考。",
            "按名称或关键词搜索[完整参考内容](content.md)，然后阅读相关条目及其来源上下文。",
            "仅在查询需要时阅读更多章节。本参考可独立于任何流程性技能使用。",
            "应用条目时请保持文档原意，并遵循正文中的适用条件和限制。",
        ]
        return "\n".join(lines) + "\n"
    lines = [
        f"# {reference.title}", "", "## Knowledge scope", "", description, "",
        "## On-demand lookup", "",
        "Consult this Reference whenever information in its scope is needed during any task.",
        "Search [the complete reference](content.md) by name or keyword, then read the relevant entries and their source context.",
        "Read further sections only when the query needs them. This Reference can be used independently of any procedural Skill.",
        "Preserve the documented meaning and follow the applicability conditions and restrictions in the text.",
    ]
    return "\n".join(lines) + "\n"


def _reference_content(reference: ReferenceCandidate) -> str:
    """Render text-only knowledge; provenance and citation quotes stay in audit JSON."""
    presentation = reference_presentation_content(reference)
    if presentation:
        return f"# {reference.title}\n\n{reference.description}\n\n{presentation.rstrip()}\n"
    reference = text_only_reference(reference)
    lines = [f"# {reference.title}", "", reference.description]
    if reference.content.strip():
        lines.extend(["", reference.content])
    for entry in reference.entries:
        lines.extend(["", f"## {entry.display_title or entry.title}", "", entry.content])
    return "\n".join(lines) + "\n"


def _table_text(value: str) -> str:
    return " ".join(value.split()).replace("\\", "\\\\").replace("|", "\\|").replace("[", "\\[").replace("]", "\\]").replace("`", "\\`")


def _index(target_profile: TargetProfile, records: Sequence[dict[str, Any]], language: str = "English") -> str:
    if _is_chinese_output(language):
        lines = [
            f"# {target_profile.software_name} 代理条目", "",
            f"软件版本：{target_profile.target_version}", "",
            "每个链接目录都是独立条目。技能（Skill）条目提供任务流程；参考（Reference）条目提供全局按需知识查阅。",
            "", "| 类型 | 标题 | 名称 | 描述 |", "| --- | --- | --- | --- |",
        ]
    else:
        lines = [
            f"# {target_profile.software_name} Agent Entries", "",
            f"Software version: {target_profile.target_version}", "",
            "Each linked directory is an independent entry. Skill entries provide task procedures; Reference entries provide global on-demand knowledge lookup.",
            "", "| Kind | Title | Name | Description |", "| --- | --- | --- | --- |",
        ]
    for record in records:
        lines.append(
            f"| {record['kind']} | [{_table_text(record['title'])}]({record['entry_path']}) | "
            f"`{record['name']}` | {_table_text(record['description'])} |"
        )
    if not records:
        lines.extend(["", "未生成任何条目。" if _is_chinese_output(language) else "No entries were generated."])
    return "\n".join(lines) + "\n"


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
