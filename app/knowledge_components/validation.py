"""Deterministic evidence, Task, Reference, Trajectory, and Skill validation."""

from __future__ import annotations

import re
from typing import Callable

from ..models import (
    CandidateValidation, EvidenceCitation, EvidenceExcerpt, EvidenceTree,
    ReferenceCandidate, SkillCandidate, SkillDraft, TaskCandidate, TrajectoryCandidate,
)
from .common import T, _compact, catalog_description_errors
from .reference_catalogs import _NUMBERED_REFERENCE_ENTRY, _reference_field_labels
from ..reference_images import hydrate_reference_images


_SKILL_SECTION_TITLE = re.compile(
    r"^(?:(?:chapter|section|appendix)\b|第\s*[一二三四五六七八九十百千0-9]+\s*[章节篇]|\d+(?:\.\d+)+(?:\s|$))",
    flags=re.IGNORECASE,
)
_SKILL_DOCUMENT_TITLE = re.compile(
    r"\b(?:manual|handbook|documentation|introduction|acknowledg(?:e)?ments?|license|contents)\b|手册|翻译|简介|致谢|许可|目录",
    flags=re.IGNORECASE,
)
_GENERIC_SKILL_PHRASES = (
    "use the documented operation or concept in the cited source material",
    "use when the cited source material applies",
    "follow the documented operation identified by the cited evidence",
    "the documented operation is available in the target software",
    "cited source material applies to the target software",
)

_TEMPLATE_PLACEHOLDER = re.compile(r"\{\{[^{}]{0,160}\}\}")
_LOOKUP_ACTION_MARKERS = (
    "look up",
    "lookup",
    "find",
    "search",
    "查找",
    "查阅",
    "查询",
    "搜索",
    "检索",
)
_LOOKUP_RESULT_MARKERS = (
    "definition",
    "parameter",
    "entry",
    "reference",
    "catalog",
    "manual",
    "documentation",
    "module",
    "section",
    "function",
    "operation",
    "command",
    "定义",
    "参数",
    "条目",
    "函数",
    "操作",
    "命令",
)
_REFERENCE_LOOKUP_ACTION_MARKERS = (
    "look up",
    "lookup",
    "find",
    "search",
    "read",
    "consult",
    "query",
    "locate",
    "查找",
    "查阅",
    "查询",
    "搜索",
    "检索",
    "阅读",
    "查看",
    "定位",
)
_REFERENCE_LOOKUP_OBJECT_MARKERS = (
    "linked reference",
    "reference catalog",
    "reference entry",
    "command catalog",
    "function catalog",
    "syntax catalog",
    "command entry",
    "function entry",
    "module entry",
    "parameter entry",
    "command table",
    "function table",
    "documentation",
    "manual",
    "关联 reference",
    "reference 目录",
    "reference 条目",
    "参考资料",
    "参考目录",
    "参考条目",
    "命令目录",
    "命令表",
    "命令条目",
    "函数目录",
    "函数表",
    "函数条目",
    "模块条目",
    "参数条目",
    "语法目录",
    "目录条目",
    "手册",
    "文档",
)
_OPEN_INPUT_ACTION_MARKERS = (
    "enter",
    "type",
    "input",
    "write",
    "键入",
    "输入",
    "录入",
    "编写",
)
_OPEN_KNOWLEDGE_OBJECT_MARKERS = (
    "expression",
    "function",
    "command",
    "formula",
    "syntax",
    "operator",
    "operation",
    "api",
    "schema",
    "parameter",
    "argument",
    "option",
    "file format",
    "表达式",
    "函数",
    "命令",
    "公式",
    "语法",
    "算子",
    "运算",
    "接口",
    "参数",
    "选项",
    "字段",
    "文件格式",
)
_OPEN_REFERENCE_USAGE_TAGS = {
    "api_reference",
    "command_catalog",
    "command_reference",
    "data_schema_reference",
    "expression_syntax",
    "file_format_reference",
    "function_catalog",
    "operator_catalog",
    "parameter_reference",
    "syntax_reference",
    "ui_reference",
}
_LOOKUP_REFERENCE_USAGE_TAGS = {
    "api_reference",
    "commands",
    "command_catalog",
    "command_reference",
    "functions",
    "function_catalog",
    "keyboard_shortcuts",
    "menu_commands",
    "operators",
    "operator_catalog",
    "parameters",
    "parameter_reference",
    "source_catalog",
    "syntax",
    "syntax_reference",
    "ui_reference",
    "valid_expressions",
}
_SOURCE_CATALOG_DEFAULT_USAGE_TAGS = ["source_catalog"]
_GENERIC_CAPABILITY_CATEGORIES = {"", "general", "operation", "task", "unknown", "其他", "通用"}


def _contains_generic_skill_language(value: str) -> bool:
    normalized = _compact(value).casefold()
    return any(fragment in normalized for fragment in _GENERIC_SKILL_PHRASES)


def _literal_occurs(values: list[str], literal: str) -> bool:
    normalized = _compact(literal)
    if len(normalized) < 2:
        return False
    text = "\n".join(values)
    if re.fullmatch(r"[A-Za-z0-9_.+/-]+", normalized):
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])",
                text,
                flags=re.IGNORECASE,
            )
        )
    return normalized.casefold() in text.casefold()


def _requires_lookup(values: list[str]) -> bool:
    normalized = _compact(" ".join(values)).casefold()
    return any(marker in normalized for marker in _LOOKUP_ACTION_MARKERS) and any(
        marker in normalized for marker in _LOOKUP_RESULT_MARKERS
    )


def _requires_reference_lookup(values: list[str]) -> bool:
    """Identify explicit Reference/catalog lookup, not target-software data queries."""

    def contains_marker(text: str, marker: str) -> bool:
        # English words must match as words: "read" inside "ready" is not a lookup.
        if marker.isascii():
            return bool(re.search(r"(?<![a-z0-9_])" + re.escape(marker) + r"(?![a-z0-9_])", text))
        return marker in text

    # A lookup verb in one observation must not combine with a catalog noun in
    # another, unrelated instruction to turn an input operation into a lookup.
    for value in values:
        for clause in re.split(r"[.!?;。！？；\n]+", _compact(value).casefold()):
            if any(contains_marker(clause, marker) for marker in _REFERENCE_LOOKUP_ACTION_MARKERS) and any(
                contains_marker(clause, marker) for marker in _REFERENCE_LOOKUP_OBJECT_MARKERS
            ):
                return True
    return False

def _contains_template_placeholder(values: list[str]) -> bool:
    return any("{{" in value or "}}" in value for value in values if value)


def _template_placeholder_errors(values: list[str], *, label: str) -> list[str]:
    if not _contains_template_placeholder(values):
        return []
    placeholders = sorted(
        {
            match.group(0)
            for value in values
            for match in _TEMPLATE_PLACEHOLDER.finditer(value or "")
        }
    )
    if placeholders:
        return [
            f"{label} uses template placeholder syntax "
            f"{', '.join(placeholders)}; write generalized natural-language scope without variables."
        ]
    return [
        f"{label} uses template placeholder delimiters; write generalized natural-language scope without variables."
    ]


def _catalog_entry_label(evidence: EvidenceExcerpt) -> str | None:
    match = _NUMBERED_REFERENCE_ENTRY.match(_compact(evidence.title))
    if not match or len(_reference_field_labels(evidence.content)) < 2:
        return None
    label = _compact(match.group("label"))
    if not label or len(label.split()) > 5:
        return None
    return label


def _introduced_catalog_scope_errors(
    child_citations: list[EvidenceCitation],
    parent_citations: list[EvidenceCitation],
    evidence_by_id: dict[str, EvidenceExcerpt],
    *,
    parent_text: list[str],
    child_text: list[str],
    child_label: str,
) -> list[str]:
    if not _requires_lookup(parent_text):
        return []
    parent_evidence_ids = {item.evidence_id for item in parent_citations}
    errors: list[str] = []
    for citation in child_citations:
        if citation.evidence_id in parent_evidence_ids:
            continue
        evidence = evidence_by_id.get(citation.evidence_id)
        label = _catalog_entry_label(evidence) if evidence else None
        if (
            label
            and not _literal_occurs(parent_text, label)
            and _literal_occurs(child_text, label)
        ):
            errors.append(
                f"{child_label} narrows the parent lookup scope to catalog example {label!r}; "
                "write the operation generically and use concrete entries only as supporting evidence"
            )
    return errors


def _skill_quality_errors(candidate: SkillCandidate) -> list[str]:
    """Reject document headings and boilerplate before a candidate becomes a Skill."""
    errors: list[str] = []
    title = _compact(candidate.title)
    if _SKILL_SECTION_TITLE.search(title) or _SKILL_DOCUMENT_TITLE.search(title):
        errors.append("Skill title is a document heading or document descriptor, not a reusable capability")
    if _contains_generic_skill_language(candidate.description):
        errors.append("Skill description is a generic evidence placeholder, not a capability description")
    if _contains_generic_skill_language(candidate.applicability):
        errors.append("Skill applicability is a generic evidence placeholder, not an operating context")
    for step in candidate.procedure:
        if _contains_generic_skill_language(step.instruction):
            errors.append(f"Procedure step {step.ordinal} is a generic placeholder")
    return errors

def _validate_citations(
    citations: list[EvidenceCitation], evidence_by_id: dict[str, EvidenceExcerpt]
) -> list[str]:
    """Validate citation identities and image ownership, without matching quote text."""
    errors: list[str] = []
    for citation in citations:
        evidence = evidence_by_id.get(citation.evidence_id)
        if not evidence:
            errors.append(f"Unknown Evidence ID: {citation.evidence_id}")
            continue
        if citation.image_artifact_id:
            allowed_image_ids = {item.artifact_id for item in evidence.image_refs}
            if citation.image_artifact_id not in allowed_image_ids:
                errors.append(
                    f"Image Artifact {citation.image_artifact_id} is not attached to Evidence {citation.evidence_id}"
                )
    return errors


def _deduplicate_by_key(items: list[T], key: Callable[[T], str]) -> list[T]:
    unique: dict[str, T] = {}
    for item in items:
        unique.setdefault(key(item), item)
    return list(unique.values())


def task_category_evidence_index(
    evidence_by_id: dict[str, EvidenceExcerpt], evidence_trees: list[EvidenceTree],
) -> dict[str, set[str]]:
    """Resolve exact authored first-level chapter names and their citable spans."""
    evidence_by_part: dict[tuple[str, str], list[EvidenceExcerpt]] = {}
    for evidence in evidence_by_id.values():
        evidence_by_part.setdefault((evidence.source_id, evidence.source_part_id), []).append(evidence)
    categories: dict[str, set[str]] = {}
    for tree in evidence_trees:
        nodes = {node.evidence_node_id: node for node in tree.nodes}
        for node in tree.nodes:
            if node.parent_node_id == tree.root_node_id:
                categories.setdefault(node.title, set())
        for node in tree.nodes:
            branch = node
            while branch.parent_node_id and branch.parent_node_id != tree.root_node_id:
                branch = nodes[branch.parent_node_id]
            if branch.parent_node_id != tree.root_node_id:
                continue
            for ref in node.source_refs:
                for evidence in evidence_by_part.get((tree.source_id, ref.source_part_id), []):
                    if evidence.start_offset < ref.end_offset and ref.start_offset < evidence.end_offset:
                        categories[branch.title].add(evidence.evidence_id)
    return categories


def validate_tasks(
    candidates: list[TaskCandidate], evidence_by_id: dict[str, EvidenceExcerpt],
    evidence_trees: list[EvidenceTree],
) -> tuple[list[TaskCandidate], list[CandidateValidation]]:
    accepted: list[TaskCandidate] = []
    validation: list[CandidateValidation] = []
    categories = task_category_evidence_index(evidence_by_id, evidence_trees)
    for candidate in _deduplicate_by_key(
        candidates,
        lambda item: (
            f"{_compact(item.task).lower()}|{'|'.join(sorted(c.evidence_id for c in item.citations))}"
        ),
    ):
        errors = _validate_citations(candidate.citations, evidence_by_id)
        if candidate.category not in categories:
            errors.append("Task category must exactly match an existing first-level EvidenceTree chapter title")
        elif not {citation.evidence_id for citation in candidate.citations} & categories[candidate.category]:
            errors.append("Task category must have supporting cited Evidence in that first-level chapter")
        task_text = [candidate.task]
        errors.extend(_template_placeholder_errors(task_text, label="Task"))
        validation.append(
            CandidateValidation(
                candidate_kind="task",
                candidate_id=candidate.task_id,
                accepted=not errors,
                errors=errors,
            )
        )
        if not errors:
            accepted.append(candidate.model_copy(update={"status": "validated"}))
    return accepted, validation


def validate_references(
    candidates: list[ReferenceCandidate], evidence_by_id: dict[str, EvidenceExcerpt]
) -> tuple[list[ReferenceCandidate], list[CandidateValidation]]:
    accepted: list[ReferenceCandidate] = []
    validation: list[CandidateValidation] = []
    for candidate in _deduplicate_by_key(
        candidates,
        lambda item: (
            f"{item.reference_mode}|{item.source_catalog_id or ''}|{_compact(item.title).lower()}|"
            f"{'|'.join(sorted(c.evidence_id for c in item.citations))}"
        ),
    ):
        from ..reference_layout import presentation_errors
        errors = _validate_citations(candidate.citations, evidence_by_id)
        errors.extend(presentation_errors(candidate))
        if candidate.reference_mode == "source_catalog":
            errors.extend(catalog_description_errors(candidate.description))
            if not candidate.source_catalog_id:
                errors.append("Source catalog Reference is missing source_catalog_id")
            if not candidate.entries:
                errors.append("Source catalog Reference has no preserved entries")
            entry_ids: set[str] = set()
            top_level_evidence_ids = {citation.evidence_id for citation in candidate.citations}
            for entry in candidate.entries:
                if entry.entry_id in entry_ids:
                    errors.append(f"Duplicate Reference entry ID: {entry.entry_id}")
                entry_ids.add(entry.entry_id)
                source_evidence = evidence_by_id.get(entry.source_evidence_id)
                if not source_evidence:
                    errors.append(
                        f"Reference entry {entry.entry_id} cites unknown source Evidence {entry.source_evidence_id}"
                    )
                    continue
                if entry.content != source_evidence.content and entry.content not in source_evidence.content:
                    errors.append(
                        f"Reference entry {entry.entry_id} is not an exact source Evidence text span"
                    )
                if entry.source_evidence_id not in top_level_evidence_ids:
                    errors.append(
                        f"Reference entry {entry.entry_id} is missing a top-level citation for {entry.source_evidence_id}"
                    )
                if not any(
                    citation.evidence_id == entry.source_evidence_id for citation in entry.citations
                ):
                    errors.append(
                        f"Reference entry {entry.entry_id} has no citation to its source Evidence"
                    )
                errors.extend(
                    f"Reference entry {entry.entry_id}: {error}"
                    for error in _validate_citations(entry.citations, evidence_by_id)
                )
        elif candidate.entries:
            errors.append("Summary Reference cannot contain source catalog entries")
        for unit in [candidate, *candidate.entries]:
            cited_ids = {c.evidence_id for c in unit.citations}
            for image in unit.image_refs:
                evidence = evidence_by_id.get(image.evidence_id)
                if (image.evidence_id not in cited_ids or evidence is None
                        or not any(ref.artifact_id == image.artifact_id
                                   and ref.media_type == image.media_type for ref in evidence.image_refs)):
                    errors.append(f"Reference image {image.artifact_id} is not attached to its cited Evidence")
        validation.append(
            CandidateValidation(
                candidate_kind="reference",
                candidate_id=candidate.reference_id,
                accepted=not errors,
                errors=errors,
            )
        )
        if not errors:
            accepted.append(hydrate_reference_images(candidate, evidence_by_id).model_copy(update={"status": "validated"}))
    return accepted, validation


def validate_trajectories(
    candidates: list[TrajectoryCandidate],
    evidence_by_id: dict[str, EvidenceExcerpt],
    tasks: set[str] | dict[str, TaskCandidate],
) -> tuple[list[TrajectoryCandidate], list[CandidateValidation]]:
    task_ids = set(tasks)
    tasks_by_id = tasks if isinstance(tasks, dict) else {}
    accepted: list[TrajectoryCandidate] = []
    validation: list[CandidateValidation] = []
    for candidate in _deduplicate_by_key(candidates, lambda item: item.task_id):
        errors: list[str] = []
        if candidate.task_id not in task_ids:
            errors.append(f"Unknown Task ID: {candidate.task_id}")
        errors.extend(_validate_citations(candidate.citations, evidence_by_id))
        trajectory_text = [
            *(step.instruction for step in candidate.steps),
            *(step.expected_observation for step in candidate.steps),
        ]
        errors.extend(_template_placeholder_errors(trajectory_text, label="Trajectory"))
        task = tasks_by_id.get(candidate.task_id)
        if task:
            task_text = [task.task]
            errors.extend(
                _introduced_catalog_scope_errors(
                    candidate.citations,
                    task.citations,
                    evidence_by_id,
                    parent_text=task_text,
                    child_text=trajectory_text,
                    child_label="Trajectory",
                )
            )
        ordinals = [step.ordinal for step in candidate.steps]
        if ordinals != list(range(1, len(ordinals) + 1)):
            errors.append("Trajectory step ordinals must be consecutive starting at 1")
        for step in candidate.steps:
            errors.extend(
                f"Step {step.ordinal}: {error}"
                for error in _validate_citations(step.citations, evidence_by_id)
            )
        validation.append(
            CandidateValidation(
                candidate_kind="trajectory",
                candidate_id=candidate.trajectory_id,
                accepted=not errors,
                errors=errors,
            )
        )
        if not errors:
            accepted.append(candidate.model_copy(update={"status": "validated"}))
    return accepted, validation


def _skill_citations(candidate: SkillDraft | SkillCandidate) -> list[EvidenceCitation]:
    return [
        citation
        for item in candidate.procedure
        for citation in item.citations
    ]


def validate_skills(
    candidates: list[SkillCandidate],
    evidence_by_id: dict[str, EvidenceExcerpt],
    tasks: set[str] | dict[str, TaskCandidate],
    trajectories: set[str] | dict[str, TrajectoryCandidate],
) -> tuple[list[SkillCandidate], list[CandidateValidation]]:
    task_ids = set(tasks)
    trajectory_ids = set(trajectories)
    tasks_by_id = tasks if isinstance(tasks, dict) else {}
    trajectories_by_id = trajectories if isinstance(trajectories, dict) else {}
    accepted: list[SkillCandidate] = []
    validation: list[CandidateValidation] = []
    for candidate in _deduplicate_by_key(
        candidates,
        lambda item: f"{_compact(item.title).lower()}|{'|'.join(sorted(item.trajectory_ids))}",
    ):
        errors: list[str] = []
        warnings: list[str] = []
        missing_trajectories = sorted(set(candidate.trajectory_ids) - trajectory_ids)
        if missing_trajectories:
            errors.append(f"Unknown Trajectory IDs: {', '.join(missing_trajectories)}")
        skill_usage_text = [
            candidate.preconditions,
            *(step.instruction for step in candidate.procedure),
        ]
        skill_scope_text = [
            candidate.title,
            candidate.description,
            candidate.applicability,
            *skill_usage_text,
        ]
        errors.extend(_template_placeholder_errors(skill_scope_text, label="Skill"))
        linked_trajectories = [
            trajectories_by_id[trajectory_id]
            for trajectory_id in candidate.trajectory_ids
            if trajectory_id in trajectories_by_id
        ]
        errors.extend(_skill_quality_errors(candidate))
        linked_parent_task_ids = {item.task_id for item in linked_trajectories}
        missing_parent_tasks = sorted(linked_parent_task_ids - task_ids)
        if missing_parent_tasks:
            errors.append(
                "Linked Trajectories reference unknown Task IDs: "
                + ", ".join(missing_parent_tasks)
            )
        linked_tasks = [
            tasks_by_id[task_id]
            for task_id in sorted(linked_parent_task_ids)
            if task_id in tasks_by_id
        ]
        if linked_tasks:
            parent_text = [
                value
                for task in linked_tasks
                for value in (task.task,)
            ]
            parent_citations = [
                citation for task in linked_tasks for citation in task.citations
            ]
            errors.extend(
                _introduced_catalog_scope_errors(
                    _skill_citations(candidate),
                    parent_citations,
                    evidence_by_id,
                    parent_text=parent_text,
                    child_text=skill_scope_text,
                    child_label="Skill",
                )
            )
        ordinals = [step.ordinal for step in candidate.procedure]
        if ordinals != list(range(1, len(ordinals) + 1)):
            errors.append("Skill procedure ordinals must be consecutive starting at 1")
        for step in candidate.procedure:
            errors.extend(
                f"Procedure step {step.ordinal}: {error}"
                for error in _validate_citations(step.citations, evidence_by_id)
            )
        validation.append(
            CandidateValidation(
                candidate_kind="skill",
                candidate_id=candidate.skill_id,
                accepted=not errors,
                errors=errors,
                warnings=warnings,
            )
        )
        if not errors:
            accepted.append(candidate.model_copy(update={"status": "validated"}))
    return accepted, validation
