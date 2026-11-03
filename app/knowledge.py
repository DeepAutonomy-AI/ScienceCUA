from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from .config import Settings
from .agent_export import export_agent_entries
from .models import (
    CandidateValidation,
    CorpusSnapshot,
    EvidenceCitation,
    EvidenceExcerpt,
    EvidenceNode,
    EvidenceTree,
    KnowledgeSnapshot,
    ModelRuntimeConfig,
    ReferenceCandidate,
    ReferenceDraft,
    ReferenceEntry,
    SkillCandidate,
    SkillDraft,
    TaskCandidate,
    TaskDraft,
    TargetProfile,
    TrajectoryCandidate,
    TrajectoryDraft,
    TrajectoryStepDraft,
    utc_now,
)
from .storage import RunStore
from .reference_images import hydrate_reference_images
from .reference_text import reference_text, reference_text_payload, text_only_reference


# Keep the established app.knowledge import surface while implementation lives
# in modules organized by responsibility.
from .knowledge_components.common import (
    T,
    CATALOG_DESCRIPTION_MIN_CHARS,
    GENERIC_CATALOG_DESCRIPTION_PREFIX,
    REFERENCE_ORDINARY_REVIEW_LIMIT,
    REFERENCE_ORDINARY_CONTEXT_CHARS,
    REFERENCE_CATALOG_SAMPLE_SIZE,
    REFERENCE_PARTITION_REVIEW_CHARS,
    REFERENCE_PARTITION_OUTLINE_LIMIT,
    REFERENCE_PARTITION_MAX_GROUPS,
    catalog_description_fallback,
    catalog_description_errors,
    resolved_catalog_description,
    KnowledgeStage,
    _stable_id,
    _compact,
    _safe_filename,
    _WINDOWS_RESERVED_FILENAMES,
    _safe_skill_markdown_stem,
    _excerpt_quote,
)
from .knowledge_components.reference_catalogs import (
    _REFERENCE_REVIEW_SIGNAL_PATTERNS,
    _representative_reference_excerpt,
    _NUMBERED_REFERENCE_ENTRY,
    _INLINE_ENTRY_LABEL,
    _INLINE_CATALOG_MARKERS,
    _INLINE_SECTION_HEADINGS,
    _INLINE_TABLE_COLUMN_HEADINGS,
    _INLINE_SYMBOL_KEYS,
    _INLINE_COMBO_LABEL,
    _INLINE_COMBO_PREFIX,
    _INLINE_SPECIAL_KEY_LABEL,
    _INLINE_FUNCTION_KEY_LABEL,
    _INLINE_SINGLE_KEY_LABEL,
    _INLINE_ARROW_KEY_LABEL,
    _INLINE_MOUSE_OR_JOYSTICK_LABEL,
    _INLINE_NUMERIC_RANGE_LABEL,
    _INLINE_QUALIFIED_KEY_LABEL,
    _INLINE_BARE_WORD_LABEL,
    _INLINE_ARROW_PREFIX,
    _INLINE_MOUSE_PREFIX,
    _INLINE_JOYSTICK_PREFIX,
    _INLINE_QUALIFIED_KEY_PREFIX,
    _REFERENCE_LIKE_TITLE_MARKERS,
    _REFERENCE_FIELD_LABEL,
    _reference_entry_sort_key,
    _reference_field_labels,
    _strip_inline_code_marks,
    _normalize_inline_entry_label,
    _contains_inline_catalog_marker,
    _is_inline_catalog_heading,
    _is_inline_table_column_heading,
    _evidence_has_inline_catalog_context,
    _is_inline_entry_label,
    _split_inline_entry_line,
    _line_spans,
    _KEYWORD_INDEX_IDENTIFIER,
    _CATALOG_CHUNK_SUFFIX,
    _ENVIRONMENT_VARIABLE_LABEL,
    _AGGREGATE_CATALOG_URL_MARKERS,
    _keyword_index_identifier_list,
    _keyword_index_reference_entries,
    _markdown_table_reference_entries,
    _inline_reference_entries,
    _catalog_entry_identity,
    _catalog_display_stem,
    _merge_inline_catalog_group,
    _catalog_fragment_key,
    _merge_fragmented_reference_catalogs,
    _catalog_preference_key,
    _deduplicate_reference_catalogs,
    _environment_catalog,
    _shortcut_label,
    _shortcut_catalog,
    _catalog_capability_branch,
    _merge_reference_catalog_families,
    _AUTOMATIC_CATALOG_CONTEXT_MARKERS,
    _INLINE_CATALOG_PROSE_LABELS,
    _INLINE_CATALOG_STRUCTURED_LABEL,
    _inline_catalog_quality_profile,
    _reference_catalog_automation_profile,
    _automatic_reference_catalog_metadata,
    _metadata_only_repeated_catalog,
    _reference_like_evidence_profile,
    _build_reference_catalogs,
    _infer_reference_usage_tags,
)
from .knowledge_components.validation import (
    _SKILL_SECTION_TITLE,
    _SKILL_DOCUMENT_TITLE,
    _GENERIC_SKILL_PHRASES,
    _TEMPLATE_PLACEHOLDER,
    _LOOKUP_ACTION_MARKERS,
    _LOOKUP_RESULT_MARKERS,
    _REFERENCE_LOOKUP_ACTION_MARKERS,
    _REFERENCE_LOOKUP_OBJECT_MARKERS,
    _OPEN_INPUT_ACTION_MARKERS,
    _OPEN_KNOWLEDGE_OBJECT_MARKERS,
    _OPEN_REFERENCE_USAGE_TAGS,
    _LOOKUP_REFERENCE_USAGE_TAGS,
    _SOURCE_CATALOG_DEFAULT_USAGE_TAGS,
    _GENERIC_CAPABILITY_CATEGORIES,
    _contains_generic_skill_language,
    _literal_occurs,
    _requires_lookup,
    _requires_reference_lookup,
    _contains_template_placeholder,
    _template_placeholder_errors,
    _catalog_entry_label,
    _introduced_catalog_scope_errors,
    _skill_quality_errors,
    _validate_citations,
    _deduplicate_by_key,
    task_category_evidence_index,
    validate_tasks,
    validate_references,
    validate_trajectories,
    _skill_citations,
    validate_skills,
)
from .knowledge_components.quality import (
    build_pipeline_quality_report,
)
from .knowledge_components.registry import (
    SkillRegistry,
    _skill_markdown_labels,
)
from .knowledge_components.tool_catalog import (
    _knowledge_tool_catalog,
)


def load_corpus_snapshot(store: RunStore, artifact_id: str) -> CorpusSnapshot:
    record = store.artifact_record(artifact_id)
    if record.kind != "corpus_snapshot":
        raise ValueError(f"Artifact {artifact_id} is not a corpus snapshot")
    payload = json.loads(store.artifact_path(artifact_id).read_text(encoding="utf-8"))
    return CorpusSnapshot.model_validate(payload)


def load_knowledge_snapshot(store: RunStore, artifact_id: str) -> KnowledgeSnapshot:
    record = store.artifact_record(artifact_id)
    if record.kind != "knowledge_snapshot":
        raise ValueError(f"Artifact {artifact_id} is not a knowledge snapshot")
    payload = json.loads(store.artifact_path(artifact_id).read_text(encoding="utf-8"))
    return KnowledgeSnapshot.model_validate(payload)


class KnowledgeWorkspace:
    """Read-only Evidence view plus bounded candidate submissions for one downstream stage."""

    def _id(self, short_prefix: str, legacy_prefix: str, *identity: str) -> str:
        return self.store.canonical_id(
            short_prefix, _stable_id(legacy_prefix, *identity), legacy_prefix, *identity
        )

    def __init__(
        self,
        *,
        store: RunStore,
        settings: Settings,
        target_profile: TargetProfile,
        stage: KnowledgeStage,
        corpus: CorpusSnapshot,
        corpus_artifact_id: str,
        knowledge: KnowledgeSnapshot | None = None,
        knowledge_artifact_id: str | None = None,
        task: TaskCandidate | None = None,
        trajectory_tasks: list[TaskCandidate] | None = None,
        trajectory_branch_node_id: str | None = None,
        trajectory_packet_full_evidence_ids: set[str] | None = None,
        trajectories: list[TrajectoryCandidate] | None = None,
        candidate_tasks: list[TaskCandidate] | None = None,
        candidate_references: list[ReferenceCandidate] | None = None,
    ):
        self.store = store
        self.settings = settings
        self.target_profile = target_profile
        self.stage = stage
        self.corpus = corpus
        self.corpus_artifact_id = corpus_artifact_id
        self.knowledge = knowledge
        self.knowledge_artifact_id = knowledge_artifact_id
        self.task = task
        self.trajectory_tasks = list(trajectory_tasks) if trajectory_tasks is not None else ([task] if task else [])
        self.trajectory_slots = {index: item for index, item in enumerate(self.trajectory_tasks, 1)}
        self.trajectory_branch_node_id = trajectory_branch_node_id
        self.trajectory_packet_full_evidence_ids = set(trajectory_packet_full_evidence_ids or ())
        self.trajectories = trajectories or []
        self.evidence_by_id = {item.evidence_id: item for item in corpus.evidence}
        self._evidence_by_part_id: dict[str, list[EvidenceExcerpt]] = {}
        for evidence in corpus.evidence:
            self._evidence_by_part_id.setdefault(evidence.source_part_id, []).append(evidence)
        self._source_order_by_id = {
            item.source_id: index for index, item in enumerate(corpus.source_results)
        }
        self._evidence_order_by_id = {
            item.evidence_id: index for index, item in enumerate(corpus.evidence)
        }
        self.reference_catalogs = _build_reference_catalogs(
            corpus.corpus_snapshot_id, list(self.evidence_by_id.values())
        )
        for catalog in self.reference_catalogs:
            legacy_id = catalog["catalog_id"]
            catalog["catalog_id"] = self.store.canonical_id(
                "C", legacy_id, legacy_id
            )
        self.reference_catalogs_by_id = {
            item["catalog_id"]: item for item in self.reference_catalogs
        }
        available_tasks = candidate_tasks if candidate_tasks is not None else (knowledge.tasks if knowledge else [])
        available_references = (
            candidate_references if candidate_references is not None else (knowledge.references if knowledge else [])
        )
        if self.stage == "skill_induction":
            available_references = []
        self.tasks_by_id = {item.task_id: item for item in available_tasks}
        self.references_by_id = {
            item.reference_id: hydrate_reference_images(item, self.evidence_by_id) for item in available_references
        }
        self.initial_catalog_ids = {
            item.source_catalog_id
            for item in self.references_by_id.values()
            if item.reference_mode == "source_catalog" and item.source_catalog_id
        }
        self.trajectories_by_id = {item.trajectory_id: item for item in self.trajectories}
        self.initial_task_count = len(self.tasks_by_id)
        self.initial_reference_count = len(self.references_by_id)
        self.submitted_tasks: list[TaskCandidate] = []
        self.submitted_references: list[ReferenceCandidate] = []
        self.submitted_catalog_ids: set[str] = set(self.initial_catalog_ids)
        self.submitted_trajectories: list[TrajectoryCandidate] = []
        self.trajectory_task_results: dict[str, dict[str, Any]] = {}
        self.submitted_skills: list[SkillCandidate] = []
        self.reference_catalog_review: dict[str, dict[str, str]] = {}
        self.reference_high_confidence_catalog_ids: set[str] = set()
        self.reference_review_catalog_ids: set[str] = set()
        self.reference_candidate_review: dict[str, dict[str, Any]] = {}
        self.reference_review_candidate_ids: set[str] = set()
        self.reference_partition_review: dict[str, dict[str, Any]] = {}
        self.reference_preparation: dict[str, Any] | None = None
        self.reference_base_units = self._build_reference_units()
        self.reference_base_units_by_node_id = {
            item["evidence_node_id"]: item for item in self.reference_base_units
        }
        self._task_node_ids_by_evidence_id: dict[str, list[str]] = {}
        for unit in self.reference_base_units:
            for evidence_id in unit["evidence_ids"]:
                self._task_node_ids_by_evidence_id.setdefault(evidence_id, []).append(
                    unit["evidence_node_id"]
                )
        # Reference Build starts from these lossless one-node units, then replaces
        # the dispatch surface with the semantic partition proposed by the first
        # turn of the same Reference Child Session. Task Discovery reuses the
        # node-to-Evidence mapping for selective tree navigation.
        self.reference_units = list(self.reference_base_units)
        self.reference_units_by_id = {
            item["unit_id"]: item for item in self.reference_units
        }
        self.reference_reading_plan: dict[str, Any] | None = None
        self.reference_reading_plan_artifact_id: str | None = None
        self.reference_unit_reviews: dict[str, dict[str, Any]] = {}
        self.active_reference_unit_id: str | None = None
        self.task_unit_reads: set[str] = set()
        self.task_evidence_reads: set[str] = set()
        self.task_node_reads: set[str] = set()
        self._task_evidence_listed_ranges: dict[
            tuple[str, str, bool, int, str], list[tuple[int, int]]
        ] = {}
        # Retained for backward-compatible reading of older attempt artifacts;
        # selective Task Discovery no longer requires per-unit dispositions.
        self.task_unit_reviews: dict[str, dict[str, Any]] = {}
        self.active_task_unit_id: str | None = None
        self.skill_coverage_review: dict[str, dict[str, str]] = {}
        self.stage_result: dict[str, Any] | None = None
        self.trajectory_attempt_id: str | None = None
        self.trajectory_checkpoint_artifact_id: str | None = None
        self.trajectory_checkpoint_artifact_ids: dict[str, str] = {}
        self._trajectory_terminal_request: dict[str, Any] | None = None

    @property
    def is_finished(self) -> bool:
        return self.stage_result is not None

    def context(self) -> dict[str, Any]:
        evidence_by_source: dict[str, int] = {}
        evidence_by_kind: dict[str, int] = {}
        for evidence in self.evidence_by_id.values():
            evidence_by_source[evidence.source_id] = evidence_by_source.get(evidence.source_id, 0) + 1
            evidence_by_kind[evidence.kind] = evidence_by_kind.get(evidence.kind, 0) + 1
        context = {
            "stage": self.stage,
            "target_profile": self.target_profile.model_dump(mode="json"),
            "corpus_snapshot_id": self.corpus.corpus_snapshot_id,
            "corpus_snapshot_artifact_id": self.corpus_artifact_id,
            "evidence_count": len(self.evidence_by_id),
            "multimodal_evidence_count": sum(
                1 for item in self.evidence_by_id.values() if item.image_refs
            ),
            "evidence_image_count": len(
                {
                    image.artifact_id
                    for item in self.evidence_by_id.values()
                    for image in item.image_refs
                }
            ),
            "source_status_counts": self.corpus.source_status_counts,
            "knowledge_snapshot_id": self.knowledge.knowledge_snapshot_id if self.knowledge else None,
            "current_task_id": self.task.task_id if self.task else None,
            "available_task_count": len(self.tasks_by_id),
            "available_reference_count": len(self.references_by_id),
            "available_trajectory_count": len(self.trajectories_by_id),
            "evidence_inventory": {
                "by_source": evidence_by_source,
                "by_kind": evidence_by_kind,
            },
            "remaining_task_capacity": max(
                0, self.settings.max_task_candidates - self.initial_task_count - len(self.submitted_tasks)
            ),
            "remaining_reference_capacity": max(
                0,
                self.settings.max_knowledge_candidates
                - self.initial_reference_count
                - len(self.submitted_references),
            ),
        }
        if self.stage in {"task_discovery", "task_trajectory"}:
            context["task_reading_policy"] = {
                "mode": "evidence_tree_node_reading",
                "unit": "EvidenceTree node with exact, individually citable source segments",
                "evidence_count": len(self.evidence_by_id),
                "evidence_tree_count": len(self.corpus.evidence_trees),
                "source_parts_are_immutable": True,
                "coverage": "inspect major branches, then selectively read source-grounded Evidence; unread nodes are allowed",
                "selected_evidence_count": len(self.task_evidence_reads),
                "selected_node_count": len(self.task_node_reads),
                "remaining_task_capacity": context["remaining_task_capacity"],
            }
        elif self.stage == "reference_build":
            context["reference_reading_policy"] = {
                "mode": "python_candidates_agent_review",
                "candidate_summary": self.reference_candidate_summary(),
                "source_parts_are_immutable": True,
                "coverage": "Every prepared candidate requires an explicit Agent preserve/exclude decision",
            }
            context["reference_catalog_candidate_count"] = len(self.reference_catalogs)
            context["reference_catalog_coverage"] = self.inspect_reference_coverage()
        if self.stage == "skill_induction":
            context.pop("available_reference_count")
            context.pop("remaining_reference_capacity")
            context["remaining_skill_capacity"] = max(
                0, self.settings.max_knowledge_candidates - len(self.submitted_skills)
            )
            context["skill_coverage"] = self.inspect_skill_coverage()
        return context

    @staticmethod
    def _reference_total_chars(reference: ReferenceCandidate) -> int:
        return len(reference.content) + sum(len(entry.content) for entry in reference.entries)

    def _partition_rows_for_catalog(self, catalog: dict[str, Any]) -> list[dict[str, Any]]:
        raw_entries = list(catalog.get("catalog_entries") or [])
        if raw_entries:
            return [
                {
                    "title": str(item["title"]),
                    "content": str(item["content"]),
                    "source_evidence_id": str(item["source_evidence_id"]),
                    "locator": dict(item.get("locator") or {}),
                }
                for item in raw_entries
            ]
        return [
            {
                "title": self.evidence_by_id[evidence_id].title,
                "content": self.evidence_by_id[evidence_id].content,
                "source_evidence_id": evidence_id,
                "locator": dict(self.evidence_by_id[evidence_id].locator),
            }
            for evidence_id in catalog.get("evidence_ids") or []
            if evidence_id in self.evidence_by_id
        ]

    def _ordinary_reference_evidence_ids(self, unit: dict[str, Any]) -> list[str]:
        """Use the same ordinary source selection for review and materialization."""
        structural_ids = {
            str(evidence_id)
            for catalog in self.reference_catalogs
            if not catalog.get("catalog_entries")
            for evidence_id in catalog.get("evidence_ids", [])
        }
        return [evidence_id for evidence_id in unit.get("evidence_ids", [])
                if evidence_id in self.evidence_by_id and evidence_id not in structural_ids]

    def _partition_rows_for_unit(self, unit: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "title": self.evidence_by_id[evidence_id].title,
                "content": self.evidence_by_id[evidence_id].content,
                "source_evidence_id": evidence_id,
                "locator": dict(self.evidence_by_id[evidence_id].locator),
            }
            for evidence_id in self._ordinary_reference_evidence_ids(unit)
            if evidence_id in self.evidence_by_id
        ]

    def _partition_manifest_item(
        self,
        *,
        subject_id: str,
        source_kind: str,
        title: str,
        description: str,
        rows: list[dict[str, Any]],
        structure: str | None = None,
    ) -> dict[str, Any] | None:
        total_chars = len(description) + sum(len(str(item["content"])) for item in rows)
        if total_chars <= REFERENCE_PARTITION_REVIEW_CHARS:
            return None
        total = len(rows)
        if total <= REFERENCE_PARTITION_OUTLINE_LIMIT:
            indices = list(range(total))
        else:
            indices = sorted(
                {
                    round(index * (total - 1) / (REFERENCE_PARTITION_OUTLINE_LIMIT - 1))
                    for index in range(REFERENCE_PARTITION_OUTLINE_LIMIT)
                }
            )
        return {
            "subject_id": subject_id,
            "source_kind": source_kind,
            "title": title,
            "structure": structure,
            "total_chars": total_chars,
            "entry_count": total,
            "threshold_chars": REFERENCE_PARTITION_REVIEW_CHARS,
            "outline_complete": len(indices) == total,
            "outline_sampling": (
                "complete" if len(indices) == total else "deterministic_stratified"
            ),
            "outline": [
                {
                    "entry_index": index,
                    "source_evidence_id": str(rows[index]["source_evidence_id"]),
                    "title": str(rows[index]["title"]),
                    "char_count": len(str(rows[index]["content"])),
                    "structure_locator": {
                        key: value
                        for key, value in dict(rows[index].get("locator") or {}).items()
                        if key
                        in {
                            "page",
                            "page_start",
                            "page_end",
                            "heading",
                            "section_path",
                            "chapter_heading",
                        }
                    },
                }
                for index in indices
            ],
        }

    def prepare_reference_build(self) -> dict[str, Any]:
        """Prepare exact source candidates; only the Agent can accept References."""
        if self.stage != "reference_build":
            raise PermissionError("Reference preparation is only allowed in reference_build")
        if self.reference_preparation is not None:
            return self.reference_preparation
        catalogs: list[dict[str, Any]] = []
        partitions: list[dict[str, Any]] = []
        for catalog in self.reference_catalogs:
            catalog_id = str(catalog["catalog_id"])
            proof = _reference_catalog_automation_profile(catalog, self.evidence_by_id)
            high_confidence = bool(proof.get("automatic"))
            self.reference_review_catalog_ids.add(catalog_id)
            if high_confidence:
                self.reference_high_confidence_catalog_ids.add(catalog_id)
            descriptor = {
                **self._reference_catalog_descriptor(catalog),
                "confidence_hint": "high" if high_confidence else "ambiguous",
                "structure_proof": {key: value for key, value in proof.items() if key != "automatic"},
                "samples": self._catalog_manifest_samples(catalog_id),
            }
            catalogs.append(descriptor)
            partition = self._partition_manifest_item(
                subject_id=catalog_id, source_kind="catalog_candidate",
                title=str(descriptor.get("suggested_title") or catalog.get("suggested_title") or catalog_id),
                description="", rows=self._partition_rows_for_catalog(catalog),
                structure=str(catalog.get("structure") or ""),
            )
            if partition is not None:
                partitions.append(partition)
        ordinary = [item for item in self._reference_candidate_descriptors(min_reference_like_score=2)
                    if item["candidate_kind"] != "catalog_candidate"]
        self.reference_review_candidate_ids = {str(item["candidate_id"]) for item in ordinary}
        for item in ordinary:
            candidate_id = str(item["candidate_id"])
            unit = self.reference_base_units_by_node_id[candidate_id]
            partition = self._partition_manifest_item(
                subject_id=candidate_id, source_kind="ordinary_candidate",
                title=str(item.get("title") or unit["title"]), description="",
                rows=self._partition_rows_for_unit(unit), structure="evidence_tree_reading_unit",
            )
            if partition is not None:
                partitions.append(partition)
        self.reference_preparation = {
            "mode": "python_candidates_agent_review",
            "catalog_candidate_count": len(catalogs),
            "high_confidence_catalog_count": len(self.reference_high_confidence_catalog_ids),
            "ordinary_candidate_count": len(ordinary),
            "catalog_candidates": catalogs, "ordinary_candidates": ordinary,
            "partition_review_threshold_chars": REFERENCE_PARTITION_REVIEW_CHARS,
            "partition_review_candidates": partitions,
            "source_parts_are_immutable": True,
        }
        self.store.emit("reference_preparation_completed", {
            "catalog_candidate_count": len(catalogs),
            "high_confidence_catalog_count": len(self.reference_high_confidence_catalog_ids),
            "ordinary_candidate_count": len(ordinary),
            "partition_review_candidate_count": len(partitions),
        }, agent="deterministic_reference_preparer")
        return self.reference_preparation

    def _catalog_manifest_samples(self, catalog_id: str) -> list[dict[str, Any]]:
        catalog = self.reference_catalogs_by_id[catalog_id]
        raw_entries = list(catalog.get("catalog_entries") or [])
        if raw_entries:
            entries = [
                {
                    "evidence_id": item["source_evidence_id"],
                    "title": item["title"],
                    "preview": _excerpt_quote(str(item["content"]), maximum=320),
                    "locator": self._reference_locator_summary(dict(item.get("locator") or {})),
                }
                for item in raw_entries
            ]
        else:
            entries = [
                {
                    "evidence_id": evidence_id,
                    "title": self.evidence_by_id[evidence_id].title,
                    "preview": _excerpt_quote(
                        self.evidence_by_id[evidence_id].content, maximum=320
                    ),
                    "locator": self._reference_locator_summary(
                        self.evidence_by_id[evidence_id].locator
                    ),
                }
                for evidence_id in catalog.get("evidence_ids", [])
                if evidence_id in self.evidence_by_id
            ]
        total = len(entries)
        indices = sorted({0, total // 2, total - 1}) if total else []
        return [{**entries[index], "sample_index": index} for index in indices]

    def _apply_reference_partition_decisions(
        self,
        subject_references: dict[str, ReferenceCandidate],
        decisions: dict[str, dict[str, Any]],
        *, emit_event: bool = True,
    ) -> dict[str, Any]:
        replacements: dict[str, list[ReferenceCandidate]] = {}
        review: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        for subject_id, raw in decisions.items():
            reference = subject_references[subject_id]
            decision = str(raw.get("decision") or "").strip()
            groups = list(raw.get("groups") or [])
            if decision not in {"keep_whole", "split"}:
                errors.append(f"Partition subject {subject_id} has invalid decision: {decision or '<empty>'}")
                continue
            if decision == "keep_whole":
                if groups:
                    errors.append(f"Partition subject {subject_id} must not provide groups for keep_whole")
                review[subject_id] = {
                    "decision": "keep_whole",
                    "original_reference_id": reference.reference_id,
                    "result_reference_ids": [reference.reference_id],
                    "total_chars": self._reference_total_chars(reference),
                }
                continue
            if not 2 <= len(groups) <= REFERENCE_PARTITION_MAX_GROUPS:
                errors.append(
                    f"Partition subject {subject_id} must contain between 2 and "
                    f"{REFERENCE_PARTITION_MAX_GROUPS} groups"
                )
                continue

            partitioned: list[ReferenceCandidate] = []
            flattened_indices: list[int] = []
            seen_titles: set[str] = set()
            for group_index, group in enumerate(groups):
                if not isinstance(group, dict):
                    errors.append(f"Partition subject {subject_id} group {group_index} must be an object")
                    continue
                title = _compact(str(group.get("title") or ""))
                description = _compact(str(group.get("description") or ""))
                raw_indices = list(group.get("entry_indices") or [])
                indices: list[int] = []
                for value in raw_indices:
                    if isinstance(value, bool) or not isinstance(value, int):
                        errors.append(
                            f"Partition subject {subject_id} group {group_index} contains a non-integer entry index"
                        )
                        continue
                    indices.append(value)
                if not title:
                    errors.append(f"Partition subject {subject_id} group {group_index} requires a title")
                if title.casefold() in seen_titles:
                    errors.append(f"Partition subject {subject_id} has duplicate group title: {title}")
                seen_titles.add(title.casefold())
                if not description or len(description) > 120:
                    errors.append(f"Partition subject {subject_id} group {group_index} requires a description of 1 to 120 characters, including spaces and punctuation")
                if not indices:
                    errors.append(f"Partition subject {subject_id} group {group_index} has no entry indices")
                    continue
                if indices != sorted(set(indices)):
                    errors.append(
                        f"Partition subject {subject_id} group {group_index} entry indices must be unique and increasing"
                    )
                    continue
                if indices[0] < 0 or indices[-1] >= len(reference.entries):
                    errors.append(f"Partition subject {subject_id} group {group_index} has an out-of-range entry index")
                    continue
                flattened_indices.extend(indices)
                selected_entries = [reference.entries[index] for index in indices]
                partition_catalog_id = self._id(
                    "C", "rcat",
                    reference.source_catalog_id or reference.reference_id,
                    "semantic_partition",
                    *[entry.entry_id for entry in selected_entries],
                )
                citations: list[EvidenceCitation] = []
                citation_keys: set[tuple[str, str]] = set()
                for entry in selected_entries:
                    for citation in entry.citations:
                        key = (citation.evidence_id, citation.quote)
                        if key not in citation_keys:
                            citation_keys.add(key)
                            citations.append(citation)
                try:
                    partitioned.append(
                        ReferenceCandidate(
                            reference_id=self._id(
                                "R", "ref",
                                self.corpus.corpus_snapshot_id,
                                "source_catalog",
                                partition_catalog_id,
                                title.casefold(),
                            ),
                            corpus_snapshot_id=self.corpus.corpus_snapshot_id,
                            title=title,
                            category=_compact(str(group.get("category") or reference.category)),
                            description=description,
                            content="",
                            reference_mode="source_catalog",
                            injection_policy=_compact(
                                str(group.get("injection_policy") or reference.injection_policy)
                            ),
                            usage_tags=group.get("usage_tags") or reference.usage_tags,
                            source_catalog_id=partition_catalog_id,
                            entries=selected_entries,
                            citations=citations,
                        )
                    )
                except Exception as exc:
                    errors.append(f"Partition subject {subject_id} group {group_index} is invalid: {exc}")
            if flattened_indices != list(range(len(reference.entries))):
                errors.append(
                    f"Partition subject {subject_id} groups must cover every entry exactly once in original order"
                )
            if len(partitioned) == len(groups):
                replacements[reference.reference_id] = partitioned
                review[subject_id] = {
                    "decision": "split",
                    "original_reference_id": reference.reference_id,
                    "result_reference_ids": [item.reference_id for item in partitioned],
                    "total_chars": self._reference_total_chars(reference),
                }

        added = sum(len(items) - 1 for items in replacements.values())
        remaining = (
            self.settings.max_knowledge_candidates
            - self.initial_reference_count
            - len(self.submitted_references)
        )
        if added > remaining:
            errors.append(
                f"Reference partitions require {added} additional candidate slots but only {max(0, remaining)} remain"
            )
        if errors:
            raise ValueError("; ".join(errors))

        if replacements:
            materialized: list[ReferenceCandidate] = []
            for reference in self.submitted_references:
                materialized.extend(replacements.get(reference.reference_id, [reference]))
            self.submitted_references = materialized
        self.reference_partition_review = review
        if emit_event:
            self.store.emit(
                "reference_partition_review_completed",
                {
                    "threshold_chars": REFERENCE_PARTITION_REVIEW_CHARS,
                    "reviewed_count": len(review),
                    "split_count": sum(item["decision"] == "split" for item in review.values()),
                    "kept_whole_count": sum(
                        item["decision"] == "keep_whole" for item in review.values()
                    ),
                    "reviews": review,
                },
                agent="reference_build_agent",
            )
        return {
            "reviewed_count": len(review),
            "split_count": sum(item["decision"] == "split" for item in review.values()),
            "kept_whole_count": sum(item["decision"] == "keep_whole" for item in review.values()),
        }

    def submit_reference_review(
        self, catalog_decisions: list[dict[str, Any]],
        ordinary_decisions: list[dict[str, Any]] | None,
        partition_decisions: list[dict[str, Any]],
        status: Literal["succeeded", "degraded", "failed"], summary: str,
        gaps: list[str] | None = None, warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate all Agent decisions, then publish one complete Reference batch."""
        if self.stage_result is not None:
            raise ValueError("Reference stage result is already complete")
        preparation = self.prepare_reference_build()
        errors: list[str] = []

        def decisions_by_id(raw_items, id_key, allowed_ids):
            decisions = {}
            for index, raw in enumerate(raw_items or []):
                if not isinstance(raw, dict):
                    errors.append(f"{id_key} decision {index} must be an object")
                    continue
                candidate_id = str(raw.get(id_key) or "").strip()
                if candidate_id not in allowed_ids:
                    errors.append(f"Unknown {id_key} decision: {candidate_id or '<empty>'}")
                    continue
                if candidate_id in decisions:
                    errors.append(f"Duplicate {id_key} decision: {candidate_id}")
                    continue
                disposition = raw.get("disposition")
                if disposition not in {"preserve", "exclude"}:
                    errors.append(f"{candidate_id} has invalid disposition")
                if disposition == "preserve":
                    title = _compact(str(raw.get("title") or ""))
                    description = _compact(str(raw.get("description") or ""))
                    if not 3 <= len(title) <= 300:
                        errors.append(f"{candidate_id} requires a title of 3 to 300 characters")
                    if not description or len(description) > 120:
                        errors.append(f"{candidate_id} requires an Agent description of 1 to 120 characters, including spaces and punctuation; use one or two short sentences about knowledge scope and when to consult it")
                # Ignore the retired field when replaying a legacy decision payload.
                decisions[candidate_id] = {key: value for key, value in raw.items() if key != "rationale"}
            missing = sorted(allowed_ids - decisions.keys())
            if missing:
                errors.append(f"Missing {id_key} decisions: " + ", ".join(missing))
            return decisions

        by_catalog = decisions_by_id(catalog_decisions, "catalog_id", self.reference_review_catalog_ids)
        by_ordinary = decisions_by_id(ordinary_decisions, "candidate_id", self.reference_review_candidate_ids)
        partition_manifest = {str(item["subject_id"]): item
                              for item in preparation["partition_review_candidates"]}
        retained = {candidate_id for candidate_id, raw in {**by_catalog, **by_ordinary}.items()
                    if raw.get("disposition") == "preserve"}
        required = retained & partition_manifest.keys()
        partition_by_subject = {}
        for index, raw in enumerate(partition_decisions or []):
            if not isinstance(raw, dict):
                errors.append(f"Partition decision {index} must be an object")
                continue
            subject_id = str(raw.get("subject_id") or "").strip()
            if subject_id not in required:
                errors.append(f"Partition decision is not required for a preserved oversized Reference: {subject_id}")
                continue
            if subject_id in partition_by_subject:
                errors.append(f"Partition subject {subject_id} has duplicate decisions")
                continue
            partition_by_subject[subject_id] = raw
        missing = sorted(required - partition_by_subject.keys())
        if missing:
            errors.append("Missing mandatory partition decisions for References above "
                          f"{REFERENCE_PARTITION_REVIEW_CHARS} characters: " + ", ".join(missing))
        final_count = len(retained)
        for raw in partition_by_subject.values():
            if raw.get("decision") == "split":
                final_count += max(0, len(raw.get("groups") or []) - 1)
        remaining = self.settings.max_knowledge_candidates - self.initial_reference_count - len(self.submitted_references)
        if final_count > remaining:
            errors.append(f"Reference candidate capacity requires {final_count} slots but only {max(0, remaining)} remain")
        if status not in {"succeeded", "degraded", "failed"}:
            errors.append("Invalid Reference terminal status")
        if errors:
            raise ValueError("; ".join(errors))

        # Isolate mutable submission state until source and partition validation succeeds.
        staged = copy.copy(self)
        staged.submitted_references = list(self.submitted_references)
        staged.submitted_catalog_ids = set(self.submitted_catalog_ids)
        staged.reference_catalog_review = dict(self.reference_catalog_review)
        staged.reference_partition_review = dict(self.reference_partition_review)
        subject_references = {}
        catalog_ids = []
        collection_ids = []
        excluded = []
        for candidate_id, raw in by_catalog.items():
            if raw["disposition"] == "exclude":
                excluded.append(candidate_id)
                staged.reference_catalog_review[candidate_id] = {"disposition": "excluded"}
                continue
            result = staged.submit_reference_catalog(
                catalog_id=candidate_id, title=_compact(str(raw["title"])),
                description=_compact(str(raw["description"])), category=raw.get("category") or "source_catalog",
                injection_policy=raw.get("injection_policy") or "on_demand", usage_tags=raw.get("usage_tags"),
            )
            catalog_ids.append(str(result["candidate_id"]))
            staged.reference_catalog_review[candidate_id] = {"disposition": "covered"}
            if candidate_id in required:
                subject_references[candidate_id] = next(item for item in staged.submitted_references
                                                        if item.reference_id == result["candidate_id"])
        for candidate_id, raw in by_ordinary.items():
            if raw["disposition"] != "preserve":
                continue
            unit = self.reference_base_units_by_node_id[candidate_id]
            result = staged.submit_reference_collection(
                evidence_ids=self._ordinary_reference_evidence_ids(unit), title=_compact(str(raw["title"])),
                description=_compact(str(raw["description"])), category=raw.get("category") or "reference",
                injection_policy=raw.get("injection_policy") or "on_demand", usage_tags=raw.get("usage_tags"),
            )
            collection_ids.append(str(result["candidate_id"]))
            if candidate_id in required:
                subject_references[candidate_id] = next(item for item in staged.submitted_references
                                                        if item.reference_id == result["candidate_id"])
        partition_result = staged._apply_reference_partition_decisions(
            subject_references, partition_by_subject, emit_event=False,
        )
        validated, validation = validate_references(staged.submitted_references, self.evidence_by_id)
        if len(validated) != len(staged.submitted_references):
            details = [error for item in validation for error in item.errors]
            raise ValueError("Reference validation failed: " + "; ".join(details or ["duplicate Reference"]))
        coverage_errors = staged._reference_coverage_review_errors()
        if coverage_errors:
            raise ValueError("; ".join(coverage_errors))
        terminal = staged.submit_stage_result(status=status, summary=summary, gaps=gaps, warnings=warnings)
        self.submitted_references = staged.submitted_references
        self.submitted_catalog_ids = staged.submitted_catalog_ids
        self.reference_catalog_review = staged.reference_catalog_review
        self.reference_partition_review = staged.reference_partition_review
        self.reference_candidate_review = {candidate_id: dict(raw) for candidate_id, raw in by_ordinary.items()}
        self.stage_result = staged.stage_result
        self.stage_result["ordinary_candidate_review"] = self.reference_candidate_review
        self.store.emit("reference_review_completed", {
            "catalog_decision_count": len(by_catalog), "ordinary_decision_count": len(by_ordinary),
            "reference_count": len(self.submitted_references), "partition_review": partition_result,
        }, agent="reference_build_agent")
        return {
            "accepted": True, "catalog_candidate_count": preparation["catalog_candidate_count"],
            "materialized_catalog_ids": catalog_ids, "excluded_catalog_ids": excluded,
            "materialized_collection_ids": collection_ids,
            "partition_review": partition_result, "result": terminal["result"],
        }

    def _build_reference_units(self) -> list[dict[str, Any]]:
        """Partition tree-owned Evidence into semantic reading units.

        SourceParts remain immutable and independently citable.  The unit only
        groups every direct source span owned by the smallest source-bearing
        EvidenceNode so the model never receives one fragmented row at a time.
        Structural ancestors contribute their title path, not their descendants'
        text, which prevents a root-sized observation.
        """

        units: list[dict[str, Any]] = []
        source_names = {
            item.source_id: item.display_name for item in self.corpus.source_results
        }
        for tree in self.corpus.evidence_trees:
            nodes = {item.evidence_node_id: item for item in tree.nodes}

            def title_path(node_id: str) -> list[str]:
                path: list[str] = []
                seen: set[str] = set()
                current = nodes.get(node_id)
                while current is not None and current.evidence_node_id not in seen:
                    seen.add(current.evidence_node_id)
                    path.append(current.title)
                    current = nodes.get(current.parent_node_id or "")
                return list(reversed(path))

            for node in tree.nodes:
                if not node.source_refs:
                    continue
                evidence_ids: list[str] = []
                source_parts: list[dict[str, Any]] = []
                for source_ref in node.source_refs:
                    matches: list[EvidenceExcerpt] = []
                    for evidence in self._evidence_by_part_id.get(
                        source_ref.source_part_id, []
                    ):
                        if evidence.source_id != tree.source_id or not evidence.content:
                            continue
                        if (
                            evidence.start_offset < source_ref.end_offset
                            and source_ref.start_offset < evidence.end_offset
                        ):
                            matches.append(evidence)
                    matches.sort(key=self._evidence_reading_order_key)
                    matched_ids = [item.evidence_id for item in matches]
                    evidence_ids.extend(matched_ids)
                    source_parts.append(
                        {
                            "source_part_id": source_ref.source_part_id,
                            "start_offset": source_ref.start_offset,
                            "end_offset": source_ref.end_offset,
                            "content_sha256": source_ref.content_sha256,
                            "locator": dict(source_ref.locator),
                            "evidence_ids": matched_ids,
                        }
                    )
                evidence_ids = list(dict.fromkeys(evidence_ids))
                unit_id = self._id(
                    "U", "reu",
                    self.corpus.corpus_snapshot_id,
                    tree.evidence_tree_id,
                    node.evidence_node_id,
                )
                units.append(
                    {
                        "unit_id": unit_id,
                        "source_id": tree.source_id,
                        "source_name": source_names.get(tree.source_id, tree.source_id),
                        "evidence_tree_id": tree.evidence_tree_id,
                        "evidence_node_id": node.evidence_node_id,
                        "node_key": node.node_key,
                        "node_type": node.node_type,
                        "title": node.title,
                        "title_path": title_path(node.evidence_node_id),
                        "is_leaf": not bool(node.child_node_ids),
                        "child_count": len(node.child_node_ids),
                        "direct_char_count": node.direct_char_count,
                        "source_part_count": len(source_parts),
                        "source_parts": source_parts,
                        "evidence_ids": evidence_ids,
                    }
                )
        return units

    def reference_reading_plan_context(self) -> dict[str, Any]:
        """Return a compact EvidenceTree map with conservative Reference signals.

        Python never makes the semantic keep/skip decision here.  It only exposes
        high-recall structural signals so the Child can avoid reading obvious
        non-candidates while treating ambiguous nodes conservatively.
        """

        source_names = {
            item.source_id: item.display_name for item in self.corpus.source_results
        }
        unit_by_node = self.reference_base_units_by_node_id
        evidence_profiles = {
            evidence_id: _reference_like_evidence_profile(evidence)
            for evidence_id, evidence in self.evidence_by_id.items()
        }
        evidence_to_nodes: dict[str, list[str]] = {}
        for unit in self.reference_base_units:
            for evidence_id in unit["evidence_ids"]:
                evidence_to_nodes.setdefault(evidence_id, []).append(
                    unit["evidence_node_id"]
                )
        catalog_ids_by_node: dict[str, list[str]] = {}
        catalog_member_nodes: dict[str, list[str]] = {}
        for catalog in self.reference_catalogs:
            members = list(
                dict.fromkeys(
                    node_id
                    for evidence_id in catalog.get("evidence_ids", [])
                    for node_id in evidence_to_nodes.get(evidence_id, [])
                )
            )
            catalog_member_nodes[catalog["catalog_id"]] = members
            for node_id in members:
                catalog_ids_by_node.setdefault(node_id, []).append(catalog["catalog_id"])
        node_to_tree: dict[str, str] = {}
        parent_by_node_id: dict[str, str | None] = {}
        title_paths: dict[str, list[str]] = {}
        sources: list[dict[str, Any]] = []
        for tree in self.corpus.evidence_trees:
            nodes = {item.evidence_node_id: item for item in tree.nodes}

            def title_path(node_id: str) -> list[str]:
                cached = title_paths.get(node_id)
                if cached is not None:
                    return cached
                path: list[str] = []
                seen: set[str] = set()
                current = nodes.get(node_id)
                while current is not None and current.evidence_node_id not in seen:
                    seen.add(current.evidence_node_id)
                    path.append(current.title)
                    current = nodes.get(current.parent_node_id or "")
                value = list(reversed(path))
                title_paths[node_id] = value
                return value

            descriptors: list[dict[str, Any]] = []
            for node in tree.nodes:
                node_to_tree[node.evidence_node_id] = tree.evidence_tree_id
                parent_by_node_id[node.evidence_node_id] = node.parent_node_id
                base = unit_by_node.get(node.evidence_node_id)
                profiles = [
                    evidence_profiles[evidence_id]
                    for evidence_id in (base or {}).get("evidence_ids", [])
                    if evidence_id in evidence_profiles
                ]
                score = max((int(item["score"]) for item in profiles), default=0)
                signal_tags = list(
                    dict.fromkeys(
                        tag for item in profiles for tag in item.get("tags", [])
                    )
                )
                signal_reasons = list(
                    dict.fromkeys(
                        reason
                        for item in profiles
                        for reason in item.get("reasons", [])
                    )
                )
                inline_entry_count = sum(
                    int(item.get("inline_entry_count", 0)) for item in profiles
                )
                descriptors.append(
                    {
                        "node_id": node.evidence_node_id,
                        "title": node.title,
                        "node_type": node.node_type,
                        "parent_node_id": node.parent_node_id,
                        "child_node_ids": list(node.child_node_ids),
                        "ordinal": node.ordinal,
                        "title_path": title_path(node.evidence_node_id),
                        "has_direct_source": base is not None,
                        "source_part_count": int(base["source_part_count"]) if base else 0,
                        "direct_char_count": node.direct_char_count,
                        "descendant_char_count": node.descendant_char_count,
                        "reference_signal": {
                            "score": score,
                            "tags": signal_tags,
                            "reasons": signal_reasons[:6],
                            "inline_entry_count": inline_entry_count,
                            "catalog_ids": catalog_ids_by_node.get(
                                node.evidence_node_id, []
                            ),
                            "python_candidate_boundary": bool(
                                score > 0
                                or catalog_ids_by_node.get(node.evidence_node_id)
                            ),
                            "python_recommendation": (
                                "preserve_catalog_or_read"
                                if catalog_ids_by_node.get(node.evidence_node_id)
                                else "read"
                                if score >= 2
                                else "read_if_semantically_uncertain"
                            ),
                        },
                    }
                )
            sources.append(
                {
                    "source_id": tree.source_id,
                    "source_name": source_names.get(tree.source_id, tree.source_id),
                    "evidence_tree_id": tree.evidence_tree_id,
                    "root_node_id": tree.root_node_id,
                    "nodes": descriptors,
                }
            )

        catalog_hints: list[dict[str, Any]] = []

        def common_ancestor(member_node_ids: list[str]) -> str | None:
            if not member_node_ids:
                return None
            if len({node_to_tree.get(item) for item in member_node_ids}) != 1:
                return None

            def chain(node_id: str) -> list[str]:
                result: list[str] = []
                seen: set[str] = set()
                current: str | None = node_id
                while current and current not in seen:
                    seen.add(current)
                    result.append(current)
                    current = parent_by_node_id.get(current)
                return result

            shared = set(chain(member_node_ids[0]))
            for node_id in member_node_ids[1:]:
                shared.intersection_update(chain(node_id))
            return next(
                (item for item in chain(member_node_ids[0]) if item in shared), None
            )

        for catalog in self.reference_catalogs:
            member_node_ids = catalog_member_nodes[catalog["catalog_id"]]
            common_root_id = common_ancestor(member_node_ids)
            catalog_hints.append(
                {
                    "catalog_id": catalog["catalog_id"],
                    "source_id": catalog["source_id"],
                    "structure": catalog["structure"],
                    "detail": catalog["detail"],
                    "entry_count": catalog["entry_count"],
                    "member_node_count": len(member_node_ids),
                    "member_node_id_samples": member_node_ids[:8],
                    "common_ancestor_node_id": common_root_id,
                    "common_ancestor_title_path": title_paths.get(
                        common_root_id or "", []
                    ),
                    "common_field_labels": list(
                        catalog.get("common_field_labels", [])
                    ),
                    "sample_titles": [
                        item.get("title", "")
                        for item in catalog.get("sample_entries", [])
                    ],
                }
            )

        return {
            "target_profile": self.target_profile.model_dump(mode="json"),
            "corpus_snapshot_id": self.corpus.corpus_snapshot_id,
            "base_source_bearing_node_count": len(self.reference_base_units),
            "sources": sources,
            "catalog_hints": catalog_hints,
            "required_partition": {
                "member_domain": "nodes where has_direct_source is true",
                "coverage": "every member-domain node must resolve exactly once to read, preserved_catalog, or skip_for_reference",
                "source_boundary": "one source and one EvidenceTree per unit",
                "selectors": "use node, all_source_bearing_descendants, or all_non_catalog_source_bearing_descendants; Python expands and validates them",
                "uncertainty_policy": "when tree semantics or Python signals leave doubt, choose read rather than skip",
                "content_policy": "planning changes reading context only; no SourcePart is merged, rewritten, or omitted",
            },
        }

    def _reference_candidate_descriptors(
        self, *, min_reference_like_score: int = 1
    ) -> list[dict[str, Any]]:
        """Build a compact, high-recall candidate inventory for Agent review.

        The inventory uses the EvidenceTree only as a semantic grouping surface.
        It does not merge SourceParts and it does not make the final keep/skip
        decision. Detected catalogs are always included; other nodes are included
        when at least one directly owned Evidence excerpt has a matching signal.
        """

        threshold = max(0, int(min_reference_like_score))
        catalog_ids_by_evidence: dict[str, list[str]] = {}
        structural_catalog_evidence_ids: set[str] = set()
        for catalog in self.reference_catalogs:
            for evidence_id in catalog.get("evidence_ids", []):
                catalog_ids_by_evidence.setdefault(evidence_id, []).append(
                    str(catalog["catalog_id"])
                )
                # A catalog made from many separate Evidence rows is represented
                # once below. Inline catalogs retain their surrounding Evidence
                # node because the source section can contain reusable context
                # outside the extracted entries.
                if not catalog.get("catalog_entries"):
                    structural_catalog_evidence_ids.add(str(evidence_id))

        descriptors: list[dict[str, Any]] = []
        for catalog_order, catalog in enumerate(self.reference_catalogs):
            catalog_evidence = [
                self.evidence_by_id[evidence_id]
                for evidence_id in catalog.get("evidence_ids", [])
                if evidence_id in self.evidence_by_id
            ]
            catalog_profiles = [
                _reference_like_evidence_profile(item) for item in catalog_evidence
            ]
            descriptors.append(
                {
                    "candidate_id": catalog["catalog_id"],
                    "candidate_kind": "catalog_candidate",
                    "source_id": catalog["source_id"],
                    "source_name": next(
                        (
                            item["source_name"]
                            for item in self.reference_base_units
                            if item["source_id"] == catalog["source_id"]
                        ),
                        catalog["source_id"],
                    ),
                    "evidence_tree_id": None,
                    "evidence_node_id": None,
                    "title": catalog.get("suggested_title") or catalog["detail"],
                    "title_path": [],
                    "source_part_count": len(catalog_evidence),
                    "evidence_count": len(catalog_evidence),
                    "evidence_id_samples": list(catalog.get("evidence_ids", [])[:8]),
                    "reference_like_score": max(
                        (int(item["score"]) for item in catalog_profiles), default=1
                    ),
                    "reference_like_tags": list(
                        dict.fromkeys(
                            ["source_catalog"]
                            + [
                                tag
                                for profile in catalog_profiles
                                for tag in profile.get("tags", [])
                            ]
                        )
                    ),
                    "reference_like_reasons": [
                        f"detected_catalog:{catalog['structure']}",
                        catalog["detail"],
                    ],
                    "inline_entry_count": len(catalog.get("catalog_entries") or []),
                    "catalog_ids": [catalog["catalog_id"]],
                    "samples": [
                        {
                            "evidence_id": item.evidence_id,
                            "title": item.title,
                            "char_count": len(item.content),
                            "preview": _excerpt_quote(item.content, maximum=240),
                        }
                        for item in catalog_evidence[:3]
                    ],
                    "source_order": catalog_order,
                }
            )
        for source_order, unit in enumerate(self.reference_base_units):
            evidence = [
                self.evidence_by_id[evidence_id]
                for evidence_id in unit["evidence_ids"]
                if evidence_id in self.evidence_by_id
                and evidence_id not in structural_catalog_evidence_ids
            ]
            if not evidence:
                continue
            profiles = [_reference_like_evidence_profile(item) for item in evidence]
            score = max((int(item["score"]) for item in profiles), default=0)
            tree_text = _compact(" ".join(str(item) for item in unit["title_path"])).casefold()
            tree_tags: list[str] = []
            tree_reasons: list[str] = []
            tree_score = 0
            for tag, markers in _REFERENCE_LIKE_TITLE_MARKERS:
                hits = [marker for marker in markers if marker in tree_text]
                if hits:
                    tree_tags.append(tag)
                    tree_score += 2 if tag != "concept_reference" else 1
                    tree_reasons.append(f"evidence_tree_title:{hits[0]}")
            score = max(score, tree_score)
            catalog_ids = list(
                dict.fromkeys(
                    catalog_id
                    for evidence_id in (item.evidence_id for item in evidence)
                    for catalog_id in catalog_ids_by_evidence.get(evidence_id, [])
                )
            )
            if score < threshold and not catalog_ids:
                continue
            tags = list(
                dict.fromkeys(
                    [*tree_tags]
                    + [
                        tag
                        for profile in profiles
                        for tag in profile.get("tags", [])
                    ]
                )
            )
            reasons = list(
                dict.fromkeys(
                    [*tree_reasons]
                    + [
                        reason
                        for profile in profiles
                        for reason in profile.get("reasons", [])
                    ]
                )
            )
            inline_entry_count = sum(
                int(profile.get("inline_entry_count", 0)) for profile in profiles
            )
            combined_content = "\n\n".join(item.content for item in evidence)
            representative_excerpt = _representative_reference_excerpt(
                combined_content,
                maximum=REFERENCE_ORDINARY_CONTEXT_CHARS,
            )
            descriptors.append(
                {
                    "candidate_id": unit["evidence_node_id"],
                    "candidate_kind": (
                        "collection_candidate"
                        if len(evidence) > 1 or inline_entry_count > 1
                        else "summary_candidate"
                    ),
                    "source_id": unit["source_id"],
                    "source_name": unit["source_name"],
                    "evidence_tree_id": unit["evidence_tree_id"],
                    "evidence_node_id": unit["evidence_node_id"],
                    "title": unit["title"],
                    "title_path": list(unit["title_path"]),
                    "source_part_count": unit["source_part_count"],
                    "evidence_count": len(evidence),
                    "evidence_id_samples": [item.evidence_id for item in evidence[:8]],
                    "reference_like_score": score,
                    "reference_like_tags": tags,
                    "reference_like_reasons": reasons[:6],
                    "inline_entry_count": inline_entry_count,
                    "catalog_ids": catalog_ids,
                    "candidate_char_count": len(combined_content),
                    "representative_excerpt": representative_excerpt,
                    "representative_excerpt_char_count": len(representative_excerpt),
                    "representative_excerpt_limit": REFERENCE_ORDINARY_CONTEXT_CHARS,
                    "representative_excerpt_strategy": (
                        "full_content"
                        if len(combined_content.strip()) <= REFERENCE_ORDINARY_CONTEXT_CHARS
                        else "beginning_middle_end_and_reference_signals"
                    ),
                    "source_order": source_order,
                }
            )
        descriptors.sort(
            key=lambda item: (
                0 if item["catalog_ids"] else 1,
                -int(item["reference_like_score"]),
                int(item["source_order"]),
            )
        )
        return descriptors

    def reference_candidate_summary(self) -> dict[str, Any]:
        candidates = self._reference_candidate_descriptors()
        return {
            "candidate_count": len(candidates),
            "catalog_candidate_count": sum(
                item["candidate_kind"] == "catalog_candidate" for item in candidates
            ),
            "collection_candidate_count": sum(
                item["candidate_kind"] == "collection_candidate" for item in candidates
            ),
            "summary_candidate_count": sum(
                item["candidate_kind"] == "summary_candidate" for item in candidates
            ),
            "detected_catalog_count": len(self.reference_catalogs),
            "source_bearing_node_count": len(self.reference_base_units),
            "policy": "Every prepared catalog and ordinary candidate requires an explicit Agent decision",
        }

    def list_reference_candidates(
        self,
        page: int = 1,
        page_size: int = 40,
        min_reference_like_score: int = 1,
        include_catalog_candidates: bool = True,
    ) -> dict[str, Any]:
        if self.stage != "reference_build":
            raise PermissionError("Reference candidates are only available in reference_build")
        page = max(1, int(page))
        page_size = max(1, min(20, int(page_size)))
        candidates = self._reference_candidate_descriptors(
            min_reference_like_score=min_reference_like_score
        )
        if not include_catalog_candidates:
            candidates = [
                item for item in candidates if item["candidate_kind"] != "catalog_candidate"
            ]
        start = (page - 1) * page_size
        return {
            "page": page,
            "page_size": page_size,
            "total": len(candidates),
            "items": [
                {key: value for key, value in item.items() if key != "samples"}
                for item in candidates[start : start + page_size]
            ],
            "truncated": start + page_size < len(candidates),
            "next_page": page + 1 if start + page_size < len(candidates) else None,
            "min_reference_like_score": max(0, int(min_reference_like_score)),
            "include_catalog_candidates": bool(include_catalog_candidates),
        }

    def inspect_reference_candidates(
        self, candidate_ids: list[str], limit: int = 8
    ) -> dict[str, Any]:
        if self.stage != "reference_build":
            raise PermissionError("Reference candidates are only available in reference_build")
        unique_ids, failures = self._normalized_id_batch(candidate_ids, "candidate_ids", 8)
        failures += [
            f"{candidate_id} (unknown Reference candidate ID)"
            for candidate_id in unique_ids
            if candidate_id not in self.reference_base_units_by_node_id
        ]
        if failures:
            raise ValueError(
                "Whole batch rejected; invalid candidate_ids: " + "; ".join(failures)
            )
        limit = max(1, min(8, int(limit)))
        return {
            "candidates": [
                self.inspect_reference_candidate(candidate_id, 0, limit)
                for candidate_id in unique_ids
            ],
            "limit": limit,
        }

    @staticmethod
    def _normalized_id_batch(
        raw_ids: Any, field: str, maximum: int
    ) -> tuple[list[str], list[str]]:
        if not isinstance(raw_ids, list):
            raise ValueError(f"{field} must be an array of 1 to {maximum} IDs")
        unique_ids: list[str] = []
        failures: list[str] = []
        for item in raw_ids:
            value = item.strip() if isinstance(item, str) else ""
            if not value:
                failures.append(f"{item!r} (empty or non-string ID)")
            elif value not in unique_ids:
                unique_ids.append(value)
        if not failures and (not unique_ids or len(unique_ids) > maximum):
            raise ValueError(f"{field} must contain 1 to {maximum} IDs")
        return unique_ids, failures

    def inspect_reference_candidate(
        self, candidate_id: str, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        if self.stage != "reference_build":
            raise PermissionError("Reference candidates are only available in reference_build")
        unit = self.reference_base_units_by_node_id.get(candidate_id)
        if not unit:
            raise ValueError(f"Unknown Reference candidate ID: {candidate_id}")
        offset = max(0, int(offset))
        limit = max(1, min(8, int(limit)))
        evidence_ids = self._ordinary_reference_evidence_ids(unit)
        selected = evidence_ids[offset : offset + limit]
        return {
            "candidate_id": candidate_id,
            "title": unit["title"],
            "title_path": list(unit["title_path"]),
            "source_id": unit["source_id"],
            "source_part_count": unit["source_part_count"],
            "evidence_count": len(evidence_ids),
            "offset": offset,
            "next_offset": min(len(evidence_ids), offset + limit),
            "truncated": offset + limit < len(evidence_ids),
            "evidence": [self._evidence_descriptor(self.evidence_by_id[item]) for item in selected],
        }

    def _submit_reference_reading_plan_legacy(
        self, units: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Validate and materialize the Reference Child's exhaustive partition."""

        if self.stage != "reference_build":
            raise PermissionError(
                "Reference reading plans are only allowed in reference_build"
            )
        if self.reference_reading_plan is not None:
            raise ValueError("The Reference reading plan has already been accepted")
        if not isinstance(units, list) or not units:
            raise ValueError("The Reference reading plan requires at least one unit")

        nodes_by_id: dict[str, Any] = {}
        tree_by_node_id: dict[str, str] = {}
        source_by_node_id: dict[str, str] = {}
        title_path_by_node_id: dict[str, list[str]] = {}
        for tree in self.corpus.evidence_trees:
            tree_nodes = {item.evidence_node_id: item for item in tree.nodes}
            for node in tree.nodes:
                nodes_by_id[node.evidence_node_id] = node
                tree_by_node_id[node.evidence_node_id] = tree.evidence_tree_id
                source_by_node_id[node.evidence_node_id] = tree.source_id
                path: list[str] = []
                seen: set[str] = set()
                current = node
                while current is not None and current.evidence_node_id not in seen:
                    seen.add(current.evidence_node_id)
                    path.append(current.title)
                    current = tree_nodes.get(current.parent_node_id or "")
                title_path_by_node_id[node.evidence_node_id] = list(reversed(path))

        expected_node_ids = set(self.reference_base_units_by_node_id)
        assigned_node_ids: set[str] = set()
        errors: list[str] = []
        normalized: list[dict[str, Any]] = []
        allowed_kinds = {"single_node", "semantic_subtree", "catalog_subtree"}

        def is_ancestor(root_node_id: str, member_node_id: str) -> bool:
            seen: set[str] = set()
            current = nodes_by_id.get(member_node_id)
            while current is not None and current.evidence_node_id not in seen:
                if current.evidence_node_id == root_node_id:
                    return True
                seen.add(current.evidence_node_id)
                current = nodes_by_id.get(current.parent_node_id or "")
            return False

        for index, raw in enumerate(units):
            if not isinstance(raw, dict):
                errors.append(f"Unit {index} must be an object")
                continue
            title = _compact(str(raw.get("title") or ""))
            unit_kind = _compact(str(raw.get("unit_kind") or ""))
            root_node_id = str(raw.get("root_node_id") or "").strip()
            rationale = _compact(str(raw.get("rationale") or ""))
            member_node_ids = list(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw.get("member_node_ids") or []
                    if str(item).strip()
                )
            )
            if len(title) < 3:
                errors.append(f"Unit {index} requires a concrete title")
            if unit_kind not in allowed_kinds:
                errors.append(
                    f"Unit {index} has invalid unit_kind {unit_kind or '<empty>'}"
                )
            if root_node_id not in nodes_by_id:
                errors.append(
                    f"Unit {index} has unknown root_node_id {root_node_id or '<empty>'}"
                )
            if len(rationale) < 3:
                errors.append(f"Unit {index} requires a concrete rationale")
            if not member_node_ids:
                errors.append(f"Unit {index} has no member_node_ids")
                continue
            unknown = [
                item for item in member_node_ids if item not in expected_node_ids
            ]
            if unknown:
                errors.append(
                    f"Unit {index} contains unknown or source-free member nodes: "
                    + ", ".join(unknown[:10])
                )
                continue
            duplicates = [
                item for item in member_node_ids if item in assigned_node_ids
            ]
            if duplicates:
                errors.append(
                    f"Unit {index} repeats nodes assigned by an earlier unit: "
                    + ", ".join(duplicates[:10])
                )
            assigned_node_ids.update(member_node_ids)
            if root_node_id in nodes_by_id:
                tree_ids = {tree_by_node_id[item] for item in member_node_ids}
                source_ids = {source_by_node_id[item] for item in member_node_ids}
                if len(tree_ids) != 1 or len(source_ids) != 1:
                    errors.append(
                        f"Unit {index} crosses a Source or EvidenceTree boundary"
                    )
                elif (
                    tree_by_node_id[root_node_id] not in tree_ids
                    or source_by_node_id[root_node_id] not in source_ids
                ):
                    errors.append(
                        f"Unit {index} root_node_id is outside the members' tree"
                    )
                not_descendants = [
                    item
                    for item in member_node_ids
                    if not is_ancestor(root_node_id, item)
                ]
                if not_descendants:
                    errors.append(
                        f"Unit {index} has members outside root_node_id's subtree: "
                        + ", ".join(not_descendants[:10])
                    )
            if unit_kind == "single_node" and (
                len(member_node_ids) != 1 or member_node_ids[0] != root_node_id
            ):
                errors.append(
                    f"Unit {index} single_node must contain only its root_node_id"
                )
            normalized.append(
                {
                    "title": title[:300],
                    "unit_kind": unit_kind,
                    "root_node_id": root_node_id,
                    "member_node_ids": member_node_ids,
                    "rationale": rationale[:600],
                }
            )

        missing = sorted(expected_node_ids - assigned_node_ids)
        extra = sorted(assigned_node_ids - expected_node_ids)
        if missing:
            errors.append(
                "Plan omits source-bearing nodes: "
                + ", ".join(missing[:20])
                + (f" (+{len(missing) - 20} more)" if len(missing) > 20 else "")
            )
        if extra:
            errors.append(
                "Plan includes invalid nodes: " + ", ".join(extra[:20])
            )
        if errors:
            raise ValueError("; ".join(errors))

        base_order = {
            item["evidence_node_id"]: index
            for index, item in enumerate(self.reference_base_units)
        }
        planned_units: list[dict[str, Any]] = []
        for planned in normalized:
            member_ids = sorted(
                planned["member_node_ids"], key=lambda item: base_order[item]
            )
            members = [self.reference_base_units_by_node_id[item] for item in member_ids]
            root = nodes_by_id[planned["root_node_id"]]
            evidence_ids = list(
                dict.fromkeys(
                    evidence_id
                    for member in members
                    for evidence_id in member["evidence_ids"]
                )
            )
            source_parts: list[dict[str, Any]] = []
            seen_spans: set[tuple[str, int, int]] = set()
            for member in members:
                for source_part in member["source_parts"]:
                    key = (
                        str(source_part["source_part_id"]),
                        int(source_part["start_offset"]),
                        int(source_part["end_offset"]),
                    )
                    if key in seen_spans:
                        continue
                    seen_spans.add(key)
                    source_parts.append(source_part)
            unit_id = self._id(
                "U", "rru",
                self.corpus.corpus_snapshot_id,
                planned["root_node_id"],
                *member_ids,
            )
            planned_units.append(
                {
                    "unit_id": unit_id,
                    "source_id": source_by_node_id[member_ids[0]],
                    "source_name": members[0]["source_name"],
                    "evidence_tree_id": tree_by_node_id[member_ids[0]],
                    "evidence_node_id": planned["root_node_id"],
                    "root_node_id": planned["root_node_id"],
                    "node_key": root.node_key,
                    "node_type": root.node_type,
                    "title": planned["title"],
                    "title_path": title_path_by_node_id[planned["root_node_id"]],
                    "unit_kind": planned["unit_kind"],
                    "planning_rationale": planned["rationale"],
                    "member_node_count": len(member_ids),
                    "member_node_ids": member_ids,
                    "member_nodes": [
                        {
                            "node_id": member["evidence_node_id"],
                            "title": member["title"],
                            "title_path": member["title_path"],
                            "node_type": member["node_type"],
                            "source_part_count": member["source_part_count"],
                            "direct_char_count": member["direct_char_count"],
                        }
                        for member in members
                    ],
                    "is_leaf": len(member_ids) == 1 and bool(members[0]["is_leaf"]),
                    "child_count": len(root.child_node_ids),
                    "direct_char_count": sum(
                        int(member["direct_char_count"]) for member in members
                    ),
                    "source_part_count": len(source_parts),
                    "source_parts": source_parts,
                    "evidence_ids": evidence_ids,
                    "first_base_ordinal": min(base_order[item] for item in member_ids),
                }
            )
        planned_units.sort(key=lambda item: int(item["first_base_ordinal"]))
        for unit in planned_units:
            unit.pop("first_base_ordinal", None)

        plan = {
            "schema_version": "reference-reading-plan.v1",
            "corpus_snapshot_id": self.corpus.corpus_snapshot_id,
            "base_node_count": len(self.reference_base_units),
            "reading_unit_count": len(planned_units),
            "units": normalized,
            "dispatch_units": [
                {
                    "unit_id": item["unit_id"],
                    "source_id": item["source_id"],
                    "evidence_tree_id": item["evidence_tree_id"],
                    "title": item["title"],
                    "unit_kind": item["unit_kind"],
                    "root_node_id": item["root_node_id"],
                    "member_node_ids": list(item["member_node_ids"]),
                    "source_part_count": item["source_part_count"],
                    "evidence_count": len(item["evidence_ids"]),
                }
                for item in planned_units
            ],
            "created_at": utc_now(),
        }
        artifact = self.store.put_json_artifact(
            plan,
            kind="reference_reading_plan",
            metadata={"stage": "reference_build"},
            preserve_text_integrity=True,
        )
        self.reference_reading_plan = plan
        self.reference_reading_plan_artifact_id = artifact.artifact_id
        self.reference_units = planned_units
        self.reference_units_by_id = {
            item["unit_id"]: item for item in self.reference_units
        }
        return {
            "accepted": True,
            "artifact_id": artifact.artifact_id,
            "base_node_count": len(self.reference_base_units),
            "reading_unit_count": len(self.reference_units),
            "reduction_count": len(self.reference_base_units)
            - len(self.reference_units),
            "units": [
                {
                    "unit_id": item["unit_id"],
                    "title": item["title"],
                    "unit_kind": item["unit_kind"],
                    "member_node_count": item["member_node_count"],
                    "source_part_count": item["source_part_count"],
                }
                for item in self.reference_units
            ],
        }

    def submit_reference_reading_plan(
        self,
        reading_units: list[dict[str, Any]],
        catalog_decisions: list[dict[str, Any]],
        skip_decisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Accept an exhaustive disposition plan while reading only candidates.

        The Child selects compact tree/catalog selectors.  Python expands them,
        proves that every source-bearing node has one and only one disposition,
        and directly preserves accepted detector catalogs from immutable source
        data.  Only ``read`` dispositions become later model turns.
        """

        if self.stage != "reference_build":
            raise PermissionError(
                "Reference reading plans are only allowed in reference_build"
            )
        if self.reference_reading_plan is not None:
            raise ValueError("The Reference reading plan has already been accepted")
        if not all(
            isinstance(value, list)
            for value in (reading_units, catalog_decisions, skip_decisions)
        ):
            raise ValueError("reading_units, catalog_decisions, and skip_decisions must be arrays")

        nodes_by_id: dict[str, Any] = {}
        tree_by_node_id: dict[str, str] = {}
        source_by_node_id: dict[str, str] = {}
        title_path_by_node_id: dict[str, list[str]] = {}
        for tree in self.corpus.evidence_trees:
            tree_nodes = {item.evidence_node_id: item for item in tree.nodes}
            for node in tree.nodes:
                nodes_by_id[node.evidence_node_id] = node
                tree_by_node_id[node.evidence_node_id] = tree.evidence_tree_id
                source_by_node_id[node.evidence_node_id] = tree.source_id
                path: list[str] = []
                seen: set[str] = set()
                current = node
                while current is not None and current.evidence_node_id not in seen:
                    seen.add(current.evidence_node_id)
                    path.append(current.title)
                    current = tree_nodes.get(current.parent_node_id or "")
                title_path_by_node_id[node.evidence_node_id] = list(reversed(path))

        expected_node_ids = set(self.reference_base_units_by_node_id)
        base_order = {
            item["evidence_node_id"]: index
            for index, item in enumerate(self.reference_base_units)
        }

        def is_ancestor(root_node_id: str, member_node_id: str) -> bool:
            seen: set[str] = set()
            current = nodes_by_id.get(member_node_id)
            while current is not None and current.evidence_node_id not in seen:
                if current.evidence_node_id == root_node_id:
                    return True
                seen.add(current.evidence_node_id)
                current = nodes_by_id.get(current.parent_node_id or "")
            return False

        def expand_selector(
            root_node_id: str, selection: str, label: str, errors: list[str]
        ) -> list[str]:
            if root_node_id not in nodes_by_id:
                errors.append(f"{label} has unknown root_node_id {root_node_id or '<empty>'}")
                return []
            if selection == "node":
                if root_node_id not in expected_node_ids:
                    errors.append(f"{label} selects a source-free node")
                    return []
                return [root_node_id]
            if selection not in {
                "all_source_bearing_descendants",
                "all_non_catalog_source_bearing_descendants",
            }:
                errors.append(f"{label} has invalid selection {selection or '<empty>'}")
                return []
            members = [
                node_id
                for node_id in expected_node_ids
                if is_ancestor(root_node_id, node_id)
            ]
            if selection == "all_non_catalog_source_bearing_descendants":
                members = [
                    node_id
                    for node_id in members
                    if assignments.get(node_id, {}).get("kind")
                    != "preserved_catalog"
                ]
            members.sort(key=lambda item: base_order[item])
            if not members:
                errors.append(f"{label} selector contains no source-bearing descendants")
            return members

        evidence_to_nodes: dict[str, list[str]] = {}
        for unit in self.reference_base_units:
            for evidence_id in unit["evidence_ids"]:
                evidence_to_nodes.setdefault(evidence_id, []).append(
                    unit["evidence_node_id"]
                )
        catalog_members = {
            catalog["catalog_id"]: list(
                dict.fromkeys(
                    node_id
                    for evidence_id in catalog.get("evidence_ids", [])
                    for node_id in evidence_to_nodes.get(evidence_id, [])
                )
            )
            for catalog in self.reference_catalogs
        }
        catalog_member_node_ids = {
            node_id for members in catalog_members.values() for node_id in members
        }
        node_reference_scores = {
            unit["evidence_node_id"]: max(
                (
                    int(_reference_like_evidence_profile(self.evidence_by_id[evidence_id])["score"])
                    for evidence_id in unit["evidence_ids"]
                ),
                default=0,
            )
            for unit in self.reference_base_units
        }

        errors: list[str] = []
        normalized_catalogs: list[dict[str, Any]] = []
        seen_catalog_ids: set[str] = set()
        for index, raw in enumerate(catalog_decisions):
            if not isinstance(raw, dict):
                errors.append(f"Catalog decision {index} must be an object")
                continue
            catalog_id = str(raw.get("catalog_id") or "").strip()
            disposition = str(raw.get("disposition") or "").strip()
            rationale = _compact(str(raw.get("rationale") or ""))
            if catalog_id not in self.reference_catalogs_by_id:
                errors.append(f"Catalog decision {index} has unknown catalog_id {catalog_id or '<empty>'}")
                continue
            if catalog_id in seen_catalog_ids:
                errors.append(f"Catalog {catalog_id} has more than one decision")
                continue
            seen_catalog_ids.add(catalog_id)
            if disposition not in {"preserve", "exclude"}:
                errors.append(f"Catalog {catalog_id} has invalid disposition {disposition or '<empty>'}")
                continue
            if len(rationale) < 3:
                errors.append(f"Catalog {catalog_id} requires a concrete rationale")
            normalized = {
                "catalog_id": catalog_id,
                "disposition": disposition,
                "cover_member_nodes": bool(raw.get("cover_member_nodes", False)),
                "rationale": rationale[:600],
            }
            if disposition == "preserve":
                normalized.update(
                    {
                        "title": _compact(str(raw.get("title") or ""))[:300],
                        "description": _compact(str(raw.get("description") or "")),
                        "category": _compact(str(raw.get("category") or "source_catalog"))[:120],
                        "injection_policy": _compact(str(raw.get("injection_policy") or "on_demand")),
                        "usage_tags": list(
                            dict.fromkeys(
                                str(item).strip()
                                for item in raw.get("usage_tags") or []
                                if str(item).strip()
                            )
                        )[:16],
                    }
                )
                if len(normalized["title"]) < 3:
                    errors.append(f"Preserved catalog {catalog_id} requires a concrete title")
                errors.extend(
                    f"Catalog {catalog_id}: {message}"
                    for message in catalog_description_errors(normalized["description"])
                )
                if normalized["injection_policy"] not in {"always", "on_demand", "contextual"}:
                    errors.append(f"Catalog {catalog_id} has invalid injection_policy")
            elif normalized["cover_member_nodes"]:
                errors.append(
                    f"Excluded catalog {catalog_id} cannot cover member nodes; read or skip them explicitly"
                )
            normalized_catalogs.append(normalized)

        missing_catalogs = sorted(set(self.reference_catalogs_by_id) - seen_catalog_ids)
        if missing_catalogs:
            errors.append(
                "Plan omits catalog decisions: " + ", ".join(missing_catalogs[:20])
            )
        preserved_count = sum(
            item["disposition"] == "preserve"
            and item["catalog_id"] not in self.submitted_catalog_ids
            for item in normalized_catalogs
        )
        remaining_capacity = max(
            0,
            self.settings.max_knowledge_candidates
            - self.initial_reference_count
            - len(self.submitted_references),
        )
        if preserved_count > remaining_capacity:
            errors.append(
                f"Plan preserves {preserved_count} catalogs but only {remaining_capacity} Reference slots remain"
            )

        assignments: dict[str, dict[str, Any]] = {}

        def assign(node_ids: list[str], disposition: dict[str, Any], label: str) -> None:
            for node_id in node_ids:
                prior = assignments.get(node_id)
                if prior is not None:
                    errors.append(
                        f"Node {node_id} is assigned more than once ({prior['kind']} and {disposition['kind']})"
                    )
                else:
                    assignments[node_id] = disposition

        for item in normalized_catalogs:
            if item["disposition"] == "preserve" and item["cover_member_nodes"]:
                assign(
                    catalog_members.get(item["catalog_id"], []),
                    {"kind": "preserved_catalog", "catalog_id": item["catalog_id"]},
                    f"Catalog {item['catalog_id']}",
                )

        normalized_reads: list[dict[str, Any]] = []
        for index, raw in enumerate(reading_units):
            if not isinstance(raw, dict):
                errors.append(f"Reading unit {index} must be an object")
                continue
            title = _compact(str(raw.get("title") or ""))
            unit_kind = _compact(str(raw.get("unit_kind") or ""))
            root_node_id = str(raw.get("root_node_id") or "").strip()
            selection = str(raw.get("selection") or "").strip()
            rationale = _compact(str(raw.get("rationale") or ""))
            if len(title) < 3:
                errors.append(f"Reading unit {index} requires a concrete title")
            if unit_kind not in {"single_node", "semantic_subtree"}:
                errors.append(f"Reading unit {index} has invalid unit_kind {unit_kind or '<empty>'}")
            if len(rationale) < 3:
                errors.append(f"Reading unit {index} requires a concrete rationale")
            members = expand_selector(
                root_node_id, selection, f"Reading unit {index}", errors
            )
            if unit_kind == "single_node" and (
                selection != "node" or members != [root_node_id]
            ):
                errors.append(f"Reading unit {index} single_node must use selection=node")
            normalized = {
                "title": title[:300],
                "unit_kind": unit_kind,
                "root_node_id": root_node_id,
                "selection": selection,
                "member_node_ids": members,
                "rationale": rationale[:600],
            }
            normalized_reads.append(normalized)
            assign(members, {"kind": "read", "reading_index": index}, f"Reading unit {index}")

        normalized_skips: list[dict[str, Any]] = []
        for index, raw in enumerate(skip_decisions):
            if not isinstance(raw, dict):
                errors.append(f"Skip decision {index} must be an object")
                continue
            root_node_id = str(raw.get("root_node_id") or "").strip()
            selection = str(raw.get("selection") or "").strip()
            rationale = _compact(str(raw.get("rationale") or ""))
            if len(rationale) < 12:
                errors.append(
                    f"Skip decision {index} requires a specific rationale of at least 12 characters"
                )
            members = expand_selector(
                root_node_id, selection, f"Skip decision {index}", errors
            )
            protected = [
                node_id
                for node_id in members
                if node_reference_scores.get(node_id, 0) > 0
                or node_id in catalog_member_node_ids
            ]
            if protected:
                errors.append(
                    f"Skip decision {index} crosses Python Reference-candidate boundary; "
                    "read or preserve these nodes: " + ", ".join(protected[:12])
                )
            normalized = {
                "root_node_id": root_node_id,
                "selection": selection,
                "member_node_ids": members,
                "rationale": rationale[:600],
            }
            normalized_skips.append(normalized)
            assign(members, {"kind": "skip_for_reference", "skip_index": index}, f"Skip decision {index}")

        missing_nodes = sorted(expected_node_ids - set(assignments), key=base_order.get)
        if missing_nodes:
            errors.append(
                "Plan omits source-bearing nodes: "
                + ", ".join(missing_nodes[:20])
                + (f" (+{len(missing_nodes) - 20} more)" if len(missing_nodes) > 20 else "")
            )
        if errors:
            raise ValueError("; ".join(errors))

        planned_units: list[dict[str, Any]] = []
        for planned in normalized_reads:
            member_ids = planned["member_node_ids"]
            members = [self.reference_base_units_by_node_id[item] for item in member_ids]
            root = nodes_by_id[planned["root_node_id"]]
            evidence_ids = list(
                dict.fromkeys(
                    evidence_id for member in members for evidence_id in member["evidence_ids"]
                )
            )
            source_parts: list[dict[str, Any]] = []
            seen_spans: set[tuple[str, int, int]] = set()
            for member in members:
                for source_part in member["source_parts"]:
                    key = (
                        str(source_part["source_part_id"]),
                        int(source_part["start_offset"]),
                        int(source_part["end_offset"]),
                    )
                    if key not in seen_spans:
                        seen_spans.add(key)
                        source_parts.append(source_part)
            unit_id = self._id(
                "U", "rru", self.corpus.corpus_snapshot_id, planned["root_node_id"], *member_ids
            )
            planned_units.append(
                {
                    "unit_id": unit_id,
                    "source_id": source_by_node_id[member_ids[0]],
                    "source_name": members[0]["source_name"],
                    "evidence_tree_id": tree_by_node_id[member_ids[0]],
                    "evidence_node_id": planned["root_node_id"],
                    "root_node_id": planned["root_node_id"],
                    "node_key": root.node_key,
                    "node_type": root.node_type,
                    "title": planned["title"],
                    "title_path": title_path_by_node_id[planned["root_node_id"]],
                    "unit_kind": planned["unit_kind"],
                    "selection": planned["selection"],
                    "planning_rationale": planned["rationale"],
                    "member_node_count": len(member_ids),
                    "member_node_ids": member_ids,
                    "member_nodes": [
                        {
                            "node_id": member["evidence_node_id"],
                            "title": member["title"],
                            "title_path": member["title_path"],
                            "node_type": member["node_type"],
                            "source_part_count": member["source_part_count"],
                            "direct_char_count": member["direct_char_count"],
                        }
                        for member in members
                    ],
                    "is_leaf": len(member_ids) == 1 and bool(members[0]["is_leaf"]),
                    "child_count": len(root.child_node_ids),
                    "direct_char_count": sum(int(member["direct_char_count"]) for member in members),
                    "source_part_count": len(source_parts),
                    "source_parts": source_parts,
                    "evidence_ids": evidence_ids,
                    "first_base_ordinal": min(base_order[item] for item in member_ids),
                }
            )
        planned_units.sort(key=lambda item: int(item["first_base_ordinal"]))
        for unit in planned_units:
            unit.pop("first_base_ordinal", None)

        materialized_catalogs: list[dict[str, Any]] = []
        for item in normalized_catalogs:
            catalog_id = item["catalog_id"]
            if item["disposition"] == "preserve":
                materialized_catalogs.append(
                    self.submit_reference_catalog(
                        catalog_id=catalog_id,
                        title=item["title"],
                        description=item["description"],
                        category=item["category"],
                        injection_policy=item["injection_policy"],
                        usage_tags=item["usage_tags"],
                    )
                )
                self.reference_catalog_review[catalog_id] = {
                    "disposition": "covered",
                    "rationale": item["rationale"],
                }
            else:
                self.reference_catalog_review[catalog_id] = {
                    "disposition": "excluded",
                    "rationale": item["rationale"],
                }

        plan = {
            "schema_version": "reference-reading-plan.v2",
            "corpus_snapshot_id": self.corpus.corpus_snapshot_id,
            "base_node_count": len(self.reference_base_units),
            "reading_unit_count": len(planned_units),
            "preserved_catalog_count": sum(item["disposition"] == "preserve" for item in normalized_catalogs),
            "skipped_node_count": sum(len(item["member_node_ids"]) for item in normalized_skips),
            "reading_units": normalized_reads,
            "catalog_decisions": normalized_catalogs,
            "skip_decisions": normalized_skips,
            "disposition_counts": {
                "read": sum(item["kind"] == "read" for item in assignments.values()),
                "preserved_catalog": sum(item["kind"] == "preserved_catalog" for item in assignments.values()),
                "skip_for_reference": sum(item["kind"] == "skip_for_reference" for item in assignments.values()),
            },
            "dispatch_units": [
                {
                    "unit_id": item["unit_id"],
                    "source_id": item["source_id"],
                    "evidence_tree_id": item["evidence_tree_id"],
                    "title": item["title"],
                    "unit_kind": item["unit_kind"],
                    "root_node_id": item["root_node_id"],
                    "selection": item["selection"],
                    "member_node_count": item["member_node_count"],
                    "source_part_count": item["source_part_count"],
                    "evidence_count": len(item["evidence_ids"]),
                }
                for item in planned_units
            ],
            "created_at": utc_now(),
        }
        artifact = self.store.put_json_artifact(
            plan,
            kind="reference_reading_plan",
            metadata={"stage": "reference_build"},
            preserve_text_integrity=True,
        )
        self.reference_reading_plan = plan
        self.reference_reading_plan_artifact_id = artifact.artifact_id
        self.reference_units = planned_units
        self.reference_units_by_id = {item["unit_id"]: item for item in planned_units}
        return {
            "accepted": True,
            "artifact_id": artifact.artifact_id,
            "base_node_count": len(self.reference_base_units),
            "reading_unit_count": len(planned_units),
            "preserved_catalog_count": plan["preserved_catalog_count"],
            "skipped_node_count": plan["skipped_node_count"],
            "disposition_counts": plan["disposition_counts"],
            "materialized_catalogs": materialized_catalogs,
            "units": [
                {
                    "unit_id": item["unit_id"],
                    "title": item["title"],
                    "unit_kind": item["unit_kind"],
                    "member_node_count": item["member_node_count"],
                    "source_part_count": item["source_part_count"],
                }
                for item in planned_units
            ],
        }

    def inspect_reference_units(
        self,
        offset: int = 0,
        limit: int = 80,
        status: str | None = None,
    ) -> dict[str, Any]:
        if self.stage != "reference_build":
            raise PermissionError("Reference reading units are only available in reference_build")
        offset = max(0, int(offset))
        limit = max(1, min(20, int(limit)))
        items = self.reference_units
        normalized_status = str(status or "").strip().casefold()
        if normalized_status == "reviewed":
            items = [item for item in items if item["unit_id"] in self.reference_unit_reviews]
        elif normalized_status == "pending":
            items = [item for item in items if item["unit_id"] not in self.reference_unit_reviews]
        elif normalized_status:
            raise ValueError("status must be reviewed, pending, or omitted")
        descriptors = []
        for item in items[offset : offset + limit]:
            descriptors.append(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"source_parts", "evidence_ids"}
                }
                | {
                    "reviewed": item["unit_id"] in self.reference_unit_reviews,
                    "evidence_count": len(item["evidence_ids"]),
                }
            )
        return {
            "offset": offset,
            "limit": limit,
            "total": len(items),
            "items": descriptors,
            "next_offset": min(len(items), offset + limit),
            "truncated": offset + limit < len(items),
        }

    def _reference_unit_payload(self, unit_id: str) -> dict[str, Any]:
        unit = self.reference_units_by_id.get(unit_id)
        if unit is None:
            raise ValueError(f"Unknown Reference reading unit: {unit_id}")
        evidence_payloads: list[dict[str, Any]] = []
        for evidence_id in unit["evidence_ids"]:
            evidence = self.evidence_by_id[evidence_id]
            evidence_payloads.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "source_part_id": evidence.source_part_id,
                    "title": evidence.title,
                    "kind": evidence.kind,
                    "content": evidence.content,
                    "locator": dict(evidence.locator),
                    "start_offset": evidence.start_offset,
                    "end_offset": evidence.end_offset,
                    "content_sha256": evidence.content_sha256,
                }
            )
        return {
            "unit": {
                key: value
                for key, value in unit.items()
                if key not in {"source_parts", "evidence_ids"}
            },
            "source_parts": list(unit["source_parts"]),
            "evidence": evidence_payloads,
            "instruction": (
                "Read this complete semantic unit as one context. SourceParts remain separate "
                "citation anchors; do not merge or rewrite them."
            ),
        }

    def read_reference_unit(self, unit_id: str) -> dict[str, Any]:
        if self.stage != "reference_build":
            raise PermissionError("Reference reading units are only available in reference_build")
        return self._reference_unit_payload(unit_id)

    def activate_reference_unit(self, unit_id: str) -> dict[str, Any]:
        if self.stage == "reference_build" and self.reference_reading_plan is None:
            raise ValueError(
                "Submit and validate the Reference reading plan before dispatching units"
            )
        payload = self._reference_unit_payload(unit_id)
        self.active_reference_unit_id = unit_id
        return payload

    def submit_reference_unit_review(
        self,
        unit_id: str,
        candidates: list[dict[str, Any]] | None = None,
        no_reference_reason: str = "",
    ) -> dict[str, Any]:
        if self.stage != "reference_build":
            raise PermissionError("Reference unit reviews are only allowed in reference_build")
        if self.reference_reading_plan is None:
            raise ValueError(
                "Submit and validate the Reference reading plan before reviewing units"
            )
        if unit_id != self.active_reference_unit_id:
            raise ValueError("unit_id must match the currently dispatched Reference reading unit")
        unit = self.reference_units_by_id.get(unit_id)
        if unit is None:
            raise ValueError(f"Unknown Reference reading unit: {unit_id}")
        normalized_candidates: list[dict[str, Any]] = []
        allowed_evidence = set(unit["evidence_ids"])
        for index, raw in enumerate(candidates or []):
            if not isinstance(raw, dict):
                raise ValueError(f"Candidate note {index} must be an object")
            title = _compact(str(raw.get("title") or ""))
            category = _compact(str(raw.get("category") or "reference"))
            rationale = _compact(str(raw.get("rationale") or ""))
            evidence_ids = list(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw.get("evidence_ids") or []
                    if str(item).strip()
                )
            )
            if len(title) < 3 or len(rationale) < 3:
                raise ValueError("Each candidate note requires a concrete title and rationale")
            if not evidence_ids or not set(evidence_ids).issubset(allowed_evidence):
                raise ValueError(
                    "Candidate evidence_ids must be a non-empty subset of the current unit"
                )
            normalized_candidates.append(
                {
                    "title": title[:300],
                    "category": category[:120] or "reference",
                    "rationale": rationale[:600],
                    "evidence_ids": evidence_ids,
                    "suggested_usage_tags": list(
                        dict.fromkeys(
                            str(item).strip()
                            for item in raw.get("suggested_usage_tags") or []
                            if str(item).strip()
                        )
                    )[:16],
                }
            )
        reason = _compact(no_reference_reason)
        if not normalized_candidates and len(reason) < 3:
            raise ValueError(
                "A unit with no Reference candidates requires a concrete no_reference_reason"
            )
        review = {
            "unit_id": unit_id,
            "evidence_node_id": unit["evidence_node_id"],
            "title_path": unit["title_path"],
            "candidates": normalized_candidates,
            "no_reference_reason": reason[:600],
            "reviewed_at": utc_now(),
        }
        self.reference_unit_reviews[unit_id] = review
        self.active_reference_unit_id = None
        return {
            "accepted": True,
            "review": review,
            "coverage": self.inspect_reference_unit_coverage(include_reviews=False),
        }

    def submit_reference_unit_result(
        self,
        unit_id: str,
        references: list[dict[str, Any]] | None = None,
        collections: list[dict[str, Any]] | None = None,
        no_reference_reason: str = "",
    ) -> dict[str, Any]:
        """Resolve one selected read directly into final Reference candidates."""

        if self.stage != "reference_build":
            raise PermissionError("Reference unit results are only allowed in reference_build")
        if self.reference_reading_plan is None:
            raise ValueError("Submit the Reference reading plan before resolving units")
        if unit_id != self.active_reference_unit_id:
            raise ValueError("unit_id must match the currently dispatched Reference reading unit")
        if unit_id in self.reference_unit_reviews:
            raise ValueError(f"Reference reading unit {unit_id} has already been resolved")
        unit = self.reference_units_by_id.get(unit_id)
        if unit is None:
            raise ValueError(f"Unknown Reference reading unit: {unit_id}")

        raw_references = references or []
        raw_collections = collections or []
        if not isinstance(raw_references, list) or not isinstance(raw_collections, list):
            raise ValueError("references and collections must be arrays")
        if len(raw_references) > 12 or len(raw_collections) > 12:
            raise ValueError("One reading unit may submit at most 12 summaries and 12 collections")
        reason = _compact(no_reference_reason)
        if not raw_references and not raw_collections and len(reason) < 3:
            raise ValueError(
                "A unit with no reusable Reference requires a concrete no_reference_reason"
            )

        allowed_evidence = set(unit["evidence_ids"])
        normalized_references: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_references):
            draft = ReferenceDraft.model_validate(raw)
            if draft.reference_mode != "summary" or draft.entries:
                raise ValueError(
                    f"Reference {index} must be a summary; use collections for exact inventories"
                )
            cited_ids = {item.evidence_id for item in draft.citations}
            if not cited_ids.issubset(allowed_evidence):
                raise ValueError(
                    f"Reference {index} cites Evidence outside the current reading unit"
                )
            normalized_references.append(draft.model_dump(mode="json"))

        normalized_collections: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_collections):
            if not isinstance(raw, dict):
                raise ValueError(f"Collection {index} must be an object")
            evidence_ids = list(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw.get("evidence_ids") or []
                    if str(item).strip()
                )
            )
            if not evidence_ids or not set(evidence_ids).issubset(allowed_evidence):
                raise ValueError(
                    f"Collection {index} evidence_ids must be a non-empty subset of the current unit"
                )
            title = _compact(str(raw.get("title") or ""))
            description = _compact(str(raw.get("description") or ""))
            if len(title) < 3:
                raise ValueError(f"Collection {index} requires a concrete title")
            description_errors = catalog_description_errors(description)
            if description_errors:
                raise ValueError(f"Collection {index}: {'; '.join(description_errors)}")
            injection_policy = _compact(str(raw.get("injection_policy") or "on_demand"))
            if injection_policy not in {"always", "on_demand", "contextual"}:
                raise ValueError(f"Collection {index} has invalid injection_policy")
            normalized_collections.append(
                {
                    "evidence_ids": evidence_ids,
                    "title": title[:300],
                    "description": description,
                    "category": _compact(str(raw.get("category") or "source_catalog"))[:120],
                    "injection_policy": injection_policy,
                    "usage_tags": list(
                        dict.fromkeys(
                            str(item).strip()
                            for item in raw.get("usage_tags") or []
                            if str(item).strip()
                        )
                    )[:16],
                }
            )

        remaining = max(
            0,
            self.settings.max_knowledge_candidates
            - self.initial_reference_count
            - len(self.submitted_references),
        )
        if len(normalized_references) + len(normalized_collections) > remaining:
            raise ValueError(
                "This unit submits more References than the remaining candidate capacity"
            )

        accepted_ids: list[str] = []
        if normalized_references:
            result = self.submit_reference_batch(normalized_references)
            if result["errors"]:
                raise ValueError(
                    "Reference submission failed validation: " + json.dumps(result["errors"])
                )
            accepted_ids.extend(result["candidate_ids"])
        for collection in normalized_collections:
            result = self.submit_reference_collection(**collection)
            accepted_ids.append(str(result["candidate_id"]))

        resolution = {
            "unit_id": unit_id,
            "evidence_node_id": unit["evidence_node_id"],
            "title_path": unit["title_path"],
            "reference_ids": accepted_ids,
            "summary_reference_count": len(normalized_references),
            "collection_reference_count": len(normalized_collections),
            "no_reference_reason": reason[:600],
            "resolved_at": utc_now(),
        }
        self.reference_unit_reviews[unit_id] = resolution
        self.active_reference_unit_id = None
        return {
            "accepted": True,
            "resolution": resolution,
            "remaining_unit_count": sum(
                item["unit_id"] not in self.reference_unit_reviews
                for item in self.reference_units
            ),
        }

    def inspect_reference_unit_coverage(
        self,
        offset: int = 0,
        limit: int = 80,
        include_reviews: bool = True,
    ) -> dict[str, Any]:
        if self.stage != "reference_build":
            raise PermissionError("Reference unit coverage is only available in reference_build")
        offset = max(0, int(offset))
        limit = max(1, min(80, int(limit)))
        pending = [
            item["unit_id"]
            for item in self.reference_units
            if item["unit_id"] not in self.reference_unit_reviews
        ]
        reviews = [
            self.reference_unit_reviews[item["unit_id"]]
            for item in self.reference_units
            if item["unit_id"] in self.reference_unit_reviews
        ]
        return {
            "total_unit_count": len(self.reference_units),
            "reviewed_unit_count": len(reviews),
            "pending_unit_count": len(pending),
            "pending_unit_ids": pending[:80],
            "candidate_note_count": sum(len(item.get("candidates", [])) for item in reviews),
            "submitted_reference_count": sum(
                len(item.get("reference_ids", [])) for item in reviews
            ),
            "reviews": reviews[offset : offset + limit] if include_reviews else [],
            "reviews_next_offset": min(len(reviews), offset + limit),
            "reviews_truncated": include_reviews and offset + limit < len(reviews),
        }

    def inspect_task_units(
        self,
        offset: int = 0,
        limit: int = 80,
        status: str | None = None,
    ) -> dict[str, Any]:
        if self.stage != "task_discovery":
            raise PermissionError("Task reading units are only available in task_discovery")
        offset = max(0, int(offset))
        limit = max(1, min(80, int(limit)))
        items = self.reference_units
        normalized_status = str(status or "").strip().casefold()
        if normalized_status in {"read", "reviewed"}:
            items = [item for item in items if item["unit_id"] in self.task_unit_reads]
        elif normalized_status in {"unread", "pending"}:
            items = [item for item in items if item["unit_id"] not in self.task_unit_reads]
        elif normalized_status:
            raise ValueError("status must be read, unread, or omitted")
        descriptors = []
        for item in items[offset : offset + limit]:
            descriptors.append(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"source_parts", "evidence_ids"}
                }
                | {
                    "read": item["unit_id"] in self.task_unit_reads,
                    "evidence_count": len(item["evidence_ids"]),
                }
            )
        return {
            "offset": offset,
            "limit": limit,
            "total": len(items),
            "items": descriptors,
            "next_offset": min(len(items), offset + limit),
            "truncated": offset + limit < len(items),
        }

    def read_task_unit(self, unit_id: str) -> dict[str, Any]:
        if self.stage != "task_discovery":
            raise PermissionError("Task reading units are only available in task_discovery")
        payload = self._reference_unit_payload(unit_id)
        self.task_unit_reads.add(unit_id)
        return payload | {
            "selection_policy": "selective_evidence_tree",
            "selected_unit_count": len(self.task_unit_reads),
        }

    def task_unit_ids_read(self) -> list[str]:
        return [
            item["unit_id"]
            for item in self.reference_units
            if item["unit_id"] in self.task_unit_reads
        ]

    def activate_task_unit(self, unit_id: str) -> dict[str, Any]:
        payload = self._reference_unit_payload(unit_id)
        self.active_task_unit_id = unit_id
        return payload

    def submit_task_unit_review(
        self,
        unit_id: str,
        candidates: list[dict[str, Any]] | None = None,
        no_task_reason: str = "",
    ) -> dict[str, Any]:
        if self.stage != "task_discovery":
            raise PermissionError("Task unit reviews are only allowed in task_discovery")
        if unit_id != self.active_task_unit_id:
            raise ValueError("unit_id must match the currently dispatched Task reading unit")
        unit = self.reference_units_by_id.get(unit_id)
        if unit is None:
            raise ValueError(f"Unknown Task reading unit: {unit_id}")
        normalized_candidates: list[dict[str, Any]] = []
        allowed_evidence = set(unit["evidence_ids"])
        for index, raw in enumerate(candidates or []):
            if not isinstance(raw, dict):
                raise ValueError(f"Candidate note {index} must be an object")
            task_text = _compact(str(raw.get("task") or ""))
            category = str(raw.get("category") or "")
            rationale = _compact(str(raw.get("rationale") or ""))
            citations: list[EvidenceCitation] = []
            citation_errors: list[str] = []
            for raw_citation in raw.get("citations") or []:
                try:
                    citation = EvidenceCitation.model_validate(raw_citation)
                    if citation.evidence_id not in allowed_evidence:
                        citation_errors.append(
                            f"Evidence {citation.evidence_id} is outside the current Task unit"
                        )
                        continue
                    errors = _validate_citations([citation], self.evidence_by_id)
                    if errors:
                        citation_errors.extend(errors)
                        continue
                    citations.append(citation)
                except Exception as exc:
                    citation_errors.append(str(exc))
            if not 3 <= len(task_text) <= 300 or len(rationale) < 3:
                raise ValueError(
                    "Each Task candidate note requires task text in 3–300 characters and a concrete rationale"
                )
            if citation_errors:
                raise ValueError("; ".join(citation_errors))
            if not citations:
                raise ValueError(
                    "Each Task candidate note requires at least one citation from the current unit"
                )
            categories = task_category_evidence_index(self.evidence_by_id, self.corpus.evidence_trees)
            if category not in categories or not {citation.evidence_id for citation in citations} & categories[category]:
                raise ValueError("Task candidate category must match a first-level chapter containing cited Evidence")
            normalized_candidates.append(
                {
                    "task": task_text,
                    "category": category,
                    "rationale": rationale[:600],
                    "citations": [item.model_dump(mode="json") for item in citations],
                }
            )
        reason = _compact(no_task_reason)
        if not normalized_candidates and len(reason) < 3:
            raise ValueError(
                "A unit with no Task candidates requires a concrete no_task_reason"
            )
        review = {
            "unit_id": unit_id,
            "evidence_node_id": unit["evidence_node_id"],
            "title_path": unit["title_path"],
            "candidates": normalized_candidates,
            "no_task_reason": reason[:600],
            "reviewed_at": utc_now(),
        }
        self.task_unit_reviews[unit_id] = review
        self.active_task_unit_id = None
        return {
            "accepted": True,
            "review": review,
            "coverage": self.inspect_task_unit_coverage(include_reviews=False),
        }

    def inspect_task_unit_coverage(
        self,
        offset: int = 0,
        limit: int = 80,
        include_reviews: bool = True,
    ) -> dict[str, Any]:
        if self.stage != "task_discovery":
            raise PermissionError("Task unit coverage is only available in task_discovery")
        offset = max(0, int(offset))
        limit = max(1, min(80, int(limit)))
        pending = [
            item["unit_id"]
            for item in self.reference_units
            if item["unit_id"] not in self.task_unit_reviews
        ]
        reviews = [
            self.task_unit_reviews[item["unit_id"]]
            for item in self.reference_units
            if item["unit_id"] in self.task_unit_reviews
        ]
        return {
            "total_unit_count": len(self.reference_units),
            "reviewed_unit_count": len(reviews),
            "pending_unit_count": len(pending),
            "pending_unit_ids": pending[:80],
            "candidate_note_count": sum(len(item["candidates"]) for item in reviews),
            "reviews": reviews[offset : offset + limit] if include_reviews else [],
            "reviews_next_offset": min(len(reviews), offset + limit),
            "reviews_truncated": include_reviews and offset + limit < len(reviews),
        }

    def _task_tree_node(self, node_id: str) -> tuple[EvidenceTree, EvidenceNode]:
        if self.stage not in {"task_discovery", "task_trajectory", "trajectory_inference", "skill_induction"}:
            raise PermissionError("EvidenceTree branch lookup is not available in this stage")
        for tree in self.corpus.evidence_trees:
            for node in tree.nodes:
                if node.evidence_node_id == node_id:
                    return tree, node
        raise ValueError(f"Unknown EvidenceTree node ID: {node_id}")

    def _task_tree_descriptor(
        self,
        node: EvidenceNode,
        *,
        tree: EvidenceTree | None = None,
        include_evidence_ids: bool = True,
    ) -> dict[str, Any]:
        unit = self.reference_base_units_by_node_id.get(node.evidence_node_id)
        evidence_ids = unit["evidence_ids"] if unit else []
        descriptor: dict[str, Any] = {
            "node_id": node.evidence_node_id,
                    "title": node.title,
            "node_type": node.node_type,
            "child_count": len(node.child_node_ids),
            "direct_evidence_count": len(evidence_ids),
            "direct_char_count": node.direct_char_count,
            "descendant_char_count": node.descendant_char_count,
        }
        if self.stage in {"task_discovery", "task_trajectory", "skill_induction"}:
            descriptor["read_node_id"] = node.evidence_node_id if evidence_ids else None
        elif include_evidence_ids:
            descriptor["direct_evidence_ids"] = evidence_ids[:4]
        if tree is not None:
            catalog = self._task_catalog_branch(tree, node)
            if catalog is not None:
                descriptor["catalog_entry_count"] = catalog["entry_count"]
        return descriptor

    def _task_catalog_branch(
        self, tree: EvidenceTree, node: EvidenceNode
    ) -> dict[str, Any] | None:
        """Summarize a proven lookup catalog without exposing its rows as Tasks."""

        if len(node.child_node_ids) < 12:
            return None
        nodes = {item.evidence_node_id: item for item in tree.nodes}
        children = [nodes[child_id] for child_id in node.child_node_ids]
        if any(child.child_node_ids for child in children):
            return None
        evidence_ids: list[str] = []
        for child in children:
            unit = self.reference_base_units_by_node_id.get(child.evidence_node_id)
            child_evidence_ids = unit["evidence_ids"] if unit else []
            if len(child_evidence_ids) != 1:
                return None
            evidence_ids.extend(child_evidence_ids)
        if len(set(evidence_ids)) != len(children):
            return None
        entry_ids = set(evidence_ids)
        if not any(
            catalog["source_id"] == tree.source_id
            and catalog["structure"] in {"numbered_heading_series", "repeated_field_schema"}
            and set(catalog["evidence_ids"]) == entry_ids
            for catalog in self.reference_catalogs
        ):
            return None
        sample_indices = sorted({0, len(children) // 2, len(children) - 1})
        return {
            "node_id": node.evidence_node_id,
            "title": node.title[:120],
            "entry_count": len(children),
            "sample_titles": [children[index].title[:120] for index in sample_indices],
        }

    def inspect_task_outline(self) -> dict[str, Any]:
        if self.stage not in {"task_discovery", "task_trajectory", "skill_induction"}:
            raise PermissionError("EvidenceTree navigation is not available in this stage")
        trees = self.corpus.evidence_trees
        sources = {item.source_id: item.display_name for item in self.corpus.source_results}
        tree_source_ids = {tree.source_id for tree in trees}
        evidence_source_ids = {item.source_id for item in self.evidence_by_id.values()}
        items = []
        for tree in trees:
            nodes: dict[str, dict[str, Any]] = {}
            for node in tree.nodes:
                descriptor: dict[str, Any] = {
                    "node_id": node.evidence_node_id,
                    "title": node.title,
                    "direct_char_count": node.direct_char_count,
                    "children": [],
                }
                unit = self.reference_base_units_by_node_id.get(node.evidence_node_id)
                if unit and unit["evidence_ids"]:
                    descriptor["read_node_id"] = node.evidence_node_id
                catalog = self._task_catalog_branch(tree, node)
                if catalog is not None:
                    descriptor["catalog_entry_count"] = catalog["entry_count"]
                nodes[node.evidence_node_id] = descriptor
            for node in tree.nodes:
                nodes[node.evidence_node_id]["children"] = [
                    nodes[child_id] for child_id in node.child_node_ids
                ]
            items.append({
                "source_id": tree.source_id,
                "source_name": sources.get(tree.source_id, tree.source_id),
                "tree_id": tree.evidence_tree_id,
                "node_count": len(tree.nodes),
                "leaf_count": sum(not node.child_node_ids for node in tree.nodes),
                "root": nodes[tree.root_node_id],
            })
        return {
            "tree_count": len(trees),
            "items": items,
            "task_categories": sorted(task_category_evidence_index(self.evidence_by_id, trees)),
            "sources_without_tree": [
                {"source_id": source_id, "source_name": sources.get(source_id, source_id)}
                for source_id in sorted(evidence_source_ids - tree_source_ids)
            ],
        }

    def read_task_nodes(self, node_ids: list[str]) -> dict[str, Any]:
        if self.stage not in {"task_discovery", "task_trajectory", "skill_induction"}:
            raise PermissionError("EvidenceTree node reading is not available in this stage")
        unique_ids, failures = self._normalized_id_batch(node_ids, "node_ids", 8)
        known_ids = {
            node.evidence_node_id
            for tree in self.corpus.evidence_trees
            for node in tree.nodes
        }
        failures += [
            f"{node_id} (unknown EvidenceTree node ID)"
            for node_id in unique_ids
            if node_id not in known_ids
        ]
        if failures:
            raise ValueError(
                "Whole batch rejected; invalid node_ids: " + "; ".join(failures)
            )
        return {"nodes": [self.read_task_node(node_id) for node_id in unique_ids]}

    def read_task_node(self, node_id: str) -> dict[str, Any]:
        """Read direct node content in authored order with exact E citation boundaries.

        Descendant text is deliberately excluded: a structural parent is a
        directory, not a request to load an entire manual.
        """
        if self.stage not in {"task_discovery", "task_trajectory", "skill_induction"}:
            raise PermissionError("EvidenceTree node reading is not available in this stage")
        tree, node = self._task_tree_node(node_id)
        unit = self.reference_base_units_by_node_id.get(node_id)
        payload: dict[str, Any] = {
            "node": self._task_tree_descriptor(node, tree=tree),
            "source_id": tree.source_id,
            "title_path": [str(title)[:120] for title in unit["title_path"]] if unit else [],
            "segments": [],
            "hint": "Cite an E evidence_id and a nonempty supporting quote from its segment content. Quote text is retained without exact substring validation.",
        }
        if not node.source_refs:
            payload["status"] = "structural_only"
            payload["hint"] = "This node has no direct text. Select a readable descendant from inspect_task_outline."
            return payload

        cursor = 0
        for index, source_ref in enumerate(node.source_refs):
            if index:
                cursor += 2  # EvidenceTree materializes spans with two newlines.
            matches = sorted(
                self._evidence_by_part_id.get(source_ref.source_part_id, []),
                key=lambda item: item.start_offset,
            )
            for evidence in matches:
                if evidence.source_id != tree.source_id:
                    continue
                span_start = max(source_ref.start_offset, evidence.start_offset)
                span_end = min(source_ref.end_offset, evidence.end_offset)
                if span_end <= span_start:
                    continue
                node_offset = cursor + span_start - source_ref.start_offset
                evidence_offset = span_start - evidence.start_offset
                content = evidence.content[evidence_offset : evidence_offset + span_end - span_start]
                if content != node.content[node_offset : node_offset + len(content)]:
                    raise ValueError("EvidenceTree content does not match its citable source span")
                segment: dict[str, Any] = {
                    "evidence_id": evidence.evidence_id,
                    "title": evidence.title[:120],
                    "node_offset": node_offset,
                    "evidence_offset": evidence_offset,
                    "resolved_version": evidence.resolved_version,
                    "content": content,
                }
                for field in ("version_relation", "evidence_applicability", "source_origin", "source_priority"):
                    if evidence.locator.get(field) is not None:
                        segment[field] = evidence.locator[field]
                if evidence.image_refs:
                    segment["image_refs"] = [
                        {"artifact_id": ref.artifact_id, "role": ref.role}
                        for ref in evidence.image_refs
                    ]
                payload["segments"].append(segment)
            cursor += source_ref.end_offset - source_ref.start_offset
        for segment in payload["segments"]:
            self.task_evidence_reads.add(segment["evidence_id"])
        if payload["segments"]:
            self.task_node_reads.add(node_id)
        return payload

    def task_node_ids_read(self) -> list[str]:
        return [
            node.evidence_node_id
            for tree in self.corpus.evidence_trees
            for node in tree.nodes
            if node.evidence_node_id in self.task_node_reads
        ]

    def _task_branch_evidence_ids(self, node_id: str) -> set[str]:
        tree, _ = self._task_tree_node(node_id)
        nodes = {item.evidence_node_id: item for item in tree.nodes}
        selected: set[str] = set()
        pending = [node_id]
        visited: set[str] = set()
        while pending:
            current_id = pending.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            node = nodes[current_id]
            unit = self.reference_base_units_by_node_id.get(current_id)
            if unit:
                selected.update(unit["evidence_ids"])
            pending.extend(node.child_node_ids)
        return selected

    def list_evidence(
        self,
        page: int = 1,
        page_size: int = 20,
        source_id: str | None = None,
        kind: str | None = None,
        reference_like_only: bool = False,
        min_reference_like_score: int = 1,
        branch_node_id: str | None = None,
        search_scope: Literal["group", "global"] = "group",
    ) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = max(1, min(
            8 if self.stage == "trajectory_inference" else
            (24 if self.stage in {"task_discovery", "task_trajectory"} else 80), int(page_size)))
        if search_scope not in {"group", "global"}:
            raise ValueError("search_scope must be group or global")
        if self.stage == "trajectory_inference" and search_scope == "group" and not branch_node_id:
            branch_node_id = self.trajectory_branch_node_id
        if (
            self.stage in {"task_discovery", "task_trajectory"}
            and branch_node_id
            and not kind
            and not reference_like_only
        ):
            tree, node = self._task_tree_node(branch_node_id)
            catalog = self._task_catalog_branch(tree, node)
            if catalog is not None and (source_id is None or source_id == tree.source_id):
                direct_unit = self.reference_base_units_by_node_id.get(node.evidence_node_id)
                direct_count = len(direct_unit["evidence_ids"]) if direct_unit else 0
                return {
                    "page": page,
                    "page_size": page_size,
                    "total": catalog["entry_count"] + direct_count,
                    "items": [],
                    "catalog": catalog,
                    "collapsed": True,
                    "status": "catalog_summary",
                    "hint": (
                        "This branch is a lookup catalog; do not paginate its entries for Task discovery. "
                        "Use search_evidence with branch_node_id for a specific entry, then "
                        "read_task_node on the matching node for exact citable text."
                    ),
                }
        items = sorted(self.evidence_by_id.values(), key=self._evidence_reading_order_key)
        if self.stage == "skill_induction":
            tree_sources = {tree.source_id for tree in self.corpus.evidence_trees}
            if source_id in tree_sources:
                raise ValueError("This source has an EvidenceTree; use its directory and read_task_node")
            items = [item for item in items if item.source_id not in tree_sources]
        if branch_node_id:
            branch_ids = self._task_branch_evidence_ids(branch_node_id)
            items = [item for item in items if item.evidence_id in branch_ids]
        if source_id:
            items = [item for item in items if item.source_id == source_id]
        if kind:
            items = [item for item in items if item.kind == kind]
        if reference_like_only:
            threshold = max(0, int(min_reference_like_score))
            items = [
                item
                for item in items
                if _reference_like_evidence_profile(item)["score"] >= threshold
            ]
        start = (page - 1) * page_size
        end = min(len(items), start + page_size)
        if self.stage in {"task_discovery", "task_trajectory"}:
            signature = (
                str(source_id or ""),
                str(kind or ""),
                bool(reference_like_only),
                max(0, int(min_reference_like_score)) if reference_like_only else 0,
                str(branch_node_id or ""),
            )
            ranges = self._task_evidence_listed_ranges.setdefault(signature, [])
            cursor = start
            for range_start, range_end in sorted(ranges):
                if range_end <= cursor:
                    continue
                if range_start > cursor:
                    break
                cursor = max(cursor, range_end)
                if cursor >= end:
                    break
            if start >= len(items) or (end > start and cursor >= end):
                return {
                    "page": page,
                    "page_size": page_size,
                    "total": len(items),
                    "items": [],
                    "status": "skipped",
                    "skipped": True,
                    "reason": "evidence_inventory_range_already_listed",
                    "message": (
                        "This Evidence inventory range was already returned in this Child Session. "
                        "Use the earlier descriptors; do not relist it with another page size."
                    ),
                    "_gateway_budget_charge": False,
                }
            ranges.append((start, end))
        return {
            "page": page,
            "page_size": page_size,
            "total": len(items),
            "items": [
                self._task_evidence_descriptor(item, preview_chars=120)
                if self.stage in {"task_discovery", "task_trajectory"} else (
                    self._trajectory_evidence_descriptor(item, preview_chars=120)
                    if self.stage == "trajectory_inference" else self._evidence_descriptor(item))
                for item in items[start:end]
            ],
        }

    def search_evidence(
        self, query: str, limit: int = 12, branch_node_id: str | None = None,
        search_scope: Literal["group", "global"] = "group",
    ) -> dict[str, Any]:
        normalized = _compact(query).lower()
        if not normalized:
            raise ValueError("query cannot be empty")
        limit = max(1, min(
            8 if self.stage == "trajectory_inference" else
            (12 if self.stage in {"task_discovery", "task_trajectory"} else 30), int(limit)))
        if search_scope not in {"group", "global"}:
            raise ValueError("search_scope must be group or global")
        if self.stage == "trajectory_inference" and search_scope == "group" and not branch_node_id:
            branch_node_id = self.trajectory_branch_node_id
        branch_ids = self._task_branch_evidence_ids(branch_node_id) if branch_node_id else None
        terms = [term for term in re.findall(r"[\w-]+", normalized, flags=re.UNICODE)
                 if len(term) > 1 and (self.stage != "trajectory_inference" or term != self.target_profile.software_name.casefold())]
        scored: list[tuple[int, EvidenceExcerpt]] = []
        for evidence in self.evidence_by_id.values():
            if branch_ids is not None and evidence.evidence_id not in branch_ids:
                continue
            title = evidence.title.lower()
            content = evidence.content.lower()
            hierarchy = " ".join(
                str(value)
                for value in (
                    evidence.locator.get("section_path"),
                    evidence.locator.get("chapter_heading"),
                    " ".join(str(item) for item in evidence.locator.get("ancestor_headings", [])),
                )
                if value
            ).lower()
            score = 0
            if normalized in title:
                score += 12
            if normalized in content:
                score += 6
            if normalized in hierarchy:
                score += 8
            for term in terms:
                score += title.count(term) * 3 + hierarchy.count(term) * 2 + content.count(term)
            if score:
                scored.append((score, evidence))
        scored.sort(key=lambda pair: (-pair[0], *self._evidence_reading_order_key(pair[1])))
        return {
            "query": query,
            "total_matches": len(scored),
            "items": [
                {
                    **(
                        self._task_evidence_descriptor(evidence, preview_chars=160)
                        if self.stage in {"task_discovery", "task_trajectory"} else (
                            self._trajectory_evidence_descriptor(evidence, preview_chars=240)
                            if self.stage == "trajectory_inference" else self._evidence_descriptor(evidence))
                    ),
                    "score": score,
                    **({} if self.stage in {"task_discovery", "task_trajectory", "trajectory_inference"} else {
                        "preview": _excerpt_quote(evidence.content, maximum=480)
                    }),
                }
                for score, evidence in scored[:limit]
            ],
        }

    def read_evidence(self, evidence_id: str, offset: int = 0, limit: int = 8_000) -> dict[str, Any]:
        evidence = self.evidence_by_id.get(evidence_id)
        if not evidence:
            raise ValueError(f"Unknown Evidence ID: {evidence_id}")
        if self.stage in {"task_discovery", "task_trajectory", "skill_induction"}:
            node_ids = self._task_node_ids_by_evidence_id.get(evidence_id, [])
            if node_ids:
                return {
                    "ok": False,
                    "error": "task_node_read_required",
                    "node_ids": node_ids,
                    "hint": "This Evidence belongs to an EvidenceTree node. Call read_task_node with one of these node IDs to read the semantic unit with E citation boundaries.",
                    "_gateway_budget_charge": False,
                }
            self.task_evidence_reads.add(evidence_id)
        offset = max(0, int(offset))
        limit = max(1, min(
            self.settings.max_evidence_read_chars,
            12_000 if self.stage == "trajectory_inference" else self.settings.max_evidence_read_chars,
            int(limit)))
        content = evidence.content
        end = min(len(content), offset + limit)
        return {
            "evidence": (
                self._task_evidence_descriptor(evidence, include_image_refs=True)
                if self.stage in {"task_discovery", "task_trajectory"} else (
                    self._trajectory_evidence_descriptor(evidence, include_image_refs=True)
                    if self.stage == "trajectory_inference" else self._evidence_descriptor(evidence))
            ),
            "content": content[offset:end],
            "offset": offset,
            "end_offset": end,
            "total_chars": len(content),
            "truncated": end < len(content),
        }

    def task_evidence_ids_read(self) -> list[str]:
        return [
            item.evidence_id
            for item in sorted(
                self.evidence_by_id.values(), key=self._evidence_reading_order_key
            )
            if item.evidence_id in self.task_evidence_reads
        ]

    def view_evidence_image(self, evidence_id: str, artifact_id: str) -> dict[str, Any]:
        evidence = self.evidence_by_id.get(evidence_id)
        if not evidence:
            raise ValueError(f"Unknown Evidence ID: {evidence_id}")
        image_ref = next(
            (item for item in evidence.image_refs if item.artifact_id == artifact_id),
            None,
        )
        if not image_ref:
            raise ValueError(f"Image Artifact {artifact_id} is not attached to Evidence {evidence_id}")
        record = self.store.artifact_record(artifact_id)
        if not record.media_type.startswith("image/"):
            raise ValueError(f"Artifact {artifact_id} is not an image")
        deliverable = record.byte_count <= self.settings.max_vision_image_bytes
        return {
            "evidence_id": evidence_id,
            "image": image_ref.model_dump(mode="json"),
            "artifact": {
                "artifact_id": record.artifact_id,
                "media_type": record.media_type,
                "byte_count": record.byte_count,
                "sha256": record.sha256,
            },
            "delivery": (
                "multimodal_attachment" if deliverable else "unavailable_image_too_large"
            ),
            "delivery_limit_bytes": self.settings.max_vision_image_bytes,
        }

    def list_reference_catalogs(self, offset: int = 0, limit: int = 40) -> dict[str, Any]:
        """List source-structure candidates that can be preserved as Reference assets."""

        offset = max(0, int(offset))
        limit = max(1, min(80, int(limit)))
        return {
            "total": len(self.reference_catalogs),
            "offset": offset,
            "next_offset": min(len(self.reference_catalogs), offset + limit),
            "items": [
                self._reference_catalog_descriptor(item)
                for item in self.reference_catalogs[offset : offset + limit]
            ],
            "truncated": offset + limit < len(self.reference_catalogs),
        }

    def inspect_reference_catalog(
        self, catalog_id: str, offset: int = 0, limit: int = 8
    ) -> dict[str, Any]:
        catalog = self.reference_catalogs_by_id.get(catalog_id)
        if not catalog:
            raise ValueError(f"Unknown Reference catalog ID: {catalog_id}")
        offset = max(0, int(offset))
        limit = max(1, min(20, int(limit)))
        catalog_entries = list(catalog.get("catalog_entries") or [])
        if catalog_entries:
            selected_entries = catalog_entries[offset : offset + limit]
            return {
                "catalog": self._reference_catalog_descriptor(catalog),
                "offset": offset,
                "limit": limit,
                "total_entries": len(catalog_entries),
                "entries": [
                    {
                        "entry_index": offset + entry_index,
                        "evidence_id": entry["source_evidence_id"],
                        "title": entry["title"],
                        "content": entry["content"],
                        "locator": self._reference_locator_summary(entry["locator"]),
                    }
                    for entry_index, entry in enumerate(selected_entries)
                ],
                "truncated": offset + limit < len(catalog_entries),
            }

        evidence_ids = list(catalog["evidence_ids"])
        selected = evidence_ids[offset : offset + limit]
        return {
            "catalog": self._reference_catalog_descriptor(catalog),
            "offset": offset,
            "limit": limit,
            "total_entries": len(evidence_ids),
            "entries": [
                {
                    "entry_index": offset + entry_index,
                    "evidence_id": evidence_id,
                    "title": self.evidence_by_id[evidence_id].title,
                    "content": self.evidence_by_id[evidence_id].content,
                    "locator": self._reference_locator_summary(
                        self.evidence_by_id[evidence_id].locator
                    ),
                }
                for entry_index, evidence_id in enumerate(selected)
            ],
            "truncated": offset + limit < len(evidence_ids),
        }

    def inspect_reference_catalog_batch(
        self,
        catalog_ids: list[str],
        entries_per_catalog: int = REFERENCE_CATALOG_SAMPLE_SIZE,
    ) -> dict[str, Any]:
        """Inspect up to three catalogs with deterministic stratified samples.

        A catalog may contain hundreds of entries. Sampling six positions
        uniformly across its complete ordered range gives the Reference Child
        a representative view while keeping one observation compact enough
        for a model turn.
        """

        if not isinstance(catalog_ids, list) or not catalog_ids:
            raise ValueError("catalog_ids must contain at least one Reference catalog ID")
        normalized_ids = [str(item).strip() for item in catalog_ids if str(item).strip()]
        if len(normalized_ids) != len(catalog_ids):
            raise ValueError("catalog_ids cannot contain empty values")
        if len(normalized_ids) > 3:
            raise ValueError("A Reference catalog batch may contain at most three catalogs")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("catalog_ids must be unique within one batch")

        sample_size = max(1, min(REFERENCE_CATALOG_SAMPLE_SIZE, int(entries_per_catalog)))

        def compact_entry(
            *, evidence_id: str, title: str, content: str, locator: dict[str, Any]
        ) -> dict[str, Any]:
            content = str(content)
            maximum = 900
            return {
                "evidence_id": evidence_id,
                "title": title,
                "content": content[:maximum],
                "content_char_count": len(content),
                "content_truncated": len(content) > maximum,
                "locator": self._reference_locator_summary(locator),
            }

        catalogs: list[dict[str, Any]] = []
        for catalog_id in normalized_ids:
            catalog = self.reference_catalogs_by_id.get(catalog_id)
            if not catalog:
                raise ValueError(f"Unknown Reference catalog ID: {catalog_id}")

            catalog_entries = list(catalog.get("catalog_entries") or [])
            if catalog_entries:
                entries = [
                    compact_entry(
                        evidence_id=entry["source_evidence_id"],
                        title=entry["title"],
                        content=entry["content"],
                        locator=entry["locator"],
                    )
                    for entry in catalog_entries
                ]
            else:
                entries = [
                    compact_entry(
                        evidence_id=evidence_id,
                        title=self.evidence_by_id[evidence_id].title,
                        content=self.evidence_by_id[evidence_id].content,
                        locator=self.evidence_by_id[evidence_id].locator,
                    )
                    for evidence_id in catalog["evidence_ids"]
                ]

            total = len(entries)
            count = min(sample_size, total)
            if count <= 1:
                indices = [0] if count else []
            else:
                indices = sorted(
                    {
                        round(position * (total - 1) / (count - 1))
                        for position in range(count)
                    }
                )
            catalogs.append(
                {
                    "catalog": self._reference_catalog_descriptor(catalog),
                    "total_entries": total,
                    "sample_indices": indices,
                    "sampling": "stratified_uniform_full_range",
                    "entries": [entries[index] for index in indices],
                }
            )

        return {
            "catalog_count": len(catalogs),
            "entries_per_catalog": sample_size,
            "catalogs": catalogs,
        }

    def inspect_reference_coverage(
        self, catalog_offset: int = 0, catalog_limit: int = 40
    ) -> dict[str, Any]:
        catalog_offset = max(0, int(catalog_offset))
        catalog_limit = max(1, min(80, int(catalog_limit)))
        covered = sorted(
            catalog_id
            for catalog_id in self.reference_catalogs_by_id
            if catalog_id in self.submitted_catalog_ids
        )
        excluded = sorted(
            catalog_id
            for catalog_id, review in self.reference_catalog_review.items()
            if review.get("disposition") == "excluded" and catalog_id not in covered
        )
        unreviewed = sorted(
            set(self.reference_catalogs_by_id) - set(covered) - set(excluded)
        )
        covered_entry_count = sum(
            int(self.reference_catalogs_by_id[catalog_id]["entry_count"])
            for catalog_id in covered
        )
        return {
            "total_catalog_count": len(self.reference_catalogs),
            "covered_catalog_ids": covered,
            "covered_entry_count": covered_entry_count,
            "excluded": [{"catalog_id": catalog_id} for catalog_id in excluded],
            "unreviewed_catalog_ids": unreviewed,
            "catalog_offset": catalog_offset,
            "catalog_next_offset": min(
                len(self.reference_catalogs), catalog_offset + catalog_limit
            ),
            "catalogs": [
                self._reference_catalog_descriptor(item)
                for item in self.reference_catalogs[
                    catalog_offset : catalog_offset + catalog_limit
                ]
            ],
            "catalogs_truncated": catalog_offset + catalog_limit < len(self.reference_catalogs),
        }

    def review_reference_coverage(self, dispositions: list[dict[str, Any]]) -> dict[str, Any]:
        if self.stage != "reference_build":
            raise PermissionError("Reference coverage review is only allowed in the reference build stage")
        expected_ids = set(self.reference_catalogs_by_id)
        reviews: dict[str, dict[str, str]] = {}
        errors: list[str] = []
        for index, raw in enumerate(dispositions):
            catalog_id = str(raw.get("catalog_id") or "").strip()
            disposition = str(raw.get("disposition") or "").strip()
            if catalog_id not in expected_ids:
                errors.append(f"Disposition {index} references an unknown Reference catalog: {catalog_id or '<empty>'}")
                continue
            if catalog_id in reviews:
                errors.append(f"Reference catalog {catalog_id} has more than one coverage disposition")
                continue
            if disposition not in {"covered", "excluded"}:
                errors.append(f"Reference catalog {catalog_id} has invalid disposition: {disposition or '<empty>'}")
                continue
            if disposition == "covered" and catalog_id not in self.submitted_catalog_ids:
                errors.append(f"Reference catalog {catalog_id} is marked covered but no source catalog was submitted")
                continue
            reviews[catalog_id] = {"disposition": disposition}
        missing = sorted(expected_ids - set(reviews))
        if missing:
            errors.append(f"Reference coverage review is missing catalogs: {', '.join(missing)}")
        if errors:
            raise ValueError("; ".join(errors))
        self.reference_catalog_review = reviews
        review_errors = self._reference_coverage_review_errors()
        if review_errors:
            raise ValueError("; ".join(review_errors))
        return {
            "coverage": self.inspect_reference_coverage(),
            "next_step": "Call submit_stage_result after this reviewed Reference coverage is final.",
        }

    def list_tasks(self, page: int = 1, page_size: int = 30) -> dict[str, Any]:
        return self._list_candidates(list(self.tasks_by_id.values()), "task_id", page, page_size)

    def read_task(self, task_id: str) -> dict[str, Any]:
        task = self.tasks_by_id.get(task_id)
        if not task:
            raise ValueError(f"Unknown Task ID: {task_id}")
        return task.model_dump(mode="json")

    def list_references(
        self, page: int = 1, page_size: int = 30,
        scope: Literal["related", "global"] = "related",
    ) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = max(1, min(80, int(page_size)))
        items = sorted(self._available_references().values(), key=lambda item: item.reference_id)
        if scope not in {"related", "global"}:
            raise ValueError("scope must be related or global")
        if self.stage == "trajectory_inference" and scope == "related":
            cited = {citation.evidence_id for task in self.trajectory_tasks
                     for citation in task.citations}
            items = [item for item in items
                     if any(citation.evidence_id in cited for citation in item.citations)]
        start = (page - 1) * page_size
        return {
            "page": page,
            "page_size": page_size,
            "total": len(items),
            "items": [
                {
                    "reference_id": item.reference_id,
                    "title": reference_text(item.title),
                    "description": reference_text(item.description),
                    "category": item.category,
                    "reference_mode": item.reference_mode,
                    "injection_policy": item.injection_policy,
                    "entry_count": len(item.entries),
                    "citation_ids": [citation.evidence_id for citation in item.citations],
                }
                for item in items[start : start + page_size]
            ],
        }

    def read_reference(
        self, reference_id: str, entry_offset: int = 0, entry_limit: int = 12
    ) -> dict[str, Any]:
        reference = self._available_references().get(reference_id)
        if not reference:
            raise ValueError(f"Unknown Reference ID: {reference_id}")
        presentation = reference.presentation
        reference = text_only_reference(reference)
        if presentation is not None:
            # The formatted view is an additional, source-backed reading route.
            # Keep the original entries addressable via read_reference_entry.
            lines = presentation.content.splitlines(keepends=True)
            content_offset = min(len(lines), max(0, int(entry_offset)))
            content_limit = max(1, min(30, int(entry_limit))) * 20
            content_end = min(len(lines), content_offset + content_limit)
            return {
                "reference_id": reference_id,
                "title": reference.title,
                "description": reference.description,
                "reference_mode": reference.reference_mode,
                "formatting_mode": presentation.mode,
                "content": "".join(lines[content_offset:content_end]),
                "content_offset": content_offset,
                "content_limit": content_limit,
                "content_total_lines": len(lines),
                "next_offset": content_end if content_end < len(lines) else None,
                "content_truncated": content_end < len(lines),
                "citation_ids": [citation.evidence_id for citation in reference.citations],
                "instruction": (
                    "Formatted Reference pagination uses zero-based line offsets: pass next_offset "
                    "as entry_offset; each entry_limit unit reads 20 lines (maximum 30 units). "
                    "Use read_reference_entry to inspect an original source-preserved entry."
                ),
            }
        if reference.reference_mode != "source_catalog":
            if self.stage == "trajectory_inference":
                return {"reference_id": reference_id, "title": reference.title, "description": reference.description,
                        "content": reference.content[:8_000],
                        "content_truncated": len(reference.content) > 8_000,
                        "citation_ids": [item.evidence_id for item in reference.citations]}
            return reference_text_payload(reference.model_dump(mode="json"))
        entry_offset = max(0, int(entry_offset))
        entry_limit = max(1, min(30, int(entry_limit)))
        payload = reference_text_payload(reference.model_dump(mode="json", exclude={"entries"}))
        entries = reference.entries[entry_offset : entry_offset + entry_limit]
        if self.stage == "trajectory_inference":
            entry_payloads = []
            for entry in entries[:8]:
                value = {"entry_id": entry.entry_id, "title": entry.title,
                         "display_title": entry.display_title,
                         "source_evidence_id": entry.source_evidence_id}
                packet_evidence = self.evidence_by_id.get(entry.source_evidence_id)
                if (entry.source_evidence_id in self.trajectory_packet_full_evidence_ids
                        and packet_evidence is not None
                        and entry.content in packet_evidence.content):
                    value["content_status"] = "already_in_evidence_packet"
                elif len(reference.entries) == 1 and len(entry.content) <= 4_000:
                    value["content"] = entry.content
                else:
                    value["preview"] = _excerpt_quote(entry.content, maximum=240)
                entry_payloads.append(value)
            return {"reference_id": reference_id, "title": reference.title, "description": reference.description,
                    "reference_mode": reference.reference_mode,
                    "entry_total": len(reference.entries), "entry_offset": entry_offset,
                    "entries": entry_payloads,
                    "entries_truncated": entry_offset + len(entry_payloads) < len(reference.entries)}
        payload.update(
            {
                "entry_total": len(reference.entries),
                "entry_offset": entry_offset,
                "entry_limit": entry_limit,
                "entries": [self._reference_entry_descriptor(entry) for entry in entries],
                "entries_truncated": entry_offset + entry_limit < len(reference.entries),
                "instruction": "Use search_reference_entries or read_reference_entry for exact source text.",
            }
        )
        return payload

    def search_reference_entries(
        self, reference_id: str, query: str, limit: int = 12
    ) -> dict[str, Any]:
        reference = self._available_references().get(reference_id)
        if not reference:
            raise ValueError(f"Unknown Reference ID: {reference_id}")
        reference = text_only_reference(reference)
        if reference.reference_mode != "source_catalog":
            raise ValueError("Reference is a summary and has no queryable catalog entries")
        normalized = _compact(query).casefold()
        if not normalized:
            raise ValueError("query cannot be empty")
        limit = max(1, min(30, int(limit)))
        terms = [term for term in re.findall(r"[\w-]+", normalized, flags=re.UNICODE) if len(term) > 1]
        matches: list[tuple[int, ReferenceEntry]] = []
        for entry in reference.entries:
            title = entry.title.casefold()
            content = entry.content.casefold()
            score = 0
            if normalized in title:
                score += 12
            if normalized in content:
                score += 6
            for term in terms:
                score += title.count(term) * 3 + content.count(term)
            if score:
                matches.append((score, entry))
        matches.sort(key=lambda item: (-item[0], item[1].entry_id))
        return {
            "reference_id": reference_id,
            "query": query,
            "total_matches": len(matches),
            "items": [
                ({"entry_id": entry.entry_id, "title": entry.title,
                  "display_title": entry.display_title,
                  "source_evidence_id": entry.source_evidence_id,
                  "preview": _excerpt_quote(entry.content, maximum=240),
                  "score": score}
                 if self.stage == "trajectory_inference" else
                 {**self._reference_entry_descriptor(entry), "score": score})
                for score, entry in matches[:(min(limit, 8) if self.stage == "trajectory_inference" else limit)]
            ],
        }

    def read_reference_entry(
        self, reference_id: str, entry_id: str,
        offset: int = 0, limit: int = 8_000,
    ) -> dict[str, Any]:
        reference = self._available_references().get(reference_id)
        if not reference:
            raise ValueError(f"Unknown Reference ID: {reference_id}")
        reference = text_only_reference(reference)
        for entry in reference.entries:
            if entry.entry_id == entry_id:
                if self.stage == "trajectory_inference":
                    packet_evidence = self.evidence_by_id.get(entry.source_evidence_id)
                    if (entry.source_evidence_id in self.trajectory_packet_full_evidence_ids
                            and packet_evidence is not None
                            and entry.content in packet_evidence.content):
                        return {"reference_id": reference_id, "entry_id": entry_id,
                                "title": entry.title,
                                "display_title": entry.display_title,
                                "source_evidence_id": entry.source_evidence_id,
                                "content_status": "already_in_evidence_packet"}
                    offset = max(0, int(offset))
                    limit = max(1, min(8_000, int(limit)))
                    end = min(len(entry.content), offset + limit)
                    return {"reference_id": reference_id, "entry_id": entry_id,
                            "title": entry.title,
                            "display_title": entry.display_title,
                            "source_evidence_id": entry.source_evidence_id,
                            "content": entry.content[offset:end], "offset": offset,
                            "total_chars": len(entry.content), "truncated": end < len(entry.content)}
                return {
                    "reference_id": reference_id,
                    "reference_title": reference.title,
                    "entry": reference_text_payload(entry.model_dump(mode="json")),
                }
        raise ValueError(f"Unknown Reference entry ID: {entry_id}")

    def list_trajectories(self, page: int = 1, page_size: int = 30) -> dict[str, Any]:
        return self._list_candidates(list(self.trajectories_by_id.values()), "trajectory_id", page, page_size)

    def read_trajectory(self, trajectory_id: str) -> dict[str, Any]:
        trajectory = self.trajectories_by_id.get(trajectory_id)
        if not trajectory:
            raise ValueError(f"Unknown Trajectory ID: {trajectory_id}")
        return trajectory.model_dump(mode="json")

    def inspect_skill_coverage(self) -> dict[str, Any]:
        """Report which validated trajectories are linked to a submitted Skill."""
        links: dict[str, list[str]] = {trajectory_id: [] for trajectory_id in self.trajectories_by_id}
        for skill in self.submitted_skills:
            for trajectory_id in skill.trajectory_ids:
                if trajectory_id in links:
                    links[trajectory_id].append(skill.skill_id)
        covered = sorted(trajectory_id for trajectory_id, skill_ids in links.items() if skill_ids)
        excluded = sorted(
            trajectory_id
            for trajectory_id, review in self.skill_coverage_review.items()
            if review.get("disposition") == "excluded" and trajectory_id not in covered
        )
        uncovered = sorted(set(links) - set(covered) - set(excluded))
        return {
            "total_trajectory_count": len(links),
            "covered_trajectory_ids": covered,
            "covered_by_skill": {trajectory_id: links[trajectory_id] for trajectory_id in covered},
            "excluded": [
                {
                    "trajectory_id": trajectory_id,
                    "rationale": self.skill_coverage_review[trajectory_id]["rationale"],
                }
                for trajectory_id in excluded
            ],
            "uncovered_trajectory_ids": uncovered,
            "reviewed_trajectory_ids": sorted(self.skill_coverage_review),
            "remaining_skill_capacity": max(
                0, self.settings.max_knowledge_candidates - len(self.submitted_skills)
            ),
        }

    def _auto_review_covered_skills(self) -> None:
        """Record coverage that follows directly from accepted Skill links."""
        linked_ids = {
            trajectory_id
            for skill in self.submitted_skills
            for trajectory_id in skill.trajectory_ids
        }
        for trajectory_id in self.trajectories_by_id:
            if trajectory_id in linked_ids:
                self.skill_coverage_review[trajectory_id] = {
                    "disposition": "covered",
                    "rationale": "A validated Skill links this trajectory.",
                }
            elif self.skill_coverage_review.get(trajectory_id, {}).get("disposition") == "covered":
                self.skill_coverage_review.pop(trajectory_id)

    def submit_task_batch(self, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        if self.stage != "task_discovery":
            raise PermissionError("Task submissions are not allowed in this stage")
        candidates: list[TaskCandidate] = []
        errors: list[dict[str, Any]] = []
        remaining = max(
            0,
            self.settings.max_task_candidates - self.initial_task_count - len(self.submitted_tasks),
        )
        for index, raw in enumerate(tasks[:remaining]):
            try:
                draft = TaskDraft.model_validate(raw)
                candidate = TaskCandidate(
                    **draft.model_dump(mode="json"),
                    task_id=self._id(
                        "T", "tsk",
                        self.corpus.corpus_snapshot_id,
                        _compact(draft.task).lower(),
                        *sorted(citation.evidence_id for citation in draft.citations),
                    ),
                    corpus_snapshot_id=self.corpus.corpus_snapshot_id,
                )
                accepted, validation = validate_tasks(
                    [candidate], self.evidence_by_id, self.corpus.evidence_trees
                )
                if accepted:
                    candidates.append(candidate)
                else:
                    errors.append(
                        {
                            "index": index,
                            "error": "; ".join(validation[0].errors),
                        }
                    )
            except Exception as exc:
                errors.append({"index": index, "error": str(exc)})
        self.submitted_tasks.extend(candidates)
        return {
            "accepted_by_schema": len(candidates),
            "candidate_ids": [item.task_id for item in candidates],
            "errors": errors,
            "remaining_task_capacity": max(
                0,
                self.settings.max_task_candidates - self.initial_task_count - len(self.submitted_tasks),
            ),
            "next_step": "Call submit_stage_result after all intended task batches are submitted.",
        }

    def submit_reference_batch(self, references: list[dict[str, Any]]) -> dict[str, Any]:
        if self.stage != "reference_build":
            raise PermissionError("Reference submissions are not allowed in this stage")
        candidates: list[ReferenceCandidate] = []
        errors: list[dict[str, Any]] = []
        remaining = max(
            0,
            self.settings.max_knowledge_candidates
            - self.initial_reference_count
            - len(self.submitted_references),
        )
        for index, raw in enumerate(references[:remaining]):
            try:
                draft = ReferenceDraft.model_validate(raw)
                if draft.reference_mode != "summary" or draft.entries:
                    raise ValueError(
                        "Use submit_reference_catalog for source-preserved catalog entries"
                    )
                payload = draft.model_dump(mode="json")
                if not payload.get("usage_tags"):
                    payload["usage_tags"] = _infer_reference_usage_tags(
                        draft.title,
                        draft.category,
                        draft.content,
                        source_catalog=False,
                    )
                candidate = ReferenceCandidate(
                    **payload,
                    reference_id=self._id(
                        "R", "ref",
                        self.corpus.corpus_snapshot_id,
                        _compact(draft.title).lower(),
                        *sorted(citation.evidence_id for citation in draft.citations),
                    ),
                    corpus_snapshot_id=self.corpus.corpus_snapshot_id,
                )
                candidates.append(candidate)
            except Exception as exc:
                errors.append({"index": index, "error": str(exc)})
        self.submitted_references.extend(candidates)
        return {
            "accepted_by_schema": len(candidates),
            "candidate_ids": [item.reference_id for item in candidates],
            "errors": errors,
            "remaining_reference_capacity": max(
                0,
                self.settings.max_knowledge_candidates
                - self.initial_reference_count
                - len(self.submitted_references),
            ),
            "next_step": (
                "Call inspect_reference_coverage and review_reference_coverage before finalizing."
                if self.stage == "reference_build"
                else "Call submit_stage_result after all intended reference batches are submitted."
            ),
        }

    def submit_reference_catalog(
        self,
        catalog_id: str,
        title: str,
        description: str | None = None,
        category: str = "source_catalog",
        injection_policy: str = "on_demand",
        usage_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        if self.stage != "reference_build":
            raise PermissionError("Reference catalog submissions are not allowed in this stage")
        catalog = self.reference_catalogs_by_id.get(catalog_id)
        if not catalog:
            raise ValueError(f"Unknown Reference catalog ID: {catalog_id}")
        return self._submit_source_catalog(
            catalog_id=catalog_id,
            evidence_ids=list(catalog["evidence_ids"]),
            catalog_entries=list(catalog.get("catalog_entries") or []),
            title=title,
            description=description,
            category=category,
            injection_policy=injection_policy,
            usage_tags=usage_tags,
            counts_toward_coverage=True,
        )

    def submit_reference_catalog_batch(self, catalogs: list[dict[str, Any]]) -> dict[str, Any]:
        """Materialize several already-inspected source catalogs in one ReAct turn.

        Each catalog remains an independent Reference. The batch is deliberately
        non-atomic so one invalid selection does not discard otherwise valid,
        source-preserved catalogs.
        """

        if self.stage != "reference_build":
            raise PermissionError("Reference catalog submissions are not allowed in this stage")
        if not catalogs:
            raise ValueError("catalogs cannot be empty")
        if len(catalogs) > 64:
            raise ValueError("A Reference catalog batch can contain at most 64 catalogs")

        accepted: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for index, raw in enumerate(catalogs):
            if not isinstance(raw, dict):
                errors.append({"index": index, "error": "Catalog submission must be an object"})
                continue
            try:
                result = self.submit_reference_catalog(
                    catalog_id=str(raw.get("catalog_id") or "").strip(),
                    title=str(raw.get("title") or "").strip(),
                    description=str(raw.get("description") or "").strip() or None,
                    category=str(raw.get("category") or "source_catalog").strip(),
                    injection_policy=str(raw.get("injection_policy") or "on_demand").strip(),
                    usage_tags=raw.get("usage_tags"),
                )
                accepted.append(result)
            except Exception as exc:
                errors.append({"index": index, "error": str(exc)})
        return {
            "accepted_by_schema": len(accepted),
            "catalogs": accepted,
            "errors": errors,
            "remaining_reference_capacity": max(
                0,
                self.settings.max_knowledge_candidates
                - self.initial_reference_count
                - len(self.submitted_references),
            ),
            "next_step": (
                "Call inspect_reference_coverage, then review_reference_coverage before finalizing."
                if self.stage == "reference_build"
                else "Continue with the bounded coverage-gap audit stage."
            ),
        }

    def submit_reference_collection(
        self,
        evidence_ids: list[str],
        title: str,
        description: str | None = None,
        category: str = "source_catalog",
        injection_policy: str = "on_demand",
        usage_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Preserve an Agent-selected Evidence collection when no detector applies."""

        if self.stage != "reference_build":
            raise PermissionError("Reference collection submissions are not allowed in this stage")
        normalized_ids = list(dict.fromkeys(str(item).strip() for item in evidence_ids if str(item).strip()))
        if not normalized_ids:
            raise ValueError("evidence_ids cannot be empty")
        if len(normalized_ids) > 512:
            raise ValueError("A source-preserved Reference collection can contain at most 512 Evidence entries")
        unknown_ids = [item for item in normalized_ids if item not in self.evidence_by_id]
        if unknown_ids:
            raise ValueError(f"Unknown Evidence IDs: {', '.join(unknown_ids)}")
        image_only_ids = [
            item for item in normalized_ids if not self.evidence_by_id[item].content.strip()
        ]
        if image_only_ids:
            raise ValueError(
                "Source-preserved Reference collections require textual Evidence; inspect "
                "image-only Evidence with view_evidence_image instead: "
                + ", ".join(image_only_ids)
            )
        collection_id = self._id(
            "C", "rcat", self.corpus.corpus_snapshot_id, "agent_selected", *sorted(normalized_ids)
        )
        return self._submit_source_catalog(
            catalog_id=collection_id,
            evidence_ids=normalized_ids,
            catalog_entries=[],
            title=title,
            description=description,
            category=category,
            injection_policy=injection_policy,
            usage_tags=usage_tags,
            counts_toward_coverage=False,
        )

    def _submit_source_catalog(
        self,
        *,
        catalog_id: str,
        evidence_ids: list[str],
        catalog_entries: list[dict[str, Any]],
        title: str,
        description: str | None,
        category: str,
        injection_policy: str,
        usage_tags: list[str] | None,
        counts_toward_coverage: bool,
    ) -> dict[str, Any]:
        description = _compact(str(description or ""))
        if not description or len(description) > 120:
            raise ValueError("Reference description must contain 1 to 120 characters, including spaces and punctuation")
        for reference in [*self.references_by_id.values(), *self.submitted_references]:
            if reference.reference_mode == "source_catalog" and reference.source_catalog_id == catalog_id:
                if counts_toward_coverage:
                    self.submitted_catalog_ids.add(catalog_id)
                return {
                    "accepted_by_schema": True,
                    "candidate_id": reference.reference_id,
                    "catalog_id": catalog_id,
                    "entry_count": len(reference.entries),
                    "already_submitted": True,
                }
        remaining = max(
            0,
            self.settings.max_knowledge_candidates
            - self.initial_reference_count
            - len(self.submitted_references),
        )
        if not remaining:
            raise ValueError("Reference candidate capacity is exhausted")

        entries: list[ReferenceEntry] = []
        citations_by_evidence_id: dict[str, EvidenceCitation] = {}
        for evidence_id in dict.fromkeys(evidence_ids):
            evidence = self.evidence_by_id[evidence_id]
            citations_by_evidence_id[evidence.evidence_id] = EvidenceCitation(
                evidence_id=evidence.evidence_id,
                quote=_excerpt_quote(evidence.content, maximum=1_600),
                rationale="Source-preserved Reference catalog entry.",
            )

        if catalog_entries:
            for raw_entry in catalog_entries:
                evidence_id = str(raw_entry["source_evidence_id"])
                evidence = self.evidence_by_id[evidence_id]
                entry_content = str(raw_entry["content"])
                entry_title = str(raw_entry["title"])
                citation = EvidenceCitation(
                    evidence_id=evidence.evidence_id,
                    quote=_excerpt_quote(entry_content, maximum=1_600),
                    rationale="Source-preserved Reference catalog entry.",
                )
                entries.append(
                    ReferenceEntry(
                        entry_id=self._id(
                            "RE", "rfe",
                            catalog_id,
                            evidence.evidence_id,
                            entry_title,
                            entry_content,
                        ),
                        title=entry_title,
                        content=entry_content,
                        source_evidence_id=evidence.evidence_id,
                        locator=dict(raw_entry.get("locator") or evidence.locator),
                        citations=[citation],
                    )
                )
        else:
            for evidence_id in evidence_ids:
                evidence = self.evidence_by_id[evidence_id]
                citation = EvidenceCitation(
                    evidence_id=evidence.evidence_id,
                    quote=_excerpt_quote(evidence.content, maximum=1_600),
                    rationale="Source-preserved Reference catalog entry.",
                )
                entries.append(
                    ReferenceEntry(
                        entry_id=self._id("RE", "rfe", catalog_id, evidence.evidence_id),
                        title=evidence.title,
                        content=evidence.content,
                        source_evidence_id=evidence.evidence_id,
                        locator=dict(evidence.locator),
                        citations=[citation],
                    )
                )

        citations = list(citations_by_evidence_id.values())
        if not citations:
            for entry in entries:
                citations.extend(entry.citations)

        candidate = ReferenceCandidate(
            reference_id=self._id(
                "R", "ref", self.corpus.corpus_snapshot_id, "source_catalog", catalog_id, _compact(title).lower()
            ),
            corpus_snapshot_id=self.corpus.corpus_snapshot_id,
            title=title,
            category=category,
            description=_compact(str(description or "")),
            content="",
            reference_mode="source_catalog",
            injection_policy=injection_policy,
            usage_tags=usage_tags
            or _infer_reference_usage_tags(
                title,
                category,
                " ".join(entry.title for entry in entries[:20]),
                source_catalog=True,
            ),
            source_catalog_id=catalog_id,
            entries=entries,
            citations=citations,
        )
        candidate = hydrate_reference_images(candidate, self.evidence_by_id)
        self.submitted_references.append(candidate)
        if counts_toward_coverage:
            self.submitted_catalog_ids.add(catalog_id)
        return {
            "accepted_by_schema": True,
            "candidate_id": candidate.reference_id,
            "catalog_id": catalog_id,
            "entry_count": len(entries),
            "citation_count": len(citations),
            "remaining_reference_capacity": max(
                0,
                self.settings.max_knowledge_candidates
                - self.initial_reference_count
                - len(self.submitted_references),
            ),
            "next_step": (
                "Call inspect_reference_coverage, then review_reference_coverage before finalizing."
                if self.stage == "reference_build"
                else "Continue with the bounded coverage-gap audit stage."
            ),
        }

    def _trajectory_task_for_slot(self, task_slot: int) -> TaskCandidate:
        if self.stage != "trajectory_inference" or not self.trajectory_tasks or not self.knowledge:
            raise PermissionError("Trajectory submission is not allowed in this stage")
        if self.stage_result is not None:
            raise ValueError("Trajectory stage is already complete")
        if isinstance(task_slot, bool) or not isinstance(task_slot, int):
            raise ValueError("task_slot must be an integer from the dispatched Task list")
        task = self.trajectory_slots.get(task_slot)
        if task is None:
            raise ValueError("task_slot is not in the dispatched Task list")
        if task.task_id in self.trajectory_task_results:
            raise ValueError("This Task slot already has an accepted outcome")
        return task

    def _record_trajectory_task_outcome(
        self, task: TaskCandidate, candidates: list[TrajectoryCandidate],
        *, status: Literal["succeeded", "degraded"], summary: str,
        gaps: list[str] | None = None,
    ) -> None:
        result = {
            "stage": "trajectory_inference", "status": status,
            "summary": summary, "gaps": gaps or [], "warnings": [],
            "candidate_count": len(candidates), "completed_at": utc_now(),
        }
        if self.trajectory_attempt_id:
            if not self.knowledge or not self.knowledge_artifact_id:
                raise ValueError("Trajectory checkpoint requires a frozen Knowledge Snapshot")
            checkpoint_artifact_id = self.store.put_trajectory_checkpoint(
                task_id=task.task_id,
                value={
                    "schema_version": "trajectory-task-checkpoint.v6",
                    "run_id": self.store.run_id,
                    "attempt_id": self.trajectory_attempt_id,
                    "task_id": task.task_id,
                    "corpus_snapshot_artifact_id": self.corpus_artifact_id,
                    "knowledge_snapshot_artifact_id": self.knowledge_artifact_id,
                    "knowledge_snapshot_id": self.knowledge.knowledge_snapshot_id,
                    "trajectories": [item.model_dump(mode="json") for item in candidates],
                    "stage_result": result,
                },
            )
            self.trajectory_checkpoint_artifact_id = checkpoint_artifact_id
            self.trajectory_checkpoint_artifact_ids[task.task_id] = checkpoint_artifact_id
        self.trajectory_task_results[task.task_id] = result
        self.submitted_trajectories.extend(candidates)

    def submit_trajectory(self, task_slot: int, **trajectory: Any) -> dict[str, Any]:
        task = self._trajectory_task_for_slot(task_slot)
        if "task_id" in trajectory:
            raise ValueError("Trajectory task_id is assigned by the dispatched Task")
        steps = [TrajectoryStepDraft.model_validate(item)
                 for item in trajectory.get("steps", [])]
        citations = list(dict.fromkeys(
            (citation.evidence_id, citation.quote, citation.image_artifact_id)
            for step in steps for citation in step.citations
        ))
        if not citations:
            raise ValueError("Each trajectory step requires an Evidence citation")
        derived_citations = [
            {"evidence_id": evidence_id, "quote": quote,
             "image_artifact_id": image_artifact_id}
            for evidence_id, quote, image_artifact_id in citations
        ]
        supplied_citations = trajectory.pop("citations", None)
        if supplied_citations is not None:
            # The v3/v4 tool surface supplied these explicitly. Keep old audit
            # submissions replayable while the v5 public schema omits the field.
            derived_citations = [EvidenceCitation.model_validate(item).model_dump(mode="json")
                                 for item in supplied_citations]
        draft = TrajectoryDraft.model_validate({**trajectory, "task_id": task.task_id,
                                               "steps": steps,
                                               "citations": derived_citations})
        candidate = TrajectoryCandidate(
            **draft.model_dump(mode="json"),
            trajectory_id=self._id(
                "TR", "traj",
                self.knowledge.knowledge_snapshot_id,
                draft.task_id,
                *sorted(citation.evidence_id for citation in draft.citations),
            ),
            knowledge_snapshot_id=self.knowledge.knowledge_snapshot_id,
        )
        accepted, validation = validate_trajectories(
            [candidate],
            self.evidence_by_id,
            {task.task_id: task},
        )
        if not accepted:
            raise ValueError("; ".join(validation[0].errors))
        self._record_trajectory_task_outcome(
            task, [candidate], status="succeeded",
            summary="One evidence-backed trajectory was accepted for the dispatched Task.",
        )
        return {
            "accepted_by_schema": True,
            "candidate_id": candidate.trajectory_id,
            "task_slot": task_slot,
            "status": "succeeded",
            "remaining_task_slots": [slot for slot, item in self.trajectory_slots.items()
                                     if item.task_id not in self.trajectory_task_results],
            "next_step": "Complete the remaining Task slots, then call submit_stage_result.",
        }

    def report_trajectory_gap(self, task_slot: int, reason: str) -> dict[str, Any]:
        task = self._trajectory_task_for_slot(task_slot)
        reason = str(reason).strip()
        if not reason:
            raise ValueError("A precise reason is required for a Task without a trajectory")
        self._record_trajectory_task_outcome(
            task, [], status="degraded", summary="No grounded trajectory was submitted.",
            gaps=[reason[:2_000]],
        )
        return {
            "accepted": True, "task_slot": task_slot,
            "remaining_task_slots": [slot for slot, item in self.trajectory_slots.items()
                                     if item.task_id not in self.trajectory_task_results],
            "next_step": "Complete the remaining Task slots, then call submit_stage_result.",
        }

    def submit_skill_batch(self, skills: list[dict[str, Any]]) -> dict[str, Any]:
        if self.stage != "skill_induction" or not self.knowledge:
            raise PermissionError("Skill submissions are not allowed in this stage")
        candidates: list[SkillCandidate] = []
        errors: list[dict[str, Any]] = []
        remaining = max(0, self.settings.max_knowledge_candidates - len(self.submitted_skills))
        for index, raw in enumerate(skills[:remaining]):
            try:
                draft = SkillDraft.model_validate(raw)
                candidate = SkillCandidate(
                    **draft.model_dump(mode="json"),
                    skill_id=self._id(
                        "SK", "skl",
                        self.knowledge.knowledge_snapshot_id,
                        _compact(draft.title).lower(),
                        *sorted(draft.trajectory_ids),
                    ),
                    knowledge_snapshot_id=self.knowledge.knowledge_snapshot_id,
                )
                accepted, validation = validate_skills(
                    [candidate],
                    self.evidence_by_id,
                    self.tasks_by_id,
                    self.trajectories_by_id,
                )
                if accepted:
                    candidates.append(candidate)
                else:
                    errors.append(
                        {
                            "index": index,
                            "error": "; ".join(validation[0].errors),
                        }
                    )
            except Exception as exc:
                errors.append({"index": index, "error": str(exc)})
        self.submitted_skills.extend(candidates)
        self._auto_review_covered_skills()
        coverage = self.inspect_skill_coverage()
        if errors:
            next_step = (
                "Some Skill candidates were rejected; inspect the indexed errors and submit "
                "corrected Skills only where needed. Use the returned coverage to resolve "
                "uncovered trajectories before submit_stage_result."
            )
        elif coverage["uncovered_trajectory_ids"]:
            next_step = (
                "Submit source-backed Skills for uncovered trajectories, or call "
                "review_skill_coverage only for trajectories that must be excluded with a "
                "concrete rationale. Then call submit_stage_result."
            )
        else:
            next_step = "Coverage is complete. Call submit_stage_result."
        return {
            "accepted_by_schema": len(candidates),
            "candidate_ids": [item.skill_id for item in candidates],
            "errors": errors,
            "remaining_skill_capacity": max(
                0, self.settings.max_knowledge_candidates - len(self.submitted_skills)
            ),
            "coverage": coverage,
            "next_step": next_step,
        }

    def review_skill_coverage(self, dispositions: list[dict[str, Any]]) -> dict[str, Any]:
        if self.stage != "skill_induction":
            raise PermissionError("Skill coverage review is only allowed in the skill induction stage")
        expected_ids = set(self.trajectories_by_id)
        reviews = dict(self.skill_coverage_review)
        supplied_ids: set[str] = set()
        errors: list[str] = []
        for index, raw in enumerate(dispositions):
            trajectory_id = str(raw.get("trajectory_id") or "").strip()
            disposition = str(raw.get("disposition") or "").strip()
            rationale = str(raw.get("rationale") or "").strip()
            if trajectory_id not in expected_ids:
                errors.append(f"Disposition {index} references an unknown trajectory: {trajectory_id or '<empty>'}")
                continue
            if trajectory_id in supplied_ids:
                errors.append(f"Trajectory {trajectory_id} has more than one coverage disposition")
                continue
            supplied_ids.add(trajectory_id)
            if disposition not in {"covered", "excluded"}:
                errors.append(f"Trajectory {trajectory_id} has invalid disposition: {disposition or '<empty>'}")
                continue
            if disposition == "excluded" and len(rationale) < 3:
                errors.append(f"Excluded trajectory {trajectory_id} requires a concrete rationale")
                continue
            reviews[trajectory_id] = {"disposition": disposition, "rationale": rationale}
        if errors:
            raise ValueError("; ".join(errors))
        previous_reviews = self.skill_coverage_review
        self.skill_coverage_review = reviews
        review_errors = self._skill_coverage_review_errors(require_complete=False)
        if review_errors:
            self.skill_coverage_review = previous_reviews
            raise ValueError("; ".join(review_errors))
        return {
            "coverage": self.inspect_skill_coverage(),
            "next_step": "Call submit_stage_result after this reviewed coverage is final.",
        }

    def submit_stage_result(
        self,
        status: Literal["succeeded", "degraded", "failed"],
        summary: str | None = None,
        gaps: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        if summary is None and self.stage != "skill_induction":
            raise ValueError("summary is required outside Skill Induction")
        if self.stage == "trajectory_inference" and self.stage_result is not None:
            request = {"status": status, "summary": summary, "gaps": gaps,
                       "warnings": warnings}
            if request == self._trajectory_terminal_request:
                return {"accepted": True, "result": self.stage_result}
            raise ValueError("Trajectory stage result is already complete")
        terminal_request = {"status": status, "summary": summary, "gaps": gaps,
                            "warnings": warnings}
        produced = self._submission_count()
        if status == "succeeded" and produced == 0:
            status = "degraded"
            gaps = [*(gaps or []), "No candidate was submitted."]
        coverage: dict[str, Any] | None = None
        if self.stage == "trajectory_inference":
            missing_slots = [slot for slot, task in self.trajectory_slots.items()
                             if task.task_id not in self.trajectory_task_results]
            if missing_slots and status != "failed":
                raise ValueError(
                    "Trajectory coverage is incomplete. Submit a trajectory or report a gap "
                    f"for Task slots: {missing_slots}"
                )
            degraded = [task.task_id for task in self.trajectory_tasks
                        if self.trajectory_task_results.get(task.task_id, {}).get("status") == "degraded"]
            if degraded and status == "succeeded":
                status = "degraded"
            coverage = {
                "total_task_count": len(self.trajectory_tasks),
                "completed_task_ids": [task.task_id for task in self.trajectory_tasks
                                       if task.task_id in self.trajectory_task_results],
                "gap_task_ids": degraded,
            }
            gaps = [*(gaps or []), *(gap for task_id in degraded
                                     for gap in self.trajectory_task_results[task_id]["gaps"])]
        elif self.stage == "reference_build":
            coverage = (
                self.inspect_reference_unit_coverage()
                if self.reference_reading_plan is not None
                else self.inspect_reference_coverage()
            )
            coverage_errors = self._reference_coverage_review_errors()
            if coverage_errors:
                if self.reference_reading_plan is not None:
                    raise ValueError(
                        "Reference coverage is not ready for finalization. Every selected "
                        "EvidenceTree candidate region must be resolved. Details: "
                        + "; ".join(coverage_errors)
                    )
                raise ValueError(
                    "Reference coverage is not ready for finalization. "
                    "Materialize or explicitly exclude each detected source catalog, then call "
                    f"review_reference_coverage. Details: {'; '.join(coverage_errors)}"
                )
        elif self.stage == "skill_induction":
            coverage = self.inspect_skill_coverage()
            coverage_errors = self._skill_coverage_review_errors()
            if coverage_errors:
                raise ValueError(
                    "Skill coverage is not ready for finalization. "
                    "Submit Skills for supported gaps, or call review_skill_coverage "
                    "with justified exclusions for uncovered trajectories. "
                    f"Details: {'; '.join(coverage_errors)}"
                )
        if summary is None:
            # Skill completion needs only status; retain a compact internal record.
            summary = (
                f"Skill induction {status}: {produced} Skill candidate(s), "
                f"{len(coverage['covered_trajectory_ids'])} covered and "
                f"{len(coverage['excluded'])} excluded Trajectory(s)."
            )
        result = {
            "stage": self.stage,
            "status": status,
            "summary": str(summary).strip()[:2_000],
            "gaps": list(dict.fromkeys(gaps or [])),
            "warnings": list(dict.fromkeys(warnings or [])),
            "candidate_count": produced,
            "completed_at": utc_now(),
        }
        if coverage is not None:
            result["coverage"] = coverage
        if self.stage == "trajectory_inference":
            self._trajectory_terminal_request = terminal_request
        self.stage_result = result
        return {"accepted": True, "result": self.stage_result}

    def _task_coverage_review_errors(self) -> list[str]:
        if self.stage != "task_discovery":
            return []
        expected_ids = set(self.reference_units_by_id)
        reviewed_ids = set(self.task_unit_reviews)
        missing = sorted(expected_ids - reviewed_ids)
        return (
            [
                f"Unreviewed EvidenceTree reading units: {', '.join(missing[:20])}"
                + (f" (+{len(missing) - 20} more)" if len(missing) > 20 else "")
            ]
            if missing
            else []
        )

    def _reference_coverage_review_errors(self) -> list[str]:
        if self.stage != "reference_build":
            return []
        if self.reference_reading_plan is not None:
            expected_ids = set(self.reference_units_by_id)
            reviewed_ids = set(self.reference_unit_reviews)
            missing = sorted(expected_ids - reviewed_ids)
            errors = []
            if missing:
                errors.append(
                    f"Unresolved EvidenceTree reading units: {', '.join(missing[:20])}"
                    + (f" (+{len(missing) - 20} more)" if len(missing) > 20 else "")
                )
            expected_catalog_ids = set(self.reference_catalogs_by_id)
            missing_catalogs = sorted(
                expected_catalog_ids - set(self.reference_catalog_review)
            )
            if missing_catalogs:
                errors.append(
                    "Unreviewed Reference catalogs: " + ", ".join(missing_catalogs)
                )
            for catalog_id in sorted(
                expected_catalog_ids & set(self.reference_catalog_review)
            ):
                disposition = self.reference_catalog_review[catalog_id].get("disposition")
                if disposition == "covered" and catalog_id not in self.submitted_catalog_ids:
                    errors.append(
                        f"Reference catalog {catalog_id} is covered but was not materialized"
                    )
                if disposition == "excluded" and catalog_id in self.submitted_catalog_ids:
                    errors.append(
                        f"Reference catalog {catalog_id} is excluded but was materialized"
                    )
            return errors
        expected_ids = set(self.reference_catalogs_by_id)
        if not expected_ids:
            return []
        reviewed_ids = set(self.reference_catalog_review)
        if not reviewed_ids:
            return ["No Reference catalog coverage review was submitted."]
        errors: list[str] = []
        missing = sorted(expected_ids - reviewed_ids)
        if missing:
            errors.append(f"Unreviewed Reference catalogs: {', '.join(missing)}")
        for catalog_id in sorted(expected_ids & reviewed_ids):
            disposition = self.reference_catalog_review[catalog_id].get("disposition")
            if disposition == "covered" and catalog_id not in self.submitted_catalog_ids:
                errors.append(
                    f"Reference catalog {catalog_id} is marked covered but no source catalog was submitted"
                )
            if disposition == "excluded" and catalog_id in self.submitted_catalog_ids:
                errors.append(
                    f"Reference catalog {catalog_id} is marked excluded but a source catalog was submitted"
                )
        return errors

    def _skill_coverage_review_errors(self, *, require_complete: bool = True) -> list[str]:
        if self.stage != "skill_induction":
            return []
        expected_ids = set(self.trajectories_by_id)
        reviewed_ids = set(self.skill_coverage_review)
        if not expected_ids:
            return []
        if not reviewed_ids and require_complete:
            return ["No skill coverage review was submitted."]
        errors: list[str] = []
        missing = sorted(expected_ids - reviewed_ids)
        if missing and require_complete:
            errors.append(f"Unreviewed trajectories: {', '.join(missing)}")
        linked_ids = {
            trajectory_id
            for skill in self.submitted_skills
            for trajectory_id in skill.trajectory_ids
        }
        for trajectory_id in sorted(expected_ids & reviewed_ids):
            disposition = self.skill_coverage_review[trajectory_id].get("disposition")
            if disposition == "covered" and trajectory_id not in linked_ids:
                errors.append(f"Trajectory {trajectory_id} is marked covered but no submitted Skill links it")
            if disposition == "excluded" and trajectory_id in linked_ids:
                errors.append(f"Trajectory {trajectory_id} is marked excluded but is linked to a submitted Skill")
        return errors

    def _available_references(self) -> dict[str, ReferenceCandidate]:
        references = dict(self.references_by_id)
        references.update({item.reference_id: item for item in self.submitted_references})
        return references

    def _submission_count(self) -> int:
        return (
            len(self.submitted_tasks)
            + len(self.submitted_references)
            + len(self.submitted_trajectories)
            + len(self.submitted_skills)
        )

    def _task_evidence_descriptor(
        self,
        evidence: EvidenceExcerpt,
        *,
        preview_chars: int = 0,
        include_image_refs: bool = False,
    ) -> dict[str, Any]:
        locator = evidence.locator
        section = locator.get("section_path") or locator.get("chapter_heading")
        if isinstance(section, list):
            section = " / ".join(str(item) for item in section)
        descriptor: dict[str, Any] = {
            "evidence_id": evidence.evidence_id,
            "source_id": evidence.source_id,
            "kind": evidence.kind,
            "title": evidence.title[:160],
            "section": str(section)[:160] if section else None,
            "char_count": len(evidence.content),
        }
        node_ids = self._task_node_ids_by_evidence_id.get(evidence.evidence_id, [])
        if node_ids:
            descriptor["read_node_ids"] = node_ids
        page = locator.get("page_start", locator.get("page"))
        if page is not None:
            descriptor["page"] = page
        if evidence.resolved_version:
            descriptor["resolved_version"] = evidence.resolved_version
        if locator.get("source_origin") not in (None, "user"):
            descriptor["source_origin"] = locator["source_origin"]
        if locator.get("source_priority") is not None:
            descriptor["source_priority"] = locator["source_priority"]
        if locator.get("version_relation"):
            descriptor["version_relation"] = locator["version_relation"]
        if locator.get("evidence_applicability"):
            descriptor["evidence_applicability"] = locator["evidence_applicability"]
        if preview_chars:
            descriptor["preview"] = _excerpt_quote(evidence.content, maximum=preview_chars)
        if evidence.image_refs:
            descriptor["image_count"] = len(evidence.image_refs)
            if include_image_refs:
                descriptor["image_refs"] = [
                    item.model_dump(mode="json") for item in evidence.image_refs
                ]
        return descriptor

    def _trajectory_evidence_descriptor(
        self, evidence: EvidenceExcerpt, *, preview_chars: int = 0,
        include_image_refs: bool = False,
    ) -> dict[str, Any]:
        locator = evidence.locator
        section = locator.get("section_path") or locator.get("chapter_heading")
        descriptor: dict[str, Any] = {
            "evidence_id": evidence.evidence_id,
            "title": evidence.title[:160],
            "section": str(section)[:160] if section else None,
            "char_count": len(evidence.content),
            "source_resolved_version": evidence.resolved_version,
        }
        if locator.get("version_relation"):
            descriptor["version_relation"] = locator["version_relation"]
        if preview_chars:
            descriptor["preview"] = _excerpt_quote(evidence.content, maximum=preview_chars)
        if evidence.image_refs:
            descriptor["image_count"] = len(evidence.image_refs)
            if include_image_refs:
                descriptor["image_refs"] = [
                    {"artifact_id": image.artifact_id, "role": image.role}
                    for image in evidence.image_refs
                ]
        return descriptor

    def _evidence_descriptor(self, evidence: EvidenceExcerpt) -> dict[str, Any]:
        image_ref_count_total = evidence.locator.get(
            "image_ref_count_total", len(evidence.image_refs)
        )
        reference_like = _reference_like_evidence_profile(evidence)
        descriptor = {
            "evidence_id": evidence.evidence_id,
            "source_id": evidence.source_id,
            "source_part_id": evidence.source_part_id,
            "kind": evidence.kind,
            "title": evidence.title,
            "locator": evidence.locator,
            "resolved_version": evidence.resolved_version,
            "source_origin": evidence.locator.get("source_origin", "user"),
            "source_priority": evidence.locator.get("source_priority"),
            "target_version": evidence.locator.get("target_version"),
            "source_resolved_version": evidence.locator.get(
                "source_resolved_version", evidence.resolved_version
            ),
            "version_relation": evidence.locator.get("version_relation"),
            "evidence_applicability": evidence.locator.get("evidence_applicability"),
            "char_count": len(evidence.content),
            "image_count": image_ref_count_total,
            "image_refs_returned": len(evidence.image_refs),
            "image_refs": [item.model_dump(mode="json") for item in evidence.image_refs],
            "source_order": self._evidence_order_by_id.get(evidence.evidence_id, 0),
            "preview": _excerpt_quote(evidence.content, maximum=320),
            "reference_like_score": reference_like["score"],
            "reference_like_tags": reference_like["tags"],
            "reference_like_reasons": reference_like["reasons"],
            "inline_entry_count": reference_like["inline_entry_count"],
        }
        for key in (
            "image_refs_overflow_count",
            "image_refs_overflow_artifact_id",
        ):
            if key in evidence.locator:
                descriptor[key] = evidence.locator[key]
        for key in (
            "heading_level",
            "section_path",
            "ancestor_headings",
            "parent_section_part_id",
            "chapter_heading",
            "chapter_part_id",
        ):
            if key in evidence.locator:
                descriptor[key] = evidence.locator[key]
        return descriptor

    def _evidence_reading_order_key(self, evidence: EvidenceExcerpt) -> tuple[int, float, float, int, str]:
        """Keep broad corpus browsing close to the source's reading order.

        Evidence IDs are content hashes and therefore unsuitable as a browsing
        order. Page and timeline locators are preferred when available, with the
        corpus insertion order providing a deterministic fallback for HTML and
        unstructured material.
        """

        locator = evidence.locator

        def numeric(*keys: str) -> float:
            for key in keys:
                value = locator.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return float("inf")

        return (
            self._source_order_by_id.get(evidence.source_id, len(self._source_order_by_id)),
            numeric("page_start", "page"),
            numeric("time_start"),
            self._evidence_order_by_id.get(evidence.evidence_id, len(self._evidence_order_by_id)),
            evidence.evidence_id,
        )

    @staticmethod
    def _reference_catalog_descriptor(catalog: dict[str, Any]) -> dict[str, Any]:
        return {
            "catalog_id": catalog["catalog_id"],
            "source_id": catalog["source_id"],
            "structure": catalog["structure"],
            "detail": catalog["detail"],
            "entry_count": catalog["entry_count"],
            "common_field_labels": catalog["common_field_labels"],
            "suggested_title": catalog["suggested_title"],
        }

    @staticmethod
    def _reference_locator_summary(locator: dict[str, Any]) -> dict[str, Any]:
        """Return only the source coordinates needed for semantic catalog review."""

        return {
            key: locator[key]
            for key in (
                "page",
                "page_start",
                "page_end",
                "heading",
                "section_path",
                "chapter_heading",
                "entry_label",
                "entry_category",
                "url",
            )
            if key in locator
        }

    @staticmethod
    def _reference_entry_descriptor(entry: ReferenceEntry) -> dict[str, Any]:
        return {
            "entry_id": entry.entry_id,
            "title": entry.title,
            "display_title": entry.display_title,
            "source_evidence_id": entry.source_evidence_id,
            "locator": entry.locator,
            "citation_ids": [citation.evidence_id for citation in entry.citations],
            "preview": _excerpt_quote(entry.content, maximum=480),
        }

    @staticmethod
    def _list_candidates(
        items: list[Any], identifier: str, page: int, page_size: int
    ) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = max(1, min(80, int(page_size)))
        items = sorted(items, key=lambda item: getattr(item, identifier))
        start = (page - 1) * page_size
        compact: list[dict[str, Any]] = []
        for item in items[start : start + page_size]:
            dumped = item.model_dump(mode="json")
            citations = (
                [citation.model_dump(mode="json") for citation in _skill_citations(item)]
                if isinstance(item, SkillCandidate)
                else dumped.get("citations", [])
            )
            entry = {
                    identifier: dumped[identifier],
                    "status": dumped.get("status"),
                    "citation_ids": list(dict.fromkeys(citation["evidence_id"] for citation in citations)),
                }
            if isinstance(item, TaskCandidate):
                entry.update(task=item.task, category=item.category)
            elif isinstance(item, TrajectoryCandidate):
                entry.update(task_id=item.task_id)
            else:
                entry["title"] = dumped.get("title", "")
            compact.append(entry)
        return {"page": page, "page_size": page_size, "total": len(items), "items": compact}


class KnowledgeToolExecutor:
    COMMON_TOOLS = {
        "inspect_knowledge_scope",
        "list_evidence",
        "search_evidence",
        "read_evidence",
        "view_evidence_image",
        "submit_stage_result",
    }
    STAGE_TOOLS = {
        "task_trajectory": {
            "inspect_task_outline", "read_task_node", "read_task_nodes",
            "list_references", "read_reference", "search_reference_entries",
            "read_reference_entry", "submit_task_trajectory_pair",
            "inspect_pair_coverage", "supersede_pair", "report_discovery_gap",
        },
        "task_discovery": {
            "inspect_task_outline",
            "read_task_node",
            "read_task_nodes",
            "submit_task_batch",
        },
        "reference_build": {
            "inspect_reference_candidate",
            "inspect_reference_candidates",
            "inspect_reference_catalog",
            "inspect_reference_catalog_batch",
            "submit_reference_review",
        },
        "trajectory_inference": {
            "list_references",
            "read_reference",
            "search_reference_entries",
            "read_reference_entry",
            "submit_trajectory",
            "report_trajectory_gap",
        },
        "skill_induction": {
            "read_task_node",
            "read_task_nodes",
            "list_tasks",
            "read_task",
            "list_trajectories",
            "read_trajectory",
            "submit_skill_batch",
            "review_skill_coverage",
        },
    }

    def __init__(self, workspace: KnowledgeWorkspace, agent_name: str):
        self.workspace = workspace
        self.store = workspace.store
        self.settings = workspace.settings
        self.agent_name = agent_name
        self.allowed_tools = self.COMMON_TOOLS | self.STAGE_TOOLS[workspace.stage]
        if workspace.stage == "trajectory_inference":
            # The dispatched prompt already contains the scoped Task and target.
            self.allowed_tools.discard("inspect_knowledge_scope")
        if workspace.stage == "skill_induction":
            # The dispatch contains the scope, and Skill submissions return coverage.
            self.allowed_tools.discard("inspect_knowledge_scope")
            # Cited Trajectory Evidence is preloaded; further context comes from
            # complete tree nodes, not flat SourcePart search results.
            self.allowed_tools.discard("search_evidence")
            self.allowed_tools.discard("inspect_task_outline")
            tree_sources = {tree.source_id for tree in workspace.corpus.evidence_trees}
            evidence_sources = {item.source_id for item in workspace.corpus.evidence}
            if not evidence_sources - tree_sources:
                # No source needs the flat-Evidence fallback in this Child.
                self.allowed_tools.discard("list_evidence")
                self.allowed_tools.discard("read_evidence")
        if workspace.stage == "reference_build":
            # Reference Build is an exception-review surface. Scope, discovery,
            # automatic materialization, coverage, and finalization are already
            # represented in the prepared manifest and the atomic submit tool.
            self.allowed_tools = {
                "inspect_reference_candidate",
                "inspect_reference_candidates",
                "inspect_reference_catalog",
                "inspect_reference_catalog_batch",
                "read_evidence",
                "submit_reference_review",
            }
        self._reference_candidate_pages_seen: set[int] = set()
        self._reference_catalog_pages_seen: set[int] = set()
        self._reference_catalog_inspections: dict[str, int] = {}
        self._reference_catalog_batch_seen: set[str] = set()
        self._reference_candidate_inspection_count = 0
        self._reference_evidence_reads: set[tuple[str, int]] = set()
        self._reference_scope_inspected = False
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "submit_task_trajectory_pair": lambda task, trajectory, candidate_id=None: self.workspace.submit_task_trajectory_pair(task, trajectory, candidate_id=candidate_id),
            "inspect_pair_coverage": lambda: self.workspace.inspect_pair_coverage(),
            "supersede_pair": lambda old_task_id, new_task_id: self.workspace.supersede_pair(old_task_id, new_task_id),
            "report_discovery_gap": lambda title, reason, citations: self.workspace.report_discovery_gap(title, reason, citations),
            "inspect_knowledge_scope": lambda: {"context": self.workspace.context()},
            "inspect_task_outline": self.workspace.inspect_task_outline,
            "read_task_node": self.workspace.read_task_node,
            "read_task_nodes": self.workspace.read_task_nodes,
            "inspect_task_units": self.workspace.inspect_task_units,
            "read_task_unit": self.workspace.read_task_unit,
            "submit_task_unit_review": self.workspace.submit_task_unit_review,
            "inspect_task_unit_coverage": self.workspace.inspect_task_unit_coverage,
            "inspect_reference_units": self.workspace.inspect_reference_units,
            "submit_reference_reading_plan": self.workspace.submit_reference_reading_plan,
            "submit_reference_unit_result": self.workspace.submit_reference_unit_result,
            "read_reference_unit": self.workspace.read_reference_unit,
            "submit_reference_unit_review": self.workspace.submit_reference_unit_review,
            "inspect_reference_unit_coverage": self.workspace.inspect_reference_unit_coverage,
            "list_reference_candidates": self.workspace.list_reference_candidates,
            "inspect_reference_candidate": self.workspace.inspect_reference_candidate,
            "inspect_reference_candidates": self.workspace.inspect_reference_candidates,
            "list_evidence": self.workspace.list_evidence,
            "search_evidence": self.workspace.search_evidence,
            "read_evidence": self.workspace.read_evidence,
            "view_evidence_image": self.workspace.view_evidence_image,
            "list_reference_catalogs": self.workspace.list_reference_catalogs,
            "inspect_reference_catalog": self.workspace.inspect_reference_catalog,
            "inspect_reference_catalog_batch": self.workspace.inspect_reference_catalog_batch,
            "inspect_reference_coverage": self.workspace.inspect_reference_coverage,
            "review_reference_coverage": self.workspace.review_reference_coverage,
            "submit_reference_review": self.workspace.submit_reference_review,
            "list_tasks": self.workspace.list_tasks,
            "read_task": self.workspace.read_task,
            "list_references": self.workspace.list_references,
            "read_reference": self.workspace.read_reference,
            "search_reference_entries": self.workspace.search_reference_entries,
            "read_reference_entry": self.workspace.read_reference_entry,
            "list_trajectories": self.workspace.list_trajectories,
            "read_trajectory": self.workspace.read_trajectory,
            "submit_task_batch": self.workspace.submit_task_batch,
            "submit_reference_batch": self.workspace.submit_reference_batch,
            "submit_reference_catalog": self.workspace.submit_reference_catalog,
            "submit_reference_catalog_batch": self.workspace.submit_reference_catalog_batch,
            "submit_reference_collection": self.workspace.submit_reference_collection,
            "submit_trajectory": self.workspace.submit_trajectory,
            "report_trajectory_gap": self.workspace.report_trajectory_gap,
            "submit_skill_batch": self.workspace.submit_skill_batch,
            "inspect_skill_coverage": self.workspace.inspect_skill_coverage,
            "review_skill_coverage": self.workspace.review_skill_coverage,
            "submit_stage_result": self._submit_stage_result,
        }

    @property
    def is_finished(self) -> bool:
        return self.workspace.is_finished

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        arguments = dict(arguments or {})
        base_payload = {"stage": self.workspace.stage, "tool": name, "arguments": arguments}
        self.store.emit("knowledge_tool_call", base_payload, agent=self.agent_name)
        if name not in self.allowed_tools:
            result = {"ok": False, "error": "tool_not_allowed", "tool": name}
            self.store.emit(
                "knowledge_tool_policy_denied",
                {"stage": self.workspace.stage, **result},
                agent=self.agent_name,
                level="warning",
            )
            return result
        try:
            guarded = self._guard_reference_review_call(name, arguments)
            if guarded is not None:
                return self._record_observation(name, guarded)
            result = self._handlers[name](**arguments)
            result.setdefault("ok", True)
            result = self._compact_reference_review_result(name, result)
        except Exception as exc:
            result = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        return self._record_observation(name, result)

    def _guard_reference_review_call(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Bound one persistent Reference Child without relying on prompt compliance."""

        if self.workspace.stage != "reference_build":
            return None

        if name == "inspect_knowledge_scope":
            if self._reference_scope_inspected:
                return {
                    "ok": True,
                    "already_provided": True,
                    "message": "The Reference scope is already in this session; continue with candidate review.",
                }
            self._reference_scope_inspected = True
            return None

        if name == "list_reference_candidates":
            arguments["page"] = max(1, int(arguments.get("page", 1)))
            arguments["page_size"] = min(20, max(1, int(arguments.get("page_size", 20))))
            arguments["min_reference_like_score"] = max(2, int(arguments.get("min_reference_like_score", 2)))
            arguments["include_catalog_candidates"] = False
            return None

        if name == "list_reference_catalogs":
            offset = max(0, int(arguments.get("offset", 0)))
            arguments["offset"] = offset
            arguments["limit"] = 20
            if offset in self._reference_catalog_pages_seen:
                return {
                    "ok": True,
                    "already_provided": True,
                    "offset": offset,
                    "message": "This compact catalog page is already in the session; inspect candidate catalogs as needed.",
                }
            self._reference_catalog_pages_seen.add(offset)
            return None

        if name == "inspect_reference_catalog":
            catalog_id = str(arguments.get("catalog_id") or "").strip()
            if catalog_id not in self.workspace.reference_review_catalog_ids:
                return {
                    "ok": False,
                    "error": "catalog_not_reviewable",
                    "catalog_id": catalog_id,
                    "message": "Only catalogs in the prepared candidate manifest may be inspected.",
                }
            arguments["offset"] = max(0, int(arguments.get("offset", 0)))
            arguments["limit"] = min(8, max(1, int(arguments.get("limit", 8))))
            return None

        if name == "inspect_reference_catalog_batch":
            raw_ids = arguments.get("catalog_ids")
            if not isinstance(raw_ids, list):
                return {
                    "ok": False,
                    "error": "invalid_catalog_batch",
                    "message": "catalog_ids must be a non-empty array with at most three unique IDs.",
                }
            catalog_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
            if not catalog_ids or len(catalog_ids) > 3 or len(set(catalog_ids)) != len(catalog_ids):
                return {
                    "ok": False,
                    "error": "invalid_catalog_batch",
                    "message": "catalog_ids must contain one to three unique, non-empty IDs.",
                }
            outside_manifest = sorted(
                set(catalog_ids) - self.workspace.reference_review_catalog_ids
            )
            if outside_manifest:
                return {
                    "ok": False,
                    "error": "catalog_not_reviewable",
                    "catalog_ids": outside_manifest,
                    "message": "Only catalogs in the prepared candidate manifest may be inspected.",
                }
            unseen = [
                catalog_id
                for catalog_id in catalog_ids
                if catalog_id not in self._reference_catalog_batch_seen
            ]
            if not unseen:
                return {
                    "ok": True,
                    "already_provided": True,
                    "catalog_ids": catalog_ids,
                    "message": "All requested catalogs already have stratified samples in this session.",
                }
            self._reference_catalog_batch_seen.update(unseen)
            arguments["catalog_ids"] = unseen
            arguments["entries_per_catalog"] = REFERENCE_CATALOG_SAMPLE_SIZE
            return None

        if name == "inspect_reference_candidate":
            candidate_id = str(arguments.get("candidate_id") or "").strip()
            if candidate_id not in self.workspace.reference_review_candidate_ids:
                return {
                    "ok": False,
                    "error": "candidate_not_reviewable",
                    "candidate_id": candidate_id,
                    "message": "Only ordinary candidates in the prepared manifest may be inspected.",
                }
            arguments["offset"] = max(0, int(arguments.get("offset", 0)))
            arguments["limit"] = min(8, max(1, int(arguments.get("limit", 8))))
            return None

        if name == "inspect_reference_candidates":
            raw_ids = arguments.get("candidate_ids")
            if not isinstance(raw_ids, list):
                return {
                    "ok": False,
                    "error": "invalid_candidate_batch",
                    "message": "candidate_ids must be an array of one to eight IDs.",
                }
            unique_ids: list[str] = []
            invalid_items: list[str] = []
            for item in raw_ids:
                value = item.strip() if isinstance(item, str) else ""
                if not value:
                    invalid_items.append(f"{item!r} (empty or non-string ID)")
                elif value not in unique_ids:
                    unique_ids.append(value)
            if invalid_items or not unique_ids or len(unique_ids) > 8:
                detail = "; ".join(invalid_items) or "batch is empty or exceeds eight IDs"
                return {
                    "ok": False,
                    "error": "invalid_candidate_batch",
                    "message": f"Whole batch rejected; invalid candidate_ids: {detail}.",
                }
            outside_manifest = [
                candidate_id
                for candidate_id in unique_ids
                if candidate_id not in self.workspace.reference_review_candidate_ids
            ]
            if outside_manifest:
                return {
                    "ok": False,
                    "error": "candidates_not_reviewable",
                    "candidate_ids": outside_manifest,
                    "message": (
                        "Whole batch rejected; these candidate IDs are not in the "
                        "prepared manifest: " + ", ".join(outside_manifest)
                    ),
                }
            arguments["candidate_ids"] = unique_ids
            arguments["limit"] = min(8, max(1, int(arguments.get("limit", 8))))
            return None

        if name == "read_evidence":
            evidence_id = str(arguments.get("evidence_id") or "").strip()
            offset = max(0, int(arguments.get("offset", 0)))
            key = (evidence_id, offset)
            if key in self._reference_evidence_reads:
                return {
                    "ok": True,
                    "already_provided": True,
                    "evidence_id": evidence_id,
                    "offset": offset,
                    "message": "This Evidence span is already in the session; reuse the earlier exact text.",
                }
            self._reference_evidence_reads.add(key)
            arguments["offset"] = offset
        return None

    def _compact_reference_review_result(
        self, name: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        if self.workspace.stage != "reference_build" or not result.get("ok", True):
            return result

        def compact_evidence(raw: dict[str, Any]) -> dict[str, Any]:
            compact = {
                key: value
                for key, value in raw.items()
                if key not in {"locator", "image_refs"}
            }
            compact["locator"] = self.workspace._reference_locator_summary(
                dict(raw.get("locator") or {})
            )
            return compact

        if name == "read_evidence" and isinstance(result.get("evidence"), dict):
            result["evidence"] = compact_evidence(result["evidence"])
        elif name == "inspect_reference_candidate":
            result["evidence"] = [
                compact_evidence(item)
                for item in result.get("evidence", [])
                if isinstance(item, dict)
            ]
        elif name == "inspect_reference_candidates":
            for candidate in result.get("candidates", []):
                if isinstance(candidate, dict):
                    candidate["evidence"] = [
                        compact_evidence(item)
                        for item in candidate.get("evidence", [])
                        if isinstance(item, dict)
                    ]
        return result

    def tool_specs(self) -> list[dict[str, Any]]:
        catalog = _knowledge_tool_catalog()
        specs = json.loads(
            json.dumps([catalog[name] for name in sorted(self.allowed_tools)])
        )
        if self.workspace.stage != "trajectory_inference":
            for item in specs:
                name = item["function"]["name"]
                properties = item["function"]["parameters"]["properties"]
                if name in {"list_evidence", "search_evidence"}:
                    properties.pop("search_scope", None)
                elif name == "list_references":
                    properties.pop("scope", None)
        if self.workspace.stage in {"task_discovery", "task_trajectory"}:
            by_name = {item["function"]["name"]: item for item in specs}
            categories = sorted(task_category_evidence_index(
                self.workspace.evidence_by_id, self.workspace.corpus.evidence_trees
            ))
            for name in ("submit_task_batch", "submit_task_trajectory_pair"):
                if name in by_name and categories:
                    properties = by_name[name]["function"]["parameters"]["properties"]
                    task_spec = properties["tasks"]["items"] if name == "submit_task_batch" else properties["task"]
                    task_spec["properties"]["category"]["enum"] = categories
            by_name["read_evidence"]["function"]["description"] = (
                "Read a single exact Evidence excerpt only when its source has no EvidenceTree. "
                "Tree-owned excerpts redirect to read_task_node so the semantic unit is read together. "
                'Required JSON: {"evidence_id":"E1"}; copy a real Evidence ID from list_evidence or search_evidence. '
                "Optional offset and limit count characters, not entries. Never pass a node ID as evidence_id."
            )
            if self.workspace.stage == "task_trajectory":
                read_spec = by_name["read_evidence"]["function"]
                read_spec["description"] = (
                    "Read a complete exact Evidence fallback unit assigned by the host when it is not fully covered by the EvidenceTree. "
                    "Fully tree-covered excerpts redirect to read_task_node. The response contains the entire "
                    "fallback scan unit with no offset or character limit. "
                    'Required JSON: {"evidence_id":"E1"}; copy a real fallback Evidence ID from list_evidence/search_evidence. '
                    'This tool accepts no offset or limit in this stage. Never call with {}. Each batched call '
                    'needs its own evidence_id. Cite the returned E evidence_id, never an N node ID.'
                )
                read_spec["parameters"]["properties"].pop("offset", None)
                read_spec["parameters"]["properties"].pop("limit", None)
            list_properties = by_name["list_evidence"]["function"]["parameters"]["properties"]
            list_properties.pop("reference_like_only", None)
            list_properties.pop("min_reference_like_score", None)
            for tool_name, property_name, maximum in (
                ("list_evidence", "page_size", 24),
                ("search_evidence", "limit", 12),
            ):
                by_name[tool_name]["function"]["parameters"]["properties"][property_name][
                    "maximum"
                ] = maximum
        elif self.workspace.stage == "skill_induction":
            by_name = {item["function"]["name"]: item for item in specs}
            terminal_spec = by_name["submit_stage_result"]["function"]
            terminal_spec["description"] = (
                "Finish this Skill Induction session after all intended submissions. "
                'Required JSON: {"status":"succeeded"}; use degraded or failed when appropriate. '
                "Send status only; do not write a summary or completion report. "
                "The host validates the result and ends the session automatically on acceptance. "
                "If rejected, resolve the returned errors before calling again."
            )
            terminal_spec["parameters"] = {
                "type": "object", "additionalProperties": False,
                "required": ["status"],
                "properties": {
                    "status": terminal_spec["parameters"]["properties"]["status"],
                },
            }
            if "list_evidence" in by_name:
                by_name["list_evidence"]["function"]["description"] = (
                    "List Evidence only for sources_without_tree in the supplied directory. "
                    "For tree sources, select a node ID and call read_task_node."
                )
                by_name["list_evidence"]["function"]["parameters"]["properties"].pop(
                    "branch_node_id", None
                )
            if "read_evidence" in by_name:
                by_name["read_evidence"]["function"]["description"] = (
                    "Read an individual Evidence excerpt only for sources_without_tree. "
                    "Tree-owned excerpts redirect to read_task_node; cited Trajectory Evidence "
                    "is already included in the dispatch. "
                    'Required JSON: {"evidence_id":"E1"}; copy the real E ID from the supplied packet. '
                    "Optional offset and limit count text characters. Never call with {} or a node ID."
                )
        elif self.workspace.stage != "trajectory_inference":
            for item in specs:
                if item["function"]["name"] in {"list_evidence", "search_evidence"}:
                    item["function"]["parameters"]["properties"].pop(
                        "branch_node_id", None
                    )
        if self.workspace.stage == "trajectory_inference":
            by_name = {item["function"]["name"]: item for item in specs}
            for name, field in (("list_evidence", "page_size"),
                                ("search_evidence", "limit")):
                by_name[name]["function"]["parameters"]["properties"][field]["maximum"] = 8
            by_name["read_evidence"]["function"]["parameters"]["properties"]["limit"]["maximum"] = 12_000
            for name, field in (("list_references", "page_size"),
                                ("read_reference", "entry_limit"),
                                ("search_reference_entries", "limit")):
                by_name[name]["function"]["parameters"]["properties"][field]["maximum"] = 8
        if self.workspace.stage == "reference_build":
            by_name = {item["function"]["name"]: item for item in specs}
            for tool_name, property_name, maximum in (
                ("list_reference_catalogs", "limit", 20),
                ("inspect_reference_catalog", "limit", 3),
                (
                    "inspect_reference_catalog_batch",
                    "entries_per_catalog",
                    REFERENCE_CATALOG_SAMPLE_SIZE,
                ),
            ):
                spec = by_name.get(tool_name)
                if spec is not None:
                    spec["function"]["parameters"]["properties"][property_name][
                        "maximum"
                    ] = maximum
                    if tool_name == "inspect_reference_catalog" and property_name == "limit":
                        spec["function"]["parameters"]["properties"]["limit"]["description"] = (
                            "Number of catalog entries to return, from 1 to 3 in Reference Build. "
                            "Not a character count."
                        )
        return specs

    def _submit_stage_result(
        self,
        status: Literal["succeeded", "degraded", "failed"],
        summary: str | None = None,
        gaps: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.workspace.submit_stage_result(
            status=status,
            summary=summary,
            gaps=gaps,
            warnings=warnings,
        )

    def _record_observation(self, tool: str, result: dict[str, Any]) -> dict[str, Any]:
        serialized = json.dumps(result, ensure_ascii=False, default=str)
        returned = result
        if len(serialized) > self.settings.audit_inline_chars:
            artifact = self.store.put_artifact(
                serialized.encode("utf-8"),
                kind="knowledge_tool_observation",
                media_type="application/json",
                suffix=".json",
                metadata={"stage": self.workspace.stage, "tool": tool},
            )
            audit_payload: dict[str, Any] = {
                "stage": self.workspace.stage,
                "tool": tool,
                "ok": result.get("ok", True),
                "observation_artifact_id": artifact.artifact_id,
                "preview": serialized[:1_000],
                "char_count": len(serialized),
            }
        else:
            audit_payload = {"stage": self.workspace.stage, "tool": tool, "observation": result}
        self.store.emit(
            "knowledge_tool_observation",
            audit_payload,
            agent=self.agent_name,
            level="error" if not result.get("ok", True) else "info",
        )
        return returned
