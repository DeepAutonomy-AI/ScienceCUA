from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import model_defaults


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


Modality = Literal["pdf", "html", "github", "unknown"]
SourceStatus = Literal["pending", "processing", "succeeded", "degraded", "excluded", "failed"]
MAX_EVIDENCE_IMAGE_REFS = 200


def normalize_saved_target_profile_payload(value: Any) -> Any:
    """Ignore retired path preferences when projecting historical saved profiles."""
    if not isinstance(value, dict):
        return value
    return {key: item for key, item in value.items()
            if key not in {"operation_path_preference", "operation_modes"}}


class TargetProfile(BaseModel):
    software_name: str = Field(min_length=1, max_length=200)
    software_id: str | None = Field(default=None, max_length=200)
    target_version: str = Field(min_length=1, max_length=100)
    platform: str = Field(min_length=1, max_length=100)
    ui_locale: str = Field(default="unknown", min_length=1, max_length=100)


class StagedInput(BaseModel):
    input_id: str
    kind: Literal["uploaded_file", "url"]
    display_name: str
    user_declared_version: str = "unknown"
    staged_path: str | None = None
    url: str | None = None
    declared_media_type: str | None = None
    source_origin: Literal["user", "websearch"] = "user"
    source_priority: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class InputAlias(BaseModel):
    input_id: str
    kind: Literal["uploaded_file", "url"]
    display_name: str
    location: str | None = None
    user_declared_version: str = "unknown"


class ArtifactRecord(BaseModel):
    artifact_id: str
    sha256: str
    kind: str
    media_type: str
    byte_count: int
    relative_path: str
    source_id: str | None = None
    created_at: str = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceManifest(BaseModel):
    source_id: str
    canonical_input_id: str
    display_name: str
    primary_modality: Modality
    media_type: str
    raw_artifact_id: str | None = None
    content_sha256: str | None = None
    byte_count: int = 0
    status: SourceStatus = "pending"
    aliases: list[InputAlias] = Field(default_factory=list)
    user_declared_version: str = "unknown"
    document_declared_version: str = "unknown"
    resolved_version: str = "unknown"
    version_resolution_basis: Literal["document", "user", "unknown"] = "unknown"
    version_conflict: bool = False
    origin_url: str | None = None
    final_url: str | None = None
    acquisition: dict[str, Any] = Field(default_factory=dict)
    source_origin: Literal["user", "websearch"] = "user"
    source_priority: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    exclusion_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class SourcePart(BaseModel):
    part_id: str
    source_id: str
    kind: str
    label: str
    locator: dict[str, Any]
    parent_part_id: str | None = None
    ordinal: int = 0
    artifact_id: str | None = None
    text_artifact_id: str | None = None
    content_sha256: str | None = None
    char_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceImageRef(BaseModel):
    """One immutable image attachment associated with a semantic Evidence unit."""

    artifact_id: str = Field(pattern=r"^art_[0-9a-f]{24}$")
    source_part_id: str
    media_type: str = Field(pattern=r"^image/")
    page: int = Field(ge=1)
    bboxes: list[list[float]] = Field(default_factory=list, max_length=32)
    pixel_width: int = Field(default=0, ge=0)
    pixel_height: int = Field(default=0, ge=0)
    role: Literal["informative", "decorative", "unknown"] = "unknown"
    association_method: str = Field(default="page_layout_nearest_semantic_unit", max_length=120)
    anchor_offset: int = Field(default=0, ge=0)


class EvidenceExcerpt(BaseModel):
    """A source-grounded semantic unit containing text, images, or both.

    The class name is retained for API compatibility. Image bytes remain in the
    Artifact store and are referenced here instead of being embedded in JSON.
    """

    evidence_id: str
    source_id: str
    source_part_id: str
    text_artifact_id: str | None = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    content_sha256: str
    kind: str
    title: str
    content: str = ""
    image_refs: list[EvidenceImageRef] = Field(
        default_factory=list, max_length=MAX_EVIDENCE_IMAGE_REFS
    )
    locator: dict[str, Any]
    resolved_version: str = "unknown"
    extraction_method: str
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_modal_content(self) -> "EvidenceExcerpt":
        if not self.content.strip() and not self.image_refs:
            raise ValueError("Evidence must contain text or at least one image reference")
        if self.content:
            if not self.text_artifact_id:
                raise ValueError("Text Evidence requires text_artifact_id")
            if self.end_offset <= self.start_offset:
                raise ValueError("Text Evidence requires a positive source span")
        elif self.start_offset != 0 or self.end_offset != 0:
            raise ValueError("Image-only Evidence must use zero text offsets")
        return self


EvidenceNodeType = Literal[
    "document",
    "section",
    "concept",
    "procedure",
    "command_reference",
    "parameter_reference",
    "example",
    "warning",
    "reference",
    "other",
]


class EvidenceSourceRef(BaseModel):
    """An Agent-authored reference to parser-owned text, never rewritten text."""

    model_config = ConfigDict(extra="forbid")

    source_part_id: str = Field(min_length=1, max_length=200)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_optional_span(self) -> "EvidenceSourceRef":
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("start_offset and end_offset must be supplied together")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset <= self.start_offset
        ):
            raise ValueError("end_offset must be greater than start_offset")
        return self


class EvidencePlanNode(BaseModel):
    """One proposed Evidence node. The Agent supplies references, not content."""

    model_config = ConfigDict(extra="forbid")

    node_key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")
    title: str = Field(min_length=1, max_length=300)
    title_origin: Literal["source", "derived"] = "source"
    node_type: EvidenceNodeType = "section"
    parent_node_key: str | None = Field(
        default=None, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$"
    )
    # Agent-authored plans normally contain far fewer references. Python-authored
    # skeleton nodes may intentionally aggregate a complete page, so the canonical
    # contract must not force that page back into artificial 64-part fragments.
    source_refs: list[EvidenceSourceRef] = Field(default_factory=list)


class EvidencePlanExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_part_id: str = Field(min_length=1, max_length=200)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    reason: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def validate_optional_span(self) -> "EvidencePlanExclusion":
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("start_offset and end_offset must be supplied together")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset <= self.start_offset
        ):
            raise ValueError("end_offset must be greater than start_offset")
        return self


class EvidencePlan(BaseModel):
    """Agent-authored assembly instructions for one Source's Evidence tree."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["evidence-plan.v1"] = "evidence-plan.v1"
    source_id: str = Field(min_length=1, max_length=200)
    nodes: list[EvidencePlanNode] = Field(min_length=1)
    excluded_parts: list[EvidencePlanExclusion] = Field(default_factory=list)
    summary: str = Field(default="", max_length=2_000)

    @model_validator(mode="after")
    def validate_tree_shape(self) -> "EvidencePlan":
        keys = [node.node_key for node in self.nodes]
        if len(keys) != len(set(keys)):
            raise ValueError("EvidencePlan node_key values must be unique")
        nodes_by_key = {node.node_key: node for node in self.nodes}
        roots = [node for node in self.nodes if node.parent_node_key is None]
        if len(roots) != 1:
            raise ValueError("EvidencePlan must contain exactly one root node")
        children: dict[str, list[str]] = {key: [] for key in keys}
        for node in self.nodes:
            parent = node.parent_node_key
            if parent is None:
                continue
            if parent == node.node_key:
                raise ValueError(f"Node {node.node_key!r} cannot parent itself")
            if parent not in nodes_by_key:
                raise ValueError(
                    f"Node {node.node_key!r} references unknown parent {parent!r}"
                )
            children[parent].append(node.node_key)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visiting:
                raise ValueError("EvidencePlan parent relationships contain a cycle")
            if key in visited:
                return
            visiting.add(key)
            for child in children[key]:
                visit(child)
            visiting.remove(key)
            visited.add(key)

        visit(roots[0].node_key)
        if len(visited) != len(nodes_by_key):
            raise ValueError("Every EvidencePlan node must be connected to the root")
        for node in self.nodes:
            if not node.source_refs and not children[node.node_key]:
                raise ValueError(
                    f"Leaf node {node.node_key!r} must contain at least one source_ref"
                )
        return self


EvidenceRevisionOperationType = Literal[
    "create_node",
    "update_node",
    "move_node",
    "merge_nodes",
    "exclude_subtree",
    "assign_content",
    "exclude_content",
]


class EvidenceRevisionOperation(BaseModel):
    """One bounded structural edit to a Python-authored Evidence skeleton.

    Raw SourcePart references are deliberately absent. Python exposes opaque
    ``content_key`` values for existing HTML/PDF units. The model may place or
    exclude those registered units, but cannot manufacture or replace content.
    """

    model_config = ConfigDict(extra="forbid")

    operation: EvidenceRevisionOperationType
    node_key: str | None = Field(
        default=None, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$"
    )
    title: str | None = Field(default=None, min_length=1, max_length=300)
    node_type: EvidenceNodeType | None = None
    parent_node_key: str | None = Field(
        default=None, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$"
    )
    before_node_key: str | None = Field(
        default=None, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$"
    )
    source_node_keys: list[str] = Field(default_factory=list, max_length=32)
    content_keys: list[str] = Field(default_factory=list, max_length=64)
    reason: str | None = Field(default=None, min_length=1, max_length=600)

    @model_validator(mode="after")
    def validate_operation_shape(self) -> "EvidenceRevisionOperation":
        if self.operation == "create_node":
            if not self.node_key or not self.title or not self.parent_node_key:
                raise ValueError(
                    "create_node requires node_key, title, and parent_node_key"
                )
            if self.source_node_keys:
                raise ValueError("create_node does not accept source_node_keys")
        elif self.operation == "update_node":
            if not self.node_key or (self.title is None and self.node_type is None):
                raise ValueError("update_node requires node_key and title or node_type")
        elif self.operation == "move_node":
            if not self.node_key or not self.parent_node_key:
                raise ValueError("move_node requires node_key and parent_node_key")
        elif self.operation == "merge_nodes":
            if not self.node_key or not self.source_node_keys:
                raise ValueError("merge_nodes requires a target node_key and source_node_keys")
            if self.node_key in self.source_node_keys:
                raise ValueError("merge_nodes target cannot also be a source node")
            if len(self.source_node_keys) != len(set(self.source_node_keys)):
                raise ValueError("merge_nodes source_node_keys must be unique")
        elif self.operation == "exclude_subtree":
            if not self.node_key or not self.reason:
                raise ValueError("exclude_subtree requires node_key and reason")
        elif self.operation == "assign_content":
            if not self.node_key or not self.content_keys:
                raise ValueError("assign_content requires node_key and content_keys")
        elif self.operation == "exclude_content":
            if not self.content_keys or not self.reason:
                raise ValueError("exclude_content requires content_keys and reason")
            if self.node_key is not None:
                raise ValueError("exclude_content does not accept node_key")
        if self.operation not in {"create_node", "assign_content", "exclude_content"} and self.content_keys:
            raise ValueError(f"{self.operation} does not accept content_keys")
        return self


class EvidenceRevision(BaseModel):
    """Agent-authored incremental edits over one immutable baseline plan."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["evidence-revision.v1"] = "evidence-revision.v1"
    source_id: str = Field(min_length=1, max_length=200)
    base_plan_artifact_id: str = Field(pattern=r"^art_[0-9a-f]{24}$")
    operations: list[EvidenceRevisionOperation] = Field(default_factory=list)
    summary: str = Field(default="", max_length=2_000)


class EvidenceSourceSpan(BaseModel):
    """A Python-resolved exact span used by one materialized Evidence node."""

    source_part_id: str
    text_artifact_id: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locator: dict[str, Any] = Field(default_factory=dict)


class EvidenceNode(BaseModel):
    """A hierarchical Evidence unit materialized only from validated source spans."""

    evidence_node_id: str = Field(pattern=r"^(?:evn_[0-9a-f]{24}|N[1-9][0-9]*)$")
    source_id: str
    node_key: str
    parent_node_id: str | None = None
    child_node_ids: list[str] = Field(default_factory=list)
    ordinal: int = Field(ge=0)
    node_type: EvidenceNodeType
    title: str
    title_origin: Literal["source", "derived"]
    source_refs: list[EvidenceSourceSpan] = Field(default_factory=list)
    content_artifact_id: str | None = None
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    content: str = ""
    direct_char_count: int = Field(default=0, ge=0)
    descendant_char_count: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_materialized_content(self) -> "EvidenceNode":
        if self.source_refs:
            if not self.content or not self.content_artifact_id or not self.content_sha256:
                raise ValueError("A sourced EvidenceNode requires materialized content")
        elif self.content or self.content_artifact_id or self.content_sha256:
            raise ValueError("A structural EvidenceNode cannot contain ungrounded content")
        if not self.source_refs and not self.child_node_ids:
            raise ValueError("An EvidenceNode must contain source_refs or child nodes")
        return self


class EvidenceTree(BaseModel):
    schema_version: Literal["evidence-tree.v1"] = "evidence-tree.v1"
    evidence_tree_id: str = Field(pattern=r"^(?:evt_[0-9a-f]{24}|ET[1-9][0-9]*)$")
    source_id: str
    root_node_id: str
    nodes: list[EvidenceNode]
    plan_artifact_id: str
    validation_artifact_id: str
    excluded_parts: list[EvidencePlanExclusion] = Field(default_factory=list)
    referenced_part_count: int = Field(default=0, ge=0)
    leaf_count: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_materialized_tree(self) -> "EvidenceTree":
        nodes_by_id = {node.evidence_node_id: node for node in self.nodes}
        if len(nodes_by_id) != len(self.nodes):
            raise ValueError("EvidenceTree node IDs must be unique")
        root = nodes_by_id.get(self.root_node_id)
        if root is None or root.parent_node_id is not None:
            raise ValueError("EvidenceTree root_node_id must identify the root node")
        for node in self.nodes:
            if node.parent_node_id is not None:
                parent = nodes_by_id.get(node.parent_node_id)
                if parent is None:
                    raise ValueError("EvidenceNode parent_node_id is not in this tree")
                if node.evidence_node_id not in parent.child_node_ids:
                    raise ValueError("EvidenceTree parent/child links must be reciprocal")
            for child_id in node.child_node_ids:
                child = nodes_by_id.get(child_id)
                if child is None or child.parent_node_id != node.evidence_node_id:
                    raise ValueError("EvidenceTree child/parent links must be reciprocal")

        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("EvidenceTree contains a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for child_id in nodes_by_id[node_id].child_node_ids:
                visit(child_id)
            visiting.remove(node_id)
            visited.add(node_id)

        visit(self.root_node_id)
        if len(visited) != len(self.nodes):
            raise ValueError("Every EvidenceTree node must be connected to the root")
        return self


class EvidencePlanIssue(BaseModel):
    code: str
    message: str
    node_key: str | None = None
    source_part_id: str | None = None


class EvidencePlanValidationReport(BaseModel):
    schema_version: Literal["evidence-plan-validation.v1"] = (
        "evidence-plan-validation.v1"
    )
    source_id: str
    valid: bool
    issues: list[EvidencePlanIssue] = Field(default_factory=list)
    eligible_part_count: int = Field(default=0, ge=0)
    referenced_part_count: int = Field(default=0, ge=0)
    excluded_part_count: int = Field(default=0, ge=0)
    coverage_complete: bool = False


class Coverage(BaseModel):
    discovered_parts: int = 0
    inventoried_parts: int = 0
    normalized_parts: int = 0
    semantic_units: int = 0
    inspected_parts: int = 0
    evidence_bearing_parts: int = 0
    skipped_parts: int = 0
    skipped: list[dict[str, Any]] = Field(default_factory=list)


class SourceResult(BaseModel):
    source_id: str
    display_name: str
    modality: Modality
    status: SourceStatus
    summary: str = ""
    evidence: list[EvidenceExcerpt] = Field(default_factory=list)
    evidence_tree: EvidenceTree | None = None
    corpus_artifact_id: str | None = None
    source_part_count: int = 0
    normalization_version: str = "v1"
    coverage: Coverage = Field(default_factory=Coverage)
    gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    manifest: SourceManifest | None = None
    completed_at: str = Field(default_factory=utc_now)


class CorpusSnapshot(BaseModel):
    corpus_snapshot_id: str
    run_id: str
    target_profile: TargetProfile
    source_results: list[SourceResult]
    evidence: list[EvidenceExcerpt]
    evidence_trees: list[EvidenceTree] = Field(default_factory=list)
    evidence_count: int
    source_part_count: int
    source_status_counts: dict[str, int]
    created_at: str = Field(default_factory=utc_now)


# Downstream candidates are intentionally separate from Evidence. Evidence is exact,
# parser-materialized source text; the following objects are model-produced and must
# retain nonempty quotes or image references linked to existing Evidence.
# Quote text is stored for traceability without exact substring matching.
CandidateStatus = Literal["candidate", "validated", "rejected", "committed"]


class EvidenceCitation(BaseModel):
    evidence_id: str = Field(pattern=r"^(?:evi_[A-Za-z0-9_-]+|E[1-9][0-9]*)$")
    quote: str = Field(default="", max_length=1_600)
    image_artifact_id: str | None = Field(default=None, pattern=r"^art_[0-9a-f]{24}$")

    @field_validator("image_artifact_id", mode="before")
    @classmethod
    def empty_optional_image_id(cls, value: Any) -> Any:
        # Model-generated tool calls sometimes include an empty optional field.
        # A text quote is sufficient support and needs no image Artifact.
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def require_text_or_image_support(self) -> "EvidenceCitation":
        if not self.quote.strip() and not self.image_artifact_id:
            raise ValueError("Citation requires nonempty quote text or image_artifact_id")
        return self


class TaskDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=3, max_length=300)
    category: str = Field(min_length=1, description="Exact first-level chapter title from the EvidenceTree directory. Validated against the frozen corpus and cited Evidence.")
    citations: list[EvidenceCitation] = Field(min_length=1, max_length=24)


def normalize_saved_task_payload(value: Any) -> Any:
    """Project historical saved Tasks into the current shape without changing files."""
    if not isinstance(value, dict):
        return value
    migrated = dict(value)
    if "task" not in migrated and "title" in migrated:
        migrated["task"] = migrated["title"]
        migrated.setdefault("category", "unknown")
    for key in ("title", "objective", "confidence", "caveats", "operation_mode"):
        migrated.pop(key, None)
    return normalize_saved_citations(migrated)


class TaskCandidate(TaskDraft):
    task_id: str = Field(pattern=r"^(?:tsk_[A-Za-z0-9_-]+|T[1-9][0-9]*)$")
    corpus_snapshot_id: str
    status: CandidateStatus = "candidate"
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_task(cls, value: Any) -> Any:
        return normalize_saved_task_payload(value)


ReferenceMode = Literal["summary", "source_catalog"]
ReferenceInjectionPolicy = Literal["always", "on_demand", "contextual"]


def _normalize_usage_tag(value: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", value.strip().casefold())
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _normalize_usage_tags(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items = [value] if isinstance(value, str) else value
    if not isinstance(raw_items, list):
        raise ValueError("usage_tags must be a list of strings")
    normalized: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, str):
            raise ValueError("usage_tags entries must be strings")
        tag = _normalize_usage_tag(raw_item)
        if not tag:
            continue
        if len(tag) > 80:
            raise ValueError("usage_tags entries must be 80 characters or shorter")
        normalized.append(tag)
    return list(dict.fromkeys(normalized))


class ReferenceImageRef(BaseModel):
    """An image retained from cited Evidence, scoped to one Reference reading unit."""

    artifact_id: str = Field(pattern=r"^art_[0-9a-f]{24}$")
    evidence_id: str
    media_type: str = Field(pattern=r"^image/")
    marker: str = ""


class ReferenceEntry(BaseModel):
    """One source-preserved entry within a queryable Reference catalog."""

    entry_id: str = Field(pattern=r"^(?:rfe_[A-Za-z0-9_-]+|RE[1-9][0-9]*)$")
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=40_000)
    source_evidence_id: str = Field(pattern=r"^(?:evi_[A-Za-z0-9_-]+|E[1-9][0-9]*)$")
    locator: dict[str, Any] = Field(default_factory=dict)
    citations: list[EvidenceCitation] = Field(min_length=1, max_length=8)
    image_refs: list[ReferenceImageRef] = Field(default_factory=list)
    display_title: str = ""


class ReferenceDraft(BaseModel):
    title: str = Field(min_length=3, max_length=300)
    description: str = Field(min_length=1, max_length=120)
    category: str = Field(default="reference", min_length=1, max_length=120)
    content: str = Field(default="", max_length=8_000)
    reference_mode: ReferenceMode = "summary"
    injection_policy: ReferenceInjectionPolicy = "on_demand"
    usage_tags: list[str] = Field(default_factory=list, max_length=16)
    source_catalog_id: str | None = Field(default=None, max_length=160)
    # Large authored indexes remain one logical Reference. Retrieval is already
    # paged/search-based, so an arbitrary 512-entry validation ceiling must not
    # split or reject an otherwise coherent source catalog.
    entries: list[ReferenceEntry] = Field(default_factory=list, max_length=4_096)
    citations: list[EvidenceCitation] = Field(min_length=1, max_length=512)
    image_refs: list[ReferenceImageRef] = Field(default_factory=list)

    @field_validator("description")
    @classmethod
    def require_description(cls, value: str) -> str:
        description = value.strip()
        if not description:
            raise ValueError("Reference description must contain knowledge scope and lookup guidance")
        return description

    @field_validator("usage_tags", mode="before")
    @classmethod
    def normalize_usage_tags(cls, value: Any) -> list[str]:
        return _normalize_usage_tags(value)

    @model_validator(mode="after")
    def require_summary_content(self) -> "ReferenceDraft":
        if self.reference_mode == "summary" and not self.content.strip():
            raise ValueError("Summary Reference requires nonempty knowledge content")
        return self


class ReferencePresentation(BaseModel):
    """Validated layout derived from immutable source units, alongside originals."""

    mode: Literal["agent_formatted"] = "agent_formatted"
    content: str = Field(min_length=1, max_length=500_000)
    source_artifact_id: str = Field(pattern=r"^art_[0-9a-f]{24}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_ids: list[str] = Field(min_length=1)


class ReferenceCandidate(ReferenceDraft):
    presentation: ReferencePresentation | None = None
    # Saved References keep their original descriptions; new submissions allow 120 characters while prompts request 100.
    description: str = Field(min_length=1, max_length=1_024)
    reference_id: str = Field(pattern=r"^(?:ref_[A-Za-z0-9_-]+|R[1-9][0-9]*)$")
    corpus_snapshot_id: str
    status: CandidateStatus = "candidate"
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_description(cls, value: Any) -> Any:
        """Read saved References whose lookup description was stored in content."""
        if not isinstance(value, dict) or "description" in value:
            return value
        migrated = dict(value)
        title = str(value.get("title") or "Reference")
        content = str(value.get("content") or "")
        if value.get("reference_mode") == "source_catalog":
            legacy_description = content.strip() or title
            # Older catalog intros allowed up to 8,000 characters. Keep the full
            # text when it cannot fit in discovery metadata, so reading or
            # re-exporting a saved Reference never discards the remainder.
            migrated["content"] = (
                content if len(" ".join(legacy_description.split())) > 1_024 else ""
            )
        else:
            legacy_description = f"{title}: {content}" if content.strip() else title
        # Only historical Candidate reads get this compatibility description.
        # New Drafts and explicitly empty descriptions must fail validation.
        migrated["description"] = " ".join(legacy_description.split())[:1_024]
        return migrated


class TrajectoryStepDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordinal: int = Field(ge=1, le=100)
    instruction: str = Field(min_length=3, max_length=2_000)
    expected_observation: str = Field(default="", max_length=1_000)
    citations: list[EvidenceCitation] = Field(min_length=1, max_length=16)


class TrajectoryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(pattern=r"^(?:tsk_[A-Za-z0-9_-]+|T[1-9][0-9]*)$")
    steps: list[TrajectoryStepDraft] = Field(min_length=1, max_length=40)
    citations: list[EvidenceCitation] = Field(min_length=1, max_length=32)


RETIRED_TRAJECTORY_FIELDS = frozenset({
    "goal", "trajectory_type", "reference_ids", "reference_trigger_type", "confidence", "caveats", "unsupported_facets",
})


def normalize_saved_trajectory_payload(value: Any) -> Any:
    """Read historical trajectories while keeping Task binding and step Evidence."""
    if not isinstance(value, dict):
        return value
    return normalize_saved_citations({key: item for key, item in value.items()
                                      if key not in RETIRED_TRAJECTORY_FIELDS})


class TrajectoryCandidate(TrajectoryDraft):
    trajectory_id: str = Field(pattern=r"^(?:traj_[A-Za-z0-9_-]+|TR[1-9][0-9]*)$")
    knowledge_snapshot_id: str
    status: CandidateStatus = "candidate"
    inferred: bool = True
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_trajectory(cls, value: Any) -> Any:
        return normalize_saved_trajectory_payload(value)


class SkillProcedureStepDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordinal: int = Field(ge=1, le=100)
    instruction: str = Field(min_length=3, max_length=2_000)
    citations: list[EvidenceCitation] = Field(min_length=1, max_length=24)


class SkillDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=3, max_length=300)
    description: str = Field(min_length=3, max_length=100)
    applicability: str = Field(min_length=1, max_length=2_000)
    preconditions: str = Field(
        max_length=24_016,
        description="Required requirements before the first procedure step, combined into one string. Submit an empty string explicitly when none are required.",
    )
    procedure: list[SkillProcedureStepDraft] = Field(min_length=1, max_length=50)
    trajectory_ids: list[str] = Field(min_length=1, max_length=64)


RETIRED_SKILL_FIELDS = frozenset({
    "citations", "reference_ids", "reference_trigger_type", "task_ids",
    "confidence", "caveats", "inferred", "completion_criteria",
})


def normalize_saved_citations(value: Any) -> Any:
    """Drop retired citation explanations without changing coverage-review rationales."""
    if isinstance(value, list):
        return [normalize_saved_citations(item) for item in value]
    if isinstance(value, dict):
        return {
            key: normalize_saved_citations(item) for key, item in value.items()
            if not (key == "rationale" and "evidence_id" in value)
        }
    return value


def normalize_saved_skill_payload(value: Any) -> Any:
    """Read legacy Skill layouts without changing the saved input."""
    if not isinstance(value, dict):
        return value
    normalized = {
        key: item for key, item in value.items()
        if key not in RETIRED_SKILL_FIELDS
    }
    # Saved Skills predate the required field. New SkillDraft submissions never
    # pass through this migration and must supply the field explicitly.
    normalized.setdefault("preconditions", "")
    preconditions = normalized["preconditions"]
    if isinstance(preconditions, list) and all(
        isinstance(item, dict) and isinstance(item.get("criterion"), str)
        for item in preconditions
    ):
        normalized["preconditions"] = "\n".join(item["criterion"] for item in preconditions)
    if isinstance(normalized.get("procedure"), list):
        normalized["procedure"] = [
            {key: item for key, item in step.items() if key != "expected_observation"}
            if isinstance(step, dict) else step
            for step in normalized["procedure"]
        ]
    return normalize_saved_citations(normalized)


class SkillCandidate(SkillDraft):
    # Keep historical Skills readable; every new submission validates SkillDraft first.
    description: str = Field(min_length=3, max_length=4_000)
    skill_id: str = Field(pattern=r"^(?:skl_[A-Za-z0-9_-]+|SK[1-9][0-9]*)$")
    knowledge_snapshot_id: str
    status: CandidateStatus = "candidate"
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_layout(cls, value: Any) -> Any:
        """Read saved Skills using the current procedure, precondition and citation layout."""
        return normalize_saved_skill_payload(value)


class CandidateValidation(BaseModel):
    candidate_kind: Literal["task", "reference", "trajectory", "skill"]
    candidate_id: str
    accepted: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class KnowledgeSnapshot(BaseModel):
    knowledge_snapshot_id: str
    run_id: str
    corpus_snapshot_id: str
    corpus_snapshot_artifact_id: str
    target_profile: TargetProfile
    tasks: list[TaskCandidate] = Field(default_factory=list)
    references: list[ReferenceCandidate] = Field(default_factory=list)
    task_validation: list[CandidateValidation] = Field(default_factory=list)
    reference_validation: list[CandidateValidation] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class ModelRuntimeConfig(BaseModel):
    mode: Literal["api"] = "api"
    api_protocol: Literal["chat_completions", "responses"] = model_defaults.api_protocol
    base_url: str = model_defaults.base_url
    model: str = model_defaults.model
    api_key: str | None = Field(default=None, repr=False)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = model_defaults.reasoning_effort
    websearch: bool = False

    @model_validator(mode="before")
    @classmethod
    def resolve_default_protocol(cls, value: Any) -> Any:
        if isinstance(value, dict) and not value.get("api_protocol"):
            normalized = dict(value)
            normalized["api_protocol"] = model_defaults.protocol_for(
                str(value.get("base_url", model_defaults.base_url))
            )
            return normalized
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("base_url cannot contain embedded credentials")
        return value.strip().rstrip("/")

    def public_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "api_protocol": self.api_protocol,
            "base_url": self.base_url,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "websearch": self.websearch,
            "api_key_configured": bool(self.api_key),
        }
