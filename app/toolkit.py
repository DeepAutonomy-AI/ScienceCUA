"""Source tool dispatch and shared Evidence review.

Modality parsing, workspace persistence, and tool schemas live in source_tools.
Explicit re-exports keep existing app.toolkit imports available.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Callable

from .models import (
    EvidenceImageRef,
    EvidencePlan,
    EvidenceRevision,
    SourcePart,
    SourceResult,
)
from .source_tools.html import (
    HTML_HEADING_TAGS,
    HTML_INLINE_HEADING_TAGS,
    HTML_SECTION_HEADING_TAGS,
    HtmlToolsMixin,
)
from .source_tools.pdf import (
    PDF_VISUAL_GROUP_MIN_IMAGES,
    PDF_VISUAL_GROUP_PADDING,
    PDF_VISUAL_GROUP_RENDER_SCALE,
    PdfToolsMixin,
)
from .source_tools.pdf_structure import (
    _classify_pdf_image,
    _clip_pdf_bbox,
    _expand_pdf_bbox,
    _is_pdf_contents_page,
    _is_pdf_furniture,
    _is_pdf_margin_image_furniture,
    _join_pdf_block_texts,
    _pdf_bbox_area,
    _pdf_body_font_size,
    _pdf_bookmark_outline,
    _pdf_enrich_generic_heading,
    _pdf_find_heading_parent,
    _pdf_heading_candidates,
    _pdf_heading_outline,
    _pdf_heading_parent_matches,
    _pdf_heading_text,
    _pdf_inline_text,
    _pdf_match_block_style,
    _pdf_page_styled_blocks,
    _pdf_printed_toc_entries,
    _pdf_repeated_furniture_texts,
    _pdf_section_is_substantive,
    _pdf_text_key,
    _pdf_tiny_display_image_reason,
    _union_bboxes,
    _vertical_bbox_distance,
)
from .source_tools.schemas import _function, _tool_spec_catalog
from .source_tools.text import (
    _compact_text,
    _excerpt_ranges,
    _split_long_text,
    _stable_id,
)
from .source_tools.workspace import SourceWorkspace


class ToolExecutor(HtmlToolsMixin, PdfToolsMixin):
    # Default T5 tool surface. The Child understands authored information
    # architecture and submits assembly instructions; Python owns parsing,
    # validation, hashing, and canonical Evidence text.
    ARCHITECT_REVIEW_TOOLS = {
        "inspect_source",
        "get_root_structure",
        "read_evidence_node",
        "submit_evidence_revision",
    }
    ARCHITECT_PDF_REVIEW_TOOLS = {
        "inspect_source",
        "get_pdf_document_structure",
        "read_pdf_content",
        "read_evidence_node",
        "submit_evidence_revision",
    }
    # Retained comparison route in which the model constructs the complete plan.
    FULL_PLAN_REVIEW_TOOLS = {
        "inspect_source",
        "get_source_outline",
        "read_source_content",
        "search_source",
        "submit_evidence_plan",
    }
    # The original T5 Child received seven common review tools plus exactly
    # one modality-specific structure/navigation tool. Inventory remains a
    # deterministic parser responsibility and is never granted to the Child.
    LEGACY_REVIEW_COMMON_TOOLS = {
        "inspect_source",
        "read_artifact",
        "list_source_parts",
        "read_source_part",
        "search_source",
        "submit_source_structure",
        "submit_source_result",
    }
    LEGACY_REVIEW_MODAL_TOOLS = {
        "pdf": {"pdf_list_structure"},
        "html": {"html_list_navigation"},
    }
    # T5.1's retained compact/curated comparison profile.
    CURATED_REVIEW_TOOLS = {
        "inspect_source",
        "list_source_parts",
        "read_source_part",
        "search_source",
        "curate_evidence",
        "submit_source_result",
    }
    # Compatibility alias now denotes the default Agent surface.
    REVIEW_TOOLS = ARCHITECT_REVIEW_TOOLS
    ALL_REVIEW_TOOLS = (
        ARCHITECT_REVIEW_TOOLS
        | ARCHITECT_PDF_REVIEW_TOOLS
        | FULL_PLAN_REVIEW_TOOLS
        | LEGACY_REVIEW_COMMON_TOOLS
        | CURATED_REVIEW_TOOLS
        | LEGACY_REVIEW_MODAL_TOOLS["pdf"]
        | LEGACY_REVIEW_MODAL_TOOLS["html"]
    )
    COMMON_TOOLS = {
        "inspect_source",
        "read_artifact",
        "list_source_parts",
        "read_source_part",
        "search_source",
        "curate_evidence",
        "submit_source_structure",
        "submit_source_result",
        "get_source_outline",
        "read_source_content",
        "submit_evidence_plan",
        "get_root_structure",
        "get_pdf_document_structure",
        "get_evidence_skeleton",
        "read_pdf_content",
        "read_evidence_node",
        "submit_evidence_revision",
    }
    MODAL_TOOLS = {
        "pdf": {"pdf_inventory", "pdf_list_structure"},
        "html": {"html_inventory", "html_list_navigation"},
    }

    def __init__(
        self,
        workspace: SourceWorkspace,
        agent_name: str,
        *,
        review_profile: str = "architect",
    ):
        if review_profile not in {"architect", "planner", "legacy", "curated"}:
            raise ValueError(f"Unknown Source Review profile: {review_profile!r}")
        self.workspace = workspace
        self.store = workspace.store
        self.settings = workspace.settings
        self.agent_name = agent_name
        self.review_profile = review_profile
        self._pdf_structure_inspected = False
        self._pdf_unassigned_seen: set[int] = set()
        self._pdf_unmatched_seen: set[int] = set()
        self._pdf_required_unassigned_count = 0
        self._pdf_required_unmatched_count = 0
        self._allow_unreviewed_pdf_fallback = False
        self.allowed_tools = self.COMMON_TOOLS | self.MODAL_TOOLS.get(
            workspace.manifest.primary_modality, set()
        )
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "inspect_source": self._inspect_source,
            "read_artifact": self._read_artifact,
            "list_source_parts": self._list_source_parts,
            "read_source_part": self._read_source_part,
            "search_source": self._search_source,
            "curate_evidence": self._curate_evidence,
            "submit_source_structure": self._submit_source_structure,
            "submit_source_result": self._submit_source_result,
            "get_source_outline": self._get_source_outline,
            "read_source_content": self._read_source_content,
            "submit_evidence_plan": self._submit_evidence_plan,
            "get_root_structure": self._get_root_structure,
            "get_pdf_document_structure": self._get_pdf_document_structure,
            "get_evidence_skeleton": self._get_evidence_skeleton,
            "read_pdf_content": self._read_pdf_content,
            "read_evidence_node": self._read_evidence_node,
            "submit_evidence_revision": self._submit_evidence_revision,
            "pdf_inventory": self._pdf_inventory,
            "pdf_list_structure": self._pdf_list_structure,
            "html_inventory": self._html_inventory,
            "html_list_navigation": self._html_list_navigation,
        }

    @property
    def source_review_tools(self) -> set[str]:
        if self.review_profile == "architect":
            if self.workspace.manifest.primary_modality == "pdf":
                return set(self.ARCHITECT_PDF_REVIEW_TOOLS)
            return set(self.ARCHITECT_REVIEW_TOOLS)
        if self.review_profile == "planner":
            return set(self.FULL_PLAN_REVIEW_TOOLS)
        if self.review_profile == "curated":
            return set(self.CURATED_REVIEW_TOOLS)
        modality = self.workspace.manifest.primary_modality
        return set(self.LEGACY_REVIEW_COMMON_TOOLS) | set(
            self.LEGACY_REVIEW_MODAL_TOOLS.get(modality, set())
        )

    @property
    def is_finished(self) -> bool:
        return self.workspace.final_result is not None

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments or {}
        source_id = self.workspace.manifest.source_id
        self.store.emit(
            "tool_call",
            {"tool": name, "arguments": args},
            source_id=source_id,
            agent=self.agent_name,
        )
        if name not in self.allowed_tools:
            result = {"ok": False, "error": "tool_not_allowed", "tool": name}
            self.store.emit(
                "tool_policy_denied",
                result,
                source_id=source_id,
                agent=self.agent_name,
                level="warning",
            )
            return result
        self.store.emit(
            "tool_policy_allowed",
            {"tool": name, "source_scope": source_id},
            source_id=source_id,
            agent=self.agent_name,
        )
        try:
            result = self._handlers[name](**args)
            result.setdefault("ok", True)
        except Exception as exc:
            result = {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }
        return self._record_observation(name, result)

    def force_result(self, status: str, summary: str, gaps: list[str], errors: list[str]) -> SourceResult:
        self.workspace.materialize_evidence_excerpts()
        if status == "succeeded" and not self.workspace.evidence:
            status = "degraded"
        self.workspace.manifest.status = status
        corpus_artifact_id = self.workspace.build_corpus_artifact()
        result = SourceResult(
            source_id=self.workspace.manifest.source_id,
            display_name=self.workspace.manifest.display_name,
            modality=self.workspace.manifest.primary_modality,
            status=status,
            summary=summary,
            evidence=list(self.workspace.evidence.values()),
            corpus_artifact_id=corpus_artifact_id,
            source_part_count=len(self.workspace.parts),
            normalization_version=self.workspace.NORMALIZATION_VERSION,
            coverage=self.workspace.coverage(),
            gaps=list(dict.fromkeys(self.workspace.gaps + gaps)),
            warnings=list(dict.fromkeys(self.workspace.warnings)),
            errors=errors,
            manifest=self.workspace.manifest,
        )
        self.workspace.final_result = result
        self.workspace.persist()
        self.store.write_source_json(
            result.source_id,
            "result.json",
            result.model_dump(mode="json"),
        )
        self.store.emit(
            "source_result_forced",
            {"status": status, "summary": summary, "gaps": gaps, "errors": errors},
            source_id=result.source_id,
            agent=self.agent_name,
            level="warning" if status != "succeeded" else "info",
        )
        return result

    def tool_specs(self) -> list[dict[str, Any]]:
        specs = _tool_spec_catalog()
        return [specs[name] for name in sorted(self.allowed_tools)]

    def _record_observation(self, tool: str, result: dict[str, Any]) -> dict[str, Any]:
        serialized = json.dumps(result, ensure_ascii=False, default=str)
        source_id = self.workspace.manifest.source_id
        audit_payload: dict[str, Any]
        returned = result
        if len(serialized) > self.settings.audit_inline_chars:
            artifact = self.store.put_artifact(
                serialized.encode("utf-8"),
                kind="tool_observation",
                media_type="application/json",
                suffix=".json",
                source_id=source_id,
                metadata={"tool": tool},
            )
            self.workspace.allowed_artifacts.add(artifact.artifact_id)
            audit_payload = {
                "tool": tool,
                "ok": result.get("ok", True),
                "observation_artifact_id": artifact.artifact_id,
                "preview": serialized[:1000],
                "char_count": len(serialized),
            }
        else:
            audit_payload = {"tool": tool, "observation": result}
        self.store.emit(
            "tool_observation",
            audit_payload,
            source_id=source_id,
            agent=self.agent_name,
            level="error" if not result.get("ok", True) else "info",
        )
        return returned

    def _inspect_source(self) -> dict[str, Any]:
        manifest = self.workspace.manifest.model_dump(mode="json")
        candidates = self._architect_parts()
        if self.review_profile == "architect":
            return {
                "manifest": {
                    "source_id": manifest.get("source_id"),
                    "display_name": manifest.get("display_name"),
                    "primary_modality": manifest.get("primary_modality"),
                    "source_origin": manifest.get("source_origin"),
                    "origin_url": manifest.get("origin_url"),
                    "final_url": manifest.get("final_url"),
                    "declared_version": manifest.get("declared_version"),
                    "detected_version": manifest.get("detected_version"),
                    "resolved_version": manifest.get("resolved_version"),
                    "version_conflict": manifest.get("version_conflict"),
                },
                "target_profile": self.workspace.target_profile.model_dump(mode="json"),
                "allowed_tools": sorted(self.source_review_tools),
                "workspace": {
                    "part_count": len(self.workspace.parts),
                    "evidence_candidate_count": len(candidates),
                    "candidate_page_count": len(
                        {self._candidate_page_key(part) for part in candidates}
                    ),
                    "inventory_completed": self.workspace.inventory_completed,
                    "gap_count": len(self.workspace.gaps),
                    "warning_count": len(self.workspace.warnings),
                },
                "evidence_architecture": {
                    "root_structure_artifact_id": self.workspace.root_structure_artifact_id,
                    "pdf_document_structure_artifact_id": (
                        self.workspace.pdf_document_structure_artifact_id
                    ),
                    "base_plan_artifact_id": (
                        self.workspace.evidence_skeleton_plan_artifact_id
                    ),
                    "candidate_tree_id": (
                        self.workspace.evidence_tree.evidence_tree_id
                        if self.workspace.evidence_tree is not None
                        else None
                    ),
                    "content_ownership": (
                        "Python owns exact Evidence content and complete SourcePart coverage; "
                        "the model may revise hierarchy and place only parser-issued "
                        "opaque PDF content keys."
                    ),
                    "next_tool": (
                        "get_pdf_document_structure"
                        if self.workspace.manifest.primary_modality == "pdf"
                        else "get_root_structure"
                    ),
                    "completion_tool": "submit_evidence_revision",
                },
            }
        page_profiles: dict[str, dict[str, Any]] = {}
        for part in candidates:
            page_key = self._candidate_page_key(part)
            profile = page_profiles.setdefault(
                page_key,
                {
                    "page_key": page_key,
                    "url": part.locator.get("url"),
                    "page": part.locator.get("page") or part.locator.get("page_start"),
                    "candidate_count": 0,
                    "candidate_chars": 0,
                    "labels": [],
                },
            )
            profile["candidate_count"] += 1
            profile["candidate_chars"] += part.char_count
            if len(profile["labels"]) < 4 and part.label not in profile["labels"]:
                profile["labels"].append(part.label)
        profile_values = list(page_profiles.values())
        authored_navigation_groups: list[dict[str, Any]] = []
        if self.workspace.manifest.primary_modality == "html":
            html_pages = sorted(
                (
                    part
                    for part in self.workspace.parts.values()
                    if part.kind == "html_page"
                ),
                key=lambda item: (item.ordinal, item.part_id),
            )
            root_page = next(
                (
                    part
                    for part in html_pages
                    if int(part.metadata.get("navigation_depth", 0)) == 0
                ),
                html_pages[0] if html_pages else None,
            )
            fetched_by_label = Counter(
                _compact_text(str(part.metadata.get("capability_branch") or ""))
                for part in html_pages
                if int(part.metadata.get("navigation_depth", 0)) > 0
            )
            for group in (
                root_page.metadata.get("navigation_groups", [])
                if root_page is not None
                else []
            ):
                label = _compact_text(str(group.get("label") or ""))
                authored_labels = [
                    _compact_text(str(item))
                    for item in group.get("labels", [])
                    if _compact_text(str(item))
                ]
                authored_navigation_groups.append(
                    {
                        "label": label,
                        "authored_labels": authored_labels,
                        "link_count": int(group.get("link_count", 0)),
                        "fetched_page_count": sum(
                            int(fetched_by_label.get(item, 0))
                            for item in (authored_labels or [label])
                        ),
                        "container_tag": group.get("container_tag"),
                    }
                )
        return {
            "manifest": manifest,
            "target_profile": self.workspace.target_profile.model_dump(mode="json"),
            "allowed_tools": sorted(self.source_review_tools),
            "workspace": {
                "part_count": len(self.workspace.parts),
                "normalized_part_count": len(self.workspace.normalized_parts),
                "evidence_candidate_count": len(candidates),
                "page_count": len(page_profiles),
                "evidence_count": len(self.workspace.evidence),
                "inventory_completed": self.workspace.inventory_completed,
                "corpus_artifact_id": self.workspace.corpus_artifact_id,
                "root_structure_artifact_id": self.workspace.root_structure_artifact_id,
            },
            "candidate_profile": profile_values[:60],
            "candidate_profile_truncated": len(profile_values) > 60,
            "evidence_architecture": {
                "primary_structure": (
                    "Python has materialized a complete-coverage candidate Evidence "
                    "skeleton. For HTML, the root-structure artifact is the neutral "
                    "authored input from which the model decides semantic hierarchy."
                ),
                "content_ownership": (
                    "The model may submit bounded structural edits but cannot supply or "
                    "replace Evidence text or SourcePart references."
                ),
                "skeleton_tree_id": (
                    self.workspace.evidence_tree.evidence_tree_id
                    if self.workspace.evidence_tree is not None
                    else None
                ),
                "base_plan_artifact_id": self.workspace.evidence_skeleton_plan_artifact_id,
                "root_structure_artifact_id": self.workspace.root_structure_artifact_id,
                "authored_navigation_groups": authored_navigation_groups,
                "completion_tool": "submit_evidence_revision",
            },
        }

    def _architect_parts(self) -> list[SourcePart]:
        # Local import avoids a module cycle: the validation engine depends on
        # SourceWorkspace, while ToolExecutor only needs it at call time.
        from .evidence_planning import EvidencePlanEngine

        return EvidencePlanEngine(self.workspace).eligible_parts()

    def _get_root_structure(self) -> dict[str, Any]:
        """Return the complete root facts together with the editable candidate tree."""

        skeleton = self._get_evidence_skeleton(depth=2, limit=80)
        if self.workspace.manifest.primary_modality == "html":
            artifact_id = self.workspace.root_structure_artifact_id
            if not artifact_id:
                raise ValueError("HTML root structure is not available")
            root_structure = json.loads(
                self.store.artifact_path(artifact_id).read_text(encoding="utf-8")
            )
        else:
            artifact_id = None
            root_structure = {
                "schema_version": "root-structure.v1",
                "source_id": self.workspace.manifest.source_id,
                "modality": self.workspace.manifest.primary_modality,
                "document_title": self.workspace.manifest.display_name,
                "instructions": (
                    "PDF has no HTML root navigation. Use the candidate Evidence "
                    "skeleton and bounded node reads for semantic corrections."
                ),
            }
        return {
            "root_structure_artifact_id": artifact_id,
            "root_structure": root_structure,
            "candidate_evidence_skeleton": skeleton,
            "base_plan_artifact_id": skeleton["base_plan_artifact_id"],
            "revision_operations": [
                "create_node",
                "update_node",
                "move_node",
                "merge_nodes",
                "exclude_subtree",
            ],
            "instruction": (
                "Treat root_structure as authored facts and the candidate skeleton "
                "as an editable starting point. Submit only the revisions needed to "
                "make Evidence follow the meaningful root-visible hierarchy."
            ),
        }


    def _get_evidence_skeleton(
        self,
        node_key: str | None = None,
        depth: int = 2,
        offset: int = 0,
        limit: int = 80,
    ) -> dict[str, Any]:
        """Return a compact projection of the Python-authored Evidence skeleton."""

        tree = self.workspace.evidence_tree
        base_plan_id = self.workspace.evidence_skeleton_plan_artifact_id
        if tree is None or self.workspace.evidence_skeleton_plan is None or not base_plan_id:
            raise ValueError("Python Evidence skeleton is not available")
        nodes = {node.node_key: node for node in tree.nodes}
        selected_key = node_key or next(
            node.node_key for node in tree.nodes if node.evidence_node_id == tree.root_node_id
        )
        if selected_key not in nodes:
            raise ValueError("Unknown Evidence skeleton node_key")
        child_keys: dict[str, list[str]] = {}
        key_by_id = {node.evidence_node_id: node.node_key for node in tree.nodes}
        for node in tree.nodes:
            if node.parent_node_id:
                parent_key = key_by_id[node.parent_node_id]
                child_keys.setdefault(parent_key, []).append(node.node_key)

        depth = max(0, min(int(depth), 8))
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 400))
        descendant_keys: list[str] = []

        def visit(key: str, remaining: int) -> None:
            if remaining <= 0:
                return
            for child_key in child_keys.get(key, []):
                descendant_keys.append(child_key)
                visit(child_key, remaining - 1)

        visit(selected_key, depth)

        def describe(key: str) -> dict[str, Any]:
            node = nodes[key]
            parent_key = key_by_id.get(node.parent_node_id or "")
            return {
                "node_key": node.node_key,
                "parent_node_key": parent_key,
                "title": node.title,
                "node_type": node.node_type,
                "child_count": len(child_keys.get(key, [])),
                "source_part_count": len(node.source_refs),
                "direct_char_count": node.direct_char_count,
                "descendant_char_count": node.descendant_char_count,
            }

        page = descendant_keys[offset : offset + limit]
        return {
            "evidence_tree_id": tree.evidence_tree_id,
            "base_plan_artifact_id": base_plan_id,
            "selected_node": describe(selected_key),
            "depth": depth,
            "total_descendants_in_view": len(descendant_keys),
            "offset": offset,
            "next_offset": min(len(descendant_keys), offset + len(page)),
            "truncated": offset + len(page) < len(descendant_keys),
            "nodes": [describe(key) for key in page],
            "revision_operations": [
                "create_node",
                "update_node",
                "move_node",
                "merge_nodes",
                "exclude_subtree",
                "assign_content",
                "exclude_content",
            ],
            "instruction": (
                "The candidate skeleton already has complete SourcePart coverage, but "
                "its hierarchy is not authoritative. Compare it with get_root_structure, "
                "then submit an empty operations list only when they are semantically coherent."
            ),
        }

    def _read_evidence_node(
        self,
        node_key: str,
        include_descendants: bool = False,
        offset: int = 0,
        limit: int = 8_000,
        content_offset: int = 0,
        content_limit: int = 40,
    ) -> dict[str, Any]:
        """Read bounded Python-materialized exact Evidence for an ambiguous node."""

        tree = self.workspace.evidence_tree
        if tree is None:
            raise ValueError("Python Evidence skeleton is not available")
        nodes = {node.node_key: node for node in tree.nodes}
        if node_key not in nodes:
            raise ValueError("Unknown Evidence skeleton node_key")
        selected = nodes[node_key]
        selected_ids = {selected.evidence_node_id}
        if include_descendants:
            changed = True
            while changed:
                changed = False
                for node in tree.nodes:
                    if node.parent_node_id in selected_ids and node.evidence_node_id not in selected_ids:
                        selected_ids.add(node.evidence_node_id)
                        changed = True
        selected_nodes = [node for node in tree.nodes if node.evidence_node_id in selected_ids]
        blocks = [
            f"[{node.node_key}] {node.title}\n{node.content}"
            for node in selected_nodes
            if node.content
        ]
        content = "\n\n".join(blocks)
        offset = max(0, int(offset))
        limit = max(500, min(int(limit), 12_000))
        end = min(len(content), offset + limit)
        from .evidence_planning import EvidencePlanEngine

        engine = EvidencePlanEngine(self.workspace)
        eligible = {part.part_id: part for part in engine.eligible_parts()}
        assignments = []
        seen_parts: set[str] = set()
        for node in selected_nodes:
            for ref in node.source_refs:
                part = eligible.get(ref.source_part_id)
                if part is None or part.part_id in seen_parts:
                    continue
                seen_parts.add(part.part_id)
                assignments.append({
                    "content_key": engine.content_key(part),
                    "node_key": node.node_key,
                    "heading": part.locator.get("heading") or part.label,
                    "heading_path": part.locator.get("heading_path") or part.locator.get("section_path"),
                    "url": part.locator.get("url"),
                    "page_start": part.locator.get("page_start") or part.locator.get("page"),
                    "page_end": part.locator.get("page_end") or part.locator.get("page"),
                    "char_count": part.char_count,
                })
        content_offset = max(0, int(content_offset))
        content_limit = max(1, min(int(content_limit), 80))
        assignment_page = assignments[content_offset:content_offset + content_limit]
        next_content_offset = min(len(assignments), content_offset + len(assignment_page))
        return {
            "node_key": node_key,
            "title": selected.title,
            "include_descendants": bool(include_descendants),
            "node_count": len(selected_nodes),
            "offset": offset,
            "next_offset": end,
            "total_chars": len(content),
            "truncated": end < len(content),
            "content": content[offset:end],
            "content_assignments": {
                "items": assignment_page,
                "total": len(assignments),
                "offset": content_offset,
                "next_offset": next_content_offset,
                "truncated": next_content_offset < len(assignments),
            },
            "instruction": (
                "This is exact Python-materialized Evidence for semantic inspection; "
                "do not return rewritten content in the revision. Use returned content_keys "
                "with create_node or assign_content to resolve ambiguous placement. "
                "A key moves the whole existing unit, including all its current spans."
            ),
        }


    def _submit_evidence_revision(
        self,
        base_plan_artifact_id: str,
        operations: list[dict[str, Any]] | None = None,
        summary: str = "",
        gaps: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        """Apply Agent structural edits over the complete Python baseline plan."""

        if (
            self.review_profile == "architect"
            and self.workspace.manifest.primary_modality == "pdf"
            and not self._allow_unreviewed_pdf_fallback
        ):
            pending_unassigned = next(
                (
                    index for index in range(self._pdf_required_unassigned_count)
                    if index not in self._pdf_unassigned_seen
                ),
                None,
            )
            pending_unmatched = next(
                (
                    index for index in range(self._pdf_required_unmatched_count)
                    if index not in self._pdf_unmatched_seen
                ),
                None,
            )
            if (
                not self._pdf_structure_inspected
                or pending_unassigned is not None
                or pending_unmatched is not None
            ):
                return {
                    "ok": False,
                    "error": "pdf_structure_review_incomplete",
                    "message": "Inspect the compact PDF structure and every residual or unmatched assignment page before submitting a revision.",
                    "next_unassigned_offset": pending_unassigned,
                    "next_unmatched_offset": pending_unmatched,
                }
        base_plan = self.workspace.evidence_skeleton_plan
        expected_artifact_id = self.workspace.evidence_skeleton_plan_artifact_id
        if base_plan is None or expected_artifact_id is None:
            return {"ok": False, "error": "evidence_skeleton_required"}
        if base_plan_artifact_id != expected_artifact_id:
            return {
                "ok": False,
                "error": "stale_evidence_skeleton",
                "expected_base_plan_artifact_id": expected_artifact_id,
            }
        try:
            revision = EvidenceRevision.model_validate(
                {
                    "source_id": self.workspace.manifest.source_id,
                    "base_plan_artifact_id": base_plan_artifact_id,
                    "operations": operations or [],
                    "summary": _compact_text(summary)[:2_000],
                }
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": "invalid_evidence_revision",
                "validation_errors": [str(exc)[:4_000]],
                "instruction": "Correct the revision operations and resubmit; there is no operation-count cap.",
            }

        from .evidence_planning import EvidencePlanEngine, EvidencePlanRejected

        engine = EvidencePlanEngine(self.workspace)
        try:
            revised_plan = engine.apply_revision(base_plan, revision)
            artifacts = engine.materialize(revised_plan)
        except EvidencePlanRejected as exc:
            return {
                "ok": False,
                "error": "evidence_revision_validation_failed",
                "validation_artifact_id": exc.validation_artifact_id,
                "report": exc.report.model_dump(mode="json"),
                "instruction": "Correct the structural edits and resubmit the revision.",
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": "invalid_evidence_revision",
                "validation_errors": [str(exc)[:4_000]],
                "instruction": "Correct the revision operations and resubmit; there is no operation-count cap.",
            }

        revision_artifact = self.store.put_json_artifact(
            revision.model_dump(mode="json"),
            kind="evidence_revision",
            source_id=self.workspace.manifest.source_id,
            metadata={
                "base_plan_artifact_id": base_plan_artifact_id,
                "revised_plan_artifact_id": artifacts.plan_artifact_id,
                "operation_count": len(revision.operations),
            },
            preserve_text_integrity=True,
        )
        self.workspace.allowed_artifacts.add(revision_artifact.artifact_id)
        return self._commit_materialized_evidence(
            revised_plan,
            artifacts,
            summary=summary,
            gaps=gaps,
            warnings=warnings,
            accepted_event="evidence_revision_accepted",
            accepted_payload={
                "revision_artifact_id": revision_artifact.artifact_id,
                "base_plan_artifact_id": base_plan_artifact_id,
                "operation_count": len(revision.operations),
            },
        )

    def _source_outline_nodes(self) -> tuple[str, dict[str, dict[str, Any]]]:
        """Build a compact, modality-neutral authored outline projection."""

        source_node_id = f"source:{self.workspace.manifest.source_id}"
        nodes: dict[str, dict[str, Any]] = {}

        def add(
            node_id: str,
            parent_node_id: str | None,
            kind: str,
            label: str,
            **extra: Any,
        ) -> None:
            nodes[node_id] = {
                "node_id": node_id,
                "parent_node_id": parent_node_id,
                "kind": kind,
                "label": label[:300],
                "ordinal": int(extra.pop("ordinal", 0)),
                **extra,
            }

        add(
            source_node_id,
            None,
            "source",
            self.workspace.manifest.display_name,
            modality=self.workspace.manifest.primary_modality,
            source_id=self.workspace.manifest.source_id,
            ordinal=0,
        )
        eligible_ids = {part.part_id for part in self._architect_parts()}

        if self.workspace.manifest.primary_modality == "html":
            pages = sorted(
                (part for part in self.workspace.parts.values() if part.kind == "html_page"),
                key=lambda item: (item.ordinal, item.part_id),
            )
            page_by_url = {
                str(part.locator.get("url")): part
                for part in pages
                if part.locator.get("url")
            }
            root_pages = [
                part for part in pages if int(part.metadata.get("navigation_depth", 0)) == 0
            ]
            root_page = root_pages[0] if root_pages else (pages[0] if pages else None)
            branch_node_by_label: dict[str, str] = {}
            branch_node_by_url: dict[str, str] = {}
            if root_page is not None:
                root_page_id = f"page:{root_page.part_id}"
                add(
                    root_page_id,
                    source_node_id,
                    "document",
                    root_page.label,
                    source_part_id=root_page.part_id,
                    url=root_page.locator.get("url"),
                    navigation_depth=0,
                    referenceable=False,
                    ordinal=root_page.ordinal,
                )
                authored_groups = (
                    root_page.metadata.get("navigation_groups")
                    or root_page.metadata.get("navigation_branches", [])
                )
                for index, branch in enumerate(authored_groups, start=1):
                    branch_id = str(
                        branch.get("group_id")
                        or branch.get("branch_id")
                        or f"group_{index:03d}"
                    )
                    node_id = f"group:{branch_id}"
                    label = _compact_text(str(branch.get("label") or branch_id))
                    links = [str(item) for item in branch.get("links", []) if item]
                    member_labels = [
                        _compact_text(str(item))
                        for item in branch.get("labels", [label])
                        if _compact_text(str(item))
                    ]
                    fetched = [url for url in links if url in page_by_url]
                    add(
                        node_id,
                        root_page_id,
                        "navigation_group",
                        label,
                        authored_basis="root_navigation_group",
                        authored_labels=member_labels,
                        container_tag=branch.get("container_tag"),
                        container_classes=branch.get("container_classes", []),
                        link_count=len(links),
                        fetched_link_count=len(fetched),
                        unfetched_links=[url for url in links if url not in page_by_url][:40],
                        ordinal=index,
                    )
                    for member_label in member_labels:
                        branch_node_by_label[member_label.casefold()] = node_id
                    for url in links:
                        branch_node_by_url.setdefault(url, node_id)

            for page in pages:
                if root_page is not None and page.part_id == root_page.part_id:
                    continue
                url = str(page.locator.get("url") or "")
                parent_url = str(page.metadata.get("navigation_parent_url") or "")
                branch_label = _compact_text(
                    str(page.metadata.get("capability_branch") or "")
                )
                parent_page = page_by_url.get(parent_url)
                if parent_page is not None and (
                    root_page is None or parent_page.part_id != root_page.part_id
                ):
                    parent_node_id = f"page:{parent_page.part_id}"
                else:
                    parent_node_id = (
                        branch_node_by_url.get(url)
                        or branch_node_by_label.get(branch_label.casefold())
                        or (f"page:{root_page.part_id}" if root_page is not None else source_node_id)
                    )
                add(
                    f"page:{page.part_id}",
                    parent_node_id,
                    "page",
                    page.label,
                    source_part_id=page.part_id,
                    url=page.locator.get("url"),
                    navigation_depth=int(page.metadata.get("navigation_depth", 0)),
                    capability_branch=page.metadata.get("capability_branch"),
                    referenceable=False,
                    ordinal=page.ordinal,
                )

            for part in self.workspace.parts.values():
                if part.kind == "html_page" or (
                    part.part_id not in eligible_ids
                    and part.kind != "html_heading"
                ):
                    continue
                node_id = f"part:{part.part_id}"
                parent_node_id = (
                    f"part:{part.parent_part_id}"
                    if part.parent_part_id and f"part:{part.parent_part_id}" in nodes
                    else None
                )
                if parent_node_id is None:
                    page = page_by_url.get(str(part.locator.get("url") or ""))
                    parent_node_id = (
                        f"page:{page.part_id}" if page is not None else source_node_id
                    )
                add(
                    node_id,
                    parent_node_id,
                    "section",
                    part.label,
                    source_part_id=part.part_id,
                    url=part.locator.get("url"),
                    heading_level=part.locator.get("heading_level"),
                    heading_path=part.locator.get("heading_path")
                    or part.locator.get("section_path"),
                    char_count=part.char_count,
                    referenceable=part.part_id in eligible_ids,
                    ordinal=part.ordinal,
                )
        else:
            relevant = [
                part
                for part in self.workspace.parts.values()
                if part.part_id in eligible_ids
                or part.kind in {"pdf_section", "pdf_page_content"}
            ]
            relevant_ids = {part.part_id for part in relevant}
            for part in sorted(relevant, key=lambda item: (item.ordinal, item.part_id)):
                parent_part_id = (
                    part.locator.get("parent_section_part_id")
                    or part.locator.get("chapter_part_id")
                )
                parent_node_id = (
                    f"part:{parent_part_id}"
                    if parent_part_id in relevant_ids
                    else source_node_id
                )
                add(
                    f"part:{part.part_id}",
                    parent_node_id,
                    "section" if part.kind == "pdf_section" else "page",
                    part.label,
                    source_part_id=part.part_id,
                    page=part.locator.get("page") or part.locator.get("page_start"),
                    page_end=part.locator.get("page_end"),
                    heading_level=part.locator.get("heading_level"),
                    section_path=part.locator.get("section_path"),
                    char_count=part.char_count,
                    referenceable=part.part_id in eligible_ids,
                    ordinal=part.ordinal,
                )

        child_counts: Counter[str] = Counter(
            str(node["parent_node_id"])
            for node in nodes.values()
            if node.get("parent_node_id") is not None
        )
        for node in nodes.values():
            node["child_count"] = child_counts.get(node["node_id"], 0)
        return source_node_id, nodes

    def _get_source_outline(
        self,
        node_id: str | None = None,
        depth: int = 2,
        offset: int = 0,
        limit: int = 80,
    ) -> dict[str, Any]:
        if not self.workspace.inventory_completed:
            raise ValueError("Source inventory is not complete")
        root_id, nodes = self._source_outline_nodes()
        selected_id = node_id or root_id
        if selected_id not in nodes:
            raise ValueError("Unknown outline node_id")
        depth = max(0, min(int(depth), 4))
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 100))
        children: dict[str, list[dict[str, Any]]] = {}
        for node in nodes.values():
            parent = node.get("parent_node_id")
            if parent is not None:
                children.setdefault(str(parent), []).append(node)
        for items in children.values():
            items.sort(key=lambda item: (item.get("ordinal", 0), item["node_id"]))
        descendants: list[dict[str, Any]] = []

        def visit(parent_id: str, remaining: int) -> None:
            if remaining <= 0:
                return
            for child in children.get(parent_id, []):
                descendants.append(child)
                visit(child["node_id"], remaining - 1)

        visit(selected_id, depth)
        page = descendants[offset : offset + limit]
        return {
            "root_node_id": root_id,
            "selected_node": nodes[selected_id],
            "depth": depth,
            "total_descendants_in_view": len(descendants),
            "offset": offset,
            "next_offset": min(len(descendants), offset + len(page)),
            "truncated": offset + len(page) < len(descendants),
            "nodes": page,
            "instruction": (
                "Expand a navigation group, page, or section by passing its node_id. "
                "Use read_source_content to read exact text for a selected subtree."
            ),
        }

    def _read_source_content(
        self,
        outline_node_id: str | None = None,
        part_ids: list[str] | None = None,
        part_offset: int = 0,
        char_offset: int = 0,
        max_chars: int = 12_000,
    ) -> dict[str, Any]:
        if not self.workspace.inventory_completed:
            raise ValueError("Source inventory is not complete")
        root_id, nodes = self._source_outline_nodes()
        eligible = {part.part_id: part for part in self._architect_parts()}
        requested_ids = list(dict.fromkeys(str(item) for item in (part_ids or [])))
        selected_node_id = outline_node_id or (root_id if not requested_ids else None)
        if selected_node_id is not None:
            if selected_node_id not in nodes:
                raise ValueError("Unknown outline_node_id")
            selected_outline_ids = {selected_node_id}
            changed = True
            while changed:
                changed = False
                for node in nodes.values():
                    if (
                        node.get("parent_node_id") in selected_outline_ids
                        and node["node_id"] not in selected_outline_ids
                    ):
                        selected_outline_ids.add(node["node_id"])
                        changed = True
            requested_ids.extend(
                str(nodes[node_id].get("source_part_id"))
                for node_id in selected_outline_ids
                if nodes[node_id].get("source_part_id") in eligible
            )
        requested_ids = [part_id for part_id in dict.fromkeys(requested_ids) if part_id]
        unknown = [part_id for part_id in requested_ids if part_id not in eligible]
        if unknown:
            return {
                "ok": False,
                "error": "source_part_not_referenceable",
                "unknown_or_structural_part_ids": unknown[:50],
            }
        order = {part_id: index for index, part_id in enumerate(self.workspace.parts)}
        requested_ids.sort(key=lambda part_id: order.get(part_id, 10**9))
        part_offset = max(0, int(part_offset))
        char_offset = max(0, int(char_offset))
        max_chars = max(500, min(int(max_chars), 14_000))
        items: list[dict[str, Any]] = []
        remaining = max_chars
        next_part_offset = part_offset
        next_char_offset = char_offset
        while next_part_offset < len(requested_ids) and remaining > 0:
            part_id = requested_ids[next_part_offset]
            part = eligible[part_id]
            text = self.workspace.part_text[part_id]
            start = min(next_char_offset, len(text))
            if start >= len(text):
                next_part_offset += 1
                next_char_offset = 0
                continue
            end = min(len(text), start + remaining)
            path: list[str] = []
            cursor = f"part:{part_id}"
            visited: set[str] = set()
            while cursor in nodes and cursor not in visited:
                visited.add(cursor)
                path.append(str(nodes[cursor]["label"]))
                parent = nodes[cursor].get("parent_node_id")
                if parent is None:
                    break
                cursor = str(parent)
            path.reverse()
            items.append(
                {
                    "source_part_id": part_id,
                    "kind": part.kind,
                    "label": part.label,
                    "outline_path": path,
                    "locator": part.locator,
                    "content_start_offset": start,
                    "content_end_offset": end,
                    "total_chars": len(text),
                    "content": text[start:end],
                    "complete_part": start == 0 and end == len(text),
                    "plan_reference": (
                        {"source_part_id": part_id}
                        if start == 0 and end == len(text)
                        else {
                            "source_part_id": part_id,
                            "start_offset": start,
                            "end_offset": end,
                        }
                    ),
                }
            )
            self.workspace.inspected_parts.add(part_id)
            remaining -= end - start
            if end < len(text):
                next_char_offset = end
                break
            next_part_offset += 1
            next_char_offset = 0
        truncated = next_part_offset < len(requested_ids)
        self.workspace.persist()
        return {
            "outline_node_id": selected_node_id,
            "selected_part_count": len(requested_ids),
            "part_offset": part_offset,
            "char_offset": char_offset,
            "items": items,
            "returned_chars": max_chars - remaining,
            "truncated": truncated,
            "next_part_offset": next_part_offset,
            "next_char_offset": next_char_offset,
            "instruction": (
                "If truncated, call again with next_part_offset and next_char_offset. "
                "EvidencePlan may reference a complete SourcePart without reading it in chunks."
            ),
        }

    def _read_artifact(self, artifact_id: str, offset: int = 0, limit: int = 12_000) -> dict[str, Any]:
        if artifact_id not in self.workspace.allowed_artifacts:
            raise PermissionError("Artifact is outside the current Source scope")
        scope = self.workspace.manifest.source_metadata.get("extraction_scope") or {}
        if scope.get("mode") == "selected":
            record = self.store.artifact_record(artifact_id)
            if (artifact_id == self.workspace.manifest.raw_artifact_id
                    or record.kind in {"raw_source", "raw_html_page", "raw_source_page", "raw_html_part",
                                       "html_source_snapshot", "html_source_snapshot_chunk"}):
                raise PermissionError("Original material is outside the selected chapter scope; read normalized content")
        limit = max(1, min(int(limit), 20_000))
        offset = max(0, int(offset))
        record = self.store.artifact_record(artifact_id)
        path = self.store.artifact_path(artifact_id)
        if not (
            record.media_type.startswith("text/")
            or record.media_type in {"application/json", "application/xml"}
        ):
            return {
                "artifact": record.model_dump(mode="json"),
                "readable_as_text": False,
            }
        text = path.read_text(encoding="utf-8", errors="replace")
        return {
            "artifact": record.model_dump(mode="json"),
            "offset": offset,
            "next_offset": min(len(text), offset + limit),
            "total_chars": len(text),
            "content": text[offset : offset + limit],
        }

    def _list_source_parts(self, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 25))
        parts = self.workspace.evidence_candidates()
        descriptors = [
            {
                "part_id": item.part_id,
                "kind": item.kind,
                "label": item.label,
                "locator": {
                    key: item.locator[key]
                    for key in (
                        "url",
                        "page",
                        "page_start",
                        "page_end",
                        "heading",
                        "heading_level",
                        "section_index",
                        "chunk",
                        "navigation_depth",
                        "capability_branch",
                    )
                    if key in item.locator
                },
                "ordinal": item.ordinal,
                "char_count": item.char_count,
                "content_kinds": item.metadata.get("content_kinds", []),
                "preview": self.workspace.part_text.get(item.part_id, "")[:280],
            }
            for item in parts[offset : offset + limit]
        ]
        return {
            "total": len(parts),
            "offset": offset,
            "next_offset": min(len(parts), offset + limit),
            "truncated": offset + len(descriptors) < len(parts),
            "parts": descriptors,
        }

    def _read_source_part(self, part_id: str, offset: int = 0, limit: int = 12_000) -> dict[str, Any]:
        if part_id not in self.workspace.parts:
            raise ValueError("Unknown source_part_id")
        part = self.workspace.parts[part_id]
        if not part.text_artifact_id:
            return {
                "part": part.model_dump(mode="json"),
                "readable_as_text": False,
                "instruction": "Use the modality-specific visual enrichment tool for this preserved binary part.",
            }
        text = self.workspace.part_text.get(part_id)
        if text is None:
            text = self.store.artifact_path(part.text_artifact_id).read_text(encoding="utf-8")
            self.workspace.part_text[part_id] = text
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 12_000))
        self.workspace.inspected_parts.add(part_id)
        self.workspace.persist()
        return {
            "part": part.model_dump(mode="json"),
            "offset": offset,
            "next_offset": min(len(text), offset + limit),
            "total_chars": len(text),
            "content": text[offset : offset + limit],
        }

    def _search_source(self, query: str, limit: int = 10) -> dict[str, Any]:
        query = _compact_text(query)
        if not query:
            raise ValueError("query is required")
        limit = max(1, min(int(limit), 15))
        needle = query.casefold()
        matches: list[dict[str, Any]] = []
        for part_id, text in self.workspace.part_text.items():
            index = text.casefold().find(needle)
            if index < 0:
                continue
            self.workspace.inspected_parts.add(part_id)
            start = max(0, index - 220)
            end = min(len(text), index + len(query) + 420)
            matches.append(
                {
                    "part_id": part_id,
                    "label": self.workspace.parts[part_id].label,
                    "locator": self.workspace.parts[part_id].locator,
                    "start_offset": start,
                    "end_offset": end,
                    "snippet": text[start:end],
                }
            )
            if len(matches) >= limit:
                break
        self.workspace.persist()
        return {"query": query, "match_count": len(matches), "matches": matches}

    @staticmethod
    def _candidate_page_key(part: SourcePart) -> str:
        if part.locator.get("url"):
            return f"url:{part.locator['url']}"
        page = part.locator.get("page") or part.locator.get("page_start")
        if page is not None:
            return f"page:{page}"
        return f"parent:{part.parent_part_id or part.part_id}"

    def _curate_evidence(
        self,
        selection_mode: str,
        selected_part_ids: list[str] | None = None,
        excluded: list[dict[str, Any]] | None = None,
        merge_groups: list[dict[str, Any]] | None = None,
        rationale: str = "",
    ) -> dict[str, Any]:
        if not self.workspace.inventory_completed:
            return {"ok": False, "error": "inventory_required"}
        if selection_mode not in {"all_except_excluded", "only_selected"}:
            return {"ok": False, "error": "invalid_selection_mode"}

        candidates = self.workspace.evidence_candidates()
        candidate_by_id = {part.part_id: part for part in candidates}
        candidate_ids = set(candidate_by_id)
        declared_selected = list(dict.fromkeys(selected_part_ids or []))
        exclusion_reasons: dict[str, str] = {}
        for item in excluded or []:
            part_id = str(item.get("part_id") or "")
            reason = _compact_text(str(item.get("reason") or "not useful evidence"))
            if part_id:
                exclusion_reasons[part_id] = reason[:500]

        merge_references: set[str] = set()
        referenced = set(declared_selected) | set(exclusion_reasons)
        for group in merge_groups or []:
            group_ids = {str(item) for item in group.get("part_ids", [])}
            merge_references.update(group_ids)
            referenced.update(group_ids)
        unknown = sorted(referenced - candidate_ids)
        if unknown:
            return {
                "ok": False,
                "error": "unknown_candidate_ids",
                "unknown_part_ids": unknown[:50],
            }

        if selection_mode == "only_selected":
            selected = set(declared_selected) | merge_references
            if not selected:
                return {"ok": False, "error": "selected_part_ids_required"}
        else:
            selected = set(candidate_ids)
        selected.difference_update(exclusion_reasons)
        if not selected:
            return {"ok": False, "error": "curation_selected_no_evidence"}

        page_order: dict[str, list[str]] = {}
        for part in candidates:
            page_order.setdefault(self._candidate_page_key(part), []).append(part.part_id)
        selected_groups: list[list[str]] = []
        grouped_ids: set[str] = set()
        for raw_group in merge_groups or []:
            group = list(dict.fromkeys(str(item) for item in raw_group.get("part_ids", [])))
            if len(group) < 2:
                return {"ok": False, "error": "merge_group_requires_at_least_two_parts"}
            if len(group) > 16:
                return {"ok": False, "error": "merge_group_too_large"}
            if not set(group).issubset(selected):
                return {"ok": False, "error": "merge_group_contains_unselected_part"}
            if grouped_ids.intersection(group):
                return {"ok": False, "error": "candidate_used_in_multiple_merge_groups"}
            page_keys = {self._candidate_page_key(candidate_by_id[item]) for item in group}
            if len(page_keys) != 1:
                return {"ok": False, "error": "merge_group_crosses_pages"}
            ordered = page_order[next(iter(page_keys))]
            positions = [ordered.index(item) for item in group]
            if positions != sorted(positions) or positions != list(
                range(min(positions), max(positions) + 1)
            ):
                return {"ok": False, "error": "merge_group_must_be_adjacent_and_ordered"}
            selected_groups.append(group)
            grouped_ids.update(group)

        curated_ids: list[str] = []
        merge_by_first = {group[0]: group for group in selected_groups}
        for candidate in candidates:
            part_id = candidate.part_id
            if part_id not in selected:
                continue
            if part_id in grouped_ids and part_id not in merge_by_first:
                continue
            group = merge_by_first.get(part_id)
            if not group:
                curated_ids.append(part_id)
                continue
            source_parts = [candidate_by_id[item] for item in group]
            source_texts = [self.workspace.part_text[item] for item in group]
            merged_text = "\n\n".join(source_texts)
            first = source_parts[0]
            last = source_parts[-1]
            merged_id = _stable_id(
                "part",
                self.workspace.manifest.source_id,
                "curated",
                *group,
            )
            merged_locator = dict(first.locator)
            merged_locator.update(
                {
                    "curated_from_part_ids": group,
                    "curated_end_label": last.label,
                }
            )
            merged_image_refs: list[dict[str, Any]] = []
            running_offset = 0
            for index, source_part in enumerate(source_parts):
                for raw_ref in source_part.metadata.get("image_refs", []):
                    adjusted = dict(raw_ref)
                    adjusted["anchor_offset"] = running_offset + int(
                        adjusted.get("anchor_offset", 0)
                    )
                    merged_image_refs.append(adjusted)
                running_offset += len(source_texts[index])
                if index < len(source_parts) - 1:
                    running_offset += 2
            merged = SourcePart(
                part_id=merged_id,
                source_id=self.workspace.manifest.source_id,
                kind=f"{self.workspace.manifest.primary_modality}_curated_section",
                label=(first.label if first.label == last.label else f"{first.label} — {last.label}")[:300],
                locator=merged_locator,
                parent_part_id=first.parent_part_id,
                ordinal=first.ordinal,
                artifact_id=first.artifact_id,
                char_count=len(merged_text),
                metadata={
                    "curated_projection": True,
                    "source_candidate_part_ids": group,
                    "image_refs": merged_image_refs,
                },
            )
            self.workspace.add_part(merged, merged_text)
            self.workspace.inventoried_parts.add(merged_id)
            curated_ids.append(merged_id)

        for part_id in candidate_ids:
            self.workspace.skipped.pop(part_id, None)
        for part_id in candidate_ids - selected:
            self.workspace.skipped[part_id] = exclusion_reasons.get(
                part_id, "not selected by Source Review Child"
            )
        self.workspace.curated_part_ids = curated_ids
        submission = {
            "selection_mode": selection_mode,
            "selected_candidate_count": len(selected),
            "excluded_candidate_count": len(candidate_ids - selected),
            "merge_group_count": len(selected_groups),
            "evidence_unit_count_before_long_text_splitting": len(curated_ids),
            "rationale": _compact_text(rationale)[:2_000],
        }
        self.workspace.curation_submission = submission
        self.workspace.materialize_evidence_excerpts()
        artifact = self.store.put_json_artifact(
            {
                **submission,
                "selected_part_ids": declared_selected,
                "excluded": excluded or [],
                "merge_groups": merge_groups or [],
                "curated_part_ids": curated_ids,
            },
            kind="source_evidence_curation",
            source_id=self.workspace.manifest.source_id,
        )
        self.workspace.allowed_artifacts.add(artifact.artifact_id)
        self.workspace.persist()
        return {
            "accepted": True,
            **submission,
            "materialized_evidence_count": len(self.workspace.evidence),
            "curation_artifact_id": artifact.artifact_id,
        }

    def _commit_materialized_evidence(
        self,
        plan: EvidencePlan,
        artifacts: Any,
        *,
        summary: str,
        gaps: list[str] | None,
        warnings: list[str] | None,
        accepted_event: str,
        accepted_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Commit a Python-validated plan and its flat compatibility projection."""

        self.workspace.evidence_tree = artifacts.tree
        self.workspace.evidence_tree_artifact_id = artifacts.tree_artifact_id
        self.workspace.evidence.clear()
        self.workspace.evidence_parts.clear()
        for node in artifacts.tree.nodes:
            for span in node.source_refs:
                source_part = self.workspace.parts[span.source_part_id]
                image_refs = [
                    EvidenceImageRef.model_validate(item)
                    for item in source_part.metadata.get("image_refs", [])
                    if span.start_offset
                    <= int(item.get("anchor_offset", 0))
                    < span.end_offset
                ]
                self.workspace.add_evidence_excerpt(
                    span.source_part_id,
                    span.start_offset,
                    span.end_offset,
                    image_refs,
                )

        # Preserve parser-owned skip decisions (for example, deterministic noise
        # filtering).  A revision may add semantic subtree exclusions, but it must
        # not erase provenance that was established before the model ran.
        for exclusion in plan.excluded_parts:
            self.workspace.skipped[exclusion.source_part_id] = exclusion.reason
        combined_gaps = list(dict.fromkeys([*self.workspace.gaps, *(gaps or [])]))
        combined_warnings = list(
            dict.fromkeys([*self.workspace.warnings, *(warnings or [])])
        )
        status = "degraded" if combined_gaps else "succeeded"
        result_summary = _compact_text(summary)[:4_000] or (
            f"Python materialized {len(artifacts.tree.nodes)} hierarchical Evidence "
            f"nodes from {artifacts.tree.referenced_part_count} SourceParts after "
            "bounded Evidence Architect revision."
        )
        self.workspace.manifest.status = status
        self.workspace.manifest.warnings = combined_warnings
        actual = self.workspace.coverage()
        corpus_artifact_id = self.workspace.build_corpus_artifact()
        result = SourceResult(
            source_id=self.workspace.manifest.source_id,
            display_name=self.workspace.manifest.display_name,
            modality=self.workspace.manifest.primary_modality,
            status=status,
            summary=result_summary,
            evidence=list(self.workspace.evidence.values()),
            evidence_tree=artifacts.tree,
            corpus_artifact_id=corpus_artifact_id,
            source_part_count=len(self.workspace.parts),
            normalization_version=self.workspace.NORMALIZATION_VERSION,
            coverage=actual,
            gaps=combined_gaps,
            warnings=combined_warnings,
            manifest=self.workspace.manifest,
        )
        self.workspace.final_result = result
        self.workspace.persist()
        self.store.write_source_json(
            result.source_id,
            "result.json",
            result.model_dump(mode="json"),
        )
        event_payload = {
            "evidence_tree_id": artifacts.tree.evidence_tree_id,
            "tree_artifact_id": artifacts.tree_artifact_id,
            "plan_artifact_id": artifacts.plan_artifact_id,
            "validation_artifact_id": artifacts.validation_artifact_id,
            "node_count": len(artifacts.tree.nodes),
            "leaf_count": artifacts.tree.leaf_count,
            "referenced_part_count": artifacts.tree.referenced_part_count,
            "excluded_part_count": len(plan.excluded_parts),
            **(accepted_payload or {}),
        }
        self.store.emit(
            accepted_event,
            event_payload,
            source_id=result.source_id,
            agent=self.agent_name,
            level="warning" if status != "succeeded" else "info",
        )
        self.store.emit(
            "source_result_accepted",
            {
                "status": status,
                "evidence_count": len(result.evidence),
                "evidence_tree_id": artifacts.tree.evidence_tree_id,
                "coverage": actual.model_dump(mode="json"),
                "coverage_mismatches": {},
            },
            source_id=result.source_id,
            agent=self.agent_name,
            level="warning" if status != "succeeded" else "info",
        )
        return {
            "accepted": True,
            "status": status,
            "evidence_tree_id": artifacts.tree.evidence_tree_id,
            "tree_artifact_id": artifacts.tree_artifact_id,
            "node_count": len(artifacts.tree.nodes),
            "leaf_count": artifacts.tree.leaf_count,
            "referenced_part_count": artifacts.tree.referenced_part_count,
            "excluded_part_count": len(plan.excluded_parts),
            "compatibility_evidence_count": len(result.evidence),
            "corpus_artifact_id": corpus_artifact_id,
            **(accepted_payload or {}),
        }

    def _submit_evidence_plan(
        self,
        nodes: list[dict[str, Any]],
        excluded_parts: list[dict[str, Any]] | None = None,
        summary: str = "",
        gaps: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate, materialize, and commit one Agent-authored Evidence plan."""

        if not self.workspace.inventory_completed:
            return {"ok": False, "error": "inventory_required"}
        try:
            plan = EvidencePlan.model_validate(
                {
                    "source_id": self.workspace.manifest.source_id,
                    "nodes": nodes,
                    "excluded_parts": excluded_parts or [],
                    "summary": _compact_text(summary)[:2_000],
                }
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": "invalid_evidence_plan",
                "validation_errors": [str(exc)[:4_000]],
                "instruction": "Correct the plan shape and call submit_evidence_plan again.",
            }

        from .evidence_planning import EvidencePlanEngine, EvidencePlanRejected

        engine = EvidencePlanEngine(self.workspace)
        try:
            artifacts = engine.materialize(plan)
        except EvidencePlanRejected as exc:
            return {
                "ok": False,
                "error": "evidence_plan_validation_failed",
                "validation_artifact_id": exc.validation_artifact_id,
                "report": exc.report.model_dump(mode="json"),
                "instruction": (
                    "Use the reported node_key and source_part_id values to correct "
                    "coverage, ordering, ranges, or overlap, then resubmit the full plan."
                ),
            }

        return self._commit_materialized_evidence(
            plan,
            artifacts,
            summary=summary,
            gaps=gaps,
            warnings=warnings,
            accepted_event="evidence_plan_accepted",
        )

    def _submit_source_structure(
        self,
        summary: str,
        outline: list[dict[str, Any]] | None = None,
        coverage_plan: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self.workspace.inventory_completed:
            return {"ok": False, "error": "inventory_required"}
        submission = {
            "summary": _compact_text(summary)[:4000],
            "outline": (outline or [])[:100],
            "coverage_plan": (coverage_plan or [])[:100],
        }
        self.workspace.structure_submission = submission
        artifact = self.store.put_json_artifact(
            submission,
            kind="source_structure",
            source_id=self.workspace.manifest.source_id,
        )
        self.workspace.allowed_artifacts.add(artifact.artifact_id)
        self.workspace.persist()
        return {"accepted": True, "structure_artifact_id": artifact.artifact_id}

    def _submit_source_result(
        self,
        status: str,
        summary: str,
        coverage: dict[str, Any] | None = None,
        gaps: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        if status not in {"succeeded", "degraded", "excluded", "failed"}:
            return {"ok": False, "error": "invalid_status"}
        if status not in {"excluded", "failed"} and not self.workspace.inventory_completed:
            return {"ok": False, "error": "inventory_required"}
        if (
            self.review_profile == "curated"
            and status in {"succeeded", "degraded"}
            and self.workspace.curation_submission is None
        ):
            return {
                "ok": False,
                "error": "evidence_curation_required",
                "instruction": "Call curate_evidence before submitting a usable Source result.",
            }

        declared = coverage or {}
        for skipped in declared.get("skipped", []):
            part_id = str(skipped.get("part_id") or "")
            reason = _compact_text(str(skipped.get("reason") or "unspecified"))
            if part_id:
                self.workspace.skipped[part_id] = reason

        # The Harness, not the model, materializes every EvidenceExcerpt from
        # normalized text spans before evaluating the terminal result.
        self.workspace.materialize_evidence_excerpts()
        actual = self.workspace.coverage()
        unresolved = (
            self.workspace.processable_part_ids()
            - self.workspace.normalized_parts
            - set(self.workspace.skipped)
        )
        validation_errors: list[str] = []
        if status == "succeeded" and not self.workspace.evidence:
            validation_errors.append("succeeded requires at least one exact EvidenceExcerpt")
        if status == "succeeded" and unresolved:
            validation_errors.append(
                f"succeeded has {len(unresolved)} unnormalized and unexplained SourcePart(s)"
            )
        if validation_errors:
            self.store.emit(
                "source_result_rejected",
                {"errors": validation_errors, "actual_coverage": actual.model_dump(mode="json")},
                source_id=self.workspace.manifest.source_id,
                agent=self.agent_name,
                level="warning",
            )
            return {
                "ok": False,
                "error": "result_validation_failed",
                "validation_errors": validation_errors,
                "actual_coverage": actual.model_dump(mode="json"),
                "instruction": "Normalize remaining parts or submit degraded with explicit gaps.",
            }

        numeric_coverage_keys = {
            "discovered_parts",
            "inventoried_parts",
            "normalized_parts",
            "semantic_units",
            "inspected_parts",
            "evidence_bearing_parts",
        }
        ignored_numeric_coverage = sorted(numeric_coverage_keys.intersection(declared))
        mismatches: dict[str, dict[str, Any]] = {}
        combined_warnings = list(dict.fromkeys(self.workspace.warnings + (warnings or [])))
        if ignored_numeric_coverage:
            combined_warnings.append("Agent-provided coverage counts were ignored; measured Harness values were used.")

        self.workspace.manifest.status = status
        self.workspace.manifest.warnings = combined_warnings
        corpus_artifact_id = self.workspace.build_corpus_artifact()
        result = SourceResult(
            source_id=self.workspace.manifest.source_id,
            display_name=self.workspace.manifest.display_name,
            modality=self.workspace.manifest.primary_modality,
            status=status,
            summary=_compact_text(summary)[:4000],
            evidence=list(self.workspace.evidence.values()),
            corpus_artifact_id=corpus_artifact_id,
            source_part_count=len(self.workspace.parts),
            normalization_version=self.workspace.NORMALIZATION_VERSION,
            coverage=actual,
            gaps=list(dict.fromkeys(self.workspace.gaps + (gaps or []))),
            warnings=combined_warnings,
            manifest=self.workspace.manifest,
        )
        self.workspace.final_result = result
        self.workspace.persist()
        self.store.write_source_json(
            result.source_id,
            "result.json",
            result.model_dump(mode="json"),
        )
        self.store.emit(
            "source_result_accepted",
            {
                "status": status,
                "evidence_count": len(result.evidence),
                "coverage": actual.model_dump(mode="json"),
                "coverage_mismatches": mismatches,
            },
            source_id=result.source_id,
            agent=self.agent_name,
            level="warning" if status != "succeeded" else "info",
        )
        return {
            "accepted": True,
            "status": status,
            "evidence_count": len(result.evidence),
            "corpus_artifact_id": corpus_artifact_id,
            "actual_coverage": actual.model_dump(mode="json"),
        }


    def _require_raw_artifact(self) -> str:
        artifact_id = self.workspace.manifest.raw_artifact_id
        if not artifact_id:
            raise ValueError("Source has no RawSourceArtifact")
        return artifact_id
