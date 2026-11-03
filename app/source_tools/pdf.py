"""PDF inventory, semantic projections, visual evidence, and directory lookup."""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from ..models import EvidenceImageRef, SourcePart
from .pdf_structure import (
    _clip_pdf_bbox,
    _expand_pdf_bbox,
    _is_pdf_contents_page,
    _is_pdf_furniture,
    _is_pdf_margin_image_furniture,
    _join_pdf_block_texts,
    _pdf_bbox_area,
    _pdf_body_font_size,
    _pdf_enrich_generic_heading,
    _pdf_find_heading_parent,
    _pdf_heading_candidates,
    _pdf_heading_outline,
    _pdf_heading_text,
    _pdf_printed_toc_entries,
    _pdf_repeated_furniture_texts,
    _pdf_section_is_substantive,
    _pdf_tiny_display_image_reason,
    _union_bboxes,
    _vertical_bbox_distance,
)
from .text import _compact_text, _stable_id


PDF_VISUAL_GROUP_MIN_IMAGES = 8
PDF_VISUAL_GROUP_RENDER_SCALE = 1.5
PDF_VISUAL_GROUP_PADDING = 8.0


class PdfToolsMixin:
    """PDF methods used by the Source-scoped ToolExecutor.

    The executor supplies workspace, store, settings, and common review helpers.
    """

    def _get_pdf_document_structure(
        self,
        navigation_node_id: str | None = None,
        offset: int = 0,
        limit: int = 12,
        unassigned_offset: int = 0,
        unassigned_limit: int = 6,
        unmatched_offset: int = 0,
        unmatched_limit: int = 24,
        assignment_offset: int = 0,
        assignment_limit: int = 6,
        excluded_offset: int = 0,
    ) -> dict[str, Any]:
        """Page one joined PDF directory while keeping residual content visible."""

        if self.workspace.manifest.primary_modality != "pdf":
            raise ValueError("Current Source is not PDF")
        if navigation_node_id is not None and (
            not isinstance(navigation_node_id, str) or not navigation_node_id.strip()
        ):
            raise ValueError(
                "navigation_node_id must be a non-empty string: use "
                '{"navigation_node_id": "root"} for the PDF root directory, '
                "or copy a chapter ID from navigation.items[].navigation_node_id."
            )
        # The public root ID maps to the parser's existing root representation.
        # Omitted/None arguments remain compatible with older callers.
        if navigation_node_id == "root":
            navigation_node_id = None
        artifact_id = self.workspace.pdf_document_structure_artifact_id
        if not artifact_id:
            raise ValueError("PDF document structure is not available")
        structure = self.workspace.pdf_document_structure
        if structure is None:
            structure = json.loads(
                self.store.artifact_path(artifact_id).read_text(encoding="utf-8")
            )
        agent_view = structure.get("agent_view") or structure
        navigation = agent_view.get("navigation_tree") or {}
        navigation_nodes = list(navigation.get("nodes") or [])
        nodes_by_id = {
            str(item["navigation_node_id"]): item
            for item in navigation_nodes
            if item.get("navigation_node_id")
        }
        if navigation_node_id and navigation_node_id not in nodes_by_id:
            raise ValueError(
                f"Unknown PDF navigation_node_id: {navigation_node_id}. "
                'Use {"navigation_node_id": "root"} for the root directory, '
                "or copy an exact chapter ID from navigation.items[].navigation_node_id. "
                "Evidence node keys such as document are not navigation IDs."
            )
        assignments = list(agent_view.get("content_assignments") or [])
        assignments_by_node: dict[str, list[dict[str, Any]]] = {}
        unmatched_assignments: list[dict[str, Any]] = []
        for item in assignments:
            node_id = str(item.get("navigation_node_id") or "")
            if node_id in nodes_by_id:
                assignments_by_node.setdefault(node_id, []).append(item)
            else:
                unmatched_assignments.append(item)
        children_by_parent: dict[str | None, list[dict[str, Any]]] = {}
        for item in navigation_nodes:
            parent_id = item.get("parent_navigation_node_id")
            if parent_id not in nodes_by_id:
                parent_id = None
            children_by_parent.setdefault(parent_id, []).append(item)

        def describe_node(item: dict[str, Any]) -> dict[str, Any]:
            node_id = str(item["navigation_node_id"])
            attached = assignments_by_node.get(node_id, [])
            return {
                "navigation_node_id": node_id,
                "parent_navigation_node_id": item.get("parent_navigation_node_id"),
                "evidence_node_keys": [
                    str(value.get("evidence_node_key"))
                    for value in attached[:4]
                    if value.get("evidence_node_key")
                ],
                "evidence_parent_keys": list(
                    dict.fromkeys(
                        str(value.get("parent_evidence_node_key"))
                        for value in attached
                        if value.get("parent_evidence_node_key")
                    )
                )[:4],
                "title": str(item.get("title") or "")[:120],
                "level": item.get("level"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "child_count": len(children_by_parent.get(node_id, [])),
                "assigned_content_count": len(attached),
                "assigned_char_count": sum(
                    int(value.get("char_count") or 0) for value in attached
                ),
            }

        def bounded_page(
            items: list[dict[str, Any]], start: int, size: int
        ) -> dict[str, Any]:
            page = items[start : start + size]
            return {
                "total": len(items),
                "offset": start,
                "items": page,
                "next_offset": start + len(page) if start + len(page) < len(items) else None,
            }

        offset = max(0, int(offset))
        limit = max(1, min(12, int(limit)))
        unassigned_offset = max(0, int(unassigned_offset))
        unassigned_limit = max(1, min(6, int(unassigned_limit)))
        unmatched_offset = max(0, int(unmatched_offset))
        unmatched_limit = max(1, min(48, int(unmatched_limit)))
        assignment_offset = max(0, int(assignment_offset))
        assignment_limit = max(1, min(6, int(assignment_limit)))
        excluded_offset = max(0, int(excluded_offset))
        selected_children = children_by_parent.get(navigation_node_id, [])
        child_page = bounded_page(
            [describe_node(item) for item in selected_children], offset, limit
        )
        residual = [
            {
                "content_key": item.get("content_key"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "char_count": item.get("char_count"),
                "preview": str(item.get("preview") or "")[:200],
                "default_attachment": item.get("default_attachment"),
            }
            for item in agent_view.get("unassigned_content") or []
        ]
        unmatched = [
            {
                "evidence_node_key": item.get("evidence_node_key"),
                "parent_evidence_node_key": item.get("parent_evidence_node_key"),
                "title": str(item.get("title") or "")[:120],
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "char_count": item.get("char_count"),
            }
            for item in unmatched_assignments
        ]
        selected_assignments = [
            {
                "evidence_node_key": item.get("evidence_node_key"),
                "parent_evidence_node_key": item.get("parent_evidence_node_key"),
                "title": str(item.get("title") or "")[:120],
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "char_count": item.get("char_count"),
            }
            for item in assignments_by_node.get(navigation_node_id or "", [])
        ]
        excluded = list(agent_view.get("deterministically_excluded_content") or [])
        excluded_by_kind: dict[str, int] = {}
        for item in excluded:
            kind = str(item.get("kind") or "unknown")
            excluded_by_kind[kind] = excluded_by_kind.get(kind, 0) + 1
        excluded_descriptors = [
            {
                "page": item.get("page"),
                "kind": str(item.get("kind") or "unknown"),
                "char_count": item.get("char_count"),
                "reason": str(item.get("reason") or "")[:160],
            }
            for item in excluded
        ]
        skeleton = self.workspace.evidence_tree
        base_plan_id = self.workspace.evidence_skeleton_plan_artifact_id
        if skeleton is None or not base_plan_id:
            raise ValueError("Python Evidence skeleton is not available")
        supporting = dict(agent_view.get("supporting_evidence") or {})
        toc_pages = list(supporting.pop("printed_toc_pages", []) or [])
        supporting["printed_toc_page_count"] = len(toc_pages)
        result = {
            "pdf_document_structure_artifact_id": artifact_id,
            "document_title": str(agent_view.get("document_title") or "")[:160],
            "page_count": agent_view.get("page_count"),
            "text_layer": agent_view.get("text_layer"),
            "navigation_source": navigation.get("selected_source"),
            "navigation_node_count": len(navigation_nodes),
            "content_assignment_count": len(assignments),
            "supporting_evidence": supporting,
            "candidate_evidence_tree_id": skeleton.evidence_tree_id,
            "candidate_evidence_node_count": len(skeleton.nodes),
            "base_plan_artifact_id": base_plan_id,
            "navigation": {
                "selected_node": (
                    describe_node(nodes_by_id[navigation_node_id])
                    if navigation_node_id else {
                        "navigation_node_id": "root",
                        "evidence_node_keys": ["document"],
                        "title": str(agent_view.get("document_title") or "")[:120],
                        "child_count": len(children_by_parent.get(None, [])),
                    }
                ),
                **child_page,
            },
            "unassigned_content": bounded_page(
                residual, unassigned_offset, unassigned_limit
            ),
            "unmatched_content_assignments": bounded_page(
                unmatched, unmatched_offset, unmatched_limit
            ),
            "selected_content_assignments": bounded_page(
                selected_assignments, assignment_offset, assignment_limit
            ),
            "excluded_content": {
                **bounded_page(excluded_descriptors, excluded_offset, 6),
                "by_kind": excluded_by_kind,
            },
            "instruction": (
                "Review unassigned_content and unmatched_content_assignments first. "
                "Page navigation with offset, residuals with unassigned_offset, "
                "unmatched assignments with unmatched_offset, and selected assignments "
                "with assignment_offset; page exclusions with excluded_offset. "
                "Unmatched pages default to 24 items (unmatched_limit up to 48); "
                "the complete-JSON size bound may shorten any page, so always follow "
                "the returned next_offset. Independent read-only pages may be requested "
                "together; do not repeat pages already inspected. "
                "Use navigation_node_id='root' for root navigation and root pagination. "
                "To inspect a branch, copy its exact navigation_node_id from navigation.items; "
                "do not use evidence_node_keys as navigation IDs. "
                "Submit edits using evidence_node_keys or parser-issued content_keys."
            ),
        }
        # A truncated JSON prefix can hide the very residual content the Child
        # must review. Keep whole items and paginate instead of relying on the
        # generic observation-artifact fallback (read_artifact is not granted).
        return_limit = 12_000
        while (
            len(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
            > return_limit
        ):
            for field in (
                "navigation",
                "unmatched_content_assignments",
                "selected_content_assignments",
                "excluded_content",
                "unassigned_content",
            ):
                page = result[field]
                if len(page["items"]) > 1:
                    page["items"].pop()
                    end = page["offset"] + len(page["items"])
                    page["next_offset"] = end if end < page["total"] else None
                    break
            else:
                return {
                    "ok": False,
                    "error": "pdf_structure_page_too_large",
                    "message": "Even a one-item PDF structure page exceeds the PDF structure page budget.",
                }
        if navigation_node_id is None:
            self._pdf_structure_inspected = True
        self._pdf_required_unassigned_count = len(residual)
        self._pdf_required_unmatched_count = len(unmatched)
        for field, seen in (
            ("unassigned_content", self._pdf_unassigned_seen),
            ("unmatched_content_assignments", self._pdf_unmatched_seen),
        ):
            page = result[field]
            seen.update(range(page["offset"], page["offset"] + len(page["items"])))
        return result

    def _read_pdf_content(
        self,
        content_key: str,
        offset: int = 0,
        limit: int = 8_000,
    ) -> dict[str, Any]:
        """Read one parser-registered residual PDF content unit by opaque key."""

        if self.workspace.manifest.primary_modality != "pdf":
            raise ValueError("Current Source is not PDF")
        part = next(
            (
                candidate
                for candidate in self._architect_parts()
                if candidate.kind == "pdf_page_content"
                and candidate.metadata.get("content_key") == content_key
            ),
            None,
        )
        if part is None:
            raise ValueError("Unknown or ineligible PDF content_key")
        content = self.workspace.part_text.get(part.part_id, "")
        offset = max(0, int(offset))
        limit = max(500, min(int(limit), 12_000))
        end = min(len(content), offset + limit)
        return {
            "content_key": content_key,
            "page_start": part.locator.get("page"),
            "page_end": part.locator.get("page"),
            "offset": offset,
            "next_offset": end,
            "total_chars": len(content),
            "truncated": end < len(content),
            "content": content[offset:end],
            "instruction": (
                "This is exact parser-owned residual content. Refer to it only by "
                "content_key in assign_content or exclude_content."
            ),
        }

    def _pdf_inventory(self) -> dict[str, Any]:
        if self.workspace.manifest.primary_modality != "pdf":
            raise ValueError("Current Source is not PDF")
        from ..pdf_sourcepart.pipeline import inventory_pages, attach_semantic_structure

        raw_id = self._require_raw_artifact()
        pages, layout_pages, page_contexts, _raw_toc, extraction_summary = inventory_pages(self, raw_id)
        version_sample = [self.workspace.manifest.display_name] + [c['raw_text'] for c in page_contexts[:5]]
        text_page_count = sum(
            1
            for context in page_contexts
            if context.get("raw_text", "").strip()
        )
        body_font_size = _pdf_body_font_size(page_contexts)
        # Embedded outlines can be partial or stale. Neither navigation nor
        # content boundaries may depend on them, including selected chapters.
        bookmark_outline: list[dict[str, Any]] = []
        extraction_scope = self.workspace.manifest.source_metadata.get('extraction_scope') or {}
        selected_scope = extraction_scope.get('mode') == 'selected'
        printed_toc = _pdf_printed_toc_entries(page_contexts)
        heading_candidates = _pdf_heading_candidates(
            page_contexts,
            body_font_size=body_font_size,
            bookmark_outline=bookmark_outline,
            printed_toc_entries=printed_toc["entries"],
        )
        sections = self._build_pdf_semantic_units(page_contexts, heading_candidates)
        assigned_blocks = {
            block_id for section in sections
            for block_id in self.workspace.parts[section['part_id']].locator.get('source_block_ids', [])
        }
        repeated_furniture = _pdf_repeated_furniture_texts(page_contexts)
        unassigned_content: list[dict[str, Any]] = []
        excluded_content: list[dict[str, Any]] = []
        for context in page_contexts:
            page_content_part_id = context["page_content_part_id"]
            if page_content_part_id and page_content_part_id in self.workspace.parts:
                page_content = self.workspace.parts[page_content_part_id]
                page_number = int(context["page"])
                meaningful_blocks = [
                    block
                    for block in context.get("blocks", [])
                    if not _is_pdf_furniture(block, repeated_furniture)
                ]
                meaningful_text = _join_pdf_block_texts(meaningful_blocks)
                residual_blocks = [b for b in meaningful_blocks if b['part_id'] not in assigned_blocks]
                if meaningful_blocks and not residual_blocks:
                    # The exact text is already represented by semantic section
                    # projections.  The full-page copy remains readable for audit,
                    # but does not participate in Evidence coverage a second time.
                    page_content.metadata.update(
                        {
                            "evidence_default": False,
                            "content_status": "assigned_to_navigation",
                        }
                    )
                elif context.get("is_contents_page") or _is_pdf_contents_page(context):
                    page_content.metadata.update(
                        {
                            "evidence_default": False,
                            "content_status": "excluded_navigation",
                        }
                    )
                    self.workspace.skipped[page_content_part_id] = (
                        "authored_navigation_projection"
                    )
                    excluded_content.append(
                        {
                            "page": page_number,
                            "kind": "table_of_contents",
                            "char_count": page_content.char_count,
                            "reason": "Authored navigation is represented by the navigation tree.",
                        }
                    )
                elif not meaningful_text.strip():
                    page_content.metadata.update(
                        {
                            "evidence_default": False,
                            "content_status": "excluded_page_furniture",
                        }
                    )
                    self.workspace.skipped[page_content_part_id] = (
                        "repeated_page_furniture_only"
                    )
                    excluded_content.append(
                        {
                            "page": page_number,
                            "kind": "page_furniture",
                            "char_count": page_content.char_count,
                            "reason": "Only repeated header, footer, or page-number text remains.",
                        }
                    )
                else:
                    # A section on this page does not cover its preceding prose.
                    # Keep exactly the still-unassigned blocks as the residual.
                    if len(residual_blocks) != len(meaningful_blocks):
                        meaningful_text = _join_pdf_block_texts(residual_blocks)
                        page_content.locator['source_block_ids'] = [b['part_id'] for b in residual_blocks]
                        self.workspace.add_part(page_content, meaningful_text)
                    # Opaque keys let the PDF Child place residual content without
                    # receiving or fabricating SourcePart identifiers.
                    content_key = _stable_id(
                        "pdfcontent",
                        self.workspace.manifest.source_id,
                        str(page_number),
                        page_content_part_id,
                        length=16,
                    )
                    page_content.metadata.update(
                        {
                            "evidence_default": True,
                            "content_status": "unassigned",
                            "content_key": content_key,
                        }
                    )
                    unassigned_content.append(
                        {
                            "content_key": content_key,
                            "page_start": page_number,
                            "page_end": page_number,
                            "char_count": page_content.char_count,
                            "preview": _compact_text(meaningful_text)[:500],
                            "default_attachment": "document",
                        }
                    )
            page_part = self.workspace.parts[context["page_part_id"]]
            page_part.metadata["semantic_section_count"] = sum(
                1
                for section in sections
                if section["page_start"] <= context["page"] <= section["page_end"]
            )

        self._reclassify_pdf_repeated_image_furniture()
        # Only ordinary structured pages may use the legacy non-overlapping
        # fragment grouping. Recovery must never reopen an unreadable PDF.
        if not selected_scope and extraction_summary['opening_error'] is None and all(c['extraction_mode']=='structured' for c in page_contexts):
            try:
                self._create_pdf_visual_groups(page_contexts, raw_id)
            except OSError:
                raise
            except Exception as exc:
                self.workspace.warnings.append(
                    f'Optional PDF visual grouping failed ({type(exc).__name__}); original image Artifacts remain available.'
                )
        self._filter_pdf_tiny_display_images()
        attach_semantic_structure(self.workspace, page_contexts)
        image_association = self._associate_pdf_images(page_contexts)
        def page_informative_image_count(page_payload: dict[str, Any]) -> int:
            return sum(
                1
                for part_id in page_payload["image_part_ids"]
                if self.workspace.parts[part_id].metadata.get("role") == "informative"
            )

        self.workspace.inventory_completed = True
        self.workspace.update_document_version("\n".join(version_sample))
        authored_outline_source = "printed_toc" if printed_toc["entries"] else "layout_headings"
        authored_outline = (
            printed_toc["entries"]
            if printed_toc["entries"]
            else [
                {
                    "node_id": item["node_id"],
                    "parent_node_id": item.get("parent_node_id"),
                    "title": item["title"],
                    "level": item["level"],
                    "page_start": item["page"],
                    "page_end": item["page"],
                    "source": item["source"],
                }
                for item in heading_candidates.values()
            ]
        )
        navigation_nodes = [
            {
                "navigation_node_id": item["node_id"],
                "parent_navigation_node_id": item.get("parent_node_id"),
                "title": item["title"],
                "level": item["level"],
                "page_start": item.get("page_start", item.get("page")),
                "page_end": item.get("page_end", item.get("page")),
                "provenance": item.get("source", authored_outline_source),
            }
            for item in authored_outline
        ]
        evidence_key_by_part_id = {
            item["part_id"]: f"section_{index:03d}"
            for index, item in enumerate(sections, start=1)
        }
        content_assignments = [
            {
                "evidence_node_key": evidence_key_by_part_id[item["part_id"]],
                "navigation_node_id": item.get("structure_node_id"),
                "title": item["heading"],
                "parent_evidence_node_key": evidence_key_by_part_id.get(
                    item.get("parent_section_part_id"), "document"
                ),
                "page_start": item["page_start"],
                "page_end": item["page_end"],
                "char_count": item["char_count"],
            }
            for index, item in enumerate(sections, start=1)
        ]
        agent_view = {
            "schema_version": "pdf-agent-structure.v1",
            "source_id": self.workspace.manifest.source_id,
            "document_title": self.workspace.manifest.display_name,
            "page_count": extraction_summary["summary"]["pages"],
            "text_layer": {
                "supported": text_page_count > 0,
                "text_page_count": text_page_count,
                "textless_page_count": len(pages) - text_page_count,
                "ocr_used": False,
            },
            "navigation_tree": {
                "selected_source": authored_outline_source,
                "nodes": navigation_nodes,
            },
            "content_assignments": content_assignments,
            "unassigned_content": unassigned_content,
            "deterministically_excluded_content": excluded_content,
            "supporting_evidence": {
                "native_bookmark_count": len(bookmark_outline),
                "printed_toc_pages": printed_toc["pages"],
                "printed_toc_entry_count": len(printed_toc["entries"]),
                "layout_heading_count": len(heading_candidates),
            },
            "instructions": (
                "navigation_tree alone defines visible Evidence topology; "
                "content_assignments attach exact parser-owned text; unassigned_content "
                "may be attached or excluded only through its opaque content_key."
            ),
        }
        pdf_document_structure = {
            "schema_version": "pdf-document-structure.v1",
            "source_id": self.workspace.manifest.source_id,
            "document_title": self.workspace.manifest.display_name,
            "page_count": extraction_summary["summary"]["pages"],
            "text_layer": {
                "supported": text_page_count > 0,
                "text_page_count": text_page_count,
                "textless_page_count": len(pages) - text_page_count,
                "ocr_used": False,
            },
            "body_font_size": body_font_size,
            "extraction": extraction_summary,
            "authored_outline_source": authored_outline_source,
            "authored_outline": authored_outline,
            "native_bookmarks": bookmark_outline,
            "bookmark_policy": "ignored",
            "printed_toc_pages": printed_toc["pages"],
            "printed_toc_entries": printed_toc["entries"],
            "layout_headings": list(heading_candidates.values()),
            "semantic_units": sections,
            "agent_view": agent_view,
            "pages": [
                {
                    "page": page["page"],
                    "text_chars": page["text_chars"],
                    "text_block_count": page["text_block_count"],
                    "image_count": page["image_count"],
                }
                for page in pages
            ],
            "instructions": (
                "These are parser-grounded structure facts. Printed TOC and layout "
                "headings determine navigation and content boundaries. Native PDF "
                "bookmarks are ignored entirely."
            ),
        }
        structure_artifact = self.store.put_json_artifact(
            pdf_document_structure,
            kind="pdf_document_structure",
            source_id=self.workspace.manifest.source_id,
            metadata={
                "page_count": extraction_summary["summary"]["pages"],
                "authored_outline_source": authored_outline_source,
                "outline_node_count": len(authored_outline),
            },
            preserve_text_integrity=True,
        )
        self.workspace.pdf_document_structure_artifact_id = structure_artifact.artifact_id
        self.workspace.pdf_document_structure = pdf_document_structure
        self.workspace.allowed_artifacts.add(structure_artifact.artifact_id)
        self.store.emit(
            "pdf_document_structure_materialized",
            {
                "artifact_id": structure_artifact.artifact_id,
                "page_count": extraction_summary["summary"]["pages"],
                "authored_outline_source": authored_outline_source,
                "outline_node_count": len(authored_outline),
                "semantic_unit_count": len(sections),
            },
            source_id=self.workspace.manifest.source_id,
            agent=self.agent_name,
        )

        inventory_payload = {
            "page_count": extraction_summary["summary"]["pages"],
            "text_page_count": text_page_count,
            "image_heavy_page_count": sum(
                1 for page in pages if page_informative_image_count(page) > 2
            ),
            **image_association,
            "semantic_section_count": len(sections),
            "semantic_units": sections,
            "pdf_document_structure_artifact_id": structure_artifact.artifact_id,
            "pages": pages,
        }
        layout_artifact = self.store.put_json_artifact(
            {"schema_version": self.workspace.NORMALIZATION_VERSION, "pages": layout_pages},
            kind="pdf_layout_index",
            source_id=self.workspace.manifest.source_id,
            preserve_text_integrity=True,
        )
        artifact = self.store.put_json_artifact(
            {**inventory_payload, "layout_artifact_id": layout_artifact.artifact_id},
            kind="pdf_inventory",
            source_id=self.workspace.manifest.source_id,
            preserve_text_integrity=True,
        )
        self.workspace.allowed_artifacts.add(layout_artifact.artifact_id)
        self.workspace.allowed_artifacts.add(artifact.artifact_id)
        self.workspace.persist()
        page_summaries = [
            {
                "page": page["page"],
                "text_chars": page["text_chars"],
                "text_block_count": page["text_block_count"],
                "image_count": page["image_count"],
                "informative_image_count": page_informative_image_count(page),
                "semantic_section_count": self.workspace.parts[page["part_id"]].metadata.get(
                    "semantic_section_count", 0
                ),
            }
            for page in pages
        ]
        return {
            "inventory_artifact_id": artifact.artifact_id,
            "layout_artifact_id": layout_artifact.artifact_id,
            "pdf_document_structure_artifact_id": structure_artifact.artifact_id,
            "page_count": inventory_payload["page_count"],
            "text_page_count": inventory_payload["text_page_count"],
            "image_heavy_page_count": inventory_payload["image_heavy_page_count"],
            "informative_image_count": inventory_payload["informative_image_count"],
            "decorative_image_count": inventory_payload["decorative_image_count"],
            "multimodal_unit_count": inventory_payload["multimodal_unit_count"],
            "visual_only_unit_count": inventory_payload["visual_only_unit_count"],
            "semantic_section_count": len(sections),
            "semantic_unit_count": len(sections) + sum(
                1 for context in page_contexts if context["page_content_part_id"]
            ),
            "resolved_version": self.workspace.manifest.resolved_version,
            "version_conflict": self.workspace.manifest.version_conflict,
            "pages": page_summaries,
            "note": (
                "The full page, layout, and section records are in inventory_artifact_id. "
                "Use pdf_list_structure for a paginated semantic-unit view."
            ),
        }

    def _build_pdf_semantic_units(
        self,
        page_contexts: list[dict[str, Any]],
        heading_candidates: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build conservative, exact section projections from ordered PDF blocks.

        A section is created only for an explicit heading pattern. This keeps a
        full-page projection as the safe fallback for documents whose visual
        layout is not reliable enough to infer a chapter boundary.
        """
        repeated_furniture = _pdf_repeated_furniture_texts(page_contexts)
        heading_candidates = heading_candidates or {}
        candidates: list[dict[str, Any]] = []
        active: dict[str, Any] | None = None
        for context in page_contexts:
            if _is_pdf_contents_page(context):
                if active and _pdf_section_is_substantive(active):
                    candidates.append(active)
                active = None
                context["is_contents_page"] = True
                continue
            for block in context["blocks"]:
                if _is_pdf_furniture(block, repeated_furniture):
                    continue
                structure_heading = heading_candidates.get(str(block["part_id"]))
                heading = (
                    str(structure_heading["title"])
                    if structure_heading is not None
                    else (None if block.get('structure_owned') and block.get('structured_block',{}).get('type') not in {'paragraph','heading'}
                          else _pdf_heading_text(block["text"]))
                )
                if heading:
                    if active and _pdf_section_is_substantive(active):
                        candidates.append(active)
                    active = {
                        "heading": heading,
                        "blocks": [block],
                        "page_start": block["page"],
                        "page_end": block["page"],
                        "structure_heading": structure_heading,
                    }
                    continue
                if active:
                    active["blocks"].append(block)
                    active["page_end"] = block["page"]
        if active and _pdf_section_is_substantive(active):
            candidates.append(active)

        prepared_sections: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates, start=1):
            blocks = candidate["blocks"]
            content = _join_pdf_block_texts(blocks)
            if not content.strip():
                continue
            heading = _pdf_enrich_generic_heading(str(candidate["heading"]), blocks)
            source_block_ids = [str(block["part_id"]) for block in blocks]
            page_start = int(candidate["page_start"])
            page_end = int(candidate["page_end"])
            part_id = f"{self.workspace.manifest.source_id}_section_{index:04d}"
            outline = _pdf_heading_outline(heading)
            structure_heading = candidate.get("structure_heading")
            if structure_heading is not None:
                outline.update(
                    {
                        "heading_kind": structure_heading.get(
                            "heading_kind", structure_heading.get("source", "layout")
                        ),
                        "heading_number": list(
                            structure_heading.get("heading_number") or []
                        ),
                        "heading_level": int(structure_heading.get("level") or 1),
                        "is_chapter_heading": int(
                            structure_heading.get("level") or 1
                        )
                        == 1,
                        "structure_node_id": structure_heading.get("node_id"),
                        "structure_source": structure_heading.get("source"),
                    }
                )
            prepared_sections.append(
                {
                    "part_id": part_id,
                    "heading": heading,
                    "blocks": blocks,
                    "content": content,
                    "source_block_ids": source_block_ids,
                    "page_start": page_start,
                    "page_end": page_end,
                    **outline,
                }
            )

        sections: list[dict[str, Any]] = []
        active_hierarchy: list[dict[str, Any]] = []
        for ordinal, prepared in enumerate(prepared_sections, start=1):
            parent = _pdf_find_heading_parent(active_hierarchy, prepared)
            if parent:
                parent_index = active_hierarchy.index(parent)
                active_hierarchy = active_hierarchy[: parent_index + 1]
            else:
                active_hierarchy = []

            ancestor_headings = [
                str(item["heading"]) for item in active_hierarchy if str(item.get("heading") or "")
            ]
            chapter = (
                prepared
                if prepared.get("is_chapter_heading")
                else next(
                    (
                        item
                        for item in reversed(active_hierarchy)
                        if item.get("is_chapter_heading") or int(item.get("heading_level") or 0) == 1
                    ),
                    None,
                )
            )
            section_path = " > ".join([*ancestor_headings, str(prepared["heading"])])
            hierarchy_locator = {
                "heading_level": int(prepared["heading_level"]),
                "section_path": section_path,
                "ancestor_headings": ancestor_headings,
            }
            heading_number = prepared.get("heading_number")
            if heading_number:
                hierarchy_locator["heading_number"] = list(heading_number)
            if prepared.get("structure_node_id"):
                hierarchy_locator["structure_node_id"] = prepared[
                    "structure_node_id"
                ]
            if prepared.get("structure_source"):
                hierarchy_locator["structure_source"] = prepared[
                    "structure_source"
                ]
            if parent:
                hierarchy_locator["parent_section_part_id"] = parent["part_id"]
            if chapter:
                hierarchy_locator["chapter_heading"] = chapter["heading"]
                hierarchy_locator["chapter_part_id"] = chapter["part_id"]

            active_hierarchy.append(prepared)
            part = SourcePart(
                part_id=str(prepared["part_id"]),
                source_id=self.workspace.manifest.source_id,
                kind="pdf_section",
                label=str(prepared["heading"])[:300],
                locator={
                    "page": int(prepared["page_start"]),
                    "page_start": int(prepared["page_start"]),
                    "page_end": int(prepared["page_end"]),
                    "heading": prepared["heading"],
                    "source_block_ids": prepared["source_block_ids"],
                    **hierarchy_locator,
                },
                parent_part_id=str(parent["part_id"]) if parent else None,
                ordinal=ordinal,
                artifact_id=self.workspace.manifest.raw_artifact_id,
                metadata={
                    "semantic_unit": True,
                    "semantic_granularity": "section",
                    "evidence_default": True,
                    "source_block_count": len(prepared["source_block_ids"]),
                    "cross_page": int(prepared["page_end"]) > int(prepared["page_start"]),
                    "heading_kind": prepared.get("heading_kind"),
                    **hierarchy_locator,
                },
            )
            self.workspace.add_part(part, str(prepared["content"]))
            self.workspace.inventoried_parts.add(str(prepared["part_id"]))
            sections.append(
                {
                    "part_id": prepared["part_id"],
                    "heading": prepared["heading"],
                    "page_start": prepared["page_start"],
                    "page_end": prepared["page_end"],
                    "heading_level": hierarchy_locator["heading_level"],
                    "heading_number": hierarchy_locator.get("heading_number", []),
                    "section_path": section_path,
                    "ancestor_headings": ancestor_headings,
                    "parent_section_part_id": hierarchy_locator.get("parent_section_part_id"),
                    "chapter_heading": hierarchy_locator.get("chapter_heading"),
                    "chapter_part_id": hierarchy_locator.get("chapter_part_id"),
                    "structure_node_id": hierarchy_locator.get("structure_node_id"),
                    "structure_source": hierarchy_locator.get("structure_source"),
                    "source_block_count": len(prepared["source_block_ids"]),
                    "text_artifact_id": part.text_artifact_id,
                    "char_count": part.char_count,
                }
            )
        return sections

    def _skip_pdf_image_part(self, image_part: SourcePart, reason: str) -> None:
        image_part.metadata["role"] = "decorative"
        image_part.metadata["classification_reason"] = reason
        self.workspace.skipped[image_part.part_id] = reason
        self.workspace.normalized_parts.discard(image_part.part_id)

    def _reclassify_pdf_repeated_image_furniture(self) -> None:
        """Drop repeated page chrome such as headers, footers, and page ribbons."""

        image_parts = [
            part
            for part in self.workspace.parts.values()
            if part.kind == "pdf_image_region"
            and part.metadata.get("role") == "informative"
            and part.artifact_id
        ]
        artifact_counts = Counter(str(part.artifact_id) for part in image_parts)
        for image_part in image_parts:
            if artifact_counts[str(image_part.artifact_id)] < 3:
                continue
            parent = self.workspace.parts.get(str(image_part.parent_part_id or ""))
            if not parent:
                continue
            bbox = _union_bboxes(image_part.locator.get("bboxes", []))
            if not bbox:
                continue
            page_width = float(parent.metadata.get("width") or 0)
            page_height = float(parent.metadata.get("height") or 0)
            if _is_pdf_margin_image_furniture(bbox, page_width, page_height):
                self._skip_pdf_image_part(image_part, "repeated_page_furniture")

    def _create_pdf_visual_groups(
        self, page_contexts: list[dict[str, Any]], raw_artifact_id: str
    ) -> None:
        """Render a single visual reference for pages made from many image fragments."""

        import fitz

        raw_path = self.store.artifact_path(raw_artifact_id)
        with fitz.open(raw_path) as document:
            for context in page_contexts:
                page_number = int(context["page"])
                candidate_parts = [
                    self.workspace.parts[part_id]
                    for part_id in context.get("image_part_ids", [])
                    if part_id in self.workspace.parts
                    and self.workspace.parts[part_id].kind == "pdf_image_region"
                    and self.workspace.parts[part_id].metadata.get("role") == "informative"
                ]
                if len(candidate_parts) < PDF_VISUAL_GROUP_MIN_IMAGES:
                    continue
                bboxes = [
                    bbox
                    for part in candidate_parts
                    for bbox in part.locator.get("bboxes", [])
                    if len(bbox) == 4
                ]
                union = _union_bboxes(bboxes)
                if not union:
                    continue
                page = document[page_number - 1]
                clip_values = _clip_pdf_bbox(
                    _expand_pdf_bbox(union, PDF_VISUAL_GROUP_PADDING),
                    float(page.rect.width),
                    float(page.rect.height),
                )
                if not clip_values:
                    continue
                clip = fitz.Rect(*clip_values)
                render_scale = min(PDF_VISUAL_GROUP_RENDER_SCALE,
                                   math.sqrt(4_000_000 / max(1.0, clip.width * clip.height)))
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(
                        render_scale,
                        render_scale,
                    ),
                    clip=clip,
                    alpha=False,
                )
                image_bytes = pixmap.tobytes("png")
                source_image_part_ids = [part.part_id for part in candidate_parts]
                image_artifact = self.store.put_artifact(
                    image_bytes,
                    kind="pdf_visual_group_render",
                    media_type="image/png",
                    suffix=".png",
                    source_id=self.workspace.manifest.source_id,
                    metadata={
                        "page": page_number,
                        "bboxes": [clip_values],
                        "source_image_part_ids": source_image_part_ids,
                        "render_scale": render_scale,
                    },
                )
                group_part_id = f"{context['page_part_id']}_visual_group_0001"
                page_area = max(1.0, float(page.rect.width) * float(page.rect.height))
                group_part = SourcePart(
                    part_id=group_part_id,
                    source_id=self.workspace.manifest.source_id,
                    kind="pdf_visual_group",
                    label=f"Page {page_number} visual group",
                    locator={
                        "page": page_number,
                        "bboxes": [clip_values],
                        "source_image_part_ids": source_image_part_ids,
                    },
                    parent_part_id=str(context["page_part_id"]),
                    ordinal=10_000 + page_number,
                    artifact_id=image_artifact.artifact_id,
                    metadata={
                        "preserved_binary": True,
                        "derived_from_pdf_render": True,
                        "media_type": "image/png",
                        "pixel_width": int(pixmap.width),
                        "pixel_height": int(pixmap.height),
                        "role": "informative",
                        "classification_reason": "fragmented_image_group_render",
                        "display_area_ratio": round(_pdf_bbox_area(clip_values) / page_area, 6),
                        "source_image_part_ids": source_image_part_ids,
                    },
                )
                self.workspace.add_part(group_part)
                self.workspace.inventoried_parts.add(group_part_id)
                context.setdefault("image_part_ids", []).append(group_part_id)
                for image_part in candidate_parts:
                    self._skip_pdf_image_part(image_part, "grouped_into_visual_region")

    def _filter_pdf_tiny_display_images(self) -> None:
        """Filter high-resolution images that are displayed only as tiny fragments."""

        for image_part in list(self.workspace.parts.values()):
            if (
                image_part.kind != "pdf_image_region"
                or image_part.metadata.get("role") != "informative"
                or image_part.metadata.get('parser_preserved_visual')
            ):
                continue
            reason = _pdf_tiny_display_image_reason(image_part)
            if reason:
                self._skip_pdf_image_part(image_part, reason)

    def _associate_pdf_images(self, page_contexts: list[dict[str, Any]]) -> dict[str, int]:
        """Attach informative images to the nearest downstream-readable PDF unit."""

        contexts = {int(item["page"]): item for item in page_contexts}
        semantic_parts = [
            part
            for part in self.workspace.parts.values()
            if part.kind in {"pdf_page_content", "pdf_section"}
            and part.metadata.get("evidence_default") is not False
        ]
        informative = 0
        decorative = 0
        visual_only = 0
        associated_part_ids: set[str] = set()
        image_parts = [
            part
            for part in self.workspace.parts.values()
            if part.kind in {"pdf_image_region", "pdf_visual_group"}
        ]
        for image_part in image_parts:
            if image_part.metadata.get("role") != "informative":
                decorative += 1
                continue
            informative += 1
            page = int(image_part.locator.get("page") or 0)
            candidates = [
                part
                for part in semantic_parts
                if int(part.locator.get("page_start", part.locator.get("page", 0)))
                <= page
                <= int(part.locator.get("page_end", part.locator.get("page", 0)))
            ]
            target = min(
                candidates,
                key=lambda part: self._pdf_image_part_distance(image_part, part, contexts.get(page)),
                default=None,
            )
            association_method = (
                "page_visual_group_render_nearest_semantic_unit"
                if image_part.kind == "pdf_visual_group"
                else "page_layout_nearest_semantic_unit"
            )
            if target is None:
                visual_only += 1
                association_method = (
                    "visual_group_only_page"
                    if image_part.kind == "pdf_visual_group"
                    else "visual_only_page"
                )
                target = SourcePart(
                    part_id=f"{image_part.part_id}_visual",
                    source_id=self.workspace.manifest.source_id,
                    kind="pdf_visual",
                    label=f"Page {page} visual",
                    locator={"page": page, "bboxes": image_part.locator.get("bboxes", [])},
                    parent_part_id=image_part.parent_part_id,
                    ordinal=image_part.ordinal,
                    artifact_id=image_part.artifact_id,
                    metadata={
                        "semantic_unit": True,
                        "semantic_granularity": "visual_region",
                        "visual_semantic_unit": True,
                        "evidence_default": True,
                    },
                )
                self.workspace.add_part(target)
                self.workspace.inventoried_parts.add(target.part_id)
                self.workspace.normalized_parts.add(target.part_id)
                semantic_parts.append(target)
            anchor_offset = self._pdf_image_anchor_offset(image_part, target, contexts.get(page))
            image_ref = EvidenceImageRef(
                artifact_id=str(image_part.artifact_id),
                source_part_id=image_part.part_id,
                media_type=str(image_part.metadata.get("media_type") or "image/png"),
                page=page,
                bboxes=image_part.locator.get("bboxes", []),
                pixel_width=int(image_part.metadata.get("pixel_width") or 0),
                pixel_height=int(image_part.metadata.get("pixel_height") or 0),
                role="informative",
                association_method=association_method,
                anchor_offset=anchor_offset,
            )
            target.metadata.setdefault("image_refs", []).append(image_ref.model_dump(mode="json"))
            target.metadata["multimodal"] = bool(target.text_artifact_id)
            image_part.metadata["associated_evidence_part_id"] = target.part_id
            image_part.metadata["association_method"] = association_method
            self.workspace.normalized_parts.add(image_part.part_id)
            self.workspace.skipped.pop(image_part.part_id, None)
            associated_part_ids.add(target.part_id)
        return {
            "informative_image_count": informative,
            "decorative_image_count": decorative,
            "multimodal_unit_count": sum(
                1
                for part_id in associated_part_ids
                if self.workspace.parts[part_id].text_artifact_id
            ),
            "visual_only_unit_count": visual_only,
        }

    def _pdf_image_part_distance(
        self,
        image_part: SourcePart,
        semantic_part: SourcePart,
        page_context: dict[str, Any] | None,
    ) -> float:
        if not page_context:
            return 0.0
        source_block_ids = set(semantic_part.locator.get("source_block_ids", []))
        blocks = [
            block
            for block in page_context.get("blocks", [])
            if block.get("part_id") in source_block_ids and block.get("bbox")
        ]
        if not blocks:
            return float("inf")
        image_bbox = _union_bboxes(image_part.locator.get("bboxes", []))
        if not image_bbox:
            return 0.0
        distances = [
            _vertical_bbox_distance(image_bbox, block["bbox"], prefer_preceding=True)
            for block in blocks
        ]
        return min(distances, default=float("inf"))

    def _pdf_image_anchor_offset(
        self,
        image_part: SourcePart,
        semantic_part: SourcePart,
        page_context: dict[str, Any] | None,
    ) -> int:
        content = self.workspace.part_text.get(semantic_part.part_id, "")
        if not content or not page_context:
            return 0
        source_block_ids = set(semantic_part.locator.get("source_block_ids", []))
        blocks = [
            block
            for block in page_context.get("blocks", [])
            if block.get("part_id") in source_block_ids and block.get("bbox")
        ]
        image_bbox = _union_bboxes(image_part.locator.get("bboxes", []))
        if not blocks or not image_bbox:
            return 0
        nearest = min(
            blocks,
            key=lambda block: _vertical_bbox_distance(
                image_bbox, block["bbox"], prefer_preceding=True
            ),
        )
        needle = str(nearest.get("text") or "").strip()
        offset = content.find(needle) if needle else 0
        return max(0, offset)

    def _pdf_list_structure(self, offset: int = 0, limit: int = 40) -> dict[str, Any]:
        if not self.workspace.inventory_completed:
            raise ValueError("Run pdf_inventory first")
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 50))
        all_units = [
            {
                "part_id": part.part_id,
                "kind": part.kind,
                "label": part.label,
                "locator": {
                    key: part.locator[key]
                    for key in (
                        "page",
                        "page_start",
                        "page_end",
                        "heading",
                        "heading_level",
                        "heading_number",
                        "section_path",
                        "ancestor_headings",
                        "parent_section_part_id",
                        "chapter_heading",
                        "chapter_part_id",
                    )
                    if key in part.locator
                },
                "parent_part_id": part.parent_part_id,
                "char_count": part.char_count,
                "evidence_default": part.metadata.get("evidence_default", True),
            }
            for part in self.workspace.parts.values()
            if part.kind in {"pdf_page_content", "pdf_section"}
        ]
        return {
            "unit_count": len(all_units),
            "section_count": sum(1 for item in all_units if item["kind"] == "pdf_section"),
            "page_projection_count": sum(
                1 for item in all_units if item["kind"] == "pdf_page_content"
            ),
            "offset": offset,
            "next_offset": min(len(all_units), offset + limit),
            "units": all_units[offset : offset + limit],
        }
