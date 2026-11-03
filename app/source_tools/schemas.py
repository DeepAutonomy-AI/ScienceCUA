"""Declarative JSON schemas for the Source tool surface."""
from __future__ import annotations

from typing import Any


def _function(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


def _tool_spec_catalog() -> dict[str, dict[str, Any]]:
    string = {"type": "string"}
    integer = {"type": "integer"}
    number = {"type": "number"}
    return {
        "inspect_source": _function("inspect_source", "Inspect the current Source manifest, target profile, and allowed tools.", {}),
        "get_root_structure": _function(
            "get_root_structure",
            "Read the complete compact root-page structure and the Python candidate Evidence skeleton in one call. The root structure preserves authored headings, nested lists, containers, links, and fragments without deciding the Evidence hierarchy.",
            {},
        ),
        "get_pdf_document_structure": _function(
            "get_pdf_document_structure",
            "Read a bounded, joined PDF structure page. Set navigation_node_id to 'root' to read the root directory, including its major branches, unassigned content, and unmatched assignments. To inspect a branch's children and attached content, copy its navigation_node_id from navigation.items. These IDs differ from evidence_node_keys. Unmatched pages default to 24 items, up to 48 with unmatched_limit; the JSON size bound can shorten pages. Follow returned next_offset values and batch independent read-only calls rather than repeating pages.",
            {
                "navigation_node_id": {
                    "type": "string", "minLength": 1, "pattern": r"\S",
                    "default": "root",
                    "description": (
                        'Use {"navigation_node_id": "root"} for the root directory and root '
                        "pagination. For a chapter, copy its exact ID from a previous response's "
                        "navigation.items[].navigation_node_id. Empty or whitespace-only strings "
                        "are invalid. Do not substitute document, ROOT, the string null, or "
                        "an evidence_node_key such as section_001."
                    ),
                },
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 12},
                "unassigned_offset": {"type": "integer", "minimum": 0},
                "unassigned_limit": {"type": "integer", "minimum": 1, "maximum": 6},
                "unmatched_offset": {"type": "integer", "minimum": 0},
                "unmatched_limit": {"type": "integer", "minimum": 1, "maximum": 48, "default": 24},
                "assignment_offset": {"type": "integer", "minimum": 0},
                "assignment_limit": {"type": "integer", "minimum": 1, "maximum": 6},
                "excluded_offset": {"type": "integer", "minimum": 0},
            },
        ),
        "get_evidence_skeleton": _function(
            "get_evidence_skeleton",
            "Inspect the compact Python-authored Evidence hierarchy. It already covers every eligible SourcePart. Start at the root; expand only ambiguous nodes.",
            {
                "node_key": string,
                "depth": integer,
                "offset": integer,
                "limit": integer,
            },
        ),
        "read_pdf_content": _function(
            "read_pdf_content",
            "Read one bounded exact residual PDF content unit by a content_key returned from get_pdf_document_structure. Use only for a concrete placement ambiguity.",
            {
                "content_key": string,
                "offset": integer,
                "limit": integer,
            },
            ["content_key"],
        ),
        "read_evidence_node": _function(
            "read_evidence_node",
            "Read bounded exact text and paginated opaque content_keys for an ambiguous HTML/PDF skeleton node. Keys select existing parser units without exposing source IDs or ranges.",
            {
                "node_key": string,
                "include_descendants": {"type": "boolean"},
                "offset": integer,
                "limit": integer,
                "content_offset": {"type": "integer", "minimum": 0},
                "content_limit": {"type": "integer", "minimum": 1, "maximum": 80},
            },
            ["node_key"],
        ),
        "submit_evidence_revision": _function(
            "submit_evidence_revision",
            "Revise or accept the complete-coverage Python candidate skeleton. There is no operation-count cap. create_node binds an unambiguous authored heading subtree from its parent, or accepts explicit content_keys. HTML/PDF assign_content and exclude_content use only returned opaque keys. Never send source text or SourcePart IDs.",
            {
                "base_plan_artifact_id": string,
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": [
                                    "create_node",
                                    "update_node",
                                    "move_node",
                                    "merge_nodes",
                                    "exclude_subtree",
                                    "assign_content",
                                    "exclude_content",
                                ],
                            },
                            "node_key": string,
                            "title": string,
                            "node_type": {
                                "type": "string",
                                "enum": [
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
                                ],
                            },
                            "parent_node_key": string,
                            "before_node_key": string,
                            "source_node_keys": {
                                "type": "array",
                                "items": string,
                                "maxItems": 32,
                            },
                            "content_keys": {
                                "type": "array",
                                "items": string,
                                "maxItems": 64,
                            },
                            "reason": string,
                        },
                        "required": ["operation"],
                        "additionalProperties": False,
                    },
                },
                "summary": string,
                "gaps": {"type": "array", "items": string},
                "warnings": {"type": "array", "items": string},
            },
            ["base_plan_artifact_id", "operations"],
        ),
        "get_source_outline": _function(
            "get_source_outline",
            "Read the authored information architecture for the current Source. HTML preserves root navigation groups, links, pages, and headings; PDF exposes parser-grounded sections and pages. Expand a returned node_id to inspect its subtree.",
            {
                "node_id": string,
                "depth": integer,
                "offset": integer,
                "limit": integer,
            },
        ),
        "read_source_content": _function(
            "read_source_content",
            "Batch-read exact parser-owned content for an outline subtree or explicit SourcePart IDs. Use the returned continuation offsets when truncated. This tool never summarizes or rewrites text.",
            {
                "outline_node_id": string,
                "part_ids": {"type": "array", "items": string, "maxItems": 100},
                "part_offset": integer,
                "char_offset": integer,
                "max_chars": integer,
            },
        ),
        "read_artifact": _function(
            "read_artifact",
            "Read a bounded text range from an Artifact in the current Source scope.",
            {"artifact_id": string, "offset": integer, "limit": integer},
            ["artifact_id"],
        ),
        "list_source_parts": _function(
            "list_source_parts",
            "List compact descriptors and previews for parser-owned Evidence candidates in source order.",
            {"offset": integer, "limit": integer},
        ),
        "read_source_part": _function(
            "read_source_part",
            "Read a bounded exact range from one normalized SourcePart. This never creates a summary.",
            {"part_id": string, "offset": integer, "limit": integer},
            ["part_id"],
        ),
        "search_source": _function(
            "search_source",
            "Search exact extracted text when the authored outline is ambiguous or a concept spans multiple branches. This is optional and is not a per-SourcePart audit tool.",
            {"query": string, "limit": integer},
            ["query"],
        ),
        "submit_evidence_plan": _function(
            "submit_evidence_plan",
            "Submit the complete hierarchical EvidencePlan. Supply structure and exact SourcePart references, never Evidence prose. Python validates all source coverage, ranges, ordering, and overlap; an invalid plan returns actionable errors and may be resubmitted. A valid plan is materialized and completes the Child session.",
            {
                "nodes": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_key": {
                                "type": "string",
                                "pattern": "^[A-Za-z][A-Za-z0-9_-]{0,79}$",
                            },
                            "title": string,
                            "title_origin": {
                                "type": "string",
                                "enum": ["source", "derived"],
                            },
                            "node_type": {
                                "type": "string",
                                "enum": [
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
                                ],
                            },
                            "parent_node_key": string,
                            "source_refs": {
                                "type": "array",
                                "maxItems": 64,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_part_id": string,
                                        "start_offset": integer,
                                        "end_offset": integer,
                                    },
                                    "required": ["source_part_id"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["node_key", "title"],
                        "additionalProperties": False,
                    },
                },
                "excluded_parts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_part_id": string,
                            "start_offset": integer,
                            "end_offset": integer,
                            "reason": string,
                        },
                        "required": ["source_part_id", "reason"],
                        "additionalProperties": False,
                    },
                },
                "summary": string,
                "gaps": {"type": "array", "items": string},
                "warnings": {"type": "array", "items": string},
            },
            ["nodes"],
        ),
        "curate_evidence": _function(
            "curate_evidence",
            "Select the exact parser-owned candidate units that should become Evidence, exclude noise with reasons, and optionally merge related adjacent candidates on one page. The Harness validates IDs and materializes all text; never send rewritten source content.",
            {
                "selection_mode": {
                    "type": "string",
                    "enum": ["all_except_excluded", "only_selected"],
                },
                "selected_part_ids": {"type": "array", "items": string},
                "excluded": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"part_id": string, "reason": string},
                        "required": ["part_id", "reason"],
                        "additionalProperties": False,
                    },
                },
                "merge_groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "part_ids": {
                                "type": "array",
                                "items": string,
                                "minItems": 2,
                                "maxItems": 16,
                            }
                        },
                        "required": ["part_ids"],
                        "additionalProperties": False,
                    },
                },
                "rationale": string,
            },
            ["selection_mode"],
        ),
        "submit_source_structure": _function(
            "submit_source_structure",
            "Submit a concise structure assessment after the complete inventory.",
            {
                "summary": string,
                "outline": {"type": "array", "items": {"type": "object"}},
                "coverage_plan": {"type": "array", "items": string},
            },
            ["summary"],
        ),
        "submit_source_result": _function(
            "submit_source_result",
            "Declare terminal Source status and material gaps. In the curated review profile, call curate_evidence first. The Harness measures coverage and preserves exact parser-owned Evidence.",
            {
                "status": {"type": "string", "enum": ["succeeded", "degraded", "excluded", "failed"]},
                "summary": string,
                "coverage": {
                    "type": "object",
                    "properties": {
                        "skipped": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"part_id": string, "reason": string},
                                "required": ["part_id", "reason"],
                            },
                        },
                    },
                },
                "gaps": {"type": "array", "items": string},
                "warnings": {"type": "array", "items": string},
            },
            ["status", "summary"],
        ),
        "pdf_inventory": _function(
            "pdf_inventory",
            "Normalize every PDF into preserved layout blocks plus downstream-readable page and explicit-section projections. The text is parser-derived, never model-written.",
            {},
        ),
        "pdf_list_structure": _function(
            "pdf_list_structure",
            "List a bounded page of page and section corpus units created by pdf_inventory. Use read_source_part for detailed inspection; raw layout block references stay in the Artifact.",
            {"offset": integer, "limit": integer},
        ),
        "html_inventory": _function(
            "html_inventory",
            "Inventory the static HTML root plus a bounded child-page budget. Discover every top-level capability branch first, then allocate child pages breadth-first and round-robin across branches, using a run-seeded random order within each branch and depth.",
            {"max_pages": integer},
        ),
        "html_list_navigation": _function(
            "html_list_navigation",
            "List normalized HTML pages, their navigation depth and capability branch, semantic DOM and Markdown projections, and controlled tutorial links.",
            {},
        ),
    }
