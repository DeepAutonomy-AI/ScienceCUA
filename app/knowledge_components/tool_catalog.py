"""Knowledge-stage tool definitions exposed through the capability gateway."""

from __future__ import annotations

from typing import Any

from .common import (
    CATALOG_DESCRIPTION_MIN_CHARS, REFERENCE_CATALOG_SAMPLE_SIZE,
    REFERENCE_PARTITION_MAX_GROUPS, REFERENCE_PARTITION_REVIEW_CHARS,
)


def _knowledge_tool_catalog() -> dict[str, dict[str, Any]]:
    citation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["evidence_id"],
        "anyOf": [
            {
                "required": ["quote"],
                "properties": {"quote": {"type": "string", "minLength": 1, "pattern": r"\S"}},
            },
            {"required": ["image_artifact_id"]},
        ],
        "properties": {
            "evidence_id": {"type": "string"},
            "quote": {
                "type": "string", "maxLength": 1600,
                "description": "Required for a text citation: one nonempty JSON string copied from supporting Evidence. Example: \"quote\": \"Press the button.\" Never use null, a number, an array, or an object. Retained for traceability without exact substring validation.",
            },
            "image_artifact_id": {
                "type": "string",
                "pattern": r"^art_[0-9a-f]{24}$",
                "description": "Optional exact image Artifact ID returned by view_evidence_image. Omit this field entirely for text quotes; never send an empty string.",
            },
        },
    }
    stage_result = {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "summary"],
        "properties": {
            "status": {"type": "string", "enum": ["succeeded", "degraded", "failed"]},
            "summary": {"type": "string"},
            "gaps": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    }
    task = {
        "type": "object",
        "additionalProperties": False,
        "required": ["task", "category", "citations"],
        "properties": {
            "task": {"type": "string", "minLength": 3, "maxLength": 300,
                     "description": "A complete reusable user task expressed in 3–300 characters."},
            "category": {"type": "string", "minLength": 1,
                         "description": "Copy an exact task_categories value from inspect_task_outline: the title of a direct child of an EvidenceTree root. Cite supporting Evidence from that chapter. Never invent a category or use the tree root or a deeper heading."},
            "citations": {"type": "array", "minItems": 1, "maxItems": 24, "items": citation},
        },
    }
    reference = {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "description", "content", "citations"],
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string", "minLength": 1, "maxLength": 100, "description": "One or two short sentences, at most 100 characters including spaces and punctuation. Briefly state knowledge scope and when to consult it. Do not list commands, shortcuts, or examples."},
            "category": {"type": "string"},
            "content": {"type": "string"},
            "injection_policy": {"type": "string", "enum": ["always", "on_demand", "contextual"]},
            "usage_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Generic lower-snake-case Know-what roles, for example command_catalog, "
                    "function_catalog, syntax_reference, parameter_reference, ui_reference, "
                    "file_format_reference, configuration_reference, or concept_reference."
                ),
            },
            "citations": {"type": "array", "minItems": 1, "items": citation},
        },
    }
    trajectory_step = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ordinal", "instruction", "citations"],
        "properties": {
            "ordinal": {"type": "integer", "minimum": 1},
            "instruction": {"type": "string"},
            "expected_observation": {"type": "string"},
            "citations": {"type": "array", "minItems": 1, "items": citation},
        },
    }
    skill_step = {
        "type": "object",
        "description": "One procedure-step JSON object; use the declared fields, with every citation as an object containing evidence_id and a string quote or an inspected image_artifact_id.",
        "additionalProperties": False,
        "required": ["ordinal", "instruction", "citations"],
        "properties": {
            "ordinal": {"type": "integer", "minimum": 1},
            "instruction": {"type": "string"},
            "citations": {"type": "array", "minItems": 1, "items": citation},
        },
    }
    def function(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}

    return {
        "inspect_knowledge_scope": function(
            "inspect_knowledge_scope",
            "Read the current stage, target profile, snapshot identifiers, and permitted data scope.",
            {"type": "object", "additionalProperties": False, "properties": {}},
        ),
        "inspect_task_outline": function(
            "inspect_task_outline",
            "Show the complete EvidenceTree directory for every source in one call. Each nested node has its node_id, title, direct text size, and children. A readable node also has read_node_id with the same ID value; pass that value as node_id to read_task_node. This directory is not citable text. If no tree exists, use list_evidence.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        ),
        "read_task_node": function(
            "read_task_node",
            "Purpose: read the complete text directly assigned to ONE EvidenceTree node. Follow-up form only; read_task_nodes is the default reading tool.\nRequired JSON: {\"node_id\":\"N1\"}. Replace N1 with a real readable node from this run.\nID source: the supplied evidence_tree directory, or inspect_task_outline when that tool is available. read_node_id repeats the same value; the argument key is node_id. An E evidence_id is a citation ID, not a node ID.\nReturns: ordered text segments with E evidence_id values for citations and attached image IDs. Children are read separately.\nLimits: no offset or limit. Never call with {}. Correct a rejected call using the named missing field.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["node_id"],
                "properties": {
                    "node_id": {"type": "string", "description": "Required real EvidenceTree node ID from the supplied evidence_tree directory or inspect_task_outline when available. Copy the node_id/read_node_id value into the node_id argument; do not use an E evidence_id."},
                },
            },
        ),
        "read_task_nodes": function(
            "read_task_nodes",
            "Purpose: read the complete text directly assigned to one to eight EvidenceTree nodes in one call. This is the default node-reading tool; use read_task_node only for a genuinely ambiguous single-node follow-up.\nRequired JSON: {\"node_ids\":[\"N1\",\"N2\"]}. Replace each entry with a real readable node from this run; use an array even for one node. Duplicate entries inside one call are ignored.\nID source: the supplied evidence_tree directory, or inspect_task_outline when that tool is available. read_node_id repeats the same value. An E evidence_id is a citation ID, not a node ID.\nReturns: one entry per node in request order, each with the same node content as read_task_node. Children are read separately.\nLimits: validation is atomic. If any node_id is invalid or unreadable, the WHOLE call is rejected and the error names every failing ID with its reason; correct those IDs and resubmit the batch. Never call with {} or an empty array.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["node_ids"],
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {"type": "string"},
                        "description": "Array of 1 to 8 real EvidenceTree node IDs from the supplied evidence_tree directory or inspect_task_outline. Use an array even for one node. Do not use E evidence_id values.",
                    },
                },
            },
        ),
        "inspect_task_units": function(
            "inspect_task_units",
            "List the compact EvidenceTree semantic directory for selective Task discovery. Inspect broadly, then choose only representative or ambiguous units to read; unread units are allowed and SourceParts remain separate.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 80},
                    "status": {"type": "string", "enum": ["read", "unread"]},
                },
            },
        ),
        "read_task_unit": function(
            "read_task_unit",
            "Select and read all fine-grained SourceParts and citation-ready Evidence belonging directly to one semantic unit. Use it when that capability area is representative, important, or ambiguous enough to ground a Task.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["unit_id"],
                "properties": {"unit_id": {"type": "string"}},
            },
        ),
        "submit_task_unit_review": function(
            "submit_task_unit_review",
            "Complete the currently dispatched Task reading unit. Record zero or more supported user-goal candidate notes grounded only in this unit's Evidence IDs; otherwise explain why it contains no distinct Task.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["unit_id"],
                "properties": {
                    "unit_id": {"type": "string"},
                    "candidates": {
                        "type": "array",
                        "maxItems": 12,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "task",
                                "category",
                                "rationale",
                                "citations",
                            ],
                            "properties": {
                                "task": {"type": "string", "minLength": 3, "maxLength": 300},
                                "category": {"type": "string"},
                                "rationale": {"type": "string"},
                                "citations": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": citation,
                                },
                            },
                        },
                    },
                    "no_task_reason": {"type": "string"},
                },
            },
        ),
        "inspect_task_unit_coverage": function(
            "inspect_task_unit_coverage",
            "Inspect exhaustive EvidenceTree-unit coverage and compact Task candidate notes accumulated across the shared Task Child Session.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 80},
                    "include_reviews": {"type": "boolean"},
                },
            },
        ),
        "submit_reference_reading_plan": function(
            "submit_reference_reading_plan",
            "Submit an exhaustive Reference disposition plan before reading source content. Compact selectors are expanded by Python; every source-bearing node must resolve exactly once to read, preserved_catalog, or skip_for_reference. Preserved detector catalogs are materialized directly from immutable Evidence.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["reading_units", "catalog_decisions", "skip_decisions"],
                "properties": {
                    "reading_units": {
                        "type": "array",
                        "maxItems": 80,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "title",
                                "unit_kind",
                                "root_node_id",
                                "selection",
                                "rationale",
                            ],
                            "properties": {
                                "title": {"type": "string"},
                                "unit_kind": {
                                    "type": "string",
                                    "enum": ["single_node", "semantic_subtree"],
                                },
                                "root_node_id": {"type": "string"},
                                "selection": {
                                    "type": "string",
                                    "enum": [
                                        "node",
                                        "all_source_bearing_descendants",
                                        "all_non_catalog_source_bearing_descendants",
                                    ],
                                },
                                "rationale": {"type": "string"},
                            },
                        },
                    },
                    "catalog_decisions": {
                        "type": "array",
                        "maxItems": 80,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "catalog_id",
                                "disposition",
                                "cover_member_nodes",
                                "rationale",
                            ],
                            "properties": {
                                "catalog_id": {"type": "string"},
                                "disposition": {
                                    "type": "string",
                                    "enum": ["preserve", "exclude"],
                                },
                                "cover_member_nodes": {"type": "boolean"},
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "category": {"type": "string"},
                                "injection_policy": {
                                    "type": "string",
                                    "enum": ["always", "on_demand", "contextual"],
                                },
                                "usage_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "rationale": {"type": "string"},
                            },
                        },
                    },
                    "skip_decisions": {
                        "type": "array",
                        "maxItems": 80,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["root_node_id", "selection", "rationale"],
                            "properties": {
                                "root_node_id": {"type": "string"},
                                "selection": {
                                    "type": "string",
                                    "enum": ["node", "all_source_bearing_descendants"],
                                },
                                "rationale": {"type": "string"},
                            },
                        },
                    },
                },
            },
        ),
        "inspect_reference_units": function(
            "inspect_reference_units",
            "List the accepted Reference-Child-planned reading units. No SourcePart text is merged.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 80},
                    "status": {"type": "string", "enum": ["reviewed", "pending"]},
                },
            },
        ),
        "read_reference_unit": function(
            "read_reference_unit",
            "Read all fine-grained SourceParts and citation-ready Evidence belonging directly to one smallest EvidenceTree reading unit. Use only to revisit an already dispatched ambiguity.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["unit_id"],
                "properties": {"unit_id": {"type": "string"}},
            },
        ),
        "submit_reference_unit_review": function(
            "submit_reference_unit_review",
            "Complete the currently dispatched reading unit. Record zero or more reusable Reference candidate notes grounded only in this unit's Evidence IDs; otherwise explain why it has no reusable Reference knowledge.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["unit_id"],
                "properties": {
                    "unit_id": {"type": "string"},
                    "candidates": {
                        "type": "array",
                        "maxItems": 12,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["title", "category", "rationale", "evidence_ids"],
                            "properties": {
                                "title": {"type": "string"},
                                "category": {"type": "string"},
                                "rationale": {"type": "string"},
                                "evidence_ids": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "string"},
                                },
                                "suggested_usage_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "no_reference_reason": {"type": "string"},
                },
            },
        ),
        "submit_reference_unit_result": function(
            "submit_reference_unit_result",
            "Resolve the currently dispatched candidate region directly into final References. Submit concise summaries with Evidence citations, exact source-preserved collections, or a concrete no-reference reason. All Evidence IDs must belong to this unit.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["unit_id", "references", "collections", "no_reference_reason"],
                "properties": {
                    "unit_id": {"type": "string"},
                    "references": {
                        "type": "array",
                        "maxItems": 12,
                        "items": reference,
                    },
                    "collections": {
                        "type": "array",
                        "maxItems": 12,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["evidence_ids", "title", "description"],
                            "properties": {
                                "evidence_ids": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 512,
                                    "items": {"type": "string"},
                                },
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "category": {"type": "string"},
                                "injection_policy": {
                                    "type": "string",
                                    "enum": ["always", "on_demand", "contextual"],
                                },
                                "usage_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "no_reference_reason": {"type": "string"},
                },
            },
        ),
        "inspect_reference_unit_coverage": function(
            "inspect_reference_unit_coverage",
            "Inspect exhaustive EvidenceTree-unit coverage and the compact candidate notes accumulated across the shared Reference Child Session.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 80},
                    "include_reviews": {"type": "boolean"},
                },
            },
        ),
        "list_reference_candidates": function(
            "list_reference_candidates",
            "List the one compact ranked page of positive-score ordinary Python Reference candidates. In Reference Build, already-reviewed catalog candidates are excluded and pagination is intentionally disabled.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "page": {"type": "integer", "minimum": 1},
                    "page_size": {"type": "integer", "minimum": 1, "maximum": 20},
                    "min_reference_like_score": {"type": "integer", "minimum": 1},
                    "include_catalog_candidates": {"type": "boolean"},
                },
            },
        ),
        "inspect_reference_candidate": function(
            "inspect_reference_candidate",
            "Inspect one Python-selected EvidenceTree candidate and its Evidence descriptors before selectively reading exact source text. Follow-up form only; inspect_reference_candidates is the default candidate-reading tool. Returns Evidence descriptors, not their full text; read_evidence reads the text. Call shape: {\"candidate_id\":\"N1\",\"offset\":0,\"limit\":2}; copy the actual candidate ID from the manifest. This does not merge or rewrite SourceParts.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_id"],
                "properties": {
                    "candidate_id": {"type": "string", "description": "Ordinary Reference source-region ID. Copy ordinary_candidates[].candidate_id from the supplied manifest. Its evidence_node_id may repeat this same value. Do not pass an E evidence_id or a C catalog_id."},
                    "offset": {"type": "integer", "minimum": 0, "description": "Zero-based Evidence descriptor index within this candidate: 0 starts at the first descriptor. Counts descriptors skipped, not characters."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Number of Evidence descriptors to return, from 1 to 8. Not a character count; do not use a text length such as 4000."},
                },
            },
        ),
        "inspect_reference_candidates": function(
            "inspect_reference_candidates",
            "Inspect one to eight Python-selected EvidenceTree candidates and their Evidence descriptors in one call, starting at the first descriptor of each candidate. This is the default candidate-reading tool; use inspect_reference_candidate only when a candidate needs a nonzero offset follow-up. Returns Evidence descriptors, not their full text; read_evidence reads the text. Call shape: {\"candidate_ids\":[\"N1\",\"N2\"],\"limit\":8}; copy actual candidate IDs from the manifest and use an array even for one candidate. Duplicate entries inside one call are ignored. Validation is atomic: if any candidate_id is not reviewable, the WHOLE call is rejected and the error names every failing ID; correct those IDs and resubmit the batch. This does not merge or rewrite SourceParts.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_ids"],
                "properties": {
                    "candidate_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {"type": "string"},
                        "description": "Array of 1 to 8 distinct ordinary Reference source-region IDs copied from ordinary_candidates[].candidate_id in the supplied manifest. Use an array even for one candidate. Do not pass E evidence_id or C catalog_id values.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": "Number of Evidence descriptors to return per candidate, from 1 to 8, always starting at descriptor 0. Not a character count; do not use a text length such as 4000.",
                    },
                },
            },
        ),
        "list_evidence": function(
            "list_evidence",
            "List compact Evidence descriptors. In Trajectory Inference, the current Evidence Tree branch is the default; use search_scope='global' only if that branch lacks a needed fact.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "page": {"type": "integer", "minimum": 1},
                    "page_size": {"type": "integer", "minimum": 1, "maximum": 80},
                    "source_id": {"type": "string"},
                    "branch_node_id": {"type": "string"},
                    "search_scope": {"type": "string", "enum": ["group", "global"]},
                    "kind": {"type": "string"},
                    "reference_like_only": {"type": "boolean"},
                    "min_reference_like_score": {"type": "integer", "minimum": 0},
                },
            },
        ),
        "search_evidence": function(
            "search_evidence",
            "Search Evidence for a missing fact. In Trajectory Inference, the current Evidence Tree branch is the default; use search_scope='global' only when needed. Results are compact source-derived previews.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                    "branch_node_id": {"type": "string"},
                    "search_scope": {"type": "string", "enum": ["group", "global"]},
                },
            },
        ),
        "read_evidence": function(
            "read_evidence",
            "Read bounded exact Evidence text when the dispatch packet marks it needs_read or a search finds a new supporting source. Quotes may come from this response or full text in the dispatch packet. Call shape: {\"evidence_id\":\"E1\",\"offset\":0,\"limit\":4000}; copy an actual Evidence ID from a descriptor or evidence_id_samples.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id"],
                "properties": {
                    "evidence_id": {"type": "string", "description": "Citable Evidence excerpt ID. Copy evidence_id from an Evidence descriptor or one value from evidence_id_samples; catalog entries may label it source_evidence_id. Do not pass a node_id, candidate_id, or catalog_id."},
                    "offset": {"type": "integer", "minimum": 0, "description": "Zero-based character offset within this Evidence text. Counts characters skipped, not bytes, tokens, or Evidence entries."},
                    "limit": {"type": "integer", "minimum": 1, "description": "Maximum number of text characters to read from offset, subject to the configured reading limit. Not a count of Evidence entries. Use end_offset to continue when truncated."},
                },
            },
        ),
        "view_evidence_image": function(
            "view_evidence_image",
            "Purpose: inspect ONE image attached to a known Evidence item.\nRequired JSON: {\"evidence_id\":\"E1\",\"artifact_id\":\"art_000000000000000000000000\"}. Replace both example IDs with real IDs from this run.\nID source: copy evidence_id and one of its image_refs[].artifact_id from the supplied trajectory_evidence packet or a returned Evidence descriptor. The image must belong to that SAME Evidence.\nReturns: multimodal image input and metadata.\nLimits: no filesystem paths or {}. Each batched call needs both IDs. On permission denial, report the unavailable image; do not repeatedly retry the same IDs. Use sufficient text evidence when available.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id", "artifact_id"],
                "properties": {
                    "evidence_id": {"type": "string", "description": "Required E ID of the Evidence item owning this image."},
                    "artifact_id": {"type": "string", "description": "Required real image_refs[].artifact_id attached to that same Evidence item, not its text artifact ID."},
                },
            },
        ),
        "list_reference_catalogs": function(
            "list_reference_catalogs",
            "List a bounded page of source-structure candidates with repeated or sequential entries. These are structural observations from the current Evidence corpus, not fixed software categories; paginate when truncated.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 80},
                },
            },
        ),
        "inspect_reference_catalog": function(
            "inspect_reference_catalog",
            "Inspect a bounded sample of a source-structure candidate before deciding whether it should become a source-preserved Reference catalog. Call shape: {\"catalog_id\":\"C1\",\"offset\":0,\"limit\":3}; copy the actual catalog ID from the manifest.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["catalog_id"],
                "properties": {
                    "catalog_id": {"type": "string", "description": "One catalog ID copied from catalog_candidates[].catalog_id. A catalog is a repeated-entry lookup collection, not an ordinary candidate or Evidence excerpt."},
                    "offset": {"type": "integer", "minimum": 0, "description": "Zero-based catalog entry index: 0 starts at the first entry. Counts entries skipped, not characters."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Number of catalog entries to return, from 1 to 20 before stage-specific limits. Not a character count."},
                },
            },
        ),
        "inspect_reference_catalog_batch": function(
            "inspect_reference_catalog_batch",
            "Inspect one to three Reference catalogs in one call. Each catalog returns up to six deterministic samples distributed uniformly across its complete ordered range. Call shape: {\"catalog_ids\":[\"C1\"],\"entries_per_catalog\":6}; copy actual catalog IDs. This single call accepts an ID array, even for one catalog, and has no offset or limit. Use this as the default catalog-reading tool; use inspect_reference_catalog only for a genuinely ambiguous follow-up.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["catalog_ids"],
                "properties": {
                    "catalog_ids": {
                        "type": "array",
                        "description": "Array of 1 to 3 distinct catalog_candidates[].catalog_id values. Use an array even for one catalog. Do not use ordinary candidate IDs or Evidence IDs.",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string"},
                    },
                    "entries_per_catalog": {
                        "type": "integer",
                        "description": "Number of sampled entries per catalog, from 1 to 6. Samples are spread across the whole catalog; this is not a character count or a pagination offset.",
                        "minimum": 1,
                        "maximum": REFERENCE_CATALOG_SAMPLE_SIZE,
                    },
                },
            },
        ),
        "inspect_reference_coverage": function(
            "inspect_reference_coverage",
            "Report which detected source-structure catalogs have been preserved, excluded, or still need review before the Reference stage can finish. Catalog descriptors are paginated; the full ID lists remain available for review.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "catalog_offset": {"type": "integer", "minimum": 0},
                    "catalog_limit": {"type": "integer", "minimum": 1, "maximum": 80},
                },
            },
        ),
        "list_tasks": function(
            "list_tasks",
            "List validated Task descriptors available in the immutable knowledge snapshot.",
            {"type": "object", "additionalProperties": False, "properties": {"page": {"type": "integer"}, "page_size": {"type": "integer"}}},
        ),
        "read_task": function(
            "read_task",
            "Read one validated Task and its evidence citations.",
            {"type": "object", "additionalProperties": False, "required": ["task_id"], "properties": {"task_id": {"type": "string"}}},
        ),
        "list_references": function(
            "list_references",
            "List Reference descriptors. In Trajectory Inference, only References linked to cited Evidence are shown by default; use scope='global' if needed.",
            {"type": "object", "additionalProperties": False, "properties": {"page": {"type": "integer"}, "page_size": {"type": "integer"}, "scope": {"type": "string", "enum": ["related", "global"]}}},
        ),
        "read_reference": function(
            "read_reference",
            "Read one validated global Reference. Source catalogs return bounded entry descriptors; use the entry tools for exact text.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["reference_id"],
                "properties": {
                    "reference_id": {"type": "string"},
                    "entry_offset": {"type": "integer", "minimum": 0},
                    "entry_limit": {"type": "integer", "minimum": 1, "maximum": 30},
                },
            },
        ),
        "search_reference_entries": function(
            "search_reference_entries",
            "Lexically search entries inside one validated source catalog Reference, then use read_reference_entry for exact source text.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["reference_id", "query"],
                "properties": {
                    "reference_id": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                },
            },
        ),
        "read_reference_entry": function(
            "read_reference_entry",
            "Read bounded exact source text for one Reference entry. Returns already_in_evidence_packet when that exact text is already in the dispatch packet.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["reference_id", "entry_id"],
                "properties": {"reference_id": {"type": "string"}, "entry_id": {"type": "string"},
                               "offset": {"type": "integer", "minimum": 0},
                               "limit": {"type": "integer", "minimum": 1, "maximum": 8_000}},
            },
        ),
        "list_trajectories": function(
            "list_trajectories",
            "List validated inferred trajectory descriptors for cross-task Skill induction.",
            {"type": "object", "additionalProperties": False, "properties": {"page": {"type": "integer"}, "page_size": {"type": "integer"}}},
        ),
        "read_trajectory": function(
            "read_trajectory",
            "Read one validated inferred trajectory and its cited steps.",
            {"type": "object", "additionalProperties": False, "required": ["trajectory_id"], "properties": {"trajectory_id": {"type": "string"}}},
        ),
        "inspect_skill_coverage": function(
            "inspect_skill_coverage",
            "Report trajectory links and uncovered trajectories across submitted Skills. Skill submissions already return this coverage; this diagnostic is not needed in the normal Skill Child workflow.",
            {"type": "object", "additionalProperties": False, "properties": {}},
        ),
        "review_skill_coverage": function(
            "review_skill_coverage",
            "Record justified exclusions for uncovered trajectories that cannot support a grounded Skill. Covered trajectories are reviewed automatically after accepted Skill submissions; do not resubmit covered dispositions.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["dispositions"],
                "properties": {
                    "dispositions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["trajectory_id", "disposition", "rationale"],
                            "properties": {
                                "trajectory_id": {"type": "string"},
                                "disposition": {"type": "string", "enum": ["excluded"]},
                                "rationale": {"type": "string"},
                            },
                        },
                    }
                },
            },
        ),
        "review_reference_coverage": function(
            "review_reference_coverage",
            "Record a final disposition for every detected source-structure catalog. Mark covered only after submit_reference_catalog has preserved it; exclude only when it is not a reusable global Reference.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["dispositions"],
                "properties": {
                    "dispositions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["catalog_id", "disposition"],
                            "properties": {
                                "catalog_id": {"type": "string"},
                                "disposition": {"type": "string", "enum": ["covered", "excluded"]},
                            },
                        },
                    }
                },
            },
        ),
        "submit_reference_review": function(
            "submit_reference_review",
            f"Atomically review every prepared catalog and ordinary candidate. Explicitly preserve or exclude every candidate; preserved candidates require an Agent-written title and a description of one or two short sentences, at most 100 characters including spaces and punctuation, stating knowledge scope and when to consult it without listing commands, shortcuts, or examples. Provide keep_whole or semantic split decisions only for preserved candidates above {REFERENCE_PARTITION_REVIEW_CHARS} characters.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "catalog_decisions",
                    "ordinary_decisions",
                    "partition_decisions",
                    "status",
                    "summary",
                ],
                "properties": {
                    "catalog_decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["catalog_id", "disposition"],
                            "properties": {
                                "catalog_id": {"type": "string"},
                                "disposition": {
                                    "type": "string",
                                    "enum": ["preserve", "exclude"],
                                },
                                "title": {"type": "string"},
                                "description": {"type": "string", "minLength": 1, "maxLength": 100, "description": "Required when preserving. One or two short sentences, at most 100 characters including spaces and punctuation. Briefly state knowledge scope and when to consult it. Do not list commands, shortcuts, or examples."},
                                "category": {"type": "string"},
                                "injection_policy": {
                                    "type": "string",
                                    "enum": ["always", "on_demand", "contextual"],
                                },
                                "usage_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "ordinary_decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["candidate_id", "disposition"],
                            "properties": {
                                "candidate_id": {"type": "string"},
                                "disposition": {
                                    "type": "string",
                                    "enum": ["preserve", "exclude"],
                                },
                                "title": {"type": "string"},
                                "description": {"type": "string", "minLength": 1, "maxLength": 100, "description": "Required when preserving. One or two short sentences, at most 100 characters including spaces and punctuation. Briefly state knowledge scope and when to consult it. Do not list commands, shortcuts, or examples."},
                                "category": {"type": "string"},
                                "injection_policy": {
                                    "type": "string",
                                    "enum": ["always", "on_demand", "contextual"],
                                },
                                "usage_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "partition_decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["subject_id", "decision", "groups"],
                            "properties": {
                                "subject_id": {"type": "string"},
                                "decision": {
                                    "type": "string",
                                    "enum": ["keep_whole", "split"],
                                },
                                "groups": {
                                    "type": "array",
                                    "maxItems": REFERENCE_PARTITION_MAX_GROUPS,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": [
                                            "title",
                                            "description",
                                            "entry_indices",
                                        ],
                                        "properties": {
                                            "title": {"type": "string"},
                                            "description": {
                                                "type": "string",
                                                "minLength": CATALOG_DESCRIPTION_MIN_CHARS,
                                                "maxLength": 100,
                                                "description": "One or two short sentences, at most 100 characters including spaces and punctuation. Briefly state knowledge scope and when to consult it. Do not list commands, shortcuts, or examples.",
                                            },
                                            "entry_indices": {
                                                "type": "array",
                                                "items": {"type": "integer"},
                                            },
                                            "category": {"type": "string"},
                                            "injection_policy": {
                                                "type": "string",
                                                "enum": ["always", "on_demand", "contextual"],
                                            },
                                            "usage_tags": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "status": {
                        "type": "string",
                        "enum": ["succeeded", "degraded", "failed"],
                    },
                    "summary": {"type": "string"},
                    "gaps": {"type": "array", "items": {"type": "string"}},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
            },
        ),
        "submit_task_batch": function(
            "submit_task_batch",
            "Submit evidence-backed capability-family Tasks. Write generalized natural-language goals without template variables; do not scope a Task to one script, file, object, or example value unless the source-backed capability is exactly that item.",
            {"type": "object", "additionalProperties": False, "required": ["tasks"], "properties": {"tasks": {"type": "array", "minItems": 1, "items": task}}},
        ),
        "submit_task_trajectory_pair": function(
            "submit_task_trajectory_pair",
            "Submit one source-supported Task with one executable, cited Trajectory. The server assigns their IDs, binds the Trajectory.task_id to this Task, and derives Trajectory citations from its steps. This is atomic: if either side fails validation, neither is accepted. Submit only steps on the Trajectory; the task description comes from the paired Task.task.",
            {"type": "object", "additionalProperties": False,
             "required": ["task", "trajectory"],
             "properties": {
                 "task": task,
                 "trajectory": {"type": "object", "additionalProperties": False,
                     "required": ["steps"],
                     "properties": {
                         "steps": {"type": "array", "minItems": 1, "items": trajectory_step},
                     }},
             }},
        ),
        "inspect_pair_coverage": function(
            "inspect_pair_coverage",
            "Inspect accepted Task-Trajectory pairs, remaining Task capacity, and major EvidenceTree branch reading. Read more branches when the exploration report is not ready for succeeded; unread individual nodes are allowed.",
            {"type": "object", "additionalProperties": False, "properties": {}},
        ),
        "supersede_pair": function(
            "supersede_pair", "Replace one accepted pair with a newer merged pair. Both pairs stay in the audit trail; only the newer pair enters the final snapshot.",
            {"type": "object", "additionalProperties": False,
             "required": ["old_task_id", "new_task_id"],
             "properties": {"old_task_id": {"type": "string"}, "new_task_id": {"type": "string"}}},
        ),
        "report_discovery_gap": function(
            "report_discovery_gap", "Record a useful source-supported goal that cannot be given a grounded operation path. It does not enter the final Task list.",
            {"type": "object", "additionalProperties": False,
             "required": ["title", "reason", "citations"],
             "properties": {"title": {"type": "string"}, "reason": {"type": "string"},
                            "citations": {"type": "array", "minItems": 1, "items": citation}}},
        ),
        "submit_reference_batch": function(
            "submit_reference_batch",
            "Submit concise evidence-backed global Reference summaries. Do not use this for repeated source entries; use submit_reference_catalog for those.",
            {"type": "object", "additionalProperties": False, "required": ["references"], "properties": {"references": {"type": "array", "minItems": 1, "items": reference}}},
        ),
        "submit_reference_catalog": function(
            "submit_reference_catalog",
            "Materialize one detected source-structure catalog as exact, independently queryable Reference entries. Provide a one- or two-sentence description of its knowledge scope and when to query it during any task. The Harness creates every entry and citation directly from current Evidence; do not rewrite or enumerate the entries yourself.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["catalog_id", "title", "description"],
                "properties": {
                    "catalog_id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {
                        "type": "string",
                        "minLength": CATALOG_DESCRIPTION_MIN_CHARS,
                        "maxLength": 100,
                        "description": "One or two short sentences, at most 100 characters including spaces and punctuation. Briefly state knowledge scope and when to consult it. Do not list commands, shortcuts, or examples.",
                    },
                    "category": {"type": "string"},
                    "injection_policy": {"type": "string", "enum": ["always", "on_demand", "contextual"]},
                    "usage_tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        ),
        "submit_reference_catalog_batch": function(
            "submit_reference_catalog_batch",
            "Materialize several already-inspected detected source catalogs in one turn. Each selected catalog becomes a separate exact, queryable Reference; the Harness creates entries and citations directly from current Evidence.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["catalogs"],
                "properties": {
                    "catalogs": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 64,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["catalog_id", "title", "description"],
                            "properties": {
                                "catalog_id": {"type": "string"},
                                "title": {"type": "string"},
                                "description": {
                                    "type": "string",
                                    "minLength": CATALOG_DESCRIPTION_MIN_CHARS,
                                    "maxLength": 100,
                                    "description": "One or two short sentences, at most 100 characters including spaces and punctuation. Briefly state knowledge scope and when to consult it. Do not list commands, shortcuts, or examples.",
                                },
                                "category": {"type": "string"},
                                "injection_policy": {
                                    "type": "string",
                                    "enum": ["always", "on_demand", "contextual"],
                                },
                                "usage_tags": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    }
                },
            },
        ),
        "submit_reference_collection": function(
            "submit_reference_collection",
            "Materialize an Agent-selected set of already-read textual Evidence as exact, queryable Reference entries when no structural catalog candidate applies. Provide a one- or two-sentence description of its knowledge scope and when to query it during any task. Image-only Evidence remains available through view_evidence_image and cannot become a text Reference entry.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_ids", "title", "description"],
                "properties": {
                    "evidence_ids": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "string"}},
                    "title": {"type": "string"},
                    "description": {
                        "type": "string",
                        "minLength": CATALOG_DESCRIPTION_MIN_CHARS,
                        "maxLength": 100,
                        "description": "One or two short sentences, at most 100 characters including spaces and punctuation. Briefly state knowledge scope and when to consult it. Do not list commands, shortcuts, or examples.",
                    },
                    "category": {"type": "string"},
                    "injection_policy": {"type": "string", "enum": ["always", "on_demand", "contextual"]},
                    "usage_tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        ),
        "submit_trajectory": function(
            "submit_trajectory",
            "Submit one Trajectory for a dispatched task_slot. Supply steps. Cite Evidence on every step. The server binds task_id to the dispatched Task, uses its Task.task as the task description and derives top-level citations from steps. If a grounded path is unavailable, report_trajectory_gap for this Task slot instead.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["task_slot", "steps"],
                "properties": {
                    "task_slot": {"type": "integer", "minimum": 1},
                    "steps": {"type": "array", "minItems": 1, "items": trajectory_step},
                },
            },
        ),
        "report_trajectory_gap": function(
            "report_trajectory_gap",
            "Record a precise evidence gap for one dispatched task_slot when a source-backed operating path cannot be established. The slot is then complete without a trajectory.",
            {
                "type": "object", "additionalProperties": False,
                "required": ["task_slot", "reason"],
                "properties": {
                    "task_slot": {"type": "integer", "minimum": 1},
                    "reason": {"type": "string", "minLength": 1},
                },
            },
        ),
        "submit_skill_batch": function(
            "submit_skill_batch",
            "Purpose: save source-grounded, reusable Skills for the dispatched trajectories.\nRequired JSON shape (replace all example text and IDs with source-supported values): {\"skills\":[{\"title\":\"Example operation\",\"description\":\"When to use it.\",\"applicability\":\"Documented environment.\",\"preconditions\":\"\",\"trajectory_ids\":[\"TR1\"],\"procedure\":[{\"ordinal\":1,\"instruction\":\"Perform the documented action.\",\"citations\":[{\"evidence_id\":\"E1\",\"quote\":\"Supporting source text.\"}]}]}]}.\nTypes: skills, procedure, and citations are arrays of OBJECTS. Every quote must be one nonempty STRING, including numeric-looking source text. Never use null, a number, array, or object as quote. Image citations instead use a real inspected image_artifact_id.\nFields: only skills at the outer level. Each Skill uses the fields in the example; derive Task links on the server. description must be at most 100 characters including spaces and punctuation, briefly stating when to use the Skill. preconditions is a required plain string; use \"\" when none apply. Each procedure step has only ordinal, instruction, and citations; do not submit expected_observation.\nRecovery: correct only rejected Skills and field paths named in validation errors; do not resubmit accepted Skills. The result reports validated coverage. Finish when all dispatched trajectories are covered or explicitly excluded.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["skills"],
                "properties": {
                    "skills": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["title", "description", "applicability", "preconditions", "procedure", "trajectory_ids"],
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string", "minLength": 3, "maxLength": 100, "description": "Briefly state when to use this Skill. At most 100 characters, including spaces and punctuation."},
                                "applicability": {"type": "string"},
                                "preconditions": {
                                    "type": "string",
                                    "description": "Required starting conditions before the first procedure step, combined into one string. Submit an empty string explicitly if none apply.",
                                    "maxLength": 24_016,
                                },
                                "procedure": {"type": "array", "minItems": 1, "description": "An array of step OBJECTS, never strings or nested arrays. Each object has only ordinal (integer), instruction (string), and citations (array of citation objects).", "items": skill_step},
                                "trajectory_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                            },
                        },
                    }
                },
            },
        ),
        "submit_stage_result": function(
            "submit_stage_result",
            "Declare the controlled terminal result for the current stage after all intended candidate submissions.",
            stage_result,
        ),
    }
