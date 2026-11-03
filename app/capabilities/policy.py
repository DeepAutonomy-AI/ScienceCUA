from __future__ import annotations

from typing import Final


SOURCE_COMMON: Final[frozenset[str]] = frozenset(
    {
        "inspect_source",
        "submit_evidence_revision",
        "get_source_outline",
        "read_source_content",
        "submit_evidence_plan",
        # Retained below for the explicitly selected legacy comparison modes.
        "read_artifact",
        "list_source_parts",
        "read_source_part",
        "search_source",
        "curate_evidence",
        "submit_source_structure",
        "submit_source_result",
    }
)
SOURCE_MODAL: Final[dict[str, frozenset[str]]] = {
    "pdf": frozenset(
        {
            "pdf_list_structure",
            "get_pdf_document_structure",
            "read_pdf_content",
            "read_evidence_node",
        }
    ),
    "html": frozenset({"html_list_navigation", "get_root_structure", "read_evidence_node"}),
}

KNOWLEDGE_COMMON: Final[frozenset[str]] = frozenset(
    {
        "inspect_knowledge_scope",
        "list_evidence",
        "search_evidence",
        "read_evidence",
        "view_evidence_image",
        "submit_stage_result",
    }
)
KNOWLEDGE_STAGE: Final[dict[str, frozenset[str]]] = {
    "task_trajectory": frozenset(
        {
            "inspect_task_outline", "read_task_node", "read_task_nodes",
            "list_references", "read_reference", "search_reference_entries",
            "read_reference_entry", "submit_task_trajectory_pair",
            "inspect_pair_coverage", "supersede_pair", "report_discovery_gap",
        }
    ),
    "task_discovery": frozenset(
        {
            "inspect_task_outline",
            "read_task_node",
            "read_task_nodes",
            "submit_task_batch",
        }
    ),
    "reference_build": frozenset(
        {
            "inspect_reference_candidate",
            "inspect_reference_candidates",
            "inspect_reference_catalog",
            "inspect_reference_catalog_batch",
            "submit_reference_review",
        }
    ),
    "trajectory_inference": frozenset(
        {
            "list_references",
            "read_reference",
            "search_reference_entries",
            "read_reference_entry",
            "submit_trajectory",
            "report_trajectory_gap",
        }
    ),
    "skill_induction": frozenset(
        {
            "read_task_node",
            "read_task_nodes",
            "list_tasks",
            "read_task",
            "list_trajectories",
            "read_trajectory",
            "submit_skill_batch",
            "review_skill_coverage",
        }
    ),
}


def tools_for_stage(stage: str, *, modality: str | None = None) -> frozenset[str]:
    """Return the maximum business-tool set a stage may ever receive.

    This list mirrors the two legacy Python executors. It is deliberately kept
    independent from an executor instance so a widened executor cannot silently
    widen a Child Session's authority.
    """

    normalized = stage.strip().casefold()
    if normalized == "root":
        return frozenset()
    if normalized == "source_discovery":
        # Native web tools are provided by the isolated DSH overlay; this is
        # the only application capability the discovery Child can receive.
        return frozenset({"save_source_candidate", "submit_source_candidates",
                          "list_discovered_sources", "inspect_source_candidate", "next_source_validation"})
    if normalized in {"source_review", "pdf_source_review", "html_source_review"}:
        selected = modality
        if normalized.startswith("pdf_"):
            selected = "pdf"
        elif normalized.startswith("html_"):
            selected = "html"
        selected = (selected or "").strip().casefold()
        if selected not in SOURCE_MODAL:
            raise ValueError("source_review requires modality 'pdf' or 'html'")
        return SOURCE_COMMON | SOURCE_MODAL[selected]
    if normalized not in KNOWLEDGE_STAGE:
        raise ValueError(f"No capability policy is defined for stage {stage!r}")
    if normalized == "skill_induction":
        return (KNOWLEDGE_COMMON - {"inspect_knowledge_scope", "search_evidence"}) | KNOWLEDGE_STAGE[normalized]
    if normalized == "trajectory_inference":
        return (KNOWLEDGE_COMMON - {"inspect_knowledge_scope"}) | KNOWLEDGE_STAGE[normalized]
    return KNOWLEDGE_COMMON | KNOWLEDGE_STAGE[normalized]
