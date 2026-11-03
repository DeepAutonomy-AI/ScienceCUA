"""Source-only provenance, coverage, and linkage diagnostics for a pipeline run."""

from __future__ import annotations

from typing import Any

from ..models import (
    CandidateValidation, CorpusSnapshot, EvidenceCitation, KnowledgeSnapshot,
    SkillCandidate, TrajectoryCandidate,
)
from .common import _compact
from .validation import _contains_template_placeholder


def build_pipeline_quality_report(
    corpus: CorpusSnapshot,
    knowledge: KnowledgeSnapshot,
    trajectories: list[TrajectoryCandidate],
    skills: list[SkillCandidate],
    validations: list[CandidateValidation],
    *,
    reference_coverage: dict[str, Any] | None = None,
    skill_coverage: dict[str, Any] | None = None,
    version_alignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a source-only, non-generative health report for one pipeline run.

    This intentionally measures provenance and linkage rather than claiming that
    every source fact should become a Task or Skill. It is written alongside the
    exported Skills so later runs can diagnose undercoverage without consulting a
    benchmark or sending benchmark data to an Agent.
    """

    citation_counts = {
        "task": 0,
        "reference": 0,
        "trajectory": 0,
        "trajectory_step": 0,
        "skill_step": 0,
    }
    cited_evidence_ids: set[str] = set()

    def record(citations: list[EvidenceCitation], category: str) -> None:
        citation_counts[category] += len(citations)
        cited_evidence_ids.update(item.evidence_id for item in citations)

    for item in knowledge.tasks:
        record(item.citations, "task")
    for item in knowledge.references:
        record(item.citations, "reference")
    for item in trajectories:
        record(item.citations, "trajectory")
        for step in item.steps:
            record(step.citations, "trajectory_step")
    for item in skills:
        for step in item.procedure:
            record(step.citations, "skill_step")

    task_ids = {item.task_id for item in knowledge.tasks}
    trajectory_ids = {item.trajectory_id for item in trajectories}
    tasks_by_id = {item.task_id: item for item in knowledge.tasks}
    trajectories_by_id = {item.trajectory_id: item for item in trajectories}
    def skill_task_ids(skill: SkillCandidate) -> set[str]:
        return {
            trajectories_by_id[trajectory_id].task_id
            for trajectory_id in skill.trajectory_ids
            if trajectory_id in trajectories_by_id
        }

    linked_task_ids = {task_id for item in skills for task_id in skill_task_ids(item)}
    linked_trajectory_ids = {
        trajectory_id for item in skills for trajectory_id in item.trajectory_ids
    }
    invalid_validation_count = sum(1 for item in validations if not item.accepted)
    reference_assets = {
        "summary_count": sum(1 for item in knowledge.references if item.reference_mode == "summary"),
        "source_catalog_count": sum(
            1 for item in knowledge.references if item.reference_mode == "source_catalog"
        ),
        "source_catalog_entry_count": sum(len(item.entries) for item in knowledge.references),
        "injection_policy_counts": {},
    }
    for reference in knowledge.references:
        policy = reference.injection_policy
        reference_assets["injection_policy_counts"][policy] = (
            reference_assets["injection_policy_counts"].get(policy, 0) + 1
        )

    source_coverage: dict[str, dict[str, Any]] = {}
    for evidence in corpus.evidence:
        source = source_coverage.setdefault(
            evidence.source_id,
            {
                "total_evidence": 0,
                "cited_evidence": 0,
                "evidence_kinds": {},
                "source_origin": evidence.locator.get("source_origin", "user"),
                "version_relation": evidence.locator.get("version_relation"),
                "evidence_applicability": evidence.locator.get("evidence_applicability"),
                "resolved_version": evidence.locator.get(
                    "source_resolved_version", evidence.resolved_version
                ),
            },
        )
        source["total_evidence"] += 1
        source["evidence_kinds"][evidence.kind] = source["evidence_kinds"].get(evidence.kind, 0) + 1
        if evidence.evidence_id in cited_evidence_ids:
            source["cited_evidence"] += 1
    for source in source_coverage.values():
        total = source["total_evidence"]
        source["cited_evidence_ratio"] = round(source["cited_evidence"] / total, 4) if total else 0.0

    return {
        "schema_version": "v6",
        "scope": "source_only_pipeline_quality",
        "corpus_snapshot_id": corpus.corpus_snapshot_id,
        "knowledge_snapshot_id": knowledge.knowledge_snapshot_id,
        "counts": {
            "evidence": len(corpus.evidence),
            "tasks": len(knowledge.tasks),
            "references": len(knowledge.references),
            "trajectories": len(trajectories),
            "skills": len(skills),
        },
        "citations": {
            **citation_counts,
            "unique_evidence_cited": len(cited_evidence_ids),
            "evidence_citation_ratio": round(len(cited_evidence_ids) / len(corpus.evidence), 4)
            if corpus.evidence
            else 0.0,
        },
        "skill_contract": {
            "skills_with_preconditions": sum(1 for item in skills if item.preconditions),
            "skills_linking_multiple_tasks": sum(
                1 for item in skills if len(skill_task_ids(item)) > 1
            ),
            "skills_linking_multiple_trajectories": sum(
                1 for item in skills if len(set(item.trajectory_ids)) > 1
            ),
            "procedure_steps": sum(len(item.procedure) for item in skills),
            "skills_with_placeholder_syntax": sum(
                1
                for item in skills
                if _contains_template_placeholder(
                    [
                        item.title,
                        item.description,
                        item.applicability,
                        item.preconditions,
                        *(step.instruction for step in item.procedure),
                    ]
                )
            ),
            "atomic_singleton_skills": 0,
        },
        "linkage": {
            "tasks_without_skill": sorted(task_ids - linked_task_ids),
            "trajectories_without_skill": sorted(trajectory_ids - linked_trajectory_ids),
            "unknown_task_links": sorted(linked_task_ids - task_ids),
            "unknown_trajectory_links": sorted(linked_trajectory_ids - trajectory_ids),
        },
        "validation": {
            "total": len(validations),
            "accepted": len(validations) - invalid_validation_count,
            "rejected": invalid_validation_count,
        },
        "reference_assets": reference_assets,
        "source_evidence_coverage": source_coverage,
        "source_status_counts": corpus.source_status_counts,
        "version_alignment": version_alignment or {},
        "reference_coverage": reference_coverage or {},
        "skill_coverage": skill_coverage or {},
        "notes": [
            "This report is generated only from the current run's corpus and candidates.",
            "Evidence citation coverage is a diagnostic, not a requirement that every EvidenceExcerpt become a Skill.",
            "All exported trajectories and Skills remain inferred until verified in the target software.",
        ],
    }
