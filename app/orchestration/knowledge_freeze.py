from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..knowledge import validate_references, validate_tasks
from ..models import CorpusSnapshot, KnowledgeSnapshot, ReferenceCandidate, TargetProfile, TaskCandidate
from ..storage import RunStore


@dataclass(frozen=True, slots=True)
class KnowledgeFreezeArtifacts:
    knowledge: KnowledgeSnapshot
    knowledge_artifact_id: str
    validation_artifact_id: str


def freeze_knowledge(
    *,
    store: RunStore,
    target_profile: TargetProfile,
    corpus: CorpusSnapshot,
    corpus_artifact_id: str,
    tasks: list[TaskCandidate],
    references: list[ReferenceCandidate],
) -> KnowledgeFreezeArtifacts:
    """Validate the two completed candidate batches and freeze their exact union."""

    if any(item.corpus_snapshot_id != corpus.corpus_snapshot_id for item in [*tasks, *references]):
        raise ValueError("Task or Reference belongs to a different CorpusSnapshot")
    evidence_by_id = {item.evidence_id: item for item in corpus.evidence}
    validated_tasks, task_validation = validate_tasks(tasks, evidence_by_id, corpus.evidence_trees)
    validated_references, reference_validation = validate_references(references, evidence_by_id)
    if len(validated_tasks) != len(tasks) or len(validated_references) != len(references):
        raise ValueError("Previously validated Task or Reference failed KnowledgeSnapshot revalidation")

    identity = {
        "run_id": store.run_id,
        "corpus_snapshot_id": corpus.corpus_snapshot_id,
        "task_ids": [item.task_id for item in validated_tasks],
        "reference_ids": [item.reference_id for item in validated_references],
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    knowledge = KnowledgeSnapshot(
        knowledge_snapshot_id=store.canonical_id(
            "KP", f"knowledge_{digest[:24]}", "knowledge_snapshot", digest
        ),
        run_id=store.run_id,
        corpus_snapshot_id=corpus.corpus_snapshot_id,
        corpus_snapshot_artifact_id=corpus_artifact_id,
        target_profile=target_profile,
        tasks=validated_tasks,
        references=validated_references,
        task_validation=task_validation,
        reference_validation=reference_validation,
    )
    validation_artifact = store.put_json_artifact(
        {
            "schema_version": "knowledge-freeze-validation.v1",
            "corpus_snapshot_id": corpus.corpus_snapshot_id,
            "task_count": len(validated_tasks),
            "reference_count": len(validated_references),
            "task_validation": [item.model_dump(mode="json") for item in task_validation],
            "reference_validation": [item.model_dump(mode="json") for item in reference_validation],
        },
        kind="knowledge_freeze_validation",
        metadata={"stage": "knowledge_freeze"},
        preserve_text_integrity=True,
    )
    knowledge_artifact = store.put_json_artifact(
        knowledge.model_dump(mode="json"),
        kind="knowledge_snapshot",
        metadata={"stage": "knowledge_freeze"},
        preserve_text_integrity=True,
    )
    return KnowledgeFreezeArtifacts(
        knowledge=knowledge,
        knowledge_artifact_id=knowledge_artifact.artifact_id,
        validation_artifact_id=validation_artifact.artifact_id,
    )
