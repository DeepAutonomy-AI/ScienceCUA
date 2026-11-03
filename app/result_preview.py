"""Read-only frontend projection of committed artifacts and completed Children."""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .knowledge import _compact, _stable_id, load_corpus_snapshot, load_knowledge_snapshot, validate_skills
from .models import (
    SkillCandidate, SkillDraft, TrajectoryCandidate,
    normalize_saved_citations, normalize_saved_skill_payload,
    normalize_saved_task_payload,
    normalize_saved_target_profile_payload,
    normalize_saved_trajectory_payload,
)
from .storage import RunStore
from .reference_text import reference_text_payload


def read_source_previews(store: RunStore, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose source metadata before a validated corpus exists, without creating results."""
    manifests = [json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted((store.root / "sources").glob("*/manifest.json"))]
    if not manifests and state.get("acquisition_artifact_id"):
        manifests = store.read_json_artifact(state["acquisition_artifact_id"]).get("manifests", [])
    if manifests:
        return [{
            "source_id": item["source_id"], "display_name": item["display_name"],
            "modality": item["primary_modality"], "status": item.get("status", "pending"),
            "is_preview": True,
            "manifest": {key: item[key] for key in (
                "origin_url", "final_url", "resolved_version", "version_conflict",
                "source_origin", "source_priority", "source_metadata",
            ) if key in item},
        } for item in manifests]

    if state.get("selected_source_artifact_id"):
        selected = store.read_json_artifact(state["selected_source_artifact_id"])
        inputs = [selected["input"]]
    else:
        request_path = store.root / "request.json"
        inputs = json.loads(request_path.read_text(encoding="utf-8")).get("inputs", []) if request_path.exists() else []
    sources = []
    for item in inputs:
        metadata = item.get("source_metadata") or {}
        media_type = (item.get("declared_media_type") or "").split(";", 1)[0].lower()
        name = str(item.get("display_name") or "")
        modality = metadata.get("modality") or (
            "pdf" if media_type == "application/pdf" or name.lower().endswith(".pdf") else
            "html" if media_type in {"text/html", "application/xhtml+xml"} or name.lower().endswith((".html", ".htm")) else
            "unknown"
        )
        sources.append({
            "input_id": item["input_id"], "display_name": name or item.get("url") or item["input_id"],
            "modality": modality, "status": "pending", "is_preview": True,
            "manifest": {
                "origin_url": item.get("url"),
                "resolved_version": metadata.get("document_version") or item.get("user_declared_version") or "unknown",
                "source_origin": item.get("source_origin", "user"),
                "source_priority": item.get("source_priority"), "source_metadata": metadata,
            },
        })
    return sources


def _with_text_only_references(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return result
    if 'references' not in result and 'references' not in (result.get('knowledge_snapshot') or {}):
        return result
    enriched = dict(result)
    for key in (None, "knowledge_snapshot"):
        container = result if key is None else (result.get(key) or {})
        if "references" not in container:
            continue
        updated = {**container, "references": [reference_text_payload(r)
                                               for r in container["references"]]}
        if key is None:
            enriched.update(updated)
        else:
            enriched[key] = updated
    return enriched


def completed_child_skills(store: RunStore, state: dict[str, Any], knowledge: Any, corpus: Any) -> tuple[list[dict[str, Any]], int, int]:
    attempts = [item for item in store.list_attempts() if item.stage == "skill_induction"]
    completed = {}
    for attempt in attempts:
        if attempt.status != "succeeded":
            continue
        capsule = store.read_json_artifact(attempt.capsule_artifact_id)
        if set(capsule["input_artifact_ids"]) == {
            state["corpus_snapshot_artifact_id"], state["knowledge_snapshot_artifact_id"]
        }:
            completed[attempt.attempt_id] = set(capsule["scope"].get("trajectory_ids", []))
    # Match historical acceptance to the exact candidate ID, not merely to a
    # model proposal. Open/failed Children and rejected proposals stay hidden.
    pending: dict[str, SkillCandidate] = {}
    accepted: dict[str, SkillCandidate] = {}
    if store.audit_path.exists():
        with store.audit_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A live writer may have an incomplete last line.
                payload = event.get("payload", {})
                if payload.get("tool") != "submit_skill_batch":
                    continue
                if event["event_type"] == "capability_tool_call":
                    scope = completed.get(payload.get("attempt_id"))
                    if scope is None:
                        continue
                    for raw in payload.get("arguments", {}).get("skills", []):
                        try:
                            draft = SkillDraft.model_validate(normalize_saved_skill_payload(raw))
                            if not set(draft.trajectory_ids) <= scope:
                                continue
                            identity = (
                                knowledge.knowledge_snapshot_id, _compact(draft.title).lower(),
                                # Retired fields remain part of historical IDs only.
                                *sorted(raw.get("reference_ids", [])), *sorted(draft.trajectory_ids),
                            )
                            legacy_id = _stable_id("skl", *identity)
                            skill_id = (
                                store.canonical_id("SK", legacy_id, "skl", *identity)
                                if hasattr(store, "canonical_id") else legacy_id
                            )
                            candidate = SkillCandidate(
                                **draft.model_dump(mode="json"),
                                skill_id=skill_id,
                                knowledge_snapshot_id=knowledge.knowledge_snapshot_id,
                                created_at=event["timestamp"],
                            )
                            pending[candidate.skill_id] = candidate
                        except (ValidationError, TypeError, ValueError):
                            continue
                elif event["event_type"] == "knowledge_tool_observation" and payload.get("stage") == "skill_induction":
                    for candidate_id in payload.get("observation", {}).get("candidate_ids", []):
                        if candidate_id in pending:
                            accepted[candidate_id] = pending.pop(candidate_id)
    trajectories = [TrajectoryCandidate.model_validate(item) for item in state.get("validated_trajectories", [])]
    validated, _ = validate_skills(list(accepted.values()),
        {item.evidence_id: item for item in corpus.evidence},
        {item.task_id: item for item in knowledge.tasks},
        {item.trajectory_id: item for item in trajectories})
    return [item.model_dump(mode="json") for item in validated], len(completed), len(attempts)


def read_result_preview(store: RunStore, state: dict[str, Any]) -> dict[str, Any] | None:
    result = store.read_result()
    if result:
        normalized = normalize_saved_citations(result)
        for container in (normalized, normalized.get("snapshot"), normalized.get("knowledge_snapshot")):
            if isinstance(container, dict) and "target_profile" in container:
                container["target_profile"] = normalize_saved_target_profile_payload(container["target_profile"])
        for task_container in (normalized, normalized.get("knowledge_snapshot", {})):
            if isinstance(task_container, dict) and "tasks" in task_container:
                task_container["tasks"] = [normalize_saved_task_payload(task) for task in task_container["tasks"]]
        if "skills" in normalized:
            normalized["skills"] = [normalize_saved_skill_payload(skill) for skill in normalized["skills"]]
        if "trajectories" in normalized:
            normalized["trajectories"] = [normalize_saved_trajectory_payload(trajectory) for trajectory in normalized["trajectories"]]
        if normalized != result:
            result = normalized
    if state.get("phase") == "completed" or not state.get("corpus_snapshot_artifact_id"):
        return _with_text_only_references(result)
    preview = dict(result or {})
    corpus = load_corpus_snapshot(store, state["corpus_snapshot_artifact_id"])
    preview.setdefault("snapshot", corpus.model_dump(mode="json"))
    children_done = children_total = 0
    if state.get("knowledge_snapshot_artifact_id"):
        knowledge = load_knowledge_snapshot(store, state["knowledge_snapshot_artifact_id"])
        preview.setdefault("knowledge_snapshot", knowledge.model_dump(mode="json"))
        preview.setdefault("tasks", [item.model_dump(mode="json") for item in knowledge.tasks])
        preview.setdefault("references", [item.model_dump(mode="json") for item in knowledge.references])
        preview.setdefault("trajectories", [normalize_saved_trajectory_payload(trajectory)
                                             for trajectory in state.get("validated_trajectories", [])])
        if not preview.get("skills"):
            skills, children_done, children_total = completed_child_skills(store, state, knowledge, corpus)
            preview["skills"] = skills
    else:
        for key, artifact_key in (("references", "validated_reference_artifact_id"), ("tasks", "validated_task_artifact_id")):
            if state.get(artifact_key) and key not in preview:
                artifact = store.read_json_artifact(state[artifact_key])
                preview[key] = ([normalize_saved_task_payload(task) for task in artifact[key]]
                                if key == "tasks" else artifact[key])
    preview["preview"] = {"is_partial": True, "completed_skill_children": children_done,
        "total_skill_children": children_total, "final_skills_committed": bool(state.get("t7_skill_complete"))}
    return _with_text_only_references(preview)
