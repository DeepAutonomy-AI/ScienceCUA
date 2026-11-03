from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.knowledge import KnowledgeWorkspace, _knowledge_tool_catalog, validate_trajectories
from app.models import (
    CorpusSnapshot, EvidenceCitation, EvidenceExcerpt, KnowledgeSnapshot, TargetProfile,
    TaskCandidate, TrajectoryCandidate, TrajectoryDraft, normalize_saved_trajectory_payload,
)
from app.storage import RunStore
from tests.knowledge_fixtures import trajectory_submission, trajectory_draft


QUOTE = "Press Ctrl+R to render the graph."
RETIRED = {"trajectory_type": "reference_guided", "goal": "Retired duplicate task description", "reference_ids": ["R1"], "reference_trigger_type": "required", "confidence": 0.9,
           "caveats": ["Old note"], "unsupported_facets": ["Old unsupported component"]}


def submitted(**updates):
    citation = {"evidence_id": "E1", "quote": QUOTE}
    return trajectory_submission(instruction=QUOTE, citations=[citation],
                                 expected_observation="The graph is rendered.", **updates)


def draft(**updates):
    return trajectory_draft(task_id="T1", instruction=QUOTE,
                            citations=[{"evidence_id": "E1", "quote": QUOTE}],
                            expected_observation="The graph is rendered.", **updates)


def workspace(tmp_path):
    target = TargetProfile(software_name="Example", target_version="1", platform="General")
    ev = EvidenceExcerpt(evidence_id="E1", source_id="SRC1", source_part_id="part_one",
                         text_artifact_id="art_test", start_offset=0, end_offset=len(QUOTE),
                         content_sha256="test", kind="instruction", title="Render", content=QUOTE,
                         locator={}, extraction_method="test")
    corpus = CorpusSnapshot(corpus_snapshot_id="CP1", run_id="run_trajectory_model", target_profile=target,
                            source_results=[], evidence=[ev], evidence_count=1, source_part_count=1,
                            source_status_counts={})
    tasks = [TaskCandidate(task_id=f"T{index}", corpus_snapshot_id="CP1", task=f"Render graph {index}",
                           category="Rendering", citations=[EvidenceCitation(evidence_id="E1", quote=QUOTE)])
             for index in (1, 2)]
    knowledge = KnowledgeSnapshot(knowledge_snapshot_id="KP1", run_id=corpus.run_id,
                                  corpus_snapshot_id="CP1", corpus_snapshot_artifact_id="art_corpus",
                                  target_profile=target, tasks=tasks)
    store = RunStore(tmp_path, corpus.run_id)
    store.initialize({"target_profile": target.model_dump(mode="json")})
    return KnowledgeWorkspace(store=store, settings=Settings(data_dir=tmp_path), target_profile=target,
                              stage="trajectory_inference", corpus=corpus, corpus_artifact_id="art_corpus",
                              knowledge=knowledge, knowledge_artifact_id="art_knowledge", trajectory_tasks=tasks)


def test_simplified_fields_and_both_submission_contracts():
    assert set(TrajectoryDraft.model_fields) == {"task_id", "steps", "citations"}
    assert set(TrajectoryCandidate.model_fields) == set(TrajectoryDraft.model_fields) | {
        "trajectory_id", "knowledge_snapshot_id", "status", "inferred", "created_at"}
    tools = _knowledge_tool_catalog()
    pair = tools["submit_task_trajectory_pair"]["function"]["parameters"]["properties"]["trajectory"]
    assert set(pair["properties"]) == {"steps"}
    assert set(pair["required"]) == {"steps"}
    standalone = tools["submit_trajectory"]["function"]["parameters"]
    assert set(standalone["properties"]) == {"task_slot", "steps"}
    assert set(standalone["required"]) == {"task_slot", "steps"}


def test_new_drafts_and_standalone_submissions_reject_deleted_fields(tmp_path):
    view = workspace(tmp_path)
    for field, value in RETIRED.items():
        with pytest.raises(ValidationError, match='Extra inputs'):
            TrajectoryDraft.model_validate(draft(**{field: value}))
        with pytest.raises(ValidationError, match='Extra inputs'):
            view.submit_trajectory(1, **submitted(**{field: value}))
    assert view.submitted_trajectories == []


def test_saved_trajectories_keep_binding_steps_and_inferred_flag_without_mutating_history():
    raw = {**draft(), **RETIRED, "trajectory_id": "TR1", "knowledge_snapshot_id": "KP1"}
    before = copy.deepcopy(raw)
    saved = TrajectoryCandidate.model_validate(raw).model_dump(mode="json")
    assert not set(RETIRED) & saved.keys()
    assert saved["task_id"] == "T1" and saved["trajectory_id"] == "TR1"
    assert saved["steps"] and saved["citations"] and saved["inferred"] is True
    assert normalize_saved_trajectory_payload(raw)["task_id"] == "T1"
    assert raw == before


def test_server_binds_trajectory_to_selected_task_and_derives_citations(tmp_path):
    view = workspace(tmp_path)
    result = view.submit_trajectory(2, **submitted())
    assert result["accepted_by_schema"] is True
    trajectory = view.submitted_trajectories[0]
    assert trajectory.task_id == "T2"
    assert trajectory.citations == trajectory.steps[0].citations
    assert not set(RETIRED) & trajectory.model_dump().keys()
    view.trajectories_by_id[trajectory.trajectory_id] = trajectory
    listed = view.list_trajectories()["items"][0]
    assert listed["task_id"] == "T2"
    assert not {"goal", "title", "trajectory_type"} & listed.keys()
    assert "goal" not in view.read_trajectory(trajectory.trajectory_id)
    with pytest.raises(ValueError, match="assigned by the dispatched Task"):
        view.submit_trajectory(1, **submitted(task_id="T2"))


def test_unknown_task_binding_is_rejected_and_trajectory_needs_grounded_steps(tmp_path):
    view = workspace(tmp_path)
    value = TrajectoryCandidate(trajectory_id="TR1", knowledge_snapshot_id="KP1",
                                **draft())
    accepted, report = validate_trajectories([value], view.evidence_by_id, view.tasks_by_id)
    assert len(accepted) == 1, report
    accepted, report = validate_trajectories([value.model_copy(update={"task_id": "T99"})],
                                            view.evidence_by_id, view.tasks_by_id)
    assert not accepted
    assert report[0].errors == ["Unknown Task ID: T99"]


