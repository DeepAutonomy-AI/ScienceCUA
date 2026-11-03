from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.knowledge import KnowledgeToolExecutor, KnowledgeWorkspace, _knowledge_tool_catalog, validate_tasks
from app.models import CorpusSnapshot, EvidenceExcerpt, TaskCandidate, TaskDraft, TargetProfile, normalize_saved_task_payload
from app.storage import RunStore
from task_model_helpers import first_level_tree

from app.contracts import TargetProfileSnapshot


def evidence(evidence_id="E1", *, source_id="SRC1", part_id="part_one", start=0, end=20):
    return EvidenceExcerpt(
        evidence_id=evidence_id, source_id=source_id, source_part_id=part_id,
        text_artifact_id="art_test", start_offset=start, end_offset=end,
        content_sha256="test", kind="instruction", title="Render", content="Render the graph.",
        locator={}, extraction_method="test",
    )


def payload(**updates):
    return {"task": "Render a graph", "category": "Operations",
            "citations": [{"evidence_id": "E1", "quote": "Render the graph."}], **updates}


def candidate(**updates):
    return TaskCandidate(task_id="T1", corpus_snapshot_id="CP1", **payload(**updates))


def test_task_contract_fields_and_required_defaults():
    assert set(TaskDraft.model_fields) == {"task", "category", "citations"}
    assert {name for name, field in TaskDraft.model_fields.items() if field.is_required()} == {"task", "category", "citations"}
    assert set(TaskDraft.model_validate(payload()).model_dump()) == {"task", "category", "citations"}
    assert set(TaskCandidate.model_fields) == set(TaskDraft.model_fields) | {"task_id", "corpus_snapshot_id", "status", "created_at"}
    tools = _knowledge_tool_catalog()
    for spec in (tools["submit_task_batch"]["function"]["parameters"]["properties"]["tasks"]["items"],
                 tools["submit_task_trajectory_pair"]["function"]["parameters"]["properties"]["task"]):
        assert set(spec["properties"]) == set(TaskDraft.model_fields)
        assert set(spec["required"]) == {"task", "category", "citations"}
        assert spec["properties"]["task"]["minLength"] == 3
        assert spec["properties"]["task"]["maxLength"] == 300




def test_new_task_submissions_reject_retired_fields():
    for field, value in {'title': 'Old name', 'objective': 'Old goal', 'confidence': 0.9,
                         'caveats': ['Old note'], 'operation_mode': 'gui'}.items():
        with pytest.raises(ValidationError, match='Extra inputs'):
            TaskDraft.model_validate(payload(**{field: value}))


@pytest.mark.parametrize("length", [2, 301])
def test_task_text_length_limits(length):
    with pytest.raises(ValidationError):
        TaskDraft.model_validate(payload(task="x" * length))


def test_saved_task_migration_is_read_only_and_removes_old_fields():
    raw = {"title": "Render a graph", "objective": "Render the current graph.", "category": "Operations",
           "confidence": 0.9, "caveats": ["Legacy note"], "operation_mode": "gui", "task_id": "T1", "corpus_snapshot_id": "CP1",
           "citations": [{"evidence_id": "E1", "quote": "Render the graph.", "rationale": "Legacy support"}]}
    before = copy.deepcopy(raw)
    saved = TaskCandidate.model_validate(raw).model_dump(mode="json")
    assert saved["task"] == raw["title"]
    assert not {"title", "objective", "confidence", "caveats", "operation_mode"} & saved.keys()
    assert "rationale" not in saved["citations"][0]
    assert raw == before
    assert normalize_saved_task_payload(raw)["task"] == raw["title"]


@pytest.mark.parametrize("category", ["Manual", "rendering", " Operations", "operations", "invented"])
def test_category_rejects_root_invented_or_modified_names(category):
    ev = evidence()
    accepted, report = validate_tasks([candidate(category=category)], {"E1": ev}, [first_level_tree(ev)])
    assert not accepted
    assert "exactly match" in report[0].errors[0]


def test_category_requires_cited_evidence_from_selected_chapter_and_respects_offsets():
    first, second = evidence(), evidence("E2", start=20, end=40)
    tree = first_level_tree(first)
    other = first_level_tree(second, "Saving").nodes[1].model_copy(update={
        "evidence_node_id": "N3", "node_key": "saving", "ordinal": 2,
    })
    tree.nodes[0].child_node_ids.append("N3")
    tree.nodes.append(other)
    by_id = {"E1": first, "E2": second}
    accepted, _ = validate_tasks([candidate()], by_id, [tree])
    assert len(accepted) == 1
    accepted, report = validate_tasks([candidate(category="Saving")], by_id, [tree])
    assert not accepted
    assert "supporting cited Evidence" in report[0].errors[0]


def test_task_category_rejects_a_source_without_a_tree():
    accepted, report = validate_tasks([candidate()], {"E1": evidence()}, [])
    assert not accepted
    assert "first-level" in report[0].errors[0]


def test_only_first_level_title_is_allowed_when_evidence_is_in_a_deeper_node():
    ev = evidence()
    tree = first_level_tree(ev)
    leaf = tree.nodes[1].model_copy(update={
        "evidence_node_id": "N3", "node_key": "deep", "parent_node_id": "N2", "title": "Deep heading",
    })
    tree.nodes[1] = tree.nodes[1].model_copy(update={
        "source_refs": [], "content": "", "content_artifact_id": None,
        "content_sha256": None, "child_node_ids": ["N3"],
    })
    tree.nodes.append(leaf)
    accepted, _ = validate_tasks([candidate()], {"E1": ev}, [tree])
    assert len(accepted) == 1
    accepted, report = validate_tasks([candidate(category="Deep heading")], {"E1": ev}, [tree])
    assert not accepted
    assert "exactly match" in report[0].errors[0]


def test_a_flat_tree_root_cannot_be_used_as_a_category():
    ev = evidence()
    tree = first_level_tree(ev)
    root = tree.nodes[1].model_copy(update={"parent_node_id": None})
    tree = tree.model_copy(update={"root_node_id": root.evidence_node_id, "nodes": [root]})
    accepted, _ = validate_tasks([candidate()], {"E1": ev}, [tree])
    assert not accepted


def test_runtime_tool_enum_and_directory_keep_full_chapter_names(tmp_path):
    ev = evidence()
    name = "Chapter " + "x" * 180
    tree = first_level_tree(ev, name)
    target = TargetProfile(software_name="Example", target_version="1", platform="General")
    corpus = CorpusSnapshot(corpus_snapshot_id="CP1", run_id="run_task_contract", target_profile=target,
                            source_results=[], evidence=[ev], evidence_trees=[tree], evidence_count=1,
                            source_part_count=1, source_status_counts={})
    store = RunStore(tmp_path, corpus.run_id)
    store.initialize({"target_profile": target.model_dump(mode="json")})
    workspace = KnowledgeWorkspace(store=store, settings=Settings(data_dir=tmp_path), target_profile=target,
                                   stage="task_discovery", corpus=corpus, corpus_artifact_id="art_corpus")
    outline = workspace.inspect_task_outline()
    assert outline["task_categories"] == [name]
    assert outline["items"][0]["root"]["children"][0]["title"] == name
    assert "operation_path_policy" not in workspace.context()
    assert "operation_path_preference" not in workspace.context()["target_profile"]
    spec = next(s for s in KnowledgeToolExecutor(workspace, "task_discovery_agent").tool_specs()
                if s["function"]["name"] == "submit_task_batch")
    assert spec["function"]["parameters"]["properties"]["tasks"]["items"]["properties"]["category"]["enum"] == [name]
    result = workspace.submit_task_batch([payload(category=name)])
    assert result["accepted_by_schema"] == 1, result
    saved = workspace.submitted_tasks[0].model_dump(mode="json")
    assert saved["category"] == name
    assert not {"title", "objective", "confidence", "caveats"} & saved.keys()


def test_saved_profiles_ignore_retired_path_fields_without_modifying_input():
    raw = {"software_name": "Example", "target_version": "1.0", "platform": "Linux",
           "operation_path_preference": "gui", "operation_modes": ["GUI", "Python"]}
    before = copy.deepcopy(raw)
    for model in (TargetProfile, TargetProfileSnapshot):
        projected = model.model_validate(raw).model_dump(mode="json")
        assert set(projected) == {"software_name", "software_id", "target_version", "platform", "ui_locale"}
    assert raw == before
    with pytest.raises(ValidationError, match="Extra inputs"):
        TargetProfileSnapshot.model_validate({**raw, "unexpected": "Still forbidden"})
