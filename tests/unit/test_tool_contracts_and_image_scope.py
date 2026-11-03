from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.capabilities import CapabilityGateway, CapabilityGatewayError
from app.knowledge import KnowledgeToolExecutor, KnowledgeWorkspace
from app.knowledge_components.tool_catalog import _knowledge_tool_catalog
from app.models import EvidenceImageRef, SkillCandidate, SkillDraft
from app.orchestration.unified_stage import UNIFIED_PROMPT
from app.orchestration.downstream_stage import SKILL_PROMPT
from app.runtime.dsh_events import RuntimeEventStream
from test_t4_capability_gateway import FakeExecutor, _invocation, _store


@pytest.mark.parametrize("stage", ["task_trajectory", "skill_induction"])
def test_snapshot_images_are_authorized_but_unrelated_images_are_not(tmp_path, stage):
    store, _ = _store(tmp_path, "run_images")
    def image(data, source="SRC1"):
        return store.put_artifact(data, kind="pdf_embedded_image", media_type="image/png",
                                  suffix=".png", source_id=source)
    owned = image(b"owned image")
    unrelated = image(b"unrelated image")
    foreign = image(b"foreign image", "SRC2")
    snapshot = store.put_json_artifact({"evidence": [{
        "evidence_id": "E1", "source_id": "SRC1",
        "image_refs": [{"artifact_id": owned.artifact_id}, {"artifact_id": foreign.artifact_id}],
    }]}, kind="corpus_snapshot")
    tools = ("view_evidence_image",)
    attempt, capsule, _ = store.begin_child_attempt(
        stage=stage, input_artifact_ids=(snapshot.artifact_id,), scope={"trajectory_ids": ["TR1"]},
        allowed_tools=tools, output_contract="test-output.v1", prompt_version="test.v1",
    )
    executor = FakeExecutor(set(tools))
    gateway = CapabilityGateway()
    registered = gateway.register_attempt(store, attempt.attempt_id, executor)
    assert owned.artifact_id in registered.record.resource_scope["artifact_ids"]
    result = gateway.invoke(registered.token, _invocation(capsule, "view_evidence_image", {
        "evidence_id": "E1", "artifact_id": owned.artifact_id,
    }, "owned"))
    assert result["ok"]
    for other in [unrelated, foreign]:
        with pytest.raises(CapabilityGatewayError, match="outside this Child Attempt"):
            gateway.invoke(registered.token, _invocation(capsule, "view_evidence_image", {
                "evidence_id": "E1", "artifact_id": other.artifact_id,
            }, other.artifact_id))
    assert len(executor.calls) == 1


def test_authorized_image_still_has_to_belong_to_the_named_evidence(tmp_path):
    store, _ = _store(tmp_path, "run_attachment")
    image = store.put_artifact(b"attached", kind="pdf_embedded_image", media_type="image/png",
                               suffix=".png", source_id="SRC1")
    ref = EvidenceImageRef(artifact_id=image.artifact_id, source_part_id="SP1", media_type="image/png", page=1)
    workspace = SimpleNamespace(evidence_by_id={"E1": SimpleNamespace(image_refs=[ref]),
        "E2": SimpleNamespace(image_refs=[])}, store=store,
        settings=SimpleNamespace(max_vision_image_bytes=1000), inspected_images=set())
    with pytest.raises(ValueError, match="not attached"):
        KnowledgeWorkspace.view_evidence_image(workspace, "E2", image.artifact_id)


def _skill():
    return {"title": "Inspect a graph", "description": "Inspect the documented graph.",
        "applicability": "A documented graph is available.", "preconditions": "",
        "trajectory_ids": ["TR1"], "procedure": [{"ordinal": 1,
        "instruction": "Inspect the graph.",
        "citations": [{"evidence_id": "E1", "quote": "Inspect the graph."}]}]}


@pytest.mark.parametrize("location,bad_value", [
    ("quote", 123), ("quote", None), ("quote", ["source text"]),
    ("citation", "E1"), ("citation", ["E1", "source text"]),
    ("step", "Inspect the graph."), ("step", ["Inspect the graph."]),
])
def test_malformed_skill_fields_are_rejected_and_corrected_without_coercion(location, bad_value):
    payload = _skill()
    if location == "step":
        payload["procedure"][0] = bad_value
    elif location == "citation":
        payload["procedure"][0]["citations"][0] = bad_value
    else:
        payload["procedure"][0]["citations"][0]["quote"] = bad_value
    before = deepcopy(payload)
    with pytest.raises(ValidationError):
        SkillDraft.model_validate(payload)
    assert payload == before
    corrected = _skill()
    assert SkillDraft.model_validate(corrected).procedure[0].citations[0].quote == "Inspect the graph."


@pytest.mark.parametrize("description,accepted", [
    ("d" * 99 + ".", True), ("d" * 100 + ".", False),
    ("\u4e2d" * 100, True), ("\u4e2d" * 101, False),
    ("d" * 99 + " ", True), ("d" * 100 + " ", False),
])
def test_new_skill_description_counts_characters_including_spaces(description, accepted):
    payload = _skill()
    payload["description"] = description
    if accepted:
        assert SkillDraft.model_validate(payload).description == description
    else:
        with pytest.raises(ValidationError, match="at most 100 characters"):
            SkillDraft.model_validate(payload)


def test_saved_skill_description_remains_readable_and_new_tool_contract_is_short():
    payload = {**_skill(), "description": "Historical description. " * 20,
               "skill_id": "SK1", "knowledge_snapshot_id": "KS1"}
    saved = SkillCandidate.model_validate(payload)
    assert saved.description == payload["description"]
    draft = {k: v for k, v in payload.items() if k not in {"skill_id", "knowledge_snapshot_id"}}
    with pytest.raises(ValidationError, match="at most 100 characters"):
        SkillDraft.model_validate(draft)
    spec = _knowledge_tool_catalog()["submit_skill_batch"]["function"]
    field = spec["parameters"]["properties"]["skills"]["items"]["properties"]["description"]
    assert field["maxLength"] == 100
    assert "100 characters" in SKILL_PROMPT
    assert "call submit_stage_result with status only" in SKILL_PROMPT


def test_reading_contract_is_shared_and_pagination_remains_stage_specific():
    catalog = _knowledge_tool_catalog()
    assert "Reading IDs and argument units:" not in UNIFIED_PROMPT
    contracts = []
    for stage in ["task_trajectory", "skill_induction"]:
        workspace = MagicMock(spec=KnowledgeWorkspace)
        workspace.stage = stage
        workspace.store = None
        workspace.settings = None
        workspace.evidence_by_id = {}
        workspace.corpus = SimpleNamespace(evidence_trees=[], evidence=[])
        specs = {s["function"]["name"]: s["function"]
                 for s in KnowledgeToolExecutor(workspace, "test").tool_specs()}
        node = specs["read_task_node"]
        contracts.append(node)
        assert node["parameters"]["required"] == ["node_id"]
        assert "supplied evidence_tree" in node["description"]
        assert '{"node_id":"N1"}' in node["description"]
        if "read_evidence" in specs and stage == "task_trajectory":
            assert "offset" not in specs["read_evidence"]["parameters"]["properties"]
    assert contracts[0] == contracts[1]
    assert "offset" in catalog["read_evidence"]["function"]["parameters"]["properties"]
    citation = catalog["submit_skill_batch"]["function"]["parameters"]["properties"]["skills"]["items"]["properties"]["procedure"]["items"]["properties"]["citations"]["items"]
    assert citation["properties"]["quote"]["type"] == "string"


def test_business_tool_validation_errors_and_recovery_are_audited_without_duplicate_success():
    frames = []
    stream = RuntimeEventStream({"session_id": "child", "stream_events": True,
        "api_key": "secret-to-redact"}, frames.append)
    def send(kind, data):
        stream.notification(SimpleNamespace(method="session.event", payload={
            "sessionId": "child", "event": {"type": kind, "data": data}}))
    def result(cid, failed, text):
        send("tool/result", {"message": {"source": {"callId": cid}, "content": [{
            "type": "tool-result", "isError": failed, "content": [{"type": "text", "text": text}]}]},
            **({"error": {"code": "INVALID_ARGS"}} if failed else {})})
    send("tool/call", {"callId": "bad", "name": "read_task_node", "arguments": "{}"})
    result("bad", True, 'missing node_id secret-to-redact')
    send("tool/call", {"callId": "good", "name": "read_task_node", "arguments": '{"node_id":"N1"}'})
    result("good", False, '{"ok":true}')
    send("tool/call", {"callId": "next", "name": "read_task_node", "arguments": '{"node_id":"N2"}'})
    result("next", False, '{"ok":true}')
    assert [f["event"] for f in frames] == ["native_tool_error", "native_tool_recovered"]
    assert frames[0]["payload"]["ok"] is False
    assert frames[0]["payload"]["error_category"] == "validation"
    assert "secret-to-redact" not in str(frames)
    assert frames[1]["payload"]["previous_error_call_id"] == "bad"
