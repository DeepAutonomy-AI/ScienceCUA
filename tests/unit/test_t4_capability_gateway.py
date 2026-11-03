from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.capabilities import (
    CapabilityGateway,
    CapabilityGatewayError,
    CapabilityInvocation,
    tools_for_stage,
)
from app.capabilities.http import create_capability_router
from app.capabilities.authorization import CapabilityAuthorizationError
from app.knowledge import KnowledgeToolExecutor
from app.models import EvidenceCitation
from app.storage import RunStore
from app.toolkit import ToolExecutor


@dataclass
class FakeExecutor:
    allowed_tools: set[str]
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((name, arguments or {}))
        return self.results.get(name, {"ok": True, "tool": name, "arguments": arguments or {}})

    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Test schema for {name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in sorted(self.allowed_tools)
        ]


def _request(artifact_id: str) -> dict[str, Any]:
    return {
        "target_profile": {
            "software_name": "Example",
            "target_version": "1.0",
            "platform": "Linux",
            "ui_locale": "English",
        },
        "inputs": [{"artifact_id": artifact_id}],
        "model": {
            "provider": "openai-compatible",
            "model": "gpt-test",
            "api_protocol": "responses",
            "reasoning_effort": "medium",
            "base_url": "https://example.invalid/v1",
            "api_key": "t4-model-key-must-not-persist",
        },
        "prompt_set_version": "test-prompts.v1",
    }


def _store(tmp_path, run_id: str) -> tuple[RunStore, str]:
    store = RunStore(tmp_path, run_id)
    artifact = store.put_json_artifact({"input": run_id}, kind="test_input")
    store.initialize(_request(artifact.artifact_id))
    return store, artifact.artifact_id


def _registered(
    tmp_path,
    *,
    run_id: str = "run_t4",
    stage: str = "trajectory_inference",
    tools: tuple[str, ...] = ("read_evidence", "submit_trajectory"),
    scope: dict[str, Any] | None = None,
    ttl_seconds: float | None = None,
    keep_alive: bool = False,
    max_tool_calls: int | None = None,
    tool_call_limits: dict[str, int] | None = None,
    budget_exhaustion_as_skip: bool = False,
    gateway: CapabilityGateway | None = None,
    executor: FakeExecutor | None = None,
):
    store, artifact_id = _store(tmp_path, run_id)
    attempt, capsule, _ = store.begin_child_attempt(
        stage=stage,
        input_artifact_ids=(artifact_id,),
        scope=scope or {"task_id": "task_allowed"},
        allowed_tools=tools,
        output_contract="test-output.v1",
        prompt_version="test-prompt.v1",
    )
    actual_executor = executor or FakeExecutor(set(tools))
    actual_gateway = gateway or CapabilityGateway()
    registered = actual_gateway.register_attempt(
        store,
        attempt.attempt_id,
        actual_executor,
        max_tool_calls=max_tool_calls,
        tool_call_limits=tool_call_limits,
        budget_exhaustion_as_skip=budget_exhaustion_as_skip,
        ttl_seconds=ttl_seconds,
        keep_alive=keep_alive,
    )
    return actual_gateway, store, attempt, capsule, actual_executor, registered


def _invocation(capsule, tool: str, arguments=None, call_id: str = "call_1"):
    return CapabilityInvocation(
        run_id=capsule.run_id,
        root_session_id=capsule.parent_root_session_id,
        child_session_id=capsule.child_session_id,
        attempt_id=capsule.attempt_id,
        stage=capsule.stage,
        tool_name=tool,
        arguments=arguments or {},
        call_id=call_id,
    )


def test_stage_policy_matches_legacy_executors_and_root_has_no_tools():
    assert "pdf_inventory" not in tools_for_stage("source_review", modality="pdf")
    assert "html_inventory" not in tools_for_stage("source_review", modality="pdf")
    assert "html_inventory" not in tools_for_stage("source_review", modality="html")
    assert "curate_evidence" in tools_for_stage("source_review", modality="html")
    assert "submit_task_batch" in tools_for_stage("task_discovery")
    assert "submit_task_batch" not in tools_for_stage("trajectory_inference")
    assert "submit_trajectory" in tools_for_stage("trajectory_inference")
    assert "submit_skill_batch" in tools_for_stage("skill_induction")
    assert "inspect_knowledge_scope" not in tools_for_stage("skill_induction")
    assert "inspect_skill_coverage" not in tools_for_stage("skill_induction")
    assert "review_skill_coverage" in tools_for_stage("skill_induction")
    assert tools_for_stage("root") == frozenset()


def test_policy_covers_both_existing_python_executors():
    for modality in ("pdf", "html"):
        architect_tools = (
            ToolExecutor.ARCHITECT_PDF_REVIEW_TOOLS
            if modality == "pdf"
            else ToolExecutor.ARCHITECT_REVIEW_TOOLS
        )
        expected = (
            architect_tools
            | ToolExecutor.FULL_PLAN_REVIEW_TOOLS
            | ToolExecutor.LEGACY_REVIEW_COMMON_TOOLS
            | ToolExecutor.CURATED_REVIEW_TOOLS
            | ToolExecutor.LEGACY_REVIEW_MODAL_TOOLS[modality]
        )
        assert tools_for_stage("source_review", modality=modality) == frozenset(expected)
    for stage in KnowledgeToolExecutor.STAGE_TOOLS:
        workspace = MagicMock()
        workspace.stage = stage
        executor = KnowledgeToolExecutor(workspace, "test_agent")
        assert executor.allowed_tools <= tools_for_stage(stage)
    skill_workspace = MagicMock()
    skill_workspace.stage = "skill_induction"
    skill_executor = KnowledgeToolExecutor(skill_workspace, "test_agent")
    assert skill_executor.allowed_tools <= tools_for_stage("skill_induction")
    assert "read_task_node" in skill_executor.allowed_tools


def test_stage_cannot_grant_another_stage_tool(tmp_path):
    store, artifact_id = _store(tmp_path, "run_wrong_tool")
    attempt, _, _ = store.begin_child_attempt(
        stage="task_discovery",
        input_artifact_ids=(artifact_id,),
        scope={},
        allowed_tools=("submit_skill_batch",),
        output_contract="test-output.v1",
        prompt_version="test-prompt.v1",
    )
    executor = FakeExecutor({"submit_skill_batch"})

    with pytest.raises(Exception, match="stage policy"):
        CapabilityGateway().register_attempt(store, attempt.attempt_id, executor)


def test_tool_whitelist_is_enforced_again_at_gateway(tmp_path):
    gateway, _, _, capsule, executor, registered = _registered(tmp_path)
    denied = _invocation(capsule, "submit_skill_batch")

    with pytest.raises(CapabilityGatewayError) as caught:
        gateway.invoke(registered.token, denied)

    assert caught.value.code == "tool_not_allowed"
    assert executor.calls == []


def test_gateway_enforces_call_budget_but_keeps_completion_tool_available(tmp_path):
    gateway, _, _, capsule, executor, registered = _registered(
        tmp_path,
        max_tool_calls=1,
    )

    first = gateway.invoke(
        registered.token,
        _invocation(capsule, "read_evidence", call_id="call_1"),
    )
    exhausted = gateway.invoke(
        registered.token,
        _invocation(capsule, "read_evidence", call_id="call_2"),
    )
    submitted = gateway.invoke(
        registered.token,
        _invocation(capsule, "submit_trajectory", call_id="call_3"),
    )

    assert first["ok"] is True
    assert exhausted["ok"] is False
    assert exhausted["error"] == "tool_call_budget_exhausted"
    assert submitted["ok"] is True
    assert [name for name, _ in executor.calls] == [
        "read_evidence",
        "submit_trajectory",
    ]


def test_skill_tree_node_result_is_returned_without_gateway_truncation(tmp_path):
    content = "A" * 20_000
    executor = FakeExecutor(
        {"read_task_node"},
        results={"read_task_node": {"ok": True, "segments": [{"content": content}]}},
    )
    gateway, _, _, capsule, _, registered = _registered(
        tmp_path, stage="skill_induction", tools=("read_task_node",),
        gateway=CapabilityGateway(), executor=executor,
    )
    result = gateway.invoke(
        registered.token,
        _invocation(capsule, "read_task_node", {"node_id": "N2"}),
    )
    assert result["segments"][0]["content"] == content
    assert "truncated" not in result


def test_gateway_can_close_one_inspection_tool_as_skipped_and_keep_submission_open(
    tmp_path,
):
    gateway, _, _, capsule, executor, registered = _registered(
        tmp_path,
        max_tool_calls=10,
        tool_call_limits={"read_evidence": 1},
        budget_exhaustion_as_skip=True,
    )

    first = gateway.invoke(
        registered.token,
        _invocation(capsule, "read_evidence", call_id="read_once"),
    )
    skipped = gateway.invoke(
        registered.token,
        _invocation(capsule, "read_evidence", call_id="read_again"),
    )
    submitted = gateway.invoke(
        registered.token,
        _invocation(capsule, "submit_trajectory", call_id="submit_after_skip"),
    )

    assert first["ok"] is True
    assert skipped["ok"] is True
    assert skipped["status"] == "skipped"
    assert skipped["reason"] == "tool_call_budget_exhausted"
    assert skipped["mandatory_next_action"] == "submit_candidates_then_submit_stage_result"
    assert submitted["ok"] is True
    assert [name for name, _ in executor.calls] == [
        "read_evidence",
        "submit_trajectory",
    ]


def test_atomic_reference_review_remains_available_after_inspection_budget(tmp_path):
    tools = (
        "read_evidence",
        "submit_reference_review",
    )
    gateway, _, _, capsule, executor, registered = _registered(
        tmp_path,
        stage="reference_build",
        tools=tools,
        max_tool_calls=1,
    )

    first = gateway.invoke(
        registered.token,
        _invocation(capsule, "read_evidence", call_id="reference_read"),
    )
    submitted = gateway.invoke(
        registered.token,
        _invocation(capsule, "submit_reference_review", call_id="reference_complete"),
    )
    exhausted = gateway.invoke(
        registered.token,
        _invocation(capsule, "read_evidence", call_id="late_read"),
    )

    assert first["ok"] is True
    assert submitted["ok"] is True
    assert exhausted["error"] == "tool_call_budget_exhausted"
    assert [name for name, _ in executor.calls] == [
        "read_evidence",
        "submit_reference_review",
    ]


@pytest.mark.parametrize(
    ("field", "changed", "code"),
    [
        ("run_id", "run_foreign", "binding_mismatch"),
        ("stage", "skill_induction", "binding_mismatch"),
        ("attempt_id", "attempt_foreign", "binding_mismatch"),
        ("child_session_id", "child_foreign", "binding_mismatch"),
    ],
)
def test_cross_run_stage_attempt_and_child_are_denied(tmp_path, field, changed, code):
    gateway, _, _, capsule, executor, registered = _registered(tmp_path)
    data = _invocation(capsule, "read_evidence").model_dump()
    data[field] = changed

    with pytest.raises(CapabilityGatewayError) as caught:
        gateway.invoke(registered.token, CapabilityInvocation.model_validate(data))

    assert caught.value.code == code
    assert executor.calls == []


def test_cross_task_read_and_nested_submission_are_denied(tmp_path):
    gateway, _, _, capsule, executor, registered = _registered(tmp_path)

    for index, arguments in enumerate(
        (
            {"task_id": "task_foreign"},
            {"trajectory": {"task_id": "task_foreign", "steps": []}},
        ),
        start=1,
    ):
        with pytest.raises(CapabilityGatewayError) as caught:
            gateway.invoke(
                registered.token,
                _invocation(capsule, "submit_trajectory", arguments, f"call_task_{index}"),
            )
        assert caught.value.code == "task_scope_denied"
    assert executor.calls == []


def test_foreign_artifact_and_filesystem_path_are_denied(tmp_path):
    gateway, _, _, capsule, executor, registered = _registered(tmp_path)
    other, other_artifact = _store(tmp_path, "run_other")
    assert other.artifact_record(other_artifact)

    with pytest.raises(CapabilityGatewayError) as artifact_error:
        gateway.invoke(
            registered.token,
            _invocation(capsule, "read_evidence", {"artifact_id": other_artifact}, "foreign_art"),
        )
    with pytest.raises(CapabilityGatewayError) as path_error:
        gateway.invoke(
            registered.token,
            _invocation(capsule, "read_evidence", {"file_path": "C:/secrets.txt"}, "bad_path"),
        )

    assert artifact_error.value.code == "artifact_not_found"
    assert path_error.value.code == "path_argument_denied"
    assert executor.calls == []


def test_empty_optional_image_id_does_not_block_text_citation(tmp_path):
    gateway, _, _, capsule, executor, registered = _registered(tmp_path)
    citation = {"evidence_id": "E1", "quote": "Quoted source text",
                "image_artifact_id": ""}
    result = gateway.invoke(
        registered.token,
        _invocation(capsule, "submit_trajectory", {"citation": citation},
                    "empty_optional_image"),
    )
    assert result["ok"] is True
    assert EvidenceCitation.model_validate(citation).image_artifact_id is None
    with pytest.raises(ValueError, match="nonempty quote text or image_artifact_id"):
        EvidenceCitation.model_validate({"evidence_id": "E1", "image_artifact_id": ""})
    assert executor.calls == [("submit_trajectory", {"citation": citation})]


def test_expired_revoked_and_terminal_attempt_tokens_are_denied(tmp_path):
    gateway, store, attempt, capsule, executor, registered = _registered(
        tmp_path, ttl_seconds=0.001
    )
    time.sleep(0.01)
    with pytest.raises(CapabilityGatewayError) as expired:
        gateway.invoke(registered.token, _invocation(capsule, "read_evidence"))
    assert expired.value.code == "expired_token"

    gateway2, store2, attempt2, capsule2, _, registered2 = _registered(
        tmp_path, run_id="run_revoked"
    )
    gateway2.revoke(registered2, reason="test")
    with pytest.raises(CapabilityGatewayError) as revoked:
        gateway2.invoke(registered2.token, _invocation(capsule2, "read_evidence"))
    assert revoked.value.code == "revoked_token"

    gateway3, store3, attempt3, capsule3, _, registered3 = _registered(
        tmp_path, run_id="run_terminal"
    )
    store3.finish_attempt(attempt3.attempt_id, status="cancelled")
    with pytest.raises(CapabilityGatewayError) as terminal:
        gateway3.invoke(registered3.token, _invocation(capsule3, "read_evidence"))
    assert terminal.value.code == "attempt_terminal"


@pytest.mark.parametrize("ended", ["expired", "revoked", "terminal"])
def test_renewal_never_revives_ended_capability(tmp_path, monkeypatch, ended):
    clock = [10_000.0]
    monkeypatch.setattr("app.capabilities.authorization.time.time", lambda: clock[0])
    gateway, store, attempt, capsule, _, registered = _registered(tmp_path)
    if ended == "expired":
        clock[0] += 900
        code = "expired_token"
    elif ended == "revoked":
        gateway.revoke(registered, reason="cancelled")
        code = "revoked_token"
    else:
        store.finish_attempt(attempt.attempt_id, status="cancelled")
        code = "attempt_terminal"
    with pytest.raises(CapabilityAuthorizationError) as renewal:
        gateway.authority.renew(registered.record.grant_id)
    assert renewal.value.code == code
    with pytest.raises(CapabilityGatewayError) as invocation:
        gateway.invoke(registered.token, _invocation(capsule, "read_evidence"))
    assert invocation.value.code == code


@pytest.mark.asyncio
async def test_live_child_lease_covers_queue_execution_and_retries(tmp_path, monkeypatch):
    """Advance a fake clock beyond an hour without models or wall-clock waits."""
    clock = [10_000.0]
    monkeypatch.setattr("app.capabilities.authorization.time.time", lambda: clock[0])
    sleeps = asyncio.Queue()

    async def controlled_sleep(delay):
        wake = asyncio.get_running_loop().create_future()
        await sleeps.put((delay, wake))
        await wake

    monkeypatch.setattr("app.capabilities.gateway.asyncio.sleep", controlled_sleep)
    gateway, store, attempt, capsule, _, registered = _registered(
        tmp_path, keep_alive=True
    )
    lease = gateway._lease_tasks[registered.record.grant_id]
    initial_expiry = registered.record.expires_at_epoch
    pending_sleep = await sleeps.get()
    try:
        # The lease remains active while waiting for the runtime semaphore, during
        # execution, protocol reminders, and upstream retry backoff alike.
        for phase in ("queued", "running", "reminder", "retry"):
            for step in range(4):
                delay, wake = pending_sleep
                assert delay == 60
                clock[0] += 300
                wake.set_result(None)
                pending_sleep = await sleeps.get()  # Renewal completed.
                assert gateway.invoke(
                    registered.token,
                    _invocation(capsule, "read_evidence", call_id=f"{phase}_{step}"),
                )["ok"]
        assert clock[0] > initial_expiry
        renewed = gateway.authority.renew(registered.record.grant_id)
        assert renewed.expires_at_epoch == clock[0] + 900
        assert renewed.allowed_tools == registered.record.allowed_tools
        assert renewed.resource_scope == registered.record.resource_scope
        assert renewed.token_sha256 == registered.record.token_sha256
        assert renewed.child_session_id == capsule.child_session_id
    finally:
        gateway.revoke(registered, reason="attempt_finished")
        with pytest.raises(asyncio.CancelledError):
            await lease
    assert not gateway._lease_tasks
    assert not gateway._bindings
    with pytest.raises(CapabilityGatewayError) as revoked:
        gateway.invoke(registered.token, _invocation(capsule, "read_evidence", call_id="after"))
    assert revoked.value.code == "revoked_token"


@pytest.mark.asyncio
async def test_lease_stops_when_attempt_becomes_terminal(tmp_path, monkeypatch):
    sleeps = asyncio.Queue()

    async def controlled_sleep(delay):
        wake = asyncio.get_running_loop().create_future()
        await sleeps.put(wake)
        await wake

    monkeypatch.setattr("app.capabilities.gateway.asyncio.sleep", controlled_sleep)
    gateway, store, attempt, capsule, _, registered = _registered(tmp_path, keep_alive=True)
    lease = gateway._lease_tasks[registered.record.grant_id]
    wake = await sleeps.get()
    store.finish_attempt(attempt.attempt_id, status="cancelled")
    wake.set_result(None)
    await lease
    assert not gateway._lease_tasks
    with pytest.raises(CapabilityGatewayError) as terminal:
        gateway.invoke(registered.token, _invocation(capsule, "read_evidence"))
    assert terminal.value.code == "attempt_terminal"
    gateway.revoke(registered, reason="cancelled")


@pytest.mark.parametrize("stage,tool,recorder_name", [
    ("source_review", "get_root_structure", "source"),
    ("reference_build", "inspect_reference_catalog", "knowledge"),
])
def test_large_tool_result_reaches_child_in_full_and_keeps_audit_artifact(tmp_path, stage, tool, recorder_name):
    from types import SimpleNamespace
    from app.config import Settings

    # Cross both the former 16k global cutoff and the old 100k root special case.
    payload = {"ok": True, "nodes": [{"title": 'quoted "text"' * 12_000}],
               "root_structure": {"headings": ["Introduction", "Commands"]}}
    executor = FakeExecutor({tool})
    gateway, store, _, capsule, _, registered = _registered(
        tmp_path, stage=stage, tools=(tool,), executor=executor,
        scope={"modality": "html"} if stage == "source_review" else None,
    )
    recorder_class = ToolExecutor if recorder_name == "source" else KnowledgeToolExecutor
    recorder = object.__new__(recorder_class)
    recorder.store = store
    recorder.settings = Settings()
    recorder.agent_name = "observation_test"
    recorder.workspace = SimpleNamespace(stage=stage, allowed_artifacts=set(),
        manifest=SimpleNamespace(source_id="SRC1"))
    executor.results[tool] = recorder._record_observation(tool, payload)
    result = gateway.invoke(registered.token, _invocation(capsule, tool))
    assert result == payload
    assert "preview" not in result and "truncated" not in result
    audit = [json.loads(line) for line in store.audit_path.read_text(encoding="utf-8").splitlines()]
    recorded = next(row["payload"] for row in audit
                    if row["event_type"] in {"tool_observation", "knowledge_tool_observation"})
    artifact = store.artifact_record(recorded["observation_artifact_id"])
    saved = json.loads(store.artifact_path(artifact.artifact_id).read_text(encoding="utf-8"))
    assert saved == payload
    delivered = next(row["payload"] for row in audit if row["event_type"] == "capability_tool_result")
    assert delivered["observation_chars"] == delivered["returned_chars"]


def test_long_task_node_read_reaches_child_without_gateway_truncation(tmp_path):
    body = "z" * 40_000
    executor = FakeExecutor(
        {"read_task_node"}, results={"read_task_node": {"ok": True, "segments": [{"evidence_id": "E1", "content": body}]}}
    )
    gateway = CapabilityGateway()
    gateway, _, _, capsule, _, registered = _registered(
        tmp_path,
        stage="task_trajectory",
        tools=("read_task_node",),
        gateway=gateway,
        executor=executor,
    )

    result = gateway.invoke(registered.token, _invocation(capsule, "read_task_node", {"node_id": "N1"}))

    assert result["ok"] is True
    assert result["segments"][0]["content"] == body
    assert result.get("truncated") is not True


def test_complete_task_outline_reaches_child_without_gateway_truncation(tmp_path):
    nodes = [{"node_id": f"N{i}", "title": "Capability " + "x" * 120, "children": []}
             for i in range(1, 200)]
    executor = FakeExecutor(
        {"inspect_task_outline"},
        results={"inspect_task_outline": {"ok": True, "items": [{"root": {"node_id": "N0", "children": nodes}}]}},
    )
    gateway = CapabilityGateway()
    gateway, _, _, capsule, _, registered = _registered(
        tmp_path,
        stage="task_trajectory",
        tools=("inspect_task_outline",),
        gateway=gateway,
        executor=executor,
    )

    result = gateway.invoke(registered.token, _invocation(capsule, "inspect_task_outline"))

    assert result["ok"] is True
    assert len(result["items"][0]["root"]["children"]) == 199
    assert result.get("truncated") is not True


def test_oversized_arguments_are_rejected_before_executor(tmp_path):
    gateway = CapabilityGateway(max_argument_bytes=1_024)
    gateway, _, _, capsule, executor, registered = _registered(tmp_path, gateway=gateway)

    with pytest.raises(CapabilityGatewayError) as caught:
        gateway.invoke(
            registered.token,
            _invocation(capsule, "read_evidence", {"query": "x" * 2_000}),
        )

    assert caught.value.code == "arguments_too_large"
    assert caught.value.status_code == 413
    assert executor.calls == []


def test_call_id_is_idempotent_but_conflicting_reuse_is_denied(tmp_path):
    gateway, _, _, capsule, executor, registered = _registered(tmp_path)
    first = gateway.invoke(
        registered.token, _invocation(capsule, "read_evidence", {"task_id": "task_allowed"})
    )
    replay = gateway.invoke(
        registered.token, _invocation(capsule, "read_evidence", {"task_id": "task_allowed"})
    )
    assert replay == first
    assert len(executor.calls) == 1

    with pytest.raises(CapabilityGatewayError) as conflict:
        gateway.invoke(
            registered.token,
            _invocation(capsule, "read_evidence", {"task_id": "task_allowed", "page": 2}),
        )
    assert conflict.value.code == "call_id_conflict"


def test_http_proxy_contract_uses_bearer_header(tmp_path):
    gateway, _, _, capsule, _, registered = _registered(tmp_path)
    app = FastAPI()
    app.include_router(create_capability_router(gateway), prefix="/internal/capabilities")
    client = TestClient(app)
    body = _invocation(capsule, "read_evidence", {"task_id": "task_allowed"}).model_dump()

    missing = client.post("/internal/capabilities/invoke", json=body)
    accepted = client.post(
        "/internal/capabilities/invoke",
        json={**body, "call_id": "http_call"},
        headers={"Authorization": f"Bearer {registered.token}"},
    )

    assert missing.status_code == 403
    assert missing.json()["error"]["code"] == "missing_token"
    assert accepted.status_code == 200
    assert accepted.json()["result"]["ok"] is True


def test_credentials_never_enter_artifacts_sessions_or_audit(tmp_path):
    secret = "secret-value-that-must-never-persist"
    gateway, store, _, capsule, executor, registered = _registered(tmp_path)

    with pytest.raises(CapabilityGatewayError) as caught:
        gateway.invoke(
            registered.token,
            _invocation(capsule, "read_evidence", {"authorization": f"Bearer {secret}"}),
        )
    assert caught.value.code == "credential_argument_denied"
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in store.root.rglob("*")
        if path.is_file()
    )
    assert secret not in persisted
    assert "t4-model-key-must-not-persist" not in persisted
    assert registered.token not in persisted
    capability_record = store.read_capability_record(registered.record.grant_id)
    assert capability_record["token_sha256"]
    assert "capability_token" not in capability_record
    assert executor.calls == []
