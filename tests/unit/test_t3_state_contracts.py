from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.contracts import ChildRunCapsule, StageReceipt
from app.models import ModelRuntimeConfig, StagedInput, TargetProfile
from app.model_errors import NoUsableSourcesError
from app.orchestration import (
    InvalidStateTransitionError,
    PipelineRuntime,
    PipelineStateMachine,
    RunPhase,
)
from app.storage import (
    ArtifactLimitError,
    ReceiptConflictError,
    RunStore,
    RunStoreConflictError,
    StateProjectionMutationError,
    _progress_for_phase,
)


def _request(*artifact_ids: str) -> dict[str, object]:
    return {
        "run_id": "ignored_by_contract",
        "target_profile": {
            "software_name": "Example",
            "target_version": "1.0",
            "platform": "Linux",
            "ui_locale": "English",
        },
        "inputs": [{"artifact_id": artifact_id} for artifact_id in artifact_ids],
        "model": {
            "provider": "openai-compatible",
            "model": "gpt-test",
            "api_protocol": "responses",
            "reasoning_effort": "medium",
            "base_url": "https://example.invalid/v1",
            "api_key": "must-not-persist",
        },
        "prompt_set_version": "test-prompts.v1",
    }


def _initialized_store(tmp_path, run_id: str = "run_t3") -> tuple[RunStore, str]:
    store = RunStore(tmp_path, run_id)
    artifact = store.put_json_artifact({"input": "small"}, kind="test_input")
    store.initialize(_request(artifact.artifact_id))
    return store, artifact.artifact_id


def test_unified_task_trajectory_phase_has_progress_between_reference_and_skill() -> None:
    assert (_progress_for_phase("references_ready")
            < _progress_for_phase("discovering_task_trajectories")
            < _progress_for_phase("inducing_skills"))


def test_retired_preferences_do_not_enter_new_request_state_or_capsule(tmp_path):
    request = _request()
    request["target_profile"].update(operation_path_preference="gui", operation_modes=["GUI"])
    original = json.dumps(request, sort_keys=True)
    store = RunStore(tmp_path, "run_retired_preferences")
    store.initialize(request)
    for profile in (store.read_state()["target_profile"],
                    json.loads((store.root / "request.json").read_text(encoding="utf-8"))["target_profile"],
                    store.read_run_context().target_profile.model_dump(mode="json")):
        assert "operation_path_preference" not in profile
        assert "operation_modes" not in profile
    assert json.dumps(request, sort_keys=True) == original
    # A repeated obsolete preference must not change the effective request.
    request["target_profile"]["operation_path_preference"] = "instruction"
    store.initialize(request)


def test_old_run_context_capsule_ignores_retired_preferences_without_changing_artifact(tmp_path):
    store, _ = _initialized_store(tmp_path, "run_old_profile")
    old_capsule = store.read_run_context().model_dump(mode="json")
    old_capsule["target_profile"].update(operation_path_preference="gui", operation_modes=["GUI"])
    artifact = store.put_json_artifact(old_capsule, kind="run_context_capsule")
    state = store.read_state()
    state["run_context_artifact_id"] = artifact.artifact_id
    (store.root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    profile = store.read_run_context().target_profile.model_dump(mode="json")
    assert "operation_path_preference" not in profile
    assert "operation_modes" not in profile
    assert store.read_json_artifact(artifact.artifact_id) == old_capsule


def _receipt(
    *,
    run_id: str = "run_t3",
    receipt_id: str = "rcpt_acquire",
    idempotency_key: str = "acquisition:input-v1",
    output_artifact_id: str,
    counts: dict[str, int] | None = None,
) -> StageReceipt:
    return StageReceipt(
        receipt_id=receipt_id,
        idempotency_key=idempotency_key,
        run_id=run_id,
        stage="acquisition",
        status="succeeded",
        input_fingerprint=hashlib.sha256(b"input-v1").hexdigest(),
        accepted_output_artifact_ids=(output_artifact_id,),
        counts=counts or {"sources": 1},
    )


def test_initialize_creates_one_root_and_credential_free_capsule(tmp_path):
    store, input_artifact_id = _initialized_store(tmp_path)

    state = store.read_state()
    sessions = store.list_sessions()
    capsule = store.read_run_context()
    serialized = json.dumps(capsule.model_dump(mode="json"), ensure_ascii=False)

    assert state["phase"] == "created"
    assert state["revision"] == 0
    assert len(sessions) == 1
    assert sessions[0].session_kind == "root"
    assert sessions[0].session_id == state["root_session_id"]
    assert capsule.input_artifact_ids == (input_artifact_id,)
    assert capsule.root_session_id == sessions[0].session_id
    assert "must-not-persist" not in serialized
    assert "api_key" not in serialized
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in store.root.rglob("*")
        if path.is_file()
    )
    assert "must-not-persist" not in persisted
    with pytest.raises(ValidationError):
        capsule.run_id = "run_changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        capsule.target_profile.software_name = "Changed"  # type: ignore[misc]


def test_each_run_gets_a_distinct_root_even_for_same_request(tmp_path):
    first, _ = _initialized_store(tmp_path, "run_first")
    second, _ = _initialized_store(tmp_path, "run_second")

    assert first.read_state()["root_session_id"] != second.read_state()["root_session_id"]


def test_child_capsule_has_only_structured_handoff_and_owned_artifact_ids(tmp_path):
    store, input_artifact_id = _initialized_store(tmp_path)

    attempt, capsule, session = store.begin_child_attempt(
        stage="task_discovery",
        input_artifact_ids=(input_artifact_id,),
        scope={"corpus_snapshot_id": "corpus_1"},
        allowed_tools=("list_evidence", "submit_task_batch"),
        output_contract="task-candidate.v1",
        prompt_version="task-discovery.v1",
    )

    payload = capsule.model_dump(mode="json")
    assert payload["input_artifact_ids"] == [input_artifact_id]
    assert not {"messages", "events", "history"}.intersection(payload)
    assert attempt.child_session_id == session.session_id == capsule.child_session_id
    assert session.parent_root_session_id == store.read_state()["root_session_id"]
    assert attempt.attempt_number == 1

    with pytest.raises(ValidationError, match="forbidden field"):
        ChildRunCapsule(
            run_id="run_t3",
            parent_root_session_id="root_1",
            child_session_id="child_1",
            attempt_id="attempt_1",
            stage="task_discovery",
            scope={"messages": [{"role": "user", "content": "leak"}]},
            output_contract="task-candidate.v1",
            prompt_version="task-discovery.v1",
        )
    with pytest.raises(ValidationError, match="forbidden field"):
        ChildRunCapsule(
            run_id="run_t3",
            parent_root_session_id="root_1",
            child_session_id="child_1",
            attempt_id="attempt_1",
            stage="task_discovery",
            scope={"api_key": "must-not-cross-handoff"},
            output_contract="task-candidate.v1",
            prompt_version="task-discovery.v1",
        )


def test_large_inline_scope_is_rejected_in_favor_of_artifact(tmp_path):
    store, input_artifact_id = _initialized_store(tmp_path)

    with pytest.raises(ValidationError, match="store large content as an Artifact"):
        store.begin_child_attempt(
            stage="reference_build",
            input_artifact_ids=(input_artifact_id,),
            scope={"inline_corpus": "x" * 9_000},
            allowed_tools=(),
            output_contract="reference-candidate.v1",
            prompt_version="reference-build.v1",
        )


def test_retry_creates_a_new_attempt_and_child_session(tmp_path):
    store, input_artifact_id = _initialized_store(tmp_path)
    first, _, first_session = store.begin_child_attempt(
        stage="trajectory_inference",
        input_artifact_ids=(input_artifact_id,),
        scope={"task_id": "tsk_1"},
        allowed_tools=("read_evidence",),
        output_contract="trajectory-candidate.v1",
        prompt_version="trajectory.v1",
    )
    store.start_attempt(first.attempt_id)
    store.finish_attempt(
        first.attempt_id,
        status="failed",
        error_category="transport",
        termination_reason="injected failure",
    )

    retry, _, retry_session = store.begin_child_attempt(
        stage="trajectory_inference",
        input_artifact_ids=(input_artifact_id,),
        scope={"task_id": "tsk_1"},
        allowed_tools=("read_evidence",),
        output_contract="trajectory-candidate.v1",
        prompt_version="trajectory.v1",
        retry_of_attempt_id=first.attempt_id,
    )

    assert retry.attempt_id != first.attempt_id
    assert retry_session.session_id != first_session.session_id
    assert retry.retry_of_attempt_id == first.attempt_id
    assert retry.attempt_number == 2


def test_retry_rejects_changed_inputs(tmp_path):
    store, input_artifact_id = _initialized_store(tmp_path)
    first, _, _ = store.begin_child_attempt(
        stage="trajectory_inference",
        input_artifact_ids=(input_artifact_id,),
        scope={"task_id": "tsk_1"},
        allowed_tools=(),
        output_contract="trajectory-candidate.v1",
        prompt_version="trajectory.v1",
    )
    store.finish_attempt(first.attempt_id, status="failed")

    with pytest.raises(RunStoreConflictError, match="fingerprint"):
        store.begin_child_attempt(
            stage="trajectory_inference",
            input_artifact_ids=(input_artifact_id,),
            scope={"task_id": "tsk_changed"},
            allowed_tools=(),
            output_contract="trajectory-candidate.v1",
            prompt_version="trajectory.v1",
            retry_of_attempt_id=first.attempt_id,
        )


def test_receipt_gates_state_transition_and_replays_idempotently(tmp_path):
    store, _ = _initialized_store(tmp_path)
    output = store.put_json_artifact({"sources": []}, kind="acquisition_result")
    receipt = _receipt(output_artifact_id=output.artifact_id)
    machine = PipelineStateMachine(store)

    committed = machine.advance(RunPhase.ACQUIRING, receipt)
    replayed = machine.advance(RunPhase.ACQUIRING, receipt)

    assert committed["phase"] == "acquiring"
    assert committed["revision"] == 1
    assert committed["receipt_replayed"] is False
    assert replayed["revision"] == 1
    assert replayed["receipt_replayed"] is True
    assert len(store.list_receipt_commits()) == 1


def test_idempotency_key_conflict_is_rejected(tmp_path):
    store, _ = _initialized_store(tmp_path)
    output = store.put_json_artifact({"sources": []}, kind="acquisition_result")
    machine = PipelineStateMachine(store)
    machine.advance(RunPhase.ACQUIRING, _receipt(output_artifact_id=output.artifact_id))
    conflicting = _receipt(
        receipt_id="rcpt_conflict",
        output_artifact_id=output.artifact_id,
        counts={"sources": 2},
    )

    with pytest.raises(ReceiptConflictError, match="Idempotency key"):
        machine.advance(RunPhase.ACQUIRING, conflicting)


def test_illegal_transition_and_foreign_artifact_are_rejected(tmp_path):
    store, _ = _initialized_store(tmp_path)
    other, _ = _initialized_store(tmp_path, "run_other")
    foreign = other.put_json_artifact({"foreign": True}, kind="test")
    machine = PipelineStateMachine(store)

    with pytest.raises(InvalidStateTransitionError, match="Illegal Run transition"):
        machine.advance(
            RunPhase.SOURCE_REVIEW,
            _receipt(output_artifact_id=foreign.artifact_id),
        )
    with pytest.raises(FileNotFoundError):
        machine.advance(
            RunPhase.ACQUIRING,
            _receipt(output_artifact_id=foreign.artifact_id),
        )


def test_recovery_projects_state_from_last_durable_receipt(tmp_path):
    store, _ = _initialized_store(tmp_path)
    output = store.put_json_artifact({"sources": []}, kind="acquisition_result")
    machine = PipelineStateMachine(store)
    machine.advance(RunPhase.ACQUIRING, _receipt(output_artifact_id=output.artifact_id))
    stale = store.read_state()
    stale.update(
        {
            "phase": "created",
            "stage": "registered",
            "status": "queued",
            "revision": 0,
            "last_receipt_id": None,
            "recovery_cursor": None,
        }
    )
    store._write_json(store.root / "state.json", stale)  # deliberate crash fault injection

    recovered = machine.recover()

    assert recovered["phase"] == "acquiring"
    assert recovered["revision"] == 1
    assert recovered["last_receipt_id"] == "rcpt_acquire"


def test_critical_state_projection_cannot_bypass_receipt_gate(tmp_path):
    store, _ = _initialized_store(tmp_path)

    with pytest.raises(StateProjectionMutationError, match="cannot be updated directly"):
        store.update_state(phase="completed", status="completed", revision=999)


def test_full_declared_state_path_reaches_completed_and_then_is_terminal(tmp_path):
    store, _ = _initialized_store(tmp_path)
    output = store.put_json_artifact({"checkpoint": True}, kind="checkpoint")
    machine = PipelineStateMachine(store)
    phases = [
        RunPhase.ACQUIRING,
        RunPhase.NORMALIZING,
        RunPhase.SOURCE_REVIEW,
        RunPhase.CORPUS_FROZEN,
        RunPhase.REFERENCE_BUILDING,
        RunPhase.REFERENCES_READY,
        RunPhase.DISCOVERING_TASK_TRAJECTORIES,
        RunPhase.INDUCING_SKILLS,
        RunPhase.VALIDATING,
        RunPhase.COMMITTING,
        RunPhase.COMPLETED,
    ]
    for index, phase in enumerate(phases, start=1):
        receipt = StageReceipt(
            receipt_id=f"rcpt_step_{index}",
            idempotency_key=f"state-path:step-{index}",
            run_id="run_t3",
            stage=f"step_{index}",
            status="succeeded",
            input_fingerprint=hashlib.sha256(f"step-{index}".encode()).hexdigest(),
            accepted_output_artifact_ids=(output.artifact_id,),
        )
        machine.advance(phase, receipt)

    assert machine.phase is RunPhase.COMPLETED
    assert machine.terminal is True
    assert store.read_state()["revision"] == len(phases)
    with pytest.raises(InvalidStateTransitionError):
        machine.advance(
            RunPhase.FAILED,
            StageReceipt(
                receipt_id="rcpt_after_terminal",
                idempotency_key="state-path:after-terminal",
                run_id="run_t3",
                stage="after_terminal",
                status="failed",
                input_fingerprint=hashlib.sha256(b"after").hexdigest(),
            ),
        )


def test_artifact_size_limit_is_enforced_before_write(tmp_path):
    store = RunStore(tmp_path, "run_limited", max_artifact_bytes=4)

    with pytest.raises(ArtifactLimitError):
        store.put_artifact(b"12345", kind="too_large", media_type="application/octet-stream")
    assert not list((store.root / "artifacts" / "objects").iterdir())


def test_content_addressed_artifact_cannot_be_overwritten(tmp_path):
    store = RunStore(tmp_path, "run_immutable")
    first = store.put_artifact(
        b"immutable",
        kind="first_kind",
        media_type="application/octet-stream",
        metadata={"version": 1},
    )
    replay = store.put_artifact(
        b"immutable",
        kind="different_kind",
        media_type="text/plain",
        metadata={"version": 2},
    )

    assert replay == first
    assert store.artifact_path(first.artifact_id).read_bytes() == b"immutable"
    assert store.artifact_record(first.artifact_id).metadata == {"version": 1}


@pytest.mark.asyncio
async def test_t5_acquisition_failure_is_also_receipt_gated(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",

    )
    runtime = PipelineRuntime(settings)
    store = runtime.create_store("run_scaffold")
    store.initialize(_request())
    target = TargetProfile(
        software_name="Example",
        target_version="1.0",
        platform="Linux",
        ui_locale="English",
    )
    model = ModelRuntimeConfig(
        base_url="https://example.invalid/v1",
        model="gpt-test",
        api_key="test-only",
    )

    with pytest.raises(NoUsableSourcesError):
        await runtime.execute(
            "run_scaffold",
            target,
            [
                StagedInput(
                    input_id="inp_1",
                    kind="uploaded_file",
                    display_name="test.pdf",
                    staged_path="test.pdf",
                )
            ],
            model,
        )

    state = store.read_state()
    commits = store.list_receipt_commits()
    assert state["phase"] == "failed"
    assert state["revision"] == 1
    assert len(commits) == 1
    assert commits[0]["receipt"]["status"] == "failed"
    assert state["error"]["category"] == "no_usable_sources"
