from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.orchestration.pipeline as pipeline_module
from app.config import Settings
from app.contracts import StageReceipt
from app.models import ModelRuntimeConfig, StagedInput, TargetProfile
from app.orchestration import PipelineRuntime, PipelineStateMachine, RunPhase
from app.orchestration.source_discovery_stage import (
    SOURCE_DISCOVERY_CONTRACT,
    SOURCE_DISCOVERY_PROMPT_VERSION,
    SourceDiscoveryArtifacts,
)
from app.source_discovery_contracts import SourceCandidateSubmission
from app.storage import RunStore


MODEL_KEY = "source-selection-model-key-that-must-not-persist"
TARGET = {"software_name": "Example", "target_version": "1.9", "platform": "General"}


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        
        skill_output_language="English",
    )


def _form(**overrides: str) -> dict[str, str]:
    return {
        "software_name": "Example",
        "target_version": "1.9",
        "platform": "General",
        "source_mode": "discovery",
        "output_language": "Chinese",
        "api_base_url": "https://model.example.invalid/v1",
        "api_model": "gpt-test",
        "api_key": MODEL_KEY,
        **overrides,
    }


def _receipt(store: RunStore, phase: RunPhase, *, status: str = "succeeded") -> StageReceipt:
    return StageReceipt(
        receipt_id=f"rcpt_{phase.value}",
        idempotency_key=f"source-selection:{phase.value}",
        run_id=store.run_id,
        stage=phase.value,
        status=status,
        input_fingerprint=hashlib.sha256(phase.value.encode()).hexdigest(),
    )


def _waiting_state(store: RunStore) -> None:
    machine = PipelineStateMachine(store)
    machine.advance(RunPhase.DISCOVERING_SOURCES, _receipt(store, RunPhase.DISCOVERING_SOURCES))
    machine.advance(
        RunPhase.AWAITING_SOURCE_SELECTION,
        _receipt(store, RunPhase.AWAITING_SOURCE_SELECTION),
    )


def _poll_phase(client: TestClient, run_id: str, expected: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        payload = response.json()
        phase = payload["state"]["phase"]
        if phase == expected:
            return payload
        assert phase not in {"failed", "cancelled"}, payload
        time.sleep(0.01)
    pytest.fail(f"Run {run_id} did not reach {expected}; last payload: {payload}")


def _persisted_text(store: RunStore) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in store.root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )


class RecordingPipelineRuntime(PipelineRuntime):
    """Keep real discovery/state handling, stopping at downstream dispatch."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.dispatched: list[tuple[str, TargetProfile, list[StagedInput], ModelRuntimeConfig]] = []

    def start(self, run_id, target_profile, inputs, model_config) -> None:
        if not inputs:
            super().start(run_id, target_profile, inputs, model_config)
            return
        self.dispatched.append((run_id, target_profile, inputs, model_config))
        store = self.get_store(run_id)
        PipelineStateMachine(store).advance(RunPhase.ACQUIRING, _receipt(store, RunPhase.ACQUIRING))


@pytest.fixture
def api_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path)
    runtime = RecordingPipelineRuntime(settings)
    environment = SimpleNamespace(runtime=runtime, calls=[], candidate_count=3)

    class FakeSourceDiscoveryService:
        def __init__(self, *, store, settings, **kwargs):
            self.store = store
            self.settings = settings

        async def discover(self, *, target_profile, output_language):
            environment.calls.append((self.store.run_id, output_language, self.settings.skill_output_language))
            attempt, _, _ = self.store.begin_child_attempt(
                stage="source_discovery", input_artifact_ids=(),
                scope={"output_language": output_language},
                allowed_tools=("submit_source_candidates",),
                output_contract=SOURCE_DISCOVERY_CONTRACT, prompt_version=SOURCE_DISCOVERY_PROMPT_VERSION,
            )
            self.store.start_attempt(attempt.attempt_id)
            batch = len(environment.calls)
            candidates = [
                {
                    "title": f"Example manual {index + 1}",
                    "url": f"https://manual.example.invalid/v{batch}/{index + 1}/",
                    "modality": "pdf" if index == 1 else "html",
                    "document_version": "1.9" if index == 0 else "1.8",
                    "version_relation": "exact" if index == 0 else "older",
                    "version_evidence": "Version stated in the official manual directory.",
                    "language": "English",
                    "platform": "General",
                    "source_type": "official",
                    "reason": "Official manual with version information.",
                    "root_assessment": "Contents page links to all major chapters." if index != 1 else "PDF manual.",
                }
                for index in range(environment.candidate_count)
            ]
            submission = SourceCandidateSubmission(
                candidates=candidates, summary="Select one manual.", search_notes=[f"Search batch {batch}"],
            )
            identified = submission.identified_candidates()
            payload = {
                "schema_version": SOURCE_DISCOVERY_CONTRACT,
                "candidates": [item.model_dump(mode="json") for item in identified],
                "summary": submission.summary, "search_notes": submission.search_notes,
            }
            artifact = self.store.put_json_artifact(payload, kind="source_discovery")
            self.store.finish_attempt(attempt.attempt_id, status="succeeded")
            return SourceDiscoveryArtifacts(
                candidate_artifact_id=artifact.artifact_id, candidates=identified,
                summary=payload["summary"], search_notes=tuple(payload["search_notes"]), attempt_id=attempt.attempt_id,
            )

    monkeypatch.setattr(main_module, "settings", settings)
    monkeypatch.setattr(main_module, "runtime", runtime)
    monkeypatch.setattr(pipeline_module, "SourceDiscoveryService", FakeSourceDiscoveryService)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(main_module.app) as client:
        environment.client = client
        yield environment
    runtime.close()


def _create_discovery(environment) -> tuple[str, dict]:
    response = environment.client.post("/api/runs", data=_form())
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    return run_id, _poll_phase(environment.client, run_id, "awaiting_source_selection")


@pytest.mark.parametrize("candidate_count", [1, 2, 3])
def test_discovery_returns_small_candidate_batches_and_waits_without_model_task(api_environment, candidate_count: int) -> None:
    api_environment.candidate_count = candidate_count
    run_id, payload = _create_discovery(api_environment)

    assert payload["state"]["status"] == "waiting_for_input"
    assert payload["state"]["active_task"] is False
    assert payload["state"]["requires_api_key"] is False
    assert len(payload["source_discovery"]["candidates"]) == candidate_count
    assert payload["sources"] == []
    assert api_environment.calls == [(run_id, "Chinese", "Chinese")]
    assert not api_environment.runtime.dispatched
    assert MODEL_KEY not in _persisted_text(api_environment.runtime.get_store(run_id))


def test_retired_path_preference_is_absent_from_api_contract_and_new_runs(api_environment) -> None:
    client = api_environment.client
    document = client.get("/openapi.json").json()
    schema_ref = document["paths"]["/api/runs"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    form_schema = document["components"]["schemas"][schema_ref.rsplit("/", 1)[1]]
    assert "operation_path_preference" not in form_schema["properties"]
    response = client.post(
        "/api/runs", data=_form(source_mode="manual", operation_path_preference="gui"),
        files={"files": ("manual.html", b"<h1>Example manual</h1>", "text/html")},
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    store = api_environment.runtime.get_store(run_id)
    assert "operation_path_preference" not in store.read_state()["target_profile"]
    assert "operation_path_preference" not in store.read_run_context().target_profile.model_dump()
    assert "operation_path_preference" not in api_environment.runtime.dispatched[0][1].model_dump()
    assert "operation_path_preference" not in _persisted_text(store)


def test_old_run_api_profiles_ignore_retired_preferences_without_rewriting_state(api_environment) -> None:
    client = api_environment.client
    response = client.post(
        "/api/runs", data=_form(source_mode="manual"),
        files={"files": ("manual.html", b"<h1>Example manual</h1>", "text/html")},
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    store = api_environment.runtime.get_store(run_id)
    state_path = store.root / "state.json"
    old_state = store.read_state()
    old_state["target_profile"].update(operation_path_preference="gui", operation_modes=["GUI"])
    state_path.write_text(json.dumps(old_state), encoding="utf-8")
    before = state_path.read_bytes()
    profiles = [client.get(f"/api/runs/{run_id}").json()["state"]["target_profile"],
                next(row for row in client.get("/api/runs").json()["runs"] if row["run_id"] == run_id)["target_profile"]]
    for profile in profiles:
        assert "operation_path_preference" not in profile
        assert "operation_modes" not in profile
    assert state_path.read_bytes() == before


def test_manual_upload_skips_search_and_explicit_manual_accepts_only_one_source(api_environment) -> None:
    client = api_environment.client
    response = client.post(
        "/api/runs", data=_form(source_mode="manual"),
        files={"files": ("manual.html", b"<h1>Example manual</h1>", "text/html")},
    )
    assert response.status_code == 202, response.text
    assert not api_environment.calls
    assert len(api_environment.runtime.dispatched) == 1
    staged = api_environment.runtime.dispatched[0][2]
    assert len(staged) == 1
    assert staged[0].source_origin == "user"

    rejected = client.post(
        "/api/runs", data=_form(source_mode="manual", urls_json=json.dumps([
            "https://manual.example.invalid/one/", "https://manual.example.invalid/two/",
        ])),
    )
    assert rejected.status_code == 422
    assert len(api_environment.runtime.dispatched) == 1


def test_discovery_rejects_manual_sources_and_unknown_candidate_ids(api_environment) -> None:
    with_sources = api_environment.client.post(
        "/api/runs", data=_form(urls_json='["https://manual.example.invalid/"]'),
    )
    assert with_sources.status_code == 422
    assert not api_environment.calls

    run_id, _ = _create_discovery(api_environment)
    rejected = api_environment.client.post(
        f"/api/runs/{run_id}/select-source",
        json={"candidate_id": "https://arbitrary.example.invalid/manual.html"},
    )
    assert rejected.status_code == 422
    assert api_environment.runtime.get_store(run_id).read_state()["phase"] == "awaiting_source_selection"
    assert not api_environment.runtime.dispatched


@pytest.mark.parametrize("candidate_index,modality", [(0, "html"), (1, "pdf")])
def test_selection_dispatches_one_persisted_source_and_is_idempotent(api_environment, candidate_index: int, modality: str) -> None:
    run_id, payload = _create_discovery(api_environment)
    candidate = payload["source_discovery"]["candidates"][candidate_index]
    candidate_id = candidate["candidate_id"]
    response = api_environment.client.post(f"/api/runs/{run_id}/select-source", json={"candidate_id": candidate_id})
    assert response.status_code == 202, response.text
    assert len(api_environment.runtime.dispatched) == 1
    _, target, inputs, model = api_environment.runtime.dispatched[0]
    assert model.api_key == MODEL_KEY
    assert target.target_version == "1.9"
    assert len(inputs) == 1
    assert inputs[0].url == candidate["url"]
    assert inputs[0].user_declared_version == candidate["document_version"]
    assert inputs[0].source_origin == "websearch"
    assert inputs[0].source_metadata["modality"] == modality
    assert inputs[0].source_metadata["target_version"] == "1.9"

    selected_payload = api_environment.client.get(f"/api/runs/{run_id}").json()
    assert selected_payload["result"] is None
    assert len(selected_payload["sources"]) == 1
    source = selected_payload["sources"][0]
    assert source["is_preview"] is True
    assert source["display_name"] == candidate["title"]
    assert source["modality"] == modality
    assert source["manifest"]["origin_url"] == candidate["url"]
    assert source["manifest"]["resolved_version"] == candidate["document_version"]
    assert MODEL_KEY not in json.dumps(selected_payload)

    repeated = api_environment.client.post(f"/api/runs/{run_id}/select-source", json={"candidate_id": candidate_id})
    assert repeated.status_code == 202
    assert repeated.json()["already_selected"] is True
    assert len(api_environment.runtime.dispatched) == 1
    other = payload["source_discovery"]["candidates"][1 - candidate_index]["candidate_id"]
    changed = api_environment.client.post(f"/api/runs/{run_id}/select-source", json={"candidate_id": other})
    assert changed.status_code == 409
    assert len(api_environment.runtime.dispatched) == 1


def test_cold_selection_restores_public_model_route_and_requires_new_key(api_environment, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id, initial = _create_discovery(api_environment)
    candidate_id = initial["source_discovery"]["candidates"][0]["candidate_id"]
    previous_runtime = api_environment.runtime
    cold_runtime = RecordingPipelineRuntime(previous_runtime.settings)
    monkeypatch.setattr(main_module, "runtime", cold_runtime)
    try:
        restored = api_environment.client.get(f"/api/runs/{run_id}").json()
        assert restored["source_discovery"] == initial["source_discovery"]
        assert restored["state"]["requires_api_key"] is True
        missing_key = api_environment.client.post(f"/api/runs/{run_id}/select-source", json={"candidate_id": candidate_id})
        assert missing_key.status_code == 422
        assert cold_runtime.get_store(run_id).read_state()["phase"] == "awaiting_source_selection"
        assert not cold_runtime.dispatched

        accepted = api_environment.client.post(
            f"/api/runs/{run_id}/select-source",
            json={"candidate_id": candidate_id, "api_key": "replacement-key-do-not-save"},
        )
        assert accepted.status_code == 202, accepted.text
        _, _, _, model = cold_runtime.dispatched[0]
        assert model.base_url == "https://model.example.invalid/v1"
        assert model.model == "gpt-test"
        assert model.api_key == "replacement-key-do-not-save"
        saved_text = _persisted_text(cold_runtime.get_store(run_id))
        assert MODEL_KEY not in saved_text
        assert "replacement-key-do-not-save" not in saved_text
    finally:
        cold_runtime.close()


def test_search_again_replaces_batch_and_waiting_run_can_be_cancelled(api_environment) -> None:
    run_id, initial = _create_discovery(api_environment)
    searched = api_environment.client.post(f"/api/runs/{run_id}/search-again", json={})
    assert searched.status_code == 202, searched.text
    updated = _poll_phase(api_environment.client, run_id, "awaiting_source_selection")
    assert len(api_environment.calls) == 2
    assert updated["source_discovery"]["candidates"][0]["url"] != initial["source_discovery"]["candidates"][0]["url"]
    assert updated["state"]["revision"] > initial["state"]["revision"]
    cancelled = api_environment.client.post(f"/api/runs/{run_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    no_late_selection = api_environment.client.post(
        f"/api/runs/{run_id}/select-source",
        json={"candidate_id": updated["source_discovery"]["candidates"][0]["candidate_id"]},
    )
    assert no_late_selection.status_code == 409
    assert not api_environment.runtime.dispatched


def test_waiting_source_selection_is_durable_and_cancelable(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "run_waiting_selection")
    store.initialize({
        "target_profile": {
            "software_name": "Example", "target_version": "1.9", "platform": "General",
        },
        "source_mode": "discovery",
        "output_language": "Chinese",
    })
    _waiting_state(store)

    reopened = RunStore(tmp_path, store.run_id)
    restored = PipelineStateMachine(reopened).recover()
    assert restored["phase"] == "awaiting_source_selection"
    assert restored["status"] == "waiting_for_input"
    assert restored["output_language"] == "Chinese"

    PipelineStateMachine(reopened).advance(
        RunPhase.CANCELLED,
        _receipt(reopened, RunPhase.CANCELLED, status="cancelled"),
    )
    assert reopened.read_state()["status"] == "cancelled"


def test_discovery_waiting_state_does_not_change_manual_state_path(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "run_manual_state")
    store.initialize({"target_profile": TARGET, "source_mode": "manual", "inputs": [{"input_id": "IN1"}]})
    machine = PipelineStateMachine(store)
    machine.advance(RunPhase.ACQUIRING, _receipt(store, RunPhase.ACQUIRING))

    assert store.read_state()["phase"] == "acquiring"
    assert store.read_state()["source_mode"] == "manual"
    assert not any(session.session_kind == "child" for session in store.list_sessions())


def test_output_languages_are_run_scoped_and_legacy_capsules_default_to_english(tmp_path: Path) -> None:
    shared_settings = _settings(tmp_path)
    first = RunStore(tmp_path, "run_chinese")
    second = RunStore(tmp_path, "run_english")
    legacy = RunStore(tmp_path, "run_legacy_language")
    first.initialize({"target_profile": TARGET, "output_language": "Chinese"})
    second.initialize({"target_profile": TARGET, "output_language": "English"})
    legacy.initialize({"target_profile": TARGET})

    assert first.read_run_context().output_language == "Chinese"
    assert second.read_run_context().output_language == "English"
    assert legacy.read_run_context().output_language == "English"
    assert shared_settings.skill_output_language == "English"


def test_deadline_checkpoint_reaches_selection_and_dispatches_only_after_user_confirmation(api_environment, monkeypatch):
    import asyncio
    import httpx
    from dataclasses import replace
    from app.capabilities import CapabilityInvocation
    from app.runtime import AgentRuntime, AgentRuntimeBusinessCompleted
    from app.orchestration.source_discovery_stage import SourceDiscoveryService
    from test_source_discovery import candidate
    from app.source_candidate_probe import SourceCandidateProbe
    import app.orchestration.source_discovery_stage as discovery_stage

    original_probe = SourceCandidateProbe

    def inspected_fixture(store, **kwargs):
        return original_probe(store, **kwargs, transport=httpx.MockTransport(lambda request:
            httpx.Response(200, content=b'<html><main><h1>Example Manual</h1><p>Complete user instructions.</p></main></html>')))

    monkeypatch.setattr(discovery_stage, 'SourceCandidateProbe', inspected_fixture)

    runtime = api_environment.runtime
    runtime.settings = replace(runtime.settings, source_discovery_timeout_seconds=1)

    class CheckpointRuntime(AgentRuntime):
        ended = None

        async def run(self, invocation):
            self.ended = asyncio.Event()
            raw = invocation.metadata["capability"]
            inspected = runtime.capability_gateway.invoke(raw["capability_token"], CapabilityInvocation(
                **raw["context"], tool_name="inspect_source_candidate", arguments={"url": candidate()["url"]}, call_id="inspect_deadline"))
            assert inspected["ok"] and inspected["status"] == "available"
            reply = runtime.capability_gateway.invoke(raw["capability_token"], CapabilityInvocation(
                **raw["context"], tool_name="save_source_candidate", arguments={"candidate": candidate()}, call_id="save_deadline"))
            assert reply["ok"]
            await self.ended.wait()
            raise AgentRuntimeBusinessCompleted("Checkpoint accepted")

        async def cancel(self, session_id):
            self.ended.set()
            return True

    runtime.agent_runtime = CheckpointRuntime()
    monkeypatch.setattr(pipeline_module, "SourceDiscoveryService", SourceDiscoveryService)
    run_id, payload = _create_discovery(api_environment)
    assert payload["state"]["status"] == "waiting_for_input"
    assert payload["source_discovery"]["completion_reason"] == "time_budget_reached"
    assert len(payload["source_discovery"]["candidates"]) == 1
    assert not runtime.dispatched
    selected = payload["source_discovery"]["candidates"][0]["candidate_id"]
    response = api_environment.client.post(f"/api/runs/{run_id}/select-source", json={"candidate_id": selected})
    assert response.status_code == 202, response.text
    assert len(runtime.dispatched) == 1
    assert runtime.dispatched[0][2][0].source_origin == "websearch"
    assert runtime.dispatched[0][2][0].source_metadata["html_scope"]["kind"] == "user_manual"
    assert runtime.dispatched[0][2][0].source_metadata["verification"]["html_snapshot_artifact_id"]
