"""Exercise the model/runtime boundary of the opt-in HTML acquisition stage."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace

import pytest

from app.acquisition import AcquisitionService
from app.config import Settings
from app.models import ModelRuntimeConfig, SourceManifest, TargetProfile
from app.orchestration.html_guided_stage import (
    normalize_guided_html_source,
    parse_plan_response,
)
from app.orchestration.source_stage import SourceStageError, normalize_source
from app.runtime import AgentTurnResult, FakeAgentRuntime
from app.storage import RunStore


FAKE_KEY = "sk-unit-test-only-do-not-persist"
HTML = """<html><head><title>Example Manual</title></head><body><main>
<h1>Example Manual</h1><h2>Open data</h2><p>Choose File, then Open.</p>
<h2>Save data</h2><p>Choose File, then Save.</p>
</main></body></html>"""


class PlanningRuntime(FakeAgentRuntime):
    def __init__(self, *, responses=(), cancel=False):
        super().__init__()
        self.responses = iter(responses)
        self.cancel_in_run = cancel
        self.closed_sessions = []

    async def run(self, invocation):
        self.invocations.append(invocation)
        if self.cancel_in_run:
            raise asyncio.CancelledError()
        response = next(self.responses, None)
        if response is None:
            payload = json.loads(str(invocation.prompt).rsplit("\n", 1)[-1])
            page = payload["page"]
            main = next(item for item in page["containers"] if item["tag"] == "main")
            response = json.dumps({
                "page_type": "single_page_manual",
                "content_root_id": main["container_id"],
                "links": [],
                "summary": "Invented summary prose must never become source evidence.",
            })
        return AgentTurnResult(
            session_id=invocation.session_id,
            final_response=response,
            finish_reason="completed",
            usage={"input_tokens": 17, "output_tokens": 9, "nested": {}, "flag": True},
        )

    async def close_session(self, session_id):
        self.closed_sessions.append(session_id)


def stage_inputs(tmp_path, monkeypatch, **settings_overrides):
    settings = Settings(
        data_dir=tmp_path / "data", output_dir=tmp_path / "output",
        html_processing_mode="model_guided", max_html_pages=0,
        html_guided_planner_transport="runtime",
        **settings_overrides,
    )
    target = TargetProfile(software_name="Example", target_version="1", platform="General")
    config = ModelRuntimeConfig(
        base_url="https://api.deepseek.com", model="deepseek-flash", api_key=FAKE_KEY,
        reasoning_effort="high",
    )
    store = RunStore(settings.data_dir, "run_guided_stage")
    store.initialize({"target_profile": target.model_dump(mode="json"), "model": config.public_dict()})
    raw = store.put_artifact(HTML.encode("utf-8"), kind="raw_source", media_type="text/html", suffix=".html")
    manifest = SourceManifest(
        source_id="SRC1", canonical_input_id="IN1", display_name="Example Manual",
        primary_modality="html", media_type="text/html", raw_artifact_id=raw.artifact_id,
        origin_url="https://example.org/manual.html", final_url="https://example.org/manual.html",
    )
    acquisition = AcquisitionService(store, settings)

    def unexpected_fetch(*args, **kwargs):
        pytest.fail("This complete single-page manual must not fetch additional URLs")

    monkeypatch.setattr(acquisition, "_read_url", unexpected_fetch)
    return store, settings, acquisition, target, manifest, config


def test_plan_json_accepts_plain_or_fenced_object():
    plan = {"content_root_id": "known", "links": []}
    assert parse_plan_response(json.dumps(plan)) == plan
    assert parse_plan_response("```json\n" + json.dumps(plan) + "\n```") == plan


@pytest.mark.parametrize("response", ["not JSON", "[]", "null", '"prose"', '{"broken":'])
def test_plan_json_rejects_non_object_or_malformed_response(response):
    with pytest.raises((ValueError, TypeError)):
        parse_plan_response(response)


@pytest.mark.asyncio
async def test_runtime_receives_snapshot_and_stage_persists_grounded_prepared_source(tmp_path, monkeypatch):
    store, settings, acquisition, target, manifest, config = stage_inputs(tmp_path, monkeypatch)
    runtime = PlanningRuntime()
    prepared = await normalize_guided_html_source(
        store, settings, acquisition, target, manifest, runtime=runtime, model_config=config,
    )
    assert len(runtime.invocations) == 1
    invocation = runtime.invocations[0]
    assert invocation.role == "html_acquisition_planning"
    assert invocation.allowed_tools == ()
    assert invocation.model["api_key"] == FAKE_KEY
    assert invocation.model["reasoning_effort"] == "none"
    assert config.reasoning_effort == "high"
    assert callable(invocation.metadata["runtime_event_sink"])
    payload = json.loads(str(invocation.prompt).rsplit("\n", 1)[-1])
    assert payload["source_id"] == "SRC1"
    assert payload["page"]["url"] == manifest.final_url
    assert payload["page"]["headings"][1]["title"] == "Open data"
    assert prepared.workspace.inventory_completed
    assert prepared.workspace.evidence_tree is not None
    assert prepared.workspace.evidence_tree.referenced_part_count > 0
    text = "\n".join(prepared.workspace.part_text.values())
    assert "Choose File, then Open." in text
    assert "Choose File, then Save." in text
    assert "Invented summary prose" not in text
    normalization = store.read_json_artifact(prepared.normalization_artifact_id)
    assert normalization["html_processing_mode"] == "model_guided"
    assert prepared.manifest.source_metadata["html_guided_model_policy"] == {
        "planner_transport": "runtime", "planner_reasoning_effort": "none", "review_reasoning_effort": "low",
    }
    attempts = store.list_attempts()
    assert len(attempts) == 1 and attempts[0].status == "succeeded"
    session = store.read_session(attempts[0].child_session_id)
    assert session.usage == {"input_tokens": 17, "output_tokens": 9}
    assert runtime.closed_sessions == [invocation.session_id]
    assert runtime.cancelled_sessions == []
    for path in store.root.rglob("*"):
        if path.is_file():
            assert FAKE_KEY.encode("utf-8") not in path.read_bytes(), path


@pytest.mark.asyncio
async def test_invalid_json_twice_fails_attempt_before_source_text_materializes(tmp_path, monkeypatch):
    store, settings, acquisition, target, manifest, config = stage_inputs(tmp_path, monkeypatch)
    runtime = PlanningRuntime(responses=["invented source text", "still not valid JSON"])
    with pytest.raises(SourceStageError) as error:
        await normalize_guided_html_source(
            store, settings, acquisition, target, manifest, runtime=runtime, model_config=config,
        )
    assert error.value.category == "html_guided_plan_invalid"
    assert len(runtime.invocations) == 2
    assert runtime.invocations[0].session_id == runtime.invocations[1].session_id
    assert "valid JSON object" in str(runtime.invocations[1].prompt)
    attempts = store.list_attempts()
    assert len(attempts) == 1 and attempts[0].status == "failed"
    assert attempts[0].error_category == "html_guided_plan_invalid"
    assert runtime.closed_sessions == [attempts[0].child_session_id]
    assert json.loads((store.root / "sources" / "SRC1" / "parts.json").read_text(encoding="utf-8")) == []
    assert not list((store.root / "sources" / "SRC1").glob("result.json"))


@pytest.mark.asyncio
async def test_cancellation_cancels_runtime_and_closes_persisted_attempt(tmp_path, monkeypatch):
    store, settings, acquisition, target, manifest, config = stage_inputs(tmp_path, monkeypatch)
    runtime = PlanningRuntime(cancel=True)
    with pytest.raises(asyncio.CancelledError):
        await normalize_guided_html_source(
            store, settings, acquisition, target, manifest, runtime=runtime, model_config=config,
        )
    attempts = store.list_attempts()
    assert len(attempts) == 1 and attempts[0].status == "cancelled"
    assert attempts[0].error_category == "cancelled"
    session_id = attempts[0].child_session_id
    assert store.read_session(session_id).status == "cancelled"
    assert runtime.cancelled_sessions == [session_id]
    assert runtime.closed_sessions == [session_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("incompatible", ["selected_scope", "adaptive_profile"])
async def test_incompatible_checkpoint_or_profile_stops_before_model_call(tmp_path, monkeypatch, incompatible):
    overrides = {"t5_processing_mode": "agent"} if incompatible == "adaptive_profile" else {}
    store, settings, acquisition, target, manifest, config = stage_inputs(tmp_path, monkeypatch, **overrides)
    if incompatible == "selected_scope":
        manifest.source_metadata["extraction_scope"] = {"mode": "selected"}
    runtime = PlanningRuntime()
    with pytest.raises(SourceStageError):
        await normalize_guided_html_source(
            store, settings, acquisition, target, manifest, runtime=runtime, model_config=config,
        )
    assert runtime.invocations == []
    assert store.list_attempts() == []


def test_legacy_html_route_remains_the_default():
    if "PREPROCESS_HTML_PROCESSING_MODE" in os.environ:
        pytest.skip("The runner explicitly configured an HTML pipeline")
    assert Settings.__dataclass_fields__["html_processing_mode"].default == "legacy"


def test_resuming_guided_manifest_through_legacy_clears_guided_metadata(tmp_path, monkeypatch):
    store, settings, acquisition, target, manifest, config = stage_inputs(tmp_path, monkeypatch)
    settings = replace(settings, html_processing_mode="legacy")
    manifest.source_metadata.update({
        "html_processing_mode": "model_guided",
        "acquisition_coverage": {
            "decision_artifact_id": "stale-guided-plan",
            "max_planning_rounds": 999,
            "acquisition_complete": False,
            "unfetched_discovered_urls": ["https://example.org/old-guided-gap.html"],
        },
    })
    prepared = normalize_source(store, settings, acquisition, target, manifest)
    assert prepared.manifest.source_metadata["html_processing_mode"] == "legacy"
    assert prepared.workspace.NORMALIZATION_VERSION.startswith("v16-html-structured")
    coverage = prepared.manifest.source_metadata["acquisition_coverage"]
    assert "decision_artifact_id" not in coverage
    assert "max_planning_rounds" not in coverage
    assert coverage["acquisition_complete"] is True
    assert coverage["unfetched_discovered_urls"] == []
    assert prepared.workspace.evidence_tree is not None
    assert len(prepared.workspace.evidence_tree.nodes) == 1
    assert "Choose File, then Open." in prepared.workspace.evidence_tree.nodes[0].content
    assert "Choose File, then Save." in prepared.workspace.evidence_tree.nodes[0].content
    record = store.read_json_artifact(prepared.normalization_artifact_id)
    assert record["html_processing_mode"] == "legacy"
