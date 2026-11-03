from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import fitz
import httpx
import pytest

import app.orchestration.pipeline as pipeline_module
from app.acquisition import AcquisitionService
from app.capabilities import CapabilityGateway, CapabilityGatewayServer
from app.config import Settings
from app.models import (
    ModelRuntimeConfig,
    SourceResult,
    StagedInput,
    TargetProfile,
)
from app.orchestration.pipeline import PipelineRuntime
from app.orchestration.source_stage import (
    PDF_SOURCE_REVIEW_PROMPT_VERSION,
    SOURCE_RESULT_CONTRACT,
    SOURCE_REVIEW_PROMPT_VERSION,
    SourceStageError,
    SourceReviewProtocolError,
    SourceReviewService,
    normalize_source,
    validate_source_results,
)
from app.runtime import (
    AgentInvocation,
    AgentRuntime,
    AgentRuntimeExecutionError,
    AgentTurnResult,
)
from app.toolkit import ToolExecutor


class GatewayDrivingRuntime(AgentRuntime):
    """A deterministic model substitute that still uses the real HTTP gateway."""

    def __init__(
        self,
        *,
        pause_seconds: float = 0.0,
        review_profile: str = "architect",
    ) -> None:
        self.pause_seconds = pause_seconds
        self.review_profile = review_profile
        self.invocations: list[AgentInvocation] = []
        self.tool_calls: dict[str, list[str]] = {}
        self.closed_sessions: list[str] = []
        self.active = 0
        self.max_active = 0

    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        self.invocations.append(invocation)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.pause_seconds:
                await asyncio.sleep(self.pause_seconds)
            capability = dict(invocation.metadata["capability"])
            context = dict(capability["context"])
            if self.review_profile == "legacy":
                structure_tool = (
                    "pdf_list_structure"
                    if "pdf_list_structure" in invocation.allowed_tools
                    else "html_list_navigation"
                )
                calls = [
                    ("inspect_source", {}),
                    (structure_tool, {}),
                    ("list_source_parts", {"offset": 0, "limit": 100}),
                    (
                        "submit_source_structure",
                        {
                            "summary": "The parser-owned structure is usable.",
                            "coverage_plan": ["Review the complete normalized source."],
                        },
                    ),
                    (
                        "submit_source_result",
                        {
                            "status": "succeeded",
                            "summary": "Source normalization is complete and evidence is parser-owned.",
                            "coverage": {"skipped": []},
                            "gaps": [],
                            "warnings": [],
                        },
                    ),
                ]
            elif self.review_profile == "curated":
                calls = [
                    ("inspect_source", {}),
                    ("list_source_parts", {"offset": 0, "limit": 100}),
                    (
                        "curate_evidence",
                        {
                            "selection_mode": "all_except_excluded",
                            "excluded": [],
                            "merge_groups": [],
                            "rationale": "The compact fixture contains only useful documentation.",
                        },
                    ),
                    (
                        "submit_source_result",
                        {
                            "status": "succeeded",
                            "summary": "Source normalization is complete and evidence is parser-owned.",
                            "coverage": {"skipped": []},
                            "gaps": [],
                            "warnings": [],
                        },
                    ),
                ]
            else:
                calls = []
            async with httpx.AsyncClient(timeout=10) as client:
                call_index = 0

                async def invoke(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                    nonlocal call_index
                    call_index += 1
                    response = await client.post(
                        str(capability["gateway_url"]),
                        headers={"Authorization": f"Bearer {capability['capability_token']}"},
                        json={
                            **context,
                            "tool_name": name,
                            "arguments": arguments,
                            "call_id": f"{invocation.session_id}.{call_index}",
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    assert payload["ok"] is True, payload
                    result = payload["result"]
                    assert result.get("ok", True) is True, payload
                    self.tool_calls.setdefault(invocation.session_id, []).append(name)
                    return result

                if self.review_profile == "architect":
                    await invoke("inspect_source", {})
                    structure_tool = (
                        "get_pdf_document_structure"
                        if "get_pdf_document_structure" in invocation.allowed_tools
                        else "get_root_structure"
                    )
                    structure = await invoke(structure_tool, {})
                    await invoke(
                        "submit_evidence_revision",
                        {
                            "base_plan_artifact_id": structure["base_plan_artifact_id"],
                            "operations": [],
                            "summary": "The Python-authored fixture hierarchy is coherent.",
                        },
                    )
                else:
                    for name, arguments in calls:
                        await invoke(name, arguments)
            return AgentTurnResult(
                session_id=invocation.session_id,
                final_response="source-review-complete",
                finish_reason="completed",
                event_count=len(self.tool_calls.get(invocation.session_id, [])),
                usage={"input_tokens": 12, "output_tokens": 3},
            )
        finally:
            self.active -= 1

    async def cancel(self, session_id: str) -> bool:
        return True

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)


class BlockingRuntime(AgentRuntime):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.closed_sessions: list[str] = []

    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def cancel(self, session_id: str) -> bool:
        return True

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)


class SilentRuntime(AgentRuntime):
    def __init__(self) -> None:
        self.invocations: list[AgentInvocation] = []
        self.closed_sessions: list[str] = []

    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        self.invocations.append(invocation)
        return AgentTurnResult(
            session_id=invocation.session_id,
            final_response="I finished without tools.",
            finish_reason="completed",
        )

    async def cancel(self, session_id: str) -> bool:
        return True

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)


class FailingRuntime(AgentRuntime):
    def __init__(self) -> None:
        self.invocations: list[AgentInvocation] = []
        self.closed_sessions: list[str] = []

    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        self.invocations.append(invocation)
        raise AgentRuntimeExecutionError("provider rejected accumulated prompt")

    async def cancel(self, session_id: str) -> bool:
        return True

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)


def _settings(tmp_path: Path, *, t5_processing_mode: str = "agent_architect") -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",

        max_html_pages=2,
        t5_processing_mode=t5_processing_mode,
        t6_reference_enabled=False,
    )


def _target() -> TargetProfile:
    return TargetProfile(
        software_name="ExampleBio",
        target_version="2.0",
        platform="Linux",
        ui_locale="English",
    )


def _model() -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        api_protocol="responses",
        base_url="https://model.example.invalid/v1",
        model="gpt-test",
        api_key="t5-api-key-must-never-persist",
        reasoning_effort="medium",
    )


def _initialize(runtime: PipelineRuntime, run_id: str, inputs: list[StagedInput]):
    store = runtime.create_store(run_id)
    store.initialize(
        {
            "run_id": run_id,
            "target_profile": _target().model_dump(mode="json"),
            "inputs": [
                {
                    "input_id": item.input_id,
                    "kind": item.kind,
                    "display_name": item.display_name,
                }
                for item in inputs
            ],
            "model": _model().public_dict(),
        }
    )
    return store


def _stage(runtime: PipelineRuntime, run_id: str, filename: str, data: bytes, index: int):
    store = runtime.create_store(run_id)
    input_id = f"inp_{index}"
    path = store.stage_upload(input_id, filename, data)
    media_type = "application/pdf" if filename.endswith(".pdf") else "text/html"
    return StagedInput(
        input_id=input_id,
        kind="uploaded_file",
        display_name=filename,
        staged_path=str(path),
        declared_media_type=media_type,
    )


def _pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 90), "ExampleBio 2.0 - Import a sequence file")
    page.insert_text((72, 120), "Choose File > Import, select sample.fasta, then confirm.")
    second = document.new_page()
    second.insert_text((72, 90), "Review the alignment and export the report.")
    data = document.tobytes()
    document.close()
    return data


def _outlined_pdf_bytes() -> bytes:
    document = fitz.open()
    document.new_page().insert_text((72, 90), "ExampleBio 2.0 Manual")
    page = document.new_page()
    for y, text in (
        (72, "Chapter 1"),
        (96, "Commands"),
        (130, "1.1"),
        (148, "and"),
        (172, "The and command combines both conditions."),
        (210, "1.2"),
        (228, "min"),
        (252, "The min command returns the smallest value."),
    ):
        heading_size = 18 if text == "Commands" else 14 if text in {"and", "min"} else 11
        page.insert_text((72, y), text, fontsize=heading_size,
                         fontname="hebo" if heading_size > 11 else "helv")
    document.set_toc(
        [
            [1, "Commands", 2],
            [2, "and", 2],
            [2, "min", 2],
        ]
    )
    data = document.tobytes()
    document.close()
    return data


def _textless_pdf_bytes() -> bytes:
    document = fitz.open()
    document.new_page()
    data = document.tobytes()
    document.close()
    return data


HTML = b"""<!doctype html><html><body>
<h1>ExampleBio 2.0 User Guide</h1>
<h2>Run alignment</h2><p>Open a FASTA file and choose Align.</p>
<pre>examplebio align sample.fasta</pre>
<table><tr><th>Option</th><th>Meaning</th></tr><tr><td>--fast</td><td>Quick mode</td></tr></table>
</body></html>"""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_t5_deterministic_mode_opens_no_child_session(tmp_path):
    driver = GatewayDrivingRuntime()
    settings = _settings(tmp_path, t5_processing_mode="deterministic")
    runtime = PipelineRuntime(settings, driver)
    run_id = "run_t5_deterministic"
    inputs = [
        _stage(runtime, run_id, "guide.html", HTML, 1),
        _stage(runtime, run_id, "manual.pdf", _pdf_bytes(), 2),
    ]
    store = _initialize(runtime, run_id, inputs)

    snapshot = await runtime.execute(run_id, _target(), inputs, _model())
    runtime.close()

    assert Settings().t5_processing_mode == "agent_architect"
    assert store.read_state()["phase"] == "corpus_frozen"
    assert store.read_state()["t5_processing_mode"] == "deterministic"
    assert driver.invocations == []
    assert store.list_attempts() == []
    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].session_kind == "root"
    assert list((store.root / "capabilities").glob("*.json")) == []
    assert {item["modality"] for item in snapshot["source_results"]} == {"html", "pdf"}
    assert snapshot["evidence_count"] > 0
    assert all(
        item["extraction_method"] == "deterministic_structural_group"
        for item in snapshot["evidence"]
    )
    for source in snapshot["source_results"]:
        workspace = json.loads(
            (store.root / "sources" / source["source_id"] / "workspace.json").read_text(
                encoding="utf-8"
            )
        )
        assert workspace["deterministic_evidence_plan"]["group_count"] > 0
    for evidence in snapshot["evidence"]:
        text = store.artifact_path(evidence["text_artifact_id"]).read_text(encoding="utf-8")
        assert evidence["content"] == text[evidence["start_offset"] : evidence["end_offset"]]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_explicit_architect_comparison_uses_compact_tool_surface(tmp_path):
    driver = GatewayDrivingRuntime(review_profile="architect")
    runtime = PipelineRuntime(_settings(tmp_path, t5_processing_mode="agent_architect"), driver)
    run_id = "run_t5_original_agent"
    inputs = [_stage(runtime, run_id, "guide.html", HTML, 1)]
    store = _initialize(runtime, run_id, inputs)

    snapshot = await runtime.execute(run_id, _target(), inputs, _model())
    runtime.close()

    assert Settings().t5_processing_mode == "agent_architect"
    assert store.read_state()["t5_processing_mode"] == "agent_architect"
    assert store.read_state()["phase"] == "corpus_frozen"
    assert snapshot["evidence_count"] > 0
    assert len(driver.invocations) == 1
    tools = set(driver.invocations[0].allowed_tools)
    assert tools == ToolExecutor.ARCHITECT_REVIEW_TOOLS
    assert len(tools) == 4
    assert "read_artifact" not in tools
    assert "submit_source_structure" not in tools
    assert "html_list_navigation" not in tools
    assert "curate_evidence" not in tools
    assert "html_inventory" not in tools
    assert "pdf_list_structure" not in tools
    assert "search_source" not in tools
    assert "read_evidence_node" in tools
    calls = driver.tool_calls[driver.invocations[0].session_id]
    assert calls == [
        "inspect_source",
        "get_root_structure",
        "submit_evidence_revision",
    ]
    assert snapshot["evidence_trees"]
    attempt = store.list_attempts()[0]
    capsule = store.read_json_artifact(attempt.capsule_artifact_id)
    assert capsule["prompt_version"] == SOURCE_REVIEW_PROMPT_VERSION


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_architect_uses_layout_headings_and_ignores_native_outline(tmp_path):
    driver = GatewayDrivingRuntime(review_profile="architect")
    runtime = PipelineRuntime(_settings(tmp_path, t5_processing_mode="agent_architect"), driver)
    run_id = "run_t5_pdf_outline"
    inputs = [_stage(runtime, run_id, "outlined.pdf", _outlined_pdf_bytes(), 1)]
    store = _initialize(runtime, run_id, inputs)

    snapshot = await runtime.execute(run_id, _target(), inputs, _model())
    runtime.close()

    assert store.read_state()["phase"] == "corpus_frozen"
    assert snapshot["evidence_trees"]
    invocation = driver.invocations[0]
    assert set(invocation.allowed_tools) == ToolExecutor.ARCHITECT_PDF_REVIEW_TOOLS
    assert driver.tool_calls[invocation.session_id] == [
        "inspect_source",
        "get_pdf_document_structure",
        "submit_evidence_revision",
    ]
    attempt = store.list_attempts()[0]
    capsule = store.read_json_artifact(attempt.capsule_artifact_id)
    assert capsule["prompt_version"] == PDF_SOURCE_REVIEW_PROMPT_VERSION

    source = snapshot["source_results"][0]
    workspace = json.loads(
        (
            store.root / "sources" / source["source_id"] / "workspace.json"
        ).read_text(encoding="utf-8")
    )
    structure = store.read_json_artifact(
        workspace["pdf_document_structure_artifact_id"]
    )
    assert structure["authored_outline_source"] == "layout_headings"
    assert structure["native_bookmarks"] == []
    assert structure["bookmark_policy"] == "ignored"
    assert [item["title"] for item in structure["layout_headings"]] == [
        "Chapter 1 Commands",
        "and",
        "min",
    ]
    units = {item["heading"]: item for item in structure["semantic_units"]}
    assert set(units) == {"Chapter 1 Commands", "and", "min"}
    assert units["and"]["parent_section_part_id"] == units["Chapter 1 Commands"]["part_id"]
    assert units["min"]["parent_section_part_id"] == units["Chapter 1 Commands"]["part_id"]


def test_empty_pdf_preserves_explicit_blank_record_before_agent(tmp_path):
    driver = GatewayDrivingRuntime(review_profile="architect")
    settings = _settings(tmp_path, t5_processing_mode="agent_architect")
    runtime = PipelineRuntime(settings, driver)
    run_id = "run_t5_textless_pdf"
    inputs = [_stage(runtime, run_id, "scan.pdf", _textless_pdf_bytes(), 1)]
    store = _initialize(runtime, run_id, inputs)
    outcome = AcquisitionService(store, settings).acquire(inputs, _target())

    prepared = normalize_source(
        store, settings, AcquisitionService(store, settings), _target(), outcome.manifests[0],
    )
    assert prepared.workspace.evidence_skeleton_plan is not None
    assert prepared.workspace.gaps
    assert prepared.workspace.pdf_document_structure['text_layer']['supported'] is False
    assert store.artifact_path(outcome.manifests[0].raw_artifact_id).read_bytes()
    runtime.close()
    assert driver.invocations == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_t5_runtime_failure_retains_python_candidate_as_degraded_result(tmp_path):
    driver = FailingRuntime()
    runtime = PipelineRuntime(_settings(tmp_path), driver)
    run_id = "run_t5_runtime_fallback"
    inputs = [_stage(runtime, run_id, "guide.html", HTML, 1)]
    store = _initialize(runtime, run_id, inputs)

    snapshot = await runtime.execute(run_id, _target(), inputs, _model())
    runtime.close()

    assert store.read_state()["phase"] == "corpus_frozen"
    assert snapshot["evidence_count"] > 0
    result = snapshot["source_results"][0]
    assert result["status"] == "degraded"
    assert result["evidence_tree"] is not None
    assert any("DSH Evidence Architect did not complete" in gap for gap in result["gaps"])
    attempt = store.list_attempts()[0]
    assert attempt.status == "failed"
    assert attempt.error_category == "runtime_error"
    assert attempt.child_session_id in driver.closed_sessions
    assert "source_review_runtime_fallback_committed" in store.audit_path.read_text(
        encoding="utf-8"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_source_review_uses_fixed_modality_budgets(tmp_path):
    driver = GatewayDrivingRuntime()
    runtime = PipelineRuntime(_settings(tmp_path), driver)
    run_id = "run_t5_fixed_budgets"
    inputs = [
        _stage(runtime, run_id, "guide.html", HTML, 1),
        _stage(runtime, run_id, "manual.pdf", _pdf_bytes(), 2),
    ]
    store = _initialize(runtime, run_id, inputs)
    model = _model()
    try:
        snapshot = await runtime.execute(run_id, _target(), inputs, model)
    finally:
        runtime.close()

    assert {item["modality"] for item in snapshot["source_results"]} == {"html", "pdf"}
    modalities = {
        attempt.attempt_id: store.read_json_artifact(attempt.capsule_artifact_id)["scope"]["modality"]
        for attempt in store.list_attempts()
    }
    events, _ = store.read_events(limit=2000)
    budgets = {
        modalities[event["payload"]["attempt_id"]]: event["payload"]["max_tool_calls"]
        for event in events
        if event["event_type"] == "capability_tool_budget_configured"
    }
    assert budgets == {"html": 3, "pdf": 80}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_t5_mixed_pdf_html_freezes_exact_corpus_with_isolated_parallel_children(tmp_path):
    driver = GatewayDrivingRuntime(pause_seconds=0.05)
    runtime = PipelineRuntime(_settings(tmp_path), driver)
    run_id = "run_t5_mixed"
    inputs = [
        _stage(runtime, run_id, "guide.html", HTML, 1),
        _stage(runtime, run_id, "manual.pdf", _pdf_bytes(), 2),
    ]
    store = _initialize(runtime, run_id, inputs)

    snapshot = await runtime.execute(run_id, _target(), inputs, _model())
    runtime.close()

    state = store.read_state()
    assert state["phase"] == "corpus_frozen"
    assert state["revision"] == 4
    assert state["t5_stage_complete"] is True
    assert snapshot["evidence_count"] >= 3
    assert {item["modality"] for item in snapshot["source_results"]} == {"html", "pdf"}
    assert driver.max_active == 2
    html_source = next(
        item for item in snapshot["source_results"] if item["modality"] == "html"
    )
    html_text = "\n".join(item["content"] for item in html_source["evidence"])
    assert "examplebio align sample.fasta" in html_text
    assert "--fast" in html_text and "Quick mode" in html_text
    html_parts = json.loads(
        (store.root / "sources" / html_source["source_id"] / "parts.json").read_text(
            encoding="utf-8"
        )
    )
    html_page = next(item for item in html_parts if item["kind"] == "html_page")
    assert store.artifact_record(html_page["metadata"]["semantic_dom_artifact_id"]).kind == "html_semantic_dom"
    assert store.artifact_record(html_page["metadata"]["markdown_artifact_id"]).kind == "html_markdown_projection"
    assert store.artifact_record(html_page["metadata"]["canonical_page_artifact_id"]).kind == "html_canonical_page"

    sessions = store.list_sessions()
    root = [item for item in sessions if item.session_kind == "root"]
    children = [item for item in sessions if item.session_kind == "child"]
    assert len(root) == 1
    assert len(children) == 2
    assert {item.parent_root_session_id for item in children} == {root[0].session_id}
    assert all(item.stage == "source_review" for item in children)
    assert all(item.session_id in driver.closed_sessions for item in children)

    for invocation in driver.invocations:
        tools = set(invocation.allowed_tools)
        assert invocation.role == "source_review"
        assert "submit_task_batch" not in tools
        expected_tools = (
            ToolExecutor.ARCHITECT_PDF_REVIEW_TOOLS
            if "get_pdf_document_structure" in tools
            else ToolExecutor.ARCHITECT_REVIEW_TOOLS
        )
        assert tools == expected_tools
        assert "pdf_inventory" not in tools
        assert "html_inventory" not in tools
        assert "submit_source_structure" not in tools
        assert driver.tool_calls[invocation.session_id][-1] == "submit_evidence_revision"
        assert driver.tool_calls[invocation.session_id][1] == (
            "get_pdf_document_structure"
            if "get_pdf_document_structure" in tools
            else "get_root_structure"
        )

    assert len(snapshot["evidence_trees"]) == 2

    for evidence in snapshot["evidence"]:
        if not evidence["content"]:
            continue
        text = store.artifact_path(evidence["text_artifact_id"]).read_text(encoding="utf-8")
        assert evidence["content"] == text[evidence["start_offset"] : evidence["end_offset"]]

    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in store.root.rglob("*")
        if path.is_file()
    )
    assert "t5-api-key-must-never-persist" not in persisted
    for invocation in driver.invocations:
        token = str(invocation.metadata["capability"]["capability_token"])
        assert token not in persisted


@pytest.mark.integration
@pytest.mark.asyncio
async def test_t5_grounded_credential_shaped_urls_survive_immutable_batch_roundtrip(tmp_path):
    driver = GatewayDrivingRuntime()
    runtime = PipelineRuntime(_settings(tmp_path), driver)
    run_id = "run_t5_grounded_url"
    grounded_url = "https://example-user:example-password@example.invalid/manual"
    html = (
        "<!doctype html><html><body><h1>Proxy configuration</h1>"
        f"<pre>curl {grounded_url}</pre></body></html>"
    ).encode("utf-8")
    inputs = [_stage(runtime, run_id, "proxy-guide.html", html, 1)]
    store = _initialize(runtime, run_id, inputs)

    snapshot = await runtime.execute(run_id, _target(), inputs, _model())
    runtime.close()

    assert store.read_state()["phase"] == "corpus_frozen"
    assert grounded_url in "\n".join(item["content"] for item in snapshot["evidence"])
    batch = store.read_json_artifact(
        store.read_state()["validated_source_batch_artifact_id"]
    )
    batch_text = "\n".join(
        evidence["content"]
        for result in batch["source_results"]
        for evidence in result["evidence"]
    )
    assert grounded_url in batch_text
    assert "[REDACTED]" not in batch_text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_t5_deduplicates_identical_html_before_child_creation(tmp_path):
    driver = GatewayDrivingRuntime()
    runtime = PipelineRuntime(_settings(tmp_path), driver)
    run_id = "run_t5_dedup"
    inputs = [
        _stage(runtime, run_id, "one.html", HTML, 1),
        _stage(runtime, run_id, "two.html", HTML, 2),
    ]
    store = _initialize(runtime, run_id, inputs)

    snapshot = await runtime.execute(run_id, _target(), inputs, _model())
    runtime.close()

    assert len(driver.invocations) == 1
    assert len([item for item in store.list_sessions() if item.session_kind == "child"]) == 1
    assert len(snapshot["source_results"]) == 1
    manifest = snapshot["source_results"][0]["manifest"]
    assert len(manifest["aliases"]) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_t5_cancel_closes_child_revokes_capability_and_commits_cancel_receipt(tmp_path):
    driver = BlockingRuntime()
    runtime = PipelineRuntime(_settings(tmp_path), driver)
    run_id = "run_t5_cancel"
    inputs = [_stage(runtime, run_id, "guide.html", HTML, 1)]
    store = _initialize(runtime, run_id, inputs)

    runtime.start(run_id, _target(), inputs, _model())
    await asyncio.wait_for(driver.entered.wait(), timeout=5)
    response = runtime.cancel(run_id)
    assert response["cancel_requested"] is True
    for _ in range(100):
        if store.read_state().get("phase") == "cancelled":
            break
        await asyncio.sleep(0.02)
    runtime.close()

    assert store.read_state()["phase"] == "cancelled"
    attempt = store.list_attempts()[0]
    assert attempt.status == "cancelled"
    assert attempt.child_session_id in driver.closed_sessions
    capability = json.loads(next((store.root / "capabilities").glob("*.json")).read_text())
    assert capability["revoked_at"] is not None
    assert store.list_receipt_commits()[-1]["receipt"]["status"] == "cancelled"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_t5_abandoned_source_attempt_is_failed_and_retried_with_new_child(tmp_path):
    driver = GatewayDrivingRuntime()
    settings = _settings(tmp_path)
    runtime = PipelineRuntime(settings, driver)
    run_id = "run_t5_recovery"
    inputs = [_stage(runtime, run_id, "guide.html", HTML, 1)]
    store = _initialize(runtime, run_id, inputs)
    outcome = AcquisitionService(store, settings).acquire(inputs, _target())
    prepared = normalize_source(
        store,
        settings,
        AcquisitionService(store, settings),
        _target(),
        outcome.manifests[0],
    )
    abandoned, _, _ = store.begin_child_attempt(
        stage="source_review",
        input_artifact_ids=(
            prepared.manifest.raw_artifact_id,
            prepared.normalization_artifact_id,
        ),
        scope={"source_id": prepared.manifest.source_id, "modality": "html"},
        allowed_tools=tuple(sorted(prepared.executor.allowed_tools)),
        output_contract=SOURCE_RESULT_CONTRACT,
        prompt_version=SOURCE_REVIEW_PROMPT_VERSION,
    )
    store.start_attempt(abandoned.attempt_id)
    service = SourceReviewService(
        store=store,
        runtime=driver,
        gateway=runtime.capability_gateway,
        gateway_server=runtime.capability_server,
        model_config=_model(),
    )

    result = await service.review(prepared)
    runtime.close()

    attempts = store.list_attempts()
    assert result.status == "succeeded"
    assert len(attempts) == 2
    assert attempts[0].status == "failed"
    assert attempts[0].error_category == "process_recovery"
    assert attempts[1].status == "succeeded"
    assert attempts[1].retry_of_attempt_id == attempts[0].attempt_id
    assert attempts[1].child_session_id != attempts[0].child_session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_t5_validator_quarantines_one_tampered_evidence(tmp_path):
    driver = GatewayDrivingRuntime()
    runtime = PipelineRuntime(_settings(tmp_path), driver)
    run_id = "run_t5_tamper"
    inputs = [_stage(runtime, run_id, "manual.pdf", _pdf_bytes(), 1)]
    store = _initialize(runtime, run_id, inputs)
    snapshot = await runtime.execute(run_id, _target(), inputs, _model())
    runtime.close()

    payload = copy.deepcopy(snapshot["source_results"][0])
    assert len(payload["evidence"]) > 1
    payload["evidence"][0]["content"] += " invented"
    tampered = SourceResult.model_validate(payload)
    validated, summary = validate_source_results(store, [tampered])

    assert validated[0].status == "degraded"
    assert len(validated[0].evidence) == len(payload["evidence"]) - 1
    assert "evidence_validation_rejected:1" in validated[0].gaps
    assert summary[0]["rejected_evidence_count"] == 1
    assert any("not an exact normalized text excerpt" in item for item in summary[0]["errors"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_t5_validator_fails_source_when_every_evidence_is_tampered(tmp_path):
    driver = GatewayDrivingRuntime()
    runtime = PipelineRuntime(_settings(tmp_path), driver)
    run_id = "run_t5_all_tampered"
    inputs = [_stage(runtime, run_id, "guide.html", HTML, 1)]
    store = _initialize(runtime, run_id, inputs)
    snapshot = await runtime.execute(run_id, _target(), inputs, _model())
    runtime.close()

    payload = copy.deepcopy(snapshot["source_results"][0])
    for evidence in payload["evidence"]:
        evidence["content"] += " invented"
    tampered = SourceResult.model_validate(payload)
    validated, summary = validate_source_results(store, [tampered])

    assert validated[0].status == "failed"
    assert validated[0].evidence == []
    assert "branch_validation_failed" in validated[0].gaps
    assert summary[0]["rejected_evidence_count"] == len(payload["evidence"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_t5_model_cannot_finish_without_structured_source_submission(tmp_path):
    driver = SilentRuntime()
    runtime = PipelineRuntime(_settings(tmp_path), driver)
    run_id = "run_t5_no_submit"
    inputs = [_stage(runtime, run_id, "guide.html", HTML, 1)]
    store = _initialize(runtime, run_id, inputs)

    with pytest.raises(SourceReviewProtocolError, match="did not submit a result"):
        await runtime.execute(run_id, _target(), inputs, _model())
    runtime.close()

    assert len(driver.invocations) == 2
    assert len({item.session_id for item in driver.invocations}) == 1
    attempt = store.list_attempts()[0]
    assert attempt.status == "failed"
    assert attempt.error_category == "source_review_protocol_error"
    assert attempt.child_session_id in driver.closed_sessions
    assert store.read_state()["phase"] == "failed"
    capability = json.loads(next((store.root / "capabilities").glob("*.json")).read_text())
    assert capability["revoked_at"] is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_t5_crash_after_review_receipt_recovers_from_immutable_batch(tmp_path, monkeypatch):
    class SimulatedProcessCrash(BaseException):
        pass

    driver = GatewayDrivingRuntime()
    runtime = PipelineRuntime(_settings(tmp_path), driver)
    run_id = "run_t5_receipt_recovery"
    inputs = [_stage(runtime, run_id, "guide.html", HTML, 1)]
    store = _initialize(runtime, run_id, inputs)
    real_builder = pipeline_module.build_corpus_snapshot

    def crash_before_freeze(*_args, **_kwargs):
        raise SimulatedProcessCrash()

    monkeypatch.setattr(pipeline_module, "build_corpus_snapshot", crash_before_freeze)
    with pytest.raises(SimulatedProcessCrash):
        await runtime.execute(run_id, _target(), inputs, _model())
    assert store.read_state()["phase"] == "source_review"
    assert len(driver.invocations) == 1

    # Simulate loss of the convenience projection and corruption of the mutable
    # source projection. Recovery must use the immutable Artifact accepted by Receipt.
    state_path = store.root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("validated_source_batch_artifact_id", None)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    mutable_result = next((store.root / "sources").glob("*/validated_result.json"))
    mutable_payload = json.loads(mutable_result.read_text(encoding="utf-8"))
    mutable_payload["evidence"][0]["content"] += " invented mutable projection"
    mutable_result.write_text(json.dumps(mutable_payload), encoding="utf-8")

    monkeypatch.setattr(pipeline_module, "build_corpus_snapshot", real_builder)
    snapshot = await runtime.execute(run_id, _target(), inputs, _model())
    runtime.close()

    assert store.read_state()["phase"] == "corpus_frozen"
    assert len(driver.invocations) == 1
    assert all("invented mutable projection" not in item["content"] for item in snapshot["evidence"])
