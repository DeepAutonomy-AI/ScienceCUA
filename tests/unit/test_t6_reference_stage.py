from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from app.capabilities import CapabilityGateway, CapabilityInvocation
from app.config import Settings
from app.knowledge import (
    KnowledgeWorkspace,
    REFERENCE_CATALOG_SAMPLE_SIZE,
    REFERENCE_ORDINARY_CONTEXT_CHARS,
    REFERENCE_PARTITION_REVIEW_CHARS,
    _build_reference_catalogs,
    _inline_catalog_quality_profile,
    _knowledge_tool_catalog,
    _reference_catalog_automation_profile,
    _representative_reference_excerpt,
)
from app.models import (
    CorpusSnapshot,
    EvidenceCitation,
    EvidenceExcerpt,
    ModelRuntimeConfig,
    ReferenceDraft,
    ReferenceEntry,
    SourceResult,
    StagedInput,
    TargetProfile,
)
from app.orchestration.pipeline import PipelineRuntime
from app.orchestration.reference_stage import REFERENCE_BUILD_PROMPT
from app.runtime import AgentInvocation, AgentRuntime, AgentTurnResult, AgentRuntimeBusinessCompleted
from app.storage import RunStore


HTML = b"""<!doctype html><html><body>
<h1>ExampleCalc Manual</h1>
<h2>Keyboard shortcuts</h2>
<table><tr><th>Key</th><th>Action</th></tr>
<tr><td>Ctrl+O</td><td>Open a document.</td></tr>
<tr><td>Ctrl+S</td><td>Save the current document.</td></tr>
<tr><td>Ctrl+P</td><td>Print the current document.</td></tr>
<tr><td>Ctrl+Q</td><td>Quit the application.</td></tr></table>
<h2>About</h2><p>ExampleCalc is a scientific calculator.</p>
<h2>Acknowledgements</h2><p>Written by the ExampleCalc documentation team.</p>
</body></html>"""


class ReferenceDrivingRuntime(AgentRuntime):
    """Deterministic DSH substitute that uses the real capability boundary."""

    def __init__(
        self,
        gateway: CapabilityGateway,
    ) -> None:
        self.gateway = gateway
        self.invocations: list[AgentInvocation] = []
        self.closed_sessions: list[str] = []
        self.cancelled_sessions: list[str] = []
        self.call_index = 0

    def _invoke(
        self, invocation: AgentInvocation, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        capability = dict(invocation.metadata["capability"])
        context = dict(capability["context"])
        self.call_index += 1
        return self.gateway.invoke(
            str(capability["capability_token"]),
            CapabilityInvocation(
                **context,
                tool_name=name,
                arguments=arguments,
                call_id=f"test.{self.call_index}",
            ),
        )

    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        assert callable(invocation.metadata.get("runtime_event_sink"))
        self.invocations.append(invocation)
        prompt = str(invocation.prompt)
        if invocation.role == "source_review":
            self._invoke(invocation, "inspect_source", {})
            structure = self._invoke(invocation, "get_root_structure", {})
            self._invoke(
                invocation,
                "submit_evidence_revision",
                {
                    "base_plan_artifact_id": structure["base_plan_artifact_id"],
                    "operations": [],
                    "summary": "The authored fixture hierarchy is coherent.",
                },
            )
        elif invocation.role == "reference_build":
            payload = json.loads(prompt.rsplit("\n\n", 1)[-1])
            manifest = payload["reference_manifest"]
            catalog_decisions = [
                {
                    "catalog_id": item["catalog_id"],
                    "disposition": "preserve",
                    "title": item["suggested_title"],
                    "description": "ExampleCalc control mappings. Consult to check documented controls and actions.",
                }
                for item in manifest["catalog_candidates"]
            ]
            ordinary_decisions = [
                {
                    "candidate_id": candidate["candidate_id"],
                    "disposition": "exclude",
                }
                for candidate in manifest["ordinary_candidates"]
            ]
            if not catalog_decisions and ordinary_decisions:
                ordinary_decisions[0].update(
                    {
                        "disposition": "preserve",
                        "title": "ExampleCalc reusable controls",
                        "description": "ExampleCalc controls. Consult when operating the documented calculator interface.",
                        "category": "reference",
                        "injection_policy": "on_demand",
                        "usage_tags": ["ui_reference"],
                    }
                )
            preserved_ids = {
                item["catalog_id"] for item in catalog_decisions
            } | {
                item["candidate_id"]
                for item in ordinary_decisions
                if item["disposition"] == "preserve"
            }
            partition_decisions = [
                {
                    "subject_id": item["subject_id"],
                    "decision": "keep_whole",
                    "groups": [],
                }
                for item in manifest["partition_review_candidates"]
                if item["subject_id"] in preserved_ids
            ]
            self._invoke(
                invocation,
                "submit_reference_review",
                {
                    "catalog_decisions": catalog_decisions,
                    "ordinary_decisions": ordinary_decisions,
                    "partition_decisions": partition_decisions,
                    "status": "succeeded",
                    "summary": "Every prepared Reference candidate received an Agent decision.",
                },
            )
        return AgentTurnResult(
            session_id=invocation.session_id,
            final_response="controlled-reference-step-complete",
            finish_reason="completed",
            event_count=1,
            usage={"input_tokens": 10, "output_tokens": 2},
        )

    async def cancel(self, session_id: str) -> bool:
        self.cancelled_sessions.append(session_id)
        return True

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)


class ReferenceCompletionRuntime(ReferenceDrivingRuntime):
    def __init__(self, gateway, store, mode):
        super().__init__(gateway)
        self.store, self.mode = store, mode
        self.stopped = asyncio.Event()

    def _invoke(self, invocation, name, arguments):
        if name == "submit_reference_review":
            if self.mode == "rejected":
                arguments = {**arguments, "catalog_decisions": [], "ordinary_decisions": []}
            elif self.mode == "degraded":
                arguments = {**arguments, "status": "degraded", "gaps": ["Source coverage is partial."]}
        return super()._invoke(invocation, name, arguments)

    async def run(self, invocation):
        if invocation.role != "reference_build":
            return await super().run(invocation)
        sink = invocation.metadata["runtime_event_sink"]
        sink("model_waiting", {"status": "waiting_for_model", "elapsed_seconds": 15,
                               "seconds_since_event": 15})
        sink("model_retry_scheduled", {"status": "retry_wait", "retry_number": 1,
                                       "max_retries": 5, "delay_seconds": 1,
                                       "message": "HTTP 503 service unavailable"})
        if self.mode == "error_before":
            raise RuntimeError("Provider failed before Reference review")
        result = await super().run(invocation)
        if self.mode == "error_after":
            raise RuntimeError("Provider failed after accepted Reference review")
        if self.mode == "linger":
            await self.stopped.wait()
            raise AgentRuntimeBusinessCompleted("Reference model stopped after review")
        return result

    async def stop_completed_session(self, session_id):
        events = self.store.read_events(limit=1000)[0]
        accepted = next(e for e in events if e["event_type"] == "reference_business_terminal_accepted")
        terminal = self.store.read_json_artifact(accepted["payload"]["terminal_artifact_id"])
        assert terminal["stage_result"]["status"] in {"succeeded", "degraded"}
        assert len(terminal["references"]) == 1
        self.stopped.set()
        return True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("completion_mode", ["normal", "linger", "error_after", "degraded", "rejected", "error_before"])
async def test_reference_stage_selects_regions_preserves_sourceparts_and_pauses(
    tmp_path: Path, completion_mode: str,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",

        t5_processing_mode="agent_architect",
        t6_reference_enabled=True,
        t6_task_enabled=False,
        max_html_pages=0,
    )
    gateway = CapabilityGateway()
    driver = ReferenceDrivingRuntime(gateway)
    runtime = PipelineRuntime(settings, driver, capability_gateway=gateway)
    run_id = "run_t6_reference_tree"
    store = runtime.create_store(run_id)
    driver = ReferenceCompletionRuntime(gateway, store, completion_mode)
    runtime.agent_runtime = driver
    path = store.stage_upload("inp_reference", "guide.html", HTML)
    staged = StagedInput(
        input_id="inp_reference",
        kind="uploaded_file",
        display_name="guide.html",
        user_declared_version="1.0",
        staged_path=str(path),
        declared_media_type="text/html",
    )
    target = TargetProfile(
        software_name="ExampleCalc",
        target_version="1.0",
        platform="General",
        ui_locale="English",
    )
    model = ModelRuntimeConfig(
        api_protocol="responses",
        base_url="https://model.example.invalid/v1",
        model="gpt-test",
        api_key="test-key-never-persisted",
        reasoning_effort="medium",
    )
    store.initialize(
        {
            "run_id": run_id,
            "target_profile": target.model_dump(mode="json"),
            "inputs": [staged.model_dump(mode="json")],
            "model": model.public_dict(),
        }
    )

    if completion_mode in {"rejected", "error_before"}:
        with pytest.raises(RuntimeError, match="before Reference review|terminal stage result"):
            await asyncio.wait_for(runtime.execute(run_id, target, [staged], model), timeout=10)
        result = None
    else:
        result = await asyncio.wait_for(runtime.execute(run_id, target, [staged], model), timeout=10)
    runtime.close()

    state = store.read_state()
    events = store.read_events(limit=1000)[0]
    reference_attempt = next(a for a in store.list_attempts() if a.stage == "reference_build")
    if completion_mode in {"rejected", "error_before"}:
        assert state["phase"] == "failed"
        assert reference_attempt.status == "failed"
        assert not any(e["event_type"] == "reference_business_terminal_accepted" for e in events)
        assert not driver.stopped.is_set()
        return
    assert state["phase"] == "references_ready"
    assert state["status"] == "paused"
    assert result["outcome"] == "references_ready"
    assert len(result["references"]) == 1
    assert result["reference_validation"][0]["accepted"] is True
    assert reference_attempt.status == "succeeded"
    accepted = next(e for e in events if e["event_type"] == "reference_business_terminal_accepted")
    terminal = store.read_json_artifact(accepted["payload"]["terminal_artifact_id"])
    assert len(terminal["references"]) == 1
    assert any(e["event_type"] == "reference_business_completion_finished" for e in events)
    assert any(e["event_type"] == "model_waiting" and e["agent"] == "reference_build" for e in events)
    assert any(e["event_type"] == "model_retry_scheduled" and e["level"] == "warning" for e in events)
    if completion_mode == "linger":
        assert driver.stopped.is_set()
    if completion_mode == "degraded":
        assert terminal["stage_result"]["status"] == "degraded"

    reference_invocations = [
        item for item in driver.invocations if item.role == "reference_build"
    ]
    assert len(reference_invocations) == 1
    assert "REFERENCE CANDIDATE REVIEW" in str(reference_invocations[0].prompt)
    assert "REFERENCE READING PLAN" not in str(reference_invocations[0].prompt)
    assert "REFERENCE READING UNIT" not in str(reference_invocations[0].prompt)
    assert len({item.session_id for item in reference_invocations}) == 1
    reference_session_id = reference_invocations[0].session_id
    assert driver.closed_sessions.count(reference_session_id) == 1

    candidate_artifact = store.read_json_artifact(
        state["reference_candidate_artifact_id"]
    )
    assert (
        candidate_artifact["selection_mode"]
        == "python_candidates_agent_review"
    )
    assert candidate_artifact["candidate_inventory"]["candidate_count"] >= 1
    assert candidate_artifact["candidate_inventory"]["detected_catalog_count"] == 1
    assert "reading_plan" not in candidate_artifact
    assert candidate_artifact["catalog_coverage"]["unreviewed_catalog_ids"] == []
    referenced_parts = {
        citation["evidence_id"]
        for reference in result["references"]
        for citation in reference["citations"]
    }
    assert all(
        entry["source_evidence_id"] in referenced_parts
        for entry in result["references"][0]["entries"]
    )




@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_count", [8, 67])
async def test_reference_single_session_reviews_every_candidate(tmp_path: Path, candidate_count: int) -> None:
    """Small and large inventories must reach one reviewer without losing candidates."""
    from types import SimpleNamespace

    from app.orchestration.reference_stage import ReferenceBuildService
    from test_reference_review_submission import _workspace

    workspace = _workspace(tmp_path, catalog_count=0, ordinary_count=candidate_count, capacity=24)
    store = workspace.store
    model = ModelRuntimeConfig()
    store.initialize({
        "target_profile": workspace.target_profile.model_dump(mode="json"),
        "model": model.public_dict(), "inputs": [],
    })
    corpus_artifact = store.put_json_artifact(workspace.corpus.model_dump(mode="json"), kind="corpus_snapshot")
    gateway = CapabilityGateway()

    class WholeInventoryRuntime(ReferenceDrivingRuntime):
        async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
            manifest = json.loads(str(invocation.prompt).rsplit("\n\n", 1)[-1])["reference_manifest"]
            assert len(manifest["ordinary_candidates"]) == candidate_count
            return await super().run(invocation)

    driver = WholeInventoryRuntime(gateway)
    service = ReferenceBuildService(
        store=store, settings=workspace.settings, runtime=driver, gateway=gateway,
        gateway_server=SimpleNamespace(endpoint_url="http://unused.invalid/invoke"), model_config=model,
    )
    result = await service.build(
        target_profile=workspace.target_profile, corpus=workspace.corpus,
        corpus_artifact_id=corpus_artifact.artifact_id,
    )

    assert len(driver.invocations) == 1
    assert driver.closed_sessions == [driver.invocations[0].session_id]
    attempts = store.list_attempts()
    assert len(attempts) == 1 and attempts[0].status == "succeeded"
    assert len(result.references) == 1
    assert all(item.accepted for item in result.validation)
    artifact = store.read_json_artifact(result.candidate_artifact_id)
    assert artifact["selection_mode"] == "python_candidates_agent_review"
    assert len(artifact["stage_result"]["ordinary_candidate_review"]) == candidate_count
    assert result.references[0].entries[0].content == workspace.corpus.evidence[0].content


def _reference_catalog_evidence(
    ordinal: int,
    *,
    title: str,
    content: str,
    url: str,
    section_path: str | None = None,
    capability_branch: str | None = None,
) -> EvidenceExcerpt:
    locator: dict[str, Any] = {
        "heading": title,
        "section_path": section_path or title,
        "url": url,
    }
    if capability_branch:
        locator["capability_branch"] = capability_branch
    return EvidenceExcerpt(
        evidence_id=f"evi_catalog_normalization_{ordinal}",
        source_id="src_catalog_normalization",
        source_part_id=f"part_catalog_normalization_{ordinal}",
        text_artifact_id=f"art_catalog_normalization_{ordinal}",
        start_offset=0,
        end_offset=len(content),
        content_sha256=f"sha_catalog_normalization_{ordinal}",
        kind="html_section",
        title=title,
        content=content,
        locator=locator,
        extraction_method="test",
    )


def test_reference_catalogs_deduplicate_merge_chunks_and_filter_metadata() -> None:
    general = "\n".join(
        f"g.command{index}\nDocumented general command {index}." for index in range(4)
    )
    raster_first = "\n".join(
        f"r.command{index}\nDocumented raster command {index}." for index in range(4)
    )
    raster_second = "\n".join(
        f"r.command{index}\nDocumented raster command {index}." for index in range(4, 8)
    )
    keyword_first = (
        "projection\ng.proj\n\nraster\nr.mapcalc, r.info\n\n"
        "vector\nv.buffer\n\ndatabase\ndb.select"
    )
    keyword_second = (
        "display\nd.rast\n\nimagery\ni.albedo\n\n"
        "temporal\nt.create\n\nstatistics\nr.univar, v.univar"
    )
    evidence = [
        _reference_catalog_evidence(
            1,
            title="General commands:",
            content=general,
            url="https://docs.example/manuals/general.html",
        ),
        _reference_catalog_evidence(
            2,
            title="General commands (g.*)",
            content=general,
            url="https://docs.example/manuals/full_index.html",
        ),
        _reference_catalog_evidence(
            3,
            title="Raster commands (1)",
            content=raster_first,
            url="https://docs.example/manuals/raster.html",
            section_path="Reference Manual > Raster commands",
        ),
        _reference_catalog_evidence(
            4,
            title="Raster commands (2)",
            content=raster_second,
            url="https://docs.example/manuals/raster.html",
            section_path="Reference Manual > Raster commands",
        ),
        _reference_catalog_evidence(
            5,
            title="Keywords - module index (1)",
            content=keyword_first,
            url="https://docs.example/manuals/keywords.html",
            section_path="Keywords - module index",
        ),
        _reference_catalog_evidence(
            6,
            title="Keywords - module index (2)",
            content=keyword_second,
            url="https://docs.example/manuals/keywords.html",
            section_path="Keywords - module index",
        ),
    ]
    for ordinal in range(7, 11):
        evidence.append(
            _reference_catalog_evidence(
                ordinal,
                title="SOURCE CODE",
                content=(
                    f"Available at: component {ordinal} source code\n"
                    f"Latest change: revision {ordinal}"
                ),
                url=f"https://docs.example/manuals/component-{ordinal}.html",
            )
        )

    catalogs = _build_reference_catalogs("corpus_catalog_normalization", evidence)

    assert len(catalogs) == 3
    by_title = {item["suggested_title"]: item for item in catalogs}
    general_catalog = by_title["General commands: lookup catalog"]
    assert general_catalog["entry_count"] == 4
    assert general_catalog["evidence_ids"] == ["evi_catalog_normalization_1"]
    assert by_title["Raster commands lookup catalog"]["entry_count"] == 8
    keyword_catalog = by_title["Keywords - module index lookup catalog"]
    assert keyword_catalog["entry_count"] == 8
    assert keyword_catalog["catalog_entries"][0]["title"] == "projection"
    assert "g.proj" in keyword_catalog["catalog_entries"][0]["content"]
    assert all("SOURCE CODE" not in item["suggested_title"] for item in catalogs)


def test_exact_authored_inline_catalogs_have_a_high_confidence_review_hint() -> None:
    content = "\n".join(
        f"r.command{index}\nDocumented raster command {index}." for index in range(6)
    )
    evidence = _reference_catalog_evidence(
        1,
        title="Raster commands manual",
        content=content,
        url="https://docs.example/manuals/raster.html",
    )
    catalogs = _build_reference_catalogs("corpus_automatic_catalog", [evidence])

    assert len(catalogs) == 1
    profile = _reference_catalog_automation_profile(
        catalogs[0], {evidence.evidence_id: evidence}
    )
    assert profile["automatic"] is True
    assert profile["confidence"] == 1.0


def test_inline_prose_fragments_are_not_treated_as_reference_catalogs() -> None:
    catalog = {
        "suggested_title": "Narrative section lookup catalog",
        "detail": "inline entry rows",
        "catalog_entries": [
            {"title": "the", "content": "the following section explains the workflow"},
            {"title": "and", "content": "and then applies the selected option"},
            {"title": "from", "content": "from the current document state"},
            {"title": "with", "content": "with an optional configuration value"},
            {"title": "this", "content": "this command updates the active view"},
            {"title": "when", "content": "when the operation has completed"},
        ],
    }

    profile = _inline_catalog_quality_profile(catalog)

    assert profile["obvious_fragment"] is True
    assert "labels_continue_as_prose" in profile["reasons"]


def test_clean_identifier_catalog_has_high_confidence_without_product_specific_markers() -> None:
    content = "\n".join(
        f"get_value_{index}\nReturn documented value {index}." for index in range(8)
    )
    evidence = _reference_catalog_evidence(
        1,
        title="Available operations",
        content=content,
        url="https://docs.example/manuals/operations.html",
    )
    catalogs = _build_reference_catalogs("corpus_identifier_catalog", [evidence])

    assert len(catalogs) == 1
    profile = _reference_catalog_automation_profile(
        catalogs[0], {evidence.evidence_id: evidence}
    )
    assert profile["automatic"] is True
    assert profile["quality"]["structured_label_ratio"] == 1.0


def test_mixed_shortcut_catalog_is_preserved_for_semantic_review() -> None:
    evidence = _reference_catalog_evidence(
        1,
        title="Keyboard command summary",
        content="Exact source content for mixed keyboard rows.",
        url="https://docs.example/manuals/keyboard.html",
    )
    rows = [
        ("Ctrl+O", "Ctrl+O\nOpen the current document."),
        ("Ctrl+S", "Ctrl+S\nSave the current document."),
        ("select", "select the object under the pointer"),
        ("change", "change the current orientation"),
        ("toggle", "toggle display of the active overlay"),
        ("bring", "bring the selected item into view"),
    ]
    catalog = {
        "catalog_id": "rcat_mixed_shortcuts",
        "source_id": evidence.source_id,
        "structure": "inline_entry_catalog",
        "detail": "inline keyboard entries",
        "entry_count": len(rows),
        "common_field_labels": [],
        "suggested_title": "Keyboard command summary lookup catalog",
        "evidence_ids": [evidence.evidence_id],
        "catalog_entries": [
            {
                "title": title,
                "content": content,
                "source_evidence_id": evidence.evidence_id,
            }
            for title, content in rows
        ],
    }
    evidence.content = "\n\n".join(content for _, content in rows)

    quality = _inline_catalog_quality_profile(catalog)
    profile = _reference_catalog_automation_profile(
        catalog, {evidence.evidence_id: evidence}
    )

    assert quality["obvious_fragment"] is False
    assert profile["automatic"] is False
    assert any(
        reason in profile["reasons"]
        for reason in (
            "lookup_semantics_require_review",
            "entry_boundaries_require_review",
        )
    )


def test_repeated_field_catalogs_remain_semantic_exceptions() -> None:
    evidence = _reference_catalog_evidence(
        1,
        title="Documented object 1",
        content="Name: Object one\nValue: 1\nDescription: Example object.",
        url="https://docs.example/manuals/objects.html",
    )
    catalog = {
        "catalog_id": "rcat_repeated_fields",
        "source_id": evidence.source_id,
        "structure": "repeated_field_catalog",
        "detail": "repeated fields: name, value, description",
        "entry_count": 1,
        "common_field_labels": ["name", "value", "description"],
        "suggested_title": "Documented objects lookup catalog",
        "evidence_ids": [evidence.evidence_id],
        "catalog_entries": [],
    }

    profile = _reference_catalog_automation_profile(
        catalog, {evidence.evidence_id: evidence}
    )
    assert profile["automatic"] is False
    assert profile["reasons"] == ["semantic_review_required"]


def test_large_logical_reference_catalog_is_not_rejected_at_512_entries() -> None:
    citation = EvidenceCitation(
        evidence_id="evi_large_catalog",
        quote="Exact source catalog content.",
    )
    entries = [
        ReferenceEntry(
            entry_id=f"rfe_large_{index}",
            title=f"keyword-{index}",
            content=f"keyword-{index}\nr.command{index}",
            source_evidence_id="evi_large_catalog",
            citations=[citation],
        )
        for index in range(685)
    ]

    draft = ReferenceDraft(
        title="Large keyword-to-command index",
        description="Exact source-authored keyword mappings. Consult when identifying a documented command.",
        content="",
        reference_mode="source_catalog",
        source_catalog_id="rcat_large",
        entries=entries,
        citations=[citation],
    )

    assert len(draft.entries) == 685


def test_reference_entry_accepts_a_40000_character_evidence_unit() -> None:
    citation = EvidenceCitation(
        evidence_id="evi_long_reference",
        quote="Exact long source section.",
    )

    entry = ReferenceEntry(
        entry_id="rfe_long_reference",
        title="Long source section",
        content="x" * 40_000,
        source_evidence_id="evi_long_reference",
        citations=[citation],
    )

    assert len(entry.content) == 40_000
    with pytest.raises(ValueError, match="at most 40000 characters"):
        ReferenceEntry(
            entry_id="rfe_too_long_reference",
            title="Too-long source section",
            content="x" * 40_001,
            source_evidence_id="evi_long_reference",
            citations=[citation],
        )


def test_reference_review_contract_requires_candidate_review_and_narrow_know_what() -> None:
    tool = _knowledge_tool_catalog()["submit_reference_review"]["function"]
    schema = tool["parameters"]

    assert "rationale" not in tool["description"]
    assert "rationale" not in REFERENCE_BUILD_PROMPT
    for field in ("catalog_decisions", "ordinary_decisions", "partition_decisions"):
        decision = schema["properties"][field]["items"]
        assert "rationale" not in decision["properties"]
        assert "rationale" not in decision["required"]
        assert decision["additionalProperties"] is False
    coverage = _knowledge_tool_catalog()["review_reference_coverage"]["function"]["parameters"]
    disposition = coverage["properties"]["dispositions"]["items"]
    assert set(disposition["properties"]) == {"catalog_id", "disposition"}
    assert disposition["required"] == ["catalog_id", "disposition"]

    assert "maxItems" not in schema["properties"]["ordinary_decisions"]
    assert "maxItems" not in schema["properties"]["catalog_decisions"]
    assert "At most 6 are supplied" not in REFERENCE_BUILD_PROMPT
    assert "already materialized" not in REFERENCE_BUILD_PROMPT
    assert "keyboard or mouse shortcuts mapped to actions" in REFERENCE_BUILD_PROMPT
    assert "command tables" in REFERENCE_BUILD_PROMPT
    assert "function/API tables" in REFERENCE_BUILD_PROMPT
    assert "syntax, grammar, operator, or valid-expression rules" in REFERENCE_BUILD_PROMPT
    assert "merely because they are stable or potentially reusable" in REFERENCE_BUILD_PROMPT
    assert REFERENCE_ORDINARY_CONTEXT_CHARS == 500
    assert "capped at\n   500 characters" in REFERENCE_BUILD_PROMPT
    assert REFERENCE_PARTITION_REVIEW_CHARS == 10_000
    assert "partition_decisions" in schema["required"]
    assert "10,000-character threshold" in REFERENCE_BUILD_PROMPT


def test_reference_catalog_batch_samples_six_uniform_positions() -> None:
    workspace = KnowledgeWorkspace.__new__(KnowledgeWorkspace)
    workspace.reference_catalogs_by_id = {
        "rcat_test": {
            "catalog_id": "rcat_test",
            "source_id": "src_test",
            "structure": "inline_entry_catalog",
            "detail": "test catalog",
            "entry_count": 74,
            "common_field_labels": [],
            "suggested_title": "Test catalog",
            "evidence_ids": [],
            "catalog_entries": [
                {
                    "source_evidence_id": f"evi_{index}",
                    "title": f"Entry {index}",
                    "content": f"Content {index}",
                    "locator": {"entry_label": f"Entry {index}"},
                }
                for index in range(74)
            ],
        }
    }

    result = workspace.inspect_reference_catalog_batch(["rcat_test"])

    assert REFERENCE_CATALOG_SAMPLE_SIZE == 6
    assert result["entries_per_catalog"] == 6
    assert result["catalogs"][0]["sample_indices"] == [0, 15, 29, 44, 58, 73]
    assert len(result["catalogs"][0]["entries"]) == 6
    schema = _knowledge_tool_catalog()["inspect_reference_catalog_batch"]["function"][
        "parameters"
    ]
    assert schema["properties"]["catalog_ids"]["maxItems"] == 3
    assert schema["properties"]["entries_per_catalog"]["maximum"] == 6


def _oversized_reference_workspace(tmp_path: Path, run_id: str) -> tuple[KnowledgeWorkspace, Any]:
    target = TargetProfile(
        software_name="ExampleCalc",
        target_version="1.0",
        platform="General",
        ui_locale="English",
    )
    evidence = []
    for index in range(3):
        content = f"Semantic section {index}. " + (chr(97 + index) * 3_900)
        evidence.append(
            EvidenceExcerpt(
                evidence_id=f"evi_partition_{index}",
                source_id="src_partition",
                source_part_id=f"part_partition_{index}",
                text_artifact_id=f"art_partition_{index}",
                start_offset=0,
                end_offset=len(content),
                content_sha256=f"sha_partition_{index}",
                kind="html_section",
                title=f"Semantic section {index}",
                content=content,
                locator={"heading": f"Semantic section {index}"},
                extraction_method="test",
            )
        )
    corpus = CorpusSnapshot(
        corpus_snapshot_id=f"corpus_{run_id}",
        run_id=run_id,
        target_profile=target,
        source_results=[
            SourceResult(
                source_id="src_partition",
                display_name="partition.html",
                modality="html",
                status="succeeded",
            )
        ],
        evidence=evidence,
        evidence_count=len(evidence),
        source_part_count=len(evidence),
        source_status_counts={"succeeded": 1},
    )
    workspace = KnowledgeWorkspace(
        store=RunStore(tmp_path, run_id),
        settings=Settings(data_dir=tmp_path, output_dir=tmp_path / "output"),
        target_profile=target,
        stage="reference_build",
        corpus=corpus,
        corpus_artifact_id="art_partition_corpus",
    )
    submitted = workspace.submit_reference_collection(
        evidence_ids=[item.evidence_id for item in evidence],
        title="Oversized semantic reference",
        description="Exact source-preserved sections. Consult when reviewing semantic regions.",
    )
    reference = next(
        item
        for item in workspace.submitted_references
        if item.reference_id == submitted["candidate_id"]
    )
    assert workspace._reference_total_chars(reference) > REFERENCE_PARTITION_REVIEW_CHARS
    return workspace, reference


def test_keep_whole_partition_decision_preserves_reference_byte_for_byte(tmp_path: Path) -> None:
    workspace, reference = _oversized_reference_workspace(tmp_path, "run_partition_keep")
    before = reference.model_dump(mode="json")

    result = workspace._apply_reference_partition_decisions(
        {"subject": reference},
        {
            "subject": {
                "decision": "keep_whole",
                "groups": [],
            }
        },
    )

    assert result == {"reviewed_count": 1, "split_count": 0, "kept_whole_count": 1}
    assert workspace.submitted_references[0].model_dump(mode="json") == before


def test_semantic_partition_preserves_every_entry_and_citation_in_order(tmp_path: Path) -> None:
    workspace, reference = _oversized_reference_workspace(tmp_path, "run_partition_split")
    original_entries = [item.model_dump(mode="json") for item in reference.entries]

    result = workspace._apply_reference_partition_decisions(
        {"subject": reference},
        {
            "subject": {
                "decision": "split",
                "groups": [
                    {
                        "title": "Semantic reference A",
                        "description": "First semantic region. Consult for its source-preserved entries.",
                        "entry_indices": [0],
                    },
                    {
                        "title": "Semantic reference B",
                        "description": "Second semantic region. Consult for its source-preserved entries.",
                        "entry_indices": [1, 2],
                    },
                ],
            }
        },
    )

    assert result == {"reviewed_count": 1, "split_count": 1, "kept_whole_count": 0}
    assert len(workspace.submitted_references) == 2
    assert all(item.description and item.content == "" for item in workspace.submitted_references)
    assert [
        entry.model_dump(mode="json")
        for item in workspace.submitted_references
        for entry in item.entries
    ] == original_entries


def test_ordinary_reference_excerpt_is_full_when_short_and_representative_when_long() -> None:
    short = "Syntax: f(x) -> x + 1. Ctrl+R runs the graph."
    assert _representative_reference_excerpt(short) == short

    long = (
        "BEGINNING overview "
        + ("a" * 260)
        + " Syntax rule: (x,y)->x*y is a valid expression. "
        + ("b" * 260)
        + " Keyboard shortcut: Ctrl+R renders the graph. "
        + ("c" * 260)
        + " ENDING constraints"
    )
    excerpt = _representative_reference_excerpt(long)

    assert len(excerpt) <= REFERENCE_ORDINARY_CONTEXT_CHARS
    assert "BEGINNING" in excerpt
    assert "Syntax rule" in excerpt
    assert "Ctrl+R" in excerpt
    assert "ENDING constraints" in excerpt


@pytest.mark.asyncio
async def test_cancel_dispatches_to_running_child_session(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",

    )
    gateway = CapabilityGateway()
    driver = ReferenceDrivingRuntime(gateway)
    runtime = PipelineRuntime(settings, driver, capability_gateway=gateway)
    run_id = "run_cached_reference_cancel"
    store = runtime.create_store(run_id)
    store.initialize(
        {
            "run_id": run_id,
            "target_profile": {
                "software_name": "ExampleCalc",
                "target_version": "1.0",
                "platform": "General",
                "ui_locale": "English",
            },
            "inputs": [],
            "model": ModelRuntimeConfig().public_dict(),
        }
    )
    attempt, capsule, _ = store.begin_child_attempt(
        stage="reference_build",
        input_artifact_ids=(),
        scope={"execution_mode": "full_pipeline_from_frozen_t5"},
        allowed_tools=(),
        output_contract="reference-stage-result.v2",
        prompt_version="test",
    )
    store.start_attempt(attempt.attempt_id)
    blocker = asyncio.create_task(asyncio.Event().wait())
    runtime._track_task(run_id, blocker)

    response = runtime.cancel(run_id)
    await asyncio.sleep(0)
    await asyncio.gather(blocker, return_exceptions=True)

    assert response["status"] == "cancelling"
    assert capsule.child_session_id in driver.cancelled_sessions
    assert runtime.has_active_task(run_id) is False
    runtime.close()


def test_reference_catalogs_merge_environment_and_sibling_shortcut_families() -> None:
    environment_general = "\n".join(
        f"APP_SETTING_{index}\nControls general setting {index}." for index in range(4)
    )
    environment_render = "\n".join(
        f"APP_RENDER_{index}\nControls render setting {index}." for index in range(4)
    )
    layer_shortcuts = (
        "Ctrl+O\nOpen a layer.\nCtrl+S\nSave layers.\n"
        "Ctrl+R\nRender layers.\nCtrl+Q\nClose the manager."
    )
    display_shortcuts = (
        "F5\nRender the display.\nF11\nToggle fullscreen.\n"
        "Ctrl+W\nClose the display.\nCtrl+P\nPrint the display."
    )
    gui_modules = "\n".join(
        f"g.gui.tool{index}\nLaunches GUI tool {index}." for index in range(4)
    )
    evidence = [
        _reference_catalog_evidence(
            1,
            title="General environment variables",
            content=environment_general,
            url="https://docs.example/manuals/environment.html",
            capability_branch="Configuration",
        ),
        _reference_catalog_evidence(
            2,
            title="Rendering environment variables",
            content=environment_render,
            url="https://docs.example/manuals/render-environment.html",
            capability_branch="Configuration",
        ),
        _reference_catalog_evidence(
            3,
            title="Layer Manager keyboard shortcuts",
            content=layer_shortcuts,
            url="https://docs.example/manuals/layer-manager.html",
            capability_branch="Graphical User Interface",
        ),
        _reference_catalog_evidence(
            4,
            title="Map Display keyboard shortcuts",
            content=display_shortcuts,
            url="https://docs.example/manuals/map-display.html",
            capability_branch="Graphical User Interface",
        ),
        _reference_catalog_evidence(
            5,
            title="GUI commands manual",
            content=gui_modules,
            url="https://docs.example/manuals/gui.html",
            capability_branch="Graphical User Interface",
        ),
    ]

    catalogs = _build_reference_catalogs("corpus_catalog_families", evidence)

    assert len(catalogs) == 3
    by_title = {item["suggested_title"]: item for item in catalogs}
    assert by_title["Environment variables lookup catalog"]["entry_count"] == 8
    assert (
        by_title[
            "Graphical User Interface keyboard and mouse shortcuts lookup catalog"
        ]["entry_count"]
        == 8
    )
    assert by_title["GUI commands manual lookup catalog"]["entry_count"] == 4
