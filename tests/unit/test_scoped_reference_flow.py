from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.agent_export import export_agent_entries
from app.capabilities import CapabilityGateway
from app.config import Settings
from app.models import EvidenceExcerpt, ModelRuntimeConfig, ReferenceCandidate, StagedInput, TargetProfile
from app.orchestration.pipeline import PipelineRuntime

from test_t6_reference_stage import ReferenceDrivingRuntime


HTML = b"""<!doctype html><html><head><title>ExampleCalc Manual</title></head>
<body><main><h1>ExampleCalc Manual</h1>
<nav class="contents"><ul>
<li><a href="#chapter-a">Chapter A</a></li>
<li><a href="#chapter-b">Chapter B</a></li>
</ul></nav>
<h2 id="chapter-a">Chapter A</h2>
<h3>Keyboard shortcuts</h3>
<table><tr><th>Key</th><th>Action</th></tr>
<tr><td>Ctrl+O</td><td>ALPHA_SELECTED_MAPPING: Open a document.</td></tr>
<tr><td>Ctrl+S</td><td>Save the current document.</td></tr>
<tr><td>Ctrl+P</td><td>Print the current document.</td></tr>
<tr><td>Ctrl+Q</td><td>Quit the application.</td></tr></table>
<h3>Save a document</h3><ol>
<li>Select File and then Save.</li>
<li>Choose a destination and confirm Save to finish ALPHA_SELECTED_OPERATION.</li>
</ol>
<h2 id="chapter-b">Chapter B</h2>
<h3>Advanced keyboard shortcuts</h3>
<p>BETA_UNSELECTED_BODY must never enter a selected Chapter A result.</p>
<table><tr><th>Key</th><th>Action</th></tr>
<tr><td>Alt+A</td><td>BETA_UNSELECTED_MAPPING: Open the advanced panel.</td></tr>
<tr><td>Alt+B</td><td>Change the advanced display.</td></tr>
<tr><td>Alt+C</td><td>Reset the advanced calculation.</td></tr>
<tr><td>Alt+D</td><td>Close the advanced panel.</td></tr></table>
</main></body></html>"""


@pytest.mark.asyncio
@pytest.mark.integration
async def test_selected_html_chapter_stays_scoped_through_reference_and_export(tmp_path: Path) -> None:
    """Use real stage/tool boundaries; only model decisions are deterministic substitutes."""
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        scope_selection_enabled=True,
        # Selected chapters must route the default adaptive profile through the
        # fixed parser and Evidence Architect before preparing any References.
        t5_processing_mode="agent",
        t6_reference_enabled=True,
        t6_task_enabled=False,
        max_html_pages=80,
    )
    gateway = CapabilityGateway()
    driver = ReferenceDrivingRuntime(gateway)
    runtime = PipelineRuntime(settings, driver, capability_gateway=gateway)
    run_id = "run_scoped_reference_flow"
    store = runtime.create_store(run_id)
    target = TargetProfile(software_name="ExampleCalc", target_version="1.0", platform="General")
    model = ModelRuntimeConfig(base_url="https://model.example.invalid/v1", model="controlled-model",
                               api_key="scoped-reference-no-network-key")
    path = store.stage_upload("IN1", "manual.html", HTML)
    staged = StagedInput(input_id="IN1", kind="uploaded_file", display_name="manual.html",
                         declared_media_type="text/html", user_declared_version="1.0", staged_path=str(path))
    store.initialize({
        "target_profile": target.model_dump(mode="json"),
        "model": model.public_dict(),
        "inputs": [staged.model_dump(mode="json")],
        "scope_selection_enabled": True,
    })

    try:
        preview_result = await asyncio.wait_for(runtime.execute(run_id, target, [staged], model), timeout=15)
        assert preview_result["status"] == "waiting_for_input"
        assert store.read_state()["phase"] == "awaiting_scope_selection"
        assert [invocation.role for invocation in driver.invocations] == ["source_review"]
        assert list((store.root / "sources").glob("*/parts.json"))

        outline = runtime.source_scope_payload(run_id)
        assert outline["outline_available"]
        assert outline["structure_review_complete"]
        selected_id = next(section["section_id"] for section in outline["sections"] if section["title"] == "Chapter A")
        runtime.select_scope(run_id, outline["outline_artifact_id"], "selected", [selected_id])
        # select_scope dispatches the actual background pipeline continuation.
        continuation = runtime._tasks[run_id]
        result = await asyncio.wait_for(continuation, timeout=15)

        state = store.read_state()
        assert state["phase"] == "references_ready"
        assert state["status"] == "paused"
        assert result["outcome"] == "references_ready"
        assert [invocation.role for invocation in driver.invocations] == ["source_review", "reference_build"]
        assert all(attempt.status == "succeeded" for attempt in store.list_attempts())

        frozen = store.read_json_artifact(state["corpus_snapshot_artifact_id"])
        evidence = [EvidenceExcerpt.model_validate(item) for item in frozen["evidence"]]
        assert evidence
        evidence_text = "\n".join(item.content for item in evidence)
        assert "ALPHA_SELECTED_MAPPING" in evidence_text
        assert "ALPHA_SELECTED_OPERATION" in evidence_text
        assert "BETA_UNSELECTED" not in evidence_text
        assert frozen["source_results"][0]["manifest"]["source_metadata"]["extraction_scope"]["section_ids"] == [selected_id]

        references = [ReferenceCandidate.model_validate(item) for item in result["references"]]
        assert references
        assert all(item["accepted"] for item in result["reference_validation"])
        entries = [entry for reference in references for entry in reference.entries]
        assert entries, "The selected authored shortcut table must become source-preserved entries."
        reference_text = "\n".join([*(reference.content for reference in references), *(entry.content for entry in entries)])
        assert "ALPHA_SELECTED_MAPPING" in reference_text
        assert "Ctrl+S" in reference_text
        assert "BETA_UNSELECTED" not in reference_text
        evidence_by_id = {item.evidence_id: item for item in evidence}
        assert all(entry.source_evidence_id in evidence_by_id for entry in entries)
        assert all("BETA_UNSELECTED" not in evidence_by_id[entry.source_evidence_id].content for entry in entries)

        exported = export_agent_entries(settings.output_dir, run_id=run_id, target_profile=target, skills=[],
                                        references=references, evidence_by_id=evidence_by_id)
        bodies = [Path(item["content_file"]).read_text(encoding="utf-8") for item in exported["agent_reference_entries"]]
        assert len(bodies) == len(references)
        assert "ALPHA_SELECTED_MAPPING" in "\n".join(bodies)
        assert all("BETA_UNSELECTED" not in body for body in bodies)
    finally:
        runtime.close()
