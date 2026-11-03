"""Large chapter selections must not become inline ChildRunCapsule payloads."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from app.acquisition import AcquisitionService
from app.capabilities import CapabilityGateway, CapabilityGatewayServer
from app.config import Settings
from app.contracts import ChildRunCapsule, MAX_SCOPE_BYTES
from app.evidence_planning import EvidencePlanEngine
from app.models import ModelRuntimeConfig, StagedInput, TargetProfile
from app.orchestration.source_stage import SourceReviewService, normalize_source
from app.storage import RunStore, RunStoreConflictError

from test_t5_source_pipeline import GatewayDrivingRuntime, _pdf_bytes


HTML = b"""<html><head><title>Example Manual</title></head><body><main>
<h1>Example Manual</h1><h2>Selected instructions</h2>
<p>Keep these complete instructions and their exact source ranges.</p>
</main></body></html>"""


def large_scope(mode="full", roots=8, children=24):
    def entry(index):
        return {
            "section_id": f"section_{index:05d}", "title": f"Authored chapter {index}",
            "url": f"https://example.invalid/manual/chapter-{index}.html",
            "fragment": "", "known_page_count": 1, "children": [],
        }

    sections = [entry(index) for index in range(roots)]
    for index, section in enumerate(sections):
        section["children"] = [entry(roots + index * children + child)
                               for child in range(children)]
    return {
        "mode": mode,
        "section_ids": [item["section_id"] for item in sections[:1200]] if mode == "selected" else [],
        "sections": sections,
        "outline_artifact_id": "art_000000000000000000000001",
        "max_child_pages": 80,
        "page_budget_scope": "child_pages_only",
    }


def prepare_source(tmp_path, modality="html"):
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "out",
                        max_html_pages=0, t5_processing_mode="agent_architect")
    store = RunStore(settings.data_dir, "run_scope_handoff")
    target = TargetProfile(software_name="Example", target_version="1", platform="General")
    store.initialize({"target_profile": target.model_dump(mode="json"), "inputs": []})
    data = HTML if modality == "html" else _pdf_bytes()
    path = store.stage_upload("IN1", f"manual.{modality}", data)
    acquisition = AcquisitionService(store, settings)
    manifest = acquisition.acquire([
        StagedInput(input_id="IN1", kind="uploaded_file", display_name=path.name,
                    staged_path=str(path), declared_media_type="text/html" if modality == "html" else "application/pdf"),
    ], target).manifests[0]
    return store, normalize_source(store, settings, acquisition, target, manifest)


def review_service(store, gateway, server, driver):
    return SourceReviewService(
        store=store, runtime=driver, gateway=gateway, gateway_server=server,
        model_config=ModelRuntimeConfig(base_url="https://model.example.invalid/v1", model="offline",
                                        api_key="no-external-api"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("modality", "mode", "roots", "children"), [
    pytest.param("html", "full", 8, 24, id="deep-html-directory"),
    pytest.param("html", "full", 52, 0, id="wide-html-directory"),
    pytest.param("html", "selected", 1300, 0, id="many-selected-html-chapters"),
    pytest.param("pdf", "selected", 1300, 0, id="many-selected-pdf-chapters"),
])
async def test_large_scope_reaches_review_without_losing_selection_or_source_ranges(
    tmp_path, modality, mode, roots, children,
):
    store, prepared = prepare_source(tmp_path, modality)
    original = large_scope(mode, roots, children)
    prepared.manifest.source_metadata["extraction_scope"] = deepcopy(original)
    old_scope = {"source_id": prepared.manifest.source_id, "modality": modality,
                 "extraction_scope": original}
    with pytest.raises(ValueError, match="exceeds 8192 bytes"):
        ChildRunCapsule.validate_scope(old_scope)
    if mode == "selected":
        assert len(json.dumps(original["section_ids"]).encode()) > MAX_SCOPE_BYTES
    before_parts = {key: part.model_dump(mode="json") for key, part in prepared.workspace.parts.items()}
    before_text = dict(prepared.workspace.part_text)
    before_refs = [ref.model_dump(mode="json") for node in prepared.workspace.evidence_tree.nodes
                   for ref in node.source_refs]
    gateway = CapabilityGateway()
    server = CapabilityGatewayServer(gateway)
    driver = GatewayDrivingRuntime()
    try:
        result = await review_service(store, gateway, server, driver).review(prepared)
    finally:
        server.close()

    assert result.status == "succeeded"
    assert len(driver.invocations) == 1
    attempt = store.list_attempts()[0]
    assert attempt.status == "succeeded"
    capsule = ChildRunCapsule.model_validate(store.read_json_artifact(attempt.capsule_artifact_id))
    assert len(json.dumps(capsule.scope).encode()) < 256
    reference = capsule.scope["extraction_scope"]
    assert reference["mode"] == mode
    assert set(reference) == {"mode", "artifact_id"}
    assert reference["artifact_id"] in capsule.input_artifact_ids
    assert reference["artifact_id"] in prepared.workspace.allowed_artifacts
    assert store.artifact_record(reference["artifact_id"]).source_id == prepared.manifest.source_id
    assert store.read_json_artifact(reference["artifact_id"]) == original
    assert prepared.manifest.source_metadata["extraction_scope"] == original
    assert (prepared.manifest.raw_artifact_id in capsule.input_artifact_ids) is (mode == "full")
    assert prepared.normalization_artifact_id in capsule.input_artifact_ids
    assert before_parts == {key: part.model_dump(mode="json") for key, part in prepared.workspace.parts.items()}
    assert before_text == prepared.workspace.part_text
    assert before_refs == [ref.model_dump(mode="json") for node in prepared.workspace.evidence_tree.nodes
                           for ref in node.source_refs]
    assert EvidencePlanEngine(prepared.workspace).validate(prepared.workspace.evidence_skeleton_plan).coverage_complete


class InterruptedReviewRuntime(GatewayDrivingRuntime):
    async def run(self, invocation):
        self.invocations.append(invocation)
        raise ValueError("Intentional offline review interruption")


@pytest.mark.asyncio
async def test_scope_reference_is_stable_on_retry_but_binds_to_every_selected_id(tmp_path):
    store, prepared = prepare_source(tmp_path)
    original = large_scope("selected", roots=1300, children=0)
    prepared.manifest.source_metadata["extraction_scope"] = deepcopy(original)
    gateway = CapabilityGateway()
    server = CapabilityGatewayServer(gateway)
    driver = InterruptedReviewRuntime()
    service = review_service(store, gateway, server, driver)
    try:
        for _ in range(2):
            with pytest.raises(ValueError, match="Intentional offline"):
                await service.review(prepared)
            # Equivalent JSON maps produce the same immutable Artifact on retry.
            current = prepared.manifest.source_metadata["extraction_scope"]
            prepared.manifest.source_metadata["extraction_scope"] = dict(reversed(list(current.items())))
        attempts = store.list_attempts()
        assert len(attempts) == 2
        assert attempts[1].retry_of_attempt_id == attempts[0].attempt_id
        assert attempts[0].input_fingerprint == attempts[1].input_fingerprint
        scopes = [store.read_json_artifact(item.capsule_artifact_id)["scope"] for item in attempts]
        assert scopes[0] == scopes[1]
        assert store.read_json_artifact(scopes[0]["extraction_scope"]["artifact_id"]) == original

        changed = prepared.manifest.source_metadata["extraction_scope"]
        # Same mode and number of selections, but one different valid chapter.
        changed["section_ids"] = changed["section_ids"][1:] + [changed["sections"][1200]["section_id"]]
        with pytest.raises(RunStoreConflictError, match="Retry input fingerprint must match"):
            await service.review(prepared)
        assert len(driver.invocations) == 2
        assert len(store.list_attempts()) == 2
    finally:
        server.close()
