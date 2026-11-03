from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.models import (
    CorpusSnapshot,
    Coverage,
    EvidenceExcerpt,
    EvidenceImageRef,
    ModelRuntimeConfig,
    SourceManifest,
    SourcePart,
    SourceResult,
    TargetProfile,
)
from app.orchestration.pipeline import PipelineRuntime
from app.runtime import UnavailableAgentRuntime
from app.storage import RunStore


class OfflineGatewayServer:
    @property
    def endpoint_url(self) -> str:
        raise AssertionError("Cache revalidation must not bind a gateway or call a model")

    def close(self) -> None:
        pass


@pytest.fixture
def runtime(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        
    )
    result = PipelineRuntime(
        settings,
        UnavailableAgentRuntime(),
        capability_server=OfflineGatewayServer(),
    )
    yield result
    result.close()


def _target() -> TargetProfile:
    return TargetProfile(
        software_name="Example",
        target_version="1.0",
        platform="General",
        ui_locale="English",
    )


def _new_store(runtime: PipelineRuntime, run_id: str) -> RunStore:
    store = runtime.create_store(run_id)
    store.initialize(
        {
            "run_id": run_id,
            "target_profile": _target().model_dump(mode="json"),
            "inputs": [],
            "model": ModelRuntimeConfig().public_dict(),
        }
    )
    return store


def _source_fixture(
    runtime: PipelineRuntime, *, wrong_image_part: bool = False
) -> tuple[RunStore, str, dict[str, str]]:
    store = _new_store(runtime, "run_original_html")
    source_id = "SRC1"
    raw = store.put_artifact(
        b"<h1>Controls</h1><p>Move the model.</p><img src='figure.png'>",
        kind="raw_source",
        media_type="text/html",
        suffix=".html",
        source_id=source_id,
    )
    manifest = SourceManifest(
        source_id=source_id,
        canonical_input_id="IN1",
        display_name="manual.html",
        primary_modality="html",
        media_type="text/html",
        raw_artifact_id=raw.artifact_id,
        status="degraded",
    )
    image = store.put_artifact(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aV1sAAAAASUVORK5CYII="
        ),
        kind="html_image",
        media_type="image/png",
        suffix=".png",
        source_id=source_id,
    )
    parts = []
    evidence = []
    for ordinal, content in enumerate(
        ["Choose the Controls menu.", "Drag the depicted handle to move the model."],
        start=1,
    ):
        text = store.put_artifact(
            content.encode("utf-8"),
            kind="source_part_text",
            media_type="text/plain",
            suffix=".txt",
            source_id=source_id,
        )
        part = SourcePart(
            part_id=f"SP{ordinal}",
            source_id=source_id,
            kind="html_section",
            label=f"Control {ordinal}",
            locator={"page": 1},
            ordinal=ordinal,
            artifact_id=raw.artifact_id,
            text_artifact_id=text.artifact_id,
            content_sha256=text.sha256,
            char_count=len(content),
            metadata=(
                {
                    "images": {
                        "I1": {
                            "status": "saved",
                            "artifact_id": image.artifact_id,
                            "media_type": "image/png",
                        }
                    }
                }
                if ordinal == 2
                else {}
            ),
        )
        parts.append(part)
        evidence.append(
            EvidenceExcerpt(
                evidence_id=f"E{ordinal}",
                source_id=source_id,
                source_part_id=part.part_id,
                text_artifact_id=text.artifact_id,
                start_offset=0,
                end_offset=len(content),
                content_sha256=text.sha256,
                kind="section",
                title=part.label,
                content=content,
                locator=part.locator,
                extraction_method="test_exact_source",
                image_refs=(
                    [
                        EvidenceImageRef(
                            artifact_id=image.artifact_id,
                            source_part_id="SP1" if wrong_image_part else part.part_id,
                            media_type="image/png",
                            page=1,
                            association_method="html_structure",
                        )
                    ]
                    if ordinal == 2
                    else []
                ),
            )
        )
    coverage = Coverage(
        discovered_parts=2,
        inventoried_parts=2,
        normalized_parts=2,
        semantic_units=2,
        evidence_bearing_parts=2,
    )
    corpus = store.put_json_artifact(
        {
            "schema_version": "v14-html-structured.v11",
            "source_id": source_id,
            "raw_artifact_id": raw.artifact_id,
            "manifest": manifest.model_dump(mode="json"),
            "parts": [part.model_dump(mode="json") for part in parts],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "evidence_tree": None,
            "coverage": coverage.model_dump(mode="json"),
            "gaps": ["Child-page budget was exhausted"],
            "warnings": ["One remote stylesheet was unavailable"],
        },
        kind="normalized_source_corpus",
        source_id=source_id,
        preserve_text_integrity=True,
    )
    raw_result = SourceResult(
        source_id=source_id,
        display_name=manifest.display_name,
        modality="html",
        status="degraded",
        summary="The manual contains useful instructions despite capture gaps.",
        evidence=evidence,
        corpus_artifact_id=corpus.artifact_id,
        source_part_count=2,
        normalization_version="v14-html-structured.v11",
        coverage=coverage,
        gaps=["Child-page budget was exhausted"],
        warnings=["One remote stylesheet was unavailable"],
        manifest=manifest,
    )
    old_result = raw_result.model_copy(deep=True)
    old_result.evidence = old_result.evidence[:1]
    old_result.coverage.evidence_bearing_parts = 1
    old_result.gaps.append("evidence_validation_rejected:1")
    old_result.errors = ["Evidence E2 cites an unknown or mismatched image Artifact"]
    snapshot = CorpusSnapshot(
        corpus_snapshot_id="CP1",
        run_id=store.run_id,
        target_profile=_target(),
        source_results=[old_result],
        evidence=old_result.evidence,
        evidence_count=1,
        source_part_count=2,
        source_status_counts={"degraded": 1},
    )
    snapshot_artifact = store.put_json_artifact(
        snapshot, kind="corpus_snapshot", preserve_text_integrity=True
    )
    for name, payload in (
        ("manifest.json", manifest.model_dump(mode="json")),
        ("parts.json", [part.model_dump(mode="json") for part in parts]),
        ("result.json", raw_result.model_dump(mode="json")),
        ("validated_result.json", old_result.model_dump(mode="json")),
    ):
        store.write_source_json(source_id, name, payload)
    store.update_state(
        t5_stage_complete=True,
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        corpus_snapshot_artifact_id=snapshot_artifact.artifact_id,
    )
    return store, snapshot_artifact.artifact_id, {
        "corpus": corpus.artifact_id,
        "text": parts[1].text_artifact_id,
        "image": image.artifact_id,
    }


def _file_contents(store: RunStore) -> dict[str, bytes]:
    return {
        path.relative_to(store.root).as_posix(): path.read_bytes()
        for path in store.root.rglob("*")
        if path.is_file()
    }


def _import(
    runtime: PipelineRuntime, source: RunStore, snapshot_id: str, run_id: str
) -> tuple[RunStore, CorpusSnapshot]:
    target = _new_store(runtime, run_id)
    snapshot = runtime.import_frozen_corpus(
        run_id=run_id, source_store=source, corpus_artifact_id=snapshot_id
    )
    return target, snapshot


def test_cache_import_recovers_html_attachment_from_unfiltered_corpus(runtime):
    source, old_snapshot_id, ids = _source_fixture(runtime)
    before = _file_contents(source)

    target, restored = _import(runtime, source, old_snapshot_id, "run_recovered")

    assert restored.run_id == target.run_id
    assert restored.evidence_count == 2
    assert [item.evidence_id for item in restored.evidence] == ["E1", "E2"]
    assert restored.evidence[1].image_refs[0].artifact_id == ids["image"]
    assert restored.source_results[0].errors == []
    assert restored.source_results[0].gaps == ["Child-page budget was exhausted"]
    assert restored.source_results[0].warnings == ["One remote stylesheet was unavailable"]
    assert restored.source_results[0].status == "degraded"
    assert restored.source_results[0].coverage.evidence_bearing_parts == 2
    state = target.read_state()
    assert state["phase"] == "corpus_frozen"
    assert state["t5_cache_revalidated"] is True
    assert state["t5_cache_original_corpus_snapshot_artifact_id"] == old_snapshot_id
    assert state["t5_cache_restored_evidence_count"] == 1
    assert state["corpus_snapshot_id"] == restored.corpus_snapshot_id
    assert state["corpus_snapshot_artifact_id"] != old_snapshot_id
    receipt = target.list_receipt_commits()[0]["receipt"]
    assert receipt["stage"] == "corpus_cache_import"
    assert state["corpus_snapshot_artifact_id"] in receipt["accepted_output_artifact_ids"]
    assert receipt["validation_artifact_id"]
    assert target.artifact_path(ids["image"]).read_bytes() == source.artifact_path(ids["image"]).read_bytes()
    assert (target.root / "sources" / "SRC1" / "parts.json").exists()
    assert _file_contents(source) == before


def test_cache_import_still_rejects_image_attached_to_the_wrong_part(runtime):
    source, snapshot_id, _ = _source_fixture(runtime, wrong_image_part=True)

    target, restored = _import(runtime, source, snapshot_id, "run_invalid_attachment")

    assert restored.evidence_count == 1
    assert [item.evidence_id for item in restored.evidence] == ["E1"]
    assert restored.source_results[0].status == "degraded"
    assert any("E2" in error and "image Artifact" in error for error in restored.source_results[0].errors)
    assert "evidence_validation_rejected:1" in restored.source_results[0].gaps
    assert target.read_state()["t5_cache_restored_evidence_count"] == 0


@pytest.mark.parametrize("corrupt_artifact", ["corpus", "text", "image"])
def test_cache_import_refuses_corrupted_source_artifacts(runtime, corrupt_artifact):
    source, snapshot_id, ids = _source_fixture(runtime)
    corrupt_path = source.artifact_path(ids[corrupt_artifact])
    corrupt_path.write_bytes(corrupt_path.read_bytes() + b"tampered")
    before = _file_contents(source)
    target = _new_store(runtime, "run_corrupt_cache")

    with pytest.raises((RuntimeError, ValueError), match="(?i)immutable record"):
        runtime.import_frozen_corpus(
            run_id=target.run_id, source_store=source, corpus_artifact_id=snapshot_id
        )

    assert target.read_state()["phase"] == "created"
    assert not target.list_receipt_commits()
    assert _file_contents(source) == before


def test_cache_import_preserves_legacy_minimal_snapshot(runtime):
    source = _new_store(runtime, "run_legacy_cache")
    content = "Select Controls to configure the model."
    text = source.put_artifact(
        content.encode("utf-8"),
        kind="evidence_text",
        media_type="text/plain",
        suffix=".txt",
        source_id="SRC1",
    )
    evidence = EvidenceExcerpt(
        evidence_id="E1",
        source_id="SRC1",
        source_part_id="SP1",
        text_artifact_id=text.artifact_id,
        start_offset=0,
        end_offset=len(content),
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        kind="section",
        title="Controls",
        content=content,
        locator={"page": 1},
        extraction_method="legacy_fixture",
    )
    old = CorpusSnapshot(
        corpus_snapshot_id="corpus_legacy",
        run_id=source.run_id,
        target_profile=_target(),
        source_results=[],
        evidence=[evidence],
        evidence_count=1,
        source_part_count=1,
        source_status_counts={},
    )
    artifact = source.put_json_artifact(old, kind="corpus_snapshot", preserve_text_integrity=True)
    before = _file_contents(source)

    target, imported = _import(runtime, source, artifact.artifact_id, "run_legacy_import")

    assert imported.model_dump(mode="json") == old.model_dump(mode="json")
    assert target.read_state()["t5_cache_revalidated"] is False
    assert target.read_state()["corpus_snapshot_artifact_id"] == artifact.artifact_id
    assert target.artifact_path(text.artifact_id).read_text(encoding="utf-8") == content
    assert _file_contents(source) == before


def test_revalidated_cache_can_be_imported_into_another_fresh_run(runtime):
    source, old_snapshot_id, _ = _source_fixture(runtime)
    first, first_snapshot = _import(runtime, source, old_snapshot_id, "run_first_recovery")
    before = _file_contents(first)
    first_snapshot_id = first.read_state()["corpus_snapshot_artifact_id"]

    second, second_snapshot = _import(runtime, first, first_snapshot_id, "run_second_recovery")

    assert first_snapshot.evidence_count == second_snapshot.evidence_count == 2
    assert second_snapshot.run_id == second.run_id
    assert second.read_state()["t5_cache_revalidated"] is True
    assert second.read_state()["t5_cache_restored_evidence_count"] == 0
    assert second.read_state()["corpus_snapshot_artifact_id"] != first_snapshot_id
    assert _file_contents(first) == before


def test_cache_import_recovers_a_source_wholly_rejected_by_old_validation(runtime):
    source, old_snapshot_id, _ = _source_fixture(runtime)
    old = CorpusSnapshot.model_validate(source.read_json_artifact(old_snapshot_id))
    failed = old.source_results[0]
    failed.status = "failed"
    failed.evidence = []
    failed.evidence_tree = None
    failed.gaps = ["Child-page budget was exhausted", "branch_validation_failed"]
    failed.coverage.evidence_bearing_parts = 0

    content = "Save the configured model using File > Save."
    text = source.put_artifact(
        content.encode("utf-8"),
        kind="source_part_text",
        media_type="text/plain",
        suffix=".txt",
        source_id="SRC2",
    )
    manifest = SourceManifest(
        source_id="SRC2",
        canonical_input_id="IN2",
        display_name="healthy.html",
        primary_modality="html",
        media_type="text/html",
        status="succeeded",
    )
    part = SourcePart(
        part_id="SP3",
        source_id="SRC2",
        kind="html_section",
        label="Save",
        locator={"page": 1},
        text_artifact_id=text.artifact_id,
        content_sha256=text.sha256,
        char_count=len(content),
    )
    evidence = EvidenceExcerpt(
        evidence_id="E3",
        source_id="SRC2",
        source_part_id=part.part_id,
        text_artifact_id=text.artifact_id,
        start_offset=0,
        end_offset=len(content),
        content_sha256=text.sha256,
        kind="section",
        title="Save",
        content=content,
        locator={"page": 1},
        extraction_method="test_exact_source",
    )
    coverage = Coverage(
        discovered_parts=1,
        inventoried_parts=1,
        normalized_parts=1,
        semantic_units=1,
        evidence_bearing_parts=1,
    )
    corpus = source.put_json_artifact(
        {
            "schema_version": "v14-html-structured.v11",
            "source_id": "SRC2",
            "manifest": manifest.model_dump(mode="json"),
            "parts": [part.model_dump(mode="json")],
            "evidence": [evidence.model_dump(mode="json")],
            "evidence_tree": None,
            "coverage": coverage.model_dump(mode="json"),
            "gaps": [],
            "warnings": [],
        },
        kind="normalized_source_corpus",
        source_id="SRC2",
        preserve_text_integrity=True,
    )
    healthy = SourceResult(
        source_id="SRC2",
        display_name="healthy.html",
        modality="html",
        status="succeeded",
        evidence=[evidence],
        source_part_count=1,
        corpus_artifact_id=corpus.artifact_id,
        coverage=coverage,
        manifest=manifest,
    )
    old.source_results.append(healthy)
    old.evidence = [evidence]
    old.evidence_count = 1
    old.source_part_count = 3
    old.source_status_counts = {"failed": 1, "succeeded": 1}
    snapshot_artifact = source.put_json_artifact(old, kind="corpus_snapshot", preserve_text_integrity=True)
    before = _file_contents(source)

    target, restored = _import(runtime, source, snapshot_artifact.artifact_id, "run_failed_source_recovery")

    assert restored.evidence_count == 3
    assert {item.evidence_id for item in restored.evidence} == {"E1", "E2", "E3"}
    recovered = next(item for item in restored.source_results if item.source_id == "SRC1")
    assert recovered.status == "degraded"
    assert recovered.gaps == ["Child-page budget was exhausted"]
    assert recovered.errors == []
    assert recovered.coverage.evidence_bearing_parts == 2
    assert target.read_state()["t5_cache_restored_evidence_count"] == 2
    assert _file_contents(source) == before


def test_rerun_api_returns_the_revalidated_snapshot(runtime, monkeypatch):
    import app.main as main_module

    source, old_snapshot_id, _ = _source_fixture(runtime)
    before = _file_contents(source)
    dispatched = []
    monkeypatch.setattr(main_module, "runtime", runtime)
    monkeypatch.setattr(main_module, "settings", runtime.settings)
    monkeypatch.setattr(
        runtime,
        "start_reference_from_frozen",
        lambda run_id, target, model: dispatched.append(run_id),
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            f"/api/runs/{source.run_id}/rerun-reference",
            data={
                "api_key": "test-key",
                "api_base_url": "https://example.invalid/v1",
                "api_model": "test-model",
            },
        )

    assert response.status_code == 202, response.text
    payload = response.json()
    target = runtime.get_store(payload["run_id"])
    state = target.read_state()
    assert dispatched == [target.run_id]
    assert payload["corpus_snapshot_id"] == state["corpus_snapshot_id"]
    assert payload["corpus_snapshot_artifact_id"] == state["corpus_snapshot_artifact_id"]
    assert payload["corpus_snapshot_artifact_id"] != old_snapshot_id
    assert state["evidence_count"] == 2
    assert state["t5_cache_revalidated"] is True
    assert _file_contents(source) == before


def test_rerun_api_reports_invalid_cache_before_dispatch(runtime, monkeypatch):
    import app.main as main_module

    source, _, ids = _source_fixture(runtime)
    source.artifact_path(ids["corpus"]).write_bytes(b"corrupted cached source")
    before = _file_contents(source)
    dispatched = []
    monkeypatch.setattr(main_module, "runtime", runtime)
    monkeypatch.setattr(main_module, "settings", runtime.settings)
    monkeypatch.setattr(
        runtime,
        "start_reference_from_frozen",
        lambda run_id, target, model: dispatched.append(run_id),
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            f"/api/runs/{source.run_id}/rerun-reference",
            data={
                "api_key": "test-key",
                "api_base_url": "https://example.invalid/v1",
                "api_model": "test-model",
            },
        )

    assert response.status_code == 409, response.text
    assert "immutable record" in response.json()["detail"]
    assert dispatched == []
    assert _file_contents(source) == before


def test_rerun_api_reports_cache_with_no_valid_evidence(runtime, monkeypatch):
    import app.main as main_module

    source, snapshot_id, ids = _source_fixture(runtime, wrong_image_part=True)
    normalized = source.read_json_artifact(ids["corpus"])
    normalized["evidence"][0]["content"] = "Invented text absent from the source artifact."
    invalid_corpus = source.put_json_artifact(
        normalized,
        kind="normalized_source_corpus",
        source_id="SRC1",
        preserve_text_integrity=True,
    )
    old = CorpusSnapshot.model_validate(source.read_json_artifact(snapshot_id))
    old.source_results[0].corpus_artifact_id = invalid_corpus.artifact_id
    old_artifact = source.put_json_artifact(old, kind="corpus_snapshot", preserve_text_integrity=True)
    source.update_state(corpus_snapshot_artifact_id=old_artifact.artifact_id)
    before = _file_contents(source)
    dispatched = []
    monkeypatch.setattr(main_module, "runtime", runtime)
    monkeypatch.setattr(main_module, "settings", runtime.settings)
    monkeypatch.setattr(
        runtime,
        "start_reference_from_frozen",
        lambda run_id, target, model: dispatched.append(run_id),
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            f"/api/runs/{source.run_id}/rerun-reference",
            data={
                "api_key": "test-key",
                "api_base_url": "https://example.invalid/v1",
                "api_model": "test-model",
            },
        )

    assert response.status_code == 409, response.text
    assert "no validated Evidence" in response.json()["detail"]
    assert dispatched == []
    assert _file_contents(source) == before
