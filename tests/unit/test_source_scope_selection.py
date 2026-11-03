from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.orchestration.pipeline as pipeline_module
from app.config import Settings
from app.models import ModelRuntimeConfig, SourceManifest, StagedInput, TargetProfile
from app.orchestration.pipeline import PipelineRuntime, _receipt
from app.orchestration.source_scope_stage import apply_scope
from app.orchestration.state import PipelineStateMachine, RunPhase
from app.storage import RunStore
from app.acquisition import AcquisitionService
from app.orchestration.source_stage import load_persisted_manifests, normalize_source
from app.toolkit import ToolExecutor


TARGET = TargetProfile(software_name="Example", target_version="1", platform="General")
SECRET = "scope-selection-model-key-do-not-persist"
MODEL = ModelRuntimeConfig(base_url="https://model.example.invalid/v1", model="test", api_key=SECRET)


@pytest.mark.parametrize('value,enabled', [(None, False), ('0', False), ('false', False),
                                          ('1', True), ('true', True)])
def test_scope_feature_switch_is_disabled_by_default(value, enabled):
    environment = dict(os.environ)
    environment.pop('PREPROCESS_SCOPE_SELECTION_ENABLED', None)
    if value is not None:
        environment['PREPROCESS_SCOPE_SELECTION_ENABLED'] = value
    result = subprocess.run([sys.executable, '-c',
                             'from app.config import Settings; print(int(Settings().scope_selection_enabled))'],
                            cwd=Path(__file__).resolve().parents[2], env=environment,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(int(enabled))


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "output",
                    scope_selection_enabled=True, t5_processing_mode="deterministic",
                    t6_reference_enabled=False, t6_task_enabled=False)


def _initialize(runtime: PipelineRuntime, run_id: str = "run_scope") -> RunStore:
    store = runtime.create_store(run_id)
    store.initialize({"target_profile": TARGET.model_dump(mode="json"), "model": MODEL.public_dict(),
                      "inputs": [], "scope_selection_enabled": True})
    return store


def _waiting(runtime: PipelineRuntime, run_id: str = "run_scope", *, available: bool = True):
    store = _initialize(runtime, run_id)
    raw = store.put_artifact(b"<html>Chapter A and B</html>", kind="raw_source", media_type="text/html", suffix=".html")
    manifest = SourceManifest(source_id="SRC1", canonical_input_id="IN1", display_name="Manual",
                              primary_modality="html", media_type="text/html", raw_artifact_id=raw.artifact_id)
    store.write_source_json("SRC1", "manifest.json", manifest.model_dump(mode="json"))
    outline = store.put_json_artifact({
        "schema_version": "source-outline.v1", "source_id": "SRC1", "modality": "html", "title": "Manual",
        "raw_artifact_id": raw.artifact_id, "raw_sha256": raw.sha256, "outline_available": available,
        "sections": [{"section_id": "A", "title": "Chapter A", "depth": 0},
                     {"section_id": "A1", "title": "Nested A1", "depth": 1},
                     {"section_id": "A11", "title": "Deep A11", "depth": 2},
                     {"section_id": "B", "title": "Chapter B", "depth": 0}]
                    if available else [], "warnings": [] if available else ["No reliable outline"],
        "max_child_pages": 80, "page_budget_scope": "child_pages_only",
    }, kind="source_outline_preview")
    machine = PipelineStateMachine(store)
    for phase in (RunPhase.ACQUIRING, RunPhase.PREVIEWING_SCOPE):
        machine.advance(phase, _receipt(run_id=run_id, stage=phase.value, fingerprint_value=phase.value))
    machine.advance(RunPhase.AWAITING_SCOPE_SELECTION,
                    _receipt(run_id=run_id, stage="scope_preview", fingerprint_value=outline.artifact_id,
                             artifact_ids=(outline.artifact_id,)))
    store.update_state(source_outline_artifact_id=outline.artifact_id)
    return store, manifest, outline.artifact_id


class RecordingRuntime(PipelineRuntime):
    def __init__(self, settings):
        super().__init__(settings)
        self.dispatched = []

    def start(self, run_id, target_profile, inputs, model_config):
        self.dispatched.append((run_id, inputs, model_config))


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('source_mode', ['manual', 'discovery'])
def test_new_runs_use_the_server_scope_switch(tmp_path, monkeypatch, enabled, source_mode):
    runtime = RecordingRuntime(replace(_settings(tmp_path), scope_selection_enabled=enabled))
    monkeypatch.setattr(main_module, 'runtime', runtime)
    monkeypatch.setattr(main_module, 'settings', runtime.settings)
    form = {'software_name': 'Example', 'target_version': '1', 'platform': 'General',
            'api_base_url': MODEL.base_url, 'api_model': MODEL.model, 'api_key': SECRET,
            'source_mode': source_mode, 'scope_selection_enabled': 'true'}
    files = {'files': ('manual.html', b'<h1>Manual</h1>', 'text/html')} if source_mode == 'manual' else None
    try:
        with TestClient(main_module.app) as client:
            response = client.post('/api/runs', data=form, files=files)
            assert response.status_code == 202, response.text
            store = runtime.get_store(response.json()['run_id'])
            assert store.read_state()['scope_selection_enabled'] is enabled
            request = json.loads((store.root / 'request.json').read_text(encoding='utf-8'))
            assert request['scope_selection_enabled'] is enabled
    finally:
        runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('modality', ['html', 'pdf'])
async def test_disabled_scope_skips_preview_and_extracts_the_complete_manual(tmp_path, monkeypatch, modality):
    runtime = PipelineRuntime(replace(_settings(tmp_path), scope_selection_enabled=False))
    store = _initialize(runtime)  # A saved/caller-supplied True cannot enable the feature.
    if modality == 'html':
        data = b'<h1>Manual</h1><h2>Alpha</h2><p>ALPHA_BODY.</p><h2>Beta</h2><p>BETA_BODY.</p>'
    else:
        with fitz.open() as document:
            for title, body in [('Alpha', 'ALPHA_BODY.'), ('Beta', 'BETA_BODY.')]:
                page = document.new_page()
                page.insert_text((60, 90), title, fontsize=20)
                page.insert_text((60, 130), body)
            data = document.tobytes()
    path = store.stage_upload('IN1', f'manual.{modality}', data)
    staged = StagedInput(input_id='IN1', kind='uploaded_file', display_name='Manual', staged_path=str(path))
    monkeypatch.setattr(pipeline_module, 'build_outline_preview', lambda *args, **kwargs: pytest.fail('Disabled preview'))
    try:
        result = await runtime.execute(store.run_id, TARGET, [staged], MODEL)
        state = store.read_state()
        assert state['phase'] == 'corpus_frozen'
        assert state['scope_selection_enabled'] is False
        assert not state.get('source_outline_artifact_id')
        assert not state.get('scope_selection_artifact_id')
        text = '\n'.join(item['content'] for item in result['evidence'])
        assert 'ALPHA_BODY' in text and 'BETA_BODY' in text
    finally:
        runtime.close()


def test_disabled_scope_rejects_new_chapters_but_allows_full_continuation(tmp_path, monkeypatch):
    runtime = RecordingRuntime(replace(_settings(tmp_path), scope_selection_enabled=False))
    store, _, outline_id = _waiting(runtime)
    monkeypatch.setattr(main_module, 'runtime', runtime)
    monkeypatch.setattr(main_module, 'settings', runtime.settings)
    monkeypatch.setattr(main_module, '_resolve_api_key', lambda supplied, base_url: supplied)
    try:
        with TestClient(main_module.app) as client:
            assert client.get('/api/health').json()['scope_selection_enabled'] is False
            payload = client.get(f'/api/runs/{store.run_id}').json()
            assert payload['source_scope']['selection_enabled'] is False
            endpoint = f'/api/runs/{store.run_id}/select-scope'
            response = client.post(endpoint, json={'outline_artifact_id': outline_id, 'mode': 'selected',
                                                   'section_ids': ['A'], 'api_key': SECRET})
            assert response.status_code == 409
            assert 'disabled' in response.json()['detail']
            assert store.read_state()['phase'] == 'awaiting_scope_selection'
            assert runtime.dispatched == []
            assert client.post(endpoint, json={'outline_artifact_id': outline_id, 'mode': 'full',
                                               'section_ids': [], 'api_key': SECRET}).status_code == 202
    finally:
        runtime.close()


def test_disabling_scope_preserves_an_already_committed_selection(tmp_path):
    enabled = RecordingRuntime(_settings(tmp_path))
    store, _, outline_id = _waiting(enabled)
    enabled.select_scope(store.run_id, outline_id, 'selected', ['A'], SECRET)
    disabled = RecordingRuntime(replace(enabled.settings, scope_selection_enabled=False))
    try:
        payload = disabled.source_scope_payload(store.run_id)
        assert payload['selection'] == {'mode': 'selected', 'section_ids': ['A']}
        assert disabled.select_scope(store.run_id, outline_id, 'selected', ['A'], SECRET)['already_selected']
        with pytest.raises(RuntimeError, match='disabled'):
            disabled.select_scope(store.run_id, outline_id, 'selected', ['B'], SECRET)
        assert disabled.source_scope_payload(store.run_id)['selection'] == payload['selection']
    finally:
        enabled.close()
        disabled.close()


@pytest.mark.asyncio
async def test_scope_choices_pause_after_complete_parsing_and_review(tmp_path, monkeypatch):
    runtime = PipelineRuntime(_settings(tmp_path))
    store = _initialize(runtime)
    path = store.stage_upload("IN1", "manual.html", b"<html><h1>Manual</h1><h2>A</h2><p>A body</p></html>")

    build = pipeline_module.build_outline_preview
    def preview(store, results):
        assert list((store.root / "sources").glob("*/parts.json"))
        assert results[0].evidence_tree is not None
        assert "A body" in "\n".join(item.content for item in results[0].evidence)
        assert any(commit["receipt"]["stage"] == "source_review" for commit in store.list_receipt_commits())
        return build(store, results)

    monkeypatch.setattr(pipeline_module, "build_outline_preview", preview)
    result = await runtime.execute(store.run_id, TARGET, [StagedInput(
        input_id="IN1", kind="uploaded_file", display_name="Manual", staged_path=str(path),
    )], MODEL)
    assert result["status"] == "waiting_for_input"
    assert store.read_state()["phase"] == "awaiting_scope_selection"
    assert not store.list_attempts()
    assert list((store.root / "sources").glob("*/parts.json"))
    # Re-entering a waiting Run does no acquisition, preview, or parsing.
    monkeypatch.setattr(pipeline_module, "build_outline_preview", lambda *args, **kwargs: pytest.fail("No repeated preview"))
    assert (await runtime.execute(store.run_id, TARGET, [], MODEL))["status"] == "waiting_for_input"
    runtime.close()


def test_selection_persists_and_replays_from_receipts_after_restart(tmp_path):
    runtime = RecordingRuntime(_settings(tmp_path))
    store, manifest, outline_id = _waiting(runtime)
    runtime._model_configs[store.run_id] = MODEL
    result = runtime.select_scope(store.run_id, outline_id, "selected", ["B", "A"])
    assert result["status"] == "running"
    assert runtime.dispatched[0][1] == []  # Original cached material is resumed, never reacquired.
    store.update_state(source_outline_artifact_id=None, scope_selection_artifact_id=None)
    restored = RecordingRuntime(_settings(tmp_path))
    payload = restored.source_scope_payload(store.run_id)
    assert payload["selection"] == {"mode": "selected", "section_ids": ["A", "B"]}
    assert store.read_state()["phase"] == "scope_selected"
    with pytest.raises(ValueError, match="API Key"):
        restored.select_scope(store.run_id, outline_id, "selected", ["A", "B"])
    assert restored.select_scope(store.run_id, outline_id, "selected", ["A", "B"], SECRET)["already_selected"]
    assert len(restored.dispatched) == 1
    apply_scope(store, [manifest], {"outline_artifact_id": outline_id, "mode": "selected", "section_ids": ["A", "B"]})
    assert manifest.source_metadata["extraction_scope"]["section_ids"] == ["A", "B"]
    with pytest.raises(RuntimeError, match="not waiting"):
        restored.select_scope(store.run_id, outline_id, "full", [], SECRET)
    persisted = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in store.root.rglob("*.json"))
    assert SECRET not in persisted
    runtime.close()
    restored.close()


@pytest.mark.parametrize("mode,ids", [("selected", []), ("selected", ["missing"]),
                                     ("selected", ["A", "A"]), ("selected", ["A1"]),
                                     ("selected", ["A11"]), ("full", ["A"])])
def test_invalid_scope_cannot_advance(tmp_path, mode, ids):
    runtime = RecordingRuntime(_settings(tmp_path))
    store, _, outline_id = _waiting(runtime)
    with pytest.raises(ValueError):
        runtime.select_scope(store.run_id, outline_id, mode, ids, SECRET)
    assert store.read_state()["phase"] == "awaiting_scope_selection"
    assert runtime.dispatched == []
    runtime.close()


def test_unavailable_outline_offers_only_full_and_stale_preview_is_rejected(tmp_path):
    runtime = RecordingRuntime(_settings(tmp_path))
    store, _, outline_id = _waiting(runtime, available=False)
    with pytest.raises(ValueError, match="no reliable"):
        runtime.select_scope(store.run_id, outline_id, "selected", ["A"], SECRET)
    with pytest.raises(RuntimeError, match="stale"):
        runtime.select_scope(store.run_id, "art_" + "0" * 24, "full", [], SECRET)
    assert runtime.select_scope(store.run_id, outline_id, "full", [], SECRET)["status"] == "running"
    runtime.close()


def test_changed_raw_material_is_rejected(tmp_path):
    runtime = RecordingRuntime(_settings(tmp_path))
    store, manifest, outline_id = _waiting(runtime)
    store.artifact_path(manifest.raw_artifact_id).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="source changed"):
        runtime.select_scope(store.run_id, outline_id, "full", [], SECRET)
    assert store.read_state()["phase"] == "awaiting_scope_selection"
    runtime.close()


def test_committed_selection_cannot_change_after_state_write_crash(tmp_path):
    runtime = RecordingRuntime(_settings(tmp_path))
    store, _, outline_id = _waiting(runtime)
    old_state = store.read_state()
    runtime.select_scope(store.run_id, outline_id, "selected", ["A"], SECRET)
    # The receipt is durable, but the state replacement was interrupted.
    (store.root / "state.json").write_text(json.dumps(old_state), encoding="utf-8")
    restarted = RecordingRuntime(runtime.settings)
    with pytest.raises(RuntimeError, match="not waiting"):
        restarted.select_scope(store.run_id, outline_id, "selected", ["B"], SECRET)
    assert store.read_state()["phase"] == "scope_selected"
    assert restarted.select_scope(store.run_id, outline_id, "selected", ["A"], SECRET)["already_selected"]
    revisions = [commit["revision"] for commit in store.list_receipt_commits()]
    assert revisions == [1, 2, 3, 4]
    PipelineStateMachine(store).recover()
    runtime.close()
    restarted.close()


def test_historical_committed_nested_scope_can_only_resume_its_exact_choice(tmp_path):
    runtime = RecordingRuntime(_settings(tmp_path))
    store, _, outline_id = _waiting(runtime)
    selection = {"schema_version": "source-scope-selection.v1", "outline_artifact_id": outline_id,
                 "source_id": "SRC1", "mode": "selected", "section_ids": ["A1"]}
    artifact = store.put_json_artifact(selection, kind="source_scope_selection", source_id="SRC1")
    PipelineStateMachine(store).advance(RunPhase.SCOPE_SELECTED, _receipt(
        run_id=store.run_id, stage="scope_selection", fingerprint_value=selection,
        artifact_ids=(artifact.artifact_id,)))
    try:
        payload = runtime.source_scope_payload(store.run_id)
        assert [item["section_id"] for item in payload["sections"]] == ["A", "B"]
        assert payload["selection"]["section_ids"] == ["A1"]
        assert runtime.select_scope(store.run_id, outline_id, "selected", ["A1"], SECRET)["already_selected"]
        assert len(runtime.dispatched) == 1
        with pytest.raises(ValueError, match="non-first-level"):
            runtime.select_scope(store.run_id, outline_id, "selected", ["A11"], SECRET)
    finally:
        runtime.close()


def test_scope_api_refresh_recovery_and_validation(tmp_path, monkeypatch):
    runtime = RecordingRuntime(_settings(tmp_path))
    store, _, outline_id = _waiting(runtime)
    monkeypatch.setattr(main_module, "runtime", runtime)
    monkeypatch.setattr(main_module, "settings", runtime.settings)
    monkeypatch.setattr(main_module, "_resolve_api_key", lambda supplied, base_url: supplied)
    with TestClient(main_module.app) as client:
        payload = client.get(f"/api/runs/{store.run_id}").json()
        assert payload["state"]["requires_api_key"] is True
        assert payload["source_scope"]["outline_artifact_id"] == outline_id
        endpoint = f"/api/runs/{store.run_id}/select-scope"
        assert client.post(endpoint, json={"outline_artifact_id": outline_id, "mode": "selected"}).status_code == 422
        assert client.post(endpoint, json={"outline_artifact_id": outline_id, "mode": "full", "sections": []}).status_code == 422
        assert client.post(endpoint, json={"outline_artifact_id": "art_" + "0" * 24, "mode": "full"}).status_code == 409
        assert [item["section_id"] for item in payload["source_scope"]["sections"]] == ["A", "B"]
        for nested_id in ("A1", "A11"):
            rejected = client.post(endpoint, json={"outline_artifact_id": outline_id, "mode": "selected",
                                                   "section_ids": [nested_id], "api_key": SECRET})
            assert rejected.status_code == 422
            assert "first-level" in rejected.json()["detail"]
        assert client.post(endpoint, json={"outline_artifact_id": outline_id, "mode": "selected", "section_ids": ["A"],
                                          "api_key": SECRET}).status_code == 202
        assert client.get(f"/api/runs/{store.run_id}").json()["source_scope"]["selection"]["section_ids"] == ["A"]
    runtime.close()


def test_cancel_while_waiting_for_scope(tmp_path):
    runtime = RecordingRuntime(_settings(tmp_path))
    store, _, outline_id = _waiting(runtime)
    runtime.cancel(store.run_id)
    assert store.read_state()["phase"] == "cancelled"
    with pytest.raises(RuntimeError):
        runtime.select_scope(store.run_id, outline_id, "full", [], SECRET)
    runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", ["pdf", "html"])
@pytest.mark.parametrize("mode", ["full", "selected"])
async def test_acquired_preview_selection_to_frozen_corpus(tmp_path, monkeypatch, modality, mode):
    """Exercise the real acquisition, outline, scoped parser, validation and freeze."""
    runtime = RecordingRuntime(_settings(tmp_path))
    store = _initialize(runtime)
    if modality == "html":
        data = b'''<!doctype html><html><head><title>Manual</title></head><body><main>
        <h1>Manual</h1><nav class="contents"><ul><li><a href="#a">Chapter A</a></li>
        <li><a href="#b">Chapter B</a></li></ul></nav>
        <h2 id="a">Chapter A</h2><p>ALPHA_SELECTED_BODY.</p>
        <h2 id="b">Chapter B</h2><p>BETA_UNSELECTED_BODY.</p></main></body></html>'''
    else:
        document = fitz.open()
        page = document.new_page()
        page.insert_text((60, 80), "Chapter A", fontsize=20)
        page.insert_text((60, 120), "ALPHA_SELECTED_BODY.")
        page.insert_text((60, 350), "Chapter B", fontsize=20)
        page.insert_text((60, 390), "BETA_UNSELECTED_BODY.")
        document.set_toc([[1, "Chapter A", 1], [1, "Chapter B", 1]])
        data = document.tobytes()
        document.close()
    path = store.stage_upload("IN1", f"manual.{modality}", data)
    staged = StagedInput(input_id="IN1", kind="uploaded_file", display_name="Manual", staged_path=str(path))
    assert (await runtime.execute(store.run_id, TARGET, [staged], MODEL))["status"] == "waiting_for_input"
    payload = runtime.source_scope_payload(store.run_id)
    assert payload["outline_available"], payload
    assert payload["schema_version"] == "source-outline.v2"
    assert any("Chapter A" in section["title"] for section in payload["sections"])
    section_ids = [next(section["section_id"] for section in payload["sections"] if "Chapter A" in section["title"])] if mode == "selected" else []
    full_corpus_id = payload["reviewed_corpus_artifact_id"]
    full_corpus_bytes = store.artifact_path(full_corpus_id).read_bytes()
    runtime.select_scope(store.run_id, payload["outline_artifact_id"], mode, section_ids, SECRET)
    # Resume from disk after a process restart. Acquisition must not run again.
    restarted = PipelineRuntime(runtime.settings)
    monkeypatch.setattr(AcquisitionService, "acquire", lambda *args: pytest.fail("Do not reacquire on scope resume"))
    monkeypatch.setattr(pipeline_module, "normalize_source", lambda *args: pytest.fail("Do not parse on scope resume"))
    result = await restarted.execute(store.run_id, TARGET, [], MODEL)
    assert store.read_state()["phase"] == "corpus_frozen"
    content = "\n".join(item["content"] for item in result["evidence"])
    assert "ALPHA_SELECTED_BODY" in content
    assert ("BETA_UNSELECTED_BODY" in content) is (mode == "full")
    assert not any("unselected" in gap for source in result["source_results"] for gap in source["gaps"])
    manifest = load_persisted_manifests(store)[0]
    assert manifest.source_metadata["extraction_scope"]["mode"] == mode
    assert store.artifact_path(full_corpus_id).read_bytes() == full_corpus_bytes
    assert "BETA_UNSELECTED_BODY" in "\n".join(item["content"] for item in store.read_json_artifact(full_corpus_id)["evidence"])
    if mode == "selected":
        cache_store = _initialize(restarted, "run_scoped_cache")
        reused = restarted.import_frozen_corpus(
            run_id=cache_store.run_id, source_store=store,
            corpus_artifact_id=store.read_state()["corpus_snapshot_artifact_id"],
        )
        assert "BETA_UNSELECTED_BODY" not in "\n".join(item.content for item in reused.evidence)
        assert reused.source_results[0].manifest.source_metadata["extraction_scope"]["section_ids"] == section_ids
        assert cache_store.artifact_path(payload["outline_artifact_id"]).exists()
        normalized = store.read_json_artifact(result["source_results"][0]["corpus_artifact_id"])
        assert all("BETA_UNSELECTED_BODY" not in store.artifact_path(part["text_artifact_id"]).read_text(encoding="utf-8")
                   for part in normalized["parts"] if part.get("text_artifact_id"))
    runtime.close()
    restarted.close()


@pytest.mark.asyncio
async def test_missing_html_fragments_do_not_filter_chapters_and_all_pages_precede_selection(tmp_path, monkeypatch):
    runtime = RecordingRuntime(_settings(tmp_path))
    store = _initialize(runtime)
    root = "https://example.org/lean/"
    html = b'''<html><head><title>Lean 4</title></head><body><main><h1>Lean 4</h1>
    <nav class="contents"><ul><li><a href="Introduction.html#Intro">Introduction</a></li>
    <li><a href="Tactics.html#Nonexistent">Tactics</a></li></ul></nav></main></body></html>'''
    pages = {root + "Introduction.html": b"<main><h1>Introduction</h1><p>INTRODUCTION_BODY.</p></main>",
             root + "Tactics.html": b"<main><h1>Tactics</h1><p>TACTICS_BODY.</p></main>"}
    acquire = AcquisitionService.acquire
    def acquire_root(service, *args):
        batch = acquire(service, *args)
        for manifest in batch.manifests:
            manifest.final_url = root
            service.store.write_source_json(manifest.source_id, "manifest.json", manifest.model_dump(mode="json"))
        return batch
    calls = []
    def fetch(service, source_id, root_url, url):
        calls.append(url)
        data = pages[url]
        artifact = service.store.put_artifact(data, kind="raw_html_part", media_type="text/html", suffix=".html", source_id=source_id)
        return {"status": "succeeded", "artifact_id": artifact.artifact_id, "data": data, "final_url": url}
    monkeypatch.setattr(AcquisitionService, "acquire", acquire_root)
    monkeypatch.setattr(AcquisitionService, "fetch_html_child", fetch)
    path = store.stage_upload("IN1", "lean.html", html)
    try:
        paused = await runtime.execute(store.run_id, TARGET, [StagedInput(
            input_id="IN1", kind="uploaded_file", display_name="Lean 4", staged_path=str(path))], MODEL)
        assert paused["status"] == "waiting_for_input"
        assert set(calls) == set(pages)
        outline = runtime.source_scope_payload(store.run_id)
        assert outline["acquisition_coverage"]["page_count"] == 3
        assert outline["acquisition_coverage"]["acquisition_complete"]
        intro = next(item for item in outline["sections"] if item["title"].startswith("Introduction"))
        runtime.select_scope(store.run_id, outline["outline_artifact_id"], "selected", [intro["section_id"]], SECRET)
        monkeypatch.setattr(AcquisitionService, "fetch_html_child", lambda *args: pytest.fail("No crawl after selection"))
        result = await runtime.execute(store.run_id, TARGET, [], MODEL)
        text = "\n".join(item["content"] for item in result["evidence"])
        assert "INTRODUCTION_BODY" in text
        assert "TACTICS_BODY" not in text
    finally:
        runtime.close()
