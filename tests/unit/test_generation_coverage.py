from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.orchestration.pipeline as pipeline_module
from app.config import Settings
from app.generation_coverage import (
    generation_coverage_state, generation_limits, saved_generation_coverage_state,
    settings_for_generation_coverage,
)
from app.models import ModelRuntimeConfig, TargetProfile
from app.orchestration import PipelineRuntime
from app.storage import RunStore, StateProjectionMutationError


PROFILES = [('low', 8, 16), ('medium', 16, 24), ('high', 22, 28), ('ultra', 28, 36)]
TARGET = TargetProfile(software_name='Example', target_version='1', platform='General')
MODEL = ModelRuntimeConfig(base_url='https://example.invalid/v1', model='test-model', api_key='test-only-key')
FORM = {'software_name': 'Example', 'target_version': '1', 'platform': 'General',
        'api_base_url': MODEL.base_url, 'api_model': MODEL.model, 'api_key': MODEL.api_key}


class RecordingRuntime(PipelineRuntime):
    def __init__(self, settings):
        super().__init__(settings)
        self.dispatched = []

    def start(self, run_id, target_profile, inputs, model_config):
        self.dispatched.append(run_id)

    def start_reference_from_frozen(self, run_id, target_profile, model_config):
        self.dispatched.append(run_id)


@pytest.fixture
def api(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / 'data', output_dir=tmp_path / 'out',
                        max_task_candidates=5, max_knowledge_candidates=7)
    runtime = RecordingRuntime(settings)
    monkeypatch.setattr(main_module, 'runtime', runtime)
    monkeypatch.setattr(main_module, 'settings', settings)
    try:
        with TestClient(main_module.app) as client:
            yield client, runtime
    finally:
        runtime.close()


@pytest.mark.parametrize('level,tasks,references', PROFILES)
@pytest.mark.parametrize('modality', ['html', 'pdf'])
def test_api_saves_server_owned_limits_for_both_material_types(api, level, tasks, references, modality):
    client, runtime = api
    response = client.post('/api/runs', data={**FORM, 'source_mode': 'manual',
        'generation_coverage': level, 'max_task_candidates': '99999',
        'max_knowledge_candidates': '99999', 'generation_limits': '{"max_task_candidates":99999}'},
        files={'files': ('manual.' + modality, b'<h1>Manual</h1>' if modality == 'html' else b'%PDF-1.7',
                         'text/html' if modality == 'html' else 'application/pdf')})
    assert response.status_code == 202, response.text
    store = runtime.get_store(response.json()['run_id'])
    expected = generation_coverage_state(level)
    for payload in [store.read_state(), json.loads((store.root / 'request.json').read_text(encoding='utf-8'))]:
        assert payload['generation_coverage'] == level
        assert payload['generation_limits'] == {'max_task_candidates': tasks, 'max_knowledge_candidates': references}
    assert saved_generation_coverage_state(runtime.settings, store.read_state()) == expected
    assert generation_limits(runtime.settings) == {'max_task_candidates': 5, 'max_knowledge_candidates': 7}


def test_default_and_discovery_registration_use_medium_without_sending_a_number(api):
    client, runtime = api
    response = client.post('/api/runs', data={**FORM, 'source_mode': 'discovery'})
    assert response.status_code == 202, response.text
    state = runtime.get_store(response.json()['run_id']).read_state()
    assert state['generation_coverage'] == 'medium'
    assert state['generation_limits'] == {'max_task_candidates': 16, 'max_knowledge_candidates': 24}


@pytest.mark.parametrize('value', ['maximum', 'ULTRA', '29', 'mediem'])
def test_unknown_coverage_does_not_create_or_dispatch_a_run(api, value):
    client, runtime = api
    response = client.post('/api/runs', data={**FORM, 'source_mode': 'discovery', 'generation_coverage': value})
    assert response.status_code == 422
    assert runtime.dispatched == []
    assert not list(runtime.settings.data_dir.glob('runs/*/request.json'))


@pytest.mark.parametrize('level,tasks,references', PROFILES)
def test_coverage_and_limits_are_durable_and_cannot_be_changed_in_place(tmp_path, level, tasks, references):
    store = RunStore(tmp_path, 'run_coverage')
    saved = generation_coverage_state(level)
    store.initialize({'target_profile': TARGET.model_dump(mode='json'), **saved})
    reopened = RunStore(tmp_path, store.run_id)
    assert reopened.read_state()['generation_limits'] == saved['generation_limits']
    reopened.update_state(**saved)
    with pytest.raises(StateProjectionMutationError, match='fixed for this run'):
        reopened.update_state(generation_coverage='low' if level != 'low' else 'ultra')
    with pytest.raises(StateProjectionMutationError, match='fixed for this run'):
        reopened.update_state(generation_limits={'max_task_candidates': tasks + 1, 'max_knowledge_candidates': references})
    with pytest.raises(ValueError, match='do not match'):
        settings_for_generation_coverage(Settings(), {**saved, 'generation_limits': {'max_task_candidates': 999}})


def test_legacy_custom_limits_are_snapshotted_and_survive_changed_process_defaults(tmp_path):
    original = Settings(max_task_candidates=11, max_knowledge_candidates=19)
    legacy = RunStore(tmp_path, 'run_old')
    legacy.initialize({'target_profile': TARGET.model_dump(mode='json')})
    first = saved_generation_coverage_state(original, legacy.read_state())
    legacy.update_state(**first)
    restarted = settings_for_generation_coverage(Settings(max_task_candidates=28, max_knowledge_candidates=36),
                                                 RunStore(tmp_path, legacy.run_id).read_state())
    assert generation_limits(restarted) == {'max_task_candidates': 11, 'max_knowledge_candidates': 19}
    assert 'generation_coverage' not in legacy.read_state()


@pytest.mark.asyncio
async def test_concurrent_pipeline_runs_receive_isolated_saved_budgets(tmp_path, monkeypatch):
    shared = Settings(data_dir=tmp_path / 'data', output_dir=tmp_path / 'out',
                      max_task_candidates=99, max_knowledge_candidates=101)
    runtime = PipelineRuntime(shared)
    observed = {}

    class ProbeDiscovery:
        def __init__(self, *, store, settings, **kwargs):
            observed[store.run_id] = generation_limits(settings)

        async def discover(self, **kwargs):
            await asyncio.sleep(0)
            raise RuntimeError('Settings probe complete; no provider call')

    monkeypatch.setattr(pipeline_module, 'SourceDiscoveryService', ProbeDiscovery)
    try:
        runs = []
        for level, tasks, references in PROFILES:
            run_id = 'run_' + level
            store = runtime.create_store(run_id)
            store.initialize({'target_profile': TARGET.model_dump(mode='json'), 'model': MODEL.public_dict(),
                              'source_mode': 'discovery', **generation_coverage_state(level)})
            runs.append(runtime.execute(run_id, TARGET, [], MODEL))
        results = await asyncio.gather(*runs, return_exceptions=True)
        assert all(isinstance(result, RuntimeError) for result in results)
        assert observed == {'run_' + level: {'max_task_candidates': tasks, 'max_knowledge_candidates': references}
                            for level, tasks, references in PROFILES}
        assert generation_limits(runtime.settings) == {'max_task_candidates': 99, 'max_knowledge_candidates': 101}
        for level, tasks, references in PROFILES:
            store = RunStore(shared.data_dir, 'run_' + level)
            assert store.read_state()['generation_limits'] == observed[store.run_id]
    finally:
        runtime.close()


@pytest.mark.parametrize('source_level,override,expected', [
    ('high', None, 'high'), ('low', 'ultra', 'ultra'), (None, 'medium', 'medium'), (None, None, None),
])
def test_frozen_source_rerun_inherits_or_explicitly_selects_its_own_budget(api, monkeypatch,
                                                                       source_level, override, expected):
    client, runtime = api
    source = runtime.create_store('run_source')
    configuration = generation_coverage_state(source_level) if source_level else {
        'generation_limits': {'max_task_candidates': 11, 'max_knowledge_candidates': 19}}
    source.initialize({'target_profile': TARGET.model_dump(mode='json'), **configuration})
    source.update_state(t5_stage_complete=True, corpus_snapshot_artifact_id='art_source')
    frozen = SimpleNamespace(target_profile=TARGET, corpus_snapshot_id='corpus_source')
    monkeypatch.setattr(main_module, 'load_corpus_snapshot', lambda *args: frozen)

    def fake_import(*, run_id, source_store, corpus_artifact_id):
        runtime.get_store(run_id).update_state(corpus_snapshot_artifact_id='art_imported')
        return frozen

    monkeypatch.setattr(runtime, 'import_frozen_corpus', fake_import)
    response = client.post('/api/runs/run_source/rerun-reference',
                           data={**FORM, **({'generation_coverage': override} if override else {})})
    assert response.status_code == 202, response.text
    state = runtime.get_store(response.json()['run_id']).read_state()
    wanted = generation_coverage_state(expected) if expected else configuration
    assert state.get('generation_coverage') == expected
    assert state['generation_limits'] == wanted['generation_limits']
    assert source.read_state()['generation_limits'] == configuration['generation_limits']
    assert state['run_id'] != source.run_id


@pytest.mark.parametrize('level,tasks,references', PROFILES)
def test_selected_task_budget_is_enforced_at_the_pair_submission_boundary(tmp_path, level, tasks, references):
    from test_unified_task_trajectory import setup, render_pair
    from app.orchestration.unified_stage import UnifiedKnowledgeWorkspace

    store, settings, target, corpus, corpus_id, reference_id, *_ = setup(tmp_path)
    workspace = UnifiedKnowledgeWorkspace(
        store=store, settings=settings_for_generation_coverage(settings, generation_coverage_state(level)),
        target_profile=target, corpus=corpus, corpus_artifact_id=corpus_id, references=[],
        reference_artifact_id=reference_id, attempt_id='capacity-check')
    workspace.inspect_task_outline()
    workspace.read_task_node('N2')
    for number in range(tasks):
        workspace.submit_task_trajectory_pair(**render_pair(title=f'Render documented graph {number}'))
    assert workspace.inspect_pair_coverage()['remaining_task_capacity'] == 0
    with pytest.raises(ValueError, match='capacity reached'):
        workspace.submit_task_trajectory_pair(**render_pair(title='One graph beyond the selected budget'))


@pytest.mark.parametrize('level,tasks,references', PROFILES)
def test_selected_reference_budget_rejects_an_over_capacity_review(tmp_path, level, tasks, references):
    from test_reference_review_submission import _workspace, _request

    workspace = _workspace(tmp_path, catalog_count=references + 1)
    workspace.settings = settings_for_generation_coverage(workspace.settings, generation_coverage_state(level))
    with pytest.raises(ValueError, match='capacity'):
        workspace.submit_reference_review(**_request(workspace))
    assert workspace.submitted_references == []
