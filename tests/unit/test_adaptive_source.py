from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
import json
from pathlib import Path

import fitz
import pytest

pytestmark = pytest.mark.integration

from app.acquisition import AcquisitionService
from app.adaptive_source import SourceContext, validate_output, write_json
from app.config import Settings
from app.models import CorpusSnapshot, ModelRuntimeConfig, StagedInput, TargetProfile
from app.orchestration.pipeline import PipelineRuntime
from app.orchestration.source_stage import normalize_source, validate_source_results
from app.orchestration.adaptive_source_stage import ADAPTIVE_SOURCE_PROMPT_VERSION, materialize_adaptive
from app.runtime import AgentInvocation, AgentRuntime, AgentTurnResult
from app.runtime.dsh_client import DshBridgeConfig
from app.source_extraction_runner import run_extraction
from app.storage import RunStore
from tests.knowledge_fixtures import trajectory_submission


HTML = b'''<html><body><header>Important startup instruction</header>
<div class="features"><section class="feature"><h6>Import</h6><p>Choose File &gt; Import.</p></section>
<section class="feature"><h5>Export</h5><pre>  save --format png\n  confirm</pre></section></div>
<article><p>Additional body region</p></article></body></html>'''

SCRIPT = '''
def extract(ctx):
    units = ctx.units()
    groups = [ctx.select_html('r1', '.feature:nth-of-type(1)'), ctx.select_html('r1', '.feature:nth-of-type(2)')]
    grouped = {u['unit_key'] for group in groups for u in group}
    rest = [u for u in units if u['unit_key'] not in grouped]
    parts = []
    nodes = [{'node_key': 'document', 'title': 'Guide', 'title_origin': 'derived', 'node_type': 'document', 'part_keys': []}]
    all_groups = [[u] for u in rest] + groups
    for group in sorted(all_groups, key=lambda g: g[0]['ordinal']):
        key = 'p' + str(len(parts))
        parts.append({'part_key': key, 'label': group[0]['text'].strip(), 'ordinal': len(parts),
                      'source_refs': [ctx.ref(u) for u in group]})
        nodes.append({'node_key': 'n' + str(len(parts)), 'title': group[0]['text'].strip(),
                      'parent_node_key': 'document', 'part_keys': [key]})
    return {'source_id': ctx.assignment['source_id'], 'parts': parts, 'tree_nodes': nodes}
'''

ALL_SCRIPT = '''
def extract(ctx):
    units = [unit for resource in ctx.resources() for unit in ctx.units(resource['resource_key'])]
    return {'source_id': ctx.assignment['source_id'], 'parts': [
        {'part_key': 'body', 'label': 'Body', 'ordinal': 0,
         'source_refs': [ctx.ref(u) for u in units if u['kind'] == 'text'],
         'image_refs': [ctx.ref(u) for u in units if u['kind'] == 'image']}],
        'tree_nodes': [{'node_key': 'document', 'title': 'Guide', 'title_origin': 'derived',
                        'node_type': 'document', 'part_keys': ['body']}]}
'''


def setup_source(tmp_path, data=HTML, *, pdf=False, max_pages=0):
    settings = Settings(data_dir=tmp_path / 'data', output_dir=tmp_path / 'out',
                        max_html_pages=max_pages, t5_processing_mode='agent',
                        t6_reference_enabled=False, t6_task_enabled=False)
    store = RunStore(settings.data_dir, 'run_adaptive')
    target = TargetProfile(software_name='ExampleCalc', target_version='1.0', platform='General', ui_locale='English')
    model = ModelRuntimeConfig(base_url='https://model.invalid/v1', model='fake', api_key='fake')
    path = store.stage_upload('input_1', 'guide.pdf' if pdf else 'guide.html', data)
    staged = StagedInput(input_id='input_1', kind='uploaded_file', display_name=path.name, staged_path=str(path),
                         declared_media_type='application/pdf' if pdf else 'text/html')
    store.initialize({'run_id': store.run_id, 'target_profile': target.model_dump(mode='json'),
                      'inputs': [staged.model_dump(mode='json')], 'model': model.public_dict()})
    acquisition = AcquisitionService(store, settings)
    manifest = acquisition.acquire([staged], target).manifests[0]
    prepared = normalize_source(store, settings, acquisition, target, manifest)
    return settings, store, prepared, target, model, staged


def execute(prepared, script=SCRIPT):
    assignment = prepared.adaptive_assignment
    path = assignment.parent / 'work' / 'extract.py'
    path.write_text(script, encoding='utf-8')
    output = assignment.parent / 'candidate' / 'extraction.json'
    report = run_extraction(assignment, path, output)
    return report, json.loads(output.read_text(encoding='utf-8'))


def test_preparation_defers_semantic_parts_and_retains_all_heading_and_body_regions(tmp_path):
    _, _, prepared, *_ = setup_source(tmp_path)
    assert prepared.workspace.parts == {}
    assert prepared.workspace.evidence_tree is None
    units = SourceContext(prepared.adaptive_assignment).units()
    assert any('Important startup' in u['text'] for u in units)
    assert any('Additional body region' in u['text'] for u in units)
    assert {'h5', 'h6'} <= {ancestor['tag'] for u in units for ancestor in u['locator']['ancestors']}
    report, raw = execute(prepared)
    assert report['valid'], report
    assert len(raw['tree_nodes']) == 5  # Two features and the two separate surrounding regions.


@pytest.mark.parametrize('fault,code', [('omit', 'source_unit_unaccounted_for'), ('duplicate', 'overlapping_source_ownership'),
    ('unknown', 'unknown_source_unit'), ('orphan', 'tree_part_ownership'), ('cycle', 'invalid_tree_shape')])
def test_host_rejects_provenance_coverage_and_tree_errors(tmp_path, fault, code):
    _, _, prepared, *_ = setup_source(tmp_path)
    report, raw = execute(prepared)
    assert report['valid']
    if fault == 'omit':
        raw['parts'][0]['source_refs'].pop()
    elif fault == 'duplicate':
        raw['parts'][0]['source_refs'].append(copy.deepcopy(raw['parts'][0]['source_refs'][0]))
    elif fault == 'unknown':
        raw['parts'][0]['source_refs'][0]['unit_key'] = 'missing'
    elif fault == 'orphan':
        raw['tree_nodes'][1]['part_keys'] = []
    else:
        raw['tree_nodes'][1]['parent_node_key'] = raw['tree_nodes'][1]['node_key']
    rejected, _ = validate_output(SourceContext(prepared.adaptive_assignment), raw)
    assert not rejected['valid']
    assert code in {item['code'] for item in rejected['issues']}


def test_raw_snapshot_mutation_is_detected(tmp_path):
    _, _, prepared, *_ = setup_source(tmp_path)
    _, raw = execute(prepared)
    ctx = SourceContext(prepared.adaptive_assignment)
    ctx.raw_path().write_text('<p>replacement</p>', encoding='utf-8')
    report, _ = validate_output(SourceContext(prepared.adaptive_assignment), raw)
    assert not report['valid']
    assert report['issues'][0]['code'] == 'source_snapshot_changed'


def test_interleaved_parts_are_rejected_and_code_whitespace_is_preserved(tmp_path):
    _, _, prepared, *_ = setup_source(tmp_path, b'<pre>  <b>first</b>\n  <b>second</b></pre><p>Last</p>')
    ctx = SourceContext(prepared.adaptive_assignment)
    units = ctx.units()
    assert units[0]['text'] == '  '
    assert units[2]['text'] == '\n  '
    report, raw = execute(prepared, ALL_SCRIPT)
    assert report['valid'], report
    raw['parts'][0]['source_refs'] = [ctx.ref(units[0]), ctx.ref(units[-1])]
    raw['parts'].append({'part_key': 'middle', 'label': 'Middle', 'ordinal': 1,
                         'source_refs': [ctx.ref(u) for u in units[1:-1]]})
    raw['tree_nodes'][0]['part_keys'].append('middle')
    report, _ = validate_output(SourceContext(prepared.adaptive_assignment), raw)
    assert 'parts_out_of_source_order' in {item['code'] for item in report['issues']}


def test_pdf_exact_text_and_image_survive_existing_source_validation(tmp_path):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), 'Open the image and choose Export.')
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16))
    pixmap.clear_with(255)
    page.insert_image(fitz.Rect(72, 100, 104, 132), stream=pixmap.tobytes('png'))
    data = document.tobytes()
    document.close()
    _, store, prepared, *_ = setup_source(tmp_path, data, pdf=True)
    report, raw = execute(prepared, ALL_SCRIPT)
    assert report['valid'], report
    _, output = validate_output(SourceContext(prepared.adaptive_assignment, replay=True), raw)
    result = materialize_adaptive(prepared, output, report)
    accepted, summaries = validate_source_results(store, [result])
    assert accepted[0].status == 'succeeded', summaries
    assert accepted[0].evidence[0].image_refs
    assert 'Open the image' in accepted[0].evidence[0].content


class ScriptRuntime(AgentRuntime):
    def __init__(self, *, script=SCRIPT, repair=False):
        self.script = script
        self.repair = repair
        self.invocations = []
        self.closed_sessions = []

    async def run(self, invocation):
        self.invocations.append(invocation)
        assert invocation.allowed_tools == ()
        assert 'capability' not in invocation.metadata
        assignment = Path(invocation.metadata['assignment_path'])
        script = assignment.parent / 'work' / 'extract.py'
        script.write_text(self.script, encoding='utf-8')
        candidate = assignment.parent / 'candidate' / 'extraction.json'
        report = await asyncio.to_thread(run_extraction, assignment, script, candidate)
        if self.repair and len(self.invocations) == 1:
            raw = json.loads(candidate.read_text(encoding='utf-8'))
            raw['parts'][0]['source_refs'].pop()
            write_json(candidate, raw)  # Host must reject the stale, previously passing report.
        return AgentTurnResult(session_id=invocation.session_id, final_response='done', finish_reason='completed')

    async def cancel(self, session_id):
        return True

    async def close_session(self, session_id):
        self.closed_sessions.append(session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize('repair', [False, True])
async def test_default_pipeline_requires_host_validation_replay_and_canonical_freeze(tmp_path, repair):
    settings, store, _, target, model, staged = setup_source(tmp_path)
    driver = ScriptRuntime(repair=repair)
    runtime = PipelineRuntime(settings, driver)
    try:
        snapshot = await runtime.execute(store.run_id, target, [staged], model)
    finally:
        runtime.close()
    assert snapshot['evidence_trees']
    assert len(driver.invocations) == (2 if repair else 1)
    assert store.read_state()['phase'] == 'corpus_frozen'
    attempt = store.list_attempts()[-1]
    assert attempt.status == 'succeeded'
    capsule = store.read_json_artifact(attempt.capsule_artifact_id)
    assert capsule['prompt_version'] == ADAPTIVE_SOURCE_PROMPT_VERSION
    result = snapshot['source_results'][0]
    assert result['status'] == 'succeeded'
    assert all(e['text_artifact_id'] for e in result['evidence'])


@pytest.mark.asyncio
async def test_nondeterministic_script_cannot_complete_stage(tmp_path):
    settings, store, _, target, model, staged = setup_source(tmp_path)
    script = ALL_SCRIPT.replace("'title': 'Guide'", "'title': str(__import__('uuid').uuid4())")
    driver = ScriptRuntime(script=script)
    runtime = PipelineRuntime(settings, driver)
    try:
        with pytest.raises(Exception, match='replayable validated result'):
            await runtime.execute(store.run_id, target, [staged], model)
    finally:
        runtime.close()
    assert store.read_state()['phase'] == 'failed'
    assert store.list_attempts()[-1].status == 'failed'
    assert not (store.root / 'sources' / 'SRC1' / 'result.json').exists()


def test_script_timeout_returns_feedback_without_accepting_a_candidate(tmp_path):
    _, _, prepared, *_ = setup_source(tmp_path)
    config = json.loads(prepared.adaptive_assignment.read_text(encoding='utf-8'))
    config['execution_timeout_seconds'] = 0.3
    write_json(prepared.adaptive_assignment, config)
    script = prepared.adaptive_assignment.parent / 'work' / 'extract.py'
    script.write_text('import time\ndef extract(ctx):\n    time.sleep(30)\n', encoding='utf-8')
    output = prepared.adaptive_assignment.parent / 'candidate' / 'extraction.json'
    report = run_extraction(prepared.adaptive_assignment, script, output)
    assert not report['valid']
    assert report['issues'][0]['code'] == 'extraction_execution_failed'
    assert not output.exists()


@pytest.mark.asyncio
async def test_source_cancellation_does_not_commit_evidence(tmp_path):
    settings, store, _, target, model, staged = setup_source(tmp_path)
    entered = asyncio.Event()

    class WaitingRuntime(ScriptRuntime):
        async def run(self, invocation):
            self.invocations.append(invocation)
            entered.set()
            await asyncio.Event().wait()

        async def cancel(self, session_id):
            self.cancelled_session = session_id
            return True

    driver = WaitingRuntime()
    runtime = PipelineRuntime(settings, driver)
    task = asyncio.create_task(runtime.execute(store.run_id, target, [staged], model))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        runtime.close()
    assert store.read_state()['phase'] == 'cancelled'
    assert store.list_attempts()[-1].status == 'cancelled'
    assert driver.cancelled_session == driver.invocations[0].session_id
    assert driver.closed_sessions == [driver.cancelled_session]
    assert not (store.root / 'sources' / 'SRC1' / 'result.json').exists()


def test_native_tools_are_source_scoped_and_downstream_stays_capability_only(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    config = DshBridgeConfig()
    base = dict(run_id='run_test', session_id='child_test', prompt='extract', model={'model': 'fake'})
    payload = config.worker_payload(AgentInvocation(role='source_review', **base,
        metadata={'native_source_tools': True, 'source_workspace': str(root)}))
    assert payload['cwd'] == str(root.resolve())
    assert payload['patches'][-1].endswith('adaptive-source.patch.yml')
    ordinary = config.worker_payload(AgentInvocation(role='trajectory_inference', **base))
    assert ordinary['patches'][-1].endswith('business-tools.patch.yml')
    with pytest.raises(Exception, match='only available'):
        config.worker_payload(AgentInvocation(role='skill_induction', **base,
            metadata={'native_source_tools': True, 'source_workspace': str(root)}))


def test_child_acquisition_is_cached_bounded_and_replayable(tmp_path, monkeypatch):
    _, _, prepared, *_ = setup_source(tmp_path, max_pages=1)
    assignment = prepared.adaptive_assignment
    config = json.loads(assignment.read_text(encoding='utf-8'))
    config['origin_url'] = 'https://manual.example/docs/index.html'
    write_json(assignment, config)
    state = SourceContext(assignment).state()
    state['resources'][0]['url'] = config['origin_url']
    write_json(assignment.parent / 'resources.json', state)
    calls = []

    def download(self, url):
        calls.append(url)
        return b'<h6>Child topic</h6><p>Exact child content.</p>', 'text/html', url

    monkeypatch.setattr(SourceContext, '_download', download)
    ctx = SourceContext(assignment)
    assert ctx.fetch_html('child.html') == 'r2'
    assert ctx.fetch_html('child.html') == 'r2'
    assert ctx.fetch_html('other.html') is None
    assert len(calls) == 1
    assert SourceContext(assignment, replay=True).fetch_html('child.html') == 'r2'
    assert SourceContext(assignment, replay=True).fetch_html('other.html') is None
    with pytest.raises(ValueError, match='outside'):
        ctx.fetch_html('../outside.html')
    report, raw = execute(prepared, ALL_SCRIPT)
    assert report['valid'], report
    assert any(gap['code'] == 'budget_exhausted' for gap in report['gaps'])


def test_native_html_observations_resolve_base_and_report_link_roles(tmp_path, monkeypatch):
    html = b'''<html><head><base href="../"></head><body><main>
    <h1>Intro <span class="permalink-widget"><a title="Permalink" href="find/?name=intro">link</a></span></h1>
    <p>See <a href="Theory/#types">the theory chapter</a>.</p>
    <ul class="section-toc"><li><a href="find/">Finding Data</a></li></ul>
    </main></body></html>'''
    _, _, prepared, *_ = setup_source(tmp_path, html, max_pages=2)
    assignment = prepared.adaptive_assignment
    config = json.loads(assignment.read_text(encoding='utf-8'))
    config['origin_url'] = 'https://manual.example/book/'
    write_json(assignment, config)
    ctx = SourceContext(assignment)
    state = ctx.state()
    state['resources'][0]['url'] = 'https://manual.example/book/Intro/'
    write_json(assignment.parent / 'resources.json', state)
    links = ctx.links()
    assert [link['link_role'] for link in links] == ['heading_permalink', 'cross_reference', 'directory_entry']
    assert links[0]['url'] == 'https://manual.example/book/find/?name=intro'
    assert links[1]['url'] == 'https://manual.example/book/Theory/'
    calls = []

    def download(self, url):
        calls.append(url)
        return b'<html><body><p>Search instructions.</p></body></html>', 'text/html', url

    monkeypatch.setattr(SourceContext, '_download', download)
    assert ctx.fetch_html('find/') == 'r2'
    assert ctx.fetch_html('find/') == 'r2'
    assert calls == ['https://manual.example/book/find/']


@pytest.mark.asyncio
async def test_adaptive_source_reaches_reference_task_trajectory_skill_and_export(tmp_path):
    from test_t6_reference_stage import ReferenceDrivingRuntime

    html = b'''<h1>ExampleCalc Manual</h1><h6>Keyboard shortcuts</h6>
    <p>Press Ctrl+R to render the graph.</p><h6>Supported modes</h6>
    <table><tr><th>Mode</th><th>Meaning</th></tr>
    <tr><td>Standard</td><td>Regular plotting.</td></tr>
    <tr><td>Detailed</td><td>Detailed plotting.</td></tr></table>'''
    settings, store, _, target, model, staged = setup_source(tmp_path, html)
    settings = replace(settings, t6_reference_enabled=True, t6_task_enabled=True)

    class FullRuntime(ReferenceDrivingRuntime):
        def __init__(self, gateway):
            super().__init__(gateway)
            script = ALL_SCRIPT.replace("'label': 'Body'", "'label': 'Keyboard shortcuts'")
            # This end-to-end fixture supplies an authored first-level chapter,
            # as required by the current Task category contract.
            script = script.replace(
                "'part_keys': ['body']}]}",
                "'part_keys': []}, {'node_key': 'shortcuts', 'title': 'Keyboard shortcuts', "
                "'title_origin': 'source', 'node_type': 'section', "
                "'parent_node_key': 'document', 'part_keys': ['body']}]}",
            )
            self.extractor = ScriptRuntime(script=script)

        async def run(self, invocation):
            if invocation.role == 'source_review':
                self.invocations.append(invocation)
                return await self.extractor.run(invocation)
            if invocation.role == 'reference_build':
                return await super().run(invocation)
            self.invocations.append(invocation)
            assert not invocation.metadata.get('native_source_tools')
            state = store.read_state()
            corpus = store.read_json_artifact(state['corpus_snapshot_artifact_id'])
            evidence = next(e for e in corpus['evidence'] if 'Press Ctrl+R' in e['content'])
            citation = {'evidence_id': evidence['evidence_id'], 'quote': 'Press Ctrl+R to render the graph.'}
            step = {'ordinal': 1, 'instruction': citation['quote'], 'expected_observation': 'The graph is rendered.', 'citations': [citation]}
            if invocation.role == 'task_trajectory':
                self._invoke(invocation, 'inspect_knowledge_scope', {})
                outline = self._invoke(invocation, 'inspect_task_outline', {})
                chapter = outline['items'][0]['root']['children'][0]
                self._invoke(invocation, 'read_task_node', {'node_id': chapter['read_node_id']})
                submitted = self._invoke(invocation, 'submit_task_trajectory_pair', {
                    'task': {'task': 'Render a graph', 'category': chapter['title'],
                             'citations': [citation]},
                    'trajectory': trajectory_submission(
                        instruction=step['instruction'], citations=step['citations'],
                        expected_observation=step['expected_observation']),
                })
                assert submitted['accepted'], submitted
            elif invocation.role == 'skill_induction':
                batch = json.loads(str(invocation.prompt).split('\n\nDISPATCHED BATCH\n', 1)[1])
                trajectory = batch['trajectories'][0]
                submitted = self._invoke(invocation, 'submit_skill_batch', {'skills': [{
                    'title': 'Render a graph', 'description': 'Render an ExampleCalc graph using its documented shortcut.',
                    'applicability': 'Use when an ExampleCalc graph is ready to render.',
                    'preconditions': 'The graph view is open.',
                    'procedure': [{key: value for key, value in step.items() if key != 'expected_observation'}],
                    'trajectory_ids': [trajectory['trajectory_id']]
                }]})
                assert submitted['accepted_by_schema'] == 1, submitted
            else:
                raise AssertionError(invocation.role)
            terminal = self._invoke(invocation, 'submit_stage_result', {'status': 'succeeded', 'summary': 'Grounded result.'})
            assert terminal['accepted'], terminal
            return AgentTurnResult(invocation.session_id, 'done', 'completed')

    from app.capabilities import CapabilityGateway
    gateway = CapabilityGateway()
    driver = FullRuntime(gateway)
    runtime = PipelineRuntime(settings, driver, capability_gateway=gateway)
    try:
        result = await runtime.execute(store.run_id, target, [staged], model)
    finally:
        runtime.close()
    assert store.read_state()['phase'] == 'completed', store.read_state()
    assert {'source_review', 'reference_build', 'task_trajectory', 'skill_induction'} <= {
        invocation.role for invocation in driver.invocations
    }
    assert result['skills']
    assert any(settings.output_dir.rglob('SKILL.md'))
