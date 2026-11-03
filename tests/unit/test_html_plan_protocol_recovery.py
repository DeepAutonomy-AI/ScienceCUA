"""Malformed planner output must be repaired with context or fail audibly."""
import json
from dataclasses import replace

import httpx
import pytest

from app.config import Settings
from app.html_guided.model import HtmlPlanRuntime, HtmlPlanTruncatedError
from app.orchestration.html_guided_stage import normalize_guided_html_source, parse_plan_response
from app.orchestration.source_stage import SourceStageError
from app.runtime import AgentTurnResult
from test_html_guided_model import SECRET, invocation
from test_html_guided_stage import FAKE_KEY, PlanningRuntime, stage_inputs


@pytest.mark.parametrize('text', [
    '{"links": [], "links": [{"link_id":"invented"}]}',
    '{"links":[], "nested":{"x":1,"x":2}}',
    '{"temperature":NaN}', '{"temperature":Infinity}',
    'Plan: {"links":[]}', '{"links":[]}\n{"other":true}',
    '```python\n{"links":[]}\n```',
    '```json\n{"links":[]}\n```\nExtra commentary',
])
def test_complete_json_parser_rejects_ambiguous_or_untrusted_wrappers(text):
    with pytest.raises((ValueError, TypeError)):
        parse_plan_response(text)


@pytest.mark.parametrize('text', ['\ufeff {"links":[]}', '```\r\n{"links":[]}\r\n```',
                                '```JSON\n{"links":[]}\n```'])
def test_complete_json_parser_accepts_only_whole_object_or_one_json_fence(text):
    assert parse_plan_response(text) == {'links': []}


def outputs(store):
    found = []
    for path in (store.root / 'artifacts/meta').glob('*.json'):
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta['kind'] == 'html_guided_model_output':
            found.append(store.read_json_artifact(meta['artifact_id']))
    return found


@pytest.mark.asyncio
async def test_json_repair_explicitly_retains_page_ids_scope_and_error_without_runtime_history(tmp_path, monkeypatch):
    store, settings, acquisition, target, manifest, config = stage_inputs(tmp_path, monkeypatch)
    runtime = PlanningRuntime(responses=['Not a JSON object; reflected key ' + FAKE_KEY])
    prepared = await normalize_guided_html_source(store, settings, acquisition, target, manifest,
        runtime=runtime, model_config=config)
    assert prepared.workspace.inventory_completed
    assert len(runtime.invocations) == 2
    repair = runtime.invocations[1]
    original = json.loads(runtime.invocations[0].prompt.rsplit('\n', 1)[-1])
    repaired = json.loads(repair.prompt.rsplit('\n', 1)[-1])
    assert repaired == original
    assert 'required_response_shape' in repair.prompt
    assert 'Invalid JSON at line 1, column 1' in repair.prompt
    assert runtime.invocations[0].session_id == repair.session_id
    evidence = outputs(store)
    assert len(evidence) == 1
    assert '[REDACTED]' in evidence[0]['response_text']
    assert evidence[0]['finish_reason'] == 'completed'
    assert evidence[0]['input_artifact_id']
    assert all(FAKE_KEY.encode() not in p.read_bytes() for p in store.root.rglob('*') if p.is_file())


@pytest.mark.asyncio
async def test_two_invalid_responses_are_bounded_and_both_preserved(tmp_path, monkeypatch):
    store, settings, acquisition, target, manifest, config = stage_inputs(tmp_path, monkeypatch)
    runtime = PlanningRuntime(responses=['not JSON one', 'not JSON two'])
    with pytest.raises(SourceStageError) as caught:
        await normalize_guided_html_source(store, settings, acquisition, target, manifest,
            runtime=runtime, model_config=config)
    assert caught.value.category == 'html_guided_plan_invalid'
    assert len(runtime.invocations) == 2
    assert len(outputs(store)) == 2
    assert store.list_attempts()[0].status == 'failed'


@pytest.mark.asyncio
async def test_runtime_length_result_is_distinct_from_invalid_json_and_never_accepted(tmp_path, monkeypatch):
    store, settings, acquisition, target, manifest, config = stage_inputs(tmp_path, monkeypatch)
    class TruncatedRuntime(PlanningRuntime):
        async def run(self, request):
            self.invocations.append(request)
            return AgentTurnResult(request.session_id, '{"key": "' + FAKE_KEY, finish_reason='length')
    runtime = TruncatedRuntime()
    with pytest.raises(SourceStageError) as caught:
        await normalize_guided_html_source(store, settings, acquisition, target, manifest,
            runtime=runtime, model_config=config)
    assert caught.value.category == 'html_guided_plan_truncated'
    assert len(runtime.invocations) == 1
    audit = outputs(store)[0]
    assert audit['parse_error'] == 'output_token_limit'
    assert audit['finish_reason'] == 'length'
    assert FAKE_KEY not in audit['response_text']


@pytest.mark.asyncio
@pytest.mark.parametrize('protocol', ['chat_completions', 'responses'])
async def test_direct_token_limit_has_safe_auditable_category_without_retry(protocol):
    requests = []
    text = '{"key":"' + SECRET
    def serve(request):
        requests.append(request)
        body = json.loads(request.content)
        messages = body.get('messages', body.get('input'))
        assert messages[0]['role'] == 'system'
        if protocol == 'responses':
            payload = {'status':'incomplete', 'incomplete_details': {'reason':'max_output_tokens'},
                       'output_text':text}
        else:
            payload = {'choices':[{'message':{'content':text}, 'finish_reason':'length'}]}
        return httpx.Response(200, json=payload)
    runtime = HtmlPlanRuntime(Settings(), transport=httpx.MockTransport(serve))
    try:
        with pytest.raises(HtmlPlanTruncatedError) as caught:
            await runtime.run(invocation(protocol=protocol, base_url='https://generic.example/v1'))
        assert caught.value.category == 'html_guided_plan_truncated'
        assert SECRET not in caught.value.response_text
        assert '[REDACTED]' in caught.value.response_text
        assert len(requests) == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_self_contained_repair_does_not_repeat_the_large_snapshot_in_history():
    bodies = []
    def serve(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]})
    runtime = HtmlPlanRuntime(Settings(), transport=httpx.MockTransport(serve))
    first = invocation(prompt='A long original snapshot', base_url='https://generic.example/v1')
    try:
        await runtime.run(first)
        await runtime.run(replace(first, prompt='Repair plus the same original snapshot',
                                  metadata={'html_plan_repair': True}))
        assert len(bodies[1]['messages']) == 2
        assert bodies[1]['messages'][0]['role'] == 'system'
        assert bodies[1]['messages'][1]['content'] == 'Repair plus the same original snapshot'
    finally:
        await runtime.close()
