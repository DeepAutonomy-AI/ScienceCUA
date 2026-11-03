import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from app.capabilities import CapabilityGateway, CapabilityInvocation
from app.config import Settings
from app.models import ModelRuntimeConfig, TargetProfile
from app.orchestration.source_discovery_stage import (SourceDiscoveryProtocolError,
    SourceDiscoveryService, SourceDiscoveryToolExecutor)
from app.runtime import AgentRuntime, AgentRuntimeBusinessCompleted, AgentRuntimeTimeoutError, AgentTurnResult
from app.source_discovery_contracts import SOURCE_DISCOVERY_TOOLS, SourceCandidateSubmission
from app.source_candidate_probe import SourceCandidateProbe
from app.storage import RunStore


HTML_MANUAL = b"""<!doctype html><html lang="en"><head><title>Example 1.0 Manual</title></head>
<body><main><h1>Example 1.0 Manual</h1><p>This manual teaches all core operations.</p>
<ol><li><a href="loading.html">Loading images</a></li>
<li><a href="analysis.html">Analyzing images</a></li></ol></main></body></html>"""


def html_manual_response(request):
    assert request.headers.get("authorization") is None
    return httpx.Response(200, content=HTML_MANUAL, headers={"content-type": "text/html"})


@pytest.fixture(autouse=True)
def local_discovery_html_transport(monkeypatch):
    """Exercise the real probe and scope gate without contacting any website."""
    import app.orchestration.source_discovery_stage as stage

    def probe(*args, **kwargs):
        return SourceCandidateProbe(*args, **kwargs, transport=httpx.MockTransport(html_manual_response))

    monkeypatch.setattr(stage, "SourceCandidateProbe", probe)


def candidate(**changes):
    return {"title": "Example 1.0 Manual", "url": "https://example.org/manual/1.0/",
        "modality": "html", "source_type": "official", "document_version": "1.0",
        "version_relation": "exact", "version_evidence": "Official 1.0 documentation directory.",
        "language": "English", "platform": "General", "root_assessment": "Contents lists all chapters.",
        "reason": "Exact official manual", "html_scope": {
            "kind": "user_manual", "scope_summary": "Complete manual for the application's core operations.",
            "entry_evidence_ids": ["H1", "P1"],
            "included_sections": ["Loading images", "Analyzing images"], "excluded_sections": [],
            "comparisons": [], "comparison_note": "No distinct broader or narrower learning entry is established by these links.",
            "limitations": [],
        }, **changes}


def setup_discovery(tmp_path):
    store = RunStore(tmp_path, "run_discovery")
    target = TargetProfile(software_name="Example", target_version="1.0", platform="General", ui_locale="English")
    model = ModelRuntimeConfig(model="gpt-test", base_url="https://example.org/v1", api_key="model-test-key")
    store.initialize({"target_profile": target.model_dump(), "inputs": [], "model": model.public_dict()})
    return store, target, model


class SubmittingRuntime(AgentRuntime):
    def __init__(self, gateway, payload, linger=False):
        self.gateway, self.payload, self.linger = gateway, payload, linger
        self.invocations, self.stopped = [], []
        self.inspections = []
        self.ended = asyncio.Event()

    async def run(self, invocation):
        self.invocations.append(invocation)
        raw = invocation.metadata["capability"]
        for index, item in enumerate(self.payload.get("candidates", [])):
            if item["modality"] == "html":
                inspected = self.gateway.invoke(raw["capability_token"], CapabilityInvocation(
                    **raw["context"], tool_name="inspect_source_candidate", arguments={"url": item["url"]},
                    call_id=f"inspect_submit_{index}"))
                assert inspected["ok"] and inspected["status"] == "available"
                assert inspected["html_snapshot"]["main_content"]["evidence_id"] == "P1"
                self.inspections.append(inspected)
        result = self.gateway.invoke(raw["capability_token"], CapabilityInvocation(
            **raw["context"], tool_name="submit_source_candidates", arguments=self.payload, call_id="submit_1"))
        assert result["ok"]
        if self.linger:
            await self.ended.wait()
            raise AgentRuntimeBusinessCompleted("Business result accepted")
        return AgentTurnResult(invocation.session_id, "done", "completed", usage={"tokens": 2})

    async def cancel(self, session_id):
        self.ended.set()
        return True

    async def stop_completed_session(self, session_id):
        self.stopped.append(session_id)
        self.ended.set()
        return True


@pytest.mark.asyncio
async def test_search_deadline_remains_live_during_a_blocked_candidate_inspection(tmp_path, monkeypatch):
    import threading
    import time
    import app.orchestration.source_discovery_stage as stage

    store, target, model = setup_discovery(tmp_path)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    class BlockedProbe:
        def __init__(self, *args, **kwargs):
            self.html_probe = SourceCandidateProbe(*args, **kwargs,
                transport=httpx.MockTransport(html_manual_response))
        def inspect(self, url):
            if url == candidate()["url"]:
                return self.html_probe.inspect(url)
            entered.set()
            release.wait(4)
            return {'status': 'failed', 'url': url, 'error': 'timeout'}
    monkeypatch.setattr(stage, 'SourceCandidateProbe', BlockedProbe)
    gateway = CapabilityGateway()

    class InspectingRuntime(SubmittingRuntime):
        async def run(self, invocation):
            raw = invocation.metadata['capability']
            def invoke(name, arguments, call_id):
                return self.gateway.invoke(raw['capability_token'], CapabilityInvocation(
                    **raw['context'], tool_name=name, arguments=arguments, call_id=call_id))
            assert invoke('inspect_source_candidate', {'url': candidate()['url']}, 'inspect_html')['status'] == 'available'
            assert invoke('save_source_candidate', {'candidate': candidate()}, 'save')['ok']
            def inspect():
                try:
                    invoke('inspect_source_candidate', {'url': 'https://example.org/stalled.pdf'}, 'inspect')
                finally:
                    finished.set()
            threading.Thread(target=inspect, daemon=True).start()
            await self.ended.wait()
            raise AgentRuntimeBusinessCompleted('Search deadline saved verified candidates')

    runtime = InspectingRuntime(gateway, {})
    service = SourceDiscoveryService(store=store, settings=replace(Settings(), source_discovery_timeout_seconds=1),
        runtime=runtime, gateway=gateway, gateway_server=SimpleNamespace(endpoint_url='http://127.0.0.1:1'), model_config=model)
    # Release even on a regression that blocks the event loop, so the test fails
    # on elapsed time instead of hanging the whole suite.
    fail_safe = threading.Timer(4, release.set)
    fail_safe.start()
    started = time.monotonic()
    try:
        result = await service.discover(target_profile=target, output_language='English')
        assert entered.is_set()
        assert time.monotonic() - started < 3
        assert not release.is_set()
        assert len(result.candidates) == 1
        assert store.read_json_artifact(result.candidate_artifact_id)['completion_reason'] == 'time_budget_reached'
    finally:
        release.set()
        fail_safe.cancel()
        assert finished.wait(1)


@pytest.mark.parametrize("outcomes,count,expected", [
    ([False], 0, "unavailable"),
    ([True], 0, "completed"),
    ([False, True], 1, "completed"),
    ([True, False], 1, "partial"),
    ([], 0, "not_attempted"),
])
@pytest.mark.asyncio
async def test_discovery_persists_host_search_outcomes_independently_of_agent_summary(tmp_path, outcomes, count, expected):
    store, target, model = setup_discovery(tmp_path)
    gateway = CapabilityGateway()

    class SearchRuntime(SubmittingRuntime):
        async def run(self, invocation):
            for ok in outcomes:
                invocation.metadata["runtime_event_sink"]("native_tool_result", {
                    "tool": "web_search", "ok": ok,
                    "output": "Retrieved results" if ok else "Error: Search failed\nDISCOVERY SEARCH FAILURE\n" +
                        json.dumps({"category": "unavailable", "retryable": True, "fatal": False}),
                })
            return await super().run(invocation)

    runtime = SearchRuntime(gateway, {"candidates": [candidate()] if count else [],
        "summary": "Agent-written assessment", "search_notes": []}, linger=True)
    service = SourceDiscoveryService(store=store, settings=Settings(), runtime=runtime,
        gateway=gateway, gateway_server=SimpleNamespace(endpoint_url="http://127.0.0.1:1"), model_config=model)
    result = await asyncio.wait_for(service.discover(target_profile=target, output_language="Chinese"), 2)
    saved = store.read_json_artifact(result.candidate_artifact_id)
    assert saved["search_status"] == expected
    assert saved["search_counts"] == {"completed": outcomes.count(True), "failed": outcomes.count(False)}
    assert len(saved["candidates"]) == count
    assert saved["summary"] == "Agent-written assessment"
    if outcomes and outcomes[-1] is False:
        assert saved["search_error"]["category"] == "unavailable"
    else:
        assert saved["search_error"] is None


@pytest.mark.parametrize("count", [0, 1, 2, 3])
@pytest.mark.asyncio
async def test_discovery_persists_zero_to_three_candidates_and_closes_completed_work(tmp_path, count):
    store, target, model = setup_discovery(tmp_path)
    gateway = CapabilityGateway()
    runtime = SubmittingRuntime(gateway, {"candidates": [candidate(url=f"https://example.org/manual/{index}/")
        for index in range(count)], "summary": "发现候选资料" if count else "未找到可用资料", "search_notes": []}, linger=True)
    service = SourceDiscoveryService(store=store, settings=Settings(), runtime=runtime,
        gateway=gateway, gateway_server=SimpleNamespace(endpoint_url="http://127.0.0.1:1"), model_config=model)
    result = await asyncio.wait_for(service.discover(target_profile=target, output_language="Chinese"), 2)
    assert len(result.candidates) == count
    saved = store.read_json_artifact(result.candidate_artifact_id)
    assert saved["summary"] == result.summary
    assert saved["output_language"] == "Chinese"
    assert all(item.candidate_id.startswith("candidate_") for item in result.candidates)
    assert store.read_attempt(result.attempt_id).status == "succeeded"
    invocation = runtime.invocations[0]
    assert invocation.role == "source_discovery"
    assert invocation.allowed_tools == SOURCE_DISCOVERY_TOOLS
    assert invocation.model["api_key"] == "model-test-key"
    assert invocation.metadata["native_web_tools"] is True
    assert "Never fabricate a parent by truncating a URL" in invocation.prompt
    assert "ui_locale" not in invocation.prompt
    assert runtime.stopped == [invocation.session_id]
    assert len(runtime.inspections) == count
    for item in saved["candidates"]:
        snapshot_id = item["verification"]["html_snapshot_artifact_id"]
        assert store.read_json_artifact(snapshot_id)["main_content"]["evidence_id"] == "P1"


def test_contract_rejects_duplicates_excess_and_invalid_transport_but_does_not_judge_roots():
    with pytest.raises(ValueError):
        SourceCandidateSubmission(candidates=[candidate(), candidate(url="https://example.org/manual/1.0/#chapter")], summary="duplicates")
    with pytest.raises(ValueError):
        SourceCandidateSubmission(candidates=[candidate(url=f"https://example.org/{index}") for index in range(4)], summary="too many")
    for invalid in ("file:///manual.html", "https://user:password@example.org/", "https://example.org:invalid/"):
        with pytest.raises(ValueError):
            SourceCandidateSubmission(candidates=[candidate(url=invalid)], summary="invalid")
    # Root and version decisions are intentionally semantic Agent decisions.
    submitted = SourceCandidateSubmission(candidates=[candidate(url="https://example.org/chapter.html",
        version_relation="unknown", root_assessment="Agent's judgment")], summary="Agent-owned assessment")
    assert submitted.candidates[0].url.endswith("chapter.html")


def test_submit_is_durable_idempotent_and_does_not_accept_agent_supplied_ids(tmp_path):
    store, _, _ = setup_discovery(tmp_path)
    executor = SourceDiscoveryToolExecutor(store, "attempt_test", "English")
    payload = {"candidates": [candidate()], "summary": "Found one"}
    bad = executor.call("submit_source_candidates", {**payload, "candidates": [candidate(candidate_id="invented")]})
    assert not bad["ok"] and not executor.is_finished
    assert executor.call("inspect_source_candidate", {"url": candidate()["url"]})["status"] == "available"
    result = executor.call("submit_source_candidates", payload)
    assert result["ok"] and executor.is_finished
    assert executor.call("submit_source_candidates", payload) == result
    assert store.read_json_artifact(result["candidate_artifact_id"])["candidates"][0]["title"] == candidate()["title"]
    assert not executor.call("submit_source_candidates", {**payload, "summary": "Changed"})["ok"]
    assert executor.tool_specs()[0]["function"]["name"] == "submit_source_candidates"


@pytest.mark.asyncio
async def test_discovery_timeout_and_user_cancellation_stop_the_child(tmp_path):
    gateway = CapabilityGateway()

    class WaitingRuntime(SubmittingRuntime):
        async def run(self, invocation):
            self.invocations.append(invocation)
            await self.ended.wait()

    for mode in ("timeout", "cancel"):
        store, target, model = setup_discovery(tmp_path / mode)
        runtime = WaitingRuntime(gateway, {})
        service = SourceDiscoveryService(store=store,
            settings=replace(Settings(), source_discovery_timeout_seconds=1), runtime=runtime,
            gateway=gateway, gateway_server=SimpleNamespace(endpoint_url="http://127.0.0.1:1"), model_config=model)
        task = asyncio.create_task(service.discover(target_profile=target, output_language="English"))
        if mode == "cancel":
            while not runtime.invocations:
                await asyncio.sleep(0)
            task.cancel()
        if mode == "timeout":
            result = await task
            assert not result.candidates
            assert store.read_json_artifact(result.candidate_artifact_id)["completion_reason"] == "time_budget_reached"
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
        assert runtime.ended.is_set()
        attempt = store.read_attempt(runtime.invocations[0].metadata["capability"]["context"]["attempt_id"])
        assert attempt.status == ("succeeded" if mode == "timeout" else "cancelled")


@pytest.mark.asyncio
async def test_saved_candidates_survive_session_cleanup_error(tmp_path):
    store, target, model = setup_discovery(tmp_path)
    gateway = CapabilityGateway()

    class CleanupErrorRuntime(SubmittingRuntime):
        async def close_session(self, session_id):
            raise RuntimeError("cleanup failed after persisted candidates")

    runtime = CleanupErrorRuntime(gateway, {"candidates": [candidate()], "summary": "Found one manual"})
    service = SourceDiscoveryService(store=store, settings=Settings(), runtime=runtime, gateway=gateway,
        gateway_server=SimpleNamespace(endpoint_url="http://127.0.0.1:1"), model_config=model)
    result = await service.discover(target_profile=target, output_language="English")
    assert len(result.candidates) == 1
    assert store.read_attempt(result.attempt_id).status == "succeeded"
    assert store.read_json_artifact(result.candidate_artifact_id)["candidates"]


def test_checkpoint_is_durable_updates_one_url_and_terminal_cannot_discard_it(tmp_path):
    store, _, _ = setup_discovery(tmp_path)
    executor = SourceDiscoveryToolExecutor(store, "attempt_checkpoint", "English")
    assert executor.call("inspect_source_candidate", {"url": candidate()["url"]})["status"] == "available"
    first = executor.call("save_source_candidate", {"candidate": candidate(version_relation="unknown")})
    assert first["ok"] and not executor.is_finished
    state = store.read_state()
    assert state["source_discovery_checkpoint_artifact_id"] == first["candidate_artifact_id"]
    assert store.read_json_artifact(first["candidate_artifact_id"])["candidates"][0]["version_relation"] == "unknown"
    updated = executor.call("save_source_candidate", {"candidate": candidate(version_relation="exact")})
    assert updated["candidate_count"] == 1
    assert updated["candidate_artifact_id"] != first["candidate_artifact_id"]
    # Empty final output and late errors cannot lose a successfully saved manual.
    finished = executor.call("submit_source_candidates", {"candidates": [], "summary": "Finished"})
    assert finished["candidate_count"] == 1
    assert executor.result.candidates[0].version_relation == "exact"
    assert not executor.call("save_source_candidate", {"candidate": candidate(title="Late overwrite")})["ok"]
    assert executor.result.candidates[0].title == candidate()["title"]


def test_checkpoint_limit_and_validation_do_not_erase_saved_candidates(tmp_path):
    store, _, _ = setup_discovery(tmp_path)
    executor = SourceDiscoveryToolExecutor(store, "attempt_checkpoint", "English")
    assert executor.call("inspect_source_candidate", {"url": candidate()["url"]})["status"] == "available"
    assert not executor.call("save_source_candidate", {"candidate": candidate(root_assessment="")})["ok"]
    for index in range(3):
        url = f"https://example.org/{index}/"
        assert executor.call("inspect_source_candidate", {"url": url})["status"] == "available"
        assert executor.call("save_source_candidate", {"candidate": candidate(url=url)})["ok"]
    assert executor.call("inspect_source_candidate", {"url": "https://example.org/fourth/"})["status"] == "available"
    assert not executor.call("save_source_candidate", {"candidate": candidate(url="https://example.org/fourth/")})["ok"]
    result = executor.finalize_saved("time_budget_reached")
    assert len(result.candidates) == 3
    assert store.read_json_artifact(result.candidate_artifact_id)["completion_reason"] == "time_budget_reached"
    assert executor.finalize_saved("runtime_interrupted") is result


@pytest.mark.parametrize("ending", ["timeout", "runtime_error", "no_submission", "cancel"])
@pytest.mark.asyncio
async def test_saved_candidate_survives_interruption_but_user_cancel_is_respected(tmp_path, ending):
    store, target, model = setup_discovery(tmp_path)
    gateway = CapabilityGateway()

    class CheckpointRuntime(SubmittingRuntime):
        async def run(self, invocation):
            self.invocations.append(invocation)
            raw = invocation.metadata["capability"]
            inspected = self.gateway.invoke(raw["capability_token"], CapabilityInvocation(
                **raw["context"], tool_name="inspect_source_candidate", arguments={"url": candidate()["url"]}, call_id="inspect_checkpoint"))
            assert inspected["status"] == "available" and inspected["html_snapshot_artifact_id"]
            result = self.gateway.invoke(raw["capability_token"], CapabilityInvocation(
                **raw["context"], tool_name="save_source_candidate", arguments={"candidate": candidate()}, call_id="save_1"))
            assert result["ok"]
            if ending == "runtime_error":
                raise RuntimeError("Provider connection interrupted after checkpoint")
            if ending == "no_submission":
                return AgentTurnResult(invocation.session_id, "Done", "completed")
            await self.ended.wait()

    runtime = CheckpointRuntime(gateway, {})
    service = SourceDiscoveryService(store=store, settings=replace(Settings(), source_discovery_timeout_seconds=1),
        runtime=runtime, gateway=gateway, gateway_server=SimpleNamespace(endpoint_url="http://127.0.0.1:1"), model_config=model)
    task = asyncio.create_task(service.discover(target_profile=target, output_language="Chinese"))
    if ending == "cancel":
        while not runtime.invocations:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        attempt_id = runtime.invocations[0].metadata["capability"]["context"]["attempt_id"]
        assert store.read_attempt(attempt_id).status == "cancelled"
        assert store.read_state()["source_discovery_checkpoint_artifact_id"]
        return
    result = await task
    assert len(result.candidates) == 1
    assert result.candidates[0].url == candidate()["url"]
    assert store.read_attempt(result.attempt_id).status == "succeeded"
    saved = store.read_json_artifact(result.candidate_artifact_id)
    expected = {"timeout": "time_budget_reached", "runtime_error": "runtime_interrupted", "no_submission": "session_ended"}
    assert saved["completion_reason"] == expected[ending]
    assert saved["output_language"] == "Chinese"
    assert "model-test-key" not in store.audit_path.read_text(encoding="utf-8")
    budget = runtime.invocations[0].metadata["discovery_budget"]
    assert budget["max_search_rounds"] == 3
    assert budget["max_fetches_per_round"] == 6


@pytest.mark.parametrize("failure", ["uninspected", "invented_evidence"])
def test_failed_scope_replacement_preserves_the_original_checkpoint(tmp_path, failure):
    store, _, _ = setup_discovery(tmp_path)
    executor = SourceDiscoveryToolExecutor(store, "attempt_replacement", "English")
    original = candidate()
    assert executor.call("inspect_source_candidate", {"url": original["url"]})["status"] == "available"
    saved = executor.call("save_source_candidate", {"candidate": original})
    assert saved["ok"]
    checkpoint = store.read_json_artifact(saved["candidate_artifact_id"])
    replacement = candidate(url="https://example.org/lessons/")
    if failure == "invented_evidence":
        assert executor.call("inspect_source_candidate", {"url": replacement["url"]})["status"] == "available"
        replacement["html_scope"]["entry_evidence_ids"] = ["H99999"]
    rejected = executor.call("save_source_candidate", {"candidate": replacement, "replaces_url": original["url"]})
    assert rejected["error"] == {
        "uninspected": "html_candidate_inspection_required", "invented_evidence": "html_scope_evidence_invalid",
    }[failure]
    assert store.read_state()["source_discovery_checkpoint_artifact_id"] == saved["candidate_artifact_id"]
    assert store.read_json_artifact(saved["candidate_artifact_id"]) == checkpoint
    result = executor.finalize_saved("runtime_interrupted")
    assert store.read_json_artifact(result.candidate_artifact_id)["candidates"] == checkpoint["candidates"]


def test_invalid_inspection_reports_fields_without_echoing_input(tmp_path):
    store, _, _ = setup_discovery(tmp_path)
    executor = SourceDiscoveryToolExecutor(store, 'attempt_validation', 'English')
    result = executor.call('inspect_source_candidate', {'action': 'inspect'})
    assert result['error'] == 'invalid_candidate_inspection'
    assert result['errors'][0]['field'] == ['url']
    assert result['errors'][0]['type'] == 'missing'
    assert 'required' in result['errors'][0]['message'].lower()
    assert all('input' not in item for item in result['errors'])
    executor.observe_search_result({'ok': False, 'output': 'DISCOVERY SEARCH FAILURE\n' +
        json.dumps({'category': 'model_arguments', 'model_can_fix': True})})
    assert executor._search_failed == 0


def test_failed_entry_preserves_observed_recovery_links_and_cached_failure(tmp_path):
    store, _, _ = setup_discovery(tmp_path)
    calls = []

    class FailedEntryProbe:
        def inspect(self, url):
            calls.append(url)
            return {'status': 'failed', 'url': url, 'error': 'http_error', 'http_status': 404,
                    'recovery_links': [{'url': 'https://example.org/real-manual/', 'title': 'Manual home'}]}

    executor = SourceDiscoveryToolExecutor(store, 'attempt_recovery', 'English', probe=FailedEntryProbe())
    url = 'https://example.org/outdated-entry/'
    first = executor.call('inspect_source_candidate', {'url': url})
    assert first['status'] == 'failed'
    assert 'not an available manual' in first['instruction']
    assert executor.call('inspect_source_candidate', {'url': url}) == first
    assert calls == [url]
    queue = executor.call('list_discovered_sources', {'modality': 'html'})
    assert {item['url'] for item in queue['items']} == {url, 'https://example.org/real-manual/'}
    rejected = executor.call('save_source_candidate', {'candidate': candidate(url=url)})
    assert rejected['error'] == 'html_candidate_unavailable'
    result = executor.finalize_saved('session_ended')
    summary = store.read_json_artifact(result.candidate_artifact_id)['failure_summary']
    assert summary['primary_reason'] == 'source_not_found'
    assert summary['inspection_counts']['total'] == 1


@pytest.mark.parametrize('statuses,reason', [
    ([403], 'source_access_denied'),
    ([404, 410], 'source_not_found'),
    ([401], 'source_authentication'),
    ([403, 404], 'source_unavailable'),
])
def test_empty_discovery_reports_host_access_facts_without_claiming_search_failed(tmp_path, statuses, reason):
    store, _, _ = setup_discovery(tmp_path)

    class AccessProbe:
        def inspect(self, url):
            return {'status': 'failed', 'url': url, 'error': 'http_error',
                    'http_status': statuses[int(url.rsplit('/', 1)[-1])]}

    executor = SourceDiscoveryToolExecutor(store, 'attempt_access', 'English', probe=AccessProbe())
    executor.observe_search_result({'ok': True})
    for index in range(len(statuses)):
        executor.call('inspect_source_candidate', {'url': f'https://example.org/manual/{index}'})
    saved = executor.call('submit_source_candidates', {'candidates': [], 'summary': 'Untrusted model explanation'})
    payload = store.read_json_artifact(saved['candidate_artifact_id'])
    assert payload['search_status'] == 'completed'
    assert payload['failure_summary']['primary_reason'] == reason
    assert payload['failure_summary']['inspection_counts']['failed'] == len(statuses)


def test_search_failure_is_distinct_from_failed_official_entry_and_summary_is_host_owned(tmp_path):
    store, _, _ = setup_discovery(tmp_path)
    executor = SourceDiscoveryToolExecutor(store, 'attempt_service_failure', 'English')
    executor.observe_search_result({'ok': False, 'output': 'DISCOVERY SEARCH FAILURE\n' + json.dumps({
        'category': 'unsupported', 'fatal': True, 'retryable': False, 'message': 'do not show this secret detail'})})
    saved = executor.call('submit_source_candidates', {'candidates': [], 'summary': 'No manual exists'})
    payload = store.read_json_artifact(saved['candidate_artifact_id'])
    assert payload['failure_summary']['primary_reason'] == 'search_unsupported'
    assert payload['failure_summary']['inspection_counts']['total'] == 0
    assert 'secret detail' not in json.dumps(payload['failure_summary'])
    assert 'No manual exists' not in json.dumps(payload['failure_summary'])


def test_final_comparison_order_overrides_checkpoint_order_without_losing_saved_manuals(tmp_path):
    store, _, _ = setup_discovery(tmp_path)
    executor = SourceDiscoveryToolExecutor(store, 'attempt_ranking', 'English')
    early = candidate(url='https://example.org/reference/')
    preferred = candidate(url='https://example.org/tutorials/')
    for item in (early, preferred):
        assert executor.call('inspect_source_candidate', {'url': item['url']})['status'] == 'available'
        assert executor.call('save_source_candidate', {'candidate': item})['ok']
    submitted = executor.call('submit_source_candidates', {'candidates': [preferred], 'summary': 'Prefer the teaching collection'})
    payload = store.read_json_artifact(submitted['candidate_artifact_id'])
    assert [item['url'] for item in payload['candidates']] == [preferred['url'], early['url']]


def test_protocol_negotiation_authentication_is_not_misreported_as_missing_search_capability(tmp_path):
    store, _, _ = setup_discovery(tmp_path)
    executor = SourceDiscoveryToolExecutor(store, 'attempt_auth', 'English')
    executor.observe_search_result({'ok': False, 'output': 'DISCOVERY SEARCH FAILURE\n' + json.dumps({
        'category': 'authentication', 'reason': 'capability_unavailable', 'fatal': True})})
    submitted = executor.call('submit_source_candidates', {'candidates': [], 'summary': 'Search failed'})
    payload = store.read_json_artifact(submitted['candidate_artifact_id'])
    assert payload['failure_summary']['primary_reason'] == 'search_authentication'

