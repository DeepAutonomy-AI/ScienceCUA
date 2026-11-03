import asyncio
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import httpx
import pymupdf
import pytest

from app.orchestration.source_discovery_stage import SourceDiscoveryToolExecutor
from app.source_candidate_probe import SourceCandidateProbe
from app.source_validation import SourceValidationQueue
from test_source_discovery import candidate, html_manual_response, setup_discovery


def test_parallel_validation_delivers_fast_html_keeps_pdf_and_blocks_early_submit(tmp_path):
    store, _, _ = setup_discovery(tmp_path)
    release = threading.Event()
    pdf_entered = threading.Event()
    calls = []; active = 0; peak = 0; lock = threading.Lock()
    html = SourceCandidateProbe(store, transport=httpx.MockTransport(html_manual_response))
    class Probe:
        def inspect(self, url):
            nonlocal active, peak
            with lock:
                active += 1; peak = max(peak, active); calls.append(url)
            try:
                if url.endswith('.pdf'):
                    pdf_entered.set(); assert release.wait(5)
                    return {'status': 'available', 'url': url, 'modality': 'pdf', 'page_count': 116, 'header_verified': True}
                return html.inspect(url)
            finally:
                with lock: active -= 1
    executor = SourceDiscoveryToolExecutor(store, 'parallel', 'English', probe=Probe(), parallel_validation=True)
    url = candidate()['url']; pdf = 'https://example.org/manual.pdf'
    try:
        executor.observe_search_sources([{'url': pdf}, {'url': url}, {'url': 'https://example.org/other.html'}, {'url': 'https://example.org/third.html'}])
        assert pdf_entered.wait(1)
        first = executor.call('next_source_validation', {'wait_seconds': 2})
        assert first['modality'] == 'html'
        # Read the desired candidate from the existing automatic job, without another GET.
        executor.call('inspect_source_candidate', {'url': url})
        assert executor.call('save_source_candidate', {'candidate': candidate(version_relation='exact')})['ok']
        assert not release.is_set()
        early = executor.call('submit_source_candidates', {'candidates': [], 'summary': 'One exact HTML found'})
        assert early['error'] == 'source_validations_unreviewed'
        assert early['validation_state']['active'] >= 1
        executor.observe_search_sources([{'url': url + '#chapter'}])
        assert calls.count(url) == 1
        release.set()
        seen_pdf = False
        for _ in range(6):
            result = executor.call('next_source_validation', {'wait_seconds': 2})
            seen_pdf |= result.get('modality') == 'pdf'
            if result['validation_state']['active'] == result['validation_state']['unreviewed'] == 0:
                break
        assert seen_pdf and peak <= 3
        assert executor.call('submit_source_candidates', {'candidates': [], 'summary': 'All results reviewed'})['ok']
        assert calls.count(pdf) == calls.count(url) == 1
    finally:
        release.set(); executor.close()


def test_queue_concurrency_and_network_caps_and_close(tmp_path):
    release = threading.Event(); three = threading.Event(); lock = threading.Lock()
    active = 0; peak = 0
    def inspect(url):
        nonlocal active, peak
        with lock:
            active += 1; peak = max(peak, active)
            if active == 3: three.set()
        try:
            release.wait(3)
            return {'url': url, 'status': 'available'}
        finally:
            with lock: active -= 1
    queue = SourceValidationQueue(inspect)
    try:
        for batch in range(4):
            queue.start([f'https://example.org/{batch}/{index}.html' for index in range(15)])
        assert three.wait(1)
        assert queue.counts()['total'] == 36 and peak == 3
        queue.close()
        assert queue.take(wait_seconds=0) is None
    finally:
        release.set(); queue.close()


def test_pdf_has_longer_deadline_and_only_one_transient_retry(tmp_path):
    store, _, _ = setup_discovery(tmp_path)
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), 'HFSS 2025 R1')
        pdf = doc.tobytes()
    class SlowBody(httpx.SyncByteStream):
        def __iter__(self):
            time.sleep(.09); yield pdf
    # Content-Type also promotes an extensionless PDF to the PDF deadline.
    probe = SourceCandidateProbe(store, timeout=.03, pdf_timeout=.3,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, headers={'content-type': 'application/pdf'}, stream=SlowBody())))
    result = probe.inspect('https://example.org/document')
    assert result['status'] == 'available' and result['page_count'] == 1
    calls = []
    def transport(request):
        calls.append(str(request.url))
        if len(calls) == 1: raise httpx.ReadTimeout('synthetic timeout')
        return httpx.Response(200, content=pdf)
    retry = SourceCandidateProbe(store, pdf_timeout=.3, transient_retries=1, transport=httpx.MockTransport(transport))
    result = retry.inspect('https://example.org/retry.pdf')
    assert result['status'] == 'available' and result['attempt_count'] == 2 and len(calls) == 2
    calls.clear()
    def unavailable(request):
        calls.append(str(request.url)); raise httpx.ReadTimeout('synthetic timeout')
    retry = SourceCandidateProbe(store, transient_retries=1, transport=httpx.MockTransport(unavailable))
    assert retry.inspect('https://example.org/down.pdf')['attempt_count'] == 2
    assert len(calls) == 2


def test_deadline_finalization_stops_late_validation_updates(tmp_path):
    store, _, _ = setup_discovery(tmp_path)
    release = threading.Event(); entered = threading.Event()
    class Probe:
        def inspect(self, url):
            entered.set(); release.wait(3)
            return {'status': 'failed', 'url': url, 'error': 'timeout'}
    executor = SourceDiscoveryToolExecutor(store, 'timeout', 'English', probe=Probe(), parallel_validation=True)
    try:
        executor.observe_search_sources([{'url': 'https://example.org/slow.pdf'}])
        assert entered.wait(1)
        result = executor.finalize_saved('time_budget_reached')
        revision = store.read_state()['revision']
        assert any('partial result' in note for note in result.search_notes)
        release.set(); time.sleep(.1)
        assert store.read_state()['revision'] == revision
    finally:
        release.set(); executor.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_sdk_reviews_html_while_pdf_downloads_and_waits_before_submitting(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from app.capabilities import CapabilityGateway, CapabilityGatewayServer
    from app.orchestration.source_discovery_stage import SourceDiscoveryService
    from app.runtime import DshAgentRuntime, DshBridgeConfig
    store, target, model = setup_discovery(tmp_path / 'run')
    release = threading.Event(); pdf_entered = threading.Event()
    requests = []; errors = []; downloads = []
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), 'Example User Guide\nFor version 1.0')
        pdf = doc.tobytes()
    html_url = pdf_url = ''
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            downloads.append(self.path)
            if self.path == '/manual.pdf':
                pdf_entered.set(); release.wait(20)
                data, kind = pdf, 'application/pdf'
            else:
                data, kind = html_manual_response(httpx.Request('GET', html_url)).content, 'text/html'
            self.send_response(200); self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/v1/responses':
                    value = {'status': 'completed', 'output': [{'type': 'web_search_call', 'status': 'completed',
                        'action': {'sources': [{'url': html_url}, {'url': pdf_url}]}}]}
                    data = json.dumps(value).encode(); kind = 'application/json'
                else:
                    requests.append(payload); turn = len(requests)
                    text = json.dumps(payload['messages'])
                    if turn == 1: name, args = 'web_search', {'queries': ['Example official manual']}
                    elif turn == 2: name, args = 'next_source_validation', {'wait_seconds': 5}
                    elif turn == 3:
                        assert 'html_snapshot' in text and pdf_entered.is_set() and not release.is_set()
                        name, args = 'save_source_candidate', {'candidate': candidate(url=html_url, version_relation='exact')}
                    elif turn == 4: name, args = 'submit_source_candidates', {'candidates': [], 'summary': 'Premature exact HTML'}
                    elif turn == 5:
                        assert 'source_validations_unreviewed' in text
                        release.set(); name, args = 'next_source_validation', {'wait_seconds': 10}
                    elif turn == 6:
                        assert 'page_count' in json.dumps(payload['messages'][-1])
                        name, args = 'save_source_candidate', {'candidate': candidate(url=pdf_url, modality='pdf', version_relation='exact', root_assessment='Complete PDF preview verified')}
                    elif turn == 7: name, args = 'submit_source_candidates', {'candidates': [], 'summary': 'HTML and PDF both reviewed'}
                    else: name, args = None, None
                    delta = {'role': 'assistant', 'tool_calls': [{'index': 0, 'id': f'call_{turn}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]} if name else {'role': 'assistant', 'content': 'Done'}
                    chunk = {'id': f'parallel_{turn}', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'gpt-test',
                        'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}
                    final = {**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if name else 'stop'}]}
                    data = (''.join('data: ' + json.dumps(value) + '\n\n' for value in [chunk, final]) + 'data: [DONE]\n\n').encode()
                    kind = 'text/event-stream'
                self.send_response(200); self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
            except BaseException as error:
                errors.append(repr(error)); self.send_error(500)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    base = f'http://127.0.0.1:{server.server_port}'
    html_url, pdf_url = base + '/manual.html', base + '/manual.pdf'
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv('DSH_WEB_SEARCH_PROVIDER', 'openai-responses')
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    model = model.model_copy(update={'base_url': base + '/v1'})
    native = DshAgentRuntime(replace(DshBridgeConfig(), dsh_home=str(tmp_path / 'home'), http_trace_enabled=False,
        request_timeout_seconds=35, api_max_retries=0))
    gateway = CapabilityGateway(); gateway_server = CapabilityGatewayServer(gateway)
    try:
        result = await SourceDiscoveryService(store=store, settings=SimpleNamespace(source_discovery_timeout_seconds=45),
            runtime=native, gateway=gateway, gateway_server=gateway_server, model_config=model).discover(target_profile=target, output_language='English')
        assert not errors, errors
        assert {value.modality for value in result.candidates} == {'html', 'pdf'}
        assert downloads.count('/manual.html') == downloads.count('/manual.pdf') == 1
    finally:
        release.set(); gateway_server.close(); server.shutdown(); server.server_close()
