"""Real SDK negotiation against local scripted provider APIs, with separate search credentials."""
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pymupdf
import pytest

from app.capabilities import CapabilityGateway, CapabilityGatewayServer
from app.orchestration.source_discovery_stage import SourceDiscoveryService
from app.runtime import DshAgentRuntime, DshBridgeConfig
from test_source_discovery import candidate, setup_discovery


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize('working_protocol', ['messages', 'chat_completions', 'unavailable', 'intermittent', 'cooldown_recovery', 'probe_recovery'])
async def test_sdk_negotiates_search_protocol_without_changing_generation_route(tmp_path, monkeypatch, working_protocol):
    store, target, model = setup_discovery(tmp_path / 'run')
    searches, generations, probes = [], [], []
    response_attempts = 0
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), 'Example complete user manual for version 1.0')
        pdf = document.tobytes()
    source = candidate(modality='pdf', root_assessment='Downloaded full user guide and inspected the opening page.')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value, status=200, content_type='application/json'):
            data = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            probes.append(self.path)
            if self.path.startswith('/broken-'):
                self.respond(b'<html><title>Not found</title><p>This entry moved.</p></html>', status=404, content_type='text/html')
                return
            self.respond(pdf, content_type='application/pdf')

        def do_POST(self):
            nonlocal response_attempts
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            is_search = self.path != '/v1/chat/completions' or 'web_search_options' in payload
            if is_search:
                assert payload['model'] == 'retrieval-model'
                assert (self.headers.get('x-api-key') or self.headers.get('Authorization')) in (
                    'search-only-private', 'Bearer search-only-private')
                searches.append(self.path)
                if working_protocol == 'probe_recovery' and self.path == '/v1/responses':
                    self.respond({'output': [{'type': 'web_search_call', 'status': 'completed',
                        'action': {'sources': [{'url': source['url'], 'title': source['title']}]}}]})
                elif working_protocol in {'intermittent', 'cooldown_recovery'} and self.path == '/v1/responses':
                    response_attempts += 1
                    if response_attempts <= (6 if working_protocol == 'cooldown_recovery' else 1):
                        self.respond({'error': {'type': 'invalid_request_error', 'message': '请求被拒绝'}}, status=400)
                    else:
                        self.respond({'output': [{'type': 'web_search_call', 'status': 'completed',
                            'action': {'sources': [{'url': source['url'], 'title': source['title']}]}}]})
                elif working_protocol == 'cooldown_recovery':
                    if self.path == '/v1/messages':
                        self.respond({}, status=404)
                    else:
                        self.respond({'choices': [{'message': {'content': 'No live web search available.'}}]})
                elif working_protocol == 'messages' and self.path == '/v1/messages':
                    self.respond({'content': [
                        {'type': 'server_tool_use', 'id': 'search1', 'name': 'web_search'},
                        {'type': 'web_search_tool_result', 'tool_use_id': 'search1', 'content': [
                            {'type': 'web_search_result', 'url': source['url'], 'title': source['title']}]}]})
                elif working_protocol == 'chat_completions' and self.path == '/v1/chat/completions':
                    self.respond({'choices': [{'message': {'content': 'A full user guide.', 'annotations': [
                        {'type': 'url_citation', 'url_citation': {'url': source['url'], 'title': source['title']}}]}}]})
                else:
                    self.respond({'error': {'code': 'service_unavailable', 'message':
                        'Search route temporarily unavailable; reflected search-only-private'}}, status=503)
                return
            assert payload['model'] == 'writer-model'
            assert self.headers.get('Authorization') == 'Bearer model-test-key'
            generations.append(payload)
            turn = len(generations)
            text = json.dumps(payload['messages'])
            if working_protocol == 'probe_recovery':
                calls = {1: ('web_search', {'queries': ['Example official manual']}),
                    **{index + 2: ('inspect_source_candidate', {'url':
                        f'http://127.0.0.1:{server.server_port}/broken-{index}'}) for index in range(6)},
                    8: ('inspect_source_candidate', {'url': source['url']}),
                    9: ('inspect_source_candidate', {'url': source['url']}),
                    10: ('save_source_candidate', {'candidate': source}),
                    11: ('submit_source_candidates', {'candidates': [], 'summary': 'Recovered the real manual after failed entries.'})}
                if turn == 8:
                    assert 'Verifications remaining this round: 0' in text
                    assert 'Recovery inspections remaining: 3' in text
                if turn in (9, 10):
                    assert 'Recovery inspections remaining: 2' in text
            elif working_protocol == 'unavailable':
                calls = {1: ('web_search', {'queries': ['Example official manual']}),
                    2: ('web_search', {'queries': ['Example full manual PDF']}),
                    3: ('web_search', {'queries': ['Example handbook']}),
                    4: ('web_search', {'queries': ['Example recovery search']}),
                    5: ('submit_source_candidates', {'candidates': [], 'summary': 'Search backend failed; no verified candidates.'})}
                if turn in (2, 3):
                    assert 'Search rounds used: 0/3' in text
                    assert 'service_unavailable' in text
                if turn == 3:
                    assert 'One recovery web_search is available' in text
                if turn == 5:
                    assert 'Search backend is blocked' in text
            elif working_protocol == 'cooldown_recovery':
                calls = {1: ('web_search', {'queries': ['Example official manual PDF']}),
                    2: ('web_search', {'queries': ['Example full manual PDF']}),
                    3: ('web_search', {'queries': ['Example handbook']}),
                    4: ('inspect_source_candidate', {'url': source['url']}),
                    5: ('save_source_candidate', {'candidate': source}),
                    6: ('submit_source_candidates', {'candidates': [], 'summary': 'Verified manual after recovery.'})}
                if turn == 3:
                    assert 'One recovery web_search is available' in text
                if turn == 4:
                    assert 'Search rounds used: 1/3' in text
                    assert 'DISCOVERY SEARCH EVIDENCE' in text
            else:
                calls = {1: ('web_search', {'queries': ['Example official manual PDF']}),
                    2: ('inspect_source_candidate', {'url': source['url']}),
                    3: ('save_source_candidate', {'candidate': source}),
                    4: ('submit_source_candidates', {'candidates': [], 'summary': 'Verified full manual.'})}
                if turn == 2:
                    assert 'Search rounds used: 1/3' in text
                    assert 'DISCOVERY SEARCH EVIDENCE' in text
            name, arguments = calls.get(turn, (None, None))
            delta = {'role': 'assistant'}
            if name:
                delta['tool_calls'] = [{'index': 0, 'id': f'call_{turn}', 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(arguments)}}]
            else:
                delta['content'] = 'Done'
            chunk = {'id': f'negotiation_{turn}', 'object': 'chat.completion.chunk', 'created': 1,
                'model': 'writer-model', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}
            final = {**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if name else 'stop'}]}
            data = (''.join('data: ' + json.dumps(item) + '\n\n' for item in [chunk, final]) + 'data: [DONE]\n\n').encode()
            self.respond(data, content_type='text/event-stream')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    source['url'] = f'http://127.0.0.1:{server.server_port}/manual.pdf'
    model = model.model_copy(update={'base_url': f'http://127.0.0.1:{server.server_port}/v1', 'model': 'writer-model'})
    monkeypatch.setenv('DSH_WEB_SEARCH_PROVIDER', 'openai-responses')
    monkeypatch.setenv('PREPROCESS_SEARCH_MODEL', 'retrieval-model')
    monkeypatch.setenv('PREPROCESS_SEARCH_API_KEY_ENV', 'TEST_NEGOTIATION_SEARCH_KEY')
    monkeypatch.setenv('TEST_NEGOTIATION_SEARCH_KEY', 'search-only-private')
    monkeypatch.delenv('PREPROCESS_SEARCH_PROTOCOL', raising=False)
    monkeypatch.delenv('PREPROCESS_SEARCH_BASE_URL', raising=False)
    native = DshAgentRuntime(replace(DshBridgeConfig(), dsh_home=str(tmp_path / 'dsh-home'),
        request_timeout_seconds=40, api_max_retries=0))
    gateway = CapabilityGateway()
    gateway_server = CapabilityGatewayServer(gateway)
    try:
        result = await SourceDiscoveryService(store=store, settings=SimpleNamespace(source_discovery_timeout_seconds=50),
            runtime=native, gateway=gateway, gateway_server=gateway_server, model_config=model).discover(
                target_profile=target, output_language='English')
        assert len(result.candidates) == (0 if working_protocol == 'unavailable' else 1)
        expected_calls = {'messages': 4, 'chat_completions': 7, 'unavailable': 27, 'intermittent': 2, 'cooldown_recovery': 10, 'probe_recovery': 1}
        assert len(searches) == expected_calls[working_protocol]
        saved = store.read_json_artifact(result.candidate_artifact_id)
        assert saved['search_status'] == ('unavailable' if working_protocol == 'unavailable' else 'completed')
        audit = store.audit_path.read_text(encoding='utf-8')
        assert 'search-only-private' not in audit and 'model-test-key' not in audit
        errors = []
        for path in (tmp_path / 'dsh-home/http-traces').rglob('*.meta.json'):
            meta = json.loads(path.read_text(encoding='utf-8'))
            if meta.get('status') == (400 if working_protocol in {'intermittent', 'cooldown_recovery'} else 503):
                errors.append(meta)
                assert meta['complete'] and meta['captured_bytes'] > 0
                body = path.with_name(meta['response_body_file']).read_text(encoding='utf-8')
                assert ('请求被拒绝' in json.loads(body)['error'].get('message', '') or 'service_unavailable' in body)
                assert 'search-only-private' not in body
        if working_protocol == 'probe_recovery':
            assert len(probes) == 7
            assert probes.count('/manual.pdf') == 1
        else:
            assert errors
        assert native._workers == {}
    finally:
        gateway_server.close()
        server.shutdown()
        server.server_close()
