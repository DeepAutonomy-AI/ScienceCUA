"""Exercise shipped DSH tools against a local scripted provider, without API credentials."""
import asyncio
import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import sys
import threading

import pytest

pytestmark = pytest.mark.integration
import httpx

from app.orchestration.pipeline import PipelineRuntime
from app.runtime import AgentInvocation, DshAgentRuntime, DshBridgeConfig
from test_adaptive_source import ALL_SCRIPT, setup_source


@pytest.mark.asyncio
@pytest.mark.skipif(not shutil.which('pwsh' if sys.platform == 'win32' else 'bash'), reason='Native shell unavailable')
async def test_shipped_native_tools_execute_extraction_and_stay_out_of_downstream(tmp_path, monkeypatch):
    settings, store, _, target, model, staged = setup_source(tmp_path / "source files ' & $x")
    requests = []
    assigned = {}
    shell = 'pwsh' if sys.platform == 'win32' else 'bash'
    model_paused = threading.Event()
    release_model = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(payload)
            turn = len(requests)
            if turn == 1:
                name, arguments = shell, {'description': 'Inspect source assignment', 'command': 'Get-Content assignment.json' if shell == 'pwsh' else 'cat assignment.json'}
            elif turn == 2:
                model_paused.set()
                release_model.wait(timeout=15)
                name, arguments = 'str_replace_editor', {'command': 'create',
                    'path': str(assigned['root'] / 'work' / 'extract.py'), 'file_text': ALL_SCRIPT}
            elif turn == 3:
                name, arguments = shell, {'description': 'Execute extraction and validation', 'command': assigned['run_command']}
            else:
                name, arguments = None, None
            delta = {'role': 'assistant'}
            if name:
                delta['tool_calls'] = [{'index': 0, 'id': f'call_{turn}', 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(arguments)}}]
            else:
                delta['content'] = 'Extraction finished.'
            chunk = {'id': f'smoke_{turn}', 'object': 'chat.completion.chunk', 'created': 1,
                     'model': 'smoke', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}
            final = {**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if name else 'stop'}]}
            data = ''.join('data: ' + json.dumps(item) + '\n\n' for item in [chunk, final]) + 'data: [DONE]\n\n'
            encoded = data.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    class CapturedRuntime(DshAgentRuntime):
        async def run(self, invocation):
            if invocation.metadata.get('assignment_path'):
                assignment = Path(invocation.metadata['assignment_path'])
                assigned.update(root=assignment.parent, **json.loads(assignment.read_text(encoding='utf-8')))
            return await super().run(invocation)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    native = CapturedRuntime(replace(DshBridgeConfig(), dsh_home=str(tmp_path / 'dsh-home'),
        request_timeout_seconds=45, api_max_retries=0, shutdown_timeout_seconds=5))
    model = model.model_copy(update={'base_url': f'http://127.0.0.1:{server.server_port}/v1', 'api_key': 'local-test'})
    pipeline = PipelineRuntime(settings, native)
    execution = asyncio.create_task(pipeline.execute(store.run_id, target, [staged], model))
    try:
        assert await asyncio.to_thread(model_paused.wait, 30), 'DSH did not complete the first native tool'
        from app import main
        monkeypatch.setattr(main, '_existing_store', lambda run_id: store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as client:
            for _ in range(40):
                response = await client.get(f'/api/runs/{store.run_id}/events?limit=500')
                live = response.json()
                if any(e['event_type'] == 'native_tool_result' for e in live['events']):
                    break
                await asyncio.sleep(0.05)
            assert response.status_code == 200
            assert not execution.done(), 'Progress was only published after the Agent finished'
            kinds = {e['event_type'] for e in live['events']}
            assert {'native_tool_call', 'native_tool_result', 'model_request_started'} <= kinds
            result = next(e for e in live['events'] if e['event_type'] == 'native_tool_result')
            assert result['payload']['ok'] is True
            assert 'assignment' in str(result['payload']['output']).lower()
            assert result['source_id'] == 'SRC1' and result['payload']['attempt_id']
            cursor = live['next_offset']
            release_model.set()
            snapshot = await execution
            later = (await client.get(f'/api/runs/{store.run_id}/events?after={cursor}&limit=500')).json()
            assert any(e['event_type'] == 'native_tool_result' and e['payload']['tool'] == 'str_replace_editor'
                       for e in later['events'])
            assert any(e['event_type'] == 'model_turn_finished' for e in later['events'])
        assert store.read_state()['phase'] == 'corpus_frozen'
        assert snapshot['evidence_count'] > 0
        assert json.loads((assigned['root'] / 'validation' / 'host-report.json').read_text(encoding='utf-8'))['replay_verified']
        names = {tool['function']['name'] for tool in requests[0]['tools']}
        assert names == {shell, 'str_replace_editor'}
        await native.run(AgentInvocation(run_id=store.run_id, session_id='downstream-surface',
            role='trajectory_inference', prompt='Reply done.', model=model.model_dump(mode='json')))
        assert not requests[-1].get('tools')
    finally:
        release_model.set()
        if not execution.done():
            execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)
        (tmp_path / 'provider-requests.json').write_text(json.dumps(requests, indent=2), encoding='utf-8')
        pipeline.close()
        await native.close_session('downstream-surface')
        server.shutdown()
        server.server_close()
