"""Shipped DSH web plugins against local scripted model/search services.

No external provider credentials or official searches are needed for this smoke test.
"""
import json
import re
import threading
import fitz
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from app.capabilities import CapabilityGateway, CapabilityGatewayServer
from app.orchestration.source_discovery_stage import SourceDiscoveryService
from app.runtime import AgentInvocation, AgentRuntimeConfigurationError, DshAgentRuntime, DshBridgeConfig
from app.runtime.dsh_events import RuntimeEventStream
from test_source_discovery import candidate, setup_discovery


def test_search_credentials_are_required_independently_of_the_model(monkeypatch):
    monkeypatch.setenv("DSH_WEB_SEARCH_PROVIDER", "deepseek-official")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from app.runtime.dsh_client import native_web_environment
    with pytest.raises(AgentRuntimeConfigurationError, match="deployment-side DEEPSEEK_API_KEY"):
        native_web_environment()


def test_gpt_only_credentials_select_responses_search(monkeypatch):
    from app.runtime.dsh_client import native_web_environment
    monkeypatch.delenv("DSH_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = native_web_environment({"base_url": "http://example.org/v1", "model": "gpt-test", "api_key": "model-key"})
    assert result["DSH_WEB_SEARCH_PROVIDER"] == "openai-responses"
    assert json.loads(result["RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG"])["apiKey"] == "model-key"
    assert "DEEPSEEK_API_KEY" not in result
    monkeypatch.setenv("DEEPSEEK_API_KEY", "search-only-test-key")
    monkeypatch.delenv("DEEPSEEK_SEARCH_BASE_URL", raising=False)
    assert native_web_environment()["DEEPSEEK_SEARCH_BASE_URL"] == "https://api.deepseek.com/anthropic/v1"
    monkeypatch.setenv("DSH_WEB_SEARCH_PROVIDER", "other-provider")
    with pytest.raises(AgentRuntimeConfigurationError, match="deepseek-official"):
        native_web_environment()


def test_native_web_surface_is_restricted_to_discovery():
    config = DshBridgeConfig()
    with pytest.raises(AgentRuntimeConfigurationError, match="Source Discovery"):
        config.worker_payload(AgentInvocation(run_id="run", session_id="child", role="skill_induction",
            prompt="", model={}, allowed_tools=("submit_source_candidates",), metadata={"native_web_tools": True}))


def test_native_web_events_redact_search_secrets_and_skip_business_duplicates():
    frames = []
    stream = RuntimeEventStream({"stream_events": True, "native_web_tools": True,
        "session_id": "child", "api_key": "model-only-test", "native_web_env": {"DEEPSEEK_API_KEY": "search-only-test"}}, frames.append)
    def send(kind, data):
        stream.notification(SimpleNamespace(method="session.event", payload={"sessionId": "child",
            "event": {"type": kind, "data": data}}))
    send("tool/call", {"name": "web_search", "callId": "search1", "arguments": {"queries": ["Example manual"]}})
    send("tool/result", {"message": {"source": {"callId": "search1"}, "content": [{
        "type": "tool-result", "content": [{"type": "text", "text": "search-only-test model-only-test"}]}]}})
    send("tool/call", {"name": "submit_source_candidates", "callId": "submit1", "arguments": {}})
    send("tool/result", {"message": {"source": {"callId": "submit1"}, "content": []}})
    assert len(frames) == 2 and frames[1]["payload"]["tool"] == "web_search"
    assert "search-only-test" not in json.dumps(frames)
    assert "model-only-test" not in json.dumps(frames)


def test_long_search_result_retains_pdf_links_outside_the_audit_preview():
    frames = []
    stream = RuntimeEventStream({'stream_events': True, 'native_web_tools': True,
        'session_id': 'child'}, frames.append)
    text = 'Untrusted retrieval context. ' * 500 + '\nDISCOVERY SEARCH EVIDENCE\n' + json.dumps({
        'sources': [{'url': 'https://example.org/manual.pdf', 'title': 'Full handbook',
                     'snippet': 'Version 1.3.2'}]})
    for kind, data in [('tool/call', {'name': 'web_search', 'callId': 'long_search', 'arguments': {}}),
                       ('tool/result', {'message': {'source': {'callId': 'long_search'}, 'content': [
                           {'type': 'tool-result', 'content': [{'type': 'text', 'text': text}]}]}})]:
        stream.notification(SimpleNamespace(method='session.event', payload={'sessionId': 'child',
            'event': {'type': kind, 'data': data}}))
    result = frames[-1]['payload']
    assert result['output']['truncated'] is True
    assert result['retrieved_sources'] == [{'url': 'https://example.org/manual.pdf',
        'title': 'Full handbook', 'snippet': 'Version 1.3.2'}]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("search_backend", ["deepseek-official", "openai-responses"])
@pytest.mark.parametrize("search_timed_out", [False, True], ids=["normal", "timeout"])
async def test_pinned_dsh_native_search_uses_separate_route_and_only_discovery_tools(
    tmp_path, monkeypatch, search_backend, search_timed_out,
):
    store, target, model = setup_discovery(tmp_path / "run")
    model_requests, searches = [], []
    source = candidate(url="https://example.org/manual-1.0.pdf", modality="pdf", root_assessment="PDF listing confirms file.")
    release_search = threading.Event()
    with fitz.open() as document:
        document.new_page().insert_text((72, 72), 'Example User Guide\nFor version 1.0')
        pdf_bytes = document.tobytes()
    if search_timed_out:
        # Exercise the production overlay/policy with a short test-only deadline.
        # Without the mounted policy, the slow request succeeds and this test fails.
        overlay = tmp_path / "discovery-timeout.patch.yml"
        original_overlay = DshBridgeConfig().workspace / "runtime/dsh/source-discovery.patch.yml"
        overlay.write_text(
            re.sub(r"(?m)^(\s*searchTimeoutMs:).*$", r"\1 250",
                   original_overlay.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        original_payload = DshBridgeConfig.worker_payload

        def short_search_timeout(self, invocation):
            payload = original_payload(self, invocation)
            payload["patches"] = [
                str(overlay) if path.endswith("source-discovery.patch.yml") else path
                for path in payload["patches"]
            ]
            return payload

        monkeypatch.setattr(DshBridgeConfig, "worker_payload", short_search_timeout)

    def wait_for_search():
        if search_timed_out:
            release_search.wait(timeout=2)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def write_response(self, data):
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                if not search_timed_out:
                    raise

        def do_GET(self):
            assert self.path == '/manual-1.0.pdf'
            self.send_response(200)
            self.send_header('Content-Type', 'application/pdf')
            self.send_header('Content-Length', str(len(pdf_bytes)))
            self.end_headers()
            self.write_response(pdf_bytes)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/responses":
                assert self.headers.get("Authorization") == "Bearer model-test-key"
                searches.append(payload)
                wait_for_search()
                value = {"status": "completed", "output": [
                    {"type": "web_search_call", "status": "completed", "action": {
                        "type": "search", "sources": [{"url": source["url"], "title": source["title"]}]}},
                    {"type": "message", "content": [{"type": "output_text", "text": "Search completed",
                        "annotations": [{"type": "url_citation", "url": source["url"], "title": source["title"]}]}]},
                ]}
                data = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.write_response(data)
                return
            if self.path == "/anthropic/v1/messages":
                assert self.headers.get("x-api-key") == "search-only-test-key"
                assert "model-test-key" not in str(self.headers)
                searches.append(payload)
                wait_for_search()
                value = {"content": [{"type": "web_search_tool_result", "tool_use_id": "search_1", "content": [{
                    "type": "web_search_result", "url": source["url"], "title": source["title"], "page_age": "2026-01-01"}]}]}
                data = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.write_response(data)
                return
            assert self.path == "/v1/chat/completions"
            assert self.headers.get("Authorization") == "Bearer model-test-key"
            model_requests.append(payload)
            turn = len(model_requests)
            if turn == 1:
                name, arguments = "web_search", {"queries": ["Example 1.0 official manual PDF"]}
            elif turn == 2 and not search_timed_out:
                assert "DISCOVERY BUDGET UPDATE" in json.dumps(payload["messages"])
                if search_backend == 'openai-responses':
                    assert 'DISCOVERY SEARCH EVIDENCE' in json.dumps(payload['messages'])
                    assert 'Search completed' in json.dumps(payload['messages'])
                name, arguments = 'inspect_source_candidate', {'url': source['url']}
            elif turn == 3 and not search_timed_out:
                assert 'For version 1.0' in json.dumps(payload['messages'])
                name, arguments = "save_source_candidate", {"candidate": source}
            elif turn == 2 or (turn == 4 and not search_timed_out):
                if search_timed_out:
                    assert "tool call timed out after 250ms" in json.dumps(payload["messages"])
                name, arguments = "submit_source_candidates", {
                    "candidates": [],  # Prior checkpoints must be retained.
                    "summary": "Search timed out" if search_timed_out else "One exact PDF",
                    "search_notes": ["Search deadline exceeded."] if search_timed_out else ["Official listing located."],
                }
            else:
                name, arguments = None, None
            delta = {"role": "assistant"}
            if name:
                encoded_arguments = json.dumps(arguments)
                split_at = len(encoded_arguments) // 2
                delta["tool_calls"] = [{"index": 0, "id": f"call_{turn}", "type": "function",
                    "function": {"name": name, "arguments": encoded_arguments[:split_at]}}]
            else:
                delta["content"] = "Done"
            chunk = {"id": f"web_smoke_{turn}", "object": "chat.completion.chunk", "created": 1,
                "model": "gpt-test", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            final = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if name else "stop"}]}
            chunks = [chunk]
            if name:
                chunks.append({**chunk, "choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": 0, "function": {"arguments": encoded_arguments[split_at:]},
                }]}, "finish_reason": None}]})
            chunks.append(final)
            data = ("".join("data: " + json.dumps(item) + "\n\n" for item in chunks) + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    source['url'] = f'http://127.0.0.1:{server.server_port}/manual-1.0.pdf'
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "search-only-test-key")
    monkeypatch.setenv("DSH_WEB_SEARCH_PROVIDER", search_backend)
    if search_backend == "openai-responses":
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_SEARCH_BASE_URL", f"http://127.0.0.1:{server.server_port}/anthropic/v1")
    native = DshAgentRuntime(replace(DshBridgeConfig(), dsh_home=str(tmp_path / "dsh-home"),
        initialize_timeout_seconds=30, request_timeout_seconds=45, api_max_retries=0, shutdown_timeout_seconds=5))
    gateway = CapabilityGateway()
    gateway_server = CapabilityGatewayServer(gateway)
    model = model.model_copy(update={"base_url": f"http://127.0.0.1:{server.server_port}/v1"})
    try:
        result = await SourceDiscoveryService(store=store, settings=SimpleNamespace(source_discovery_timeout_seconds=55),
            runtime=native, gateway=gateway, gateway_server=gateway_server, model_config=model).discover(
                target_profile=target, output_language="English")
        assert len(result.candidates) == (0 if search_timed_out else 1)
        if not search_timed_out:
            assert result.candidates[0].modality == "pdf"
            assert store.read_state()["source_discovery_checkpoint_artifact_id"]
        assert len(searches) == 1
        assert searches[0]["tools"][0]["type"] == (
            "web_search_20250305" if search_backend == "deepseek-official" else "web_search")
        assert {tool["function"]["name"] for tool in model_requests[0]["tools"]} == {
            "web_search", "web_fetch", "save_source_candidate", "submit_source_candidates",
            "list_discovered_sources", "inspect_source_candidate", "next_source_validation"}
        assert native._workers == {} and not native._run_semaphore.locked()
        audit = store.audit_path.read_text(encoding="utf-8")
        assert "native_tool_result" in audit and "web_search" in audit
        if search_timed_out:
            events = [json.loads(line) for line in audit.splitlines()]
            tool_result = next(row["payload"] for row in events
                               if row["event_type"] == "native_tool_result" and row["payload"].get("tool") == "web_search")
            assert tool_result["ok"] is False
            assert "tool call timed out after 250ms" in tool_result["output"]
        assert "search-only-test-key" not in audit and "model-test-key" not in audit
        trace_root = tmp_path / "dsh-home" / "http-traces" / store.run_id
        trace_files = list(trace_root.rglob("*.meta.json"))
        assert trace_files, "The real SDK must capture HTTP before protocol adaptation"
        traces = [json.loads(path.read_text(encoding="utf-8")) for path in trace_files]
        model_trace = next(path for path in trace_files if
            json.loads(path.read_text(encoding="utf-8"))["url"].endswith("/chat/completions"))
        request_trace = json.loads(model_trace.with_name(
            model_trace.name.replace(".meta.json", ".request.json")).read_text(encoding="utf-8"))
        web_tool = next(tool["function"] for tool in request_trace["tools"]
                        if tool["function"]["name"] == "web_search")
        assert web_tool["parameters"]["required"] == ["queries"]
        assert any(item["complete"] and item["status"] == 200 for item in traces)
        bodies = "\n".join(path.read_text(encoding="utf-8")
                            for path in trace_root.rglob("*.response.body"))
        assert "call_1" in bodies and "queries" in bodies
        all_trace_text = "\n".join(path.read_text(encoding="utf-8")
                                    for path in trace_root.rglob("*") if path.is_file())
        assert "model-test-key" not in all_trace_text and "search-only-test-key" not in all_trace_text
        # A subsequent downstream session receives no web tools.
        await native.run(AgentInvocation(run_id=store.run_id, session_id="downstream", role="skill_induction",
            prompt="Reply done.", model=model.model_dump(mode="json")))
        assert not model_requests[-1].get("tools")
    finally:
        release_search.set()
        await native.close_session("downstream")
        gateway_server.close()
        server.shutdown()
        server.server_close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_http_trace_retains_empty_wire_arguments_and_matches_the_executed_call(tmp_path, monkeypatch):
    store, target, model = setup_discovery(tmp_path / "run")
    replies = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/chat/completions"
            turn = len(replies) + 1
            if turn == 1:
                name, arguments = "web_search", {}
            elif turn == 2:
                assert 'missing required property' in json.dumps(payload["messages"])
                assert 'TOOL_INPUT_SCHEMA_ERROR' in json.dumps(payload["messages"])
                assert 'queries' in json.dumps(payload["messages"])
                name, arguments = "submit_source_candidates", {
                    "candidates": [], "summary": "The search call had no queries.",
                    "search_notes": ["Invalid search arguments."],
                }
            else:
                name, arguments = None, None
            delta = {"role": "assistant"}
            if name:
                delta["tool_calls"] = [{"index": 0, "id": f"empty_call_{turn}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)}}]
            else:
                delta["content"] = "Done"
            chunk = {"id": f"empty_trace_{turn}", "object": "chat.completion.chunk", "created": 1,
                "model": "gpt-test", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            end = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if name else "stop"}]}
            data = ("".join("data: " + json.dumps(item) + "\n\n" for item in [chunk, end])
                    + "data: [DONE]\n\n").encode()
            replies.append(data)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("DSH_WEB_SEARCH_PROVIDER", "openai-responses")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    native = DshAgentRuntime(replace(DshBridgeConfig(), dsh_home=str(tmp_path / "dsh-home"),
        initialize_timeout_seconds=30, request_timeout_seconds=30, api_max_retries=0))
    gateway = CapabilityGateway()
    gateway_server = CapabilityGatewayServer(gateway)
    model = model.model_copy(update={"base_url": f"http://127.0.0.1:{server.server_port}/v1"})
    try:
        result = await SourceDiscoveryService(store=store,
            settings=SimpleNamespace(source_discovery_timeout_seconds=40), runtime=native,
            gateway=gateway, gateway_server=gateway_server, model_config=model).discover(
                target_profile=target, output_language="English")
        assert result.candidates == ()
        assert 2 <= len(replies) <= 3
        trace_root = tmp_path / "dsh-home/http-traces" / store.run_id
        first = next(trace_root.rglob("http_000001_*.meta.json"))
        meta = json.loads(first.read_text(encoding="utf-8"))
        body = first.with_name(meta["response_body_file"]).read_bytes()
        assert body == replies[0]
        call = json.loads(body.decode().split("data: ", 1)[1].split("\n\n", 1)[0])
        wire_call = call["choices"][0]["delta"]["tool_calls"][0]
        assert wire_call["function"]["arguments"] == "{}"
        events = [json.loads(line) for line in store.audit_path.read_text(encoding="utf-8").splitlines()]
        executed = next(event["payload"] for event in events if event["event_type"] == "native_tool_call"
                        and event["payload"].get("call_id") == wire_call["id"])
        assert executed["arguments"] == {}
        assert meta["complete"] and not meta["truncated"]
    finally:
        gateway_server.close()
        server.shutdown()
        server.server_close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pinned_dsh_round_limit_and_deadline_preserve_saved_manual(tmp_path, monkeypatch):
    """A real DSH loop cannot issue a fourth search or lose a checkpoint on timeout."""
    store, target, model = setup_discovery(tmp_path / "run")
    source = candidate(modality="html", url="https://example.org/manual/", root_assessment="Full manual entry and chapter contents verified.")
    requests, searches = [], []
    release = threading.Event()
    extra_search_denied = threading.Event()
    original_payload = DshBridgeConfig.worker_payload

    def test_budget(self, invocation):
        payload = original_payload(self, invocation)
        # Only accelerate the test's clock; production keeps its 90-second wrap-up.
        payload["capability_config"]["discoveryBudget"]["wrap_up_seconds"] = 0
        return payload

    monkeypatch.setattr(DshBridgeConfig, "worker_payload", test_budget)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/responses":
                searches.append(payload)
                data = json.dumps({"status": "completed", "output": [
                    {"type": "web_search_call", "status": "completed", "action": {
                        "sources": [{"url": source["url"], "title": source["title"]}]}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            requests.append(payload)
            turn = len(requests)
            if turn == 2:
                name, arguments = "save_source_candidate", {"candidate": source}
            elif turn <= 5:
                name, arguments = "web_search", {"queries": [f"Example exact manual followup {turn}"]}
            else:
                text = json.dumps(payload["messages"])
                assert "Search rounds are exhausted" in text
                assert "Saved candidates: 1" in text
                extra_search_denied.set()
                release.wait(timeout=20)
                name, arguments = None, None
            delta = {"role": "assistant"}
            if name:
                delta["tool_calls"] = [{"index": 0, "id": f"call_{turn}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)}}]
            else:
                delta["content"] = "Done"
            chunk = {"id": f"checkpoint_{turn}", "object": "chat.completion.chunk", "created": 1,
                "model": "gpt-test", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            final = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if name else "stop"}]}
            data = ("".join("data: " + json.dumps(item) + "\n\n" for item in [chunk, final]) + "data: [DONE]\n\n").encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("DSH_WEB_SEARCH_PROVIDER", "openai-responses")
    model = model.model_copy(update={"base_url": f"http://127.0.0.1:{server.server_port}/v1"})
    native = DshAgentRuntime(replace(DshBridgeConfig(), dsh_home=str(tmp_path / "dsh-home"),
        request_timeout_seconds=30, api_max_retries=0))
    gateway = CapabilityGateway()
    gateway_server = CapabilityGatewayServer(gateway)
    try:
        result = await SourceDiscoveryService(store=store, settings=SimpleNamespace(source_discovery_timeout_seconds=12),
            runtime=native, gateway=gateway, gateway_server=gateway_server, model_config=model).discover(
                target_profile=target, output_language="English")
        assert extra_search_denied.is_set()
        assert len(searches) == 3
        assert len(result.candidates) == 1
        assert result.candidates[0].url == source["url"]
        saved = store.read_json_artifact(result.candidate_artifact_id)
        assert saved["completion_reason"] == "time_budget_reached"
        assert store.read_attempt(result.attempt_id).status == "succeeded"
        assert native._workers == {} and not native._run_semaphore.locked()
    finally:
        release.set()
        gateway_server.close()
        server.shutdown()
        server.server_close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pinned_discovery_reserves_broad_search_and_recovers_overlooked_pdf(tmp_path, monkeypatch):
    store, target, model = setup_discovery(tmp_path / 'run')
    searches, requests = [], []
    with fitz.open() as document:
        document.new_page().insert_text((72, 72), 'Example user manual\nFor version 0.9\nFull handbook')
        pdf = document.tobytes()
    source = candidate(modality='pdf', document_version='0.9', version_relation='older',
        root_assessment='The inspected PDF is a full user handbook for an older version.')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/pdf')
            self.send_header('Content-Length', str(len(pdf)))
            self.end_headers()
            self.wfile.write(pdf)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/v1/responses':
                searches.append(payload['input'])
                hits = [{'url': f'https://example.org/unrelated/{index}', 'title': 'Unrelated',
                         'snippet': 'Unrelated retrieval context. ' * 45} for index in range(14)]
                hits.append({'url': source['url'], 'title': 'Example User Guide 0.9', 'snippet': 'Full user handbook PDF.'})
                body = json.dumps({'status': 'completed', 'output': [
                    {'type': 'web_search_call', 'status': 'completed', 'action': {'sources': hits}},
                    {'type': 'message', 'content': [{'type': 'output_text', 'text': 'A nearby-version full handbook is available.'}]},
                ]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            requests.append(payload)
            turn = len(requests)
            text = json.dumps(payload['messages'])
            calls = {
                1: ('web_search', {'queries': ['Example 1.0 official manual']}),
                2: ('web_search', {'queries': ['Example 1.0 handbook']}),
                3: ('web_search', {'queries': ['Example 1.0 user guide PDF']}),
                4: ('web_search', {'queries': ['Example user handbook filetype:pdf']}),
                5: ('list_discovered_sources', {}),
                6: ('inspect_source_candidate', {'url': source['url']}),
                7: ('save_source_candidate', {'candidate': source}),
                8: ('submit_source_candidates', {'candidates': [], 'summary': 'Older handbook verified and retained'}),
            }
            if turn == 4:
                assert 'Reserve this final search for a version-free' in text
            if turn == 6:
                assert 'pending_count' in text and source['url'] in text
            if turn == 7:
                assert 'For version 0.9' in text
            name, arguments = calls.get(turn, (None, None))
            delta = {'role': 'assistant'}
            if name:
                delta['tool_calls'] = [{'index': 0, 'id': f'call_{turn}', 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(arguments)}}]
            else:
                delta['content'] = 'Done'
            chunk = {'id': f'recovery_{turn}', 'object': 'chat.completion.chunk', 'created': 1,
                'model': 'gpt-test', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}
            final = {**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if name else 'stop'}]}
            body = (''.join('data: ' + json.dumps(item) + '\n\n' for item in [chunk, final]) + 'data: [DONE]\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    source['url'] = f'http://127.0.0.1:{server.server_port}/UserGuide-0.9.pdf'
    monkeypatch.setenv('DSH_WEB_SEARCH_PROVIDER', 'openai-responses')
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    model = model.model_copy(update={'base_url': f'http://127.0.0.1:{server.server_port}/v1'})
    native = DshAgentRuntime(replace(DshBridgeConfig(), dsh_home=str(tmp_path / 'dsh-home'),
        request_timeout_seconds=45, api_max_retries=0))
    gateway = CapabilityGateway()
    gateway_server = CapabilityGatewayServer(gateway)
    try:
        result = await SourceDiscoveryService(store=store, settings=SimpleNamespace(source_discovery_timeout_seconds=50),
            runtime=native, gateway=gateway, gateway_server=gateway_server, model_config=model).discover(
                target_profile=target, output_language='English')
        assert len(searches) == 3
        assert searches[-1].endswith('Example user handbook filetype:pdf')
        assert len(result.candidates) == 1 and result.candidates[0].version_relation == 'older'
        assert result.candidates[0].verification['header_verified']
        queue = store.read_json_artifact(store.read_state()['source_discovery_queue_artifact_id'])
        assert any(item['url'] == source['url'] and item['review_status'] == 'inspected' for item in queue['sources'])
        events, _ = store.read_events(limit=500)
        first_search = next(row['payload'] for row in events if row['event_type'] == 'native_tool_result'
                            and row['payload'].get('tool') == 'web_search')
        assert first_search['output']['truncated'] is True
        assert any(item['url'] == source['url'] for item in first_search['retrieved_sources'])
    finally:
        gateway_server.close()
        server.shutdown()
        server.server_close()
