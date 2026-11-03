import asyncio
import json
from types import SimpleNamespace

import pytest

from app.runtime import AgentInvocation, DshAgentRuntime, DshBridgeConfig
from app.runtime.dsh_events import RuntimeEventStream


def test_public_events_keep_tool_results_bounded_and_hide_credentials_and_reasoning():
    frames = []
    stream = RuntimeEventStream({"stream_events": True, "native_source_tools": True,
        "session_id": "child", "model": "test", "api_key": "secret-value"}, frames.append)

    def send(kind, data, session="child"):
        stream.notification(SimpleNamespace(method="session.event", payload={
            "sessionId": session, "event": {"type": kind, "time": 123, "data": data}}))

    send("step/start", {"turn": 1, "step": 1})
    send("request/header", {"system": "private system", "api_key": "secret-value"})
    send("assistant/chunk", {"text": "private reasoning", "step": 1})
    send("assistant/chunk", {"text": "private reasoning", "step": 1})
    send("tool/call", {"name": "pwsh", "callId": "other", "arguments": "{}"}, "other-session")
    send("tool/call", {"name": "pwsh", "callId": "c1", "arguments": json.dumps({
        "command": "echo secret-value", "api_key": "different-secret"})})
    send("tool/result", {"message": {"source": {"callId": "c1"}, "content": [{
        "type": "tool-result", "isError": True, "content": [{"type": "text",
        "text": "secret-value Bearer access-token " + "结果" * 20000}]}]}, "error": {"code": "EFAIL"}})
    serialized = json.dumps(frames, ensure_ascii=False)
    assert "secret-value" not in serialized
    assert "different-secret" not in serialized
    assert "access-token" not in serialized
    assert "private reasoning" not in serialized
    assert "private system" not in serialized
    assert "other-session" not in serialized
    assert [f["event"] for f in frames].count("model_response_streaming") == 1
    result = frames[-1]["payload"]
    assert result["tool"] == "pwsh" and result["call_id"] == "c1"
    assert result["ok"] is False and result["output"]["truncated"] is True
    assert len(json.dumps(frames[-1], ensure_ascii=False).encode("utf-8")) < 64000


def test_sdk_retry_events_expose_counts_delay_and_redacted_error():
    frames = []
    stream = RuntimeEventStream({"stream_events": True, "session_id": "child", "api_key": "test-secret"}, frames.append)
    for kind, data in [
        ("llm/retry", {"turn": 1, "step": 2, "retry": 3, "maxRetries": 5, "delayMs": 2500,
                       "failure": {"message": "HTTP 503 test-secret Bearer hidden-token"}}),
        ("llm/retry-started", {"turn": 1, "step": 2, "retry": 3}),
        ("turn/end", {"reason": {"kind": "error", "error": {"message": "HTTP 503 test-secret"}}}),
    ]:
        stream.notification(SimpleNamespace(method="session.event", payload={
            "sessionId": "child", "event": {"type": kind, "data": data}}))
    assert [f["event"] for f in frames] == ["model_retry_scheduled", "model_retry_started", "model_turn_finished"]
    retry = frames[0]["payload"]
    assert retry["retry_number"] == 3 and retry["max_retries"] == 5 and retry["delay_seconds"] == 2.5
    assert "test-secret" not in str(frames) and "hidden-token" not in str(frames)
    assert frames[-1]["payload"]["status"] == "error"


@pytest.mark.asyncio
async def test_bridge_delivers_events_and_heartbeat_before_final_response(monkeypatch):
    from app.runtime import dsh_client
    monkeypatch.setattr(dsh_client, "_EVENT_HEARTBEAT_SECONDS", 0.02)
    events = []
    reader = asyncio.StreamReader()
    runtime = DshAgentRuntime()
    invocation = AgentInvocation(run_id="run", session_id="child", role="source_review", prompt="",
        model={}, metadata={"runtime_event_sink": lambda e, p: events.append((e, p))})
    task = asyncio.create_task(runtime._read_response(SimpleNamespace(stdout=reader), invocation,
                                                     {"api_key": "top-secret"}))
    reader.feed_data((json.dumps({"type": "runtime_event", "event": "native_tool_call", "payload": {
        "child_session_id": "child", "tool": "pwsh", "arguments": "top-secret"}}) + "\n").encode())
    await asyncio.sleep(0.06)
    assert not task.done()
    assert events[0][0] == "native_tool_call"
    assert events[-1][0] == "model_waiting"
    assert events[-1][1]["status"] == "executing_tool"
    assert "top-secret" not in json.dumps(events)
    response = b'{"ok":true,"result":{}}\n'
    reader.feed_data(response)
    assert await task == response


@pytest.mark.asyncio
async def test_bridge_stream_does_not_reset_absolute_timeout(monkeypatch):
    from app.runtime import dsh_client
    # The bridge reserves 5 seconds after the configured SDK deadline.
    runtime = DshAgentRuntime(DshBridgeConfig(request_timeout_seconds=-4.92))
    monkeypatch.setattr(dsh_client, "_EVENT_HEARTBEAT_SECONDS", 0.01)
    reader = asyncio.StreamReader()
    invocation = AgentInvocation(run_id="run", session_id="child", role="source_review", prompt="", model={})
    task = asyncio.create_task(runtime._read_response(SimpleNamespace(stdout=reader), invocation, {}))
    try:
        for _ in range(15):
            reader.feed_data(b'{"type":"runtime_event","event":"model_response_streaming","payload":{"child_session_id":"child"}}\n')
            await asyncio.sleep(0.01)
            if task.done():
                break
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, 0.2)
        # A timed out read must not leave a concurrent reader on this pipe.
        reader.feed_data(b'next\n')
        assert (await reader.readline()).endswith(b'\n')
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_stream_releases_pending_pipe_reader():
    reader = asyncio.StreamReader()
    runtime = DshAgentRuntime()
    invocation = AgentInvocation(run_id="run", session_id="child", role="source_review", prompt="", model={})
    task = asyncio.create_task(runtime._read_response(SimpleNamespace(stdout=reader), invocation, {}))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    reader.feed_data(b'next\n')
    assert await reader.readline() == b'next\n'
