from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from app.runtime import dsh_worker


@dataclass
class _Result:
    finish_reason: str
    events: list[dict[str, Any]]
    session_id: str = "child_test"
    final_response: str = ""


class _Harness:
    def __init__(self, results: list[_Result]) -> None:
        self.results = list(results)
        self.calls: list[tuple[Any, str]] = []

    def run(self, prompt: Any, *, session_id: str, on_notification=None) -> _Result:
        self.calls.append((prompt, session_id))
        return self.results.pop(0)


def _failure(detail: str) -> _Result:
    return _Result(
        finish_reason="error",
        events=[
            {
                "type": "turn/end",
                "data": {"reason": {"kind": "error", "error": {"message": detail}}},
            }
        ],
    )


def _success() -> _Result:
    return _Result(
        finish_reason="completed",
        final_response="done",
        events=[
            {"type": "turn/end", "data": {"reason": {"kind": "completed"}}},
        ],
    )




def test_retry_notice_is_streamed_before_cooldown_and_redacts_key(monkeypatch):
    frames = []
    sleeps = []
    harness = _Harness([_failure("HTTP 503 api unavailable: test-secret"), _success()])
    monkeypatch.setattr(dsh_worker, "_write_frame", frames.append)

    def sleep(delay):
        assert frames[-1]["event"] == "model_retry_scheduled"
        assert len(harness.calls) == 1
        sleeps.append(delay)

    monkeypatch.setattr(dsh_worker.time, "sleep", sleep)
    result, _ = dsh_worker._run({"prompt": "task", "session_id": "child_test",
        "stream_events": True, "api_key": "test-secret", "api_max_retries": 1,
        "api_retry_base_delay_seconds": 10}, harness)
    assert result["ok"] is True and sleeps == [10]
    assert frames[-1]["payload"]["retry_number"] == 1
    assert "test-secret" not in str(frames)


@pytest.mark.parametrize(
    "request_id",
    (
        "905e6071-ddeb-4014-b703-654aa7a2ac69",
        "905e6071-ddeb-4038-b703-654aa7a2ac69",
    ),
)
def test_request_id_digits_do_not_turn_pi_ai_error_into_authentication_failure(
    request_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = (
        "An error occurred while processing your request. "
        f"Please include the request ID {request_id}. code: PI_AI_ERROR"
    )
    harness = _Harness([_failure(message), _success()])
    monkeypatch.setattr(dsh_worker.time, "sleep", lambda _: None)

    response, _ = dsh_worker._run(
        {
            "prompt": "original task",
            "session_id": "child_test",
            "api_max_retries": 12,
            "api_retry_base_delay_seconds": 0,
        },
        harness,
    )

    assert response["result"]["runtime_retry_count"] == 1
    assert dsh_worker._error_category(RuntimeError(message)) == "runtime_error"
    assert len(harness.calls) == 2


@pytest.mark.parametrize(
    "message",
    (
        "HTTP 401 unauthorized",
        "status code: 403",
        "code=401 invalid authentication",
        "401 status code (no body)",
        '{"status":403,"message":"forbidden"}',
    ),
)
def test_real_authentication_statuses_remain_non_retryable(message: str) -> None:
    assert dsh_worker._is_retryable_provider_failure(message) is False
    assert dsh_worker._error_category(RuntimeError(message)) == "authentication"


@pytest.mark.parametrize(
    "message",
    [
        "CONTEXT_WINDOW_EXCEEDED",
        "Invalid prompt: usage policy",
        "401 unauthorized",
        "insufficient_quota",
        "unsupported JSON schema",
        "HTTP 502: The 'gpt-5.5' model is not supported when using Codex with a ChatGPT account.",
        '{"code":"SERVER","message":"Unsupported model: example"}',
        '{"error":{"code":"model_not_found","message":"The model does not exist"}}',
    ],
)
def test_deterministic_provider_failures_are_not_retried(
    message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness([_failure(message), _success()])
    monkeypatch.setattr(dsh_worker.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="DSH turn ended"):
        dsh_worker._run(
            {
                "prompt": "original task",
                "session_id": "child_test",
                "api_max_retries": 5,
            },
            harness,
        )

    assert len(harness.calls) == 1


@pytest.mark.parametrize("stream_events", [True, False])
def test_unsupported_model_stops_sdk_at_first_retry_notice(monkeypatch, stream_events):
    frames = []
    message = "502: The 'gpt-5.5' model is not supported when using Codex with a ChatGPT account."

    class RetryHarness(_Harness):
        closed = False

        def run(self, prompt, *, session_id, on_notification):
            self.calls.append((prompt, session_id))
            on_notification(SimpleNamespace(method="session.event", payload={
                "sessionId": session_id, "event": {"type": "llm/retry", "data": {
                    "retry": 1, "maxRetries": 5, "delayMs": 500,
                    "failure": {"code": "SERVER", "message": message},
                }}}))
            pytest.fail("The native model request must stop before retrying")

        def close(self):
            self.closed = True

    harness = RetryHarness([])
    monkeypatch.setattr(dsh_worker, "_write_frame", frames.append)
    monkeypatch.setattr(dsh_worker.time, "sleep", lambda _: pytest.fail("Unsupported models cannot enter cooldown"))
    with pytest.raises(RuntimeError, match="not supported"):
        dsh_worker._run({"prompt": "task", "session_id": "child_test",
                         "stream_events": stream_events, "api_max_retries": 12}, harness)
    assert len(harness.calls) == 1 and harness.closed
    if stream_events:
        assert [f["event"] for f in frames] == ["model_retry_stopped"]
    else:
        assert frames == []


def test_temporary_model_unavailability_still_retries():
    assert dsh_worker._is_retryable_provider_failure("HTTP 503: model temporarily unavailable")


@pytest.mark.integration
def test_real_sdk_does_not_repeat_unsupported_model_request(tmp_path, monkeypatch):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path

    pytest.importorskip("deepseek_harness_runtime")
    from app.runtime import AgentInvocation, DshBridgeConfig

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append(self.path)
            body = json.dumps({"error": {
                "message": "The 'model-test' model is not supported when using Codex with a ChatGPT account.",
                "type": "upstream_error", "code": "unsupported_model",
            }}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    frames = []
    monkeypatch.setattr(dsh_worker, "_write_frame", frames.append)
    try:
        payload = DshBridgeConfig(workspace=Path.cwd(), dsh_home=str(tmp_path / "dsh"),
                                  api_max_retries=0, request_timeout_seconds=10).worker_payload(
            AgentInvocation(run_id="run_test", session_id="child_test", role="root", prompt="Test request.",
                            model={"model": "model-test", "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                                   "api_key": "fake-test-key", "api_protocol": "chat_completions"},
                            metadata={"runtime_event_sink": lambda *_args: None}))
        with pytest.raises(RuntimeError, match="not supported"):
            dsh_worker._run(payload, None)
        assert received == ["/v1/chat/completions"]
        assert any(f["event"] == "model_retry_stopped" for f in frames)
        assert not any(f["event"] == "model_retry_scheduled" for f in frames)
        assert "fake-test-key" not in str(frames)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_transient_provider_failure_ramps_retries_in_same_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness([_failure("HTTP 503") for _ in range(12)] + [_success()])
    delays: list[float] = []
    monkeypatch.setattr(dsh_worker.time, "sleep", delays.append)

    response, returned_harness = dsh_worker._run(
        {
            "prompt": "original task",
            "session_id": "child_test",
            "api_max_retries": 12,
            "api_retry_base_delay_seconds": 10,
        },
        harness,
    )

    assert returned_harness is harness
    assert response["result"]["runtime_retry_count"] == 12
    assert len(harness.calls) == 13
    assert all(session_id == "child_test" for _, session_id in harness.calls)
    assert harness.calls[0][0] == "original task"
    assert all(prompt == dsh_worker._RETRY_CONTINUATION_PROMPT for prompt, _ in harness.calls[1:])
    assert delays == [10, 20, *([30] * 10)]
