"""Wire-level contract checks for the tool-free HTML planning client."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.html_guided import model as model_module
from app.html_guided.model import HtmlPlanRuntime
from app.models import ModelRuntimeConfig
from app.runtime import AgentInvocation, AgentRuntimeConfigurationError


SECRET = "sk-unit-test-html-planner-header-only"
PLAN = {"page_type": "article", "content_root_id": "root_existing", "links": [], "summary": "Keep the captured article."}


def invocation(*, base_url="https://api.deepseek.com", protocol="chat_completions", prompt="Plan the captured HTML snapshot.", events=None, tools=()):
    model = ModelRuntimeConfig(
        base_url=base_url, api_protocol=protocol, model="deepseek-flash",
        api_key=SECRET, reasoning_effort="none",
    ).model_dump(mode="json")
    return AgentInvocation(
        run_id="run_direct_html", session_id="child_html_plan",
        role="html_acquisition_planning", prompt=prompt, model=model,
        allowed_tools=tools,
        metadata={"runtime_event_sink": lambda event, payload: events.append((event, payload))} if events is not None else {},
    )


def chat_response(content=None):
    return {
        "choices": [{"message": {"role": "assistant", "content": content or json.dumps(PLAN)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32,
                  "prompt_tokens_details": {"cached_tokens": 5},
                  "completion_tokens_details": {"reasoning_tokens": 3}},
    }


def client(handler, **overrides):
    settings = Settings(api_timeout_seconds=5, **overrides)
    return HtmlPlanRuntime(settings, transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", ["https://api.deepseek.com", "https://api.deepseek.com/chat/completions"])
async def test_chat_uses_configured_model_and_explicitly_disables_deepseek_thinking(base_url):
    requests, events = [], []

    def serve(request):
        requests.append(request)
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert request.headers["authorization"] == "Bearer " + SECRET
        body = json.loads(request.content)
        assert body["model"] == "deepseek-flash"
        assert body["thinking"] == {"type": "disabled"}
        assert body["reasoning_effort"] == "none"
        assert body["max_tokens"] == 8192
        assert body.get("stream", False) is False
        assert body["response_format"] == {"type": "json_object"}
        assert not body.get("tools")
        assert any(message["content"] == "Plan the captured HTML snapshot." for message in body["messages"])
        assert SECRET not in request.content.decode("utf-8")
        return httpx.Response(200, json=chat_response())

    runtime = client(serve)
    result = await runtime.run(invocation(base_url=base_url, events=events))
    assert len(requests) == 1
    assert json.loads(result.final_response) == PLAN
    assert result.session_id == "child_html_plan"
    assert result.usage["inputTokens"] == 15
    assert result.usage["cacheReadTokens"] == 5
    assert result.usage["outputTokens"] == 12
    assert result.usage["totalTokens"] == 32
    assert result.usage["reasoningTokens"] == 3
    assert events
    assert SECRET not in json.dumps(events)
    await runtime.close_session(result.session_id)


@pytest.mark.asyncio
async def test_responses_uses_explicit_none_without_changing_endpoint_or_model():
    requests = []

    def serve(request):
        requests.append(request)
        assert str(request.url) == "https://model.example/v1/responses"
        body = json.loads(request.content)
        assert body["model"] == "deepseek-flash"
        assert body["reasoning"] == {"effort": "none"}
        assert body["max_output_tokens"] == 8192
        assert body["text"]["format"] == {"type": "json_object"}
        assert "thinking" not in body
        assert not body.get("tools")
        assert SECRET not in request.content.decode("utf-8")
        return httpx.Response(200, json={
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": json.dumps(PLAN)}]}],
            "usage": {"input_tokens": 22, "output_tokens": 7, "total_tokens": 29,
                      "input_tokens_details": {"cached_tokens": 2}},
        })

    runtime = client(serve)
    result = await runtime.run(invocation(base_url="https://model.example/v1/responses", protocol="responses"))
    assert len(requests) == 1
    assert json.loads(result.final_response) == PLAN
    assert result.usage["inputTokens"] == 20
    assert result.usage["cacheReadTokens"] == 2
    assert result.usage["outputTokens"] == 7
    await runtime.close_session(result.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_authentication_error_is_not_retried_and_never_exposes_provider_echo(status):
    requests, events = [], []

    def serve(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "Rejected secret " + SECRET}})

    runtime = client(serve, api_max_attempts=99)
    with pytest.raises(Exception) as error:
        await runtime.run(invocation(events=events))
    assert len(requests) == 1
    assert getattr(error.value, "category", None) == "authentication"
    assert SECRET not in str(error.value)
    assert SECRET not in json.dumps(error.value.__dict__, default=str)
    assert SECRET not in json.dumps(events)
    await runtime.close_session("child_html_plan")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [429, 503, "timeout"])
async def test_transient_failures_have_at_most_three_attempts_and_safe_errors(monkeypatch, failure):
    requests, events, delays = [], [], []

    async def no_wait(delay):
        delays.append(delay)

    monkeypatch.setattr(model_module.asyncio, "sleep", no_wait)

    def serve(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("Provider echoed " + SECRET, request=request)
        return httpx.Response(failure, json={"error": {"message": "Provider echoed " + SECRET}})

    runtime = client(serve, api_max_attempts=99)
    with pytest.raises(Exception) as error:
        await runtime.run(invocation(events=events))
    assert len(requests) == 3
    assert len(delays) == 2
    assert SECRET not in str(error.value)
    assert SECRET not in json.dumps(error.value.__dict__, default=str)
    assert SECRET not in json.dumps(events)
    await runtime.close_session("child_html_plan")


@pytest.mark.asyncio
async def test_transient_failure_can_recover_on_second_attempt(monkeypatch):
    requests = []

    async def no_wait(delay):
        pass

    monkeypatch.setattr(model_module.asyncio, "sleep", no_wait)

    def serve(request):
        requests.append(request)
        return httpx.Response(503) if len(requests) == 1 else httpx.Response(200, json=chat_response())

    runtime = client(serve, api_max_attempts=3)
    result = await runtime.run(invocation())
    assert len(requests) == 2
    assert json.loads(result.final_response) == PLAN
    await runtime.close_session(result.session_id)


@pytest.mark.asyncio
async def test_runtime_cancel_propagates_to_inflight_request():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def serve(request):
        calls.append(request)
        started.set()
        await release.wait()
        return httpx.Response(200, json=chat_response())

    runtime = client(serve)
    task = asyncio.create_task(runtime.run(invocation()))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert await runtime.cancel("child_html_plan") is True
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(calls) == 1
    finally:
        task.cancel()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close_session("child_html_plan")


@pytest.mark.asyncio
async def test_tool_bearing_invocation_is_rejected_before_request():
    requests = []
    runtime = client(lambda request: requests.append(request) or httpx.Response(200, json=chat_response()))
    with pytest.raises(AgentRuntimeConfigurationError):
        await runtime.run(invocation(tools=("invented_tool",)))
    assert requests == []
