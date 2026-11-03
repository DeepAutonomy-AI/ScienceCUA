import asyncio
import sys
from pathlib import Path

import pytest

from app.runtime import (
    AgentInvocation,
    AgentRuntimeConfigurationError,
    AgentTurnResult,
)
from app.runtime.dsh_client import DshAgentRuntime, DshBridgeConfig
from app.capabilities import CapabilityGateway, CapabilityGatewayServer


def test_native_bridge_uses_local_python_and_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_NATIVE_HOME", str(tmp_path / "dsh-home"))
    config = DshBridgeConfig.from_environment()

    assert config.worker_argv() == (
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "app" / "runtime" / "dsh_worker.py"),
    )
    assert config.dsh_home == str(tmp_path / "dsh-home")
    assert config.runtime_path(tmp_path) == str(tmp_path.resolve())


def test_native_gateway_listens_only_on_loopback():
    server = CapabilityGatewayServer(CapabilityGateway())
    try:
        assert server.endpoint_url.startswith("http://127.0.0.1:")
        assert server._server is not None
        assert server._server.server_address[0] == "127.0.0.1"
    finally:
        server.close()


def test_http_trace_configuration_is_scoped_and_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_NATIVE_HOME", str(tmp_path / "dsh-home"))
    monkeypatch.setenv("PREPROCESS_HTTP_TRACE", "1")
    monkeypatch.setenv("PREPROCESS_HTTP_TRACE_MAX_RESPONSE_BYTES", "4096")
    monkeypatch.setenv("PREPROCESS_HTTP_TRACE_MAX_SESSION_BYTES", "8192")
    invocation = AgentInvocation(
        run_id="run_trace", session_id="child_trace", role="source_discovery",
        prompt="test", model={"model": "test", "api_key": "never-write-to-patch"},
    )
    config = DshBridgeConfig.from_environment()
    payload = config.worker_payload(invocation)
    trace = payload["http_trace_config"]
    assert Path(trace["directory"]) == tmp_path / "dsh-home/http-traces/run_trace/child_trace"
    assert trace["max_response_bytes"] == 4096
    assert trace["max_session_bytes"] == 8192
    assert "never-write-to-patch" not in str(trace)
    from app.runtime.dsh_worker import _http_trace_patch
    patch = _http_trace_patch(payload)
    assert "http-trace.mjs" in Path(patch).read_text(encoding="utf-8")
    assert "never-write-to-patch" not in Path(patch).read_text(encoding="utf-8")
    monkeypatch.setenv("PREPROCESS_HTTP_TRACE", "0")
    disabled = DshBridgeConfig.from_environment().worker_payload(invocation)
    assert disabled["http_trace_config"] is None
    assert _http_trace_patch(disabled) is None


def test_worker_payload_keeps_openai_compatible_route_and_secret_off_argv(tmp_path):
    patch = tmp_path / "production.patch.yml"
    patch.write_text("[]\n", encoding="utf-8")
    config = DshBridgeConfig(workspace=tmp_path, patch_paths=(patch,))
    invocation = AgentInvocation(
        run_id="run_1",
        session_id="root_1",
        role="root",
        prompt="hello",
        model={
            "provider": "deepseek-official",
            "model": "gpt-compatible-model",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "top-secret",
            "reasoning_effort": "high",
        },
    )

    payload = config.worker_payload(invocation)

    assert payload["base_url"] == "http://127.0.0.1:8000/v1"
    assert payload["model"] == "gpt-compatible-model"
    assert payload["api_key"] == "top-secret"
    assert "top-secret" not in " ".join(config.worker_argv())


def test_business_tools_require_attempt_scoped_capability(tmp_path):
    patch = tmp_path / "production.patch.yml"
    patch.write_text("[]\n", encoding="utf-8")
    config = DshBridgeConfig(workspace=tmp_path, patch_paths=(patch,))
    invocation = AgentInvocation(
        run_id="run_1",
        session_id="child_1",
        role="task_discovery",
        prompt="discover",
        model={"model": "test"},
        allowed_tools=("list_evidence",),
    )

    with pytest.raises(AgentRuntimeConfigurationError, match="attempt-scoped capability"):
        config.worker_payload(invocation)


def test_business_tools_install_only_the_capability_proxy(tmp_path):
    patch = tmp_path / "production.patch.yml"
    patch.write_text("[]\n", encoding="utf-8")
    proxy = tmp_path / "runtime" / "dsh" / "capability-proxy.mjs"
    proxy.parent.mkdir(parents=True)
    proxy.write_text("export function apply() {}\n", encoding="utf-8")
    config = DshBridgeConfig(workspace=tmp_path, patch_paths=(patch,))
    invocation = AgentInvocation(
        run_id="run_1",
        session_id="child_1",
        role="task_discovery",
        prompt="discover",
        model={"model": "test"},
        allowed_tools=("list_evidence",),
        metadata={
            "capability": {
                "gateway_url": "http://172.20.0.1:48123/internal/capabilities/invoke",
                "capability_token": "x" * 48,
                "tool_specs": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_evidence",
                            "description": "List evidence",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "context": {
                    "run_id": "run_1",
                    "root_session_id": "root_1",
                    "child_session_id": "child_1",
                    "attempt_id": "attempt_1",
                    "stage": "task_discovery",
                },
            }
        },
    )

    payload = config.worker_payload(invocation)

    assert payload["capability_config"]["token"] == "x" * 48
    assert Path(payload["capability_proxy_path"]) == proxy.resolve()
    assert payload["capability_config"]["toolSpecs"][0]["function"]["name"] == "list_evidence"


def test_capability_proxy_does_not_emit_unsupported_numeric_bounds():
    proxy = Path(__file__).resolve().parents[2] / "runtime" / "dsh" / "capability-proxy.mjs"
    source = proxy.read_text(encoding="utf-8")

    assert "converted.min =" not in source
    assert "converted.max =" not in source
    assert "converted.minLength =" not in source
    assert "converted.maxLength =" not in source
    assert "bounds remain enforced by the Capability" in source


def test_tool_free_root_rejects_capability_credentials(tmp_path):
    patch = tmp_path / "production.patch.yml"
    patch.write_text("[]\n", encoding="utf-8")
    config = DshBridgeConfig(workspace=tmp_path, patch_paths=(patch,))
    invocation = AgentInvocation(
        run_id="run_1",
        session_id="root_1",
        role="root",
        prompt="hello",
        model={"model": "test"},
        metadata={"capability": {"capability_token": "x" * 48}},
    )

    with pytest.raises(AgentRuntimeConfigurationError, match="tool-free"):
        config.worker_payload(invocation)


def test_openai_compatible_route_is_explicit_and_maps_xhigh(tmp_path):
    patch = tmp_path / "production.patch.yml"
    patch.write_text("[]\n", encoding="utf-8")
    config = DshBridgeConfig(workspace=tmp_path, patch_paths=(patch,))
    invocation = AgentInvocation(
        run_id="run_1",
        session_id="root_1",
        role="root",
        prompt="hello",
        model={
            "model": "gpt-compatible-model",
            "api_protocol": "chat_completions",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "top-secret",
            "reasoning_effort": "xhigh",
        },
    )

    payload = config.worker_payload(invocation)

    assert payload["provider"].startswith("resource-to-skill-openai-")
    assert payload["provider_config"]["api"] == "openai-completions"
    assert payload["provider_config"]["input"] == ["text", "image"]
    assert payload["provider_config"]["max_effort_wire_value"] == "xhigh"
    assert payload["reasoning_effort"] == "max"


def test_responses_protocol_maps_to_pi_ai_route(tmp_path):
    patch = tmp_path / "production.patch.yml"
    patch.write_text("[]\n", encoding="utf-8")
    config = DshBridgeConfig(workspace=tmp_path, patch_paths=(patch,))
    invocation = AgentInvocation(
        run_id="run_1",
        session_id="root_1",
        role="root",
        prompt="hello",
        model={
            "model": "responses-model",
            "api_protocol": "responses",
            "base_url": "http://127.0.0.1:8000/v1",
        },
    )

    payload = config.worker_payload(invocation)

    assert payload["provider_config"]["api"] == "openai-responses"
    assert "compat" not in payload["provider_config"]


@pytest.mark.asyncio
async def test_runtime_serializes_model_runs_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch = tmp_path / "production.patch.yml"
    patch.write_text("[]\n", encoding="utf-8")
    runtime = DshAgentRuntime(
        DshBridgeConfig(workspace=tmp_path, patch_paths=(patch,))
    )
    active = 0
    peak = 0

    async def fake_run(
        _runtime: DshAgentRuntime, invocation: AgentInvocation
    ) -> AgentTurnResult:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return AgentTurnResult(
            session_id=invocation.session_id,
            final_response="done",
            finish_reason="completed",
            event_count=1,
        )

    monkeypatch.setattr(DshAgentRuntime, "_run_serialized", fake_run)
    invocations = [
        AgentInvocation(
            run_id=f"run_{index}",
            session_id=f"child_{index}",
            role="root",
            prompt="test",
            model={"model": "test"},
        )
        for index in range(3)
    ]

    await asyncio.gather(*(runtime.run(item) for item in invocations))

    assert peak == 1
