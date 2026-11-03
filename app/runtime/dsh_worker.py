from __future__ import annotations

import json
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

if __package__:
    from .dsh_events import RuntimeEventStream
else:
    from dsh_events import RuntimeEventStream


_AUTH_STATUS_PATTERN = re.compile(
    r"(?:"
    r"\bhttp(?:\s+status)?\s*[:=]?\s*(?:401|403)\b|"
    r"\bstatus(?:\s+code)?[\"']?\s*[:=]?\s*(?:401|403)\b|"
    r"\bcode[\"']?\s*[:=]\s*(?:401|403)\b|"
    r"\b(?:401|403)\s+status\s+code\b"
    r")",
    flags=re.IGNORECASE,
)

_UNSUPPORTED_MODEL_PATTERN = re.compile(
    r"\b(?:unsupported[_ -]+model|model[_ -]+not[_ -]+found|invalid[_ -]+model)\b|"
    r"\bmodel\b.{0,160}?\b(?:not supported|unsupported|does not exist|not found)\b",
    flags=re.IGNORECASE,
)


def _is_unsupported_model_failure(message: str) -> bool:
    return bool(_UNSUPPORTED_MODEL_PATTERN.search(" ".join(str(message or "").split())))


def _is_authentication_failure(message: str) -> bool:
    normalized = " ".join(str(message or "").casefold().split())
    return bool(_AUTH_STATUS_PATTERN.search(normalized)) or any(
        marker in normalized
        for marker in (
            "api key",
            "unauthorized",
            "authentication failed",
            "invalid authentication",
            "invalid_api_key",
        )
    )


def _error_category(exc: BaseException) -> str:
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    if "no adapter registered" in message or "plugin tree failed to load" in message:
        return "protocol"
    if "timeout" in name or "timed out" in message:
        return "timeout"
    if "protocol" in name or "jsonrpc" in name:
        return "protocol"
    if _is_authentication_failure(message):
        return "authentication"
    if "transport" in name or "closed" in message or "connection" in message:
        return "transport"
    return "runtime_error"


def _usage_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        usage = data.get("usage")
        if isinstance(usage, dict):
            return usage
    return {}


def _turn_end_detail(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") != "turn/end":
            continue
        data = event.get("data")
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))[:2_000]
    return ""


_NON_RETRYABLE_PROVIDER_MARKERS = (
    "context_window_exceeded",
    "context overflow",
    "invalid prompt",
    "usage policy",
    "authentication",
    "unauthorized",
    "api key",
    "insufficient_quota",
    "insufficient quota",
    "quota exceeded",
    "credit balance",
    "unsupported json schema",
    "tool budget",
    "cancelled",
    "canceled",
)

_RETRYABLE_PROVIDER_MARKERS = (
    "pi_ai_error",
    '"code":"server"',
    "service temporarily unavailable",
    "internal server error",
    "rate limit",
    "too many requests",
    "timed out",
    "timeout",
    "connection reset",
    "connection closed",
    "connection error",
    "network error",
    "http 429",
    "status code: 429",
    "status code 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
    "status code 500",
    "status code 502",
    "status code 503",
    "status code 504",
)

_RETRY_CONTINUATION_PROMPT = (
    "Continue the same task from the existing session state. The previous model turn ended "
    "because of a transient provider/API failure. Do not repeat completed tool calls. "
    "Continue from the last successful step and finish the requested stage."
)


def _is_retryable_provider_failure(message: str) -> bool:
    normalized = " ".join(str(message or "").casefold().split())
    if _is_authentication_failure(normalized) or _is_unsupported_model_failure(normalized):
        return False
    if any(marker in normalized for marker in _NON_RETRYABLE_PROVIDER_MARKERS):
        return False
    return any(marker in normalized for marker in _RETRYABLE_PROVIDER_MARKERS)


def _retry_delay(base_seconds: float, retry_number: int) -> float:
    """Return base, 2*base, then a capped 3*base provider cooldown."""

    return max(0.0, base_seconds) * min(3, max(1, int(retry_number)))


def _yaml_scalar(value: str) -> str:
    # JSON strings are valid YAML scalars and avoid executable YAML tags.
    return json.dumps(value, ensure_ascii=False)


def _provider_patch(request: dict[str, Any]) -> str | None:
    config = request.get("provider_config")
    if not isinstance(config, dict):
        return None
    provider = str(request["provider"])
    model = str(request["model"])
    api = str(config["api"])
    base_url = str(config["base_url"])
    api_key_env = str(config["api_key_env"])
    compat = config.get("compat") if isinstance(config.get("compat"), dict) else {}
    lines = [
        "- id: llm-pi-ai",
        "  config:",
        "    providers:",
        f"      {_yaml_scalar(provider)}:",
        f"        apiKeyEnv: {_yaml_scalar(api_key_env)}",
        f"        api: {_yaml_scalar(api)}",
        f"        baseURL: {_yaml_scalar(base_url)}",
        "        defaultInput: [text, image]",
    ]
    if compat:
        lines.extend(
            [
                "        compat:",
                "          supportsDeveloperRole: "
                + str(bool(compat.get("supportsDeveloperRole"))).lower(),
                "          supportsStore: " + str(bool(compat.get("supportsStore"))).lower(),
                f"          maxTokensField: {_yaml_scalar(str(compat.get('maxTokensField') or 'max_tokens'))}",
            ]
        )
    lines.extend(
        [
            "        models:",
            f"          - id: {_yaml_scalar(model)}",
            "            input: [text, image]",
        ]
    )
    effort = request.get("reasoning_effort")
    if isinstance(effort, str) and effort:
        max_wire = str(config.get("max_effort_wire_value") or "max")
        lines.extend(
            [
                "            reasoningEfforts:",
                "              off:",
                "              minimal: minimal",
                "              low: low",
                "              medium: medium",
                "              high: high",
                f"              max: {_yaml_scalar(max_wire)}",
            ]
        )
    content = "\n".join(lines) + "\n"
    patch_digest = hashlib.sha256(content.encode()).hexdigest()[:20]
    patch_dir = Path(request["dsh_home"]) / "bridge-patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / f"{provider}-{patch_digest}.yml"
    if not patch_path.exists():
        temporary = patch_path.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(patch_path)
    return str(patch_path)


def _capability_patch(request: dict[str, Any]) -> str | None:
    proxy_path = request.get("capability_proxy_path")
    config = request.get("capability_config")
    if proxy_path is None and config is None:
        return None
    if not isinstance(proxy_path, str) or not Path(proxy_path).is_absolute():
        raise ValueError("Capability proxy must be an absolute path")
    if not isinstance(config, dict):
        raise ValueError("Capability configuration is missing")
    path = Path(proxy_path)
    if not path.is_file():
        raise ValueError("Capability proxy does not exist")
    # Node's Windows ESM loader requires a file URL for an absolute drive path.
    module_name = path.as_uri() if os.name == "nt" else str(path)
    content = "\n".join(
        [
            "- insert:",
            "    - id: resource-to-skill-capability-proxy",
            f"      name: {_yaml_scalar(module_name)}",
        ]
    ) + "\n"
    patch_digest = hashlib.sha256(content.encode()).hexdigest()[:20]
    patch_dir = Path(request["dsh_home"]) / "bridge-patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / f"capability-{patch_digest}.yml"
    if not patch_path.exists():
        temporary = patch_path.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(patch_path)
    return str(patch_path)


def _write_frame(frame: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _native_search_patch(request: dict[str, Any]) -> str | None:
    env = request.get("native_web_env") or {}
    module_name = env.get("RESOURCE_TO_SKILL_OPENAI_SEARCH_MODULE")
    if env.get("DSH_WEB_SEARCH_PROVIDER") != "openai-responses" or not module_name:
        return None
    content = "- insert:\n    - id: discovery-web-search-openai\n      name: " + _yaml_scalar(module_name) + "\n"
    patch_dir = Path(request["dsh_home"]) / "bridge-patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    path = patch_dir / ("search-" + hashlib.sha256(content.encode()).hexdigest()[:20] + ".yml")
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return str(path)


def _http_trace_patch(request: dict[str, Any]) -> str | None:
    if not isinstance(request.get("http_trace_config"), dict):
        return None
    module = Path(__file__).resolve().parents[2] / "runtime" / "dsh" / "http-trace.mjs"
    content = "- insert:\n    - id: model-http-trace\n      name: " + _yaml_scalar(module.as_uri()) + "\n"
    patch_dir = Path(request["dsh_home"]) / "bridge-patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    path = patch_dir / ("http-trace-" + hashlib.sha256(content.encode()).hexdigest()[:20] + ".yml")
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return str(path)


def _tool_diagnostics_patch(request: dict[str, Any]) -> str:
    module = Path(__file__).resolve().parents[2] / "runtime" / "dsh" / "tool-diagnostics.mjs"
    content = "- insert:\n    - id: resource-to-skill-tool-diagnostics\n      name: " + _yaml_scalar(module.as_uri()) + "\n"
    patch_dir = Path(request["dsh_home"]) / "bridge-patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    path = patch_dir / ("tool-diagnostics-" + hashlib.sha256(content.encode()).hexdigest()[:20] + ".yml")
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return str(path)


def _parallel_calls_patch(request: dict[str, Any]) -> str | None:
    if not request.get("disable_parallel_tool_calls"):
        return None
    module = Path(__file__).resolve().parents[2] / "runtime" / "dsh" / "parallel-calls-guard.mjs"
    content = "- insert:\n    - id: model-parallel-calls-guard\n      name: " + _yaml_scalar(module.as_uri()) + "\n"
    patch_dir = Path(request["dsh_home"]) / "bridge-patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    path = patch_dir / ("parallel-calls-" + hashlib.sha256(content.encode()).hexdigest()[:20] + ".yml")
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return str(path)


def _run(request: dict[str, Any], harness: Any | None) -> tuple[dict[str, Any], Any]:
    events = RuntimeEventStream(request, _write_frame)
    if harness is None:
        from deepseek_harness import DeepSeekHarness

        provider_patch = _provider_patch(request)
        patches = [str(item) for item in request.get("patches") or ()]
        if provider_patch is not None:
            patches.append(provider_patch)
        capability_patch = _capability_patch(request)
        if capability_patch is not None:
            patches.append(capability_patch)
        search_patch = _native_search_patch(request)
        if search_patch is not None:
            patches.append(search_patch)
        patches.append(_tool_diagnostics_patch(request))
        trace_patch = _http_trace_patch(request)
        if trace_patch is not None:
            patches.append(trace_patch)
        parallel_patch = _parallel_calls_patch(request)
        if parallel_patch is not None:
            # Applied after the trace patch so the guard wraps it: the trace
            # records the final outgoing body with parallel_tool_calls disabled.
            patches.append(parallel_patch)
        env: dict[str, str] = {}
        if request.get("native_web_tools"):
            web_env = request.get("native_web_env")
            if not isinstance(web_env, dict) or not (
                web_env.get("DEEPSEEK_API_KEY") or web_env.get("RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG")
            ):
                raise ValueError("Source discovery is missing the deployment search credential")
            env.update({str(key): str(value) for key, value in web_env.items()
                        if key in {"DSH_WEB_SEARCH_PROVIDER", "DEEPSEEK_API_KEY", "DEEPSEEK_SEARCH_BASE_URL",
                                   "RESOURCE_TO_SKILL_OPENAI_SEARCH_MODULE", "RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG"}})
        provider_config = request.get("provider_config")
        if isinstance(provider_config, dict) and request.get("api_key"):
            env[str(provider_config["api_key_env"])] = str(request["api_key"])
        capability_config = request.get("capability_config")
        if isinstance(capability_config, dict):
            env["RESOURCE_TO_SKILL_CAPABILITY_CONFIG"] = json.dumps(
                capability_config, ensure_ascii=False, separators=(",", ":")
            )
            env["RESOURCE_TO_SKILL_DSH_TOOLS_MODULE"] = str(
                Path(request["dsh_home"])
                / "profiles"
                / "node_modules"
                / "@deepseek-ai"
                / "dsh-tools"
                / "entry-0.js"
            )
        if trace_patch is not None:
            trace_config = dict(request["http_trace_config"])
            trace_config["secrets"] = [value for value in (
                request.get("api_key"), (capability_config or {}).get("token"),
                (request.get("native_web_env") or {}).get("DEEPSEEK_API_KEY"),
            ) if isinstance(value, str) and value]
            try:
                search = json.loads((request.get('native_web_env') or {}).get('RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG', '{}'))
                if isinstance(search, dict) and search.get('apiKey'):
                    trace_config['secrets'].append(str(search['apiKey']))
            except (ValueError, TypeError):
                pass
            env["RESOURCE_TO_SKILL_HTTP_TRACE_CONFIG"] = json.dumps(trace_config)
        harness = DeepSeekHarness(
            provider=request["provider"],
            model=request["model"],
            reasoning_effort=request.get("reasoning_effort"),
            max_tokens=request.get("max_tokens"),
            cwd=request["cwd"],
            dsh_home=request["dsh_home"],
            profile=request.get("profile", "sdk-minimal"),
            patches=tuple(patches),
            base_url=request.get("base_url") if provider_patch is None else None,
            api_key=request.get("api_key") if provider_patch is None else None,
            env=env,
            initialize_timeout_seconds=float(request.get("initialize_timeout_seconds", 30)),
            request_timeout_seconds=request.get("request_timeout_seconds"),
            shutdown_timeout_seconds=float(request.get("shutdown_timeout_seconds", 2)),
        )
        events.emit("model_runtime_starting", status="initializing")
        harness.start()
        events.emit("model_runtime_ready", status="ready")
        if trace_patch is not None:
            events.emit("model_http_trace_ready", status="enabled",
                        directory=request["http_trace_config"]["directory"],
                        max_response_bytes=request["http_trace_config"]["max_response_bytes"],
                        max_session_bytes=request["http_trace_config"]["max_session_bytes"])
    max_retries = max(0, min(12, int(request.get("api_max_retries", 12))))
    retry_base_delay = max(
        0.0, float(request.get("api_retry_base_delay_seconds", 10.0))
    )
    retry_count = 0
    prompt = request["prompt"]

    def on_notification(notification: Any) -> None:
        # The SDK otherwise retries SERVER errors five times before returning a
        # result to this outer retry loop. Inspect those notices even when public
        # telemetry is disabled, and stop deterministic model failures immediately.
        if notification.method == "session.event" and notification.payload.get("sessionId") == request["session_id"]:
            event = notification.payload.get("event") or {}
            if event.get("type") == "llm/retry":
                failure = (event.get("data") or {}).get("failure") or {}
                message = str(failure.get("message") or "")
                if _is_unsupported_model_failure(message):
                    events.emit("model_retry_stopped", status="failed", retry_scope="sdk",
                                message=events.bounded(message, 2000))
                    try:
                        harness.close()
                    finally:
                        raise RuntimeError(message)
        events.notification(notification)

    while True:
        try:
            result = harness.run(prompt, session_id=request["session_id"], on_notification=on_notification)
        except Exception as exc:
            failure = str(exc)
            if retry_count >= max_retries or not _is_retryable_provider_failure(failure):
                raise
        else:
            if result.finish_reason == "completed":
                break
            detail = _turn_end_detail(result.events)
            failure = (
                f"DSH turn ended with {result.finish_reason or 'no-finish-reason'}"
                + (f": {detail}" if detail else "")
            )
            if retry_count >= max_retries or not _is_retryable_provider_failure(failure):
                raise RuntimeError(failure)
        retry_count += 1
        delay = _retry_delay(retry_base_delay, retry_count)
        events.emit("model_retry_scheduled", status="retry_wait", retry_number=retry_count,
                    max_retries=max_retries, retry_scope="worker",
                    delay_seconds=delay, message=events.bounded(failure, 2000))
        time.sleep(delay)
        prompt = _RETRY_CONTINUATION_PROMPT
    usage = _usage_from_events(result.events)
    if retry_count:
        usage = dict(usage)
        usage["runtime_retry_count"] = retry_count
        usage["runtime_attempt_count"] = retry_count + 1
    return (
        {
            "ok": True,
            "result": {
                "session_id": result.session_id,
                "final_response": result.final_response,
                "finish_reason": result.finish_reason,
                "event_count": len(result.events),
                "usage": usage,
                "runtime_retry_count": retry_count,
            },
        },
        harness,
    )


def main() -> int:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    harness: Any | None = None
    try:
        for line in sys.stdin:
            request: dict[str, Any] = {}
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Bridge request must be a JSON object")
                if request.get("op") == "shutdown":
                    response = {"ok": True, "result": {"shutdown": True}}
                    json.dump(response, sys.stdout, ensure_ascii=False, separators=(",", ":"))
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    break
                if request.get("op") != "run":
                    raise ValueError("Unknown bridge operation")
                response, harness = _run(request, harness)
            except BaseException as exc:
                message = str(exc)[:2_000]
                api_key = str(request.get("api_key") or "")
                if api_key:
                    message = message.replace(api_key, "[REDACTED]")
                web_env = request.get("native_web_env") or {}
                search_key = str(web_env.get("DEEPSEEK_API_KEY") or "")
                if search_key:
                    message = message.replace(search_key, "[REDACTED]")
                capability = request.get("capability_config") if isinstance(request, dict) else None
                if isinstance(capability, dict):
                    token = str(capability.get("token") or "")
                    if token:
                        message = message.replace(token, "[REDACTED]")
                response = {
                    "ok": False,
                    "error": {
                        "category": _error_category(exc),
                        "message": message,
                        "exception_type": type(exc).__name__,
                    },
                }
            json.dump(response, sys.stdout, ensure_ascii=False, separators=(",", ":"))
            sys.stdout.write("\n")
            sys.stdout.flush()
    finally:
        if harness is not None:
            harness.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
