from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from ..source_discovery_contracts import SOURCE_DISCOVERY_TOOLS

from .base import (
    AgentInvocation,
    AgentRuntime,
    AgentRuntimeAuthenticationError,
    AgentRuntimeBusinessCompleted,
    AgentRuntimeCancelledError,
    AgentRuntimeConfigurationError,
    AgentRuntimeExecutionError,
    AgentRuntimeProtocolError,
    AgentRuntimeTimeoutError,
    AgentRuntimeTransportError,
    AgentTurnResult,
)


# DSH writes its generated profile at startup. Keep runtime state beside this
# checkout's other data so launching from a restricted workspace stays writable.
_DEFAULT_NATIVE_DSH_HOME = str(Path(__file__).resolve().parents[2] / "data" / "dsh-home")
_CUSTOM_PROVIDER_PREFIX = "resource-to-skill-openai"
_EVENT_HEARTBEAT_SECONDS = 15.0
DEEPSEEK_SEARCH_DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic/v1"


def native_web_environment(model: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Ephemeral search credentials, independent from the conversation route."""
    search_overrides = any(os.getenv(name, '').strip() for name in (
        'PREPROCESS_SEARCH_BASE_URL', 'PREPROCESS_SEARCH_MODEL', 'PREPROCESS_SEARCH_API_KEY_ENV',
        'PREPROCESS_SEARCH_PROTOCOL'))
    provider = os.getenv("DSH_WEB_SEARCH_PROVIDER", "").strip() or (
        "deepseek-official" if os.getenv("DEEPSEEK_API_KEY", "").strip() and not search_overrides else "openai-responses")
    if provider == "openai-responses":
        model = model or {}
        base_url = os.getenv('PREPROCESS_SEARCH_BASE_URL', '').strip() or str(model.get('base_url') or '')
        search_model = os.getenv('PREPROCESS_SEARCH_MODEL', '').strip() or str(model.get('model') or '')
        key_name = os.getenv('PREPROCESS_SEARCH_API_KEY_ENV', '').strip()
        api_key = os.getenv(key_name, '').strip() if key_name else model.get('api_key')
        protocol = os.getenv('PREPROCESS_SEARCH_PROTOCOL', 'auto').strip() or 'auto'
        parsed, original = urlparse(base_url), urlparse(str(model.get('base_url') or ''))
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or
                parsed.password or parsed.query or parsed.fragment):
            raise AgentRuntimeConfigurationError('Search Base URL must be HTTP(S) without credentials, query or fragment')
        def origin(value):
            return (value.scheme, value.hostname, value.port or (443 if value.scheme == 'https' else 80))
        if not key_name and origin(parsed) != origin(original):
            raise AgentRuntimeConfigurationError('A separate search endpoint requires PREPROCESS_SEARCH_API_KEY_ENV; generation credentials are not forwarded to another origin')
        if protocol not in {'auto', 'responses', 'messages', 'chat_completions'}:
            raise AgentRuntimeConfigurationError('PREPROCESS_SEARCH_PROTOCOL must be auto, responses, messages or chat_completions')
        if not api_key or not search_model:
            raise AgentRuntimeConfigurationError("Search requires a Base URL, model and API Key; check the search credential configuration")
        plugin = Path(__file__).resolve().parents[2] / "runtime" / "dsh" / "openai-search-provider.mjs"
        return {"DSH_WEB_SEARCH_PROVIDER": provider,
                "RESOURCE_TO_SKILL_OPENAI_SEARCH_MODULE": plugin.as_uri(),
                "RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG": json.dumps({
                    "baseURL": base_url, "model": search_model, "apiKey": api_key, 'protocol': protocol,
                }, ensure_ascii=False)}
    if provider != "deepseek-official":
        raise AgentRuntimeConfigurationError("Search provider must be deepseek-official or openai-responses")
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise AgentRuntimeConfigurationError("Automatic source discovery requires a deployment-side DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_SEARCH_BASE_URL", "").strip() or DEEPSEEK_SEARCH_DEFAULT_BASE_URL
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise AgentRuntimeConfigurationError("DEEPSEEK_SEARCH_BASE_URL must be an HTTP(S) endpoint without URL credentials")
    return {"DSH_WEB_SEARCH_PROVIDER": provider, "DEEPSEEK_API_KEY": key,
            "DEEPSEEK_SEARCH_BASE_URL": base_url.rstrip("/")}


def native_web_secrets(web_env: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        config = json.loads(str(web_env.get('RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG') or '{}'))
        api_key = config.get('apiKey') if isinstance(config, dict) else None
    except (ValueError, TypeError):
        api_key = None
    return tuple(str(value) for value in (web_env.get('DEEPSEEK_API_KEY'), api_key) if value)


def _payload_secrets(payload: Mapping[str, Any]) -> tuple[str, ...]:
    web_env = payload.get("native_web_env") or {}
    capability = payload.get("capability_config") or {}
    return tuple(str(item) for item in (payload.get("api_key"), capability.get("token"),
                                       *native_web_secrets(web_env)) if item)


def _redact_text(value: str, payload: Mapping[str, Any]) -> str:
    for secret in _payload_secrets(payload):
        value = value.replace(secret, "[REDACTED]")
    return value


@dataclass(frozen=True, slots=True)
class DshBridgeConfig:
    """Launch the pinned DSH SDK with the application's local Python."""

    python_executable: str = field(default_factory=lambda: sys.executable)
    dsh_home: str = _DEFAULT_NATIVE_DSH_HOME
    workspace: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    profile: str = "sdk-minimal"
    patch_paths: tuple[Path, ...] = field(
        default_factory=lambda: (
            Path(__file__).resolve().parents[2] / "runtime" / "dsh" / "production.patch.yml",
        )
    )
    initialize_timeout_seconds: float = 30.0
    request_timeout_seconds: float | None = None
    shutdown_timeout_seconds: float = 2.0
    # Number of retries after the initial DSH turn when the provider reports a
    # transient API failure. The same Child Session is retained so completed
    # tool work is not discarded.
    # Retry once after 10 seconds, once after 20 seconds, then allow ten more
    # retries at the 30-second plateau.
    api_max_retries: int = 12
    # Base delay for the bounded linear schedule: base, 2*base, then 3*base.
    api_retry_base_delay_seconds: float = 10.0
    # Serialize model-bearing DSH turns across all runs sharing this runtime.
    # Deterministic parsing and local tool execution remain unaffected.
    max_concurrent_runs: int = 1
    http_trace_enabled: bool = True
    http_trace_max_response_bytes: int = 32 * 1024 * 1024
    http_trace_max_session_bytes: int = 256 * 1024 * 1024
    # Force parallel_tool_calls: false on tool-bearing requests; the relay drops
    # tool-call arguments on the streaming parallel path.
    disable_parallel_tool_calls: bool = True

    @classmethod
    def from_environment(cls) -> "DshBridgeConfig":
        root = Path(__file__).resolve().parents[2]
        timeout_value = os.getenv("DSH_REQUEST_TIMEOUT_SECONDS", "").strip()
        retry_value = os.getenv("DSH_API_MAX_RETRIES", "12").strip()
        retry_delay_value = os.getenv("DSH_API_RETRY_BASE_DELAY_SECONDS", "10").strip()
        concurrency_value = os.getenv("DSH_MAX_CONCURRENT_RUNS", "1").strip()
        native_home = os.getenv("DSH_NATIVE_HOME", "").strip() or _DEFAULT_NATIVE_DSH_HOME
        return cls(
            python_executable=sys.executable,
            dsh_home=str(Path(native_home).expanduser().resolve()),
            workspace=root,
            request_timeout_seconds=float(timeout_value) if timeout_value else None,
            api_max_retries=max(0, int(retry_value or "12")),
            api_retry_base_delay_seconds=max(0.0, float(retry_delay_value or "10")),
            max_concurrent_runs=max(1, int(concurrency_value or "1")),
            http_trace_enabled=os.getenv("PREPROCESS_HTTP_TRACE", "1").strip().lower()
            not in {"0", "false", "no", "off"},
            http_trace_max_response_bytes=max(1, int(os.getenv(
                "PREPROCESS_HTTP_TRACE_MAX_RESPONSE_BYTES", str(32 * 1024 * 1024)))),
            http_trace_max_session_bytes=max(1, int(os.getenv(
                "PREPROCESS_HTTP_TRACE_MAX_SESSION_BYTES", str(256 * 1024 * 1024)))),
            disable_parallel_tool_calls=os.getenv("DSH_DISABLE_PARALLEL_TOOL_CALLS", "1").strip().lower()
            not in {"0", "false", "no", "off"},
        )

    def worker_argv(self) -> tuple[str, ...]:
        worker = self.runtime_path(Path(__file__).with_name("dsh_worker.py"))
        return (self.python_executable, worker)

    def runtime_path(self, path: Path) -> str:
        return str(path.resolve())

    def worker_payload(self, invocation: AgentInvocation) -> dict[str, Any]:
        model = dict(invocation.model)
        api_key = model.pop("api_key", None)
        patches = [self.runtime_path(path) for path in self.patch_paths]
        native_source = invocation.metadata.get("native_source_tools") is True
        native_web = invocation.metadata.get("native_web_tools") is True
        if native_web and (native_source or invocation.role != "source_discovery"
                           or invocation.allowed_tools not in {SOURCE_DISCOVERY_TOOLS, ("submit_source_candidates",)}):
            raise AgentRuntimeConfigurationError("Native web tools are only available to the Source Discovery Child")
        working_directory = self.workspace
        if native_source:
            if invocation.role != "source_review" or invocation.allowed_tools:
                raise AgentRuntimeConfigurationError("Native extraction tools are only available to tool-free Source sessions")
            requested_workspace = Path(str(invocation.metadata.get("source_workspace") or "")).resolve()
            if not requested_workspace.is_dir():
                raise AgentRuntimeConfigurationError("Source extraction workspace is missing")
            working_directory = requested_workspace
        stage_patch = Path(__file__).resolve().parents[2] / "runtime" / "dsh" / (
            "adaptive-source.patch.yml" if native_source else
            "source-discovery.patch.yml" if native_web else "business-tools.patch.yml"
        )
        if not stage_patch.is_file():
            raise AgentRuntimeConfigurationError("DSH stage tool patch is missing")
        patches.append(self.runtime_path(stage_patch))
        missing = [str(path) for path in self.patch_paths if not path.is_file()]
        if missing:
            raise AgentRuntimeConfigurationError(
                f"DSH patch file does not exist: {', '.join(missing)}"
            )
        base_url = model.get("base_url")
        api_protocol = str(model.get("api_protocol") or "chat_completions")
        explicit_provider = str(model.get("provider") or "").strip()
        provider_config: dict[str, Any] | None = None
        reasoning_effort = model.get("reasoning_effort")
        if base_url and not explicit_provider:
            provider_digest = hashlib.sha256(
                f"{base_url}|{api_protocol}|{model.get('model')}".encode()
            ).hexdigest()[:16]
            provider = f"{_CUSTOM_PROVIDER_PREFIX}-{provider_digest}"
            provider_api = {
                "chat_completions": "openai-completions",
                "responses": "openai-responses",
            }.get(api_protocol, api_protocol)
            provider_config = {
                "api": provider_api,
                "base_url": str(base_url),
                "api_key_env": "RESOURCE_TO_SKILL_OPENAI_API_KEY",
                "input": ["text", "image"],
            }
            if provider_api == "openai-completions":
                provider_config["compat"] = {
                    "supportsDeveloperRole": False,
                    "supportsStore": False,
                    "maxTokensField": "max_tokens",
                }
            if reasoning_effort == "xhigh":
                reasoning_effort = "max"
                provider_config["max_effort_wire_value"] = "xhigh"
            elif reasoning_effort == "none":
                reasoning_effort = None
        else:
            provider = explicit_provider or "deepseek-official"
        capability_config: dict[str, Any] | None = None
        capability_proxy_path: str | None = None
        if invocation.allowed_tools:
            raw = invocation.metadata.get("capability")
            if not isinstance(raw, Mapping):
                raise AgentRuntimeConfigurationError(
                    "Business tools require attempt-scoped capability metadata"
                )
            gateway_url = str(raw.get("gateway_url") or "")
            parsed_gateway = urlparse(gateway_url)
            if parsed_gateway.scheme not in {"http", "https"} or not parsed_gateway.hostname:
                raise AgentRuntimeConfigurationError("Capability gateway URL is invalid")
            token = str(raw.get("capability_token") or "")
            if len(token) < 32:
                raise AgentRuntimeConfigurationError("Capability token is invalid")
            context = raw.get("context")
            specs = raw.get("tool_specs")
            if not isinstance(context, Mapping) or not isinstance(specs, list):
                raise AgentRuntimeConfigurationError("Capability context or tool schemas are missing")
            expected_context = {
                "run_id": invocation.run_id,
                "child_session_id": invocation.session_id,
                "stage": invocation.role,
            }
            if any(str(context.get(key) or "") != value for key, value in expected_context.items()):
                raise AgentRuntimeConfigurationError(
                    "Capability metadata does not match the Agent invocation"
                )
            spec_names: list[str] = []
            for spec in specs:
                function = spec.get("function") if isinstance(spec, Mapping) else None
                if not isinstance(function, Mapping) or not function.get("name"):
                    raise AgentRuntimeConfigurationError("Capability tool schema is invalid")
                spec_names.append(str(function["name"]))
            if sorted(spec_names) != sorted(invocation.allowed_tools):
                raise AgentRuntimeConfigurationError(
                    "Capability tool schemas do not match allowed_tools"
                )
            proxy = self.workspace / "runtime" / "dsh" / "capability-proxy.mjs"
            if not proxy.is_file():
                raise AgentRuntimeConfigurationError("DSH capability proxy is missing")
            capability_proxy_path = self.runtime_path(proxy)
            capability_config = {
                "gatewayUrl": gateway_url,
                "token": token,
                "toolSpecs": specs,
                "context": dict(context),
            }
            if native_web and invocation.metadata.get("discovery_budget"):
                capability_config["discoveryBudget"] = dict(invocation.metadata["discovery_budget"])
        elif invocation.metadata.get("capability") is not None:
            raise AgentRuntimeConfigurationError(
                "A tool-free invocation cannot carry capability credentials"
            )
        trace_config = None
        if self.http_trace_enabled:
            def trace_component(value: str) -> str:
                if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
                    return value
                return hashlib.sha256(value.encode()).hexdigest()[:32]

            trace_config = {
                "directory": str(Path(self.dsh_home) / "http-traces"
                                 / trace_component(invocation.run_id)
                                 / trace_component(invocation.session_id)),
                "run_id": invocation.run_id, "session_id": invocation.session_id,
                "stage": invocation.role,
                "max_response_bytes": self.http_trace_max_response_bytes,
                "max_session_bytes": self.http_trace_max_session_bytes,
            }
        return {
            "session_id": invocation.session_id,
            "prompt": invocation.prompt,
            "provider": provider,
            "provider_config": provider_config,
            "model": str(model.get("model") or "deepseek-v4-flash"),
            "reasoning_effort": reasoning_effort,
            "max_tokens": model.get("max_tokens"),
            "base_url": model.get("base_url"),
            "api_key": api_key,
            "profile": self.profile,
            "patches": patches,
            "dsh_home": self.dsh_home,
            "cwd": self.runtime_path(working_directory),
            "initialize_timeout_seconds": self.initialize_timeout_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            "api_max_retries": self.api_max_retries,
            "api_retry_base_delay_seconds": self.api_retry_base_delay_seconds,
            "capability_proxy_path": capability_proxy_path,
            "capability_config": capability_config,
            "native_source_tools": native_source,
            "native_web_tools": native_web,
            "native_web_env": native_web_environment(invocation.model) if native_web else None,
            "http_trace_config": trace_config,
            "disable_parallel_tool_calls": self.disable_parallel_tool_calls,
            "stream_events": callable(invocation.metadata.get("runtime_event_sink")),
        }


_ERROR_TYPES: Mapping[str, type[AgentRuntimeExecutionError]] = {
    "authentication": AgentRuntimeAuthenticationError,
    "cancelled": AgentRuntimeCancelledError,
    "protocol": AgentRuntimeProtocolError,
    "timeout": AgentRuntimeTimeoutError,
    "transport": AgentRuntimeTransportError,
}


@dataclass(slots=True)
class _SessionWorker:
    process: asyncio.subprocess.Process
    route_signature: tuple[Any, ...]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class DshAgentRuntime(AgentRuntime):
    """Run one official DSH SDK turn per live session and normalize its result."""

    def __init__(self, config: DshBridgeConfig | None = None) -> None:
        self.config = config or DshBridgeConfig.from_environment()
        self._workers: dict[str, _SessionWorker] = {}
        self._workers_lock = asyncio.Lock()
        self._cancelled_sessions: set[str] = set()
        self._completed_sessions: set[str] = set()
        self._run_semaphore = asyncio.Semaphore(
            max(1, int(self.config.max_concurrent_runs))
        )

    @staticmethod
    def _route_signature(payload: Mapping[str, Any]) -> tuple[Any, ...]:
        secret = str(payload.get("api_key") or "").encode()
        capability = payload.get("capability_config")
        capability_secret = ""
        capability_public = ""
        if isinstance(capability, Mapping):
            capability_secret = str(capability.get("token") or "")
            capability_public = json.dumps(
                {key: value for key, value in capability.items() if key != "token"},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        return (
            payload.get("provider"),
            payload.get("model"),
            payload.get("reasoning_effort"),
            payload.get("max_tokens"),
            payload.get("base_url"),
            hashlib.sha256(secret).hexdigest(),
            payload.get("profile"),
            tuple(payload.get("patches") or ()),
            payload.get("dsh_home"),
            payload.get("cwd"),
            payload.get("capability_proxy_path"),
            hashlib.sha256(capability_secret.encode()).hexdigest(),
            hashlib.sha256(capability_public.encode()).hexdigest(),
            hashlib.sha256(json.dumps(payload.get("native_web_env"), sort_keys=True).encode()).hexdigest(),
            json.dumps(payload.get("http_trace_config"), sort_keys=True),
        )

    async def _get_or_start_worker(
        self, session_id: str, payload: Mapping[str, Any]
    ) -> _SessionWorker:
        signature = self._route_signature(payload)
        async with self._workers_lock:
            existing = self._workers.get(session_id)
            if existing is not None and existing.process.returncode is None:
                if existing.route_signature != signature:
                    raise AgentRuntimeConfigurationError(
                        "A live DSH session cannot change provider, model, endpoint, or workspace"
                    )
                return existing
            try:
                process = await asyncio.create_subprocess_exec(
                    *self.config.worker_argv(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except (FileNotFoundError, OSError) as exc:
                raise AgentRuntimeTransportError(
                    "Unable to start the DSH worker", detail=str(exc)
                ) from exc
            worker = _SessionWorker(process=process, route_signature=signature)
            self._workers[session_id] = worker
            return worker

    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        queued_at = time.monotonic()
        sink = invocation.metadata.get("runtime_event_sink")
        def emit(event: str, payload: dict[str, Any]) -> None:
            if callable(sink):
                try:
                    sink(event, payload)
                except Exception:
                    pass  # Timing telemetry must not interrupt the Child.
        emit("model_queued", {"status": "queued", "child_session_id": invocation.session_id})
        async with self._run_semaphore:
            active_at = time.monotonic()
            emit("child_runtime_slot_acquired", {
                "queue_seconds": active_at - queued_at,
                "turn_kind": invocation.metadata.get("runtime_turn_kind", "stage_work"),
            })
            try:
                return await self._run_serialized(invocation)
            except (asyncio.CancelledError, AgentRuntimeCancelledError):
                emit("model_cancelled", {"status": "cancelled", "child_session_id": invocation.session_id})
                raise
            except AgentRuntimeBusinessCompleted:
                emit("model_business_completed", {"status": "completed", "child_session_id": invocation.session_id})
                raise
            except Exception as exc:
                emit("model_failed", {"status": "failed", "child_session_id": invocation.session_id,
                                      "error_type": type(exc).__name__})
                raise
            finally:
                emit("child_runtime_slot_released", {
                    "active_seconds": time.monotonic() - active_at,
                    "turn_kind": invocation.metadata.get("runtime_turn_kind", "stage_work"),
                })

    async def _run_serialized(self, invocation: AgentInvocation) -> AgentTurnResult:
        payload = self.config.worker_payload(invocation)
        api_key = str(payload.get("api_key") or "")
        capability = payload.get("capability_config")
        capability_token = (
            str(capability.get("token") or "") if isinstance(capability, Mapping) else ""
        )
        worker = await self._get_or_start_worker(invocation.session_id, payload)
        process = worker.process
        async with worker.lock:
            if process.stdin is None or process.stdout is None:
                raise AgentRuntimeTransportError("DSH bridge pipes are unavailable")
            encoded = (
                json.dumps({"op": "run", **payload}, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            process.stdin.write(encoded)
            await process.stdin.drain()
            try:
                stdout = await self._read_response(process, invocation, payload)
            except asyncio.TimeoutError as exc:
                await self._terminate_worker(invocation.session_id, process)
                raise AgentRuntimeTimeoutError("DSH turn exceeded its configured timeout") from exc
            except asyncio.CancelledError:
                await self._terminate_worker(invocation.session_id, process)
                async with self._workers_lock:
                    self._completed_sessions.discard(invocation.session_id)
                raise

        stderr_text = ""
        if not stdout and process.stderr is not None:
            stderr = await process.stderr.read()
            stderr_text = stderr.decode("utf-8", errors="replace")[-8_000:]
        stderr_text = _redact_text(stderr_text, payload)
        if not stdout:
            async with self._workers_lock:
                self._workers.pop(invocation.session_id, None)
                was_cancelled = invocation.session_id in self._cancelled_sessions
                self._cancelled_sessions.discard(invocation.session_id)
                was_completed = invocation.session_id in self._completed_sessions
                self._completed_sessions.discard(invocation.session_id)
            if was_completed:
                raise AgentRuntimeBusinessCompleted(
                    f"DSH session stopped after business completion: {invocation.session_id}"
                )
            if was_cancelled:
                raise AgentRuntimeCancelledError(
                    f"DSH session was cancelled: {invocation.session_id}"
                )
            raise AgentRuntimeTransportError(
                f"DSH bridge exited with code {process.returncode}", detail=stderr_text or None
            )
        try:
            response = json.loads(stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise AgentRuntimeProtocolError(
                "DSH bridge returned invalid JSON", detail=stderr_text or _redact_text(
                    stdout[-2_000:].decode("utf-8", errors="replace"), payload)
            ) from exc
        if not isinstance(response, dict):
            raise AgentRuntimeProtocolError("DSH bridge response must be a JSON object")
        async with self._workers_lock:
            self._completed_sessions.discard(invocation.session_id)
        if not response.get("ok"):
            error = response.get("error") if isinstance(response.get("error"), dict) else {}
            category = str(error.get("category") or "runtime_error")
            error_type = _ERROR_TYPES.get(category, AgentRuntimeExecutionError)
            message = str(error.get("message") or "DSH runtime failed")
            worker_detail = str(error.get("detail") or "")
            message = _redact_text(message, payload)
            worker_detail = _redact_text(worker_detail, payload)
            raise error_type(
                message,
                detail=stderr_text or worker_detail or None,
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise AgentRuntimeProtocolError("DSH bridge response is missing result")
        return AgentTurnResult(
            session_id=str(result.get("session_id") or invocation.session_id),
            final_response=_redact_text(str(result.get("final_response") or ""), payload),
            finish_reason=(
                str(result["finish_reason"]) if result.get("finish_reason") is not None else None
            ),
            event_count=int(result.get("event_count") or 0),
            usage=result.get("usage") if isinstance(result.get("usage"), dict) else {},
        )

    async def _read_response(
        self, process: asyncio.subprocess.Process, invocation: AgentInvocation, payload: dict[str, Any]
    ) -> bytes:
        """Drain event frames while retaining one absolute deadline for the final response."""
        sink = invocation.metadata.get("runtime_event_sink")
        started = time.monotonic()
        deadline = (started + self.config.request_timeout_seconds + 5.0
                    if self.config.request_timeout_seconds is not None else None)
        status = "starting"
        last_event = started
        pending = None

        def emit(event: str, data: dict[str, Any]) -> None:
            if callable(sink):
                # Defense in depth: never allow known invocation credentials into audit logs.
                encoded = json.dumps(data, ensure_ascii=False)
                for secret in _payload_secrets(payload):
                    if secret:
                        encoded = encoded.replace(json.dumps(str(secret), ensure_ascii=False)[1:-1], "[REDACTED]")
                try:
                    sink(event, json.loads(encoded))
                except Exception:
                    pass  # Telemetry failures must not change the business result.

        try:
            while True:
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    raise asyncio.TimeoutError
                wait = _EVENT_HEARTBEAT_SECONDS
                if deadline is not None:
                    wait = min(wait, deadline - now)
                if pending is None:
                    pending = asyncio.create_task(process.stdout.readline())
                try:
                    line = await asyncio.wait_for(asyncio.shield(pending), timeout=wait)
                except asyncio.TimeoutError:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise
                    emit("model_waiting", {"child_session_id": invocation.session_id,
                         "status": status, "elapsed_seconds": round(time.monotonic() - started, 1),
                         "seconds_since_event": round(time.monotonic() - last_event, 1)})
                    continue
                pending = None
                if not line:
                    return line
                try:
                    frame = json.loads(line.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    return line  # Existing final-response validation reports the protocol error.
                if not isinstance(frame, dict) or frame.get("type") != "runtime_event":
                    return line
                data = frame.get("payload")
                event = frame.get("event")
                if (not isinstance(data, dict) or not isinstance(event, str)
                    or not event.startswith(("model_", "native_tool_"))
                    or data.get("child_session_id") != invocation.session_id):
                    raise AgentRuntimeProtocolError("Invalid DSH runtime event frame")
                status = str(data.get("status") or status)
                if event == "native_tool_call":
                    status = "executing_tool"
                elif event == "native_tool_result":
                    status = "tool_finished"
                last_event = time.monotonic()
                emit(event, data)
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def cancel(self, session_id: str) -> bool:
        async with self._workers_lock:
            worker = self._workers.get(session_id)
            process = worker.process if worker is not None else None
            if process is None or process.returncode is not None:
                return False
            self._cancelled_sessions.add(session_id)
        await self._terminate_worker(session_id, process)
        return True

    async def stop_completed_session(self, session_id: str) -> bool:
        async with self._workers_lock:
            worker = self._workers.get(session_id)
            process = worker.process if worker is not None else None
            if process is None or process.returncode is not None:
                return False
            self._completed_sessions.add(session_id)
        await self._terminate_worker(session_id, process)
        return True

    async def close_session(self, session_id: str) -> bool:
        """Flush and close one persistent SDK runtime after its logical session ends."""

        async with self._workers_lock:
            worker = self._workers.get(session_id)
        if worker is None or worker.process.returncode is not None:
            async with self._workers_lock:
                self._completed_sessions.discard(session_id)
            return False
        async with worker.lock:
            process = worker.process
            if process.stdin is None or process.stdout is None:
                await self._terminate_worker(session_id, process)
                return True
            process.stdin.write(b'{"op":"shutdown"}\n')
            await process.stdin.drain()
            try:
                await asyncio.wait_for(
                    process.stdout.readline(), timeout=self.config.shutdown_timeout_seconds
                )
                await asyncio.wait_for(
                    process.wait(), timeout=self.config.shutdown_timeout_seconds
                )
            except asyncio.TimeoutError:
                await self._terminate_worker(session_id, process)
        async with self._workers_lock:
            self._workers.pop(session_id, None)
            self._completed_sessions.discard(session_id)
        return True

    async def _terminate_worker(
        self, session_id: str, process: asyncio.subprocess.Process
    ) -> None:
        if process.returncode is None:
            if sys.platform == "win32" and getattr(process, "pid", None):
                # Windows venv Python launches a second Python process, which in
                # turn owns the SDK runtime. Stop the owned tree so inherited
                # pipes close and the model slot is actually released.
                killer = await asyncio.create_subprocess_exec(
                    "taskkill", "/PID", str(process.pid), "/T", "/F",
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                await killer.wait()
            elif process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=self.config.shutdown_timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        async with self._workers_lock:
            current = self._workers.get(session_id)
            if current is not None and current.process is process:
                self._workers.pop(session_id, None)
