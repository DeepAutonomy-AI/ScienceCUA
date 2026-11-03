"""Bounded public runtime telemetry; model reasoning and request context stay private."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

if __package__:
    from .search_evidence import retrieved_search_sources
else:  # dsh_worker is also launched as a standalone script.
    from search_evidence import retrieved_search_sources


class RuntimeEventStream:
    def __init__(self, request: dict[str, Any], write: Callable[[dict[str, Any]], None]):
        self.request = request
        self.write = write
        self.enabled = bool(request.get("stream_events"))
        capability = request.get("capability_config") or {}
        self.secrets = tuple(str(value) for value in (
            request.get("api_key"), capability.get("token"),
            (request.get("native_web_env") or {}).get("DEEPSEEK_API_KEY"),
        ) if value)
        try:
            search = json.loads((request.get('native_web_env') or {}).get('RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG', '{}'))
            if isinstance(search, dict) and search.get('apiKey'):
                self.secrets += (str(search['apiKey']),)
        except (ValueError, TypeError):
            pass
        self.calls: dict[str, tuple[str, float]] = {}
        self.pending_tool_errors: dict[str, list[str]] = {}
        self.last_activity = 0.0
        self.step: Any = None

    def clean(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): "[REDACTED]" if str(key).lower() in {
                "api_key", "apikey", "x-api-key", "deepseek_api_key", "authorization", "capability_token", "password", "secret",
            } else self.clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(item) for item in value]
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            return re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
        return value

    def bounded(self, value: Any, limit: int = 8000) -> Any:
        value = self.clean(value)
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        if len(serialized) <= limit:
            return value
        return {"preview": serialized[:limit], "truncated": True, "total_chars": len(serialized)}

    def emit(self, event: str, **payload: Any) -> None:
        if self.enabled:
            self.write({"type": "runtime_event", "event": event, "payload": self.clean({
                "child_session_id": self.request["session_id"], **payload,
            })})

    def notification(self, notification: Any) -> None:
        if not self.enabled or notification.method != "session.event":
            return
        envelope = notification.payload
        if envelope.get("sessionId") != self.request["session_id"]:
            return
        event = envelope.get("event") or {}
        kind, data = event.get("type"), event.get("data") or {}
        common = {key: data[key] for key in ("turn", "step") if key in data}
        if isinstance(event.get("time"), (int, float)):
            common["runtime_time_ms"] = event["time"]
        if kind == "turn/start":
            self.emit("model_turn_started", status="running", **common)
        elif kind == "step/start":
            self.step = data.get("step")
            self.last_activity = 0.0
            self.emit("model_request_started", status="waiting_for_model",
                      model=self.request.get("model"), **common)
        elif kind == "assistant/chunk":
            # Only record activity. Never forward raw chunks, which can contain reasoning.
            now = time.monotonic()
            if now - self.last_activity >= 5.0:
                self.last_activity = now
                self.emit("model_response_streaming", status="receiving_response", **common)
        elif kind == "step/end":
            self.emit("model_step_completed", status="step_completed", **common)
        elif kind == "llm/retry":
            failure = data.get("failure") or {}
            self.emit("model_retry_scheduled", status="retry_wait", retry_scope="sdk",
                      retry_number=data.get("retry"), max_retries=data.get("maxRetries"),
                      delay_seconds=round(float(data.get("delayMs") or 0) / 1000, 2),
                      message=self.bounded(failure.get("message") or "Model request failed", 2000),
                      **common)
        elif kind == "llm/retry-started":
            self.emit("model_retry_started", status="waiting_for_model", retry_scope="sdk",
                      retry_number=data.get("retry"), **common)
        elif kind == "turn/end":
            reason = data.get("reason") or {}
            error = reason.get("error") or {}
            self.emit("model_turn_finished", status=reason.get("kind", "finished"),
                      message=self.bounded(error.get("message") or "", 2000), **common)
        elif kind == "tool/call":
            call_id = str(data.get("callId") or "")
            name = str(data.get("name") or "")
            is_native = bool(self.request.get("native_source_tools") or (
                self.request.get("native_web_tools") and name in {"web_search", "web_fetch"}))
            self.calls[call_id] = (name, time.monotonic())
            if not is_native:
                return  # Keep the call ID for SDK validation-error telemetry; successful business calls use gateway audit.
            arguments = data.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    pass
            self.emit("native_tool_call", tool=name, call_id=call_id, status="running",
                      arguments=self.bounded(arguments), **common)
        elif kind == "tool/result":
            message = data.get("message") or {}
            call_id = str((message.get("source") or {}).get("callId") or "")
            if call_id not in self.calls:
                return
            name, started = self.calls.pop(call_id, ("unknown", time.monotonic()))
            blocks = [item for item in message.get("content", []) if item.get("type") == "tool-result"]
            ok = not data.get("error") and not any(item.get("isError") for item in blocks)
            text = "\n".join(str(content.get("text", "")) for item in blocks
                             for content in item.get("content", []) if content.get("type") == "text")
            is_native = bool(self.request.get("native_source_tools") or (
                self.request.get("native_web_tools") and name in {"web_search", "web_fetch"}))
            if not is_native:
                if not ok:
                    self.pending_tool_errors.setdefault(name, []).append(call_id)
                    self.emit("native_tool_error", tool=name, call_id=call_id,
                              error_category=("validation" if (data.get("error") or {}).get("code") == "INVALID_ARGS"
                                              else "authorization" if "artifact_scope_denied" in text else "execution"),
                              ok=False, status="failed", output=self.bounded(text, 2000),
                              error=self.bounded(data.get("error"), 2000), **common)
                elif self.pending_tool_errors.get(name):
                    try:
                        value = json.loads(text)
                    except (ValueError, TypeError):
                        value = None
                    if isinstance(value, dict) and (value.get("ok") is False or value.get("errors")):
                        return  # A business validation failure is not a successful recovery.
                    previous = self.pending_tool_errors[name].pop(0)
                    self.emit("native_tool_recovered", tool=name, call_id=call_id,
                              previous_error_call_id=previous, status="succeeded",
                              message="A later call to this tool passed DSH validation and execution.",
                              **common)
                return
            self.emit("native_tool_result", tool=name, call_id=call_id, ok=bool(ok),
                      status="succeeded" if ok else "failed", output=self.bounded(text),
                      **({'retrieved_sources': retrieved_search_sources(text)} if name == 'web_search' and ok else {}),
                      error=self.bounded(data.get("error"), 2000),
                      duration_seconds=round(time.monotonic() - started, 3), **common)
