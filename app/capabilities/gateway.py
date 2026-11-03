from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..storage import RunStore, sanitize
from .authorization import (
    CapabilityAuthorizationError,
    CapabilityAuthority,
    CapabilityRecord,
)


class BusinessToolExecutor(Protocol):
    allowed_tools: set[str] | frozenset[str]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def tool_specs(self) -> list[dict[str, Any]]: ...


class CapabilityGatewayError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int = 403) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class CapabilityInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    root_session_id: str
    child_session_id: str
    attempt_id: str
    stage: str
    tool_name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9._:-]+$")


@dataclass(frozen=True, slots=True)
class RegisteredCapability:
    record: CapabilityRecord
    token: str = field(repr=False)
    tool_specs: tuple[dict[str, Any], ...]

    def runtime_metadata(self, gateway_url: str) -> dict[str, Any]:
        """Build ephemeral runtime metadata; callers must never persist this value."""

        return {
            "gateway_url": gateway_url,
            "capability_token": self.token,
            "tool_specs": list(self.tool_specs),
            "context": {
                "run_id": self.record.run_id,
                "root_session_id": self.record.root_session_id,
                "child_session_id": self.record.child_session_id,
                "attempt_id": self.record.attempt_id,
                "stage": self.record.stage,
            },
        }


@dataclass(slots=True)
class _Binding:
    store: RunStore
    executor: BusinessToolExecutor
    allowed_artifact_ids: set[str]
    max_tool_calls: int | None = None
    tool_call_count: int = 0
    tool_call_limits: dict[str, int] = field(default_factory=dict)
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    budget_exhaustion_as_skip: bool = False
    cached_calls: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    on_tool_result: Callable[[CapabilityInvocation, dict[str, Any]], None] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


_SCOPED_RESOURCES = ("source", "task", "reference", "trajectory", "evidence")
_PATH_KEYS = {"path", "file_path", "filesystem_path", "cwd", "working_directory"}
_CREDENTIAL_KEYS = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "access_token",
    "refresh_token",
    "capability_token",
    "cookie",
    "set-cookie",
}


def _tool_name(spec: Mapping[str, Any]) -> str:
    function = spec.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "")
    return str(spec.get("name") or "")


def _walk_fields(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_fields(item)


def _id_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if isinstance(item, str))
    return ()


class CapabilityGateway:
    """Only execution path from a DSH Child tool call to Python business code."""

    def __init__(
        self,
        authority: CapabilityAuthority | None = None,
        *,
        max_argument_bytes: int = 256 * 1024,
    ) -> None:
        self.authority = authority or CapabilityAuthority()
        if max_argument_bytes < 1_024:
            raise ValueError("Gateway argument limit is too small")
        self.max_argument_bytes = max_argument_bytes
        self._bindings: dict[str, _Binding] = {}
        self._lease_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = threading.RLock()

    def register_attempt(
        self,
        store: RunStore,
        attempt_id: str,
        executor: BusinessToolExecutor,
        *,
        allowed_tools: tuple[str, ...] | None = None,
        max_tool_calls: int | None = None,
        tool_call_limits: Mapping[str, int] | None = None,
        budget_exhaustion_as_skip: bool = False,
        ttl_seconds: float | None = None,
        keep_alive: bool = False,
        on_tool_result: Callable[[CapabilityInvocation, dict[str, Any]], None] | None = None,
    ) -> RegisteredCapability:
        # A lease must be owned by the same async Child lifecycle that revokes it.
        loop = asyncio.get_running_loop() if keep_alive else None
        if max_tool_calls is not None and max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        requested = tuple(
            sorted(allowed_tools if allowed_tools is not None else executor.allowed_tools)
        )
        if not set(requested).issubset(set(executor.allowed_tools)):
            raise CapabilityGatewayError(
                "executor_tool_mismatch", "Gateway cannot grant tools missing from the executor"
            )
        normalized_tool_limits: dict[str, int] = {}
        for raw_name, raw_limit in (tool_call_limits or {}).items():
            name = str(raw_name).strip()
            limit = int(raw_limit)
            if name not in requested:
                raise ValueError(f"Tool call limit references an ungranted tool: {name}")
            if limit < 1:
                raise ValueError("Per-tool call limits must be positive")
            normalized_tool_limits[name] = limit
        issued = self.authority.issue(
            store, attempt_id, allowed_tools=requested, ttl_seconds=ttl_seconds
        )
        catalog: dict[str, dict[str, Any]] = {}
        for spec in executor.tool_specs():
            name = _tool_name(spec)
            if name:
                catalog[name] = spec
        missing = sorted(set(requested) - set(catalog))
        if missing:
            self.authority.revoke(issued.record.grant_id, reason="missing_tool_schema")
            raise CapabilityGatewayError(
                "missing_tool_schema", f"Executor has no schema for: {', '.join(missing)}"
            )
        binding = _Binding(
            store=store,
            executor=executor,
            allowed_artifact_ids=set(issued.record.resource_scope.get("artifact_ids", ())),
            max_tool_calls=max_tool_calls,
            tool_call_limits=normalized_tool_limits,
            budget_exhaustion_as_skip=bool(budget_exhaustion_as_skip),
            on_tool_result=on_tool_result,
        )
        with self._lock:
            self._bindings[issued.record.grant_id] = binding
        store.emit(
            "capability_tool_budget_configured",
            {
                "grant_id": issued.record.grant_id,
                "attempt_id": issued.record.attempt_id,
                "max_tool_calls": max_tool_calls,
                "tool_call_limits": normalized_tool_limits,
                "budget_exhaustion_as_skip": bool(budget_exhaustion_as_skip),
                "completion_tools_exempt_after_exhaustion": True,
            },
            agent=issued.record.stage,
        )
        registered = RegisteredCapability(
            record=issued.record,
            token=issued.token,
            tool_specs=tuple(catalog[name] for name in sorted(requested)),
        )
        if loop is not None:
            ttl = self.authority.default_ttl_seconds if ttl_seconds is None else ttl_seconds
            self._lease_tasks[issued.record.grant_id] = loop.create_task(
                self._keep_alive(registered, store, ttl),
                name=f"capability-lease-{issued.record.grant_id}",
            )
        return registered

    async def _keep_alive(
        self, registered: RegisteredCapability, store: RunStore, ttl: float
    ) -> None:
        """Keep queued/running Children authorized across turns and retry backoff."""
        grant_id = registered.record.grant_id
        try:
            while True:
                await asyncio.sleep(min(60.0, ttl / 3.0))
                self.authority.renew(grant_id, ttl_seconds=ttl)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # No resurrection or silent task failure. Existing expiry checks remain.
            store.emit(
                "capability_lease_ended",
                {"grant_id": grant_id, "attempt_id": registered.record.attempt_id,
                 "reason": getattr(exc, "code", type(exc).__name__)},
                agent=registered.record.stage,
                level="warning",
            )
        finally:
            with self._lock:
                if self._lease_tasks.get(grant_id) is asyncio.current_task():
                    self._lease_tasks.pop(grant_id, None)

    def invoke(self, token: str, invocation: CapabilityInvocation) -> dict[str, Any]:
        argument_size = len(
            json.dumps(
                invocation.arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        if argument_size > self.max_argument_bytes:
            raise CapabilityGatewayError(
                "arguments_too_large",
                f"Tool arguments exceed {self.max_argument_bytes} bytes",
                status_code=413,
            )
        try:
            record = self.authority.authorize(
                token,
                run_id=invocation.run_id,
                root_session_id=invocation.root_session_id,
                child_session_id=invocation.child_session_id,
                attempt_id=invocation.attempt_id,
                stage=invocation.stage,
                tool_name=invocation.tool_name,
            )
        except CapabilityAuthorizationError as exc:
            raise CapabilityGatewayError(exc.code, str(exc), status_code=403) from exc
        with self._lock:
            binding = self._bindings.get(record.grant_id)
        if binding is None:
            raise CapabilityGatewayError(
                "unbound_grant", "Capability has no live executor", status_code=410
            )
        if binding.store.run_id != invocation.run_id:
            raise CapabilityGatewayError("run_mismatch", "Executor belongs to another Run")
        if invocation.tool_name not in binding.executor.allowed_tools:
            raise CapabilityGatewayError(
                "executor_tool_denied", "Executor denied this tool", status_code=403
            )

        self._validate_arguments(record, binding, invocation.arguments)
        request_digest = hashlib.sha256(
            json.dumps(
                {"tool": invocation.tool_name, "arguments": invocation.arguments},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        with binding.lock:
            cached = binding.cached_calls.get(invocation.call_id)
            if cached:
                if cached[0] != request_digest:
                    raise CapabilityGatewayError(
                        "call_id_conflict",
                        "Tool call ID was reused with different arguments",
                        status_code=409,
                    )
                return cached[1]

            binding.store.emit(
                "capability_tool_call",
                {
                    "grant_id": record.grant_id,
                    "attempt_id": record.attempt_id,
                    "child_session_id": record.child_session_id,
                    "stage": record.stage,
                    "tool": invocation.tool_name,
                    "call_id": invocation.call_id,
                    "arguments": invocation.arguments,
                },
                agent=record.stage,
            )
            is_completion_tool = invocation.tool_name.startswith("submit_") or invocation.tool_name in {
                "report_discovery_gap",
                "inspect_pair_coverage",
                "supersede_pair",
                "report_trajectory_gap",
                "inspect_reference_coverage",
                "review_reference_coverage",
                "inspect_skill_coverage",
                "review_skill_coverage",
            }
            tool_limit = binding.tool_call_limits.get(invocation.tool_name)
            tool_count = binding.tool_call_counts.get(invocation.tool_name, 0)
            total_exhausted = (
                binding.max_tool_calls is not None
                and binding.tool_call_count >= binding.max_tool_calls
            )
            tool_exhausted = tool_limit is not None and tool_count >= tool_limit
            if not is_completion_tool and (total_exhausted or tool_exhausted):
                configured_limit = tool_limit if tool_exhausted else binding.max_tool_calls
                budget_kind = "tool" if tool_exhausted else "inspection"
                bounded = {
                    "ok": bool(binding.budget_exhaustion_as_skip),
                    "status": "skipped" if binding.budget_exhaustion_as_skip else "rejected",
                    "skipped": bool(binding.budget_exhaustion_as_skip),
                    "error": None if binding.budget_exhaustion_as_skip else "tool_call_budget_exhausted",
                    "reason": "tool_call_budget_exhausted",
                    "message": (
                        f"The {budget_kind} budget of {configured_limit} call(s) is exhausted. "
                        "Do not retry this inspection tool. Use the evidence already observed, "
                        "submit any supported candidates, and call submit_stage_result now."
                    ),
                    "mandatory_next_action": "submit_candidates_then_submit_stage_result",
                    "completion_tools_remain_available": True,
                }
                binding.cached_calls[invocation.call_id] = (request_digest, bounded)
                binding.store.emit(
                    "capability_tool_budget_exhausted",
                    {
                        "grant_id": record.grant_id,
                        "attempt_id": record.attempt_id,
                        "tool": invocation.tool_name,
                        "max_tool_calls": binding.max_tool_calls,
                        "tool_call_limit": tool_limit,
                        "budget_kind": budget_kind,
                        "returned_as_skipped": binding.budget_exhaustion_as_skip,
                    },
                    agent=record.stage,
                    level="warning",
                )
                binding.store.emit(
                    "capability_tool_result",
                    {
                        "grant_id": record.grant_id,
                        "attempt_id": record.attempt_id,
                        "tool": invocation.tool_name,
                        "call_id": invocation.call_id,
                        "ok": False,
                        "truncated": False,
                        "observation_artifact_id": None,
                    },
                    agent=record.stage,
                    level="info" if bounded.get("ok") else "error",
                )
                return bounded
            result = binding.executor.call(invocation.tool_name, invocation.arguments)
            if not isinstance(result, dict):
                raise CapabilityGatewayError(
                    "invalid_tool_result",
                    "Business tools must return a JSON object",
                    status_code=500,
                )
            charge_budget = bool(result.pop("_gateway_budget_charge", True))
            if charge_budget:
                binding.tool_call_count += 1
                binding.tool_call_counts[invocation.tool_name] = tool_count + 1
            bounded = sanitize(result)
            binding.cached_calls[invocation.call_id] = (request_digest, bounded)
            binding.store.emit(
                "capability_tool_result",
                {
                    "grant_id": record.grant_id,
                    "attempt_id": record.attempt_id,
                    "tool": invocation.tool_name,
                    "call_id": invocation.call_id,
                    "ok": bounded.get("ok", True),
                    "truncated": bounded.get("truncated", False),
                    "observation_artifact_id": bounded.get("observation_artifact_id"),
                    "observation_chars": (bounded.get("char_count")
                                          if bounded.get("observation_artifact_id") else
                                          len(json.dumps(sanitize(result), ensure_ascii=False,
                                                         sort_keys=True, default=str))),
                    "returned_chars": len(json.dumps(bounded, ensure_ascii=False,
                                                      sort_keys=True, default=str)),
                },
                agent=record.stage,
                level="error" if not bounded.get("ok", True) else "info",
            )
        if binding.on_tool_result is not None:
            try:
                # Observe the committed business result.
                binding.on_tool_result(invocation, result)
            except Exception as exc:
                binding.store.emit(
                    "capability_result_observer_failed",
                    {"attempt_id": record.attempt_id, "tool": invocation.tool_name,
                     "call_id": invocation.call_id, "error": str(exc)[:500]},
                    agent=record.stage, level="warning",
                )
        return bounded

    def revoke(self, registered: RegisteredCapability, *, reason: str = "revoked") -> None:
        self.authority.revoke(registered.record.grant_id, reason=reason)
        with self._lock:
            self._bindings.pop(registered.record.grant_id, None)
            lease = self._lease_tasks.pop(registered.record.grant_id, None)
        if lease is not None:
            lease.get_loop().call_soon_threadsafe(lease.cancel)

    def _validate_arguments(
        self,
        record: CapabilityRecord,
        binding: _Binding,
        arguments: Mapping[str, Any],
    ) -> None:
        for key, value in _walk_fields(arguments):
            normalized = key.strip().casefold()
            if normalized in _CREDENTIAL_KEYS:
                raise CapabilityGatewayError(
                    "credential_argument_denied",
                    "Credentials are not valid business-tool arguments",
                    status_code=403,
                )
            if normalized in _PATH_KEYS or normalized.endswith("_path"):
                raise CapabilityGatewayError(
                    "path_argument_denied",
                    "Business tool calls cannot select filesystem paths",
                    status_code=403,
                )
            if normalized in {"artifact_id", "artifact_ids"} or normalized.endswith(
                "_artifact_id"
            ) or normalized.endswith("_artifact_ids"):
                if (normalized == "image_artifact_id" and isinstance(value, str)
                        and not value.strip()):
                    # Optional image support is absent for text-only citations.
                    # EvidenceCitation still requires a nonempty text quote.
                    continue
                for artifact_id in _id_values(value):
                    try:
                        artifact_record = binding.store.artifact_record(artifact_id)
                    except FileNotFoundError as exc:
                        raise CapabilityGatewayError(
                            "artifact_not_found", "Artifact is unavailable in this Run", status_code=404
                        ) from exc
                    source_scope = set(record.resource_scope.get("source_ids", ()))
                    source_owned = bool(
                        source_scope and artifact_record.source_id in source_scope
                    )
                    if artifact_id not in binding.allowed_artifact_ids and not source_owned:
                        raise CapabilityGatewayError(
                            "artifact_scope_denied",
                            "Artifact is outside this Child Attempt's scope",
                            status_code=403,
                        )
            for resource in _SCOPED_RESOURCES:
                if normalized not in {f"{resource}_id", f"{resource}_ids"}:
                    continue
                permitted = set(record.resource_scope.get(f"{resource}_ids", ()))
                if permitted:
                    for identifier in _id_values(value):
                        if identifier not in permitted:
                            raise CapabilityGatewayError(
                                f"{resource}_scope_denied",
                                f"{resource.title()} is outside this Child Attempt's scope",
                                status_code=403,
                            )
