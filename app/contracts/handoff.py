from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..models import normalize_saved_target_profile_payload, utc_now


ArtifactId = str
SessionKind = Literal["root", "child"]
SessionStatus = Literal["created", "running", "succeeded", "failed", "cancelled"]
AttemptStatus = Literal["created", "running", "succeeded", "failed", "cancelled"]
ReceiptStatus = Literal["succeeded", "degraded", "failed", "cancelled"]

MAX_RUN_CAPSULE_BYTES = 64 * 1024
MAX_CHILD_CAPSULE_BYTES = 64 * 1024
MAX_RECEIPT_BYTES = 128 * 1024
MAX_SCOPE_BYTES = 8 * 1024

_ARTIFACT_PATTERN = r"^art_[0-9a-f]{24}$"
_RUN_PATTERN = r"^run_[A-Za-z0-9_-]+$"
_ROOT_PATTERN = r"^root_[A-Za-z0-9_-]+$"
_CHILD_PATTERN = r"^child_[A-Za-z0-9_-]+$"
_ATTEMPT_PATTERN = r"^attempt_[A-Za-z0-9_-]+$"
_RECEIPT_PATTERN = r"^rcpt_[A-Za-z0-9_-]+$"
_VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_TOOL_PATTERN = r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$"
_FORBIDDEN_HANDOFF_KEYS = {
    "messages",
    "events",
    "history",
    "chat_history",
    "conversation",
    "api_key",
    "authorization",
    "password",
    "secret",
    "access_token",
    "refresh_token",
    "capability_token",
}


def _json_size(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _assert_size(value: Any, limit: int, label: str) -> None:
    size = _json_size(value)
    if size > limit:
        raise ValueError(f"{label} exceeds {limit} bytes; store large content as an Artifact")


def _find_forbidden_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            if normalized in _FORBIDDEN_HANDOFF_KEYS:
                return normalized
            nested = _find_forbidden_key(item)
            if nested:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _find_forbidden_key(item)
            if nested:
                return nested
    return None


def _unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    return values


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelRoute(FrozenContract):
    """Persistable model routing metadata. It deliberately excludes credentials."""

    provider: str | None = Field(default=None, max_length=160)
    model: str = Field(min_length=1, max_length=240)
    api_protocol: Literal["chat_completions", "responses"]
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"]
    endpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    max_tokens: int | None = Field(default=None, ge=1)

    @classmethod
    def from_public_config(cls, value: Mapping[str, Any]) -> "ModelRoute":
        endpoint = str(value.get("base_url") or "").strip()
        return cls(
            provider=str(value.get("provider") or "").strip() or None,
            model=str(value.get("model") or "unconfigured").strip() or "unconfigured",
            api_protocol=str(value.get("api_protocol") or "chat_completions"),
            reasoning_effort=str(value.get("reasoning_effort") or "medium"),
            endpoint_sha256=hashlib.sha256(endpoint.encode()).hexdigest() if endpoint else None,
            max_tokens=int(value["max_tokens"]) if value.get("max_tokens") is not None else None,
        )


class TargetProfileSnapshot(FrozenContract):
    software_name: str = Field(min_length=1, max_length=200)
    software_id: str | None = Field(default=None, max_length=200)
    target_version: str = Field(min_length=1, max_length=100)
    platform: str = Field(min_length=1, max_length=100)
    ui_locale: str = Field(default="unknown", min_length=1, max_length=100)

    @model_validator(mode="before")
    @classmethod
    def ignore_retired_preferences(cls, value: Any) -> Any:
        return normalize_saved_target_profile_payload(value)


class RunContextCapsule(FrozenContract):
    schema_version: Literal["run-context.v1"] = "run-context.v1"
    run_id: str = Field(pattern=_RUN_PATTERN)
    root_session_id: str = Field(pattern=_ROOT_PATTERN)
    target_profile: TargetProfileSnapshot
    input_artifact_ids: tuple[ArtifactId, ...] = Field(default=(), max_length=256)
    model_route: ModelRoute
    output_language: Literal["Chinese", "English"] = "English"
    prompt_set_version: str = Field(pattern=_VERSION_PATTERN)
    created_at: str = Field(default_factory=utc_now)

    @field_validator("input_artifact_ids")
    @classmethod
    def validate_artifact_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not re.fullmatch(_ARTIFACT_PATTERN, value):
                raise ValueError(f"Invalid Artifact ID: {value}")
        return _unique(values, "input_artifact_ids")

    @model_validator(mode="after")
    def bound_payload(self) -> "RunContextCapsule":
        _assert_size(self.model_dump(mode="json"), MAX_RUN_CAPSULE_BYTES, "RunContextCapsule")
        return self


class ChildRunCapsule(FrozenContract):
    schema_version: Literal["child-run.v1"] = "child-run.v1"
    run_id: str = Field(pattern=_RUN_PATTERN)
    parent_root_session_id: str = Field(pattern=_ROOT_PATTERN)
    child_session_id: str = Field(pattern=_CHILD_PATTERN)
    attempt_id: str = Field(pattern=_ATTEMPT_PATTERN)
    stage: str = Field(min_length=1, max_length=120, pattern=_VERSION_PATTERN)
    input_artifact_ids: tuple[ArtifactId, ...] = Field(default=(), max_length=256)
    scope: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: tuple[str, ...] = Field(default=(), max_length=64)
    output_contract: str = Field(pattern=_VERSION_PATTERN)
    prompt_version: str = Field(pattern=_VERSION_PATTERN)
    created_at: str = Field(default_factory=utc_now)

    @field_validator("input_artifact_ids")
    @classmethod
    def validate_artifact_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not re.fullmatch(_ARTIFACT_PATTERN, value):
                raise ValueError(f"Invalid Artifact ID: {value}")
        return _unique(values, "input_artifact_ids")

    @field_validator("allowed_tools")
    @classmethod
    def validate_tools(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not re.fullmatch(_TOOL_PATTERN, value):
                raise ValueError(f"Invalid tool name: {value}")
        return _unique(values, "allowed_tools")

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, value: dict[str, Any]) -> dict[str, Any]:
        forbidden = _find_forbidden_key(value)
        if forbidden:
            raise ValueError(f"Child scope cannot contain forbidden field: {forbidden}")
        _assert_size(value, MAX_SCOPE_BYTES, "ChildRunCapsule.scope")
        return value

    @model_validator(mode="after")
    def bound_payload(self) -> "ChildRunCapsule":
        _assert_size(self.model_dump(mode="json"), MAX_CHILD_CAPSULE_BYTES, "ChildRunCapsule")
        return self


class StageReceipt(FrozenContract):
    schema_version: Literal["stage-receipt.v1"] = "stage-receipt.v1"
    receipt_id: str = Field(pattern=_RECEIPT_PATTERN)
    idempotency_key: str = Field(min_length=1, max_length=240, pattern=r"^[A-Za-z0-9._:-]+$")
    run_id: str = Field(pattern=_RUN_PATTERN)
    stage: str = Field(min_length=1, max_length=120, pattern=_VERSION_PATTERN)
    status: ReceiptStatus
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_id: str | None = Field(default=None, pattern=_ATTEMPT_PATTERN)
    accepted_output_artifact_ids: tuple[ArtifactId, ...] = Field(default=(), max_length=512)
    validation_artifact_id: ArtifactId | None = Field(default=None, pattern=_ARTIFACT_PATTERN)
    counts: dict[str, int] = Field(default_factory=dict)
    gaps: tuple[str, ...] = Field(default=(), max_length=128)
    warnings: tuple[str, ...] = Field(default=(), max_length=128)
    completed_at: str = Field(default_factory=utc_now)

    @field_validator("accepted_output_artifact_ids")
    @classmethod
    def unique_outputs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(values, "accepted_output_artifact_ids")

    @field_validator("counts")
    @classmethod
    def nonnegative_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("Receipt counts cannot be negative")
        return value

    @model_validator(mode="after")
    def bound_payload(self) -> "StageReceipt":
        _assert_size(self.model_dump(mode="json"), MAX_RECEIPT_BYTES, "StageReceipt")
        return self

    def semantic_digest(self) -> str:
        payload = self.model_dump(mode="json", exclude={"receipt_id", "completed_at"})
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class SessionRecord(FrozenContract):
    schema_version: Literal["session-record.v1"] = "session-record.v1"
    run_id: str = Field(pattern=_RUN_PATTERN)
    session_id: str = Field(pattern=r"^(root|child)_[A-Za-z0-9_-]+$")
    session_kind: SessionKind
    parent_root_session_id: str | None = Field(default=None, pattern=_ROOT_PATTERN)
    stage: str | None = Field(default=None, max_length=120, pattern=_VERSION_PATTERN)
    attempt_id: str | None = Field(default=None, pattern=_ATTEMPT_PATTERN)
    attempt_number: int | None = Field(default=None, ge=1)
    status: SessionStatus = "created"
    model_route: ModelRoute
    prompt_version: str = Field(pattern=_VERSION_PATTERN)
    allowed_tools: tuple[str, ...] = Field(default=(), max_length=64)
    started_at: str | None = None
    ended_at: str | None = None
    termination_reason: str | None = Field(default=None, max_length=2_000)
    dsh_log_artifact_id: str | None = Field(default=None, pattern=_ARTIFACT_PATTERN)
    usage: dict[str, int | float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_relationship(self) -> "SessionRecord":
        if self.session_kind == "root":
            if not self.session_id.startswith("root_"):
                raise ValueError("Root Session ID must start with root_")
            if self.parent_root_session_id or self.attempt_id or self.attempt_number:
                raise ValueError("Root Session cannot have a parent or Child attempt")
        else:
            if not self.session_id.startswith("child_"):
                raise ValueError("Child Session ID must start with child_")
            if not self.parent_root_session_id or not self.attempt_id or not self.attempt_number:
                raise ValueError("Child Session requires parent_root_session_id and attempt fields")
            if not self.stage:
                raise ValueError("Child Session requires a stage")
        if self.status in {"succeeded", "failed", "cancelled"} and not self.ended_at:
            raise ValueError("Terminal Session requires ended_at")
        return self


class AttemptRecord(FrozenContract):
    schema_version: Literal["attempt-record.v1"] = "attempt-record.v1"
    attempt_id: str = Field(pattern=_ATTEMPT_PATTERN)
    run_id: str = Field(pattern=_RUN_PATTERN)
    stage: str = Field(min_length=1, max_length=120, pattern=_VERSION_PATTERN)
    attempt_number: int = Field(ge=1)
    child_session_id: str = Field(pattern=_CHILD_PATTERN)
    parent_root_session_id: str = Field(pattern=_ROOT_PATTERN)
    retry_of_attempt_id: str | None = Field(default=None, pattern=_ATTEMPT_PATTERN)
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    capsule_artifact_id: str = Field(pattern=_ARTIFACT_PATTERN)
    status: AttemptStatus = "created"
    error_category: str | None = Field(default=None, max_length=120)
    started_at: str | None = None
    ended_at: str | None = None
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_terminal_time(self) -> "AttemptRecord":
        if self.status in {"succeeded", "failed", "cancelled"} and not self.ended_at:
            raise ValueError("Terminal Attempt requires ended_at")
        return self
