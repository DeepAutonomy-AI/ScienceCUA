from __future__ import annotations

import hashlib
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import ChildRunCapsule
from ..models import utc_now
from ..storage import RunStore
from .policy import tools_for_stage


class CapabilityAuthorizationError(PermissionError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CapabilityRecord(BaseModel):
    """Persistable grant metadata. The bearer credential is never part of it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "capability-record.v1"
    grant_id: str = Field(pattern=r"^cap_[A-Za-z0-9_-]+$")
    token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str
    root_session_id: str
    child_session_id: str
    attempt_id: str
    stage: str
    allowed_tools: tuple[str, ...]
    resource_scope: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    issued_at: str
    issued_at_epoch: float
    expires_at_epoch: float
    revoked_at: str | None = None
    revocation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class IssuedCapability:
    record: CapabilityRecord
    token: str = field(repr=False)


_RESOURCE_KEYS = (
    "artifact",
    "source",
    "task",
    "reference",
    "trajectory",
    "evidence",
)


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value)
    return ()


def resource_scope_from_capsule(
    capsule: ChildRunCapsule, store: RunStore | None = None,
) -> dict[str, tuple[str, ...]]:
    scope: dict[str, tuple[str, ...]] = {
        "artifact_ids": tuple(sorted(set(capsule.input_artifact_ids)))
    }
    raw = capsule.scope
    for resource in _RESOURCE_KEYS:
        values: list[str] = []
        for key in (
            f"{resource}_id",
            f"{resource}_ids",
            f"allowed_{resource}_id",
            f"allowed_{resource}_ids",
        ):
            values.extend(_strings(raw.get(key)))
        if resource == "artifact":
            values.extend(capsule.input_artifact_ids)
        if values:
            scope[f"{resource}_ids"] = tuple(sorted(set(values)))
    if store is not None and "view_evidence_image" in capsule.allowed_tools:
        images: set[str] = set(scope.get("artifact_ids", ()))
        for snapshot_id in capsule.input_artifact_ids:
            record = store.artifact_record(snapshot_id)
            if record.kind != "corpus_snapshot":
                continue
            snapshot = store.read_json_artifact(snapshot_id)
            for evidence in snapshot.get("evidence", []):
                if scope.get("source_ids") and evidence.get("source_id") not in scope["source_ids"]:
                    continue
                if scope.get("evidence_ids") and evidence.get("evidence_id") not in scope["evidence_ids"]:
                    continue
                for image in evidence.get("image_refs", []):
                    image_id = image.get("artifact_id")
                    try:
                        image_record = store.artifact_record(image_id)
                    except (FileNotFoundError, ValueError, TypeError):
                        continue
                    if (image_record.media_type.startswith("image/")
                            and image_record.source_id == evidence.get("source_id")):
                        images.add(image_id)
        scope["artifact_ids"] = tuple(sorted(images))
    return scope


class CapabilityAuthority:
    """Issues short-lived opaque credentials bound to exactly one Child Attempt."""

    def __init__(self, *, default_ttl_seconds: float = 900.0) -> None:
        if default_ttl_seconds <= 0:
            raise ValueError("Capability TTL must be positive")
        self.default_ttl_seconds = default_ttl_seconds
        self._records_by_digest: dict[str, CapabilityRecord] = {}
        self._stores_by_grant: dict[str, RunStore] = {}
        self._lock = threading.RLock()

    def issue(
        self,
        store: RunStore,
        attempt_id: str,
        *,
        allowed_tools: tuple[str, ...] | None = None,
        ttl_seconds: float | None = None,
    ) -> IssuedCapability:
        attempt = store.read_attempt(attempt_id)
        session = store.read_session(attempt.child_session_id)
        capsule = ChildRunCapsule.model_validate(
            store.read_json_artifact(attempt.capsule_artifact_id)
        )
        if attempt.run_id != store.run_id or capsule.run_id != store.run_id:
            raise CapabilityAuthorizationError("run_mismatch", "Attempt belongs to another Run")
        if (
            session.session_kind != "child"
            or session.attempt_id != attempt_id
            or capsule.attempt_id != attempt_id
            or capsule.child_session_id != attempt.child_session_id
            or capsule.parent_root_session_id != attempt.parent_root_session_id
            or capsule.stage != attempt.stage
        ):
            raise CapabilityAuthorizationError(
                "attempt_binding_mismatch", "Child Session binding is inconsistent"
            )
        if attempt.status not in {"created", "running"}:
            raise CapabilityAuthorizationError(
                "attempt_terminal", "A terminal Attempt cannot receive a capability"
            )

        modality = str(capsule.scope.get("modality") or "") or None
        policy_tools = tools_for_stage(attempt.stage, modality=modality)
        requested = tuple(allowed_tools if allowed_tools is not None else capsule.allowed_tools)
        if len(set(requested)) != len(requested):
            raise CapabilityAuthorizationError("duplicate_tool", "Tool grants must be unique")
        if not set(requested).issubset(set(capsule.allowed_tools)):
            raise CapabilityAuthorizationError(
                "capsule_tool_mismatch", "Requested tools exceed the Child Capsule"
            )
        if not set(requested).issubset(policy_tools):
            raise CapabilityAuthorizationError(
                "stage_tool_denied", "Requested tools exceed the stage policy"
            )

        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if ttl <= 0:
            raise ValueError("Capability TTL must be positive")
        token = secrets.token_urlsafe(48)
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.time()
        record = CapabilityRecord(
            grant_id=f"cap_{uuid.uuid4().hex}",
            token_sha256=digest,
            run_id=store.run_id,
            root_session_id=attempt.parent_root_session_id,
            child_session_id=attempt.child_session_id,
            attempt_id=attempt.attempt_id,
            stage=attempt.stage,
            allowed_tools=tuple(sorted(requested)),
            resource_scope=resource_scope_from_capsule(capsule, store),
            issued_at=utc_now(),
            issued_at_epoch=now,
            expires_at_epoch=now + ttl,
        )
        with self._lock:
            self._records_by_digest[digest] = record
            self._stores_by_grant[record.grant_id] = store
            store.write_capability_record(record.grant_id, record.model_dump(mode="json"))
        store.emit(
            "capability_issued",
            {
                "grant_id": record.grant_id,
                "attempt_id": record.attempt_id,
                "child_session_id": record.child_session_id,
                "stage": record.stage,
                "allowed_tools": list(record.allowed_tools),
                "expires_at_epoch": record.expires_at_epoch,
            },
            agent=record.stage,
        )
        return IssuedCapability(record=record, token=token)

    def authorize(
        self,
        token: str,
        *,
        run_id: str,
        root_session_id: str,
        child_session_id: str,
        attempt_id: str,
        stage: str,
        tool_name: str,
    ) -> CapabilityRecord:
        if not token:
            raise CapabilityAuthorizationError("missing_token", "Capability token is required")
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock:
            record = self._records_by_digest.get(digest)
        if record is None or not secrets.compare_digest(record.token_sha256, digest):
            raise CapabilityAuthorizationError("invalid_token", "Capability token is invalid")
        if record.revoked_at:
            raise CapabilityAuthorizationError("revoked_token", "Capability token is revoked")
        if time.time() >= record.expires_at_epoch:
            raise CapabilityAuthorizationError("expired_token", "Capability token is expired")
        actual = (
            record.run_id,
            record.root_session_id,
            record.child_session_id,
            record.attempt_id,
            record.stage,
        )
        expected = (run_id, root_session_id, child_session_id, attempt_id, stage)
        if actual != expected:
            raise CapabilityAuthorizationError(
                "binding_mismatch", "Capability does not belong to this Run/Session/Attempt"
            )
        if tool_name not in record.allowed_tools:
            raise CapabilityAuthorizationError(
                "tool_not_allowed", f"Tool {tool_name!r} is not granted to this Attempt"
            )
        store = self._stores_by_grant[record.grant_id]
        attempt = store.read_attempt(record.attempt_id)
        if attempt.status not in {"created", "running"}:
            raise CapabilityAuthorizationError(
                "attempt_terminal", "Capability ended with its Attempt"
            )
        return record

    def renew(self, grant_id: str, *, ttl_seconds: float | None = None) -> CapabilityRecord:
        """Host-only renewal; never revive an expired, revoked, or terminal grant."""
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if ttl <= 0:
            raise ValueError("Capability TTL must be positive")
        with self._lock:
            current = next(
                (item for item in self._records_by_digest.values() if item.grant_id == grant_id),
                None,
            )
            if current is None:
                raise CapabilityAuthorizationError("unknown_grant", "Capability grant not found")
            if current.revoked_at:
                raise CapabilityAuthorizationError("revoked_token", "Capability token is revoked")
            now = time.time()
            if now >= current.expires_at_epoch:
                raise CapabilityAuthorizationError("expired_token", "Capability token is expired")
            store = self._stores_by_grant[grant_id]
            if store.read_attempt(current.attempt_id).status not in {"created", "running"}:
                raise CapabilityAuthorizationError("attempt_terminal", "Capability ended with its Attempt")
            updated = current.model_copy(update={"expires_at_epoch": now + ttl})
            store.write_capability_record(grant_id, updated.model_dump(mode="json"))
            self._records_by_digest[current.token_sha256] = updated
        return updated

    def revoke(self, grant_id: str, *, reason: str = "revoked") -> CapabilityRecord:
        with self._lock:
            current = next(
                (item for item in self._records_by_digest.values() if item.grant_id == grant_id),
                None,
            )
            if current is None:
                raise CapabilityAuthorizationError("unknown_grant", "Capability grant not found")
            if current.revoked_at:
                return current
            updated = current.model_copy(
                update={"revoked_at": utc_now(), "revocation_reason": reason[:240]}
            )
            self._records_by_digest[current.token_sha256] = updated
            store = self._stores_by_grant[grant_id]
            store.write_capability_record(grant_id, updated.model_dump(mode="json"))
        store.emit(
            "capability_revoked",
            {"grant_id": grant_id, "attempt_id": updated.attempt_id, "reason": reason[:240]},
            agent=updated.stage,
        )
        return updated
