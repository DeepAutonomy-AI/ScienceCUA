from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from .contracts import (
    AttemptRecord,
    ChildRunCapsule,
    ModelRoute,
    RunContextCapsule,
    SessionRecord,
    StageReceipt,
)
from .models import ArtifactRecord, normalize_saved_target_profile_payload, utc_now


SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "set-cookie",
    "password",
    "secret",
    "access_token",
    "refresh_token",
    "capability_token",
}

DEFAULT_MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
PROTECTED_STATE_PROJECTION_FIELDS = {
    "id_format",
    "status",
    "stage",
    "phase",
    "revision",
    "last_receipt_id",
    "recovery_cursor",
    "progress",
    "root_session_id",
    "run_context_artifact_id",
}


class RunStoreConflictError(RuntimeError):
    """A persisted immutable Run record conflicts with a new submission."""


class ReceiptConflictError(RunStoreConflictError):
    """A Receipt was replayed with different content or against the wrong phase."""


class ArtifactLimitError(ValueError):
    """An Artifact exceeds the configured immutable object size limit."""


class StateProjectionMutationError(RunStoreConflictError):
    """Critical Run state may only be projected from a committed Receipt."""


def sanitize(value: Any, *, redact_text: bool = True) -> Any:
    """Remove credentials from persisted operational data.

    ``redact_text=False`` is reserved for source-grounded domain artifacts whose
    text, hashes, and offsets form one integrity boundary. Explicitly sensitive
    fields are still redacted in that mode, but ordinary strings must round-trip
    byte-for-byte.
    """

    if hasattr(value, "model_dump"):
        return sanitize(value.model_dump(mode="json"), redact_text=redact_text)
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if key.lower() in SENSITIVE_KEYS
                else sanitize(item, redact_text=redact_text)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize(item, redact_text=redact_text) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item, redact_text=redact_text) for item in value]
    if isinstance(value, str) and redact_text:
        value = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
        return re.sub(r"(?i)(https?://)[^/\s:@]+:[^@\s/]+@", r"\1[REDACTED]@", value)
    return value


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


class RunStore:
    def __init__(
        self,
        data_dir: Path,
        run_id: str,
        *,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    ):
        self.run_id = run_id
        self.max_artifact_bytes = max_artifact_bytes
        self.root = data_dir / "runs" / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "incoming").mkdir(exist_ok=True)
        (self.root / "artifacts" / "objects").mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts" / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "sources").mkdir(exist_ok=True)
        (self.root / "sessions").mkdir(exist_ok=True)
        (self.root / "attempts").mkdir(exist_ok=True)
        (self.root / "receipts").mkdir(exist_ok=True)
        (self.root / "capabilities").mkdir(exist_ok=True)
        self._lock = threading.RLock()

    @property
    def audit_path(self) -> Path:
        return self.root / "audit.jsonl"

    def canonical_id(self, prefix: str, legacy_id: str, *identity: str) -> str:
        """Allocate a persistent, run-scoped ID for a domain object.

        Old runs retain their original IDs. New runs persist the short ID itself;
        the registry stores only an identity fingerprint for retry stability.
        """

        state_path = self.root / "state.json"
        if not state_path.exists() or json.loads(state_path.read_text(encoding="utf-8")).get("id_format") != "short_v1":
            return legacy_id
        if not re.fullmatch(r"[A-Z]{1,4}", prefix):
            raise ValueError("Invalid short ID prefix")
        fingerprint = hashlib.sha256("\x1f".join(identity).encode("utf-8")).hexdigest()
        path = self.root / "id_registry.sqlite3"
        with self._lock:
            with closing(sqlite3.connect(path, timeout=30)) as db:
                with db:
                    db.execute("""CREATE TABLE IF NOT EXISTS ids (
                        namespace TEXT NOT NULL,
                        fingerprint TEXT NOT NULL,
                        ordinal INTEGER NOT NULL,
                        PRIMARY KEY (namespace, fingerprint),
                        UNIQUE (namespace, ordinal)
                    )""")
                    db.execute("BEGIN IMMEDIATE")
                    row = db.execute(
                        "SELECT ordinal FROM ids WHERE namespace = ? AND fingerprint = ?",
                        (prefix, fingerprint),
                    ).fetchone()
                    if row is None:
                        number = db.execute(
                            "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM ids WHERE namespace = ?",
                            (prefix,),
                        ).fetchone()[0]
                        db.execute(
                            "INSERT INTO ids (namespace, fingerprint, ordinal) VALUES (?, ?, ?)",
                            (prefix, fingerprint, number),
                        )
                    else:
                        number = row[0]
            return f"{prefix}{number}"

    def initialize(self, request: dict[str, Any]) -> None:
        sanitized_request = sanitize(request)
        if "target_profile" in sanitized_request:
            sanitized_request["target_profile"] = normalize_saved_target_profile_payload(sanitized_request["target_profile"])
        with self._lock:
            request_path = self.root / "request.json"
            state_path = self.root / "state.json"
            if request_path.exists() or state_path.exists():
                existing = (
                    json.loads(request_path.read_text(encoding="utf-8"))
                    if request_path.exists()
                    else None
                )
                if isinstance(existing, dict) and "target_profile" in existing:
                    existing["target_profile"] = normalize_saved_target_profile_payload(existing["target_profile"])
                if existing == sanitized_request and state_path.exists():
                    return
                raise RunStoreConflictError(f"Run {self.run_id} is already initialized")

            root_session_id = f"root_{uuid.uuid4().hex}"
            model_route = ModelRoute.from_public_config(request.get("model", {}))
            capsule = RunContextCapsule(
                run_id=self.run_id,
                root_session_id=root_session_id,
                target_profile=sanitized_request.get("target_profile", {}),
                input_artifact_ids=tuple(
                    str(item["artifact_id"])
                    for item in request.get("inputs", [])
                    if isinstance(item, dict) and item.get("artifact_id")
                ),
                model_route=model_route,
                output_language=request.get("output_language", "English"),
                prompt_set_version=str(
                    request.get("prompt_set_version") or "resource-to-skill.v1"
                ),
            )
            for artifact_id in capsule.input_artifact_ids:
                self.artifact_record(artifact_id)
            capsule_artifact = self.put_json_artifact(
                capsule,
                kind="run_context_capsule",
                metadata={"schema_version": capsule.schema_version},
            )
            root_session = SessionRecord(
                run_id=self.run_id,
                session_id=root_session_id,
                session_kind="root",
                model_route=model_route,
                prompt_version=capsule.prompt_set_version,
            )
            self._write_json(request_path, sanitized_request)
            self._write_json(
                self.root / "sessions" / f"{root_session_id}.json",
                root_session.model_dump(mode="json"),
            )
            self._write_json(
                state_path,
                {
                    "run_id": self.run_id,
                    "id_format": "short_v1",
                    "status": "queued",
                    "stage": "registered",
                    "phase": "created",
                    "revision": 0,
                    "last_receipt_id": None,
                    "recovery_cursor": None,
                    "root_session_id": root_session_id,
                    "run_context_artifact_id": capsule_artifact.artifact_id,
                    "progress": 0,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "target_profile": sanitized_request.get("target_profile", {}),
                    "model": sanitized_request.get("model", {}),
                    "input_count": len(request.get("inputs", [])),
                    "source_mode": sanitized_request.get("source_mode", "manual"),
                    "scope_selection_enabled": bool(sanitized_request.get("scope_selection_enabled", False)),
                    **({"output_language": sanitized_request["output_language"]}
                       if "output_language" in sanitized_request else {}),
                    **{key: sanitized_request[key] for key in ("generation_coverage", "generation_limits")
                       if key in sanitized_request},
                },
            )

    def stage_upload(self, input_id: str, filename: str, data: bytes) -> Path:
        safe_name = Path(filename or "upload.bin").name
        destination = self.root / "incoming" / input_id / safe_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination

    def update_state(self, **changes: Any) -> dict[str, Any]:
        forbidden = PROTECTED_STATE_PROJECTION_FIELDS.intersection(changes)
        if forbidden:
            fields = ", ".join(sorted(forbidden))
            raise StateProjectionMutationError(
                f"Receipt-gated state fields cannot be updated directly: {fields}"
            )
        with self._lock:
            state = self.read_state()
            for key in ("generation_coverage", "generation_limits"):
                if key in changes and key in state and changes[key] != state[key]:
                    raise StateProjectionMutationError(f"Generation configuration is fixed for this run: {key}")
            state.update(sanitize(changes))
            state["updated_at"] = utc_now()
            self._write_json(self.root / "state.json", state)
            return state

    def read_state(self) -> dict[str, Any]:
        path = self.root / "state.json"
        if not path.exists():
            return {"run_id": self.run_id, "status": "unknown"}
        return json.loads(path.read_text(encoding="utf-8"))

    def cancellation_requested(self) -> bool:
        return self.read_state().get("status") in {"cancelling", "cancelled"}

    def read_run_context(self) -> RunContextCapsule:
        state = self.read_state()
        artifact_id = state.get("run_context_artifact_id")
        if not isinstance(artifact_id, str):
            raise FileNotFoundError("Run context has not been initialized")
        return RunContextCapsule.model_validate(self.read_json_artifact(artifact_id))

    def create_session(self, record: SessionRecord) -> SessionRecord:
        if record.run_id != self.run_id:
            raise RunStoreConflictError("Session belongs to another Run")
        path = self.root / "sessions" / f"{record.session_id}.json"
        with self._lock:
            if path.exists():
                existing = SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
                if existing == record:
                    return existing
                raise RunStoreConflictError(f"Session already exists: {record.session_id}")
            if record.session_kind == "child":
                parent = self.read_session(str(record.parent_root_session_id))
                if parent.session_kind != "root":
                    raise RunStoreConflictError("Child parent must be this Run's Root Session")
            self._write_json(path, record.model_dump(mode="json"))
        return record

    def read_session(self, session_id: str) -> SessionRecord:
        if not re.fullmatch(r"(root|child)_[A-Za-z0-9_-]+", session_id):
            raise FileNotFoundError("Invalid Session ID")
        path = self.root / "sessions" / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(session_id)
        record = SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.run_id != self.run_id:
            raise RunStoreConflictError("Session ownership mismatch")
        return record

    def list_sessions(self) -> list[SessionRecord]:
        records = [
            SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in (self.root / "sessions").glob("*.json")
        ]
        return sorted(records, key=lambda item: item.created_at)

    def read_attempt(self, attempt_id: str) -> AttemptRecord:
        if not re.fullmatch(r"attempt_[A-Za-z0-9_-]+", attempt_id):
            raise FileNotFoundError("Invalid Attempt ID")
        path = self.root / "attempts" / f"{attempt_id}.json"
        if not path.exists():
            raise FileNotFoundError(attempt_id)
        record = AttemptRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.run_id != self.run_id:
            raise RunStoreConflictError("Attempt ownership mismatch")
        return record

    def list_attempts(self) -> list[AttemptRecord]:
        records = [
            AttemptRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in (self.root / "attempts").glob("*.json")
        ]
        return sorted(records, key=lambda item: item.created_at)

    def begin_child_attempt(
        self,
        *,
        stage: str,
        input_artifact_ids: tuple[str, ...],
        scope: dict[str, Any],
        allowed_tools: tuple[str, ...],
        output_contract: str,
        prompt_version: str,
        model_route: ModelRoute | None = None,
        retry_of_attempt_id: str | None = None,
    ) -> tuple[AttemptRecord, ChildRunCapsule, SessionRecord]:
        with self._lock:
            state = self.read_state()
            root_session_id = state.get("root_session_id")
            if not isinstance(root_session_id, str):
                raise RunStoreConflictError("Run has no Root Session")
            root_session = self.read_session(root_session_id)
            for artifact_id in input_artifact_ids:
                self.artifact_record(artifact_id)

            retry_of: AttemptRecord | None = None
            attempt_number = 1
            if retry_of_attempt_id:
                retry_of = self.read_attempt(retry_of_attempt_id)
                if retry_of.stage != stage:
                    raise RunStoreConflictError("Retry must use the same stage")
                if retry_of.status not in {"failed", "cancelled"}:
                    raise RunStoreConflictError("Only failed or cancelled Attempts may be retried")
                attempt_number = retry_of.attempt_number + 1

            fingerprint_payload = {
                "stage": stage,
                "input_artifact_ids": input_artifact_ids,
                "scope": scope,
            }
            input_fingerprint = hashlib.sha256(
                json.dumps(
                    fingerprint_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if retry_of and retry_of.input_fingerprint != input_fingerprint:
                raise RunStoreConflictError("Retry input fingerprint must match the prior Attempt")

            attempt_id = f"attempt_{uuid.uuid4().hex}"
            child_session_id = f"child_{uuid.uuid4().hex}"
            capsule = ChildRunCapsule(
                run_id=self.run_id,
                parent_root_session_id=root_session_id,
                child_session_id=child_session_id,
                attempt_id=attempt_id,
                stage=stage,
                input_artifact_ids=input_artifact_ids,
                scope=scope,
                allowed_tools=allowed_tools,
                output_contract=output_contract,
                prompt_version=prompt_version,
            )
            capsule_artifact = self.put_json_artifact(
                capsule,
                kind="child_run_capsule",
                metadata={
                    "schema_version": capsule.schema_version,
                    "stage": stage,
                    "attempt_id": attempt_id,
                },
            )
            session = SessionRecord(
                run_id=self.run_id,
                session_id=child_session_id,
                session_kind="child",
                parent_root_session_id=root_session_id,
                stage=stage,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                model_route=model_route or root_session.model_route,
                prompt_version=prompt_version,
                allowed_tools=allowed_tools,
            )
            attempt = AttemptRecord(
                attempt_id=attempt_id,
                run_id=self.run_id,
                stage=stage,
                attempt_number=attempt_number,
                child_session_id=child_session_id,
                parent_root_session_id=root_session_id,
                retry_of_attempt_id=retry_of_attempt_id,
                input_fingerprint=input_fingerprint,
                capsule_artifact_id=capsule_artifact.artifact_id,
            )
            self.create_session(session)
            self._write_json(
                self.root / "attempts" / f"{attempt_id}.json",
                attempt.model_dump(mode="json"),
            )
            self.emit(
                "attempt_created",
                {
                    "attempt_id": attempt_id,
                    "child_session_id": child_session_id,
                    "parent_root_session_id": root_session_id,
                    "stage": stage,
                    "attempt_number": attempt_number,
                    "retry_of_attempt_id": retry_of_attempt_id,
                    "capsule_artifact_id": capsule_artifact.artifact_id,
                },
                agent=stage,
            )
            return attempt, capsule, session

    def start_attempt(self, attempt_id: str) -> AttemptRecord:
        with self._lock:
            attempt = self.read_attempt(attempt_id)
            if attempt.status == "running":
                return attempt
            if attempt.status != "created":
                raise RunStoreConflictError(f"Cannot start Attempt in status {attempt.status}")
            now = utc_now()
            updated = AttemptRecord.model_validate(
                {**attempt.model_dump(mode="json"), "status": "running", "started_at": now}
            )
            session = self.read_session(attempt.child_session_id)
            updated_session = SessionRecord.model_validate(
                {**session.model_dump(mode="json"), "status": "running", "started_at": now}
            )
            self._write_json(
                self.root / "attempts" / f"{attempt_id}.json",
                updated.model_dump(mode="json"),
            )
            self._write_json(
                self.root / "sessions" / f"{session.session_id}.json",
                updated_session.model_dump(mode="json"),
            )
            return updated

    def finish_attempt(
        self,
        attempt_id: str,
        *,
        status: str,
        error_category: str | None = None,
        termination_reason: str | None = None,
        usage: dict[str, int | float] | None = None,
        dsh_log_artifact_id: str | None = None,
    ) -> AttemptRecord:
        if status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("Attempt terminal status must be succeeded, failed, or cancelled")
        with self._lock:
            attempt = self.read_attempt(attempt_id)
            if attempt.status in {"succeeded", "failed", "cancelled"}:
                if attempt.status == status:
                    return attempt
                raise RunStoreConflictError("Attempt is already terminal")
            if dsh_log_artifact_id:
                self.artifact_record(dsh_log_artifact_id)
            now = utc_now()
            updated = AttemptRecord.model_validate(
                {
                    **attempt.model_dump(mode="json"),
                    "status": status,
                    "error_category": error_category,
                    "ended_at": now,
                }
            )
            session = self.read_session(attempt.child_session_id)
            updated_session = SessionRecord.model_validate(
                {
                    **session.model_dump(mode="json"),
                    "status": status,
                    "ended_at": now,
                    "termination_reason": termination_reason,
                    "usage": usage or {},
                    "dsh_log_artifact_id": dsh_log_artifact_id,
                }
            )
            self._write_json(
                self.root / "attempts" / f"{attempt_id}.json",
                updated.model_dump(mode="json"),
            )
            self._write_json(
                self.root / "sessions" / f"{session.session_id}.json",
                updated_session.model_dump(mode="json"),
            )
            self.emit(
                "attempt_finished",
                {
                    "attempt_id": attempt_id,
                    "child_session_id": attempt.child_session_id,
                    "status": status,
                    "error_category": error_category,
                },
                agent=attempt.stage,
                level="error" if status == "failed" else "info",
            )
            return updated

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        source_id: str | None = None,
        agent: str | None = None,
        level: str = "info",
    ) -> dict[str, Any]:
        event = {
            "event_id": f"evt_{hashlib.sha256(f'{utc_now()}:{threading.get_ident()}'.encode()).hexdigest()[:16]}",
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "event_type": event_type,
            "level": level,
            "source_id": source_id,
            "agent": agent,
            "payload": sanitize(payload or {}),
        }
        line = json.dumps(event, ensure_ascii=False, default=jsonable)
        with self._lock:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return event

    def put_artifact(
        self,
        data: bytes,
        *,
        kind: str,
        media_type: str,
        suffix: str = ".bin",
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        if len(data) > self.max_artifact_bytes:
            raise ArtifactLimitError(
                f"Artifact has {len(data)} bytes; limit is {self.max_artifact_bytes}"
            )
        sha256 = hashlib.sha256(data).hexdigest()
        artifact_id = f"art_{sha256[:24]}"
        clean_suffix = suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,12}", suffix) else ".bin"
        object_path = self.root / "artifacts" / "objects" / f"{artifact_id}{clean_suffix}"
        meta_path = self.root / "artifacts" / "meta" / f"{artifact_id}.json"
        with self._lock:
            if not object_path.exists() and not meta_path.exists():
                object_path.write_bytes(data)
                record = ArtifactRecord(
                    artifact_id=artifact_id,
                    sha256=sha256,
                    kind=kind,
                    media_type=media_type,
                    byte_count=len(data),
                    relative_path=object_path.relative_to(self.root).as_posix(),
                    source_id=source_id,
                    metadata=sanitize(metadata or {}),
                )
                self._write_json(meta_path, record.model_dump(mode="json"))
                return record
            if meta_path.exists():
                return ArtifactRecord.model_validate_json(meta_path.read_text(encoding="utf-8"))
            existing = next((self.root / "artifacts" / "objects").glob(f"{artifact_id}.*"))
            record = ArtifactRecord(
                artifact_id=artifact_id,
                sha256=sha256,
                kind=kind,
                media_type=media_type,
                byte_count=len(data),
                relative_path=existing.relative_to(self.root).as_posix(),
                source_id=source_id,
                metadata=sanitize(metadata or {}),
            )
            self._write_json(meta_path, record.model_dump(mode="json"))
            return record

    def put_json_artifact(
        self,
        value: Any,
        *,
        kind: str,
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        preserve_text_integrity: bool = False,
    ) -> ArtifactRecord:
        # Source-grounded artifacts may contain credential-shaped examples from
        # documentation. Rewriting those strings invalidates Evidence offsets and
        # hashes. Callers must opt in; sensitive dictionary fields remain redacted.
        data = json.dumps(
            sanitize(value, redact_text=not preserve_text_integrity),
            ensure_ascii=False,
            sort_keys=True,
            default=jsonable,
        ).encode("utf-8")
        return self.put_artifact(
            data,
            kind=kind,
            media_type="application/json",
            suffix=".json",
            source_id=source_id,
            metadata=metadata,
        )

    def artifact_record(self, artifact_id: str) -> ArtifactRecord:
        if not re.fullmatch(r"art_[0-9a-f]{24}", artifact_id):
            raise FileNotFoundError("Invalid artifact id")
        path = self.root / "artifacts" / "meta" / f"{artifact_id}.json"
        if not path.exists():
            raise FileNotFoundError(artifact_id)
        return ArtifactRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def artifact_path(self, artifact_id: str) -> Path:
        record = self.artifact_record(artifact_id)
        path = (self.root / record.relative_path).resolve()
        if self.root.resolve() not in path.parents:
            raise FileNotFoundError(artifact_id)
        return path

    def read_json_artifact(self, artifact_id: str) -> Any:
        record = self.artifact_record(artifact_id)
        if record.media_type != "application/json":
            raise ValueError(f"Artifact {artifact_id} is not JSON")
        return json.loads(self.artifact_path(artifact_id).read_text(encoding="utf-8"))

    def put_trajectory_checkpoint(self, *, task_id: str, value: dict[str, Any]) -> str:
        """Publish a completed Task result before acknowledging its terminal tool call."""
        if not re.fullmatch(r"(?:tsk_[A-Za-z0-9_-]+|T[1-9][0-9]*)", task_id):
            raise ValueError("Invalid Task ID")
        with self._lock:
            artifact = self.put_json_artifact(
                value, kind="trajectory_task_checkpoint",
                metadata={"stage": "trajectory_inference", "task_id": task_id,
                          "attempt_id": value["attempt_id"]},
                preserve_text_integrity=True,
            )
            path = self.root / "trajectory_checkpoints" / f"{task_id}.json"
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous.get("artifact_id") != artifact.artifact_id:
                    raise RunStoreConflictError(f"Task {task_id} already has a different checkpoint")
                return artifact.artifact_id
            self._write_json(path, {"task_id": task_id, "artifact_id": artifact.artifact_id})
            return artifact.artifact_id

    def read_trajectory_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"(?:tsk_[A-Za-z0-9_-]+|T[1-9][0-9]*)", task_id):
            raise ValueError("Invalid Task ID")
        path = self.root / "trajectory_checkpoints" / f"{task_id}.json"
        with self._lock:
            if not path.exists():
                return None
            pointer = json.loads(path.read_text(encoding="utf-8"))
            if pointer.get("task_id") != task_id:
                raise RunStoreConflictError("Trajectory checkpoint Task ID mismatch")
            artifact_id = pointer["artifact_id"]
            record = self.artifact_record(artifact_id)
            if record.kind != "trajectory_task_checkpoint":
                raise RunStoreConflictError("Trajectory checkpoint artifact kind mismatch")
            data = self.artifact_path(artifact_id).read_bytes()
            if hashlib.sha256(data).hexdigest() != record.sha256:
                raise RunStoreConflictError("Trajectory checkpoint artifact hash mismatch")
            checkpoint = json.loads(data)
            if checkpoint.get("task_id") != task_id or checkpoint.get("run_id") != self.run_id:
                raise RunStoreConflictError("Trajectory checkpoint ownership mismatch")
            return checkpoint

    def put_task_trajectory_pair(self, *, task_id: str, value: dict[str, Any]) -> str:
        """Persist a validated pair before acknowledging its tool submission."""
        if not re.fullmatch(r"(?:tsk_[A-Za-z0-9_-]+|T[1-9][0-9]*)", task_id):
            raise ValueError("Invalid Task ID")
        with self._lock:
            path = self.root / "task_trajectory_pairs" / f"{task_id}.json"
            if path.exists():
                previous = self.read_task_trajectory_pair(task_id)
                if {key: item for key, item in previous.items() if key != "superseded_by"} != value:
                    raise RunStoreConflictError(f"Task {task_id} already has a different pair")
                return json.loads(path.read_text(encoding="utf-8"))["artifact_id"]
            artifact = self.put_json_artifact(
                value, kind="task_trajectory_pair_checkpoint",
                metadata={"stage": "task_trajectory", "task_id": task_id,
                          "attempt_id": value["attempt_id"]},
                preserve_text_integrity=True,
            )
            self._write_json(path, {"task_id": task_id, "artifact_id": artifact.artifact_id,
                                    "superseded_by": None})
            return artifact.artifact_id

    def read_task_trajectory_pair(self, task_id: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"(?:tsk_[A-Za-z0-9_-]+|T[1-9][0-9]*)", task_id):
            raise ValueError("Invalid Task ID")
        path = self.root / "task_trajectory_pairs" / f"{task_id}.json"
        with self._lock:
            if not path.exists():
                return None
            pointer = json.loads(path.read_text(encoding="utf-8"))
            if pointer.get("task_id") != task_id:
                raise RunStoreConflictError("Task-Trajectory pair Task ID mismatch")
            artifact_id = pointer["artifact_id"]
            record = self.artifact_record(artifact_id)
            if record.kind != "task_trajectory_pair_checkpoint":
                raise RunStoreConflictError("Task-Trajectory pair artifact kind mismatch")
            data = self.artifact_path(artifact_id).read_bytes()
            if hashlib.sha256(data).hexdigest() != record.sha256:
                raise RunStoreConflictError("Task-Trajectory pair artifact hash mismatch")
            checkpoint = json.loads(data)
            if checkpoint.get("task_id") != task_id or checkpoint.get("run_id") != self.run_id:
                raise RunStoreConflictError("Task-Trajectory pair ownership mismatch")
            return checkpoint | {"superseded_by": pointer.get("superseded_by")}

    def list_task_trajectory_pairs(self) -> list[dict[str, Any]]:
        directory = self.root / "task_trajectory_pairs"
        return [self.read_task_trajectory_pair(path.stem)
                for path in sorted(directory.glob("*.json"))] if directory.exists() else []

    def supersede_task_trajectory_pair(self, old_task_id: str, new_task_id: str) -> None:
        with self._lock:
            old = self.read_task_trajectory_pair(old_task_id)
            new = self.read_task_trajectory_pair(new_task_id)
            if old is None or new is None or old_task_id == new_task_id:
                raise ValueError("Supersession requires two existing distinct pairs")
            if old["superseded_by"] not in {None, new_task_id} or new["superseded_by"]:
                raise RunStoreConflictError("Pair supersession conflicts with an existing decision")
            path = self.root / "task_trajectory_pairs" / f"{old_task_id}.json"
            pointer = json.loads(path.read_text(encoding="utf-8"))
            pointer["superseded_by"] = new_task_id
            self._write_json(path, pointer)

    def put_task_trajectory_terminal(self, value: dict[str, Any]) -> str:
        with self._lock:
            path = self.root / "task_trajectory_terminal.json"
            if path.exists():
                pointer = json.loads(path.read_text(encoding="utf-8"))
                previous = self.read_json_artifact(pointer["artifact_id"])
                if previous != value:
                    raise RunStoreConflictError("Task-Trajectory terminal already differs")
                return pointer["artifact_id"]
            artifact = self.put_json_artifact(value, kind="task_trajectory_terminal",
                                              metadata={"stage": "task_trajectory"},
                                              preserve_text_integrity=True)
            self._write_json(path, {"artifact_id": artifact.artifact_id})
            return artifact.artifact_id

    def put_task_trajectory_gap(self, value: dict[str, Any]) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:24]
        with self._lock:
            path = self.root / "task_trajectory_gaps" / f"{digest}.json"
            if path.exists():
                pointer = json.loads(path.read_text(encoding="utf-8"))
                if self.read_json_artifact(pointer["artifact_id"]) != value:
                    raise RunStoreConflictError("Discovery gap hash collision")
                return pointer["artifact_id"]
            artifact = self.put_json_artifact(value, kind="task_trajectory_discovery_gap",
                                              metadata={"stage": "task_trajectory"},
                                              preserve_text_integrity=True)
            self._write_json(path, {"artifact_id": artifact.artifact_id})
            return artifact.artifact_id

    def list_task_trajectory_gaps(self) -> list[dict[str, Any]]:
        directory = self.root / "task_trajectory_gaps"
        if not directory.exists():
            return []
        with self._lock:
            return [self.read_json_artifact(
                json.loads(path.read_text(encoding="utf-8"))["artifact_id"])
                for path in sorted(directory.glob("*.json"))]

    def read_task_trajectory_terminal(self) -> dict[str, Any] | None:
        path = self.root / "task_trajectory_terminal.json"
        with self._lock:
            if not path.exists():
                return None
            pointer = json.loads(path.read_text(encoding="utf-8"))
            record = self.artifact_record(pointer["artifact_id"])
            if record.kind != "task_trajectory_terminal":
                raise RunStoreConflictError("Task-Trajectory terminal artifact kind mismatch")
            data = self.artifact_path(pointer["artifact_id"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != record.sha256:
                raise RunStoreConflictError("Task-Trajectory terminal artifact hash mismatch")
            terminal = json.loads(data)
            if terminal.get("run_id") != self.run_id:
                raise RunStoreConflictError("Task-Trajectory terminal ownership mismatch")
            return terminal

    def import_artifact(self, source_store: "RunStore", artifact_id: str) -> ArtifactRecord:
        """Copy one immutable Artifact from another Run without changing its identity."""

        record = source_store.artifact_record(artifact_id)
        source_path = source_store.artifact_path(artifact_id)
        imported = self.put_artifact(
            source_path.read_bytes(),
            kind=record.kind,
            media_type=record.media_type,
            suffix=source_path.suffix,
            source_id=record.source_id,
            metadata={
                **record.metadata,
                "imported_from_run_id": source_store.run_id,
            },
        )
        if imported.artifact_id != artifact_id or imported.sha256 != record.sha256:
            raise RunStoreConflictError(
                f"Imported Artifact identity changed: {artifact_id}"
            )
        return imported

    def commit_stage_receipt(
        self,
        receipt: StageReceipt,
        *,
        expected_phase: str,
        next_phase: str,
    ) -> dict[str, Any]:
        if receipt.run_id != self.run_id:
            raise ReceiptConflictError("Receipt belongs to another Run")
        with self._lock:
            state = self.read_state()
            if state.get("status") == "unknown":
                raise ReceiptConflictError("Run must be initialized before committing a Receipt")
            receipt_path = self.root / "receipts" / f"{receipt.receipt_id}.json"
            semantic_digest = receipt.semantic_digest()
            if receipt_path.exists():
                existing = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (
                    existing.get("semantic_digest") != semantic_digest
                    or existing.get("to_phase") != next_phase
                ):
                    raise ReceiptConflictError(
                        f"Receipt ID was reused with different content: {receipt.receipt_id}"
                    )
                recovered = self.recover_from_receipts()
                return {**recovered, "receipt_replayed": True}

            for path in (self.root / "receipts").glob("*.json"):
                existing = json.loads(path.read_text(encoding="utf-8"))
                existing_receipt = existing.get("receipt", {})
                if existing_receipt.get("idempotency_key") != receipt.idempotency_key:
                    continue
                if (
                    existing.get("semantic_digest") == semantic_digest
                    and existing.get("to_phase") == next_phase
                ):
                    recovered = self.recover_from_receipts()
                    return {**recovered, "receipt_replayed": True}
                raise ReceiptConflictError(
                    f"Idempotency key was reused with different content: {receipt.idempotency_key}"
                )

            current_phase = str(state.get("phase") or "created")
            if current_phase != expected_phase:
                raise ReceiptConflictError(
                    f"Receipt expected phase {expected_phase}, current phase is {current_phase}"
                )
            artifact_ids = list(receipt.accepted_output_artifact_ids)
            if receipt.validation_artifact_id:
                artifact_ids.append(receipt.validation_artifact_id)
            for artifact_id in artifact_ids:
                self.artifact_record(artifact_id)
            if receipt.attempt_id:
                attempt = self.read_attempt(receipt.attempt_id)
                if receipt.status in {"succeeded", "degraded"} and attempt.status != "succeeded":
                    raise ReceiptConflictError(
                        "A successful Receipt requires a succeeded Attempt"
                    )

            revision = int(state.get("revision") or 0) + 1
            envelope = {
                "schema_version": "receipt-commit.v1",
                "revision": revision,
                "from_phase": expected_phase,
                "to_phase": next_phase,
                "semantic_digest": semantic_digest,
                "commit_digest": _receipt_commit_digest(
                    revision, expected_phase, next_phase, semantic_digest
                ),
                "committed_at": utc_now(),
                "receipt": receipt.model_dump(mode="json"),
            }
            # The Receipt is authoritative and must become durable before state advances.
            self._write_json(receipt_path, envelope)
            state.update(
                {
                    "phase": next_phase,
                    "stage": next_phase,
                    "status": _status_for_phase(next_phase),
                    "revision": revision,
                    "last_receipt_id": receipt.receipt_id,
                    "recovery_cursor": receipt.receipt_id,
                    "progress": _progress_for_phase(next_phase),
                    "updated_at": utc_now(),
                }
            )
            self._write_json(self.root / "state.json", state)
            self.emit(
                "stage_receipt_committed",
                {
                    "receipt_id": receipt.receipt_id,
                    "idempotency_key": receipt.idempotency_key,
                    "from_phase": expected_phase,
                    "to_phase": next_phase,
                    "revision": revision,
                },
                agent=receipt.stage,
            )
            return {**state, "receipt_replayed": False}

    def list_receipt_commits(self) -> list[dict[str, Any]]:
        commits = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (self.root / "receipts").glob("*.json")
        ]
        return sorted(commits, key=lambda item: int(item.get("revision") or 0))

    def recover_from_receipts(self) -> dict[str, Any]:
        with self._lock:
            state = self.read_state()
            if state.get("status") == "unknown":
                raise ReceiptConflictError("Run must be initialized before recovery")
            projected_phase = "created"
            projected_revision = 0
            last_receipt_id: str | None = None
            for commit in self.list_receipt_commits():
                revision = int(commit.get("revision") or 0)
                if revision != projected_revision + 1:
                    raise ReceiptConflictError("Receipt revision chain is not contiguous")
                if commit.get("from_phase") != projected_phase:
                    raise ReceiptConflictError("Receipt phase chain is inconsistent")
                receipt = StageReceipt.model_validate(commit.get("receipt"))
                if receipt.run_id != self.run_id:
                    raise ReceiptConflictError("Receipt ownership mismatch")
                if receipt.semantic_digest() != commit.get("semantic_digest"):
                    raise ReceiptConflictError("Receipt semantic digest mismatch")
                if commit.get("commit_digest") != _receipt_commit_digest(
                    revision,
                    str(commit.get("from_phase")),
                    str(commit.get("to_phase")),
                    str(commit.get("semantic_digest")),
                ):
                    raise ReceiptConflictError("Receipt commit digest mismatch")
                artifact_ids = list(receipt.accepted_output_artifact_ids)
                if receipt.validation_artifact_id:
                    artifact_ids.append(receipt.validation_artifact_id)
                for artifact_id in artifact_ids:
                    self.artifact_record(artifact_id)
                projected_phase = str(commit.get("to_phase"))
                projected_revision = revision
                last_receipt_id = receipt.receipt_id

            state.update(
                {
                    "phase": projected_phase,
                    "stage": projected_phase if projected_revision else "registered",
                    "status": _status_for_phase(projected_phase)
                    if projected_revision
                    else "queued",
                    "revision": projected_revision,
                    "last_receipt_id": last_receipt_id,
                    "recovery_cursor": last_receipt_id,
                    "progress": _progress_for_phase(projected_phase)
                    if projected_revision
                    else 0,
                    "updated_at": utc_now(),
                }
            )
            self._write_json(self.root / "state.json", state)
            return state

    def write_source_json(self, source_id: str, name: str, value: Any) -> None:
        if not re.fullmatch(r"(?:src_[A-Za-z0-9_-]+|SRC[1-9][0-9]*)", source_id):
            raise ValueError("Invalid source id")
        directory = self.root / "sources" / source_id
        directory.mkdir(parents=True, exist_ok=True)
        self._write_json(directory / name, value)

    def write_result(self, value: Any) -> None:
        self._write_json(self.root / "result.json", value)

    def read_result(self) -> dict[str, Any] | None:
        path = self.root / "result.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def read_events(self, after: int = 0, limit: int = 200) -> tuple[list[dict[str, Any]], int]:
        if not self.audit_path.exists():
            return [], after
        events: list[dict[str, Any]] = []
        next_offset = after
        with self.audit_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index < after:
                    continue
                if len(events) >= limit:
                    break
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                next_offset = index + 1
        return events, next_offset

    def write_capability_record(self, grant_id: str, value: dict[str, Any]) -> None:
        """Persist a redacted capability lifecycle record, never the bearer token."""

        if not re.fullmatch(r"cap_[A-Za-z0-9_-]+", grant_id):
            raise ValueError("Invalid Capability grant ID")
        clean = sanitize(value)
        serialized = json.dumps(clean, ensure_ascii=False).casefold()
        if "capability_token" in serialized or "bearer " in serialized:
            raise ValueError("Capability records cannot contain bearer credentials")
        self._write_json(self.root / "capabilities" / f"{grant_id}.json", clean)

    def read_capability_record(self, grant_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"cap_[A-Za-z0-9_-]+", grant_id):
            raise FileNotFoundError("Invalid Capability grant ID")
        path = self.root / "capabilities" / f"{grant_id}.json"
        if not path.exists():
            raise FileNotFoundError(grant_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=jsonable),
            encoding="utf-8",
        )
        temporary.replace(path)


def _status_for_phase(phase: str) -> str:
    if phase in {"awaiting_source_selection", "awaiting_scope_selection"}:
        return "waiting_for_input"
    if phase in {"references_ready", "tasks_ready"}:
        return "paused"
    if phase in {"completed", "degraded", "failed", "cancelled"}:
        return phase
    return "running"


def _receipt_commit_digest(
    revision: int,
    from_phase: str,
    to_phase: str,
    semantic_digest: str,
) -> str:
    value = f"{revision}\0{from_phase}\0{to_phase}\0{semantic_digest}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _progress_for_phase(phase: str) -> int:
    progress = {
        "created": 0,
        "discovering_sources": 1,
        "awaiting_source_selection": 3,
        "source_selected": 4,
        "acquiring": 5,
        "previewing_scope": 7,
        "awaiting_scope_selection": 9,
        "scope_selected": 10,
        "normalizing": 15,
        "source_review": 25,
        "corpus_frozen": 35,
        "reference_building": 45,
        "references_ready": 50,
        "discovering_task_trajectories": 60,
        "discovering_knowledge": 52,
        "tasks_ready": 55,
        "aligning_knowledge": 58,
        "knowledge_frozen": 65,
        "inferring_trajectories": 75,
        "inducing_skills": 85,
        "validating": 92,
        "committing": 97,
        "completed": 100,
        "degraded": 100,
        "failed": 100,
        "cancelled": 100,
    }
    return progress.get(phase, 0)


def list_run_ids(data_dir: Path) -> list[str]:
    root = data_dir / "runs"
    if not root.exists():
        return []
    directories = [item for item in root.iterdir() if item.is_dir() and item.name.startswith("run_")]
    directories.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return [item.name for item in directories]
