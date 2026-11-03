from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from dataclasses import replace
from typing import Any, TypedDict

from ..acquisition import AcquisitionService
from ..capabilities import CapabilityGateway, CapabilityGatewayServer
from ..config import Settings, normalize_output_language
from ..generation_coverage import saved_generation_coverage_state, settings_for_generation_coverage
from ..contracts import StageReceipt
from ..knowledge import (
    SkillRegistry,
    build_pipeline_quality_report,
    load_corpus_snapshot,
    load_knowledge_snapshot,
)
from ..model_errors import NoUsableSourcesError, public_pipeline_error
from ..models import (
    CandidateValidation,
    ModelRuntimeConfig,
    ReferenceCandidate,
    SkillCandidate,
    SourceManifest,
    SourceResult,
    StagedInput,
    TargetProfile,
    TrajectoryCandidate,
)
from ..runtime import AgentRuntime, UnavailableAgentRuntime
from ..storage import RunStore
from .source_stage import (
    PreparedSource,
    SourceReviewService,
    build_corpus_snapshot,
    finalize_deterministic_source,
    load_persisted_manifests,
    load_persisted_source_results,
    normalize_source,
    validate_source_results,
)
from .html_guided_stage import normalize_guided_html_source
from .reference_stage import REFERENCE_BUILD_PROMPT_VERSION, ReferenceBuildService
from .unified_stage import UNIFIED_PROMPT_VERSION, UnifiedTaskTrajectoryService
from .downstream_stage import (
    SkillInductionService,
)
from .state import PipelineStateMachine, RunPhase, TERMINAL_PHASES
from .source_discovery_stage import SourceDiscoveryService
from .source_scope_stage import (
    build_outline_preview, project_reviewed_scope, receipt_artifact, scope_payload, selected_scope, validate_selection,
)


class PipelineState(TypedDict, total=False):
    run_id: str
    status: str
    stage: str
    target_profile: dict[str, Any]
    inputs: list[dict[str, Any]]
    model_config: dict[str, Any]
    phase: str
    revision: int
    root_session_id: str
    last_receipt_id: str | None
    corpus_snapshot_artifact_id: str


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _receipt(
    *,
    run_id: str,
    stage: str,
    fingerprint_value: Any,
    artifact_ids: tuple[str, ...] = (),
    validation_artifact_id: str | None = None,
    status: str = "succeeded",
    counts: dict[str, int] | None = None,
    gaps: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    attempt_id: str | None = None,
) -> StageReceipt:
    fingerprint = _fingerprint(fingerprint_value)
    semantic = _fingerprint(
        {
            "stage": stage,
            "fingerprint": fingerprint,
            "artifacts": artifact_ids,
            "validation": validation_artifact_id,
            "status": status,
        }
    )
    return StageReceipt(
        receipt_id=f"rcpt_{semantic[:24]}",
        idempotency_key=f"t5.{stage}.{fingerprint[:32]}",
        run_id=run_id,
        stage=stage,
        status=status,  # type: ignore[arg-type]
        attempt_id=attempt_id,
        input_fingerprint=fingerprint,
        accepted_output_artifact_ids=artifact_ids,
        validation_artifact_id=validation_artifact_id,
        counts=counts or {},
        gaps=gaps[:128],
        warnings=warnings[:128],
    )


class PipelineRuntime:
    """Receipt-gated PDF/HTML pipeline from acquisition through Skill export."""

    def __init__(
        self,
        settings: Settings,
        agent_runtime: AgentRuntime | None = None,
        *,
        capability_gateway: CapabilityGateway | None = None,
        capability_server: CapabilityGatewayServer | None = None,
    ) -> None:
        self.settings = settings
        self.agent_runtime = agent_runtime or UnavailableAgentRuntime()
        self.capability_gateway = capability_gateway or CapabilityGateway()
        self.capability_server = capability_server or CapabilityGatewayServer(
            self.capability_gateway,
        )
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self._stores: dict[str, RunStore] = {}
        self._stores_lock = threading.Lock()
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._tasks_lock = threading.Lock()
        self._source_action_lock = threading.RLock()
        self._model_configs: dict[str, ModelRuntimeConfig] = {}

    def create_store(self, run_id: str) -> RunStore:
        with self._stores_lock:
            store = self._stores.get(run_id)
            if store is None:
                store = RunStore(
                    self.settings.data_dir,
                    run_id,
                    max_artifact_bytes=self.settings.max_source_bytes,
                )
                self._stores[run_id] = store
            return store

    def get_store(self, run_id: str) -> RunStore:
        return self.create_store(run_id)

    def start(
        self,
        run_id: str,
        target_profile: TargetProfile,
        inputs: list[StagedInput],
        model_config: ModelRuntimeConfig,
    ) -> None:
        self._model_configs[run_id] = model_config
        task = asyncio.create_task(self.execute(run_id, target_profile, inputs, model_config))
        self._track_task(run_id, task)

    def start_reference_from_frozen(
        self,
        run_id: str,
        target_profile: TargetProfile,
        model_config: ModelRuntimeConfig,
    ) -> None:
        """Run all post-T5 stages for a Run holding an imported T5 snapshot.

        The method name is retained for API compatibility with the original
        Reference-only cache action. It no longer pauses after Reference Build.
        """

        task = asyncio.create_task(
            self.execute(
                run_id,
                target_profile,
                [],
                model_config,
                from_frozen_t5=True,
            )
        )
        self._track_task(run_id, task)

    def _track_task(self, run_id: str, task: asyncio.Task[Any]) -> None:
        with self._tasks_lock:
            self._tasks[run_id] = task

        def cleanup(done: asyncio.Task[Any]) -> None:
            with self._tasks_lock:
                if self._tasks.get(run_id) is done:
                    self._tasks.pop(run_id, None)
            try:
                done.result()
            except (asyncio.CancelledError, Exception):
                pass
            if self.get_store(run_id).read_state().get("phase") in {
                phase.value for phase in TERMINAL_PHASES
            }:
                self._model_configs.pop(run_id, None)

        task.add_done_callback(cleanup)

    def has_active_task(self, run_id: str) -> bool:
        """Report in-memory execution, including runs resumed from a T5 cache."""

        with self._tasks_lock:
            task = self._tasks.get(run_id)
        return bool(task is not None and not task.done())

    def has_model_credentials(self, run_id: str) -> bool:
        model = self._model_configs.get(run_id)
        return bool(model and model.api_key)

    def source_discovery_payload(self, run_id: str) -> dict[str, Any] | None:
        store = self.get_store(run_id)
        artifact_id = store.read_state().get("source_discovery_artifact_id")
        if not artifact_id:
            # Receipt outputs remain authoritative if a process stops between writes.
            for commit in reversed(store.list_receipt_commits()):
                receipt = commit.get("receipt", {})
                if receipt.get("stage") == "source_discovery":
                    outputs = receipt.get("accepted_output_artifact_ids", [])
                    artifact_id = outputs[0] if outputs else None
                    break
        return store.read_json_artifact(str(artifact_id)) if artifact_id else None

    def _source_action_model(self, run_id: str, api_key: str | None) -> ModelRuntimeConfig:
        model = self._model_configs.get(run_id)
        if model is None:
            public = dict(self.get_store(run_id).read_state().get("model") or {})
            public.pop("api_key_configured", None)
            model = ModelRuntimeConfig.model_validate(public)
        if api_key and api_key.strip():
            model = model.model_copy(update={"api_key": api_key.strip()})
        if not model.api_key:
            raise ValueError("Model API Key is required to continue this Run after a server restart")
        return model

    def source_scope_payload(self, run_id: str) -> dict[str, Any] | None:
        payload = scope_payload(self.get_store(run_id))
        if payload is not None:
            payload["selection_enabled"] = self.settings.scope_selection_enabled
        return payload

    def recover_scope_state(self, run_id: str) -> None:
        """Repair a stopped scope checkpoint whose receipt preceded its state write."""
        with self._source_action_lock:
            if self.has_active_task(run_id):
                return
            store = self.get_store(run_id)
            state = store.read_state()
            if not state.get("scope_selection_enabled"):
                return
            commits = store.list_receipt_commits()
            if commits and (commits[-1]["revision"] != state.get("revision")
                            or commits[-1]["to_phase"] != state.get("phase")):
                PipelineStateMachine(store).recover()

    def select_scope(self, run_id: str, outline_artifact_id: str, mode: str,
                     section_ids: list[str], api_key: str | None = None) -> dict[str, Any]:
        with self._source_action_lock:
            store = self.get_store(run_id)
            self.recover_scope_state(run_id)
            phase = PipelineStateMachine(store).phase
            outline = self.source_scope_payload(run_id)
            if not outline or outline["outline_artifact_id"] != outline_artifact_id:
                raise RuntimeError("This outline preview is stale or does not belong to this Run")
            previous = selected_scope(store)
            replaying = (phase is not RunPhase.AWAITING_SCOPE_SELECTION and previous
                         and previous["outline_artifact_id"] == outline_artifact_id
                         and previous["mode"] == mode
                         and len(section_ids) == len(set(section_ids))
                         and set(previous["section_ids"]) == set(section_ids))
            # A new choice is first-level only. Exact replays of historical
            # committed scopes must not silently broaden or invalidate a Run.
            ordered_ids = previous["section_ids"] if replaying else validate_selection(outline, mode, section_ids)
            selection = {"schema_version": "source-scope-selection.v1",
                         "outline_artifact_id": outline_artifact_id,
                         "source_id": outline["source_id"], "mode": mode, "section_ids": ordered_ids}
            if (not self.settings.scope_selection_enabled and mode == "selected"
                    and previous != selection):
                raise RuntimeError("Chapter selection is disabled; continue with the whole material")
            if phase is not RunPhase.AWAITING_SCOPE_SELECTION:
                if phase in TERMINAL_PHASES or previous != selection:
                    raise RuntimeError("This Run is not waiting for chapter selection")
                if phase is RunPhase.SCOPE_SELECTED and not self.has_active_task(run_id):
                    model = self._source_action_model(run_id, api_key)
                    self.start(run_id, TargetProfile.model_validate(store.read_state()["target_profile"]), [], model)
                return {"run_id": run_id, "status": store.read_state()["status"], "already_selected": True}
            if self.has_active_task(run_id):
                raise RuntimeError("The outline preview is still finishing; please retry")
            model = self._source_action_model(run_id, api_key)
            # Check that the preview still describes the persisted acquired bytes.
            manifests = load_persisted_manifests(store)
            manifest = next((item for item in manifests if item.source_id == outline["source_id"]), None)
            if (manifest is None or manifest.raw_artifact_id != outline["raw_artifact_id"]
                    or hashlib.sha256(store.artifact_path(manifest.raw_artifact_id).read_bytes()).hexdigest()
                    != outline["raw_sha256"]):
                raise RuntimeError("The source changed after the outline preview; start a new Run")
            artifact = store.put_json_artifact(selection, kind="source_scope_selection", source_id=manifest.source_id)
            PipelineStateMachine(store).advance(
                RunPhase.SCOPE_SELECTED,
                _receipt(run_id=run_id, stage="scope_selection", fingerprint_value=selection,
                         artifact_ids=(artifact.artifact_id,), counts={"selected_chapters": len(ordered_ids)}),
            )
            store.update_state(scope_selection_artifact_id=artifact.artifact_id)
            store.emit("source_scope_selected", {"mode": mode, "section_ids": ordered_ids,
                       "outline_artifact_id": outline_artifact_id}, agent="source_scope")
            self.start(run_id, TargetProfile.model_validate(store.read_state()["target_profile"]), [], model)
            return {"run_id": run_id, "status": "running"}

    def select_source(self, run_id: str, candidate_id: str,
                      api_key: str | None = None) -> dict[str, Any]:
        with self._source_action_lock:
            store = self.get_store(run_id)
            state = store.read_state()
            phase = RunPhase(state["phase"])
            if state.get("selected_candidate_id") and phase not in {
                RunPhase.AWAITING_SOURCE_SELECTION, RunPhase.DISCOVERING_SOURCES,
                RunPhase.CANCELLED, RunPhase.FAILED,
            }:
                if state["selected_candidate_id"] != candidate_id:
                    raise RuntimeError("A different source has already been selected")
                if phase is RunPhase.SOURCE_SELECTED and not self.has_active_task(run_id):
                    model = self._source_action_model(run_id, api_key)
                    selected = store.read_json_artifact(state["selected_source_artifact_id"])
                    self.start(run_id, TargetProfile.model_validate(state["target_profile"]),
                               [StagedInput.model_validate(selected["input"])], model)
                return {"run_id": run_id, "status": store.read_state()["status"], "already_selected": True}
            if phase is not RunPhase.AWAITING_SOURCE_SELECTION:
                raise RuntimeError("This Run is not waiting for source selection")
            if self.has_active_task(run_id):
                raise RuntimeError("The search task is still finishing; please retry")
            payload = self.source_discovery_payload(run_id) or {}
            candidate = next((item for item in payload.get("candidates", [])
                              if item.get("candidate_id") == candidate_id), None)
            if candidate is None:
                raise ValueError("Unknown source candidate")
            model = self._source_action_model(run_id, api_key)
            staged = StagedInput(
                input_id="IN1", kind="url", display_name=candidate["title"],
                url=candidate["url"], user_declared_version=candidate["document_version"],
                source_origin="websearch", source_priority="user_selected",
                source_metadata={**candidate, "discovery_candidate_id": candidate_id,
                                 "selected_by_user": True, "target_version": state["target_profile"]["target_version"]},
            )
            selected = store.put_json_artifact(
                {"schema_version": "selected-source.v1", "candidate": candidate,
                 "input": staged.model_dump(mode="json")}, kind="selected_source",
            )
            store.update_state(selected_candidate_id=candidate_id,
                               selected_source_artifact_id=selected.artifact_id, input_count=1)
            PipelineStateMachine(store).advance(
                RunPhase.SOURCE_SELECTED,
                _receipt(run_id=run_id, stage="source_selection",
                         fingerprint_value={"candidate": candidate_id,
                                            "artifact": state.get("source_discovery_artifact_id")},
                         artifact_ids=(selected.artifact_id,)),
            )
            store.emit("source_selected", {"candidate_id": candidate_id,
                       "url": staged.url, "modality": candidate["modality"]}, agent="source_discovery")
            self.start(run_id, TargetProfile.model_validate(state["target_profile"]), [staged], model)
            return {"run_id": run_id, "status": "running"}

    def search_again(self, run_id: str, api_key: str | None = None) -> dict[str, Any]:
        with self._source_action_lock:
            store = self.get_store(run_id)
            machine = PipelineStateMachine(store)
            if machine.phase is not RunPhase.AWAITING_SOURCE_SELECTION or self.has_active_task(run_id):
                raise RuntimeError("This Run is not available for another search")
            model = self._source_action_model(run_id, api_key)
            state = store.read_state()
            machine.advance(
                RunPhase.DISCOVERING_SOURCES,
                _receipt(run_id=run_id, stage="source_discovery_retry",
                         fingerprint_value={"run_id": run_id, "revision": state["revision"]}),
            )
            self.start(run_id, TargetProfile.model_validate(state["target_profile"]), [], model)
            return {"run_id": run_id, "status": "running"}

    def _cancel_running_child_sessions(self, store: RunStore) -> None:
        session_ids = tuple(
            dict.fromkeys(
                attempt.child_session_id
                for attempt in store.list_attempts()
                if attempt.status == "running"
            )
        )
        if not session_ids:
            return

        async def cancel_sessions() -> None:
            outcomes: dict[str, bool] = {}
            for session_id in session_ids:
                try:
                    outcomes[session_id] = bool(
                        await self.agent_runtime.cancel(session_id)
                    )
                except Exception as exc:
                    outcomes[session_id] = False
                    store.emit(
                        "child_session_cancel_failed",
                        {"child_session_id": session_id, "error": str(exc)[:500]},
                        level="warning",
                    )
            store.emit(
                "child_session_cancel_dispatched",
                {"sessions": outcomes},
                level="warning",
            )

        cancellation = asyncio.create_task(cancel_sessions())

        def consume_result(done: asyncio.Task[None]) -> None:
            try:
                done.result()
            except (asyncio.CancelledError, Exception):
                pass

        cancellation.add_done_callback(consume_result)

    def import_frozen_corpus(
        self,
        *,
        run_id: str,
        source_store: RunStore,
        corpus_artifact_id: str,
    ) -> CorpusSnapshot:
        """Reuse T5 inputs, revalidating complete immutable source corpora when present."""

        store = self.get_store(run_id)
        machine = PipelineStateMachine(store)
        if machine.phase is not RunPhase.CREATED:
            raise RuntimeError("A frozen CorpusSnapshot can only be imported into a new Run")
        corpus = load_corpus_snapshot(source_store, corpus_artifact_id)
        self._verify_cached_artifact(source_store, corpus_artifact_id)
        dependency_ids = self._corpus_dependency_artifact_ids(corpus)
        restored_sources: dict[str, SourceResult] = {}
        source_parts: dict[str, list[dict[str, Any]]] = {}
        for source in corpus.source_results:
            failed_validation = source.status == "failed" and "branch_validation_failed" in source.gaps
            if (source.status not in {"succeeded", "degraded"} and not failed_validation) or not source.corpus_artifact_id:
                continue
            record = source_store.artifact_record(source.corpus_artifact_id)
            if record.kind != "normalized_source_corpus":
                continue  # Older snapshot-only caches retain their original behavior.
            self._verify_cached_artifact(source_store, source.corpus_artifact_id)
            normalized = source_store.read_json_artifact(source.corpus_artifact_id)
            if (
                not isinstance(normalized, dict)
                or record.source_id != source.source_id
                or normalized.get("source_id") != source.source_id
                or not isinstance(normalized.get("parts"), list)
                or not isinstance(normalized.get("evidence"), list)
                or not isinstance(normalized.get("manifest"), dict)
            ):
                raise ValueError(f"Source {source.source_id} has an invalid normalized corpus cache")
            review_status = normalized["manifest"].get("status")
            if failed_validation and review_status not in {"succeeded", "degraded"}:
                continue  # A failed Source review must not be promoted by cache import.
            restored_payload = source.model_dump(mode="json")
            # Preserve the accepted Source review prose and semantic gaps. Only
            # deterministic validation diagnostics are regenerated in the new Run.
            gaps = [gap for gap in source.gaps
                    if not gap.startswith("evidence_validation_rejected:") and gap != "branch_validation_failed"]
            gaps = list(dict.fromkeys([*normalized.get("gaps", []), *gaps]))
            restored_payload.update(
                evidence=normalized["evidence"],
                evidence_tree=normalized.get("evidence_tree"),
                coverage=normalized.get("coverage", {}),
                manifest=normalized["manifest"],
                source_part_count=len(normalized["parts"]),
                normalization_version=normalized.get("schema_version", source.normalization_version),
                gaps=gaps,
                warnings=list(dict.fromkeys([*normalized.get("warnings", []), *source.warnings])),
                errors=[],
                status="degraded" if gaps or review_status == "degraded" else "succeeded",
            )
            restored = SourceResult.model_validate(restored_payload)
            if restored.manifest.source_id != source.source_id:
                raise ValueError(f"Source {source.source_id} has a mismatched cached manifest")
            restored_sources[source.source_id] = restored
            source_parts[source.source_id] = normalized["parts"]
            dependency_ids.add(source.corpus_artifact_id)
            dependency_ids.update(self._cached_source_artifact_ids(normalized))
            dependency_ids.update(self._cached_source_artifact_ids(restored_payload))

        # Verify every file before creating any destination artifacts. Never repair
        # the old Run's mutable projections or overwrite its frozen checkpoint.
        for artifact_id in sorted(dependency_ids):
            self._verify_cached_artifact(source_store, artifact_id)
        for artifact_id in sorted(dependency_ids):
            store.import_artifact(source_store, artifact_id)
        imported_snapshot = store.import_artifact(source_store, corpus_artifact_id)
        original_evidence_count = corpus.evidence_count
        original_corpus_artifact_id = corpus_artifact_id
        validation_artifact_id = None
        if restored_sources:
            for source_id, source in restored_sources.items():
                store.write_source_json(source_id, "parts.json", source_parts[source_id])
                store.write_source_json(source_id, "manifest.json", source.manifest.model_dump(mode="json"))
                store.write_source_json(source_id, "result.json", source.model_dump(mode="json"))
            validated, summary = validate_source_results(store, restored_sources.values())
            restored_sources = {item.source_id: item for item in validated}
            results = [restored_sources.get(item.source_id, item) for item in corpus.source_results]
            validation_artifact = store.put_json_artifact(
                {"schema_version": "source-validation.v1", "sources": summary},
                kind="source_validation",
            )
            validation_artifact_id = validation_artifact.artifact_id
            corpus, corpus_artifact_id = build_corpus_snapshot(store, corpus.target_profile, results)
            imported_snapshot = store.artifact_record(corpus_artifact_id)
        machine.advance(
            RunPhase.CORPUS_FROZEN,
            _receipt(
                run_id=run_id,
                stage="corpus_cache_import",
                fingerprint_value={
                    "source_run_id": source_store.run_id,
                    "original_corpus_snapshot_artifact_id": original_corpus_artifact_id,
                    "corpus_snapshot_id": corpus.corpus_snapshot_id,
                    "corpus_snapshot_artifact_id": corpus_artifact_id,
                    "corpus_snapshot_sha256": imported_snapshot.sha256,
                },
                artifact_ids=(corpus_artifact_id,),
                validation_artifact_id=validation_artifact_id,
                status="degraded" if any(item.status in {"degraded", "failed"} for item in corpus.source_results) else "succeeded",
                counts={
                    "sources": len(corpus.source_results),
                    "evidence": corpus.evidence_count,
                    "parts": corpus.source_part_count,
                    "dependencies": len(dependency_ids),
                },
            ),
        )
        snapshot_payload = corpus.model_dump(mode="json")
        store.update_state(
            corpus_snapshot_id=corpus.corpus_snapshot_id,
            corpus_snapshot_artifact_id=corpus_artifact_id,
            evidence_count=corpus.evidence_count,
            source_part_count=corpus.source_part_count,
            source_status_counts=corpus.source_status_counts,
            t5_stage_complete=True,
            t5_cache_hit=True,
            t5_cache_source_run_id=source_store.run_id,
            t5_cache_dependency_count=len(dependency_ids),
            t5_cache_revalidated=bool(restored_sources),
            t5_cache_revalidated_source_ids=list(restored_sources),
            t5_cache_original_corpus_snapshot_artifact_id=original_corpus_artifact_id,
            t5_cache_restored_evidence_count=max(0, corpus.evidence_count - original_evidence_count),
            **({"source_validation_artifact_id": validation_artifact_id} if validation_artifact_id else {}),
        )
        store.write_result(
            {
                "schema_version": "t5-result.v1",
                "outcome": "corpus_frozen",
                "snapshot": snapshot_payload,
                "corpus_snapshot_artifact_id": corpus_artifact_id,
                "next_stage": "reference_build",
                "cache": {
                    "hit": True,
                    "source_run_id": source_store.run_id,
                    "revalidated": bool(restored_sources),
                    "original_corpus_snapshot_artifact_id": original_corpus_artifact_id,
                    "corpus_snapshot_id": corpus.corpus_snapshot_id,
                    "corpus_snapshot_sha256": imported_snapshot.sha256,
                },
            }
        )
        store.emit(
            "corpus_snapshot_cache_imported",
            {
                "source_run_id": source_store.run_id,
                "corpus_snapshot_id": corpus.corpus_snapshot_id,
                "corpus_snapshot_artifact_id": corpus_artifact_id,
                "dependency_count": len(dependency_ids),
                "revalidated": bool(restored_sources),
                "original_corpus_snapshot_artifact_id": original_corpus_artifact_id,
                "original_evidence_count": original_evidence_count,
                "evidence_count": corpus.evidence_count,
                "restored_evidence_count": max(0, corpus.evidence_count - original_evidence_count),
            },
        )
        return corpus

    @staticmethod
    def _verify_cached_artifact(store: RunStore, artifact_id: str) -> None:
        record = store.artifact_record(artifact_id)
        data = store.artifact_path(artifact_id).read_bytes()
        if len(data) != record.byte_count or hashlib.sha256(data).hexdigest() != record.sha256:
            raise ValueError(f"Cached Artifact {artifact_id} bytes do not match its immutable record")

    @staticmethod
    def _cached_source_artifact_ids(value: Any) -> set[str]:
        """Collect declared Artifact references, excluding IDs quoted in source prose."""

        artifact_ids: set[str] = set()

        def walk(item: Any, *, artifact_field: bool = False) -> None:
            if isinstance(item, str):
                if artifact_field and re.fullmatch(r"art_[0-9a-f]{24}", item):
                    artifact_ids.add(item)
            elif isinstance(item, list):
                for child in item:
                    walk(child, artifact_field=artifact_field)
            elif isinstance(item, dict):
                for key, child in item.items():
                    walk(child, artifact_field=(key == "artifact_id" or key.endswith("_artifact_id")
                                                or key == "artifact_ids" or key.endswith("_artifact_ids")))

        walk(value)
        return artifact_ids

    async def execute(
        self,
        run_id: str,
        target_profile: TargetProfile,
        inputs: list[StagedInput],
        model_config: ModelRuntimeConfig,
        *,
        from_frozen_t5: bool = False,
    ) -> dict[str, Any]:
        store = self.get_store(run_id)
        machine = PipelineStateMachine(store)
        if (not self.settings.scope_selection_enabled
                and machine.phase in {RunPhase.CREATED, RunPhase.SOURCE_SELECTED, RunPhase.ACQUIRING}
                and store.read_state().get("scope_selection_enabled")):
            # Process policy wins over caller-supplied per-run flags. Existing
            # committed selections are resumed through their immutable receipts.
            store.update_state(scope_selection_enabled=False)
        self._model_configs[run_id] = model_config
        output_language = normalize_output_language(
            str(store.read_state().get("output_language") or self.settings.skill_output_language)
        )
        generation_state = saved_generation_coverage_state(self.settings, store.read_state())
        store.update_state(output_language=output_language, **generation_state)
        run_settings = replace(
            settings_for_generation_coverage(self.settings, generation_state),
            skill_output_language=output_language,
        )
        store.emit(
            "run_started",
            {
                "target_profile": target_profile.model_dump(mode="json"),
                "input_count": len(inputs),
                "model": model_config.public_dict(),
                "migration_stage": "T7",
                **generation_state,
                "execution_mode": (
                    "full_pipeline_from_frozen_t5"
                    if from_frozen_t5
                    else "full_pipeline"
                ),
            },
        )
        try:
            manifests: list[SourceManifest]
            initial_results: list[SourceResult]
            if machine.phase is RunPhase.CREATED and store.read_state().get("source_mode") == "discovery":
                machine.advance(
                    RunPhase.DISCOVERING_SOURCES,
                    _receipt(run_id=run_id, stage="source_discovery_dispatch",
                             fingerprint_value={"target": target_profile.model_dump(mode="json"),
                                                "output_language": output_language}),
                )
            if machine.phase is RunPhase.DISCOVERING_SOURCES:
                discovered = await SourceDiscoveryService(
                    store=store, settings=run_settings, runtime=self.agent_runtime,
                    gateway=self.capability_gateway, gateway_server=self.capability_server,
                    model_config=model_config,
                ).discover(target_profile=target_profile, output_language=output_language)
                store.update_state(source_discovery_artifact_id=discovered.candidate_artifact_id,
                                   selected_candidate_id=None, selected_source_artifact_id=None)
                machine.advance(
                    RunPhase.AWAITING_SOURCE_SELECTION,
                    _receipt(run_id=run_id, stage="source_discovery",
                             fingerprint_value={"artifact": discovered.candidate_artifact_id,
                                                "attempt_id": discovered.attempt_id},
                             artifact_ids=(discovered.candidate_artifact_id,),
                             attempt_id=discovered.attempt_id,
                             counts={"candidates": len(discovered.candidates)}),
                )
                store.emit("source_selection_required", {"candidate_count": len(discovered.candidates)},
                           agent="source_discovery")
                return {"run_id": run_id, "status": "waiting_for_input"}
            if machine.phase is RunPhase.AWAITING_SOURCE_SELECTION:
                return {"run_id": run_id, "status": "waiting_for_input"}
            if machine.phase in {RunPhase.CREATED, RunPhase.SOURCE_SELECTED}:
                if machine.phase is RunPhase.SOURCE_SELECTED and not inputs:
                    selected = store.read_json_artifact(store.read_state()["selected_source_artifact_id"])
                    inputs = [StagedInput.model_validate(selected["input"])]
                outcome = await asyncio.to_thread(
                    AcquisitionService(store, run_settings).acquire,
                    inputs,
                    target_profile,
                )
                manifests = outcome.manifests
                initial_results = outcome.initial_results
                active = self._active_manifests(manifests)
                if not active:
                    raise self._no_usable_sources(inputs, manifests)
                acquisition_artifact = store.put_json_artifact(
                    {
                        "schema_version": "acquisition-outcome.v1",
                        "manifests": [item.model_dump(mode="json") for item in manifests],
                        "initial_results": [
                            item.model_dump(mode="json") for item in initial_results
                        ],
                    },
                    kind="acquisition_outcome",
                )
                counts = {
                    "pdf": sum(item.primary_modality == "pdf" for item in active),
                    "html": sum(item.primary_modality == "html" for item in active),
                    "terminal": len(initial_results),
                }
                machine.advance(
                    RunPhase.ACQUIRING,
                    _receipt(
                        run_id=run_id,
                        stage="acquisition",
                        fingerprint_value={
                            "inputs": [item.model_dump(mode="json") for item in inputs],
                            "target": target_profile.model_dump(mode="json"),
                        },
                        artifact_ids=(acquisition_artifact.artifact_id,),
                        counts=counts,
                    ),
                )
                store.update_state(
                    acquisition_artifact_id=acquisition_artifact.artifact_id,
                    source_counts=counts,
                    canonical_source_count=len(manifests),
                )
            else:
                manifests = load_persisted_manifests(store)
                initial_results = [
                    item
                    for item in load_persisted_source_results(store)
                    if item.status in {"excluded", "failed"}
                ]

            if machine.phase is RunPhase.AWAITING_SCOPE_SELECTION:
                return {"run_id": run_id, "status": "waiting_for_input"}
            selection = selected_scope(store)
            reviewed_checkpoint = receipt_artifact(store, "source_review", "validated_source_batch_artifact_id")
            if machine.phase is RunPhase.SCOPE_SELECTED and not selection:
                raise RuntimeError("The committed chapter selection is missing")
            # Old v1 checkpoints resume through full parsing, then ask again from
            # reviewed Evidence. Their guessed anchors are never applied to text.
            legacy_scope = machine.phase in {RunPhase.SCOPE_SELECTED, RunPhase.PREVIEWING_SCOPE} and not reviewed_checkpoint
            if legacy_scope:
                # Even if the feature was disabled after this old checkpoint,
                # never silently turn a committed chapter choice into full scope.
                run_settings = replace(run_settings, scope_selection_enabled=True)
                for manifest in manifests:
                    manifest.source_metadata.pop("extraction_scope", None)

            prepared: list[PreparedSource] = []
            if machine.phase in {RunPhase.ACQUIRING, RunPhase.NORMALIZING} or legacy_scope:
                acquisition = AcquisitionService(store, run_settings)
                persisted_ids = {
                    item.source_id for item in load_persisted_source_results(store)
                }
                active = [
                    item
                    for item in self._active_manifests(manifests)
                    if item.source_id not in persisted_ids
                ]
                for manifest in active:
                    if run_settings.html_processing_mode not in {"legacy", "model_guided"}:
                        raise ValueError("PREPROCESS_HTML_PROCESSING_MODE must be 'legacy' or 'model_guided'")
                    if manifest.primary_modality == "html" and run_settings.html_processing_mode == "model_guided":
                        prepared.append(await normalize_guided_html_source(
                            store, run_settings, acquisition, target_profile, manifest,
                            runtime=self.agent_runtime, model_config=model_config,
                        ))
                    else:
                        prepared.append(await asyncio.to_thread(
                            normalize_source,
                            store,
                            run_settings,
                            acquisition,
                            target_profile,
                            manifest,
                        ))
                normalization_artifact = store.put_json_artifact(
                    {
                        "schema_version": "normalization-batch.v1",
                        "sources": [
                            {
                                "source_id": item.manifest.source_id,
                                "modality": item.manifest.primary_modality,
                                "normalization_artifact_id": item.normalization_artifact_id,
                                "part_count": len(item.workspace.parts),
                            }
                            for item in prepared
                        ],
                    },
                    kind="normalization_batch",
                )
                if machine.phase in {RunPhase.ACQUIRING, RunPhase.SCOPE_SELECTED, RunPhase.PREVIEWING_SCOPE}:
                    machine.advance(
                        RunPhase.NORMALIZING,
                        _receipt(
                            run_id=run_id,
                            stage="normalization",
                            fingerprint_value={
                                "sources": [
                                    {
                                        "source_id": item.manifest.source_id,
                                        "raw_artifact_id": item.manifest.raw_artifact_id,
                                        "extraction_scope": item.manifest.source_metadata.get("extraction_scope"),
                                    }
                                    for item in prepared
                                ]
                            },
                            artifact_ids=(normalization_artifact.artifact_id,),
                            counts={
                                "sources": len(prepared),
                                "parts": sum(len(item.workspace.parts) for item in prepared),
                            },
                        ),
                    )
                    store.update_state(
                        normalization_artifact_id=normalization_artifact.artifact_id
                    )

                processing_mode = run_settings.t5_processing_mode
                if processing_mode == "deterministic":
                    reviewed = await asyncio.gather(
                        *(
                            asyncio.to_thread(finalize_deterministic_source, item)
                            for item in prepared
                        )
                    )
                elif processing_mode in {
                    "agent",
                    "agent_architect",
                    "agent_planner",
                    "agent_legacy",
                    "agent_curated",
                }:
                    reviewed = await self._review_prepared_sources(
                        store,
                        prepared,
                        model_config,
                    )
                else:
                    raise ValueError(
                        "PREPROCESS_T5_PROCESSING_MODE must be 'agent', 'agent_architect', "
                        "'agent_planner', 'agent_legacy', 'agent_curated', or "
                        "'deterministic'; "
                        f"received {processing_mode!r}"
                    )
                combined = self._ordered_results(
                    manifests,
                    [*initial_results, *load_persisted_source_results(store), *reviewed],
                )
                validated, validation_summary = validate_source_results(store, combined)
                validation_artifact = store.put_json_artifact(
                    {
                        "schema_version": "source-validation.v1",
                        "sources": validation_summary,
                    },
                    kind="source_validation",
                )
                validated_batch_artifact = store.put_json_artifact(
                    {
                        "schema_version": "validated-source-batch.v1",
                        "source_results": [
                            item.model_dump(mode="json") for item in validated
                        ],
                    },
                    kind="validated_source_batch",
                    metadata={"source_count": len(validated)},
                    preserve_text_integrity=True,
                )
                accepted_outputs = tuple(
                    dict.fromkeys(
                        [
                            *(
                                item.corpus_artifact_id
                                for item in validated
                                if item.corpus_artifact_id
                            ),
                            validated_batch_artifact.artifact_id,
                        ]
                    )
                )
                review_status = (
                    "degraded"
                    if any(item.status in {"degraded", "failed"} for item in validated)
                    else "succeeded"
                )
                machine.advance(
                    RunPhase.SOURCE_REVIEW,
                    _receipt(
                        run_id=run_id,
                        stage="source_review",
                        fingerprint_value={
                            "processing_mode": processing_mode,
                            "sources": [
                                {
                                    "source_id": item.source_id,
                                    "status": item.status,
                                    "corpus_artifact_id": item.corpus_artifact_id,
                                }
                                for item in validated
                            ]
                        },
                        artifact_ids=accepted_outputs,
                        validation_artifact_id=validation_artifact.artifact_id,
                        status=review_status,
                        counts={
                            "sources": len(validated),
                            "evidence": sum(len(item.evidence) for item in validated),
                        },
                        gaps=tuple(gap for item in validated for gap in item.gaps),
                        warnings=tuple(
                            warning for item in validated for warning in item.warnings
                        ),
                    ),
                )
                store.update_state(
                    t5_processing_mode=processing_mode,
                    source_validation_artifact_id=validation_artifact.artifact_id,
                    validated_source_batch_artifact_id=validated_batch_artifact.artifact_id,
                )

            if (machine.phase is RunPhase.SOURCE_REVIEW
                    and run_settings.scope_selection_enabled
                    and store.read_state().get("scope_selection_enabled")):
                machine.advance(
                    RunPhase.PREVIEWING_SCOPE,
                    _receipt(run_id=run_id, stage="scope_preview_dispatch",
                             fingerprint_value={"reviewed_batch": store.read_state().get("validated_source_batch_artifact_id")}),
                )
            if machine.phase is RunPhase.PREVIEWING_SCOPE:
                results = self._load_receipted_source_results(store)
                outline_id = await asyncio.to_thread(build_outline_preview, store, results)
                machine.advance(
                    RunPhase.AWAITING_SCOPE_SELECTION,
                    _receipt(run_id=run_id, stage="scope_preview", fingerprint_value={"outline": outline_id},
                             artifact_ids=(outline_id,)),
                )
                store.update_state(source_outline_artifact_id=outline_id)
                store.emit("source_scope_selection_required", {"outline_artifact_id": outline_id}, agent="source_scope")
                return {"run_id": run_id, "status": "waiting_for_input"}

            if machine.phase in {RunPhase.SOURCE_REVIEW, RunPhase.SCOPE_SELECTED}:
                results = self._load_receipted_source_results(store)
                if machine.phase is RunPhase.SCOPE_SELECTED:
                    results = await asyncio.to_thread(
                        project_reviewed_scope, store, results, selected_scope(store),
                        settings=run_settings, target=target_profile,
                    )
                results, _ = validate_source_results(store, results)
                snapshot, snapshot_artifact_id = build_corpus_snapshot(
                    store, target_profile, results
                )
                snapshot_payload = snapshot.model_dump(mode="json")
                machine.advance(
                    RunPhase.CORPUS_FROZEN,
                    _receipt(
                        run_id=run_id,
                        stage="corpus_freeze",
                        fingerprint_value=snapshot_payload,
                        artifact_ids=(snapshot_artifact_id,),
                        counts={
                            "sources": len(snapshot.source_results),
                            "evidence": snapshot.evidence_count,
                            "parts": snapshot.source_part_count,
                        },
                    ),
                )
                store.update_state(
                    corpus_snapshot_id=snapshot.corpus_snapshot_id,
                    corpus_snapshot_artifact_id=snapshot_artifact_id,
                    evidence_count=snapshot.evidence_count,
                    source_part_count=snapshot.source_part_count,
                    source_status_counts=snapshot.source_status_counts,
                    t5_stage_complete=True,
                )
                store.write_result(
                    {
                        "schema_version": "t5-result.v1",
                        "outcome": "corpus_frozen",
                        "snapshot": snapshot_payload,
                        "corpus_snapshot_artifact_id": snapshot_artifact_id,
                        "next_stage": "task_and_reference",
                    }
                )
                store.emit(
                    "corpus_snapshot_created",
                    {
                        "corpus_snapshot_id": snapshot.corpus_snapshot_id,
                        "corpus_snapshot_artifact_id": snapshot_artifact_id,
                        "evidence_count": snapshot.evidence_count,
                        "source_part_count": snapshot.source_part_count,
                    },
                )

            if (
                run_settings.t6_reference_enabled
                and machine.phase is RunPhase.CORPUS_FROZEN
            ):
                corpus_artifact_id = str(
                    store.read_state().get("corpus_snapshot_artifact_id") or ""
                )
                if not corpus_artifact_id:
                    raise RuntimeError("Reference Build requires a frozen CorpusSnapshot")
                machine.advance(
                    RunPhase.REFERENCE_BUILDING,
                    _receipt(
                        run_id=run_id,
                        stage="reference_build_dispatch",
                        fingerprint_value={
                            "corpus_snapshot_artifact_id": corpus_artifact_id,
                            "prompt_version": REFERENCE_BUILD_PROMPT_VERSION,
                            "reference_formatting_mode": run_settings.reference_formatting_mode,
                            **generation_state,
                        },
                        artifact_ids=(corpus_artifact_id,),
                    ),
                )

            if machine.phase is RunPhase.REFERENCE_BUILDING:
                corpus_artifact_id = str(
                    store.read_state().get("corpus_snapshot_artifact_id") or ""
                )
                corpus = load_corpus_snapshot(store, corpus_artifact_id)
                reference_artifacts = await ReferenceBuildService(
                    store=store,
                    settings=run_settings,
                    runtime=self.agent_runtime,
                    gateway=self.capability_gateway,
                    gateway_server=self.capability_server,
                    model_config=model_config,
                ).build(
                    target_profile=target_profile,
                    corpus=corpus,
                    corpus_artifact_id=corpus_artifact_id,
                )
                rejected = sum(
                    1 for item in reference_artifacts.validation if not item.accepted
                )
                receipt_status = (
                    "degraded"
                    if reference_artifacts.stage_result.get("status") == "degraded" or rejected
                    else "succeeded"
                )
                machine.advance(
                    RunPhase.REFERENCES_READY,
                    _receipt(
                        run_id=run_id,
                        stage="reference_build",
                        fingerprint_value={
                            "corpus_snapshot_id": corpus.corpus_snapshot_id,
                            "validated_reference_artifact_id": (
                                reference_artifacts.validated_artifact_id
                            ),
                            "reference_formatting_mode": run_settings.reference_formatting_mode,
                            "validation_artifact_id": (
                                reference_artifacts.validation_artifact_id
                            ),
                        },
                        artifact_ids=(reference_artifacts.validated_artifact_id,),
                        validation_artifact_id=(
                            reference_artifacts.validation_artifact_id
                        ),
                        status=receipt_status,
                        counts={
                            "candidates": len(reference_artifacts.validation),
                            "references": len(reference_artifacts.references),
                            "rejected": rejected,
                        },
                        gaps=tuple(reference_artifacts.stage_result.get("gaps") or ()),
                        warnings=tuple(
                            [*(reference_artifacts.stage_result.get("warnings") or ())]
                            + (
                                [f"{rejected} Reference candidate(s) failed validation."]
                                if rejected
                                else []
                            )
                        ),
                        attempt_id=reference_artifacts.attempt_id,
                    ),
                )
                snapshot_payload = corpus.model_dump(mode="json")
                references_payload = [
                    item.model_dump(mode="json")
                    for item in reference_artifacts.references
                ]
                validation_payload = [
                    item.model_dump(mode="json")
                    for item in reference_artifacts.validation
                ]
                store.update_state(
                    t6_reference_complete=True,
                    reference_count=len(references_payload),
                    reference_candidate_artifact_id=(
                        reference_artifacts.candidate_artifact_id
                    ),
                    validated_reference_artifact_id=(
                        reference_artifacts.validated_artifact_id
                    ),
                    reference_validation_artifact_id=(
                        reference_artifacts.validation_artifact_id
                    ),
                    reference_stage_result=reference_artifacts.stage_result,
                )
                store.write_result(
                    {
                        "schema_version": "t6-reference-result.v1",
                        "outcome": "references_ready",
                        "snapshot": snapshot_payload,
                        "corpus_snapshot_artifact_id": corpus_artifact_id,
                        "references": references_payload,
                        "reference_validation": validation_payload,
                        "reference_stage_result": reference_artifacts.stage_result,
                        "next_stage": "task_trajectory",
                    }
                )
                store.emit(
                    "references_ready",
                    {
                        "reference_count": len(references_payload),
                        "candidate_count": len(validation_payload),
                        "rejected_count": rejected,
                        "validated_reference_artifact_id": (
                            reference_artifacts.validated_artifact_id
                        ),
                    },
                    level="warning" if receipt_status == "degraded" else "info",
                )

            if (
                run_settings.t6_task_enabled
                and machine.phase is RunPhase.REFERENCES_READY
            ):
                current_state = store.read_state()
                corpus_artifact_id = str(current_state.get("corpus_snapshot_artifact_id") or "")
                reference_artifact_id = str(current_state.get("validated_reference_artifact_id") or "")
                if not corpus_artifact_id or not reference_artifact_id:
                    raise RuntimeError("Unified Task-Trajectory stage requires Corpus and Reference artifacts")
                store.update_state(task_trajectory_mode="unified")
                machine.advance(
                    RunPhase.DISCOVERING_TASK_TRAJECTORIES,
                    _receipt(
                        run_id=run_id, stage="task_trajectory_dispatch",
                        fingerprint_value={"corpus": corpus_artifact_id,
                                           "references": reference_artifact_id,
                                           "prompt_version": UNIFIED_PROMPT_VERSION,
                                           **generation_state},
                        artifact_ids=(corpus_artifact_id, reference_artifact_id),
                    ),
                )

            if machine.phase is RunPhase.DISCOVERING_TASK_TRAJECTORIES:
                current_state = store.read_state()
                corpus_artifact_id = str(current_state["corpus_snapshot_artifact_id"])
                reference_artifact_id = str(current_state["validated_reference_artifact_id"])
                corpus = load_corpus_snapshot(store, corpus_artifact_id)
                reference_payload = store.read_json_artifact(reference_artifact_id)
                references = [ReferenceCandidate.model_validate(item)
                              for item in reference_payload.get("references", [])]
                unified = await UnifiedTaskTrajectoryService(
                    store=store, settings=run_settings, runtime=self.agent_runtime,
                    gateway=self.capability_gateway, gateway_server=self.capability_server,
                    model_config=model_config,
                ).run(
                    target_profile=target_profile, corpus=corpus,
                    corpus_artifact_id=corpus_artifact_id,
                    references=references, reference_artifact_id=reference_artifact_id,
                )
                store.update_state(
                    t6_task_complete=True,
                    t6_knowledge_freeze_complete=True,
                    t7_trajectory_complete=True,
                    task_count=len(unified.tasks),
                    trajectory_count=len(unified.trajectories),
                    validated_task_artifact_id=unified.task_artifact_id,
                    task_validation_artifact_id=unified.task_validation_artifact_id,
                    knowledge_snapshot_id=unified.knowledge.knowledge_snapshot_id,
                    knowledge_snapshot_artifact_id=unified.knowledge_artifact_id,
                    knowledge_freeze_validation_artifact_id=unified.knowledge_validation_artifact_id,
                    trajectory_candidate_artifact_id=unified.trajectory_artifact_id,
                    trajectory_validation_artifact_id=unified.trajectory_validation_artifact_id,
                    validated_trajectories=[item.model_dump(mode="json")
                                            for item in unified.trajectories],
                    task_trajectory_stage_result=unified.stage_result,
                    task_stage_result=unified.stage_result,
                    trajectory_stage_results=[unified.stage_result],
                    trajectory_attempt_ids=([unified.attempt_id] if unified.attempt_id else []),
                )
                machine.advance(
                    RunPhase.INDUCING_SKILLS,
                    _receipt(
                        run_id=run_id, stage="task_trajectory",
                        fingerprint_value={"corpus": corpus_artifact_id,
                                           "references": reference_artifact_id,
                                           "knowledge": unified.knowledge_artifact_id,
                                           "trajectories": unified.trajectory_artifact_id},
                        artifact_ids=(unified.task_artifact_id,
                                      unified.knowledge_artifact_id,
                                      unified.trajectory_artifact_id),
                        validation_artifact_id=unified.trajectory_validation_artifact_id,
                        status="degraded" if unified.stage_result.get("status") == "degraded"
                               else "succeeded",
                        counts={"tasks": len(unified.tasks),
                                "trajectories": len(unified.trajectories)},
                        gaps=tuple(unified.stage_result.get("gaps") or ()),
                        attempt_id=unified.attempt_id,
                    ),
                )

            if machine.phase is RunPhase.INDUCING_SKILLS:
                current_state = store.read_state()
                corpus_artifact_id = str(current_state["corpus_snapshot_artifact_id"])
                knowledge_artifact_id = str(current_state["knowledge_snapshot_artifact_id"])
                corpus = load_corpus_snapshot(store, corpus_artifact_id)
                knowledge = load_knowledge_snapshot(store, knowledge_artifact_id)
                trajectories = [
                    TrajectoryCandidate.model_validate(item)
                    for item in current_state.get("validated_trajectories", [])
                ]
                skill_artifacts = await SkillInductionService(
                    store=store,
                    settings=run_settings,
                    runtime=self.agent_runtime,
                    gateway=self.capability_gateway,
                    gateway_server=self.capability_server,
                    model_config=model_config,
                ).induce(
                    target_profile=target_profile,
                    corpus=corpus,
                    corpus_artifact_id=corpus_artifact_id,
                    knowledge=knowledge,
                    knowledge_artifact_id=knowledge_artifact_id,
                    trajectories=trajectories,
                )
                skill_rejected = sum(
                    1 for item in skill_artifacts.validation if not item.accepted
                )
                store.update_state(
                    t7_skill_complete=True,
                    skill_count=len(skill_artifacts.skills),
                    skill_candidate_artifact_id=skill_artifacts.candidate_artifact_id,
                    skill_validation_artifact_id=skill_artifacts.validation_artifact_id,
                    skill_stage_result=skill_artifacts.stage_result,
                    skill_attempt_ids=list(skill_artifacts.attempt_ids),
                    validated_skills=[
                        item.model_dump(mode="json") for item in skill_artifacts.skills
                    ],
                )
                machine.advance(
                    RunPhase.VALIDATING,
                    _receipt(
                        run_id=run_id,
                        stage="skill_induction",
                        fingerprint_value={
                            "candidate_artifact": skill_artifacts.candidate_artifact_id,
                            "validation_artifact": skill_artifacts.validation_artifact_id,
                        },
                        artifact_ids=(skill_artifacts.candidate_artifact_id,),
                        validation_artifact_id=skill_artifacts.validation_artifact_id,
                        status=(
                            "degraded"
                            if skill_rejected
                            or skill_artifacts.stage_result.get("status") == "degraded"
                            else "succeeded"
                        ),
                        counts={
                            "skills": len(skill_artifacts.skills),
                            "rejected": skill_rejected,
                        },
                        gaps=tuple(skill_artifacts.stage_result.get("gaps") or ()),
                        warnings=tuple(skill_artifacts.stage_result.get("warnings") or ()),
                    ),
                )

            if machine.phase is RunPhase.VALIDATING:
                current_state = store.read_state()
                corpus = load_corpus_snapshot(
                    store, str(current_state["corpus_snapshot_artifact_id"])
                )
                knowledge = load_knowledge_snapshot(
                    store, str(current_state["knowledge_snapshot_artifact_id"])
                )
                trajectories = [
                    TrajectoryCandidate.model_validate(item)
                    for item in current_state.get("validated_trajectories", [])
                ]
                skills = (
                    list(skill_artifacts.skills)
                    if "skill_artifacts" in locals()
                    else [
                        SkillCandidate.model_validate(item)
                        for item in current_state.get("validated_skills", [])
                    ]
                )
                if not skills:
                    skill_payload = store.read_json_artifact(
                        str(current_state["skill_candidate_artifact_id"])
                    )
                    skills = [
                        SkillCandidate.model_validate(item)
                        for item in skill_payload.get("skills", [])
                    ]
                validations: list[CandidateValidation] = [
                    *knowledge.task_validation,
                    *knowledge.reference_validation,
                ]
                for key in (
                    "task_validation_artifact_id",
                    "reference_validation_artifact_id",
                    "trajectory_validation_artifact_id",
                    "skill_validation_artifact_id",
                ):
                    artifact_id = current_state.get(key)
                    if isinstance(artifact_id, str) and artifact_id:
                        payload = store.read_json_artifact(artifact_id)
                        validations.extend(
                            CandidateValidation.model_validate(item)
                            for item in payload.get("validation", [])
                        )
                validation_by_id = {
                    (item.candidate_kind, item.candidate_id): item
                    for item in validations
                }
                validations = list(validation_by_id.values())
                quality_report = build_pipeline_quality_report(
                    corpus,
                    knowledge,
                    trajectories,
                    skills,
                    validations,
                    reference_coverage=(store.read_result() or {}).get(
                        "reference_stage_result", {}
                    ).get("coverage", {}),
                    skill_coverage=current_state.get("skill_stage_result", {}).get(
                        "coverage", {}
                    ),
                )
                quality_artifact = store.put_json_artifact(
                    quality_report,
                    kind="pipeline_quality_report",
                    metadata={"stage": "validation"},
                )
                store.update_state(
                    pipeline_quality_artifact_id=quality_artifact.artifact_id,
                    validated_skills=[item.model_dump(mode="json") for item in skills],
                )
                machine.advance(
                    RunPhase.COMMITTING,
                    _receipt(
                        run_id=run_id,
                        stage="pipeline_validation",
                        fingerprint_value=quality_report,
                        artifact_ids=(quality_artifact.artifact_id,),
                        counts={
                            "tasks": len(knowledge.tasks),
                            "references": len(knowledge.references),
                            "trajectories": len(trajectories),
                            "skills": len(skills),
                        },
                    ),
                )

            if machine.phase is RunPhase.COMMITTING:
                current_state = store.read_state()
                corpus = load_corpus_snapshot(
                    store, str(current_state["corpus_snapshot_artifact_id"])
                )
                knowledge = load_knowledge_snapshot(
                    store, str(current_state["knowledge_snapshot_artifact_id"])
                )
                trajectories = [
                    TrajectoryCandidate.model_validate(item)
                    for item in current_state.get("validated_trajectories", [])
                ]
                skills = [
                    SkillCandidate.model_validate(item)
                    for item in current_state.get("validated_skills", [])
                ]
                validations: list[CandidateValidation] = [
                    *knowledge.task_validation,
                    *knowledge.reference_validation,
                ]
                for key in (
                    "trajectory_validation_artifact_id",
                    "skill_validation_artifact_id",
                ):
                    artifact_id = current_state.get(key)
                    if isinstance(artifact_id, str) and artifact_id:
                        payload = store.read_json_artifact(artifact_id)
                        validations.extend(
                            CandidateValidation.model_validate(item)
                            for item in payload.get("validation", [])
                        )
                validations = list({
                    (item.candidate_kind, item.candidate_id): item
                    for item in validations
                }.values())
                quality_artifact_id = str(current_state["pipeline_quality_artifact_id"])
                quality_report = store.read_json_artifact(quality_artifact_id)
                registry_result = await asyncio.to_thread(
                    SkillRegistry(
                        run_settings.output_dir,
                        artifact_store=store,

                        skill_output_language=run_settings.skill_output_language,
                    ).commit,
                    run_id=run_id,
                    target_profile=target_profile,
                    corpus=corpus,
                    knowledge=knowledge,
                    trajectories=trajectories,
                    skills=skills,
                    validations=validations,
                    quality_report=quality_report,
                )
                commit_artifact = store.put_json_artifact(
                    registry_result, kind="skill_registry_commit",
                    metadata={"stage": "commit"},
                )
                outcome = (
                    "succeeded"
                    if skills
                    and corpus.evidence_count
                    and all(item.status == "succeeded" for item in corpus.source_results)
                    and current_state.get("task_trajectory_stage_result", {}).get("status")
                    == "succeeded"
                    else "degraded"
                )
                terminal = RunPhase.COMPLETED if outcome == "succeeded" else RunPhase.DEGRADED
                machine.advance(
                    terminal,
                    _receipt(
                        run_id=run_id,
                        stage="skill_registry_commit",
                        fingerprint_value=registry_result,
                        artifact_ids=(commit_artifact.artifact_id,),
                        status=outcome,
                        counts={"skills": len(skills)},
                    ),
                )
                result_payload = {
                    "schema_version": "pipeline-result.v7",
                    "run_id": run_id,
                    "outcome": outcome,
                    "snapshot": corpus.model_dump(mode="json"),
                    "corpus_snapshot_artifact_id": current_state["corpus_snapshot_artifact_id"],
                    "knowledge_snapshot": knowledge.model_dump(mode="json"),
                    "knowledge_snapshot_artifact_id": current_state["knowledge_snapshot_artifact_id"],
                    "tasks": [item.model_dump(mode="json") for item in knowledge.tasks],
                    "references": [item.model_dump(mode="json") for item in knowledge.references],
                    "trajectories": [item.model_dump(mode="json") for item in trajectories],
                    "skills": [
                        item.model_copy(update={"status": "committed"}).model_dump(
                            mode="json"
                        )
                        for item in skills
                    ],
                    "stage_results": {
                        "task_trajectory": current_state.get("task_trajectory_stage_result", {}),
                        "task_discovery": current_state.get("task_stage_result", {}),
                        "reference_build": current_state.get("reference_stage_result", {}),
                        "knowledge_freeze": current_state.get("knowledge_freeze_result", {}),
                        "trajectory_inference": current_state.get("trajectory_stage_results", []),
                        "skill_induction": current_state.get("skill_stage_result", {}),
                    },
                    "skill_registry": registry_result,
                    "skill_registry_commit_artifact_id": commit_artifact.artifact_id,
                    "pipeline_quality": quality_report,
                    "pipeline_quality_artifact_id": quality_artifact_id,
                }
                store.write_result(result_payload)
                store.update_state(
                    outcome=outcome,
                    output_directory=registry_result["output_directory"],
                    audit_directory=registry_result["audit_directory"],
                    agent_skill_directory=registry_result.get("agent_skill_directory"),
                    agent_entry_directory=registry_result.get("agent_entry_directory"),
                    agent_entry_manifest_file=registry_result.get("agent_entry_manifest_file"),
                    agent_entry_count=registry_result.get("agent_entry_count"),
                )
                store.emit(
                    "run_completed",
                    {"outcome": outcome, "skill_count": len(skills)},
                    level="warning" if outcome != "succeeded" else "info",
                )
                return result_payload

            state = store.read_state()
            snapshot_artifact_id = state.get("corpus_snapshot_artifact_id")
            if not isinstance(snapshot_artifact_id, str):
                raise RuntimeError(
                    f"T5 cannot resume from phase {machine.phase.value}: CorpusSnapshot is missing"
                )
            if machine.phase is RunPhase.REFERENCES_READY:
                return store.read_result() or {}
            return store.read_json_artifact(snapshot_artifact_id)
        except asyncio.CancelledError:
            if machine.phase not in TERMINAL_PHASES:
                machine.advance(
                    RunPhase.CANCELLED,
                    _receipt(
                        run_id=run_id,
                        stage="run_cancel",
                        fingerprint_value={"run_id": run_id, "phase": machine.phase.value},
                        status="cancelled",
                    ),
                )
            store.update_state(outcome="cancelled")
            store.emit("run_cancelled", {"reason": "user_requested"}, level="warning")
            raise
        except Exception as exc:
            error = public_pipeline_error(exc)
            if machine.phase not in TERMINAL_PHASES:
                machine.advance(
                    RunPhase.FAILED,
                    _receipt(
                        run_id=run_id,
                        stage="pipeline_failure",
                        fingerprint_value={
                            "run_id": run_id,
                            "phase": machine.phase.value,
                            "category": error.get("category"),
                            "message": error.get("message"),
                        },
                        status="failed",
                        gaps=(str(error.get("message") or "T5 failed"),),
                    ),
                )
            store.update_state(error=error, outcome="failed")
            store.emit("run_failed", error, level="error")
            raise

    async def _review_prepared_sources(
        self,
        store: RunStore,
        prepared: list[PreparedSource],
        model_config: ModelRuntimeConfig,
    ) -> list[SourceResult]:
        if not prepared:
            return []
        service = SourceReviewService(
            store=store,
            runtime=self.agent_runtime,
            gateway=self.capability_gateway,
            gateway_server=self.capability_server,
            model_config=model_config,
        )
        tasks = [asyncio.create_task(service.review(item)) for item in prepared]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def cancel(self, run_id: str) -> dict[str, Any]:
        self.recover_scope_state(run_id)
        store = self.get_store(run_id)
        state = store.read_state()
        if state.get("status") in {"completed", "failed", "cancelled"}:
            return {
                "run_id": run_id,
                "status": state.get("status"),
                "cancel_requested": False,
                "active_task": False,
                "message": "Run is already terminal.",
            }
        with self._tasks_lock:
            task = self._tasks.get(run_id)
        active = bool(task and not task.done())
        store.emit(
            "run_cancel_requested",
            {"active_task": active, "previous_status": state.get("status")},
            level="warning",
        )
        if task is not None and active:
            # Stopping the orchestration coroutine alone can leave a persistent DSH
            # worker finishing its current model request. Explicitly terminate every
            # running Child Session as a separate task before cancelling the pipeline.
            self._cancel_running_child_sessions(store)
            task.cancel()
            return {
                "run_id": run_id,
                "status": "cancelling",
                "cancel_requested": True,
                "active_task": True,
            }
        machine = PipelineStateMachine(store)
        if machine.phase not in TERMINAL_PHASES:
            machine.advance(
                RunPhase.CANCELLED,
                _receipt(
                    run_id=run_id,
                    stage="run_cancel",
                    fingerprint_value={"run_id": run_id, "phase": machine.phase.value},
                    status="cancelled",
                ),
            )
        store.update_state(outcome="cancelled")
        self._model_configs.pop(run_id, None)
        return {
            "run_id": run_id,
            "status": "cancelled",
            "cancel_requested": True,
            "active_task": False,
        }

    def close(self) -> None:
        self._model_configs.clear()
        self.capability_server.close()

    @staticmethod
    def _active_manifests(manifests: list[SourceManifest]) -> list[SourceManifest]:
        return [
            item
            for item in manifests
            if item.primary_modality in {"pdf", "html"}
            and item.raw_artifact_id
            and item.exclusion_reason is None
            and item.status not in {"excluded", "failed"}
        ]

    @staticmethod
    def _no_usable_sources(
        inputs: list[StagedInput], manifests: list[SourceManifest]
    ) -> NoUsableSourcesError:
        reasons: dict[str, int] = {}
        for item in manifests:
            reason = item.exclusion_reason or item.status
            reasons[reason] = reasons.get(reason, 0) + 1
        return NoUsableSourcesError(
            input_count=len(inputs),
            excluded_count=sum(item.status == "excluded" for item in manifests),
            failed_count=sum(item.status == "failed" for item in manifests),
            reason_counts=reasons,
        )

    @staticmethod
    def _ordered_results(
        manifests: list[SourceManifest], results: list[SourceResult]
    ) -> list[SourceResult]:
        by_source = {item.source_id: item for item in results}
        return [by_source[item.source_id] for item in manifests if item.source_id in by_source]

    @staticmethod
    def _load_receipted_source_results(store: RunStore) -> list[SourceResult]:
        """Load the immutable validated batch accepted by the Source Review Receipt."""

        candidate_ids: list[str] = []
        state_id = store.read_state().get("validated_source_batch_artifact_id")
        if isinstance(state_id, str):
            candidate_ids.append(state_id)
        for commit in reversed(store.list_receipt_commits()):
            receipt = commit.get("receipt") if isinstance(commit, dict) else None
            if not isinstance(receipt, dict) or receipt.get("stage") != "source_review":
                continue
            candidate_ids.extend(
                str(item) for item in receipt.get("accepted_output_artifact_ids", [])
            )
            break
        for artifact_id in dict.fromkeys(candidate_ids):
            try:
                record = store.artifact_record(artifact_id)
            except FileNotFoundError:
                continue
            if record.kind != "validated_source_batch":
                continue
            raw = store.artifact_path(artifact_id).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record.sha256:
                raise RuntimeError("Validated Source batch Artifact failed its content hash")
            payload = json.loads(raw)
            if payload.get("schema_version") != "validated-source-batch.v1":
                raise RuntimeError("Validated Source batch has an unsupported schema")
            return [
                SourceResult.model_validate(item)
                for item in payload.get("source_results", [])
            ]
        raise RuntimeError("Source Review Receipt has no validated Source batch Artifact")

    @staticmethod
    def _corpus_dependency_artifact_ids(corpus: CorpusSnapshot) -> set[str]:
        """Artifacts needed to inspect cached Evidence without the original Run."""

        artifact_ids: set[str] = set()

        def add_evidence_artifacts(evidence: Any) -> None:
            if evidence.text_artifact_id:
                artifact_ids.add(evidence.text_artifact_id)
            artifact_ids.update(image.artifact_id for image in evidence.image_refs)

        for evidence in corpus.evidence:
            add_evidence_artifacts(evidence)
        for source in corpus.source_results:
            for evidence in source.evidence:
                add_evidence_artifacts(evidence)
        for tree in corpus.evidence_trees:
            artifact_ids.add(tree.plan_artifact_id)
            artifact_ids.add(tree.validation_artifact_id)
            for node in tree.nodes:
                if node.content_artifact_id:
                    artifact_ids.add(node.content_artifact_id)
                artifact_ids.update(ref.text_artifact_id for ref in node.source_refs)
        artifact_ids.discard("")
        return artifact_ids
