from __future__ import annotations

from enum import Enum

from ..contracts import StageReceipt
from ..storage import RunStore


class RunPhase(str, Enum):
    CREATED = "created"
    DISCOVERING_SOURCES = "discovering_sources"
    AWAITING_SOURCE_SELECTION = "awaiting_source_selection"
    SOURCE_SELECTED = "source_selected"
    ACQUIRING = "acquiring"
    PREVIEWING_SCOPE = "previewing_scope"
    AWAITING_SCOPE_SELECTION = "awaiting_scope_selection"
    SCOPE_SELECTED = "scope_selected"
    NORMALIZING = "normalizing"
    SOURCE_REVIEW = "source_review"
    CORPUS_FROZEN = "corpus_frozen"
    REFERENCE_BUILDING = "reference_building"
    REFERENCES_READY = "references_ready"
    DISCOVERING_TASK_TRAJECTORIES = "discovering_task_trajectories"
    INDUCING_SKILLS = "inducing_skills"
    VALIDATING = "validating"
    COMMITTING = "committing"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_PHASES = {
    RunPhase.COMPLETED,
    RunPhase.DEGRADED,
    RunPhase.FAILED,
    RunPhase.CANCELLED,
}

_HAPPY_PATH = (
    RunPhase.CREATED,
    RunPhase.ACQUIRING,
    RunPhase.NORMALIZING,
    RunPhase.SOURCE_REVIEW,
    RunPhase.CORPUS_FROZEN,
    RunPhase.REFERENCE_BUILDING,
    RunPhase.REFERENCES_READY,
    RunPhase.DISCOVERING_TASK_TRAJECTORIES,
    RunPhase.INDUCING_SKILLS,
    RunPhase.VALIDATING,
    RunPhase.COMMITTING,
)

ALLOWED_TRANSITIONS: dict[RunPhase, frozenset[RunPhase]] = {
    phase: frozenset({next_phase, RunPhase.FAILED, RunPhase.CANCELLED})
    for phase, next_phase in zip(_HAPPY_PATH, _HAPPY_PATH[1:])
}
# A new Run may explicitly import an immutable, already-frozen T5 checkpoint.
# The import is still receipt-gated and never carries sessions or chat history.
ALLOWED_TRANSITIONS[RunPhase.CREATED] = frozenset(
    {*ALLOWED_TRANSITIONS[RunPhase.CREATED], RunPhase.CORPUS_FROZEN, RunPhase.DISCOVERING_SOURCES}
)
ALLOWED_TRANSITIONS[RunPhase.DISCOVERING_SOURCES] = frozenset(
    {RunPhase.AWAITING_SOURCE_SELECTION, RunPhase.FAILED, RunPhase.CANCELLED}
)
ALLOWED_TRANSITIONS[RunPhase.AWAITING_SOURCE_SELECTION] = frozenset(
    {RunPhase.SOURCE_SELECTED, RunPhase.DISCOVERING_SOURCES, RunPhase.CANCELLED, RunPhase.FAILED}
)
ALLOWED_TRANSITIONS[RunPhase.SOURCE_SELECTED] = frozenset(
    {RunPhase.ACQUIRING, RunPhase.FAILED, RunPhase.CANCELLED}
)
ALLOWED_TRANSITIONS[RunPhase.ACQUIRING] = frozenset(
    {*ALLOWED_TRANSITIONS[RunPhase.ACQUIRING], RunPhase.PREVIEWING_SCOPE}
)
ALLOWED_TRANSITIONS[RunPhase.PREVIEWING_SCOPE] = frozenset(
    {RunPhase.AWAITING_SCOPE_SELECTION, RunPhase.NORMALIZING, RunPhase.FAILED, RunPhase.CANCELLED}
)
ALLOWED_TRANSITIONS[RunPhase.AWAITING_SCOPE_SELECTION] = frozenset(
    {RunPhase.SCOPE_SELECTED, RunPhase.FAILED, RunPhase.CANCELLED}
)
ALLOWED_TRANSITIONS[RunPhase.SCOPE_SELECTED] = frozenset(
    {RunPhase.NORMALIZING, RunPhase.CORPUS_FROZEN, RunPhase.FAILED, RunPhase.CANCELLED}
)
ALLOWED_TRANSITIONS[RunPhase.SOURCE_REVIEW] = frozenset(
    {*ALLOWED_TRANSITIONS[RunPhase.SOURCE_REVIEW], RunPhase.PREVIEWING_SCOPE}
)
ALLOWED_TRANSITIONS[RunPhase.COMMITTING] = frozenset(
    {RunPhase.COMPLETED, RunPhase.DEGRADED, RunPhase.FAILED, RunPhase.CANCELLED}
)
for phase in TERMINAL_PHASES:
    ALLOWED_TRANSITIONS[phase] = frozenset()


class InvalidStateTransitionError(RuntimeError):
    pass


class PipelineStateMachine:
    """Authoritative, receipt-gated state projection for one business Run."""

    def __init__(self, store: RunStore):
        self.store = store

    @property
    def phase(self) -> RunPhase:
        state = self.store.read_state()
        return RunPhase(str(state.get("phase") or RunPhase.CREATED.value))

    @property
    def terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES

    def allowed_next(self) -> frozenset[RunPhase]:
        return ALLOWED_TRANSITIONS[self.phase]

    def advance(self, next_phase: RunPhase, receipt: StageReceipt) -> dict[str, object]:
        current = self.phase
        if next_phase == current:
            return self.store.commit_stage_receipt(
                receipt,
                expected_phase="__idempotent_replay__",
                next_phase=next_phase.value,
            )
        if next_phase not in ALLOWED_TRANSITIONS[current]:
            raise InvalidStateTransitionError(
                f"Illegal Run transition: {current.value} -> {next_phase.value}"
            )
        expected_receipt_status = {
            RunPhase.COMPLETED: "succeeded",
            RunPhase.DEGRADED: "degraded",
            RunPhase.FAILED: "failed",
            RunPhase.CANCELLED: "cancelled",
        }.get(next_phase)
        if expected_receipt_status and receipt.status != expected_receipt_status:
            raise InvalidStateTransitionError(
                f"Transition to {next_phase.value} requires a {expected_receipt_status} receipt"
            )
        if not expected_receipt_status and receipt.status not in {"succeeded", "degraded"}:
            raise InvalidStateTransitionError(
                f"Transition to {next_phase.value} requires a successful receipt"
            )
        return self.store.commit_stage_receipt(
            receipt,
            expected_phase=current.value,
            next_phase=next_phase.value,
        )

    def recover(self) -> dict[str, object]:
        return self.store.recover_from_receipts()
