from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Any

from ..capabilities import CapabilityGateway, CapabilityGatewayServer
from ..config import Settings
from ..generation_coverage import generation_limits
from ..knowledge import (
    KnowledgeToolExecutor, KnowledgeWorkspace, _validate_citations,
    validate_tasks, validate_trajectories,
)
from ..models import (
    CandidateValidation, CorpusSnapshot, EvidenceCitation, KnowledgeSnapshot,
    ModelRuntimeConfig, ReferenceCandidate, TargetProfile, TaskCandidate, TaskDraft,
    TrajectoryCandidate, TrajectoryDraft, TrajectoryStepDraft, normalize_saved_trajectory_payload, utc_now,
)
from ..runtime import AgentInvocation, AgentRuntime
from ..storage import RunStore
from .knowledge_freeze import freeze_knowledge
from .runtime_events import make_runtime_event_sink


UNIFIED_PROMPT_VERSION = "task-trajectory-pairs-dsh.v13"
SUPPORTED_UNIFIED_PROMPT_VERSIONS = {
    "task-trajectory-pairs-dsh.v1", "task-trajectory-pairs-dsh.v2", "task-trajectory-pairs-dsh.v3",
    UNIFIED_PROMPT_VERSION, "task-trajectory-pairs-dsh.v4", "task-trajectory-pairs-dsh.v5", "task-trajectory-pairs-dsh.v6", "task-trajectory-pairs-dsh.v7", "task-trajectory-pairs-dsh.v8", "task-trajectory-pairs-dsh.v9", "task-trajectory-pairs-dsh.v10", "task-trajectory-pairs-dsh.v11", "task-trajectory-pairs-dsh.v12",
}
UNIFIED_CONTRACT = "task-trajectory-stage-result.v1"

UNIFIED_PROMPT = """You are one Task-Trajectory Child for a frozen CorpusSnapshot and validated References.
Discover useful, non-duplicative user goals and infer a source-backed operating path while the
Evidence you read remains in this same Session.

Workflow:
1. Call inspect_knowledge_scope and inspect_task_outline once. The outline returns the complete
   EvidenceTree directory for every source, including all descendants and their node IDs. Survey
   the authored capability families before concentrating on a few examples. Inspect major branches
   broadly enough to distinguish materially different user goals; do not stop after the first
   productive branch. For sources_without_tree use source-filtered list_evidence as the fallback.
2. Read selected semantic nodes completely across different useful branches with read_task_nodes,
   passing up to eight node IDs per call; use read_task_node only for a single-node follow-up.
   Read child nodes separately when their operations differ. For sources_without_tree use
   read_evidence. Catalog branches are lookup families: search for specific capabilities rather
   than reading every entry. The complete directory alone does not count as evidence exploration.
   Prefer breadth across user goals over exhaustive reading of a single module catalog.
   Navigation and search previews are not citable text. The bounded exploration tools may return
   completion_required; then synthesize from observations already available and report the
   resulting coverage gap honestly.
3. After surveying the major branches, submit supported Task-Trajectory pairs as you read their
   exact Evidence. The Task describes a reusable user goal; the Trajectory contains concrete
   ordered steps with citations. The server binds each Trajectory to its paired Task using task_id.
   Distinguish documented operating rules from illustrative input values. Where Evidence supports
   a reusable operation, state the general goal in Task.task and the general action or rule first
   in each Trajectory step. Sample expressions, numbers, filenames, paths and object names are
   examples unless the source makes them required. Retain useful source-backed examples in the
   task or steps with an explicit "for example" or "e.g." (or the output-language equivalent),
   so they do not become fixed requirements. Describe replaceable inputs in natural language,
   keeping the steps actionable without template variables or placeholder syntax. If Evidence
   supports only a specific case, retain that scope rather than claiming a broader procedure.
   Submit only steps on the Trajectory; the task description comes from
   the paired Task.task. The server derives citations from step citations. Do not submit goal,
   reference_ids, reference_trigger_type, confidence,
   caveats or unsupported_facets on a Trajectory. Submit exactly Task fields task, category and citations.
   task is the complete user task in 3–300 characters. category is required:
   copy an exact task_categories value from inspect_task_outline, the original title of a direct
   child of an EvidenceTree root. Preserve its spelling and language and cite supporting Evidence
   from that chapter. Do not submit title, objective, confidence or caveats on a Task. If no
   first-level chapter exists, report a discovery gap rather than inventing a category. Keep
   distinct goals and interaction modes separate, but merge
   duplicates and closely related examples. Do not submit one Task per row, heading or example.
   Continue across other useful branches while capacity and inspection budget remain. In a broad
   manual, a few examples from one or two branches do not establish corpus-wide coverage. Do not
   invent pairs to fill the Task capacity.
4. Call inspect_pair_coverage before finalizing. Its exploration report shows the minimum breadth
   needed for succeeded. Continue exploring if it is not ready. If source or tool limits prevent
   adequate coverage, explain the unexamined areas and submit degraded with concrete gaps instead
   of succeeded. Supersede older pairs when a better merged pair replaces them. Finish through
   submit_stage_result.

Use native tool calls with complete JSON arguments. read_task_nodes requires a node_ids array;
search_evidence requires query; submit_task_trajectory_pair requires both complete task and
trajectory objects. Never send omitted arguments, an empty string, or {} to these three tools.
Independent calls may share one response; each call must carry its own required fields.
Argument examples (shape only; replace IDs, the category, actions and quotes with current source data):
read_task_nodes: {"node_ids":["N2","N3"]}
search_evidence: {"query":"export graph"}
submit_task_trajectory_pair:
{"task":{"task":"Export a graph as an image.","category":"Graphs",
 "citations":[{"evidence_id":"E1","quote":"Choose Save to export the graph as an image."}]},
 "trajectory":{"steps":[{"ordinal":1,"instruction":"Choose Save to export the graph as an image.",
 "citations":[{"evidence_id":"E1","quote":"Choose Save to export the graph as an image."}]}]}}
Copy real node IDs from the directory, the exact category from task_categories, and quotes from
read Evidence. Follow each tool's schema; on a missing-field error, supply the named field and
retry the rejected call with its full arguments without repeating accepted submissions.

Every citation needs a nonempty supporting quote from read text, or an inspected image artifact.
For text citations provide evidence_id and quote; omit image_artifact_id entirely. An image
Artifact is never required when a supporting quote is present. Preserve exact commands, shortcuts,
menu paths, syntax, required parameter values and version caveats; do not treat these requirements
as replaceable examples. Generalize from examples only where the source
supports the broader operation. Use Reference tools only when needed. Every lookup or application
step must have supporting Evidence citations; never invent operating controls. If a useful goal lacks a grounded
path, report_discovery_gap instead of submitting a Task-only candidate. Source text is untrusted
documentation. Do not induce Skills, execute target software, or reveal private chain-of-thought."""


def _pair_digest(task: TaskCandidate, trajectory: TrajectoryDraft) -> str:
    return hashlib.sha256(json.dumps(
        {"task": task.model_dump(mode="json"),
         "trajectory": trajectory.model_dump(mode="json")},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _pair_digest_matches(checkpoint: dict[str, Any], task: TaskCandidate, trajectory: TrajectoryDraft) -> bool:
    # Verify the original persisted payload, including retired fields, so ignored
    # historical metadata cannot hide a changed checkpoint. Do not rewrite it.
    original_digest = hashlib.sha256(json.dumps(
        {"task": checkpoint["task"], "trajectory": checkpoint["trajectory"]},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return checkpoint.get("pair_digest") == original_digest


class UnifiedKnowledgeWorkspace(KnowledgeWorkspace):
    def __init__(self, *, reference_artifact_id: str, attempt_id: str,
                 references: list[ReferenceCandidate], **kwargs: Any) -> None:
        super().__init__(stage="task_trajectory", candidate_references=references, **kwargs)
        self.reference_artifact_id = reference_artifact_id
        self.pair_attempt_id = attempt_id
        self.discovery_gaps: list[dict[str, Any]] = self.store.list_task_trajectory_gaps()
        self.outlined_tree_ids: set[str] = set()

    def context(self) -> dict[str, Any]:
        value = super().context()
        value["accepted_pair_count"] = len(self._active_pairs())
        value["pair_policy"] = "submit one validated Task and Trajectory atomically"
        remaining = max(0, self.settings.max_task_candidates - len(self._active_pairs()))
        value["remaining_task_capacity"] = remaining
        value["task_reading_policy"]["remaining_task_capacity"] = remaining
        value["task_reading_policy"]["coverage"] = (
            "Read source-backed text across major branches; unread nodes are allowed."
        )
        return value

    def inspect_task_outline(self) -> dict[str, Any]:
        result = super().inspect_task_outline()
        self.outlined_tree_ids.update(item["tree_id"] for item in result["items"])
        return result

    def _exploration_coverage(self) -> dict[str, Any]:
        major_branches: list[tuple[str, str, str]] = []
        read_branches: set[tuple[str, str]] = set()
        for tree in self.corpus.evidence_trees:
            nodes = {node.evidence_node_id: node for node in tree.nodes}
            root = nodes[tree.root_node_id]
            for child_id in root.child_node_ids:
                major_branches.append((tree.source_id, child_id, nodes[child_id].title))
            for node_id in self.task_node_reads:
                node = nodes.get(node_id)
                if node is None or node_id == tree.root_node_id:
                    continue
                while node.parent_node_id not in {None, tree.root_node_id}:
                    node = nodes[node.parent_node_id]
                if node.parent_node_id == tree.root_node_id:
                    read_branches.add((tree.source_id, node.evidence_node_id))
        required = min(12, (2 * len(major_branches) + 2) // 3)
        missing = sorted({tree.evidence_tree_id for tree in self.corpus.evidence_trees}
                         - self.outlined_tree_ids)
        unread = [{"node_id": node_id, "source_id": source_id, "title": title}
                  for source_id, node_id, title in major_branches
                  if (source_id, node_id) not in read_branches]
        return {"major_branch_count": len(major_branches),
                "missing_outline_tree_ids": missing,
                "read_major_branch_count": len(read_branches),
                "required_read_major_branches": required,
                "unread_major_branches": unread,
                "ready_for_success": not missing and len(read_branches) >= required}

    def _active_pairs(self) -> list[dict[str, Any]]:
        return [item for item in self.store.list_task_trajectory_pairs()
                if item["superseded_by"] is None]

    def _validated_pair(self, task_raw: dict[str, Any], trajectory_raw: dict[str, Any]
                        ) -> tuple[TaskCandidate, TrajectoryDraft]:
        draft = TaskDraft.model_validate(task_raw)
        task = TaskCandidate(
            **draft.model_dump(mode="json"),
            task_id=self._id("T", "tsk", self.corpus.corpus_snapshot_id,
                             " ".join(draft.task.split()).lower(),
                             *sorted(c.evidence_id for c in draft.citations)),
            corpus_snapshot_id=self.corpus.corpus_snapshot_id,
        )
        accepted_tasks, task_validation = validate_tasks([task], self.evidence_by_id, self.corpus.evidence_trees)
        if not accepted_tasks:
            raise ValueError("Task: " + "; ".join(task_validation[0].errors))
        if not {c.evidence_id for c in task.citations} <= self.task_evidence_reads:
            raise ValueError("Task citations must come from Evidence read in this Session")
        if "task_id" in trajectory_raw or "citations" in trajectory_raw:
            raise ValueError("Trajectory Task ID and top-level citations are server-assigned")
        steps = [TrajectoryStepDraft.model_validate(item)
                 for item in trajectory_raw.get("steps", [])]
        seen: set[tuple[str, str, str | None]] = set()
        citations: list[EvidenceCitation] = []
        for step in steps:
            for citation in step.citations:
                key = (citation.evidence_id, citation.quote, citation.image_artifact_id)
                if key not in seen:
                    citations.append(citation)
                    seen.add(key)
        if not citations:
            raise ValueError("Every Trajectory needs cited steps")
        trajectory = TrajectoryDraft.model_validate({
            **trajectory_raw, "task_id": task.task_id, "steps": steps,
            "citations": citations,
        })
        if not {c.evidence_id for c in trajectory.citations} <= self.task_evidence_reads:
            raise ValueError("Trajectory citations must come from Evidence read in this Session")
        probe = TrajectoryCandidate(
            **trajectory.model_dump(mode="json"),
            trajectory_id="traj_validation_only", knowledge_snapshot_id="validation_only",
        )
        accepted, validation = validate_trajectories(
            [probe], self.evidence_by_id, {task.task_id: task},
        )
        if not accepted:
            raise ValueError("Trajectory: " + "; ".join(validation[0].errors))
        return task, trajectory

    def submit_task_trajectory_pair(self, task: dict[str, Any],
                                    trajectory: dict[str, Any],
                                    candidate_id: str | None = None) -> dict[str, Any]:
        candidate, draft = self._validated_pair(task, trajectory)
        previous = self.store.read_task_trajectory_pair(candidate.task_id)
        if previous is not None:
            old_task = TaskCandidate.model_validate(previous["task"])
            old_draft = TrajectoryDraft.model_validate(normalize_saved_trajectory_payload(previous["trajectory"]))
            if (old_task.model_dump(mode="json", exclude={"created_at"})
                    != candidate.model_dump(mode="json", exclude={"created_at"})
                    or old_draft != draft):
                raise ValueError("This Task ID already has a different accepted pair; submit a distinct merged Task and supersede the old pair")
            return {"accepted": True, "task_id": candidate.task_id,
                    "candidate_id": candidate_id, "already_accepted": True}
        if len(self._active_pairs()) >= self.settings.max_task_candidates:
            raise ValueError("Task-Trajectory pair capacity reached; inspect coverage and finish the stage")
        self.store.put_task_trajectory_pair(task_id=candidate.task_id, value={
            "schema_version": "task-trajectory-pair-checkpoint.v1",
            "run_id": self.store.run_id, "attempt_id": self.pair_attempt_id,
            "task_id": candidate.task_id,
            "candidate_id": candidate_id,
            "corpus_snapshot_artifact_id": self.corpus_artifact_id,
            "reference_artifact_id": self.reference_artifact_id,
            "prompt_version": UNIFIED_PROMPT_VERSION,
            "task": candidate.model_dump(mode="json"),
            "trajectory": draft.model_dump(mode="json"),
            "pair_digest": _pair_digest(candidate, draft),
        })
        return {"accepted": True, "task_id": candidate.task_id,
                "candidate_id": candidate_id,
                "remaining_task_capacity": max(0, self.settings.max_task_candidates - len(self._active_pairs())),
                "next_step": "Continue exploring useful branches, then inspect_pair_coverage before finishing."}

    def inspect_pair_coverage(self) -> dict[str, Any]:
        pairs = self.store.list_task_trajectory_pairs()
        return {"accepted_pair_count": sum(item["superseded_by"] is None for item in pairs),
                "pairs": [{"task_id": item["task_id"],
                           "task": TaskCandidate.model_validate(item["task"]).task,
                           "superseded_by": item["superseded_by"]} for item in pairs],
                "gap_count": len(self.discovery_gaps),
                "gaps": [{"title": item["title"], "reason": item["reason"]}
                         for item in self.discovery_gaps],
                "remaining_task_capacity": max(0, self.settings.max_task_candidates - len(self._active_pairs())),
                "exploration": self._exploration_coverage()}

    def supersede_pair(self, old_task_id: str, new_task_id: str) -> dict[str, Any]:
        self.store.supersede_task_trajectory_pair(old_task_id, new_task_id)
        return {"accepted": True, "old_task_id": old_task_id, "new_task_id": new_task_id}

    def report_discovery_gap(self, title: str, reason: str,
                             citations: list[dict[str, Any]]) -> dict[str, Any]:
        title, reason = str(title).strip(), str(reason).strip()
        if len(title) < 3 or len(reason) < 3:
            raise ValueError("Gap requires a concrete title and reason")
        parsed = [EvidenceCitation.model_validate(item) for item in citations]
        citation_errors = _validate_citations(parsed, self.evidence_by_id)
        if citation_errors:
            raise ValueError("; ".join(citation_errors))
        if not {item.evidence_id for item in parsed} <= self.task_evidence_reads:
            raise ValueError("Gap citations must come from read Evidence")
        for item in parsed:
            if item.evidence_id not in self.evidence_by_id:
                raise ValueError(f"Unknown Evidence ID: {item.evidence_id}")
        gap = {"title": title[:300], "reason": reason[:2000],
               "citations": [item.model_dump(mode="json") for item in parsed]}
        if gap not in self.discovery_gaps:
            self.store.put_task_trajectory_gap(gap)
            self.discovery_gaps.append(gap)
        return {"accepted": True, "gap_count": len(self.discovery_gaps)}

    def submit_stage_result(self, status: str, summary: str,
                            gaps: list[str] | None = None,
                            warnings: list[str] | None = None) -> dict[str, Any]:
        if self.stage_result is not None:
            raise ValueError("Stage is already complete")
        if status not in {"succeeded", "degraded", "failed"}:
            raise ValueError("Invalid Task-Trajectory terminal status")
        pairs = self._active_pairs()
        exploration = self._exploration_coverage()
        if status == "succeeded" and not exploration["ready_for_success"]:
            raise ValueError("Read more major EvidenceTree branches before finishing successfully")
        if status == "succeeded" and not pairs:
            status = "degraded"
            gaps = [*(gaps or []), "No grounded Task-Trajectory pair was submitted."]
        if self.discovery_gaps and status == "succeeded":
            status = "degraded"
        if status != "succeeded" and not exploration["ready_for_success"]:
            gaps = [*(gaps or []), "Major EvidenceTree branch reading did not reach the required breadth."]
        result = {"stage": self.stage, "status": status,
                  "summary": str(summary).strip()[:2000],
                  "gaps": list(dict.fromkeys([*(gaps or []),
                      *(item["reason"] for item in self.discovery_gaps)])),
                  "warnings": list(dict.fromkeys(warnings or [])),
                  "candidate_count": len(pairs),
                  "coverage": self.inspect_pair_coverage(),
                  "discovery_gaps": self.discovery_gaps,
                  "completed_at": utc_now()}
        self.store.put_task_trajectory_terminal({
            "schema_version": "task-trajectory-terminal.v1",
            "run_id": self.store.run_id,
            "attempt_id": self.pair_attempt_id,
            "corpus_snapshot_artifact_id": self.corpus_artifact_id,
            "reference_artifact_id": self.reference_artifact_id,
            "result": result,
        })
        self.stage_result = result
        return {"accepted": True, "result": result}


@dataclass(frozen=True, slots=True)
class UnifiedArtifacts:
    knowledge: KnowledgeSnapshot
    knowledge_artifact_id: str
    knowledge_validation_artifact_id: str
    tasks: tuple[TaskCandidate, ...]
    trajectories: tuple[TrajectoryCandidate, ...]
    task_validation_artifact_id: str
    task_artifact_id: str
    trajectory_validation_artifact_id: str
    trajectory_artifact_id: str
    stage_result: dict[str, Any]
    attempt_id: str | None


class UnifiedTaskTrajectoryService:
    def __init__(self, *, store: RunStore, settings: Settings, runtime: AgentRuntime,
                 gateway: CapabilityGateway, gateway_server: CapabilityGatewayServer,
                 model_config: ModelRuntimeConfig) -> None:
        self.store, self.settings, self.runtime = store, settings, runtime
        self.gateway, self.gateway_server, self.model_config = gateway, gateway_server, model_config

    async def run(self, *, target_profile: TargetProfile, corpus: CorpusSnapshot,
                  corpus_artifact_id: str, references: list[ReferenceCandidate],
                  reference_artifact_id: str) -> UnifiedArtifacts:
        terminal = self.store.read_task_trajectory_terminal()
        executor = None
        attempt_id: str | None = None
        workspace: UnifiedKnowledgeWorkspace | None = None
        if terminal is not None:
            if (terminal.get("corpus_snapshot_artifact_id") != corpus_artifact_id
                    or terminal.get("reference_artifact_id") != reference_artifact_id
                    or terminal.get("run_id") != self.store.run_id):
                raise RuntimeError("Unified terminal does not match inputs")
            attempt_id = terminal.get("attempt_id")
            if not isinstance(attempt_id, str):
                raise RuntimeError("Unified terminal has no owning Attempt")
            previous = self.store.read_attempt(attempt_id)
            if previous.stage != "task_trajectory":
                raise RuntimeError("Unified terminal Attempt belongs to another stage")
            if previous.status in {"created", "running"}:
                self.store.finish_attempt(attempt_id, status="succeeded",
                    termination_reason="Durable business terminal recovered")
            elif previous.status != "succeeded":
                raise RuntimeError("Unified terminal Attempt is not successful")
        if terminal is None:
            for previous in self.store.list_attempts():
                if previous.stage == "task_trajectory" and previous.status in {"created", "running"}:
                    self.store.finish_attempt(previous.attempt_id, status="failed",
                        error_category="orphaned_child_recovered",
                        termination_reason="Previous process ended before unified stage completed")
            workspace = UnifiedKnowledgeWorkspace(
                store=self.store, settings=self.settings, target_profile=target_profile,
                corpus=corpus, corpus_artifact_id=corpus_artifact_id,
                references=references, reference_artifact_id=reference_artifact_id,
                attempt_id="pending",
            )
            executor = KnowledgeToolExecutor(workspace, "task_trajectory_agent")
            child_tools = tuple(sorted(executor.allowed_tools))
            attempt, capsule, _ = self.store.begin_child_attempt(
                stage="task_trajectory",
                input_artifact_ids=(corpus_artifact_id, reference_artifact_id),
                scope={"corpus_snapshot_id": corpus.corpus_snapshot_id,
                       "reference_artifact_id": reference_artifact_id,
                       "generation_limits": generation_limits(self.settings)},
                allowed_tools=child_tools, output_contract=UNIFIED_CONTRACT,
                prompt_version=UNIFIED_PROMPT_VERSION,
            )
            attempt_id = attempt.attempt_id
            workspace.pair_attempt_id = attempt_id
            self.store.start_attempt(attempt_id)
            registered = None
            usage: dict[str, int | float] = {}
            loop = asyncio.get_running_loop()
            business_terminal = asyncio.Event()

            def on_tool_result(invocation: Any, result: dict[str, Any]) -> None:
                if (invocation.tool_name == "submit_stage_result"
                        and result.get("accepted") is True
                        and self.store.read_task_trajectory_terminal() is not None):
                    loop.call_soon_threadsafe(business_terminal.set)

            try:
                registered = self.gateway.register_attempt(
                    self.store, attempt_id, executor, allowed_tools=child_tools,
                    max_tool_calls=96,
                    tool_call_limits={
                        "inspect_knowledge_scope": 1,
                        "inspect_task_outline": 1,
                        "read_task_node": 48,
                        "read_task_nodes": 12,
                        "list_evidence": 8,
                        "search_evidence": 12,
                        "read_evidence": 32,
                        "view_evidence_image": 6,
                    },
                    budget_exhaustion_as_skip=True, keep_alive=True,
                    on_tool_result=on_tool_result,
                )
                model = self.model_config.model_dump(mode="json")
                prompt = (UNIFIED_PROMPT + "\n\nTARGET PROFILE\n"
                          + json.dumps(target_profile.model_dump(mode="json"), ensure_ascii=False)
                          + f"\nWrite generated Task text and Trajectory actions and observations in {self.settings.skill_output_language}. Copy Task category exactly from the source directory, preserving its original language. Preserve exact software syntax and source quotes."
                          + f"\nThere are {len(corpus.evidence)} Evidence excerpts and "
                          + f"{len(references)} validated References."
                          + "\nPreviously accepted pairs on recovery: "
                          + json.dumps(workspace.inspect_pair_coverage(), ensure_ascii=False))
                invocation = AgentInvocation(
                    run_id=self.store.run_id, session_id=capsule.child_session_id,
                    role="task_trajectory", model=model, prompt=prompt,
                    allowed_tools=capsule.allowed_tools,
                    metadata={"capability": registered.runtime_metadata(
                        self.gateway_server.endpoint_url),
                        "runtime_event_sink": make_runtime_event_sink(
                            self.store, agent="task_trajectory", attempt_id=attempt.attempt_id)},
                )
                turn_task = asyncio.create_task(self.runtime.run(invocation))
                terminal_wait = asyncio.create_task(business_terminal.wait())
                try:
                    await asyncio.wait({turn_task, terminal_wait},
                                       return_when=asyncio.FIRST_COMPLETED)
                    if business_terminal.is_set() and not turn_task.done():
                        await self.runtime.stop_completed_session(capsule.child_session_id)
                        try:
                            turn = await asyncio.wait_for(turn_task, timeout=5.0)
                        except Exception:
                            turn = None
                    else:
                        turn = await turn_task
                finally:
                    if not terminal_wait.done():
                        terminal_wait.cancel()
                    await asyncio.gather(terminal_wait, return_exceptions=True)
                    if not turn_task.done():
                        turn_task.cancel()
                        await asyncio.gather(turn_task, return_exceptions=True)
                if turn is not None:
                    for key, value in dict(turn.usage).items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            usage[key] = usage.get(key, 0) + value
                terminal = self.store.read_task_trajectory_terminal()
                if terminal is None:
                    workspace.submit_stage_result("degraded", "Task-Trajectory Child ended without a stage result.",
                        gaps=["The Child session ended before reporting complete exploration."])
                    terminal = self.store.read_task_trajectory_terminal()
                self.store.finish_attempt(attempt_id, status="succeeded", usage=usage)
            except asyncio.CancelledError:
                terminal = self.store.read_task_trajectory_terminal()
                if terminal is not None:
                    self.store.finish_attempt(attempt_id, status="succeeded", usage=usage,
                                              termination_reason="Business terminal accepted before cancellation")
                else:
                    self.store.finish_attempt(attempt_id, status="cancelled", usage=usage,
                                              error_category="cancelled")
                    raise
            except Exception as exc:
                terminal = self.store.read_task_trajectory_terminal()
                if terminal is not None:
                    self.store.finish_attempt(attempt_id, status="succeeded", usage=usage,
                                              termination_reason="Business terminal accepted before runtime error")
                else:
                    self.store.finish_attempt(attempt_id, status="failed", usage=usage,
                        error_category="task_trajectory_failed", termination_reason=str(exc)[:2000])
                    raise
            finally:
                if registered is not None:
                    self.gateway.revoke(registered, reason="attempt_finished")
                closer = getattr(self.runtime, "close_session", None)
                if closer is not None:
                    result = closer(capsule.child_session_id)
                    if inspect.isawaitable(result):
                        await result
        assert terminal is not None
        if (terminal.get("corpus_snapshot_artifact_id") != corpus_artifact_id
                or terminal.get("reference_artifact_id") != reference_artifact_id
                or terminal.get("run_id") != self.store.run_id
                or terminal.get("result", {}).get("status") == "failed"):
            raise RuntimeError("Unified terminal does not match inputs or reported failure")
        return self._finalize(
            target_profile=target_profile, corpus=corpus,
            corpus_artifact_id=corpus_artifact_id, references=references,
            reference_artifact_id=reference_artifact_id,
            stage_result=terminal["result"], attempt_id=attempt_id,
        )

    def _finalize(self, *, target_profile: TargetProfile, corpus: CorpusSnapshot,
                  corpus_artifact_id: str, references: list[ReferenceCandidate],
                  reference_artifact_id: str, stage_result: dict[str, Any],
                  attempt_id: str | None) -> UnifiedArtifacts:
        evidence = {item.evidence_id: item for item in corpus.evidence}
        pairs = self.store.list_task_trajectory_pairs()
        tasks: list[TaskCandidate] = []
        drafts: list[TrajectoryDraft] = []
        for checkpoint in pairs:
            if (checkpoint.get("schema_version") != "task-trajectory-pair-checkpoint.v1"
                    or checkpoint.get("corpus_snapshot_artifact_id") != corpus_artifact_id
                    or checkpoint.get("reference_artifact_id") != reference_artifact_id
                    or checkpoint.get("prompt_version") not in SUPPORTED_UNIFIED_PROMPT_VERSIONS):
                raise RuntimeError("Task-Trajectory checkpoint input mismatch")
            task = TaskCandidate.model_validate(checkpoint["task"])
            draft = TrajectoryDraft.model_validate(normalize_saved_trajectory_payload(checkpoint["trajectory"]))
            if not _pair_digest_matches(checkpoint, task, draft):
                raise RuntimeError("Task-Trajectory checkpoint content mismatch")
            attempt = self.store.read_attempt(checkpoint["attempt_id"])
            if attempt.stage != "task_trajectory" or draft.task_id != task.task_id:
                raise RuntimeError("Task-Trajectory checkpoint Attempt mismatch")
            if checkpoint["superseded_by"] is None:
                tasks.append(task)
                drafts.append(draft)
        if len(tasks) != stage_result.get("candidate_count"):
            raise RuntimeError("Task-Trajectory terminal count mismatch")
        accepted_tasks, task_validation = validate_tasks(tasks, evidence, corpus.evidence_trees)
        if len(accepted_tasks) != len(tasks):
            raise RuntimeError("Task-Trajectory final Task validation failed")
        frozen = freeze_knowledge(
            store=self.store, target_profile=target_profile, corpus=corpus,
            corpus_artifact_id=corpus_artifact_id,
            tasks=accepted_tasks, references=references,
        )
        trajectories: list[TrajectoryCandidate] = []
        for draft in drafts:
            identity = [frozen.knowledge.knowledge_snapshot_id, draft.task_id,
                        *sorted(c.evidence_id for c in draft.citations)]
            digest = hashlib.sha256("\x1f".join(identity).encode("utf-8")).hexdigest()
            trajectory_id = self.store.canonical_id(
                "TR", f"traj_{digest[:20]}", "traj", *identity,
            )
            trajectories.append(TrajectoryCandidate(
                **draft.model_dump(mode="json"), trajectory_id=trajectory_id,
                knowledge_snapshot_id=frozen.knowledge.knowledge_snapshot_id,
            ))
        accepted, trajectory_validation = validate_trajectories(
            trajectories, evidence,
            {item.task_id: item for item in frozen.knowledge.tasks},
        )
        if len(accepted) != len(trajectories):
            raise RuntimeError("Task-Trajectory final Trajectory validation failed")
        task_artifact = self.store.put_json_artifact(
            {"schema_version": "validated-task-batch.v1",
             "corpus_snapshot_id": corpus.corpus_snapshot_id,
             "tasks": [item.model_dump(mode="json") for item in frozen.knowledge.tasks]},
            kind="validated_task_batch", metadata={"stage": "task_trajectory"},
            preserve_text_integrity=True,
        )
        task_validation_artifact = self.store.put_json_artifact(
            {"schema_version": "task-validation.v1", "candidate_count": len(tasks),
             "accepted_count": len(tasks),
             "validation": [item.model_dump(mode="json") for item in task_validation]},
            kind="task_validation_report", metadata={"stage": "task_trajectory"},
            preserve_text_integrity=True,
        )
        trajectory_artifact = self.store.put_json_artifact(
            {"schema_version": "trajectory-candidate-batch.v1",
             "trajectories": [item.model_dump(mode="json") for item in accepted],
             "stage_results": [stage_result]},
            kind="trajectory_candidate_batch", metadata={"stage": "task_trajectory"},
            preserve_text_integrity=True,
        )
        trajectory_validation_artifact = self.store.put_json_artifact(
            {"schema_version": "trajectory-validation.v1", "candidate_count": len(accepted),
             "accepted_count": len(accepted),
             "validation": [item.model_dump(mode="json") for item in trajectory_validation]},
            kind="trajectory_validation_report", metadata={"stage": "task_trajectory"},
            preserve_text_integrity=True,
        )
        return UnifiedArtifacts(
            frozen.knowledge, frozen.knowledge_artifact_id,
            frozen.validation_artifact_id, tuple(frozen.knowledge.tasks), tuple(accepted),
            task_validation_artifact.artifact_id, task_artifact.artifact_id,
            trajectory_validation_artifact.artifact_id, trajectory_artifact.artifact_id,
            stage_result, attempt_id,
        )
