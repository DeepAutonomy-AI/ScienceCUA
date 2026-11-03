from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.capabilities import CapabilityGateway, CapabilityInvocation
from app.config import Settings
from app.knowledge import (
    KnowledgeToolExecutor, KnowledgeWorkspace, SkillRegistry, build_pipeline_quality_report,
    load_corpus_snapshot, load_knowledge_snapshot,
)
from app.models import (
    CorpusSnapshot, EvidenceCitation, EvidenceExcerpt, KnowledgeSnapshot,
    ModelRuntimeConfig, ReferenceCandidate, SourceResult, TargetProfile, TaskCandidate,
    TrajectoryCandidate,
)
from app.orchestration.downstream_stage import DownstreamStageProtocolError, SkillInductionService
from app.runtime import (
    AgentInvocation, AgentRuntime, AgentRuntimeBusinessCompleted,
    AgentRuntimeCancelledError, AgentTurnResult,
)
from app.storage import RunStore
from tests.knowledge_fixtures import trajectory_draft


QUOTE = "Press Ctrl+R to render the graph."


class SkillTerminalRuntime(AgentRuntime):
    def __init__(self, store: RunStore, gateway: CapabilityGateway, *,
                 finish_in_reminder: bool = False, mode: str = "linger",
                 status: str = "succeeded") -> None:
        self.store, self.gateway = store, gateway
        self.finish_in_reminder, self.mode, self.status = finish_in_reminder, mode, status
        self.slot = asyncio.Semaphore(1)
        self.stops: dict[str, asyncio.Event] = {}
        self.stopped: list[str] = []
        self.closed: list[str] = []
        self.turn_kinds: list[str] = []
        self.dispatches: dict[str, dict] = {}
        self.allowed_tools: dict[str, set[str]] = {}
        self.candidate_submitted = asyncio.Event()
        self.allow_terminal = asyncio.Event()
        self.stop_started = asyncio.Event()
        self.calls = 0
        self.released = 0

    def call(self, invocation: AgentInvocation, name: str, arguments: dict) -> dict:
        self.calls += 1
        capability = invocation.metadata["capability"]
        return self.gateway.invoke(
            capability["capability_token"],
            CapabilityInvocation(
                **capability["context"], tool_name=name, arguments=arguments,
                call_id=f"skill-test.{self.calls}",
            ),
        )

    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        async with self.slot:
            try:
                kind = str(invocation.metadata["runtime_turn_kind"])
                self.turn_kinds.append(kind)
                if kind == "stage_work":
                    self.allowed_tools[invocation.session_id] = set(invocation.allowed_tools)
                    self.dispatches[invocation.session_id] = json.loads(
                        str(invocation.prompt).split("DISPATCHED BATCH\n", 1)[1]
                    )
                if self.finish_in_reminder and kind == "stage_work":
                    return AgentTurnResult(invocation.session_id, "incomplete", "completed", 1)
                self.stops[invocation.session_id] = asyncio.Event()
                context = invocation.metadata["capability"]["context"]
                attempt = self.store.read_attempt(context["attempt_id"])
                scope = self.store.read_json_artifact(attempt.capsule_artifact_id)["scope"]
                if self.mode == "rejected":
                    rejected = self.call(invocation, "submit_stage_result", {
                        "status": "succeeded",
                    })
                    assert rejected["ok"] is False
                    raise RuntimeError("provider failed after rejected terminal")
                if self.mode == "excluded":
                    reviewed = self.call(invocation, "review_skill_coverage", {
                        "dispositions": [{
                            "trajectory_id": trajectory_id, "disposition": "excluded",
                            "rationale": "No reusable operating unit can be induced in this scope.",
                        } for trajectory_id in scope["trajectory_ids"]],
                    })
                    assert not reviewed["coverage"]["uncovered_trajectory_ids"], reviewed
                else:
                    dispatch = self.dispatches[invocation.session_id]
                    citation = {"evidence_id": "E1", "quote": QUOTE}
                    submitted = self.call(invocation, "submit_skill_batch", {"skills": [{
                        "title": f"Render graph for {trajectory['task_id']}",
                        "description": "Render the current graph.",
                        "applicability": "When a graph is ready to render.",
                        "preconditions": "The graph view is open.",
                        "procedure": [{
                            "ordinal": 1, "instruction": QUOTE,
                            "citations": [citation],
                        }],
                        "trajectory_ids": [trajectory["trajectory_id"]],
                    } for trajectory in dispatch["trajectories"]]})
                    assert submitted["accepted_by_schema"] == len(scope["trajectory_ids"]), submitted
                self.candidate_submitted.set()
                if self.mode == "gate":
                    await self.allow_terminal.wait()
                elif self.mode == "before_terminal_error":
                    raise RuntimeError("provider failed before terminal")
                terminal = self.call(invocation, "submit_stage_result", {
                    "status": self.status,
                })
                assert terminal["accepted"] is True, terminal
                if self.mode == "error":
                    raise RuntimeError("EMPTY_RESPONSE after accepted skill result")
                if self.mode == "normal":
                    return AgentTurnResult(invocation.session_id, "ok", "completed", 1)
                await self.stops[invocation.session_id].wait()
                if self.mode == "cancelled":
                    raise AgentRuntimeCancelledError("worker stopped after business completion")
                raise AgentRuntimeBusinessCompleted("worker stopped after business completion")
            finally:
                self.released += 1

    async def stop_completed_session(self, session_id: str) -> bool:
        # The business payload must be readable from disk before stopping model work.
        accepted = [event for event in self.store.read_events(limit=1000)[0]
                    if event["event_type"] == "skill_business_terminal_accepted"]
        assert accepted
        terminal = self.store.read_json_artifact(accepted[-1]["payload"]["terminal_artifact_id"])
        assert terminal["stage_result"]["status"] == self.status
        self.stopped.append(session_id)
        self.stop_started.set()
        if self.mode == "slow_stop":
            await asyncio.Event().wait()
        self.stops[session_id].set()
        return True

    async def cancel(self, session_id: str) -> bool:
        self.stops[session_id].set()
        return True

    async def close_session(self, session_id: str) -> bool:
        self.closed.append(session_id)
        return True


def setup_skill_run(tmp_path: Path, count: int = 1, *, with_references: bool = False):
    settings = Settings(data_dir=tmp_path / "data", skill_induction_concurrency=5)
    store = RunStore(settings.data_dir, "run_skill_terminal")
    target = TargetProfile(software_name="Example", target_version="1",
                           platform="General", ui_locale="English")
    store.initialize({"target_profile": target.model_dump(mode="json")})
    evidence = EvidenceExcerpt(
        evidence_id="E1", source_id="SRC1", source_part_id="part_one",
        kind="instruction", title="Render", content=QUOTE, start_offset=0,
        end_offset=len(QUOTE), extraction_method="test",
        text_artifact_id="art_000000000000000000000000", content_sha256="2" * 64,
        locator={},
    )
    corpus = CorpusSnapshot(
        corpus_snapshot_id="CP1", run_id=store.run_id, target_profile=target,
        source_results=[SourceResult(source_id="SRC1", display_name="manual", modality="html",
                                     status="succeeded", evidence=[evidence], source_part_count=1)],
        evidence=[evidence], evidence_count=1, source_part_count=1,
        source_status_counts={"succeeded": 1},
    )
    corpus_id = store.put_json_artifact(corpus.model_dump(mode="json"), kind="corpus_snapshot").artifact_id
    citation = EvidenceCitation(evidence_id="E1", quote=QUOTE)
    tasks = [TaskCandidate(task_id=f'T{index}', corpus_snapshot_id='CP1', task=f'Render graph {index}', citations=[citation], category='Operations') for index in range(1, count + 1)]
    knowledge = KnowledgeSnapshot(
        knowledge_snapshot_id="KP1", run_id=store.run_id, corpus_snapshot_id="CP1",
        corpus_snapshot_artifact_id=corpus_id, target_profile=target, tasks=tasks,
        references=[ReferenceCandidate(
            reference_id="R1", corpus_snapshot_id="CP1", title="Graph rendering guidance",
            content=QUOTE, citations=[citation],
        )] if with_references else [],
    )
    knowledge_id = store.put_json_artifact(knowledge.model_dump(mode="json"),
                                         kind="knowledge_snapshot").artifact_id
    trajectories = [TrajectoryCandidate(
        trajectory_id=f"TR{index}", knowledge_snapshot_id="KP1",
        **trajectory_draft(task_id=task.task_id, instruction=QUOTE,
                           citations=[citation.model_dump(mode="json")],
                           expected_observation="The graph is rendered."),
    ) for index, task in enumerate(tasks, 1)]
    gateway = CapabilityGateway()

    async def induce(runtime: AgentRuntime):
        return await SkillInductionService(
            store=store, settings=settings, runtime=runtime, gateway=gateway,
            gateway_server=type("Server", (), {"endpoint_url": "http://127.0.0.1:1"})(),
            model_config=ModelRuntimeConfig(base_url="https://example.invalid/v1", model="fake"),
        ).induce(target_profile=target, corpus=corpus, corpus_artifact_id=corpus_id,
                 knowledge=knowledge, knowledge_artifact_id=knowledge_id, trajectories=trajectories)

    return store, gateway, induce


@pytest.mark.parametrize("status", ["succeeded", "degraded"])
@pytest.mark.parametrize("mode", ["linger", "cancelled", "error"])
async def test_saved_skill_result_survives_runtime_stop_or_failure(tmp_path, status, mode):
    store, gateway, induce = setup_skill_run(tmp_path)
    runtime = SkillTerminalRuntime(store, gateway, mode=mode, status=status)
    result = await asyncio.wait_for(induce(runtime), timeout=2)
    assert len(result.skills) == 1
    assert result.stage_result["status"] == status
    assert runtime.turn_kinds == ["stage_work"]
    assert runtime.released == 1 and not runtime.slot.locked()
    assert len(runtime.closed) == 1
    attempt = store.read_attempt(result.attempt_ids[0])
    assert attempt.status == "succeeded" and attempt.error_category is None
    log = store.read_json_artifact(store.read_session(attempt.child_session_id).dsh_log_artifact_id)
    assert log["finish_reason"] == "business_complete"
    assert log["usage_complete"] is False
    if mode == "error":
        assert "EMPTY_RESPONSE" in log["runtime_stop_detail"]
    else:
        assert runtime.stopped == [attempt.child_session_id]
    terminal = store.read_json_artifact(log["terminal_artifact_id"])
    assert terminal["scope"]["trajectory_ids"] == ["TR1"]
    assert terminal["stage_result"] == result.stage_result["batches"][0]
    assert terminal["stage_result"]["summary"] == (
        f"Skill induction {status}: 1 Skill candidate(s), 1 covered and 0 excluded Trajectory(s)."
    )
    assert terminal["skills"] == [dict(item.model_dump(mode="json"), status="candidate")
                                  for item in result.skills]


async def test_skill_completion_in_reminder_stops_same_session(tmp_path):
    store, gateway, induce = setup_skill_run(tmp_path)
    runtime = SkillTerminalRuntime(store, gateway, finish_in_reminder=True)
    result = await asyncio.wait_for(induce(runtime), timeout=2)
    assert runtime.turn_kinds == ["stage_work", "protocol_completion_reminder"]
    assert len(result.skills) == 1 and len(runtime.stopped) == 1
    attempt = store.read_attempt(result.attempt_ids[0])
    log = store.read_json_artifact(store.read_session(attempt.child_session_id).dsh_log_artifact_id)
    assert log["turns"][-1]["kind"] == "protocol_completion_reminder"


async def test_each_completed_batch_releases_slot_for_next_child(tmp_path):
    store, gateway, induce = setup_skill_run(tmp_path, count=3)
    runtime = SkillTerminalRuntime(store, gateway)
    result = await asyncio.wait_for(induce(runtime), timeout=2)
    assert len(result.skills) == 3
    assert len(set(runtime.stopped)) == 3 and runtime.released == 3
    assert runtime.turn_kinds == ["stage_work"] * 3
    assert not runtime.slot.locked()


async def test_candidate_submission_alone_does_not_stop_child(tmp_path):
    store, gateway, induce = setup_skill_run(tmp_path)
    runtime = SkillTerminalRuntime(store, gateway, mode="gate")
    task = asyncio.create_task(induce(runtime))
    try:
        await asyncio.wait_for(runtime.candidate_submitted.wait(), timeout=1)
        await asyncio.sleep(0)
        assert not task.done() and not runtime.stopped
        assert runtime.slot.locked()
        runtime.allow_terminal.set()
        result = await asyncio.wait_for(task, timeout=2)
        assert len(result.skills) == 1 and len(runtime.stopped) == 1
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("mode", ["rejected", "before_terminal_error"])
async def test_unaccepted_stage_result_does_not_hide_runtime_failure(tmp_path, mode):
    store, gateway, induce = setup_skill_run(tmp_path)
    runtime = SkillTerminalRuntime(store, gateway, mode=mode)
    with pytest.raises(RuntimeError, match="provider failed"):
        await induce(runtime)
    assert not runtime.stopped
    assert store.list_attempts()[0].status == "failed"
    assert not any(e["event_type"] == "skill_business_terminal_accepted"
                   for e in store.read_events(limit=1000)[0])


async def test_failure_to_save_terminal_does_not_mark_business_complete(tmp_path, monkeypatch):
    store, gateway, induce = setup_skill_run(tmp_path)
    runtime = SkillTerminalRuntime(store, gateway, mode="error")
    original = store.put_json_artifact

    def save(value, **kwargs):
        if kwargs["kind"] == "skill_batch_terminal":
            raise OSError("terminal disk write failed")
        return original(value, **kwargs)

    monkeypatch.setattr(store, "put_json_artifact", save)
    with pytest.raises(RuntimeError, match="EMPTY_RESPONSE"):
        await induce(runtime)
    assert not runtime.stopped
    assert store.list_attempts()[0].status == "failed"


async def test_explicit_exclusion_can_finish_without_skills(tmp_path):
    store, gateway, induce = setup_skill_run(tmp_path)
    runtime = SkillTerminalRuntime(store, gateway, mode="excluded", status="degraded")
    result = await asyncio.wait_for(induce(runtime), timeout=2)
    assert result.skills == () and result.stage_result["status"] == "degraded"
    assert len(runtime.stopped) == 1
    assert result.stage_result["coverage"]["excluded"][0]["trajectory_id"] == "TR1"


@pytest.mark.parametrize("mode", ["linger", "error"])
async def test_saved_failed_result_stops_child_and_preserves_business_failure(tmp_path, mode):
    store, gateway, induce = setup_skill_run(tmp_path)
    runtime = SkillTerminalRuntime(store, gateway, mode=mode, status="failed")
    with pytest.raises(DownstreamStageProtocolError, match="Child reported failure: Skill induction failed"):
        await asyncio.wait_for(induce(runtime), timeout=2)
    attempt = store.list_attempts()[0]
    assert attempt.status == "failed" and attempt.error_category == "skill_induction_failed"
    session = store.read_session(attempt.child_session_id)
    assert "Child reported failure" in session.termination_reason
    log = store.read_json_artifact(session.dsh_log_artifact_id)
    terminal = store.read_json_artifact(log["terminal_artifact_id"])
    assert terminal["stage_result"]["status"] == "failed"
    assert not runtime.slot.locked() and runtime.released == 1
    assert runtime.turn_kinds == ["stage_work"]


async def test_run_cancellation_before_terminal_keeps_attempt_cancelled(tmp_path):
    store, gateway, induce = setup_skill_run(tmp_path)
    runtime = SkillTerminalRuntime(store, gateway, mode="gate")
    task = asyncio.create_task(induce(runtime))
    await asyncio.wait_for(runtime.candidate_submitted.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.list_attempts()[0].status == "cancelled"
    assert not runtime.stopped and not runtime.slot.locked()


async def test_run_cancellation_after_terminal_keeps_completed_attempt(tmp_path):
    store, gateway, induce = setup_skill_run(tmp_path)
    runtime = SkillTerminalRuntime(store, gateway, mode="slow_stop")
    task = asyncio.create_task(induce(runtime))
    await asyncio.wait_for(runtime.stop_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    attempt = store.list_attempts()[0]
    assert attempt.status == "succeeded"
    assert not runtime.slot.locked()
    log = store.read_json_artifact(store.read_session(attempt.child_session_id).dsh_log_artifact_id)
    assert log["finish_reason"] == "business_complete"


async def test_cleanup_error_after_saved_result_does_not_fail_business(tmp_path, monkeypatch):
    store, gateway, induce = setup_skill_run(tmp_path)
    runtime = SkillTerminalRuntime(store, gateway)

    async def close(session_id):
        raise RuntimeError("worker pipe closed during cleanup")

    monkeypatch.setattr(runtime, "close_session", close)
    result = await asyncio.wait_for(induce(runtime), timeout=2)
    assert result.stage_result["status"] == "succeeded" and len(result.skills) == 1
    assert store.read_attempt(result.attempt_ids[0]).status == "succeeded"
    assert not runtime.slot.locked()
    assert any(e["event_type"] == "skill_runtime_cleanup_failed_after_completion"
               for e in store.read_events(limit=1000)[0])


async def test_duplicate_accepted_skills_still_use_existing_aggregate_deduplication(tmp_path, monkeypatch):
    store, gateway, induce = setup_skill_run(tmp_path)
    runtime = SkillTerminalRuntime(store, gateway)
    original = runtime.call

    def repeat(invocation, name, arguments):
        result = original(invocation, name, arguments)
        if name == "submit_skill_batch":
            assert original(invocation, name, arguments)["accepted_by_schema"] == 1
        return result

    monkeypatch.setattr(runtime, "call", repeat)
    result = await asyncio.wait_for(induce(runtime), timeout=2)
    assert len(result.skills) == 1 and result.stage_result["status"] == "succeeded"
    assert len(runtime.stopped) == 1
    attempt = store.read_attempt(result.attempt_ids[0])
    log = store.read_json_artifact(store.read_session(attempt.child_session_id).dsh_log_artifact_id)
    terminal = store.read_json_artifact(log["terminal_artifact_id"])
    assert len(terminal["skills"]) == 2


async def test_skill_induction_and_exports_do_not_inherit_required_reference_links(tmp_path):
    store, gateway, induce = setup_skill_run(tmp_path, with_references=True)
    runtime = SkillTerminalRuntime(store, gateway, mode="normal")
    result = await asyncio.wait_for(induce(runtime), timeout=2)
    assert len(result.skills) == 1 and result.stage_result["status"] == "succeeded"
    dispatch = next(iter(runtime.dispatches.values()))
    for trajectory in dispatch["trajectories"]:
        assert "reference_ids" not in trajectory
        assert "trajectory_type" not in trajectory
        assert "reference_trigger_type" not in trajectory
    reference_tools = {"list_references", "read_reference", "search_reference_entries", "read_reference_entry"}
    assert all(not tools & reference_tools for tools in runtime.allowed_tools.values())
    assert "available_reference_count" not in dispatch["scope"]

    attempt = store.read_attempt(result.attempt_ids[0])
    capsule = store.read_json_artifact(attempt.capsule_artifact_id)
    corpus = load_corpus_snapshot(store, capsule["input_artifact_ids"][0])
    knowledge = load_knowledge_snapshot(store, capsule["input_artifact_ids"][1])
    trajectories = [TrajectoryCandidate.model_validate({
        **item, "trajectory_type": "reference_guided", "reference_ids": ["R1"], "reference_trigger_type": "required",
    }) for item in dispatch["trajectories"]]
    quality = build_pipeline_quality_report(
        corpus, knowledge, trajectories, list(result.skills), list(result.validation),
    )
    assert not any("reference" in key for key in quality["skill_contract"])
    registry = SkillRegistry(tmp_path / "exports")
    exported = registry.commit(
        run_id=store.run_id, target_profile=knowledge.target_profile, corpus=corpus,
        knowledge=knowledge, trajectories=trajectories, skills=list(result.skills),
        validations=list(result.validation), quality_report=quality,
    )
    payload = json.loads(Path(exported["skills_file"]).read_text(encoding="utf-8"))
    assert payload["references"][0]["reference_id"] == "R1"
    for field in ("trajectory_type", "reference_ids", "reference_trigger_type", "confidence", "caveats", "unsupported_facets"):
        assert field not in payload["trajectories"][0]
    assert "reference_ids" not in payload["skills"][0]
    assert "reference_trigger_type" not in payload["skills"][0]
    markdown = Path(exported["audit_directory"]) / "skills" / f"{result.skills[0].skill_id}.md"
    agent_markdown = Path(exported["agent_skill_entries"][0]["entry_file"])
    for path in [markdown, agent_markdown]:
        text = path.read_text(encoding="utf-8")
        assert "References" not in text and "reference_trigger_type" not in text
        assert "references/" not in text and "`R1`" not in text


async def test_excluded_trajectory_does_not_generate_a_fallback_skill(tmp_path):
    store, gateway, induce = setup_skill_run(tmp_path, with_references=True)
    runtime = SkillTerminalRuntime(store, gateway, mode="excluded", status="degraded")
    result = await asyncio.wait_for(induce(runtime), timeout=2)
    assert not result.skills
    assert result.stage_result["candidate_count"] == 0
    assert result.stage_result["coverage"]["excluded"][0]["trajectory_id"] == "TR1"


def test_only_skill_terminal_tool_omits_report_fields():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from app.knowledge_components.tool_catalog import _knowledge_tool_catalog

    for stage in ["skill_induction", "task_trajectory", "trajectory_inference"]:
        workspace = MagicMock(spec=KnowledgeWorkspace)
        workspace.stage, workspace.store, workspace.settings = stage, None, None
        workspace.evidence_by_id = {}
        workspace.corpus = SimpleNamespace(evidence_trees=[], evidence=[])
        executor = KnowledgeToolExecutor(workspace, "test")
        specs = {s["function"]["name"]: s["function"] for s in executor.tool_specs()}
        terminal = specs["submit_stage_result"]
        if stage == "skill_induction":
            assert terminal["parameters"]["required"] == ["status"]
            assert set(terminal["parameters"]["properties"]) == {"status"}
            assert terminal["parameters"]["additionalProperties"] is False
            assert terminal["parameters"]["properties"]["status"]["enum"] == [
                "succeeded", "degraded", "failed",
            ]
            assert '"status":"succeeded"' in terminal["description"]
            assert "do not write a summary" in terminal["description"]
            assert "automatically on acceptance" in terminal["description"]
        else:
            assert terminal["parameters"]["required"] == ["status", "summary"]
            with pytest.raises(ValueError, match="summary is required"):
                KnowledgeWorkspace.submit_stage_result(workspace, status="succeeded")
    assert _knowledge_tool_catalog()["submit_stage_result"]["function"]["parameters"]["required"] == [
        "status", "summary",
    ]
