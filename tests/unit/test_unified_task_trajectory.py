from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.capabilities import CapabilityGateway, CapabilityInvocation, tools_for_stage
from app.config import Settings
from app.contracts import StageReceipt
from app.knowledge import KnowledgeToolExecutor
from app.models import (
    CorpusSnapshot, EvidenceExcerpt, EvidenceNode, EvidenceSourceSpan,
    EvidenceTree, ModelRuntimeConfig, SourceResult, TargetProfile,
)
from app.orchestration.pipeline import PipelineRuntime
from app.orchestration.state import PipelineStateMachine, RunPhase
from app.orchestration.unified_stage import (
    UNIFIED_PROMPT_VERSION, UnifiedKnowledgeWorkspace, UnifiedTaskTrajectoryService,
)
from app.runtime import AgentInvocation, AgentRuntime, AgentTurnResult
from app.storage import RunStore
from task_model_helpers import first_level_tree
from tests.knowledge_fixtures import trajectory_submission


QUOTE = "Press Ctrl+R to render the graph."


def setup(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "out")
    store = RunStore(settings.data_dir, "run_unified")
    target = TargetProfile(software_name="Example", target_version="1.0",
                           platform="General", ui_locale="English")
    store.initialize({"target_profile": target.model_dump(mode="json"), "inputs": [],
                      "model": {"mode": "api", "model": "fake",
                                "base_url": "https://example.invalid/v1"}})
    evidence = EvidenceExcerpt(
        evidence_id="evi_one", source_id="src_one", source_part_id="part_one",
        kind="instruction", title="Render", content=QUOTE, start_offset=0,
        end_offset=len(QUOTE), extraction_method="deterministic_test",
        text_artifact_id="art_000000000000000000000000",
        content_sha256="2" * 64, locator={},
    )
    source = SourceResult(source_id="src_one", display_name="manual", modality="html",
                          status="succeeded", evidence=[evidence], source_part_count=1)
    corpus = CorpusSnapshot(corpus_snapshot_id="corpus_one", run_id=store.run_id,
                            target_profile=target, source_results=[source],
                            evidence=[evidence], evidence_count=1, source_part_count=1,
                            evidence_trees=[first_level_tree(evidence)],
                            source_status_counts={"succeeded": 1})
    corpus_id = store.put_json_artifact(corpus.model_dump(mode="json"),
                                        kind="corpus_snapshot").artifact_id
    reference_id = store.put_json_artifact(
        {"schema_version": "validated-reference-batch.v1", "references": []},
        kind="validated_reference_batch").artifact_id
    gateway = CapabilityGateway()
    server = type("Server", (), {"endpoint_url": "http://127.0.0.1:1"})()
    config = ModelRuntimeConfig(base_url="https://example.invalid/v1", model="fake",
                                api_key="fake", reasoning_effort="medium")
    return store, settings, target, corpus, corpus_id, reference_id, gateway, server, config


def render_pair(*, title: str = "Render a graph", evidence_ids: tuple[str, ...] = ("evi_one",)) -> dict:
    citations = [{"evidence_id": evidence_id, "quote": QUOTE} for evidence_id in evidence_ids]
    return {
        "task": {"task": title, "category": "Operations",
                 "citations": citations},
        "trajectory": trajectory_submission(instruction=QUOTE, citations=citations,
                                            expected_observation="The graph is rendered."),
    }


class PairedRuntime(AgentRuntime):
    def __init__(self, gateway: CapabilityGateway, *, fail_after_pair: bool = False,
                 fail_after_terminal: bool = False):
        self.gateway = gateway
        self.fail_after_pair = fail_after_pair
        self.fail_after_terminal = fail_after_terminal
        self.sessions: list[str] = []
        self.calls = 0

    def call(self, invocation: AgentInvocation, name: str, arguments: dict) -> dict:
        self.calls += 1
        capability = invocation.metadata["capability"]
        return self.gateway.invoke(
            capability["capability_token"],
            CapabilityInvocation(**capability["context"], tool_name=name,
                                 arguments=arguments, call_id=f"pair.{self.calls}"),
        )

    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        assert callable(invocation.metadata.get("runtime_event_sink"))
        self.sessions.append(invocation.session_id)
        assert self.call(invocation, "inspect_knowledge_scope", {})["ok"]
        assert self.call(invocation, "inspect_task_outline", {})["ok"]
        assert self.call(invocation, "read_task_node", {"node_id": "N2"})["ok"]
        pair = self.call(invocation, "submit_task_trajectory_pair", render_pair())
        assert pair["accepted"] is True, pair
        if self.fail_after_pair:
            raise RuntimeError("provider stopped before stage terminal")
        coverage = self.call(invocation, "inspect_pair_coverage", {})
        assert coverage["accepted_pair_count"] == 1
        terminal = self.call(invocation, "submit_stage_result", {
            "status": "succeeded", "summary": "One grounded pair."})
        assert terminal["accepted"] is True, terminal
        if self.fail_after_terminal:
            raise RuntimeError("provider failed after business terminal")
        return AgentTurnResult(invocation.session_id, "ok", "completed", 1)

    async def cancel(self, session_id: str) -> bool:
        return True


class RecoveryRuntime(PairedRuntime):
    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        self.sessions.append(invocation.session_id)
        assert self.call(invocation, "inspect_task_outline", {})["ok"]
        assert self.call(invocation, "read_task_node", {"node_id": "N2"})["ok"]
        coverage = self.call(invocation, "inspect_pair_coverage", {})
        assert coverage["accepted_pair_count"] == 1
        terminal = self.call(invocation, "submit_stage_result", {
            "status": "succeeded", "summary": "Recovered the accepted pair."})
        assert terminal["accepted"] is True, terminal
        return AgentTurnResult(invocation.session_id, "ok", "completed", 1)


class PipelineRuntimeDriver(PairedRuntime):
    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        if invocation.role == "task_trajectory":
            return await super().run(invocation)
        assert invocation.role == "skill_induction"
        self.calls += 1
        if "DISPATCHED BATCH\n" in str(invocation.prompt):
            import json
            dispatch = json.loads(str(invocation.prompt).split("DISPATCHED BATCH\n", 1)[1])
            assert dispatch["trajectory_evidence"][0]["content"] == QUOTE
            assert "evidence_tree" in dispatch
            trajectory_id = dispatch["trajectories"][0]["trajectory_id"]
            citation = {"evidence_id": "evi_one", "quote": QUOTE}
            result = self.call(invocation, "submit_skill_batch", {"skills": [{
                "title": "Render a graph", "description": "Render the current graph.",
                "applicability": "When a graph is ready to render.",
                "preconditions": "The graph view is open.",
                "procedure": [{"ordinal": 1, "instruction": QUOTE,
                               "citations": [citation]}],
                "trajectory_ids": [trajectory_id],
            }]})
            assert result["accepted_by_schema"] == 1, result
        else:
            result = self.call(invocation, "submit_stage_result", {
                "status": "succeeded", "summary": "The trajectory is covered."})
            assert result["accepted"] is True, result
        return AgentTurnResult(invocation.session_id, "ok", "completed", 1)


def test_pair_submission_is_atomic_and_recovery_is_inspectable(tmp_path: Path) -> None:
    store, settings, target, corpus, corpus_id, reference_id, gateway, _, _ = setup(tmp_path)
    store.commit_stage_receipt(StageReceipt(
        receipt_id="rcpt_unified_sync", idempotency_key="unified.sync",
        run_id=store.run_id, stage="test_setup", status="succeeded",
        input_fingerprint="1" * 64,
        accepted_output_artifact_ids=(corpus_id, reference_id),
    ), expected_phase="created", next_phase="discovering_task_trajectories")
    store.update_state(task_trajectory_mode="unified",
                       corpus_snapshot_artifact_id=corpus_id,
                       validated_reference_artifact_id=reference_id)
    attempt, capsule, _ = store.begin_child_attempt(
        stage="task_trajectory", input_artifact_ids=(corpus_id, reference_id),
        scope={}, allowed_tools=("submit_task_trajectory_pair",),
        output_contract="task-trajectory-stage-result.v1",
        prompt_version=UNIFIED_PROMPT_VERSION,
    )
    workspace = UnifiedKnowledgeWorkspace(
        store=store, settings=settings, target_profile=target, corpus=corpus,
        corpus_artifact_id=corpus_id, references=[], reference_artifact_id=reference_id,
        attempt_id=attempt.attempt_id,
    )
    pair = render_pair()
    with pytest.raises(ValueError, match="read in this Session"):
        workspace.submit_task_trajectory_pair(**pair)
    assert store.list_task_trajectory_pairs() == []
    workspace.inspect_task_outline()
    workspace.read_task_node("N2")
    retired = render_pair()
    retired["trajectory"]["goal"] = retired["task"]["task"]
    with pytest.raises(ValueError, match="Extra inputs"):
        workspace.submit_task_trajectory_pair(**retired)
    assert store.list_task_trajectory_pairs() == []
    executor = KnowledgeToolExecutor(workspace, "task_trajectory_agent")
    registered = gateway.register_attempt(
        store, attempt.attempt_id, executor,
        allowed_tools=("submit_task_trajectory_pair",),
    )
    pair["task"]["citations"][0]["image_artifact_id"] = ""
    pair["trajectory"]["steps"][0]["citations"][0]["image_artifact_id"] = ""
    accepted = gateway.invoke(registered.token, CapabilityInvocation(
        run_id=capsule.run_id, root_session_id=capsule.parent_root_session_id,
        child_session_id=capsule.child_session_id, attempt_id=capsule.attempt_id,
        stage=capsule.stage, tool_name="submit_task_trajectory_pair",
        arguments=pair, call_id="quoted_citation_with_empty_optional_image",
    ))
    assert accepted["accepted"] is True
    assert workspace.inspect_pair_coverage()["remaining_task_capacity"] == 15
    with pytest.raises(ValueError, match="Invalid Task-Trajectory terminal status"):
        workspace.submit_stage_result("unknown", "Invalid")
    workspace.submit_stage_result("succeeded", "One grounded pair")
    state = store.read_state()
    pairs = store.list_task_trajectory_pairs()
    terminal = store.read_task_trajectory_terminal()
    assert state["task_trajectory_mode"] == "unified"
    assert len(pairs) == 1
    assert terminal["attempt_id"] == attempt.attempt_id


def test_unified_success_requires_major_branch_breadth(tmp_path: Path) -> None:
    store, settings, target, corpus, corpus_id, reference_id, _, _, _ = setup(tmp_path)
    root_id = "N1"
    branch_ids = [f"N{index}" for index in range(2, 15)]
    nodes = [EvidenceNode(
        evidence_node_id=root_id, source_id="src_one", node_key="root",
        child_node_ids=branch_ids, ordinal=0, node_type="document",
        title="Manual", title_origin="source",
    )]
    evidence = []
    for index, branch_id in enumerate(branch_ids, 1):
        part_id = f"part_{index}"
        artifact_id = f"art_{index:024x}"
        evidence.append(EvidenceExcerpt(
            evidence_id=f"evi_{index}", source_id="src_one", source_part_id=part_id,
            kind="instruction", title=f"Capability {index}", content=QUOTE,
            start_offset=0, end_offset=len(QUOTE), extraction_method="test",
            text_artifact_id=artifact_id,
            content_sha256=hashlib.sha256(QUOTE.encode()).hexdigest(), locator={},
        ))
        nodes.append(EvidenceNode(
            evidence_node_id=branch_id, source_id="src_one",
            node_key=f"branch_{index}", parent_node_id=root_id,
            ordinal=index, node_type="section", title=f"Capability {index}",
            title_origin="source", source_refs=[EvidenceSourceSpan(
                source_part_id=part_id, text_artifact_id=artifact_id,
                start_offset=0, end_offset=len(QUOTE),
                content_sha256=hashlib.sha256(QUOTE.encode()).hexdigest(),
            )], content_artifact_id=artifact_id,
            content_sha256=hashlib.sha256(QUOTE.encode()).hexdigest(),
            content=QUOTE, direct_char_count=len(QUOTE),
        ))
    corpus.evidence = evidence
    corpus.evidence_count = len(evidence)
    corpus.source_part_count = len(evidence)
    corpus.source_results[0].evidence = evidence
    corpus.evidence_trees = [EvidenceTree(
        evidence_tree_id="ET1", source_id="src_one", root_node_id=root_id,
        nodes=nodes, plan_artifact_id="plan", validation_artifact_id="validation",
    )]
    workspace = UnifiedKnowledgeWorkspace(
        store=store, settings=settings, target_profile=target,
        corpus=corpus, corpus_artifact_id=corpus_id, references=[],
        reference_artifact_id=reference_id, attempt_id="attempt_test",
    )
    outline = workspace.inspect_task_outline()
    assert [node["node_id"] for node in outline["items"][0]["root"]["children"]] == branch_ids
    assert outline["items"][0]["node_count"] == 14
    for branch_id in branch_ids[:8]:
        workspace.read_task_node(branch_id)
    coverage = workspace.inspect_pair_coverage()["exploration"]
    assert coverage["required_read_major_branches"] == 9
    assert coverage["read_major_branch_count"] == 8
    assert coverage["ready_for_success"] is False
    with pytest.raises(ValueError, match="Read more major"):
        workspace.submit_stage_result("succeeded", "Only eight branches")
    workspace.read_task_node(branch_ids[8])
    assert workspace.inspect_pair_coverage()["exploration"]["ready_for_success"] is True
    pair = render_pair(evidence_ids=("evi_1",))
    pair["task"]["category"] = "Capability 1"
    workspace.submit_task_trajectory_pair(**pair)
    assert workspace.submit_stage_result("succeeded", "Nine branches and one pair")["accepted"]


def test_unified_pair_capacity_is_restored(tmp_path: Path) -> None:
    store, settings, target, corpus, corpus_id, reference_id, _, _, _ = setup(tmp_path)
    settings = replace(settings, max_task_candidates=1)
    attempt, _, _ = store.begin_child_attempt(
        stage="task_trajectory", input_artifact_ids=(corpus_id, reference_id),
        scope={}, allowed_tools=("submit_task_trajectory_pair",),
        output_contract="task-trajectory-stage-result.v1",
        prompt_version=UNIFIED_PROMPT_VERSION,
    )
    workspace = UnifiedKnowledgeWorkspace(
        store=store, settings=settings, target_profile=target, corpus=corpus,
        corpus_artifact_id=corpus_id, references=[], reference_artifact_id=reference_id,
        attempt_id=attempt.attempt_id,
    )
    workspace.inspect_task_outline()
    workspace.read_task_node("N2")
    workspace.submit_task_trajectory_pair(**render_pair())
    assert workspace.inspect_pair_coverage()["remaining_task_capacity"] == 0
    with pytest.raises(ValueError, match="capacity reached"):
        workspace.submit_task_trajectory_pair(**render_pair(title="Render another graph"))


@pytest.mark.parametrize('prompt_version', ['task-trajectory-pairs-dsh.v7', 'task-trajectory-pairs-dsh.v10'])
@pytest.mark.parametrize('tamper', [None, 'digest', 'content', 'retired_metadata', 'retired_goal', 'retired_type'])
def test_historical_pair_digest_with_citation_rationale_is_revalidated(tmp_path: Path, tamper, prompt_version) -> None:
    store, settings, target, corpus, corpus_id, reference_id, gateway, server, config = setup(tmp_path)
    attempt, _, _ = store.begin_child_attempt(
        stage='task_trajectory', input_artifact_ids=(corpus_id, reference_id),
        scope={}, allowed_tools=('submit_task_trajectory_pair',),
        output_contract='task-trajectory-stage-result.v1',
        prompt_version=prompt_version,
    )
    workspace = UnifiedKnowledgeWorkspace(
        store=store, settings=settings, target_profile=target, corpus=corpus,
        corpus_artifact_id=corpus_id, references=[], reference_artifact_id=reference_id,
        attempt_id=attempt.attempt_id,
    )
    workspace.inspect_task_outline()
    workspace.read_task_node('N2')
    workspace.submit_task_trajectory_pair(**render_pair())
    terminal = workspace.submit_stage_result('succeeded', 'One grounded historical pair')['result']
    checkpoint = store.list_task_trajectory_pairs()[0]
    checkpoint.pop('superseded_by')
    checkpoint['prompt_version'] = prompt_version
    checkpoint['task']['title'] = checkpoint['task'].pop('task')
    checkpoint['task']['objective'] = 'Render the graph using the documented control.'
    checkpoint['task']['confidence'] = 0.9
    checkpoint['task']['caveats'] = ['Legacy note']
    checkpoint['task']['operation_mode'] = 'instruction'
    checkpoint['trajectory'].update(goal='Render the graph using the documented control.', trajectory_type='reference_guided',
                                    reference_ids=['R1'], reference_trigger_type='required',
                                    confidence=0.9, caveats=['Legacy note'], unsupported_facets=['Legacy gap'])
    for candidate in (checkpoint['task'], checkpoint['trajectory']):
        for citation in candidate['citations']:
            citation['rationale'] = 'Legacy supporting explanation.'
    for step in checkpoint['trajectory']['steps']:
        for citation in step['citations']:
            citation['rationale'] = 'Legacy step explanation.'
    checkpoint['pair_digest'] = hashlib.sha256(json.dumps(
        {'task': checkpoint['task'], 'trajectory': checkpoint['trajectory']},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')).hexdigest()
    if tamper == 'digest':
        checkpoint['pair_digest'] = '0' * 64
    elif tamper == 'content':
        checkpoint['task']['objective'] = 'A modified objective after digest creation.'
    elif tamper == 'retired_metadata':
        checkpoint['trajectory']['reference_ids'] = ['R2']
    elif tamper == 'retired_goal':
        checkpoint['trajectory']['goal'] = 'A modified goal after digest creation.'
    elif tamper == 'retired_type':
        checkpoint['trajectory']['trajectory_type'] = 'operation'
    artifact = store.put_json_artifact(
        checkpoint, kind='task_trajectory_pair_checkpoint',
        metadata={'stage': 'task_trajectory', 'attempt_id': attempt.attempt_id,
                  'task_id': checkpoint['task_id']}, preserve_text_integrity=True,
    )
    pointer = store.root / 'task_trajectory_pairs' / f"{checkpoint['task_id']}.json"
    pointer.write_text(json.dumps({
        'task_id': checkpoint['task_id'], 'artifact_id': artifact.artifact_id, 'superseded_by': None,
    }), encoding='utf-8')
    service = UnifiedTaskTrajectoryService(
        store=store, settings=settings, runtime=PairedRuntime(gateway), gateway=gateway,
        gateway_server=server, model_config=config,
    )
    if tamper:
        with pytest.raises(RuntimeError, match='checkpoint content mismatch'):
            service._finalize(
                target_profile=target, corpus=corpus, corpus_artifact_id=corpus_id,
                references=[], reference_artifact_id=reference_id,
                stage_result=terminal, attempt_id=attempt.attempt_id,
            )
    else:
        result = service._finalize(
            target_profile=target, corpus=corpus, corpus_artifact_id=corpus_id,
            references=[], reference_artifact_id=reference_id,
            stage_result=terminal, attempt_id=attempt.attempt_id,
        )
        assert len(result.tasks) == len(result.trajectories) == 1
        assert 'rationale' not in result.tasks[0].model_dump()['citations'][0]
        assert 'rationale' not in result.trajectories[0].model_dump()['steps'][0]['citations'][0]
        for field in ('goal', 'trajectory_type', 'reference_ids', 'reference_trigger_type', 'confidence', 'caveats', 'unsupported_facets'):
            assert field not in result.trajectories[0].model_dump()
        assert result.trajectories[0].task_id == result.tasks[0].task_id
        assert store.read_task_trajectory_pair(checkpoint['task_id'])['task']['citations'][0]['rationale']


@pytest.mark.asyncio
async def test_unified_stage_uses_one_session_and_freezes_pairs(tmp_path: Path) -> None:
    store, settings, target, corpus, corpus_id, reference_id, gateway, server, config = setup(tmp_path)
    runtime = PairedRuntime(gateway)
    service = UnifiedTaskTrajectoryService(
        store=store, settings=settings, runtime=runtime, gateway=gateway,
        gateway_server=server, model_config=config,
    )
    artifacts = await service.run(target_profile=target, corpus=corpus,
                                  corpus_artifact_id=corpus_id, references=[],
                                  reference_artifact_id=reference_id)
    assert len(runtime.sessions) == 1
    assert len(artifacts.tasks) == len(artifacts.trajectories) == 1
    assert artifacts.trajectories[0].task_id == artifacts.tasks[0].task_id
    assert artifacts.trajectories[0].knowledge_snapshot_id == artifacts.knowledge.knowledge_snapshot_id
    toolset = tools_for_stage("task_trajectory")
    assert {"submit_task_trajectory_pair", "read_task_node"} <= toolset
    assert {"record_task_candidates", "resolve_task_candidate", "finish_task_batch"}.isdisjoint(toolset)


@pytest.mark.asyncio
async def test_unified_terminal_survives_provider_error(tmp_path: Path) -> None:
    store, settings, target, corpus, corpus_id, reference_id, gateway, server, config = setup(tmp_path)
    runtime = PairedRuntime(gateway, fail_after_terminal=True)
    service = UnifiedTaskTrajectoryService(
        store=store, settings=settings, runtime=runtime, gateway=gateway,
        gateway_server=server, model_config=config,
    )
    first = await service.run(target_profile=target, corpus=corpus,
                              corpus_artifact_id=corpus_id, references=[],
                              reference_artifact_id=reference_id)
    second = await service.run(target_profile=target, corpus=corpus,
                               corpus_artifact_id=corpus_id, references=[],
                               reference_artifact_id=reference_id)
    assert len(runtime.sessions) == 1
    assert [item.task_id for item in first.tasks] == [item.task_id for item in second.tasks]


@pytest.mark.asyncio
async def test_unified_stage_recovers_accepted_pair_after_interruption(tmp_path: Path) -> None:
    store, settings, target, corpus, corpus_id, reference_id, gateway, server, config = setup(tmp_path)
    interrupted = UnifiedTaskTrajectoryService(
        store=store, settings=settings,
        runtime=PairedRuntime(gateway, fail_after_pair=True),
        gateway=gateway, gateway_server=server, model_config=config,
    )
    with pytest.raises(RuntimeError, match="provider stopped"):
        await interrupted.run(target_profile=target, corpus=corpus,
                              corpus_artifact_id=corpus_id, references=[],
                              reference_artifact_id=reference_id)
    assert len(store.list_task_trajectory_pairs()) == 1
    resumed_runtime = RecoveryRuntime(gateway)
    resumed = UnifiedTaskTrajectoryService(
        store=store, settings=settings, runtime=resumed_runtime,
        gateway=gateway, gateway_server=server, model_config=config,
    )
    result = await resumed.run(target_profile=target, corpus=corpus,
                               corpus_artifact_id=corpus_id, references=[],
                               reference_artifact_id=reference_id)
    assert len(result.tasks) == len(result.trajectories) == 1
    assert len(store.list_task_trajectory_pairs()) == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_routes_unified_pairs_into_skill_induction(tmp_path: Path) -> None:
    store, settings, target, corpus, corpus_id, reference_id, gateway, server, config = setup(tmp_path)
    reference_validation = store.put_json_artifact(
        {"validation": []}, kind="reference_validation_report")
    machine = PipelineStateMachine(store)
    for index, phase in enumerate((
        RunPhase.ACQUIRING, RunPhase.NORMALIZING, RunPhase.SOURCE_REVIEW,
        RunPhase.CORPUS_FROZEN, RunPhase.REFERENCE_BUILDING,
        RunPhase.REFERENCES_READY,
    ), 1):
        machine.advance(phase, StageReceipt(
            receipt_id=f"rcpt_unified_{index}", idempotency_key=f"unified.{index}",
            run_id=store.run_id, stage=f"setup_{index}", status="succeeded",
            input_fingerprint=str(index).zfill(64),
            accepted_output_artifact_ids=(corpus_id,),
        ))
    store.update_state(corpus_snapshot_artifact_id=corpus_id,
                       validated_reference_artifact_id=reference_id,
                       reference_validation_artifact_id=reference_validation.artifact_id)
    runtime = PipelineRuntimeDriver(gateway)
    pipeline = PipelineRuntime(settings, runtime, capability_gateway=gateway,
                               capability_server=server)
    pipeline._stores[store.run_id] = store
    result = await pipeline.execute(store.run_id, target, [], config)
    assert result["outcome"] == "succeeded", result
    assert store.read_state()["task_trajectory_mode"] == "unified"
    assert len(result["tasks"]) == len(result["trajectories"]) == len(result["skills"]) == 1
    assert result["trajectories"][0]["task_id"] == result["tasks"][0]["task_id"]
    for field in ('goal', 'trajectory_type', 'reference_ids', 'reference_trigger_type', 'confidence', 'caveats', 'unsupported_facets'):
        assert field not in result["trajectories"][0]
    assert set(result["tasks"][0]) == {"task", "category", "citations", "task_id", "corpus_snapshot_id", "status", "created_at"}
    assert "operation_path_preference" not in result["snapshot"]["target_profile"]
    assert "citations" not in result["skills"][0]
    for field in ('task_ids', 'caveats', 'confidence', 'inferred', 'completion_criteria'):
        assert field not in result['skills'][0]
    assert result["skills"][0]["preconditions"] == "The graph view is open."
    assert result["skills"][0]["procedure"][0]["citations"]
    exported = json.loads(Path(result["skill_registry"]["skills_file"]).read_text(encoding="utf-8"))
    assert 'goal' not in exported['trajectories'][0]
    assert "operation_mode" not in exported["tasks"][0]
    assert "operation_path_preference" not in exported["target_profile"]
    assert "citations" not in exported["skills"][0]
    for field in ('task_ids', 'caveats', 'confidence', 'inferred', 'completion_criteria'):
        assert field not in exported['skills'][0]
    assert result['pipeline_quality']['linkage']['tasks_without_skill'] == []
    assert result['pipeline_quality']['linkage']['unknown_task_links'] == []
    assert exported["skills"][0]["preconditions"] == "The graph view is open."
    assert exported["skills"][0]["procedure"][0]["citations"]
    assert "skill" not in result["pipeline_quality"]["citations"]
    assert "skill_precondition" not in result["pipeline_quality"]["citations"]
    assert result["pipeline_quality"]["citations"]["skill_step"] == 1
    markdown_path = Path(result["skill_registry"]["skills_file"]).parent / "skills" / f"{result['skills'][0]['skill_id']}.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Evidence" not in markdown
    assert "`evi_one`" in markdown
    preconditions_section = markdown.split("## Preconditions\n", 1)[1].split("## Procedure", 1)[0]
    assert "The graph view is open." in preconditions_section
    assert "`evi_one`" not in preconditions_section
