from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from app.config import Settings
from app.knowledge import KnowledgeWorkspace
from app.models import (
    CorpusSnapshot,
    EvidenceCitation,
    EvidenceExcerpt,
    KnowledgeSnapshot,
    SourceResult,
    TargetProfile,
    TaskCandidate,
    TrajectoryCandidate,
    TrajectoryStepDraft,
)
from app.storage import RunStore

from types import SimpleNamespace as Obj
from app.orchestration.downstream_stage import (
    SkillInductionService,
    _skill_trajectory_evidence_packet,
)

import unittest
from app.knowledge import _requires_reference_lookup


def _exercise_coverage_without_shortcut(tmp_path: Path):
    content = "Use List mode to add a graph. Press Ctrl+S to save the graph."
    evidence = EvidenceExcerpt(
        evidence_id="evi_graph_controls",
        source_id="src_graph_controls",
        source_part_id="part_graph_controls",
        text_artifact_id="art_graph_controls",
        start_offset=0,
        end_offset=len(content),
        content_sha256="test-sha",
        kind="html_section",
        title="Graph controls",
        content=content,
        locator={},
        extraction_method="test",
    )
    add_citation = EvidenceCitation(
        evidence_id=evidence.evidence_id,
        quote="Use List mode to add a graph.",
    )
    save_citation = EvidenceCitation(
        evidence_id=evidence.evidence_id,
        quote="Press Ctrl+S to save the graph.",
    )
    target = TargetProfile(
        software_name="Example", target_version="1", platform="General", ui_locale="English"
    )
    corpus = CorpusSnapshot(
        corpus_snapshot_id="corpus_graph_controls",
        run_id="run_graph_controls",
        target_profile=target,
        source_results=[SourceResult(
            source_id=evidence.source_id,
            display_name="manual.html",
            modality="html",
            status="succeeded",
        )],
        evidence=[evidence],
        evidence_count=1,
        source_part_count=1,
        source_status_counts={"succeeded": 1},
    )
    task = TaskCandidate(task_id='tsk_graph_controls', corpus_snapshot_id=corpus.corpus_snapshot_id, task='Manage and save a graph', citations=[add_citation, save_citation], category='Operations')
    knowledge = KnowledgeSnapshot(
        knowledge_snapshot_id="knowledge_graph_controls",
        run_id=corpus.run_id,
        corpus_snapshot_id=corpus.corpus_snapshot_id,
        corpus_snapshot_artifact_id="art_corpus_graph_controls",
        target_profile=target,
        tasks=[task],
    )
    trajectory = TrajectoryCandidate(
        trajectory_id="traj_graph_controls",
        knowledge_snapshot_id=knowledge.knowledge_snapshot_id,
        task_id=task.task_id,
        steps=[
            TrajectoryStepDraft(
                ordinal=1,
                instruction="Use List mode to add a graph.",
                expected_observation="The graph appears in the view.",
                citations=[add_citation],
            ),
            TrajectoryStepDraft(
                ordinal=2,
                instruction="Press Ctrl+S to save the graph.",
                expected_observation="The graph is saved.",
                citations=[save_citation],
            ),
        ],
        citations=[add_citation, save_citation],
    )
    store = RunStore(tmp_path, corpus.run_id)
    store.initialize({"target_profile": target.model_dump(mode="json")})
    workspace = KnowledgeWorkspace(
        store=store,
        settings=Settings(data_dir=tmp_path, output_dir=tmp_path / "outputs"),
        target_profile=target,
        stage="skill_induction",
        corpus=corpus,
        corpus_artifact_id="art_corpus_graph_controls",
        knowledge=knowledge,
        knowledge_artifact_id="art_knowledge_graph_controls",
        trajectories=[trajectory],
    )

    def submit_skill(title: str, instruction: str, citation: EvidenceCitation):
        return workspace.submit_skill_batch([{
            "title": title,
            "description": f"Use the graph view to {title.lower()}.",
            "applicability": "Use when working with a graph in the view.",
            "preconditions": "The graph view is open.",
            "procedure": [{
                "ordinal": 1,
                "instruction": instruction,
                "citations": [citation.model_dump(mode="json")],
            }],
            "trajectory_ids": [trajectory.trajectory_id],
        }])

    add_result = submit_skill(
        "Add another graph", "Use List mode to add a graph.", add_citation
    )
    assert add_result["accepted_by_schema"] == 1
    assert add_result["candidate_ids"] == ["SK1"]
    assert "missing_operational_controls" not in add_result["coverage"]
    assert add_result["coverage"]["covered_trajectory_ids"] == [trajectory.trajectory_id]
    assert add_result["coverage"]["uncovered_trajectory_ids"] == []
    assert add_result["coverage"]["reviewed_trajectory_ids"] == [trajectory.trajectory_id]
    assert add_result["next_step"] == "Coverage is complete. Call submit_stage_result."
    disposition = [{
        "trajectory_id": trajectory.trajectory_id,
        "disposition": "covered",
        "rationale": "The accepted Skill links this trajectory.",
    }]
    reviewed = workspace.review_skill_coverage(disposition)
    assert reviewed["coverage"]["reviewed_trajectory_ids"] == [trajectory.trajectory_id]
    final = workspace.submit_stage_result(
        status="succeeded"
    )["result"]
    assert final["status"] == "succeeded"
    assert final["candidate_count"] == 1
    assert final["coverage"]["reviewed_trajectory_ids"] == [trajectory.trajectory_id]


class SkillCoverageTests(TestCase):
    def test_linked_skill_can_complete_without_trajectory_shortcut(self):
        with TemporaryDirectory() as directory:
            _exercise_coverage_without_shortcut(Path(directory))

    def test_only_uncovered_trajectory_needs_explicit_exclusion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            content = "Press Ctrl+R to render the graph."
            evidence = EvidenceExcerpt(
                evidence_id="evi_render", source_id="src_render",
                source_part_id="part_render", text_artifact_id="art_render",
                start_offset=0, end_offset=len(content), content_sha256="test-sha",
                kind="html_section", title="Render", content=content,
                locator={}, extraction_method="test",
            )
            citation = EvidenceCitation(evidence_id=evidence.evidence_id, quote=content)
            target = TargetProfile(
                software_name="Example", target_version="1", platform="General",
                ui_locale="English",
            )
            corpus = CorpusSnapshot(
                corpus_snapshot_id="corpus_render", run_id="run_render",
                target_profile=target, source_results=[SourceResult(
                    source_id=evidence.source_id, display_name="manual.html",
                    modality="html", status="succeeded",
                )], evidence=[evidence], evidence_count=1, source_part_count=1,
                source_status_counts={"succeeded": 1},
            )
            task = TaskCandidate(task_id='tsk_render', corpus_snapshot_id=corpus.corpus_snapshot_id, task='Render and style a graph', citations=[citation], category='Operations')
            knowledge = KnowledgeSnapshot(
                knowledge_snapshot_id="knowledge_render", run_id=corpus.run_id,
                corpus_snapshot_id=corpus.corpus_snapshot_id,
                corpus_snapshot_artifact_id="art_corpus_render", target_profile=target,
                tasks=[task],
            )
            step = TrajectoryStepDraft(
                ordinal=1, instruction=content,
                expected_observation="The graph is rendered.", citations=[citation],
            )
            render = TrajectoryCandidate(
                trajectory_id="traj_render", knowledge_snapshot_id=knowledge.knowledge_snapshot_id,
                task_id=task.task_id, steps=[step],
                citations=[citation],
            )
            styling = TrajectoryCandidate(
                trajectory_id="traj_styling", knowledge_snapshot_id=knowledge.knowledge_snapshot_id,
                task_id=task.task_id, steps=[step],
                citations=[citation],
            )
            store = RunStore(root, corpus.run_id)
            store.initialize({"target_profile": target.model_dump(mode="json")})
            workspace = KnowledgeWorkspace(
                store=store, settings=Settings(data_dir=root, output_dir=root / "outputs"),
                target_profile=target, stage="skill_induction", corpus=corpus,
                corpus_artifact_id="art_corpus_render", knowledge=knowledge,
                knowledge_artifact_id="art_knowledge_render",
                trajectories=[render, styling],
            )
            submitted = workspace.submit_skill_batch([{
                "title": "Render a graph", "description": "Render the current graph.",
                "applicability": "Use when a graph needs rendering.",
                "preconditions": "The graph view is open.",
                "procedure": [{
                    "ordinal": 1, "instruction": content,
                    "citations": [citation.model_dump(mode="json")],
                }],
                "trajectory_ids": [render.trajectory_id],
            }])
            self.assertEqual(submitted["accepted_by_schema"], 1)
            self.assertEqual(submitted["coverage"]["reviewed_trajectory_ids"], [render.trajectory_id])
            self.assertEqual(submitted["coverage"]["uncovered_trajectory_ids"], [styling.trajectory_id])
            with self.assertRaisesRegex(ValueError, "Unreviewed trajectories"):
                workspace.submit_stage_result(status="succeeded")

            reviewed = workspace.review_skill_coverage([{
                "trajectory_id": styling.trajectory_id,
                "disposition": "excluded",
                "rationale": "The source does not document graph styling controls.",
            }])
            self.assertEqual(
                reviewed["coverage"]["reviewed_trajectory_ids"],
                [render.trajectory_id, styling.trajectory_id],
            )
            final = workspace.submit_stage_result(
                status="degraded"
            )["result"]
            self.assertEqual(final["status"], "degraded")
            self.assertEqual(final["coverage"]["excluded"][0]["trajectory_id"], styling.trajectory_id)


def _citation(evidence_id: str) -> Obj:
    return Obj(evidence_id=evidence_id)


def _trajectory(index: int, evidence_id: str) -> Obj:
    citation = _citation(evidence_id)
    return Obj(
        trajectory_id=f"TR{index}", task_id=f"T{index}",
        citations=[citation], steps=[Obj(ordinal=1, citations=[citation])],
    )


def _corpus() -> Obj:
    evidence = [
        Obj(evidence_id=f"E{index}", source_id="SRC1", source_part_id=f"part_{index}",
            start_offset=0, end_offset=30, content=f"Evidence text {index}",
            title=f"Evidence {index}", image_refs=[], resolved_version="1")
        for index in range(1, 4)
    ]
    nodes = [
        Obj(evidence_node_id="N1", parent_node_id=None, title="Manual", source_refs=[
            Obj(source_part_id="part_3", start_offset=0, end_offset=30),
        ]),
        Obj(evidence_node_id="N2", parent_node_id="N1", title="Raster", source_refs=[]),
        Obj(evidence_node_id="N3", parent_node_id="N2", title="Raster import", source_refs=[
            Obj(source_part_id="part_1", start_offset=0, end_offset=30),
        ]),
        Obj(evidence_node_id="N4", parent_node_id="N1", title="Vector", source_refs=[
            Obj(source_part_id="part_2", start_offset=0, end_offset=30),
        ]),
    ]
    return Obj(evidence=evidence, evidence_trees=[
        Obj(source_id="SRC1", root_node_id="N1", nodes=nodes),
    ])


def test_skill_batches_follow_first_level_tree_ownership_without_count_splitting() -> None:
    corpus = _corpus()
    trajectories = [_trajectory(index, "E1") for index in range(1, 5)]
    trajectories += [_trajectory(5, "E1"), _trajectory(6, "E2")]
    tasks = {
        f"T{index}": Obj(
            category=f"unrelated free text {index}",
            citations=[_citation("E3" if index < 5 else "E2")],
        )
        for index in range(1, 7)
    }
    service = SkillInductionService.__new__(SkillInductionService)
    batches = service._batches(trajectories, tasks, corpus)
    assert len(batches) == 3
    assert [item.trajectory_id for item in batches[0]["trajectories"]] == [
        "TR1", "TR2", "TR3", "TR4",
    ]
    assert batches[0]["branch_node_ids"] == ["N2"]
    assert batches[1]["batch_id"] == "cross_branch_TR5"
    assert batches[1]["branch_node_ids"] == ["N2", "N4"]
    assert [item.trajectory_id for item in batches[2]["trajectories"]] == ["TR6"]


def test_skill_evidence_packet_deduplicates_and_keeps_step_links() -> None:
    corpus = _corpus()
    first = _trajectory(1, "E1")
    second = _trajectory(2, "E1")
    second.steps.append(Obj(ordinal=2, citations=[_citation("E2")]))
    packet = _skill_trajectory_evidence_packet([first, second], corpus)
    assert [item["evidence_id"] for item in packet] == ["E1", "E2"]
    assert packet[0]["content"] == "Evidence text 1"
    assert packet[0]["used_by"] == [
        {"trajectory_id": "TR1", "step_ordinals": [1]},
        {"trajectory_id": "TR2", "step_ordinals": [1]},
    ]
    assert packet[1]["used_by"] == [
        {"trajectory_id": "TR2", "step_ordinals": [2]},
    ]


class ReferenceLookupIntentTests(unittest.TestCase):
    def test_graph_input_does_not_require_catalog(self) -> None:
        self.assertFalse(
            _requires_reference_lookup(
                [
                    "Select the 2D Graph tab, then click the Add tab to start a new function entry.",
                    "The function input text box has focus and is ready for typing.",
                ]
            )
        )

    def test_actual_catalog_lookup_requires_reference(self) -> None:
        self.assertTrue(
            _requires_reference_lookup(["Search the function catalog for the required expression."])
        )
        self.assertTrue(
            _requires_reference_lookup(["查阅函数目录，找到对应的函数条目。"])
        )

    def test_unrelated_clauses_do_not_combine(self) -> None:
        self.assertFalse(
            _requires_reference_lookup(["Read the graph; then add a new function entry."])
        )
