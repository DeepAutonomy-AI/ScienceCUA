from __future__ import annotations

import hashlib
from pathlib import Path

from app.capabilities import tools_for_stage
from app.config import Settings
from app.knowledge import KnowledgeToolExecutor, KnowledgeWorkspace
from app.models import (
    CorpusSnapshot,
    EvidenceExcerpt,
    EvidenceNode,
    EvidenceSourceSpan,
    EvidenceTree,
    SourceResult,
    TargetProfile,
)
from app.storage import RunStore


def _workspace(
    tmp_path: Path,
    contents: list[str],
    *,
    ranges: list[tuple[int, int]] | None = None,
    with_tree: bool = True,
    stage: str = "task_discovery",
) -> KnowledgeWorkspace:
    evidence = []
    spans = []
    node_contents = []
    for index, content in enumerate(contents, 1):
        artifact_id = f"art_{index:024x}"
        evidence.append(
            EvidenceExcerpt(
                evidence_id=f"E{index}",
                source_id="SRC1",
                source_part_id=f"part_{index}",
                text_artifact_id=artifact_id,
                start_offset=0,
                end_offset=len(content),
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                kind="html_section",
                title=f"Section {index}",
                content=content,
                locator={},
                extraction_method="test",
            )
        )
        start, end = ranges[index - 1] if ranges else (0, len(content))
        selected = content[start:end]
        node_contents.append(selected)
        spans.append(
            EvidenceSourceSpan(
                source_part_id=f"part_{index}",
                text_artifact_id=artifact_id,
                start_offset=start,
                end_offset=end,
                content_sha256=hashlib.sha256(selected.encode()).hexdigest(),
            )
        )
    assembled = "\n\n".join(node_contents)
    trees = []
    if with_tree:
        trees.append(
            EvidenceTree(
                evidence_tree_id="ET1",
                source_id="SRC1",
                root_node_id="N1",
                plan_artifact_id="plan",
                validation_artifact_id="validation",
                nodes=[
                    EvidenceNode(
                        evidence_node_id="N1",
                        source_id="SRC1",
                        node_key="root",
                        child_node_ids=["N2"],
                        ordinal=0,
                        node_type="document",
                        title="Manual",
                        title_origin="source",
                        descendant_char_count=len(assembled),
                    ),
                    EvidenceNode(
                        evidence_node_id="N2",
                        source_id="SRC1",
                        node_key="quickstart",
                        parent_node_id="N1",
                        ordinal=0,
                        node_type="section",
                        title="Quickstart",
                        title_origin="source",
                        source_refs=spans,
                        content_artifact_id="art_0000000000000000000000ff",
                        content_sha256=hashlib.sha256(assembled.encode()).hexdigest(),
                        content=assembled,
                        direct_char_count=len(assembled),
                    ),
                ],
            )
        )
    target = TargetProfile(
        software_name="MapTool",
        target_version="1.0",
        platform="General",
        ui_locale="English",
    )
    corpus = CorpusSnapshot(
        corpus_snapshot_id="corpus_sample",
        run_id="run_sample",
        target_profile=target,
        source_results=[
            SourceResult(
                source_id="SRC1",
                display_name="manual.html",
                modality="html",
                status="succeeded",
            )
        ],
        evidence=evidence,
        evidence_trees=trees,
        evidence_count=len(evidence),
        source_part_count=len(evidence),
        source_status_counts={"succeeded": 1},
    )
    return KnowledgeWorkspace(
        store=RunStore(tmp_path, "run_sample"),
        settings=Settings(data_dir=tmp_path),
        target_profile=target,
        stage=stage,
        corpus=corpus,
        corpus_artifact_id="corpus_artifact",
    )


def test_node_read_includes_body_after_notice_and_preserves_citations(tmp_path: Path) -> None:
    notice = "This archived manual has a newer version. Read the current manual page."
    operation = "Import vector data with v.in.ogr and select an output vector map name."
    workspace = _workspace(tmp_path, [notice, operation])
    workspace.evidence_by_id["E2"].locator["version_relation"] = "exact_match"
    original = workspace.corpus.model_dump(mode="json")
    executor = KnowledgeToolExecutor(workspace, "task_discovery_agent")
    outline = executor.call("inspect_task_outline", {})
    descriptor = outline["items"][0]["root"]["children"][0]
    assert descriptor["read_node_id"] == "N2"
    assert "direct_evidence_ids" not in descriptor
    assert "next_offset" not in outline
    assert "expand_task_branch" not in executor.allowed_tools
    redirect = executor.call("read_evidence", {"evidence_id": "E1"})
    assert redirect["error"] == "task_node_read_required"
    assert redirect["node_ids"] == ["N2"]
    assert not workspace.task_evidence_ids_read()

    packet = executor.call("read_task_node", {"node_id": "N2"})
    assert packet["ok"]
    assert [s["content"] for s in packet["segments"]] == [notice, operation]
    assert [s["evidence_id"] for s in packet["segments"]] == ["E1", "E2"]
    assert packet["segments"][1]["version_relation"] == "exact_match"
    assert "next_offset" not in packet
    assert "end_offset" not in packet
    submitted = executor.call(
        "submit_task_batch",
        {"tasks": [{
            "task": "Import a dataset into a named vector map.",
            "category": descriptor["title"],
            "citations": [{"evidence_id": "E2", "quote": operation}],
        }]},
    )
    assert submitted["accepted_by_schema"] == 1
    assert not submitted["errors"]
    assert workspace.task_node_ids_read() == ["N2"]
    assert workspace.task_evidence_ids_read() == ["E1", "E2"]
    assert workspace.corpus.model_dump(mode="json") == original


def test_long_node_read_returns_every_source_character_in_one_call(tmp_path: Path) -> None:
    contents = [f"Section {i}: " + ('quoted "text"\\\n' * 220) for i in range(10)]
    workspace = _workspace(tmp_path, contents)
    executor = KnowledgeToolExecutor(workspace, "task_discovery_agent")

    packet = executor.call("read_task_node", {"node_id": "N2"})

    assert sum(map(len, contents)) > 16_000
    assert [segment["content"] for segment in packet["segments"]] == contents
    assert [segment["evidence_id"] for segment in packet["segments"]] == [
        f"E{i}" for i in range(1, 11)
    ]
    assert "offset" not in packet
    assert "end_offset" not in packet
    assert "next_offset" not in packet
    assert "truncated" not in packet
    assert workspace.task_evidence_ids_read() == [f"E{i}" for i in range(1, 11)]


def test_node_read_respects_exact_source_ranges(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, ["prefix BODY suffix"], ranges=[(7, 11)])
    packet = workspace.read_task_node("N2")
    assert packet["segments"][0]["content"] == "BODY"
    assert packet["segments"][0]["evidence_offset"] == 7
    assert packet["node"]["direct_char_count"] == 4


def test_structural_node_does_not_load_descendant_manual(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, ["Actual chapter text."])
    packet = workspace.read_task_node("N1")
    assert packet["status"] == "structural_only"
    assert not packet["segments"]
    assert not workspace.task_node_ids_read()
    assert not workspace.task_evidence_ids_read()
    assert workspace.read_task_node("N2")["segments"]


def test_single_excerpt_fallback_for_source_without_tree(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, ["Standalone documentation text."], with_tree=False)
    assert workspace.read_evidence("E1")["content"] == "Standalone documentation text."


def test_node_tool_is_available_to_task_and_skill_stages(tmp_path: Path) -> None:
    assert "read_task_node" in tools_for_stage("task_discovery")
    assert "read_task_node" in tools_for_stage("skill_induction")
    assert "search_evidence" not in tools_for_stage("skill_induction")
    assert "expand_task_branch" not in tools_for_stage("task_discovery")
    assert "expand_task_branch" not in tools_for_stage("task_trajectory")
    for stage in ("reference_build", "trajectory_inference"):
        assert "read_task_node" not in tools_for_stage(stage)
        workspace = _workspace(tmp_path / stage, ["Actual chapter text."], stage=stage)
        executor = KnowledgeToolExecutor(workspace, "test")
        assert executor.call("read_task_node", {"node_id": "N2"})["error"] == "tool_not_allowed"
    skill_workspace = _workspace(tmp_path / "skill", ["Actual chapter text."], stage="skill_induction")
    skill_executor = KnowledgeToolExecutor(skill_workspace, "skill_induction_agent")
    assert skill_executor.call("read_task_node", {"node_id": "N2"})["segments"][0]["content"] == "Actual chapter text."
    assert skill_executor.call("read_evidence", {"evidence_id": "E1"})["error"] == "tool_not_allowed"
    assert skill_executor.call("list_evidence", {})["error"] == "tool_not_allowed"
    assert skill_executor.call("search_evidence", {"query": "chapter"})["error"] == "tool_not_allowed"
    assert "read_evidence" not in {s["function"]["name"] for s in skill_executor.tool_specs()}
    fallback_workspace = _workspace(
        tmp_path / "skill_no_tree", ["Standalone documentation text."],
        with_tree=False, stage="skill_induction",
    )
    fallback_executor = KnowledgeToolExecutor(fallback_workspace, "skill_induction_agent")
    assert fallback_executor.call("list_evidence", {})["total"] == 1
    assert fallback_executor.call("read_evidence", {"evidence_id": "E1"})["content"] == "Standalone documentation text."
    assert fallback_executor.call("search_evidence", {"query": "documentation"})["error"] == "tool_not_allowed"
    executor = KnowledgeToolExecutor(_workspace(tmp_path / "task", ["Actual chapter text."]), "test")
    spec = next(s for s in executor.tool_specs() if s["function"]["name"] == "read_task_node")
    assert spec["function"]["parameters"]["required"] == ["node_id"]
    assert set(spec["function"]["parameters"]["properties"]) == {"node_id"}
    outline_spec = next(s for s in executor.tool_specs() if s["function"]["name"] == "inspect_task_outline")
    assert outline_spec["function"]["parameters"]["properties"] == {}


def test_read_task_nodes_batch_reads_in_order_and_dedupes(tmp_path: Path) -> None:
    assert "read_task_nodes" in tools_for_stage("task_discovery")
    assert "read_task_nodes" in tools_for_stage("task_trajectory")
    assert "read_task_nodes" in tools_for_stage("skill_induction")
    workspace = _workspace(tmp_path, ["Actual chapter text."])
    executor = KnowledgeToolExecutor(workspace, "task_discovery_agent")

    result = executor.call("read_task_nodes", {"node_ids": ["N2", "N2", "N1"]})

    assert result["ok"]
    assert [node["node"]["node_id"] for node in result["nodes"]] == ["N2", "N1"]
    assert result["nodes"][0]["segments"][0]["content"] == "Actual chapter text."
    assert result["nodes"][1]["status"] == "structural_only"
    spec = next(s for s in executor.tool_specs() if s["function"]["name"] == "read_task_nodes")
    assert spec["function"]["parameters"]["required"] == ["node_ids"]
    assert spec["function"]["parameters"]["properties"]["node_ids"]["maxItems"] == 8


def test_read_task_nodes_rejects_whole_batch_naming_every_failure(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, ["Actual chapter text."])
    executor = KnowledgeToolExecutor(workspace, "task_discovery_agent")

    result = executor.call("read_task_nodes", {"node_ids": ["N2", "N999", "", "N888"]})

    assert result["ok"] is False
    assert result["error"] == "ValueError"
    for failing in ("N999", "N888"):
        assert failing in result["message"]
    assert "N2" not in result["message"]
    assert workspace.task_node_ids_read() == []


def test_read_task_nodes_rejects_bad_batch_shapes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, ["Actual chapter text."])
    executor = KnowledgeToolExecutor(workspace, "task_discovery_agent")

    assert executor.call("read_task_nodes", {"node_ids": []})["ok"] is False
    assert executor.call("read_task_nodes", {"node_ids": "N2"})["ok"] is False
    oversized = executor.call("read_task_nodes", {"node_ids": [f"N{i}" for i in range(9)]})
    assert oversized["ok"] is False
    assert workspace.task_node_ids_read() == []
