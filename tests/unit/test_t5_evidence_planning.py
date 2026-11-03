from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy

import fitz
import pytest
from bs4 import BeautifulSoup

from app.acquisition import AcquisitionService
from app.config import Settings
from app.evidence_planning import EvidencePlanEngine, EvidencePlanRejected
from app.models import (
    EvidencePlan,
    EvidencePlanExclusion,
    EvidencePlanNode,
    EvidenceRevision,
    EvidenceSourceRef,
    StagedInput,
    TargetProfile,
)
from app.storage import RunStore
from app.toolkit import SourceWorkspace, ToolExecutor


HTML = b"""<!doctype html><html><head><title>GRASS manual</title></head>
<body><main>
<h1>GRASS startup program</h1><p>Launch and initialize GRASS GIS.</p>
<h2>SYNOPSIS</h2><p>Use the grass command with flags and a mapset.</p>
<h3>Flags</h3><p>Use --gui for the graphical interface.</p>
<h2>ENVIRONMENT VARIABLES</h2><p>Variables customize startup behavior.</p>
<h3>Python Environment Variables</h3><p>GRASS_PYTHON selects Python.</p>
<h4>Addon Path</h4><p>GRASS_ADDON_PATH adds scripts.</p>
<h5>Unix note</h5><p>Separate multiple paths with a colon.</p>
<h2>EXAMPLES</h2><p>Run grass --gui to start the interface.</p>
<h2>AUTHORS</h2><p>Example Author.</p>
</main></body></html>"""


def _workspace(tmp_path, html=HTML):
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",

        max_html_pages=0,
    )
    store = RunStore(settings.data_dir, "run_evidence_plan")
    staged = store.stage_upload("inp_1", "grass.html", html)
    source_input = StagedInput(
        input_id="inp_1",
        kind="uploaded_file",
        display_name="grass.html",
        staged_path=str(staged),
        declared_media_type="text/html",
    )
    target = TargetProfile(
        software_name="GRASS GIS",
        target_version="8.4",
        platform="Linux",
        ui_locale="English",
    )
    acquisition = AcquisitionService(store, settings)
    manifest = acquisition.acquire([source_input], target).manifests[0]
    workspace = SourceWorkspace(store, settings, acquisition, target, manifest)
    result = ToolExecutor(workspace, "evidence_plan_test").call(
        "html_inventory", {"max_pages": 0}
    )
    assert result["ok"] is True
    return store, workspace


def _pdf_workspace(tmp_path):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Example Handbook")
    page.insert_text((72, 110), "Example Author")
    page = document.new_page()
    page.insert_text((72, 72), "Contents")
    page.insert_text((72, 110), "Introduction . . . . . . . . . . . 4")
    page.insert_text((72, 135), "Commands . . . . . . . . . . . . . 5")
    page.insert_text((90, 160), "1.1 Run . . . . . . . . . . . . . . 5")
    page = document.new_page()
    page.insert_text((72, 72), "Abstract")
    page.insert_text((72, 110), "This handbook explains the documented workflow.")
    page = document.new_page()
    page.insert_text((72, 72), "Introduction")
    page.insert_text((72, 110), "Start the application and open a project.")
    page = document.new_page()
    page.insert_text((72, 72), "Commands")
    page.insert_text((72, 110), "1.1 Run")
    page.insert_text((72, 145), "Run the selected calculation.")
    document.set_toc(
        [
            [1, "Introduction", 4],
            [1, "Commands", 5],
            [2, "1.1 Run", 5],
        ]
    )
    data = document.tobytes()
    document.close()

    settings = Settings(
        data_dir=tmp_path / "pdf-data",
        output_dir=tmp_path / "pdf-out",

    )
    store = RunStore(settings.data_dir, "run_pdf_evidence_plan")
    staged = store.stage_upload("inp_pdf", "handbook.pdf", data)
    source_input = StagedInput(
        input_id="inp_pdf",
        kind="uploaded_file",
        display_name="handbook.pdf",
        staged_path=str(staged),
        declared_media_type="application/pdf",
    )
    target = TargetProfile(
        software_name="Example",
        target_version="1.0",
        platform="General",
        ui_locale="English",
    )
    acquisition = AcquisitionService(store, settings)
    manifest = acquisition.acquire([source_input], target).manifests[0]
    workspace = SourceWorkspace(store, settings, acquisition, target, manifest)
    result = ToolExecutor(workspace, "pdf_normalizer").call("pdf_inventory", {})
    assert result["ok"] is True
    return store, workspace


def _parts_by_label(workspace):
    return {
        part.label: part
        for part in workspace.parts.values()
        if part.kind in {"html_section", "html_code", "html_heading"}
    }


def _plan(workspace) -> EvidencePlan:
    parts = _parts_by_label(workspace)
    return EvidencePlan(
        source_id=workspace.manifest.source_id,
        nodes=[
            EvidencePlanNode(
                node_key="root",
                title="GRASS startup program",
                node_type="document",
                source_refs=[
                    EvidenceSourceRef(
                        source_part_id=parts["GRASS startup program"].part_id
                    )
                ],
            ),
            EvidencePlanNode(
                node_key="synopsis",
                title="Synopsis",
                title_origin="derived",
                node_type="command_reference",
                parent_node_key="root",
                source_refs=[
                    EvidenceSourceRef(source_part_id=parts["SYNOPSIS"].part_id),
                    EvidenceSourceRef(source_part_id=parts["Flags"].part_id),
                ],
            ),
            EvidencePlanNode(
                node_key="environment",
                title="Environment Variables",
                title_origin="derived",
                node_type="reference",
                parent_node_key="root",
                source_refs=[
                    EvidenceSourceRef(
                        source_part_id=parts["ENVIRONMENT VARIABLES"].part_id
                    ),
                    EvidenceSourceRef(
                        source_part_id=parts["Python Environment Variables"].part_id
                    ),
                    EvidenceSourceRef(source_part_id=parts["Addon Path"].part_id),
                ],
            ),
            EvidencePlanNode(
                node_key="examples",
                title="Examples",
                title_origin="derived",
                node_type="example",
                parent_node_key="root",
                source_refs=[
                    EvidenceSourceRef(source_part_id=parts["EXAMPLES"].part_id)
                ],
            ),
        ],
        excluded_parts=[
            EvidencePlanExclusion(
                source_part_id=parts["AUTHORS"].part_id,
                reason="Authorship metadata is not operational Evidence.",
            )
        ],
        summary="Organize one manual page into a document with semantic children.",
    )


def test_html_document_ir_preserves_h1_h4_parentage_and_h5_inline_text(tmp_path):
    store, workspace = _workspace(tmp_path)
    parts = _parts_by_label(workspace)
    page = next(part for part in workspace.parts.values() if part.kind == "html_page")

    assert parts["GRASS startup program"].parent_part_id == page.part_id
    assert parts["SYNOPSIS"].parent_part_id == parts["GRASS startup program"].part_id
    assert parts["Flags"].parent_part_id == parts["SYNOPSIS"].part_id
    assert parts["ENVIRONMENT VARIABLES"].parent_part_id == parts["GRASS startup program"].part_id
    assert parts["Python Environment Variables"].parent_part_id == parts["ENVIRONMENT VARIABLES"].part_id
    assert parts["Addon Path"].parent_part_id == parts["Python Environment Variables"].part_id
    assert parts["Addon Path"].locator["heading_path"] == [
        "GRASS startup program",
        "ENVIRONMENT VARIABLES",
        "Python Environment Variables",
        "Addon Path",
    ]
    addon_text = workspace.part_text[parts["Addon Path"].part_id]
    assert "Unix note" in addon_text
    assert "Separate multiple paths with a colon." in addon_text
    assert "Unix note" not in parts

    semantic_dom = store.read_json_artifact(page.metadata["semantic_dom_artifact_id"])
    unix_heading = next(node for node in semantic_dom["nodes"] if node.get("text") == "Unix note")
    assert unix_heading["heading_level"] == 5
    assert unix_heading["heading_path"][-2:] == ["Addon Path", "Unix note"]


def test_valid_plan_materializes_hierarchical_multi_part_evidence(tmp_path):
    store, workspace = _workspace(tmp_path)
    plan = _plan(workspace)
    engine = EvidencePlanEngine(workspace)

    report = engine.validate(plan)
    artifacts = engine.materialize(plan)

    assert report.valid is True
    assert report.coverage_complete is True
    assert report.eligible_part_count == 8
    assert report.referenced_part_count == 7
    assert report.excluded_part_count == 1
    tree = artifacts.tree
    assert len(tree.nodes) == 4
    root = next(node for node in tree.nodes if node.node_key == "root")
    environment = next(node for node in tree.nodes if node.node_key == "environment")
    assert root.parent_node_id is None
    assert len(root.child_node_ids) == 3
    assert root.descendant_char_count > 0
    assert len(environment.source_refs) == 3
    assert "Variables customize startup behavior." in environment.content
    assert "GRASS_PYTHON selects Python." in environment.content
    assert "Unix note" in environment.content
    assert store.artifact_path(environment.content_artifact_id).read_text(
        encoding="utf-8"
    ) == environment.content
    for span in environment.source_refs:
        source_text = workspace.part_text[span.source_part_id]
        exact = source_text[span.start_offset : span.end_offset]
        assert hashlib.sha256(exact.encode("utf-8")).hexdigest() == span.content_sha256
    assert store.artifact_record(artifacts.plan_artifact_id).kind == "evidence_plan"
    assert store.artifact_record(artifacts.validation_artifact_id).kind == "evidence_plan_validation"
    assert store.artifact_record(artifacts.tree_artifact_id).kind == "evidence_tree"


def test_invalid_plan_is_rejected_with_actionable_coverage_issues(tmp_path):
    store, workspace = _workspace(tmp_path)
    first = EvidencePlanEngine(workspace).eligible_parts()[0]
    text = workspace.part_text[first.part_id]
    plan = EvidencePlan(
        source_id=workspace.manifest.source_id,
        nodes=[
            EvidencePlanNode(
                node_key="root",
                title="Incomplete plan",
                title_origin="derived",
                node_type="document",
                source_refs=[
                    EvidenceSourceRef(
                        source_part_id=first.part_id,
                        start_offset=0,
                        end_offset=max(1, len(text) // 2),
                    )
                ],
            )
        ],
    )
    engine = EvidencePlanEngine(workspace)

    report = engine.validate(plan)
    assert report.valid is False
    assert report.coverage_complete is False
    assert any(issue.code == "uncovered_source_text" for issue in report.issues)
    with pytest.raises(EvidencePlanRejected) as caught:
        engine.materialize(plan)
    assert caught.value.validation_artifact_id
    assert store.artifact_record(caught.value.validation_artifact_id).kind == "evidence_plan_validation"


def test_evidence_architect_reviews_and_accepts_python_skeleton(tmp_path):
    store, workspace = _workspace(tmp_path)
    engine = EvidencePlanEngine(workspace)
    base_plan = engine.build_skeleton_plan()
    base_artifacts = engine.materialize(base_plan)
    workspace.evidence_skeleton_plan = base_plan
    workspace.evidence_skeleton_plan_artifact_id = base_artifacts.plan_artifact_id
    workspace.evidence_skeleton_tree_artifact_id = base_artifacts.tree_artifact_id
    workspace.evidence_tree = base_artifacts.tree
    workspace.evidence_tree_artifact_id = base_artifacts.tree_artifact_id
    executor = ToolExecutor(workspace, "html_evidence_architect")

    assert executor.source_review_tools == ToolExecutor.ARCHITECT_REVIEW_TOOLS
    skeleton = executor.call("get_evidence_skeleton", {"depth": 2})
    assert skeleton["ok"] is True
    assert skeleton["selected_node"]["node_key"] == "document"
    assert skeleton["base_plan_artifact_id"] == base_artifacts.plan_artifact_id
    assert base_artifacts.tree.referenced_part_count == len(engine.eligible_parts())

    content = executor.call(
        "read_evidence_node",
        {"node_key": "document", "limit": 8_000},
    )
    assert content["ok"] is True
    assert "Launch and initialize GRASS GIS" in content["content"]

    rejected = executor.call(
        "submit_evidence_revision",
        {
            "base_plan_artifact_id": "art_000000000000000000000000",
            "operations": [],
        },
    )
    assert rejected["ok"] is False
    assert rejected["error"] == "stale_evidence_skeleton"
    assert executor.is_finished is False

    accepted = executor.call(
        "submit_evidence_revision",
        {
            "base_plan_artifact_id": base_artifacts.plan_artifact_id,
            "operations": [
                {
                    "operation": "update_node",
                    "node_key": "document",
                    "title": "GRASS manual",
                }
            ],
            "summary": "The Python skeleton needed only a concise root title.",
        },
    )
    assert accepted["ok"] is True
    assert accepted["accepted"] is True
    assert executor.is_finished is True
    assert workspace.final_result is not None
    assert workspace.final_result.evidence_tree is not None
    root = next(
        node
        for node in workspace.final_result.evidence_tree.nodes
        if node.evidence_node_id == workspace.final_result.evidence_tree.root_node_id
    )
    assert root.title == "GRASS manual"
    assert store.artifact_record(accepted["revision_artifact_id"]).kind == "evidence_revision"


def test_revision_applies_bounded_structural_edits_without_rewriting_content(tmp_path):
    _store, workspace = _workspace(tmp_path)
    engine = EvidencePlanEngine(workspace)
    base = _plan(workspace)
    revision = EvidenceRevision(
        source_id=workspace.manifest.source_id,
        base_plan_artifact_id="art_000000000000000000000001",
        operations=[
            {
                "operation": "update_node",
                "node_key": "root",
                "title": "GRASS manual",
            },
            {
                "operation": "merge_nodes",
                "node_key": "synopsis",
                "source_node_keys": ["examples"],
            },
            {
                "operation": "exclude_subtree",
                "node_key": "environment",
                "reason": "Not needed in this bounded test projection.",
            },
        ],
        summary="Apply only structural changes to the Python-owned plan.",
    )

    revised = engine.apply_revision(base, revision)
    artifacts = engine.materialize(revised)

    assert [node.node_key for node in revised.nodes] == ["root", "synopsis"]
    assert revised.nodes[0].title == "GRASS manual"
    assert len(revised.nodes[1].source_refs) == 3
    assert len(revised.excluded_parts) == 4
    assert artifacts.tree.referenced_part_count == 4
    assert "Run grass --gui" in artifacts.tree.nodes[1].content


def test_revision_can_create_semantic_parent_and_move_existing_nodes(tmp_path):
    _store, workspace = _workspace(tmp_path)
    engine = EvidencePlanEngine(workspace)
    base = _plan(workspace)
    revision = EvidenceRevision(
        source_id=workspace.manifest.source_id,
        base_plan_artifact_id="art_000000000000000000000001",
        operations=[
            {
                "operation": "create_node",
                "node_key": "operations",
                "title": "Operating GRASS",
                "node_type": "section",
                "parent_node_key": "root",
            },
            {
                "operation": "move_node",
                "node_key": "synopsis",
                "parent_node_key": "operations",
            },
            {
                "operation": "move_node",
                "node_key": "examples",
                "parent_node_key": "operations",
            },
        ],
        summary="Create one source-faithful semantic grouping node.",
    )

    revised = engine.apply_revision(base, revision)
    artifacts = engine.materialize(revised)

    operations = next(node for node in revised.nodes if node.node_key == "operations")
    assert operations.source_refs == []
    assert operations.title_origin == "derived"
    assert {
        node.node_key
        for node in revised.nodes
        if node.parent_node_key == "operations"
    } == {"synopsis", "examples"}
    materialized = next(
        node for node in artifacts.tree.nodes if node.node_key == "operations"
    )
    assert len(materialized.child_node_ids) == 2


def test_root_navigation_outline_preserves_visual_card_groups(tmp_path):
    _store, workspace = _workspace(tmp_path)
    executor = ToolExecutor(workspace, "html_evidence_architect")
    soup = BeautifulSoup(
        """
        <main><table><tr>
          <td class="box">
            <h3>Graphical User Interface</h3><a href="gui.html">GUI</a>
            <h3>Display</h3><a href="display.html">Display commands</a>
          </td>
          <td class="box">
            <h3>General</h3><a href="general.html">General commands</a>
          </td>
        </tr></table></main>
        """,
        "html.parser",
    )
    branches = [
        {
            "branch_id": "branch_001",
            "label": "Graphical User Interface",
            "links": [{"url": "https://example.test/gui.html", "text": "GUI"}],
        },
        {
            "branch_id": "branch_002",
            "label": "Display",
            "links": [
                {"url": "https://example.test/display.html", "text": "Display commands"}
            ],
        },
        {
            "branch_id": "branch_003",
            "label": "General",
            "links": [
                {"url": "https://example.test/general.html", "text": "General commands"}
            ],
        },
    ]

    groups = executor._navigation_display_groups(
        soup,
        "https://example.test/index.html",
        "https://example.test/",
        branches,
    )

    assert [group["label"] for group in groups] == [
        "Graphical User Interface / Display",
        "General",
    ]
    assert groups[0]["container_tag"] == "td"
    assert groups[0]["container_classes"] == ["box"]


def test_pdf_skeleton_uses_navigation_nodes_and_keeps_residual_content_off_tree(tmp_path):
    _store, workspace = _pdf_workspace(tmp_path)
    engine = EvidencePlanEngine(workspace)
    plan = engine.build_skeleton_plan()

    assert [node.title for node in plan.nodes[1:]] == [
        "Introduction",
        "Commands",
        "1.1 Run",
    ]
    assert all(not node.title.startswith("Page ") for node in plan.nodes)
    assert plan.nodes[2].parent_node_key == "document"
    assert plan.nodes[3].parent_node_key == plan.nodes[2].node_key
    assert len(plan.nodes[0].source_refs) == 2
    assert engine.validate(plan).coverage_complete is True

    view = workspace.pdf_document_structure["agent_view"]
    assert set(view) >= {
        "navigation_tree",
        "content_assignments",
        "unassigned_content",
        "deterministically_excluded_content",
    }
    assert {item["page"] for item in view["deterministically_excluded_content"]} == {2}
    assert {item["page_start"] for item in view["unassigned_content"]} == {1, 3}

    artifacts = engine.materialize(plan)
    workspace.evidence_skeleton_plan = plan
    workspace.evidence_skeleton_plan_artifact_id = artifacts.plan_artifact_id
    workspace.evidence_skeleton_tree_artifact_id = artifacts.tree_artifact_id
    workspace.evidence_tree = artifacts.tree
    workspace.evidence_tree_artifact_id = artifacts.tree_artifact_id
    executor = ToolExecutor(workspace, "pdf_evidence_architect")
    payload = executor.call("get_pdf_document_structure", {})
    assert payload["ok"] is True
    assert "authored_outline" not in payload
    abstract = next(
        item
        for item in payload["unassigned_content"]["items"]
        if item["page_start"] == 3
    )
    residual = executor.call(
        "read_pdf_content", {"content_key": abstract["content_key"]}
    )
    assert residual["ok"] is True
    assert "This handbook explains" in residual["content"]


@pytest.fixture
def pdf_architect_executor(tmp_path):
    _store, workspace = _pdf_workspace(tmp_path)
    engine = EvidencePlanEngine(workspace)
    plan = engine.build_skeleton_plan()
    artifacts = engine.materialize(plan)
    workspace.evidence_skeleton_plan = plan
    workspace.evidence_skeleton_plan_artifact_id = artifacts.plan_artifact_id
    workspace.evidence_skeleton_tree_artifact_id = artifacts.tree_artifact_id
    workspace.evidence_tree = artifacts.tree
    workspace.evidence_tree_artifact_id = artifacts.tree_artifact_id
    return ToolExecutor(workspace, "pdf_evidence_architect")


@pytest.mark.parametrize("invalid_id", ["", " ", "\t\r\n", 0, False])
def test_pdf_navigation_rejects_blank_id_and_recovers_with_explicit_root(
    pdf_architect_executor, invalid_id,
):
    executor = pdf_architect_executor
    rejected = executor.call("get_pdf_document_structure", {"navigation_node_id": invalid_id})
    assert rejected["ok"] is False
    assert "non-empty string" in rejected["message"]
    assert '{"navigation_node_id": "root"}' in rejected["message"]
    base_plan_id = executor.workspace.evidence_skeleton_plan_artifact_id
    premature = executor.call("submit_evidence_revision", {
        "base_plan_artifact_id": base_plan_id, "operations": [],
    })
    assert premature["error"] == "pdf_structure_review_incomplete"

    root = executor.call("get_pdf_document_structure", {"navigation_node_id": "root"})
    assert root["ok"] is True
    assert root["navigation"]["selected_node"]["navigation_node_id"] == "root"
    assert root["navigation"]["total"] == root["navigation"]["selected_node"]["child_count"] == 2
    assert len(root["navigation"]["items"]) == 2
    accepted = executor.call("submit_evidence_revision", {
        "base_plan_artifact_id": base_plan_id, "operations": [],
    })
    assert accepted["accepted"] is True


def test_pdf_navigation_uses_explicit_root_for_pages_and_returned_ids_for_branches(
    pdf_architect_executor,
):
    executor = pdf_architect_executor
    legacy_root = executor.call("get_pdf_document_structure", {})
    explicit_root = executor.call("get_pdf_document_structure", {"navigation_node_id": "root"})
    assert explicit_root == legacy_root
    assert explicit_root["navigation"]["selected_node"]["navigation_node_id"] == "root"
    first = executor.call("get_pdf_document_structure", {"navigation_node_id": "root", "limit": 1})
    assert first["navigation"]["next_offset"] == 1
    second = executor.call("get_pdf_document_structure", {
        "navigation_node_id": "root", "offset": 1, "limit": 1,
    })
    assert second["navigation"]["total"] == 2
    command = second["navigation"]["items"][0]
    assert command["title"] == "Commands"
    branch = executor.call("get_pdf_document_structure", {
        "navigation_node_id": command["navigation_node_id"],
    })
    assert branch["ok"] is True
    assert branch["navigation"]["selected_node"]["title"] == "Commands"
    leaf = branch["navigation"]["items"][0]
    assert leaf["title"] == "1.1 Run"
    selected = executor.call("get_pdf_document_structure", {
        "navigation_node_id": leaf["navigation_node_id"],
    })
    assert selected["ok"] is True
    assert selected["navigation"]["total"] == 0
    assert selected["selected_content_assignments"]["total"] >= 1


def test_pdf_unmatched_pages_expand_without_bypassing_review_completeness(pdf_architect_executor):
    executor = pdf_architect_executor
    view = executor.workspace.pdf_document_structure["agent_view"]
    view["content_assignments"] = [
        {
            "evidence_node_key": f"unmatched_{index:03d}",
            "parent_evidence_node_key": "document",
            "title": f"Unmatched authored topic {index}",
            "page_start": 4, "page_end": 4, "char_count": 100,
        }
        for index in range(73)
    ]
    first = executor.call("get_pdf_document_structure", {"navigation_node_id": "root"})
    page = first["unmatched_content_assignments"]
    assert len(page["items"]) == 24
    assert page["next_offset"] == 24
    base_id = executor.workspace.evidence_skeleton_plan_artifact_id
    rejected = executor.call("submit_evidence_revision", {
        "base_plan_artifact_id": base_id, "operations": [],
    })
    assert rejected["error"] == "pdf_structure_review_incomplete"
    assert rejected["next_unmatched_offset"] == 24
    seen = [item["evidence_node_key"] for item in page["items"]]
    calls = 1
    while page["next_offset"] is not None:
        result = executor.call("get_pdf_document_structure", {
            "navigation_node_id": "root", "unmatched_offset": page["next_offset"],
        })
        assert result["ok"] is True
        page = result["unmatched_content_assignments"]
        seen.extend(item["evidence_node_key"] for item in page["items"])
        calls += 1
    assert calls == 4
    assert seen == [f"unmatched_{index:03d}" for index in range(73)]
    accepted = executor.call("submit_evidence_revision", {
        "base_plan_artifact_id": base_id, "operations": [],
    })
    assert accepted["accepted"] is True
    assert EvidencePlanEngine(executor.workspace).validate(
        executor.workspace.evidence_skeleton_plan
    ).coverage_complete is True


def test_pdf_large_unmatched_pages_keep_whole_items_and_accurate_next_offset(pdf_architect_executor):
    executor = pdf_architect_executor
    view = executor.workspace.pdf_document_structure["agent_view"]
    view["content_assignments"] = [
        {
            "evidence_node_key": f"unmatched_{index:03d}",
            "parent_evidence_node_key": "document",
            "title": (f"Topic {index}: " + "long heading " * 14),
            "page_start": 4, "page_end": 5, "char_count": 100,
        }
        for index in range(73)
    ]
    seen = []
    offset = 0
    while True:
        result = executor.call("get_pdf_document_structure", {
            "navigation_node_id": "root", "unmatched_offset": offset,
            "unmatched_limit": 48,
        })
        assert result["ok"] is True
        page = result["unmatched_content_assignments"]
        assert 0 < len(page["items"]) <= 48
        assert len(json.dumps(result, ensure_ascii=False)) < 12_100
        assert page["offset"] == offset
        seen.extend(item["evidence_node_key"] for item in page["items"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] == offset + len(page["items"])
        offset = page["next_offset"]
    assert seen == [f"unmatched_{index:03d}" for index in range(73)]


@pytest.mark.parametrize("invalid_id", ["document", "ROOT", "null", "section_001", "pdftoc_missing"])
def test_pdf_navigation_rejects_invented_ids_and_evidence_node_keys(pdf_architect_executor, invalid_id):
    result = pdf_architect_executor.call("get_pdf_document_structure", {
        "navigation_node_id": invalid_id,
    })
    assert result["ok"] is False
    assert "Unknown PDF navigation_node_id" in result["message"]
    assert '{"navigation_node_id": "root"}' in result["message"]


def test_pdf_revision_places_opaque_residual_content_without_source_ids(tmp_path):
    _store, workspace = _pdf_workspace(tmp_path)
    engine = EvidencePlanEngine(workspace)
    base = engine.build_skeleton_plan()
    abstract = next(
        item
        for item in workspace.pdf_document_structure["agent_view"]["unassigned_content"]
        if item["page_start"] == 3
    )
    revision = EvidenceRevision(
        source_id=workspace.manifest.source_id,
        base_plan_artifact_id="art_000000000000000000000001",
        operations=[
            {
                "operation": "create_node",
                "node_key": "abstract",
                "title": "Abstract",
                "parent_node_key": "document",
                "before_node_key": "section_001",
            },
            {
                "operation": "assign_content",
                "node_key": "abstract",
                "content_keys": [abstract["content_key"]],
            },
        ],
    )

    revised = engine.apply_revision(base, revision)
    artifacts = engine.materialize(revised)

    assert [node.node_key for node in revised.nodes[:3]] == [
        "document",
        "abstract",
        "section_001",
    ]
    abstract_node = next(node for node in artifacts.tree.nodes if node.node_key == "abstract")
    assert "This handbook explains" in abstract_node.content
    assert engine.validate(revised).coverage_complete is True


def _architect(workspace, base=None):
    engine = EvidencePlanEngine(workspace)
    base = base or engine.build_skeleton_plan()
    artifacts = engine.materialize(base)
    workspace.evidence_skeleton_plan = base
    workspace.evidence_skeleton_plan_artifact_id = artifacts.plan_artifact_id
    workspace.evidence_skeleton_tree_artifact_id = artifacts.tree_artifact_id
    workspace.evidence_tree = artifacts.tree
    workspace.evidence_tree_artifact_id = artifacts.tree_artifact_id
    return engine, base, ToolExecutor(workspace, "evidence_revision_regression")


def _frozen(workspace):
    return (
        {key: part.model_dump(mode="json") for key, part in workspace.parts.items()},
        dict(workspace.part_text),
        {part.text_artifact_id: hashlib.sha256(workspace.store.artifact_path(part.text_artifact_id).read_bytes()).hexdigest()
         for part in workspace.parts.values() if part.text_artifact_id},
    )


def _refs(plan):
    return Counter(ref.model_dump_json() for node in plan.nodes for ref in node.source_refs)


def test_large_html_revision_restores_all_authored_sections_without_operation_cap(tmp_path):
    from app.source_tools.schemas import _tool_spec_catalog

    html = '<html><title>Scientific manual</title><main><h1>Scientific manual</h1><p>Overview.</p>'
    html += ''.join(f'<h2>Topic {index}</h2><p>Exact instructions {index}.</p>' for index in range(90))
    html += '</main></html>'
    _store, workspace = _workspace(tmp_path, html.encode())
    engine, base, executor = _architect(workspace)
    frozen = _frozen(workspace)
    base_before = base.model_dump(mode="json")
    operations = [{"operation": "create_node", "node_key": f"topic_{index}",
                   "title": f"Topic {index}", "parent_node_key": "document"}
                  for index in range(90)]
    result = executor.call("submit_evidence_revision", {
        "base_plan_artifact_id": workspace.evidence_skeleton_plan_artifact_id,
        "operations": operations,
    })
    assert result["ok"], result
    assert result["accepted"]
    children = [node for node in workspace.evidence_tree.nodes if node.parent_node_id]
    assert [node.title for node in children] == [f"Topic {index}" for index in range(90)]
    assert [node.content for node in children] == [f"Exact instructions {index}." for index in range(90)]
    assert len(children) == 90
    assert _frozen(workspace) == frozen
    assert base.model_dump(mode="json") == base_before
    spec = _tool_spec_catalog()["submit_evidence_revision"]["function"]["parameters"]
    assert "maxItems" not in spec["properties"]["operations"]
    assert "maxItems" not in EvidenceRevision.model_json_schema()["properties"]["operations"]


@pytest.mark.parametrize("modality", ["html", "pdf"])
def test_create_node_moves_authored_subtree_and_preserves_frozen_units(tmp_path, modality):
    _store, workspace = _workspace(tmp_path) if modality == "html" else _pdf_workspace(tmp_path)
    engine = EvidencePlanEngine(workspace)
    skeleton = engine.build_skeleton_plan()
    # A coarse candidate can hold an entire authored branch in one node.
    base = EvidencePlan(source_id=workspace.manifest.source_id, nodes=[EvidencePlanNode(
        node_key="document", title="Scientific manual", node_type="document",
        source_refs=[deepcopy(ref) for node in skeleton.nodes for ref in node.source_refs],
    )])
    frozen = _frozen(workspace)
    titles = [("environment", "ENVIRONMENT VARIABLES", "document"),
              ("python", "Python Environment Variables", "environment")] if modality == "html" else [
                  ("commands", "Commands", "document"), ("run", "1.1 Run", "commands")]
    revision = EvidenceRevision(source_id=workspace.manifest.source_id,
        base_plan_artifact_id="art_000000000000000000000001", operations=[
            {"operation": "create_node", "node_key": key, "title": title, "parent_node_key": parent}
            for key, title, parent in titles])
    revised = engine.apply_revision(base, revision)
    tree = engine.materialize(revised).tree
    leaf = next(node for node in tree.nodes if node.node_key == titles[-1][0])
    assert ("GRASS_PYTHON" if modality == "html" else "Run the selected calculation.") in leaf.content
    assert _refs(base) == _refs(revised)
    assert engine.validate(revised).coverage_complete
    assert _frozen(workspace) == frozen


def test_ambiguous_html_heading_requires_explicit_key_and_failed_revision_is_atomic(tmp_path):
    html = b'''<html><title>Manual</title><main><h1>Manual</h1><p>Overview.</p>
    <h2>Syntax</h2><p>First context.</p><h2>Syntax</h2><p>Second context.</p></main></html>'''
    _store, workspace = _workspace(tmp_path, html)
    engine, base, executor = _architect(workspace)
    frozen = _frozen(workspace)
    before = base.model_dump(mode="json")
    operation = {"operation": "create_node", "node_key": "syntax", "title": "Syntax", "parent_node_key": "document"}
    rejected = executor.call("submit_evidence_revision", {
        "base_plan_artifact_id": workspace.evidence_skeleton_plan_artifact_id, "operations": [operation],
    })
    assert not rejected["ok"]
    assert "no unambiguous source content" in str(rejected["validation_errors"])
    assert not executor.is_finished
    assert base.model_dump(mode="json") == before
    assert _frozen(workspace) == frozen

    assignments = []
    offset = 0
    while True:
        page = executor.call("read_evidence_node", {"node_key": "document", "content_offset": offset, "content_limit": 1})
        assert page["ok"]
        view = page["content_assignments"]
        assignments.extend(view["items"])
        if not view["truncated"]:
            break
        assert view["next_offset"] > offset
        offset = view["next_offset"]
    keys = [item["content_key"] for item in assignments if item["heading"] == "Syntax"]
    assert len(keys) == len(set(keys)) == 2
    operation["content_keys"] = [keys[1]]
    accepted = executor.call("submit_evidence_revision", {
        "base_plan_artifact_id": workspace.evidence_skeleton_plan_artifact_id, "operations": [operation],
    })
    assert accepted["ok"], accepted
    assert next(node for node in workspace.evidence_tree.nodes if node.node_key == "syntax").content == "Second context."
    assert _frozen(workspace) == frozen


@pytest.mark.parametrize("modality", ["html", "pdf"])
def test_create_node_can_bind_explicit_existing_content_with_renamed_title(tmp_path, modality):
    _store, workspace = _workspace(tmp_path) if modality == "html" else _pdf_workspace(tmp_path)
    engine, base, executor = _architect(workspace)
    frozen = _frozen(workspace)
    if modality == "pdf":
        part = next(part for part in engine.eligible_parts() if part.kind == "pdf_page_content" and part.locator.get("page") == 3)
    else:
        part = next(part for part in engine.eligible_parts() if part.label == "EXAMPLES")
    revision = EvidenceRevision(source_id=workspace.manifest.source_id,
        base_plan_artifact_id=workspace.evidence_skeleton_plan_artifact_id, operations=[{
            "operation": "create_node", "node_key": "selected", "title": "A meaningful title",
            "parent_node_key": "document", "content_keys": [engine.content_key(part)],
        }])
    revised = engine.apply_revision(base, revision)
    selected = next(node for node in engine.materialize(revised).tree.nodes if node.node_key == "selected")
    assert selected.content == workspace.part_text[part.part_id]
    assert _refs(base) == _refs(revised)
    assert _frozen(workspace) == frozen


@pytest.mark.parametrize("bad_key", ["content_invented", "raw_part_id", "foreign_source"])
def test_unknown_or_invented_content_keys_cannot_change_the_baseline(tmp_path, bad_key):
    _store, workspace = _workspace(tmp_path)
    engine, base, _executor = _architect(workspace)
    part = engine.eligible_parts()[0]
    key = {"raw_part_id": part.part_id,
           "foreign_source": engine._stable_id("content", "another_source", part.part_id)}.get(bad_key, bad_key)
    before = base.model_dump(mode="json")
    revision = EvidenceRevision(source_id=workspace.manifest.source_id,
        base_plan_artifact_id=workspace.evidence_skeleton_plan_artifact_id, operations=[{
            "operation": "create_node", "node_key": "invalid", "title": "Invalid selection",
            "parent_node_key": "document", "content_keys": [key],
        }])
    with pytest.raises(ValueError, match="Unknown or ineligible content_key"):
        engine.apply_revision(base, revision)
    assert base.model_dump(mode="json") == before


def test_revision_does_not_silently_discard_empty_created_leaf(tmp_path):
    _store, workspace = _workspace(tmp_path)
    engine, base, _executor = _architect(workspace)
    revision = EvidenceRevision(source_id=workspace.manifest.source_id,
        base_plan_artifact_id=workspace.evidence_skeleton_plan_artifact_id, operations=[{
            "operation": "create_node", "node_key": "unmatched", "title": "Unfetched topic",
            "parent_node_key": "document",
        }])
    with pytest.raises(ValueError, match="unmatched"):
        engine.apply_revision(base, revision)


def test_created_group_is_not_silently_pruned_after_its_empty_child(tmp_path):
    _store, workspace = _workspace(tmp_path)
    engine = EvidencePlanEngine(workspace)
    base = _plan(workspace)
    synopsis = next(node for node in base.nodes if node.node_key == "synopsis")
    revision = EvidenceRevision(source_id=workspace.manifest.source_id,
        base_plan_artifact_id="art_000000000000000000000001", operations=[
            {"operation": "create_node", "node_key": "group", "title": "Operating the app", "parent_node_key": "root"},
            {"operation": "move_node", "node_key": "synopsis", "parent_node_key": "group"},
            {"operation": "exclude_content", "content_keys": [engine.content_key(workspace.parts[ref.source_part_id])
                                                               for ref in synopsis.source_refs],
             "reason": "Explicit exclusion in this regression case."},
        ])
    with pytest.raises(ValueError, match="group"):
        engine.apply_revision(base, revision)


def test_content_binding_preserves_all_existing_partial_spans(tmp_path):
    _store, workspace = _workspace(tmp_path)
    engine = EvidencePlanEngine(workspace)
    base = engine.build_skeleton_plan()
    root = base.nodes[0]
    part = next(part for part in engine.eligible_parts() if part.label == "EXAMPLES")
    root.source_refs = [ref for ref in root.source_refs if ref.source_part_id != part.part_id]
    middle = len(workspace.part_text[part.part_id]) // 2
    root.source_refs.extend([
        EvidenceSourceRef(source_part_id=part.part_id, start_offset=0, end_offset=middle),
        EvidenceSourceRef(source_part_id=part.part_id, start_offset=middle, end_offset=len(workspace.part_text[part.part_id])),
    ])
    revision = EvidenceRevision(source_id=workspace.manifest.source_id,
        base_plan_artifact_id="art_000000000000000000000001", operations=[{
            "operation": "create_node", "node_key": "examples", "title": "Example workflow",
            "parent_node_key": "document", "content_keys": [engine.content_key(part)],
        }])
    revised = engine.apply_revision(base, revision)
    selected = next(node for node in revised.nodes if node.node_key == "examples")
    assert [(ref.start_offset, ref.end_offset) for ref in selected.source_refs] == [
        (0, middle), (middle, len(workspace.part_text[part.part_id]))]
    assert _refs(revised) == _refs(base)
    assert engine.materialize(revised).tree.referenced_part_count == len(engine.eligible_parts())
