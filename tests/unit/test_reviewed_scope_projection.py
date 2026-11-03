"""Scope is a lossless projection of accepted Evidence, independent of outlines."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.acquisition import AcquisitionService
from app.config import Settings
from app.evidence_planning import EvidencePlanEngine
from app.models import EvidencePlan, EvidencePlanNode, EvidenceSourceRef, SourceManifest, SourcePart, TargetProfile
from app.orchestration.source_scope_stage import build_outline_preview, project_reviewed_scope, validate_selection
from app.orchestration.source_stage import validate_source_results
from app.source_tools.workspace import SourceWorkspace
from app.storage import RunStore
from app.toolkit import ToolExecutor


def workspace(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "d", output_dir=tmp_path / "o", scope_selection_enabled=True)
    store = RunStore(settings.data_dir, "run_projection")
    target = TargetProfile(software_name="Example", target_version="1", platform="General")
    store.initialize({"target_profile": target.model_dump(mode="json")})
    raw = store.put_artifact(b"original", kind="raw_source", media_type="text/html", suffix=".html", source_id="SRC1")
    manifest = SourceManifest(source_id="SRC1", canonical_input_id="IN1", display_name="Manual",
                              primary_modality="html", media_type="text/html", raw_artifact_id=raw.artifact_id)
    return SourceWorkspace(store, settings, AcquisitionService(store, settings), target, manifest)


def accept(ws, plan):
    artifacts = EvidencePlanEngine(ws).materialize(plan)
    ToolExecutor(ws, "test")._commit_materialized_evidence(
        plan, artifacts, summary="Reviewed", gaps=[], warnings=[], accepted_event="test_accepted")
    return ws.final_result


@pytest.mark.parametrize("modality", ["pdf", "html"])
def test_only_first_level_choices_include_all_descendant_content(tmp_path, modality):
    ws = workspace(tmp_path)
    ws.manifest.primary_modality = modality
    ws.add_part(SourcePart(part_id="book", source_id="SRC1", kind=f"{modality}_section", label="Book", locator={}), "ABCDE")
    hierarchy = [("root", None), ("chapter", "root"), ("subsection", "chapter"),
                 ("deep", "subsection"), ("sibling", "root")]
    plan = EvidencePlan(source_id="SRC1", nodes=[
        EvidencePlanNode(node_key=key, title=key, parent_node_key=parent,
                         source_refs=[EvidenceSourceRef(source_part_id="book", start_offset=index, end_offset=index+1)])
        for index, (key, parent) in enumerate(hierarchy)
    ])
    full = accept(ws, plan)
    outline_id = build_outline_preview(ws.store, [full])
    outline = ws.store.read_json_artifact(outline_id)
    assert [item["title"] for item in outline["sections"]] == ["chapter", "sibling"]
    assert all(item["depth"] == 0 for item in outline["sections"])
    nodes = {node.node_key: node for node in full.evidence_tree.nodes}
    for key in ("subsection", "deep"):
        with pytest.raises(ValueError, match="non-first-level"):
            validate_selection(outline, "selected", [nodes[key].evidence_node_id])
    selected = {"outline_artifact_id": outline_id, "mode": "selected",
                "section_ids": [nodes["chapter"].evidence_node_id]}
    projected = project_reviewed_scope(ws.store, [full], selected, settings=ws.settings, target=ws.target_profile)
    results, summary = validate_source_results(ws.store, projected)
    assert summary[0]["error_count"] == 0
    assert "".join(item.content for item in results[0].evidence) == "BCD"
    assert [node.node_key for node in results[0].evidence_tree.nodes] == ["root", "chapter", "subsection", "deep"]


def test_document_without_chapters_offers_whole_material_only(tmp_path):
    ws = workspace(tmp_path)
    ws.add_part(SourcePart(part_id="book", source_id="SRC1", kind="html_section", label="Book", locator={}), "Whole document")
    full = accept(ws, EvidencePlan(source_id="SRC1", nodes=[EvidencePlanNode(
        node_key="root", title="Book", source_refs=[EvidenceSourceRef(source_part_id="book")])]))
    outline = ws.store.read_json_artifact(build_outline_preview(ws.store, [full]))
    assert not outline["outline_available"]
    assert outline["sections"] == []
    assert validate_selection(outline, "full", []) == []


@pytest.mark.parametrize("mode", ["selected", "full"])
def test_shared_sourcepart_spans_preserve_exact_text_and_strip_ancestor_body(tmp_path, mode):
    ws = workspace(tmp_path)
    chunks = ["ANCESTOR_SECRET\n", "KEEP\r\n中文\n", "DROP_UNSELECTED"]
    text = "".join(chunks)
    ws.add_part(SourcePart(part_id="shared", source_id="SRC1", kind="html_section", label="Manual", locator={}), text)
    offsets = [0, len(chunks[0]), len(chunks[0]) + len(chunks[1]), len(text)]
    plan = EvidencePlan(source_id="SRC1", nodes=[EvidencePlanNode(
        node_key=key, title=title, parent_node_key=parent,
        source_refs=[EvidenceSourceRef(source_part_id="shared", start_offset=offsets[index], end_offset=offsets[index+1])],
    ) for index, (key, title, parent) in enumerate([
        ("root", "Manual", None), ("keep", "Keep", "root"), ("drop", "Drop", "root"),
    ])])
    full = accept(ws, plan)
    original = ws.store.artifact_path(full.corpus_artifact_id).read_bytes()
    outline_id = build_outline_preview(ws.store, [full])
    outline = ws.store.read_json_artifact(outline_id)
    selected_id = next(item["section_id"] for item in outline["sections"] if item["title"] == "Keep")
    selection = {"outline_artifact_id": outline_id, "mode": mode,
                 "section_ids": [selected_id] if mode == "selected" else []}
    projected = project_reviewed_scope(ws.store, [full], selection, settings=ws.settings, target=ws.target_profile)
    validated, summaries = validate_source_results(ws.store, projected)
    assert summaries[0]["error_count"] == 0
    content = "".join(item.content for item in validated[0].evidence)
    assert chunks[1] in content
    assert ("DROP_UNSELECTED" in content) is (mode == "full")
    assert ("ANCESTOR_SECRET" in content) is (mode == "full")
    assert ws.store.artifact_path(full.corpus_artifact_id).read_bytes() == original
    # Repeating after a partial projection/restart reads the immutable full corpus.
    again = project_reviewed_scope(ws.store, [full], selection, settings=ws.settings, target=ws.target_profile)
    assert [item.content for item in again[0].evidence] == [item.content for item in validated[0].evidence]


def test_1025_chapters_materialize_and_are_all_available_for_selection(tmp_path):
    ws = workspace(tmp_path)
    ws.add_part(SourcePart(part_id="book", source_id="SRC1", kind="html_section", label="Book", locator={}), "x" * 1025)
    plan = EvidencePlan(source_id="SRC1", nodes=[EvidencePlanNode(node_key="root", title="Book")] + [
        EvidencePlanNode(node_key=f"chapter_{index}", parent_node_key="root", title=f"Chapter {index}",
                         source_refs=[EvidenceSourceRef(source_part_id="book", start_offset=index, end_offset=index+1)])
        for index in range(1025)
    ])
    full = accept(ws, plan)
    assert len(full.evidence_tree.nodes) == 1026
    outline = ws.store.read_json_artifact(build_outline_preview(ws.store, [full]))
    assert len(outline["sections"]) == 1025
    assert len({item["section_id"] for item in outline["sections"]}) == 1025


def test_inline_images_stay_owned_by_projected_parts_without_reopening_full_text(tmp_path):
    ws = workspace(tmp_path)
    image = ws.store.put_artifact(b"image-fixture", kind="source_image", media_type="image/png", suffix=".png", source_id="SRC1")
    raw_ref = {"artifact_id": image.artifact_id, "source_part_id": "shared", "media_type": "image/png", "page": 1, "anchor_offset": 1}
    ws.add_part(SourcePart(part_id="shared", source_id="SRC1", kind="html_section", label="Manual", locator={},
                           metadata={"image_refs": [raw_ref], "images": {"image": {"status": "saved", "artifact_id": image.artifact_id, "media_type": "image/png"}}}), "KEEP|SECRET")
    plan = EvidencePlan(source_id="SRC1", nodes=[
        EvidencePlanNode(node_key="root", title="Manual"),
        EvidencePlanNode(node_key="keep", title="Keep", parent_node_key="root", source_refs=[EvidenceSourceRef(source_part_id="shared", start_offset=0, end_offset=5)]),
        EvidencePlanNode(node_key="drop", title="Drop", parent_node_key="root", source_refs=[EvidenceSourceRef(source_part_id="shared", start_offset=5, end_offset=11)]),
    ])
    full = accept(ws, plan)
    outline_id = build_outline_preview(ws.store, [full])
    outline = ws.store.read_json_artifact(outline_id)
    selection = {"outline_artifact_id": outline_id, "mode": "selected", "section_ids": [outline["sections"][0]["section_id"]]}
    results = project_reviewed_scope(ws.store, [full], selection, settings=ws.settings, target=ws.target_profile)
    validated, summaries = validate_source_results(ws.store, results)
    assert summaries[0]["error_count"] == 0
    assert validated[0].evidence[0].image_refs[0].artifact_id == image.artifact_id
    assert validated[0].evidence[0].image_refs[0].source_part_id != "shared"
    corpus = ws.store.read_json_artifact(validated[0].corpus_artifact_id)
    assert all(part["part_id"] != "shared" for part in corpus["parts"])
    assert all("SECRET" not in ws.store.artifact_path(part["text_artifact_id"]).read_text(encoding="utf-8")
               for part in corpus["parts"] if part.get("text_artifact_id"))
