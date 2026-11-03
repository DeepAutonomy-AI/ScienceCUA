"""Exercise PDF extraction through acquisition, native SourceParts, and Evidence."""

from __future__ import annotations

import hashlib
import json
from unittest.mock import patch

import fitz
import pytest

from app.acquisition import AcquisitionService
from app.config import Settings
from app.evidence_planning import EvidencePlanEngine
from app.models import StagedInput, TargetProfile
from app.orchestration.source_stage import validate_source_results
from app.storage import RunStore
from app.toolkit import SourceWorkspace, ToolExecutor


LINK_URL = "https://example.org/reference?view=controls#keys"
TABLE_ROWS = [("Key", "Action"), ("F1", "Rotate model"),
              ("F2", "Translate model"), ("F3", "Fit model")]


@pytest.fixture(scope="module")
def manual_pdf():
    """Two pages with authored structure, without external fixture dependencies."""
    with fitz.open() as document:
        page = document.new_page(width=540, height=720)
        page.insert_text((72, 72), "Controls", fontsize=18, fontname="hebo")
        page.insert_text((72, 110), "A wrapped paragraph begins here and continues", fontsize=11)
        page.insert_text((72, 124), "on its next source line without a new paragraph.", fontsize=11)
        xs = [72, 170, 468]
        ys = [174, 204, 234, 264, 294]
        for x in xs:
            page.draw_line((x, ys[0]), (x, ys[-1]), color=(0, 0, 0))
        for y in ys:
            page.draw_line((xs[0], y), (xs[-1], y), color=(0, 0, 0))
        for row_number, cells in enumerate(TABLE_ROWS):
            for column, text in enumerate(cells):
                page.insert_text((xs[column] + 8, ys[row_number] + 20), text, fontsize=11)
        page.insert_text((72, 330), "Read the controls reference.", fontsize=11)
        page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(72, 315, 230, 335),
                          "uri": LINK_URL})
        page = document.new_page(width=540, height=720)
        page.insert_text((72, 72), "Reference Notes", fontsize=18, fontname="hebo")
        page.insert_text((72, 110), "SECOND_PAGE_CONTENT remains available independently.", fontsize=11)
        document.set_toc([[1, "Controls", 1], [1, "Reference Notes", 2]])
        return document.tobytes()


@pytest.fixture(scope="module")
def image_only_pdf():
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), False)
    pixmap.clear_with(170)
    with fitz.open() as document:
        page = document.new_page(width=300, height=240)
        page.insert_image(fitz.Rect(30, 40, 270, 200), stream=pixmap.tobytes("png"))
        return document.tobytes()


def workspace_for(tmp_path, source, *, filename="manual.pdf"):
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "out",
                         max_html_pages=0)
    store = RunStore(settings.data_dir, "run_pdf_integration")
    staged = store.stage_upload("inp_1", filename, source)
    target = TargetProfile(software_name="Example", target_version="1",
                           platform="General", ui_locale="English")
    acquisition = AcquisitionService(store, settings)
    manifest = acquisition.acquire([
        StagedInput(input_id="inp_1", kind="uploaded_file", display_name=filename,
                    staged_path=str(staged), declared_media_type=(
                        "application/pdf" if filename.endswith(".pdf") else "text/html"))
    ], target).manifests[0]
    workspace = SourceWorkspace(store, settings, acquisition, target, manifest)
    return workspace, ToolExecutor(workspace, "pdf_integration_test")


def inventory(workspace, executor):
    response = executor.call("pdf_inventory", {})
    assert response["ok"], response
    assert workspace.inventory_completed
    candidates = workspace.evidence_candidates()
    assert candidates
    return candidates


def commit(workspace, executor):
    engine = EvidencePlanEngine(workspace)
    plan = engine.build_skeleton_plan()
    artifacts = engine.materialize(plan)
    response = executor._commit_materialized_evidence(
        plan, artifacts, summary="PDF integration fixture.", gaps=[], warnings=[],
        accepted_event="integration_verified")
    assert response["accepted"], response
    result = workspace.final_result
    validated, summaries = validate_source_results(workspace.store, [result])
    assert summaries[0]["errors"] == [], summaries
    assert summaries[0]["rejected_evidence_count"] == 0
    assert len(validated[0].evidence) == len(result.evidence)
    return result


def structure_blocks(workspace, candidates):
    blocks = []
    for part in candidates:
        artifact_id = part.metadata["structure_artifact_id"]
        assert artifact_id in workspace.allowed_artifacts
        structure = workspace.store.read_json_artifact(artifact_id)
        blocks.extend(structure["blocks"])
    return blocks


def test_pdf_pipeline_preserves_paragraph_table_relationships_and_links(tmp_path, manual_pdf):
    workspace, executor = workspace_for(tmp_path, manual_pdf)
    candidates = inventory(workspace, executor)
    blocks = structure_blocks(workspace, candidates)
    paragraphs = [block for block in blocks if block["type"] == "paragraph"]
    assert any("A wrapped paragraph begins here and continues on its next source line"
               in block.get("text", "") for block in paragraphs)
    tables = [block for block in blocks if block["type"] == "table"]
    assert tables, "Ruled tables must retain explicit row and column structure"
    for expected in TABLE_ROWS:
        assert any(all(text in json.dumps(row) for text in expected)
                   for table in tables for row in table["rows"]), expected
    text = "\n".join(workspace.part_text[part.part_id] for part in candidates if part.text_artifact_id)
    assert "| F1 | Rotate model |" in text
    assert LINK_URL in text or LINK_URL in json.dumps(blocks)
    assert "SECOND_PAGE_CONTENT" in text
    result = commit(workspace, executor)
    assert any("| F1 | Rotate model |" in evidence.content for evidence in result.evidence)
    assert result.evidence_tree.referenced_part_count == len(
        EvidencePlanEngine(workspace).eligible_parts())


def test_pdf_pipeline_persists_exact_source_and_normalized_sourcepart_content(tmp_path, manual_pdf):
    workspace, executor = workspace_for(tmp_path, manual_pdf)
    candidates = inventory(workspace, executor)
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_bytes() == manual_pdf
    for part in candidates:
        if not part.text_artifact_id:
            continue
        text = workspace.store.artifact_path(part.text_artifact_id).read_text(encoding="utf-8")
        assert text == workspace.part_text[part.part_id]
        assert part.char_count == len(text)
        assert part.content_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert part.metadata["structure_artifact_id"] in workspace.allowed_artifacts
        assert part.metadata["extraction_mode"] == "structured"


def test_image_only_pdf_materializes_grounded_evidence_image_refs(tmp_path, image_only_pdf):
    workspace, executor = workspace_for(tmp_path, image_only_pdf)
    candidates = inventory(workspace, executor)
    assert any(part.metadata.get("image_refs") for part in candidates)
    result = commit(workspace, executor)
    refs = [reference for evidence in result.evidence for reference in evidence.image_refs]
    assert refs, "Image-only pages must survive all the way into Evidence"
    for reference in refs:
        assert reference.artifact_id in workspace.allowed_artifacts
        assert workspace.store.artifact_path(reference.artifact_id).read_bytes()
        assert reference.media_type.startswith("image/")
        assert reference.page == 1
        assert reference.source_part_id in workspace.parts
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_bytes() == image_only_pdf


@pytest.mark.parametrize("force_recovery", [False, True])
def test_pdf_visual_resources_survive_long_staging_paths(
    tmp_path, image_only_pdf, force_recovery,
):
    from app.pdf_sourcepart import extractor
    from app.pdf_sourcepart.filesystem import filesystem_path

    source = tmp_path / "image-only.pdf"
    source.write_bytes(image_only_pdf)
    output = tmp_path / ("nested-pdf-staging-" * 10) / "extracted"
    assert len(str(output.resolve())) > 260
    document = extractor.extract_pdf(
        source, output, fail_pages={1} if force_recovery else None,
    )
    visual_blocks = [
        block for page in document["pages"] for block in page["blocks"]
        if block["type"] in {"image", "page_image"}
    ]
    assert visual_blocks, document["summary"]
    for block in visual_blocks:
        asset = filesystem_path(output) / "assets" / block["filename"]
        assert len(str(asset)) > 260
        assert asset.read_bytes(), block
    assert document["summary"]["all_pages_represented"] is True
    assert (filesystem_path(output) / "raw" / "source.pdf").read_bytes() == image_only_pdf


def test_optional_visual_group_failure_preserves_original_evidence(tmp_path, image_only_pdf):
    workspace, executor = workspace_for(tmp_path, image_only_pdf)
    with patch.object(executor, '_create_pdf_visual_groups', side_effect=RuntimeError('renderer unavailable')):
        candidates = inventory(workspace, executor)
    assert any(part.metadata.get('image_refs') for part in candidates)
    assert any('original image Artifacts remain available' in warning for warning in workspace.warnings)
    result = commit(workspace, executor)
    assert any(evidence.image_refs for evidence in result.evidence)


def test_one_page_structure_failure_keeps_text_image_and_other_page_evidence(tmp_path, manual_pdf):
    from app.pdf_sourcepart import extractor

    workspace, executor = workspace_for(tmp_path, manual_pdf)
    original = extractor.structured_page

    def fail_first_page(page, snapshot, *args, **kwargs):
        if page.number == 0:
            raise RuntimeError("synthetic layout parser failure")
        return original(page, snapshot, *args, **kwargs)

    with patch.object(extractor, "structured_page", side_effect=fail_first_page):
        candidates = inventory(workspace, executor)
    modes = {part.metadata["extraction_mode"] for part in candidates
             if part.kind in {"pdf_section", "pdf_page_content"}}
    assert "basic_text" in modes
    assert "structured" in modes
    result = commit(workspace, executor)
    text = "\n".join(evidence.content for evidence in result.evidence)
    assert "A wrapped paragraph begins here" in text
    assert "SECOND_PAGE_CONTENT" in text
    assert any(ref.page == 1 for evidence in result.evidence for ref in evidence.image_refs), (
        "Fallback retains a full-page visual as well as basic machine text")
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_bytes() == manual_pdf


def test_pdf_normalization_version_changes_without_invalidating_html(tmp_path, manual_pdf):
    pdf, _ = workspace_for(tmp_path / "pdf", manual_pdf)
    html, _ = workspace_for(tmp_path / "html", b"<h1>Manual</h1><p>Instructions</p>",
                            filename="manual.html")
    assert pdf.NORMALIZATION_VERSION != "v13", "The former PDF normalization cache must expire"
    assert "pdf" in pdf.NORMALIZATION_VERSION.lower()
    assert html.NORMALIZATION_VERSION == "v16-html-structured.v11-random-navigation.v2"
    assert pdf.NORMALIZATION_VERSION != html.NORMALIZATION_VERSION


def test_source_validation_preserves_authored_carriage_returns(tmp_path, manual_pdf):
    workspace, executor = workspace_for(tmp_path, manual_pdf)
    candidates = inventory(workspace, executor)
    part = next(p for p in candidates if p.text_artifact_id)
    exact = workspace.part_text[part.part_id] + '\nCarriage\rReturn\r\nMixed line endings.'
    workspace.add_part(part, exact)
    result = commit(workspace, executor)
    assert any(exact == e.content for e in result.evidence)
    assert workspace.store.artifact_path(part.text_artifact_id).read_bytes().decode('utf-8') == exact


def test_vector_diagram_preserves_visual_in_sourcepart_and_evidence(tmp_path):
    with fitz.open() as document:
        page = document.new_page(width=540, height=720)
        page.insert_text((72, 72), 'Read the diagram below alongside its explanation.')
        for i in range(10):
            x, y = 80 + (i % 5) * 80, 150 + (i // 5) * 90
            page.draw_polyline([(x,y),(x+25,y+40),(x+50,y)],color=(0,0,0))
        data = document.tobytes()
    workspace, executor = workspace_for(tmp_path, data)
    candidates = inventory(workspace, executor)
    page_part = next(p for p in workspace.parts.values() if p.kind=='pdf_page')
    structure = workspace.store.read_json_artifact(page_part.metadata['structure_artifact_id'])
    visuals = [b for b in structure['blocks'] if b.get('supplemental_visual')]
    assert visuals and 'vector_graphics_or_complex_layout' in visuals[0]['reasons']
    assert workspace.store.artifact_path(visuals[0]['artifact_id']).read_bytes().startswith(b'\x89PNG')
    assert any(p.metadata.get('image_refs') for p in candidates)
    result = commit(workspace, executor)
    assert any(e.image_refs for e in result.evidence)


def test_visual_selection_checks_original_spans_when_sorted_text_omits_glyphs():
    from app.pdf_sourcepart.extractor import visual_preservation_reasons
    from unittest.mock import Mock
    page = Mock()
    page.get_drawings.return_value = []
    page.get_image_info.return_value = []
    snapshot = {'raw_text':'Clean sorted projection', 'lines':[
        {'text':'Unmapped\x07 glyph', 'runs':[{'font':'Helvetica','text':'Unmapped\x07 glyph'}]}]}
    assert 'unmapped_text_glyphs' in visual_preservation_reasons(page,snapshot)


def test_prose_before_first_heading_survives_exactly_once_in_evidence(tmp_path):
    preface = "PREFACE_ONLY remains before the first section heading."
    body = "BODY_ONLY remains inside the authored Commands section."
    with fitz.open() as document:
        page = document.new_page(width=540, height=720)
        page.insert_text((72, 72), preface, fontsize=11)
        page.insert_text((72, 180), "Commands", fontsize=18, fontname="hebo")
        page.insert_text((72, 220), body, fontsize=11)
        document.set_toc([[1, "Commands", 1]])
        data = document.tobytes()
    workspace, executor = workspace_for(tmp_path, data)
    inventory(workspace, executor)
    result = commit(workspace, executor)
    text = "\n".join(evidence.content for evidence in result.evidence)
    assert text.count(preface) == 1, "A section later on the page cannot suppress its preceding prose"
    assert text.count(body) == 1, "The residual page projection must not duplicate assigned section text"


def test_unavailable_primary_engine_recovers_both_pages_and_visual_evidence(tmp_path, manual_pdf):
    from app.pdf_sourcepart import extractor

    workspace, executor = workspace_for(tmp_path, manual_pdf)
    with patch.object(extractor, "fitz", None), patch.object(
            executor, "_create_pdf_visual_groups",
            side_effect=AssertionError("Legacy image grouping must not reopen a failed primary document")):
        response = executor.call("pdf_inventory", {})
    assert response["ok"], response
    assert response["page_count"] == 2
    result = commit(workspace, executor)
    text = "\n".join(evidence.content for evidence in result.evidence)
    assert "A wrapped paragraph begins here" in text
    assert "SECOND_PAGE_CONTENT" in text
    refs = [reference for evidence in result.evidence for reference in evidence.image_refs]
    assert {reference.page for reference in refs} == {1, 2}
    assert all(workspace.store.artifact_path(reference.artifact_id).read_bytes() for reference in refs)
    assert result.status == "degraded"
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_bytes() == manual_pdf


def test_invalid_pdf_retains_original_with_explicit_unknown_page_count(tmp_path):
    data = b"%PDF-1.7\nGenuinely invalid PDF: no objects, catalog, or page tree.\n%%EOF\n"
    workspace, executor = workspace_for(tmp_path, data)
    assert workspace.manifest.primary_modality == "pdf"
    response = executor.call("pdf_inventory", {})
    assert response["ok"], response
    assert response["page_count"] is None, "A retained-source placeholder is not a known physical PDF page"
    structure = workspace.store.read_json_artifact(response["pdf_document_structure_artifact_id"])
    assert structure["extraction"]["summary"]["page_count_known"] is False
    assert structure["extraction"]["summary"]["all_pages_represented"] is None
    candidates = [part for part in workspace.evidence_candidates()
                  if part.kind in {"pdf_section", "pdf_page_content"}]
    assert candidates and all(part.metadata["extraction_mode"] == "raw_pdf" for part in candidates)
    assert all(not part.locator["page_number_known"] for part in workspace.parts.values()
               if part.kind == "pdf_page")
    result = commit(workspace, executor)
    assert result.status == "degraded"
    assert result.gaps
    assert result.evidence
    assert all(not evidence.image_refs for evidence in result.evidence)
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_bytes() == data


@pytest.mark.parametrize("decoder_error", [RuntimeError, OSError])
def test_failed_embedded_image_extraction_keeps_page_visual_in_evidence(tmp_path, image_only_pdf, decoder_error):
    workspace, executor = workspace_for(tmp_path, image_only_pdf)
    with patch.object(fitz.Document, "extract_image", side_effect=decoder_error("unsupported image stream")):
        candidates = inventory(workspace, executor)
    page = next(part for part in workspace.parts.values() if part.kind == "pdf_page")
    structure = workspace.store.read_json_artifact(page.metadata["structure_artifact_id"])
    assert not any(block["type"] == "blank_page" for block in structure["blocks"])
    assert any(part.metadata.get("image_refs") for part in candidates)
    result = commit(workspace, executor)
    refs = [reference for evidence in result.evidence for reference in evidence.image_refs]
    assert refs, "An unreadable embedded image must be recovered through page rendering"
    assert all(workspace.store.artifact_path(reference.artifact_id).read_bytes() for reference in refs)
    assert result.status == "degraded"


def test_small_vector_diagram_with_text_survives_in_evidence(tmp_path):
    with fitz.open() as document:
        page = document.new_page(width=300, height=240)
        page.insert_text((30, 40), "The boxed region below is part of the instruction.")
        page.draw_rect(fitz.Rect(40, 80, 260, 200), color=(0, 0, 0), fill=(0.8, 0.9, 1))
        data = document.tobytes()
    workspace, executor = workspace_for(tmp_path, data)
    candidates = inventory(workspace, executor)
    assert any(part.metadata.get("image_refs") for part in candidates)
    result = commit(workspace, executor)
    assert any(evidence.image_refs for evidence in result.evidence), (
        "A single authored vector figure is not disposable because it is below a path-count threshold")


def test_separately_positioned_heading_number_and_name_form_one_section(tmp_path):
    with fitz.open() as document:
        page = document.new_page(width=540, height=720)
        page.insert_text((72, 120), "Chapter 7", fontsize=18, fontname="hebo")
        page.insert_text((72, 160), "Supported Operations", fontsize=22, fontname="hebo")
        page.insert_text((72, 230), "7.1", fontsize=16, fontname="hebo")
        page.insert_text((115, 230), "plus", fontsize=16, fontname="hebo")
        page.insert_text((72, 265), "Addition accepts two or more parameters.", fontsize=11)
        data = document.tobytes()
    workspace, executor = workspace_for(tmp_path, data)
    inventory(workspace, executor)
    text_blocks = [workspace.part_text[part.part_id] for part in workspace.parts.values()
                   if part.kind == "pdf_text_block"]
    assert "Chapter 7 Supported Operations" in text_blocks
    assert "7.1 plus" in text_blocks
    assert "Chapter 7" not in text_blocks
    assert "7.1" not in text_blocks
    result = commit(workspace, executor)
    evidence_text = "\n".join(evidence.content for evidence in result.evidence)
    assert evidence_text.count("Addition accepts two or more parameters.") == 1
    assert all(evidence.content.strip() != "7.1" for evidence in result.evidence)


def test_unavailable_vector_render_marks_evidence_degraded(tmp_path):
    from app.pdf_sourcepart import fallback

    with fitz.open() as document:
        page = document.new_page(width=300, height=240)
        page.insert_text((30, 40), "Text survives independently of its diagram.")
        page.draw_rect(fitz.Rect(40, 80, 260, 200), color=(0, 0, 0))
        data = document.tobytes()
    workspace, executor = workspace_for(tmp_path, data)
    with patch.object(fallback, "preserve_page_visual", return_value=None):
        inventory(workspace, executor)
    result = commit(workspace, executor)
    assert result.status == "degraded"
    assert result.gaps
    assert "Text survives independently" in "\n".join(evidence.content for evidence in result.evidence)
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_bytes() == data


@pytest.mark.parametrize("label,title_size,title_x,title_y,title", [
    ("Chapter 1", 18, 72, 160, "An independent sibling heading"),
    ("Section 2", 16, 72, 160, "A smaller subsection heading"),
    ("Part IV", 22, 320, 160, "An independent column heading"),
    ("Chapter 1", 22, 72, 160, "2 Another numbered heading"),
    ("Chapter 1", 22, 72, 100, "A preceding heading in reading geometry"),
])
def test_nearby_heading_labels_do_not_absorb_independent_headings(
        tmp_path, label, title_size, title_x, title_y, title):
    with fitz.open() as document:
        page = document.new_page(width=700, height=720)
        page.insert_text((72, 120), label, fontsize=18, fontname="hebo")
        page.insert_text((title_x, title_y), title, fontsize=title_size, fontname="hebo")
        page.insert_text((72, 265), "Body text remains available.", fontsize=11)
        data = document.tobytes()
    workspace, executor = workspace_for(tmp_path, data)
    inventory(workspace, executor)
    text_blocks = [workspace.part_text[part.part_id] for part in workspace.parts.values()
                   if part.kind == "pdf_text_block"]
    assert label in text_blocks
    assert title in text_blocks
    assert f"{label} {title}" not in text_blocks
