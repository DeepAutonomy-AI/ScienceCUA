"""Grounding and acquisition coverage checks for the opt-in HTML pipeline."""
from __future__ import annotations

import hashlib
import json
import base64
import threading
import time

import pytest
from bs4 import BeautifulSoup

from app.acquisition import AcquisitionService, FetchResult
from app.config import Settings
from app.evidence_planning import EvidencePlanEngine
from app.html_guided.pipeline import _model_link, _resolve_model_plan, _validate_plan, build_snapshot, run_guided_html
from app.html_sourcepart import extractor
from app.models import SourceManifest, TargetProfile
from app.source_tools.workspace import SourceWorkspace
from app.storage import RunStore
from app.toolkit import ToolExecutor


ROOT_URL = "https://example.org/docs/catalog.html"


def workspace_for(tmp_path, html: str, *, max_pages=8, max_rounds=3, batch_size=80, fetch_concurrency=4):
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        html_processing_mode="model_guided",
        html_guided_max_rounds=max_rounds,
        html_guided_link_batch_size=batch_size,
        html_guided_fetch_concurrency=fetch_concurrency,
        max_html_pages=max_pages,
        scope_selection_enabled=False,
    )
    store = RunStore(settings.data_dir, "run_guided_test")
    raw = store.put_artifact(
        html.encode("utf-8"), kind="raw_source", media_type="text/html", suffix=".html"
    )
    manifest = SourceManifest(
        source_id="SRC1",
        canonical_input_id="IN1",
        display_name="Example Manual",
        primary_modality="html",
        media_type="text/html",
        raw_artifact_id=raw.artifact_id,
        origin_url=ROOT_URL,
        final_url=ROOT_URL,
    )
    target = TargetProfile(
        software_name="Example", target_version="1", platform="General"
    )
    return SourceWorkspace(store, settings, AcquisitionService(store, settings), target, manifest)


def install_pages(monkeypatch, workspace, pages):
    requested = []

    def read_url(url, **kwargs):
        requested.append(url)
        response = pages[url]
        if isinstance(response, Exception):
            raise response
        return FetchResult(response.encode("utf-8"), "text/html", url, 200, {})

    monkeypatch.setattr(workspace.acquisition, "_read_url", read_url)
    return requested


def plan_for(payload, *, action="fetch", choose=None):
    page = payload["page"]
    container = next(
        (item for item in page["containers"] if item["tag"] == "main"),
        page["containers"][0],
    )
    return {
        "page_type": "directory" if page["links"] else "article",
        "content_root_id": container["container_id"],
        "links": [
            {
                "link_id": link["link_id"],
                "action": choose(link) if choose else action,
                "reason": "The observed link is a manual resource.",
                "group": "Tutorials",
            }
            for link in page["links"]
        ],
        "summary": "Use the authored main content and linked instructions.",
    }


def combined_text(workspace):
    return "\n\n".join(workspace.part_text.values())


@pytest.mark.asyncio
async def test_navigation_only_boundary_is_reselected_by_model_without_changing_link_choices(tmp_path, monkeypatch):
    workspace = workspace_for(tmp_path, '<html><body><div id="catalog"><nav><ul><li><a href="lesson.html">'
        + 'Lesson chapter ' * 20 + '</a></li></ul></nav></div>'
        '<main><h1>Manual</h1><p>Authored instructions to retain.</p></main></body></html>')
    requested = install_pages(monkeypatch, workspace, {
        'https://example.org/docs/lesson.html': '<main><h1>Lesson</h1><p>Execute the command.</p></main>'})
    calls = []

    async def decide(payload):
        calls.append(payload)
        plan = plan_for(payload)
        if len(calls) == 1:
            plan['content_root_id'] = next(c['container_id'] for c in payload['page']['containers'] if c['dom_id'] == 'catalog')
        return plan

    await run_guided_html(workspace, decide)
    assert 'navigation text only' in calls[1]['validation_error']
    assert requested == ['https://example.org/docs/lesson.html']
    assert 'Authored instructions to retain.' in combined_text(workspace)
    assert 'Execute the command.' in combined_text(workspace)


def assert_grounded_tree(workspace):
    engine = EvidencePlanEngine(workspace)
    plan = engine.build_skeleton_plan()
    artifacts = engine.materialize(plan)
    assert artifacts.tree.referenced_part_count == len(engine.eligible_parts())
    for node in artifacts.tree.nodes:
        spans = []
        for ref in node.source_refs:
            text = workspace.part_text[ref.source_part_id][ref.start_offset:ref.end_offset]
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == ref.content_sha256
            spans.append(text)
        assert node.content == "\n\n".join(spans)
    return artifacts.tree


@pytest.mark.asyncio
async def test_paragraph_catalog_links_can_cross_directories_and_subdomains(tmp_path, monkeypatch):
    html = """<html><head><title>Example Tutorials</title></head><body><main>
    <h1>Example Tutorials</h1><h2>Getting started</h2>
    <p><a href="/articles/run.html">Run a simulation</a></p>
    <p><a href="https://learn.example.org/guide.html">Plot results</a></p>
    </main></body></html>"""
    workspace = workspace_for(tmp_path, html)
    pages = {
        "https://example.org/articles/run.html": "<html><main><h1>Run a simulation</h1><p>Press Execute to start the simulation.</p></main></html>",
        "https://learn.example.org/guide.html": "<html><main><h1>Plot results</h1><p>Choose the voltage trace and press Plot.</p></main></html>",
    }
    requested = install_pages(monkeypatch, workspace, pages)
    snapshots = []

    async def decide(payload):
        snapshots.append(payload)
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert set(requested) == set(pages)
    assert result["child_page_count"] == 2
    assert result["acquisition_complete"] is True
    root_links = snapshots[0]["page"]["links"]
    assert {link["url"] for link in root_links} == set(pages)
    assert all(link["link_id"] and link["context"] for link in root_links)
    assert "Press Execute to start the simulation." in combined_text(workspace)
    assert "Choose the voltage trace and press Plot." in combined_text(workspace)
    tree = assert_grounded_tree(workspace)
    assert any("Press Execute" in node.content for node in tree.nodes)
    assert any("Choose the voltage trace" in node.content for node in tree.nodes)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_field", ["link_id", "content_root_id"])
async def test_unknown_model_ids_are_rejected_before_fetch(tmp_path, monkeypatch, invalid_field):
    workspace = workspace_for(tmp_path, '<html><main><h1>Manual</h1><p><a href="part.html">Part</a></p></main></html>')
    requested = install_pages(monkeypatch, workspace, {})
    calls = []

    async def decide(payload):
        calls.append(payload)
        plan = plan_for(payload)
        if invalid_field == "link_id":
            plan["links"][0]["link_id"] = "invented_link_id"
        else:
            plan["content_root_id"] = "invented_container_id"
        return plan

    with pytest.raises(ValueError):
        await run_guided_html(workspace, decide)
    assert requested == []
    assert len(calls) == 2
    assert calls[1]["validation_error"]


@pytest.mark.asyncio
async def test_python_enforces_child_page_budget_and_reports_unfetched_resources(tmp_path, monkeypatch):
    workspace = workspace_for(
        tmp_path,
        '<html><main><h1>Manual</h1><p><a href="a.html">A</a></p><p><a href="b.html">B</a></p></main></html>',
        max_pages=1,
    )
    pages = {
        "https://example.org/docs/a.html": "<html><main><h1>A</h1><p>Procedure A.</p></main></html>",
        "https://example.org/docs/b.html": "<html><main><h1>B</h1><p>Procedure B.</p></main></html>",
    }
    requested = install_pages(monkeypatch, workspace, pages)

    async def decide(payload):
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert len(requested) == 1
    assert result["child_page_count"] == 1
    assert result["acquisition_complete"] is False
    assert result["truncated"] is True
    assert len(result["unfetched_discovered_urls"]) == 1
    assert set(result["unfetched_discovered_urls"]) == set(pages) - set(requested)
    assert workspace.gaps


@pytest.mark.asyncio
async def test_deferred_missing_and_unreviewed_links_never_report_complete(tmp_path, monkeypatch):
    workspace = workspace_for(
        tmp_path,
        '<html><main><h1>Manual</h1><p><a href="failed.html">Failed</a></p><p><a href="later.html">Later</a></p><p><a href="forgotten.html">Forgotten</a></p></main></html>',
    )
    requested = install_pages(monkeypatch, workspace, {"https://example.org/docs/failed.html": OSError("fixture unavailable")})

    async def decide(payload):
        plan = plan_for(payload, choose=lambda link: "required_missing" if "later.html" in link["url"] else "fetch")
        plan["links"] = [item for item in plan["links"] if item["link_id"] != next(
            link["link_id"] for link in payload["page"]["links"] if "forgotten.html" in link["url"]
        )]
        return plan

    result = await run_guided_html(workspace, decide)
    assert set(requested) == {"https://example.org/docs/failed.html", "https://example.org/docs/later.html"}
    assert result["acquisition_complete"] is False
    assert len(result["unfetched_discovered_urls"]) == 3
    assert result["navigation_gaps"]
    assert workspace.gaps


@pytest.mark.asyncio
async def test_same_document_fragments_reuse_captured_content(tmp_path, monkeypatch):
    html = '<html><main><h1>Manual</h1><p><a href="#setup">Setup</a> <a href="catalog.html#run">Run</a></p><h2 id="setup">Setup</h2><p>Choose a workspace.</p><h2 id="run">Run</h2><p>Press Run.</p></main></html>'
    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        assert all(link["kind"] == "same_page_anchor" for link in payload["page"]["links"])
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert requested == []
    assert result["page_count"] == 1
    assert result["acquisition_complete"] is True
    tree = assert_grounded_tree(workspace)
    nodes = {node.title: node for node in tree.nodes}
    assert "Setup" in nodes and "Run" in nodes
    assert "Choose a workspace." in nodes["Setup"].content
    assert "Press Run." in nodes["Run"].content


@pytest.mark.asyncio
async def test_main_projection_preserves_capture_and_exact_code_table_content(tmp_path, monkeypatch):
    html = """<html><head><title>Manual</title></head><body>
    <header><p>Unrelated subscription promotion.</p></header>
    <main id="manual"><h1>Manual</h1><h2>Simulation</h2>
    <p>Choose <b>Transient</b>, then press Execute.</p>
    <pre>  solve(v)\n\n    plot(v)</pre>
    <table><tr><th>Key</th><th>Action</th></tr><tr><td>F5</td><td>Run simulation</td></tr></table>
    <h3>Save results</h3><p>Save the plotted trace as CSV.</p></main>
    <footer><p>Unrelated copyright boilerplate.</p></footer></body></html>"""
    workspace = workspace_for(tmp_path, html)
    original_id = workspace.manifest.raw_artifact_id
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert requested == []
    assert workspace.store.artifact_path(original_id).read_bytes() == html.encode("utf-8")
    text = combined_text(workspace)
    assert "subscription promotion" not in text
    assert "copyright boilerplate" not in text
    assert "Choose **Transient**, then press Execute." in text
    assert "  solve(v)\n\n    plot(v)" in text
    assert "F5" in text and "Run simulation" in text
    page_part = next(part for part in workspace.parts.values() if part.kind == "html_page")
    assert page_part.metadata["original_capture_artifact_id"] == original_id
    assert result["acquisition_complete"] is True
    tree = assert_grounded_tree(workspace)
    nodes = {node.title: node for node in tree.nodes}
    assert nodes["Save results"].parent_node_id == nodes["Simulation"].evidence_node_id
    assert "Save the plotted trace as CSV." in nodes["Save results"].content


@pytest.mark.asyncio
async def test_link_batches_do_not_silently_drop_candidates(tmp_path, monkeypatch):
    workspace = workspace_for(
        tmp_path,
        '<html><main><h1>Manual</h1>' + ''.join(f'<p><a href="part{i}.html">Part {i}</a></p>' for i in range(5)) + '</main></html>',
        batch_size=2,
    )
    requested = install_pages(monkeypatch, workspace, {})
    seen = []

    async def decide(payload):
        seen.extend(link["url"] for link in payload["page"]["links"])
        return plan_for(payload, action="exclude")

    result = await run_guided_html(workspace, decide)
    assert requested == []
    assert len(seen) == len(set(seen)) == 5
    assert result["acquisition_complete"] is True


@pytest.mark.asyncio
async def test_model_extra_fields_cannot_replace_observed_destination_or_audit(tmp_path, monkeypatch):
    workspace = workspace_for(tmp_path, '<html><main><h1>Manual</h1><p><a href="part.html">Part</a></p></main></html>')
    observed_url = "https://example.org/docs/part.html"
    requested = install_pages(monkeypatch, workspace, {
        observed_url: "<html><main><h1>Part</h1><p>Open the control panel.</p></main></html>"
    })

    async def decide(payload):
        plan = plan_for(payload)
        for decision in plan["links"]:
            decision.update(url="https://invented.example/poison.html", source_url="invented", status="succeeded")
        return plan

    result = await run_guided_html(workspace, decide)
    assert requested == [observed_url]
    records = workspace.store.read_json_artifact(result["decision_artifact_id"])["decisions"]
    assert len(records) == 1
    assert records[0]["url"] == observed_url
    assert records[0]["source_url"] == ROOT_URL
    assert records[0]["status"] == "fetched"
    assert "Open the control panel." in combined_text(workspace)


def test_disclosure_catalog_groups_and_opaque_ids_survive_snapshot():
    html = """<html><body><main><h1>Tutorials</h1><details><summary>Basic workflows</summary>
    <p><a href="open.html">Open data</a></p><details><summary>Plotting</summary>
    <p><a href="plot.html">Plot a curve</a></p></details></details></main></body></html>"""
    first, roots = build_snapshot(html, ROOT_URL)
    second, _ = build_snapshot(html, ROOT_URL)
    assert first == second
    assert roots.keys() == {item["container_id"] for item in first["containers"]}
    assert [item["summary"] for item in first["disclosures"]] == ["Basic workflows", "Plotting"]
    links = {item["label"]: item for item in first["links"]}
    assert links["Open data"]["disclosure_path"] == ["Basic workflows"]
    assert links["Plot a curve"]["disclosure_path"] == ["Basic workflows", "Plotting"]


@pytest.mark.asyncio
async def test_depth_bound_preserves_parent_but_marks_unexpanded_chapter(tmp_path, monkeypatch):
    workspace = workspace_for(tmp_path, '<html><main><h1>Manual</h1><p><a href="chapter.html">Chapter</a></p></main></html>', max_rounds=2)
    requested = install_pages(monkeypatch, workspace, {
        "https://example.org/docs/chapter.html": '<html><main><h1>Chapter</h1><p>Chapter introduction.</p><p><a href="procedure.html">Procedure</a></p></main></html>',
    })

    async def decide(payload):
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert requested == ["https://example.org/docs/chapter.html"]
    assert "Chapter introduction." in combined_text(workspace)
    assert result["acquisition_complete"] is False
    assert result["truncated"] is True
    assert result["unfetched_discovered_urls"] == ["https://example.org/docs/procedure.html"]
    assert any(item["status"] == "depth_exhausted" for item in result["navigation_gaps"])


@pytest.mark.asyncio
async def test_embedded_manual_is_retained_as_unexpanded_gap(tmp_path, monkeypatch):
    workspace = workspace_for(tmp_path, '<html><main><h1>Manual</h1><p>The instructions are embedded below.</p><iframe src="guide.html" title="Guide"></iframe></main></html>')
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert requested == []
    assert result["acquisition_complete"] is False
    assert any(item["status"] == "embedded_content_pending" for item in result["navigation_gaps"])
    assert "The instructions are embedded below." in combined_text(workspace)


@pytest.mark.asyncio
@pytest.mark.parametrize("outer_tag", ["body", "[document]"])
async def test_selected_body_or_document_keeps_appendix_outside_nested_main(tmp_path, monkeypatch, outer_tag):
    content = """<main><h1>Simulation Manual</h1>
    <p>Primary instructions: choose Transient, then press Execute.</p></main>
    <section id="appendix"><h2>Appendix: key map</h2>
    <p>Appendix instructions: press F5 to run the current simulation.</p>
    <pre>  solve(v)\n\n    save(v)</pre></section>"""
    html = "<html><head><title>Simulation Manual</title></head><body>" + content + "</body></html>" if outer_tag == "body" else content
    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        plan = plan_for(payload)
        selected = next(item for item in payload["page"]["containers"] if item["tag"] == outer_tag)
        plan["content_root_id"] = selected["container_id"]
        return plan

    result = await run_guided_html(workspace, decide)
    assert requested == []
    assert result["acquisition_complete"] is True
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_bytes() == html.encode("utf-8")
    text = combined_text(workspace)
    assert "Primary instructions: choose Transient, then press Execute." in text
    assert "Appendix instructions: press F5 to run the current simulation." in text
    assert "  solve(v)\n\n    save(v)" in text
    tree = assert_grounded_tree(workspace)
    assert any("Primary instructions: choose Transient, then press Execute." in node.content for node in tree.nodes)
    assert any("Appendix instructions: press F5 to run the current simulation." in node.content for node in tree.nodes)


@pytest.mark.asyncio
async def test_large_capture_bounds_model_requests_without_losing_link_decisions(tmp_path, monkeypatch):
    headings = ''.join(f'<h2 id="section{i}">Section {i}: {"Detailed authored chapter title " * 5}</h2>' for i in range(220))
    links = []
    for index in range(110):
        href = f'https://docs.example.org/{"manual/" * 12}chapter-{index}.html'
        links.append(
            f'<p><a href="{href}">Chapter {index}: {"Complete operating instructions " * 8}</a>'
            + (" Additional context describing the procedure." * 20)
            + f'<a href="{href}">Repeat chapter {index}</a></p>'
        )
    wrappers = ''.join(f'<div class="wrapper-{index}">' for index in range(18))
    html = '<html><head><title>Large Manual</title></head><body><main>' + wrappers + headings + ''.join(links) + '</div>' * 18 + '</main></body></html>'
    workspace = workspace_for(tmp_path, html, batch_size=80)
    requested = install_pages(monkeypatch, workspace, {})
    payloads, observed_ids, selected_roots = [], [], []

    async def decide(payload):
        payloads.append(payload)
        current_container_ids = {item["container_id"] for item in payload["page"]["containers"]}
        assert all(identity.startswith("C") for identity in current_container_ids)
        assert all(set(link["content_root_ids"]) <= current_container_ids for link in payload["page"]["links"])
        assert all(len(link["landmark_context"]) <= 4 for link in payload["page"]["links"])
        observed_ids.extend(link["link_id"] for link in payload["page"]["links"])
        plan = plan_for(payload, action="exclude")
        selected_roots.append(plan["content_root_id"])
        return plan

    result = await run_guided_html(workspace, decide)
    assert requested == []
    assert result["acquisition_complete"] is True
    assert len(payloads) > 2  # Size-aware batches must be smaller than the configured 80 links.
    assert all(len(json.dumps(payload, ensure_ascii=False)) <= 48_000 for payload in payloads)
    assert len(selected_roots) > 1 and len(set(selected_roots)) == 1
    snapshots = [workspace.store.read_json_artifact(artifact_id) for artifact_id in result["pipeline_artifact_ids"]
                 if workspace.store.artifact_record(artifact_id).kind == "html_guided_snapshot"]
    assert len(snapshots) == 1
    full = snapshots[0]
    expected_ids = {link["link_id"] for link in full["links"]}
    assert len(expected_ids) == 110
    assert len(observed_ids) == len(set(observed_ids)) == 110
    assert set(observed_ids) == {f"L{index}" for index in range(1, 111)}
    audit = workspace.store.read_json_artifact(result["decision_artifact_id"])["decisions"]
    assert {record["link_id"] for record in audit} == expected_ids
    assert len(full["headings"]) == 220
    assert all(link["occurrence_count"] == 2 for link in full["links"])
    assert all(len(link["context"]) > 180 for link in full["links"])
    assert all(len(link["dom_path"]) > 300 for link in full["links"])
    assert any(len(container["text_preview"]) > 400 for container in full["containers"])


@pytest.mark.asyncio
@pytest.mark.parametrize("css_owner", ["body", "div"])
async def test_main_projection_preserves_ancestor_css_visibility(tmp_path, monkeypatch, css_owner):
    main = """<main><h1>Manual</h1><p>Visible instructions: open the control panel.</p>
    <pre>run()<span class="hidden">Inline hidden tooltip.</span>\nfinish()</pre>
    <p class="hidden">Hidden supplemental instructions.</p></main>"""
    content = '<div class="manual">' + main + '</div>' if css_owner == "div" else main
    body_class = ' class="manual"' if css_owner == "body" else ""
    html = f'<html lang="en"><head><title>Manual</title><style>{css_owner}.manual .hidden {{display:none}}</style></head><body{body_class}>' + content + '</body></html>'
    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert requested == []
    text = combined_text(workspace)
    assert "Visible instructions: open the control panel." in text
    assert "run()\nfinish()" in text
    assert "Inline hidden tooltip." not in text
    assert "Hidden supplemental instructions." in text
    assert "<summary>Content hidden in the original HTML</summary>" in text
    annotations, blocks = [], []
    for part in workspace.evidence_candidates():
        saved = workspace.store.read_json_artifact(part.metadata["structure_artifact_id"])
        annotations.extend(saved["annotations"])
        blocks.extend(extractor.all_blocks(saved["blocks"]))
    assert any("Inline hidden tooltip." in extractor.plain(item["runs"]) for item in annotations)
    assert any(item["type"] == "hidden_content" for item in blocks)
    assert result["acquisition_complete"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("image_location", ["closed_dialog", "open_dialog", "article_figure"])
async def test_empty_closed_dialog_placeholder_does_not_degrade_authored_article(tmp_path, monkeypatch, image_location):
    markup = {
        "closed_dialog": '<dialog><img src=""></dialog>',
        "open_dialog": '<dialog open><img src=""></dialog>',
        "article_figure": '<figure><img src="" alt="Required simulation diagram"></figure>',
    }[image_location]
    html = '<html><head><title>Manual</title></head><body><main><h1>Manual</h1><h2>Run simulation</h2><p>Choose Transient, then press Execute.</p>' + markup + '</main></body></html>'
    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    page = result["pages"][0]
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_bytes() == html.encode("utf-8")
    assert "Choose Transient, then press Execute." in combined_text(workspace)
    if image_location == "closed_dialog":
        assert requested == []
        assert page["extraction"]["mode"] == "structured"
        adjustments = page["projection_adjustments"]
        assert len(adjustments) == 1
        assert adjustments[0]["kind"] == "empty_closed_dialog_image"
        assert adjustments[0]["dom_path"] and '<img src=""' in adjustments[0]["raw_markup"]
        metadata = workspace.store.artifact_record(page["projection_artifact_id"]).metadata
        assert metadata["projection_adjustments"] == adjustments
        tree = assert_grounded_tree(workspace)
        assert any(node.title == "Run simulation" for node in tree.nodes)
    else:
        assert page["extraction"]["mode"] != "structured"
        assert any(item["code"] == "structured_image_source_missing" for item in page["extraction"]["reasons"])
        assert page["missing_image_count"] >= 1
        assert workspace.warnings or workspace.gaps
        assert not page.get("projection_adjustments")


@pytest.mark.asyncio
@pytest.mark.parametrize("has_visible_anchor_prose", [False, True])
async def test_image_resource_anchor_placeholder_keeps_link_and_real_figure(tmp_path, monkeypatch, has_visible_anchor_prose):
    image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2ioAAAAASUVORK5CYII="
    inside = "Visible explanation must remain a block." if has_visible_anchor_prose else '<noscript>&lt;img src="fallback.png" alt="ESCAPED-FALLBACK"&gt;</noscript>'
    html = '<html><head><title>Manual</title></head><body><main><h1>Manual</h1><h2>Run simulation</h2><p>Choose Transient, then press Execute.</p><a href="/media/high-resolution.png" class="figure-link"><div class="image-wrapper">' + inside + '</div></a><img alt="Captured circuit diagram" src="data:image/png;base64,' + image_base64 + '"></main></body></html>'
    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        return plan_for(payload, action="exclude")

    result = await run_guided_html(workspace, decide)
    page = result["pages"][0]
    assert requested == []
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_bytes() == html.encode("utf-8")
    text = combined_text(workspace)
    assert "Choose Transient, then press Execute." in text
    assert "ESCAPED-FALLBACK" not in text
    assert page["saved_image_count"] == 1
    document = workspace.store.read_json_artifact(page["structured_source_artifact_id"])
    saved_image = next(iter(document["images"].values()))
    assert workspace.store.artifact_path(saved_image["artifact_id"]).read_bytes() == base64.b64decode(image_base64)
    changes = [item for item in page.get("projection_adjustments", []) if item["kind"] == "empty_image_link_wrappers"]
    if not has_visible_anchor_prose:
        assert page["extraction"]["mode"] == "structured"
        assert any(link["url"] == "https://example.org/media/high-resolution.png" for link in document["links"].values())
        assert len(changes) == 1 and changes[0]["changed_dom_paths"]
        assert "<noscript>" in changes[0]["raw_markup"]
        tree = assert_grounded_tree(workspace)
        assert any(node.title == "Run simulation" for node in tree.nodes)
    else:
        assert "Visible explanation must remain a block." in text
        assert changes == []
        projected = workspace.store.artifact_path(page["projection_artifact_id"]).read_text(encoding="utf-8")
        assert '<div class="image-wrapper">Visible explanation must remain a block.</div>' in projected


@pytest.mark.asyncio
async def test_book_anchors_and_captured_menus_resolve_without_model_classification(tmp_path, monkeypatch):
    def book_page(title, prefix, menu):
        toc = ''.join(f'<a href="#{prefix}{index}">{title} section {index}</a>' for index in range(20))
        sections = ''.join(f'<h2 id="{prefix}{index}">{title} section {index}</h2><p>{title} instruction {index}: choose the documented control.</p>' for index in range(20))
        return '<html><head><title>' + title + '</title></head><body><main><h1>' + title + '</h1><nav>' + toc + menu + '</nav>' + sections + '</main></body></html>'

    chapter_a = "https://example.org/docs/chapter-a.html"
    chapter_b = "https://example.org/docs/chapter-b.html"
    root_html = book_page("Book", "root", '<a href="chapter-a.html">Chapter A</a>')
    workspace = workspace_for(tmp_path, root_html)
    requested = install_pages(monkeypatch, workspace, {
        chapter_a: book_page("Chapter A", "a", '<a href="catalog.html">Contents</a><a href="chapter-b.html">Chapter B</a>'),
        chapter_b: book_page("Chapter B", "b", '<a href="catalog.html">Contents</a><a href="chapter-a.html">Chapter A</a>'),
    })
    offered_by_page = {}

    async def decide(payload):
        links = payload["page"]["links"]
        assert all(link["kind"] != "same_page_anchor" for link in links)
        offered_by_page[payload["page"]["url"]] = [link["url"] for link in links]
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert requested == [chapter_a, chapter_b]
    assert offered_by_page == {ROOT_URL: [chapter_a], chapter_a: [chapter_b], chapter_b: []}
    assert result["child_page_count"] == 2
    assert result["acquisition_complete"] is True
    assert result["unfetched_discovered_urls"] == []
    snapshots = [workspace.store.read_json_artifact(artifact_id) for artifact_id in result["pipeline_artifact_ids"]
                 if workspace.store.artifact_record(artifact_id).kind == "html_guided_snapshot"]
    observed_ids = [link["link_id"] for snapshot in snapshots for link in snapshot["links"]]
    records = workspace.store.read_json_artifact(result["decision_artifact_id"])["decisions"]
    assert len(observed_ids) == 65
    assert len(records) == len({record["link_id"] for record in records}) == 65
    assert {record["link_id"] for record in records} == set(observed_ids)
    host_decisions = [record for record in records if record.get("decision_origin") == "host"]
    assert len(host_decisions) == 63
    assert sum(record["status"] == "resolved_local_anchor" for record in host_decisions) == 60
    assert sum(record["status"] == "duplicate" for record in host_decisions) == 3
    assert all(record["action"] == "host_resolve" for record in host_decisions)
    assert "Chapter A instruction 0: choose the documented control." in combined_text(workspace)
    assert "Chapter B instruction 0: choose the documented control." in combined_text(workspace)
    assert_grounded_tree(workspace)


@pytest.mark.asyncio
async def test_short_model_aliases_stay_stable_across_batches_and_map_to_canonical_audit(tmp_path, monkeypatch):
    html = '<html><main><h1>Manual</h1>' + ''.join(f'<p><a href="chapter{index}.html">Chapter {index}</a></p>' for index in range(1, 6)) + '</main></html>'
    workspace = workspace_for(tmp_path, html, batch_size=2)
    selected_url = "https://example.org/docs/chapter4.html"
    requested = install_pages(monkeypatch, workspace, {
        selected_url: '<html><main><h1>Chapter 4</h1><p>Run the fourth documented procedure.</p></main></html>',
    })
    root_batches = []

    async def decide(payload):
        page = payload["page"]
        assert all(item["container_id"].startswith("C") for item in page["containers"])
        assert all(item["link_id"].startswith("L") for item in page["links"])
        if page["url"] == ROOT_URL:
            root_batches.append([item["link_id"] for item in page["links"]])
        return plan_for(payload, choose=lambda link: "fetch" if link["url"] == selected_url else "exclude")

    result = await run_guided_html(workspace, decide)
    assert root_batches == [["L1", "L2"], ["L3", "L4"], ["L5"]]
    assert requested == [selected_url]
    snapshots = [workspace.store.read_json_artifact(artifact_id) for artifact_id in result["pipeline_artifact_ids"]
                 if workspace.store.artifact_record(artifact_id).kind == "html_guided_snapshot"]
    root_snapshot = next(item for item in snapshots if item["url"] == ROOT_URL)
    canonical_by_url = {item["url"]: item["link_id"] for item in root_snapshot["links"]}
    records = workspace.store.read_json_artifact(result["decision_artifact_id"])["decisions"]
    assert len(records) == 5
    assert {item["link_id"] for item in records} == set(canonical_by_url.values())
    assert all(item["link_id"] == canonical_by_url[item["url"]] for item in records)
    selected_record = next(item for item in records if item["status"] == "fetched")
    assert selected_record["link_id"] == canonical_by_url[selected_url]
    assert "Run the fourth documented procedure." in combined_text(workspace)
    page_part = next(part for part in workspace.parts.values() if part.kind == "html_page" and part.locator["url"] == ROOT_URL)
    assert page_part.metadata["content_root_id"] in {item["container_id"] for item in root_snapshot["containers"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_scope", ["another_batch", "previous_page"])
async def test_model_alias_must_belong_to_the_current_page_and_batch(tmp_path, monkeypatch, stale_scope):
    html = '<html><main><h1>Manual</h1><p><a href="chapter-a.html">Chapter A</a></p><p><a href="chapter-b.html">Chapter B</a></p></main></html>'
    workspace = workspace_for(tmp_path, html, batch_size=1 if stale_scope == "another_batch" else 80)
    chapter_a = "https://example.org/docs/chapter-a.html"
    requested = install_pages(monkeypatch, workspace, {
        chapter_a: '<html><main><h1>Chapter A</h1><p><a href="procedure.html">Procedure</a></p></main></html>',
    })
    rejected_calls = []

    async def decide(payload):
        plan = plan_for(payload, action="exclude")
        if stale_scope == "another_batch":
            wrong_scope = payload["batch"]["index"] == 2
            bad_alias = "L1"
        else:
            wrong_scope = payload["page"]["url"] == chapter_a
            bad_alias = "L2"
            if not wrong_scope:
                plan = plan_for(payload, choose=lambda link: "fetch" if link["url"] == chapter_a else "exclude")
        if wrong_scope:
            rejected_calls.append(payload)
            plan["links"][0]["link_id"] = bad_alias
        return plan

    with pytest.raises(ValueError):
        await run_guided_html(workspace, decide)
    assert len(rejected_calls) == 2
    assert rejected_calls[1]["validation_error"]
    assert requested == ([] if stale_scope == "another_batch" else [chapter_a])


@pytest.mark.asyncio
@pytest.mark.parametrize("default_action", ["exclude", "required_missing", "fetch"])
async def test_batch_default_never_implicitly_fetches_links(tmp_path, monkeypatch, default_action):
    workspace = workspace_for(tmp_path, '<html><main><h1>Manual</h1><p><a href="a.html">A</a></p><p><a href="b.html">B</a></p></main></html>')
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        plan = plan_for(payload)
        reason = "These are required manual pages that have not been captured." if default_action == "required_missing" else "These observed links are outside the requested material."
        plan.update(links=[], default_action=default_action, default_reason=reason)
        return plan

    if default_action == "fetch":
        with pytest.raises(ValueError):
            await run_guided_html(workspace, decide)
    else:
        result = await run_guided_html(workspace, decide)
        records = workspace.store.read_json_artifact(result["decision_artifact_id"])["decisions"]
        assert len(records) == 2
        assert all(item["status"] == ("excluded" if default_action == "exclude" else "deferred") for item in records)
        assert result["acquisition_complete"] is (default_action == "exclude")
    assert requested == []


@pytest.mark.asyncio
@pytest.mark.parametrize("selected_region", ["body", "main"])
@pytest.mark.parametrize("projection_path", ["plain", "closed_dialog", "image_link_noscript"])
async def test_lean_block_code_outside_pre_preserves_literal_indentation(tmp_path, monkeypatch, selected_region, projection_path):
    expected = (
        "def next (n : Nat) : Nat :=\n"
        "  match n with\n"
        "    | 0 => 1\n"
        "    | Nat.succ k =>\n"
        "      let result := k + 2\n"
        "      result"
    )
    code = (
        '<code class="hl lean block"><span class="keyword">def</span> <span class="definition">next</span> <span>(n : Nat) : Nat :=</span>\n'
        '  <span class="keyword">match</span> <span>n</span> <span class="keyword">with</span>\n'
        '    <span>| 0 =&gt; 1</span>\n'
        '    <span>| Nat.succ k =&gt;</span>\n'
        '      <span class="keyword">let</span> <span>result := k + 2</span>\n'
        '      <span>result</span></code>'
    )
    extra = {
        "plain": "",
        "closed_dialog": '<dialog><img src=""></dialog>',
        "image_link_noscript": '<a href="/media/highres.png"><div><noscript>&lt;img src="fallback.png"&gt;</noscript></div></a>',
    }[projection_path]
    html = '<html><head><title>Lean Manual</title></head><body><main><h1>Lean Manual</h1><h2>Pattern matching</h2><p>Keep the indentation in this Lean definition.</p>' + code + extra + '</main></body></html>'
    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        plan = plan_for(payload, action="exclude")
        selected = next(item for item in payload["page"]["containers"] if item["tag"] == selected_region)
        plan["content_root_id"] = selected["container_id"]
        return plan

    result = await run_guided_html(workspace, decide)
    page = result["pages"][0]
    assert requested == []
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_bytes() == html.encode("utf-8")
    assert page["extraction"]["mode"] == "structured"
    projected = workspace.store.artifact_path(page["projection_artifact_id"]).read_text(encoding="utf-8")
    assert BeautifulSoup(projected, "html5lib").find("code").get_text() == expected
    assert "```\n" + expected + "\n```" in combined_text(workspace)
    code_blocks = []
    for part in workspace.evidence_candidates():
        structure = workspace.store.read_json_artifact(part.metadata["structure_artifact_id"])
        code_blocks.extend(block for block in extractor.all_blocks(structure["blocks"]) if block["type"] == "code")
    assert len(code_blocks) == 1
    assert extractor.plain(code_blocks[0]["runs"]) == expected
    tree = assert_grounded_tree(workspace)
    assert any("```\n" + expected + "\n```" in node.content for node in tree.nodes)
    if projection_path != "plain":
        expected_kind = "empty_closed_dialog_image" if projection_path == "closed_dialog" else "empty_image_link_wrappers"
        assert any(item["kind"] == expected_kind for item in page["projection_adjustments"])


@pytest.mark.asyncio
async def test_guided_block_image_anchor_keeps_figure_and_link_while_legacy_recovers(tmp_path, monkeypatch):
    # The image bytes need only be retained; the parser must never fetch the linked high-resolution URL.
    image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2ioAAAAASUVORK5CYII="
    html = '<html><head><title>Manual</title></head><body><main><h1>Manual</h1><h2>Run simulation</h2><p>Choose Transient, then press Execute.</p><a href="/media/full.png"><div class="figure-container"><img alt="Circuit diagram" src="data:image/png;base64,' + image_base64 + '"></div></a></main></body></html>'
    workspace = workspace_for(tmp_path / "guided", html)
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        return plan_for(payload, action="exclude")

    result = await run_guided_html(workspace, decide)
    page = result["pages"][0]
    assert requested == []
    assert page["extraction"]["mode"] == "structured"
    document = workspace.store.read_json_artifact(page["structured_source_artifact_id"])
    outer_link = next(link_id for link_id, link in document["links"].items() if link["url"] == "https://example.org/media/full.png")
    image = next(iter(document["images"].values()))
    assert workspace.store.artifact_path(image["artifact_id"]).read_bytes() == base64.b64decode(image_base64)
    image_runs = []
    for part in workspace.evidence_candidates():
        structure = workspace.store.read_json_artifact(part.metadata["structure_artifact_id"])
        image_runs.extend(run for run in extractor.all_runs(structure["blocks"]) if run["type"] == "image")
    assert len(image_runs) == 1 and image_runs[0]["link_id"] == outer_link
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_bytes() == html.encode("utf-8")
    projected = workspace.store.artifact_path(page["projection_artifact_id"]).read_text(encoding="utf-8")
    assert '<div class="figure-container">' in projected
    assert not any(item["kind"] == "empty_image_link_wrappers" for item in page["projection_adjustments"])
    assert any(node.title == "Run simulation" for node in assert_grounded_tree(workspace).nodes)

    legacy = workspace_for(tmp_path / "legacy", html, max_pages=0)
    legacy.manifest.source_metadata["html_processing_mode"] = "legacy"
    install_pages(monkeypatch, legacy, {})
    old = ToolExecutor(legacy, "legacy_comparison").call("html_inventory", {"max_pages": 0})
    assert old["ok"]
    old_page = old["pages"][0]
    assert old_page["extraction"]["mode"] == "basic_text"
    assert any(item["code"] == "structured_link_coverage_failed" for item in old_page["extraction"]["reasons"])


@pytest.mark.asyncio
async def test_guided_clickable_card_keeps_visible_footer_and_nested_link_identity(tmp_path, monkeypatch):
    html = """<html><head><title>Manual</title></head><body><main id="manual"><h1>Manual</h1><h2>Workflows</h2>
    <a href="/workflow.html"><div class="card"><header><p>Card header: Load a model.</p></header>
    <section class="card-main"><p>Card body: Choose the simulation controls.</p></section>
    <footer><p>Card footer: Open workflow now.</p></footer>
    <p>Optional <a href="/help.html">Own nested help</a></p></div></a>
    </main></body></html>"""
    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        return plan_for(payload, action="exclude")

    result = await run_guided_html(workspace, decide)
    assert requested == []
    page = result["pages"][0]
    assert page["extraction"]["mode"] == "structured"
    text = combined_text(workspace)
    for phrase in ("Card header: Load a model.", "Card body: Choose the simulation controls.", "Card footer: Open workflow now.", "Own nested help"):
        assert phrase in text
    document = workspace.store.read_json_artifact(page["structured_source_artifact_id"])
    by_url = {item["url"]: link_id for link_id, item in document["links"].items()}
    outer = by_url["https://example.org/workflow.html"]
    nested = by_url["https://example.org/help.html"]
    assert outer != nested
    runs = []
    for part in workspace.evidence_candidates():
        structure = workspace.store.read_json_artifact(part.metadata["structure_artifact_id"])
        runs.extend(extractor.all_runs(structure["blocks"]))
    assert any("Card footer:" in run.get("text", "") and run.get("link_id") == outer for run in runs)
    assert any("Own nested help" in run.get("text", "") and run.get("link_id") == nested for run in runs)
    assert not any("Own nested help" in run.get("text", "") and run.get("link_id") == outer for run in runs)
    assert_grounded_tree(workspace)


@pytest.mark.asyncio
@pytest.mark.parametrize("code_container", ["pre", "code_block"])
async def test_guided_block_anchor_keeps_literal_code_and_its_link_sidecar(tmp_path, monkeypatch, code_container):
    expected = "def next (n : Nat) : Nat :=\n  match n with\n    | 0 => 1\n    | Nat.succ k => k + 2"
    markup = '<span>def next (n : Nat) : Nat :=</span>\n  <span>match n with</span>\n    <span>| 0 =&gt; 1</span>\n    <span>| Nat.succ k =&gt; k + 2</span>'
    code = '<pre><code>' + markup + '</code></pre>' if code_container == "pre" else '<code class="hl lean block">' + markup + '</code>'
    html = '<html><head><title>Manual</title></head><body><main><h1>Manual</h1><h2>Examples</h2><a href="/example.html"><div class="example">' + code + '</div></a></main></body></html>'
    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        return plan_for(payload, action="exclude")

    result = await run_guided_html(workspace, decide)
    page = result["pages"][0]
    assert requested == []
    assert page["extraction"]["mode"] == "structured"
    document = workspace.store.read_json_artifact(page["structured_source_artifact_id"])
    outer_link = next(link_id for link_id, link in document["links"].items() if link["url"] == "https://example.org/example.html")
    code_blocks = []
    for part in workspace.evidence_candidates():
        structure = workspace.store.read_json_artifact(part.metadata["structure_artifact_id"])
        code_blocks.extend(block for block in extractor.all_blocks(structure["blocks"]) if block["type"] == "code")
    assert len(code_blocks) == 1
    assert extractor.plain(code_blocks[0]["runs"]) == expected
    assert all(run["link_id"] == outer_link for run in code_blocks[0]["runs"] if run["type"] == "text")
    assert "```\n" + expected + "\n```" in combined_text(workspace)
    assert any("```\n" + expected + "\n```" in node.content for node in assert_grounded_tree(workspace).nodes)


@pytest.mark.asyncio
async def test_authored_categories_share_budget_across_batches_and_parallel_fetches_keep_order(tmp_path, monkeypatch):
    categories = ''.join('<h2>' + group + '</h2>' + ''.join(f'<p><a href="{group}{index}.html">{group}{index}</a></p>' for index in range(1, 6)) for group in ("A", "B", "C"))
    html = '<html><head><title>Manual</title></head><body><main><h1>Manual</h1>' + categories + '</main></body></html>'
    workspace = workspace_for(tmp_path, html, max_pages=6, batch_size=5, fetch_concurrency=2)
    expected = ["https://example.org/docs/" + name + ".html" for name in ("A1", "B1", "C1", "A2", "B2", "C2")]
    requested, completed, budgets = [], [], []
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()
    other_pages_finished = threading.Event()

    def read_url(url, **kwargs):
        assert url in expected
        with lock:
            requested.append(url)
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            # Force the first scheduled page to finish last; acquisition order must not shape the tree.
            if url == expected[0]:
                assert other_pages_finished.wait(timeout=5), "The second fetch slot must continue while the first page waits"
            else:
                time.sleep(0.01)
            name = url.rsplit("/", 1)[-1].removesuffix(".html")
            return FetchResult(f'<html><main><h1>{name}</h1><p>Exact instructions for {name}.</p></main></html>'.encode("utf-8"), "text/html", url, 200, {})
        finally:
            with lock:
                completed.append(url)
                state["active"] -= 1
                if len([item for item in completed if item != expected[0]]) == 5:
                    other_pages_finished.set()

    monkeypatch.setattr(workspace.acquisition, "_read_url", read_url)

    async def decide(payload):
        if payload["page"]["url"] == ROOT_URL:
            budgets.append(payload["remaining_child_pages"])
        # Deliberately use one model group: the authored h2 categories must control fair coverage.
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert budgets == [6, 6, 6]
    assert len(requested) == len(set(requested)) == 6
    assert set(requested) == set(expected)
    assert completed != expected and completed[-1] == expected[0]
    assert state["max_active"] == 2
    assert [page["url"] for page in result["pages"][1:]] == expected
    assert result["child_page_count"] == 6
    assert result["acquisition_complete"] is False
    assert result["truncated"] is True
    assert len(result["unfetched_discovered_urls"]) == 9
    records = workspace.store.read_json_artifact(result["decision_artifact_id"])["decisions"]
    assert len(records) == 15
    assert {record["url"] for record in records if record["status"] == "fetched"} == set(expected)
    assert sum(record["status"] == "budget_exhausted" for record in records) == 9
    for name in ("A1", "B1", "C1", "A2", "B2", "C2"):
        assert f"Exact instructions for {name}." in combined_text(workspace)
    assert_grounded_tree(workspace)


@pytest.mark.asyncio
async def test_entry_manual_identity_survives_child_titles_and_model_gets_link_context(tmp_path, monkeypatch):
    html = """<html><head><title>Complete Example Book</title></head><body><main><h1>Complete Example Book</h1>
    <nav role="doc-toc"><a href="chapter.html">Chapter One</a></nav>
    <h2>Code notation</h2><pre><code>Use <a href="/api/operator.html">operator</a> in expressions.</code></pre>
    </main></body></html>"""
    workspace = workspace_for(tmp_path, html)
    chapter = "https://example.org/docs/chapter.html"
    grandchild = "https://example.org/docs/procedure.html"
    requested = install_pages(monkeypatch, workspace, {
        chapter: '<html><head><title>A Different Chapter Title</title></head><body><main><h1>Chapter One</h1><p><a href="procedure.html">Nested procedure</a></p></main></body></html>',
        grandchild: '<html><head><title>Another Procedure Title</title></head><body><main><h1>Nested procedure</h1><p>Follow this book procedure.</p></main></body></html>',
    })
    scopes, titles = [], []

    async def decide(payload):
        scopes.append(dict(payload["entry_scope"]))
        titles.append(payload["page"]["title"])
        if payload["page"]["url"] == ROOT_URL:
            links = {link["url"]: link for link in payload["page"]["links"]}
            assert links[chapter]["authored_contents"] is True
            assert links[chapter]["navigation_hint"] == "directory_entry"
            code_link = links["https://example.org/api/operator.html"]
            assert code_link["in_code"] is True
            assert code_link["navigation_hint"] == "cross_reference"
            assert code_link["authored_contents"] is False
        return plan_for(payload, choose=lambda link: "fetch" if link["url"] in {chapter, grandchild} else "exclude")

    result = await run_guided_html(workspace, decide)
    assert requested == [chapter, grandchild]
    assert titles == ["Complete Example Book", "A Different Chapter Title", "Another Procedure Title"]
    assert len(scopes) == 3 and all(scope == scopes[0] for scope in scopes)
    assert scopes[0]["entry_url"] == ROOT_URL
    assert scopes[0]["document_title"] == "Complete Example Book"
    assert scopes[0]["entry_heading"] == "Complete Example Book"
    assert workspace.manifest.source_metadata["html_entry_scope"] == scopes[0]
    assert result["acquisition_complete"] is True
    assert "Follow this book procedure." in combined_text(workspace)


@pytest.mark.asyncio
@pytest.mark.parametrize("landmark", ["main", "article", "role_main", "fallback"])
async def test_entry_heading_prefers_document_landmark_over_header_branding(tmp_path, monkeypatch, landmark):
    if landmark == "fallback":
        content = '<div id="guide"><h1>Standalone Book</h1><p>Standalone manual instructions.</p></div>'
        expected = "Standalone Book"
    else:
        opening, closing = ('<div id="guide" role="main">', '</div>') if landmark == "role_main" else (f'<{landmark} id="guide">', f'</{landmark}>')
        content = '<header><h1>Site brand</h1></header>' + opening + '<aside><h1>Sidebar brand</h1></aside><header><h1>Actual Book</h1></header><p>Actual manual instructions.</p>' + closing
        expected = "Actual Book"
    html = '<html><head><title>Documentation site</title></head><body>' + content + '</body></html>'
    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {})
    scopes = []

    async def decide(payload):
        scopes.append(dict(payload['entry_scope']))
        assert payload['entry_scope']['entry_heading'] == expected
        plan = plan_for(payload)
        guide = next(item for item in payload['page']['containers'] if item.get('dom_id') == 'guide') if landmark != 'fallback' else payload['page']['containers'][0]
        plan['content_root_id'] = guide['container_id']
        return plan

    result = await run_guided_html(workspace, decide)
    assert requested == []
    assert result['acquisition_complete'] is True
    assert len(scopes) == 1
    assert scopes[0]['document_title'] == 'Documentation site'
    assert scopes[0]['entry_heading'] == expected
    assert workspace.manifest.source_metadata['html_entry_scope']['entry_heading'] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["websearch", "user"])
async def test_adopted_discovery_scope_is_a_bounded_model_hint_not_a_link_whitelist(tmp_path, origin):
    workspace = workspace_for(tmp_path, '<main><h1>Learning portal</h1><p>Choose a lesson.</p></main>')
    workspace.manifest.source_origin = origin
    workspace.manifest.source_metadata['html_scope'] = {
        'kind': 'tutorial_collection', 'scope_summary': 'Practical workflows',
        'entry_evidence_ids': ['H999'], 'included_sections': ['Segmentation' * 100] * 16,
        'excluded_sections': ['Development' * 100] * 16, 'comparisons': [],
        'comparison_note': 'The selected collection matches the learning goal.', 'limitations': [],
    }
    # Use contract-valid long descriptions to exercise the planner context bound.
    for key in ['included_sections', 'excluded_sections']:
        workspace.manifest.source_metadata['html_scope'][key] = [value[:600] for value in workspace.manifest.source_metadata['html_scope'][key]]

    async def decide(payload):
        hint = payload['entry_scope'].get('discovery_scope')
        if origin == 'websearch':
            assert hint['scope_summary'] == 'Practical workflows'
            assert len(json.dumps(hint, ensure_ascii=False).encode('utf8')) <= 6000
            assert 'entry_evidence_ids' not in hint  # IDs refer to the earlier inspection, not this capture.
            assert 'H999' not in json.dumps(hint)
            assert payload['page']['links'] == []  # Scope prose cannot fabricate source links.
        else:
            assert hint is None
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert result['page_count'] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("landmark,level", [("main", 2), ("article", 2), ("role_main", 3)])
async def test_entry_heading_uses_shallowest_authored_heading_without_h1(tmp_path, monkeypatch, landmark, level):
    opening, closing = ('<div role="main">', '</div>') if landmark == "role_main" else (f'<{landmark}>', f'</{landmark}>')
    html = ('<html><head><title>Complete Parent Handbook</title></head><body>'
            '<header><h1>Complete Parent Handbook</h1></header>' + opening
            + '<aside><h1>Global documentation</h1></aside><h4>Related training</h4>'
            + f'<h{level}>Chapter 2 Tutorials</h{level}><p>Follow this chapter procedure.</p>'
            + closing + '</body></html>')
    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        scope = payload["entry_scope"]
        assert scope["entry_heading"] == "Chapter 2 Tutorials"
        assert scope["document_title"] == "Complete Parent Handbook"
        assert "supplied entry and its authored descendants" in scope["policy"]
        assert "parent table of contents never expands" in scope["policy"]
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert requested == [] and result["acquisition_complete"] is True
    assert workspace.NORMALIZATION_VERSION == "v17-html-model-guided.v22"


@pytest.mark.asyncio
@pytest.mark.parametrize("navigation_first", [True, False])
async def test_grouped_page_navigation_keeps_real_directory_and_plain_contents_uses(tmp_path, monkeypatch, navigation_first):
    navigation = ('<div class="crosslinks"><p>[<a href="parent.html">table of contents</a>]'
                  '[<a href="previous.html">previous topic</a>][<a href="chapter.html">next topic</a>]'
                  '[<a href="index.html">index</a>]</p></div>')
    directory = '<div class="toc"><p><a href="chapter.html">Owned tutorial chapter</a></p></div>'
    html = ('<html><head><title>Parent Handbook</title></head><body><header><h1>Parent Handbook</h1></header>'
            '<main><h2>Chapter 2 Tutorials</h2>'
            + (navigation + directory if navigation_first else directory + navigation)
            + '<p><a href="own-contents.html">Contents</a></p></main></body></html>')
    snapshot, _ = build_snapshot(html, ROOT_URL)
    original = {item["url"]: item for item in snapshot["links"]}
    parent, chapter, contents = ["https://example.org/docs/" + name
                                 for name in ("parent.html", "chapter.html", "own-contents.html")]
    assert original[parent]["page_navigation_only"] is True
    assert original[parent]["page_navigation_relations"] == ["contents"]
    assert original[parent]["non_navigation_occurrence_count"] == 0
    facts = original[parent]["occurrences"][0]["page_navigation"]
    assert facts["basis"] == "grouped_relational_links"
    assert facts["peer_relations"] == ["contents", "previous", "next", "index"]
    assert facts["container_dom_path"]
    assert original[chapter]["label"] == "Owned tutorial chapter"
    assert original[chapter]["page_navigation_only"] is False
    assert original[chapter]["non_navigation_occurrence_count"] == 1
    assert original[chapter]["page_navigation_relations"] == ["next"]
    assert len(original[chapter]["occurrences"]) == 2
    assert original[contents]["page_navigation_only"] is False
    assert original[contents]["occurrences"][0]["page_navigation"] is None

    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {
        chapter: '<main><h1>Owned tutorial chapter</h1><p>Run the tutorial command.</p></main>',
        contents: '<main><h1>Contents</h1><p>The supplied ordinary directory remains eligible.</p></main>',
    })
    offered = []

    async def decide(payload):
        assert payload["entry_scope"]["entry_heading"] == "Chapter 2 Tutorials"
        assert len(json.dumps(payload, ensure_ascii=False)) <= 48_000
        if payload["page"]["url"] == ROOT_URL:
            offered.extend(payload["page"]["links"])
            links = {link["url"]: link for link in payload["page"]["links"]}
            assert links[parent]["page_navigation_only"] is True
            assert links[chapter]["page_navigation_only"] is False
            assert links[chapter]["authored_contents"] is True
            assert links[chapter]["label"] == "Owned tutorial chapter"
            assert links[contents]["page_navigation"] == []
        return plan_for(payload, choose=lambda link: "exclude" if link["page_navigation_only"] else "fetch")

    result = await run_guided_html(workspace, decide)
    assert set(requested) == {chapter, contents}
    assert {link["url"] for link in offered} == set(original)
    assert result["acquisition_complete"] is True
    assert "Run the tutorial command." in combined_text(workspace)
    assert "ordinary directory remains eligible" in combined_text(workspace)
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_text("utf-8") == html


def test_page_navigation_requires_relation_or_container_evidence():
    html = ('<main><h2>Chapter</h2><p><a href="contents.html">Contents</a></p>'
            '<p>For background read <a href="previous.html">previous topic</a> and '
            '<a href="next.html">next topic</a> when appropriate.</p>'
            '<nav aria-label="Breadcrumb"><a href="parent.html">Parent catalog</a></nav>'
            '<p><a rel="up" href="rel-parent.html">Overview</a></p>'
            '<table summary="Navigation footer"><tr><td><a accesskey="n" href="sibling.html">Next</a></td></tr></table>'
            '<nav aria-label="Table of contents"><a href="real-contents.html">Contents</a></nav></main>')
    snapshot, _ = build_snapshot(html, ROOT_URL)
    links = {item["url"].rsplit("/", 1)[-1]: item for item in snapshot["links"]}
    for name in ("contents.html", "previous.html", "next.html", "real-contents.html"):
        assert links[name]["page_navigation_only"] is False
        assert links[name]["non_navigation_occurrence_count"] == 1
    assert links["parent.html"]["page_navigation_relations"] == ["ancestor"]
    assert links["parent.html"]["occurrences"][0]["page_navigation"]["basis"] == "semantic_container"
    assert links["rel-parent.html"]["page_navigation_relations"] == ["up"]
    assert links["rel-parent.html"]["occurrences"][0]["page_navigation"]["basis"] == "rel_attribute"
    assert links["sibling.html"]["page_navigation_relations"] == ["next"]
    facts = links["sibling.html"]["occurrences"][0]["page_navigation"]
    assert facts["basis"] == "semantic_container"
    assert facts["container_label"] == "Navigation footer"
    assert facts["container_tag"] == "table"
    model = _model_link(links["sibling.html"])
    assert model["page_navigation_only"] is True
    assert model["page_navigation"][0]["container_label"] == "Navigation footer"


@pytest.mark.parametrize("relation", ["index", "next"])
@pytest.mark.parametrize("directory_first", [True, False])
def test_authored_toc_rel_attribute_preserves_directory_use_over_navigation(relation, directory_first):
    directory = ('<div class="contents"><ul><li><a rel="' + relation
                 + '" href="chapter.html">Owned chapter</a></li></ul></div>')
    navigation = '<p><a rel="' + relation + '" href="chapter.html">Navigation destination</a></p>'
    pure_navigation = '<p><a rel="' + relation + '" href="other.html">Only navigation</a></p>'
    html = '<main><h2>Manual</h2>' + (directory + navigation if directory_first else navigation + directory) + pure_navigation + '</main>'
    snapshot, _ = build_snapshot(html, ROOT_URL)
    links = {link["url"].rsplit("/", 1)[-1]: link for link in snapshot["links"]}
    chapter = links["chapter.html"]
    assert chapter["label"] == "Owned chapter"
    assert chapter["authored_contents"] is True
    assert chapter["page_navigation_only"] is False
    assert chapter["non_navigation_occurrence_count"] == 1
    assert chapter["page_navigation_relations"] == [relation]
    assert len(chapter["occurrences"]) == 2
    directory_use = next(item for item in chapter["occurrences"] if item["authored_contents"])
    assert directory_use["page_navigation"] is None
    model = _model_link(chapter)
    assert model["page_navigation_only"] is False and model["authored_contents"] is True
    assert model["label"] == "Owned chapter"
    other = links["other.html"]
    assert other["page_navigation_only"] is True
    assert other["non_navigation_occurrence_count"] == 0
    assert other["page_navigation_relations"] == [relation]
    assert other["occurrences"][0]["page_navigation"]["basis"] == "rel_attribute"


@pytest.mark.asyncio
async def test_heading_permalink_redirector_is_host_resolved_even_inside_authored_toc(tmp_path, monkeypatch):
    redirector = "https://lean.example/find/?domain=Lean&name=Nat.add"
    html = '<html><head><title>Lean Manual</title></head><body><main><h1>Lean Manual</h1><div class="toc"><h2>Section<span class="permalink-widget"><a href="' + redirector + '">🔗</a></span></h2><p>Actual section instructions remain here.</p></div></main></body></html>'
    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {})
    offered = []

    async def decide(payload):
        offered.extend(payload["page"]["links"])
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert requested == [] and offered == []
    assert result["acquisition_complete"] is True
    assert result["host_resolved_link_count"] == 1
    records = workspace.store.read_json_artifact(result["decision_artifact_id"])["decisions"]
    assert len(records) == 1
    record = records[0]
    assert record["url"] == redirector
    assert record["legacy_role"] == "heading_permalink"
    assert record["authored_contents"] is True
    assert record["status"] == "resolved_heading_permalink"
    assert record["decision_origin"] == "host" and record["action"] == "host_resolve"
    snapshots = [workspace.store.read_json_artifact(artifact_id) for artifact_id in result["pipeline_artifact_ids"]
                 if workspace.store.artifact_record(artifact_id).kind == "html_guided_snapshot"]
    assert snapshots[0]["links"][0]["link_id"] == record["link_id"]
    tree = assert_grounded_tree(workspace)
    assert any(node.title.startswith("Section") for node in tree.nodes)
    assert "Actual section instructions remain here." in combined_text(workspace)


@pytest.mark.asyncio
@pytest.mark.parametrize("permalink_first", [True, False])
async def test_real_directory_use_of_same_url_is_not_hidden_by_heading_permalink(tmp_path, monkeypatch, permalink_first):
    permalink = '<h2>Chapter<span class="permalink-widget"><a href="chapter.html#intro">🔗</a></span></h2>'
    directory = '<p><a href="chapter.html">Read chapter</a></p>'
    entries = permalink + directory if permalink_first else directory + permalink
    html = '<html><head><title>Manual</title></head><body><main><h1>Manual</h1><div class="toc">' + entries + '</div></main></body></html>'
    workspace = workspace_for(tmp_path, html)
    chapter = "https://example.org/docs/chapter.html"
    requested = install_pages(monkeypatch, workspace, {
        chapter: '<html><main><h1>Chapter</h1><p>The real chapter instructions must be captured.</p></main></html>',
    })
    offered = []

    async def decide(payload):
        if payload["page"]["url"] == ROOT_URL:
            offered.extend(payload["page"]["links"])
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert [link["url"] for link in offered] == [chapter]
    assert offered[0]["navigation_hint"] == "directory_entry"
    assert offered[0]["authored_contents"] is True
    assert requested == [chapter]
    assert result["acquisition_complete"] is True
    records = workspace.store.read_json_artifact(result["decision_artifact_id"])["decisions"]
    assert len(records) == 1 and records[0]["status"] == "fetched"
    assert records[0]["occurrence_count"] == 2
    assert len(records[0]["occurrences"]) == 2
    assert "The real chapter instructions must be captured." in combined_text(workspace)


@pytest.mark.asyncio
@pytest.mark.parametrize("article_first", [False, True])
async def test_landmarks_and_content_membership_distinguish_site_books_from_article_toc(tmp_path, monkeypatch, article_first):
    chapter = "https://example.org/books/user/intro.html"
    reference = "https://example.org/books/reference/index.html"
    sidebar = '<aside aria-label="All books"><div role="navigation" aria-label="Documentation books"><div class="toc"><p><a href="' + chapter + '">User chapter in site menu</a></p><p><a href="' + reference + '">Separate Reference Manual</a></p></div></div></aside>'
    article = '<main id="manual"><h1>Current User Guide</h1><div class="toc"><h2>Current guide contents</h2><p><a href="' + chapter + '">Own authored chapter</a></p></div></main>'
    brand = '<div role="navigation" aria-label="Brand navigation"><a href="https://example.org/">Home</a></div>'
    html = '<html><head><title>Current User Guide</title></head><body>' + brand + (article + sidebar if article_first else sidebar + article) + '</body></html>'
    snapshot, _ = build_snapshot(html, ROOT_URL)
    main_id = next(item["container_id"] for item in snapshot["containers"] if item["dom_id"] == "manual")
    links = {item["url"]: item for item in snapshot["links"]}
    own = links[chapter]
    assert own["in_chrome"] is False
    assert own["authored_contents"] is True
    assert main_id in own["content_root_ids"]
    assert own["occurrence_count"] == 2
    assert {item["in_chrome"] for item in own["occurrences"]} == {True, False}
    assert sum(main_id in item["content_root_ids"] for item in own["occurrences"]) == 1
    for url in (reference, "https://example.org/"):
        assert links[url]["in_chrome"] is True
        assert main_id not in links[url]["content_root_ids"]
    assert any(item["tag"] == "aside" for item in links[reference]["landmark_context"])
    assert any(item["role"] == "navigation" and item["label"] == "Documentation books" for item in links[reference]["landmark_context"])
    assert any(item["role"] == "navigation" and item["label"] == "Brand navigation" for item in links["https://example.org/"]["landmark_context"])

    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {
        chapter: '<html><main><h1>Own authored chapter</h1><p>Follow the current guide procedure.</p></main></html>',
    })

    async def decide(payload):
        containers = payload["page"]["containers"]
        aliases = {item["container_id"] for item in containers}
        current_main = next(item["container_id"] for item in containers if item["tag"] == "main")
        assert all(identity.startswith("C") for identity in aliases)
        assert len(json.dumps(payload, ensure_ascii=False)) <= 48_000
        assert all(set(link["content_root_ids"]) <= aliases for link in payload["page"]["links"])
        assert all(len(link["landmark_context"]) <= 4 for link in payload["page"]["links"])
        # This fake model uses the supplied structural facts, without a URL-prefix acquisition rule.
        return plan_for(payload, choose=lambda link: "fetch" if current_main in link["content_root_ids"] else "exclude")

    result = await run_guided_html(workspace, decide)
    assert requested == [chapter]
    assert result["acquisition_complete"] is True
    assert "Follow the current guide procedure." in combined_text(workspace)


@pytest.mark.asyncio
async def test_prior_exclusion_is_visible_but_new_authored_contents_can_override_it(tmp_path, monkeypatch):
    chapter = "https://example.org/docs/chapter.html"
    procedure = "https://example.org/docs/procedure.html"
    html = '<html><head><title>User Guide</title></head><body><aside><a href="procedure.html">Site-wide procedure link</a></aside><main><h1>User Guide</h1><div class="toc"><a href="chapter.html">Current guide chapter</a></div></main></body></html>'
    workspace = workspace_for(tmp_path, html)
    requested = install_pages(monkeypatch, workspace, {
        chapter: '<html><head><title>Current guide chapter</title></head><body><main><h1>Current guide chapter</h1><div class="toc"><a href="procedure.html">Authored chapter procedure</a></div></main></body></html>',
        procedure: '<html><main><h1>Authored chapter procedure</h1><p>The newly established manual entry is captured.</p></main></html>',
    })
    prior_reason = "Only observed in global site navigation. " + "Original scope explanation. " * 15
    hints = []

    async def decide(payload):
        assert len(json.dumps(payload, ensure_ascii=False)) <= 48_000
        if payload["page"]["url"] == ROOT_URL:
            plan = plan_for(payload, choose=lambda link: "fetch" if link["url"] == chapter else "exclude")
            rejected = next(item for item in plan["links"] if item["link_id"] == next(
                link["link_id"] for link in payload["page"]["links"] if link["url"] == procedure
            ))
            rejected["reason"] = prior_reason
            return plan
        if payload["page"]["url"] == chapter:
            link = next(item for item in payload["page"]["links"] if item["url"] == procedure)
            assert link["authored_contents"] is True
            hint = link["prior_scope_decision"]
            assert hint == {"action": "exclude", "reason": prior_reason[:240], "source_url": ROOT_URL}
            hints.append(hint)
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert requested == [chapter, procedure]
    assert len(hints) == 1
    assert result["acquisition_complete"] is True
    records = workspace.store.read_json_artifact(result["decision_artifact_id"])["decisions"]
    same_url = [record for record in records if record["url"] == procedure]
    assert [record["status"] for record in same_url] == ["excluded", "fetched"]
    assert "The newly established manual entry is captured." in combined_text(workspace)


@pytest.mark.parametrize("field", ["action", "default_action"])
def test_required_missing_model_contract_maps_to_unchanged_internal_defer(field):
    raw = {"content_root_id": "C1", "links": []}
    if field == "action":
        raw["links"] = [{"link_id": "L1", "action": "required_missing", "reason": "This chapter is required and has not been captured."}]
    else:
        raw.update(default_action="required_missing", default_reason="These chapters are required and have not been captured.")
    plan, error = _resolve_model_plan(raw, {"C1": "root_existing"}, {"L1": "link_existing"}, None)
    assert error is None
    mapped = plan["links"][0]["action"] if field == "action" else plan["default_action"]
    assert mapped == "defer"
    assert (raw["links"][0]["action"] if field == "action" else raw["default_action"]) == "required_missing"
    snapshot = {"containers": [{"container_id": "root_existing"}], "links": [{"link_id": "link_existing"}]}
    assert _validate_plan(plan, snapshot, None) is None


@pytest.mark.parametrize("field", ["action", "default_action"])
def test_old_model_defer_is_rejected_instead_of_silently_mapped(field):
    raw = {"content_root_id": "C1", "links": []}
    if field == "action":
        raw["links"] = [{"link_id": "L1", "action": "defer", "reason": "Old ambiguous model enum."}]
    else:
        raw.update(default_action="defer", default_reason="Old ambiguous model enum.")
    _, error = _resolve_model_plan(raw, {"C1": "root_existing"}, {"L1": "link_existing"}, None)
    assert error is not None
    assert "required_missing" in error


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["action", "default_action"])
async def test_old_model_defer_requires_explicit_model_repair_before_missing_status(tmp_path, monkeypatch, field):
    workspace = workspace_for(tmp_path, '<html><main><h1>Manual</h1><p><a href="required.html">Required chapter</a></p></main></html>')
    requested = install_pages(monkeypatch, workspace, {})
    calls = []

    async def decide(payload):
        calls.append(payload)
        action = "defer" if len(calls) == 1 else "required_missing"
        plan = plan_for(payload, action=action)
        if field == "default_action":
            plan.update(links=[], default_action=action, default_reason="This manual chapter is required and has not been captured.")
        return plan

    result = await run_guided_html(workspace, decide)
    assert requested == (["https://example.org/docs/required.html"] if field == 'action' else [])
    assert len(calls) == 2
    assert "required_missing" in calls[1]["validation_error"]
    assert result["acquisition_complete"] is False
    records = workspace.store.read_json_artifact(result["decision_artifact_id"])["decisions"]
    assert len(records) == 1
    assert records[0]["action"] == ('fetch' if field == 'action' else 'defer')
    assert records[0]["status"] == ('failed' if field == 'action' else 'deferred')


def test_prior_missing_hint_uses_external_enum_without_mutating_internal_decision():
    snapshot, _ = build_snapshot('<main><h1>Manual</h1><p><a href="required.html">Required chapter</a></p></main>', ROOT_URL)
    link = snapshot["links"][0]
    link["prior_scope_decision"] = {"action": "defer", "reason": "This manual chapter is required and has not been captured.", "source_url": ROOT_URL}
    model_link = _model_link(link)
    assert model_link["prior_scope_decision"]["action"] == "required_missing"
    assert link["prior_scope_decision"]["action"] == "defer"


def test_unused_fetch_default_is_removed_only_after_every_link_has_an_explicit_decision():
    raw = {
        "content_root_id": "C1", "default_action": "fetch", "default_reason": "Unused model default.",
        "links": [
            {"link_id": "L1", "action": "fetch", "reason": "An authored chapter."},
            {"link_id": "L2", "action": "exclude", "reason": "A separate optional reference."},
        ],
        "plan_normalization": [{"kind": "untrusted_model_claim"}],
    }
    unchanged = json.loads(json.dumps(raw))
    plan, error = _resolve_model_plan(raw, {"C1": "root_existing"}, {"L1": "link_a", "L2": "link_b"}, None)
    assert error is None
    assert "default_action" not in plan and "default_reason" not in plan
    assert plan["plan_normalization"] == [{"kind": "ignored_unused_fetch_default", "explicit_link_count": 2}]
    assert [(item["link_id"], item["action"]) for item in plan["links"]] == [("link_a", "fetch"), ("link_b", "exclude")]
    snapshot = {"containers": [{"container_id": "root_existing"}], "links": [{"link_id": "link_a"}, {"link_id": "link_b"}]}
    assert _validate_plan(plan, snapshot, None) is None
    assert raw == unchanged


@pytest.mark.parametrize("invalid_case", ["missing", "duplicate", "unknown"])
def test_fetch_default_is_not_normalized_when_explicit_coverage_is_invalid(invalid_case):
    links = {
        "missing": [{"link_id": "L1", "action": "fetch", "reason": "An authored chapter."}],
        "duplicate": [
            {"link_id": "L1", "action": "fetch", "reason": "An authored chapter."},
            {"link_id": "L1", "action": "exclude", "reason": "Conflicting duplicate."},
        ],
        "unknown": [
            {"link_id": "L1", "action": "fetch", "reason": "An authored chapter."},
            {"link_id": "L3", "action": "fetch", "reason": "An invented alias."},
        ],
    }[invalid_case]
    raw = {"content_root_id": "C1", "default_action": "fetch", "default_reason": "Must not fetch implicit links.", "links": links}
    unchanged = json.loads(json.dumps(raw))
    plan, error = _resolve_model_plan(raw, {"C1": "root_existing"}, {"L1": "link_a", "L2": "link_b"}, None)
    snapshot = {"containers": [{"container_id": "root_existing"}], "links": [{"link_id": "link_a"}, {"link_id": "link_b"}]}
    assert error is not None or _validate_plan(plan, snapshot, None) is not None
    assert not any(item.get("kind") == "ignored_unused_fetch_default" for item in plan.get("plan_normalization", []))
    assert raw == unchanged


def test_empty_batch_can_discard_an_unused_fetch_default_without_inventing_a_link():
    raw = {"content_root_id": "C1", "links": [], "default_action": "fetch", "default_reason": "No links are offered."}
    unchanged = json.loads(json.dumps(raw))
    plan, error = _resolve_model_plan(raw, {"C1": "root_existing"}, {}, None)
    assert error is None
    assert plan["links"] == []
    assert "default_action" not in plan and "default_reason" not in plan
    assert plan["plan_normalization"] == [{"kind": "ignored_unused_fetch_default", "explicit_link_count": 0}]
    assert _validate_plan(plan, {"containers": [{"container_id": "root_existing"}], "links": []}, None) is None
    assert raw == unchanged


@pytest.mark.parametrize("decisions", [None, [{"link_id": "L1", "action": "exclude"}],
                                      [{"link_id": "L1", "action": "fetch", "url": "https://invented.example/"}]])
def test_boundary_only_empty_batch_cannot_acquire_model_invented_links(decisions):
    raw = {"content_root_id": "C1"}
    if decisions is not None:
        raw["links"] = decisions
    unchanged = json.loads(json.dumps(raw))
    plan, error = _resolve_model_plan(raw, {"C1": "root_existing"}, {}, None)
    assert error is None
    assert plan["links"] == []
    assert plan["plan_normalization"][0]["kind"] == "ignored_unoffered_empty_batch_decisions"
    assert _validate_plan(plan, {"containers": [{"container_id": "root_existing"}], "links": []}, None) is None
    assert raw == unchanged


def test_boundary_only_call_still_requires_a_valid_supplied_content_root():
    _, error = _resolve_model_plan({"content_root_id": "C99"}, {"C1": "root_existing"}, {}, None)
    assert error == "content_root_id must name a supplied C alias."


def test_ignoring_an_unused_fetch_default_does_not_accept_an_invalid_explicit_action():
    raw = {
        "content_root_id": "C1", "default_action": "fetch", "default_reason": "Unused.",
        "links": [{"link_id": "L1", "action": "invented_action", "reason": "Every alias is present but the action is invalid."}],
    }
    plan, error = _resolve_model_plan(raw, {"C1": "root_existing"}, {"L1": "link_a"}, None)
    snapshot = {"containers": [{"container_id": "root_existing"}], "links": [{"link_id": "link_a"}]}
    assert error is not None or _validate_plan(plan, snapshot, None) is not None
