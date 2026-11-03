from __future__ import annotations

import json

import fitz
import pytest

from app.acquisition import AcquisitionService, FetchResult
from app.config import Settings
from app.evidence_planning import EvidencePlanEngine
from app.models import StagedInput, TargetProfile
from app.storage import RunStore
from app.toolkit import SourceWorkspace, ToolExecutor


def _target() -> TargetProfile:
    return TargetProfile(
        software_name="ExampleBio",
        target_version="2.0",
        platform="Linux",
        ui_locale="English",
    )


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "data_dir": tmp_path / "data",
        "output_dir": tmp_path / "out",
    }
    values.update(overrides)
    return Settings(**values)


def _workspace(tmp_path, run_id: str, filename: str, data: bytes, **settings_overrides):
    settings = _settings(tmp_path, **settings_overrides)
    store = RunStore(settings.data_dir, run_id)
    path = store.stage_upload("inp_1", filename, data)
    source_input = StagedInput(
        input_id="inp_1",
        kind="uploaded_file",
        display_name=filename,
        staged_path=str(path),
        declared_media_type=("application/pdf" if filename.endswith(".pdf") else "text/html"),
    )
    acquisition = AcquisitionService(store, settings)
    manifest = acquisition.acquire([source_input], _target()).manifests[0]
    workspace = SourceWorkspace(store, settings, acquisition, _target(), manifest)
    return store, acquisition, workspace


def _finish_pdf(tmp_path, run_id: str, data: bytes):
    store, _, workspace = _workspace(tmp_path, run_id, "manual.pdf", data)
    executor = ToolExecutor(workspace, "deterministic_pdf_test")
    assert executor.call("pdf_inventory", {})["ok"] is True
    assert executor.call("pdf_list_structure", {})["ok"] is True
    assert executor.call(
        "curate_evidence",
        {
            "selection_mode": "all_except_excluded",
            "excluded": [],
            "merge_groups": [],
        },
    )["accepted"] is True
    assert executor.call(
        "submit_source_result",
        {
            "status": "succeeded",
            "summary": "Deterministic PDF normalization completed.",
            "coverage": {"skipped": workspace.coverage().skipped},
            "gaps": [],
            "warnings": [],
        },
    )["accepted"] is True
    return store, workspace.final_result


def test_t5_html_saved_origin_and_budget_are_balanced_across_capability_branches(
    tmp_path, monkeypatch
):
    root_html = b"""<!doctype html>
    <!-- saved from url=(0043)https://docs.example.test/manual/index.html -->
    <html><head><title>ExampleBio manual</title></head><body><main>
    <h1>Manual</h1>
    <h2>Import</h2><a href="import-1.html">Import 1</a><a href="import-2.html">Import 2</a>
    <h2>Align</h2><a href="align-1.html">Align 1</a><a href="align-2.html">Align 2</a>
    <h2>Export</h2><a href="export-1.html">Export 1</a><a href="export-2.html">Export 2</a>
    </main><footer><a href="site-map.html">Site map</a></footer></body></html>"""
    store, acquisition, workspace = _workspace(
        tmp_path,
        "run_t5_html_budget",
        "saved.html",
        root_html,
        max_html_pages=4,
    )
    fetched: list[str] = []

    def fake_read_url(self, value):
        fetched.append(value)
        name = value.rsplit("/", 1)[-1]
        data = (
            f"<html><head><title>{name}</title></head><body><main>"
            f"<h1>{name}</h1><p>Operational documentation.</p></main></body></html>"
        ).encode()
        return FetchResult(data, "text/html", value, 200, {})

    monkeypatch.setattr(AcquisitionService, "_read_url", fake_read_url)
    executor = ToolExecutor(workspace, "deterministic_html_test")
    inventory = executor.call("html_inventory", {"max_pages": 4})
    navigation = executor.call("html_list_navigation", {"offset": 1, "limit": 2})

    assert workspace.manifest.origin_url == "https://docs.example.test/manual/index.html"
    assert inventory["root_page_count"] == 1
    assert inventory["child_page_count"] == 4
    assert inventory["page_count"] == 5
    assert [url.rsplit("/", 1)[-1].split("-")[0] for url in fetched] == [
        "import", "align", "export", "import",
    ]
    assert len(set(fetched)) == 4
    assert "https://docs.example.test/manual/site-map.html" not in fetched
    assert [item["fetched_pages"] for item in inventory["branch_coverage"]] == [2, 1, 1]
    assert inventory["truncated"] is True
    assert navigation["page_count"] == 5
    assert navigation["returned"] == 2
    assert navigation["truncated"] is True
    assert all(item["url"].startswith("https://docs.example.test/manual/") for item in navigation["pages"])
    del store, acquisition


def test_t5_html_details_summaries_define_authored_groups_and_balanced_budget(
    tmp_path, monkeypatch
):
    root_html = b"""<!doctype html>
    <!-- saved from url=(0043)https://docs.example.test/user/index.html -->
    <html><head><title>Scientific App Guide</title></head><body><main>
    <h2>Scientific App Guide</h2><p>Use this guide to operate the application.</p>
    <p><a href="tools/help.html">Open the embedded help viewer.</a></p>
    <details id="general" open><summary><b>General</b></summary><ul>
      <li><a href="start.html">Startup</a></li>
      <li><a href="window.html">Application Window</a></li>
    </ul></details>
    <details id="tools" open><summary><b>Tools</b></summary><ul>
      <li><a href="tools/labels.html">Labels</a></li>
      <li><a href="tools/help.html">Help Viewer</a></li>
    </ul></details>
    <details id="commands" open><summary><b>Commands</b></summary><ul>
      <li><a href="commands/open.html">open</a></li>
      <li><a href="commands/save.html">save</a></li>
    </ul></details>
    </main></body></html>"""
    store, _, workspace = _workspace(
        tmp_path,
        "run_t5_details_groups",
        "guide.html",
        root_html,
        max_html_pages=4,
    )
    fetched: list[str] = []

    def fake_read_url(self, value):
        fetched.append(value)
        name = value.rsplit("/", 1)[-1]
        data = (
            f"<html><head><title>{name}</title></head><body><main>"
            f"<h1>{name}</h1><p>Exact operational documentation for {name}.</p>"
            "</main></body></html>"
        ).encode()
        return FetchResult(data, "text/html", value, 200, {})

    monkeypatch.setattr(AcquisitionService, "_read_url", fake_read_url)
    executor = ToolExecutor(workspace, "deterministic_details_test")
    inventory = executor.call("html_inventory", {"max_pages": 4})

    assert inventory["ok"] is True, inventory
    assert [item["label"] for item in inventory["branch_coverage"]] == [
        "General",
        "Tools",
        "Commands",
    ]
    assert [item["fetched_pages"] for item in inventory["branch_coverage"]] == [
        2,
        1,
        1,
    ]
    assert set(fetched[::3]) == {
        "https://docs.example.test/user/start.html",
        "https://docs.example.test/user/window.html",
    }
    assert fetched[1] in {
        "https://docs.example.test/user/tools/labels.html",
        "https://docs.example.test/user/tools/help.html",
    }
    assert fetched[2] in {
        "https://docs.example.test/user/commands/open.html",
        "https://docs.example.test/user/commands/save.html",
    }

    root_page = next(
        part
        for part in workspace.parts.values()
        if part.kind == "html_page"
        and int(part.metadata.get("navigation_depth", 0)) == 0
    )
    assert [group["label"] for group in root_page.metadata["navigation_groups"]] == [
        "General",
        "Tools",
        "Commands",
    ]
    assert all(
        group["container_tag"] == "details"
        for group in root_page.metadata["navigation_groups"]
    )
    catalogs = {
        part.label: part
        for part in workspace.parts.values()
        if part.kind == "html_navigation_group"
    }
    assert list(catalogs) == ["General", "Tools", "Commands"]
    assert "Use this guide" not in workspace.part_text[catalogs["General"].part_id]
    assert "Startup" in workspace.part_text[catalogs["General"].part_id]
    assert "Labels" in workspace.part_text[catalogs["Tools"].part_id]
    assert "open" in workspace.part_text[catalogs["Commands"].part_id]

    plan = EvidencePlanEngine(workspace).build_skeleton_plan()
    by_key = {node.node_key: node for node in plan.nodes}
    root = by_key["document"]
    topics = [by_key[key] for key in ["topic_001", "topic_002", "topic_003"]]
    assert [node.title for node in topics] == ["General", "Tools", "Commands"]
    assert all(node.parent_node_key == root.node_key for node in topics)
    assert all(len(node.source_refs) == 1 for node in topics)
    assert [
        sum(1 for node in plan.nodes if node.parent_node_key == topic.node_key)
        for topic in topics
    ] == [2, 1, 1]
    assert len(root.source_refs) == 1
    assert "Use this guide" in workspace.part_text[root.source_refs[0].source_part_id]
    del store


def test_t5_1_agent_curation_excludes_noise_and_merges_adjacent_same_page_units(tmp_path):
    html = b"""<!doctype html><html><body><main>
    <h1>ExampleBio Guide</h1><p>This guide explains the normal workflow.</p>
    <h2>Open data</h2><p>Choose File and open sample.fasta.</p>
    <h2>Save results</h2><p>Choose Save Results after reviewing the alignment.</p>
    <nav><h2>Next topic</h2><p>Table of contents</p></nav>
    </main></body></html>"""
    store, _, workspace = _workspace(
        tmp_path, "run_t5_1_curated", "guide.html", html, max_html_pages=0
    )
    executor = ToolExecutor(workspace, "t5_1_curated_test")
    assert executor.call("html_inventory", {"max_pages": 0})["ok"] is True

    candidates = workspace.evidence_candidates()
    assert [item.label for item in candidates] == [
        "ExampleBio Guide",
        "Open data",
        "Save results",
    ]
    response = executor.call(
        "curate_evidence",
        {
            "selection_mode": "only_selected",
            "selected_part_ids": [candidates[0].part_id, candidates[1].part_id],
            "excluded": [
                {"part_id": candidates[2].part_id, "reason": "Not needed in this focused corpus"}
            ],
            "merge_groups": [
                {"part_ids": [candidates[0].part_id, candidates[1].part_id]}
            ],
            "rationale": "Keep one coherent setup unit.",
        },
    )

    assert response["accepted"] is True
    assert response["materialized_evidence_count"] == 1
    evidence = next(iter(workspace.evidence.values()))
    assert "normal workflow" in evidence.content
    assert "open sample.fasta" in evidence.content
    assert "Save Results" not in evidence.content
    assert "Next topic" not in "\n".join(workspace.part_text.values())
    assert evidence.locator["curated_from_part_ids"] == [
        candidates[0].part_id,
        candidates[1].part_id,
    ]
    exact = store.artifact_path(evidence.text_artifact_id).read_text(encoding="utf-8")
    assert evidence.content == exact[evidence.start_offset : evidence.end_offset]


def test_t5_1_normalizer_hides_exact_duplicate_candidates_but_records_them(tmp_path):
    html = b"""<!doctype html><html><body><main>
    <h1>Guide</h1><p>Shared operational sentence.</p>
    <h2>Repeated box</h2><p>Shared operational sentence.</p>
    </main></body></html>"""
    store, _, workspace = _workspace(
        tmp_path, "run_t5_1_dedup", "guide.html", html, max_html_pages=0
    )
    executor = ToolExecutor(workspace, "t5_1_dedup_test")
    result = executor.call("html_inventory", {"max_pages": 0})

    assert result["ok"] is True
    assert len(workspace.evidence_candidates()) == 1
    duplicate = next(
        part for part in workspace.parts.values() if part.metadata.get("exact_duplicate_of")
    )
    assert duplicate.metadata["evidence_default"] is False
    page = next(part for part in workspace.parts.values() if part.kind == "html_page")
    canonical = store.read_json_artifact(page.metadata["canonical_page_artifact_id"])
    assert len(canonical["candidate_units"]) == 2
    assert sum(bool(item["evidence_candidate"]) for item in canonical["candidate_units"]) == 1


def test_html_h5_h6_are_preserved_inline_but_do_not_create_source_parts(tmp_path):
    html = b"""<!doctype html><html><body><main>
    <h1>Guide</h1><p>Guide introduction.</p>
    <h4>Detailed options</h4><p>Choose the appropriate mode.</p>
    <h5>Option A</h5><p>Use option A for the standard workflow.</p>
    <h6>Caveat</h6><p>Verify the output before continuing.</p>
    <h4>Next operation</h4><p>Continue with the next operation.</p>
    </main></body></html>"""
    store, _, workspace = _workspace(
        tmp_path, "run_html_h4_boundary", "guide.html", html, max_html_pages=0
    )

    result = ToolExecutor(workspace, "html_h4_boundary_test").call(
        "html_inventory", {"max_pages": 0}
    )

    assert result["ok"] is True
    candidates = workspace.evidence_candidates()
    assert [part.label for part in candidates] == [
        "Guide",
        "Detailed options",
        "Next operation",
    ]
    detail = next(part for part in candidates if part.label == "Detailed options")
    detail_text = workspace.part_text[detail.part_id]
    assert "Option A" in detail_text
    assert "Use option A for the standard workflow." in detail_text
    assert "Caveat" in detail_text
    assert "Verify the output before continuing." in detail_text
    page = next(part for part in workspace.parts.values() if part.kind == "html_page")
    semantic_dom = store.read_json_artifact(page.metadata["semantic_dom_artifact_id"])
    assert any(
        node.get("tag") == "h5" and node.get("heading_level") == 5
        for node in semantic_dom["nodes"]
    )
    assert any(
        node.get("tag") == "h6" and node.get("heading_level") == 6
        for node in semantic_dom["nodes"]
    )


def test_t5_1_html_stream_handles_unclosed_paragraphs_without_nested_duplication(tmp_path):
    html = b"""<!doctype html><html><body><div id="container">
    <p>Version notice.</p><hr class="header">
    <h2>DESCRIPTION</h2>Bare opening sentence.<p>Second sentence.
    <h2>NOTES</h2>Direct note.<p>Another note.
    <div class="toc"><h4 class="toc">Table of contents</h4><ul><li>DESCRIPTION</li></ul></div>
    <h2>EXAMPLES</h2><p>Run this:<div><pre>example --run</pre></div>
    <hr class="header"><p>Main index | Full index</p>
    </div></body></html>"""
    _, _, workspace = _workspace(
        tmp_path, "run_t5_1_unclosed_p", "manual.html", html, max_html_pages=0
    )
    result = ToolExecutor(workspace, "t5_1_stream_test").call(
        "html_inventory", {"max_pages": 0}
    )

    assert result["ok"] is True
    text_by_label = {
        part.label: workspace.part_text[part.part_id]
        for part in workspace.evidence_candidates()
    }
    assert "Bare opening sentence" in text_by_label["DESCRIPTION"]
    assert "Second sentence" in text_by_label["DESCRIPTION"]
    assert "Direct note" not in text_by_label["DESCRIPTION"]
    assert "Direct note" in text_by_label["NOTES"]
    assert "Table of contents" not in "\n".join(text_by_label.values())
    assert text_by_label["EXAMPLES"].count("example --run") == 1
    # Source preservation now also includes the authored footer navigation.
    assert "\n".join(text_by_label.values()).count("Main index") == 1


def test_t5_html_budget_advances_breadth_first_at_next_depth(tmp_path, monkeypatch):
    root = b"""<!doctype html><!-- saved from url=(0043)https://docs.example.test/manual/index.html -->
    <html><body><main><h1>Manual</h1><h2>Alpha</h2><a href="alpha.html">Alpha</a>
    <h2>Beta</h2><a href="beta.html">Beta</a></main></body></html>"""
    pages = {
        "alpha.html": b'<html><body><main><h1>Alpha</h1><a href="alpha-deep.html">Deep A</a></main></body></html>',
        "beta.html": b'<html><body><main><h1>Beta</h1><a href="beta-deep.html">Deep B</a></main></body></html>',
        "alpha-deep.html": b"<html><body><main><h1>Deep A</h1></main></body></html>",
        "beta-deep.html": b"<html><body><main><h1>Deep B</h1></main></body></html>",
    }
    _, _, workspace = _workspace(
        tmp_path, "run_t5_html_depth", "depth.html", root, max_html_pages=4
    )
    fetched: list[str] = []

    def fake_read_url(self, value):
        fetched.append(value)
        return FetchResult(pages[value.rsplit("/", 1)[-1]], "text/html", value, 200, {})

    monkeypatch.setattr(AcquisitionService, "_read_url", fake_read_url)
    inventory = ToolExecutor(workspace, "deterministic_html_test").call(
        "html_inventory", {"max_pages": 4}
    )

    assert [value.rsplit("/", 1)[-1] for value in fetched] == [
        "alpha.html",
        "beta.html",
        "alpha-deep.html",
        "beta-deep.html",
    ]
    assert [item["navigation_depth"] for item in inventory["depth_coverage"]] == [1, 2]
    assert [item["fetched_pages"] for item in inventory["depth_coverage"]] == [2, 2]
    assert [item["fetched_pages"] for item in inventory["branch_coverage"]] == [2, 2]


def test_html_random_navigation_is_repeatable_per_run_and_samples_later_links(tmp_path, monkeypatch):
    links = [f"command-{index}.html" for index in range(24)]
    root = (
        '<!-- saved from url=(0043)https://docs.example.test/manual/index.html -->'
        '<main><h1>Commands</h1><ul>'
        + ''.join(f'<li><a href="{link}">{link}</a></li>' for link in links)
        + '</ul></main>'
    ).encode()

    def capture(directory, run_id):
        store, acquisition, workspace = _workspace(
            directory, run_id, "manual.html", root, max_html_pages=8,
        )
        fetched = []

        def read_url(url):
            fetched.append(url.rsplit('/', 1)[-1])
            return FetchResult(b'<h1>Command</h1><p>Instructions.</p>', 'text/html', url, 200, {})

        monkeypatch.setattr(acquisition, '_read_url', read_url)
        result = ToolExecutor(workspace, 'random_navigation_test').call('html_inventory', {})
        assert result['ok'], result
        if 'pages' not in result:
            result = store.read_json_artifact(result['observation_artifact_id'])
        root_part = next(p for p in workspace.parts.values()
                         if p.kind == 'html_page' and p.metadata['navigation_depth'] == 0)
        # Random fetch order must not rewrite the author's directory order.
        assert [url.rsplit('/', 1)[-1] for url in root_part.metadata['controlled_navigation']] == links
        saved_parts = json.loads((store.root / 'sources' / workspace.manifest.source_id / 'parts.json').read_text('utf-8'))
        saved_root = next(p for p in saved_parts if p['part_id'] == root_part.part_id)
        assert saved_root['metadata']['navigation_sampling'] == result['navigation_sampling']
        events, _ = store.read_events(limit=1000)
        sampling = next(e for e in events if e['event_type'] == 'html_navigation_sampling_configured')
        assert sampling['payload'] == result['navigation_sampling']
        return fetched, result

    first, inventory = capture(tmp_path / 'first', 'run_random_navigation')
    retry, repeated = capture(tmp_path / 'retry', 'run_random_navigation')
    fresh, changed = capture(tmp_path / 'fresh', 'run_other_navigation')
    assert retry == first
    assert repeated['navigation_sampling'] == inventory['navigation_sampling']
    assert changed['navigation_sampling']['seed'] != inventory['navigation_sampling']['seed']
    assert fresh != first
    assert len(first) == len(set(first)) == 8
    assert set(first) & set(links[:12])
    assert set(first) & set(links[12:])
    assert set(first).isdisjoint(url.rsplit('/', 1)[-1] for url in inventory['unfetched_discovered_urls'])
    assert len(inventory['unfetched_discovered_urls']) == 16
    assert inventory['child_page_count'] == 8 and inventory['page_count'] == 9
    assert inventory['navigation_sampling']['strategy'] == 'seeded_random_per_branch_depth.v1'
    assert 'temporary_priority_urls' not in inventory['navigation_sampling']


@pytest.mark.parametrize('budget', [0, 3, 5, 25])
def test_grass_random_sampling_does_not_force_or_exclude_d_vect(tmp_path, monkeypatch, budget):
    base = 'https://grass.osgeo.org/grass84/manuals/'
    root = (f'<!-- saved from url=(0050){base}index.html -->'
            '<main><h1>GRASS GIS</h1><h2>Display</h2>'
            '<a href="display.html">Display commands manual</a>'
            '<a href="displaydrivers.html">Display drivers</a>'
            '<h2>General</h2><a href="general.html">General commands</a></main>').encode()
    store, acquisition, workspace = _workspace(
        tmp_path, 'run_temporary_grass_priority', 'manual.html', root, max_html_pages=budget,
    )
    fetched = []

    def read_url(url):
        name = url.removeprefix(base)
        fetched.append(name)
        if name == 'display.html':
            html = '<main><h1>Display</h1><ul>' + ''.join(
                f'<li><a href="d.command-{i}.html">Command {i}</a></li>' for i in range(20)
            ) + '<li><a href="d.vect.html">d.vect</a></li></ul></main>'
        elif name == 'general.html':
            html = '<main><h1>General</h1><ul><li><a href="g.region.html">g.region</a></li></ul></main>'
        else:
            html = f'<h1>{name}</h1><p>Exact command instructions for {name}.</p>'
        return FetchResult(html.encode(), 'text/html', url, 200, {})

    monkeypatch.setattr(acquisition, '_read_url', read_url)
    inventory = ToolExecutor(workspace, 'grass_random_sampling_test').call('html_inventory', {})
    assert inventory['ok'], inventory
    if 'pages' not in inventory:
        inventory = store.read_json_artifact(inventory['observation_artifact_id'])
    assert 'temporary_priority_urls' not in inventory['navigation_sampling']
    assert inventory['navigation_sampling']['strategy'] == 'seeded_random_per_branch_depth.v1'
    assert inventory['child_page_count'] == len(fetched) == budget
    assert inventory['page_count'] == budget + 1
    assert len(set(fetched)) == len(fetched)
    if budget >= 3:
        assert set(fetched[:3]) == {'display.html', 'displaydrivers.html', 'general.html'}
    if budget == 25:
        assert fetched.count('d.vect.html') == 1
        page = next(p for p in inventory['pages'] if p['url'] == base + 'd.vect.html')
        assert page['navigation_depth'] == 2 and page['capability_branch'] == 'Display'
        assert store.artifact_path(page['artifact_id']).read_bytes().find(b'Exact command instructions') >= 0
        assert base + 'd.vect.html' not in inventory['unfetched_discovered_urls']
    else:
        # This fixed run's random sample chooses another command. The former
        # temporary override would have forced d.vect into the five-page budget.
        assert 'd.vect.html' not in fetched


def test_html_random_navigation_mixes_links_from_all_parents_at_each_depth(tmp_path, monkeypatch):
    root = b'''<!-- saved from url=(0043)https://docs.example.test/manual/index.html -->
    <main><h1>Display</h1><ul><li><a href="early.html">Early catalog</a></li>
    <li><a href="late.html">Late catalog</a></li></ul></main>'''
    store, acquisition, workspace = _workspace(
        tmp_path, 'run_random_parent_pool', 'manual.html', root, max_html_pages=8,
    )
    fetched = []

    def read_url(url):
        name = url.rsplit('/', 1)[-1]
        fetched.append(name)
        if name in {'early.html', 'late.html'}:
            prefix = name.removesuffix('.html')
            html = '<main><h1>Catalog</h1><ul>' + ''.join(
                f'<li><a href="{prefix}-{i}.html">Command {i}</a></li>' for i in range(12)
            ) + '</ul></main>'
        else:
            html = '<h1>Command</h1><p>Instructions.</p>'
        return FetchResult(html.encode(), 'text/html', url, 200, {})

    monkeypatch.setattr(acquisition, '_read_url', read_url)
    inventory = ToolExecutor(workspace, 'random_navigation_test').call('html_inventory', {})
    assert inventory['ok'], inventory
    if 'pages' not in inventory:
        inventory = store.read_json_artifact(inventory['observation_artifact_id'])
    assert set(fetched[:2]) == {'early.html', 'late.html'}
    assert len(fetched) == len(set(fetched)) == 8
    assert {name.split('-')[0] for name in fetched[2:]} == {'early', 'late'}
    assert [d['fetched_pages'] for d in inventory['depth_coverage']] == [2, 6]
    assert len(inventory['unfetched_discovered_urls']) == 18


def test_html_random_navigation_preserves_scope_deduplication_and_success_budget(tmp_path, monkeypatch):
    root = (
        '<!-- saved from url=(0043)https://docs.example.test/manual/index.html -->'
        '<main><h1>Commands</h1><ul>'
        + ''.join(f'<li><a href="command-{i}.html">Command {i}</a></li>' for i in range(6))
        + '<li><a href="command-0.html#details">Repeated link</a></li>'
        '<li><a href="https://other.test/command.html">Outside manual</a></li>'
        '<li><a href="#commands">Local anchor</a></li></ul></main>'
    ).encode()
    store, acquisition, workspace = _workspace(
        tmp_path, 'run_random_failure', 'manual.html', root, max_html_pages=3,
    )
    attempted = []

    def read_url(url):
        attempted.append(url)
        assert url.startswith('https://docs.example.test/manual/command-')
        assert '#' not in url
        if len(attempted) == 1:
            raise RuntimeError('synthetic fetch failure')
        return FetchResult(b'<h1>Command</h1><p>Instructions.</p>', 'text/html', url, 200, {})

    monkeypatch.setattr(acquisition, '_read_url', read_url)
    inventory = ToolExecutor(workspace, 'random_navigation_test').call('html_inventory', {'max_pages': 99})
    assert inventory['ok'], inventory
    if 'pages' not in inventory:
        inventory = store.read_json_artifact(inventory['observation_artifact_id'])
    assert inventory['max_child_pages'] == inventory['child_page_count'] == 3
    assert inventory['page_count'] == 4
    assert len(attempted) == len(set(attempted)) == 4
    assert len(inventory['navigation_gaps']) == 1
    assert inventory['navigation_gaps'][0]['url'] == attempted[0]
    pending = inventory['unfetched_discovered_urls']
    assert len(pending) == 2 and set(pending).isdisjoint(attempted)
    assert inventory['branch_coverage'][0]['discovered_pages'] == 6


def test_t5_authentication_html_is_excluded_before_normalization(tmp_path):
    settings = _settings(tmp_path)
    store = RunStore(settings.data_dir, "run_t5_auth")
    html = b'<html><body><form><input name="user"><input type="password"></form></body></html>'
    path = store.stage_upload("inp_1", "private.html", html)
    outcome = AcquisitionService(store, settings).acquire(
        [
            StagedInput(
                input_id="inp_1",
                kind="uploaded_file",
                display_name="private.html",
                staged_path=str(path),
                declared_media_type="text/html",
            )
        ],
        _target(),
    )

    assert outcome.manifests[0].status == "excluded"
    assert outcome.manifests[0].exclusion_reason == "authentication_required"
    assert outcome.initial_results[0].evidence == []


def test_html_inventory_keeps_cross_references_out_of_chapter_ownership(tmp_path, monkeypatch):
    root_url = "https://docs.example.test/book/"
    intro_url = root_url + "Introduction/"
    second_url = root_url + "Dependent-Type-Theory/"
    setup_url = root_url + "Introduction/Setup/"
    find_url = root_url + "find/"
    root = b'''<!-- saved from url=(0031)https://docs.example.test/book/ -->
    <html><body><main><h1>Book</h1><h2>Contents</h2><ol class="section-toc">
    <li><a href="Introduction/#intro">Introduction</a></li>
    <li><a href="Dependent-Type-Theory/#types">Dependent Type Theory</a></li>
    <li><a href="find/">Finding Data</a></li></ol></main></body></html>'''
    pages = {
        intro_url: b'''<html><head><base href="../"><title>Introduction</title></head>
        <body><main><h1>Introduction<span class="permalink-widget">
        <a title="Permalink" href="find/?domain=section&amp;name=intro">link</a></span></h1>
        <p>You will learn more in <a href="Dependent-Type-Theory/#types">dependent type theory</a>.</p>
        <p>Also see <a href="Appendix/">the appendix</a>.</p>
        <nav><a rel="next" href="Dependent-Type-Theory/#types">Next</a></nav>
        <h2>Contents</h2><ul class="section-toc"><li>
        <a href="Introduction/Setup/#setup">Setup</a></li></ul>
        </main></body></html>''',
        second_url: b'''<html><head><base href="../"><title>Dependent Type Theory</title></head>
        <body><main><h1>Dependent Type Theory</h1><p>Recall <a href="Introduction/">the introduction</a>.</p></main></body></html>''',
        find_url: b'<html><head><title>Finding Data</title></head><body><main><h1>Finding Data</h1><p>Search instructions.</p></main></body></html>',
        setup_url: b'<html><head><title>Setup</title></head><body><main><h1>Setup</h1><p>Install the tool.</p></main></body></html>',
    }
    store, _, workspace = _workspace(tmp_path, "run_html_link_roles", "book.html", root, max_html_pages=20)
    fetched = []

    def fake_read_url(self, value):
        fetched.append(value)
        return FetchResult(pages[value], "text/html", value, 200, {})

    monkeypatch.setattr(AcquisitionService, "_read_url", fake_read_url)
    executor = ToolExecutor(workspace, "deterministic_html_test")
    inventory = executor.call("html_inventory", {})
    assert inventory["ok"] is True, inventory
    assert fetched == [intro_url, second_url, find_url, setup_url]
    assert inventory["navigation_gaps"] == []
    by_url = {page["url"]: page for page in inventory["pages"]}
    assert by_url[second_url]["navigation_parent_url"] == root_url
    assert by_url[setup_url]["navigation_parent_url"] == intro_url
    observations = store.read_json_artifact(by_url[intro_url]["link_observations_artifact_id"])
    assert observations["base_url"] == root_url
    references = [link for link in observations["links"] if link["link_role"] == "cross_reference"]
    assert {link["url"] for link in references} == {second_url, root_url + "Appendix/"}
    assert all(link["crawl_candidate"] is False for link in references)
    assert by_url[intro_url]["link_role_counts"]["heading_permalink"] == 1
    plan = EvidencePlanEngine(workspace).build_skeleton_plan()
    nodes = {node.title: node for node in plan.nodes}
    assert nodes["Dependent Type Theory"].parent_node_key != nodes["Introduction"].node_key
    assert nodes["Setup"].parent_node_key == nodes["Introduction"].node_key
    assert "dependent type theory" in " ".join(workspace.part_text.values())


def test_html_contents_base_preserves_fragments_and_cannot_expand_scope(tmp_path, monkeypatch):
    from bs4 import BeautifulSoup

    root = b'''<!-- saved from url=(0031)https://docs.example.test/book/ -->
    <html><head><base href="https://other.test/book/"></head><body><main>
    <h1>Contents</h1><ul><li><a href="chapter/">Chapter</a></li></ul>
    </main></body></html>'''
    _, _, workspace = _workspace(tmp_path, "run_html_base_scope", "book.html", root, max_html_pages=20)

    def no_network(self, value):
        pytest.fail(f"Out-of-scope base caused acquisition: {value}")

    monkeypatch.setattr(AcquisitionService, "_read_url", no_network)
    executor = ToolExecutor(workspace, "deterministic_html_test")
    inventory = executor.call("html_inventory", {})
    assert inventory["ok"] is True, inventory
    assert inventory["child_page_count"] == 0
    soup = BeautifulSoup('''<head><base href="../"></head><main><h1>Contents</h1>
    <ul><li><a href="chapter/#details">Chapter</a></li></ul></main>''', "html.parser")
    structure = executor._root_structure_projection(soup, "https://docs.example.test/book/intro/", "https://docs.example.test/book/", [])
    link = structure["nested_link_trees"][0]["items"][0]
    assert link["href"] == "https://docs.example.test/book/chapter/#details"
    assert link["fragment"] == "details"
    assert link["link_role"] == "directory_entry"


def test_t5_pdf_inventory_covers_every_page_and_preserves_page_evidence(tmp_path):
    document = fitz.open()
    for number in range(1, 4):
        page = document.new_page()
        page.insert_text((72, 72), f"Lesson {number}\nOpen data file {number} and run alignment.")
    data = document.tobytes()
    document.close()

    _, result = _finish_pdf(tmp_path, "run_t5_pdf_pages", data)

    assert result.status == "succeeded"
    assert result.coverage.semantic_units == 3
    assert result.coverage.evidence_bearing_parts == 3
    assert len(result.evidence) == 3
    assert {item.locator["page"] for item in result.evidence} == {1, 2, 3}
    assert {item.kind for item in result.evidence} == {"pdf_page_content"}


def test_t5_pdf_explicit_sections_preserve_hierarchy_and_exclude_contents_projection(tmp_path):
    document = fitz.open()
    contents = document.new_page()
    contents.insert_text((72, 72), "Contents")
    contents.insert_text((72, 110), "4.1 Import . . . . . . . 2")
    contents.insert_text((72, 130), "4.2 Export . . . . . . . 2")
    page = document.new_page()
    page.insert_text((72, 72), "Chapter 4")
    page.insert_text((72, 100), "Sequence Workflows")
    page.insert_text((72, 150), "4.1 Import")
    page.insert_text((72, 180), "Choose File > Import and select a FASTA file.")
    page.insert_text((72, 235), "4.2 Export")
    page.insert_text((72, 265), "Choose Export and save the alignment report.")
    data = document.tobytes()
    document.close()

    store, result = _finish_pdf(tmp_path, "run_t5_pdf_sections", data)

    chapter = next(item for item in result.evidence if item.title == "Chapter 4: Sequence Workflows")
    imported = next(item for item in result.evidence if item.title == "4.1 Import")
    exported = next(item for item in result.evidence if item.title == "4.2 Export")
    assert not any(item.locator.get("page") == 1 for item in result.evidence)
    assert "authored_navigation_projection" in {
        item["reason"] for item in result.coverage.skipped
    }
    assert chapter.locator["heading_level"] == 1
    for child in (imported, exported):
        assert child.locator["heading_level"] == 2
        assert child.locator["parent_section_part_id"] == chapter.source_part_id
        assert child.locator["chapter_part_id"] == chapter.source_part_id
        assert child.locator["section_path"].startswith("Chapter 4: Sequence Workflows >")
    parts = json.loads(
        (store.root / "sources" / result.source_id / "parts.json").read_text(encoding="utf-8")
    )
    assert any(item["kind"] == "pdf_text_block" for item in parts)


def test_t5_pdf_tiny_and_repeated_furniture_images_are_preserved_but_filtered(tmp_path):
    document = fitz.open()
    tiny = fitz.Pixmap(fitz.csRGB, fitz.Rect(0, 0, 2, 2), 0)
    tiny.clear_with(0xFF0000)
    tiny_png = tiny.tobytes("png")
    footer = fitz.Pixmap(fitz.csRGB, fitz.Rect(0, 0, 375, 167), 0)
    footer.clear_with(0x663399)
    footer_png = footer.tobytes("png")
    for number in range(4):
        page = document.new_page()
        page.insert_text((72, 72), f"Lesson {number + 1}\nRun the documented workflow.")
        page.insert_image(fitz.Rect(72, 120, 144, 192), stream=tiny_png)
        page.insert_image(fitz.Rect(-2, 660, 412, 846), stream=footer_png)
    data = document.tobytes()
    document.close()

    store, result = _finish_pdf(tmp_path, "run_t5_pdf_decorative", data)

    parts = json.loads(
        (store.root / "sources" / result.source_id / "parts.json").read_text(encoding="utf-8")
    )
    image_parts = [item for item in parts if item["kind"] == "pdf_image_region"]
    assert image_parts
    assert all(item["metadata"]["preserved_binary"] is True for item in image_parts)
    assert not any(item.image_refs for item in result.evidence)
    reasons = {item["reason"] for item in result.coverage.skipped}
    assert "decorative_image_filtered" in reasons
    assert "repeated_page_furniture" in reasons


def test_t5_pdf_fragmented_images_are_grouped_into_one_visual_evidence_ref(tmp_path):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Use the visual control table below to identify each operation.")
    for index in range(10):
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.Rect(0, 0, 80, 60), 0)
        pixmap.clear_with(0x223344 + index)
        row, column = divmod(index, 5)
        x0, y0 = 72 + column * 72, 132 + row * 52
        page.insert_image(
            fitz.Rect(x0, y0, x0 + 44, y0 + 32), stream=pixmap.tobytes("png")
        )
    data = document.tobytes()
    document.close()

    store, result = _finish_pdf(tmp_path, "run_t5_pdf_group", data)

    evidence = next(item for item in result.evidence if item.image_refs)
    assert len(evidence.image_refs) == 1
    image_ref = evidence.image_refs[0]
    assert store.artifact_record(image_ref.artifact_id).kind == "pdf_visual_group_render"
    assert sum(
        item["reason"] == "grouped_into_visual_region" for item in result.coverage.skipped
    ) == 10


def test_t5_pdf_informative_image_is_attached_to_exact_multimodal_evidence(tmp_path):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Use the controls shown below to configure the alignment.")
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.Rect(0, 0, 320, 180), 0)
    pixmap.clear_with(0x336699)
    page.insert_image(fitz.Rect(72, 120, 520, 372), stream=pixmap.tobytes("png"))
    data = document.tobytes()
    document.close()

    store, result = _finish_pdf(tmp_path, "run_t5_pdf_multimodal", data)

    evidence = next(item for item in result.evidence if item.image_refs)
    assert evidence.content.startswith("Use the controls shown below")
    assert evidence.extraction_method == "multimodal_source_unit"
    assert evidence.image_refs[0].role == "informative"
    assert store.artifact_record(evidence.image_refs[0].artifact_id).kind == "pdf_embedded_image"


def test_t5_pdf_image_only_page_preserves_visual_content_without_ocr(tmp_path):
    document = fitz.open()
    page = document.new_page()
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.Rect(0, 0, 320, 180), 0)
    pixmap.clear_with(0x669933)
    page.insert_image(fitz.Rect(72, 100, 520, 352), stream=pixmap.tobytes("png"))
    data = document.tobytes()
    document.close()

    _store, _, workspace = _workspace(
        tmp_path, "run_t5_pdf_visual_only", "manual.pdf", data
    )
    result = ToolExecutor(workspace, "deterministic_pdf_test").call(
        "pdf_inventory", {}
    )

    assert result["ok"] is True, result
    candidates = workspace.evidence_candidates()
    assert any(part.metadata.get('image_refs') for part in candidates)
    assert any('OCR was not used' in workspace.part_text.get(part.part_id, '') for part in candidates)


def test_html_root_structure_preserves_sphinx_topic_and_subtopic_lists(tmp_path):
    root_html = b"""<!doctype html>
    <!-- saved from url=(0047)https://docs.example.test/manual/index.html -->
    <html><head><title>Scientific Manual 4.9.6</title></head><body><main>
      <h1>Welcome to the Scientific Manual<a class="headerlink" href="#welcome">&#182;</a></h1>
      <div class="toctree-wrapper compound"><p class="caption">Contents:</p><ul>
        <li class="toctree-l1"><a href="getting-started.html">Getting Started</a><ul>
          <li class="toctree-l2"><a href="configuration.html#syntax">Configuration</a></li>
          <li class="toctree-l2"><a href="first-document.html">First Document</a></li>
        </ul></li>
        <li class="toctree-l1"><a href="editing.html">Editing</a><ul>
          <li class="toctree-l2"><a href="editing.html#shortcuts">Keyboard Shortcuts</a></li>
        </ul></li>
      </ul></div>
    </main></body></html>"""
    store, _, workspace = _workspace(
        tmp_path,
        "run_t5_sphinx_root_structure",
        "manual.html",
        root_html,
        max_html_pages=0,
    )
    normalizer = ToolExecutor(workspace, "deterministic_html_test")
    inventory = normalizer.call("html_inventory", {"max_pages": 0})
    assert inventory["ok"] is True, inventory

    root_structure = store.read_json_artifact(
        inventory["root_structure_artifact_id"]
    )
    assert root_structure["headings"][0]["label"] == "Welcome to the Scientific Manual"
    assert len(root_structure["nested_link_trees"]) == 1
    topic_items = root_structure["nested_link_trees"][0]["items"]
    assert [item["label"] for item in topic_items] == ["Getting Started", "Editing"]
    assert [item["label"] for item in topic_items[0]["children"]] == [
        "Configuration",
        "First Document",
    ]
    assert topic_items[0]["children"][0]["fragment"] == "syntax"

    engine = EvidencePlanEngine(workspace)
    candidate = engine.build_skeleton_plan()
    artifacts = engine.materialize(candidate)
    workspace.evidence_skeleton_plan = candidate
    workspace.evidence_skeleton_plan_artifact_id = artifacts.plan_artifact_id
    workspace.evidence_skeleton_tree_artifact_id = artifacts.tree_artifact_id
    workspace.evidence_tree = artifacts.tree
    workspace.evidence_tree_artifact_id = artifacts.tree_artifact_id

    architect = ToolExecutor(workspace, "html_evidence_architect")
    view = architect.call("get_root_structure", {})
    assert view["ok"] is True
    assert view["root_structure"]["nested_link_trees"][0]["link_count"] == 5
    assert view["candidate_evidence_skeleton"]["selected_node"]["node_key"] == "document"
    assert "create_node" in view["revision_operations"]
