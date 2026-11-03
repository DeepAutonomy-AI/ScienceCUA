"""Compiled-help sidebars are captured as data with independent provenance."""
import json
import copy

import pytest

from app.html_guided.navigation import add_navigation_links, parse_robohelp_toc, recover_navigation, robohelp_base
from app.html_guided.pipeline import _model_link, build_snapshot, run_guided_html
from test_html_guided_pipeline import ROOT_URL, combined_text, install_pages, plan_for, workspace_for

HELP = '<html><head><meta name="generator" content="Adobe RoboHelp 2022"><title>Help</title><script src="whxdata/layoutconfig.js"></script></head><body><main><h1>Help</h1><p>Choose a topic from the sidebar.</p></main></body></html>'
BASE = 'https://example.org/docs/'


def toc(entries):
    return '(function(){var toc = ' + json.dumps(entries) + '; window.rh.model.publish("data",toc);})();'


def test_navigation_parser_reads_only_literal_json_without_executing_trailing_code():
    text = toc([{"type": "item", "name": "Chapter", "url": "chapter.htm"}]) + '; fetch("https://untrusted.invalid/")'
    assert parse_robohelp_toc(text) == [{"type": "item", "name": "Chapter", "url": "chapter.htm"}]
    with pytest.raises(ValueError):
        parse_robohelp_toc('var toc = run_arbitrary_code();')
    assert robohelp_base(HELP, ROOT_URL) == BASE
    assert robohelp_base('<script src="whxdata/layoutconfig.js"></script><main>Ordinary page</main>', ROOT_URL) is None


@pytest.mark.asyncio
async def test_sidebar_topics_and_merged_projects_reach_model_without_rewriting_original_dom(tmp_path, monkeypatch):
    workspace = workspace_for(tmp_path, HELP, max_pages=2)
    pages = {
        BASE + 'whxdata/toc.new.js': toc([{"type": "book", "name": "Tutorials", "key": "toc1"}, {"type": "project", "name": "Merged manual", "ref": "mergedProjects/Child", "key": "unused-global-key"}]),
        BASE + 'whxdata/toc1.new.js': toc([{"type": "item", "name": "Setup", "url": "setup.htm"}]),
        BASE + 'mergedProjects/Child/whxdata/toc.new.js': toc([{"type": "book", "name": "Using the tool", "key": "toc1"}]),
        BASE + 'mergedProjects/Child/whxdata/toc1.new.js': toc([{"type": "item", "name": "Use", "url": "use.htm"}]),
        BASE + 'setup.htm': '<main><h1>Setup</h1><p>Choose the exact setup command.</p></main>',
        BASE + 'mergedProjects/Child/use.htm': '<main><h1>Use</h1><p>Press Run to execute the documented procedure.</p></main>',
    }
    requested = install_pages(monkeypatch, workspace, pages)
    offered = []

    async def decide(payload):
        if payload['page']['url'] == ROOT_URL:
            assert payload['page']['external_navigation']['topic_count'] == 2
            offered.extend(payload['page']['links'])
            assert all(link['navigation_source']['artifact_id'] for link in offered)
            assert all(link['content_root_ids'] == [] and link['authored_contents'] for link in offered)
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert result['page_count'] == 3 and result['child_page_count'] == 2
    assert result['acquisition_complete'] is True
    assert set(requested) == set(pages)
    assert {link['url'] for link in offered} == {BASE + 'setup.htm', BASE + 'mergedProjects/Child/use.htm'}
    assert 'Choose the exact setup command.' in combined_text(workspace)
    assert 'Press Run to execute the documented procedure.' in combined_text(workspace)
    assert result['external_navigation']['resource_count'] == 4
    raw = workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_text('utf-8')
    assert raw == HELP and 'setup.htm' not in raw
    snapshots = [workspace.store.read_json_artifact(a) for a in result['pipeline_artifact_ids']
                 if workspace.store.artifact_record(a).kind == 'html_guided_snapshot']
    root = snapshots[0]
    assert all(link['navigation_source']['json_pointer'].startswith('/') for link in root['links'])


@pytest.mark.asyncio
async def test_navigation_resource_budget_and_fetch_failure_are_explicit_gaps(tmp_path, monkeypatch):
    import app.html_guided.navigation as navigation
    workspace = workspace_for(tmp_path, HELP)
    install_pages(monkeypatch, workspace, {BASE + 'whxdata/toc.new.js': toc([{"type": "book", "name": "Tutorials", "key": "toc1"}])})
    monkeypatch.setattr(navigation, 'MAX_NAVIGATION_FILES', 1)
    captured = await recover_navigation(workspace, HELP, ROOT_URL)
    assert captured['topics'] == []
    assert any(gap['status'] == 'navigation_metadata_budget' for gap in captured['gaps'])
    monkeypatch.setattr(navigation, 'MAX_NAVIGATION_FILES', 4)
    captured = await recover_navigation(workspace, HELP, ROOT_URL)
    assert any(gap['status'] == 'navigation_metadata_failed' for gap in captured['gaps'])


@pytest.mark.asyncio
async def test_foreign_projects_and_invalid_keys_are_not_fetched(tmp_path, monkeypatch):
    workspace = workspace_for(tmp_path, HELP)
    requested = install_pages(monkeypatch, workspace, {BASE + 'whxdata/toc.new.js': toc([
        {"type": "project", "name": "Foreign", "ref": "https://other.invalid/docs", "key": "toc1"},
        {"type": "project", "name": "Local from remote", "ref": "file:///C:/private", "key": "toc1"},
        {"type": "book", "name": "Traversal", "key": "../../private"},
    ])})
    captured = await recover_navigation(workspace, HELP, ROOT_URL)
    assert requested == [BASE + 'whxdata/toc.new.js']
    assert {gap['reason'] for gap in captured['gaps']} == {'cross_origin_project', 'invalid_book_key'}


def test_supplemental_toc_deduplicates_topics_and_keeps_data_provenance():
    snapshot, _ = build_snapshot('<main><a href="chapter.htm">Existing chapter</a></main>', ROOT_URL)
    source = {'url': BASE + 'whxdata/toc.new.js', 'artifact_id': 'captured_toc', 'json_pointer': '/0'}
    topics = [{'url': BASE + 'chapter.htm', 'label': 'Duplicate', 'groups': ['Book'], 'source': source},
              {'url': BASE + 'other.htm', 'label': 'Other', 'groups': ['Book'], 'source': source}]
    add_navigation_links(snapshot, {'topics': topics, 'resources': [], 'gaps': [], 'format': 'robohelp-json-toc'})
    assert len(snapshot['links']) == 2
    added = snapshot['links'][1]
    assert added['navigation_source'] == source and added['dom_path'] is None
    assert added['disclosure_path'] == ['Book']


@pytest.mark.parametrize('dom_use', ['header', 'heading_permalink'])
def test_external_toc_enriches_a_duplicate_url_without_replacing_dom_evidence(dom_use):
    html = ('<header><a href="chapter.htm">Next</a></header><main><h1>Manual</h1></main>'
            if dom_use == 'header' else
            '<main><h1>Manual</h1><h2>Chapter<span class="permalink-widget"><a href="chapter.htm#intro">🔗</a></span></h2></main>')
    snapshot, _ = build_snapshot(html, ROOT_URL)
    original = copy.deepcopy(snapshot['links'][0])
    first = {'url': BASE + 'whxdata/toc1.new.js', 'artifact_id': 'captured_first', 'json_pointer': '/0'}
    second = {'url': BASE + 'whxdata/toc2.new.js', 'artifact_id': 'captured_second', 'json_pointer': '/1'}
    topics = [
        {'url': BASE + 'chapter.htm', 'label': 'Authored chapter', 'groups': ['Current book'], 'source': first},
        {'url': BASE + 'chapter.htm#intro', 'label': 'Related catalog entry', 'groups': ['Second section'], 'source': second},
    ]
    navigation = {'topics': topics, 'resources': [], 'gaps': [], 'format': 'robohelp-json-toc'}
    add_navigation_links(snapshot, navigation)
    assert len(snapshot['links']) == 1
    merged = snapshot['links'][0]
    for key in ('link_id', 'url', 'href', 'fragment', 'dom_path', 'label', 'context', 'content_root_ids'):
        assert merged[key] == original[key]
    assert merged['occurrences'][:len(original['occurrences'])] == original['occurrences']
    assert merged['occurrence_count'] == original['occurrence_count'] + 2
    assert merged['authored_contents'] is True
    assert merged['legacy_role'] == 'directory_entry'
    assert merged['disclosure_path'] == ['Current book']
    assert merged['navigation_source'] == first
    assert [item['source'] for item in merged['external_navigation_entries']] == [first, second]
    assert any(item['tag'] == 'external_navigation' for item in merged['landmark_context'])
    model = _model_link(merged)
    assert model['navigation_hint'] == 'directory_entry'
    assert model['external_navigation_entries'] == [
        {'label': 'Authored chapter', 'groups': ['Current book']},
        {'label': 'Related catalog entry', 'groups': ['Second section']},
    ]
    # Re-applying the same captured metadata cannot fabricate new occurrences.
    add_navigation_links(snapshot, navigation)
    assert merged['occurrence_count'] == original['occurrence_count'] + 2


@pytest.mark.asyncio
async def test_external_toc_duplicate_reaches_model_as_authored_entry_from_header(tmp_path, monkeypatch):
    html = HELP.replace('<body>', '<body><header><h1>Site brand</h1><a rel="up" href="setup.htm">Overview</a></header>').replace('<h1>Help</h1>', '<h1>Workflow Manual</h1>')
    workspace = workspace_for(tmp_path, html, max_pages=1)
    pages = {
        BASE + 'whxdata/toc.new.js': toc([{'type': 'book', 'name': 'Getting started', 'key': 'toc1'}]),
        BASE + 'whxdata/toc1.new.js': toc([{'type': 'item', 'name': 'Setup steps', 'url': 'setup.htm'}]),
        BASE + 'setup.htm': '<main><h1>Setup steps</h1><p>Choose the documented setup command.</p></main>',
    }
    requested = install_pages(monkeypatch, workspace, pages)

    async def decide(payload):
        if payload['page']['url'] == ROOT_URL:
            link = payload['page']['links'][0]
            assert link['url'] == BASE + 'setup.htm'
            assert link['authored_contents'] is True
            assert link['navigation_hint'] == 'directory_entry'
            assert link['navigation_source']['artifact_id']
            assert link['external_navigation_entries'] == [{'label': 'Setup steps', 'groups': ['Getting started']}]
            assert link['page_navigation_only'] is False
            assert link['non_navigation_occurrence_count'] == 1
            assert link['page_navigation_relations'] == ['up']
            assert payload['entry_scope']['entry_heading'] == 'Workflow Manual'
            main_alias = next(item['container_id'] for item in payload['page']['containers'] if item['tag'] == 'main')
            assert main_alias not in link['content_root_ids']  # No invented DOM location for the external TOC.
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert result['page_count'] == 2 and result['acquisition_complete'] is True
    assert set(requested) == set(pages)
    assert workspace.store.artifact_path(workspace.manifest.raw_artifact_id).read_text('utf-8') == html
    assert 'Choose the documented setup command.' in combined_text(workspace)
