"""Described catalogs remain navigable and missing bodies stay explicit."""
import httpx
import pytest
from bs4 import BeautifulSoup

from app.acquisition import AcquisitionService, FetchResult
from app.config import Settings
from app.evidence_planning import EvidencePlanEngine
from app.html_navigation import classify_html_link
from app.models import StagedInput, TargetProfile
from app.storage import RunStore
from app.toolkit import SourceWorkspace, ToolExecutor


@pytest.mark.parametrize("markup,roles", [
    ('<ul><li><a href="item.html">Item</a> – a description with <a href="ref.html">a reference</a></li></ul>',
     ['directory_entry', 'cross_reference']),
    ('<ul><li><b><a href="item.html">Item</a></b>: description<ul><li>See <a href="ref.html">elsewhere</a>.</li></ul></li></ul>',
     ['directory_entry', 'cross_reference']),
    ('<dl><dt><a href="item.html">Item</a> — description</dt></dl>', ['directory_entry']),
    ('<table><tr><td><a href="item.html">Item</a> - description</td></tr></table>', ['directory_entry']),
    ('<details><summary>Note</summary><ul><li>Read <a href="item.html">this item</a> – for details.</li></ul></details>',
     ['cross_reference']),
    ('<ul><li><a href="item.html">Item</a> is discussed with <a href="ref.html">a reference</a>.</li></ul>',
     ['cross_reference', 'cross_reference']),
])
def test_described_entries_use_the_leading_link_and_authored_separator(markup, roles):
    soup = BeautifulSoup(markup, 'html.parser')
    assert [classify_html_link(a)[0] for a in soup.find_all('a')] == roles


def _workspace(tmp_path):
    settings = Settings(data_dir=tmp_path/'data', output_dir=tmp_path/'out', max_html_pages=8)
    store = RunStore(settings.data_dir, 'run_described_catalog')
    data = '''<!-- saved from url=(0042)https://docs.example.test/manual/index.html -->
    <html><head><title>Manual</title></head><body><h1>Manual</h1>
    <details><summary>Operations</summary><ul>
    <li><a href="first.html">First</a> – perform the first operation with <a href="reference.html">a reference</a></li>
    <li><a href="second.html">Second</a> – perform the second operation</li>
    </ul></details></body></html>'''.encode('utf-8')
    path = store.stage_upload('I1', 'manual.html', data)
    target = TargetProfile(software_name='Example', target_version='1', platform='General')
    acquisition = AcquisitionService(store, settings)
    manifest = acquisition.acquire([StagedInput(input_id='I1', kind='uploaded_file',
        display_name='manual.html', staged_path=str(path), declared_media_type='text/html')], target).manifests[0]
    workspace = SourceWorkspace(store, settings, acquisition, target, manifest)
    return workspace, ToolExecutor(workspace, 'navigation_test')


def test_described_catalog_fetches_bodies_without_borrowing_description_links(tmp_path, monkeypatch):
    workspace, executor = _workspace(tmp_path)
    fetched = []

    def read_url(self, url):
        fetched.append(url)
        name = url.rsplit('/', 1)[-1]
        return FetchResult(f'<h1>{name}</h1><p>Exact body for {name}.</p>'.encode(), 'text/html', url, 200, {})

    monkeypatch.setattr(AcquisitionService, '_read_url', read_url)
    inventory = executor.call('html_inventory', {})
    assert inventory['ok'], inventory
    assert {url.rsplit('/', 1)[-1] for url in fetched} == {'first.html', 'second.html'}
    assert inventory['child_page_count'] == 2
    assert inventory['acquisition_complete'] is True
    plan = EvidencePlanEngine(workspace).build_skeleton_plan()
    materialized = EvidencePlanEngine(workspace).materialize(plan)
    content = '\n'.join(node.content for node in materialized.tree.nodes)
    assert 'Exact body for first.html.' in content
    assert 'Exact body for second.html.' in content


def test_failed_children_preserve_diagnostics_and_entry_page_fallback(tmp_path, monkeypatch):
    workspace, executor = _workspace(tmp_path)

    def read_url(self, url):
        raise httpx.ConnectError('[Errno 11001] getaddrinfo failed')

    monkeypatch.setattr(AcquisitionService, '_read_url', read_url)
    inventory = executor.call('html_inventory', {})
    assert inventory['ok'], inventory
    assert inventory['child_page_count'] == 0
    assert inventory['acquisition_complete'] is False
    assert len(inventory['navigation_gaps']) == 2
    persisted = workspace.store.read_json_artifact(inventory['inventory_artifact_id'])
    for gap in persisted['navigation_gaps']:
        assert gap['error_type'] == 'ConnectError'
        assert 'getaddrinfo failed' in gap['message']
    assert any('Only the HTML entry page was captured' in gap for gap in workspace.gaps)
    plan = EvidencePlanEngine(workspace).build_skeleton_plan()
    materialized = EvidencePlanEngine(workspace).materialize(plan)
    content = '\n'.join(node.content for node in materialized.tree.nodes)
    assert 'perform the first operation' in content
    assert 'Exact body' not in content
