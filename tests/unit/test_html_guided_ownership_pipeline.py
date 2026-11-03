"""A model's relevance judgment cannot override a proven publisher boundary."""
import json

import pytest

from app.html_guided.pipeline import run_guided_html
from test_html_guided_pipeline import ROOT_URL, combined_text, install_pages, plan_for, workspace_for


BASE = 'https://example.org/docs/'


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during", ["metadata_probe", "projection"])
async def test_root_only_cancellation_prevents_sourcepart_materialization(tmp_path, monkeypatch, cancel_during):
    from app.html_guided import pipeline
    from app.html_guided.ownership import OwnershipBoundary

    workspace = workspace_for(tmp_path, '<main><h1>Manual</h1><p>Authored instructions.</p></main>')
    cancelled = False
    monkeypatch.setattr(workspace.store, 'cancellation_requested', lambda: cancelled)
    if cancel_during == 'metadata_probe':
        original = OwnershipBoundary.probe_ancestors

        async def cancel_after_probe(self, *args, **kwargs):
            nonlocal cancelled
            result = await original(self, *args, **kwargs)
            cancelled = True
            return result

        monkeypatch.setattr(OwnershipBoundary, 'probe_ancestors', cancel_after_probe)
    else:
        original = pipeline._projection

        def cancel_after_projection(*args, **kwargs):
            nonlocal cancelled
            result = original(*args, **kwargs)
            cancelled = True
            return result

        monkeypatch.setattr(pipeline, '_projection', cancel_after_projection)

    async def decide(payload):
        return plan_for(payload)

    with pytest.raises(RuntimeError, match='^run_cancelled$'):
        await run_guided_html(workspace, decide)
    assert combined_text(workspace) == ''


@pytest.mark.asyncio
async def test_foreign_reference_is_captured_but_never_planned_or_materialized(tmp_path, monkeypatch):
    root = '''<head><link rel="up" href="book.html"></head><main><h1>Selected chapter</h1>
      <div class="toc"><ul><li><a href="own.html">Owned lesson</a></li></ul></div>
      <p>A related optional reference is <a href="foreign.html">elsewhere</a>.</p></main>'''
    book = '''<main><div class="toc"><ul>
      <li><a href="catalog.html">Selected chapter</a><ul><li><a href="own.html">Lesson</a></li></ul></li>
      <li><a href="other.html">Other chapter</a><ul><li><a href="foreign.html">Reference</a></li></ul></li>
      </ul></div></main>'''
    foreign = '''<head><link rel="up" href="other.html"></head><main><h1>Other reference</h1>
      <p>OUTSIDE-CHAPTER-TEXT</p><a href="deep.html">Do not expand this page</a></main>'''
    pages = {BASE + 'book.html': book, BASE + 'own.html':
             '<head><link rel="up" href="catalog.html"></head><main><h1>Owned lesson</h1><p>Choose Run.</p></main>',
             BASE + 'foreign.html': foreign}
    workspace = workspace_for(tmp_path, root, max_pages=2)
    requested = install_pages(monkeypatch, workspace, pages)
    planned = []

    async def decide(payload):
        planned.append(payload['page']['url'])
        return plan_for(payload)  # Deliberately fetch even the optional reference.

    result = await run_guided_html(workspace, decide)
    assert set(requested) == set(pages)
    assert planned == [ROOT_URL, BASE + 'own.html']
    assert result['page_count'] == 2 and result['attempted_child_page_count'] == 2
    assert result['acquisition_complete'] is True and result['navigation_gaps'] == []
    assert 'Choose Run.' in combined_text(workspace)
    assert 'OUTSIDE-CHAPTER-TEXT' not in combined_text(workspace)
    ownership = result['publisher_ownership']
    assert ownership['metadata_probe']['fetch_count'] == 1
    excluded = ownership['excluded_pages']
    assert len(excluded) == 1 and excluded[0]['url'] == BASE + 'foreign.html'
    assert workspace.store.artifact_path(excluded[0]['raw_artifact_id']).read_text('utf8') == foreign
    proof = json.loads(workspace.store.artifact_path(excluded[0]['artifact_id']).read_text('utf8'))
    assert proof['status'] == 'foreign' and proof['evidence']
    assert all(item['artifact_id'] and item['dom_path'] for item in proof['evidence'])
    decisions = json.loads(workspace.store.artifact_path(result['decision_artifact_id']).read_text('utf8'))['decisions']
    rejected = next(item for item in decisions if item['url'] == BASE + 'foreign.html')
    assert rejected['action'] == 'fetch' and rejected['status'] == 'scope_excluded'
    assert rejected['ownership_decision_origin'] == 'host'


@pytest.mark.asyncio
async def test_unknown_publisher_parent_does_not_fabricate_a_scope_conflict(tmp_path, monkeypatch):
    root = '''<main><h1>Catalog</h1><div class="toc"><ul><li><a href="own.html">Lesson</a></li></ul></div>
      <p>Further text is <a href="unknown.html">here</a>.</p></main>'''
    workspace = workspace_for(tmp_path, root, max_pages=2)
    install_pages(monkeypatch, workspace, {
        BASE + 'own.html': '<main><h1>Lesson</h1><p>Owned instructions.</p></main>',
        BASE + 'unknown.html': '<head><link rel="up" href="unseen-parent.html"></head><main><h1>Unverified</h1><p>Keep the existing model decision.</p></main>',
    })

    async def decide(payload):
        return plan_for(payload)

    result = await run_guided_html(workspace, decide)
    assert result['page_count'] == 3
    assert result['publisher_ownership']['excluded_pages'] == []
    assert result['publisher_ownership']['unknown_page_count'] == 1
    assert 'Keep the existing model decision.' in combined_text(workspace)


@pytest.mark.asyncio
async def test_explicit_cross_publisher_catalog_wins_over_different_breadcrumbs(tmp_path, monkeypatch):
    external = 'https://other.example/tutorial.html'
    root = f'''<nav aria-label="breadcrumb"><a href="/">Home</a>Catalog</nav><main><h1>Reading list</h1>
      <div role="doc-toc"><ul><li><a href="{external}">Authored reading</a></li></ul></div></main>'''
    workspace = workspace_for(tmp_path, root, max_pages=1)
    install_pages(monkeypatch, workspace, {external:
        '<nav aria-label="breadcrumb"><a href="https://example.org/">Home</a><a href="https://example.org/other-book">Other book</a></nav>'
        '<main><h1>Reading</h1><p>Use the listed external tutorial.</p></main>'})

    async def decide(payload):
        return plan_for(payload, choose=lambda link: 'fetch' if link['url'] == external else 'exclude')

    result = await run_guided_html(workspace, decide)
    assert result['page_count'] == 2
    assert result['publisher_ownership']['excluded_pages'] == []
    assert result['publisher_ownership']['checks'][0]['status'] == 'owned'
    assert 'Use the listed external tutorial.' in combined_text(workspace)
