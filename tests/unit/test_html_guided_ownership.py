from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.html_guided.ownership import (
    MAX_METADATA_BYTES,
    MAX_METADATA_TOTAL_BYTES,
    MetadataFetchResult,
    OwnershipBoundary,
)


ROOT = "https://docs.example.test/chapter3.html"
INDEX = "https://docs.example.test/index.html"


def document(body: str, *, up: str | None = None) -> str:
    return '<html><head>' + (f'<link rel="up" href="{up}">' if up else '') + '</head><body>' + body + '</body></html>'


def root_boundary(body: str | None = None, *, entry: str = ROOT, up: str | None = "index.html") -> OwnershipBoundary:
    body = body or '<main><div class="toc"><ul><li><a href="child.html">Owned child</a></li></ul></div></main>'
    return OwnershipBoundary.from_root(html=document(body, up=up), entry_url=entry,
                                       selected_dom_path="main", artifact_id="art_root")


def ancestor_toc() -> str:
    return document('<div class="toc"><dl>'
                    '<dt><a href="chapter3.html">Chapter3</a></dt><dd><dl><dt><a href="child.html">Child</a></dt></dl></dd>'
                    '<dt><a href="chapter4.html">Chapter4</a></dt><dd><dl><dt><a href="foreign.html">Foreign</a></dt></dl></dd>'
                    '<dt><a href="chapter5.html">Chapter5</a></dt></dl></div>')


async def probe(boundary, pages):
    calls, saved = [], []

    async def fetch(url, max_bytes):
        calls.append((url, max_bytes))
        response = pages[url]
        return response if isinstance(response, MetadataFetchResult) else MetadataFetchResult(response.encode(), url)

    def save(data, metadata):
        saved.append((data, metadata))
        return f"art_meta_{len(saved)}"

    report = await boundary.probe_ancestors(fetch, save)
    return report, calls, saved


def test_root_explicit_toc_does_not_seed_tip_or_adjacent_reference_list():
    boundary = root_boundary('<main><div class="toc"><ul><li><a href="child.html">Child</a></li></ul></div>'
                             '<p>Read <a href="settings.html">settings tweaks</a> if needed.</p>'
                             '<ul><li><a href="citation.html">Citation</a></li></ul></main>')
    assert set(boundary.seeds) == {"https://docs.example.test/child.html"}
    assert boundary.inspect_document(html=document('<p>Settings</p>'), page_url="https://docs.example.test/settings.html", artifact_id="art_settings").status == "unknown"


def test_cross_host_plain_author_catalog_seed_overrides_foreign_breadcrumbs():
    boundary = root_boundary('<main><ul><li><a href="https://another.test/tutorial">Tutorial</a> — explanation</li></ul></main>')
    result = boundary.inspect_document(html=document('<nav aria-label="Breadcrumb"><a href="https://another.test/">Other book</a><span>Tutorial</span></nav>'),
                                       page_url="https://another.test/tutorial", artifact_id="art_foreign")
    assert result.status == "owned"
    assert result.evidence[0].document_url == ROOT


def test_external_toc_requires_captured_source_pointer_and_allows_cross_host():
    boundary = OwnershipBoundary.from_root(html=document('<main>Welcome</main>'), entry_url=ROOT,
        selected_dom_path="main", artifact_id="art_root", external_topics=[
            {"url": "https://external.test/topic", "source": {"url": "https://docs.example.test/toc.json", "artifact_id": "art_toc", "json_pointer": "/topics/0"}},
            {"url": "https://external.test/unproven", "source": {"url": "https://docs.example.test/toc.json"}},
        ])
    assert set(boundary.seeds) == {"https://external.test/topic"}
    assert boundary.seeds["https://external.test/topic"][0].dom_path == "/topics/0"


def test_child_plain_reference_dl_does_not_extend_owned_boundary():
    boundary = root_boundary()
    child = "https://docs.example.test/child.html"
    html = document('<main><dl><dt><a href="foreign.html">Tool reference</a></dt><dd>Related help</dd></dl></main>', up="chapter3.html")
    assert boundary.add_owned_contents(html=html, page_url=child, selected_dom_path="main", artifact_id="art_child") == []
    assert boundary.inspect_document(html=document('<p>foreign</p>'), page_url="https://docs.example.test/foreign.html", artifact_id="art_f").status == "unknown"


def test_explicit_nested_toc_adds_deeper_owned_descendants_even_across_hosts():
    boundary = root_boundary()
    child = "https://docs.example.test/child.html"
    html = document('<main><nav role="doc-toc"><ol><li><a href="https://deep.test/topic">Deep</a></li></ol></nav></main>', up="chapter3.html")
    assert boundary.add_owned_contents(html=html, page_url=child, selected_dom_path="main", artifact_id="art_child") == ["https://deep.test/topic"]
    assert boundary.inspect_document(html=document('<p>deep</p>', up="https://deep.test/other"), page_url="https://deep.test/topic", artifact_id="art_deep").status == "owned"


@pytest.mark.parametrize("opening,closing", [
    ('<aside>', '</aside>'), ('<header>', '</header>'), ('<footer>', '</footer>'),
    ('<div role="complementary">', '</div>'), ('<div role="banner">', '</div>'),
    ('<div role="contentinfo">', '</div>'), ('<div role="site-navigation">', '</div>'),
    ('<nav>', '</nav>'), ('<div role="navigation">', '</div>'),
])
def test_child_site_toc_cannot_override_foreign_publisher_branch(opening, closing):
    boundary = root_boundary('<main><div class="toc"><ul><li><a href="child.html">Lesson</a></li></ul></div></main>'
        '<nav aria-label="Breadcrumb"><a href="/guide">Guide</a><span>Chapter3</span></nav>')
    child = "https://docs.example.test/child.html"
    html = document('<main>' + opening + '<nav role="doc-toc"><ul><li><a href="/tools">Global tools</a></li></ul></nav>' + closing
        + '<nav role="doc-toc"><ul><li><a href="deep.html">Real lesson subchapter</a></li></ul></nav></main>', up="chapter3.html")
    assert boundary.add_owned_contents(html=html, page_url=child, selected_dom_path="main", artifact_id="art_lesson") == ["https://docs.example.test/deep.html"]
    foreign = document('<nav aria-label="Breadcrumb"><a href="/guide">Guide</a><span>Tools</span></nav>')
    assert boundary.inspect_document(html=foreign, page_url="https://docs.example.test/tools", artifact_id="art_tools").status == "foreign"
    assert boundary.inspect_document(html=document('<p>Deep</p>', up="child.html"), page_url="https://docs.example.test/deep.html", artifact_id="art_deep").status == "owned"


@pytest.mark.parametrize("opening,closing", [
    ('<aside role="complementary">', '</aside>'), ('<nav>', '</nav>'),
    ('<div role="navigation">', '</div>'),
])
def test_root_portal_chrome_catalog_still_owns_explicit_resources(opening, closing):
    boundary = root_boundary('<main>' + opening + '<div class="toc"><ul>'
        '<li><a href="https://another.test/resource">Portal resource</a></li></ul></div>' + closing + '</main>')
    assert "https://another.test/resource" in boundary.seeds


def test_breadcrumb_common_ancestor_divergence_not_url_prefix():
    root = '<main><div class="toctree-wrapper"><ul><li><a href="one">One</a></li></ul></div></main>'
    crumbs = '<nav aria-label="Breadcrumb"><a href="/guide">Guide</a><span>Tutorials</span></nav>'
    boundary = root_boundary(root + crumbs, entry="https://docs.example.test/tutorials", up=None)
    foreign = document('<nav aria-label="Breadcrumb"><a href="/guide">Guide</a><a href="/tools">Tools</a><span>Measure</span></nav>')
    result = boundary.inspect_document(html=foreign, page_url="https://docs.example.test/tutorials-looking-similar", artifact_id="art_tools")
    assert result.status == "foreign"
    assert any(p.relation == "publisher_breadcrumb" for p in result.evidence)
    descendant = document('<nav aria-label="Breadcrumb"><a href="/guide">Guide</a><a href="/tutorials">Tutorials</a><span>Nested</span></nav>')
    assert boundary.inspect_document(html=descendant, page_url="https://docs.example.test/completely-different-path", artifact_id="art_nested").status == "owned"
    # The root's publisher breadcrumb already proves this is its ancestor,
    # even when the ancestor page has no useful breadcrumb of its own.
    assert boundary.inspect_document(html=document('<p>Whole guide</p>'), page_url="https://docs.example.test/guide", artifact_id="art_guide").status == "foreign"


def test_breadcrumb_alone_without_author_catalog_does_not_enable_hard_boundary():
    boundary = root_boundary('<main><p>Article without a catalog.</p></main><nav aria-label="Breadcrumb"><a href="/guide">Guide</a><span>Article</span></nav>')
    other = document('<nav aria-label="Breadcrumb"><a href="/guide">Guide</a><a href="/other">Other</a></nav>')
    assert not boundary.strong
    assert boundary.inspect_document(html=other, page_url="https://docs.example.test/other", artifact_id="art_other").status == "unknown"


def test_incomplete_breadcrumb_with_only_parent_link_does_not_prove_sibling():
    boundary = root_boundary('<main><div class="toc"><ul><li><a href="child.html">Child</a></li></ul></div></main>'
        '<nav aria-label="Breadcrumb"><a href="/guide">Guide</a><span>Chapter3</span></nav>')
    incomplete = document('<nav aria-label="Breadcrumb"><a href="/guide">Guide</a></nav>')
    assert boundary.inspect_document(html=incomplete, page_url="https://docs.example.test/maybe-deeper", artifact_id="art_incomplete").status == "unknown"


def test_full_book_entry_owns_deeper_breadcrumbs():
    boundary = root_boundary('<main><div class="toc"><ul><li><a href="/chapter">Chapter</a></li></ul></div></main>', entry="https://docs.example.test/guide", up=None)
    nested = document('<nav aria-label="Breadcrumb"><a href="/guide">Guide</a><a href="/unlisted-chapter">Chapter</a><span>Deep</span></nav>')
    assert boundary.inspect_document(html=nested, page_url="https://docs.example.test/leaf", artifact_id="art_leaf").status == "owned"


@pytest.mark.asyncio
async def test_ancestor_toc_proves_root_siblings_and_preserves_raw_dom_proof():
    boundary = root_boundary()
    report, calls, saved = await probe(boundary, {INDEX: ancestor_toc()})
    assert calls == [(INDEX, MAX_METADATA_BYTES)]
    assert report.total_bytes == len(saved[0][0])
    assert saved[0][1]["parent_relation"]["dom_path"].endswith("link:nth-of-type(1)")
    for owner in ("chapter4.html", "chapter5.html"):
        decision = boundary.inspect_document(html=document('<p>Other chapter section</p>', up=owner),
            page_url="https://docs.example.test/inline-" + owner, artifact_id="art_section")
        assert decision.status == "foreign"
        assert any(item.artifact_id == "art_meta_1" for item in decision.evidence)
    assert boundary.inspect_document(html=document('<p>Unseen deeper node</p>', up="unknown-intermediate.html"),
        page_url="https://docs.example.test/deeper.html", artifact_id="art_deeper").status == "unknown"
    assert boundary.inspect_document(html=document('<p>Known peer itself</p>'), page_url="https://docs.example.test/chapter4.html", artifact_id="art_peer").status == "foreign"


@pytest.mark.asyncio
async def test_missing_exact_entry_or_ambiguous_entry_stays_unknown_no_index_alias_guess():
    for tree in ('<div class="toc"><ul><li><a href="chapter3/index.html">Root alias?</a></li><li><a href="chapter4.html">Other</a></li></ul></div>',
                 '<div class="toc"><ul><li><a href="chapter3.html">A</a></li><li><a href="chapter3.html">B</a></li><li><a href="chapter4.html">Other</a></li></ul></div>'):
        boundary = root_boundary()
        report, _, _ = await probe(boundary, {INDEX: document(tree)})
        assert not boundary.publisher_indexes
        assert report.gaps[0]["status"] == "scope_metadata_entry_not_located"
        assert boundary.inspect_document(html=document('<p>other</p>', up="chapter4.html"), page_url="https://docs.example.test/foreign.html", artifact_id="art_f").status == "unknown"


@pytest.mark.asyncio
async def test_metadata_toc_uncompressed_over_64_members_and_real_dt_dd_parent():
    many = ''.join(f'<dt><a href="deep-{i}.html">Deep{i}</a></dt>' for i in range(180))
    tree = '<div class="toc"><dl><dt><a href="chapter3.html">Root</a></dt><dd><dl>' + many + '</dl></dd><dt><a href="chapter4.html">Peer</a></dt></dl></div>'
    boundary = root_boundary()
    await probe(boundary, {INDEX: document(tree)})
    assert "https://docs.example.test/deep-179.html" in boundary.publisher_indexes[0]["positions"]
    assert boundary.inspect_document(html=document('<p>deep</p>'), page_url="https://docs.example.test/deep-179.html", artifact_id="art_deep").status == "owned"


@pytest.mark.asyncio
async def test_max_two_metadata_requests_and_total_byte_budget():
    first = document('<p>Not enough hierarchy here.</p>', up="book.html")
    second = ancestor_toc()
    boundary = root_boundary()
    report, calls, saved = await probe(boundary, {INDEX: first, "https://docs.example.test/book.html": second})
    assert len(calls) == len(saved) == report.fetch_count == 2
    assert all(limit == MAX_METADATA_BYTES for _, limit in calls)
    assert report.total_bytes <= MAX_METADATA_TOTAL_BYTES
    assert boundary.publisher_indexes
    await boundary.probe_ancestors(lambda *_: pytest.fail("idempotent probe must not fetch again"), lambda *_: "never")


@pytest.mark.asyncio
async def test_other_origin_redirect_credentials_and_port_never_establish_ownership():
    for target in ("https://other.test/index.html", "https://u:p@docs.example.test/index.html", "https://docs.example.test:444/index.html"):
        boundary = root_boundary(up=target)
        report, calls, _ = await probe(boundary, {})
        assert not calls
        assert not boundary.publisher_indexes
        if boundary.documents[ROOT].up:
            assert report.gaps[0]["status"] == "scope_metadata_origin_denied"
    boundary = root_boundary()
    report, _, saved = await probe(boundary, {INDEX: MetadataFetchResult(ancestor_toc().encode(), "https://other.test/index.html")})
    assert not saved
    assert report.gaps[0]["status"] == "scope_metadata_redirect_denied"


@pytest.mark.asyncio
async def test_oversized_or_failed_metadata_is_unknown_not_main_pipeline_exception():
    boundary = root_boundary()
    report, _, saved = await probe(boundary, {INDEX: MetadataFetchResult(b' ' * (MAX_METADATA_BYTES + 1), INDEX)})
    assert not saved
    assert report.gaps[0]["status"] == "scope_metadata_byte_budget"
    boundary = root_boundary()
    report, _, _ = await probe(boundary, {})
    assert report.gaps[0]["status"] == "scope_metadata_fetch_failed"
    assert not boundary.publisher_indexes


@pytest.mark.asyncio
async def test_metadata_probe_propagates_cancellation():
    boundary = root_boundary()

    async def cancelled(url, max_bytes):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await boundary.probe_ancestors(cancelled, lambda *_: "never")


@pytest.mark.asyncio
async def test_metadata_probe_propagates_acquisition_run_cancelled_sentinel():
    boundary = root_boundary()
    sentinel = RuntimeError("run_cancelled")

    async def cancelled(url, max_bytes):
        raise sentinel

    with pytest.raises(RuntimeError, match="^run_cancelled$") as caught:
        await boundary.probe_ancestors(cancelled, lambda *_: "never")
    assert caught.value is sentinel
    assert not boundary.probe_report.gaps


@pytest.mark.asyncio
async def test_safe_local_metadata_and_escape_denial(tmp_path: Path):
    entry = (tmp_path / "chapter3.html").as_uri()
    index = (tmp_path / "index.html").as_uri()
    boundary = root_boundary(entry=entry)
    report, calls, _ = await probe(boundary, {index: ancestor_toc()})
    assert calls and report.captures
    boundary = root_boundary(entry=entry, up="../outside.html")
    report, calls, _ = await probe(boundary, {})
    assert not calls and report.gaps[0]["status"] == "scope_metadata_origin_denied"


def test_conflicting_publisher_facts_stay_unknown_and_empty_selected_path_works():
    boundary = OwnershipBoundary.from_root(html=document('<div class="toc"><ul><li><a href="child.html">Child</a></li></ul></div>'
        '<nav aria-label="Breadcrumb"><a href="/guide">Guide</a><span>Chapter3</span></nav>'), entry_url=ROOT, selected_dom_path="", artifact_id="art_root")
    html = document('<nav aria-label="Breadcrumb"><a href="/guide">Guide</a><a href="/different">Different</a></nav>', up="chapter3.html")
    assert boundary.inspect_document(html=html, page_url="https://docs.example.test/not-a-seed", artifact_id="art_mixed").status == "unknown"
    assert json.loads(json.dumps(boundary.as_dict()))["entry_url"] == ROOT
    assert len(json.dumps(boundary.model_context(), ensure_ascii=False).encode()) <= 4096
