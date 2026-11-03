from unittest.mock import patch

import pytest

from app.acquisition import AcquisitionService
from app.config import Settings
from app.html_sourcepart.extractor import Extractor
from app.html_sourcepart.scope import filter_document, preview_html_outline
from app.models import StagedInput, TargetProfile
from app.storage import RunStore
from app.toolkit import SourceWorkspace, ToolExecutor


ROOT = 'https://example.org/book/index.html'


def _outline(html):
    return preview_html_outline(html.encode('utf-8'), root_url=ROOT, source_id='SRC1')


@pytest.mark.parametrize('tag,items', [
    ('ul', '<li><a href="a.html">Chapter A</a><ul><li><a href="a2.html">A2</a></li></ul></li>'
           '<li><a href="b.html">Chapter B</a></li>'),
    ('ol', '<li><a href="a.html">Chapter A</a><ol><li><a href="a2.html">A2</a></li></ol></li>'
           '<li><a href="b.html">Chapter B</a></li>'),
    ('dl', '<dt><a href="a.html">Chapter A</a></dt><dd><dl><dt><a href="a2.html">A2</a></dt></dl></dd>'
           '<dt><a href="b.html">Chapter B</a></dt>'),
])
def test_preview_preserves_first_level_chapters_and_subtree_without_acquisition(tag, items):
    html = f'<html><head><title>Manual</title></head><body><nav><a href="home.html">Home</a></nav>' \
           f'<main><h1>Manual</h1><div class="toc"><{tag}>{items}</{tag}></div>' \
           '<p>Also read <a href="unrelated.html">another page</a>.</p></main></body></html>'
    with patch.object(AcquisitionService, 'fetch_html_child', side_effect=AssertionError('preview must not acquire')):
        preview = _outline(html)
    assert preview['outline_available']
    assert [node['title'] for node in preview['sections']] == ['Chapter A', 'Chapter B']
    assert preview['sections'][0]['known_page_count'] == 2
    assert preview['sections'][0]['children'][0]['url'] == 'https://example.org/book/a2.html'
    assert preview == _outline(html)


def test_preview_uses_same_page_headings_when_no_directory_exists():
    html = '<main><h1>Manual</h1><h2>Plot</h2><p>Plot text.</p><h3>Syntax</h3>' \
           '<h2>Calculate</h2><p>Calculation text.</p></main>'
    preview = _outline(html)
    assert [node['title'] for node in preview['sections']] == ['Plot', 'Calculate']
    assert all(node['heading_dom_path'] and node['known_page_count'] == 1 for node in preview['sections'])


def test_preview_does_not_invent_chapters_from_prose_links():
    assert not _outline('<main><p>Read <a href="a.html">more</a>.</p></main>')['outline_available']


@pytest.mark.parametrize('encoding,charset', [('gb18030', 'gb18030'), ('cp1252', 'windows-1252')])
def test_preview_decoding_matches_formal_parser_for_non_utf8_chapters(encoding, charset):
    titles = ['绘图', '计算'] if encoding == 'gb18030' else ['Réseau', 'Données']
    html = f'<html><head><meta charset="{charset}"><title>Manual</title></head><body><main>' \
           f'<h1>Manual</h1><h2 id="a">{titles[0]}</h2><p>Keep.</p>' \
           f'<h2 id="b">{titles[1]}</h2><p>Omit.</p></main></body></html>'
    preview = preview_html_outline(html.encode(encoding), root_url=ROOT, source_id='SRC1')
    assert [node['title'] for node in preview['sections']] == titles
    from bs4 import UnicodeDammit
    decoded = UnicodeDammit(html.encode(encoding), is_html=True).unicode_markup
    document = filter_document(Extractor(ROOT, 'SRC1').extract(decoded), html=decoded,
                               targets=[preview['sections'][0]])
    assert document['sections'][0]['title'] == titles[0]
    assert titles[1] not in str(document['sections'])


def _workspace(tmp_path, html, selected_indexes, *, budget=80):
    # A saved source preserves its effective online base without making requests.
    source = f'<!-- saved from url=(0040){ROOT} -->{html}'
    path = tmp_path/'manual.html'
    path.write_text(source, encoding='utf-8')
    settings = Settings(data_dir=tmp_path/'data', output_dir=tmp_path/'out', max_html_pages=budget)
    store = RunStore(settings.data_dir, 'run_html_scope')
    target = TargetProfile(software_name='Example', target_version='1', platform='General', ui_locale='English')
    acquisition = AcquisitionService(store, settings)
    manifest = acquisition.acquire([StagedInput(input_id='inp_1', kind='uploaded_file', display_name='manual.html',
                                               staged_path=str(path), declared_media_type='text/html')], target).manifests[0]
    manifest.final_url = ROOT
    preview = _outline(source)
    manifest.source_metadata['extraction_scope'] = {
        'mode':'selected', 'section_ids':[preview['sections'][index]['section_id'] for index in selected_indexes],
        'sections':preview['sections'], 'outline_artifact_id':'outline_fixture',
    }
    workspace = SourceWorkspace(store, settings, acquisition, target, manifest)
    return workspace, ToolExecutor(workspace, 'scope_test')


def _fetcher(workspace, pages, calls):
    def fetch(source_id, root_url, url):
        calls.append(url)
        assert url in pages, f'Unselected or unintended URL was acquired: {url}'
        data = pages[url].encode('utf-8')
        artifact = workspace.store.put_artifact(data, kind='raw_html_part', media_type='text/html', suffix='.html', source_id=source_id)
        return {'status':'succeeded', 'artifact_id':artifact.artifact_id, 'data':data, 'final_url':url}
    return fetch


def _text(workspace):
    return '\n'.join(workspace.part_text.values())


def test_selected_crawl_excludes_other_chapters_and_expands_only_body_child_links(tmp_path):
    html = '<main><h1>Manual</h1><p>Unselected root introduction.</p><div class="toc"><ul>' \
           '<li><a href="a.html">A</a></li><li><a href="b.html">B</a></li></ul></div></main>'
    workspace, executor = _workspace(tmp_path, html, [0])
    pages = {
        'https://example.org/book/a.html':'<main><h1>A</h1><p>Selected chapter body.</p><ul>'
            '<li><a href="a2.html">Child A2</a></li><li><a href="b.html">B</a></li></ul>'
            '<p>Compare <a href="cross.html">this other topic</a>.</p>'
            '<ul><li><a href="unknown/index.html">Unowned directory</a></li></ul>'
            '<nav class="toc"><ul><li><a href="global.html">Global navigation</a></li></ul></nav></main>',
        'https://example.org/book/a2.html':'<main><h1>A2</h1><p>Selected descendant body.</p></main>',
    }
    calls = []
    with patch.object(workspace.acquisition, 'fetch_html_child', side_effect=_fetcher(workspace, pages, calls)):
        result = executor.call('html_inventory', {})
    assert result['ok'], result
    assert calls == list(pages)
    assert result['child_page_count'] == 2 and result['page_count'] == 3
    assert 'Selected chapter body.' in _text(workspace) and 'Selected descendant body.' in _text(workspace)
    assert 'Unselected root introduction.' not in _text(workspace)
    structure = workspace.store.read_json_artifact(workspace.root_structure_artifact_id)
    assert [item['label'] for item in structure['nested_link_trees'][0]['items']] == ['A']


@pytest.mark.parametrize('budget', [1, 80])
def test_shared_page_is_captured_once_and_only_selected_anchor_intervals_are_normalized(tmp_path, budget):
    html = '<main><h1>Manual</h1><div class="toc"><ul>' \
           '<li><a href="shared.html#a">A</a></li><li><a href="shared.html#b">B</a></li>' \
           '<li><a href="shared.html#c">C</a></li></ul></div></main>'
    workspace, executor = _workspace(tmp_path, html, [0, 2], budget=budget)
    pages = {'https://example.org/book/shared.html':'<main><h1>Shared manual</h1><p>Omit introduction.</p>'
             '<h2 id="a">A</h2><p>Keep A.</p><pre>  alpha()\n    beta()</pre>'
             '<h2 id="b">B</h2><p>Omit B.</p><h2 id="c">C</h2><p>Keep C.</p></main>'}
    calls = []
    with patch.object(workspace.acquisition, 'fetch_html_child', side_effect=_fetcher(workspace, pages, calls)):
        result = executor.call('html_inventory', {})
    assert result['ok'], result
    assert len(calls) == result['child_page_count'] == 1
    assert len(result['branch_coverage']) == 2
    assert all(record['discovered_pages'] == record['fetched_pages'] == 1
               and record['unfetched_pages'] == 0 for record in result['branch_coverage'])
    text = _text(workspace)
    assert 'Keep A.' in text and 'Keep C.' in text and '  alpha()\n    beta()' in text
    assert 'Omit B.' not in text and 'Omit introduction.' not in text
    page = result['pages'][1]
    semantic = workspace.store.read_json_artifact(page['semantic_dom_artifact_id'])
    assert 'Omit B.' not in str(semantic)
    markdown = workspace.store.artifact_path(page['markdown_artifact_id']).read_text(encoding='utf-8')
    assert 'Omit B.' not in markdown


def test_selected_same_page_chapter_excludes_other_body_and_child_links(tmp_path):
    html = '<main><h1>Manual</h1><h2 id="a">A</h2><p>Keep A.</p>' \
           '<ul><li><a href="a2.html">A2</a></li></ul><h2 id="b">B</h2><p>Omit B.</p>' \
           '<ul><li><a href="b2.html">B2</a></li></ul></main>'
    workspace, executor = _workspace(tmp_path, html, [0])
    calls = []
    pages = {'https://example.org/book/a2.html':'<main><h1>A2</h1><p>Keep descendant.</p></main>'}
    with patch.object(workspace.acquisition, 'fetch_html_child', side_effect=_fetcher(workspace, pages, calls)):
        result = executor.call('html_inventory', {})
    assert result['ok'], result
    assert calls == list(pages)
    assert 'Keep A.' in _text(workspace) and 'Omit B.' not in _text(workspace)


def test_child_budget_is_fair_across_selected_chapters_and_root_is_separate(tmp_path):
    html = '<main><h1>Manual</h1><div class="toc"><ul><li><a href="a.html">A</a><ul>' \
           '<li><a href="a2.html">A2</a></li><li><a href="a3.html">A3</a></li></ul></li>' \
           '<li><a href="b.html">B</a><ul><li><a href="b2.html">B2</a></li></ul></li>' \
           '<li><a href="c.html">C</a></li></ul></div></main>'
    workspace, executor = _workspace(tmp_path, html, [0, 1], budget=2)
    pages = {f'https://example.org/book/{name}.html':f'<main><h1>{name}</h1><p>{name} body.</p></main>'
             for name in ['a', 'b']}
    calls = []
    with patch.object(workspace.acquisition, 'fetch_html_child', side_effect=_fetcher(workspace, pages, calls)):
        result = executor.call('html_inventory', {})
    assert result['ok'], result
    assert calls == list(pages)
    assert result['child_page_count'] == 2 and result['root_page_count'] == 1 and result['page_count'] == 3
    assert result['truncated']
    assert all(record['fetched_pages'] == 1 for record in result['branch_coverage'])
    assert 'https://example.org/book/c.html' not in result['unfetched_discovered_urls']


def test_missing_fragment_and_degraded_parser_never_admit_whole_page():
    html = '<main><h1>Manual</h1><h2 id="a">A</h2><p>Do not accidentally include.</p></main>'
    document = Extractor(ROOT, 'SRC1').extract(html)
    filtered = filter_document(document, html=html, targets=[{'fragment':'absent'}])
    assert filtered['sections'] == []
    assert any(warning['code'] == 'selected_scope_boundary_not_found' for warning in filtered['warnings'])
    document['extraction']['mode'] = 'basic_text'
    filtered = filter_document(document, html=html, targets=[{'fragment':'a'}])
    assert not filtered['sections'] and not filtered['parts']


def test_dynamic_unlisted_sibling_chapter_is_rejected_but_owned_descendant_is_fetched(tmp_path):
    html = '<main><h1>Manual</h1><div class="toc"><ul>' \
           '<li><a href="a/index.html">A</a></li><li><a href="b/index.html">B</a></li></ul></div></main>'
    workspace, executor = _workspace(tmp_path, html, [0])
    pages = {
        'https://example.org/book/a/index.html':'<main><h1>A</h1><ul>'
            '<li><a href="topic.html">Owned topic</a></li>'
            '<li><a href="../b/unlisted.html">Unlisted sibling chapter</a></li>'
            '<li><a href="../unknown/outside.html">Unowned external branch</a></li></ul></main>',
        'https://example.org/book/a/topic.html':'<main><h1>Owned topic</h1><p>Keep body.</p></main>',
    }
    calls = []
    with patch.object(workspace.acquisition, 'fetch_html_child', side_effect=_fetcher(workspace, pages, calls)):
        result = executor.call('html_inventory', {})
    assert result['ok'], result
    assert calls == list(pages)


def test_eighty_child_limit_excludes_root_and_balances_two_chapters(tmp_path):
    def chapter(name):
        links = ''.join(f'<li><a href="{name}/{number}.html">{name} {number}</a></li>' for number in range(41))
        return f'<li><a href="{name}/index.html">{name}</a><ul>{links}</ul></li>'
    html = '<main><h1>Manual</h1><div class="toc"><ul>' + chapter('a') + chapter('b') + '</ul></div></main>'
    workspace, executor = _workspace(tmp_path, html, [0, 1], budget=80)
    pages = {f'https://example.org/book/{name}/{leaf}.html':f'<main><h1>{name} {leaf}</h1><p>Body.</p></main>'
             for name in ['a', 'b'] for leaf in ['index', *range(41)]}
    calls = []
    with patch.object(workspace.acquisition, 'fetch_html_child', side_effect=_fetcher(workspace, pages, calls)):
        result = executor.call('html_inventory', {})
    assert result['ok'], result
    assert len(calls) == len(set(calls)) == result['child_page_count'] == 80
    assert result['page_count'] == 81
    assert [record['fetched_pages'] for record in result['branch_coverage']] == [40, 40]
    assert len(result['unfetched_discovered_urls']) == 4


def test_selected_annotations_images_and_embedded_metadata_do_not_leak_other_chapters(tmp_path):
    html = '<main><h1>Manual</h1><h2 id="a">A</h2><p>Keep A.' \
           '<span style="display:none">Keep hidden A <img src="a.png"></span></p>' \
           '<iframe src="a-embedded.html"></iframe><h2 id="b">B</h2><p>Omit B.' \
           '<span style="display:none">Omit hidden B <img src="b.png"></span></p>' \
           '<iframe src="b-embedded.html"></iframe></main>'
    workspace, executor = _workspace(tmp_path, html, [0], budget=0)
    with patch.object(workspace.acquisition, '_read_url', side_effect=ValueError('fixture resource unavailable')):
        result = executor.call('html_inventory', {})
    assert result['ok'], result
    page = result['pages'][0]
    document = workspace.store.read_json_artifact(page['structured_source_artifact_id'])
    assert 'Omit hidden B' not in str(document['annotations'])
    assert [image['url'] for image in document['images'].values()] == ['https://example.org/book/a.png']
    assert len(document['embedded_resources']) == 1
    assert 'b-embedded.html' not in str(document['embedded_resources'])


def test_whole_page_selection_excludes_unselected_chapter_on_the_shared_page(tmp_path):
    html = '<main><h1>Manual</h1><div class="toc"><ul>' \
           '<li><a href="shared.html">A</a></li><li><a href="shared.html#b">B</a></li></ul></div></main>'
    workspace, executor = _workspace(tmp_path, html, [0])
    pages = {'https://example.org/book/shared.html':'<main><h1>A</h1><p>Keep A body.</p>'
             '<h2 id="b">B</h2><p>Omit B body.</p><ul><li><a href="b2.html">Omit child B2</a></li></ul></main>'}
    calls = []
    with patch.object(workspace.acquisition, 'fetch_html_child', side_effect=_fetcher(workspace, pages, calls)):
        result = executor.call('html_inventory', {})
    assert result['ok'], result
    assert calls == list(pages)
    assert 'Keep A body.' in _text(workspace) and 'Omit B body.' not in _text(workspace)


def test_ambiguous_shared_whole_page_chapters_fail_closed_with_a_gap(tmp_path):
    html = '<main><h1>Manual</h1><div class="toc"><ul>' \
           '<li><a href="shared.html">A</a></li><li><a href="shared.html">B</a></li></ul></div></main>'
    workspace, executor = _workspace(tmp_path, html, [0])
    pages = {'https://example.org/book/shared.html':'<main><h1>Shared</h1><p>Ambiguous body must be excluded.</p></main>'}
    with patch.object(workspace.acquisition, 'fetch_html_child', side_effect=_fetcher(workspace, pages, [])):
        result = executor.call('html_inventory', {})
    assert result['ok'], result
    assert 'Ambiguous body must be excluded.' not in _text(workspace)
    assert any('selected_scope_shared_page_ambiguous' in gap for gap in workspace.gaps)


def test_late_dynamic_fragment_on_captured_page_is_merged_without_a_second_fetch(tmp_path):
    html = '<main><h1>Manual</h1><div class="toc"><ul>' \
           '<li><a href="a.html">A</a></li><li><a href="b.html">B</a></li></ul></div></main>'
    workspace, executor = _workspace(tmp_path, html, [0, 1])
    pages = {
        'https://example.org/book/a.html':'<main><h1>A</h1><ul><li><a href="shared.html#a">Shared A</a></li></ul></main>',
        'https://example.org/book/b.html':'<main><h1>B</h1><ul><li><a href="b2.html">B2</a></li></ul></main>',
        'https://example.org/book/shared.html':'<main><h1>Shared</h1><h2 id="a">A section</h2><p>Keep shared A.</p>'
            '<h2 id="b">B section</h2><p>Keep shared B.</p><ul><li><a href="b3.html">B3</a></li></ul>'
            '<h2 id="c">C section</h2><p>Omit shared C.</p></main>',
        'https://example.org/book/b2.html':'<main><h1>B2</h1><ul><li><a href="shared.html#b">Shared B</a></li></ul></main>',
        'https://example.org/book/b3.html':'<main><h1>B3</h1><p>Keep shared B descendant.</p></main>',
    }
    calls = []
    with patch.object(workspace.acquisition, 'fetch_html_child', side_effect=_fetcher(workspace, pages, calls)):
        result = executor.call('html_inventory', {})
    assert result['ok'], result
    assert calls.count('https://example.org/book/shared.html') == 1
    assert len(calls) == result['child_page_count'] == 5
    assert 'Keep shared A.' in _text(workspace) and 'Keep shared B.' in _text(workspace)
    assert 'Keep shared B descendant.' in _text(workspace) and 'Omit shared C.' not in _text(workspace)
