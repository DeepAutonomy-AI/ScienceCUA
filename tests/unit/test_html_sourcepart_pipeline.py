from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import patch

import fitz
import pytest

from app.acquisition import AcquisitionService, FetchResult
from app.config import Settings
from app.evidence_planning import EvidencePlanEngine
from app.html_sourcepart import extractor
from app.html_sourcepart.pipeline import save_snapshot
from app.knowledge import _inline_reference_entries
from app.models import StagedInput, TargetProfile
from app.orchestration.source_stage import validate_source_results
from app.storage import RunStore
from app.toolkit import SourceWorkspace, ToolExecutor

from bs4 import BeautifulSoup
from app.html_navigation import classify_html_link, html_base_url


def workspace_for(tmp_path, source):
    path = tmp_path / 'manual.html'
    path.write_bytes(source.encode('utf-8'))
    settings = Settings(data_dir=tmp_path/'data', output_dir=tmp_path/'out',
                         max_html_pages=0)
    store = RunStore(settings.data_dir, 'run_html_integration')
    target = TargetProfile(software_name='Example', target_version='1', platform='General', ui_locale='English')
    acquisition = AcquisitionService(store, settings)
    manifest = acquisition.acquire([StagedInput(input_id='inp_1', kind='uploaded_file', display_name='manual.html',
                                               staged_path=str(path), declared_media_type='text/html')], target).manifests[0]
    workspace = SourceWorkspace(store, settings, acquisition, target, manifest)
    return workspace, ToolExecutor(workspace, 'integration_test')


def inventory(workspace, executor):
    result = executor.call('html_inventory', {'max_pages':0})
    assert result['ok'], result
    assert result['page_count'] == 1
    page = result['pages'][0]
    source = workspace.store.read_json_artifact(page['structured_source_artifact_id'])
    return page, source


def commit(workspace, executor):
    engine = EvidencePlanEngine(workspace)
    plan = engine.build_skeleton_plan()
    artifacts = engine.materialize(plan)
    response = executor._commit_materialized_evidence(plan, artifacts, summary='Source-preserved integration fixture.',
                                            gaps=[], warnings=[], accepted_event='integration_verified')
    assert response['accepted'], response
    return workspace.final_result


def test_native_html_roundtrip_preserves_structure_images_links_and_evidence(tmp_path):
    source = '''<!-- saved from url=(0043)https://example.org/docs/manual.html -->
    <html><head><title>Example Manual</title><link rel="stylesheet" href="manual.css"></head>
    <body><main><h1>Controls</h1><p>assigned button +<b>Shift</b> switches mode.</p>
    <p><img src="figure.png" alt="Mode diagram"><a href="next.html">details</a></p>
    <ul><li>First mode<ul><li>Nested option</li></ul></li><li>Second mode</li></ul>
    <pre>  run mode\n\n    print(value)</pre>
    <table><tr><th>Key</th><th>Action</th></tr><tr><td>F1</td><td>Rotate model</td></tr>
    <tr><td>F2</td><td>Translate model</td></tr><tr><td>F3</td><td>Fit model</td></tr>
    <tr><td>F4</td><td>Save model</td></tr></table>
    <iframe src="search.html" title="Search"></iframe></main></body></html>'''
    image = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 4, 4), False).tobytes('png')
    (tmp_path/'figure.png').write_bytes(image)
    (tmp_path/'manual.css').write_text('p {color:blue}', encoding='utf-8')
    workspace, executor = workspace_for(tmp_path, source)
    page, document = inventory(workspace, executor)
    assert page['extraction']['mode'] == 'structured'
    assert page['saved_image_count'] == 1
    assert page['embedded_count'] == 1
    assert page['extraction']['embedded_sources_pending']
    snapshot = workspace.store.read_json_artifact(page['source_snapshot_artifact_id'])
    assert snapshot['content'] == source
    assert document['links']['L1']['url'] == 'https://example.org/docs/next.html'
    image_record = document['images']['I1']
    assert workspace.store.artifact_path(image_record['artifact_id']).read_bytes() == image
    part = workspace.evidence_candidates()[0]
    structure = workspace.store.read_json_artifact(part.metadata['structure_artifact_id'])
    blocks = list(extractor.all_blocks(structure['blocks']))
    assert {'table', 'list', 'code', 'paragraph', 'embedded_content'} <= {b['type'] for b in blocks}
    assert next(b for b in blocks if b['type']=='table')['rows'][1]['cells'][1]['column'] == 1
    assert next(b for b in blocks if b['type']=='code')['runs'][0]['text'].startswith('  run mode\n\n')
    assert 'button +**Shift** switches' in workspace.part_text[part.part_id]
    assert part.locator['link_targets_artifact_id'] == page['structured_source_artifact_id']
    result = commit(workspace, executor)
    assert result.normalization_version == 'v16-html-structured.v11-random-navigation.v2'
    assert result.evidence_tree.referenced_part_count == len(EvidencePlanEngine(workspace).eligible_parts())
    assert any(e.image_refs[0].artifact_id == image_record['artifact_id'] for e in result.evidence if e.image_refs)
    entries = [entry for evidence in result.evidence for entry in _inline_reference_entries(evidence)]
    assert [e['title'] for e in entries] == ['F1', 'F2', 'F3', 'F4']
    assert all('|' in e['content'] and e['locator']['entry_structure']=='markdown_table_row' for e in entries)
    original_status = result.status
    original_evidence = [e.model_dump(mode='json') for e in result.evidence]
    validated, summaries = validate_source_results(workspace.store, [result])
    assert summaries[0]['errors'] == []
    assert summaries[0]['rejected_evidence_count'] == 0
    assert validated[0].status == original_status
    assert [e.model_dump(mode='json') for e in validated[0].evidence] == original_evidence


def test_missing_image_is_retained_without_losing_parent_structure(tmp_path):
    workspace, executor = workspace_for(tmp_path, '<h1>Manual</h1><p>Visible instructions.</p><img src="absent.png">')
    page, document = inventory(workspace, executor)
    assert page['extraction']['mode'] == 'structured'
    assert page['missing_image_count'] == 1
    assert document['images']['I1']['status'] == 'missing'
    assert document['images']['I1']['src'] == 'absent.png'
    assert workspace.warnings
    assert commit(workspace, executor).evidence


@pytest.mark.parametrize('fail_basic', [False, True])
def test_parser_failure_still_saves_native_sourceparts_and_snapshot(tmp_path, fail_basic):
    source = '<h1>Manual</h1><p>Unknown publisher content.</p>'
    workspace, executor = workspace_for(tmp_path, source)
    with patch('app.html_sourcepart.extractor.parse_html', side_effect=RuntimeError('parser unavailable')):
        with patch('app.html_sourcepart.fallback.scan', return_value=(None, [{'code':'basic_parser_failed'}])) if fail_basic else nullcontext():
            page, document = inventory(workspace, executor)
    assert page['extraction']['mode'] == ('raw_html' if fail_basic else 'basic_text')
    snapshot = workspace.store.read_json_artifact(page['source_snapshot_artifact_id'])
    assert snapshot['content'] == source
    assert workspace.evidence_candidates()
    assert workspace.gaps
    result = commit(workspace, executor)
    assert result.evidence
    assert result.status == 'degraded'
    assert document['extraction']['degraded']


@pytest.mark.parametrize('error_type', [IndexError, AttributeError, RuntimeError, OverflowError])
def test_unexpected_structure_adaptation_failure_uses_independent_recovery(tmp_path, error_type):
    source = '<h1>Manual</h1><p>Unfamiliar document structure remains available.</p>'
    workspace, executor = workspace_for(tmp_path, source)
    with patch('app.html_sourcepart.pipeline.pipeline_sections', side_effect=error_type('adaptation failed')):
        page, document = inventory(workspace, executor)
    assert page['extraction']['mode'] == 'basic_text'
    assert page['extraction']['reasons'][0]['code'] == 'pipeline_structure_adaptation_failed'
    assert page['extraction']['reasons'][0]['error_type'] == error_type.__name__
    assert workspace.store.read_json_artifact(page['source_snapshot_artifact_id'])['content'] == source
    result = commit(workspace, executor)
    assert result.status == 'degraded'
    assert 'Unfamiliar document structure remains available.' in '\n'.join(e.content for e in result.evidence)
    assert document['extraction']['degraded']


def test_structure_adaptation_does_not_hide_storage_failure(tmp_path):
    workspace, executor = workspace_for(tmp_path, '<h1>Manual</h1><p>Body text.</p>')
    from app.html_sourcepart.pipeline import extract_page
    raw_id = workspace.manifest.raw_artifact_id
    with patch('app.html_sourcepart.pipeline.pipeline_sections', side_effect=PermissionError('storage unavailable')):
        with pytest.raises(PermissionError):
            extract_page(workspace, html='<h1>Manual</h1><p>Body text.</p>',
                         soup=BeautifulSoup('<h1>Manual</h1><p>Body text.</p>', 'html.parser'),
                         page_url='https://example.test/manual.html', raw_artifact_id=raw_id,
                         page_part_id='page', page_ordinal=1, fetch_resources=False)


def test_source_attributes_and_inline_srcdoc_remain_exact_in_artifacts(tmp_path):
    source = '<object data="missing.bin" password="documentation-example">Fallback text.</object><iframe srcdoc="&lt;p&gt;Inside&lt;/p&gt;"></iframe>'
    workspace, executor = workspace_for(tmp_path, source)
    page, document = inventory(workspace, executor)
    assert document['embedded_resources'][0]['attributes']['password'] == 'documentation-example'
    assert document['embedded_resources'][1]['inline_html']['content'] == '<p>Inside</p>'
    assert page['extraction']['mode'] == 'structured'


def test_image_only_heading_gets_a_native_evidence_attachment(tmp_path):
    (tmp_path/'heading.png').write_bytes(fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 4, 4), False).tobytes('png'))
    workspace, executor = workspace_for(tmp_path, '<h1><img src="heading.png" alt="Diagram"></h1>')
    page, _ = inventory(workspace, executor)
    assert page['saved_image_count'] == 1
    result = commit(workspace, executor)
    assert len(result.evidence) == 1 and len(result.evidence[0].image_refs) == 1
    validated, summaries = validate_source_results(workspace.store, [result])
    assert summaries[0]['errors'] == []
    assert len(validated[0].evidence) == 1
    assert validated[0].evidence[0].image_refs


def test_eighty_child_budget_uses_native_structured_extractor_for_every_page(tmp_path, monkeypatch):
    root = '<!-- saved from url=(0043)https://example.org/docs/index.html -->\n<h1>Commands</h1><ul>'
    root += ''.join(f'<li><a href="command-{i}.html">Command {i}</a></li>' for i in range(85)) + '</ul>'
    workspace, executor = workspace_for(tmp_path, root)
    settings = replace(workspace.settings, max_html_pages=80)
    workspace.settings = workspace.acquisition.settings = executor.settings = settings
    fetched = []

    def read_url(url):
        fetched.append(url)
        name = url.rsplit('/', 1)[-1]
        data = f'<h1>{name}</h1><p>Exact instructions for {name}.</p>'.encode()
        return FetchResult(data, 'text/html', url, 200, {})

    monkeypatch.setattr(workspace.acquisition, '_read_url', read_url)
    result = executor.call('html_inventory', {'max_pages':80})
    assert result['ok'], result
    if 'page_count' not in result:
        result = workspace.store.read_json_artifact(result['observation_artifact_id'])
    assert result['page_count'] == 81 and result['child_page_count'] == 80
    assert len(fetched) == 80 and result['truncated']
    assert len(result['unfetched_discovered_urls']) == 5
    assert all(p['extraction']['mode'] == 'structured' and p['extractor_version'] == extractor.VERSION for p in result['pages'])
    assert all(workspace.store.read_json_artifact(p['source_snapshot_artifact_id'])['complete'] for p in result['pages'])
    assert commit(workspace, executor).evidence


def test_extreme_table_span_uses_recovery_and_preserves_all_source(tmp_path):
    source = '<h1>Manual</h1><table><tr><td colspan="999999999">Important unknown content.</td></tr></table>'
    workspace, executor = workspace_for(tmp_path, source)
    page, document = inventory(workspace, executor)
    assert page['extraction']['mode'] == 'basic_text'
    assert workspace.store.read_json_artifact(page['source_snapshot_artifact_id'])['content'] == source
    assert 'Important unknown content.' in '\n'.join(workspace.part_text.values())
    assert document['extraction']['reasons']
    assert commit(workspace, executor).evidence


def test_imported_css_keeps_cascade_order_and_hidden_code_annotations(tmp_path):
    (tmp_path/'base.css').write_text('.hidden { display:none } .shown { display:none }', encoding='utf-8')
    (tmp_path/'manual.css').write_text('@import "base.css"; .shown { display:inline }', encoding='utf-8')
    workspace, executor = workspace_for(tmp_path, '<link rel="stylesheet" href="manual.css"><h1>Example</h1><pre>x<span class="hidden">tooltip</span><span class="shown">y</span></pre>')
    page, _ = inventory(workspace, executor)
    assert page['extraction']['mode'] == 'structured'
    assert 'xy' in '\n'.join(workspace.part_text.values())
    assert 'tooltip' not in '\n'.join(workspace.part_text.values())
    part = workspace.evidence_candidates()[0]
    saved = workspace.store.read_json_artifact(part.metadata['structure_artifact_id'])
    assert any('tooltip' in extractor.plain(annotation['runs']) for annotation in saved['annotations'])


def test_structure_artifact_size_failure_recovers_before_publishing_parts(tmp_path):
    workspace, executor = workspace_for(tmp_path, '<h1>Manual</h1><p>Unfamiliar content.</p>')
    from app.storage import ArtifactLimitError
    from app.html_sourcepart import pipeline
    original = pipeline.source_json
    def limited(store, value, **kwargs):
        if kwargs['kind'] == 'html_sourcepart_structure' and value['extraction']['mode'] == 'structured':
            raise ArtifactLimitError('structured tree exceeds storage budget')
        return original(store, value, **kwargs)
    with patch('app.html_sourcepart.pipeline.source_json', side_effect=limited):
        page, _ = inventory(workspace, executor)
    assert page['extraction']['mode'] == 'basic_text'
    assert page['extraction']['reasons'][0]['code'] == 'pipeline_structure_adaptation_failed'
    assert commit(workspace, executor).evidence


def test_large_snapshot_survives_json_escaping_and_small_artifact_limit(tmp_path):
    workspace, _ = workspace_for(tmp_path, '<p>Small original.</p>')
    workspace.store.max_artifact_bytes = 20_000
    content = '\\' * 16_000
    artifact = save_snapshot(workspace, {'content':content, 'format':'html', 'complete':True, 'char_count':len(content), 'sha256':extractor.digest(content)},
                             'https://example.org/manual.html', workspace.manifest.raw_artifact_id)
    saved = workspace.store.read_json_artifact(artifact.artifact_id)
    restored = ''.join(workspace.store.read_json_artifact(chunk['artifact_id'])['content'] for chunk in saved['chunks'])
    assert restored == content and saved['complete']


def test_html_cannot_acquire_local_files_outside_its_source_directory(tmp_path, monkeypatch):
    original = tmp_path/'documents'
    original.mkdir()
    outside = tmp_path/'private.css'
    outside.write_text('body {display:none}', encoding='utf-8')
    source = f'<link rel="stylesheet" href="{outside.as_uri()}"><h1>Manual</h1><p>Readable content.</p>'
    workspace, executor = workspace_for(original, source)
    original_read = workspace.acquisition._read_url
    def read_url(url, **limits):
        assert url != outside.as_uri(), 'Unexpected out-of-scope read'
        return original_read(url, **limits)
    monkeypatch.setattr(workspace.acquisition, '_read_url', read_url)
    page, document = inventory(workspace, executor)
    assert page['extraction']['mode'] == 'structured'
    assert document['stylesheets'][0]['status'] == 'missing'
    assert 'outside' in document['stylesheets'][0]['reason']
    assert 'Readable content.' in '\n'.join(workspace.part_text.values())


def test_octet_stream_image_is_identified_without_changing_its_bytes(tmp_path, monkeypatch):
    data = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 4, 4), False).tobytes('png')
    workspace, executor = workspace_for(tmp_path, '<h1>Manual</h1><img src="https://example.org/diagram">')
    monkeypatch.setattr(workspace.acquisition, '_read_url', lambda url, **_limits: FetchResult(data, 'application/octet-stream', url, 200, {}))
    page, document = inventory(workspace, executor)
    assert page['saved_image_count'] == 1
    image = document['images']['I1']
    assert image['media_type'] == 'image/png'
    assert workspace.store.artifact_path(image['artifact_id']).read_bytes() == data
    assert commit(workspace, executor).evidence[0].image_refs[0].media_type == 'image/png'


def test_reference_table_keeps_literal_operator_names(tmp_path):
    source = '<h1>Operators</h1><table><tr><th>Operator</th><th>Meaning</th></tr>'
    source += ''.join(f'<tr><td><code>{name}</code></td><td>{meaning}</td></tr>' for name, meaning in
                      [('*','Multiply'),('+','Add'),('|','Alternative'),('^','Power')]) + '</table>'
    workspace, executor = workspace_for(tmp_path, source)
    inventory(workspace, executor)
    result = commit(workspace, executor)
    entries = [entry for evidence in result.evidence for entry in _inline_reference_entries(evidence)]
    assert [e['title'] for e in entries] == ['*', '+', '|', '^']


@pytest.mark.parametrize("html,role", [
    ('<h2>About <span class="permalink-widget inline"><a href="find/?name=about">link</a></span></h2>', 'heading_permalink'),
    ('<h2>About <a title="Permalink" href="lookup/?name=about">link</a></h2>', 'heading_permalink'),
    ('<h2>About <a class="headerlink" href="#about">#</a></h2>', 'heading_permalink'),
    ('<h2>About <a rel="bookmark" href="section.html#about">link</a></h2>', 'heading_permalink'),
    ('<p>See <a href="chapter2/">chapter 2</a> for more details.</p>', 'cross_reference'),
    ('<ul><li>See <a href="chapter2/">chapter 2</a> for more details.</li></ul>', 'cross_reference'),
    ('<nav><a rel="next" href="chapter2/">Next</a></nav>', 'page_navigation'),
    ('<ol class="section-toc"><li><a href="find/">Finding data</a></li></ol>', 'directory_entry'),
    ('<main><h1>Contents</h1><a href="topic.html">Topic</a></main>', 'directory_entry'),
    ('<main><table><tr><td><h3>Display</h3><a href="display.html">Commands</a></td></tr></table></main>', 'directory_entry'),
    ('<dl><dt><a href="topic.html">Topic</a></dt><dd>A topic description.</dd></dl>', 'directory_entry'),
    ('<ul><li><a href="topic.html">Topic</a><ul><li><a href="sub.html">Subtopic</a></li></ul></li></ul>', 'directory_entry'),
    ('<p><a href="topic.html">Unexplained link</a></p>', 'unclassified'),
    ('<details><summary>Note</summary><p>See <a href="topic.html">another topic</a>.</p></details>', 'cross_reference'),
])
def test_link_roles_use_dom_evidence_instead_of_url_names(html, role):
    soup = BeautifulSoup(html, "html.parser")
    assert classify_html_link(soup.find("a"))[0] == role


@pytest.mark.parametrize("head,page_url,expected", [
    ('<base href="./../">', 'https://example.test/book/Introduction/', 'https://example.test/book/'),
    ('<base href="/manual/">', 'https://example.test/book/chapter.html', 'https://example.test/manual/'),
    ('<base href="https://cdn.test/manual/">', 'https://example.test/book/', 'https://cdn.test/manual/'),
    ('<base target="_blank"><base href="../"><base href="/wrong/">', 'https://example.test/book/chapter/', 'https://example.test/book/'),
    ('', 'file:///C:/manual/index.html', 'file:///C:/manual/index.html'),
])
def test_base_resolution_obeys_first_href_and_final_page_url(head, page_url, expected):
    soup = BeautifulSoup(f'<html><head>{head}</head></html>', 'html.parser')
    assert html_base_url(soup, page_url) == expected


def test_unmarked_linked_paragraph_does_not_turn_later_links_into_children():
    soup = BeautifulSoup('<p><a href="a.html">A</a>, <a href="b.html">B</a></p>', 'html.parser')
    assert all(classify_html_link(anchor)[0] == 'unclassified' for anchor in soup.find_all('a'))
