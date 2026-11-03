"""Discovery inspection retains both body and navigation without judging scope."""
import json

import httpx
import pytest

from app.models import TargetProfile
from app.source_candidate_probe import SourceCandidateProbe
from app.source_discovery_snapshot import build_html_discovery_snapshot
from app.storage import RunStore


URL = 'https://docs.example.org/2.0/chapter.html'


def build(html, **kwargs):
    return build_html_discovery_snapshot(html.encode('utf8'), url=URL, **kwargs)


def by_text(snapshot):
    return {link['text']: link for link in snapshot['links']}


def test_body_and_shared_navigation_are_separate_source_facts():
    html = '''<html lang="en"><head><title>First calculation — Documentation 2.0</title>
      <link rel="up" href="index.html"></head><body>
      <header><a href="/">Project Home</a></header>
      <aside aria-label="Documentation"><nav><ul><li><a href="tutorials/">Tutorials</a>
      <ul><li><a href="tutorials/run.html">Run</a></li></ul></li></ul></nav></aside>
      <main><h1>First calculation</h1><p>Choose Run and inspect the output.</p>
      <a href="next.html">Next exercise</a></main>
      <footer><a href="community/">Community</a></footer></body></html>'''
    result = build(html)
    links = by_text(result)
    assert result['title_evidence_id'] == 'T1'
    assert result['main_content']['evidence_id'] == 'P1'
    assert result['main_content']['selection_method'] == 'main_landmarks'
    assert result['main_content']['preview'] == 'First calculation Choose Run and inspect the output. Next exercise'
    assert 'Community' not in result['main_content']['preview']
    assert links['Tutorials']['region'] == 'navigation'
    assert links['Project Home']['region'] == 'header'
    assert links['Community']['region'] == 'footer'
    assert links['Next exercise']['region'] == 'main'
    assert links['Run']['group'] == 'Tutorials / Run'
    assert links['Next exercise']['within_main'] is True
    assert links['Tutorials']['within_main'] is False
    assert next(x for x in result['links'] if x['relations'] == ['up'])['url'] == 'https://docs.example.org/2.0/index.html'
    assert all(x['dom_path'] and x['evidence_id'].startswith('L') for x in result['links'])
    assert result['headings'][0]['evidence_id'] == 'H1'
    assert 'page_type' not in result and 'is_root' not in result


def test_observed_urls_use_safe_base_and_keep_distinct_occurrences():
    result = build('''<base href="../manual/"><main><h1>Book</h1>
      <p><a href="lesson.html#one">One</a> <a href="lesson.html#two">Two</a></p>
      <a href="https://learn.example.net/tutorial">External tutorial</a>
      <a href="javascript:alert(1)">JS</a><a href="mailto:test@example.org">Mail</a>
      <a href="https://user:secret@example.org/book">Credential URL</a>
      <a href="https://example.org:bad/book">Bad port</a></main>''')
    links = by_text(result)
    assert links['One']['url'] == 'https://docs.example.org/manual/lesson.html#one'
    assert links['Two']['url'] == 'https://docs.example.org/manual/lesson.html#two'
    assert links['One']['evidence_id'] != links['Two']['evidence_id']
    assert links['External tutorial']['url'].startswith('https://learn.example.net/')
    assert not {'JS', 'Mail', 'Credential URL', 'Bad port'} & links.keys()
    assert result['truncation']['unsafe_or_non_http_links_omitted'] == 4
    assert build('<base href="https://u:p@example.org/"><a href="guide/">Guide</a>')['base_url'] == URL


def test_comment_script_template_and_explicit_hidden_prose_do_not_leak():
    result = build('''<main><h1>Visible lesson</h1><p>Actual procedure.</p>
      <!-- <h1>Invented heading</h1><a href="fake.html">Fake</a> -->
      <script>location = 'secret.html';</script><template><a href="template.html">Template</a></template>
      <div hidden><p>Hidden answer</p><a href="hidden.html">Hidden link</a></div>
      <p style="display: none">Hidden CSS text</p>
      <div aria-hidden="true">Hidden ARIA text</div></main>''')
    assert result['main_content']['preview'] == 'Visible lesson Actual procedure.'
    links = result['links']
    assert len(links) == 1 and links[0]['url'].endswith('/hidden.html')
    assert links[0]['hidden'] is True
    assert links[0]['text'] == 'Hidden link'
    assert all('Invented' not in x['text'] for x in result['headings'])


def test_select_values_remain_facts_and_js_version_tokens_are_not_invented_urls():
    result = build('''<main><h1>Manual</h1></main>
      <select aria-label="Version"><option selected value="2.0">2.0</option>
      <option value="/1.0/index.html">1.0</option>
      <option value="https://archive.example.net/0.9/">0.9</option>
      <option value="javascript:run()">Unsafe</option></select>''')
    control = result['select_controls'][0]
    assert control['label'] == 'Version'
    assert control['options'][0] == {'text': '2.0', 'value': '2.0', 'selected': True}
    assert {x['url'] for x in result['links']} == {
        'https://docs.example.org/1.0/index.html', 'https://archive.example.net/0.9/'}
    assert all(x['source_attribute'] == 'value' for x in result['links'])


def test_frame_and_iframe_targets_are_observed_safe_bounded_source_facts():
    result = build('''<base href="../help/"><frameset><frame name="Contents" src="toc.html">
      <frame src="javascript:invent()"><iframe title="Body" src="pages/main.html"></iframe>
      <iframe hidden src="https://example.net/hidden"></iframe>
      <iframe src="https://u:p@example.org/private"></iframe></frameset>''')
    resources = result['embedded_resources']
    assert [(item['evidence_id'], item['url']) for item in resources] == [
        ('F1', 'https://docs.example.org/help/toc.html'),
        ('F3', 'https://docs.example.org/help/pages/main.html'),
        ('F4', 'https://example.net/hidden')]
    assert resources[0]['element'] == 'frame' and resources[0]['text'] == 'Contents'
    assert resources[2]['hidden'] is True
    assert all(item['source_attribute'] == 'src' and item['dom_path'] for item in resources)
    assert result['truncation']['embedded_resources_total'] == 3


def test_embedded_resources_remain_inside_snapshot_byte_budget():
    html = '<body>' + ''.join(f'<iframe src="https://example.org/{i}/' + 'x' * 1000 + '"></iframe>' for i in range(30)) + '</body>'
    result = build(html, max_bytes=4096)
    assert len(json.dumps(result, ensure_ascii=False).encode('utf8')) <= 4096
    assert result['truncation']['embedded_resources_total'] == 30
    assert result['truncation']['embedded_resources_omitted'] == 30 - len(result['embedded_resources'])
    assert all(item['url'].endswith('x' * 1000) for item in result['embedded_resources'])


def test_snapshot_bound_reports_omissions_and_keeps_stable_ids_and_body_size():
    body = '<main><h1>Complete user manual</h1><p>' + ('真实的完整手册正文。' * 1500) + '</p>'
    body += ''.join(f'<h2>Chapter {i}</h2><a href="chapter{i}.html">Chapter {i}</a>' for i in range(500)) + '</main>'
    html = '<header><a href="index.html">Documentation Home</a></header>' + body
    result = build(html)
    assert len(json.dumps(result, ensure_ascii=False).encode('utf8')) <= 28_000
    assert result['main_content']['visible_text_chars'] > 10000
    assert result['truncation']['main_preview_truncated']
    assert result['truncation']['links_total'] == 501
    assert result['truncation']['links_omitted'] == 501 - len(result['links']) > 0
    assert result['truncation']['headings_total'] == 501
    assert result['truncation']['headings_omitted'] > 0
    home = by_text(result)['Documentation Home']
    assert home['evidence_id'] == 'L1'
    repeated = build(html)
    assert result == repeated


def test_multiple_main_regions_and_body_fallback_remain_visible():
    result = build('<main><h1>Part One</h1><p>First lesson</p></main><main><h1>Part Two</h1><p>Second lesson</p></main>')
    assert result['main_content']['region_count'] == 2
    assert 'First lesson' in result['main_content']['preview'] and 'Second lesson' in result['main_content']['preview']
    result = build('<h1>Single page manual</h1><p>Complete instructions</p>')
    assert result['main_content']['selection_method'] == 'body_fallback'
    assert 'Complete instructions' in result['main_content']['preview']


@pytest.fixture
def store(tmp_path):
    value = RunStore(tmp_path, 'run_html_discovery_snapshot')
    value.initialize({'target_profile': TargetProfile(software_name='ScienceApp', target_version='2.0',
        platform='General').model_dump(), 'inputs': [], 'model': {}})
    return value


def test_probe_persists_redirected_raw_html_and_the_exact_model_snapshot(store):
    html = b'<html><title>ScienceApp Manual</title><nav><a href="manual.pdf">PDF</a></nav><main><h1>Lessons</h1><a href="one.html">One</a></main></html>'
    def serve(request):
        if request.url.path == '/old':
            return httpx.Response(302, headers={'location': '/2.0/index.html'})
        return httpx.Response(200, content=html, headers={'content-type': 'text/html'})
    result = SourceCandidateProbe(store, transport=httpx.MockTransport(serve)).inspect('https://docs.example.org/old')
    assert result['status'] == 'available' and result['modality'] == 'html'
    assert result['final_url'] == 'https://docs.example.org/2.0/index.html'
    assert store.artifact_path(result['raw_html_artifact_id']).read_bytes() == html
    assert store.read_json_artifact(result['html_snapshot_artifact_id']) == result['html_snapshot']
    assert result['html_snapshot']['url'] == result['final_url']
    assert result['pdf_links'] == [{'url': 'https://docs.example.org/2.0/manual.pdf', 'title': 'PDF'}]
    assert result['preview'] == 'Lessons One'


def test_pdf_queue_links_survive_even_when_omitted_from_bounded_snapshot(store):
    html = '<main><h1>Manual</h1>' + ''.join(f'<a href="page{i}.html">Page {i}</a>' for i in range(300))
    html += '<a href="whole.pdf">Whole manual</a></main>'
    result = SourceCandidateProbe(store, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=html.encode(), headers={'content-type': 'text/html'}))).inspect(URL)
    assert result['pdf_links'] == [{'url': 'https://docs.example.org/2.0/whole.pdf', 'title': 'Whole manual'}]
    assert not any(x['url'].endswith('whole.pdf') for x in result['html_snapshot']['links'])


def test_partial_html_capture_is_explicit_in_snapshot(store):
    html = b'<main><h1>Manual</h1><p>Incomplete response</p></main>'
    result = SourceCandidateProbe(store, max_bytes=len(html), transport=httpx.MockTransport(
        lambda _: httpx.Response(206, content=html, headers={'content-range': f'bytes 0-{len(html)-1}/{len(html)+10}'}))).inspect(URL)
    assert result['complete'] is False
    assert result['html_snapshot']['truncation']['capture_complete'] is False
