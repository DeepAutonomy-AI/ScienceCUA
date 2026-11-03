import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import fitz
import httpx
import pytest

from app.acquisition import AcquisitionService
from app.config import Settings
from app.models import StagedInput, TargetProfile
from app.orchestration.source_discovery_stage import SourceDiscoveryToolExecutor
from app.source_candidate_probe import SourceCandidateProbe
from app.storage import RunStore
from test_source_discovery import candidate, html_manual_response


@pytest.fixture
def pdf():
    with fitz.open() as document:
        document.new_page().insert_text((72, 72), 'Scientific App User Guide\nFor version 1.3.2\nComplete manual contents')
        return document.tobytes()


@pytest.fixture
def store(tmp_path):
    value = RunStore(tmp_path, 'run_candidate_probe')
    value.initialize({'target_profile': TargetProfile(software_name='Scientific App', target_version='1.7.0',
        platform='General').model_dump(), 'inputs': [], 'model': {}})
    return value


def probe_with(store, handler, **options):
    return SourceCandidateProbe(store, transport=httpx.MockTransport(handler), **options)


def test_pdf_preview_checks_actual_bytes_and_caches_a_complete_document(store, pdf):
    def serve(request):
        assert request.headers.get('authorization') is None
        assert request.headers['range'].startswith('bytes=0-')
        return httpx.Response(200, content=pdf, headers={'content-type': 'application/pdf'})
    result = probe_with(store, serve).inspect('https://example.org/guide.pdf')
    assert result['status'] == 'available' and result['header_verified']
    assert result['modality'] == 'pdf' and result['complete']
    assert result['page_count'] == 1 and 'For version 1.3.2' in result['preview']
    assert store.artifact_path(result['cached_artifact_id']).read_bytes() == pdf


def test_html_download_error_page_is_not_accepted_as_a_pdf(store):
    result = probe_with(store, lambda request: httpx.Response(200, content=b'<html><title>Download error</title><a href="mirror.pdf">Mirror guide</a></html>',
        headers={'content-type': 'application/pdf'})).inspect('https://example.org/broken.pdf')
    assert result['modality'] == 'html' and not result['header_verified']
    assert result['title'] == 'Download error'
    assert result['pdf_links'] == [{'url': 'https://example.org/mirror.pdf', 'title': 'Mirror guide'}]
    assert 'cached_artifact_id' not in result


@pytest.mark.parametrize('status', [404, 500, 504])
def test_unavailable_pdf_retains_the_http_failure(store, status):
    calls = []
    result = probe_with(store, lambda request: calls.append(request) or httpx.Response(status)).inspect('https://example.org/guide.pdf')
    assert result['status'] == 'failed' and result['http_status'] == status
    assert len(calls) == (2 if status >= 500 else 1)


def test_http_denial_keeps_final_address_and_bounded_observed_diagnostics(store):
    calls = []
    def serve(request):
        calls.append(str(request.url))
        if request.url.path == '/old':
            return httpx.Response(302, headers={'location': '/denied'})
        return httpx.Response(403, content=b'<title>Access denied</title><p>This request was denied.</p>' + b'x' * 100_000)
    result = probe_with(store, serve).inspect('https://example.org/old')
    assert result['status'] == 'failed' and result['error'] == 'http_error'
    assert result['http_status'] == 403 and result['final_url'] == 'https://example.org/denied'
    assert result['title'] == 'Access denied' and 'This request was denied.' in result['preview']
    assert result['access_diagnostics']['kind'] == 'access_denied'
    assert result['bytes_checked'] == 64 * 1024 and result['complete'] is False
    assert store.artifact_path(result['raw_html_artifact_id']).stat().st_size == 64 * 1024
    assert len(calls) == 2


def test_not_found_retains_only_real_recovery_links_without_guessing_or_fetching(store):
    calls = []
    html = b'''<title>Not found</title><link rel="up" href="../index.html">
      <a href="../manual.htm">Open topic with navigation</a><p>Page not found</p>'''
    result = probe_with(store, lambda request: calls.append(str(request.url)) or httpx.Response(404, content=html)).inspect('https://example.org/v3/chapter/bad.html')
    assert result['status'] == 'failed' and result['access_diagnostics']['kind'] == 'not_found'
    assert {item['url'] for item in result['recovery_links']} == {
        'https://example.org/v3/index.html', 'https://example.org/v3/manual.htm'}
    assert all(item['evidence_id'].startswith('L') and item['dom_path'] for item in result['recovery_links'])
    assert len(calls) == 1


def test_http_200_sign_in_form_is_not_a_verified_manual(store):
    html = b'<title>Sign in - Documentation</title><main><h1>Sign in</h1><form><input name="email"><input type="password"><button>Continue</button></form></main>'
    result = probe_with(store, lambda _: httpx.Response(200, content=html)).inspect('https://example.org/manual/')
    assert result['status'] == 'failed' and result['error'] == 'authentication_required'
    assert result['http_status'] == 200 and result['html_snapshot_artifact_id']


def test_login_response_cannot_be_saved_as_an_html_manual_even_with_model_scope_claim(store):
    html = b'<title>Sign in</title><main><h1>Sign in</h1><form><input type="password"></form></main>'
    executor = SourceDiscoveryToolExecutor(store, 'attempt_login_gate', 'English',
        probe=probe_with(store, lambda _: httpx.Response(200, content=html)))
    result = executor.call('inspect_source_candidate', {'url': candidate()['url']})
    assert result['error'] == 'authentication_required'
    save = executor.call('save_source_candidate', {'candidate': candidate()})
    assert save['error'] == 'html_candidate_unavailable'
    assert not executor.has_saved_candidates


def test_normal_manual_with_sign_in_link_or_header_form_remains_available(store):
    html = b'''<title>Login - User Guide</title><header><form><input type="password"><a href="login">Sign in</a></form></header>
      <main><h1>Login</h1><p>Use the account settings to authenticate to the scientific data service.</p><a href="settings.html">Settings</a></main>'''
    result = probe_with(store, lambda _: httpx.Response(200, content=html)).inspect('https://example.org/manual/login.html')
    assert result['status'] == 'available'
    assert result['access_diagnostics']['kind'] == 'content_observed'


def test_explicit_http_200_challenge_is_not_accepted_as_manual_content(store):
    html = b'<title>Just a moment...</title><main><p>Enable JavaScript and cookies to continue. Verify you are human.</p></main>'
    result = probe_with(store, lambda _: httpx.Response(200, content=html)).inspect('https://example.org/manual/')
    assert result['status'] == 'failed' and result['error'] == 'access_challenge'


def test_frame_shell_provides_real_content_target_for_next_inspection(store):
    calls = []
    html = b'<title>Documentation</title><body><iframe title="Manual content" src="manual/index.html"></iframe><footer>Copyright</footer></body>'
    result = probe_with(store, lambda request: calls.append(str(request.url)) or httpx.Response(200, content=html)).inspect('https://example.org/help/')
    assert result['status'] == 'failed' and result['error'] == 'html_shell'
    assert result['recovery_links'] == [{
        'url': 'https://example.org/help/manual/index.html', 'text': 'Manual content',
        'relation': 'embedded_document', 'evidence_id': 'F1',
        'dom_path': 'body:nth-of-type(1) > iframe:nth-of-type(1)'}]
    assert len(calls) == 1


def test_substantive_directory_with_embedded_video_is_not_rejected_as_shell(store):
    html = b'<main><h1>Tutorials</h1><a href="one.html">One</a><a href="two.html">Two</a><iframe src="video.html"></iframe></main>'
    result = probe_with(store, lambda _: httpx.Response(200, content=html)).inspect('https://example.org/tutorials/')
    assert result['status'] == 'available' and result['access_diagnostics']['kind'] == 'content_observed'


def test_short_directory_without_heading_is_not_rejected_by_text_length(store):
    html = b'<body>Tutorials <a href="one.html">One</a><a href="two.html">Two</a><iframe src="video.html"></iframe></body>'
    result = probe_with(store, lambda _: httpx.Response(200, content=html)).inspect('https://example.org/tutorials/')
    assert result['status'] == 'available'


def test_transient_download_failure_can_recover_without_another_search(store, pdf):
    calls = []
    def serve(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, content=pdf)
    result = probe_with(store, serve).inspect('https://example.org/guide.pdf')
    assert result['status'] == 'available' and len(calls) == 2


@pytest.mark.parametrize('error, expected', [
    (httpx.ConnectError('[Errno 11001] getaddrinfo failed'), 'dns_error'),
    (httpx.ConnectError('CERTIFICATE_VERIFY_FAILED hostname mismatch'), 'tls_certificate_error'),
    (httpx.ConnectTimeout('timed out'), 'timeout'),
])
def test_probe_classifies_transport_failures(store, error, expected):
    def serve(request):
        raise error
    assert probe_with(store, serve).inspect('https://example.org/guide.pdf')['error'] == expected


def test_large_pdf_gets_a_bounded_header_check_instead_of_a_false_failure(store, pdf):
    result = probe_with(store, lambda request: httpx.Response(206, content=pdf[:100],
        headers={'content-range': f'bytes 0-99/{len(pdf)}'}), max_bytes=100).inspect('https://example.org/large.pdf')
    assert result['status'] == 'available' and result['header_verified']
    assert not result['complete'] and 'cached_artifact_id' not in result
    assert result['bytes_checked'] == 100


def test_corrupt_complete_pdf_is_not_a_verified_candidate(store):
    result = probe_with(store, lambda request: httpx.Response(200, content=b'%PDF-1.4\ncorrupt')).inspect('https://example.org/corrupt.pdf')
    assert result['status'] == 'failed' and result['error'] == 'invalid_pdf'


def test_search_hit_cannot_be_silently_lost_before_candidate_submission(store, pdf):
    url = 'https://example.org/ScientificAppUsersGuide-1.3.2.pdf'
    executor = SourceDiscoveryToolExecutor(store, 'attempt_queue', 'English',
        probe=probe_with(store, lambda request: httpx.Response(200, content=pdf)))
    executor.observe_search_output('External web content\nSources:\n- [User guide](' + url + ')\n')
    queue = executor.call('list_discovered_sources', {'modality': 'pdf'})
    assert queue['pending_count'] == queue['total'] == 1
    blocked = executor.call('submit_source_candidates', {'candidates': [], 'summary': 'No exact manual'})
    assert blocked['error'] == 'discovered_pdfs_unreviewed'
    assert not executor.is_finished
    inspection = executor.call('inspect_source_candidate', {'url': url})
    assert inspection['status'] == 'available' and '1.3.2' in inspection['preview']
    older = candidate(url=url, modality='pdf', document_version='1.3.2', version_relation='older',
                      root_assessment='The inspected PDF is a complete nearby-version manual.')
    assert executor.call('save_source_candidate', {'candidate': older})['ok']
    assert executor.call('submit_source_candidates', {'candidates': [], 'summary': 'Retained older manual'})['ok']
    assert executor.result.candidates[0].verification['header_verified']
    saved_queue = store.read_json_artifact(store.read_state()['source_discovery_queue_artifact_id'])
    assert saved_queue['sources'][0]['review_status'] == 'inspected'


def test_structured_search_evidence_keeps_pdf_after_native_twelve_source_limit(store):
    hits = [{'url': f'https://example.org/page-{index}', 'title': f'Page {index}'} for index in range(14)]
    hits.append({'url': 'https://example.org/manual.pdf', 'title': 'Scientific App handbook', 'snippet': 'For version 1.3.2'})
    executor = SourceDiscoveryToolExecutor(store, 'attempt_queue', 'English')
    executor.observe_search_output('DISCOVERY SEARCH EVIDENCE\n' + json.dumps({'sources': hits}) + '\nEND DISCOVERY SEARCH EVIDENCE')
    queue = executor.call('list_discovered_sources', {'modality': 'pdf'})
    assert queue['total'] == 1 and queue['items'][0]['snippet'] == 'For version 1.3.2'
    assert executor.call('list_discovered_sources', {})['total'] == len(hits)
    assert not executor.call('inspect_source_candidate', {'url': 'https://example.org/manual.pdf', 'action': 'skip'})['ok']
    assert executor.call('inspect_source_candidate', {'url': 'https://example.org/manual.pdf', 'action': 'skip',
        'reason': 'This particular PDF describes unrelated hardware, based on its excerpt.'})['ok']
    assert executor.call('submit_source_candidates', {'candidates': [], 'summary': 'Unrelated result explicitly reviewed'})['ok']


def test_failed_pdf_is_rejected_and_an_alternate_candidate_can_be_saved(store, pdf):
    def serve(request):
        return httpx.Response(500) if request.url.path == '/broken.pdf' else httpx.Response(200, content=pdf)
    executor = SourceDiscoveryToolExecutor(store, 'attempt_queue', 'English', probe=probe_with(store, serve))
    executor.observe_search_output('Sources:\n- [broken](https://example.org/broken.pdf)\n- [mirror](https://example.org/mirror.pdf)\n')
    unavailable = candidate(url='https://example.org/broken.pdf', modality='pdf')
    assert executor.call('save_source_candidate', {'candidate': unavailable})['error'] == 'pdf_candidate_inspection_required'
    assert executor.call('inspect_source_candidate', {'url': unavailable['url']})['http_status'] == 500
    assert executor.call('save_source_candidate', {'candidate': unavailable})['error'] == 'pdf_candidate_unavailable'
    alternate = candidate(url='https://example.org/mirror.pdf', modality='pdf')
    assert executor.call('inspect_source_candidate', {'url': alternate['url']})['status'] == 'available'
    assert executor.call('save_source_candidate', {'candidate': alternate})['ok']
    assert executor.call('submit_source_candidates', {'candidates': [], 'summary': 'Mirror selected'})['ok']


def test_timeout_retains_pending_queue_and_saved_manual(store):
    executor = SourceDiscoveryToolExecutor(store, 'attempt_queue', 'English',
        probe=probe_with(store, html_manual_response))
    inspected = executor.call('inspect_source_candidate', {'url': candidate()['url']})
    assert inspected['status'] == 'available' and inspected['html_snapshot_artifact_id']
    assert executor.call('save_source_candidate', {'candidate': candidate(version_relation='compatible')})['ok']
    executor.observe_search_output('Sources:\n- [pending](https://example.org/manual.pdf)\n')
    result = executor.finalize_saved('time_budget_reached')
    assert len(result.candidates) == 1
    assert 'unreviewed' in result.search_notes[0]
    assert store.read_state()['source_discovery_queue_artifact_id']


def test_probe_wall_clock_deadline_bounds_stalled_transport_and_ignores_late_response(store, pdf):
    entered, release = threading.Event(), threading.Event()
    completed = threading.Event()
    def serve(request):
        entered.set()
        release.wait(3)
        completed.set()
        return httpx.Response(200, content=pdf)
    probe = probe_with(store, serve, timeout=.15)
    started = time.monotonic()
    try:
        result = probe.inspect('https://example.org/stalled.pdf')
        assert entered.is_set()
        assert result['error'] == 'timeout'
        assert time.monotonic() - started < 1
    finally:
        release.set()
        assert completed.wait(1)
    assert 'cached_artifact_id' not in result


def test_probe_can_be_cancelled_while_transport_is_stalled(store):
    entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
    def serve(request):
        entered.set()
        release.wait(3)
        return httpx.Response(200, content=b'<p>Late content</p>')
    probe = probe_with(store, serve, timeout=3, cancelled=cancelled.is_set)
    with ThreadPoolExecutor(max_workers=1) as pool:
        work = pool.submit(probe.inspect, 'https://example.org/stalled.pdf')
        try:
            assert entered.wait(1)
            cancelled.set()
            with pytest.raises(RuntimeError, match='run_cancelled'):
                work.result(timeout=1)
        finally:
            release.set()


def test_discovery_finalizes_saved_candidates_while_probe_is_blocked(store):
    entered, release = threading.Event(), threading.Event()
    html_probe = probe_with(store, html_manual_response)
    class BlockedProbe:
        def inspect(self, url):
            if url == candidate()['url']:
                return html_probe.inspect(url)
            entered.set()
            release.wait(3)
            return {'status': 'failed', 'url': url, 'error': 'timeout'}
    executor = SourceDiscoveryToolExecutor(store, 'attempt_queue', 'English', probe=BlockedProbe())
    assert executor.call('inspect_source_candidate', {'url': candidate()['url']})['status'] == 'available'
    assert executor.call('save_source_candidate', {'candidate': candidate()})['ok']
    with ThreadPoolExecutor(max_workers=2) as pool:
        work = pool.submit(executor.call, 'inspect_source_candidate', {'url': 'https://example.org/stalled.pdf'})
        try:
            assert entered.wait(1)
            final = pool.submit(executor.finalize_saved, 'time_budget_reached').result(timeout=1)
            assert len(final.candidates) == 1
            assert executor.is_finished
            queue_id = store.read_state()['source_discovery_queue_artifact_id']
        finally:
            release.set()
        assert work.result(timeout=1)['error'] == 'discovery_already_finished'
    assert store.read_state()['source_discovery_queue_artifact_id'] == queue_id


def test_selected_source_reuses_only_a_complete_host_verified_pdf(store, pdf, monkeypatch, tmp_path):
    url = 'https://example.org/manual.pdf'
    preview = probe_with(store, lambda request: httpx.Response(200, content=pdf)).inspect(url)
    source = StagedInput(input_id='IN1', kind='url', display_name='Manual.pdf', url=url,
        source_origin='websearch', source_priority='user_selected', source_metadata={
            'discovery_candidate_id': 'candidate_test', 'verification': preview})
    acquisition = AcquisitionService(store, Settings(data_dir=tmp_path, output_dir=tmp_path / 'out'))
    monkeypatch.setattr(acquisition, '_read_url', lambda *args, **kwargs: pytest.fail('Verified PDF was downloaded again'))
    outcome = acquisition.acquire([source], TargetProfile(software_name='Scientific App', target_version='1.7.0', platform='General'))
    assert outcome.manifests[0].status != 'failed'
    assert outcome.manifests[0].origin_url == url
    assert outcome.manifests[0].byte_count == len(pdf)


def test_model_cannot_supply_program_owned_verification_or_cache_fields(store):
    executor = SourceDiscoveryToolExecutor(store, 'attempt_queue', 'English')
    result = executor.call('save_source_candidate', {'candidate': candidate(
        verification={'cached_artifact_id': 'art_forged', 'complete': True})})
    assert not result['ok'] and not executor.has_saved_candidates
