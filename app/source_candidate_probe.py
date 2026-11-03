"""Bounded transport checks and source previews for discovery candidates."""
from __future__ import annotations

import re
import ssl
import queue
import threading
import time
from urllib.parse import urlsplit
from typing import Any, Callable

import pymupdf
import httpx

from .source_discovery_contracts import SourceCandidateDraft
from .source_discovery_snapshot import build_html_discovery_snapshot
from .storage import RunStore


def _html_access_diagnostics(snapshot: dict[str, Any], http_status: int) -> dict[str, Any]:
    """Recognize explicit access gates, not the suitability of a manual's scope."""
    title = snapshot.get('title', '')
    preview = snapshot.get('main_content', {}).get('preview', '')
    chars = snapshot.get('main_content', {}).get('visible_text_chars', 0)
    headings = [item.get('text', '') for item in snapshot.get('headings', [])
                if not item.get('hidden') and item.get('level') == 1]
    signals = snapshot.get('access_signals', {})
    login_identity = any(re.search(r'^(?:(?:customer|member|user|account)\s+)?(?:sign[ -]?in|log[ -]?in|authentication required)(?:$|\s*[-|:–—])', text.strip(), re.I)
                         for text in [title, *headings])
    # A login link in an otherwise useful manual, or a chapter teaching login,
    # is not an access gate. Require an actual visible password form as well.
    if (signals.get('main_password_inputs', 0) and signals.get('visible_forms', 0)
            and login_identity and chars < 1200):
        return {'kind': 'authentication_required', 'evidence': ['visible_password_form', 'login_page_identity'],
                'instruction': 'This response is a sign-in page, not the requested manual. Look for a public official documentation entry or ask the user to provide material.'}
    challenge_identity = re.search(r'^(?:just a moment|attention required|security verification|verify you are human|checking your browser)\b', title.strip(), re.I)
    challenge_body = re.search(r'verify (?:that )?you are human|checking your browser|enable javascript and cookies|security verification|captcha|cloudflare', preview, re.I)
    if challenge_identity and challenge_body and chars < 1600:
        return {'kind': 'access_challenge', 'evidence': ['challenge_page_title', 'challenge_page_body'],
                'instruction': 'The response is an access challenge. Do not treat it as documentation or keep retrying it; inspect other public official entries.'}
    if http_status == 404:
        return {'kind': 'not_found', 'evidence': ['http_404'],
                'instruction': 'This exact address was not found. Use observed official navigation, directory or version links; do not infer that the software has no manual.'}
    if http_status in {401, 403}:
        return {'kind': 'access_denied', 'evidence': [f'http_{http_status}'],
                'instruction': 'This request was denied. The status alone does not establish whether the cause is authentication, an access policy or automated-request protection.'}
    if http_status >= 400:
        return {'kind': 'http_error', 'evidence': [f'http_{http_status}'],
                'instruction': 'The request failed. Preserve its HTTP status and inspect other observed official links; this response is not verified documentation.'}
    frames = [item for item in snapshot.get('embedded_resources', []) if not item.get('hidden')]
    links = [item for item in snapshot.get('links', []) if not item.get('hidden')]
    # Only a clear frame wrapper is rejected. Short directories and ordinary
    # SPA responses remain model decisions; text length alone is insufficient.
    if frames and chars < 160 and all(item.get('region') == 'footer' for item in links) and not any(
            item.get('within_main') and not item.get('hidden') for item in snapshot.get('headings', [])):
        return {'kind': 'html_shell', 'evidence': ['visible_embedded_document', 'minimal_wrapper_text'],
                'instruction': 'Only a thin frame wrapper was retrieved. Inspect its observed embedded document URL before selecting a learning entry; the wrapper is not verified manual content.'}
    return {'kind': 'content_observed', 'evidence': ['html_response'],
            'instruction': 'Judge the actual body, navigation and collection scope. A successful HTTP response alone does not establish a suitable manual.'}


def _recovery_links(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Bounded real DOM targets; never synthesize parent or release paths."""
    candidates = []
    for item in snapshot.get('embedded_resources', []):
        if not item.get('hidden'):
            candidates.append({**item, 'relation': 'embedded_document'})
    for item in snapshot.get('links', []):
        relations = set(item.get('relations', [])) & {'home', 'up', 'index', 'contents', 'canonical'}
        explicit_navigation = re.search(r'open\s+(?:topic\s+)?with\s+navigation', item.get('text', ''), re.I)
        if not item.get('hidden') and (relations or explicit_navigation):
            candidates.append({**item, 'relation': sorted(relations)[0] if relations else 'topic_navigation'})
    result, seen = [], {snapshot.get('url')}
    for item in candidates:
        if item['url'] in seen:
            continue
        seen.add(item['url'])
        result.append({key: item.get(key, '') for key in ('url', 'text', 'relation', 'evidence_id', 'dom_path')})
        if len(result) >= 12:
            break
    return result


class SourceCandidateProbe:
    def __init__(self, store: RunStore, *, timeout: float = 12,
                 max_bytes: int = 8 * 1024 * 1024,
                 transport: httpx.BaseTransport | None = None,
                 cancelled: Callable[[], bool] | None = None,
                 pdf_timeout: float | None = None, transient_retries: int = 0):
        self.store, self.timeout, self.max_bytes = store, timeout, max_bytes
        self.transport, self.cancelled = transport, cancelled
        self.pdf_timeout = timeout if pdf_timeout is None else pdf_timeout
        self.transient_retries = min(1, max(0, transient_retries))

    def inspect(self, url: str) -> dict[str, Any]:
        url = SourceCandidateDraft.validate_url(url)
        for attempt in range(self.transient_retries + 1):
            self._check_cancelled()
            result = self._inspect_once(url)
            result['attempt_count'] = attempt + 1
            if result.get('error') not in {'timeout', 'connection_error', 'dns_error'}:
                return result
        return result

    def _inspect_once(self, url: str) -> dict[str, Any]:
        self._check_cancelled()
        started = time.monotonic()
        pdf_response = threading.Event()
        if urlsplit(url).path.lower().endswith('.pdf'):
            pdf_response.set()
        outcomes: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        expired = threading.Event()

        def work() -> None:
            try:
                outcomes.put((True, self._inspect_request(url, expired, pdf_response)))
            except Exception as exc:
                outcomes.put((False, exc))

        # Socket timeouts do not bound OS DNS resolution or an entire response
        # streamed in tiny chunks. A daemon worker gives the caller a wall-clock
        # deadline, including parsing, redirects, retries and DNS.
        threading.Thread(target=work, name="source-candidate-probe", daemon=True).start()
        try:
            while True:
                self._check_cancelled()
                remaining = started + (self.pdf_timeout if pdf_response.is_set() else self.timeout) - time.monotonic()
                if remaining <= 0:
                    return {'status': 'failed', 'url': url, 'error': 'timeout'}
                try:
                    ok, outcome = outcomes.get(timeout=min(.1, remaining))
                except queue.Empty:
                    continue
                if ok:
                    return outcome
                raise outcome
        finally:
            expired.set()

    def _inspect_request(self, url: str, expired: threading.Event, pdf_response: threading.Event) -> dict[str, Any]:
        def check_active() -> None:
            self._check_cancelled()
            if expired.is_set():
                raise httpx.ReadTimeout('Candidate inspection deadline reached')

        try:
            with httpx.Client(follow_redirects=True, timeout=max(self.timeout, self.pdf_timeout), transport=self.transport) as client:
                for attempt in range(2):
                    check_active()
                    with client.stream('GET', url, headers={
                        'User-Agent': 'ModalPreprocessingAgent/0.1',
                        'Accept': 'application/pdf,text/html;q=0.9,*/*;q=0.5',
                        'Range': f'bytes=0-{self.max_bytes - 1}',
                    }) as response:
                        if 'application/pdf' in response.headers.get('content-type', '').lower():
                            pdf_response.set()
                        if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                            continue
                        final_url = SourceCandidateDraft.validate_url(str(response.url))
                        byte_limit = min(self.max_bytes, 64 * 1024) if response.is_error else self.max_bytes
                        chunks, size, complete = [], 0, True
                        for chunk in response.iter_bytes():
                            if b'%PDF-' in chunk[:1024]:
                                pdf_response.set()
                            check_active()
                            remaining = byte_limit - size
                            chunks.append(chunk[:remaining])
                            size += min(len(chunk), remaining)
                            if len(chunk) > remaining:
                                complete = False
                                break
                        data = b''.join(chunks)
                        content_range = response.headers.get('content-range', '')
                        if response.status_code == 206:
                            match = re.fullmatch(r'bytes\s+(\d+)-(\d+)/(\d+)', content_range)
                            complete = complete and bool(match) and int(match[1]) == 0 and int(match[2]) + 1 == int(match[3])
                        result = {'status': 'failed' if response.is_error else 'available', 'url': url, 'final_url': final_url,
                                  'http_status': response.status_code, 'complete': complete,
                                  'bytes_checked': len(data)}
                        if response.is_error:
                            result['error'] = 'http_error'
                        if b'%PDF-' in data[:1024]:
                            if response.is_error:
                                return result
                            result.update(modality='pdf', header_verified=True)
                            if complete:
                                try:
                                    with pymupdf.open(stream=data, filetype='pdf') as document:
                                        if document.needs_pass:
                                            return {**result, 'status': 'failed', 'error': 'encrypted_pdf'}
                                        result.update(page_count=len(document),
                                            preview='\n\n'.join(document[index].get_text() for index in range(min(2, len(document))))[:12000])
                                except Exception:
                                    return {**result, 'status': 'failed', 'error': 'invalid_pdf'}
                                check_active()
                                artifact = self.store.put_artifact(data, kind='discovery_pdf_preview',
                                    media_type='application/pdf', suffix='.pdf', metadata={
                                        'url': url, 'final_url': final_url, 'complete': True,
                                        'http_status': response.status_code,
                                    })
                                result['cached_artifact_id'] = artifact.artifact_id
                            else:
                                result['preview_note'] = 'PDF header verified; preview size limit reached. Version and full document validity remain unverified.'
                            return result
                        links: list[dict[str, str]] = []
                        snapshot = build_html_discovery_snapshot(data, url=final_url,
                            capture_complete=complete, pdf_links_out=links)
                        check_active()
                        raw = self.store.put_artifact(data, kind='discovery_html_capture',
                            media_type='text/html', suffix='.html', metadata={
                                'url': url, 'final_url': final_url, 'complete': complete,
                                'http_status': response.status_code,
                            })
                        check_active()
                        snapshot_artifact = self.store.put_json_artifact(snapshot,
                            kind='discovery_html_snapshot', metadata={
                                'url': url, 'final_url': final_url,
                                'raw_html_artifact_id': raw.artifact_id,
                            }, preserve_text_integrity=True)
                        diagnostics = _html_access_diagnostics(snapshot, response.status_code)
                        if not response.is_error and diagnostics['kind'] in {'authentication_required', 'access_challenge', 'html_shell'}:
                            result.update(status='failed', error=diagnostics['kind'])
                        return {**result, 'modality': 'html', 'header_verified': False,
                                'title': snapshot['title'], 'preview': snapshot['main_content']['preview'],
                                'pdf_links': links[:24], 'html_snapshot': snapshot,
                                'access_diagnostics': diagnostics, 'recovery_links': _recovery_links(snapshot),
                                'raw_html_artifact_id': raw.artifact_id,
                                'html_snapshot_artifact_id': snapshot_artifact.artifact_id}
            raise RuntimeError('Candidate inspection ended without a response')
        except httpx.HTTPStatusError as exc:
            return {'status': 'failed', 'url': url, 'error': 'http_error', 'http_status': exc.response.status_code}
        except httpx.TimeoutException:
            return {'status': 'failed', 'url': url, 'error': 'timeout'}
        except httpx.RequestError as exc:
            cause: BaseException | None = exc
            messages = []
            while cause is not None and len(messages) < 6:
                messages.append(str(cause))
                cause = cause.__cause__ or cause.__context__
            detail = ' '.join(messages)
            error = ('tls_certificate_error' if isinstance(exc.__cause__, ssl.SSLError) or
                     re.search(r'certificate|CERTIFICATE|CERT_ALTNAME', detail) else
                     'dns_error' if re.search(r'getaddrinfo|ENOTFOUND|Name or service not known', detail) else
                     'connection_error')
            return {'status': 'failed', 'url': url, 'error': error}

    def _check_cancelled(self) -> None:
        if self.cancelled and self.cancelled():
            raise RuntimeError('run_cancelled')
