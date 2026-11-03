from __future__ import annotations

import mimetypes
import html as html_lib
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from .config import Settings
from .models import (
    Coverage,
    InputAlias,
    SourceManifest,
    SourceResult,
    StagedInput,
    TargetProfile,
)
from .storage import RunStore


VERSION_PATTERNS = [
    re.compile(r"(?i)\bspecifically\s+v?([0-9]+(?:\.[0-9]+){0,3})\b"),
    re.compile(r"(?i)\bversion\s*[:：]?\s*v?([0-9]+(?:\.[0-9]+){0,3})\b"),
    re.compile(r"版本\s*[:：]?\s*v?([0-9]+(?:\.[0-9]+){0,3})"),
    re.compile(r"(?i)\bv([0-9]+(?:\.[0-9]+){1,3})\b"),
]
SAVED_FROM_URL_PATTERN = re.compile(r"(?is)<!--\s*saved\s+from\s+url=\(\d+\)(.*?)\s*-->")


class AuthenticationRequired(Exception):
    pass


class SourceTooLarge(Exception):
    pass


@dataclass
class FetchResult:
    data: bytes
    media_type: str
    final_url: str
    status_code: int | None
    headers: dict[str, str]


@dataclass
class AcquisitionOutcome:
    manifests: list[SourceManifest]
    initial_results: list[SourceResult]


def detect_document_version(text: str, software_name: str | None = None) -> str:
    sample = text[:200_000]
    for pattern in VERSION_PATTERNS:
        match = pattern.search(sample)
        if match:
            return match.group(1)
    if software_name and software_name.strip():
        software_pattern = re.compile(
            rf"(?i)\b{re.escape(software_name.strip())}\s+(?:version\s*|v\s*)?([0-9]+(?:\.[0-9]+){{0,3}})\b"
        )
        match = software_pattern.search(sample)
        if match:
            return match.group(1)
    return "unknown"


def resolve_version(document_version: str, user_version: str) -> tuple[str, str, bool]:
    document = (document_version or "unknown").strip()
    user = (user_version or "unknown").strip()
    if document.lower() != "unknown":
        conflict = user.lower() != "unknown" and document.casefold() != user.casefold()
        return document, "document", conflict
    if user.lower() != "unknown":
        return user, "user", False
    return "unknown", "unknown", False


class AcquisitionService:
    def __init__(self, store: RunStore, settings: Settings):
        self.store = store
        self.settings = settings

    def acquire(
        self,
        inputs: list[StagedInput],
        target_profile: TargetProfile,
        *,
        existing_sources_by_sha: dict[str, SourceManifest] | None = None,
    ) -> AcquisitionOutcome:
        manifests: list[SourceManifest] = []
        initial_results: list[SourceResult] = []
        existing_by_hash = {
            key: value
            for key, value in (existing_sources_by_sha or {}).items()
            if key and value
        }
        by_hash: dict[str, SourceManifest] = {}

        for source_input in inputs:
            self.store.emit(
                "acquisition_started",
                {
                    "input_id": source_input.input_id,
                    "kind": source_input.kind,
                    "display_name": source_input.display_name,
                    "source_origin": source_input.source_origin,
                    "source_priority": source_input.source_priority,
                },
            )
            alias = InputAlias(
                input_id=source_input.input_id,
                kind=source_input.kind,
                display_name=source_input.display_name,
                location=self._safe_public_url(source_input.url) if source_input.kind == "url" else source_input.display_name,
                user_declared_version=source_input.user_declared_version or "unknown",
            )
            try:
                fetched = self._read_input(source_input)
                auth_reason = self._authentication_reason(fetched.data, fetched.media_type)
                if auth_reason:
                    raise AuthenticationRequired(auth_reason)

                modality, media_type, suffix = self._detect_modality(
                    fetched.data,
                    fetched.media_type,
                    source_input.display_name,
                    fetched.final_url,
                )
                if modality == "unknown":
                    manifest = self._terminal_manifest(
                        source_input,
                        alias,
                        status="excluded",
                        reason="unsupported_media_type",
                        media_type=media_type,
                    )
                    manifests.append(manifest)
                    initial_results.append(self._terminal_result(manifest, "The source is not a supported PDF or HTML document."))
                    self._persist_manifest(manifest)
                    self.store.emit(
                        "source_excluded",
                        {"reason": manifest.exclusion_reason, "media_type": media_type},
                        source_id=manifest.source_id,
                        level="warning",
                    )
                    continue
                if (
                    source_input.source_origin == "websearch"
                    and modality == "html"
                    and not source_input.source_metadata.get("discovery_candidate_id")
                    and not self._websearch_html_relevant(
                        fetched.data,
                        source_input.display_name,
                        fetched.final_url,
                        target_profile,
                    )
                ):
                    manifest = self._terminal_manifest(
                        source_input,
                        alias,
                        status="excluded",
                        reason="websearch_content_not_relevant",
                        media_type=media_type,
                    )
                    manifests.append(manifest)
                    initial_results.append(
                        self._terminal_result(
                            manifest,
                            "Websearch source did not contain the target software name.",
                        )
                    )
                    self._persist_manifest(manifest)
                    self.store.emit(
                        "source_excluded",
                        {
                            "reason": manifest.exclusion_reason,
                            "media_type": media_type,
                            "source_origin": source_input.source_origin,
                        },
                        source_id=manifest.source_id,
                        level="warning",
                    )
                    continue

                inferred_html_origin_url = (
                    self._infer_uploaded_html_origin_url(fetched.data)
                    if source_input.kind == "uploaded_file" and modality == "html"
                    else None
                )
                origin_url = self._safe_public_url(source_input.url) or inferred_html_origin_url
                final_url = fetched.final_url if source_input.kind == "url" else inferred_html_origin_url

                artifact = self.store.put_artifact(
                    fetched.data,
                    kind="raw_source",
                    media_type=media_type,
                    suffix=suffix,
                    metadata={
                        "input_id": source_input.input_id,
                        "display_name": source_input.display_name,
                        "final_url": final_url,
                        "inferred_html_origin_url": inferred_html_origin_url,
                        "source_origin": source_input.source_origin,
                        "source_priority": source_input.source_priority,
                        "source_metadata": source_input.source_metadata,
                    },
                )
                if artifact.sha256 in existing_by_hash:
                    canonical = existing_by_hash[artifact.sha256]
                    manifest = self._duplicate_manifest(
                        source_input,
                        alias,
                        canonical,
                        artifact,
                        modality=modality,
                        media_type=media_type,
                        final_url=final_url,
                        status_code=fetched.status_code,
                    )
                    manifests.append(manifest)
                    initial_results.append(
                        self._terminal_result(
                            manifest,
                            "Source is an exact duplicate of an existing source and was skipped.",
                        )
                    )
                    self._persist_manifest(manifest)
                    event_type = (
                        "websearch_source_duplicate"
                        if source_input.source_origin == "websearch"
                        else "source_duplicate_excluded"
                    )
                    self.store.emit(
                        event_type,
                        {
                            "input_id": source_input.input_id,
                            "duplicate_of_source_id": canonical.source_id,
                            "duplicate_match": "sha256",
                            "content_sha256": artifact.sha256,
                            "source_origin": source_input.source_origin,
                        },
                        source_id=manifest.source_id,
                        level="warning",
                    )
                    continue
                if artifact.sha256 in by_hash:
                    canonical = by_hash[artifact.sha256]
                    canonical.aliases.append(alias)
                    declared_versions = {
                        item.user_declared_version
                        for item in canonical.aliases
                        if item.user_declared_version.lower() != "unknown"
                    }
                    if len(declared_versions) > 1:
                        warning = "Duplicate inputs have different user-declared versions. The source will be processed once, and its own version takes precedence."
                        if warning not in canonical.warnings:
                            canonical.warnings.append(warning)
                    self._persist_manifest(canonical)
                    self.store.emit(
                        "source_deduplicated",
                        {
                            "input_id": source_input.input_id,
                            "canonical_source_id": canonical.source_id,
                            "content_sha256": artifact.sha256,
                        },
                        source_id=canonical.source_id,
                    )
                    continue

                source_id = self.store.canonical_id(
                    "SRC", f"src_{artifact.sha256[:20]}", "content", artifact.sha256
                )
                name_and_text = source_input.display_name
                if modality == "html":
                    name_and_text += "\n" + self._decode_html(fetched.data)[:80_000]
                document_version = detect_document_version(
                    name_and_text,
                    target_profile.software_name,
                )
                if source_input.source_metadata.get("discovery_candidate_id"):
                    # The discovery Agent assessed the version; selection metadata
                    # preserves that basis instead of guessing from incidental numbers.
                    document_version = "unknown"
                resolved, basis, conflict = resolve_version(
                    document_version,
                    source_input.user_declared_version,
                )
                manifest = SourceManifest(
                    source_id=source_id,
                    canonical_input_id=source_input.input_id,
                    display_name=source_input.display_name,
                    primary_modality=modality,
                    media_type=media_type,
                    raw_artifact_id=artifact.artifact_id,
                    content_sha256=artifact.sha256,
                    byte_count=artifact.byte_count,
                    aliases=[alias],
                    user_declared_version=source_input.user_declared_version or "unknown",
                    document_declared_version=document_version,
                    resolved_version=resolved,
                    version_resolution_basis=basis,
                    version_conflict=conflict,
                    origin_url=origin_url,
                    final_url=final_url,
                    acquisition={
                        "kind": source_input.kind,
                        "local_resource_base_url": (
                            Path(source_input.staged_path).resolve().parent.as_uri() + "/"
                            if source_input.kind == "uploaded_file" and source_input.staged_path
                            and modality == "html" else None
                        ),
                        "status_code": fetched.status_code,
                        "headers": fetched.headers,
                        "target_software": target_profile.software_name,
                        "inferred_html_origin_url": inferred_html_origin_url,
                        "source_origin": source_input.source_origin,
                        "source_priority": source_input.source_priority,
                    },
                    source_origin=source_input.source_origin,
                    source_priority=source_input.source_priority,
                    source_metadata=source_input.source_metadata,
                )
                manifests.append(manifest)
                by_hash[artifact.sha256] = manifest
                self._persist_manifest(manifest)
                if inferred_html_origin_url:
                    self.store.emit(
                        "html_origin_url_inferred",
                        {"origin_url": inferred_html_origin_url},
                        source_id=source_id,
                    )
                self.store.emit(
                    "raw_artifact_created",
                    {
                        "artifact_id": artifact.artifact_id,
                        "sha256": artifact.sha256,
                        "byte_count": artifact.byte_count,
                        "modality": modality,
                    },
                    source_id=source_id,
                )
            except AuthenticationRequired as exc:
                manifest = self._terminal_manifest(
                    source_input,
                    alias,
                    status="excluded",
                    reason="authentication_required",
                )
                manifests.append(manifest)
                initial_results.append(self._terminal_result(manifest, str(exc)))
                self._persist_manifest(manifest)
                self.store.emit(
                    "source_excluded",
                    {"reason": "authentication_required", "detail": str(exc)},
                    source_id=manifest.source_id,
                    level="warning",
                )
            except Exception as exc:
                manifest = self._terminal_manifest(
                    source_input,
                    alias,
                    status="failed",
                    reason="acquisition_failed",
                )
                manifests.append(manifest)
                result = self._terminal_result(manifest, "Source acquisition failed.")
                result.errors.append(str(exc))
                initial_results.append(result)
                self._persist_manifest(manifest)
                self.store.emit(
                    "acquisition_failed",
                    {"error_type": type(exc).__name__, "message": str(exc)},
                    source_id=manifest.source_id,
                    level="error",
                )

        return AcquisitionOutcome(manifests=manifests, initial_results=initial_results)

    def fetch_html_child(self, source_id: str, root_url: str, candidate_url: str) -> dict[str, Any]:
        normalized = self.normalize_url(root_url, candidate_url)
        if not normalized or not self.is_within_tutorial_scope(root_url, normalized):
            return {"status": "excluded", "reason": "outside_tutorial_scope", "url": candidate_url}
        try:
            fetched = self._read_url(normalized)
            reason = self._authentication_reason(fetched.data, fetched.media_type)
            if reason:
                return {"status": "excluded", "reason": "authentication_required", "url": normalized}
            modality, media_type, suffix = self._detect_modality(
                fetched.data,
                fetched.media_type,
                Path(urlparse(normalized).path).name,
                normalized,
            )
            if modality != "html":
                return {"status": "excluded", "reason": "not_html", "url": normalized}
            artifact = self.store.put_artifact(
                fetched.data,
                kind="raw_html_part",
                media_type=media_type,
                suffix=suffix,
                source_id=source_id,
                metadata={"url": normalized},
            )
            return {
                "status": "succeeded",
                "url": normalized,
                "final_url": fetched.final_url,
                "data": fetched.data,
                "artifact_id": artifact.artifact_id,
                "media_type": media_type,
            }
        except AuthenticationRequired:
            return {"status": "excluded", "reason": "authentication_required", "url": normalized}
        except Exception as exc:
            return {
                "status": "failed",
                "reason": "fetch_failed",
                "url": normalized,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    def normalize_url(self, base_url: str, href: str) -> str | None:
        href = href.strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "data:")):
            return None
        joined = urljoin(base_url, href)
        parsed = urlparse(joined)
        if parsed.scheme not in {"http", "https", "file"}:
            return None
        return urlunparse(parsed._replace(fragment=""))

    def is_within_tutorial_scope(self, root_url: str, candidate_url: str) -> bool:
        root = urlparse(root_url)
        candidate = urlparse(candidate_url)
        if root.scheme != candidate.scheme:
            return False
        if root.scheme in {"http", "https"}:
            if root.netloc.casefold() != candidate.netloc.casefold():
                return False
            root_directory = posixpath.dirname(root.path or "/")
            prefix = root_directory.rstrip("/") + "/"
            return candidate.path == root.path or candidate.path.startswith(prefix)
        if root.scheme == "file":
            root_path = self._file_url_to_path(root_url).resolve()
            candidate_path = self._file_url_to_path(candidate_url).resolve()
            return candidate_path == root_path or root_path.parent in candidate_path.parents
        return False

    def _read_input(self, source_input: StagedInput) -> FetchResult:
        if source_input.kind == "uploaded_file":
            if not source_input.staged_path:
                raise ValueError("Uploaded input has no staged path")
            path = Path(source_input.staged_path)
            data = self._read_limited_file(path)
            media_type = source_input.declared_media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return FetchResult(data, media_type, path.name, None, {})
        if not source_input.url:
            raise ValueError("URL input has no URL")
        # Reuse only a complete host-created discovery preview from this Run.
        # The model's candidate schema cannot supply verification/cache fields.
        verification = source_input.source_metadata.get('verification') or {}
        cached_id = verification.get('cached_artifact_id')
        if (source_input.source_origin == 'websearch' and source_input.source_metadata.get('discovery_candidate_id')
                and verification.get('complete') is True and isinstance(cached_id, str)):
            try:
                record = self.store.artifact_record(cached_id)
                if (record.kind == 'discovery_pdf_preview' and record.metadata.get('complete') is True
                        and record.metadata.get('url') == source_input.url):
                    data = self._read_limited_file(self.store.artifact_path(cached_id))
                    if b'%PDF-' in data[:1024]:
                        self.store.emit('acquisition_preview_reused', {'artifact_id': cached_id, 'url': source_input.url})
                        return FetchResult(data, 'application/pdf', record.metadata['final_url'],
                                           record.metadata.get('http_status', 200), {})
            except (FileNotFoundError, KeyError):
                pass
        return self._read_url(source_input.url)

    def _read_url(self, value: str, *, max_bytes: int | None = None) -> FetchResult:
        self._raise_if_cancelled()
        maximum = min(self.settings.max_source_bytes, max_bytes) if max_bytes is not None else self.settings.max_source_bytes
        direct_path = Path(value)
        if direct_path.exists():
            data = self._read_limited_file(direct_path, maximum)
            return FetchResult(
                data,
                mimetypes.guess_type(direct_path.name)[0] or "application/octet-stream",
                direct_path.resolve().as_uri(),
                None,
                {},
            )
        parsed = urlparse(value)
        if parsed.username or parsed.password:
            raise AuthenticationRequired("The URL contains credentials and was excluded by policy.")
        if parsed.scheme == "file":
            path = self._file_url_to_path(value)
            data = self._read_limited_file(path, maximum)
            return FetchResult(
                data,
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                path.resolve().as_uri(),
                None,
                {},
            )
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http(s), file URL, or an existing local path is supported")

        headers = {"User-Agent": "ModalPreprocessingAgent/0.1", "Accept": "*/*"}
        with httpx.Client(follow_redirects=True, timeout=self.settings.fetch_timeout_seconds) as client:
            with client.stream("GET", value, headers=headers) as response:
                if response.status_code in {401, 402, 403, 407}:
                    raise AuthenticationRequired(f"HTTP {response.status_code}")
                response.raise_for_status()
                length = int(response.headers.get("content-length", "0") or 0)
                if length > maximum:
                    raise SourceTooLarge(f"Source exceeds {maximum} bytes")
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    self._raise_if_cancelled()
                    total += len(chunk)
                    if total > maximum:
                        raise SourceTooLarge(f"Source exceeds {maximum} bytes")
                    chunks.append(chunk)
                media_type = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0].strip()
                safe_headers = {
                    key: response.headers[key]
                    for key in ("content-type", "content-length", "etag", "last-modified")
                    if key in response.headers
                }
                return FetchResult(
                    b"".join(chunks),
                    media_type,
                    str(response.url),
                    response.status_code,
                    safe_headers,
                )

    def _raise_if_cancelled(self) -> None:
        if self.store.cancellation_requested():
            raise RuntimeError("run_cancelled")

    def _websearch_html_relevant(
        self,
        data: bytes,
        display_name: str,
        final_url: str,
        target_profile: TargetProfile,
    ) -> bool:
        text = " ".join(
            [
                display_name,
                final_url,
                self._decode_html(data)[:120_000],
            ]
        ).casefold()
        compact_text = re.sub(r"[^a-z0-9]+", "", text)
        for value in [target_profile.software_name, target_profile.software_id or ""]:
            raw = str(value or "").strip().casefold()
            if not raw:
                continue
            compact = re.sub(r"[^a-z0-9]+", "", raw)
            if compact and compact in compact_text:
                return True
            tokens = [
                token
                for token in re.findall(r"[a-z0-9]+", raw)
                if len(token) >= 3
            ]
            if tokens and all(token in text for token in tokens):
                return True
        return False

    def _read_limited_file(self, path: Path, maximum: int | None = None) -> bytes:
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        maximum = maximum if maximum is not None else self.settings.max_source_bytes
        if size > maximum:
            raise SourceTooLarge(f"Source exceeds {maximum} bytes")
        return path.read_bytes()

    def _file_url_to_path(self, value: str) -> Path:
        parsed = urlparse(value)
        raw_path = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:", raw_path):
            raw_path = raw_path[1:]
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            raw_path = f"//{parsed.netloc}{raw_path}"
        return Path(raw_path)

    def _authentication_reason(self, data: bytes, media_type: str) -> str | None:
        if "html" not in media_type and not data.lstrip()[:100].lower().startswith((b"<!doctype html", b"<html")):
            return None
        text = self._decode_html(data)
        soup = BeautifulSoup(text, "html.parser")
        if soup.select_one('input[type="password"]'):
            return "The page contains a password login form."
        visible = soup.get_text(" ", strip=True).casefold()
        hard_markers = (
            "sign in to continue",
            "log in to continue",
            "subscribe to continue",
            "purchase to continue",
            "请登录后继续",
            "登录后查看",
            "订阅后继续",
            "付费后查看",
        )
        if any(marker in visible for marker in hard_markers):
            return "The page requires login, a subscription, or payment to continue."
        return None

    def _detect_modality(
        self,
        data: bytes,
        declared_media_type: str,
        display_name: str,
        final_url: str,
    ) -> tuple[str, str, str]:
        prefix = data[:512].lstrip().lower()
        name = (display_name or Path(urlparse(final_url).path).name).lower()
        if data.startswith(b"%PDF-"):
            return "pdf", "application/pdf", ".pdf"
        if self._is_github_wiki_url(final_url):
            return "unknown", declared_media_type or "application/octet-stream", Path(name).suffix or ".bin"
        if self._github_repo_from_url(final_url):
            return "github", "application/vnd.github.repository+html", ".github"
        if (
            "html" in declared_media_type
            or name.endswith((".html", ".htm"))
            or prefix.startswith((b"<!doctype html", b"<html"))
        ):
            return "html", "text/html", ".html"
        return "unknown", declared_media_type or "application/octet-stream", Path(name).suffix or ".bin"

    def _github_repo_from_url(self, value: str) -> tuple[str, str] | None:
        parsed = urlparse(value)
        hostname = (parsed.hostname or "").casefold()
        if hostname not in {"github.com", "www.github.com"}:
            return None
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1]
        if owner.casefold() in {"login", "search", "features", "topics", "marketplace"}:
            return None
        repo = repo.removesuffix(".git")
        if not owner or not repo:
            return None
        return owner, repo

    def _is_github_wiki_url(self, value: str) -> bool:
        parsed = urlparse(value)
        hostname = (parsed.hostname or "").casefold()
        if hostname not in {"github.com", "www.github.com"}:
            return False
        parts = [part.casefold() for part in parsed.path.split("/") if part]
        return len(parts) >= 3 and parts[2] == "wiki"

    def _decode_html(self, data: bytes) -> str:
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    def _infer_uploaded_html_origin_url(self, data: bytes) -> str | None:
        text = self._decode_html(data)
        saved_match = SAVED_FROM_URL_PATTERN.search(text[:20_000])
        if saved_match:
            candidate = html_lib.unescape(saved_match.group(1).strip())
            if self._is_navigable_url(candidate):
                return self._safe_public_url(candidate)

        soup = BeautifulSoup(text[:200_000], "html.parser")
        for candidate in self._html_metadata_url_candidates(soup):
            if self._is_navigable_url(candidate):
                return self._safe_public_url(candidate)
        return None

    def _html_metadata_url_candidates(self, soup: BeautifulSoup) -> list[str]:
        candidates: list[str] = []
        for node in soup.find_all("link", href=True):
            rel = node.get("rel") or []
            rel_values = rel if isinstance(rel, list) else str(rel).split()
            if any(str(value).casefold() == "canonical" for value in rel_values):
                candidates.append(str(node.get("href") or "").strip())
        for node in soup.find_all("meta"):
            key = str(node.get("property") or node.get("name") or "").casefold()
            if key in {"og:url", "twitter:url"}:
                candidates.append(str(node.get("content") or "").strip())
        base = soup.find("base", href=True)
        if base is not None:
            candidates.append(str(base.get("href") or "").strip())
        return candidates

    def _is_navigable_url(self, value: str | None) -> bool:
        parsed = urlparse(str(value or "").strip())
        if parsed.scheme in {"http", "https"}:
            return bool(parsed.netloc)
        if parsed.scheme == "file":
            return bool(parsed.path)
        return False

    def _terminal_manifest(
        self,
        source_input: StagedInput,
        alias: InputAlias,
        *,
        status: str,
        reason: str,
        media_type: str = "application/octet-stream",
    ) -> SourceManifest:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", source_input.input_id)
        return SourceManifest(
            source_id=self.store.canonical_id(
                "SRC", f"src_{safe_id}", "terminal", source_input.input_id
            ),
            canonical_input_id=source_input.input_id,
            display_name=source_input.display_name,
            primary_modality="unknown",
            media_type=media_type,
            status=status,
            aliases=[alias],
            user_declared_version=source_input.user_declared_version or "unknown",
            resolved_version=source_input.user_declared_version or "unknown",
            version_resolution_basis=(
                "user" if (source_input.user_declared_version or "unknown").lower() != "unknown" else "unknown"
            ),
            origin_url=self._safe_public_url(source_input.url),
            final_url=self._safe_public_url(source_input.url),
            acquisition={
                "kind": source_input.kind,
                "source_origin": source_input.source_origin,
                "source_priority": source_input.source_priority,
            },
            source_origin=source_input.source_origin,
            source_priority=source_input.source_priority,
            source_metadata=source_input.source_metadata,
            exclusion_reason=reason,
        )

    def _duplicate_manifest(
        self,
        source_input: StagedInput,
        alias: InputAlias,
        canonical: SourceManifest,
        artifact: Any,
        *,
        modality: str,
        media_type: str,
        final_url: str | None,
        status_code: int | None,
    ) -> SourceManifest:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", source_input.input_id)
        duplicate_metadata = {
            **source_input.source_metadata,
            "duplicate_of_source_id": canonical.source_id,
            "duplicate_match": "sha256",
            "duplicate_content_sha256": artifact.sha256,
        }
        return SourceManifest(
            source_id=self.store.canonical_id(
                "SRC", f"src_{safe_id}", "duplicate", source_input.input_id
            ),
            canonical_input_id=source_input.input_id,
            display_name=source_input.display_name,
            primary_modality=modality,
            media_type=media_type,
            raw_artifact_id=artifact.artifact_id,
            content_sha256=artifact.sha256,
            byte_count=artifact.byte_count,
            status="excluded",
            aliases=[alias],
            user_declared_version=source_input.user_declared_version or "unknown",
            document_declared_version="unknown",
            resolved_version=source_input.user_declared_version or canonical.resolved_version or "unknown",
            version_resolution_basis=(
                "user" if (source_input.user_declared_version or "unknown").lower() != "unknown" else "unknown"
            ),
            origin_url=self._safe_public_url(source_input.url) or final_url,
            final_url=final_url,
            acquisition={
                "kind": source_input.kind,
                "status_code": status_code,
                "inferred_html_origin_url": final_url if source_input.kind == "uploaded_file" else None,
                "source_origin": source_input.source_origin,
                "source_priority": source_input.source_priority,
                "duplicate_of_source_id": canonical.source_id,
                "duplicate_match": "sha256",
            },
            source_origin=source_input.source_origin,
            source_priority=source_input.source_priority,
            source_metadata=duplicate_metadata,
            exclusion_reason="duplicate_of_existing_source",
        )

    def _terminal_result(self, manifest: SourceManifest, summary: str) -> SourceResult:
        return SourceResult(
            source_id=manifest.source_id,
            display_name=manifest.display_name,
            modality=manifest.primary_modality,
            status=manifest.status,
            summary=summary,
            coverage=Coverage(),
            gaps=[manifest.exclusion_reason] if manifest.exclusion_reason else [],
            manifest=manifest,
        )

    def _persist_manifest(self, manifest: SourceManifest) -> None:
        self.store.write_source_json(
            manifest.source_id,
            "manifest.json",
            manifest.model_dump(mode="json"),
        )

    def _safe_public_url(self, value: str | None) -> str | None:
        if not value:
            return value
        parsed = urlparse(value)
        if not (parsed.username or parsed.password):
            return value
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port:
            netloc = f"{netloc}:{port}"
        return urlunparse(parsed._replace(netloc=netloc))
