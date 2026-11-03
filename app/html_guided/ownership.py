"""Evidence-based document ownership for the optional guided HTML route.

Related prose links never establish ownership. Explicit root catalogs win even
across publishers; outside that catalog, only real publisher hierarchy can
prove a conflict. Missing/inconsistent hierarchy remains unknown.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import url2pathname

from bs4 import Comment, NavigableString, Tag

from ..html_navigation import classify_html_link, html_base_url, is_authored_contents
from ..html_sourcepart.extractor import parse_html

MAX_METADATA_FETCHES = 2
MAX_METADATA_BYTES = 256_000
MAX_METADATA_TOTAL_BYTES = 512_000
_LISTS = {"ul", "ol", "dl"}


def _url(value: str, base: str = "") -> str:
    try:
        parsed = urlsplit(urljoin(base, str(value).strip()))
        if parsed.scheme not in {"http", "https", "file"} or parsed.username or parsed.password:
            return ""
        if parsed.scheme in {"http", "https"} and (not parsed.hostname or "\\" in parsed.netloc):
            return ""
        return urlunsplit(parsed._replace(fragment=""))
    except (ValueError, TypeError):
        return ""


def _path(node: Tag) -> str:
    pieces = []
    while isinstance(node, Tag) and node.name != "[document]":
        index = 1 + sum(isinstance(sibling, Tag) and sibling.name == node.name for sibling in node.previous_siblings)
        pieces.append(f"{node.name}:nth-of-type({index})")
        node = node.parent
    return " > ".join(reversed(pieces))


def _selected(soup: Any, path: str) -> Tag:
    region = soup if not path or path == "[document]" else soup.select_one(path)
    if region is None:
        raise ValueError("Recorded selected content boundary does not exist")
    return region


def _child_catalog_chrome(node: Tag, selected: Tag) -> bool:
    """A child page's site landmarks cannot grant new document ownership.

    This deliberately does not apply to root portals or captured external TOCs.
    A body doc-toc remains eligible unless it is inside an explicit chrome
    landmark such as an aside or complementary region.
    """
    for owner in (node, *node.parents):
        if not isinstance(owner, Tag):
            continue
        roles = set(str(owner.get("role") or "").casefold().split())
        if (owner.name in {"aside", "header", "footer"}
                or owner.name == "nav" and "doc-toc" not in roles
                or roles & {"complementary", "banner", "contentinfo", "navigation", "site-navigation"}):
            return True
        if owner is selected:
            break
    return False


@dataclass(frozen=True)
class Evidence:
    document_url: str
    artifact_id: str
    dom_path: str
    relation: str
    target_url: str = ""

    def as_dict(self) -> dict[str, str]:
        return dict(document_url=self.document_url, artifact_id=self.artifact_id,
                    dom_path=self.dom_path, relation=self.relation, target_url=self.target_url)


@dataclass
class TocNode:
    url: str
    label: str
    evidence: Evidence
    children: list[TocNode] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"url": self.url, "label": self.label, "evidence": self.evidence.as_dict(),
                "children": [child.as_dict() for child in self.children]}


def _toc_trees(region: Tag, page_url: str, artifact_id: str) -> list[list[TocNode]]:
    """Uncompressed ul/ol/dl ownership; a dt owns the following dd subtree."""
    base = html_base_url(region.find_parent("html") or region, page_url)

    def item(node: Tag, owner: Tag) -> TocNode:
        anchor = next((a for a in node.find_all("a", href=True) if a.find_parent(list(_LISTS)) is owner), None)
        target = _url(str(anchor.get("href") or ""), base) if anchor else ""
        label = (anchor.get_text(" ", strip=True) if anchor else node.get_text(" ", strip=True))
        result = TocNode(target, label, Evidence(page_url, artifact_id, _path(anchor or node), "authored_toc", target))
        for child in node.find_all(list(_LISTS)):
            if child.find_parent(list(_LISTS)) is owner:
                result.children.extend(items(child))
        return result

    def items(owner: Tag) -> list[TocNode]:
        if owner.name != "dl":
            return [item(node, owner) for node in owner.find_all("li", recursive=False)]
        output = []
        for node in owner.find_all(["dt", "dd"], recursive=False):
            if node.name == "dt":
                output.append(item(node, owner))
            elif output:
                for child in node.find_all(list(_LISTS)):
                    if child.find_parent(list(_LISTS)) is owner:
                        output[-1].children.extend(items(child))
        return output

    lists = ([region] if region.name in _LISTS else []) + region.find_all(list(_LISTS))
    output = []
    for node in lists:
        parent_list = node.find_parent(list(_LISTS))
        if parent_list is not None and (parent_list is region or region in parent_list.parents):
            continue
        if is_authored_contents(node):
            tree = items(node)
            if tree:
                output.append(tree)
    return output


def _walk(nodes: Sequence[TocNode], prefix: tuple[int, ...] = ()):
    for index, node in enumerate(nodes):
        position = (*prefix, index)
        yield node, position
        yield from _walk(node.children, position)


@dataclass
class DocumentFacts:
    url: str
    up: list[Evidence]
    breadcrumbs: list[tuple[list[str], list[Evidence]]]

    def as_dict(self) -> dict[str, Any]:
        return {"url": self.url, "up": [item.as_dict() for item in self.up],
                "breadcrumbs": [{"urls": urls, "evidence": [item.as_dict() for item in proof]} for urls, proof in self.breadcrumbs]}


def _facts(soup: Any, page_url: str, artifact_id: str) -> DocumentFacts:
    base = html_base_url(soup, page_url)
    up = []
    for link in soup.head.find_all("link", href=True) if soup.head else []:
        if "up" in {str(value).lower() for value in link.get("rel", [])}:
            target = _url(link["href"], base)
            if target:
                up.append(Evidence(page_url, artifact_id, _path(link), "publisher_up", target))
    candidates = []
    for node in soup.find_all(["nav", "ol", "ul", "div"]):
        tokens = " ".join([str(node.get("aria-label") or ""), *node.get("class", [])]).lower()
        microdata = str(node.get("itemtype") or "").rstrip("/").endswith("/BreadcrumbList")
        if "breadcrumb" in tokens or microdata:
            candidates.append(node)
    candidate_ids = {id(node) for node in candidates}
    breadcrumbs = []
    seen = set()
    for node in candidates:
        if any(id(parent) in candidate_ids for parent in node.parents):
            continue
        urls, proof = [], []
        for anchor in node.find_all("a", href=True):
            target = _url(anchor["href"], base)
            if target and (not urls or urls[-1] != target):
                urls.append(target)
                proof.append(Evidence(page_url, artifact_id, _path(anchor), "publisher_breadcrumb", target))
        if not urls:
            continue
        if urls[-1] != page_url:
            anchors = node.find_all("a", href=True)
            last = anchors[-1] if anchors else None
            after_last, trailing = False, []
            for value in node.descendants:
                if value is last:
                    after_last = True
                if (after_last and isinstance(value, NavigableString) and not isinstance(value, Comment)
                        and last not in value.parents):
                    trailing.append(str(value))
            declared_current = bool(node.find(attrs={"aria-current": ["page", "true"]})) or any(character.isalnum() for character in "".join(trailing))
            urls.append(page_url)
            proof.append(Evidence(page_url, artifact_id, _path(node),
                                  "breadcrumb_current_document" if declared_current else "breadcrumb_current_inferred", page_url))
        key = tuple(urls)
        if key not in seen:
            seen.add(key)
            breadcrumbs.append((urls, proof))
    return DocumentFacts(page_url, up, breadcrumbs)


@dataclass
class OwnershipDecision:
    status: str
    reason: str
    evidence: list[Evidence] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason,
                "evidence": [item.as_dict() for item in self.evidence]}


@dataclass
class MetadataFetchResult:
    data: bytes
    final_url: str
    media_type: str = "text/html"


@dataclass
class MetadataProbeReport:
    captures: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    fetch_count: int = 0
    total_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"captures": self.captures, "gaps": self.gaps,
                "fetch_count": self.fetch_count, "total_bytes": self.total_bytes}


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def _same_origin(left: str, right: str) -> bool:
    try:
        def origin(url: str):
            value = urlsplit(url)
            return value.scheme, value.hostname, value.port or (443 if value.scheme == "https" else 80)
        return origin(left) == origin(right)
    except ValueError:
        return False


def _safe_metadata_url(url: str, entry: str, local_roots: Sequence[Path]) -> bool:
    target, source = urlsplit(url), urlsplit(entry)
    if target.scheme in {"http", "https"}:
        return source.scheme in {"http", "https"} and _same_origin(url, entry)
    if target.scheme != "file" or source.scheme != "file" or target.netloc.lower() not in {"", "localhost"}:
        return False
    path = Path(url2pathname(target.path)).resolve()
    defaults = [Path(url2pathname(source.path)).resolve().parent]
    return any(path == root.resolve() or path.is_relative_to(root.resolve()) for root in (local_roots or defaults))


class OwnershipBoundary:
    def __init__(self, entry_url: str, root_facts: DocumentFacts):
        self.entry_url = entry_url
        self.seeds: dict[str, list[Evidence]] = {}
        self.positive_edges: dict[str, dict[str, Evidence]] = {}
        self.documents = {entry_url: root_facts}
        self.publisher_indexes: list[dict[str, Any]] = []
        self.probe_report = MetadataProbeReport()

    @classmethod
    def from_root(cls, *, html: str, entry_url: str, selected_dom_path: str,
                  artifact_id: str, external_topics: Sequence[Mapping[str, Any]] = ()) -> OwnershipBoundary:
        entry_url = _url(entry_url)
        if not entry_url:
            raise ValueError("Entry has no supported captured URL")
        soup = parse_html(html)
        region = _selected(soup, selected_dom_path)
        boundary = cls(entry_url, _facts(soup, entry_url, artifact_id))
        trees = _toc_trees(region, entry_url, artifact_id)
        for tree in trees:
            for node, _ in _walk(tree):
                if node.url and node.url != entry_url:
                    boundary.seeds.setdefault(node.url, []).append(node.evidence)
        # Catalogs without explicit TOC markup can use physical directory-entry
        # declarations. Never turn the model's root fetch list into ownership.
        if not trees:
            for anchor in region.find_all("a", href=True):
                role, reason = classify_html_link(anchor)
                target = _url(anchor["href"], html_base_url(soup, entry_url))
                if role == "directory_entry" and target and target != entry_url:
                    proof = Evidence(entry_url, artifact_id, _path(anchor), "root_catalog:" + reason, target)
                    boundary.seeds.setdefault(target, []).append(proof)
        for topic in external_topics:
            source = topic.get("source") or topic.get("navigation_source") or {}
            capture_id = source.get("artifact_id") or source.get("source_artifact_id")
            pointer = source.get("json_pointer") or source.get("pointer") or source.get("dom_path")
            document_url = source.get("url") or source.get("resource_url") or source.get("document_url")
            target = _url(str(topic.get("url") or ""), entry_url) if topic.get("url") else ""
            if target and capture_id and pointer and document_url:
                proof = Evidence(str(document_url), str(capture_id), str(pointer), "external_authored_toc", target)
                boundary.seeds.setdefault(target, []).append(proof)
        return boundary

    @property
    def strong(self) -> bool:
        return bool(self.seeds or self.publisher_indexes)

    def _positive(self, url: str, seen: set[str] | None = None) -> list[Evidence] | None:
        if url == self.entry_url:
            return []
        if url in self.seeds:
            return list(self.seeds[url])
        seen = set() if seen is None else seen
        if url in seen:
            return None
        seen.add(url)
        for parent, children in self.positive_edges.items():
            if url in children:
                prefix = self._positive(parent, set(seen))
                if prefix is not None:
                    return [*prefix, children[url]]
        return None

    def _index_relation(self, url: str) -> tuple[str, list[Evidence]] | None:
        answers = []
        for index in self.publisher_indexes:
            positions = index["positions"]
            targets = positions.get(url, [])
            roots = positions.get(self.entry_url, [])
            if not targets or len(roots) != 1:
                continue
            root_path = roots[0][0]
            statuses, evidence = set(), []
            for target_path, proof in targets:
                statuses.add("owned" if target_path[:len(root_path)] == root_path else "foreign")
                evidence.extend([roots[0][1], proof])
            if len(statuses) == 1:
                answers.append((next(iter(statuses)), evidence))
        if answers and len({status for status, _ in answers}) == 1:
            return answers[0]
        return None

    def _publisher_decision(self, facts: DocumentFacts) -> OwnershipDecision:
        signals: list[tuple[str, list[Evidence]]] = []
        indexed_document = self._index_relation(facts.url)
        if indexed_document:
            signals.append(indexed_document)
        current, chain, seen = facts.url, [], set()
        while current not in seen:
            seen.add(current)
            if current != facts.url:
                positive = self._positive(current)
                indexed = self._index_relation(current)
                if positive is not None:
                    signals.append(("owned", [*chain, *positive]))
                    break
                if indexed:
                    signals.append((indexed[0], [*chain, *indexed[1]]))
                    break
            document = self.documents.get(current)
            targets = {item.target_url for item in document.up} if document else set()
            if len(targets) != 1:
                break
            edge = next(item for item in document.up if item.target_url in targets)
            chain.append(edge)
            current = edge.target_url
        root_breadcrumbs = self.documents[self.entry_url].breadcrumbs
        for root_urls, root_proof in root_breadcrumbs:
            if facts.url in root_urls[:-1]:
                signals.append(("foreign", root_proof))
        for target_urls, target_proof in facts.breadcrumbs:
            if self.entry_url in target_urls[:-1]:
                signals.append(("owned", target_proof))
                continue
            for root_urls, root_proof in root_breadcrumbs:
                common = 0
                for left, right in zip(root_urls, target_urls):
                    if left != right:
                        break
                    common += 1
                if common and common < len(root_urls) and (common < len(target_urls) or target_urls[-1] != self.entry_url):
                    # A breadcrumb showing only a home/parent link may omit
                    # intermediate levels; an inferred current leaf is not a
                    # publisher-declared sibling branch.
                    if ((common < len(target_proof) and target_proof[common].relation == "breadcrumb_current_inferred")
                            or (common < len(root_proof) and root_proof[common].relation == "breadcrumb_current_inferred")):
                        continue
                    signals.append(("foreign", [*root_proof, *target_proof]))
        statuses = {status for status, _ in signals}
        if len(statuses) == 1:
            status = next(iter(statuses))
            return OwnershipDecision(status, "publisher_descendant" if status == "owned" else "publisher_owner_conflict", signals[0][1])
        return OwnershipDecision("unknown", "publisher_evidence_conflicts" if signals else "publisher_owner_chain_unresolved")

    def inspect_document(self, *, html: str, page_url: str, artifact_id: str) -> OwnershipDecision:
        page_url = _url(page_url)
        facts = _facts(parse_html(html), page_url, artifact_id)
        self.documents[page_url] = facts
        positive = self._positive(page_url)
        if positive is not None:
            return OwnershipDecision("owned", "entry_or_explicit_author_catalog", positive)
        if not self.strong:
            return OwnershipDecision("unknown", "entry_has_no_strong_authored_boundary")
        return self._publisher_decision(facts)

    def add_owned_contents(self, *, html: str, page_url: str, selected_dom_path: str,
                           artifact_id: str) -> list[str]:
        page_url = _url(page_url)
        decision = self.inspect_document(html=html, page_url=page_url, artifact_id=artifact_id)
        if decision.status != "owned":
            return []
        soup = parse_html(html)
        region = _selected(soup, selected_dom_path)
        added = []
        for tree in _toc_trees(region, page_url, artifact_id):
            for node, _ in _walk(tree):
                if node.url and node.url != page_url:
                    anchor = soup.select_one(node.evidence.dom_path)
                    if anchor is None or _child_catalog_chrome(anchor, region):
                        continue
                    self.positive_edges.setdefault(page_url, {})[node.url] = node.evidence
                    added.append(node.url)
        return list(dict.fromkeys(added))

    async def probe_ancestors(self, fetch: Callable[[str, int], Awaitable[Any]],
                              save_raw: Callable[[bytes, dict[str, Any]], Any], *,
                              local_roots: Sequence[Path] = ()) -> MetadataProbeReport:
        if self.probe_report.fetch_count or self.probe_report.gaps:
            return self.probe_report
        report = self.probe_report
        document = self.documents[self.entry_url]
        visited = {self.entry_url}
        for _ in range(MAX_METADATA_FETCHES):
            targets = {edge.target_url for edge in document.up}
            if len(targets) != 1:
                break
            target = next(iter(targets))
            relation = next(edge for edge in document.up if edge.target_url == target)
            if target in visited:
                report.gaps.append({"status": "scope_metadata_cycle", "url": target})
                break
            if not _safe_metadata_url(target, self.entry_url, local_roots):
                report.gaps.append({"status": "scope_metadata_origin_denied", "url": target})
                break
            visited.add(target)
            report.fetch_count += 1
            limit = min(MAX_METADATA_BYTES, MAX_METADATA_TOTAL_BYTES - report.total_bytes)
            try:
                response = await fetch(target, limit)
                data = response["data"] if isinstance(response, Mapping) else response.data
                final_url = _url(response.get("final_url", target) if isinstance(response, Mapping) else response.final_url)
                media_type = response.get("media_type", "text/html") if isinstance(response, Mapping) else getattr(response, "media_type", "text/html")
                if len(data) > limit:
                    report.gaps.append({"status": "scope_metadata_byte_budget", "url": target, "returned_bytes": len(data)})
                    break
                report.total_bytes += len(data)
                if not final_url or not _safe_metadata_url(final_url, self.entry_url, local_roots):
                    report.gaps.append({"status": "scope_metadata_redirect_denied", "url": target, "final_url": final_url})
                    break
                if str(media_type).split(";", 1)[0].strip().lower() not in {"text/html", "application/xhtml+xml"}:
                    report.gaps.append({"status": "scope_metadata_not_html", "url": target})
                    break
                meta = {"url": target, "final_url": final_url, "sha256": hashlib.sha256(data).hexdigest(),
                        "byte_count": len(data), "parent_relation": relation.as_dict(), "purpose": "publisher_scope_metadata_only"}
                capture_id = save_raw(data, meta)
                if inspect.isawaitable(capture_id):
                    capture_id = await capture_id
                if not isinstance(capture_id, str) or not capture_id:
                    raise ValueError("scope metadata persistence did not return an artifact ID")
                report.captures.append({**meta, "artifact_id": capture_id})
                soup = parse_html(_decode(data))
                document = _facts(soup, final_url, capture_id)
                self.documents[final_url] = document
                found = False
                for tree in _toc_trees(soup, final_url, capture_id):
                    positions: dict[str, list[tuple[tuple[int, ...], Evidence]]] = {}
                    for node, position in _walk(tree):
                        if node.url:
                            positions.setdefault(node.url, []).append((position, node.evidence))
                    roots = positions.get(self.entry_url, [])
                    roots = [(path, proof) for path, proof in roots if not any(len(other) < len(path) and path[:len(other)] == other for other, _ in roots)]
                    if len(roots) != 1:
                        continue
                    positions[self.entry_url] = roots
                    positions.setdefault(final_url, []).append(((), Evidence(final_url, capture_id, _path(soup), "ancestor_toc_document", final_url)))
                    self.publisher_indexes.append({"document_url": final_url, "artifact_id": capture_id,
                                                   "positions": positions, "tree": tree})
                    found = True
                if found:
                    break
                report.gaps.append({"status": "scope_metadata_entry_not_located", "url": final_url})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if isinstance(exc, RuntimeError) and str(exc) == "run_cancelled":
                    raise
                report.gaps.append({"status": "scope_metadata_fetch_failed", "url": target, "error_type": type(exc).__name__, "message": str(exc)[:300]})
                break
        return report

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": "html-guided-ownership.v1", "entry_url": self.entry_url, "strong_boundary": self.strong,
                "seed_urls": {url: [item.as_dict() for item in proof] for url, proof in self.seeds.items()},
                "positive_child_edges": {parent: {url: proof.as_dict() for url, proof in children.items()} for parent, children in self.positive_edges.items()},
                "publisher_documents": {url: facts.as_dict() for url, facts in self.documents.items()},
                "publisher_indexes": [{"document_url": index["document_url"], "artifact_id": index["artifact_id"],
                                       "trees": [node.as_dict() for node in index["tree"]]} for index in self.publisher_indexes],
                "metadata_probe": self.probe_report.as_dict()}

    def model_context(self) -> dict[str, Any]:
        context = {"entry_url": self.entry_url[:1024], "strong_boundary": self.strong,
                   "seed_count": len(self.seeds), "publisher_index_count": len(self.publisher_indexes),
                   "seed_url_samples": [url[:256] for url in list(self.seeds)[:8]],
                   "policy": "Explicit root catalogs and proven authored descendants establish ownership, including cross-host resources. Inline relevance does not. Strong publisher-owner conflicts are excluded by the host; unresolved ownership stays model-guided and does not change the entry identity."}
        while len(json.dumps(context, ensure_ascii=False).encode("utf-8")) > 4096 and context["seed_url_samples"]:
            context["seed_url_samples"].pop()
        if len(json.dumps(context, ensure_ascii=False).encode("utf-8")) > 4096:
            context["entry_url"] = self.entry_url[:256]
        return context
