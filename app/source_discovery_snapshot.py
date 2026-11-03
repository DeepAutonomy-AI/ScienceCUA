"""Bounded, source-addressable HTML facts for a model's discovery decision.

This is a page inspection, not a crawler or a root classifier. In particular,
navigation remains visible beside main-body facts: the model must distinguish a
chapter with a shared sidebar from the landing page of a useful document set.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from typing import Any
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


SNAPSHOT_VERSION = "html-discovery-snapshot.v2"
MAX_SNAPSHOT_BYTES = 28_000
_SKIP = {"script", "style", "noscript", "template"}
_HEAD_RELATIONS = {"home", "up", "index", "contents", "prev", "next", "canonical", "alternate"}


def _compact(value: str, limit: int = 200) -> str:
    return re.sub(r"\s+", " ", value.replace("\x00", "")).strip()[:limit]


def _safe_url(value: str, base: str) -> str | None:
    try:
        value = urljoin(base, value.strip())
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or any(c.isspace() or ord(c) < 32 for c in value)):
            return None
        parsed.port
        return value if len(value) <= 4000 else None
    except (ValueError, TypeError):
        return None


def _path(node: Tag) -> str:
    pieces = []
    for item in (node, *node.parents):
        if not isinstance(item, Tag) or item.name == "[document]":
            break
        index = 1 + sum(isinstance(sibling, Tag) and sibling.name == item.name
                        for sibling in item.previous_siblings)
        pieces.append(f"{item.name}:nth-of-type({index})")
    return " > ".join(reversed(pieces))


def _hidden(node: Tag) -> bool:
    for owner in (node, *node.parents):
        if not isinstance(owner, Tag):
            continue
        style = re.sub(r"\s+", "", str(owner.get("style") or "").lower())
        if (owner.has_attr("hidden") or str(owner.get("aria-hidden") or "").lower() == "true"
                or "display:none" in style or "visibility:hidden" in style):
            return True
    return False


def _text(node: Tag, *, main_only: bool = False) -> str:
    values = []
    for child in node.descendants:
        if not isinstance(child, NavigableString) or isinstance(child, Comment):
            continue
        parent = child.parent
        if not isinstance(parent, Tag) or _hidden(parent):
            continue
        owners = (parent, *parent.parents)
        if any(owner.name in _SKIP for owner in owners if isinstance(owner, Tag)):
            continue
        if main_only and any(
            owner.name in {"nav", "aside", "header", "footer"}
            or str(owner.get("role") or "").lower() in {"navigation", "complementary", "banner", "contentinfo"}
            for owner in owners if isinstance(owner, Tag) and owner is not node
        ):
            continue
        values.append(str(child))
    return re.sub(r"\s+", " ", " ".join(values).replace("\x00", "")).strip()


def _outer(nodes: list[Tag]) -> list[Tag]:
    identities = {id(node) for node in nodes}
    return [node for node in nodes if not any(id(owner) in identities for owner in node.parents)]


def _main_roots(soup: BeautifulSoup) -> tuple[list[Tag], str]:
    for selector, method in (("main, [role=main]", "main_landmarks"), ("article", "article_landmarks")):
        roots = _outer(soup.select(selector))
        if roots:
            return roots, method
    return [soup.body or soup], "body_fallback"


def _within(node: Tag, roots: list[Tag]) -> bool:
    identities = {id(root) for root in roots}
    return any(id(owner) in identities for owner in (node, *node.parents))


def _region(node: Tag, roots: list[Tag]) -> tuple[str, str]:
    for owner in (node, *node.parents):
        if not isinstance(owner, Tag):
            continue
        role = str(owner.get("role") or "").lower()
        if owner.name == "nav" or role == "navigation":
            return "navigation", _compact(str(owner.get("aria-label") or owner.get("id") or ""), 100)
        if owner.name == "aside" or role == "complementary":
            return "sidebar", _compact(str(owner.get("aria-label") or owner.get("id") or ""), 100)
        if owner.name == "header" or role == "banner":
            return "header", _compact(str(owner.get("aria-label") or owner.get("id") or ""), 100)
        if owner.name == "footer" or role == "contentinfo":
            return "footer", _compact(str(owner.get("aria-label") or owner.get("id") or ""), 100)
        if owner.name == "head":
            return "head", ""
    return ("main" if _within(node, roots) else "body"), ""


def _group(node: Tag) -> str:
    labels = []
    for owner in node.parents:
        if not isinstance(owner, Tag) or owner.name not in {"li", "dt", "dd"}:
            continue
        pieces = []
        for child in owner.children:
            if isinstance(child, Tag) and child.name in {"ul", "ol", "dl"}:
                break
            pieces.append(child.get_text(" ", strip=True) if isinstance(child, Tag) else str(child))
        value = _compact(" ".join(pieces), 80)
        if value:
            labels.append(value)
        if len(labels) == 3:
            break
    return " / ".join(reversed(labels))[:180]


def _context(node: Tag) -> str:
    owner = node.find_parent(["p", "li", "dt", "dd", "td", "th", "summary"])
    return _compact(_text(owner or node), 180)


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def build_html_discovery_snapshot(data: bytes, *, url: str,
                                  max_bytes: int = MAX_SNAPSHOT_BYTES,
                                  capture_complete: bool = True,
                                  pdf_links_out: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Return real, stable page facts without deciding whether its scope is good.

    L/H IDs use source-DOM order and survive selection/truncation. Explicitly
    hidden links remain observed facts with a flag; hidden prose and script or
    comment text cannot become the main-body preview. CSS is not executed.
    """
    if max_bytes < 4096:
        raise ValueError("HTML discovery snapshot budget must be at least 4096 bytes")
    soup = BeautifulSoup(data, "html.parser")
    for comment in soup.find_all(string=lambda item: isinstance(item, Comment)):
        comment.extract()
    for node in list(soup.find_all(_SKIP)):
        node.decompose()
    base = url
    if (base_node := soup.find("base", href=True)) is not None:
        base = _safe_url(str(base_node["href"]), url) or url
    roots, selection = _main_roots(soup)
    main_text = " ".join(_text(root, main_only=True) for root in roots).strip()
    title = _compact(_text(soup.title), 500) if soup.title else ""
    embedded_resources = []
    embedded_total = 0
    for index, node in enumerate(soup.find_all(["frame", "iframe"]), 1):
        target = _safe_url(str(node.get("src") or ""), base) if node.get("src") else None
        if not target:
            continue
        embedded_total += 1
        if len(embedded_resources) >= 12:
            continue
        embedded_resources.append({"evidence_id": f"F{index}", "url": target,
            "text": _compact(str(node.get("title") or node.get("name") or ""), 160),
            "element": node.name, "source_attribute": "src", "dom_path": _path(node),
            "hidden": _hidden(node)})
    headings = []
    for index, node in enumerate(soup.find_all(re.compile(r"^h[1-6]$")), 1):
        region, label = _region(node, roots)
        headings.append({"evidence_id": f"H{index}", "level": int(node.name[1]),
                         "text": _compact(node.get_text(" ", strip=True), 240), "dom_path": _path(node),
                         "region": region, "within_main": _within(node, roots),
                         "hidden": _hidden(node), **({"landmark_label": label} if label else {})})

    # Pick across regions so a large sidebar cannot displace the entire body,
    # while a long chapter directory cannot erase actual home/up links.
    buckets: dict[str, deque] = defaultdict(deque)
    all_links, unsafe_links = 0, 0
    for node in soup.find_all(["a", "area", "link", "option"]):
        attribute = "value" if node.name == "option" else "href"
        if not node.has_attr(attribute):
            continue
        value = str(node.get(attribute) or "").strip()
        relations = [str(item).lower() for item in node.get("rel", [])][:8]
        if node.name == "link" and not _HEAD_RELATIONS.intersection(relations):
            continue
        # An arbitrary select value can be a JS-only version token; only
        # URL-shaped values are offered as navigable resources.
        if node.name == "option" and not value.startswith(("http://", "https://", "/", "./", "../")):
            continue
        target = _safe_url(value, base)
        if target is None:
            unsafe_links += 1
            continue
        all_links += 1
        if pdf_links_out is not None and len(pdf_links_out) < 24 and '.pdf' in target.casefold():
            pdf_links_out.append({'url': target, 'title': _compact(node.get_text(" ", strip=True), 500)})
        region, _ = _region(node, roots)
        bucket = "head" if node.name == "link" else region
        buckets[bucket].append((all_links, node, target, attribute, relations))
    chosen = []
    for _ in range(min(12, len(buckets["head"]))):
        chosen.append(buckets["head"].popleft())
    cycle = ("main", "main", "navigation", "sidebar", "header", "body", "footer", "head")
    while len(chosen) < 120 and any(buckets.values()):
        for bucket in cycle:
            if buckets[bucket] and len(chosen) < 120:
                chosen.append(buckets[bucket].popleft())
    links = []
    for index, node, target, attribute, relations in chosen:
        region, label = _region(node, roots)
        record = {"evidence_id": f"L{index}", "url": target,
                  "text": _compact(node.get_text(" ", strip=True) or str(node.get("aria-label") or node.get("title") or ""), 160),
                  "dom_path": _path(node), "region": region, "within_main": _within(node, roots),
                  "hidden": _hidden(node), "source_attribute": attribute, "relations": relations,
                  "context": _context(node), "group": _group(node)}
        if label:
            record["landmark_label"] = label
        links.append(record)
    controls = []
    for index, node in enumerate(soup.find_all("select")[:6], 1):
        options = node.find_all("option")
        controls.append({"control_id": f"S{index}", "dom_path": _path(node),
                         "label": _compact(str(node.get("aria-label") or node.get("name") or node.get("id") or ""), 100),
                         "options": [{"text": _compact(_text(option), 80),
                                      "value": str(option.get("value") or "")[:160],
                                      "selected": option.has_attr("selected")} for option in options[:12]],
                         "total_options": len(options)})
    chosen_headings = [item for item in headings if item["region"] == "main"][:28]
    chosen_headings += [item for item in headings if item not in chosen_headings][:12]
    snapshot = {"schema_version": SNAPSHOT_VERSION, "url": url, "title": title,
                "title_evidence_id": "T1" if title else None,
                "base_url": base, "html_language": str((soup.html or soup).get("lang") or "")[:100],
                "main_content": {"evidence_id": "P1", "selection_method": selection,
                                 "dom_paths": [_path(root) for root in roots[:6]],
                                 "region_count": len(roots), "preview": main_text[:2400],
                                 "visible_text_chars": len(main_text)},
                "headings": chosen_headings, "links": links, "select_controls": controls,
                "embedded_resources": embedded_resources,
                "access_signals": {
                    "visible_password_inputs": sum(1 for node in soup.find_all("input")
                        if str(node.get("type") or "").lower() == "password" and not _hidden(node)),
                    "main_password_inputs": sum(1 for node in soup.find_all("input")
                        if str(node.get("type") or "").lower() == "password" and not _hidden(node)
                        and _within(node, roots)),
                    "visible_forms": sum(1 for node in soup.find_all("form") if not _hidden(node)),
                },
                "truncation": {"capture_complete": capture_complete,
                               "headings_total": len(headings), "headings_omitted": 0,
                               "links_total": all_links, "links_omitted": 0,
                               "unsafe_or_non_http_links_omitted": unsafe_links,
                               "select_controls_total": len(soup.find_all("select")),
                               "embedded_resources_total": embedded_total,
                               "embedded_resources_omitted": 0,
                               "main_preview_truncated": len(main_text) > 2400,
                               "byte_budget": max_bytes},
                "interpretation": "Observed HTML structure only. Navigation can be shared by many chapters. IDs identify source facts, not proof of a complete manual or suitable teaching scope. No CSS/JavaScript rendering was performed."}
    if base != url:
        snapshot["base_href_dom_path"] = _path(base_node)

    def counts() -> None:
        snapshot["truncation"].update(headings_omitted=len(headings) - len(snapshot["headings"]),
                                      links_omitted=all_links - len(snapshot["links"]),
                                      embedded_resources_omitted=embedded_total - len(snapshot["embedded_resources"]))

    counts()
    # Remove complete facts rather than truncating an address or its locator.
    # The raw response is persisted by the probe for every omitted occurrence.
    while _size(snapshot) > max_bytes:
        if snapshot["links"]:
            snapshot["links"].pop()
        elif snapshot["embedded_resources"]:
            snapshot["embedded_resources"].pop()
        elif len(snapshot["headings"]) > 1:
            snapshot["headings"].pop()
        elif snapshot["select_controls"]:
            snapshot["select_controls"].pop()
        elif len(snapshot["main_content"]["preview"]) > 200:
            snapshot["main_content"]["preview"] = snapshot["main_content"]["preview"][:200]
            snapshot["truncation"]["main_preview_truncated"] = True
        elif snapshot["headings"]:
            snapshot["headings"].pop()
        else:
            raise ValueError("HTML page identity exceeds discovery snapshot budget")
        counts()
    snapshot["truncation"]["select_controls_omitted"] = (
        snapshot["truncation"]["select_controls_total"] - len(snapshot["select_controls"]))
    # This integer adds at most a few bytes, accounted for by a fixed reserve.
    while _size(snapshot) > max_bytes and snapshot["links"]:
        snapshot["links"].pop()
        counts()
    return snapshot
