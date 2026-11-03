"""Read bounded, data-only navigation published by compiled HTML help systems.

RoboHelp's topic HTML omits the sidebar. Its JSON TOC payloads are read without
executing JavaScript. This adapter depends on the generator format, never a site.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from urllib.parse import quote, unquote, urldefrag, urljoin, urlsplit, urlunsplit

from ..html_sourcepart.extractor import parse_html
from ..source_tools.text import _stable_id

MAX_NAVIGATION_FILES = 64
MAX_NAVIGATION_BYTES = 3_000_000
MAX_NAVIGATION_TOPICS = 2_000
MAX_NAVIGATION_DEPTH = 12


def _url(base: str, reference: str) -> str:
    parsed = urlsplit(urljoin(base, reference))
    if parsed.scheme not in {"http", "https", "file"} or parsed.username or parsed.password:
        raise ValueError("Navigation metadata must use an ordinary HTTP or local file URL")
    return urlunsplit(parsed._replace(path=quote(unquote(parsed.path), safe="/:")))


def robohelp_base(html: str, page_url: str) -> str | None:
    if "adobe robohelp" not in html[:30_000].casefold():
        return None
    soup = parse_html(html)
    generator = soup.find("meta", attrs={"name": re.compile("^generator$", re.I)})
    if not generator or "adobe robohelp" not in str(generator.get("content", "")).casefold():
        return None
    # The data directory is observed in the topic, not guessed from its host.
    for script in soup.find_all("script", src=True):
        address = _url(page_url, str(script["src"]))
        if "/whxdata/" in urlsplit(address).path:
            return address.rsplit("/whxdata/", 1)[0] + "/"
    return None


def parse_robohelp_toc(text: str) -> list[dict]:
    match = re.search(r"\bvar\s+toc\s*=\s*", text)
    if not match:
        raise ValueError("No data-only RoboHelp TOC assignment was found")
    value, _ = json.JSONDecoder().raw_decode(text[match.end():])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("The RoboHelp TOC must contain a JSON array of records")
    return value


async def recover_navigation(workspace, html: str, page_url: str) -> dict | None:
    base = robohelp_base(html, page_url)
    if base is None:
        return None
    store = workspace.store
    records, topics, gaps = [], [], []
    pending = deque([(base, "toc", [], 0)])
    visited = set()
    byte_count = 0
    allowed_origin = urlsplit(base).netloc.casefold()

    def permitted(address):
        parsed = urlsplit(address)
        if parsed.scheme == "file":
            if urlsplit(page_url).scheme != "file":
                return False
            original_parent = workspace.acquisition._file_url_to_path(page_url).resolve().parent
            candidate = workspace.acquisition._file_url_to_path(address).resolve()
            return candidate == original_parent or original_parent in candidate.parents
        return parsed.scheme in {"http", "https"} and parsed.netloc.casefold() == allowed_origin

    async def fetch(item):
        project, key, groups, depth = item
        url = _url(project, "whxdata/" + key + ".new.js")
        try:
            if not permitted(url):
                raise ValueError("Navigation metadata is outside its observed origin or local input directory")
            response = await asyncio.to_thread(workspace.acquisition._read_url, url, max_bytes=256_000)
            if not permitted(response.final_url):
                raise ValueError("Navigation metadata redirected outside its observed origin")
            entries = parse_robohelp_toc(workspace.acquisition._decode_html(response.data))
            artifact = store.put_artifact(response.data, kind="html_navigation_metadata", media_type="text/javascript",
                                          suffix=".js", source_id=workspace.manifest.source_id,
                                          metadata={"url": url, "final_url": response.final_url,
                                                    "format": "robohelp-json-toc"})
            workspace.allowed_artifacts.add(artifact.artifact_id)
            return item, url, artifact, entries, None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            workspace.acquisition._raise_if_cancelled()
            return item, url, None, [], {"url": url, "status": "navigation_metadata_failed", "reason": type(exc).__name__}

    while pending and len(records) < MAX_NAVIGATION_FILES and len(topics) < MAX_NAVIGATION_TOPICS:
        batch = []
        while pending and len(batch) < min(4, MAX_NAVIGATION_FILES - len(records)):
            item = pending.popleft()
            address = _url(item[0], "whxdata/" + item[1] + ".new.js")
            if address in visited:
                continue
            visited.add(address)
            batch.append(item)
        if not batch:
            continue
        for (project, key, groups, depth), url, artifact, entries, failure in await asyncio.gather(*(fetch(item) for item in batch)):
            records.append({"url": url, "artifact_id": artifact.artifact_id if artifact else None,
                            "status": "failed" if failure else "saved"})
            if failure:
                gaps.append(failure)
                continue
            byte_count += artifact.byte_count
            if byte_count > MAX_NAVIGATION_BYTES:
                gaps.append({"url": url, "status": "navigation_metadata_budget", "reason": "navigation_byte_limit"})
                pending.clear()
                break
            for index, entry in enumerate(entries):
                label = str(entry.get("name") or "").strip()[:350]
                kind = entry.get("type")
                if kind == "item" and isinstance(entry.get("url"), str):
                    try:
                        target = _url(project, entry["url"])
                    except ValueError:
                        continue
                    topics.append({"url": target, "label": label, "groups": groups,
                                   "description": str(entry.get("description") or "")[:400],
                                   "source": {"url": url, "artifact_id": artifact.artifact_id,
                                              "sha256": artifact.sha256, "json_pointer": f"/{index}"}})
                    if len(topics) >= MAX_NAVIGATION_TOPICS:
                        gaps.append({"url": url, "status": "navigation_metadata_budget", "reason": "navigation_topic_limit"})
                        break
                elif kind in {"book", "project"}:
                    if depth >= MAX_NAVIGATION_DEPTH:
                        gaps.append({"url": url, "status": "navigation_metadata_budget", "reason": "navigation_depth_limit"})
                        continue
                    next_project, next_key = project, str(entry.get("key") or "")
                    if kind == "project":
                        try:
                            next_project = _url(project, str(entry.get("ref") or "").rstrip("/") + "/")
                        except ValueError:
                            continue
                        next_key = "toc"
                        if not permitted(next_project):
                            gaps.append({"url": next_project, "status": "navigation_metadata_unsupported", "reason": "cross_origin_project"})
                            continue
                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", next_key):
                        gaps.append({"url": url, "status": "navigation_metadata_unsupported", "reason": "invalid_book_key"})
                        continue
                    pending.append((next_project, next_key, [*groups, label] if label else groups, depth + 1))
    if pending:
        gaps.append({"url": base, "status": "navigation_metadata_budget", "reason": "navigation_file_or_topic_limit"})
    result = {"format": "robohelp-json-toc", "base_url": base, "resources": records,
              "topics": topics, "gaps": gaps, "byte_count": byte_count}
    store.emit("html_guided_navigation_recovered", {"format": result["format"], "resource_count": len(records),
               "topic_count": len(topics), "gap_count": len(gaps)},
               source_id=workspace.manifest.source_id, agent="html_guided_navigation")
    return result


def add_navigation_links(snapshot: dict, navigation: dict) -> None:
    existing = {item["url"]: item for item in snapshot["links"]}
    page_url = snapshot["url"]
    for topic in navigation["topics"]:
        target, fragment = urldefrag(topic["url"])
        groups, label = topic["groups"], topic["label"]
        entry = {"label": label, "groups": list(groups), "href": topic["url"],
                 "source": dict(topic["source"])}
        landmark = {"tag": "external_navigation", "role": "doc-toc", "label": " / ".join(groups)[:120]}
        occurrence = {
            "label": label, "context": " / ".join([*groups, label]),
            "heading_path": [snapshot["title"], *groups], "disclosure_path": groups,
            "dom_path": None, "in_chrome": False, "authored_contents": True, "in_code": False,
            "content_root_ids": [], "landmark_context": [landmark],
            "navigation_source": dict(topic["source"]),
        }
        if target in existing:
            link = existing[target]
            entries = link.setdefault("external_navigation_entries", [])
            if entry in entries:
                continue
            entries.append(entry)
            # Keep the original DOM identity and occurrences. The independently
            # captured TOC adds authorship and group evidence, even when the HTML
            # occurrence was merely a header, sidebar or heading permalink.
            link.setdefault("occurrences", []).append(occurrence)
            link["occurrence_count"] = int(link.get("occurrence_count", 0)) + 1
            link["authored_contents"] = True
            link["legacy_role"] = "directory_entry"
            link["legacy_role_reason"] = "captured_compiled_help_toc"
            if not link.get("navigation_source"):
                link["navigation_source"] = dict(topic["source"])
                link["disclosure_path"] = list(groups)
            link["landmark_context"] = [
                *[item for item in link.get("landmark_context", [])
                  if item.get("tag") != "external_navigation"][:3], landmark,
            ]
            continue
        link = {
            "link_id": _stable_id("hg_link", page_url, target), "url": target, "href": topic["url"],
            "fragment": fragment or None, **occurrence,
            "kind": "document", "fetchable": True, "legacy_role": "directory_entry",
            "legacy_role_reason": "captured_compiled_help_toc", "occurrence_count": 1,
            "occurrences": [occurrence], "external_navigation_entries": [entry],
        }
        snapshot["links"].append(link)
        existing[target] = link
    snapshot["link_count"] = len(snapshot["links"])
    snapshot["external_navigation"] = {"format": navigation["format"], "resource_count": len(navigation["resources"]),
                                       "topic_count": len(navigation["topics"]), "gap_count": len(navigation["gaps"])}
