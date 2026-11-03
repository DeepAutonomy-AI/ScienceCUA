"""Neutral source observations and validation for Agent-written extraction scripts.

No headings, containers, font sizes or page boundaries become semantic parts here.
Scripts group immutable text/image handles; the host re-derives their contents.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import posixpath
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse, urlunparse

import fitz
import httpx
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from pydantic import ValidationError

from .adaptive_source_contracts import ExtractionOutput, UnitRef
from .html_navigation import classify_html_link, html_base_url
from .models import EvidencePlan, EvidencePlanNode, EvidenceSourceRef


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class SourceContext:
    """Available to extract(ctx); raw observations and cached acquisition only.

    units('r1') returns dictionaries with unit_key, kind, text, ordinal, locator.
    ref(unit) returns a contract reference. select_html selects text/image units
    below a CSS selector. fetch_html caches an in-scope page under the shared
    page budget. image_path/render_pdf_page expose source images for observation.
    """

    def __init__(self, assignment_path: str | Path, *, replay: bool = False):
        self.assignment_path = Path(assignment_path).resolve()
        self.root = self.assignment_path.parent
        self.assignment = json.loads(self.assignment_path.read_text(encoding="utf-8"))
        self.replay = replay
        self._indices: dict[str, list[dict[str, Any]]] = {}
        self._units_by_key: dict[str, dict[str, dict[str, Any]]] = {}
        self._html_bases: dict[str, str] = {}

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Source resource path leaves the extraction workspace")
        return path

    def state(self) -> dict[str, Any]:
        return json.loads((self.root / "resources.json").read_text(encoding="utf-8"))

    def resources(self) -> list[dict[str, Any]]:
        return self.state()["resources"]

    def resource(self, key: str) -> dict[str, Any]:
        return next(item for item in self.resources() if item["resource_key"] == key)

    def raw_path(self, key: str = "r1") -> Path:
        resource = self.resource(key)
        path = self._path(resource["path"])
        if digest(path.read_bytes()) != resource["sha256"]:
            raise ValueError(f"Raw source changed: {key}")
        if key == "r1" and resource["sha256"] != self.assignment["root_sha256"]:
            raise ValueError("Root source differs from its acquired snapshot")
        return path

    @staticmethod
    def ref(unit: dict[str, Any]) -> dict[str, str]:
        return {"resource_key": unit["resource_key"], "unit_key": unit["unit_key"]}

    def units(self, resource_key: str = "r1") -> list[dict[str, Any]]:
        if resource_key not in self._indices:
            resource = self.resource(resource_key)
            path = self.raw_path(resource_key)
            units = self._html_units(resource, path) if resource["modality"] == "html" else self._pdf_units(resource, path)
            self._indices[resource_key] = units
            self._units_by_key[resource_key] = {unit['unit_key']: unit for unit in units}
        return self._indices[resource_key]

    def unit(self, resource_key: str, unit_key: str) -> dict[str, Any]:
        self.units(resource_key)
        return self._units_by_key[resource_key][unit_key]

    def _html_units(self, resource: dict[str, Any], path: Path) -> list[dict[str, Any]]:
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        # Whole body, not just the first main/article. No navigation is discarded.
        root = soup.body or soup
        result: list[dict[str, Any]] = []

        def walk(node: Any, node_path: str, ancestors: list[dict[str, Any]]) -> None:
            if isinstance(node, Comment):
                return
            if isinstance(node, Tag):
                info = {"tag": node.name, "attrs": dict(node.attrs), "path": node_path}
                ancestors = [*ancestors, info]
                if node.name == "img" and node.get("src"):
                    add("image", "", node_path, ancestors, src=str(node["src"]), alt=str(node.get("alt") or ""))
                for index, child in enumerate(node.contents):
                    walk(child, f"{node_path}/{index}", ancestors)
            elif isinstance(node, NavigableString):
                text = str(node).replace("\x00", "")
                if text.strip() or (text and any(item['tag'] in {'pre', 'code'} for item in ancestors)):
                    add("text", text, node_path, ancestors)

        def add(kind: str, text: str, node_path: str, ancestors: list[dict[str, Any]], **extra: Any) -> None:
            result.append({
                "resource_key": resource["resource_key"], "unit_key": f"u{len(result) + 1}",
                "kind": kind, "text": text, "ordinal": len(result),
                "locator": {"url": resource.get("url"), "dom_path": node_path, "ancestors": ancestors, **extra},
            })

        walk(root, "body" if soup.body else "document", [])
        return result

    def select_html(self, resource_key: str, selector: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(self.raw_path(resource_key).read_bytes(), "html.parser")
        root = soup.body or soup
        paths: dict[int, str] = {}

        def walk(node: Any, path: str) -> None:
            if isinstance(node, Tag):
                paths[id(node)] = path
                for index, child in enumerate(node.contents):
                    walk(child, f"{path}/{index}")

        walk(root, "body" if soup.body else "document")
        selected = {paths[id(tag)] for tag in soup.select(selector) if id(tag) in paths}
        return [unit for unit in self.units(resource_key) if any(
            unit["locator"]["dom_path"] == path or unit["locator"]["dom_path"].startswith(path + "/")
            for path in selected
        )]

    def _pdf_units(self, resource: dict[str, Any], path: Path) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        with fitz.open(path) as document:
            for page_index, page in enumerate(document):
                for block_index, block in enumerate(page.get_text("dict", sort=True)["blocks"]):
                    locator: dict[str, Any] = {"page": page_index + 1, "bbox": list(block["bbox"]), "block_index": block_index}
                    if block["type"] == 0:
                        lines = block.get("lines", [])
                        text = "\n".join("".join(span["text"] for span in line["spans"]) for line in lines).replace("\x00", "")
                        if not text.strip():
                            continue
                        locator["spans"] = [{"size": span["size"], "font": span["font"], "flags": span["flags"]} for line in lines for span in line["spans"]]
                        unit = {"kind": "text", "text": text}
                    elif block["type"] == 1:
                        unit = {"kind": "image", "text": "", "image_sha256": digest(block["image"]),
                                "image_base64": base64.b64encode(block["image"]).decode("ascii"),
                                "media_type": f"image/{'jpeg' if block['ext'] == 'jpg' else block['ext']}",
                                "width": block["width"], "height": block["height"]}
                    else:
                        continue
                    result.append({"resource_key": resource["resource_key"], "unit_key": f"u{len(result) + 1}",
                                   "ordinal": len(result), "locator": locator, **unit})
        return result

    def pdf_outline(self, resource_key: str = "r1") -> list[list[Any]]:
        """Legacy API: validate provenance without exposing native bookmarks."""
        self.raw_path(resource_key)
        return []

    def render_pdf_page(self, page: int, resource_key: str = "r1") -> str:
        with fitz.open(self.raw_path(resource_key)) as document:
            target = self.root / "observations" / f"{resource_key}-page-{page}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            document[page - 1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(target)
        return str(target)

    def normalize_url(self, url: str, resource_key: str = "r1") -> str | None:
        resource = self.resource(resource_key)
        base = resource.get("url")
        if not base or not url or url.startswith(("#", "javascript:", "mailto:", "data:")):
            return None
        if resource.get("modality") == "html":
            if resource_key not in self._html_bases:
                soup = BeautifulSoup(self.raw_path(resource_key).read_bytes(), "html.parser")
                self._html_bases[resource_key] = html_base_url(soup, base)
            base = self._html_bases[resource_key]
        parsed = urlparse(urljoin(base, url))
        if parsed.scheme not in {"http", "https", "file"}:
            return None
        return urlunparse(parsed._replace(fragment=""))

    def in_scope(self, url: str) -> bool:
        root = urlparse(self.assignment.get("origin_url") or "")
        candidate = urlparse(url)
        directory = posixpath.dirname(root.path or "/").rstrip("/") + "/"
        normalized_path = posixpath.normpath(unquote(candidate.path))
        return bool(root.scheme) and root.scheme == candidate.scheme and root.netloc.casefold() == candidate.netloc.casefold() and (
            normalized_path == root.path or normalized_path.startswith(directory)
        )

    def links(self, resource_key: str = "r1") -> list[dict[str, Any]]:
        soup = BeautifulSoup(self.raw_path(resource_key).read_bytes(), "html.parser")
        result = []
        for tag in soup.find_all("a", href=True):
            role, reason = classify_html_link(tag)
            result.append({"label": tag.get_text(" ", strip=True), "href": str(tag["href"]),
                           "url": self.normalize_url(str(tag["href"]), resource_key),
                           "link_role": role, "role_reason": reason})
        return result

    def fetch_html(self, url: str, resource_key: str = "r1") -> str | None:
        normalized = self.normalize_url(url, resource_key)
        if normalized is None or not self.in_scope(normalized):
            raise ValueError("Requested page is outside the assigned source scope")
        state = self.state()
        for resource in state["resources"]:
            if normalized in {resource.get("url"), resource.get("requested_url")}:
                return resource["resource_key"]
        previous = next((item for item in state["requests"] if item["url"] == normalized), None)
        if previous:
            return None
        if self.replay:
            raise ValueError("Replay requested a page absent from the captured acquisition snapshot")
        if len(state["resources"]) - 1 >= self.assignment["max_html_pages"]:
            state["requests"].append({"url": normalized, "status": "budget_exhausted"})
            write_json(self.root / "resources.json", state)
            return None
        try:
            data, media_type, final_url = self._download(normalized)
            if not self.in_scope(final_url):
                raise ValueError("Redirect leaves assigned source scope")
            if "html" not in media_type and not data.lstrip().lower().startswith((b"<!doctype html", b"<html")):
                raise ValueError("Requested resource is not HTML")
            key = f"r{len(state['resources']) + 1}"
            relative = f"input/{key}.html"
            self._path(relative).write_bytes(data)
            state["resources"].append({"resource_key": key, "path": relative, "sha256": digest(data),
                                       "modality": "html", "url": final_url, "requested_url": normalized})
            state["requests"].append({"url": normalized, "status": "succeeded"})
            write_json(self.root / "resources.json", state)
            return key
        except Exception as exc:
            state["requests"].append({"url": normalized, "status": "fetch_failed", "message": str(exc)[:1000]})
            write_json(self.root / "resources.json", state)
            return None

    def _download(self, url: str) -> tuple[bytes, str, str]:
        maximum = self.assignment["max_source_bytes"]
        if urlparse(url).scheme == "file":
            path = Path(unquote(urlparse(url).path.lstrip("/") if os.name == 'nt' else urlparse(url).path))
            if path.stat().st_size > maximum:
                raise ValueError("Source resource exceeds the byte budget")
            return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "text/html", url
        with httpx.stream("GET", url, follow_redirects=True, timeout=self.assignment["fetch_timeout_seconds"]) as response:
            response.raise_for_status()
            if not self.in_scope(str(response.url)):
                raise ValueError("Redirect leaves assigned source scope")
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > maximum:
                    raise ValueError("Source resource exceeds the byte budget")
            return bytes(data), response.headers.get("content-type", ""), str(response.url)

    def image_bytes(self, unit: dict[str, Any]) -> tuple[bytes, str]:
        if unit.get("image_base64"):
            return base64.b64decode(unit["image_base64"]), unit["media_type"]
        src = unit["locator"]["src"]
        if src.startswith("data:image/"):
            header, content = src.split(",", 1)
            data = base64.b64decode(content) if ";base64" in header else unquote(content).encode()
            return data, header[5:].split(";")[0]
        url = self.normalize_url(src, unit["resource_key"])
        if url is None or not self.in_scope(url):
            raise ValueError("Image is outside the assigned source scope; record an explicit gap/exclusion")
        stem = digest(url.encode())
        path = self.root / "assets" / f"{stem}.bin"
        meta = path.with_suffix(".json")
        if not path.exists():
            if self.replay:
                raise ValueError("Image is missing from the replay snapshot")
            data, media_type, _ = self._download(url)
            if not media_type.startswith("image/"):
                raise ValueError("Image URL did not return image content")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            write_json(meta, {"media_type": media_type, "sha256": digest(data)})
        info = json.loads(meta.read_text(encoding="utf-8"))
        data = path.read_bytes()
        if digest(data) != info["sha256"]:
            raise ValueError("Cached image changed")
        return data, info["media_type"]

    def image_path(self, ref: dict[str, Any]) -> str:
        unit = next(item for item in self.units(ref["resource_key"]) if item["unit_key"] == ref["unit_key"])
        data, media_type = self.image_bytes(unit)
        path = self.root / "observations" / f"{ref['resource_key']}-{ref['unit_key']}{mimetypes.guess_extension(media_type) or '.bin'}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def fingerprint(self) -> str:
        # Refresh unit contents from raw bytes when the host creates a new context.
        resources = self.resources()
        keys = [item['resource_key'] for item in resources]
        if keys.count('r1') != 1 or len(keys) != len(set(keys)):
            raise ValueError('Acquisition snapshot must retain one root and unique resource keys')
        if len(resources) > (1 + self.assignment['max_html_pages'] if self.assignment['modality'] == 'html' else 1):
            raise ValueError('Acquisition snapshot exceeds the resource budget')
        for item in resources:
            if item['modality'] != self.assignment['modality']:
                raise ValueError('Acquisition resource modality differs from the assignment')
            if item['resource_key'] != 'r1' and not self.in_scope(item.get('url') or ''):
                raise ValueError('Acquisition resource is outside the assigned source scope')
            self.raw_path(item["resource_key"])
        assets = []
        for meta in sorted((self.root / "assets").glob("*.json")):
            info = json.loads(meta.read_text(encoding="utf-8"))
            data = meta.with_suffix(".bin").read_bytes()
            if digest(data) != info["sha256"]:
                raise ValueError("Cached image changed")
            assets.append({"key": meta.stem, **info})
        return digest(canonical_bytes({"resources": self.state(), "assets": assets}))

    def acquisition_gaps(self) -> list[dict[str, Any]]:
        state = self.state()
        gaps = [{"code": item["status"], "url": item["url"], "message": item.get("message") or item["status"]}
                for item in state["requests"] if item["status"] != "succeeded"]
        acquired = {url for item in state["resources"] for url in (item.get("url"), item.get("requested_url")) if url}
        requested = {item["url"] for item in state["requests"]}
        linked = {item["url"] for resource in state["resources"] if resource["modality"] == "html"
                  for item in self.links(resource["resource_key"]) if item["url"] and self.in_scope(item["url"])}
        # Asset links are not document coverage candidates.
        pending = sorted(url for url in linked - acquired - requested if Path(urlparse(url).path).suffix.lower() not in {
            ".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".zip", ".gz", ".pdf"
        })
        if pending:
            gaps.append({"code": "unfetched_links", "message": f"{len(pending)} in-scope linked resources were not fetched.",
                         "count": len(pending), "sample": pending[:20]})
        for resource in state["resources"]:
            if resource["modality"] == "pdf" and not any(unit["kind"] == "text" for unit in self.units(resource["resource_key"])):
                gaps.append({"code": "pdf_text_unavailable", "resource_key": resource["resource_key"],
                             "message": "PDF has no extractable text; OCR output cannot yet satisfy exact source-span validation."})
        return gaps


def validate_output(ctx: SourceContext, raw: Any) -> tuple[dict[str, Any], ExtractionOutput | None]:
    """Two independent coverage checks: raw units -> parts -> tree ownership."""
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    report: dict[str, Any] = {"schema_version": "adaptive-source-validation.v1", "valid": False,
                              "candidate_sha256": digest(canonical_bytes(raw)), "issues": issues, "warnings": warnings}

    def issue(code: str, message: str, **location: Any) -> None:
        issues.append({"severity": "error", "code": code, "message": message, **location})

    try:
        output = ExtractionOutput.model_validate(raw)
    except ValidationError as exc:
        for error in exc.errors(include_url=False):
            issue("invalid_output_schema", error["msg"], location=".".join(map(str, error["loc"])))
        return report, None
    if output.source_id != ctx.assignment["source_id"]:
        issue("source_scope_mismatch", "Result source_id differs from the assignment")
    try:
        report["input_snapshot_sha256"] = ctx.fingerprint()
        units = {(resource["resource_key"], unit["unit_key"]): unit for resource in ctx.resources() for unit in ctx.units(resource["resource_key"])}
    except Exception as exc:
        issue("source_snapshot_changed", str(exc))
        return report, output
    resource_order = {item["resource_key"]: index for index, item in enumerate(ctx.resources())}
    intervals: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    resolved_order: dict[str, list[tuple[int, int, int]]] = {}

    def resolve(ref: UnitRef, owner: str, expected_kind: str | None = None) -> tuple[int, int, int] | None:
        key = (ref.resource_key, ref.unit_key)
        unit = units.get(key)
        if unit is None:
            issue("unknown_source_unit", "Reference does not exist in acquired source", owner=owner, resource_key=key[0], unit_key=key[1])
            return None
        if expected_kind and unit["kind"] != expected_kind:
            issue("source_unit_kind_mismatch", f"Expected {expected_kind} reference", owner=owner, resource_key=key[0], unit_key=key[1])
            return None
        length = len(unit["text"]) if unit["kind"] == "text" else 1
        start, end = (0, length) if ref.start_offset is None else (ref.start_offset, ref.end_offset)
        if not 0 <= start < end <= length or (unit["kind"] == "image" and ref.start_offset is not None):
            issue("invalid_source_range", "Reference range is outside its source unit", owner=owner, resource_key=key[0], unit_key=key[1])
            return None
        intervals[key].append((start, end, owner))
        return resource_order[key[0]], unit["ordinal"], start

    keys = [part.part_key for part in output.parts]
    parts = {part.part_key: part for part in output.parts}
    if len(keys) != len(parts):
        issue("duplicate_part_key", "Each part_key must be unique")
    if len({part.ordinal for part in output.parts}) != len(output.parts):
        issue("duplicate_part_ordinal", "SourcePart ordinals must be unique")
    for part in output.parts:
        order = [value for ref in part.source_refs if (value := resolve(ref, part.part_key, "text")) is not None]
        resolved_order[part.part_key] = order
        if part.source_refs and len(order) == len(part.source_refs) and not source_text(ctx, part.source_refs).strip():
            issue("empty_text_part", "A text-bearing part must contain non-whitespace source text", part_key=part.part_key)
        if order != sorted(order):
            issue("source_refs_out_of_order", "References within a part must preserve original source order", part_key=part.part_key)
        if not part.source_refs and part.image_refs:
            issue("image_requires_text_anchor", "Attach images to a text-bearing part; record a gap for image-only content", part_key=part.part_key)
        for ref in part.image_refs:
            if resolve(ref, part.part_key, "image") is not None:
                try:
                    ctx.image_bytes(units[(ref.resource_key, ref.unit_key)])
                except Exception as exc:
                    issue("image_unavailable", str(exc), part_key=part.part_key)
        if part.parent_part_key is not None and part.parent_part_key not in parts:
            issue("unknown_part_parent", "Parent part does not exist", part_key=part.part_key)
        seen = {part.part_key}
        parent = part.parent_part_key
        while parent in parts:
            if parent in seen:
                issue("part_parent_cycle", "SourcePart parent graph contains a cycle", part_key=part.part_key)
                break
            seen.add(parent)
            parent = parts[parent].parent_part_key
    ordered_parts = sorted(output.parts, key=lambda item: item.ordinal)
    positions = [position for part in ordered_parts for position in resolved_order[part.part_key]]
    if positions != sorted(positions):
        issue("parts_out_of_source_order", "Part ordinals disagree with original source order")
    for index, exclusion in enumerate(output.exclusions):
        for ref in exclusion.source_refs:
            resolve(ref, f"exclusion:{index}")
    for key, unit in units.items():
        spans = sorted(intervals[key])
        cursor = 0
        for start, end, owner in spans:
            if start < cursor:
                issue("overlapping_source_ownership", "A source range is used or excluded more than once", resource_key=key[0], unit_key=key[1], owner=owner)
            missing = unit["text"][cursor:start].strip() if unit["kind"] == "text" else start > cursor
            if missing:
                issue("source_unit_unaccounted_for", "Source range is neither extracted nor explicitly excluded", resource_key=key[0], unit_key=key[1], start_offset=cursor, end_offset=start)
            cursor = max(cursor, end)
        length = len(unit["text"]) if unit["kind"] == "text" else 1
        if (unit["text"][cursor:].strip() if unit["kind"] == "text" else cursor < length):
            issue("source_unit_unaccounted_for", "Source range is neither extracted nor explicitly excluded", resource_key=key[0], unit_key=key[1], start_offset=cursor, end_offset=length)
    node_ownership: dict[str, list[str]] = defaultdict(list)
    children = {node.parent_node_key for node in output.tree_nodes if node.parent_node_key is not None}
    for node in output.tree_nodes:
        ordinals = []
        for key in node.part_keys:
            if key not in parts:
                issue("unknown_tree_part", "Tree node references a nonexistent part", node_key=node.node_key, part_key=key)
            else:
                node_ownership[key].append(node.node_key)
                if parts[key].source_refs:
                    ordinals.append(parts[key].ordinal)
        if ordinals != sorted(ordinals):
            issue("tree_parts_out_of_order", "Node part_keys must preserve source order", node_key=node.node_key)
        if not any(key in parts and parts[key].source_refs for key in node.part_keys) and node.node_key not in children:
            issue("empty_tree_leaf", "A leaf needs source text; an empty structural node needs children", node_key=node.node_key)
    for part in output.parts:
        if part.source_refs and len(node_ownership[part.part_key]) != 1:
            issue("tree_part_ownership", "Every text-bearing part must belong to exactly one tree node", part_key=part.part_key)
        if not part.source_refs and part.part_key not in {item.parent_part_key for item in output.parts}:
            issue("empty_structural_part", "An empty structural part needs child parts", part_key=part.part_key)
    try:
        # Reuse the existing shape validator without modifying downstream models.
        EvidencePlan(source_id=output.source_id, nodes=[EvidencePlanNode(
            node_key=node.node_key, title=node.title, title_origin=node.title_origin,
            node_type=node.node_type, parent_node_key=node.parent_node_key,
            source_refs=[EvidenceSourceRef(source_part_id=key) for key in node.part_keys
                         if key in parts and parts[key].source_refs],
        ) for node in output.tree_nodes])
    except Exception as exc:
        issue("invalid_tree_shape", str(exc)[:3000])
    text_chars = sum(len(unit["text"]) for unit in units.values())
    excluded_chars = sum(end - start for spans in intervals.values() for start, end, owner in spans if owner.startswith("exclusion:"))
    if text_chars and excluded_chars / text_chars > 0.2:
        warnings.append({"code": "large_exclusion", "message": "More than 20% of source characters are excluded; inspect substantive omissions."})
    if not any(part.source_refs for part in output.parts):
        issue("no_text_evidence", "No exact text is available for the existing Evidence Tree contract")
    report["gaps"] = [*ctx.acquisition_gaps(), *(gap.model_dump(exclude_none=True) for gap in output.gaps)]
    report["counts"] = {"resources": len(ctx.resources()), "source_units": len(units), "parts": len(output.parts),
                        "tree_nodes": len(output.tree_nodes), "excluded_characters": excluded_chars}
    report["valid"] = not issues
    report["input_snapshot_sha256"] = ctx.fingerprint()
    return report, output


def source_text(ctx: SourceContext, refs: list[UnitRef]) -> str:
    chunks = []
    for ref in refs:
        unit = ctx.unit(ref.resource_key, ref.unit_key)
        chunks.append(unit["text"] if ref.start_offset is None else unit["text"][ref.start_offset:ref.end_offset])
    return "\n".join(chunks)
