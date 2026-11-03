"""Lightweight, source-bound PDF chapter previews and exact selected regions.

Previewing reads printed navigation and text geometry only. It creates no SourceParts,
Evidence, images, or rendered page snapshots.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import re
from typing import Any

try:
    import pymupdf as fitz
except ImportError:  # Full-document recovery can still use alternate engines.
    fitz = None

from .filesystem import filesystem_path


def _key(value: str) -> str:
    return re.sub(r"[\W_]+", "", str(value).casefold())


def _lines(page) -> list[dict[str, Any]]:
    layout = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)
    return [{"text": "".join(str(span.get("text") or "") for span in line["spans"]),
             "bbox": list(line["bbox"]),
             "size": max((float(span.get("size") or 0) for span in line["spans"]), default=0),
             "bold": any(int(span.get("flags") or 0) & 16 for span in line["spans"])}
            for block in layout["blocks"] if block["type"] == 0 for line in block["lines"]]


def _heading_anchor(page, title: str, lines=None, destination_y=None) -> dict[str, Any] | None:
    lines = _lines(page) if lines is None else lines
    wanted = _key(title)
    found = []
    for index, line in enumerate(lines):
        for width in (1, 2, 3):
            chunk = lines[index:index + width]
            if not chunk:
                continue
            text = " ".join(item["text"] for item in chunk)
            if _key(text) == wanted:
                found.append({"page": page.number + 1,
                              "y": max(0.0, float(chunk[0]["bbox"][1]) - .25),
                              "heading_bbox": [min(item["bbox"][0] for item in chunk),
                                               min(item["bbox"][1] for item in chunk),
                                               max(item["bbox"][2] for item in chunk),
                                               max(item["bbox"][3] for item in chunk)],
                              "method": "matched_heading"})
    unique = {(item["page"], item["y"]): item for item in found}
    if len(unique) == 1:
        return next(iter(unique.values()))
    if unique and destination_y is not None:
        return min(unique.values(), key=lambda item: abs(item["y"] - destination_y))
    return None


def _scan_candidates(document) -> tuple[list[dict[str, Any]], str]:
    # Text-only scanning never reads native bookmarks or invokes a full parser.
    cache = {}
    printed = []
    for index in range(document.page_count):
        page = document[index]
        lines = cache[index] = _lines(page)
        if index < 24:
            for line in lines:
                match = re.match(r"^(.+?)\s*(?:\.\s*){3,}\s*(\d+)\s*$", line["text"])
                if match:
                    printed.append((match.group(1).strip(), int(match.group(2)), line["bbox"][0], index))
    chapter_entries = []
    seen_printed = set()
    for title, printed_page, indent, toc_index in printed:
        numbered = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+", title)
        if not _key(title) or (numbered and "." in numbered.group(1)):
            continue
        identity = (_key(title), printed_page, toc_index)
        if identity not in seen_printed:
            seen_printed.add(identity)
            chapter_entries.append((title, printed_page, indent, toc_index))
    if chapter_entries:
        minimum = min(item[2] for item in chapter_entries)
        chapter_entries = [item for item in chapter_entries if item[2] <= minimum + 5]
    resolved = []
    for title, printed_page, indent, toc_index in chapter_entries:
        possible = document.get_page_numbers(str(printed_page))
        if not possible and 1 <= printed_page <= document.page_count:
            possible = [printed_page - 1]
        matches = [_heading_anchor(document[index], title, cache[index]) for index in possible if index != toc_index]
        matches = [anchor for anchor in matches if anchor]
        if not matches:
            # Printed page numbers often omit front matter and PDF page labels.
            # Resolve the authored title itself rather than guessing an offset.
            matches = [_heading_anchor(document[index], title, lines) for index, lines in cache.items()
                       if index != toc_index]
            matches = [anchor for anchor in matches if anchor]
        if len(matches) == 1:
            resolved.append({"title": title, "start_anchor": matches[0], "indent": indent,
                             "source": "printed_toc"})
    if len(chapter_entries) >= 2:
        if len(resolved) != len(chapter_entries):
            # A few uniquely matched entries do not establish a complete chapter
            # set. Exposing them would let a selected range absorb omitted ones.
            return [], 'unresolved_printed_toc'
        return resolved, "printed_toc"

    font_weights = Counter()
    for lines in cache.values():
        for line in lines:
            if line['text'].strip() and not line['bold']:
                font_weights[round(line['size'], 1)] += len(line['text'])
    body_size = font_weights.most_common(1)[0][0] if font_weights else 11
    numbered_candidates, styled_candidates = [], []
    for index, lines in cache.items():
        for line in lines:
            text = line["text"].strip()
            if not text or len(text) > 180:
                continue
            numbered = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+\S|^(?:chapter|part)\s+[\wIVX]+\b", text, re.I)
            anchor = {"page": index + 1, "y": max(0.0, float(line['bbox'][1]) - .25),
                      "heading_bbox": line['bbox']}
            if numbered and (line['bold'] or line['size'] > body_size * 1.1):
                if not numbered.group(1) or '.' not in numbered.group(1):
                    numbered_candidates.append({'title': text,
                        'start_anchor': {**anchor, 'method': 'numbered_heading'}, 'source': 'layout_heading'})
            ratio = line['size'] / max(.1, body_size)
            if (len(text) <= 160 and len(text.split()) <= 18 and not re.search(r'[.!?;,]$', text)
                    and ratio >= 1.2 and (line['bold'] or ratio >= 1.5)
                    and not (len(_key(text)) < 3 or text.startswith('['))):
                styled_candidates.append({'title': text, '_font_size': round(line['size'], 1),
                    'start_anchor': {**anchor, 'method': 'styled_heading'}, 'source': 'layout_heading'})
    repeated = Counter(_key(item['title']) for item in numbered_candidates)
    candidates = [item for item in numbered_candidates if repeated[_key(item['title'])] == 1]
    if len(candidates) >= 2:
        return candidates, 'layout_headings'
    # A repeated, visibly larger heading style provides first-level boundaries
    # without importing bookmark hierarchy. Smaller subheadings remain inside
    # their enclosing chapter; a single large cover title is not a chapter set.
    repeated = Counter(_key(item['title']) for item in styled_candidates)
    styled_candidates = [item for item in styled_candidates if repeated[_key(item['title'])] == 1]
    style_counts = Counter(item['_font_size'] for item in styled_candidates)
    chapter_size = max((size for size, count in style_counts.items() if count >= 2), default=None)
    candidates = [{key: value for key, value in item.items() if key != '_font_size'}
                  for item in styled_candidates if item['_font_size'] == chapter_size]
    return candidates, "layout_headings"


def _clear_horizontal_boundary(document, anchor: dict[str, Any]) -> bool:
    """Do not cut a neighbouring text column at another chapter's heading."""
    heading = anchor.get('heading_bbox')
    if not heading:
        return True
    for line in _lines(document[anchor['page'] - 1]):
        box = line['bbox']
        nearby = box[1] < heading[3] + 16 and box[3] > heading[1] - 16
        beside = box[2] + 12 < heading[0] or heading[2] + 12 < box[0]
        if nearby and beside and len(_key(line['text'])) > 4:
            return False
    return True


def preview_pdf_outline(path: Path | str, source_id: str) -> dict[str, Any]:
    result = {"source_id": source_id, "modality": "pdf", "title": Path(path).stem,
              "outline_available": False, "sections": [], "warnings": []}
    if fitz is None:
        result['warnings'].append('The PDF geometry parser is unavailable; a chapter preview cannot be produced.')
        return result
    with fitz.open(filesystem_path(path)) as document:
        if document.needs_pass:
            result["warnings"].append("The PDF is password-protected; a chapter preview is unavailable.")
            return result
        result.update(title=document.metadata.get("title") or Path(path).stem, page_count=document.page_count)
        if document.page_count > 5000:
            result["warnings"].append("The PDF exceeds the supported page-count budget.")
            return result
        candidates, source = _scan_candidates(document)
        if len(candidates) < 2:
            if source == 'unresolved_printed_toc':
                result['warnings'].append('The printed contents could not be fully matched to unique chapter headings; process the complete PDF instead.')
            else:
                result["warnings"].append("No reliable authored chapter boundaries were found; process the complete PDF instead.")
            return result
        positions = [(item["start_anchor"]["page"], item["start_anchor"]["y"]) for item in candidates]
        if positions != sorted(set(positions)):
            result["warnings"].append("Chapter destinations overlap or are out of order; precise section selection is unavailable.")
            return result
        if any(not _clear_horizontal_boundary(document, item['start_anchor']) for item in candidates):
            result['warnings'].append('A chapter boundary shares a row with another text column; precise chapter regions are ambiguous.')
            return result
        end_of_document = {"page": document.page_count + 1, "y": 0.0, "method": "document_end"}
        for index, item in enumerate(candidates):
            start = item["start_anchor"]
            end = candidates[index + 1]["start_anchor"] if index + 1 < len(candidates) else end_of_document
            identity = f"{source_id}|{index}|{item['title']}|{start['page']}|{start['y']}"
            result["sections"].append({**item,
                "section_id": "pdfsec_" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                "page_start": start["page"],
                "page_end": end["page"] if end["y"] > 0 else end["page"] - 1,
                "end_anchor": dict(end)})
        result.update(outline_available=True, outline_source=source)
    return result


def selected_sections(scope: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not scope or scope.get("mode") != "selected":
        return []
    ids = scope.get("section_ids") or []
    sections = scope.get("sections") or []
    by_id = {item["section_id"]: item for item in sections}
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
        raise ValueError("Unknown or empty selected PDF chapter scope")
    return [item for item in sections if item["section_id"] in ids]


def page_regions(scope: dict[str, Any] | None, page_count: int) -> dict[int, list[tuple[float, float]]]:
    regions: dict[int, list[tuple[float, float]]] = {}
    for section in selected_sections(scope):
        start, end = section["start_anchor"], section["end_anchor"]
        start_page, end_page = int(start["page"]), int(end["page"])
        start_y, end_y = float(start["y"]), float(end["y"])
        if (not 1 <= start_page <= end_page <= page_count + 1 or min(start_y, end_y) < 0
                or not math.isfinite(start_y + end_y) or (start_page == end_page and end_y <= start_y)):
            raise ValueError("Invalid original PDF chapter boundary")
        for number in range(start_page, min(end_page, page_count) + 1):
            low = start_y if number == start_page else 0.0
            high = end_y if number == end_page else math.inf
            if high > low:
                regions.setdefault(number, []).append((low, high))
    for number, spans in regions.items():
        merged = []
        for low, high in sorted(spans):
            if merged and low <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
            else:
                merged.append((low, high))
        regions[number] = merged
    return regions


def filter_outline(outline: list[list[Any]], scope: dict[str, Any] | None) -> list[list[Any]]:
    """Legacy compatibility: native outlines are never used for selected scope."""
    return []


def scoped_snapshot(snapshot: dict[str, Any], regions: list[tuple[float, float]]) -> dict[str, Any]:
    def inside(box):
        return any(low <= float(box[1]) and float(box[3]) <= high for low, high in regions)
    lines = [line for line in snapshot["lines"] if inside(line["bbox"])]
    return {**snapshot, "lines": lines,
            "raw_text": "\n".join(line["text"] for line in lines),
            "links": [link for link in snapshot["links"] if inside(link["bbox"])],
            "scope_regions": [[low, min(high, snapshot["height"])] for low, high in regions]}
