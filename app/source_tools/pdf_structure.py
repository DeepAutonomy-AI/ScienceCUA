"""Pure PDF typography, navigation, section, and image-geometry helpers."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..models import SourcePart
from .text import _stable_id


def _pdf_page_styled_blocks(page: Any) -> list[dict[str, Any]]:
    """Return text-block typography facts without changing extracted source text."""

    try:
        payload = page.get_text("dict", sort=True) or {}
    except TypeError:
        payload = page.get_text("dict") or {}
    records: list[dict[str, Any]] = []
    for block in payload.get("blocks", []):
        if int(block.get("type", 0)) != 0:
            continue
        rendered_lines: list[str] = []
        spans: list[dict[str, Any]] = []
        for line in block.get("lines", []):
            line_spans = [span for span in line.get("spans", []) if span.get("text")]
            line_text = "".join(str(span.get("text") or "") for span in line_spans)
            if line_text.strip():
                rendered_lines.append(line_text)
            spans.extend(line_spans)
        text = "\n".join(rendered_lines)
        if not text.strip():
            continue
        weights: Counter[float] = Counter()
        max_size = 0.0
        bold = False
        italic = False
        for span in spans:
            size = round(float(span.get("size") or 0.0), 2)
            chars = max(1, len(str(span.get("text") or "").strip()))
            weights[size] += chars
            max_size = max(max_size, size)
            flags = int(span.get("flags") or 0)
            font_name = str(span.get("font") or "").casefold()
            bold = bold or bool(flags & 16) or "bold" in font_name
            italic = italic or bool(flags & 2) or "italic" in font_name
        dominant_size = max(weights, key=lambda value: (weights[value], value)) if weights else 0.0
        records.append(
            {
                "text": text,
                "text_key": _pdf_text_key(text),
                "bbox": [round(float(value), 2) for value in block.get("bbox", (0, 0, 0, 0))],
                "font_size": dominant_size,
                "max_font_size": max_size,
                "bold": bold,
                "italic": italic,
                "line_count": max(1, len(rendered_lines)),
            }
        )
    return records


def _pdf_match_block_style(
    text: str, bbox: list[float], styled_blocks: list[dict[str, Any]]
) -> dict[str, Any]:
    key = _pdf_text_key(text)
    candidates = [item for item in styled_blocks if item.get("text_key") == key]
    if not candidates and key:
        candidates = [
            item
            for item in styled_blocks
            if item.get("text_key")
            and (
                key in str(item["text_key"])
                or str(item["text_key"]) in key
            )
        ]
    if not candidates:
        return {}

    def distance(item: dict[str, Any]) -> float:
        other = item.get("bbox") or [0, 0, 0, 0]
        return sum(abs(float(left) - float(right)) for left, right in zip(bbox, other))

    return min(candidates, key=distance)


def _pdf_body_font_size(page_contexts: list[dict[str, Any]]) -> float | None:
    weights: Counter[float] = Counter()
    for context in page_contexts:
        for block in context.get("blocks", []):
            size = block.get("font_size")
            text = _pdf_inline_text(str(block.get("text") or ""))
            if size and text:
                weights[round(float(size), 1)] += len(text)
    if not weights:
        return None
    return max(weights, key=lambda value: (weights[value], -value))


def _pdf_text_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _pdf_bookmark_outline(
    raw_toc: list[list[Any]], page_count: int
) -> list[dict[str, Any]]:
    """Keep the legacy import compatible without consuming native bookmarks."""
    return []


def _pdf_printed_toc_entries(
    page_contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    toc_pages: list[int] = []
    seen: set[tuple[str, int]] = set()
    for context in page_contexts:
        if not _is_pdf_contents_page(context):
            continue
        toc_pages.append(int(context["page"]))
        if context.get('printed_toc_entries'):
            for item in context['printed_toc_entries']:
                title, page = item['title'], int(item['target_page'])
                entries.append({'node_id':_stable_id('pdftoc',str(len(entries)+1),title,str(page),length=16),
                    'parent_node_id':None,'title':title,'level':item.get('level',1),
                    'page_start':page,'page_end':page,'source':'printed_toc',
                    'toc_page':int(context['page']),'ordinal':len(entries)+1})
            continue
        for block in context.get("blocks", []):
            for raw_line in str(block.get("text") or "").splitlines():
                line = _pdf_inline_text(raw_line)
                match = re.match(r"^(.+?)\s*(?:\.\s*){3,}\s*(\d+)\s*$", line)
                if not match:
                    continue
                title = _pdf_inline_text(match.group(1)).rstrip(".")
                page = int(match.group(2))
                if not title or page < 1:
                    continue
                identity = (title.casefold(), page)
                if identity in seen:
                    continue
                seen.add(identity)
                numbered = re.match(r"^(\d+(?:\.\d+)*)\b", title)
                level = (
                    len(numbered.group(1).split("."))
                    if numbered is not None
                    else 1
                )
                entries.append(
                    {
                        "node_id": _stable_id(
                            "pdftoc", str(len(entries) + 1), title, str(page), length=16
                        ),
                        "parent_node_id": None,
                        "title": title,
                        "level": level,
                        "page_start": page,
                        "page_end": page,
                        "source": "printed_toc",
                        "toc_page": int(context["page"]),
                        "ordinal": len(entries) + 1,
                    }
                )
    hierarchy_stack: list[dict[str, Any]] = []
    for entry in entries:
        level = max(1, int(entry["level"]))
        while hierarchy_stack and int(hierarchy_stack[-1]["level"]) >= level:
            hierarchy_stack.pop()
        entry["parent_node_id"] = (
            hierarchy_stack[-1]["node_id"] if hierarchy_stack else None
        )
        hierarchy_stack.append(entry)
    for index, entry in enumerate(entries):
        following_page = next(
            (
                int(item["page_start"])
                for item in entries[index + 1 :]
                if int(item["level"]) <= int(entry["level"])
                and int(item["page_start"]) >= int(entry["page_start"])
            ),
            None,
        )
        if following_page is not None:
            entry["page_end"] = max(int(entry["page_start"]), following_page - 1)
    return {"pages": toc_pages, "entries": entries}


def _pdf_heading_candidates(
    page_contexts: list[dict[str, Any]],
    *,
    body_font_size: float | None,
    bookmark_outline: list[dict[str, Any]] | None = None,
    printed_toc_entries: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    repeated_furniture = _pdf_repeated_furniture_texts(page_contexts)
    # Native bookmarks, including legacy caller-supplied values, have no role
    # in boundary matching, hierarchy, or typography classification.
    authored = list(printed_toc_entries or [])
    authored_by_page: dict[int, list[dict[str, Any]]] = {}
    for node in authored:
        authored_by_page.setdefault(int(node["page_start"]), []).append(node)

    # Bind authored navigation to layout blocks once, before classifying any
    # typography-only headings. Each printed title binds at most one block,
    # so a short title cannot multiply into many matches in surrounding prose.
    authored_matches: dict[str, dict[str, Any]] = {}
    for context in page_contexts:
        page = int(context["page"])
        available = [
            block
            for block in context.get("blocks", [])
            if not _is_pdf_furniture(block, repeated_furniture)
        ]
        used_block_ids: set[str] = set()
        for node in authored_by_page.get(page, []):
            title_key = _pdf_text_key(str(node.get("title") or ""))
            if not title_key:
                continue
            ranked: list[tuple[int, int, int, dict[str, Any]]] = []
            for block_index, block in enumerate(available):
                block_id = str(block["part_id"])
                if block_id in used_block_ids:
                    continue
                raw_text = str(block.get("text") or "")
                block_key = _pdf_text_key(raw_text)
                if not block_key:
                    continue
                line_keys = [
                    _pdf_text_key(line)
                    for line in raw_text.splitlines()
                    if _pdf_text_key(line)
                ]
                if block_key == title_key:
                    match_class = 5
                elif title_key in line_keys:
                    match_class = 4
                elif block_key.startswith(title_key):
                    match_class = 3
                elif title_key in block_key:
                    match_class = 2
                else:
                    continue
                # Prefer the strongest textual match, then the least surrounding
                # text, then the original reading order.
                ranked.append(
                    (
                        match_class,
                        -max(0, len(block_key) - len(title_key)),
                        -block_index,
                        block,
                    )
                )
            if not ranked:
                continue
            selected = max(ranked, key=lambda item: item[:3])[3]
            selected_id = str(selected["part_id"])
            authored_matches[selected_id] = node
            used_block_ids.add(selected_id)

    preliminary: list[dict[str, Any]] = []
    for context in page_contexts:
        if _is_pdf_contents_page(context):
            continue
        for block in context.get("blocks", []):
            if _is_pdf_furniture(block, repeated_furniture):
                continue
            text = _pdf_inline_text(str(block.get("text") or ""))
            if not text:
                continue
            authored_match = authored_matches.get(str(block["part_id"]))
            core_hint = block.get('heading_hint')
            if block.get('structure_owned') and not core_hint and block.get('structured_block',{}).get('type') not in {'paragraph'}:
                continue
            # A PDF layout block may contain a heading line and its first body
            # paragraph together.  Authored navigation is still allowed to bind
            # that block; the short-text limit only guards heuristic headings.
            if authored_match is None and len(text) > 180:
                continue

            explicit = _pdf_heading_text(text)
            font_size = float(block.get("max_font_size") or block.get("font_size") or 0.0)
            ratio = font_size / body_font_size if body_font_size and font_size else 0.0
            words = text.split()
            looks_like_prose = len(words) > 18 or (
                len(text) > 100 and bool(re.search(r"[.!?]\s*$", text))
            )
            layout_candidate = (
                not looks_like_prose
                and int(block.get("line_count") or 1) <= 3
                and ratio >= 1.18
            )
            # A very large first-page title is document metadata, not a chapter.
            first_page_title = int(context["page"]) == 1 and ratio >= 2.0
            if authored_match is None and explicit is None and not core_hint and (
                not layout_candidate or first_page_title
            ):
                continue

            outline = _pdf_heading_outline(explicit or text)
            source = "layout_heading"
            title = explicit or text
            level: int | None = None
            parent_node_id = None
            node_id = _stable_id(
                "pdfhead", str(context["page"]), str(block["part_id"]), text, length=16
            )
            if authored_match is not None:
                source = str(authored_match["source"])
                title = str(authored_match["title"])
                level = int(authored_match["level"])
                parent_node_id = authored_match.get("parent_node_id")
                node_id = str(authored_match["node_id"])
            elif outline.get("heading_number") or outline.get("heading_kind") in {
                "chapter",
                "section",
                "appendix",
            }:
                source = "numbered_heading"
                level = int(outline["heading_level"])
            elif core_hint:
                level = int(core_hint.get('level') or 1)
                title = core_hint.get('title') or text

            preliminary.append(
                {
                    "node_id": node_id,
                    "parent_node_id": parent_node_id,
                    "block_part_id": str(block["part_id"]),
                    "title": title,
                    "level": level,
                    "page": int(context["page"]),
                    "source": source,
                    "heading_kind": outline.get("heading_kind"),
                    "heading_number": list(outline.get("heading_number") or []),
                    "font_size": round(font_size, 2) if font_size else None,
                    "font_ratio": round(ratio, 3) if ratio else None,
                    "bold": bool(block.get("bold")),
                    "bbox": block.get("bbox"),
                    "ordinal": len(preliminary) + 1,
                }
            )

    layout_sizes = sorted(
        {
            round(float(item["font_size"]), 1)
            for item in preliminary
            if item["level"] is None and item.get("font_size")
        },
        reverse=True,
    )
    size_levels: dict[float, int] = {}
    clusters: list[float] = []
    for size in layout_sizes:
        existing = next((item for item in clusters if abs(item - size) <= 0.7), None)
        if existing is None:
            clusters.append(size)
            existing = size
        size_levels[size] = min(4, clusters.index(existing) + 1)

    hierarchy_stack: list[dict[str, Any]] = []
    result: dict[str, dict[str, Any]] = {}
    for item in preliminary:
        if item["level"] is None:
            item["level"] = size_levels.get(
                round(float(item.get("font_size") or 0.0), 1), 1
            )
        level = max(1, int(item["level"]))
        item["level"] = level
        while hierarchy_stack and int(hierarchy_stack[-1]["level"]) >= level:
            hierarchy_stack.pop()
        if item.get("parent_node_id") is None:
            item["parent_node_id"] = (
                hierarchy_stack[-1]["node_id"] if hierarchy_stack else None
            )
        hierarchy_stack.append(item)
        result[str(item["block_part_id"])] = item
    return result


def _pdf_inline_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _pdf_heading_text(value: str) -> str | None:
    """Recognize only explicit heading forms; ambiguous text stays page-level."""
    text = _pdf_inline_text(value)
    if not text or len(text) > 180:
        return None
    numbered = re.match(r"^(?:\d+(?:\.\d+)+)\.?\s+(?=\S).+", text)
    named = re.match(
        r"^(?:chapter|section|appendix)\s+[A-Za-z0-9.]+(?:\s*[:.-]\s*.+|\s+.+)?$",
        text,
        flags=re.IGNORECASE,
    )
    chinese = re.match(r"^第\s*[一二三四五六七八九十百千0-9]+\s*[章节篇](?:\s*[:：.-]?\s*).+", text)
    return text if numbered or named or chinese else None


def _pdf_heading_outline(value: str) -> dict[str, Any]:
    """Return deterministic hierarchy hints for an already accepted PDF heading."""

    text = _pdf_inline_text(value)
    heading_kind = "heading"
    heading_number: list[str] = []
    named = re.match(
        r"^(chapter|section|appendix)\s+([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)\b",
        text,
        flags=re.IGNORECASE,
    )
    if named:
        heading_kind = named.group(1).casefold()
        heading_number = [
            item.casefold()
            for item in named.group(2).strip(".").split(".")
            if item
        ]
    else:
        numbered = re.match(r"^(\d+(?:\.\d+)+)\.?(?:\s|$)", text)
        if numbered:
            heading_kind = "numbered"
            heading_number = [item for item in numbered.group(1).split(".") if item]

    if heading_kind == "chapter" or (heading_kind == "appendix" and len(heading_number) <= 1):
        heading_level = 1
        is_chapter_heading = True
    elif heading_number:
        heading_level = max(1, len(heading_number))
        is_chapter_heading = heading_level == 1
    else:
        heading_level = 1
        is_chapter_heading = False

    return {
        "heading_kind": heading_kind,
        "heading_number": heading_number,
        "heading_level": heading_level,
        "is_chapter_heading": is_chapter_heading,
    }


def _pdf_find_heading_parent(
    active_hierarchy: list[dict[str, Any]], child: dict[str, Any]
) -> dict[str, Any] | None:
    child_level = int(child.get("heading_level") or 1)
    for parent in reversed(active_hierarchy):
        parent_level = int(parent.get("heading_level") or 1)
        if parent_level >= child_level:
            continue
        if _pdf_heading_parent_matches(parent, child):
            return parent
    return None


def _pdf_heading_parent_matches(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    parent_number = tuple(str(item).casefold() for item in parent.get("heading_number") or [])
    child_number = tuple(str(item).casefold() for item in child.get("heading_number") or [])
    if parent_number and child_number:
        return (
            len(parent_number) < len(child_number)
            and child_number[: len(parent_number)] == parent_number
        )
    return True


def _pdf_enrich_generic_heading(heading: str, blocks: list[dict[str, Any]]) -> str:
    """Join a bare chapter number with its adjacent document title when unambiguous.

    Many manuals use separate PDF text blocks for ``Chapter 3`` and ``Using the
    Calculator``. The section boundary is already deterministic; this only makes
    its locator readable without inventing or paraphrasing source content.
    """
    bare_heading = re.fullmatch(
        r"(?:chapter|section|appendix)\s+[A-Za-z0-9.]+", heading.strip(), flags=re.IGNORECASE
    )
    if not bare_heading or len(blocks) < 2:
        return heading
    candidate = _pdf_inline_text(str(blocks[1].get("text") or ""))
    if not candidate or len(candidate) > 120:
        return heading
    # A short title normally has no sentence-ending punctuation and no line that
    # begins a list. Longer prose remains inside the exact section content only.
    if re.search(r"[.!?;:。！？；：]", candidate) or candidate.startswith(("•", "-", "*")):
        return heading
    if len(candidate.split()) > 14:
        return heading
    return f"{heading}: {candidate}"


def _pdf_repeated_furniture_texts(page_contexts: list[dict[str, Any]]) -> set[str]:
    counts: Counter[str] = Counter()
    for context in page_contexts:
        seen_on_page: set[str] = set()
        page_height = float(context.get("page_height") or 0)
        for block in context.get("blocks", []):
            bbox = block.get("bbox")
            if not bbox or page_height <= 0:
                continue
            text = _pdf_inline_text(str(block.get("text") or ""))
            if not text or len(text) > 160:
                continue
            y0, y1 = float(bbox[1]), float(bbox[3])
            if y0 <= page_height * 0.18 or y1 >= page_height * 0.82:
                seen_on_page.add(text)
        counts.update(seen_on_page)
    return {text for text, count in counts.items() if count >= 2}


def _is_pdf_contents_page(context: dict[str, Any]) -> bool:
    if context.get('printed_toc_entries'):
        return True
    blocks = context.get("blocks") or []
    for block in blocks[:8]:
        for raw_line in str(block.get("text") or "").splitlines():
            normalized = _pdf_inline_text(raw_line).casefold()
            if not normalized or len(normalized) > 80:
                continue
            if re.fullmatch(
                r"(?:table of )?contents|\u76ee\u5f55|\u76ee\u6b21|\u76ee\u9304",
                normalized,
                flags=re.IGNORECASE,
            ):
                return True
    # A page containing several dot-leader entries is a contents/index page even
    # when it has no explicit title. Treat it as one readable page projection,
    # never as a sequence of content sections.
    leader_entries = sum(
        1
        for block in blocks
        for raw_line in str(block.get("text") or "").splitlines()
        # Require spaced dot leaders, not ordinary ``...`` syntax.  Scientific
        # software manuals commonly contain variadic signatures such as
        # ``plus(... parameters, ...)`` in real content pages.
        if re.search(r"(?:\.\s+){6,}", raw_line)
    )
    return leader_entries >= 3


def _is_pdf_furniture(block: dict[str, Any], repeated_furniture: set[str]) -> bool:
    if block.get('structure_owned'):
        # Core already separated furniture from body with line-level provenance.
        return block.get('structured_block', {}).get('type') == 'blank_page'
    text = _pdf_inline_text(str(block.get("text") or ""))
    bbox = block.get("bbox")
    page_height = float(block.get("page_height") or 0)
    if not text:
        return True
    if not bbox or page_height <= 0:
        return False
    y0, y1 = float(bbox[1]), float(bbox[3])
    in_margin = y0 <= page_height * 0.18 or y1 >= page_height * 0.82
    if text in repeated_furniture and in_margin:
        return True
    return bool(y0 >= page_height * 0.82 and re.fullmatch(r"(?:page\s+)?\d+", text, re.IGNORECASE))


def _pdf_section_is_substantive(candidate: dict[str, Any]) -> bool:
    blocks = candidate.get("blocks") or []
    if not blocks:
        return False
    # An authored outline entry may be a pure structural container whose next
    # block immediately starts a child section.  Preserve that exact heading as
    # a section node so children attach to the authored parent instead of the
    # previous sibling chapter.
    if candidate.get("structure_heading") is not None:
        return True
    if len(blocks) < 2:
        return False
    body = _join_pdf_block_texts(blocks[1:])
    return bool(body.strip())


def _join_pdf_block_texts(blocks: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        str(block.get("text") or "").strip()
        for block in blocks
        if str(block.get("text") or "").strip()
    )


def _pdf_bbox_area(bbox: list[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def _expand_pdf_bbox(bbox: list[float], padding: float) -> list[float]:
    return [
        float(bbox[0]) - padding,
        float(bbox[1]) - padding,
        float(bbox[2]) + padding,
        float(bbox[3]) + padding,
    ]


def _clip_pdf_bbox(
    bbox: list[float], page_width: float, page_height: float
) -> list[float] | None:
    if page_width <= 0 or page_height <= 0 or len(bbox) != 4:
        return None
    clipped = [
        max(0.0, min(float(bbox[0]), page_width)),
        max(0.0, min(float(bbox[1]), page_height)),
        max(0.0, min(float(bbox[2]), page_width)),
        max(0.0, min(float(bbox[3]), page_height)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return [round(value, 2) for value in clipped]


def _is_pdf_margin_image_furniture(
    bbox: list[float], page_width: float, page_height: float
) -> bool:
    if page_width <= 0 or page_height <= 0 or len(bbox) != 4:
        return False
    in_header = float(bbox[3]) <= page_height * 0.10
    in_footer = float(bbox[1]) >= page_height * 0.72 or float(bbox[3]) >= page_height * 0.97
    touches_outer_margin = float(bbox[0]) <= page_width * 0.04 or float(bbox[2]) >= page_width * 0.96
    return (in_header or in_footer) and touches_outer_margin


def _pdf_tiny_display_image_reason(image_part: SourcePart) -> str | None:
    bbox = _union_bboxes(image_part.locator.get("bboxes", []))
    if not bbox:
        return None
    metadata = image_part.metadata
    display_area_ratio = float(metadata.get("display_area_ratio") or 0.0)
    display_width = max(0.0, float(bbox[2]) - float(bbox[0]))
    display_height = max(0.0, float(bbox[3]) - float(bbox[1]))
    pixel_width = int(metadata.get("pixel_width") or 0)
    pixel_height = int(metadata.get("pixel_height") or 0)
    pixel_area = max(0, pixel_width) * max(0, pixel_height)
    if (
        (pixel_width <= 48 and pixel_height <= 48 and display_area_ratio < 0.01)
        or (pixel_area <= 1_024 and display_area_ratio < 0.02)
    ):
        return "tiny_low_information_asset"
    if (display_width <= 48 and display_height <= 48 and display_area_ratio < 0.006) or (
        min(display_width, display_height) <= 24 and display_area_ratio < 0.02
    ):
        return "tiny_display_region"
    return None


def _classify_pdf_image(
    *,
    pixel_width: int,
    pixel_height: int,
    rects: list[list[float]],
    page_width: float,
    page_height: float,
) -> tuple[str, str, float]:
    """Filter only intrinsically tiny assets; page-level rules run after inventory."""

    page_area = max(1.0, page_width * page_height)
    display_area = max(
        (_pdf_bbox_area(rect) for rect in rects if len(rect) == 4),
        default=0.0,
    )
    display_area_ratio = round(display_area / page_area, 6)
    pixel_area = max(0, pixel_width) * max(0, pixel_height)
    tiny_square_asset = pixel_width <= 48 and pixel_height <= 48 and display_area_ratio < 0.01
    extremely_low_detail = pixel_area <= 1_024 and display_area_ratio < 0.02
    if tiny_square_asset or extremely_low_detail:
        return "decorative", "tiny_low_information_asset", display_area_ratio
    return "informative", "layout_significant_image", display_area_ratio


def _union_bboxes(bboxes: list[list[float]]) -> list[float] | None:
    valid = [bbox for bbox in bboxes if len(bbox) == 4]
    if not valid:
        return None
    return [
        min(float(item[0]) for item in valid),
        min(float(item[1]) for item in valid),
        max(float(item[2]) for item in valid),
        max(float(item[3]) for item in valid),
    ]


def _vertical_bbox_distance(
    image_bbox: list[float], text_bbox: list[float], *, prefer_preceding: bool
) -> float:
    image_top, image_bottom = float(image_bbox[1]), float(image_bbox[3])
    text_top, text_bottom = float(text_bbox[1]), float(text_bbox[3])
    if text_bottom <= image_top:
        return image_top - text_bottom
    if text_top >= image_bottom:
        distance = text_top - image_bottom
        return distance + (12.0 if prefer_preceding else 0.0)
    return 0.0
