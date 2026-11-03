"""Source-preserving Reference catalog parsing, deduplication, and structural profiles."""

from __future__ import annotations

import re
from typing import Any

from ..models import EvidenceExcerpt
from .common import REFERENCE_ORDINARY_CONTEXT_CHARS, _compact, _stable_id


_REFERENCE_REVIEW_SIGNAL_PATTERNS = (
    re.compile(
        r"syntax|grammar|operator|expression|valid\s+(?:form|expression)|example|->|→",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"shortcut|hotkey|keyboard|arrow\s+keys?|mouse\s+buttons?|"
        r"(?:Ctrl|Alt|Shift|Cmd|Command)\+[A-Za-z0-9_+.-]+|"
        r"press\s+[A-Za-z0-9]",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"command|function|API|parameter|option|setting|menu|file\s+format|constraint",
        flags=re.IGNORECASE,
    ),
)


def _representative_reference_excerpt(
    content: str, maximum: int = REFERENCE_ORDINARY_CONTEXT_CHARS
) -> str:
    """Return bounded review context without pretending an opening slice is a summary.

    Short candidates are returned in full. For longer candidates, deterministic spans
    cover the beginning, middle, end, and up to two distinct Reference-like signals.
    The excerpts are kept in source order and never exceed the candidate-level budget.
    """

    stripped = content.strip()
    if len(stripped) <= maximum:
        return stripped
    compact = _compact(stripped)
    if len(compact) <= maximum:
        return compact

    def centered_span(position: int, width: int) -> tuple[int, int]:
        start = max(0, min(len(compact) - width, position - width // 2))
        return start, min(len(compact), start + width)

    spans: list[tuple[int, int]] = [
        (0, 95),
        centered_span(len(compact) // 2, 75),
        (len(compact) - 75, len(compact)),
    ]
    signal_spans: list[tuple[int, int]] = []
    for pattern in _REFERENCE_REVIEW_SIGNAL_PATTERNS:
        match = pattern.search(compact)
        if match is None:
            continue
        candidate = centered_span((match.start() + match.end()) // 2, 110)
        if any(
            max(candidate[0], existing[0]) < min(candidate[1], existing[1])
            for existing in signal_spans
        ):
            continue
        signal_spans.append(candidate)
        if len(signal_spans) == 2:
            break
    spans.extend(signal_spans)
    spans.sort()

    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    excerpt = " … ".join(compact[start:end].strip() for start, end in merged)
    return excerpt[:maximum].rstrip()


_NUMBERED_REFERENCE_ENTRY = re.compile(r"^(?P<number>\d+(?:\.\d+)+)\s+(?P<label>.+?)\s*$")
_INLINE_ENTRY_LABEL = re.compile(
    r"^(?:"
    r"[A-Za-z][A-Za-z0-9_]*(?:[._:-][A-Za-z0-9_]+)+"
    r"|[A-Za-z][A-Za-z0-9_]{1,48}\([^()\n]{0,80}\)"
    r"|(?:Ctrl|Alt|Shift|Cmd|Command)\+[A-Za-z0-9_+.-]{1,48}"
    r")$",
    flags=re.IGNORECASE,
)
_INLINE_CATALOG_MARKERS = (
    "shortcut",
    "shortcuts",
    "keyboard",
    "hotkey",
    "hotkeys",
    "command reference",
    "command table",
    "commands manual",
    "function table",
    "api reference",
    "syntax",
    "grammar",
    "快捷键",
    "键盘",
    "鼠标",
    "摇杆",
    "指令表",
    "命令表",
    "函数表",
    "语法",
    "参数",
    "数据型",
)
_INLINE_SECTION_HEADINGS = frozenset(
    {
        "鼠标控制",
        "键盘控制",
        "导航",
        "时间控制",
        "显示星体标签",
        "显示空间飞行器标签",
        "显示的项目",
        "显示选项",
        "多视窗选项",
        "运动控制",
        "其他命令",
        "摇杆控制",
        "数据型列表",
        "默认值和零参数",
        "语法",
        "Celestia 的坐标系",
        "导航(通过方向键)",
        "导航 (通过方向键)",
        "导航(通过激活数字锁后的数字小键盘)",
        "导航 (通过激活数字锁后的数字小键盘)",
        "指令表（音序）",
        "指令表(音序)",
    }
)
_INLINE_TABLE_COLUMN_HEADINGS = frozenset(
    {
        "按键",
        "快捷键",
        "指令",
        "命令",
        "语法",
        "参数",
        "说明",
        "功能",
        "作用",
        "数据型",
        "默认值",
        "鼠标动作",
        "等效的键盘命令",
        "keyboard shortcut",
        "keyboard shortcuts",
        "command",
        "description",
        "function",
    }
)
_INLINE_SYMBOL_KEYS = frozenset(
    {
        "!",
        "?",
        "-",
        "=",
        "&",
        "^",
        "/",
        ";",
        "{",
        "}",
        "[",
        "]",
        ",",
        ".",
        "*",
        "~",
        "`",
        "+",
        "%",
        "\\",
        '"',
        "'",
        ":",
        "@",
        "#",
        "$",
        "|",
        "(",
        ")",
    }
)
_INLINE_COMBO_LABEL = re.compile(
    r"^(?:(?:Ctrl|Control|Alt|Shift|Cmd|Command|Option|Meta)\+)+(?:[^\s+]{1,64})$",
    flags=re.IGNORECASE,
)
_INLINE_COMBO_PREFIX = re.compile(
    r"^(?P<label>(?:(?:Ctrl|Control|Alt|Shift|Cmd|Command|Option|Meta)\s*\+\s*)+\S{1,64})\s+(?P<description>\S.*)$",
    flags=re.IGNORECASE,
)
_INLINE_SPECIAL_KEY_LABEL = re.compile(
    r"^(?:Enter|Return|Esc|Escape|Backspace|Tab|Del|Delete|Home|End|Insert|Ins|Space|"
    r"PgUp|PgDn|PageUp|PageDown|空格键|回车键|删除键|退格键)$",
    flags=re.IGNORECASE,
)
_INLINE_FUNCTION_KEY_LABEL = re.compile(r"^F(?:1[0-9]|2[0-4]|[1-9])$", flags=re.IGNORECASE)
_INLINE_SINGLE_KEY_LABEL = re.compile(r"^[A-Z0-9]$")
_INLINE_ARROW_KEY_LABEL = re.compile(
    r"^(?:(?:Shift|Ctrl|Alt)\s*\+\s*)?(?:上/下|左/右|上下|左右)?方向键$",
    flags=re.IGNORECASE,
)
_INLINE_MOUSE_OR_JOYSTICK_LABEL = re.compile(
    r"^(?=.*(?:鼠标|左键|右键|中键|滚轮|拖动|axis|trigger|button)).{1,90}$",
    flags=re.IGNORECASE,
)
_INLINE_NUMERIC_RANGE_LABEL = re.compile(r"^\d+\s*(?:-|–|—|~|到)\s*\d+$")
_INLINE_QUALIFIED_KEY_LABEL = re.compile(r"^[A-Za-z0-9]\s*\([^()\n]{1,24}\)$")
_INLINE_BARE_WORD_LABEL = re.compile(r"^(?:[a-z][a-z0-9_-]{1,64}|Default|Null)$")
_INLINE_ARROW_PREFIX = re.compile(
    r"^(?P<label>(?:(?:Shift|Ctrl|Alt)\s*\+\s*)?(?:上/下|左/右|上下|左右)?方向键)\s+(?P<description>\S.*)$",
    flags=re.IGNORECASE,
)
_INLINE_MOUSE_PREFIX = re.compile(
    r"^(?P<label>(?:在物体上按左键|在空白处按左键|在物体上左键双击|按右键|左键拖动|"
    r"右键拖动|滚轮|鼠标中键|右键\+左键向左/右拖动|右键\+左键向上/下拖动|"
    r"Ctrl\s*键\+左键向上/下拖动))\s+(?P<description>\S.*)$",
    flags=re.IGNORECASE,
)
_INLINE_JOYSTICK_PREFIX = re.compile(
    r"^(?P<label>(?:[XYZ]\s+axis|[LR]\s+trigger|Button\s+\d+))\s+(?P<description>\S.*)$",
    flags=re.IGNORECASE,
)
_INLINE_QUALIFIED_KEY_PREFIX = re.compile(
    r"^(?P<label>[A-Za-z0-9]\s*\([^()\n]{1,24}\))\s+(?P<description>\S.*)$"
)
_REFERENCE_LIKE_TITLE_MARKERS = (
    ("command_catalog", ("command", "commands manual", "module index", "tool index", "operator")),
    ("function_catalog", ("function", "built-in", "api", "method", "dictionary")),
    ("syntax_reference", ("syntax", "grammar", "expression", "formula", "operator")),
    ("parameter_reference", ("parameter", "argument", "option", "flag", "setting")),
    ("configuration_reference", ("configuration", "preference", "environment variable", "variable")),
    ("file_format_reference", ("file format", "schema", "extension")),
    (
        "ui_reference",
        (
            "shortcut",
            "keyboard",
            "hotkey",
            "key binding",
            "menu",
            "toolbar",
            "button",
            "dialog",
            "window",
            "component",
            "快捷键",
            "键盘",
            "按键",
            "鼠标",
            "摇杆",
            "菜单",
            "按钮",
            "对话框",
            "窗口",
        ),
    ),
    ("concept_reference", ("reference", "manual", "catalog", "index", "overview")),
)
_REFERENCE_FIELD_LABEL = re.compile(r"^\s*(?:[-*•]\s*)?(?P<label>[^:\uff1a\n]{1,48})\s*[:\uff1a]")


def _reference_entry_sort_key(evidence: EvidenceExcerpt) -> tuple[Any, ...]:
    match = _NUMBERED_REFERENCE_ENTRY.match(_compact(evidence.title))
    if match:
        return (0, *(int(item) for item in match.group("number").split(".")), evidence.title)
    return (1, evidence.title.casefold(), evidence.evidence_id)


def _reference_field_labels(content: str) -> tuple[str, ...]:
    labels: list[str] = []
    for line in content.splitlines()[:24]:
        match = _REFERENCE_FIELD_LABEL.match(line)
        if not match:
            continue
        label = _compact(match.group("label")).casefold()
        if 1 < len(label) <= 48 and label not in labels:
            labels.append(label)
    return tuple(labels)


def _strip_inline_code_marks(value: str) -> str:
    stripped = _compact(value)
    if len(stripped) > 2 and stripped.startswith("`") and stripped.endswith("`"):
        return stripped[1:-1].strip()
    return stripped


def _normalize_inline_entry_label(value: str) -> str:
    label = _strip_inline_code_marks(value)
    label = re.sub(r"\s*\+\s*", "+", label)
    label = re.sub(r"\s+", " ", label)
    return label.strip().casefold()


def _contains_inline_catalog_marker(value: str) -> bool:
    text = _compact(value).casefold()
    return any(marker.casefold() in text for marker in _INLINE_CATALOG_MARKERS)


def _is_inline_catalog_heading(value: str) -> bool:
    text = _compact(value)
    if not text or len(text) > 120:
        return False
    if text in _INLINE_SECTION_HEADINGS or text.casefold() in _INLINE_TABLE_COLUMN_HEADINGS:
        return True
    if text.startswith("附录"):
        return True
    if not _contains_inline_catalog_marker(text):
        return False
    if any(mark in text for mark in ("。", "！", "？", "；", "，", ".", "!", "?", ";", ",")):
        return False
    return len(text) <= 80 and (
        text.endswith(("表", "列表", "控制", "参考", "语法", "参数", "命令", "快捷键"))
        or "指令表" in text
        or "命令表" in text
        or "函数表" in text
        or "command reference" in text.casefold()
        or "keyboard shortcuts" in text.casefold()
    )


def _is_inline_table_column_heading(value: str) -> bool:
    return _compact(value).casefold() in _INLINE_TABLE_COLUMN_HEADINGS


def _evidence_has_inline_catalog_context(evidence: EvidenceExcerpt) -> bool:
    values = (
        evidence.title,
        evidence.locator.get("heading"),
        evidence.locator.get("section_path"),
        evidence.locator.get("chapter_heading"),
        evidence.locator.get("url"),
    )
    return _contains_inline_catalog_marker(" ".join(str(value) for value in values if value))


def _is_inline_entry_label(value: str, *, allow_broad: bool = False) -> bool:
    label = _strip_inline_code_marks(value)
    if not label or len(label) > 90:
        return False
    if _is_inline_catalog_heading(label):
        return False

    normalized_plus = re.sub(r"\s*\+\s*", "+", label)
    if _INLINE_ENTRY_LABEL.match(normalized_plus):
        return True
    if not allow_broad:
        return False

    if label in _INLINE_SYMBOL_KEYS:
        return True
    if _INLINE_COMBO_LABEL.match(normalized_plus):
        return True
    if _INLINE_SPECIAL_KEY_LABEL.match(label):
        return True
    if _INLINE_FUNCTION_KEY_LABEL.match(label):
        return True
    if _INLINE_SINGLE_KEY_LABEL.match(label):
        return True
    if _INLINE_ARROW_KEY_LABEL.match(label):
        return True
    if _INLINE_MOUSE_OR_JOYSTICK_LABEL.match(label):
        return True
    if _INLINE_NUMERIC_RANGE_LABEL.match(label):
        return True
    if _INLINE_QUALIFIED_KEY_LABEL.match(label):
        return True
    if _INLINE_BARE_WORD_LABEL.match(label):
        return True
    return bool(_INLINE_ENTRY_LABEL.match(label))


def _split_inline_entry_line(text: str, *, allow_broad: bool) -> tuple[str, str] | None:
    compact = _compact(text)
    if not compact:
        return None
    if _is_inline_entry_label(compact, allow_broad=allow_broad):
        return (compact, "")
    if not allow_broad:
        return None

    for pattern in (
        _INLINE_COMBO_PREFIX,
        _INLINE_MOUSE_PREFIX,
        _INLINE_JOYSTICK_PREFIX,
        _INLINE_ARROW_PREFIX,
        _INLINE_QUALIFIED_KEY_PREFIX,
    ):
        match = pattern.match(compact)
        if match:
            label = _compact(match.group("label"))
            description = _compact(match.group("description"))
            if _is_inline_entry_label(label, allow_broad=True) and description:
                return (label, description)

    first, separator, rest = compact.partition(" ")
    if separator and _is_inline_entry_label(first, allow_broad=True) and rest.strip():
        return (first, rest.strip())
    return None


def _line_spans(content: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    cursor = 0
    for raw in content.splitlines(keepends=True):
        stripped = raw.strip()
        if stripped:
            leading = len(raw) - len(raw.lstrip(" \t"))
            start = cursor + leading
            end = cursor + len(raw.rstrip("\r\n"))
        else:
            start = cursor
            end = cursor
        spans.append({"text": stripped, "start": start, "end": end, "raw_end": cursor + len(raw)})
        cursor += len(raw)
    return spans


_KEYWORD_INDEX_IDENTIFIER = re.compile(
    r"^[A-Za-z][A-Za-z0-9_+-]*(?:\.[A-Za-z0-9_*+-]+)+$"
)
_CATALOG_CHUNK_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")
_ENVIRONMENT_VARIABLE_LABEL = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
_AGGREGATE_CATALOG_URL_MARKERS = (
    "full_index",
    "master_index",
    "all_commands",
)


def _keyword_index_identifier_list(value: str) -> bool:
    """Return whether a source line is a comma-separated identifier list.

    Keyword indexes commonly alternate a human keyword and one or more dotted
    command/API identifiers.  Treating every bare word as an inline entry label
    shifts those pairs by one line, so this shape needs a small structural parser.
    """

    compact = _compact(value)
    if not compact or len(compact) > 8_000:
        return False
    parts = [_compact(item) for item in compact.split(",") if _compact(item)]
    return bool(parts) and all(_KEYWORD_INDEX_IDENTIFIER.fullmatch(item) for item in parts)


def _keyword_index_reference_entries(
    evidence: EvidenceExcerpt,
) -> list[dict[str, Any]] | None:
    context = " ".join(
        str(value)
        for value in (
            evidence.title,
            evidence.locator.get("heading"),
            evidence.locator.get("section_path"),
            evidence.locator.get("chapter_heading"),
            evidence.locator.get("url"),
        )
        if value
    ).casefold()
    if "keyword" not in context or "index" not in context:
        return None

    lines = [item for item in _line_spans(evidence.content) if item["text"]]
    entries: list[dict[str, Any]] = []
    index = 0
    while index + 1 < len(lines):
        label_line = lines[index]
        value_line = lines[index + 1]
        label = _compact(label_line["text"])
        value = _compact(value_line["text"])
        if (
            label
            and not _keyword_index_identifier_list(label)
            and _keyword_index_identifier_list(value)
        ):
            locator = dict(evidence.locator)
            locator.update(
                {
                    "entry_label": label,
                    "entry_category": "keyword index",
                    "char_start": label_line["start"],
                    "char_end": value_line["end"],
                }
            )
            entries.append(
                {
                    "title": label,
                    "content": evidence.content[
                        label_line["start"] : value_line["end"]
                    ].strip(),
                    "source_evidence_id": evidence.evidence_id,
                    "locator": locator,
                }
            )
            index += 2
            continue
        # A chunk can start with the value line belonging to the previous
        # chunk. Skip that orphan and resynchronize on the next label/value pair.
        index += 1

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        key = _normalize_inline_entry_label(entry["title"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    # A weak match should be rejected rather than falling back to the broad
    # bare-word parser which created shifted keyword catalogs.
    return deduped if len(deduped) >= 4 else []


def _markdown_table_reference_entries(evidence: EvidenceExcerpt) -> list[dict[str, Any]]:
    """Preserve authored lookup rows from structured HTML's Markdown projection."""
    lines = _line_spans(evidence.content)
    entries = []
    columns = 0
    fence = None
    for line in lines:
        text = line["text"].strip()
        marker = re.match(r"^(`{3,}|~{3,})", text)
        if marker:
            if fence and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            elif not fence:
                fence = marker[1]
            columns = 0
            continue
        if fence:
            continue
        if not text.startswith("|") or not text.endswith("|"):
            columns = 0
            continue
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", text[1:-1])]
        if len(cells) >= 2 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            columns = len(cells)
            continue
        if not columns or len(cells) != columns or not cells[0] or not any(cells[1:]):
            continue
        label = re.sub(r"\[([^\]]+)\]\[L[0-9]+\]", r"\1", cells[0])
        code_label = re.fullmatch(r"(`+)(.*?)\1", label)
        if code_label:
            label = code_label[2].strip(" ")
        else:
            label = re.sub(r"\*\*(.+?)\*\*", r"\1", label)
            label = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", label)
        label = label.replace(r"\|", "|")
        if not label or len(label) > 300:
            continue
        locator = {**evidence.locator, "entry_label":label, "char_start":line["start"],
                   "char_end":line["end"], "entry_structure":"markdown_table_row"}
        entries.append({"title":label, "content":evidence.content[line["start"]:line["end"]].strip(),
                        "source_evidence_id":evidence.evidence_id, "locator":locator})
    return entries


def _inline_reference_entries(evidence: EvidenceExcerpt) -> list[dict[str, Any]]:
    """Extract source-preserved entries from one Evidence unit containing a lookup list."""

    table_entries = _markdown_table_reference_entries(evidence)
    if len(table_entries) >= 4:
        return table_entries
    keyword_entries = _keyword_index_reference_entries(evidence)
    if keyword_entries is not None:
        return keyword_entries

    lines = _line_spans(evidence.content)
    entries: list[dict[str, Any]] = []
    active_catalog = _evidence_has_inline_catalog_context(evidence)
    current_heading: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        line_text = line["text"]
        if _is_inline_catalog_heading(line_text):
            active_catalog = True
            if not _is_inline_table_column_heading(line_text):
                current_heading = _compact(line_text)
            index += 1
            continue

        entry_head = _split_inline_entry_line(line_text, allow_broad=active_catalog)
        if not entry_head:
            index += 1
            continue
        label, inline_description = entry_head

        description_indices: list[int] = []
        cursor = index + 1
        if inline_description:
            description = inline_description
            end_line = line
        else:
            while cursor < len(lines) and not lines[cursor]["text"]:
                cursor += 1
            while cursor < len(lines):
                current = lines[cursor]
                text = current["text"]
                if not text:
                    break
                if _is_inline_catalog_heading(text):
                    break
                if _split_inline_entry_line(text, allow_broad=active_catalog):
                    break
                description_indices.append(cursor)
                if len(description_indices) >= 3:
                    break
                cursor += 1

            if not description_indices:
                index += 1
                continue
            description = _compact(" ".join(lines[item]["text"] for item in description_indices))
            end_line = lines[description_indices[-1]]

        if not description or _is_inline_catalog_heading(description):
            index += 1
            continue
        if not active_catalog and len(description) < 8:
            index += 1
            continue

        entry_content = evidence.content[line["start"] : end_line["end"]].strip()
        if not active_catalog and "\n" not in entry_content and len(entry_content.split()) < 3:
            index += 1
            continue
        locator = dict(evidence.locator)
        locator.update(
            {
                "entry_label": label,
                "char_start": line["start"],
                "char_end": end_line["end"],
            }
        )
        if current_heading:
            locator["entry_category"] = current_heading
        entries.append(
            {
                "title": label,
                "content": entry_content,
                "source_evidence_id": evidence.evidence_id,
                "locator": locator,
            }
        )
        index = max(cursor, index + 1)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        key = "|".join(
            (
                _normalize_inline_entry_label(entry["title"]),
                _compact(str(entry["locator"].get("entry_category", ""))).casefold(),
            )
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _catalog_entry_identity(entry: dict[str, Any]) -> tuple[str, str]:
    return (
        _normalize_inline_entry_label(str(entry.get("title") or "")),
        _compact(str(entry.get("content") or "")).casefold(),
    )


def _catalog_display_stem(value: str) -> str:
    stem = re.sub(r"\s+lookup catalog\s*$", "", _compact(value), flags=re.IGNORECASE)
    stem = _CATALOG_CHUNK_SUFFIX.sub("", stem).rstrip(" :")
    return stem or "Source-preserved entries"


def _merge_inline_catalog_group(
    *,
    corpus_snapshot_id: str,
    catalogs: list[dict[str, Any]],
    evidence_order: dict[str, int],
    suggested_title: str | None = None,
    detail: str = "merged source-section inline lookup entries",
) -> dict[str, Any]:
    ordered_catalogs = sorted(
        catalogs,
        key=lambda item: min(
            (evidence_order.get(evidence_id, 10**9) for evidence_id in item["evidence_ids"]),
            default=10**9,
        ),
    )
    evidence_ids = sorted(
        {
            evidence_id
            for catalog in ordered_catalogs
            for evidence_id in catalog["evidence_ids"]
        },
        key=lambda evidence_id: (evidence_order.get(evidence_id, 10**9), evidence_id),
    )
    entries: list[dict[str, Any]] = []
    seen_entries: set[tuple[str, str]] = set()
    for catalog in ordered_catalogs:
        group_title = _catalog_display_stem(catalog["suggested_title"])
        for raw_entry in catalog.get("catalog_entries") or []:
            identity = _catalog_entry_identity(raw_entry)
            if identity in seen_entries:
                continue
            seen_entries.add(identity)
            entry = dict(raw_entry)
            locator = dict(entry.get("locator") or {})
            locator.setdefault("catalog_group", group_title)
            entry["locator"] = locator
            entries.append(entry)

    title = suggested_title or (
        f"{_catalog_display_stem(ordered_catalogs[0]['suggested_title'])} lookup catalog"
    )
    source_id = ordered_catalogs[0]["source_id"]
    catalog_id = _stable_id(
        "rcat",
        corpus_snapshot_id,
        "merged_inline_entry_catalog",
        source_id,
        *evidence_ids,
        *(
            f"{title_part}\x1e{content_part}"
            for title_part, content_part in sorted(seen_entries)[:64]
        ),
    )
    sample = [
        {
            "evidence_id": entry["source_evidence_id"],
            "title": entry["title"],
            "locator": entry["locator"],
            "char_count": len(entry["content"]),
        }
        for entry in entries[:5]
    ]
    return {
        "catalog_id": catalog_id,
        "source_id": source_id,
        "structure": "inline_entry_catalog",
        "detail": detail,
        "entry_count": len(entries),
        "evidence_ids": evidence_ids,
        "catalog_entries": entries,
        "common_field_labels": [],
        "sample_entries": sample,
        "suggested_title": title,
    }


def _catalog_fragment_key(
    catalog: dict[str, Any], evidence_by_id: dict[str, EvidenceExcerpt]
) -> tuple[str, str, str] | None:
    if catalog.get("structure") != "inline_entry_catalog":
        return None
    evidence = [
        evidence_by_id[evidence_id]
        for evidence_id in catalog.get("evidence_ids") or []
        if evidence_id in evidence_by_id
    ]
    if len(evidence) != 1:
        return None
    item = evidence[0]
    url = str(item.locator.get("url") or "").split("#", 1)[0].strip().casefold()
    section = _compact(
        str(
            item.locator.get("section_path")
            or item.locator.get("heading")
            or item.title
        )
    ).casefold()
    section = _CATALOG_CHUNK_SUFFIX.sub("", section).rstrip(" :")
    if not url or not section:
        return None
    return (catalog["source_id"], url, section)


def _merge_fragmented_reference_catalogs(
    corpus_snapshot_id: str,
    catalogs: list[dict[str, Any]],
    evidence_by_id: dict[str, EvidenceExcerpt],
    evidence_order: dict[str, int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for catalog in catalogs:
        key = _catalog_fragment_key(catalog, evidence_by_id)
        if key is None:
            passthrough.append(catalog)
        else:
            grouped.setdefault(key, []).append(catalog)
    for members in grouped.values():
        if len(members) == 1:
            passthrough.append(members[0])
        else:
            passthrough.append(
                _merge_inline_catalog_group(
                    corpus_snapshot_id=corpus_snapshot_id,
                    catalogs=members,
                    evidence_order=evidence_order,
                )
            )
    return passthrough


def _catalog_preference_key(
    catalog: dict[str, Any],
    evidence_by_id: dict[str, EvidenceExcerpt],
    evidence_order: dict[str, int],
) -> tuple[int, int, int, str]:
    urls = [
        str(evidence_by_id[evidence_id].locator.get("url") or "").casefold()
        for evidence_id in catalog.get("evidence_ids") or []
        if evidence_id in evidence_by_id
    ]
    aggregate_penalty = sum(
        any(marker in url for marker in _AGGREGATE_CATALOG_URL_MARKERS)
        for url in urls
    )
    first_order = min(
        (evidence_order.get(evidence_id, 10**9) for evidence_id in catalog["evidence_ids"]),
        default=10**9,
    )
    return (aggregate_penalty, len(catalog["evidence_ids"]), first_order, catalog["catalog_id"])


def _deduplicate_reference_catalogs(
    catalogs: list[dict[str, Any]],
    evidence_by_id: dict[str, EvidenceExcerpt],
    evidence_order: dict[str, int],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, tuple[tuple[str, str], ...]], list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for catalog in catalogs:
        entries = catalog.get("catalog_entries") or []
        if not entries:
            passthrough.append(catalog)
            continue
        signature = tuple(sorted({_catalog_entry_identity(entry) for entry in entries}))
        buckets.setdefault((catalog["source_id"], signature), []).append(catalog)
    for members in buckets.values():
        passthrough.append(
            min(
                members,
                key=lambda item: _catalog_preference_key(
                    item, evidence_by_id, evidence_order
                ),
            )
        )
    return passthrough


def _environment_catalog(catalog: dict[str, Any]) -> bool:
    entries = catalog.get("catalog_entries") or []
    if len(entries) < 3:
        return False
    matches = sum(
        bool(_ENVIRONMENT_VARIABLE_LABEL.fullmatch(_compact(str(entry.get("title") or ""))))
        for entry in entries
    )
    return matches / len(entries) >= 0.8


def _shortcut_label(value: str) -> bool:
    label = _strip_inline_code_marks(value)
    normalized_plus = re.sub(r"\s*\+\s*", "+", label)
    return bool(
        label in _INLINE_SYMBOL_KEYS
        or _INLINE_COMBO_LABEL.match(normalized_plus)
        or _INLINE_SPECIAL_KEY_LABEL.match(label)
        or _INLINE_FUNCTION_KEY_LABEL.match(label)
        or _INLINE_SINGLE_KEY_LABEL.match(label)
        or _INLINE_ARROW_KEY_LABEL.match(label)
        or _INLINE_MOUSE_OR_JOYSTICK_LABEL.match(label)
        or _INLINE_NUMERIC_RANGE_LABEL.match(label)
        or _INLINE_QUALIFIED_KEY_LABEL.match(label)
    )


def _shortcut_catalog(catalog: dict[str, Any]) -> bool:
    entries = catalog.get("catalog_entries") or []
    return bool(entries) and (
        sum(_shortcut_label(str(entry.get("title") or "")) for entry in entries)
        / len(entries)
        >= 0.75
    )


def _catalog_capability_branch(
    catalog: dict[str, Any], evidence_by_id: dict[str, EvidenceExcerpt]
) -> str:
    branches = {
        _compact(str(evidence_by_id[evidence_id].locator.get("capability_branch") or ""))
        for evidence_id in catalog.get("evidence_ids") or []
        if evidence_id in evidence_by_id
    }
    branches.discard("")
    return next(iter(branches)) if len(branches) == 1 else ""


def _merge_reference_catalog_families(
    corpus_snapshot_id: str,
    catalogs: list[dict[str, Any]],
    evidence_by_id: dict[str, EvidenceExcerpt],
    evidence_order: dict[str, int],
) -> list[dict[str, Any]]:
    environment_groups: dict[str, list[dict[str, Any]]] = {}
    remaining: list[dict[str, Any]] = []
    for catalog in catalogs:
        if _environment_catalog(catalog):
            environment_groups.setdefault(catalog["source_id"], []).append(catalog)
        else:
            remaining.append(catalog)
    for source_id, members in environment_groups.items():
        if len(members) == 1:
            remaining.append(members[0])
        else:
            remaining.append(
                _merge_inline_catalog_group(
                    corpus_snapshot_id=corpus_snapshot_id,
                    catalogs=members,
                    evidence_order=evidence_order,
                    suggested_title="Environment variables lookup catalog",
                    detail="merged environment-variable lookup entries",
                )
            )

    shortcut_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    final_catalogs: list[dict[str, Any]] = []
    for catalog in remaining:
        branch = _catalog_capability_branch(catalog, evidence_by_id)
        if branch and _shortcut_catalog(catalog):
            shortcut_groups.setdefault((catalog["source_id"], branch), []).append(catalog)
        else:
            final_catalogs.append(catalog)
    for (_, branch), members in shortcut_groups.items():
        if len(members) == 1:
            final_catalogs.append(members[0])
        else:
            final_catalogs.append(
                _merge_inline_catalog_group(
                    corpus_snapshot_id=corpus_snapshot_id,
                    catalogs=members,
                    evidence_order=evidence_order,
                    suggested_title=f"{branch} keyboard and mouse shortcuts lookup catalog",
                    detail="merged interface-shortcut lookup entries",
                )
            )
    return final_catalogs


_AUTOMATIC_CATALOG_CONTEXT_MARKERS = (
    "command",
    "function",
    "api",
    "parameter",
    "option",
    "setting",
    "shortcut",
    "keyboard",
    "mouse",
    "variable",
    "environment",
    "keyword",
    "index",
    "syntax",
    "format",
)

_INLINE_CATALOG_PROSE_LABELS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "here",
        "in",
        "into",
        "is",
        "it",
        "most",
        "of",
        "on",
        "or",
        "other",
        "see",
        "show",
        "shows",
        "start",
        "the",
        "then",
        "this",
        "to",
        "use",
        "when",
        "where",
        "with",
        "you",
    }
)
_INLINE_CATALOG_STRUCTURED_LABEL = re.compile(
    r"^(?:"
    r"--?[A-Za-z0-9][A-Za-z0-9_-]*|"
    r"[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+|"
    r"[A-Za-z][A-Za-z0-9]*[._:/+*-][A-Za-z0-9._:/+*-]*|"
    r"[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|"
    r"(?:Ctrl|Alt|Shift|Cmd|Command|F\d+)(?:[-+ ].*)?"
    r")$",
    flags=re.IGNORECASE,
)


def _inline_catalog_quality_profile(catalog: dict[str, Any]) -> dict[str, Any]:
    """Measure whether parsed inline entries have credible row boundaries.

    The inline parser is deliberately high recall, so prose can occasionally be
    split into fake label/value rows. This profile uses only generic boundary and
    identifier shapes; it never depends on a particular software product.
    """

    entries = list(catalog.get("catalog_entries") or [])
    if not entries:
        return {
            "entry_count": 0,
            "standalone_label_ratio": 0.0,
            "prose_continuation_ratio": 0.0,
            "structured_label_ratio": 0.0,
            "noise_label_ratio": 0.0,
            "obvious_fragment": False,
            "reasons": [],
        }

    standalone = 0
    prose_continuation = 0
    structured = 0
    stopword = 0
    symbol_or_numeric = 0
    for entry in entries:
        title = _compact(str(entry.get("title") or ""))
        content = str(entry.get("content") or "").strip()
        if title and re.match(
            rf"^{re.escape(title)}[ \t]*(?:\r?\n|$)", content, flags=re.IGNORECASE
        ):
            standalone += 1
        if title and content[: len(title)].casefold() == title.casefold():
            remainder = content[len(title) :]
            if re.match(r"[ \t]+[a-z]", remainder):
                prose_continuation += 1
        if _INLINE_CATALOG_STRUCTURED_LABEL.fullmatch(title):
            structured += 1
        if title.casefold() in _INLINE_CATALOG_PROSE_LABELS:
            stopword += 1
        if title and not any(character.isalpha() for character in title):
            symbol_or_numeric += 1

    total = len(entries)
    standalone_ratio = standalone / total
    prose_ratio = prose_continuation / total
    structured_ratio = structured / total
    noise_ratio = (stopword + symbol_or_numeric) / total
    title_context = _compact(
        f"{catalog.get('suggested_title', '')} {catalog.get('detail', '')}"
    ).casefold()
    control_context = any(
        marker in title_context
        for marker in ("keyboard", "mouse", "shortcut", "key assignment")
    )
    credible_control_rows = control_context and structured_ratio >= 0.03
    prose_fragment = (
        prose_ratio >= 0.55
        and standalone_ratio < 0.25
        and structured_ratio < 0.35
        and not credible_control_rows
    )
    noisy_prose_fragment = (
        prose_ratio >= 0.70
        and structured_ratio <= 0.25
        and noise_ratio >= 0.25
        and not credible_control_rows
    )
    symbol_fragment = symbol_or_numeric / total >= 0.35 and structured_ratio < 0.20
    reasons: list[str] = []
    if prose_fragment:
        reasons.append("labels_continue_as_prose")
    if noisy_prose_fragment:
        reasons.append("prose_labels_are_stopwords_or_symbols")
    if symbol_fragment:
        reasons.append("labels_are_code_punctuation_fragments")
    return {
        "entry_count": total,
        "standalone_label_ratio": round(standalone_ratio, 4),
        "prose_continuation_ratio": round(prose_ratio, 4),
        "structured_label_ratio": round(structured_ratio, 4),
        "noise_label_ratio": round(noise_ratio, 4),
        "obvious_fragment": bool(reasons),
        "reasons": reasons,
    }


def _reference_catalog_automation_profile(
    catalog: dict[str, Any], evidence_by_id: dict[str, EvidenceExcerpt]
) -> dict[str, Any]:
    """Prove when Python may preserve an exact catalog without semantic review.

    Only inline catalogs with exact, unique, source-anchored entries and an
    authored lookup context are automatic. Numbered/repeated-field groupings stay
    ambiguous because repetition alone does not establish reusable semantics.
    """

    entries = list(catalog.get("catalog_entries") or [])
    reasons: list[str] = []
    if catalog.get("structure") != "inline_entry_catalog" or len(entries) < 4:
        return {"automatic": False, "confidence": 0.0, "reasons": ["semantic_review_required"]}

    identities: set[tuple[str, str]] = set()
    exact_count = 0
    valid_count = 0
    for entry in entries:
        title = _compact(str(entry.get("title") or ""))
        content = str(entry.get("content") or "").strip()
        evidence = evidence_by_id.get(str(entry.get("source_evidence_id") or ""))
        if not title or not content or evidence is None:
            continue
        valid_count += 1
        identities.add(_catalog_entry_identity(entry))
        if content in evidence.content:
            exact_count += 1
    if valid_count != len(entries) or exact_count != len(entries):
        return {"automatic": False, "confidence": 0.0, "reasons": ["source_integrity_not_proven"]}
    if len(identities) != len(entries):
        return {"automatic": False, "confidence": 0.0, "reasons": ["duplicate_entries"]}
    reasons.extend(("all_entries_source_exact", "all_entries_unique"))

    quality = _inline_catalog_quality_profile(catalog)
    if quality["obvious_fragment"]:
        return {
            "automatic": False,
            "confidence": 0.0,
            "reasons": [*reasons, *quality["reasons"]],
            "quality": quality,
        }

    context = _compact(
        " ".join(
            str(value)
            for value in (
                catalog.get("suggested_title"),
                catalog.get("detail"),
                *(evidence_by_id[evidence_id].title for evidence_id in catalog.get("evidence_ids", []) if evidence_id in evidence_by_id),
                *(evidence_by_id[evidence_id].locator.get("heading", "") for evidence_id in catalog.get("evidence_ids", []) if evidence_id in evidence_by_id),
                *(evidence_by_id[evidence_id].locator.get("section_path", "") for evidence_id in catalog.get("evidence_ids", []) if evidence_id in evidence_by_id),
            )
            if value
        )
    ).casefold()
    labels = [str(entry.get("title") or "") for entry in entries]
    command_ratio = sum(bool(_KEYWORD_INDEX_IDENTIFIER.fullmatch(_compact(label))) for label in labels) / len(labels)
    authored_lookup_context = (
        any(marker in context for marker in _AUTOMATIC_CATALOG_CONTEXT_MARKERS)
        or command_ratio >= 0.75
        or quality["structured_label_ratio"] >= 0.80
    )
    recognized_shape = (
        command_ratio >= 0.75
        or quality["structured_label_ratio"] >= 0.75
        or _shortcut_catalog(catalog)
        or _environment_catalog(catalog)
        or ("keyword" in context and "index" in context)
    )
    if not authored_lookup_context or not recognized_shape:
        return {
            "automatic": False,
            "confidence": 0.75,
            "reasons": [*reasons, "lookup_semantics_require_review"],
            "quality": quality,
        }
    clean_boundaries = (
        quality["standalone_label_ratio"] >= 0.80
        or quality["structured_label_ratio"] >= 0.80
    )
    if not clean_boundaries:
        return {
            "automatic": False,
            "confidence": 0.8,
            "reasons": [*reasons, "entry_boundaries_require_review"],
            "quality": quality,
        }
    reasons.append("authored_lookup_semantics")
    reasons.append("catalog_entry_boundaries_clean")
    return {
        "automatic": True,
        "confidence": 1.0,
        "reasons": reasons,
        "quality": quality,
    }


def _automatic_reference_catalog_metadata(
    catalog: dict[str, Any], software_name: str
) -> dict[str, Any]:
    suggested = _compact(str(catalog.get("suggested_title") or "Reference catalog"))
    stem = _catalog_display_stem(suggested)
    context = f"{suggested} {catalog.get('detail', '')}".casefold()
    software = _compact(software_name)
    prefix = "" if stem.casefold().startswith(software.casefold()) else f"{software} "
    if "keyword" in context and "index" in context:
        title = f"{software} keyword-to-command index"
        category = "concept_reference"
        description = (
            f"Exact source-authored keyword-to-command or module mappings for {software}. "
            "Query this index when translating a scientific topic or desired capability into relevant documented operations."
        )
    elif "shortcut" in context or "keyboard" in context or "mouse" in context:
        title = f"{prefix}{stem}".strip()
        category = "shortcut_reference"
        description = (
            f"Exact documented keyboard and mouse control mappings for {software}. "
            "Query this catalog when selecting an interface shortcut or checking the action bound to a documented control."
        )
    elif "environment" in context or _environment_catalog(catalog):
        title = f"{prefix}{stem}".strip()
        category = "configuration_reference"
        description = (
            f"Exact documented environment-variable and configuration entries for {software}. "
            "Query this catalog when identifying a variable, its documented role, or the setting associated with a runtime behavior."
        )
    elif "command" in context or "module" in context:
        title = f"{prefix}{stem}".strip()
        category = "command_catalog"
        description = (
            f"Exact documented command or module names and concise capability descriptions for {software}. "
            "Query this catalog when discovering an operation by purpose or checking the documented scope of a named command."
        )
    else:
        title = f"{prefix}{stem}".strip()
        category = "source_catalog"
        description = (
            f"Exact source-authored lookup entries for {software}, preserved with their original locators. "
            "Query this catalog when a downstream task requires one of the documented names, values, or mappings in this section."
        )
    return {
        "title": title[:300],
        "description": description,
        "category": category,
        "injection_policy": "on_demand",
        "usage_tags": _infer_reference_usage_tags(title, category, stem, source_catalog=True),
    }


def _metadata_only_repeated_catalog(
    entries: list[EvidenceExcerpt], common_fields: tuple[str, ...]
) -> bool:
    fields = {_compact(item).casefold() for item in common_fields}
    generic_titles = {
        _compact(item.title).casefold()
        for item in entries
    }
    return (
        {"available at", "latest change"}.issubset(fields)
        and generic_titles.issubset({"source code", "source", "metadata"})
    )


def _reference_like_evidence_profile(evidence: EvidenceExcerpt) -> dict[str, Any]:
    text = _compact(
        " ".join(
            str(value)
            for value in (
                evidence.title,
                evidence.locator.get("heading"),
                evidence.locator.get("section_path"),
                evidence.locator.get("chapter_heading"),
                evidence.locator.get("url"),
            )
            if value
        )
    ).casefold()
    tags: list[str] = []
    reasons: list[str] = []
    score = 0
    for tag, markers in _REFERENCE_LIKE_TITLE_MARKERS:
        hits = [marker for marker in markers if marker in text]
        if hits:
            tags.append(tag)
            score += 2 if tag != "concept_reference" else 1
            reasons.append(f"title_or_locator:{hits[0]}")

    labels = _reference_field_labels(evidence.content)
    if len(labels) >= 2:
        score += 2
        tags.append("source_catalog")
        reasons.append("repeated_field_labels")

    inline_entries = _inline_reference_entries(evidence)
    if len(inline_entries) >= 4:
        score += min(6, 2 + len(inline_entries) // 8)
        tags.append("command_catalog")
        tags.append("source_catalog")
        reasons.append(f"inline_entries:{len(inline_entries)}")

    if evidence.kind in {"html_table", "pdf_table", "html_code"}:
        score += 1
        tags.append("source_catalog")
        reasons.append(f"kind:{evidence.kind}")

    return {
        "score": score,
        "tags": list(dict.fromkeys(tags)),
        "reasons": reasons[:6],
        "inline_entry_count": len(inline_entries),
    }


def _build_reference_catalogs(
    corpus_snapshot_id: str, evidence_items: list[EvidenceExcerpt]
) -> list[dict[str, Any]]:
    """Find repeated source structures that should be preserved, not summarized.

    The detector intentionally uses only heading shape and repeated field labels. It
    does not assume a software-specific concept such as a function or command.
    """

    numbered_groups: dict[tuple[str, str], list[EvidenceExcerpt]] = {}
    field_groups: dict[tuple[str, tuple[str, ...]], list[EvidenceExcerpt]] = {}
    inline_catalogs: list[tuple[EvidenceExcerpt, list[dict[str, Any]]]] = []
    for evidence in evidence_items:
        title = _compact(evidence.title)
        numbered = _NUMBERED_REFERENCE_ENTRY.match(title)
        if numbered:
            chapter = numbered.group("number").split(".", 1)[0]
            numbered_groups.setdefault((evidence.source_id, chapter), []).append(evidence)
        labels = _reference_field_labels(evidence.content)
        if len(labels) >= 2:
            field_groups.setdefault((evidence.source_id, labels), []).append(evidence)
        inline_entries = _inline_reference_entries(evidence)
        if len(inline_entries) >= 4:
            inline_catalogs.append((evidence, inline_entries[:4_096]))

    catalogs: list[dict[str, Any]] = []
    emitted_entry_sets: set[tuple[str, ...]] = set()

    def add_catalog(
        *,
        source_id: str,
        structure: str,
        entries: list[EvidenceExcerpt],
        detail: str,
        common_fields: tuple[str, ...] = (),
    ) -> None:
        ordered = sorted(entries, key=_reference_entry_sort_key)
        evidence_ids = tuple(item.evidence_id for item in ordered)
        if len(evidence_ids) < 4 or evidence_ids in emitted_entry_sets:
            return
        emitted_entry_sets.add(evidence_ids)
        catalog_id = _stable_id(
            "rcat",
            corpus_snapshot_id,
            structure,
            source_id,
            *evidence_ids,
        )
        sample = [
            {
                "evidence_id": item.evidence_id,
                "title": item.title,
                "locator": item.locator,
                "char_count": len(item.content),
            }
            for item in ordered[:5]
        ]
        catalogs.append(
            {
                "catalog_id": catalog_id,
                "source_id": source_id,
                "structure": structure,
                "detail": detail,
                "entry_count": len(ordered),
                "evidence_ids": list(evidence_ids),
                "common_field_labels": list(common_fields),
                "sample_entries": sample,
                "suggested_title": f"Source-preserved entry catalog ({detail})",
            }
        )

    for (source_id, chapter), entries in numbered_groups.items():
        common = set(_reference_field_labels(entries[0].content)) if entries else set()
        for entry in entries[1:]:
            common.intersection_update(_reference_field_labels(entry.content))
        add_catalog(
            source_id=source_id,
            structure="numbered_heading_series",
            entries=entries,
            detail=f"{chapter}.* numbered headings",
            common_fields=tuple(sorted(common)),
        )

    for (source_id, labels), entries in field_groups.items():
        if _metadata_only_repeated_catalog(entries, labels):
            continue
        add_catalog(
            source_id=source_id,
            structure="repeated_field_schema",
            entries=entries,
            detail="repeated source field schema",
            common_fields=labels,
        )

    for evidence, inline_entries in inline_catalogs:
        catalog_id = _stable_id(
            "rcat",
            corpus_snapshot_id,
            "inline_entry_catalog",
            evidence.source_id,
            evidence.evidence_id,
            *[entry["title"] for entry in inline_entries[:64]],
        )
        sample = [
            {
                "evidence_id": entry["source_evidence_id"],
                "title": entry["title"],
                "locator": entry["locator"],
                "char_count": len(entry["content"]),
            }
            for entry in inline_entries[:5]
        ]
        catalogs.append(
            {
                "catalog_id": catalog_id,
                "source_id": evidence.source_id,
                "structure": "inline_entry_catalog",
                "detail": "single Evidence inline lookup entries",
                "entry_count": len(inline_entries),
                "evidence_ids": [evidence.evidence_id],
                "catalog_entries": inline_entries,
                "common_field_labels": [],
                "sample_entries": sample,
                "suggested_title": f"{evidence.title} lookup catalog",
            }
        )

    evidence_by_id = {item.evidence_id: item for item in evidence_items}
    evidence_order = {
        item.evidence_id: index for index, item in enumerate(evidence_items)
    }
    catalogs = _merge_fragmented_reference_catalogs(
        corpus_snapshot_id,
        catalogs,
        evidence_by_id,
        evidence_order,
    )
    catalogs = _deduplicate_reference_catalogs(
        catalogs,
        evidence_by_id,
        evidence_order,
    )
    catalogs = _merge_reference_catalog_families(
        corpus_snapshot_id,
        catalogs,
        evidence_by_id,
        evidence_order,
    )
    # High-recall inline parsing can mistake wrapped prose or code punctuation
    # for label/value rows. Demote only structurally proven fragments; their
    # underlying Evidence remains intact and can still be considered as an
    # ordinary semantic candidate.
    catalogs = [
        catalog
        for catalog in catalogs
        if catalog.get("structure") != "inline_entry_catalog"
        or not _inline_catalog_quality_profile(catalog)["obvious_fragment"]
    ]
    return sorted(catalogs, key=lambda item: (item["source_id"], item["catalog_id"]))

def _infer_reference_usage_tags(
    title: str,
    category: str,
    content: str,
    *,
    source_catalog: bool,
) -> list[str]:
    text = _compact(" ".join([title, category, content])).casefold()
    tags: list[str] = []
    marker_groups = [
        (
            "command_catalog",
            (
                "command",
                "operation",
                "operator",
                "tool",
                "shortcut",
                "hotkey",
                "key binding",
                "命令",
                "指令",
                "操作",
                "算子",
                "快捷键",
                "按键",
            ),
        ),
        ("function_catalog", ("function", "built-in", "dictionary", "函数", "内置", "字典")),
        ("parameter_reference", ("parameter", "argument", "option", "flag", "参数", "选项")),
        ("syntax_reference", ("syntax", "grammar", "expression", "formula", "语法", "表达式", "公式")),
        ("file_format_reference", ("file format", "extension", "schema file", "文件格式", "扩展名")),
        (
            "ui_reference",
            (
                "menu",
                "button",
                "dialog",
                "window",
                "tab",
                "toolbar",
                "shortcut",
                "keyboard",
                "hotkey",
                "key binding",
                "mouse",
                "joystick",
                "菜单",
                "按钮",
                "对话框",
                "窗口",
                "工具栏",
                "快捷键",
                "键盘",
                "按键",
                "鼠标",
                "左键",
                "右键",
                "滚轮",
                "摇杆",
            ),
        ),
        ("configuration_reference", ("configuration", "setting", "preference", "配置", "设置")),
        ("data_schema_reference", ("schema", "field", "column", "record", "数据结构", "字段")),
        ("concept_reference", ("concept", "overview", "definition", "概念", "定义")),
    ]
    for tag, markers in marker_groups:
        if any(marker in text for marker in markers):
            tags.append(tag)
    if source_catalog:
        if "ui_reference" not in tags and re.search(
            r"(?:ctrl\s*\+|alt\s*\+|shift\s*\+|cmd\s*\+|enter|esc|backspace|tab|"
            r"\bF(?:[1-9]|1[0-9]|2[0-4])\b|快捷键|键盘|按键|鼠标|左键|右键|滚轮|"
            r"摇杆|方向键)",
            text,
            flags=re.IGNORECASE,
        ):
            tags.append("ui_reference")
        if "command_catalog" not in tags:
            identifier_tokens = {
                token
                for token in re.findall(r"\b[a-z][a-z0-9_.:-]{1,64}\b", text)
                if token
                not in {
                    "source",
                    "catalog",
                    "entry",
                    "entries",
                    "lookup",
                    "reference",
                    "manual",
                }
            }
            if len(identifier_tokens) >= 3:
                tags.append("command_catalog")
    if source_catalog:
        tags.append("source_catalog")
    elif not tags:
        tags.append("reference_summary")
    return list(dict.fromkeys(tags))
