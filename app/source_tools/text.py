"""Deterministic identifiers and exact-text boundaries shared by Source tools."""
from __future__ import annotations

import hashlib
import re


def _stable_id(prefix: str, *parts: str, length: int = 20) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8", errors="replace")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def _compact_text(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _split_long_text(text: str, maximum: int) -> list[str]:
    if len(text) <= maximum:
        return [text]
    paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for paragraph in paragraphs:
        if len(paragraph) > maximum:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_length = 0
            for offset in range(0, len(paragraph), maximum):
                chunks.append(paragraph[offset : offset + maximum])
            continue
        projected = current_length + len(paragraph) + (2 if current else 0)
        if current and projected > maximum:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_length = len(paragraph)
        else:
            current.append(paragraph)
            current_length = projected
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text[:maximum]]


def _excerpt_ranges(text: str, maximum: int) -> list[tuple[int, int]]:
    if not text.strip():
        return []
    if len(text) <= maximum:
        return [(0, len(text))]
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + maximum)
        if end < len(text):
            boundary = max(text.rfind("\n", start + maximum // 2, end), text.rfind(" ", start + maximum // 2, end))
            if boundary > start:
                end = boundary + 1
        if text[start:end].strip():
            ranges.append((start, end))
        start = end
    return ranges
