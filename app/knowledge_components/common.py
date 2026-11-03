"""Shared identifiers, descriptions, limits, and filename helpers for knowledge stages."""

from __future__ import annotations

import hashlib
import re
from typing import Literal, TypeVar


T = TypeVar("T")

CATALOG_DESCRIPTION_MIN_CHARS = 1
GENERIC_CATALOG_DESCRIPTION_PREFIX = "Source-preserved catalog with "
REFERENCE_ORDINARY_REVIEW_LIMIT = 6
REFERENCE_ORDINARY_CONTEXT_CHARS = 500
REFERENCE_CATALOG_SAMPLE_SIZE = 6
REFERENCE_PARTITION_REVIEW_CHARS = 10_000
REFERENCE_PARTITION_OUTLINE_LIMIT = 80
REFERENCE_PARTITION_MAX_GROUPS = 32


def catalog_description_fallback(entry_count: int) -> str:
    return (
        f"{GENERIC_CATALOG_DESCRIPTION_PREFIX}{entry_count} independently retrievable entries. "
        "Each entry retains exact normalized source text and its source locator."
    )


def catalog_description_errors(description: str) -> list[str]:
    compact = " ".join(str(description or "").split())
    return [] if compact else ["Catalog Reference description must not be empty"]


def resolved_catalog_description(description: str | None, entry_count: int) -> str:
    compact = " ".join(str(description or "").split())
    return compact or catalog_description_fallback(entry_count)

KnowledgeStage = Literal[
    "task_discovery",
    "task_trajectory",
    "reference_build",
    "trajectory_inference",
    "skill_induction",
]


def _stable_id(prefix: str, *parts: str, length: int = 20) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8", errors="replace")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _safe_filename(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:100] or fallback


_WINDOWS_RESERVED_FILENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _safe_skill_markdown_stem(value: str, fallback: str) -> str:
    cleaned = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "_", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    cleaned = cleaned[:150].rstrip(" .") or fallback
    if cleaned.upper() in _WINDOWS_RESERVED_FILENAMES:
        cleaned = f"{cleaned}_skill"
    return cleaned


def _excerpt_quote(content: str, maximum: int = 360) -> str:
    """Select an exact, readable source substring for deterministic fallback candidates."""
    compact = content.strip()
    if len(compact) <= maximum:
        return compact
    boundary = max(compact.rfind("\n", 0, maximum), compact.rfind(". ", 0, maximum))
    if boundary < max(80, maximum // 3):
        boundary = maximum
    return compact[:boundary].rstrip()
