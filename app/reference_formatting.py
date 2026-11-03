"""Validation primitives for experimental, evidence-backed Reference formatting.

The source-preserved pipeline remains authoritative. A formatting experiment may
use this conservative check before accepting whitespace-only model changes.
"""

from __future__ import annotations


def preserves_non_whitespace(source: str, formatted: str) -> bool:
    """Check that all non-whitespace characters remain in their original order.

    This is necessary but not sufficient for semantic equivalence: code and
    tables can depend on whitespace, so callers must also validate structure.
    """
    return bool(source.strip()) and bool(formatted.strip()) and (
        "".join(source.split()) == "".join(formatted.split())
    )
