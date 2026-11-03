"""Server-owned generation budgets, fixed for the lifetime of each run."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Literal

from .config import Settings


GenerationCoverage = Literal["low", "medium", "high", "ultra"]
GENERATION_COVERAGE_LIMITS: Mapping[str, tuple[int, int]] = MappingProxyType({
    "low": (8, 16),
    "medium": (16, 24),
    "high": (22, 28),
    "ultra": (28, 36),
})


def generation_limits(settings: Settings) -> dict[str, int]:
    return {
        "max_task_candidates": settings.max_task_candidates,
        "max_knowledge_candidates": settings.max_knowledge_candidates,
    }


def generation_coverage_state(level: GenerationCoverage) -> dict[str, Any]:
    if level not in GENERATION_COVERAGE_LIMITS:
        raise ValueError("Unknown generation coverage level")
    tasks, references = GENERATION_COVERAGE_LIMITS[level]
    return {
        "generation_coverage": level,
        "generation_limits": {"max_task_candidates": tasks, "max_knowledge_candidates": references},
    }


def settings_for_generation_coverage(settings: Settings, state: Mapping[str, Any]) -> Settings:
    """Resolve a saved budget without altering shared process settings.

    A historical run without a selection initially keeps the previous process
    budget. Its first execution saves that budget so later restarts keep it too.
    """
    level = state.get("generation_coverage")
    saved = state.get("generation_limits")
    if level is not None:
        canonical = generation_coverage_state(level)["generation_limits"]
        if saved is not None and saved != canonical:
            raise ValueError("Saved generation limits do not match the selected coverage level")
        saved = canonical
    if saved is None:
        return settings
    if (not isinstance(saved, Mapping)
            or set(saved) != {"max_task_candidates", "max_knowledge_candidates"}
            or any(type(value) is not int or value < 0 for value in saved.values())):
        raise ValueError("Invalid saved generation limits")
    return replace(settings, **dict(saved))


def saved_generation_coverage_state(settings: Settings, state: Mapping[str, Any]) -> dict[str, Any]:
    selected = state.get("generation_coverage")
    resolved = settings_for_generation_coverage(settings, state)
    result: dict[str, Any] = {"generation_limits": generation_limits(resolved)}
    if selected is not None:
        result["generation_coverage"] = selected
    return result
