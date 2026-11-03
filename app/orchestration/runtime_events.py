"""Consistent audit delivery for model waits, retries, and failures in every stage."""

from __future__ import annotations

from typing import Any, Callable

from ..storage import RunStore


def make_runtime_event_sink(
    store: RunStore, *, agent: str, attempt_id: str, source_id: str | None = None,
) -> Callable[[str, dict[str, Any]], None]:
    def sink(event: str, payload: dict[str, Any]) -> None:
        level = ("error" if event in {"model_failed", "model_retry_stopped"}
                 or payload.get("status") in {"error", "failed"} or payload.get("ok") is False else
                 "warning" if event in {"model_retry_scheduled", "model_cancelled"} else "info")
        store.emit(event, {**payload, "attempt_id": attempt_id},
                   agent=agent, source_id=source_id, level=level)

    return sink
