"""Shared valid Trajectory samples for tests that exercise stage behavior.

Keep exact field assertions and deliberately invalid or historical payloads in
contract tests. These helpers do not validate or infer fields from production
schemas, so contract changes remain visible in those tests.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def trajectory_submission(
    *, instruction: str, citations: list[dict[str, Any]],
    expected_observation: str = "", **updates: Any,
) -> dict[str, Any]:
    return {
        "steps": [{
            "ordinal": 1,
            "instruction": instruction,
            "expected_observation": expected_observation,
            "citations": deepcopy(citations),
        }],
        **updates,
    }


def trajectory_draft(*, task_id: str, **submission: Any) -> dict[str, Any]:
    payload = trajectory_submission(**submission)
    return {
        "task_id": task_id,
        "citations": deepcopy([
            citation for step in payload["steps"] for citation in step["citations"]
        ]),
        **payload,
    }
