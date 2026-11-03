from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from .base import AgentInvocation, AgentRuntime, AgentTurnResult


class FakeAgentRuntime(AgentRuntime):
    """Deterministic runtime for orchestration and contract tests."""

    def __init__(self, results: Iterable[AgentTurnResult] = ()) -> None:
        self.invocations: list[AgentInvocation] = []
        self.cancelled_sessions: list[str] = []
        self._results = deque(results)

    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        self.invocations.append(invocation)
        if self._results:
            result = self._results.popleft()
            if result.session_id != invocation.session_id:
                raise ValueError("Scripted result session_id does not match invocation")
            return result
        return AgentTurnResult(
            session_id=invocation.session_id,
            final_response="",
            finish_reason="completed",
        )

    async def cancel(self, session_id: str) -> bool:
        self.cancelled_sessions.append(session_id)
        return True

