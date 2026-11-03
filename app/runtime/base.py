from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class AgentRuntimeUnavailableError(RuntimeError):
    """Raised when a production Agent runtime has not been configured yet."""


class AgentRuntimeConfigurationError(RuntimeError):
    """Raised when an invocation cannot be represented by the selected runtime."""


class AgentRuntimeExecutionError(RuntimeError):
    """Base class for normalized runtime failures."""

    category = "runtime_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


class AgentRuntimeAuthenticationError(AgentRuntimeExecutionError):
    category = "authentication"


class AgentRuntimeProtocolError(AgentRuntimeExecutionError):
    category = "protocol"


class AgentRuntimeTimeoutError(AgentRuntimeExecutionError):
    category = "timeout"


class AgentRuntimeTransportError(AgentRuntimeExecutionError):
    category = "transport"


class AgentRuntimeCancelledError(AgentRuntimeExecutionError):
    category = "cancelled"


class AgentRuntimeBusinessCompleted(AgentRuntimeExecutionError):
    category = "business_complete"


@dataclass(frozen=True, slots=True)
class AgentInvocation:
    run_id: str
    session_id: str
    role: str
    prompt: str | Sequence[Mapping[str, Any]]
    model: Mapping[str, Any]
    allowed_tools: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    session_id: str
    final_response: str
    finish_reason: str | None = None
    event_count: int = 0
    usage: Mapping[str, Any] = field(default_factory=dict)


class AgentRuntime(ABC):
    """Stable application boundary implemented by Fake and DSH runtimes."""

    @abstractmethod
    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, session_id: str) -> bool:
        raise NotImplementedError

    async def stop_completed_session(self, session_id: str) -> bool:
        """Stop model work after the host has durably accepted the business result."""
        return await self.cancel(session_id)


class UnavailableAgentRuntime(AgentRuntime):
    async def run(self, invocation: AgentInvocation) -> AgentTurnResult:
        raise AgentRuntimeUnavailableError(
            "No Agent runtime is configured for this pipeline."
        )

    async def cancel(self, session_id: str) -> bool:
        return False
