"""Agent Trace and Execution Event domain models.

Records every step of the agent's execution loop:
- Iteration number
- Model call & usage
- Tool name, arguments, and response
- Latency (ms)
- Errors
- Termination reason
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from backend.domain.base import DomainEntity


class ExecutionEventType(StrEnum):
    """Types of events recorded during agent loop execution."""

    ITERATION_START = "iteration_start"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    MODEL_ERROR = "model_error"
    FINAL_ANSWER = "final_answer"
    TERMINATION = "termination"
    ERROR = "error"


class TerminationReason(StrEnum):
    """Reason why the agent loop terminated."""

    NORMAL = "normal"
    MAX_ITERATIONS = "max_iterations"
    MAX_TOOL_CALLS = "max_tool_calls"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    TOOL_LOOP_DETECTED = "tool_loop_detected"
    FORBIDDEN_TOOL = "forbidden_tool"
    BUDGET_EXCEEDED = "budget_exceeded"
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    ERROR = "error"


class ExecutionEvent(BaseModel):
    """A single recorded event in the agent loop execution trace."""

    id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    iteration: int = 0
    timestamp: float = Field(default_factory=time.time)
    event_type: ExecutionEventType
    model_call: dict[str, Any] | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_response: Any | None = None
    latency_ms: float = 0.0
    error: str | None = None
    termination_reason: TerminationReason | None = None

    model_config = {"use_enum_values": True}


class AgentTrace(DomainEntity):
    """Complete execution trace for an agent trial."""

    trial_id: uuid.UUID
    events: list[ExecutionEvent] = Field(default_factory=list)
    termination_reason: TerminationReason | None = None
    total_iterations: int = 0
    total_tool_calls: int = 0
    total_latency_ms: float = 0.0
    final_answer: str | None = None

    def add_event(self, event: ExecutionEvent) -> None:
        """Append an execution event and accumulate latency."""
        self.events.append(event)
        if event.latency_ms > 0:
            self.total_latency_ms += event.latency_ms

    def get_events_by_type(self, event_type: ExecutionEventType) -> list[ExecutionEvent]:
        """Filter events by event type."""
        target_val = event_type.value if hasattr(event_type, "value") else event_type
        return [e for e in self.events if e.event_type == target_val]

    def get_tool_calls(self) -> list[ExecutionEvent]:
        """Return all tool call events."""
        return [
            e for e in self.events
            if e.event_type in (ExecutionEventType.TOOL_CALL.value, ExecutionEventType.TOOL_CALL)
        ]

    def has_errors(self) -> bool:
        """Check if any error was recorded during execution."""
        error_types = {
            ExecutionEventType.TOOL_ERROR.value,
            ExecutionEventType.MODEL_ERROR.value,
            ExecutionEventType.ERROR.value,
        }
        return any(
            e.error is not None or e.event_type in error_types
            for e in self.events
        )

    def to_summary(self) -> dict[str, Any]:
        """Return a structured summary of the execution trace."""
        return {
            "trial_id": str(self.trial_id),
            "termination_reason": self.termination_reason,
            "total_iterations": self.total_iterations,
            "total_tool_calls": self.total_tool_calls,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "final_answer": self.final_answer,
            "event_count": len(self.events),
            "has_errors": self.has_errors(),
        }
