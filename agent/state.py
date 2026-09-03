"""Execution context and conversation state for a single trial.

ExecutionContext is created per-trial and carries:
  - trace_id for log correlation
  - conversation message history
  - recorded tool calls
  - cost / token counters
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ToolCallRecord:
    """In-memory record of a single tool invocation."""

    seq: int
    tool_name: str
    arguments: dict[str, Any]
    response: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    latency_ms: int = 0
    retry_count: int = 0
    trace_id: str = ""


@dataclass
class ExecutionContext:
    """Mutable state for one agent trial execution."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    messages: list[dict[str, str]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    step: int = 0

    # Counters
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    start_time: float = field(default_factory=time.time)

    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        response: dict[str, Any],
        status: str = "success",
        latency_ms: int = 0,
        retry_count: int = 0,
    ) -> ToolCallRecord:
        """Append a tool call record and return it."""
        self.step += 1
        record = ToolCallRecord(
            seq=self.step,
            tool_name=tool_name,
            arguments=arguments,
            response=response,
            status=status,
            latency_ms=latency_ms,
            retry_count=retry_count,
            trace_id=self.trace_id,
        )
        self.tool_calls.append(record)
        logger.info(
            "tool_call",
            trace_id=self.trace_id,
            seq=self.step,
            tool=tool_name,
            status=status,
            latency_ms=latency_ms,
        )
        return record

    @property
    def elapsed_seconds(self) -> float:
        return round(time.time() - self.start_time, 3)
