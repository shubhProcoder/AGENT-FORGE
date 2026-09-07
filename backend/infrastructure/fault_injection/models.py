"""Fault Injection Models and Configuration."""

from __future__ import annotations

from enum import Enum
import uuid
from pydantic import BaseModel, Field


class FaultType(str, Enum):
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    SERVER_ERROR = "SERVER_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    CORRUPTED_RESPONSE = "CORRUPTED_RESPONSE"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"


class FaultInjectionError(Exception):
    """Raised when a fault is intentionally injected into tool execution."""

    def __init__(self, fault_type: FaultType, message: str, details: dict | None = None):
        super().__init__(message)
        self.fault_type = fault_type
        self.message = message
        self.details = details or {}


class FaultRule(BaseModel):
    """Rule defining when and how to inject a fault into tool execution."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    target_tool: str = "*"  # Tool name or "*" for any tool
    fault_type: FaultType = FaultType.SERVER_ERROR
    trigger_on_call: int | None = None  # 1-indexed: fire only on Nth call (e.g. 1st attempt fails, 2nd succeeds)
    probability: float = 1.0  # 1.0 = deterministic fault
    delay_seconds: float = 0.0  # Latency simulation
    max_triggers: int | None = 1  # Total times this rule can fire; None for unlimited
    triggers_count: int = 0
    error_message: str | None = None
    http_status_code: int | None = None

    def should_trigger(self, tool_name: str, call_count: int) -> bool:
        """Evaluate whether this rule should trigger for the given tool invocation."""
        if self.target_tool != "*" and self.target_tool != tool_name:
            return False

        if self.max_triggers is not None and self.triggers_count >= self.max_triggers:
            return False

        if self.trigger_on_call is not None and call_count != self.trigger_on_call:
            return False

        if self.probability < 1.0:
            import random
            if random.random() > self.probability:
                return False

        return True

    def record_trigger(self) -> None:
        """Increment trigger counter."""
        self.triggers_count += 1
