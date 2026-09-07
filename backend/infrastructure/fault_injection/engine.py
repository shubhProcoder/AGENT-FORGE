"""Fault Injection Engine for intercepting tool execution."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from typing import Any, Callable
import uuid

from backend.infrastructure.fault_injection.models import (
    FaultInjectionError,
    FaultRule,
    FaultType,
)

logger = logging.getLogger(__name__)


class FaultInjectionEngine:
    """Interception engine that deterministically injects simulated failures into tool calls."""

    def __init__(self) -> None:
        self.rules: list[FaultRule] = []
        self.tool_call_counts: dict[str, int] = defaultdict(int)
        self.injected_fault_log: list[dict[str, Any]] = []

    def register_rule(self, rule: FaultRule) -> None:
        """Register a new fault rule."""
        self.rules.append(rule)
        logger.info(
            f"[FaultEngine] Registered fault rule for tool='{rule.target_tool}', type={rule.fault_type.value}"
        )

    def remove_rule(self, rule_id: uuid.UUID) -> bool:
        """Remove a rule by ID."""
        orig_len = len(self.rules)
        self.rules = [r for r in self.rules if r.id != rule_id]
        return len(self.rules) < orig_len

    def clear_rules(self) -> None:
        """Clear all active rules."""
        self.rules.clear()

    def reset(self) -> None:
        """Reset counters and logs, keeping rules intact."""
        self.tool_call_counts.clear()
        self.injected_fault_log.clear()
        for rule in self.rules:
            rule.triggers_count = 0

    def get_call_count(self, tool_name: str) -> int:
        """Get the total calls so far for this tool."""
        return self.tool_call_counts[tool_name]

    async def intercept(self, tool_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Intercept tool execution, evaluating rules and injecting faults if matched."""
        self.tool_call_counts[tool_name] += 1
        current_call = self.tool_call_counts[tool_name]

        # Check rules
        for rule in self.rules:
            if rule.should_trigger(tool_name, current_call):
                rule.record_trigger()
                self.injected_fault_log.append({
                    "tool_name": tool_name,
                    "call_number": current_call,
                    "fault_type": rule.fault_type.value,
                    "rule_id": str(rule.id),
                })
                logger.warning(
                    f"[FaultEngine] INJECTING FAULT: tool='{tool_name}' on call #{current_call} "
                    f"type={rule.fault_type.value}"
                )

                if rule.delay_seconds > 0:
                    await asyncio.sleep(rule.delay_seconds)

                if rule.fault_type == FaultType.TIMEOUT:
                    raise FaultInjectionError(
                        fault_type=FaultType.TIMEOUT,
                        message=rule.error_message or f"Operation timed out executing '{tool_name}'",
                        details={"tool": tool_name, "call_number": current_call, "status_code": 504},
                    )

                if rule.fault_type == FaultType.NETWORK_ERROR:
                    raise FaultInjectionError(
                        fault_type=FaultType.NETWORK_ERROR,
                        message=rule.error_message or f"Transient network failure connecting to '{tool_name}'",
                        details={"tool": tool_name, "call_number": current_call},
                    )

                if rule.fault_type == FaultType.RATE_LIMIT:
                    raise FaultInjectionError(
                        fault_type=FaultType.RATE_LIMIT,
                        message=rule.error_message or "Rate limit exceeded. Too many requests.",
                        details={"tool": tool_name, "call_number": current_call, "retry_after": 1.0, "status_code": 429},
                    )

                if rule.fault_type == FaultType.SERVER_ERROR:
                    raise FaultInjectionError(
                        fault_type=FaultType.SERVER_ERROR,
                        message=rule.error_message or f"Internal server error (HTTP 500) executing '{tool_name}'",
                        details={"tool": tool_name, "call_number": current_call, "status_code": 500},
                    )

                if rule.fault_type == FaultType.CORRUPTED_RESPONSE:
                    return {
                        "status": "error",
                        "corrupted": True,
                        "raw_bytes": b"MALFORMED_HEADER\x00\xff".decode("latin-1"),
                    }

                if rule.fault_type == FaultType.PARTIAL_FAILURE:
                    return {
                        "status": "partial_failure",
                        "error": rule.error_message or "Partial write failed; operation unverified",
                    }

        # No fault triggered: execute tool normally
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)
