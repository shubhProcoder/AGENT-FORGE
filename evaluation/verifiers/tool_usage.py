"""Tool-usage verifier — checks that the agent used the correct tools.

Verifies:
  - Required tools were called.
  - Forbidden tools were NOT called.
  - Tool call sequence (optional).
"""

from __future__ import annotations

from typing import Any

from agent.state import ExecutionContext
from evaluation.metrics import VerificationOutcome


def verify_tool_usage(
    ctx: ExecutionContext,
    required_tools: list[str] | None = None,
    forbidden_tools: list[str] | None = None,
    expected_sequence: list[str] | None = None,
) -> VerificationOutcome:
    """Verify tool usage patterns in the execution context.

    Args:
        ctx: The execution context from the agent trial.
        required_tools: Tools that MUST have been called.
        forbidden_tools: Tools that MUST NOT have been called.
        expected_sequence: Optional exact ordering of tool calls.
    """
    outcome = VerificationOutcome(passed=True)
    called = [tc.tool_name for tc in ctx.tool_calls]
    called_set = set(called)

    score = 1.0
    issues: list[str] = []

    # Check required tools
    if required_tools:
        missing = set(required_tools) - called_set
        if missing:
            score -= 0.3 * len(missing)
            issues.append(f"Missing required tools: {missing}")

    # Check forbidden tools
    if forbidden_tools:
        used_forbidden = called_set & set(forbidden_tools)
        if used_forbidden:
            score = 0.0  # critical violation
            issues.append(f"Forbidden tools used: {used_forbidden}")

    # Check sequence (if provided)
    if expected_sequence:
        if called != expected_sequence:
            score -= 0.2
            issues.append(f"Expected sequence {expected_sequence}, got {called}")

    score = max(score, 0.0)

    if issues:
        outcome.add_metric(
            "tool_correctness",
            score=score,
            details="; ".join(issues),
        )
    else:
        outcome.add_metric("tool_correctness", score=1.0)

    return outcome
