"""Custom metrics — latency, cost, token counters.

Simple in-memory counters; can be extended to push to Prometheus or StatsD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrialMetrics:
    """Aggregated metrics for one trial."""

    trace_id: str = ""
    total_latency_ms: int = 0
    tool_call_count: int = 0
    retry_count: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "total_latency_ms": self.total_latency_ms,
            "tool_call_count": self.tool_call_count,
            "retry_count": self.retry_count,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "errors": self.errors,
        }


def collect_trial_metrics(ctx: Any) -> TrialMetrics:
    """Build TrialMetrics from an ExecutionContext."""
    from agent.state import ExecutionContext

    if not isinstance(ctx, ExecutionContext):
        return TrialMetrics()

    return TrialMetrics(
        trace_id=ctx.trace_id,
        total_latency_ms=int(ctx.elapsed_seconds * 1000),
        tool_call_count=len(ctx.tool_calls),
        retry_count=sum(tc.retry_count for tc in ctx.tool_calls),
        total_tokens=ctx.total_tokens,
        prompt_tokens=ctx.prompt_tokens,
        completion_tokens=ctx.completion_tokens,
        estimated_cost_usd=ctx.estimated_cost_usd,
        errors=sum(1 for tc in ctx.tool_calls if tc.status == "error"),
    )
