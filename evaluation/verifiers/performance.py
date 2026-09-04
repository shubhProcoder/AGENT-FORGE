"""Performance verifier — scores latency and cost against budgets."""

from __future__ import annotations

from agent.state import ExecutionContext
from evaluation.metrics import VerificationOutcome


def verify_performance(
    ctx: ExecutionContext,
    max_latency_seconds: float = 30.0,
    max_cost_usd: float = 0.10,
) -> VerificationOutcome:
    """Score the trial on latency and cost efficiency.

    Both metrics are scored on a linear scale from 1.0 (within budget)
    down to 0.0 (2× over budget or worse).
    """
    outcome = VerificationOutcome(passed=True)

    # ── Latency score ────────────────────────────────────────────────────
    elapsed = ctx.elapsed_seconds
    if elapsed <= max_latency_seconds:
        latency_score = 1.0
    elif elapsed <= max_latency_seconds * 2:
        latency_score = 1.0 - ((elapsed - max_latency_seconds) / max_latency_seconds)
    else:
        latency_score = 0.0
    outcome.add_metric("latency", round(latency_score, 4))

    # ── Cost score ───────────────────────────────────────────────────────
    cost = ctx.estimated_cost_usd
    if cost <= max_cost_usd:
        cost_score = 1.0
    elif cost <= max_cost_usd * 2:
        cost_score = 1.0 - ((cost - max_cost_usd) / max_cost_usd)
    else:
        cost_score = 0.0
    outcome.add_metric("cost", round(cost_score, 4))

    return outcome
