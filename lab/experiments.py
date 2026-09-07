"""Reliability experiments — aggregated runner for all lab scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lab.concurrency import ConcurrencyResult, run_concurrency_experiment
from lab.idempotency import IdempotencyResult, run_idempotency_experiment


@dataclass
class ExperimentSuite:
    """Aggregated results from all reliability experiments."""

    concurrency: ConcurrencyResult | None = None
    idempotency: IdempotencyResult | None = None
    all_passed: bool = False
    summary: list[dict[str, Any]] = field(default_factory=list)


async def run_all_experiments() -> ExperimentSuite:
    """Run every reliability experiment and return aggregated results."""
    suite = ExperimentSuite()

    # ── Concurrency ──────────────────────────────────────────────────────
    suite.concurrency = await run_concurrency_experiment(num_requests=50)
    suite.summary.append(
        {
            "experiment": "concurrency_storm",
            "passed": suite.concurrency.passed,
            "detail": suite.concurrency.details or "50 concurrent requests → 1 order",
        }
    )

    # ── Idempotency ──────────────────────────────────────────────────────
    suite.idempotency = await run_idempotency_experiment(num_retries=5)
    suite.summary.append(
        {
            "experiment": "idempotency",
            "passed": suite.idempotency.passed,
            "detail": suite.idempotency.details or "5 retries → 1 ticket",
        }
    )

    suite.all_passed = all(item["passed"] for item in suite.summary)
    return suite
