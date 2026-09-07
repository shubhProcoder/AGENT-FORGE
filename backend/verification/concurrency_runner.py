"""Concurrency Runner and Race Condition Verifier.

Executes concurrent stress tests against the Execution Plane tools
and verifies system invariants under high contention.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ConcurrencyRunResult:
    total_requests: int
    successful_calls: int
    conflicts_or_errors: int
    duration_seconds: float
    responses: list[Any] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)


@dataclass
class RaceConditionVerificationResult:
    passed: bool
    race_detected: bool
    expected_mutations: int
    actual_mutations: int
    details: str
    metrics: dict[str, float] = field(default_factory=dict)


class ConcurrencyRunner:
    """Executes parallel tasks against async tools or agents to induce race conditions."""

    async def run(
        self,
        func: Callable[..., Any],
        num_requests: int = 20,
        *args: Any,
        **kwargs: Any,
    ) -> ConcurrencyRunResult:
        """Fire N concurrent calls of func(*args, **kwargs)."""
        logger.info(f"[ConcurrencyRunner] Spawning {num_requests} concurrent requests...")
        start_time = time.time()

        tasks = [func(*args, **kwargs) for _ in range(num_requests)]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        duration = time.time() - start_time
        successful = 0
        errors = 0
        responses = []
        exceptions = []

        for res in raw_results:
            if isinstance(res, Exception):
                errors += 1
                exceptions.append(str(res))
            elif isinstance(res, dict) and res.get("is_error"):
                errors += 1
                responses.append(res)
            else:
                successful += 1
                responses.append(res)

        logger.info(
            f"[ConcurrencyRunner] Completed {num_requests} requests in {duration:.3f}s: "
            f"{successful} successful, {errors} errors/conflicts."
        )

        return ConcurrencyRunResult(
            total_requests=num_requests,
            successful_calls=successful,
            conflicts_or_errors=errors,
            duration_seconds=duration,
            responses=responses,
            exceptions=exceptions,
        )


class RaceConditionVerifier:
    """Verifies that concurrent operations on shared resources preserved state invariants."""

    def verify(
        self,
        resource_type: str,
        state_before: dict[str, Any],
        state_after: dict[str, Any],
        resource_id: str | None = None,
        expected_max_mutations: int = 1,
    ) -> RaceConditionVerificationResult:
        actual_mutations = 0
        details = ""

        if resource_type == "orders":
            old_orders = state_before.get("orders", {})
            new_orders = state_after.get("orders", {})
            added_orders = [o for oid, o in new_orders.items() if oid not in old_orders]
            actual_mutations = len(added_orders)

            if actual_mutations > expected_max_mutations:
                details = (
                    f"Race condition detected: {actual_mutations} orders created, "
                    f"expected at most {expected_max_mutations}."
                )

        elif resource_type == "refund":
            if not resource_id:
                raise ValueError("resource_id (order_id) is required to verify refund concurrency")
            order = state_after.get("orders", {}).get(resource_id, {})
            actual_mutations = order.get("refund_count", 0)

            if actual_mutations > expected_max_mutations:
                details = (
                    f"Race condition detected: order '{resource_id}' was refunded "
                    f"{actual_mutations} times, expected at most {expected_max_mutations}."
                )

        elif resource_type == "tickets":
            old_tickets = state_before.get("tickets", {})
            new_tickets = state_after.get("tickets", {})
            added_tickets = [t for tid, t in new_tickets.items() if tid not in old_tickets]
            actual_mutations = len(added_tickets)

            if actual_mutations > expected_max_mutations:
                details = (
                    f"Race condition detected: {actual_mutations} tickets created, "
                    f"expected at most {expected_max_mutations}."
                )

        passed = actual_mutations <= expected_max_mutations
        race_detected = not passed
        score = 1.0 if passed else 0.0

        if passed and not details:
            details = (
                f"Concurrency invariant preserved: {actual_mutations} mutation(s) occurred "
                f"(<= {expected_max_mutations} expected)."
            )

        return RaceConditionVerificationResult(
            passed=passed,
            race_detected=race_detected,
            expected_mutations=expected_max_mutations,
            actual_mutations=actual_mutations,
            details=details,
            metrics={"concurrency_safety": score},
        )
