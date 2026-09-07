"""Concurrency experiment — demonstrates race-condition detection.

Spawns N concurrent calls to create_order and verifies that
idempotency / locking produces exactly 1 order.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from tools.order.create_order import create_order, get_all_orders, reset_order_store


@dataclass
class ConcurrencyResult:
    """Outcome of a concurrency stress test."""

    concurrent_requests: int
    orders_created: int
    expected_orders: int
    passed: bool
    details: str = ""


async def run_concurrency_experiment(
    num_requests: int = 50,
    customer_id: int = 8127,
    idempotency_key: str = "stress-test-key",
) -> ConcurrencyResult:
    """Fire N concurrent create_order calls with the same idempotency key.

    Expected: exactly 1 order is created.
    Failure: more than 1 order indicates a race condition.
    """
    reset_order_store()

    items = [{"name": "Widget", "amount": 9.99}]

    # Fire all requests concurrently
    tasks = [
        create_order(
            customer_id=customer_id,
            items=items,
            idempotency_key=idempotency_key,
        )
        for _ in range(num_requests)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count how many unique orders were actually created
    orders = get_all_orders()
    order_count = len(orders)

    passed = order_count == 1
    details = "" if passed else f"Race condition: {order_count} orders created instead of 1"

    return ConcurrencyResult(
        concurrent_requests=num_requests,
        orders_created=order_count,
        expected_orders=1,
        passed=passed,
        details=details,
    )
