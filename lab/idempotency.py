"""Idempotency experiment — verifies at-most-once semantics.

Sends the same request N times and checks that only one logical
operation was performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.customer.create_ticket import create_ticket, reset_ticket_store, _created_tickets


@dataclass
class IdempotencyResult:
    """Outcome of an idempotency test."""

    total_requests: int
    items_created: int
    expected_items: int
    passed: bool
    details: str = ""


async def run_idempotency_experiment(
    num_retries: int = 5,
    customer_id: int = 1827,
    idempotency_key: str = "idem-test-001",
) -> IdempotencyResult:
    """Send the same create_ticket request N times with the same idempotency key.

    Expected: exactly 1 ticket is created.
    """
    reset_ticket_store()

    for _ in range(num_retries):
        await create_ticket(
            customer_id=customer_id,
            category="billing",
            summary="Duplicate test ticket",
            idempotency_key=idempotency_key,
        )

    ticket_count = len(_created_tickets)
    passed = ticket_count == 1
    details = "" if passed else f"Idempotency violation: {ticket_count} tickets created instead of 1"

    return IdempotencyResult(
        total_requests=num_retries,
        items_created=ticket_count,
        expected_items=1,
        passed=passed,
        details=details,
    )
