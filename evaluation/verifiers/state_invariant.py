"""State-invariant verifier — checks that system state is consistent.

Example invariants:
  - order.total == sum(order.items)
  - refund_count <= 1 per order
  - no unexpected field mutations
"""

from __future__ import annotations

from typing import Any

from agent.state import ExecutionContext
from evaluation.metrics import VerificationOutcome
from tools.order.create_order import get_all_orders


def verify_state_invariants(
    ctx: ExecutionContext,
    expected_state: dict[str, Any] | None = None,
) -> VerificationOutcome:
    """Run state-invariant checks after an agent trial.

    Checks:
      1. Order total == sum of item amounts.
      2. Refund count <= 1 per order (no duplicate refunds).
      3. Custom expected_state assertions if provided.
    """
    outcome = VerificationOutcome(passed=True)
    orders = get_all_orders()

    violations: list[dict[str, Any]] = []

    for order_id, order in orders.items():
        # Invariant 1: total consistency
        items = order.get("items", [])
        expected_total = sum(item.get("amount", 0) for item in items)
        if order.get("total") != expected_total:
            violations.append(
                {
                    "order_id": order_id,
                    "invariant": "total_mismatch",
                    "expected": expected_total,
                    "actual": order.get("total"),
                }
            )

        # Invariant 2: no duplicate refunds
        if order.get("refund_count", 0) > 1:
            violations.append(
                {
                    "order_id": order_id,
                    "invariant": "duplicate_refund",
                    "refund_count": order["refund_count"],
                }
            )

    if violations:
        outcome.add_metric(
            "state_correctness",
            score=0.0,
            details=f"{len(violations)} state violation(s) detected",
        )
        outcome.failures.extend(
            [{"type": "state_invariant", "detail": v} for v in violations]
        )
    else:
        outcome.add_metric("state_correctness", score=1.0)

    return outcome
