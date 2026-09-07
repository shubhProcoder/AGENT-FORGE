"""Idempotency Verifier.

Verifies at-most-once semantics by inspecting tool call history 
and resulting system state under normal execution and fault injection retries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IdempotencyVerificationResult:
    passed: bool
    total_calls: int
    idempotent_keys_checked: list[str]
    mutations_count: int
    violations: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class IdempotencyVerifier:
    """Verifies that duplicate tool calls with the same idempotency key did not produce duplicate mutations."""

    def verify(
        self,
        call_trace: list[dict[str, Any]],
        state_before: dict[str, Any],
        state_after: dict[str, Any],
    ) -> IdempotencyVerificationResult:
        violations: list[str] = []
        idempotent_calls: dict[str, list[dict[str, Any]]] = {}

        # Group calls by idempotency_key
        for call in call_trace:
            args = call.get("arguments", {})
            key = args.get("idempotency_key")
            if key:
                if key not in idempotent_calls:
                    idempotent_calls[key] = []
                idempotent_calls[key].append(call)

        # For each key with multiple invocations, verify system state
        mutations_count = 0

        for key, calls in idempotent_calls.items():
            if len(calls) > 1:
                logger.info(
                    f"[IdempotencyVerifier] Key '{key}' was invoked {len(calls)} times. Verifying state invariants..."
                )
                tool_name = calls[0].get("tool", "")

                if tool_name == "create_order":
                    # Check orders created in state_after vs state_before
                    old_orders = state_before.get("orders", {})
                    new_orders = state_after.get("orders", {})
                    added_orders = [
                        o for oid, o in new_orders.items() if oid not in old_orders
                    ]
                    mutations_count = len(added_orders)
                    if len(added_orders) > 1:
                        violations.append(
                            f"Idempotency violation for key '{key}': {len(added_orders)} orders created instead of 1."
                        )

                elif tool_name == "refund_order":
                    # Check refund count on the target order
                    order_id = calls[0].get("arguments", {}).get("order_id")
                    order = state_after.get("orders", {}).get(order_id, {})
                    refund_count = order.get("refund_count", 0)
                    mutations_count = refund_count
                    if refund_count > 1:
                        violations.append(
                            f"Idempotency violation for key '{key}': order '{order_id}' was refunded {refund_count} times."
                        )

                elif tool_name == "create_ticket":
                    old_tickets = state_before.get("tickets", {})
                    new_tickets = state_after.get("tickets", {})
                    added_tickets = [
                        t for tid, t in new_tickets.items() if tid not in old_tickets
                    ]
                    mutations_count = len(added_tickets)
                    if len(added_tickets) > 1:
                        violations.append(
                            f"Idempotency violation for key '{key}': {len(added_tickets)} tickets created instead of 1."
                        )

        passed = len(violations) == 0
        score = 1.0 if passed else 0.0

        return IdempotencyVerificationResult(
            passed=passed,
            total_calls=len(call_trace),
            idempotent_keys_checked=list(idempotent_calls.keys()),
            mutations_count=mutations_count,
            violations=violations,
            metrics={"idempotency_correctness": score},
        )
