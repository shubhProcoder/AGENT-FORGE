"""create_order tool — simulates order creation with concurrency & idempotency.

Intentionally uses an in-memory store so the Reliability Lab can
demonstrate race conditions and idempotency handling.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from agent.tool_executor import register_tool
from tools.base import tool_schema

# ── In-memory order store ────────────────────────────────────────────────────

_orders: dict[str, dict[str, Any]] = {}
_idempotency_keys: dict[str, str] = {}  # idempotency_key → order_id
_lock = asyncio.Lock()


def reset_order_store() -> None:
    """Clear the store (used between tests)."""
    _orders.clear()
    _idempotency_keys.clear()


def get_all_orders() -> dict[str, dict[str, Any]]:
    """Return the full order store — useful for state verification."""
    return dict(_orders)


@register_tool("create_order")
async def create_order(
    customer_id: int,
    items: list[dict[str, Any]],
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Create a new order for a customer.

    Supports idempotency keys to guarantee at-most-once creation.
    Uses an async lock to prevent concurrent duplicate writes.
    """
    async with _lock:
        # Idempotency check
        if idempotency_key and idempotency_key in _idempotency_keys:
            existing = _orders[_idempotency_keys[idempotency_key]]
            return {"success": True, "data": existing, "note": "Idempotent — existing order returned"}

        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        total = sum(item.get("amount", 0) for item in items)

        order = {
            "order_id": order_id,
            "customer_id": customer_id,
            "items": items,
            "total": total,
            "status": "created",
            "refund_count": 0,
        }
        _orders[order_id] = order

        if idempotency_key:
            _idempotency_keys[idempotency_key] = order_id

    return {"success": True, "data": order}


create_order.__tool_schema__ = tool_schema(
    name="create_order",
    description="Create a new order for a customer. Provide items with 'name' and 'amount' fields.",
    parameters={
        "customer_id": {"type": "integer", "description": "Customer ID"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "amount": {"type": "number"},
                },
            },
            "description": "List of order items",
        },
        "idempotency_key": {"type": "string", "description": "Optional unique key"},
    },
    required=["customer_id", "items"],
)
