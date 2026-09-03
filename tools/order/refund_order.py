"""refund_order tool — issue a refund on an order.

Tracks refund_count on the order to catch duplicate-refund bugs.
"""

from __future__ import annotations

from typing import Any

from agent.tool_executor import register_tool
from tools.base import tool_schema
from tools.order.create_order import _orders


@register_tool("refund_order")
async def refund_order(order_id: str, reason: str = "") -> dict[str, Any]:
    """Refund an order. Increments refund_count on the order record.

    A well-behaved agent should only refund once; the Reliability Lab
    tests whether duplicate refunds are prevented.
    """
    order = _orders.get(order_id)
    if order is None:
        return {"success": False, "error": f"Order {order_id} not found"}

    order["refund_count"] += 1
    order["status"] = "refunded"

    return {
        "success": True,
        "data": {
            "order_id": order_id,
            "refund_count": order["refund_count"],
            "reason": reason,
        },
    }


refund_order.__tool_schema__ = tool_schema(
    name="refund_order",
    description="Issue a refund for an existing order. Only call once per order.",
    parameters={
        "order_id": {"type": "string", "description": "The order ID to refund"},
        "reason": {"type": "string", "description": "Reason for the refund"},
    },
    required=["order_id"],
)
