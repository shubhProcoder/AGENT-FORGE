"""get_order tool — retrieve an order by ID."""

from __future__ import annotations

from typing import Any

from agent.tool_executor import register_tool
from tools.base import tool_schema
from tools.order.create_order import _orders


@register_tool("get_order")
async def get_order(order_id: str) -> dict[str, Any]:
    """Retrieve an order by its ID."""
    order = _orders.get(order_id)
    if order is None:
        return {"success": False, "error": f"Order {order_id} not found"}
    return {"success": True, "data": order}


get_order.__tool_schema__ = tool_schema(
    name="get_order",
    description="Retrieve an order by its order ID.",
    parameters={
        "order_id": {"type": "string", "description": "The order ID (e.g. ORD-A1B2C3D4)"},
    },
    required=["order_id"],
)
