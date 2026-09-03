"""get_customer_tickets tool — retrieve tickets for a customer."""

from __future__ import annotations

from typing import Any

from agent.tool_executor import register_tool
from tools.base import tool_schema

# ── Mock ticket data ─────────────────────────────────────────────────────────

MOCK_TICKETS: dict[int, list[dict[str, Any]]] = {
    1827: [
        {"ticket_id": "T-4001", "category": "billing", "status": "open", "summary": "Overcharged on last invoice"},
        {"ticket_id": "T-3980", "category": "technical", "status": "closed", "summary": "Login issue resolved"},
    ],
    8127: [
        {"ticket_id": "T-4010", "category": "billing", "status": "open", "summary": "Unexpected charge $29.99"},
    ],
    3042: [],
}


@register_tool("get_customer_tickets")
async def get_customer_tickets(customer_id: int) -> dict[str, Any]:
    """Get all support tickets for a customer."""
    tickets = MOCK_TICKETS.get(customer_id)
    if tickets is None:
        return {"success": False, "error": f"No ticket data for customer {customer_id}"}
    return {"success": True, "data": {"customer_id": customer_id, "tickets": tickets}}


get_customer_tickets.__tool_schema__ = tool_schema(
    name="get_customer_tickets",
    description="Retrieve all support tickets (open and closed) for a given customer ID.",
    parameters={
        "customer_id": {"type": "integer", "description": "The customer's numeric ID"},
    },
    required=["customer_id"],
)
