"""search_customer tool — look up a customer by ID or query string.

Uses an in-memory mock database for the MVP.
"""

from __future__ import annotations

from typing import Any

from agent.tool_executor import register_tool
from tools.base import tool_schema

# ── Mock data ────────────────────────────────────────────────────────────────

MOCK_CUSTOMERS: dict[int, dict[str, Any]] = {
    1827: {
        "id": 1827,
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "plan": "premium",
        "status": "active",
        "balance_due": 149.99,
    },
    8127: {
        "id": 8127,
        "name": "Bob Martinez",
        "email": "bob@example.com",
        "plan": "basic",
        "status": "active",
        "balance_due": 0.0,
    },
    3042: {
        "id": 3042,
        "name": "Carol Smith",
        "email": "carol@example.com",
        "plan": "enterprise",
        "status": "suspended",
        "balance_due": 2450.00,
    },
}


@register_tool("search_customer")
async def search_customer(customer_id: int | None = None, query: str | None = None) -> dict[str, Any]:
    """Search for a customer by ID or name substring."""
    if customer_id is not None:
        customer = MOCK_CUSTOMERS.get(customer_id)
        if customer:
            return {"success": True, "data": customer}
        return {"success": False, "error": f"Customer {customer_id} not found"}

    if query:
        matches = [
            c for c in MOCK_CUSTOMERS.values()
            if query.lower() in c["name"].lower() or query.lower() in c["email"].lower()
        ]
        return {"success": True, "data": matches}

    return {"success": False, "error": "Provide customer_id or query"}


search_customer.__tool_schema__ = tool_schema(
    name="search_customer",
    description="Search for a customer by their numeric ID or a name/email substring.",
    parameters={
        "customer_id": {"type": "integer", "description": "Exact customer ID"},
        "query": {"type": "string", "description": "Name or email substring to search"},
    },
)
