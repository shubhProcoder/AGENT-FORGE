"""create_ticket tool — create a new support ticket for a customer.

Demonstrates idempotency: if a ticket with the same idempotency_key
already exists, the original ticket is returned instead of creating a duplicate.
"""

from __future__ import annotations

import uuid
from typing import Any

from agent.tool_executor import register_tool
from tools.base import tool_schema

# ── In-memory store ──────────────────────────────────────────────────────────

_created_tickets: list[dict[str, Any]] = []
_idempotency_keys: dict[str, dict[str, Any]] = {}


def reset_ticket_store() -> None:
    """Clear the store — used by tests to reset state between trials."""
    _created_tickets.clear()
    _idempotency_keys.clear()


@register_tool("create_ticket")
async def create_ticket(
    customer_id: int,
    category: str,
    summary: str,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Create a support ticket for the given customer.

    If an idempotency_key is provided and a ticket with that key
    already exists, the existing ticket is returned (no duplicate).
    """
    # Idempotency guard
    if idempotency_key and idempotency_key in _idempotency_keys:
        return {
            "success": True,
            "data": _idempotency_keys[idempotency_key],
            "note": "Duplicate request — returning existing ticket",
        }

    ticket = {
        "ticket_id": f"T-{uuid.uuid4().hex[:6].upper()}",
        "customer_id": customer_id,
        "category": category,
        "summary": summary,
        "status": "open",
    }
    _created_tickets.append(ticket)

    if idempotency_key:
        _idempotency_keys[idempotency_key] = ticket

    return {"success": True, "data": ticket}


create_ticket.__tool_schema__ = tool_schema(
    name="create_ticket",
    description="Create a new support ticket for a customer. Optionally pass an idempotency_key to prevent duplicate tickets.",
    parameters={
        "customer_id": {"type": "integer", "description": "Customer ID"},
        "category": {"type": "string", "description": "Ticket category (e.g. billing, technical)"},
        "summary": {"type": "string", "description": "Brief description of the issue"},
        "idempotency_key": {"type": "string", "description": "Optional unique key to prevent duplicates"},
    },
    required=["customer_id", "category", "summary"],
)
