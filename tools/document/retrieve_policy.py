"""retrieve_policy tool — fetch a specific company policy by name."""

from __future__ import annotations

from typing import Any

from agent.tool_executor import register_tool
from tools.base import tool_schema

# ── Policy store ─────────────────────────────────────────────────────────────

POLICIES: dict[str, str] = {
    "refund": (
        "Refunds are available within 30 days of purchase. "
        "Premium customers receive priority processing. "
        "Refunds over $500 require manager approval. "
        "Each order can only be refunded once."
    ),
    "billing_dispute": (
        "Customers disputing a charge should first contact support. "
        "A support ticket with category 'billing' must be created. "
        "If the dispute is valid, issue a refund within 5 business days."
    ),
    "suspension": (
        "Accounts are suspended after 90 days of non-payment. "
        "Suspended accounts cannot place new orders. "
        "To reactivate, the customer must clear the outstanding balance."
    ),
    "data_deletion": (
        "Customer data deletion requests must be processed within 72 hours. "
        "This is an internal-only operation — never expose to external agents."
    ),
}


@register_tool("retrieve_policy")
async def retrieve_policy(policy_name: str) -> dict[str, Any]:
    """Retrieve a company policy by its name."""
    policy = POLICIES.get(policy_name.lower())
    if policy is None:
        available = list(POLICIES.keys())
        return {"success": False, "error": f"Unknown policy '{policy_name}'", "available": available}
    return {"success": True, "data": {"policy_name": policy_name, "content": policy}}


retrieve_policy.__tool_schema__ = tool_schema(
    name="retrieve_policy",
    description="Retrieve a company policy document by name (e.g. 'refund', 'billing_dispute', 'suspension').",
    parameters={
        "policy_name": {"type": "string", "description": "Name of the policy to retrieve"},
    },
    required=["policy_name"],
)
