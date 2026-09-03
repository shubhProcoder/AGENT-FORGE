"""search_documents tool — keyword search over a mock knowledge base."""

from __future__ import annotations

from typing import Any

from agent.tool_executor import register_tool
from tools.base import tool_schema

# ── Mock knowledge base ──────────────────────────────────────────────────────

MOCK_DOCUMENTS = [
    {
        "doc_id": "DOC-001",
        "title": "Refund Policy",
        "category": "policy",
        "content": (
            "Refunds are available within 30 days of purchase. "
            "Premium customers receive priority processing. "
            "Refunds over $500 require manager approval. "
            "Each order can only be refunded once."
        ),
    },
    {
        "doc_id": "DOC-002",
        "title": "Billing Dispute Process",
        "category": "policy",
        "content": (
            "Customers disputing a charge should first contact support. "
            "A support ticket with category 'billing' must be created. "
            "If the dispute is valid, issue a refund within 5 business days."
        ),
    },
    {
        "doc_id": "DOC-003",
        "title": "Account Suspension Guidelines",
        "category": "policy",
        "content": (
            "Accounts are suspended after 90 days of non-payment. "
            "Suspended accounts cannot place new orders. "
            "To reactivate, the customer must clear the outstanding balance."
        ),
    },
    {
        "doc_id": "DOC-004",
        "title": "API Rate Limiting",
        "category": "engineering",
        "content": (
            "All API endpoints are rate-limited to 100 requests per minute "
            "per API key. Exceeding this limit returns HTTP 429."
        ),
    },
    {
        "doc_id": "DOC-005",
        "title": "Data Deletion Runbook",
        "category": "engineering",
        "content": (
            "Customer data deletion requests must be processed within 72 hours. "
            "Use the delete_customer_data() internal function. "
            "Never expose this function to external agents."
        ),
    },
]


@register_tool("search_documents")
async def search_documents(query: str, category: str | None = None) -> dict[str, Any]:
    """Search documents by keyword match. Optionally filter by category."""
    query_lower = query.lower()
    results = [
        doc for doc in MOCK_DOCUMENTS
        if query_lower in doc["title"].lower() or query_lower in doc["content"].lower()
    ]
    if category:
        results = [doc for doc in results if doc["category"] == category]

    return {
        "success": True,
        "data": {
            "query": query,
            "results": results,
            "count": len(results),
        },
    }


search_documents.__tool_schema__ = tool_schema(
    name="search_documents",
    description="Search the knowledge base by keyword. Optionally filter by category (policy, engineering).",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "category": {"type": "string", "description": "Optional category filter"},
    },
    required=["query"],
)
