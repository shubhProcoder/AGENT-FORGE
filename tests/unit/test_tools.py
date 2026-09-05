"""Unit tests for tool implementations."""

from __future__ import annotations

import pytest

from tools.customer.search_customer import search_customer
from tools.customer.get_customer_tickets import get_customer_tickets
from tools.customer.create_ticket import create_ticket, reset_ticket_store
from tools.order.create_order import create_order, reset_order_store
from tools.order.get_order import get_order
from tools.order.refund_order import refund_order
from tools.document.search_documents import search_documents
from tools.document.retrieve_policy import retrieve_policy


# ── Customer tools ───────────────────────────────────────────────────────────

class TestSearchCustomer:
    @pytest.mark.asyncio
    async def test_search_by_id_found(self):
        result = await search_customer(customer_id=1827)
        assert result["success"] is True
        assert result["data"]["name"] == "Alice Johnson"

    @pytest.mark.asyncio
    async def test_search_by_id_not_found(self):
        result = await search_customer(customer_id=99999)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_search_by_query(self):
        result = await search_customer(query="bob")
        assert result["success"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["name"] == "Bob Martinez"


class TestGetCustomerTickets:
    @pytest.mark.asyncio
    async def test_tickets_exist(self):
        result = await get_customer_tickets(customer_id=1827)
        assert result["success"] is True
        assert len(result["data"]["tickets"]) == 2

    @pytest.mark.asyncio
    async def test_no_tickets(self):
        result = await get_customer_tickets(customer_id=3042)
        assert result["success"] is True
        assert result["data"]["tickets"] == []


class TestCreateTicket:
    @pytest.mark.asyncio
    async def test_create_ticket_success(self):
        reset_ticket_store()
        result = await create_ticket(customer_id=1827, category="billing", summary="Test issue")
        assert result["success"] is True
        assert result["data"]["status"] == "open"

    @pytest.mark.asyncio
    async def test_idempotency(self):
        reset_ticket_store()
        r1 = await create_ticket(customer_id=1827, category="billing", summary="X", idempotency_key="k1")
        r2 = await create_ticket(customer_id=1827, category="billing", summary="X", idempotency_key="k1")
        assert r1["data"]["ticket_id"] == r2["data"]["ticket_id"]


# ── Order tools ──────────────────────────────────────────────────────────────

class TestCreateOrder:
    @pytest.mark.asyncio
    async def test_create_order(self):
        reset_order_store()
        result = await create_order(customer_id=8127, items=[{"name": "A", "amount": 10.0}])
        assert result["success"] is True
        assert result["data"]["total"] == 10.0

    @pytest.mark.asyncio
    async def test_idempotent_order(self):
        reset_order_store()
        r1 = await create_order(customer_id=8127, items=[{"name": "B", "amount": 5.0}], idempotency_key="ord1")
        r2 = await create_order(customer_id=8127, items=[{"name": "B", "amount": 5.0}], idempotency_key="ord1")
        assert r1["data"]["order_id"] == r2["data"]["order_id"]


class TestRefundOrder:
    @pytest.mark.asyncio
    async def test_refund_increments_count(self):
        reset_order_store()
        create_res = await create_order(customer_id=8127, items=[{"name": "C", "amount": 20.0}])
        order_id = create_res["data"]["order_id"]
        refund_res = await refund_order(order_id=order_id, reason="test")
        assert refund_res["data"]["refund_count"] == 1

    @pytest.mark.asyncio
    async def test_double_refund_detected(self):
        reset_order_store()
        create_res = await create_order(customer_id=8127, items=[{"name": "D", "amount": 30.0}])
        order_id = create_res["data"]["order_id"]
        await refund_order(order_id=order_id)
        await refund_order(order_id=order_id)
        get_res = await get_order(order_id=order_id)
        assert get_res["data"]["refund_count"] == 2  # Bug! Verifier should catch this.


# ── Document tools ───────────────────────────────────────────────────────────

class TestSearchDocuments:
    @pytest.mark.asyncio
    async def test_search_by_keyword(self):
        result = await search_documents(query="refund")
        assert result["success"] is True
        assert result["data"]["count"] >= 1

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self):
        result = await search_documents(query="rate", category="engineering")
        assert result["success"] is True
        assert all(doc["category"] == "engineering" for doc in result["data"]["results"])


class TestRetrievePolicy:
    @pytest.mark.asyncio
    async def test_known_policy(self):
        result = await retrieve_policy(policy_name="refund")
        assert result["success"] is True
        assert "30 days" in result["data"]["content"]

    @pytest.mark.asyncio
    async def test_unknown_policy(self):
        result = await retrieve_policy(policy_name="nonexistent")
        assert result["success"] is False
