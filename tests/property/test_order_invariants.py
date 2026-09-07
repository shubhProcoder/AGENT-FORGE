"""Property-based tests using Hypothesis.

Generates random order data and verifies state invariants hold.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from tools.order.create_order import create_order, get_all_orders, reset_order_store


# ── Strategies ───────────────────────────────────────────────────────────────

item_strategy = st.fixed_dictionaries(
    {
        "name": st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L",))),
        "amount": st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
    }
)

items_strategy = st.lists(item_strategy, min_size=1, max_size=10)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestOrderInvariants:
    """Property-based tests: no matter what items we generate,
    the order total must always equal the sum of item amounts."""

    @given(items=items_strategy)
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_total_equals_sum_of_items(self, items):
        reset_order_store()
        result = await create_order(customer_id=1, items=items)
        assert result["success"] is True

        order = result["data"]
        expected_total = sum(item["amount"] for item in items)
        assert order["total"] == pytest.approx(expected_total, abs=1e-6), (
            f"Invariant violated: total={order['total']}, sum={expected_total}"
        )

    @given(items=items_strategy)
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_order_fields_present(self, items):
        reset_order_store()
        result = await create_order(customer_id=42, items=items)
        order = result["data"]
        assert "order_id" in order
        assert "customer_id" in order
        assert "items" in order
        assert "total" in order
        assert "status" in order
        assert order["refund_count"] == 0
