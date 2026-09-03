"""Execution Plane Environment Provisioning.

Provides functions to seed a sandboxed environment with mock data
(e.g., customers, orders, tickets) and snapshot its state for diffing.
"""

from typing import Any
import copy
import logging

logger = logging.getLogger(__name__)


class EnvironmentSandbox:
    """A simulated Execution Plane database for the Agent to mutate."""
    
    def __init__(self, idempotency_store: Any | None = None):
        # In a real system, this might be a fresh Postgres schema or sqlite DB
        self.state: dict[str, dict[str, Any]] = {
            "customers": {},
            "orders": {},
            "tickets": {},
            "refunds": {},
        }
        from backend.infrastructure.idempotency.store import IdempotencyStore, default_idempotency_store
        self.idempotency_store: IdempotencyStore = idempotency_store or default_idempotency_store

    def provision(self) -> None:
        """Seed initial data into the sandbox."""
        logger.info("Provisioning EnvironmentSandbox with fixtures...")
        self.state["customers"]["cust_123"] = {
            "id": "cust_123",
            "name": "Alice Smith",
            "email": "alice@example.com",
            "status": "active"
        }
        self.state["orders"]["ord_456"] = {
            "id": "ord_456",
            "customer_id": "cust_123",
            "total": 100.00,
            "status": "paid",
            "refund_count": 0,
        }

    def snapshot(self) -> dict[str, Any]:
        """Freeze the current state of the sandbox for diffing."""
        logger.info("Snapshotting EnvironmentSandbox state.")
        return copy.deepcopy(self.state)

    async def create_order(
        self,
        customer_id: str,
        items: list[dict[str, Any]],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Create an order with idempotency check."""
        if idempotency_key:
            completed, cached_res = await self.idempotency_store.check_or_start(idempotency_key)
            if completed:
                logger.info(f"[Sandbox] Idempotent replay for order with key {idempotency_key}")
                return cached_res

        import uuid
        order_id = f"ord_{uuid.uuid4().hex[:8]}"
        total = sum(float(item.get("amount", item.get("price", 0.0))) for item in items)
        new_order = {
            "id": order_id,
            "customer_id": customer_id,
            "items": items,
            "total": total,
            "status": "pending",
            "refund_count": 0,
        }
        self.state["orders"][order_id] = new_order
        res = {"status": "success", "order_id": order_id, "total": total}

        if idempotency_key:
            await self.idempotency_store.complete(idempotency_key, res, resource_id=order_id)
        return res

    async def refund_order(
        self,
        order_id: str,
        reason: str = "customer_request",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Refund an order with idempotency check."""
        if idempotency_key:
            completed, cached_res = await self.idempotency_store.check_or_start(idempotency_key)
            if completed:
                logger.info(f"[Sandbox] Idempotent replay for refund on order {order_id}")
                return cached_res

        order = self.state["orders"].get(order_id)
        if not order:
            res = {"status": "error", "message": f"Order {order_id} not found"}
            if idempotency_key:
                await self.idempotency_store.complete(idempotency_key, res)
            return res

        # Mutate order state
        order["refund_count"] = order.get("refund_count", 0) + 1
        order["status"] = "refunded"

        import uuid
        refund_id = f"ref_{uuid.uuid4().hex[:8]}"
        self.state["refunds"][refund_id] = {
            "id": refund_id,
            "order_id": order_id,
            "amount": order.get("total", 0.0),
            "reason": reason,
        }

        res = {
            "status": "success",
            "refund_id": refund_id,
            "order_id": order_id,
            "refund_count": order["refund_count"],
        }

        if idempotency_key:
            await self.idempotency_store.complete(idempotency_key, res, resource_id=refund_id)
        return res

    async def create_ticket(
        self,
        customer_id: str,
        subject: str,
        category: str = "general",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Create a support ticket with idempotency check."""
        if idempotency_key:
            completed, cached_res = await self.idempotency_store.check_or_start(idempotency_key)
            if completed:
                logger.info(f"[Sandbox] Idempotent replay for ticket with key {idempotency_key}")
                return cached_res

        import uuid
        ticket_id = f"tkt_{uuid.uuid4().hex[:8]}"
        new_ticket = {
            "id": ticket_id,
            "customer_id": customer_id,
            "subject": subject,
            "category": category,
            "status": "open",
        }
        self.state["tickets"][ticket_id] = new_ticket
        res = {"status": "success", "ticket_id": ticket_id}

        if idempotency_key:
            await self.idempotency_store.complete(idempotency_key, res, resource_id=ticket_id)
        return res

