"""Unit and Integration Tests for Phase 2: Reliability & Edge Cases.

Covers:
- TASK 9: Failure Injection Engine & Idempotency Verifier
- TASK 10: Concurrency Runner & Race Condition Verifier
- TASK 11: Hidden Tests, Dataset Versioning & Regression Detection
"""

from __future__ import annotations

import asyncio
import uuid
import pytest

from backend.domain.dataset import Dataset, Task
from backend.domain.result import VerificationResult
from backend.domain.trial import Trial
from backend.application.trial_manager import TrialManager
from backend.evaluation.versioning import (
    RegressionDetector,
    create_dataset_snapshot,
)
from backend.infrastructure.concurrency.lock import KeyedAsyncLock
from backend.infrastructure.database.fixtures import EnvironmentSandbox
from backend.infrastructure.fault_injection.engine import FaultInjectionEngine
from backend.infrastructure.fault_injection.models import (
    FaultInjectionError,
    FaultRule,
    FaultType,
)
from backend.infrastructure.idempotency.store import (
    IdempotencyConflictError,
    IdempotencyStatus,
    IdempotencyStore,
)
from backend.infrastructure.mcp.server import AgentMCPServer
from backend.verification.concurrency_runner import (
    ConcurrencyRunner,
    RaceConditionVerifier,
)
from backend.verification.idempotency_verifier import IdempotencyVerifier


# ============================================================================
# TASK 9: Failure Injection Engine & Idempotency Tests
# ============================================================================

class TestFaultInjectionEngine:
    @pytest.mark.asyncio
    async def test_trigger_on_first_call_only(self):
        """Verify fault triggers on attempt 1, allowing attempt 2 to succeed."""
        engine = FaultInjectionEngine()
        rule = FaultRule(
            target_tool="refund_order",
            fault_type=FaultType.TIMEOUT,
            trigger_on_call=1,
            max_triggers=1,
        )
        engine.register_rule(rule)

        async def dummy_refund(order_id: str):
            return {"status": "refunded", "order_id": order_id}

        # Attempt 1: Should raise FaultInjectionError (TIMEOUT)
        with pytest.raises(FaultInjectionError) as exc_info:
            await engine.intercept("refund_order", dummy_refund, order_id="ord_123")
        assert exc_info.value.fault_type == FaultType.TIMEOUT

        # Attempt 2: Should succeed
        res = await engine.intercept("refund_order", dummy_refund, order_id="ord_123")
        assert res["status"] == "refunded"
        assert engine.get_call_count("refund_order") == 2

    @pytest.mark.asyncio
    async def test_mcp_server_fault_interception(self):
        """MCP server catches FaultInjectionError and returns structured MCP payload."""
        engine = FaultInjectionEngine()
        engine.register_rule(
            FaultRule(
                target_tool="create_order",
                fault_type=FaultType.RATE_LIMIT,
                error_message="Rate limit 429",
            )
        )
        mcp = AgentMCPServer(fault_engine=engine)
        mcp.register_tool("create_order", lambda **kwargs: {"status": "ok"})

        response = await mcp.execute_tool("create_order", {"amount": 100})
        assert response.get("is_error") is True
        assert response.get("fault_injected") is True
        assert response.get("fault_type") == "RATE_LIMIT"
        assert len(mcp.call_trace) == 1
        assert mcp.call_trace[0]["status"] == "fault_injected"

    @pytest.mark.asyncio
    async def test_corrupted_response_fault(self):
        engine = FaultInjectionEngine()
        engine.register_rule(
            FaultRule(target_tool="get_data", fault_type=FaultType.CORRUPTED_RESPONSE)
        )
        res = await engine.intercept("get_data", lambda: {"status": "ok"})
        assert res.get("corrupted") is True


class TestIdempotencyStoreAndVerifier:
    @pytest.mark.asyncio
    async def test_idempotency_store_lifecycle(self):
        store = IdempotencyStore()
        key = "idem-test-key-01"

        # 1. First request starts
        completed, cached = await store.check_or_start(key)
        assert not completed
        assert cached is None

        # 2. Concurrent duplicate request while in progress raises conflict
        with pytest.raises(IdempotencyConflictError):
            await store.check_or_start(key)

        # 3. Complete first request
        payload = {"order_id": "ord_999", "status": "created"}
        await store.complete(key, payload, resource_id="ord_999")

        # 4. Subsequent retry with same key returns cached response
        completed, cached = await store.check_or_start(key)
        assert completed
        assert cached == payload

    @pytest.mark.asyncio
    async def test_sandbox_idempotent_order_creation(self):
        sandbox = EnvironmentSandbox()
        sandbox.provision()

        key = "order-idem-123"
        items = [{"name": "Widget", "amount": 50.0}]

        # Call 1
        res1 = await sandbox.create_order("cust_123", items, idempotency_key=key)
        # Call 2 (retry)
        res2 = await sandbox.create_order("cust_123", items, idempotency_key=key)

        assert res1["order_id"] == res2["order_id"]
        # Only 1 order created in state
        assert len(sandbox.state["orders"]) == 2  # 1 initial ord_456 + 1 new order

    def test_idempotency_verifier_detects_violations(self):
        verifier = IdempotencyVerifier()
        state_before = {"orders": {"ord_1": {"total": 100, "refund_count": 0}}}

        # Scenario A: Violated refund (refund_count is 2 despite same idempotency key)
        state_after_bad = {"orders": {"ord_1": {"total": 100, "refund_count": 2}}}
        trace_bad = [
            {"tool": "refund_order", "arguments": {"order_id": "ord_1", "idempotency_key": "idem-ref-1"}},
            {"tool": "refund_order", "arguments": {"order_id": "ord_1", "idempotency_key": "idem-ref-1"}},
        ]
        res_bad = verifier.verify(trace_bad, state_before, state_after_bad)
        assert not res_bad.passed
        assert len(res_bad.violations) > 0

        # Scenario B: Clean idempotency (refund_count is 1)
        state_after_good = {"orders": {"ord_1": {"total": 100, "refund_count": 1}}}
        res_good = verifier.verify(trace_bad, state_before, state_after_good)
        assert res_good.passed
        assert len(res_good.violations) == 0


# ============================================================================
# TASK 10: Concurrency & Race Condition Detection Tests
# ============================================================================

class TestConcurrencyAndRaceConditions:
    @pytest.mark.asyncio
    async def test_keyed_async_lock_mutual_exclusion(self):
        lock_manager = KeyedAsyncLock()
        counter = {"count": 0}

        async def worker():
            async with lock_manager.acquire("resource_A"):
                c = counter["count"]
                await asyncio.sleep(0.01)
                counter["count"] = c + 1

        await asyncio.gather(*[worker() for _ in range(10)])
        assert counter["count"] == 10

    @pytest.mark.asyncio
    async def test_concurrency_runner_with_sandbox(self):
        sandbox = EnvironmentSandbox()
        sandbox.provision()
        runner = ConcurrencyRunner()

        key = "stress-test-order-key"
        items = [{"name": "Book", "amount": 25.0}]

        # Fire 25 concurrent order creations with the same idempotency key
        result = await runner.run(
            sandbox.create_order,
            num_requests=25,
            customer_id="cust_123",
            items=items,
            idempotency_key=key,
        )

        assert result.total_requests == 25
        assert result.successful_calls > 0

        # Verify race condition
        verifier = RaceConditionVerifier()
        state_before = {"orders": {"ord_456": {}}}
        ver_result = verifier.verify(
            resource_type="orders",
            state_before=state_before,
            state_after=sandbox.state,
            expected_max_mutations=1,
        )

        assert ver_result.passed
        assert not ver_result.race_detected
        assert ver_result.actual_mutations == 1

    def test_race_condition_verifier_catches_unprotected_race(self):
        verifier = RaceConditionVerifier()
        state_before = {"orders": {"ord_1": {"refund_count": 0}}}
        state_after_corrupted = {"orders": {"ord_1": {"refund_count": 3}}}

        ver_result = verifier.verify(
            resource_type="refund",
            state_before=state_before,
            state_after=state_after_corrupted,
            resource_id="ord_1",
            expected_max_mutations=1,
        )

        assert not ver_result.passed
        assert ver_result.race_detected
        assert "Race condition detected" in ver_result.details


# ============================================================================
# TASK 11: Hidden Tests, Dataset Versioning & Regression Detection Tests
# ============================================================================

class TestHiddenTestsAndVersioning:
    def test_task_public_view_hides_private_contract(self):
        task = Task(
            dataset_id=uuid.uuid4(),
            name="Refund Task",
            input_prompt="Please refund order ord_456",
            allowed_tools=["refund_order"],
            invariants=["no_duplicate_orders"],
            hidden_invariants=["customer_email_immutable", "internal_accounting_sync"],
            hidden_expected_state={"accounting_ledger": "reconciled"},
        )

        public_view = task.get_public_view()
        assert "hidden_invariants" not in public_view
        assert "hidden_expected_state" not in public_view
        assert public_view["allowed_tools"] == ["refund_order"]
        assert "customer_email_immutable" not in public_view["invariants"]

    def test_dataset_versioning_and_snapshot(self):
        dataset = Dataset(
            project_id=uuid.uuid4(),
            name="Reliability Benchmark",
            version="1.2.0",
        )
        task1 = Task(
            dataset_id=dataset.id,
            name="Task 1",
            input_prompt="Do something",
        )
        dataset.tasks.append(task1)

        snapshot = create_dataset_snapshot(dataset)
        assert snapshot.dataset_name == "Reliability Benchmark"
        assert snapshot.version == "1.2.0"
        assert len(snapshot.tasks) == 1
        assert len(snapshot.version_hash) > 0

    def test_regression_detector_catches_hidden_regression(self):
        trial_id = uuid.uuid4()
        baseline = VerificationResult(
            trial_id=trial_id,
            passed=True,
            overall_score=1.0,
            public_score=1.0,
            hidden_score=1.0,
        )
        # Candidate maintained public score but degraded on hidden tests (overfitting!)
        candidate = VerificationResult(
            trial_id=trial_id,
            passed=False,
            overall_score=0.6,
            public_score=1.0,
            hidden_score=0.2,
            hidden_failures=["customer_email_immutable violated"],
        )

        analysis = RegressionDetector.compare_results(baseline, candidate)
        assert analysis["has_regression"] is True
        assert analysis["is_hidden_regression"] is True
        assert analysis["overfitting_detected"] is True
        assert len(analysis["regressions"]) >= 2

    @pytest.mark.asyncio
    async def test_trial_manager_full_lifecycle_with_hidden_invariants(self):
        trial_manager = TrialManager()
        task = Task(
            dataset_id=uuid.uuid4(),
            name="Refund and Verify Invariants",
            input_prompt="Refund ord_456 safely",
            allowed_tools=["refund_order"],
            hidden_invariants=["customer_email_immutable"],
        )
        trial = Trial(run_id=uuid.uuid4(), task_id=task.id)

        # Execute trial
        result = await trial_manager.execute_trial(trial, task)
        assert result.passed
        assert result.public_passed
        assert result.hidden_passed
        assert result.overall_score == 1.0
