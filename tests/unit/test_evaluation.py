"""Unit tests for evaluation engine, verifiers, and scoring."""

from __future__ import annotations

import pytest

from agent.state import ExecutionContext, ToolCallRecord
from evaluation.engine import compute_score, detect_regressions
from evaluation.metrics import EvaluationScore, VerificationOutcome
from evaluation.verifiers.tool_usage import verify_tool_usage
from evaluation.verifiers.policy import verify_policy
from evaluation.verifiers.performance import verify_performance
from evaluation.verifiers.state_invariant import verify_state_invariants
from tools.order.create_order import reset_order_store, create_order
from tools.order.refund_order import refund_order


# ── Scoring engine ───────────────────────────────────────────────────────────

class TestScoringEngine:
    def test_perfect_score(self):
        outcome = VerificationOutcome(passed=True)
        # All metrics default to 1.0 if not set
        score = compute_score(outcome)
        assert score.overall == 1.0

    def test_custom_weights(self):
        outcome = VerificationOutcome(passed=True)
        outcome.add_metric("safety", 0.5)
        score = compute_score(outcome, weights={"safety": 1.0})
        assert score.overall < 1.0

    def test_regression_detection(self):
        prev = EvaluationScore(overall=0.9, metric_scores={"safety": 0.98, "tool_correctness": 0.94})
        curr = EvaluationScore(overall=0.85, metric_scores={"safety": 0.81, "tool_correctness": 0.82})
        regressions = detect_regressions(curr, prev, threshold=0.05)
        assert len(regressions) == 2
        assert any(r["metric"] == "safety" for r in regressions)


# ── Tool usage verifier ──────────────────────────────────────────────────────

class TestToolUsageVerifier:
    def _make_ctx(self, tools_called: list[str]) -> ExecutionContext:
        ctx = ExecutionContext()
        for i, name in enumerate(tools_called, 1):
            ctx.tool_calls.append(
                ToolCallRecord(seq=i, tool_name=name, arguments={}, status="success")
            )
        return ctx

    def test_required_tools_present(self):
        ctx = self._make_ctx(["search_customer", "create_ticket"])
        result = verify_tool_usage(ctx, required_tools=["search_customer", "create_ticket"])
        assert result.passed
        assert result.metrics["tool_correctness"] == 1.0

    def test_required_tool_missing(self):
        ctx = self._make_ctx(["search_customer"])
        result = verify_tool_usage(ctx, required_tools=["search_customer", "create_ticket"])
        assert not result.passed

    def test_forbidden_tool_used(self):
        ctx = self._make_ctx(["search_customer", "delete_customer"])
        result = verify_tool_usage(ctx, forbidden_tools=["delete_customer"])
        assert not result.passed
        assert result.metrics["tool_correctness"] == 0.0


# ── Policy verifier ──────────────────────────────────────────────────────────

class TestPolicyVerifier:
    def _make_ctx(self, tools: list[tuple[str, dict]]) -> ExecutionContext:
        ctx = ExecutionContext()
        for i, (name, args) in enumerate(tools, 1):
            ctx.tool_calls.append(
                ToolCallRecord(seq=i, tool_name=name, arguments=args, status="success")
            )
        return ctx

    def test_safe_execution(self):
        ctx = self._make_ctx([("search_customer", {"customer_id": 1})])
        result = verify_policy(ctx)
        assert result.passed
        assert result.metrics["safety"] == 1.0

    def test_forbidden_tool_detected(self):
        ctx = self._make_ctx([("delete_customer", {"customer_id": 1})])
        result = verify_policy(ctx)
        assert not result.passed
        assert result.metrics["safety"] == 0.0


# ── State invariant verifier ─────────────────────────────────────────────────

class TestStateInvariantVerifier:
    @pytest.mark.asyncio
    async def test_no_violations(self):
        reset_order_store()
        await create_order(customer_id=1, items=[{"name": "A", "amount": 10}])
        ctx = ExecutionContext()
        result = verify_state_invariants(ctx)
        assert result.passed

    @pytest.mark.asyncio
    async def test_duplicate_refund_violation(self):
        reset_order_store()
        res = await create_order(customer_id=1, items=[{"name": "B", "amount": 20}])
        oid = res["data"]["order_id"]
        await refund_order(order_id=oid)
        await refund_order(order_id=oid)
        ctx = ExecutionContext()
        result = verify_state_invariants(ctx)
        assert not result.passed
        assert any("duplicate_refund" in str(f) for f in result.failures)


# ── Performance verifier ─────────────────────────────────────────────────────

class TestPerformanceVerifier:
    def test_within_budget(self):
        ctx = ExecutionContext()
        ctx.estimated_cost_usd = 0.01
        # Fake a fast elapsed time
        import time
        ctx.start_time = time.time() - 2  # 2 seconds elapsed
        result = verify_performance(ctx, max_latency_seconds=30, max_cost_usd=0.10)
        assert result.metrics["latency"] == 1.0
        assert result.metrics["cost"] == 1.0

    def test_over_budget(self):
        ctx = ExecutionContext()
        ctx.estimated_cost_usd = 0.25
        import time
        ctx.start_time = time.time() - 90  # 90 seconds elapsed
        result = verify_performance(ctx, max_latency_seconds=30, max_cost_usd=0.10)
        assert result.metrics["latency"] < 1.0
        assert result.metrics["cost"] < 1.0
