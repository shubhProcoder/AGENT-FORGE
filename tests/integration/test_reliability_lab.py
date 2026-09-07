"""Integration tests for the reliability lab experiments."""

from __future__ import annotations

import pytest

from lab.concurrency import run_concurrency_experiment
from lab.idempotency import run_idempotency_experiment
from lab.experiments import run_all_experiments


class TestConcurrencyExperiment:
    @pytest.mark.asyncio
    async def test_concurrent_creates_produce_one_order(self):
        result = await run_concurrency_experiment(num_requests=50)
        assert result.passed, result.details
        assert result.orders_created == 1

    @pytest.mark.asyncio
    async def test_small_concurrency(self):
        result = await run_concurrency_experiment(num_requests=5)
        assert result.passed


class TestIdempotencyExperiment:
    @pytest.mark.asyncio
    async def test_retries_create_one_ticket(self):
        result = await run_idempotency_experiment(num_retries=10)
        assert result.passed, result.details
        assert result.items_created == 1


class TestExperimentSuite:
    @pytest.mark.asyncio
    async def test_all_experiments_pass(self):
        suite = await run_all_experiments()
        assert suite.all_passed, suite.summary
