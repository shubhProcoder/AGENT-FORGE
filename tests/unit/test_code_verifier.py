"""Unit tests for CodeVerifier."""

from __future__ import annotations

import uuid
import pytest

from backend.domain.result import EvaluationResult
from backend.verification.candidate_validator import ValidationResult
from backend.verification.code_verifier import CodeVerifier
from backend.verification.test_runner import TestCaseResult, TestRunResult


class TestCodeVerifier:
    def setup_method(self):
        self.verifier = CodeVerifier()
        self.trial_id = uuid.uuid4()
        self.task_id = uuid.uuid4()

    def test_perfect_solution(self):
        """All public and hidden tests pass -> 1.0 score."""
        pub_result = TestRunResult(
            passed=True,
            exit_code=0,
            total_tests=8,
            passed_count=8,
            failed_count=0,
            test_cases=[TestCaseResult(name=f"test_{i}", classname="", status="passed") for i in range(8)],
        )
        hid_result = TestRunResult(
            passed=True,
            exit_code=0,
            total_tests=6,
            passed_count=6,
            failed_count=0,
            test_cases=[TestCaseResult(name=f"test_h_{i}", classname="", status="passed") for i in range(6)],
        )

        res = self.verifier.verify(
            public_result=pub_result,
            hidden_result=hid_result,
            trial_id=self.trial_id,
            task_id=self.task_id,
        )

        assert isinstance(res, EvaluationResult)
        assert res.passed is True
        assert res.overall_score == 1.0
        assert res.public_passed is True
        assert res.hidden_passed is True
        assert res.public_score == 1.0
        assert res.hidden_score == 1.0
        assert res.failure_type is None
        assert res.failure_category is None

    def test_overfitting_detection(self):
        """Candidate passes 8/8 public tests but fails 3 hidden tests."""
        pub_result = TestRunResult(
            passed=True,
            exit_code=0,
            total_tests=8,
            passed_count=8,
            failed_count=0,
            test_cases=[TestCaseResult(name=f"test_{i}", classname="", status="passed") for i in range(8)],
        )
        hid_result = TestRunResult(
            passed=False,
            exit_code=1,
            total_tests=6,
            passed_count=3,
            failed_count=3,
            test_cases=[
                TestCaseResult(name="test_h_1", classname="", status="passed"),
                TestCaseResult(name="test_h_2", classname="", status="passed"),
                TestCaseResult(name="test_h_3", classname="", status="passed"),
                TestCaseResult(name="test_h_4", classname="", status="failed", failure_message="AssertionError: 1 != None"),
                TestCaseResult(name="test_h_5", classname="", status="failed", failure_message="AssertionError"),
                TestCaseResult(name="test_h_6", classname="", status="failed", failure_message="AssertionError"),
            ],
        )

        res = self.verifier.verify(
            public_result=pub_result,
            hidden_result=hid_result,
            trial_id=self.trial_id,
            task_id=self.task_id,
        )

        assert res.passed is False
        assert res.public_passed is True
        assert res.hidden_passed is False
        assert res.public_score == 1.0
        assert res.hidden_score == 0.5
        assert res.failure_category == "OVERFITTING / CORRECTNESS"
        assert res.failure_type == "HIDDEN_TEST_FAILURE"
        assert len(res.hidden_failures) == 3

    def test_syntax_validation_failure(self):
        """Syntax error in validation stops before sandbox and records syntax failure."""
        val_res = ValidationResult(
            valid=False,
            error_type="SYNTAX_ERROR",
            message="unmatched ')'",
            line=5,
            column=12,
        )

        res = self.verifier.verify(
            public_result=None,
            validation_result=val_res,
            trial_id=self.trial_id,
            task_id=self.task_id,
        )

        assert res.passed is False
        assert res.failure_type == "SYNTAX_ERROR"
        assert res.failure_category == "SYNTAX_ERROR"
        assert len(res.public_failures) == 1
        assert res.public_failures[0]["line"] == 5

    def test_timeout_distinction(self):
        """Sandbox timeout is recorded as TIMEOUT failure."""
        pub_result = TestRunResult(
            passed=False,
            exit_code=-1,
            failure_category="TIMEOUT",
            stderr="Execution timed out after 5.0 seconds",
        )

        res = self.verifier.verify(
            public_result=pub_result,
            trial_id=self.trial_id,
            task_id=self.task_id,
        )

        assert res.passed is False
        assert res.failure_type == "TIMEOUT"
        assert res.failure_category == "TIMEOUT"

    def test_resource_limit_distinction(self):
        """OOM is recorded as RESOURCE_LIMIT_EXCEEDED failure."""
        pub_result = TestRunResult(
            passed=False,
            exit_code=137,
            failure_category="RESOURCE_LIMIT_EXCEEDED",
            stderr="Killed (Out of memory)",
        )

        res = self.verifier.verify(
            public_result=pub_result,
            trial_id=self.trial_id,
            task_id=self.task_id,
        )

        assert res.passed is False
        assert res.failure_type == "RESOURCE_LIMIT_EXCEEDED"

    def test_sandbox_startup_failure_distinction(self):
        """Daemon failure is distinguished from candidate test failure."""
        pub_result = TestRunResult(
            passed=False,
            exit_code=125,
            failure_category="SANDBOX_ERROR",
            stderr="docker: Cannot connect to Docker daemon",
        )

        res = self.verifier.verify(
            public_result=pub_result,
            trial_id=self.trial_id,
            task_id=self.task_id,
        )

        assert res.passed is False
        assert res.failure_type == "SANDBOX_ERROR"
        assert res.failure_category == "SANDBOX_ERROR"
