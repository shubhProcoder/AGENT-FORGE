"""Code verifier for interpreting test run evidence and classifying outcomes.

Interprets machine-readable JUnit evidence from public and hidden sandboxes,
classifies distinct failure modes (syntax, runtime, test, timeout, OOM, sandbox),
detects overfitting (public pass + hidden fail), and produces structured VerificationResult / EvaluationResult.
Does NOT execute code.
"""

from __future__ import annotations

import logging
from typing import Any
import uuid

from backend.domain.dataset import Task
from backend.domain.result import EvaluationResult, VerificationResult
from backend.verification.candidate_validator import ValidationResult
from backend.verification.test_runner import TestRunResult
from evaluation.engine import compute_score
from evaluation.metrics import VerificationOutcome

logger = logging.getLogger(__name__)


class CodeVerifier:
    """Verifies candidate execution outcomes against public and hidden test contracts."""

    def verify(
        self,
        public_result: TestRunResult | None,
        hidden_result: TestRunResult | None = None,
        validation_result: ValidationResult | None = None,
        task: Task | None = None,
        trial_id: uuid.UUID | None = None,
        task_id: uuid.UUID | None = None,
    ) -> EvaluationResult:
        """Evaluate evidence from static validation and public/hidden test runs."""
        tid = trial_id or uuid.uuid4()
        tkid = task_id or (task.id if task else uuid.uuid4())

        public_failures: list[dict[str, Any]] = []
        hidden_failures: list[dict[str, Any]] = []
        metrics: dict[str, float] = {}
        evidence: dict[str, Any] = {}

        # 1. Check static syntax validation failure
        if validation_result and not validation_result.valid:
            failure_type = validation_result.error_type or "SYNTAX_ERROR"
            fail_record = {
                "type": failure_type,
                "message": validation_result.message or "Syntax error during static analysis",
                "line": validation_result.line,
                "column": validation_result.column,
            }
            public_failures.append(fail_record)
            metrics.update({
                "public_score": 0.0,
                "hidden_score": 0.0,
                "overall_score": 0.0,
                "functional_correctness": 0.0,
            })
            return EvaluationResult(
                trial_id=tid,
                task_id=tkid,
                passed=False,
                overall_score=0.0,
                public_passed=False,
                public_score=0.0,
                hidden_passed=False,
                hidden_score=0.0,
                failure_type=failure_type,
                failure_category="SYNTAX_ERROR",
                public_failures=public_failures,
                hidden_failures=[],
                metrics=metrics,
                evidence={"validation": fail_record},
            )

        # 2. Check public test execution results
        if not public_result:
            return EvaluationResult(
                trial_id=tid,
                task_id=tkid,
                passed=False,
                overall_score=0.0,
                public_passed=False,
                public_score=0.0,
                hidden_passed=False,
                hidden_score=0.0,
                failure_type="SANDBOX_ERROR",
                failure_category="SANDBOX_ERROR",
                public_failures=[{"type": "SANDBOX_ERROR", "message": "No public test result available"}],
                metrics={"overall_score": 0.0},
            )

        evidence["public_execution"] = {
            "exit_code": public_result.exit_code,
            "failure_category": public_result.failure_category,
            "total_tests": public_result.total_tests,
            "passed_count": public_result.passed_count,
            "failed_count": public_result.failed_count,
            "error_count": public_result.error_count,
            "duration_seconds": public_result.duration_seconds,
            "stdout": public_result.stdout[:2048],
            "stderr": public_result.stderr[:2048],
        }

        # Check for infrastructure/timeout/OOM in public run
        if public_result.failure_category in ("SANDBOX_ERROR", "TIMEOUT", "RESOURCE_LIMIT_EXCEEDED"):
            fail_type = public_result.failure_category
            public_failures.append({
                "type": fail_type,
                "message": public_result.stderr or public_result.output,
            })
            metrics.update({"public_score": 0.0, "hidden_score": 0.0, "overall_score": 0.0})
            return EvaluationResult(
                trial_id=tid,
                task_id=tkid,
                passed=False,
                overall_score=0.0,
                public_passed=False,
                public_score=0.0,
                hidden_passed=False,
                hidden_score=0.0,
                failure_type=fail_type,
                failure_category=fail_type,
                public_failures=public_failures,
                hidden_failures=[],
                metrics=metrics,
                evidence=evidence,
            )

        # Compute public test score
        if public_result.total_tests > 0:
            public_score = round(public_result.passed_count / public_result.total_tests, 4)
            public_passed = public_result.passed and (public_result.passed_count == public_result.total_tests)
        else:
            public_score = 1.0 if public_result.passed else 0.0
            public_passed = public_result.passed

        # Record public test failures
        for tc in public_result.test_cases:
            if tc.status in ("failed", "error"):
                public_failures.append({
                    "test_name": tc.name,
                    "status": tc.status,
                    "message": tc.failure_message,
                    "traceback": tc.traceback,
                })

        # 3. Check hidden test execution results
        hidden_passed = True
        hidden_score = 1.0

        if hidden_result:
            evidence["hidden_execution"] = {
                "exit_code": hidden_result.exit_code,
                "failure_category": hidden_result.failure_category,
                "total_tests": hidden_result.total_tests,
                "passed_count": hidden_result.passed_count,
                "failed_count": hidden_result.failed_count,
                "error_count": hidden_result.error_count,
                "duration_seconds": hidden_result.duration_seconds,
            }

            if hidden_result.failure_category in ("SANDBOX_ERROR", "TIMEOUT", "RESOURCE_LIMIT_EXCEEDED"):
                hidden_passed = False
                hidden_score = 0.0
                hidden_failures.append({
                    "type": hidden_result.failure_category,
                    "message": hidden_result.stderr or hidden_result.output,
                })
            elif hidden_result.total_tests > 0:
                hidden_score = round(hidden_result.passed_count / hidden_result.total_tests, 4)
                hidden_passed = hidden_result.passed and (hidden_result.passed_count == hidden_result.total_tests)
            else:
                hidden_score = 1.0 if hidden_result.passed else 0.0
                hidden_passed = hidden_result.passed

            for tc in hidden_result.test_cases:
                if tc.status in ("failed", "error"):
                    hidden_failures.append({
                        "test_name": tc.name,
                        "status": tc.status,
                        "message": tc.failure_message,
                    })

        # 4. Determine overall score and failure classification
        if hidden_result:
            overall_score = round((public_score * 0.5) + (hidden_score * 0.5), 4)
            passed = public_passed and hidden_passed
        else:
            overall_score = public_score
            passed = public_passed

        # Classify failure category
        failure_type = None
        failure_category = None

        if not passed:
            if not public_passed:
                failure_type = public_result.failure_category or "TEST_FAILURE"
                failure_category = failure_type
            elif not hidden_passed:
                # Public tests passed, but hidden tests failed!
                failure_type = "HIDDEN_TEST_FAILURE"
                failure_category = "OVERFITTING / CORRECTNESS"

        metrics["public_score"] = public_score
        metrics["hidden_score"] = hidden_score
        metrics["overall_score"] = overall_score
        metrics["functional_correctness"] = overall_score

        # Also pipe into VerificationOutcome for scoring engine compatibility
        outcome = VerificationOutcome(passed=passed)
        outcome.add_metric("functional_correctness", overall_score)
        scored = compute_score(outcome)

        return EvaluationResult(
            trial_id=tid,
            task_id=tkid,
            passed=passed,
            overall_score=round(scored.overall, 4),
            public_passed=public_passed,
            public_score=public_score,
            hidden_passed=hidden_passed,
            hidden_score=hidden_score,
            failure_type=failure_type,
            failure_category=failure_category,
            public_failures=public_failures,
            hidden_failures=hidden_failures,
            metrics=metrics,
            evidence=evidence,
        )
