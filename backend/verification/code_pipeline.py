"""Code evaluation pipeline.

Coordinates the complete end-to-end evaluation flow:
Task
 → Candidate Code Extraction
 → Candidate Validation (Static AST)
 → Public Test Isolation Sandbox Execution
 → Hidden Test Isolation Sandbox Execution
 → CodeVerifier Evidence Interpretation
 → Scoring Engine
 → EvaluationResult Persistence & Event Emission

Enforces strict separation: candidate code NEVER receives hidden test source.
Does not execute code in the FastAPI host process.
"""

from __future__ import annotations

import logging
from typing import Any, Callable
import uuid

from backend.domain.dataset import Task
from backend.domain.result import EvaluationResult
from backend.domain.sandbox import CodeExecutionService
from backend.verification.candidate_validator import CandidateValidator, ValidationResult
from backend.verification.code_extractor import (
    CandidateCode,
    CodeExtractionError,
    CodeExtractionService,
)
from backend.verification.code_verifier import CodeVerifier
from backend.verification.test_runner import PytestAdapter, TestRunResult

logger = logging.getLogger(__name__)


class CodeEvaluationPipeline:
    """Orchestrates the secure, multi-stage code evaluation lifecycle."""

    def __init__(
        self,
        code_executor: CodeExecutionService,
        code_extractor: CodeExtractionService | None = None,
        candidate_validator: CandidateValidator | None = None,
        code_verifier: CodeVerifier | None = None,
        event_listener: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.code_executor = code_executor
        self.extractor = code_extractor or CodeExtractionService()
        self.validator = candidate_validator or CandidateValidator()
        self.verifier = code_verifier or CodeVerifier()
        self.pytest_adapter = PytestAdapter(code_executor=code_executor)
        self.event_listener = event_listener
        self.recorded_events: list[dict[str, Any]] = []

    def _emit_event(self, event_name: str, payload: dict[str, Any]) -> None:
        """Emit structured lifecycle event for observability (TASK 15 hooks)."""
        event_record = {"event": event_name, **payload}
        self.recorded_events.append(event_record)
        if self.event_listener:
            try:
                self.event_listener(event_name, payload)
            except Exception as e:
                logger.warning(f"Error in event listener for {event_name}: {e}")

    async def evaluate(
        self,
        task: Task,
        agent_output: str | None = None,
        candidate_code: str | None = None,
        trial_id: uuid.UUID | None = None,
        run_id: uuid.UUID | None = None,
        trace_id: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> EvaluationResult:
        """Execute the complete evaluation workflow on a candidate solution."""
        trial_id = trial_id or uuid.uuid4()
        run_id = run_id or uuid.uuid4()
        trace_id = trace_id or str(uuid.uuid4())
        base_event_ctx = {
            "run_id": str(run_id),
            "trial_id": str(trial_id),
            "execution_id": str(trial_id),
            "trace_id": trace_id,
            "task_id": str(task.id),
        }

        self._emit_event("code_execution_started", base_event_ctx)

        # 1. Extract candidate code
        extracted: CandidateCode
        if candidate_code is not None:
            extracted = CandidateCode(source=candidate_code, extracted_from="direct_input")
        else:
            try:
                extracted = self.extractor.extract(agent_output)
            except CodeExtractionError as e:
                logger.warning(f"Extraction error for task {task.name}: {e}")
                self._emit_event("code_execution_failed", {**base_event_ctx, "error": str(e)})
                return EvaluationResult(
                    trial_id=trial_id,
                    task_id=task.id,
                    passed=False,
                    overall_score=0.0,
                    public_passed=False,
                    public_score=0.0,
                    hidden_passed=False,
                    hidden_score=0.0,
                    failure_type="EXTRACTION_ERROR",
                    failure_category="EXTRACTION_ERROR",
                    public_failures=[{"type": "EXTRACTION_ERROR", "message": str(e)}],
                    metrics={"overall_score": 0.0},
                )

        # 2. Validate candidate syntax (static AST check)
        validation_result: ValidationResult = self.validator.validate(extracted.source)
        if not validation_result.valid:
            logger.info(f"Candidate code failed static syntax check: {validation_result.message}")
            self._emit_event("code_execution_failed", {
                **base_event_ctx,
                "error": validation_result.message,
                "error_type": "SYNTAX_ERROR",
            })
            # Static syntax failure prevents sandbox execution
            eval_result = self.verifier.verify(
                public_result=None,
                validation_result=validation_result,
                task=task,
                trial_id=trial_id,
                task_id=task.id,
            )
            return eval_result

        # 3. Public evaluation in isolated sandbox
        public_test_code = task.expected_state.get("public_test_code", "")
        public_test_file = task.expected_state.get("public_test_file", "test_public.py")
        public_result: TestRunResult | None = None

        if public_test_code:
            self._emit_event("test_run_started", {**base_event_ctx, "suite": "public"})
            public_result = await self.pytest_adapter.run_tests(
                candidate_code=extracted.source,
                test_code=public_test_code,
                test_filename=public_test_file,
                timeout_seconds=timeout_seconds,
            )
            self._emit_event("test_run_completed", {
                **base_event_ctx,
                "suite": "public",
                "passed": public_result.passed,
                "exit_code": public_result.exit_code,
            })

            # Check for timeout or resource violations in public sandbox
            if public_result.failure_category == "TIMEOUT":
                self._emit_event("sandbox_timeout", base_event_ctx)
            elif public_result.failure_category == "RESOURCE_LIMIT_EXCEEDED":
                self._emit_event("sandbox_resource_limit", base_event_ctx)

        # 4. Hidden evaluation in a SEPARATE, PROTECTED isolated sandbox
        # Candidate execution environment never receives hidden test source
        hidden_test_code = task.hidden_expected_state.get("hidden_test_code", "")
        hidden_test_file = task.hidden_expected_state.get("hidden_test_file", "test_hidden.py")
        hidden_result: TestRunResult | None = None

        # Execute hidden tests if public tests ran (or if public passed / tests are present)
        # Only skip hidden evaluation if public execution suffered infrastructure or fatal timeout
        if hidden_test_code and public_result and public_result.failure_category not in ("SANDBOX_ERROR", "TIMEOUT", "RESOURCE_LIMIT_EXCEEDED"):
            self._emit_event("test_run_started", {**base_event_ctx, "suite": "hidden"})
            hidden_result = await self.pytest_adapter.run_tests(
                candidate_code=extracted.source,
                test_code=hidden_test_code,
                test_filename=hidden_test_file,
                timeout_seconds=timeout_seconds,
            )
            self._emit_event("test_run_completed", {
                **base_event_ctx,
                "suite": "hidden",
                "passed": hidden_result.passed,
                "exit_code": hidden_result.exit_code,
            })

            if hidden_result.failure_category == "TIMEOUT":
                self._emit_event("sandbox_timeout", base_event_ctx)
            elif hidden_result.failure_category == "RESOURCE_LIMIT_EXCEEDED":
                self._emit_event("sandbox_resource_limit", base_event_ctx)

        # 5. Verification & scoring
        self._emit_event("verification_started", base_event_ctx)
        eval_result = self.verifier.verify(
            public_result=public_result,
            hidden_result=hidden_result,
            validation_result=validation_result,
            task=task,
            trial_id=trial_id,
            task_id=task.id,
        )
        self._emit_event("verification_completed", {
            **base_event_ctx,
            "overall_score": eval_result.overall_score,
            "passed": eval_result.passed,
        })

        if eval_result.passed:
            self._emit_event("code_execution_completed", base_event_ctx)
        else:
            self._emit_event("code_execution_failed", {
                **base_event_ctx,
                "failure_type": eval_result.failure_type,
                "failure_category": eval_result.failure_category,
            })

        return eval_result
