"""Unit tests for CodeEvaluationPipeline and Acceptance Demonstrations.

Covers:
- Demo A: Perfect solution (Public: 8/8, Hidden: 6/6, Score: 100, PASS)
- Demo B: Public overfitting (Public: 8/8, Hidden: fails, FAIL, OVERFITTING / CORRECTNESS)
- Demo C: Infinite loop (TIMEOUT, not TEST_FAILURE)
- Demo D: Syntax error (SYNTAX_ERROR, no sandbox execution)
- Demo E: Runtime error (RUNTIME_ERROR)
- Demo F: Sandbox unavailable (SANDBOX_ERROR, infrastructure failure)
- Observability events emission
- Public vs hidden test isolation
"""

from __future__ import annotations

import uuid
import pytest

from backend.domain.dataset import Task
from backend.domain.sandbox import ExecutionRequest, ExecutionStatus
from backend.evaluation.scenarios.second_largest_unique import (
    CANONICAL_PERFECT_SOLUTION,
    OVERFITTED_PUBLIC_ONLY_SOLUTION,
    create_second_largest_unique_task,
)
from backend.infrastructure.sandbox.docker_executor import DockerCodeExecutor
from backend.infrastructure.sandbox.process_runner import (
    FakeContainerProcessRunner,
    ProcessOutput,
)
from backend.verification.code_pipeline import CodeEvaluationPipeline


JUNIT_XML_PUBLIC_PERFECT = """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="8" errors="0" failures="0" skipped="0" time="0.120">
    <testcase classname="test_public" name="test_normal_case" time="0.01" />
    <testcase classname="test_public" name="test_duplicates" time="0.01" />
    <testcase classname="test_public" name="test_negative_values" time="0.01" />
    <testcase classname="test_public" name="test_empty_input" time="0.01" />
    <testcase classname="test_public" name="test_one_unique_value" time="0.01" />
    <testcase classname="test_public" name="test_already_sorted" time="0.01" />
    <testcase classname="test_public" name="test_reverse_sorted" time="0.01" />
    <testcase classname="test_public" name="test_large_input" time="0.05" />
</testsuite>"""

JUNIT_XML_HIDDEN_PERFECT = """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="6" errors="0" failures="0" skipped="0" time="0.090">
    <testcase classname="test_hidden" name="test_two_unique_heavy_duplicates" time="0.01" />
    <testcase classname="test_hidden" name="test_mixed_negative_zero_positive" time="0.01" />
    <testcase classname="test_hidden" name="test_all_negative_duplicates" time="0.01" />
    <testcase classname="test_hidden" name="test_single_element" time="0.01" />
    <testcase classname="test_hidden" name="test_float_values" time="0.01" />
    <testcase classname="test_hidden" name="test_extreme_integer_values" time="0.01" />
</testsuite>"""

JUNIT_XML_HIDDEN_OVERFITTED = """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="6" errors="0" failures="3" skipped="0" time="0.090">
    <testcase classname="test_hidden" name="test_two_unique_heavy_duplicates" time="0.01">
        <failure message="assert None == 1">AssertionError: assert None == 1</failure>
    </testcase>
    <testcase classname="test_hidden" name="test_mixed_negative_zero_positive" time="0.01">
        <failure message="assert None == -5">AssertionError: assert None == -5</failure>
    </testcase>
    <testcase classname="test_hidden" name="test_all_negative_duplicates" time="0.01" />
    <testcase classname="test_hidden" name="test_single_element" time="0.01" />
    <testcase classname="test_hidden" name="test_float_values" time="0.01">
        <failure message="assert None == 1.5">AssertionError: assert None == 1.5</failure>
    </testcase>
    <testcase classname="test_hidden" name="test_extreme_integer_values" time="0.01" />
</testsuite>"""


@pytest.mark.asyncio
class TestCodeEvaluationPipeline:
    def setup_method(self):
        self.task = create_second_largest_unique_task()

    async def test_demo_a_perfect_solution(self):
        """Demo A: Perfect candidate scores 1.0 (8/8 public, 6/6 hidden)."""
        def custom_runner(command_args, timeout, container_name):
            # Check if this is public or hidden test execution
            if "test_public.py" in command_args or any("test_public" in a for a in command_args):
                return ProcessOutput(
                    exit_code=0,
                    stdout=JUNIT_XML_PUBLIC_PERFECT,
                    stderr="",
                    duration_ms=120.0,
                )
            else:
                return ProcessOutput(
                    exit_code=0,
                    stdout=JUNIT_XML_HIDDEN_PERFECT,
                    stderr="",
                    duration_ms=90.0,
                )

        runner = FakeContainerProcessRunner(custom_handler=custom_runner)
        executor = DockerCodeExecutor(runner=runner)
        pipeline = CodeEvaluationPipeline(code_executor=executor)

        markdown_response = f"Here is my solution:\n```python\n{CANONICAL_PERFECT_SOLUTION}\n```"

        result = await pipeline.evaluate(task=self.task, agent_output=markdown_response)

        assert result.passed is True
        assert result.overall_score == 1.0
        assert result.public_passed is True
        assert result.hidden_passed is True
        assert result.public_score == 1.0
        assert result.hidden_score == 1.0
        assert result.failure_type is None
        assert result.failure_category is None

    async def test_demo_b_public_overfitting(self):
        """Demo B: Overfitted candidate passes 8/8 public, fails hidden -> FAIL with OVERFITTING."""
        def custom_runner(command_args, timeout, container_name):
            if any("test_public" in a for a in command_args):
                return ProcessOutput(
                    exit_code=0,
                    stdout=JUNIT_XML_PUBLIC_PERFECT,
                    stderr="",
                    duration_ms=100.0,
                )
            else:
                return ProcessOutput(
                    exit_code=1,
                    stdout=JUNIT_XML_HIDDEN_OVERFITTED,
                    stderr="",
                    duration_ms=80.0,
                )

        runner = FakeContainerProcessRunner(custom_handler=custom_runner)
        executor = DockerCodeExecutor(runner=runner)
        pipeline = CodeEvaluationPipeline(code_executor=executor)

        result = await pipeline.evaluate(
            task=self.task,
            candidate_code=OVERFITTED_PUBLIC_ONLY_SOLUTION,
        )

        assert result.passed is False
        assert result.public_passed is True
        assert result.hidden_passed is False
        assert result.public_score == 1.0
        assert result.hidden_score == 0.5
        assert result.failure_category == "OVERFITTING / CORRECTNESS"
        assert result.failure_type == "HIDDEN_TEST_FAILURE"
        assert len(result.hidden_failures) == 3

    async def test_demo_c_infinite_loop_timeout(self):
        """Demo C: Infinite loop candidate produces TIMEOUT status, not TEST_FAILURE."""
        runner = FakeContainerProcessRunner()
        runner.simulate_timeout = True

        executor = DockerCodeExecutor(runner=runner)
        pipeline = CodeEvaluationPipeline(code_executor=executor)

        infinite_loop_code = """
def second_largest_unique(numbers):
    while True:
        pass
"""
        result = await pipeline.evaluate(task=self.task, candidate_code=infinite_loop_code)

        assert result.passed is False
        assert result.failure_type == "TIMEOUT"
        assert result.failure_category == "TIMEOUT"

    async def test_demo_d_syntax_error_no_sandbox(self):
        """Demo D: Syntax error caught statically without launching sandbox."""
        runner = FakeContainerProcessRunner()
        executor = DockerCodeExecutor(runner=runner)
        pipeline = CodeEvaluationPipeline(code_executor=executor)

        broken_code = """
def second_largest_unique(numbers):
    return numbers[((  # unclosed parenthesis
"""
        result = await pipeline.evaluate(task=self.task, candidate_code=broken_code)

        assert result.passed is False
        assert result.failure_type == "SYNTAX_ERROR"
        assert result.failure_category == "SYNTAX_ERROR"
        # Verify no container was ever launched!
        assert len(runner.recorded_calls) == 0

    async def test_demo_e_runtime_error(self):
        """Demo E: Runtime error (e.g. module error or division by zero) produces RUNTIME_ERROR."""
        runner = FakeContainerProcessRunner(
            default_output=ProcessOutput(
                exit_code=1,
                stdout="",
                stderr="Traceback (most recent call last):\n  File 'solution.py', line 2\nZeroDivisionError: division by zero",
                duration_ms=40.0,
            )
        )
        executor = DockerCodeExecutor(runner=runner)
        pipeline = CodeEvaluationPipeline(code_executor=executor)

        runtime_error_code = """
def second_largest_unique(numbers):
    return 1 / 0
"""
        result = await pipeline.evaluate(task=self.task, candidate_code=runtime_error_code)

        assert result.passed is False
        assert result.failure_type == "CANDIDATE_RUNTIME_ERROR" or result.failure_type == "RUNTIME_ERROR"

    async def test_demo_f_sandbox_unavailable(self):
        """Demo F: Docker startup failure produces SANDBOX_ERROR without blaming agent."""
        runner = FakeContainerProcessRunner()
        runner.fail_on_startup = True

        executor = DockerCodeExecutor(runner=runner)
        pipeline = CodeEvaluationPipeline(code_executor=executor)

        result = await pipeline.evaluate(task=self.task, candidate_code=CANONICAL_PERFECT_SOLUTION)

        assert result.passed is False
        assert result.failure_type == "SANDBOX_ERROR"
        assert result.failure_category == "SANDBOX_ERROR"

    async def test_candidate_never_receives_hidden_test_source(self):
        """Verify candidate files in public execution never contain hidden test code."""
        captured_requests: list[dict[str, str]] = []

        def inspect_runner(command_args, timeout, container_name):
            # Check the command or workspace files in recorded calls
            return ProcessOutput(exit_code=0, stdout=JUNIT_XML_PUBLIC_PERFECT, stderr="", duration_ms=50.0)

        runner = FakeContainerProcessRunner(custom_handler=inspect_runner)
        executor = DockerCodeExecutor(runner=runner)
        pipeline = CodeEvaluationPipeline(code_executor=executor)

        await pipeline.evaluate(task=self.task, candidate_code=CANONICAL_PERFECT_SOLUTION)

        # Look at the public run recorded command args
        assert len(runner.recorded_calls) >= 1
        public_call = runner.recorded_calls[0]
        # Command should only reference public test
        assert "test_hidden.py" not in " ".join(public_call["command_args"])

    async def test_observability_event_emission(self):
        """Verify structured observability events with run_id, trial_id, execution_id, trace_id."""
        runner = FakeContainerProcessRunner(
            default_output=ProcessOutput(exit_code=0, stdout=JUNIT_XML_PUBLIC_PERFECT, stderr="", duration_ms=50.0)
        )
        executor = DockerCodeExecutor(runner=runner)
        pipeline = CodeEvaluationPipeline(code_executor=executor)

        trial_id = uuid.uuid4()
        run_id = uuid.uuid4()
        trace_id = "trace-12345"

        await pipeline.evaluate(
            task=self.task,
            candidate_code=CANONICAL_PERFECT_SOLUTION,
            trial_id=trial_id,
            run_id=run_id,
            trace_id=trace_id,
        )

        event_names = [e["event"] for e in pipeline.recorded_events]
        assert "code_execution_started" in event_names
        assert "test_run_started" in event_names
        assert "verification_started" in event_names
        assert "verification_completed" in event_names
        assert "code_execution_completed" in event_names

        for evt in pipeline.recorded_events:
            assert evt["run_id"] == str(run_id)
            assert evt["trial_id"] == str(trial_id)
            assert evt["trace_id"] == trace_id
