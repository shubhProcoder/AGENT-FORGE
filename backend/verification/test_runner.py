"""Test Runner Interface and Pytest Sandbox Adapter.

Abstracts test suite execution and uses CodeExecutionService to execute tests
inside isolated Docker sandboxes rather than directly on the FastAPI host.
Parses canonical JUnit XML reports to extract structured, machine-readable results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Protocol
import xml.etree.ElementTree as ET

from backend.domain.sandbox import (
    CodeExecutionService,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class TestCaseResult:
    """Detailed result of an individual test case in the test suite."""

    __test__ = False

    name: str
    classname: str
    status: str  # "passed", "failed", "error", "skipped"
    duration_seconds: float = 0.0
    failure_message: str | None = None
    traceback: str | None = None


@dataclass
class TestRunResult:
    """Structured, machine-readable outcome of a test suite execution."""

    __test__ = False

    passed: bool
    exit_code: int
    output: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    total_tests: int = 0
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    duration_seconds: float = 0.0
    test_cases: list[TestCaseResult] = field(default_factory=list)
    failure_category: str | None = None  # TEST_RUNNER_ERROR, CANDIDATE_RUNTIME_ERROR, TEST_FAILURE, TIMEOUT, RESOURCE_LIMIT_EXCEEDED, SANDBOX_ERROR
    junit_xml: str | None = None
    stdout: str = ""
    stderr: str = ""


class TestRunner(Protocol):
    """Protocol for test runners."""

    async def run(self, context: Any) -> TestRunResult:
        ...


class PytestAdapter:
    """An implementation of TestRunner that executes pytest via CodeExecutionService.

    Never executes pytest directly in the host process. The canonical result is
    parsed from JUnit XML output; terminal stdout/stderr is preserved as evidence.
    """

    def __init__(
        self,
        code_executor: CodeExecutionService,
        test_filename: str = "test_suite.py",
        solution_filename: str = "solution.py",
        junit_filename: str = "junit.xml",
    ) -> None:
        self.code_executor = code_executor
        self.test_filename = test_filename
        self.solution_filename = solution_filename
        self.junit_filename = junit_filename

    async def run_tests(
        self,
        candidate_code: str,
        test_code: str,
        test_filename: str | None = None,
        timeout_seconds: float = 10.0,
        extra_files: dict[str, str] | None = None,
    ) -> TestRunResult:
        """Run candidate solution against test code in an isolated sandbox."""
        actual_test_filename = test_filename or self.test_filename
        files = {
            self.solution_filename: candidate_code,
            actual_test_filename: test_code,
            **(extra_files or {}),
        }

        # Command constructed internally by AgentForge
        command = [
            "pytest",
            actual_test_filename,
            f"--junitxml={self.junit_filename}",
            "-q",
            "--tb=short",
        ]

        request = ExecutionRequest(
            candidate_files=files,
            command=command,
            timeout_seconds=timeout_seconds,
        )

        exec_result: ExecutionResult = await self.code_executor.execute(request)
        return self._parse_execution_result(exec_result)

    def _parse_execution_result(self, exec_result: ExecutionResult) -> TestRunResult:
        """Parse execution result and JUnit XML into a structured TestRunResult."""
        combined_output = (exec_result.stdout + "\n" + exec_result.stderr).strip()

        # Handle infrastructure & execution level errors
        if exec_result.status == ExecutionStatus.TIMEOUT or exec_result.timed_out:
            return TestRunResult(
                passed=False,
                exit_code=exec_result.exit_code,
                output=combined_output,
                failure_category="TIMEOUT",
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
                metrics={"duration_seconds": exec_result.duration_ms / 1000.0},
            )

        if exec_result.status == ExecutionStatus.RESOURCE_LIMIT_EXCEEDED or exec_result.resource_limit_exceeded:
            return TestRunResult(
                passed=False,
                exit_code=exec_result.exit_code,
                output=combined_output,
                failure_category="RESOURCE_LIMIT_EXCEEDED",
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
                metrics={"duration_seconds": exec_result.duration_ms / 1000.0},
            )

        if exec_result.status == ExecutionStatus.SANDBOX_ERROR:
            return TestRunResult(
                passed=False,
                exit_code=exec_result.exit_code,
                output=combined_output,
                failure_category="SANDBOX_ERROR",
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
                metrics={"duration_seconds": exec_result.duration_ms / 1000.0},
            )

        # Check if junit.xml was captured in stdout or files
        # Often junit.xml is read or embedded in artifacts or can be parsed from stdout/artifacts
        xml_content = self._extract_junit_xml(exec_result)
        if xml_content:
            return self._parse_junit_xml(xml_content, exec_result)

        # Fallback when no junit.xml was produced (e.g. syntax error or early collection failure)
        failure_category = "TEST_RUNNER_ERROR"
        lowered = combined_output.lower()
        if "syntaxerror" in lowered or "indentationerror" in lowered:
            failure_category = "TEST_RUNNER_ERROR"
        elif "traceback" in lowered or "importerror" in lowered or "modulenotfounderror" in lowered:
            failure_category = "CANDIDATE_RUNTIME_ERROR"
        elif exec_result.exit_code == 0:
            failure_category = None

        passed = exec_result.exit_code == 0
        return TestRunResult(
            passed=passed,
            exit_code=exec_result.exit_code,
            output=combined_output,
            failure_category=failure_category,
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
            metrics={"duration_seconds": exec_result.duration_ms / 1000.0},
        )

    def _extract_junit_xml(self, exec_result: ExecutionResult) -> str | None:
        """Check if JUnit XML content is present in stdout or artifact references."""
        # Check if stdout contains full XML declaration
        if "<?xml" in exec_result.stdout and "</testsuite>" in exec_result.stdout:
            start = exec_result.stdout.find("<?xml")
            end = exec_result.stdout.find("</testsuite>") + len("</testsuite>")
            return exec_result.stdout[start:end]
        if "<testsuite" in exec_result.stdout and "</testsuite>" in exec_result.stdout:
            start = exec_result.stdout.find("<testsuite")
            end = exec_result.stdout.find("</testsuite>") + len("</testsuite>")
            return exec_result.stdout[start:end]
        return None

    def _parse_junit_xml(self, xml_content: str, exec_result: ExecutionResult) -> TestRunResult:
        """Parse JUnit XML string into normalized TestRunResult."""
        try:
            root = ET.fromstring(xml_content)
            # Root may be <testsuites> or <testsuite>
            suite = root if root.tag == "testsuite" else root.find("testsuite")
            if suite is None:
                suite = root

            total = int(suite.attrib.get("tests", 0))
            errors = int(suite.attrib.get("errors", 0))
            failures = int(suite.attrib.get("failures", 0))
            skipped = int(suite.attrib.get("skipped", 0))
            duration = float(suite.attrib.get("time", exec_result.duration_ms / 1000.0))
            passed = total - (errors + failures + skipped)

            test_cases: list[TestCaseResult] = []
            for tc in suite.findall("testcase"):
                name = tc.attrib.get("name", "unknown")
                classname = tc.attrib.get("classname", "")
                tc_time = float(tc.attrib.get("time", 0.0))

                failure_elem = tc.find("failure")
                error_elem = tc.find("error")
                skipped_elem = tc.find("skipped")

                if failure_elem is not None:
                    status = "failed"
                    msg = failure_elem.attrib.get("message", "")
                    tb = failure_elem.text
                elif error_elem is not None:
                    status = "error"
                    msg = error_elem.attrib.get("message", "")
                    tb = error_elem.text
                elif skipped_elem is not None:
                    status = "skipped"
                    msg = skipped_elem.attrib.get("message", "")
                    tb = None
                else:
                    status = "passed"
                    msg = None
                    tb = None

                test_cases.append(
                    TestCaseResult(
                        name=name,
                        classname=classname,
                        status=status,
                        duration_seconds=tc_time,
                        failure_message=msg,
                        traceback=tb,
                    )
                )

            is_passed = total > 0 and failures == 0 and errors == 0
            failure_category = None
            if not is_passed:
                if errors > 0:
                    failure_category = "CANDIDATE_RUNTIME_ERROR"
                else:
                    failure_category = "TEST_FAILURE"

            return TestRunResult(
                passed=is_passed,
                exit_code=exec_result.exit_code,
                output=(exec_result.stdout + "\n" + exec_result.stderr).strip(),
                total_tests=total,
                passed_count=passed,
                failed_count=failures,
                skipped_count=skipped,
                error_count=errors,
                duration_seconds=duration,
                test_cases=test_cases,
                failure_category=failure_category,
                junit_xml=xml_content,
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
                metrics={
                    "total_tests": total,
                    "passed_tests": passed,
                    "failed_tests": failures,
                    "duration_seconds": duration,
                },
            )

        except Exception as e:
            logger.warning(f"Failed to parse JUnit XML: {e}")
            return TestRunResult(
                passed=exec_result.exit_code == 0,
                exit_code=exec_result.exit_code,
                output=(exec_result.stdout + "\n" + exec_result.stderr).strip(),
                failure_category="TEST_RUNNER_ERROR",
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
                metrics={"duration_seconds": exec_result.duration_ms / 1000.0},
            )

    async def run(self, context: Any) -> TestRunResult:
        """Backward compatible protocol method."""
        if hasattr(context, "candidate_code") and hasattr(context, "test_code"):
            return await self.run_tests(context.candidate_code, context.test_code)
        # Default fallback
        return TestRunResult(passed=False, exit_code=-1, output="Invalid context provided")
