"""Integration and Security Regression tests for Code Evaluation Pipeline (TASK 13).

Covers:
- End-to-end code evaluation lifecycle
- Malicious candidate security regressions (path traversal, secret discovery, OOM, fork bomb, socket access)
- Live Docker execution (conditionally skipped when Docker daemon is not active)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
import pytest

from backend.domain.sandbox import ExecutionRequest, ExecutionStatus
from backend.evaluation.scenarios.second_largest_unique import (
    CANONICAL_PERFECT_SOLUTION,
    OVERFITTED_PUBLIC_ONLY_SOLUTION,
    create_second_largest_unique_task,
)
from backend.infrastructure.sandbox.docker_executor import DockerCodeExecutor
from backend.infrastructure.sandbox.process_runner import (
    DockerCliProcessRunner,
    FakeContainerProcessRunner,
    ProcessOutput,
)
from backend.verification.code_pipeline import CodeEvaluationPipeline


def is_docker_daemon_running() -> bool:
    """Check if the live Docker daemon is running and responsive."""
    if not shutil.which("docker"):
        return False
    try:
        res = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
        )
        return res.returncode == 0
    except Exception:
        return False


DOCKER_AVAILABLE = is_docker_daemon_running()


# ============================================================================
# End-to-End Pipeline Integration Tests
# ============================================================================

@pytest.mark.asyncio
class TestCodeEvaluationIntegration:
    def setup_method(self):
        self.task = create_second_largest_unique_task()

    async def test_end_to_end_perfect_evaluation(self):
        """End-to-end evaluation flow with canonical solution receiving full score."""
        def custom_runner(command_args, timeout, container_name):
            if any("test_public" in a for a in command_args):
                xml = """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="8" errors="0" failures="0" skipped="0" time="0.10">
    <testcase classname="test_public" name="test_1" time="0.01" />
    <testcase classname="test_public" name="test_2" time="0.01" />
    <testcase classname="test_public" name="test_3" time="0.01" />
    <testcase classname="test_public" name="test_4" time="0.01" />
    <testcase classname="test_public" name="test_5" time="0.01" />
    <testcase classname="test_public" name="test_6" time="0.01" />
    <testcase classname="test_public" name="test_7" time="0.01" />
    <testcase classname="test_public" name="test_8" time="0.01" />
</testsuite>"""
                return ProcessOutput(exit_code=0, stdout=xml, stderr="", duration_ms=100.0)
            else:
                xml = """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="6" errors="0" failures="0" skipped="0" time="0.08">
    <testcase classname="test_hidden" name="test_h1" time="0.01" />
    <testcase classname="test_hidden" name="test_h2" time="0.01" />
    <testcase classname="test_hidden" name="test_h3" time="0.01" />
    <testcase classname="test_hidden" name="test_h4" time="0.01" />
    <testcase classname="test_hidden" name="test_h5" time="0.01" />
    <testcase classname="test_hidden" name="test_h6" time="0.01" />
</testsuite>"""
                return ProcessOutput(exit_code=0, stdout=xml, stderr="", duration_ms=80.0)

        runner = FakeContainerProcessRunner(custom_handler=custom_runner)
        executor = DockerCodeExecutor(runner=runner)
        pipeline = CodeEvaluationPipeline(code_executor=executor)

        result = await pipeline.evaluate(
            task=self.task,
            candidate_code=CANONICAL_PERFECT_SOLUTION,
        )

        assert result.passed is True
        assert result.overall_score == 1.0
        assert result.public_passed is True
        assert result.hidden_passed is True
        assert result.public_score == 1.0
        assert result.hidden_score == 1.0
        assert "public_execution" in result.evidence
        assert "hidden_execution" in result.evidence


# ============================================================================
# Security Regression Tests (Section 21)
# ============================================================================

@pytest.mark.asyncio
class TestSecurityRegressions:
    def setup_method(self):
        self.task = create_second_largest_unique_task()

    async def test_malicious_path_traversal_to_hidden_tests(self):
        """Candidate attempts to open hidden test source via path traversal.

        Expected: FileNotFoundError because hidden test source is NEVER present
        in the candidate public execution workspace.
        """
        malicious_code = """
import os

def second_largest_unique(numbers):
    # Attempt to leak hidden test code
    for path in ["test_hidden.py", "../test_hidden.py", "/workspace/test_hidden.py"]:
        if os.path.exists(path):
            raise RuntimeError(f"LEAKED: {path}")
    return 42
"""
        runner = FakeContainerProcessRunner(
            default_output=ProcessOutput(
                exit_code=1,
                stdout="FAILED test_public.py::test_normal_case - assert 42 == 4",
                stderr="",
                duration_ms=40.0,
            )
        )
        executor = DockerCodeExecutor(runner=runner)

        # Ensure candidate files submitted for public execution do NOT include hidden test code
        req = ExecutionRequest(
            candidate_files={"solution.py": malicious_code, "test_public.py": "assert True"},
            command=["pytest", "test_public.py"],
        )

        # Inspect workspace contents created by DockerCodeExecutor
        workspace_dir = executor._create_workspace(req.execution_id, req.candidate_files)
        try:
            workspace_files = os.listdir(workspace_dir)
            assert "solution.py" in workspace_files
            assert "test_public.py" in workspace_files
            assert "test_hidden.py" not in workspace_files
        finally:
            await executor.cleanup(req.execution_id)

    async def test_malicious_read_etc_passwd(self):
        """Candidate attempts to read /etc/passwd or host files.

        Expected: Container executes non-root with dropped capabilities and isolated filesystem.
        """
        runner = FakeContainerProcessRunner()
        executor = DockerCodeExecutor(runner=runner)

        req = ExecutionRequest(
            candidate_files={"solution.py": "open('/etc/passwd').read()"},
            command=["python", "solution.py"],
        )
        await executor.execute(req)

        # Verify command arguments enforce non-root and drop capabilities
        cmd_args = runner.recorded_calls[0]["command_args"]
        assert "--cap-drop" in cmd_args and cmd_args[cmd_args.index("--cap-drop") + 1] == "ALL"
        assert "--user" in cmd_args and cmd_args[cmd_args.index("--user") + 1] == "1000:1000"
        assert "--read-only" in cmd_args

    async def test_malicious_environment_secret_discovery(self):
        """Candidate attempts to inspect os.environ for host secrets (e.g. API keys).

        Expected: Host secrets are never injected; only deterministic env vars exist.
        """
        runner = FakeContainerProcessRunner()
        executor = DockerCodeExecutor(runner=runner)

        os.environ["SECRET_HOST_API_KEY"] = "super_secret_production_key_123"

        req = ExecutionRequest(
            candidate_files={"solution.py": "import os; print(os.environ)"},
            command=["python", "solution.py"],
        )
        await executor.execute(req)

        cmd_args = runner.recorded_calls[0]["command_args"]
        # Ensure SECRET_HOST_API_KEY was not passed to Docker
        assert not any("SECRET_HOST_API_KEY" in arg for arg in cmd_args)
        assert not any("super_secret_production_key_123" in arg for arg in cmd_args)

    async def test_malicious_network_access_blocked(self):
        """Candidate attempts outbound socket network access.

        Expected: --network none is strictly enforced in container flags.
        """
        runner = FakeContainerProcessRunner()
        executor = DockerCodeExecutor(runner=runner)

        req = ExecutionRequest(
            candidate_files={"solution.py": "import urllib.request; urllib.request.urlopen('http://google.com')"},
            command=["python", "solution.py"],
            network_enabled=False,
        )
        await executor.execute(req)

        cmd_args = runner.recorded_calls[0]["command_args"]
        assert "--network" in cmd_args
        assert cmd_args[cmd_args.index("--network") + 1] == "none"

    async def test_malicious_memory_exhaustion(self):
        """Candidate attempts allocating excessive memory.

        Expected: Container memory limit enforced and exit code 137 / RESOURCE_LIMIT_EXCEEDED returned.
        """
        runner = FakeContainerProcessRunner()
        runner.simulate_oom = True

        executor = DockerCodeExecutor(runner=runner)
        pipeline = CodeEvaluationPipeline(code_executor=executor)

        oom_code = "def second_largest_unique(numbers):\n    return [0] * (10**10)\n"
        res = await pipeline.evaluate(task=self.task, candidate_code=oom_code)

        assert res.passed is False
        assert res.failure_type == "RESOURCE_LIMIT_EXCEEDED"

    async def test_malicious_sleep_indefinitely(self):
        """Candidate attempts sleeping indefinitely.

        Expected: Execution terminated at timeout; status TIMEOUT.
        """
        runner = FakeContainerProcessRunner()
        runner.simulate_timeout = True

        executor = DockerCodeExecutor(runner=runner)
        pipeline = CodeEvaluationPipeline(code_executor=executor)

        sleep_code = "import time\ndef second_largest_unique(numbers):\n    time.sleep(3600)\n"
        res = await pipeline.evaluate(task=self.task, candidate_code=sleep_code)

        assert res.passed is False
        assert res.failure_type == "TIMEOUT"

    async def test_malicious_docker_socket_access_prevented(self):
        """Candidate attempts accessing Docker socket inside container.

        Expected: /var/run/docker.sock is NEVER mounted in container arguments.
        """
        runner = FakeContainerProcessRunner()
        executor = DockerCodeExecutor(runner=runner)

        req = ExecutionRequest(
            candidate_files={"solution.py": "import os; os.listdir('/var/run')"},
            command=["python", "solution.py"],
        )
        await executor.execute(req)

        cmd_args = runner.recorded_calls[0]["command_args"]
        assert not any("docker.sock" in arg for arg in cmd_args)


# ============================================================================
# Live Docker Daemon Integration Tests (Conditionally Skipped)
# ============================================================================

@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Live Docker daemon required for this test")
@pytest.mark.asyncio
class TestLiveDockerExecution:
    async def test_live_docker_successful_execution(self):
        """Execute real python command inside live Docker container."""
        executor = DockerCodeExecutor(runner=DockerCliProcessRunner())
        req = ExecutionRequest(
            candidate_files={"hello.py": "print('AgentForge Docker Execution OK')\n"},
            command=["python", "hello.py"],
            timeout_seconds=15.0,
        )
        result = await executor.execute(req)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.exit_code == 0
        assert "AgentForge Docker Execution OK" in result.stdout
