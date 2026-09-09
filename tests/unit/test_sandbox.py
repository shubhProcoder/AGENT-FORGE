"""Unit tests for CodeExecutionService and DockerCodeExecutor."""

from __future__ import annotations

import os
import pytest

from backend.domain.sandbox import (
    CodeExecutionService,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)
from backend.infrastructure.sandbox.docker_executor import (
    DockerCodeExecutor,
    SecurityViolationError,
)
from backend.infrastructure.sandbox.process_runner import (
    FakeContainerProcessRunner,
    ProcessOutput,
)


@pytest.mark.asyncio
class TestDockerCodeExecutor:
    async def test_protocol_conformance(self):
        """Verify DockerCodeExecutor conforms to CodeExecutionService protocol."""
        executor = DockerCodeExecutor(runner=FakeContainerProcessRunner())
        assert isinstance(executor, CodeExecutionService)

    async def test_docker_command_construction(self):
        """Verify that all required security hardening and isolation flags are present."""
        runner = FakeContainerProcessRunner()
        executor = DockerCodeExecutor(runner=runner, base_image="python:3.11-slim")

        req = ExecutionRequest(
            candidate_files={"solution.py": "def solution(): pass\n"},
            command=["pytest", "test_public.py"],
            timeout_seconds=5.0,
            cpu_limit=0.5,
            memory_limit_mb=256,
            pids_limit=32,
            network_enabled=False,
            environment_variables={"CUSTOM_ENV": "123"},
        )

        res = await executor.execute(req)
        assert res.status == ExecutionStatus.SUCCESS
        assert len(runner.recorded_calls) == 1

        cmd_args = runner.recorded_calls[0]["command_args"]

        # 1. Essential flags
        assert "run" in cmd_args
        assert "--rm" in cmd_args
        assert "--network" in cmd_args and cmd_args[cmd_args.index("--network") + 1] == "none"
        assert "--read-only" in cmd_args
        assert "--cap-drop" in cmd_args and cmd_args[cmd_args.index("--cap-drop") + 1] == "ALL"
        assert "--security-opt" in cmd_args and cmd_args[cmd_args.index("--security-opt") + 1] == "no-new-privileges"
        assert "--user" in cmd_args and cmd_args[cmd_args.index("--user") + 1] == "1000:1000"

        # 2. Resource governance
        assert "--cpus=0.5" in cmd_args
        assert "--memory=256m" in cmd_args
        assert "--memory-swap=256m" in cmd_args
        assert "--pids-limit=32" in cmd_args

        # 3. Tmpfs & workspace
        assert "--tmpfs" in cmd_args
        assert "-w" in cmd_args and cmd_args[cmd_args.index("-w") + 1] == "/workspace"

        # 4. Deterministic env vars
        assert "-e" in cmd_args
        assert "PYTHONUNBUFFERED=1" in cmd_args
        assert "PYTHONDONTWRITEBYTECODE=1" in cmd_args
        assert "CUSTOM_ENV=123" in cmd_args

        # 5. Base image and command
        assert "python:3.11-slim" in cmd_args
        assert "pytest" in cmd_args
        assert "test_public.py" in cmd_args

    async def test_successful_execution(self):
        """Test successful execution handling."""
        runner = FakeContainerProcessRunner(
            default_output=ProcessOutput(
                exit_code=0,
                stdout="All tests passed\n",
                stderr="",
                duration_ms=50.0,
            )
        )
        executor = DockerCodeExecutor(runner=runner)
        req = ExecutionRequest(
            candidate_files={"main.py": "print('hello')"},
            command=["python", "main.py"],
        )
        result = await executor.execute(req)

        assert result.status == ExecutionStatus.SUCCESS
        assert result.exit_code == 0
        assert "All tests passed" in result.stdout
        assert not result.timed_out
        assert not result.resource_limit_exceeded

    async def test_syntax_error_classification(self):
        """Test that syntax errors in container output are classified as SYNTAX_ERROR."""
        runner = FakeContainerProcessRunner(
            default_output=ProcessOutput(
                exit_code=1,
                stdout="",
                stderr='  File "solution.py", line 1\n    def foo(\n           ^\nSyntaxError: was never closed',
                duration_ms=25.0,
            )
        )
        executor = DockerCodeExecutor(runner=runner)
        req = ExecutionRequest(
            candidate_files={"solution.py": "def foo("},
            command=["python", "solution.py"],
        )
        result = await executor.execute(req)

        assert result.status == ExecutionStatus.SYNTAX_ERROR
        assert result.exit_code == 1
        assert "SyntaxError" in result.stderr

    async def test_runtime_error_classification(self):
        """Test that unhandled exceptions are classified as RUNTIME_ERROR."""
        runner = FakeContainerProcessRunner(
            default_output=ProcessOutput(
                exit_code=1,
                stdout="",
                stderr="Traceback (most recent call last):\n  File 'main.py', line 1, in <module>\nZeroDivisionError: division by zero",
                duration_ms=30.0,
            )
        )
        executor = DockerCodeExecutor(runner=runner)
        req = ExecutionRequest(
            candidate_files={"main.py": "1 / 0"},
            command=["python", "main.py"],
        )
        result = await executor.execute(req)

        assert result.status == ExecutionStatus.RUNTIME_ERROR
        assert result.exit_code == 1
        assert "ZeroDivisionError" in result.stderr

    async def test_timeout_classification(self):
        """Test that container timeouts are classified as TIMEOUT."""
        runner = FakeContainerProcessRunner()
        runner.simulate_timeout = True

        executor = DockerCodeExecutor(runner=runner)
        req = ExecutionRequest(
            candidate_files={"loop.py": "while True: pass"},
            command=["python", "loop.py"],
            timeout_seconds=2.0,
        )
        result = await executor.execute(req)

        assert result.status == ExecutionStatus.TIMEOUT
        assert result.timed_out is True
        assert result.exit_code == -1
        assert "timed out" in result.stderr

    async def test_resource_limit_classification(self):
        """Test that exit code 137 (OOM / SIGKILL) is classified as RESOURCE_LIMIT_EXCEEDED."""
        runner = FakeContainerProcessRunner()
        runner.simulate_oom = True

        executor = DockerCodeExecutor(runner=runner)
        req = ExecutionRequest(
            candidate_files={"bomb.py": "a = [0] * 10**9"},
            command=["python", "bomb.py"],
        )
        result = await executor.execute(req)

        assert result.status == ExecutionStatus.RESOURCE_LIMIT_EXCEEDED
        assert result.resource_limit_exceeded is True
        assert result.exit_code == 137

    async def test_sandbox_startup_failure(self):
        """Test that failure to start container (e.g. daemon down) is classified as SANDBOX_ERROR."""
        runner = FakeContainerProcessRunner()
        runner.fail_on_startup = True

        executor = DockerCodeExecutor(runner=runner)
        req = ExecutionRequest(
            candidate_files={"main.py": "print(1)"},
            command=["python", "main.py"],
        )
        result = await executor.execute(req)

        assert result.status == ExecutionStatus.SANDBOX_ERROR
        assert result.exit_code == 125
        assert "Cannot connect to the Docker daemon" in result.stderr

    async def test_cleanup_after_execution(self):
        """Verify that the host workspace directory is cleaned up after execution."""
        runner = FakeContainerProcessRunner()
        executor = DockerCodeExecutor(runner=runner)
        req = ExecutionRequest(
            candidate_files={"file.py": "x = 1"},
            command=["python", "file.py"],
        )
        workspace_dir = os.path.join(executor.workspace_base_dir, req.execution_id)

        result = await executor.execute(req)
        assert result.status == ExecutionStatus.SUCCESS
        assert not os.path.exists(workspace_dir)

    async def test_cleanup_after_failure(self):
        """Verify that the host workspace directory is cleaned up even if execution raises an exception."""
        runner = FakeContainerProcessRunner()
        runner.simulate_timeout = True

        executor = DockerCodeExecutor(runner=runner)
        req = ExecutionRequest(
            candidate_files={"bad.py": "pass"},
            command=["python", "bad.py"],
        )
        workspace_dir = os.path.join(executor.workspace_base_dir, req.execution_id)

        await executor.execute(req)
        assert not os.path.exists(workspace_dir)

    async def test_security_rejects_path_traversal(self):
        """Verify that filenames attempting path traversal are rejected."""
        executor = DockerCodeExecutor(runner=FakeContainerProcessRunner())

        # Traversal with ..
        req1 = ExecutionRequest(
            candidate_files={"../../etc/passwd": "malicious"},
            command=["cat", "file"],
        )
        res1 = await executor.execute(req1)
        assert res1.status == ExecutionStatus.SANDBOX_ERROR
        assert "Security Violation" in res1.stderr

        # Absolute path
        req2 = ExecutionRequest(
            candidate_files={"/root/secret.py": "malicious"},
            command=["cat", "file"],
        )
        res2 = await executor.execute(req2)
        assert res2.status == ExecutionStatus.SANDBOX_ERROR
        assert "Security Violation" in res2.stderr

    async def test_bounded_output_truncation(self):
        """Verify that excessive stdout is truncated and flagged."""
        runner = FakeContainerProcessRunner(
            default_output=ProcessOutput(
                exit_code=0,
                stdout="A" * 1000,
                stderr="",
                duration_ms=10.0,
            )
        )
        executor = DockerCodeExecutor(runner=runner)
        req = ExecutionRequest(
            candidate_files={"flood.py": "print('A'*1000)"},
            command=["python", "flood.py"],
            output_limit_bytes=100,  # limit to 100 bytes
        )
        res = await executor.execute(req)

        assert res.output_truncated is True
        assert len(res.stdout) < 200
        assert "[TRUNCATED]" in res.stdout

    async def test_cancel_execution(self):
        """Verify cancelling in-flight execution terminates container and cleans up."""
        runner = FakeContainerProcessRunner()
        executor = DockerCodeExecutor(runner=runner)

        exec_id = "test-cancel-id"
        executor._active_executions[exec_id] = {
            "container_name": f"agentforge_{exec_id}",
            "workspace_dir": os.path.join(executor.workspace_base_dir, exec_id),
            "start_time": 0.0,
        }
        os.makedirs(executor._active_executions[exec_id]["workspace_dir"], exist_ok=True)

        cancelled = await executor.cancel(exec_id)
        assert cancelled is True
        assert f"agentforge_{exec_id}" in runner.stopped_containers
        assert not os.path.exists(os.path.join(executor.workspace_base_dir, exec_id))
