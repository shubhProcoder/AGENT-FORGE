"""Docker-based isolated code execution service.

Implements CodeExecutionService using strict containerization policies:
- non-root execution
- read-only root filesystem
- network disabled
- all capabilities dropped
- no new privileges
- explicit CPU, memory, and pids limits
- ephemeral temporary workspace
- automatic cleanup
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any

from backend.domain.artifacts import Artifact
from backend.domain.sandbox import (
    CodeExecutionService,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)
from backend.infrastructure.sandbox.process_runner import (
    ContainerProcessRunner,
    DockerCliProcessRunner,
    ProcessOutput,
)

logger = logging.getLogger(__name__)

# Safe relative filename pattern (letters, digits, dots, underscores, dashes)
SAFE_FILENAME_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


class SecurityViolationError(Exception):
    """Raised when a candidate file or request violates security sandbox policies."""


class DockerCodeExecutor(CodeExecutionService):
    """Executes untrusted code inside ephemeral, hardened Docker containers."""

    def __init__(
        self,
        runner: ContainerProcessRunner | None = None,
        base_image: str = "python:3.11-slim",
        workspace_base_dir: str | None = None,
    ) -> None:
        self.runner = runner or DockerCliProcessRunner()
        self.base_image = base_image
        self.workspace_base_dir = workspace_base_dir or os.path.join(
            tempfile.gettempdir(), "agentforge"
        )
        self._active_executions: dict[str, dict[str, Any]] = {}

    def _sanitize_and_validate_path(self, relative_path: str) -> str:
        """Validate that the candidate filepath is safe and strictly relative."""
        # Reject empty or non-string paths
        if not relative_path or not isinstance(relative_path, str):
            raise SecurityViolationError("Filename cannot be empty")

        # Reject absolute paths
        if relative_path.startswith("/") or relative_path.startswith("\\"):
            raise SecurityViolationError(f"Absolute paths not permitted: {relative_path}")

        # Reject path traversal
        normalized = os.path.normpath(relative_path)
        parts = normalized.split(os.sep)
        if ".." in parts or normalized.startswith(".."):
            raise SecurityViolationError(f"Path traversal not permitted: {relative_path}")

        # Validate filename format
        for part in parts:
            if not SAFE_FILENAME_REGEX.match(part):
                raise SecurityViolationError(
                    f"Filename contains illegal characters: {part} (in {relative_path})"
                )

        return normalized

    def _create_workspace(self, execution_id: str, files: dict[str, str]) -> str:
        """Create a dedicated, isolated temporary workspace on the host."""
        workspace_dir = os.path.join(self.workspace_base_dir, execution_id)
        os.makedirs(workspace_dir, exist_ok=True, mode=0o700)

        for rel_path, content in files.items():
            safe_rel = self._sanitize_and_validate_path(rel_path)
            full_path = os.path.join(workspace_dir, safe_rel)

            # Ensure subdirectories exist inside workspace
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Write file content
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

        return workspace_dir

    def _build_docker_args(
        self,
        request: ExecutionRequest,
        workspace_dir: str,
        container_name: str,
    ) -> list[str]:
        """Formulate hardened Docker run command arguments."""
        args = [
            "run",
            "--name",
            container_name,
            "--rm",
        ]

        # 1. Network policy
        if not request.network_enabled:
            args.extend(["--network", "none"])

        # 2. Security isolation
        args.extend([
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "1000:1000",
        ])

        # 3. Resource limits
        args.extend([
            f"--cpus={request.cpu_limit}",
            f"--memory={request.memory_limit_mb}m",
            f"--memory-swap={request.memory_limit_mb}m",
            f"--pids-limit={request.pids_limit}",
        ])

        # 4. Ephemeral /tmp tmpfs
        args.extend([
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
        ])

        # 5. Deterministic environment variables
        deterministic_env = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
            **request.environment_variables,
        }
        for k, v in deterministic_env.items():
            args.extend(["-e", f"{k}={v}"])

        # 6. Workspace volume mounting
        args.extend([
            "-v",
            f"{workspace_dir}:/workspace:rw",
            "-w",
            request.working_directory,
        ])

        # 7. Image and internal command
        args.append(self.base_image)
        args.extend(request.command)

        return args

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute untrusted code inside an ephemeral, hardened Docker container."""
        execution_id = request.execution_id
        container_name = f"agentforge_{execution_id}"
        workspace_dir = ""

        logger.info(f"Starting sandboxed execution {execution_id}")

        try:
            # 1. Validate files & create workspace
            workspace_dir = self._create_workspace(execution_id, request.candidate_files)
            self._active_executions[execution_id] = {
                "container_name": container_name,
                "workspace_dir": workspace_dir,
                "start_time": time.perf_counter(),
            }

            # 2. Construct security-hardened Docker arguments
            docker_args = self._build_docker_args(request, workspace_dir, container_name)

            # 3. Run container via runner
            output = await self.runner.run_container(
                docker_args,
                timeout_seconds=request.timeout_seconds,
                container_name=container_name,
            )

            # 4. Truncate outputs if exceeding bounds
            stdout = output.stdout
            stderr = output.stderr
            truncated = False

            if len(stdout.encode("utf-8")) > request.output_limit_bytes:
                stdout = stdout[: request.output_limit_bytes] + "\n... [TRUNCATED]"
                truncated = True

            if len(stderr.encode("utf-8")) > request.output_limit_bytes:
                stderr = stderr[: request.output_limit_bytes] + "\n... [TRUNCATED]"
                truncated = True

            # 5. Classify execution status
            status = self._classify_status(output, stdout, stderr)

            # 6. Discover artifacts in workspace (e.g. junit.xml)
            artifact_refs = self._collect_artifacts(workspace_dir, execution_id)

            return ExecutionResult(
                execution_id=execution_id,
                status=status,
                exit_code=output.exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_ms=output.duration_ms,
                timed_out=output.timed_out,
                resource_limit_exceeded=output.resource_limit_exceeded,
                output_truncated=truncated,
                artifact_references=artifact_refs,
                failure_reason=output.error_message or (stderr if status != ExecutionStatus.SUCCESS else None),
                container_id=container_name,
                evidence_metadata={
                    "command": request.command,
                    "exit_code": output.exit_code,
                    "timed_out": output.timed_out,
                    "resource_limit_exceeded": output.resource_limit_exceeded,
                    "duration_ms": output.duration_ms,
                },
            )

        except SecurityViolationError as e:
            logger.warning(f"Security violation in execution {execution_id}: {e}")
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.SANDBOX_ERROR,
                exit_code=1,
                stdout="",
                stderr=f"Security Violation: {e}",
                duration_ms=0.0,
                failure_reason=str(e),
            )
        except Exception as e:
            logger.error(f"Execution failed for {execution_id}: {e}")
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.SANDBOX_ERROR,
                exit_code=125,
                stdout="",
                stderr=str(e),
                duration_ms=0.0,
                failure_reason=f"Sandbox error: {e}",
            )
        finally:
            # Automatic cleanup of workspace and execution record
            await self.cleanup(execution_id)

    def _classify_status(
        self,
        output: ProcessOutput,
        stdout: str,
        stderr: str,
    ) -> ExecutionStatus:
        """Classify container execution result into fine-grained ExecutionStatus."""
        if output.timed_out:
            return ExecutionStatus.TIMEOUT

        if output.resource_limit_exceeded or output.exit_code == 137:
            return ExecutionStatus.RESOURCE_LIMIT_EXCEEDED

        # Docker startup or daemon failure (exit code 125, 127)
        if output.exit_code in (125, 127) or (output.error_message and "daemon" in output.error_message.lower()):
            return ExecutionStatus.SANDBOX_ERROR

        if output.exit_code == 0:
            return ExecutionStatus.SUCCESS

        combined_output = (stdout + "\n" + stderr).lower()

        # Syntax error detection
        if "syntaxerror" in combined_output or "indentationerror" in combined_output:
            return ExecutionStatus.SYNTAX_ERROR

        # Test failure vs runtime crash
        if "failed" in combined_output or "assert" in combined_output or "failure" in combined_output:
            return ExecutionStatus.TEST_FAILURE

        return ExecutionStatus.RUNTIME_ERROR

    def _collect_artifacts(self, workspace_dir: str, execution_id: str) -> list[str]:
        """Discover generated artifacts (like junit.xml) in the workspace before cleanup."""
        artifacts: list[str] = []
        if not workspace_dir or not os.path.exists(workspace_dir):
            return artifacts

        try:
            for root, _, files in os.walk(workspace_dir):
                for filename in files:
                    # Collect test reports and diagnostic logs
                    if filename.endswith(".xml") or filename.endswith(".log") or filename.endswith(".json"):
                        rel_path = os.path.relpath(os.path.join(root, filename), workspace_dir)
                        artifacts.append(rel_path)
        except Exception as e:
            logger.warning(f"Error scanning artifacts in {workspace_dir}: {e}")

        return artifacts

    async def cancel(self, execution_id: str) -> bool:
        """Cancel an in-flight execution and terminate its container."""
        logger.info(f"Cancelling execution {execution_id}")
        exec_info = self._active_executions.get(execution_id)
        if not exec_info:
            return False

        container_name = exec_info["container_name"]
        await self.runner.stop_container(container_name)
        await self.cleanup(execution_id)
        return True

    async def cleanup(self, execution_id: str) -> None:
        """Clean up the host workspace and active execution tracking."""
        exec_info = self._active_executions.pop(execution_id, None)
        workspace_dir = exec_info["workspace_dir"] if exec_info else os.path.join(self.workspace_base_dir, execution_id)

        if os.path.exists(workspace_dir):
            try:
                shutil.rmtree(workspace_dir, ignore_errors=True)
                logger.debug(f"Cleaned up workspace {workspace_dir}")
            except Exception as e:
                logger.warning(f"Failed to delete workspace {workspace_dir}: {e}")
