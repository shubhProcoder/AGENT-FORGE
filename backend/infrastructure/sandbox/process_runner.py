"""Process runners for Docker container execution.

Separates raw process invocation from the execution service, allowing
hermetic unit testing without requiring a live Docker daemon.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ProcessOutput:
    """Raw output from running a container process."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    resource_limit_exceeded: bool = False
    container_id: str | None = None
    error_message: str | None = None


class ContainerProcessRunner(Protocol):
    """Protocol for launching and managing container processes."""

    async def run_container(
        self,
        command_args: list[str],
        timeout_seconds: float,
        container_name: str | None = None,
    ) -> ProcessOutput:
        """Run a container with the given arguments and timeout."""
        ...

    async def stop_container(self, container_name: str) -> None:
        """Force stop / kill a running container."""
        ...


class DockerCliProcessRunner:
    """Real implementation executing the Docker CLI."""

    def __init__(self, docker_binary: str = "docker") -> None:
        self.docker_binary = docker_binary

    async def run_container(
        self,
        command_args: list[str],
        timeout_seconds: float,
        container_name: str | None = None,
    ) -> ProcessOutput:
        """Invoke docker CLI as an async subprocess."""
        cmd = [self.docker_binary] + command_args
        start_time = time.perf_counter()

        logger.debug(f"Launching container command: {' '.join(cmd[:10])}...")

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_seconds,
                )
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                exit_code = proc.returncode if proc.returncode is not None else 0

                # Exit code 137 commonly indicates SIGKILL / OOM in Docker
                resource_limit = exit_code == 137 or "oom" in stderr.lower()

                return ProcessOutput(
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    duration_ms=duration_ms,
                    timed_out=False,
                    resource_limit_exceeded=resource_limit,
                    container_id=container_name,
                )

            except asyncio.TimeoutError:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                logger.warning(f"Container {container_name} timed out after {timeout_seconds}s")
                if container_name:
                    await self.stop_container(container_name)
                if proc:
                    try:
                        proc.kill()
                        await proc.wait()
                    except ProcessLookupError:
                        pass
                return ProcessOutput(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Execution timed out after {timeout_seconds} seconds",
                    duration_ms=duration_ms,
                    timed_out=True,
                    container_id=container_name,
                )

        except FileNotFoundError:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return ProcessOutput(
                exit_code=127,
                stdout="",
                stderr=f"Docker binary '{self.docker_binary}' not found on host system",
                duration_ms=duration_ms,
                error_message="Docker CLI binary not found",
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Error launching docker container: {e}")
            return ProcessOutput(
                exit_code=125,
                stdout="",
                stderr=str(e),
                duration_ms=duration_ms,
                error_message=f"Container execution error: {e}",
            )

    async def stop_container(self, container_name: str) -> None:
        """Force kill and remove container by name."""
        try:
            kill_proc = await asyncio.create_subprocess_exec(
                self.docker_binary,
                "rm",
                "-f",
                container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(kill_proc.communicate(), timeout=5.0)
        except Exception as e:
            logger.warning(f"Failed to stop container {container_name}: {e}")


class FakeContainerProcessRunner:
    """Test double for hermetic unit testing without an active Docker daemon."""

    def __init__(
        self,
        default_output: ProcessOutput | None = None,
        custom_handler: Callable[[list[str], float, str | None], ProcessOutput] | None = None,
    ) -> None:
        self.default_output = default_output or ProcessOutput(
            exit_code=0,
            stdout="Fake container stdout",
            stderr="",
            duration_ms=45.0,
        )
        self.custom_handler = custom_handler
        self.recorded_calls: list[dict[str, Any]] = []
        self.stopped_containers: list[str] = []
        self.fail_on_startup: bool = False
        self.simulate_timeout: bool = False
        self.simulate_oom: bool = False

    async def run_container(
        self,
        command_args: list[str],
        timeout_seconds: float,
        container_name: str | None = None,
    ) -> ProcessOutput:
        """Simulate container execution and record invocation parameters."""
        call_record = {
            "command_args": list(command_args),
            "timeout_seconds": timeout_seconds,
            "container_name": container_name,
        }
        self.recorded_calls.append(call_record)

        if self.fail_on_startup:
            return ProcessOutput(
                exit_code=125,
                stdout="",
                stderr="docker: Cannot connect to the Docker daemon. Is the docker daemon running?",
                duration_ms=5.0,
                error_message="Daemon unavailable",
            )

        if self.simulate_timeout:
            return ProcessOutput(
                exit_code=-1,
                stdout="",
                stderr=f"Execution timed out after {timeout_seconds} seconds",
                duration_ms=timeout_seconds * 1000.0,
                timed_out=True,
                container_id=container_name,
            )

        if self.simulate_oom:
            return ProcessOutput(
                exit_code=137,
                stdout="",
                stderr="Killed (Out of memory)",
                duration_ms=120.0,
                resource_limit_exceeded=True,
                container_id=container_name,
            )

        if self.custom_handler:
            return self.custom_handler(command_args, timeout_seconds, container_name)

        return self.default_output

    async def stop_container(self, container_name: str) -> None:
        """Record container stop invocation."""
        self.stopped_containers.append(container_name)
