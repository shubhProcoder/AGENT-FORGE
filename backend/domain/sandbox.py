"""Domain models and protocol for isolated code execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import Any, Protocol, runtime_checkable


class ExecutionStatus(str, Enum):
    """Detailed execution status states for the Code Sandbox."""

    SUCCESS = "SUCCESS"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TEST_FAILURE = "TEST_FAILURE"
    TIMEOUT = "TIMEOUT"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    SANDBOX_ERROR = "SANDBOX_ERROR"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class ResourceProfile:
    """Resource constraints for sandboxed execution."""

    timeout_seconds: float = 10.0
    cpu_limit: float = 1.0
    memory_limit_mb: int = 512
    pids_limit: int = 64
    network_enabled: bool = False
    output_limit_bytes: int = 65536  # 64 KB
    artifact_limit_bytes: int = 1048576  # 1 MB


@dataclass
class ExecutionRequest:
    """Approved request specification for sandboxed execution.

    Command is constructed internally by AgentForge and never accepted directly
    from untrusted LLM input.
    """

    candidate_files: dict[str, str]  # relative_path -> content
    command: list[str]  # e.g. ["pytest", "test_public.py", "--junitxml=junit.xml"]
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    working_directory: str = "/workspace"
    timeout_seconds: float = 10.0
    cpu_limit: float = 1.0
    memory_limit_mb: int = 512
    pids_limit: int = 64
    network_enabled: bool = False
    environment_variables: dict[str, str] = field(default_factory=dict)
    output_limit_bytes: int = 65536
    artifact_limit_bytes: int = 1048576


@dataclass
class ExecutionResult:
    """Complete, structured outcome of a sandboxed execution."""

    execution_id: str
    status: ExecutionStatus
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    resource_limit_exceeded: bool = False
    output_truncated: bool = False
    artifact_references: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    container_id: str | None = None
    evidence_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionSpec:
    """Internal execution specification defining how a candidate task is run."""

    language: str = "python"
    entrypoint: str = "solution.py"
    test_runner: str = "pytest"
    test_suite_id: str = ""
    resource_profile: ResourceProfile = field(default_factory=ResourceProfile)


@runtime_checkable
class CodeExecutionService(Protocol):
    """Protocol for isolated execution engines."""

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute the request inside an isolated sandbox."""
        ...

    async def cancel(self, execution_id: str) -> bool:
        """Cancel an in-flight execution."""
        ...

    async def cleanup(self, execution_id: str) -> None:
        """Clean up resources associated with an execution."""
        ...
