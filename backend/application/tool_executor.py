"""Tool execution abstraction for Agent Runtime.

Decouples Agent Runtime from concrete MCP Server or Execution Plane database fixtures.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from backend.domain.model_provider import ToolDefinition


@runtime_checkable
class ToolExecutor(Protocol):
    """Protocol for executing tools on behalf of an agent in the Execution Plane.

    Any object providing an async or sync `execute_tool(name, arguments)` satisfies
    this protocol. `AgentMCPServer` implements this protocol naturally.
    """


    async def discover_tools(self, allowed_tools: list[str] | None = None) -> list[ToolDefinition]:
        """Discover tools available for the agent."""
        ...

    async def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        project_id: str | None = None,
        evaluation_id: str | None = None,
        run_id: str | None = None,
        trial_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> Any:
        """Execute a tool request and return the result."""
        ...
