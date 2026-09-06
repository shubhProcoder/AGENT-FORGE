"""Agent MCP Server (Execution Plane).

This server acts as the strict integration boundary between the Agent Runtime 
and the backend tools. The evaluated agent ONLY communicates with this server.
It does NOT have access to the Control Plane (FastAPI APIs, Evaluation controls).
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Callable

from backend.infrastructure.fault_injection.engine import FaultInjectionEngine
from backend.infrastructure.fault_injection.models import FaultInjectionError
from backend.infrastructure.idempotency.store import IdempotencyConflictError

logger = logging.getLogger(__name__)


class AgentMCPServer:
    """The Model Context Protocol Server for the Agent."""

    def __init__(self, fault_engine: FaultInjectionEngine | None = None) -> None:
        self.registered_tools: dict[str, Callable[..., Any]] = {}
        self.fault_engine = fault_engine
        self.call_trace: list[dict[str, Any]] = []

    def set_fault_engine(self, fault_engine: FaultInjectionEngine | None) -> None:
        """Attach or detach a fault injection engine."""
        self.fault_engine = fault_engine

    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        """Register a backend tool."""
        self.registered_tools[name] = func

    def register_sandbox(self, sandbox: Any) -> None:
        """Register all business operations from an EnvironmentSandbox."""
        if hasattr(sandbox, "create_order"):
            self.register_tool("create_order", sandbox.create_order)
        if hasattr(sandbox, "refund_order"):
            self.register_tool("refund_order", sandbox.refund_order)
        if hasattr(sandbox, "create_ticket"):
            self.register_tool("create_ticket", sandbox.create_ticket)

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool request from the Agent Runtime, running through fault injection."""
        if name not in self.registered_tools:
            raise ValueError(f"Tool '{name}' is not available on this MCP Server.")

        logger.info(f"[MCP Server] Executing tool '{name}' with args {arguments}")
        func = self.registered_tools[name]
        start_time = time.time()

        try:
            if self.fault_engine:
                result = await self.fault_engine.intercept(name, func, **arguments)
            else:
                if inspect.iscoroutinefunction(func):
                    result = await func(**arguments)
                else:
                    result = func(**arguments)

            duration = time.time() - start_time
            self.call_trace.append({
                "tool": name,
                "arguments": arguments,
                "status": "success",
                "result": result,
                "duration_seconds": duration,
                "timestamp": start_time,
            })
            return result

        except FaultInjectionError as fie:
            duration = time.time() - start_time
            logger.warning(f"[MCP Server] Injected fault caught during '{name}': {fie.message}")
            error_payload = {
                "is_error": True,
                "error": fie.message,
                "fault_injected": True,
                "fault_type": fie.fault_type.value,
                "details": fie.details,
            }
            self.call_trace.append({
                "tool": name,
                "arguments": arguments,
                "status": "fault_injected",
                "error": error_payload,
                "duration_seconds": duration,
                "timestamp": start_time,
            })
            return error_payload

        except IdempotencyConflictError as ice:
            duration = time.time() - start_time
            logger.warning(f"[MCP Server] Idempotency conflict during '{name}': {ice}")
            error_payload = {
                "is_error": True,
                "error": str(ice),
                "status_code": 409,
            }
            self.call_trace.append({
                "tool": name,
                "arguments": arguments,
                "status": "conflict",
                "error": error_payload,
                "duration_seconds": duration,
                "timestamp": start_time,
            })
            return error_payload

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[MCP Server] Tool '{name}' execution failed: {e}")
            error_payload = {"is_error": True, "error": str(e)}
            self.call_trace.append({
                "tool": name,
                "arguments": arguments,
                "status": "error",
                "error": error_payload,
                "duration_seconds": duration,
                "timestamp": start_time,
            })
            return error_payload

    def clear_trace(self) -> None:
        """Clear execution trace logs."""
        self.call_trace.clear()


# Global default instance for the Execution Plane
agent_mcp_server = AgentMCPServer()
