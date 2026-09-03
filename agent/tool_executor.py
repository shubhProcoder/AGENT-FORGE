"""Tool executor — dispatches tool calls to registered tool functions.

Acts as the MCP-compatible bridge between the Agent Runtime and
concrete tool implementations under tools/.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Coroutine

from loguru import logger

from agent.retry import ToolTimeoutError, with_timeout
from agent.state import ExecutionContext

# Global registry: tool_name → async callable
_TOOL_REGISTRY: dict[str, Callable[..., Coroutine[Any, Any, dict[str, Any]]]] = {}


def register_tool(name: str) -> Callable:
    """Decorator to register an async function as a named tool.

    Usage:
        @register_tool("search_customer")
        async def search_customer(query: str) -> dict[str, Any]:
            ...
    """

    def decorator(fn: Callable[..., Coroutine[Any, Any, dict[str, Any]]]) -> Callable:
        _TOOL_REGISTRY[name] = fn
        logger.debug(f"Registered tool: {name}")
        return fn

    return decorator


def get_registered_tools() -> dict[str, Callable]:
    """Return a copy of the current tool registry."""
    return dict(_TOOL_REGISTRY)


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return OpenAI-compatible function definitions for all registered tools.

    Each tool function should have a `__tool_schema__` attribute set by its
    module.  If not present, a minimal schema is generated from the name.
    """
    definitions: list[dict[str, Any]] = []
    for name, fn in _TOOL_REGISTRY.items():
        schema = getattr(fn, "__tool_schema__", None)
        if schema:
            definitions.append(schema)
        else:
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": fn.__doc__ or f"Tool: {name}",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            )
    return definitions


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    ctx: ExecutionContext,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Execute a registered tool, recording the call in the execution context.

    Returns the tool response dict.
    Raises KeyError if the tool is not registered.
    Raises ToolTimeoutError if the call exceeds the timeout.
    """
    fn = _TOOL_REGISTRY.get(tool_name)
    if fn is None:
        error_resp = {"error": f"Unknown tool: {tool_name}"}
        ctx.log_tool_call(tool_name, arguments, error_resp, status="error")
        raise KeyError(f"Tool '{tool_name}' is not registered")

    start = time.perf_counter()
    retries = 0
    status = "success"
    response: dict[str, Any] = {}

    try:
        response = await with_timeout(
            fn(**arguments),
            timeout_seconds=timeout_seconds,
            label=tool_name,
        )
    except ToolTimeoutError:
        status = "timeout"
        response = {"error": f"Tool {tool_name} timed out"}
        raise
    except Exception as exc:
        status = "error"
        response = {"error": str(exc)}
        raise
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        ctx.log_tool_call(
            tool_name,
            arguments,
            response,
            status=status,
            latency_ms=latency_ms,
            retry_count=retries,
        )

    return response
