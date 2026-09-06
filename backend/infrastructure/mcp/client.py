"""MCP Client Adapter for Agent Runtime."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from pydantic import BaseModel

from contextlib import AsyncExitStack
from mcp.client.session import ClientSession
from mcp.types import CallToolResult

from backend.domain.model_provider import ToolDefinition
from backend.domain.agent_trace import ExecutionEvent, ExecutionEventType

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base exception for MCP errors."""

class MCPConnectionError(MCPError):
    """Raised when connection fails or session is not active."""

class MCPProtocolError(MCPError):
    """Raised when protocol negotiation or RPC fails."""

class MCPToolError(MCPError):
    """Raised when an MCP tool returns isError=True."""

class MCPToolTimeout(MCPError):
    """Raised when an MCP tool execution times out."""

class MCPAuthorizationError(MCPError):
    """Raised when attempting to execute or discover an unauthorized tool."""


class MCPClientAdapter:
    """Adapts an official MCP ClientSession to the internal ToolExecutor protocol.
    
    Responsibilities:
    - Connection lifecycle (connect, disconnect)
    - Tool discovery mapping (MCP Tool -> ToolDefinition)
    - Execution mapping (ToolResult, isError)
    - Tracing correlation IDs
    - Authorization filtering
    - Timeout handling
    """

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.session: ClientSession | None = None
        self.timeout_seconds = timeout_seconds
        self._trace_log: list[ExecutionEvent] = []
        self._exit_stack = AsyncExitStack()

    async def connect_in_memory(self, mcp_server: Any) -> None:
        """Connect to an in-process MCP Server using memory streams.
        Used primarily for testing.
        """
        from mcp.client._memory import create_client_server_memory_streams
        from mcp.server.mcpserver import MCPServer
        
        client_streams, server_streams = await self._exit_stack.enter_async_context(create_client_server_memory_streams())
        
        # Start the server in the background
        if isinstance(mcp_server, MCPServer):
            # Start the MCPServer
            # MCPServer doesn't have a direct async run that takes streams, but wait...
            # We can use mcp_server._lowlevel_server to get the core Server object
            core_server = mcp_server._lowlevel_server
        else:
            core_server = mcp_server
            
        async def run_server():
            await core_server.run(
                server_streams[0], server_streams[1], core_server.create_initialization_options()
            )
            
        # Create task and ensure it gets cancelled on exit
        task = asyncio.create_task(run_server())
        self._exit_stack.callback(task.cancel)
        
        # Start the client session
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(client_streams[0], client_streams[1])
        )
        await self.session.initialize()

    async def connect_sse(self, url: str) -> None:
        """Connect to an MCP Server using Streamable HTTP (SSE).
        Used for production.
        """
        from mcp.client.sse import sse_client
        
        streams = await self._exit_stack.enter_async_context(sse_client(url))
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(streams[0], streams[1])
        )
        await self.session.initialize()

    async def disconnect(self) -> None:
        """Close the session and transport."""
        await self._exit_stack.aclose()
        self.session = None

    async def discover_tools(self, allowed_tools: list[str] | None = None) -> list[ToolDefinition]:
        """Discover tools from the MCP server, mapping them to internal format.
        
        If allowed_tools is provided, only those tools are exposed.
        """
        if not self.session:
            raise MCPConnectionError("No active MCP session.")

        try:
            # list_tools handles pagination internally in the SDK or we just use the API
            response = await self.session.list_tools()
        except Exception as e:
            raise MCPProtocolError(f"Failed to list tools: {e}") from e

        tool_defs: list[ToolDefinition] = []
        for mcp_tool in response.tools:
            # Authorization filtering
            if allowed_tools is not None and mcp_tool.name not in allowed_tools:
                continue

            tool_defs.append(
                ToolDefinition(
                    name=mcp_tool.name,
                    description=mcp_tool.description or "",
                    parameters=mcp_tool.input_schema or {},
                )
            )

        return tool_defs

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
        """Execute a tool via the MCP client, mapping results and handling timeouts.
        
        Creates a trace event for each execution.
        """
        if not self.session:
            raise MCPConnectionError("No active MCP session.")

        import time
        start_time = time.time()
        error_class: type[MCPError] | None = None
        error_msg: str | None = None
        result: Any = None
        is_error = False
        
        try:
            async with asyncio.timeout(self.timeout_seconds):
                # Call the official MCP SDK call_tool
                response: CallToolResult = await self.session.call_tool(name, arguments)
                
                # Check for is_error from MCP
                if response.is_error:
                    is_error = True
                    error_msg = self._extract_error_text(response)
                    error_class = MCPToolError
                else:
                    # Parse structured content if it is JSON text
                    result = self._extract_result(response)
                    
                    # Our compatibility bridge might encode {"is_error": True} inside the text 
                    # for custom dispatcher errors that didn't use the MCP isError flag natively
                    if isinstance(result, dict) and result.get("is_error"):
                        is_error = True
                        error_msg = str(result.get("error", "Unknown tool error"))
                        error_class = MCPToolError
                        
        except TimeoutError as e:
            is_error = True
            error_class = MCPToolTimeout
            error_msg = f"Tool execution timed out after {self.timeout_seconds} seconds"
        except Exception as e:
            is_error = True
            error_class = MCPProtocolError
            error_msg = f"Protocol or execution error: {e}"

        duration = time.time() - start_time
        
        # Record trace
        event = ExecutionEvent(
            event_type=ExecutionEventType.TOOL_ERROR if is_error else ExecutionEventType.TOOL_RESULT,
            tool_name=name,
            tool_arguments=arguments,
            tool_response=result if not is_error else {"error": error_msg},
            latency_ms=duration * 1000,
            error=error_msg,
        )
        self._trace_log.append(event)
        
        if is_error and error_class:
            raise error_class(error_msg)
            
        return result

    def _extract_error_text(self, response: CallToolResult) -> str:
        """Extract error message from CallToolResult."""
        from mcp.types import TextContent
        texts = [c.text for c in response.content if isinstance(c, TextContent)]
        return "\n".join(texts) if texts else "Unknown MCP error"

    def _extract_result(self, response: CallToolResult) -> Any:
        """Extract structured result from CallToolResult."""
        from mcp.types import TextContent
        if not response.content:
            return None
            
        # Try to parse JSON from TextContent
        for content in response.content:
            if isinstance(content, TextContent):
                try:
                    return json.loads(content.text)
                except json.JSONDecodeError:
                    return content.text
                    
        return None
