import inspect
import json
import logging
from typing import Any

from mcp.server.mcpserver import MCPServer
from backend.infrastructure.mcp.server import AgentMCPServer

logger = logging.getLogger(__name__)

def create_mcp_compatibility_server(agent_mcp_server: AgentMCPServer) -> MCPServer:
    """Creates an official MCP Server that delegates to the custom AgentMCPServer.
    
    This acts as a compatibility bridge so the existing server logic and 
    FaultInjectionEngine remain intact, but they are exposed via the real 
    MCP protocol.
    """
    mcp_server = MCPServer("agent-forge-mcp-server")

    for name, func in agent_mcp_server.registered_tools.items():
        # A closure to capture 'name' safely in the loop
        def make_wrapper(tool_name: str) -> Any:
            async def wrapper(**kwargs: Any) -> str:
                logger.debug(f"[MCP Compatibility Server] Calling tool '{tool_name}' with args {kwargs}")
                try:
                    result = await agent_mcp_server.execute_tool(tool_name, kwargs)
                    
                    # Ensure errors are correctly exposed as JSON string
                    # The Custom dispatcher natively supports returning `dict` with "is_error".
                    return json.dumps(result)
                except Exception as e:
                    logger.error(f"[MCP Compatibility Server] Tool execution error: {e}")
                    raise e
            
            # Transfer the docstring to the wrapper so MCPServer reads it as description
            wrapper.__doc__ = func.__doc__ or f"Execute tool '{tool_name}'"
            wrapper.__name__ = tool_name
            # Transfer signature for pydantic parsing done by MCPServer
            wrapper.__signature__ = inspect.signature(func) # type: ignore
            return wrapper
            
        mcp_server.add_tool(make_wrapper(name))

    return mcp_server
