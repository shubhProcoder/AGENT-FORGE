import asyncio
import pytest

from backend.domain.agent_trace import ExecutionEventType
from backend.infrastructure.mcp.client import (
    MCPClientAdapter, 
    MCPToolError, 
    MCPToolTimeout, 
    MCPConnectionError,
)
from backend.infrastructure.mcp.server import AgentMCPServer
from backend.infrastructure.mcp.adapter_server import create_mcp_compatibility_server
from backend.infrastructure.fault_injection.engine import FaultInjectionEngine
from backend.infrastructure.fault_injection.models import FaultRule, FaultType
from backend.infrastructure.database.fixtures import EnvironmentSandbox

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mcp_server():
    server = AgentMCPServer()
    # Provide a simple mock deterministic tool
    server.register_tool("echo", lambda text: {"text": text, "status": "success"})
    server.register_tool("error_tool", lambda: {"is_error": True, "error": "Intentional error"})
    
    async def slow_tool():
        await asyncio.sleep(0.5)
        return {"status": "slow_success"}
    server.register_tool("slow_tool", slow_tool)
    
    server.register_tool("unauthorized_tool", lambda: {"secret": "data"})
    return server

@pytest.fixture
def fault_engine():
    return FaultInjectionEngine()

@pytest.fixture
def adapter():
    return MCPClientAdapter(timeout_seconds=0.2)

async def test_tool_discovery(mcp_server, adapter):
    comp_server = create_mcp_compatibility_server(mcp_server)
    await adapter.connect_in_memory(comp_server)
    
    tools = await adapter.discover_tools()
    
    assert len(tools) == 4
    tool_names = {t.name for t in tools}
    assert tool_names == {"echo", "error_tool", "slow_tool", "unauthorized_tool"}
    
    # Check schema mapping
    echo_tool = next(t for t in tools if t.name == "echo")
    assert "text" in echo_tool.parameters["properties"]
    
    await adapter.disconnect()

async def test_unauthorized_tool_filtering(mcp_server, adapter):
    comp_server = create_mcp_compatibility_server(mcp_server)
    await adapter.connect_in_memory(comp_server)
    
    # Restrict to echo only
    tools = await adapter.discover_tools(allowed_tools=["echo"])
    
    assert len(tools) == 1
    assert tools[0].name == "echo"
    
    await adapter.disconnect()

async def test_valid_tool_call(mcp_server, adapter):
    comp_server = create_mcp_compatibility_server(mcp_server)
    await adapter.connect_in_memory(comp_server)
    
    result = await adapter.execute_tool(
        name="echo",
        arguments={"text": "hello mcp"},
        project_id="proj1",
        trial_id="trial1",
        tool_call_id="call1"
    )
    
    assert result["status"] == "success"
    assert result["text"] == "hello mcp"
    
    # Verify trace
    assert len(adapter._trace_log) == 1
    event = adapter._trace_log[0]
    assert event.event_type == ExecutionEventType.TOOL_RESULT
    assert event.tool_name == "echo"
    assert event.tool_arguments == {"text": "hello mcp"}
    
    await adapter.disconnect()

async def test_mcp_is_error_true(mcp_server, adapter):
    comp_server = create_mcp_compatibility_server(mcp_server)
    await adapter.connect_in_memory(comp_server)
    
    with pytest.raises(MCPToolError) as exc_info:
        await adapter.execute_tool(name="error_tool", arguments={})
        
    assert "Intentional error" in str(exc_info.value)
    
    # Verify trace logs the error
    assert len(adapter._trace_log) == 1
    event = adapter._trace_log[0]
    assert event.event_type == ExecutionEventType.TOOL_ERROR
    assert "Intentional error" in event.error
    
    await adapter.disconnect()

async def test_timeout(mcp_server, adapter):
    comp_server = create_mcp_compatibility_server(mcp_server)
    await adapter.connect_in_memory(comp_server)
    
    with pytest.raises(MCPToolTimeout):
        await adapter.execute_tool(name="slow_tool", arguments={})
        
    # Verify trace logs the timeout
    assert len(adapter._trace_log) == 1
    event = adapter._trace_log[0]
    assert event.event_type == ExecutionEventType.TOOL_ERROR
    assert "timed out" in event.error
    
    await adapter.disconnect()

async def test_injected_failure(mcp_server, fault_engine, adapter):
    # Setup fault injection
    fault_engine.register_rule(FaultRule(
        target_tool="echo",
        fault_type=FaultType.SERVER_ERROR,
        error_message="Injected server error"
    ))
    mcp_server.set_fault_engine(fault_engine)
    
    comp_server = create_mcp_compatibility_server(mcp_server)
    await adapter.connect_in_memory(comp_server)
    
    with pytest.raises(MCPToolError) as exc_info:
        await adapter.execute_tool(name="echo", arguments={"text": "test"})
        
    assert "Injected server error" in str(exc_info.value)
    
    assert len(adapter._trace_log) == 1
    event = adapter._trace_log[0]
    assert event.event_type == ExecutionEventType.TOOL_ERROR
    assert "Injected server error" in event.error
    
    await adapter.disconnect()

async def test_connection_lifecycle_failure(adapter):
    with pytest.raises(MCPConnectionError):
        await adapter.execute_tool("echo", {})
        
    with pytest.raises(MCPConnectionError):
        await adapter.discover_tools()

async def test_multiple_sequential_tool_calls(mcp_server, adapter):
    comp_server = create_mcp_compatibility_server(mcp_server)
    await adapter.connect_in_memory(comp_server)
    
    res1 = await adapter.execute_tool("echo", {"text": "one"})
    assert res1["text"] == "one"
    
    res2 = await adapter.execute_tool("echo", {"text": "two"})
    assert res2["text"] == "two"
    
    assert len(adapter._trace_log) == 2
    
    await adapter.disconnect()
