"""Unit tests for TASK 12.2: Real Agent Runtime.

Tests verify:
1. Deterministic mock agent completes task
2. Agent terminates normally
3. Max iteration termination
4. Max tool-call termination
5. Tool failure handling & recording
6. Malformed tool call handling
7. Repeated tool loop detection (including equivalent argument order)
8. Cancellation support
9. Execution timeout enforcement
10. Final answer without tools
11. Multi-step tool execution
12. Security: Hidden test data & invariants are never exposed to the agent
13. Complete telemetry recording on execution events
14. Forbidden tool invocation is blocked
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from backend.application.agent_runtime import AgentRuntime, AgentRuntimeError
from backend.domain.agent_trace import (
    ExecutionEventType,
    TerminationReason,
)
from backend.domain.dataset import Task
from backend.domain.environment import Environment, EnvironmentStatus
from backend.domain.model_provider import (
    FinishReason,
    ModelFormatError,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    UsageMetadata,
)
from backend.domain.trial import Trial
from backend.infrastructure.llm.mock_provider import MockModelProvider

# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────


class MockToolExecutor:
    """Deterministic tool executor tracking call history and scripted outputs."""

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self.responses = responses or {}
        self.raise_error = raise_error
        self.executed_calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.executed_calls.append((name, arguments))
        if self.raise_error:
            raise self.raise_error
        return self.responses.get(name, {"status": "success", "tool": name})


def _create_trial() -> Trial:
    return Trial(run_id=uuid.uuid4(), task_id=uuid.uuid4())


def _create_env(trial_id: uuid.UUID) -> Environment:
    return Environment(trial_id=trial_id, status=EnvironmentStatus.ACTIVE)


# ─────────────────────────────────────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentRuntime:
    """Test suite covering the 12 requirements of TASK 12.2."""

    @pytest.mark.asyncio
    async def test_deterministic_mock_agent_completes_task(self):
        """1. Deterministic mock agent executes a tool call, then terminates on final answer."""
        provider = MockModelProvider(
            responses=[
                ModelResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_01",
                            name="refund_order",
                            arguments={"order_id": "ord_100", "amount": 50.0},
                        )
                    ],
                    finish_reason=FinishReason.TOOL_CALLS,
                ),
                ModelResponse(
                    text_content="Order ord_100 refunded successfully.",
                    finish_reason=FinishReason.STOP,
                ),
            ]
        )
        executor = MockToolExecutor(
            responses={"refund_order": {"status": "refunded", "order_id": "ord_100"}}
        )
        runtime = AgentRuntime(model_provider=provider, tool_executor=executor)
        trial = _create_trial()
        env = _create_env(trial.id)

        trace = await runtime.execute(trial, env, allowed_tools=["refund_order"])

        assert trial.is_completed is True
        assert trial.model_calls == 2
        assert trial.total_tokens > 0
        assert trace.termination_reason == TerminationReason.NORMAL
        assert trace.final_answer == "Order ord_100 refunded successfully."
        assert trace.total_tool_calls == 1
        assert len(executor.executed_calls) == 1
        expected_call = ("refund_order", {"order_id": "ord_100", "amount": 50.0})
        assert executor.executed_calls[0] == expected_call

    @pytest.mark.asyncio
    async def test_agent_terminates_normally(self):
        """2. Agent terminates cleanly when model provides a direct final answer."""
        provider = MockModelProvider.with_text("The user inquiry has been resolved.")
        runtime = AgentRuntime(model_provider=provider)
        trial = _create_trial()
        env = _create_env(trial.id)

        trace = await runtime.execute(trial, env, allowed_tools=[])

        assert trial.is_completed is True
        assert trial.error_message is None
        assert trace.termination_reason == TerminationReason.NORMAL
        assert trace.final_answer == "The user inquiry has been resolved."
        assert trace.total_iterations == 1
        assert trace.total_tool_calls == 0

    @pytest.mark.asyncio
    async def test_max_iteration_termination(self):
        """3. Runtime halts and raises AgentRuntimeError when max_iterations is reached."""
        # Factory creates distinct arguments on each turn so loop detector is not triggered
        call_idx = 0

        def varying_tool_factory(req: ModelRequest) -> ModelResponse:
            nonlocal call_idx
            call_idx += 1
            return ModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        id=f"call_{call_idx}",
                        name="check_status",
                        arguments={"seq": call_idx},
                    )
                ],
                finish_reason=FinishReason.TOOL_CALLS,
            )

        provider = MockModelProvider(response_factory=varying_tool_factory)
        runtime = AgentRuntime(model_provider=provider, max_iterations=3)
        trial = _create_trial()
        env = _create_env(trial.id)

        with pytest.raises(AgentRuntimeError, match="Exceeded max iterations: 3"):
            await runtime.execute(trial, env, allowed_tools=["check_status"])

        assert trial.error_message == "Exceeded max iterations: 3"
        assert runtime.last_trace.termination_reason == TerminationReason.MAX_ITERATIONS
        assert runtime.last_trace.total_iterations == 3

    @pytest.mark.asyncio
    async def test_max_tool_call_termination(self):
        """4. Runtime halts and raises AgentRuntimeError when max_tool_calls is exceeded."""
        call_idx = 0

        def multi_tool_factory(req: ModelRequest) -> ModelResponse:
            nonlocal call_idx
            call_idx += 1
            return ModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        id=f"call_{call_idx}",
                        name="check_status",
                        arguments={"seq": call_idx},
                    )
                ],
                finish_reason=FinishReason.TOOL_CALLS,
            )

        provider = MockModelProvider(response_factory=multi_tool_factory)
        runtime = AgentRuntime(model_provider=provider, max_tool_calls=2)
        trial = _create_trial()
        env = _create_env(trial.id)

        with pytest.raises(AgentRuntimeError, match="Exceeded max tool calls: 2"):
            await runtime.execute(trial, env, allowed_tools=["check_status"])

        assert trial.error_message == "Exceeded max tool calls: 2"
        assert runtime.last_trace.termination_reason == TerminationReason.MAX_TOOL_CALLS

    @pytest.mark.asyncio
    async def test_tool_failure_recorded(self):
        """5. Tool execution failures are captured as TOOL_ERROR events and fed to the model."""
        provider = MockModelProvider(
            responses=[
                ModelResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_fail",
                            name="create_order",
                            arguments={"item": "widget"},
                        )
                    ],
                    finish_reason=FinishReason.TOOL_CALLS,
                ),
                ModelResponse(
                    text_content="Order creation failed due to database unavailable.",
                    finish_reason=FinishReason.STOP,
                ),
            ]
        )
        executor = MockToolExecutor(raise_error=RuntimeError("Database connection dropped"))
        runtime = AgentRuntime(model_provider=provider, tool_executor=executor)
        trial = _create_trial()
        env = _create_env(trial.id)

        trace = await runtime.execute(trial, env, allowed_tools=["create_order"])

        assert trial.is_completed is True
        assert trace.has_errors() is True
        error_events = [
            e for e in trace.events
            if e.event_type == ExecutionEventType.TOOL_ERROR.value
        ]
        assert len(error_events) == 1
        assert "Database connection dropped" in error_events[0].error
        assert error_events[0].tool_name == "create_order"
        assert trace.termination_reason == TerminationReason.NORMAL

    @pytest.mark.asyncio
    async def test_malformed_tool_call(self):
        """6. ModelFormatError from provider is captured in trace and raises AgentRuntimeError."""
        provider = MockModelProvider(
            simulate_error=ModelFormatError("Malformed JSON in tool arguments payload")
        )
        runtime = AgentRuntime(model_provider=provider)
        trial = _create_trial()
        env = _create_env(trial.id)

        with pytest.raises(ModelFormatError, match="Malformed JSON in tool arguments"):
            await runtime.execute(trial, env, allowed_tools=[])

        assert runtime.last_trace.termination_reason == TerminationReason.MODEL_ERROR
        assert runtime.last_trace.has_errors() is True
        assert "Malformed JSON" in trial.error_message

    @pytest.mark.asyncio
    async def test_repeated_tool_loop_detection(self):
        """7. Obvious tool loops with identical arguments are detected and terminated."""
        provider = MockModelProvider(
            responses=[
                ModelResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id=f"loop_{i}",
                            name="refund_order",
                            arguments={"order_id": "ord_999"},
                        )
                    ],
                    finish_reason=FinishReason.TOOL_CALLS,
                )
                for i in range(5)
            ]
        )
        runtime = AgentRuntime(model_provider=provider, loop_detection_threshold=3)
        trial = _create_trial()
        env = _create_env(trial.id)

        with pytest.raises(AgentRuntimeError, match="Tool loop detected"):
            await runtime.execute(trial, env, allowed_tools=["refund_order"])

        assert runtime.last_trace.termination_reason == TerminationReason.TOOL_LOOP_DETECTED
        assert "Tool loop detected" in trial.error_message

    @pytest.mark.asyncio
    async def test_repeated_tool_loop_with_shuffled_argument_keys(self):
        """7b. Tool loop detection handles key-ordering differences as equivalent arguments."""
        # Turn 1: {"a": 1, "b": 2}
        # Turn 2: {"b": 2, "a": 1}
        # Turn 3: {"a": 1, "b": 2} -> triggers threshold=3
        provider = MockModelProvider(
            responses=[
                ModelResponse(
                    tool_calls=[
                        ToolCallRequest(id="c1", name="sync", arguments={"a": 1, "b": 2})
                    ],
                    finish_reason=FinishReason.TOOL_CALLS,
                ),
                ModelResponse(
                    tool_calls=[
                        ToolCallRequest(id="c2", name="sync", arguments={"b": 2, "a": 1})
                    ],
                    finish_reason=FinishReason.TOOL_CALLS,
                ),
                ModelResponse(
                    tool_calls=[
                        ToolCallRequest(id="c3", name="sync", arguments={"a": 1, "b": 2})
                    ],
                    finish_reason=FinishReason.TOOL_CALLS,
                ),
            ]
        )
        runtime = AgentRuntime(model_provider=provider, loop_detection_threshold=3)
        trial = _create_trial()
        env = _create_env(trial.id)

        with pytest.raises(AgentRuntimeError, match="Tool loop detected"):
            await runtime.execute(trial, env, allowed_tools=["sync"])

        assert runtime.last_trace.termination_reason == TerminationReason.TOOL_LOOP_DETECTED

    @pytest.mark.asyncio
    async def test_cancellation(self):
        """8. Cancellation token halts execution and records CANCELLED termination."""
        cancel_token = asyncio.Event()
        cancel_token.set()  # Cancel prior to execution

        provider = MockModelProvider.with_text("Should not be returned")
        runtime = AgentRuntime(model_provider=provider)
        trial = _create_trial()
        env = _create_env(trial.id)

        with pytest.raises(AgentRuntimeError, match="cancelled"):
            await runtime.execute(trial, env, allowed_tools=[], cancellation_token=cancel_token)

        assert runtime.last_trace.termination_reason == TerminationReason.CANCELLED
        assert "cancelled" in trial.error_message

    @pytest.mark.asyncio
    async def test_timeout(self):
        """9. Execution exceeding timeout_seconds is terminated with TIMEOUT reason."""
        class SlowMockProvider:
            async def generate(self, request: ModelRequest) -> ModelResponse:
                await asyncio.sleep(0.15)
                return ModelResponse(text_content="Too slow")

        runtime = AgentRuntime(model_provider=SlowMockProvider(), timeout_seconds=0.02)
        trial = _create_trial()
        env = _create_env(trial.id)

        with pytest.raises(AgentRuntimeError, match="timed out"):
            await runtime.execute(trial, env, allowed_tools=[])

        assert runtime.last_trace.termination_reason == TerminationReason.TIMEOUT
        assert "timed out" in trial.error_message

    @pytest.mark.asyncio
    async def test_final_answer_without_tools(self):
        """10. Final answer returned on first turn with zero tool calls."""
        provider = MockModelProvider.with_text("The answer is 42.")
        runtime = AgentRuntime(model_provider=provider)
        trial = _create_trial()
        env = _create_env(trial.id)

        trace = await runtime.execute(trial, env, allowed_tools=["refund_order", "create_order"])

        assert trial.is_completed is True
        assert trial.model_calls == 1
        assert trace.total_tool_calls == 0
        assert len(trace.get_tool_calls()) == 0
        assert trace.final_answer == "The answer is 42."
        assert trace.termination_reason == TerminationReason.NORMAL

    @pytest.mark.asyncio
    async def test_multi_step_tool_execution(self):
        """11. Multi-step agent workflow dispatches tools in sequence with observations."""
        provider = MockModelProvider(
            responses=[
                ModelResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="tc_1",
                            name="search_customer",
                            arguments={"email": "alice@example.com"},
                        )
                    ],
                    finish_reason=FinishReason.TOOL_CALLS,
                ),
                ModelResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="tc_2",
                            name="create_ticket",
                            arguments={"customer_id": "cust_123", "subject": "Billing issue"},
                        )
                    ],
                    finish_reason=FinishReason.TOOL_CALLS,
                ),
                ModelResponse(
                    text_content="Ticket tkt_555 opened for customer cust_123.",
                    finish_reason=FinishReason.STOP,
                ),
            ]
        )
        executor = MockToolExecutor(
            responses={
                "search_customer": {"customer_id": "cust_123", "name": "Alice"},
                "create_ticket": {"ticket_id": "tkt_555", "status": "OPEN"},
            }
        )
        runtime = AgentRuntime(model_provider=provider, tool_executor=executor)
        trial = _create_trial()
        env = _create_env(trial.id)

        trace = await runtime.execute(
            trial, env, allowed_tools=["search_customer", "create_ticket"]
        )

        assert trial.is_completed is True
        assert trial.model_calls == 3
        assert trace.total_tool_calls == 2
        assert len(executor.executed_calls) == 2
        assert executor.executed_calls[0][0] == "search_customer"
        assert executor.executed_calls[1][0] == "create_ticket"
        assert trace.final_answer == "Ticket tkt_555 opened for customer cust_123."
        assert trace.termination_reason == TerminationReason.NORMAL

    @pytest.mark.asyncio
    async def test_hidden_test_invariants_never_exposed_to_agent(self):
        """12. Security: Hidden invariants and private contracts are NEVER sent to the agent."""
        secret_hidden_token = "CONFIDENTIAL_INVARIANT_RULE_xyz999"
        secret_hidden_state = "INTERNAL_EXPECTED_STATE_SECRET"

        task = Task(
            dataset_id=uuid.uuid4(),
            name="Public Support Task",
            description="Public task description visible to agent",
            input_prompt="Please help the customer find their order.",
            allowed_tools=["search_customer"],
            # Hidden fields that must NEVER leak
            hidden_invariants=[secret_hidden_token],
            hidden_expected_state={"secret_key": secret_hidden_state},
        )

        provider = MockModelProvider.with_text("Found customer.")
        runtime = AgentRuntime(model_provider=provider)
        trial = _create_trial()
        env = _create_env(trial.id)

        await runtime.execute(trial, env, task=task)

        # Inspect all messages received by the model provider
        for req in provider.received_requests:
            for msg in req.messages:
                content = msg.content or ""
                assert secret_hidden_token not in content, (
                    f"Hidden invariant leaked into prompt message: {content}"
                )
                assert secret_hidden_state not in content, (
                    f"Hidden expected state leaked into prompt message: {content}"
                )

    @pytest.mark.asyncio
    async def test_execution_event_telemetry_complete(self):
        """13. Telemetry: Verify all required event fields are faithfully recorded."""
        provider = MockModelProvider(
            responses=[
                ModelResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="tc_rec",
                            name="create_order",
                            arguments={"sku": "item_9"},
                        )
                    ],
                    usage=UsageMetadata(prompt_tokens=15, completion_tokens=10, total_tokens=25),
                    finish_reason=FinishReason.TOOL_CALLS,
                ),
                ModelResponse(
                    text_content="Order created.",
                    usage=UsageMetadata(prompt_tokens=20, completion_tokens=5, total_tokens=25),
                    finish_reason=FinishReason.STOP,
                ),
            ]
        )
        executor = MockToolExecutor(responses={"create_order": {"order_id": "ord_88"}})
        runtime = AgentRuntime(model_provider=provider, tool_executor=executor)
        trial = _create_trial()
        env = _create_env(trial.id)

        trace = await runtime.execute(trial, env, allowed_tools=["create_order"])

        # Check trace summary
        summary = trace.to_summary()
        assert summary["total_iterations"] == 2
        assert summary["total_tool_calls"] == 1
        assert summary["final_answer"] == "Order created."
        assert summary["termination_reason"] == "normal"
        assert summary["event_count"] >= 4

        # Check tool event fields: iteration, tool_name, tool_arguments, tool_response, latency_ms
        tool_events = trace.get_tool_calls()
        assert len(tool_events) == 1
        t_evt = tool_events[0]
        assert t_evt.iteration == 1
        assert t_evt.tool_name == "create_order"
        assert t_evt.tool_arguments == {"sku": "item_9"}
        assert t_evt.tool_response == {"order_id": "ord_88"}
        assert t_evt.latency_ms >= 0.0

        # Check model event fields
        model_events = trace.get_events_by_type(ExecutionEventType.MODEL_CALL)
        assert len(model_events) == 2
        assert model_events[0].model_call["total_tokens"] == 25
        assert model_events[0].latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_forbidden_tool_rejected(self):
        """14. Invoking a tool outside allowed_tools terminates with FORBIDDEN_TOOL."""
        provider = MockModelProvider(
            responses=[
                ModelResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="tc_bad",
                            name="delete_database",
                            arguments={},
                        )
                    ],
                    finish_reason=FinishReason.TOOL_CALLS,
                )
            ]
        )
        runtime = AgentRuntime(model_provider=provider)
        trial = _create_trial()
        env = _create_env(trial.id)

        with pytest.raises(AgentRuntimeError, match="forbidden"):
            await runtime.execute(trial, env, allowed_tools=["search_customer"])

        assert runtime.last_trace.termination_reason == TerminationReason.FORBIDDEN_TOOL
        assert "forbidden" in trial.error_message
