"""Agent Runtime with explicit stopping controls.

This is the execution loop that interacts with the LLM through a ModelProvider
and dispatches tools via a ToolExecutor (e.g. MCP Server) to the Execution Plane.

The runtime is strictly provider-agnostic and decoupled from:
- Concrete database implementations
- Concrete MCP server implementations
- Scoring engine and verifier implementations

Architecture:
    Control Plane
        ↓
    Run Manager
        ↓
    Trial Manager
        ↓
    Agent Runtime   ← this module
        ↓
    ModelProvider   (injected dependency, not hardcoded)
        ↓
    ToolExecutor    (injected protocol, not hardcoded)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from backend.domain.agent_trace import (
    AgentTrace,
    ExecutionEvent,
    ExecutionEventType,
    TerminationReason,
)
from backend.domain.environment import Environment
from backend.domain.model_provider import (
    MessageRole,
    ModelError,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ToolDefinition,
)
from backend.domain.trial import Trial

if TYPE_CHECKING:
    from backend.application.tool_executor import ToolExecutor
    from backend.domain.dataset import Task

logger = logging.getLogger(__name__)


class AgentRuntimeError(Exception):
    """Base exception for agent runtime execution failures."""


class AgentRuntime:
    """The controlled tool-using agent execution loop.

    Parameters
    ----------
    model_provider:
        Any object that satisfies the ``ModelProvider`` protocol.
        If not supplied, defaults to a ``MockModelProvider``.
    tool_executor:
        Any object that satisfies the ``ToolExecutor`` protocol or callable.
        Decouples runtime from concrete MCP server implementation.
    max_iterations, max_tool_calls, timeout_seconds, max_budget:
        Execution safety bounds.
    loop_detection_threshold:
        Number of repeated equivalent tool calls that triggers loop termination.
    """

    def __init__(
        self,
        model_provider: ModelProvider | None = None,
        tool_executor: ToolExecutor | Callable[..., Any] | None = None,
        max_iterations: int = 15,
        max_tool_calls: int = 30,
        timeout_seconds: int = 120,
        max_budget: float = 1.0,
        loop_detection_threshold: int = 3,
    ) -> None:
        if model_provider is None:
            from backend.infrastructure.llm.mock_provider import MockModelProvider
            model_provider = MockModelProvider()

        self.model_provider: ModelProvider = model_provider
        self.tool_executor: ToolExecutor | Callable[..., Any] | None = tool_executor
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.timeout_seconds = timeout_seconds
        self.max_budget = max_budget
        self.loop_detection_threshold = loop_detection_threshold

        # Last execution trace recorded by this runtime
        self.last_trace: AgentTrace | None = None

        # Legacy compatibility attribute
        self.mcp_client = None

    async def generate_model_response(self, request: ModelRequest) -> ModelResponse:
        """Call the injected provider and return a typed ModelResponse.

        This is the single point of contact between AgentRuntime and the LLM.
        Any ModelError raised by the provider propagates upward.
        """
        return await self.model_provider.generate(request)

    async def execute(
        self,
        trial: Trial,
        env: Environment,
        allowed_tools: list[str] | None = None,
        task_input: str | None = None,
        task: Task | None = None,
        cancellation_token: asyncio.Event | None = None,
    ) -> AgentTrace:
        """Run the controlled agent loop with explicit safety bounds.

        Parameters
        ----------
        trial:
            The Trial domain entity to record metrics on.
        env:
            The sandboxed Environment allocated for this execution.
        allowed_tools:
            Whitelist of tool names the agent is permitted to call.
        task_input:
            Public task prompt/instruction.
        task:
            Optional Task domain entity. If provided, strictly public contract
            views are extracted — hidden invariants/tests are never exposed.
        cancellation_token:
            Optional asyncio.Event to cancel execution mid-flight.

        Returns
        -------
        AgentTrace:
            Full step-by-step execution trace with telemetry and events.
        """
        trace = AgentTrace(trial_id=trial.id)
        self.last_trace = trace
        trial.trace = trace

        # Determine effective allowed tools
        if allowed_tools is None:
            if task is not None and task.allowed_tools:
                allowed_tools = list(task.allowed_tools)
            else:
                allowed_tools = []

        # Build initial messages strictly from public context
        prompt_content = self._extract_public_prompt(task=task, task_input=task_input)
        messages: list[ModelMessage] = [
            ModelMessage(role=MessageRole.SYSTEM, content="You are a helpful support agent."),
            ModelMessage(role=MessageRole.USER, content=prompt_content),
        ]

        # Tool definitions exposed to the model
        if self.tool_executor is not None and hasattr(self.tool_executor, "discover_tools"):
            res = self.tool_executor.discover_tools(allowed_tools=allowed_tools)
            if inspect.isawaitable(res):
                tool_definitions = await res
            else:
                tool_definitions = res
        else:
            # Fallback for mock/legacy
            tool_definitions = [
                ToolDefinition(name=name, description=f"Execute tool '{name}'")
                for name in allowed_tools
            ]

        iteration_count = 0
        tool_call_count = 0
        tool_call_history: list[tuple[str, str]] = []

        try:
            async with asyncio.timeout(self.timeout_seconds):
                while True:
                    # 1. Check cancellation token
                    if cancellation_token is not None and cancellation_token.is_set():
                        self._handle_cancellation(trace, trial, iteration_count)

                    # 2. Check max iterations
                    if iteration_count >= self.max_iterations:
                        self._handle_max_iterations(trace, trial, iteration_count)

                    # 3. Check budget limits
                    if trial.estimated_cost >= self.max_budget:
                        self._handle_budget_exceeded(trace, trial, iteration_count)

                    iteration_count += 1
                    trace.total_iterations = iteration_count

                    trace.add_event(
                        ExecutionEvent(
                            iteration=iteration_count,
                            event_type=ExecutionEventType.ITERATION_START,
                        )
                    )

                    # 4. Generate model response
                    logger.debug(f"Trial {trial.id}: Iteration {iteration_count}")
                    request = ModelRequest(
                        messages=messages,
                        tools=tool_definitions,
                    )

                    t0 = time.perf_counter()
                    try:
                        response: ModelResponse = await self.generate_model_response(request)
                    except ModelError as me:
                        model_latency = (time.perf_counter() - t0) * 1000
                        self._handle_model_error(trace, trial, iteration_count, me, model_latency)
                        raise

                    model_latency = (time.perf_counter() - t0) * 1000

                    # Update trial accounting
                    trial.model_calls += 1
                    trial.total_tokens += response.usage.total_tokens
                    trial.estimated_cost += response.usage.estimated_cost_usd

                    trace.add_event(
                        ExecutionEvent(
                            iteration=iteration_count,
                            event_type=ExecutionEventType.MODEL_CALL,
                            model_call={
                                "model_id": response.model_id,
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens,
                            },
                            latency_ms=model_latency,
                        )
                    )

                    # 5. Check cancellation after model response
                    if cancellation_token is not None and cancellation_token.is_set():
                        self._handle_cancellation(trace, trial, iteration_count)

                    # 6. Branch: Tool Calls vs Final Answer
                    if response.has_tool_calls:
                        messages.append(
                            ModelMessage(
                                role=MessageRole.ASSISTANT,
                                tool_calls=response.tool_calls,
                            )
                        )

                        for tool_call in response.tool_calls:
                            # Verify tool call limits
                            if tool_call_count >= self.max_tool_calls:
                                self._handle_max_tool_calls(trace, trial, iteration_count)

                            # Validate tool call format
                            if not tool_call.name:
                                self._handle_malformed_tool_call(
                                    trace, trial, iteration_count, "Tool call missing name"
                                )

                            # Validate tool authorization
                            if tool_call.name not in allowed_tools:
                                self._handle_forbidden_tool(
                                    trace, trial, iteration_count, tool_call.name
                                )

                            # Detect obvious tool loops
                            canonical_args = json.dumps(tool_call.arguments, sort_keys=True)
                            tool_signature = (tool_call.name, canonical_args)
                            tool_call_history.append(tool_signature)

                            if self._is_tool_loop(tool_signature, tool_call_history):
                                self._handle_tool_loop(
                                    trace,
                                    trial,
                                    iteration_count,
                                    tool_call.name,
                                    tool_call.arguments,
                                )

                            tool_call_count += 1
                            trace.total_tool_calls = tool_call_count

                            # Check cancellation immediately before dispatch
                            if cancellation_token is not None and cancellation_token.is_set():
                                self._handle_cancellation(trace, trial, iteration_count)

                            # Dispatch tool through abstraction
                            t_tool_0 = time.perf_counter()
                            tool_error: str | None = None
                            try:
                                tool_result = await self._execute_tool(
                                    env, 
                                    tool_call.name, 
                                    tool_call.arguments,
                                    trial=trial,
                                    tool_call_id=tool_call.id,
                                )
                                # Check if MCP/tool returned structured error payload
                                if isinstance(tool_result, dict) and tool_result.get("is_error"):
                                    tool_error = str(tool_result.get("error", "Tool error"))
                            except Exception as exc:
                                tool_error = str(exc)
                                tool_result = {"is_error": True, "error": str(exc)}

                            tool_latency = (time.perf_counter() - t_tool_0) * 1000

                            # Record execution event
                            if tool_error:
                                trace.add_event(
                                    ExecutionEvent(
                                        iteration=iteration_count,
                                        event_type=ExecutionEventType.TOOL_ERROR,
                                        tool_name=tool_call.name,
                                        tool_arguments=tool_call.arguments,
                                        tool_response=tool_result,
                                        latency_ms=tool_latency,
                                        error=tool_error,
                                    )
                                )
                            else:
                                trace.add_event(
                                    ExecutionEvent(
                                        iteration=iteration_count,
                                        event_type=ExecutionEventType.TOOL_CALL,
                                        tool_name=tool_call.name,
                                        tool_arguments=tool_call.arguments,
                                        tool_response=tool_result,
                                        latency_ms=tool_latency,
                                    )
                                )

                            # Append tool observation back to conversation history
                            obs_content = (
                                json.dumps(tool_result)
                                if isinstance(tool_result, (dict, list))
                                else str(tool_result)
                            )
                            messages.append(
                                ModelMessage(
                                    role=MessageRole.TOOL,
                                    tool_call_id=tool_call.id,
                                    content=obs_content,
                                )
                            )

                    else:
                        # Final answer branch — normal termination
                        final_text = response.text_content or ""
                        messages.append(
                            ModelMessage(
                                role=MessageRole.ASSISTANT,
                                content=final_text,
                            )
                        )
                        trace.final_answer = final_text
                        trace.termination_reason = TerminationReason.NORMAL
                        trace.add_event(
                            ExecutionEvent(
                                iteration=iteration_count,
                                event_type=ExecutionEventType.FINAL_ANSWER,
                                termination_reason=TerminationReason.NORMAL,
                            )
                        )
                        trial.is_completed = True
                        break

        except TimeoutError:
            self._handle_timeout(trace, trial, iteration_count)

        finally:
            trial.latency_ms = int(trace.total_latency_ms)

        return trace

    # ── Tool Execution Abstraction ─────────────────────────────────────────

    async def _execute_tool(
        self, env: Environment, tool_name: str, arguments: dict[str, Any], trial: Trial | None = None, tool_call_id: str | None = None
    ) -> Any:
        """Dispatch tool execution through the ToolExecutor abstraction.

        Decouples the AgentRuntime from concrete MCP server implementations.
        """
        if self.tool_executor is not None:
            kwargs = {}
            if trial is not None:
                kwargs["project_id"] = str(trial.project_id) if hasattr(trial, "project_id") else None
                kwargs["evaluation_id"] = str(trial.evaluation_id) if hasattr(trial, "evaluation_id") else None
                kwargs["run_id"] = str(trial.run_id) if hasattr(trial, "run_id") else None
                kwargs["trial_id"] = str(trial.id)
            if tool_call_id is not None:
                kwargs["tool_call_id"] = tool_call_id

            if hasattr(self.tool_executor, "execute_tool"):
                sig = inspect.signature(self.tool_executor.execute_tool)
                valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())}
                res = self.tool_executor.execute_tool(tool_name, arguments, **valid_kwargs)
                if inspect.isawaitable(res):
                    return await res
                return res
            elif hasattr(self.tool_executor, "execute"):
                sig = inspect.signature(self.tool_executor.execute)
                valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())}
                res = self.tool_executor.execute(tool_name, arguments, **valid_kwargs)
                if inspect.isawaitable(res):
                    return await res
                return res
            elif callable(self.tool_executor):
                res = self.tool_executor(tool_name, arguments)
                if inspect.isawaitable(res):
                    return await res
                return res

        # Check if environment carries an mcp_server reference
        if hasattr(env, "mcp_server") and getattr(env, "mcp_server") is not None:
            return await env.mcp_server.execute_tool(tool_name, arguments)

        return {"status": "success"}

    # ── Loop Detection ─────────────────────────────────────────────────────

    def _is_tool_loop(
        self,
        current_signature: tuple[str, str],
        history: list[tuple[str, str]],
    ) -> bool:
        """Detect repeated invocation of equivalent tool calls without progress."""
        if len(history) < self.loop_detection_threshold:
            return False

        # Condition 1: Consecutive identical tool calls >= threshold
        recent = history[-self.loop_detection_threshold:]
        if all(sig == current_signature for sig in recent):
            return True

        # Condition 2: Total count of this exact tool call >= threshold
        if history.count(current_signature) >= self.loop_detection_threshold:
            return True

        return False

    # ── Context Extraction (Strict Public Contract) ─────────────────────────

    def _extract_public_prompt(
        self,
        task: Task | None = None,
        task_input: str | None = None,
    ) -> str:
        """Extract only the public view of the task.

        Hidden invariants and hidden test cases are NEVER exposed to the agent.
        """
        if task_input is not None:
            return task_input

        if task is not None:
            public_view = task.get_public_view()
            parts: list[str] = []
            if public_view.get("input_prompt"):
                parts.append(public_view["input_prompt"])
            elif public_view.get("name"):
                parts.append(f"Task: {public_view['name']}")
                if public_view.get("description"):
                    parts.append(public_view["description"])
            return "\n\n".join(parts) if parts else "Complete the task."

        return "Assist the user with their request."

    # ── Termination & Error Handlers ───────────────────────────────────────

    def _handle_cancellation(self, trace: AgentTrace, trial: Trial, iteration: int) -> None:
        msg = "Agent execution was cancelled"
        trace.termination_reason = TerminationReason.CANCELLED
        trace.add_event(
            ExecutionEvent(
                iteration=iteration,
                event_type=ExecutionEventType.TERMINATION,
                error=msg,
                termination_reason=TerminationReason.CANCELLED,
            )
        )
        trial.error_message = msg
        raise AgentRuntimeError(msg)

    def _handle_max_iterations(self, trace: AgentTrace, trial: Trial, iteration: int) -> None:
        msg = f"Exceeded max iterations: {self.max_iterations}"
        trace.termination_reason = TerminationReason.MAX_ITERATIONS
        trace.add_event(
            ExecutionEvent(
                iteration=iteration,
                event_type=ExecutionEventType.TERMINATION,
                error=msg,
                termination_reason=TerminationReason.MAX_ITERATIONS,
            )
        )
        trial.error_message = msg
        raise AgentRuntimeError(msg)

    def _handle_max_tool_calls(self, trace: AgentTrace, trial: Trial, iteration: int) -> None:
        msg = f"Exceeded max tool calls: {self.max_tool_calls}"
        trace.termination_reason = TerminationReason.MAX_TOOL_CALLS
        trace.add_event(
            ExecutionEvent(
                iteration=iteration,
                event_type=ExecutionEventType.TERMINATION,
                error=msg,
                termination_reason=TerminationReason.MAX_TOOL_CALLS,
            )
        )
        trial.error_message = msg
        raise AgentRuntimeError(msg)

    def _handle_budget_exceeded(self, trace: AgentTrace, trial: Trial, iteration: int) -> None:
        msg = f"Exceeded budget: {self.max_budget}"
        trace.termination_reason = TerminationReason.BUDGET_EXCEEDED
        trace.add_event(
            ExecutionEvent(
                iteration=iteration,
                event_type=ExecutionEventType.TERMINATION,
                error=msg,
                termination_reason=TerminationReason.BUDGET_EXCEEDED,
            )
        )
        trial.error_message = msg
        raise AgentRuntimeError(msg)

    def _handle_forbidden_tool(
        self, trace: AgentTrace, trial: Trial, iteration: int, tool_name: str
    ) -> None:
        msg = f"Tool {tool_name} is forbidden in this trial."
        trace.termination_reason = TerminationReason.FORBIDDEN_TOOL
        trace.add_event(
            ExecutionEvent(
                iteration=iteration,
                event_type=ExecutionEventType.TERMINATION,
                tool_name=tool_name,
                error=msg,
                termination_reason=TerminationReason.FORBIDDEN_TOOL,
            )
        )
        trial.error_message = msg
        raise AgentRuntimeError(msg)

    def _handle_tool_loop(
        self,
        trace: AgentTrace,
        trial: Trial,
        iteration: int,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        msg = (
            f"Tool loop detected: tool '{tool_name}' invoked repeatedly "
            f"with equivalent arguments {arguments}"
        )
        trace.termination_reason = TerminationReason.TOOL_LOOP_DETECTED
        trace.add_event(
            ExecutionEvent(
                iteration=iteration,
                event_type=ExecutionEventType.TERMINATION,
                tool_name=tool_name,
                tool_arguments=arguments,
                error=msg,
                termination_reason=TerminationReason.TOOL_LOOP_DETECTED,
            )
        )
        trial.error_message = msg
        raise AgentRuntimeError(msg)

    def _handle_malformed_tool_call(
        self, trace: AgentTrace, trial: Trial, iteration: int, detail: str
    ) -> None:
        msg = f"Malformed tool call: {detail}"
        trace.termination_reason = TerminationReason.TOOL_ERROR
        trace.add_event(
            ExecutionEvent(
                iteration=iteration,
                event_type=ExecutionEventType.TOOL_ERROR,
                error=msg,
                termination_reason=TerminationReason.TOOL_ERROR,
            )
        )
        trial.error_message = msg
        raise AgentRuntimeError(msg)

    def _handle_model_error(
        self,
        trace: AgentTrace,
        trial: Trial,
        iteration: int,
        error: ModelError,
        latency_ms: float,
    ) -> None:
        msg = f"Model error: {error}"
        trace.termination_reason = TerminationReason.MODEL_ERROR
        trace.add_event(
            ExecutionEvent(
                iteration=iteration,
                event_type=ExecutionEventType.MODEL_ERROR,
                latency_ms=latency_ms,
                error=msg,
                termination_reason=TerminationReason.MODEL_ERROR,
            )
        )
        trial.error_message = str(error)

    def _handle_timeout(self, trace: AgentTrace, trial: Trial, iteration: int) -> None:
        msg = f"Agent execution timed out after {self.timeout_seconds} seconds"
        trace.termination_reason = TerminationReason.TIMEOUT
        trace.add_event(
            ExecutionEvent(
                iteration=iteration,
                event_type=ExecutionEventType.TERMINATION,
                error=msg,
                termination_reason=TerminationReason.TIMEOUT,
            )
        )
        trial.error_message = msg
        raise AgentRuntimeError(msg)

    # ── Legacy Compatibility ───────────────────────────────────────────────

    async def _generate_response(self, messages: list[dict]) -> dict[str, Any]:
        """Legacy compatibility shim."""
        typed_messages = []
        for m in messages:
            role_str = m.get("role", "user")
            try:
                role = MessageRole(role_str)
            except ValueError:
                role = MessageRole.USER
            typed_messages.append(ModelMessage(role=role, content=m.get("content")))

        request = ModelRequest(messages=typed_messages)
        response = await self.generate_model_response(request)

        result: dict[str, Any] = {
            "role": "assistant",
            "content": response.text_content or "",
            "usage": {"total_tokens": response.usage.total_tokens},
        }
        if response.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in response.tool_calls
            ]
        return result
