"""Agent runtime — the core orchestration loop.

Receives a task, calls an LLM, dispatches tool calls, and returns
the final output along with the full execution context (trace).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import openai
from loguru import logger

from agent.state import ExecutionContext
from agent.tool_executor import execute_tool, get_tool_definitions
from backend.config import settings


class AgentRuntime:
    """Orchestrates a single agent trial.

    Lifecycle:
        1. Load system prompt + task input into message list.
        2. Loop: send messages → LLM responds with text or tool_call.
           a. If tool_call → dispatch via tool_executor, append result.
           b. If text (no tool_call) → treat as final answer.
        3. Return final output + execution context.
    """

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str = "",
        allowed_tools: list[str] | None = None,
        max_steps: int | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.model = model or settings.openai_model
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools
        self.max_steps = max_steps or settings.agent_max_steps
        self.timeout_seconds = timeout_seconds or settings.agent_timeout_seconds
        self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    async def run(self, task_input: str) -> tuple[str, ExecutionContext]:
        """Execute the agent loop and return (final_output, context)."""
        ctx = ExecutionContext()

        # Seed the conversation
        if self.system_prompt:
            ctx.messages.append({"role": "system", "content": self.system_prompt})
        ctx.messages.append({"role": "user", "content": task_input})

        # Build tool definitions
        tool_defs = get_tool_definitions()
        if self.allowed_tools:
            tool_defs = [t for t in tool_defs if t["function"]["name"] in self.allowed_tools]

        logger.info("agent_start", trace_id=ctx.trace_id, model=self.model)

        final_output = ""

        for step in range(1, self.max_steps + 1):
            logger.debug(f"step={step}", trace_id=ctx.trace_id)

            # ── Call LLM ─────────────────────────────────────────────────
            chat_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": ctx.messages,
            }
            if tool_defs:
                chat_kwargs["tools"] = tool_defs
                chat_kwargs["tool_choice"] = "auto"

            response = await self._client.chat.completions.create(**chat_kwargs)
            choice = response.choices[0]
            message = choice.message

            # Track token usage
            if response.usage:
                ctx.prompt_tokens += response.usage.prompt_tokens
                ctx.completion_tokens += response.usage.completion_tokens
                ctx.total_tokens += response.usage.total_tokens

            # ── Tool call branch ─────────────────────────────────────────
            if message.tool_calls:
                # Append the assistant message (with tool_calls) to history
                ctx.messages.append(message.model_dump())

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    logger.info(
                        "tool_dispatch",
                        trace_id=ctx.trace_id,
                        tool=fn_name,
                        step=step,
                    )

                    try:
                        result = await execute_tool(fn_name, fn_args, ctx)
                    except Exception as exc:
                        result = {"error": str(exc)}

                    # Feed tool result back to LLM
                    ctx.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )
                continue  # next step — LLM will process tool results

            # ── Final answer branch ──────────────────────────────────────
            final_output = message.content or ""
            ctx.messages.append({"role": "assistant", "content": final_output})
            break

        # Estimate cost (rough: $0.15 / 1M input, $0.60 / 1M output for gpt-4o-mini)
        ctx.estimated_cost_usd = round(
            (ctx.prompt_tokens * 0.00000015) + (ctx.completion_tokens * 0.0000006),
            6,
        )

        logger.info(
            "agent_finish",
            trace_id=ctx.trace_id,
            steps=ctx.step,
            tokens=ctx.total_tokens,
            cost_usd=ctx.estimated_cost_usd,
            elapsed=ctx.elapsed_seconds,
        )

        return final_output, ctx
