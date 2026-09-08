"""Mock Model Provider for deterministic testing.

This provider is designed for unit tests that must run:
  - Without any network access
  - Without any API credentials
  - With fully deterministic, scripted responses

Usage::

    from backend.infrastructure.llm.mock_provider import MockModelProvider
    from backend.domain.model_provider import ModelResponse, UsageMetadata, FinishReason

    # Simple text responses
    provider = MockModelProvider(responses=[
        ModelResponse(text_content="Hello world", finish_reason=FinishReason.STOP),
    ])
    response = await provider.generate(request)
    assert response.text_content == "Hello world"

    # Simulate errors
    from backend.domain.model_provider import ModelTimeoutError
    provider = MockModelProvider(simulate_error=ModelTimeoutError("timeout"))
    with pytest.raises(ModelTimeoutError):
        await provider.generate(request)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable

from backend.domain.model_provider import (
    FinishReason,
    ModelError,
    ModelFormatError,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    UsageMetadata,
)

logger = logging.getLogger(__name__)


class MockModelProvider:
    """Deterministic mock provider for use in unit tests.

    Supports three modes (evaluated in priority order):
    1. ``simulate_error``: Always raises the given exception.
    2. ``response_factory``: Calls the factory callable on each generate() call.
    3. ``responses``: Cycles through a scripted queue; wraps around when exhausted.
    """

    def __init__(
        self,
        responses: list[ModelResponse] | None = None,
        response_factory: Callable[[ModelRequest], ModelResponse] | None = None,
        simulate_error: ModelError | None = None,
        default_model_id: str = "mock-model-v1",
        default_usage: UsageMetadata | None = None,
    ) -> None:
        self._responses: deque[ModelResponse] = deque(responses or [])
        self._response_factory = response_factory
        self._simulate_error = simulate_error
        self._default_model_id = default_model_id
        self._default_usage = default_usage or UsageMetadata(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            estimated_cost_usd=0.000002,
        )

        # Call recorder for test assertions
        self.received_requests: list[ModelRequest] = []
        self.call_count: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Return the next scripted response (or raise the scripted error).

        Records every incoming request in ``self.received_requests``.
        """
        self.received_requests.append(request)
        self.call_count += 1

        logger.debug(
            "[MockProvider] generate() call #%d, correlation_id=%s",
            self.call_count,
            request.correlation_id,
        )

        # Priority 1: always-error mode
        if self._simulate_error is not None:
            raise self._simulate_error

        # Priority 2: factory mode
        if self._response_factory is not None:
            response = self._response_factory(request)
            return self._stamp(response, request)

        # Priority 3: scripted queue
        if self._responses:
            response = self._responses[0]
            # Rotate: move consumed response to the back (wrap-around)
            self._responses.rotate(-1)
            return self._stamp(response, request)

        # Fallback: generic final answer
        return self._stamp(
            ModelResponse(
                text_content="Mock response: task completed.",
                finish_reason=FinishReason.STOP,
            ),
            request,
        )

    # ── Convenience builder helpers ────────────────────────────────────────

    @classmethod
    def with_text(cls, text: str, **kwargs: Any) -> "MockModelProvider":
        """Shortcut: provider that always returns a text answer."""
        return cls(
            responses=[ModelResponse(text_content=text, finish_reason=FinishReason.STOP)],
            **kwargs,
        )

    @classmethod
    def with_tool_call(
        cls,
        name: str,
        arguments: dict[str, Any],
        call_id: str = "call_mock_001",
        **kwargs: Any,
    ) -> "MockModelProvider":
        """Shortcut: provider that returns a single tool call."""
        return cls(
            responses=[
                ModelResponse(
                    tool_calls=[
                        ToolCallRequest(id=call_id, name=name, arguments=arguments)
                    ],
                    finish_reason=FinishReason.TOOL_CALLS,
                )
            ],
            **kwargs,
        )

    def reset(self) -> None:
        """Reset call history (not the scripted responses)."""
        self.received_requests.clear()
        self.call_count = 0

    # ── Internal ───────────────────────────────────────────────────────────

    def _stamp(self, response: ModelResponse, request: ModelRequest) -> ModelResponse:
        """Fill in default metadata fields if the scripted response left them empty."""
        return response.model_copy(
            update={
                "model_id": response.model_id or self._default_model_id,
                "usage": response.usage
                if response.usage.total_tokens > 0
                else self._default_usage,
                "correlation_id": response.correlation_id or request.correlation_id,
            }
        )
