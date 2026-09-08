"""Unit tests for AgentForge TASK 12.1: Model Provider Abstraction.

Test coverage:
1. MockModelProvider — deterministic text response
2. MockModelProvider — tool-call response
3. MockModelProvider — multiple scripted responses (round-robin)
4. MockModelProvider — simulate timeout error
5. MockModelProvider — simulate auth error
6. MockModelProvider — simulate format error
7. MockModelProvider — call recording (request assertions)
8. OpenAIProvider — instantiation from config without real API key
9. OpenAIProvider — translates APITimeoutError → ModelTimeoutError
10. OpenAIProvider — translates AuthenticationError → ModelAuthenticationError
11. OpenAIProvider — translates RateLimitError → ModelRateLimitError
12. OpenAIProvider — translates APIConnectionError → ModelUnavailableError
13. OpenAIProvider — malformed tool-call JSON → ModelFormatError
14. OpenAIProvider — valid tool-call response parsing
15. OpenAIProvider — usage metadata extraction
16. AgentRuntime — accepts MockModelProvider and receives typed ModelResponse
17. AgentRuntime — default-constructs MockModelProvider when none supplied
18. AgentRuntime — generate_model_response returns ModelResponse
19. ModelProvider protocol — structural subtyping check
20. Domain models — ModelMessage, ToolDefinition, ModelRequest round-trip
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.domain.model_provider import (
    FinishReason,
    MessageRole,
    ModelAuthenticationError,
    ModelError,
    ModelFormatError,
    ModelMessage,
    ModelProvider,
    ModelRateLimitError,
    ModelRequest,
    ModelResponse,
    ModelTimeoutError,
    ModelUnavailableError,
    ToolCallRequest,
    ToolDefinition,
    UsageMetadata,
)
from backend.infrastructure.llm.mock_provider import MockModelProvider
from backend.infrastructure.llm.openai_adapter import OpenAIProvider, ProviderConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _simple_request(text: str = "Hello") -> ModelRequest:
    return ModelRequest(
        messages=[ModelMessage(role=MessageRole.USER, content=text)]
    )


def _request_with_tools() -> ModelRequest:
    return ModelRequest(
        messages=[ModelMessage(role=MessageRole.USER, content="Search for order #123")],
        tools=[
            ToolDefinition(
                name="search_customer",
                description="Search customers",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1–7  MockModelProvider
# ─────────────────────────────────────────────────────────────────────────────


class TestMockModelProvider:
    @pytest.mark.asyncio
    async def test_deterministic_text_response(self):
        """Provider always returns the scripted text answer."""
        provider = MockModelProvider.with_text("Task completed successfully.")
        response = await provider.generate(_simple_request())

        assert isinstance(response, ModelResponse)
        assert response.text_content == "Task completed successfully."
        assert response.finish_reason == FinishReason.STOP
        assert not response.has_tool_calls
        assert response.is_final_answer

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        """Provider returns a typed tool call without network."""
        provider = MockModelProvider.with_tool_call(
            name="search_customer",
            arguments={"query": "john@example.com"},
            call_id="call_test_001",
        )
        response = await provider.generate(_request_with_tools())

        assert isinstance(response, ModelResponse)
        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert isinstance(tc, ToolCallRequest)
        assert tc.name == "search_customer"
        assert tc.arguments == {"query": "john@example.com"}
        assert tc.id == "call_test_001"
        assert response.finish_reason == FinishReason.TOOL_CALLS

    @pytest.mark.asyncio
    async def test_multiple_scripted_responses_round_robin(self):
        """Multiple scripted responses cycle in order."""
        r1 = ModelResponse(text_content="First", finish_reason=FinishReason.STOP)
        r2 = ModelResponse(text_content="Second", finish_reason=FinishReason.STOP)
        provider = MockModelProvider(responses=[r1, r2])

        resp_a = await provider.generate(_simple_request())
        resp_b = await provider.generate(_simple_request())
        resp_c = await provider.generate(_simple_request())  # wraps back to r1

        assert resp_a.text_content == "First"
        assert resp_b.text_content == "Second"
        assert resp_c.text_content == "First"

    @pytest.mark.asyncio
    async def test_simulate_timeout_error(self):
        """Provider raises ModelTimeoutError when configured."""
        provider = MockModelProvider(
            simulate_error=ModelTimeoutError("simulated timeout")
        )
        with pytest.raises(ModelTimeoutError, match="simulated timeout"):
            await provider.generate(_simple_request())

    @pytest.mark.asyncio
    async def test_simulate_auth_error(self):
        """Provider raises ModelAuthenticationError when configured."""
        provider = MockModelProvider(
            simulate_error=ModelAuthenticationError("bad key")
        )
        with pytest.raises(ModelAuthenticationError):
            await provider.generate(_simple_request())

    @pytest.mark.asyncio
    async def test_simulate_format_error(self):
        """Provider raises ModelFormatError when configured."""
        provider = MockModelProvider(
            simulate_error=ModelFormatError("invalid json")
        )
        with pytest.raises(ModelFormatError):
            await provider.generate(_simple_request())

    @pytest.mark.asyncio
    async def test_call_recording(self):
        """Provider records all received requests for test assertions."""
        provider = MockModelProvider.with_text("ok")
        req1 = _simple_request("Hello")
        req2 = _simple_request("World")

        await provider.generate(req1)
        await provider.generate(req2)

        assert provider.call_count == 2
        assert len(provider.received_requests) == 2
        assert provider.received_requests[0].messages[0].content == "Hello"
        assert provider.received_requests[1].messages[0].content == "World"

    @pytest.mark.asyncio
    async def test_usage_metadata_defaults(self):
        """MockModelProvider stamps default usage metadata on responses."""
        provider = MockModelProvider.with_text("done")
        response = await provider.generate(_simple_request())

        assert response.usage.total_tokens > 0
        assert response.usage.prompt_tokens >= 0
        assert response.usage.estimated_cost_usd >= 0.0

    @pytest.mark.asyncio
    async def test_correlation_id_propagated(self):
        """Response carries the same correlation_id as the request."""
        cid = uuid.uuid4()
        req = ModelRequest(
            messages=[ModelMessage(role=MessageRole.USER, content="hi")],
            correlation_id=cid,
        )
        provider = MockModelProvider.with_text("hi back")
        response = await provider.generate(req)

        assert response.correlation_id == cid

    @pytest.mark.asyncio
    async def test_response_factory_mode(self):
        """Factory callable receives the request and returns a response."""
        def factory(req: ModelRequest) -> ModelResponse:
            content = req.messages[-1].content or ""
            return ModelResponse(
                text_content=f"Echo: {content}",
                finish_reason=FinishReason.STOP,
            )

        provider = MockModelProvider(response_factory=factory)
        response = await provider.generate(_simple_request("ping"))

        assert response.text_content == "Echo: ping"

    @pytest.mark.asyncio
    async def test_no_network_access_required(self):
        """MockModelProvider works entirely in-memory — no network or env vars needed."""
        # If this test passes, no network calls were made (no real API was hit)
        provider = MockModelProvider.with_text("offline ok")
        response = await provider.generate(_simple_request())
        assert response.text_content == "offline ok"


# ─────────────────────────────────────────────────────────────────────────────
# 8–15  OpenAIProvider (no real API key required)
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenAIProvider:
    def test_instantiation_without_api_key(self):
        """OpenAIProvider can be constructed without a real API key."""
        config = ProviderConfig(api_key=None, model_name="gpt-4o-mini")
        provider = OpenAIProvider(config=config)
        assert provider.config.model_name == "gpt-4o-mini"
        assert provider._client is None  # lazy — not yet created

    def test_instantiation_from_settings(self):
        """from_settings() builds a ProviderConfig from backend.config.settings."""
        config = ProviderConfig.from_settings()
        assert isinstance(config, ProviderConfig)
        assert config.model_name  # non-empty

    @pytest.mark.asyncio
    async def test_timeout_error_translation(self):
        """APITimeoutError → ModelTimeoutError (no raw openai exception leaks)."""
        import openai

        config = ProviderConfig(api_key="sk-fake", model_name="gpt-4o-mini")
        provider = OpenAIProvider(config=config)

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = openai.APITimeoutError(
            request=MagicMock()
        )
        provider._client = mock_client

        with pytest.raises(ModelTimeoutError) as exc_info:
            await provider.generate(_simple_request())

        assert isinstance(exc_info.value, ModelError)
        assert exc_info.value.raw_error is not None

    @pytest.mark.asyncio
    async def test_auth_error_translation(self):
        """AuthenticationError → ModelAuthenticationError."""
        import openai

        config = ProviderConfig(api_key="sk-bad", model_name="gpt-4o-mini")
        provider = OpenAIProvider(config=config)

        mock_response = MagicMock()
        mock_response.request = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}},
        )
        provider._client = mock_client

        with pytest.raises(ModelAuthenticationError):
            await provider.generate(_simple_request())

    @pytest.mark.asyncio
    async def test_rate_limit_error_translation(self):
        """RateLimitError → ModelRateLimitError."""
        import openai

        config = ProviderConfig(api_key="sk-ok", model_name="gpt-4o-mini")
        provider = OpenAIProvider(config=config)

        mock_response = MagicMock()
        mock_response.request = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}},
        )
        provider._client = mock_client

        with pytest.raises(ModelRateLimitError):
            await provider.generate(_simple_request())

    @pytest.mark.asyncio
    async def test_connection_error_translation(self):
        """APIConnectionError → ModelUnavailableError."""
        import openai

        config = ProviderConfig(api_key="sk-ok", model_name="gpt-4o-mini")
        provider = OpenAIProvider(config=config)

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )
        provider._client = mock_client

        with pytest.raises(ModelUnavailableError):
            await provider.generate(_simple_request())

    @pytest.mark.asyncio
    async def test_malformed_tool_call_json_raises_format_error(self):
        """Invalid JSON in tool call arguments → ModelFormatError."""
        config = ProviderConfig(api_key="sk-ok", model_name="gpt-4o-mini")
        provider = OpenAIProvider(config=config)

        # Build a mock OpenAI response with broken tool argument JSON
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc123"
        mock_tool_call.function.name = "search_customer"
        mock_tool_call.function.arguments = "{INVALID_JSON"

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message = mock_message

        mock_raw = MagicMock()
        mock_raw.choices = [mock_choice]
        mock_raw.usage = None
        mock_raw.model = "gpt-4o-mini"
        mock_raw.id = "chatcmpl-test"
        mock_raw.object = "chat.completion"

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_raw
        provider._client = mock_client

        with pytest.raises(ModelFormatError, match="invalid JSON"):
            await provider.generate(_simple_request())

    @pytest.mark.asyncio
    async def test_valid_tool_call_parsing(self):
        """Valid tool call response is parsed into ToolCallRequest objects."""
        import json

        config = ProviderConfig(api_key="sk-ok", model_name="gpt-4o-mini")
        provider = OpenAIProvider(config=config)

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_valid_001"
        mock_tool_call.function.name = "search_customer"
        mock_tool_call.function.arguments = json.dumps({"query": "alice@example.com"})

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message = mock_message

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 25
        mock_usage.completion_tokens = 15
        mock_usage.total_tokens = 40

        mock_raw = MagicMock()
        mock_raw.choices = [mock_choice]
        mock_raw.usage = mock_usage
        mock_raw.model = "gpt-4o-mini"
        mock_raw.id = "chatcmpl-valid"
        mock_raw.object = "chat.completion"

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_raw
        provider._client = mock_client

        response = await provider.generate(_request_with_tools())

        assert isinstance(response, ModelResponse)
        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.name == "search_customer"
        assert tc.arguments == {"query": "alice@example.com"}
        assert tc.id == "call_valid_001"

    @pytest.mark.asyncio
    async def test_usage_metadata_extraction(self):
        """Usage tokens and cost are correctly extracted from the raw response."""
        config = ProviderConfig(api_key="sk-ok", model_name="gpt-4o-mini")
        provider = OpenAIProvider(config=config)

        mock_message = MagicMock()
        mock_message.content = "Here is your answer."
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message = mock_message

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150

        mock_raw = MagicMock()
        mock_raw.choices = [mock_choice]
        mock_raw.usage = mock_usage
        mock_raw.model = "gpt-4o-mini"
        mock_raw.id = "chatcmpl-usage"
        mock_raw.object = "chat.completion"

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_raw
        provider._client = mock_client

        response = await provider.generate(_simple_request())

        assert response.usage.prompt_tokens == 100
        assert response.usage.completion_tokens == 50
        assert response.usage.total_tokens == 150
        assert response.usage.estimated_cost_usd > 0
        assert response.model_id == "gpt-4o-mini"
        assert response.finish_reason == FinishReason.STOP


# ─────────────────────────────────────────────────────────────────────────────
# 16–18  AgentRuntime dependency injection
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentRuntimeProviderInjection:
    @pytest.mark.asyncio
    async def test_inject_mock_provider_returns_typed_response(self):
        """AgentRuntime.generate_model_response() returns a ModelResponse when
        MockModelProvider is injected — zero network calls."""
        from backend.application.agent_runtime import AgentRuntime

        provider = MockModelProvider.with_text("I have completed the task.")
        runtime = AgentRuntime(model_provider=provider)

        request = _simple_request("Process order #42")
        response = await runtime.generate_model_response(request)

        assert isinstance(response, ModelResponse)
        assert response.text_content == "I have completed the task."
        assert response.is_final_answer
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_default_constructs_mock_when_no_provider_given(self):
        """AgentRuntime creates a MockModelProvider by default — no credentials needed."""
        from backend.application.agent_runtime import AgentRuntime
        from backend.infrastructure.llm.mock_provider import MockModelProvider as MP

        runtime = AgentRuntime()

        assert isinstance(runtime.model_provider, MP)

    @pytest.mark.asyncio
    async def test_generate_model_response_propagates_provider_error(self):
        """ModelError raised by the provider propagates unchanged from AgentRuntime."""
        from backend.application.agent_runtime import AgentRuntime

        provider = MockModelProvider(simulate_error=ModelTimeoutError("provider timed out"))
        runtime = AgentRuntime(model_provider=provider)

        with pytest.raises(ModelTimeoutError, match="provider timed out"):
            await runtime.generate_model_response(_simple_request())

    @pytest.mark.asyncio
    async def test_runtime_execute_uses_provider(self):
        """AgentRuntime.execute() calls the injected provider, increments model_calls."""
        from backend.application.agent_runtime import AgentRuntime
        from backend.domain.trial import Trial
        from backend.domain.environment import Environment, EnvironmentStatus

        provider = MockModelProvider.with_text("Task done.")
        runtime = AgentRuntime(model_provider=provider)

        trial = Trial(
            run_id=uuid.uuid4(),
            task_id=uuid.uuid4(),
        )
        env = Environment(trial_id=trial.id, status=EnvironmentStatus.ACTIVE)

        await runtime.execute(trial, env, allowed_tools=[])

        assert trial.model_calls >= 1
        assert trial.is_completed is True
        assert provider.call_count >= 1

    @pytest.mark.asyncio
    async def test_runtime_token_accounting(self):
        """AgentRuntime accumulates total_tokens from each ModelResponse."""
        from backend.application.agent_runtime import AgentRuntime
        from backend.domain.trial import Trial
        from backend.domain.environment import Environment, EnvironmentStatus

        usage = UsageMetadata(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        provider = MockModelProvider(
            responses=[
                ModelResponse(
                    text_content="Done.",
                    finish_reason=FinishReason.STOP,
                    usage=usage,
                )
            ]
        )
        runtime = AgentRuntime(model_provider=provider)
        trial = Trial(run_id=uuid.uuid4(), task_id=uuid.uuid4())
        env = Environment(trial_id=trial.id, status=EnvironmentStatus.ACTIVE)

        await runtime.execute(trial, env, allowed_tools=[])

        assert trial.total_tokens == 30


# ─────────────────────────────────────────────────────────────────────────────
# 19  Protocol structural subtyping
# ─────────────────────────────────────────────────────────────────────────────


class TestModelProviderProtocol:
    def test_mock_provider_satisfies_protocol(self):
        """MockModelProvider satisfies the ModelProvider Protocol at runtime."""
        provider = MockModelProvider.with_text("ok")
        assert isinstance(provider, ModelProvider)

    def test_openai_provider_satisfies_protocol(self):
        """OpenAIProvider satisfies the ModelProvider Protocol at runtime."""
        provider = OpenAIProvider(config=ProviderConfig(api_key=None))
        assert isinstance(provider, ModelProvider)

    def test_arbitrary_object_with_generate_satisfies_protocol(self):
        """Any object with an async generate() method satisfies the protocol."""

        class CustomProvider:
            async def generate(self, request: ModelRequest) -> ModelResponse:
                return ModelResponse(text_content="custom")

        assert isinstance(CustomProvider(), ModelProvider)

    def test_object_without_generate_fails_protocol(self):
        """Objects lacking generate() do NOT satisfy the protocol."""

        class NotAProvider:
            def query(self, x: str) -> str:
                return x

        assert not isinstance(NotAProvider(), ModelProvider)


# ─────────────────────────────────────────────────────────────────────────────
# 20  Domain model round-trips
# ─────────────────────────────────────────────────────────────────────────────


class TestDomainModels:
    def test_model_message_role_serialization(self):
        """MessageRole values are serialized as strings in Pydantic."""
        msg = ModelMessage(role=MessageRole.ASSISTANT, content="hello")
        data = msg.model_dump()
        assert data["role"] == "assistant"

    def test_tool_definition_round_trip(self):
        """ToolDefinition serializes and deserializes cleanly."""
        td = ToolDefinition(
            name="refund_order",
            description="Refund an order",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        )
        data = td.model_dump()
        restored = ToolDefinition(**data)
        assert restored.name == td.name
        assert restored.parameters == td.parameters

    def test_model_request_defaults(self):
        """ModelRequest assigns a correlation_id by default."""
        req = ModelRequest(messages=[ModelMessage(role=MessageRole.USER, content="hi")])
        assert req.correlation_id is not None
        assert isinstance(req.correlation_id, uuid.UUID)
        assert req.tools == []

    def test_model_response_properties(self):
        """has_tool_calls and is_final_answer properties work correctly."""
        text_response = ModelResponse(text_content="Done", finish_reason=FinishReason.STOP)
        assert text_response.is_final_answer
        assert not text_response.has_tool_calls

        tool_response = ModelResponse(
            tool_calls=[ToolCallRequest(name="foo", arguments={})],
            finish_reason=FinishReason.TOOL_CALLS,
        )
        assert tool_response.has_tool_calls
        assert not tool_response.is_final_answer

    def test_model_error_hierarchy(self):
        """All typed errors are subclasses of ModelError."""
        for cls in (
            ModelTimeoutError,
            ModelAuthenticationError,
            ModelRateLimitError,
            ModelFormatError,
            ModelUnavailableError,
        ):
            err = cls("test")
            assert isinstance(err, ModelError)
            assert isinstance(err, Exception)

    def test_model_error_attributes(self):
        """ModelError stores model, correlation_id, and raw_error."""
        cid = uuid.uuid4()
        raw = ValueError("raw")
        err = ModelTimeoutError("timed out", model="gpt-4", correlation_id=cid, raw_error=raw)

        assert str(err) == "timed out"
        assert err.model == "gpt-4"
        assert err.correlation_id == cid
        assert err.raw_error is raw
