"""OpenAI Provider Adapter.

Translates between AgentForge's domain types (ModelRequest / ModelResponse)
and the OpenAI Python SDK.  No OpenAI types escape this module.

Design decisions:
- ``ProviderConfig`` is a plain Pydantic model; it reads ``OPENAI_API_KEY`` from
  the environment via ``backend.config.settings`` by default.
- The adapter can be instantiated without a live API key — the key is only
  validated when the first request is actually sent.
- All raw openai exceptions are caught here and re-raised as typed ModelError
  subclasses, so the application layer never sees ``openai.*`` exceptions.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from backend.domain.model_provider import (
    FinishReason,
    MessageRole,
    ModelAuthenticationError,
    ModelError,
    ModelFormatError,
    ModelRateLimitError,
    ModelRequest,
    ModelResponse,
    ModelTimeoutError,
    ModelUnavailableError,
    ToolCallRequest,
    UsageMetadata,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


class ProviderConfig(BaseModel):
    """Configuration for the OpenAI provider adapter.

    All fields have sensible defaults so the adapter boots without a real key.
    The API key is read from the environment (``OPENAI_API_KEY``) or from
    ``settings.openai_api_key`` if not explicitly supplied.
    """

    model_name: str = "gpt-4o-mini"
    api_key: str | None = None            # None → read from env at instantiation time
    timeout: float = 30.0                 # seconds
    max_tokens: int | None = None
    temperature: float | None = None
    max_retries: int = 2

    @classmethod
    def from_settings(cls) -> "ProviderConfig":
        """Build config from ``backend.config.settings``."""
        from backend.config import settings  # local import to avoid circular deps

        return cls(
            model_name=settings.openai_model,
            api_key=settings.openai_api_key or None,
            timeout=float(settings.agent_timeout_seconds),
            max_retries=settings.max_retries,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────


class OpenAIProvider:
    """Implements ModelProvider against the OpenAI Chat Completions API.

    This class intentionally does NOT subclass ``ModelProvider`` — it satisfies
    the protocol via structural typing (duck typing) as required.
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig.from_settings()
        self._client: Any = None  # lazy-initialised on first generate()

    # ── Lazy client init ───────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Return (or create) the async OpenAI client."""
        if self._client is None:
            try:
                import openai  # local import so tests that don't have openai installed can still import this module
            except ImportError as exc:  # pragma: no cover
                raise ModelUnavailableError(
                    "openai package is not installed. "
                    "Add 'openai>=1.30' to your dependencies.",
                    model=self.config.model_name,
                    raw_error=exc,
                )

            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    # ── Main entry point ───────────────────────────────────────────────────

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Translate ModelRequest → OpenAI call → ModelResponse.

        Raises:
            ModelTimeoutError: on APITimeoutError.
            ModelAuthenticationError: on AuthenticationError.
            ModelRateLimitError: on RateLimitError.
            ModelFormatError: when tool-call arguments cannot be parsed.
            ModelUnavailableError: on connection errors or 5xx responses.
        """
        import openai  # noqa: PLC0415 — intentional local import

        client = self._get_client()
        model_name = request.model_name or self.config.model_name
        correlation_id = request.correlation_id

        # Build payload
        payload = self._build_payload(request, model_name)

        logger.debug(
            "[OpenAIProvider] generate() model=%s correlation_id=%s",
            model_name,
            correlation_id,
        )

        try:
            raw = await client.chat.completions.create(**payload)
        except openai.APITimeoutError as exc:
            raise ModelTimeoutError(
                f"OpenAI request timed out after {self.config.timeout}s",
                model=model_name,
                correlation_id=correlation_id,
                raw_error=exc,
            ) from exc
        except openai.AuthenticationError as exc:
            raise ModelAuthenticationError(
                "OpenAI authentication failed — check OPENAI_API_KEY",
                model=model_name,
                correlation_id=correlation_id,
                raw_error=exc,
            ) from exc
        except openai.RateLimitError as exc:
            raise ModelRateLimitError(
                "OpenAI rate limit exceeded",
                model=model_name,
                correlation_id=correlation_id,
                raw_error=exc,
            ) from exc
        except openai.BadRequestError as exc:
            raise ModelFormatError(
                f"OpenAI rejected the request: {exc}",
                model=model_name,
                correlation_id=correlation_id,
                raw_error=exc,
            ) from exc
        except openai.APIConnectionError as exc:
            raise ModelUnavailableError(
                f"Could not connect to OpenAI: {exc}",
                model=model_name,
                correlation_id=correlation_id,
                raw_error=exc,
            ) from exc
        except openai.APIError as exc:
            # Catch-all for other SDK errors
            raise ModelError(
                f"OpenAI API error: {exc}",
                model=model_name,
                correlation_id=correlation_id,
                raw_error=exc,
            ) from exc

        return self._parse_response(raw, model_name, correlation_id)

    # ── Translation helpers ────────────────────────────────────────────────

    def _build_payload(self, request: ModelRequest, model_name: str) -> dict[str, Any]:
        """Convert a ModelRequest into an OpenAI-compatible dict."""
        messages: list[dict[str, Any]] = []

        for msg in request.messages:
            entry: dict[str, Any] = {"role": msg.role if isinstance(msg.role, str) else msg.role.value}

            if msg.content is not None:
                entry["content"] = msg.content

            if msg.tool_call_id is not None:
                entry["tool_call_id"] = msg.tool_call_id

            if msg.name is not None:
                entry["name"] = msg.name

            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            messages.append(entry)

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }

        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": td.name,
                        "description": td.description,
                        "parameters": td.parameters,
                    },
                }
                for td in request.tools
            ]
            payload["tool_choice"] = "auto"

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        elif self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        elif self.config.temperature is not None:
            payload["temperature"] = self.config.temperature

        return payload

    def _parse_response(
        self,
        raw: Any,
        model_name: str,
        correlation_id: uuid.UUID,
    ) -> ModelResponse:
        """Convert an OpenAI ChatCompletion object into a ModelResponse."""
        choice = raw.choices[0]
        message = choice.message

        # ── Finish reason ──────────────────────────────────────────────────
        raw_finish = (choice.finish_reason or "unknown").lower()
        try:
            finish_reason = FinishReason(raw_finish)
        except ValueError:
            finish_reason = FinishReason.UNKNOWN

        # ── Tool calls ─────────────────────────────────────────────────────
        tool_calls: list[ToolCallRequest] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    raise ModelFormatError(
                        f"Tool call '{tc.function.name}' returned invalid JSON arguments: {raw_args!r}",
                        model=model_name,
                        correlation_id=correlation_id,
                        raw_error=exc,
                    ) from exc
                tool_calls.append(
                    ToolCallRequest(id=tc.id, name=tc.function.name, arguments=args)
                )

        # ── Usage ──────────────────────────────────────────────────────────
        usage = UsageMetadata()
        if raw.usage:
            usage = UsageMetadata(
                prompt_tokens=raw.usage.prompt_tokens or 0,
                completion_tokens=raw.usage.completion_tokens or 0,
                total_tokens=raw.usage.total_tokens or 0,
                # gpt-4o-mini pricing: $0.15 / 1M input, $0.60 / 1M output
                estimated_cost_usd=round(
                    (raw.usage.prompt_tokens or 0) * 0.00000015
                    + (raw.usage.completion_tokens or 0) * 0.0000006,
                    8,
                ),
            )

        return ModelResponse(
            text_content=message.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            model_id=raw.model or model_name,
            usage=usage,
            correlation_id=correlation_id,
            raw_response={"id": raw.id, "object": raw.object},
        )
