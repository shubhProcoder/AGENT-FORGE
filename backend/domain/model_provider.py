"""Model Provider domain models and protocol.

This module defines the provider-agnostic interface for LLM interaction.
All types here are pure domain/application types — no vendor SDK types leak
into or out of this module.

Architecture:
    Control Plane
        ↓
    Run Manager
        ↓
    Trial Manager
        ↓
    Agent Runtime
        ↓
    ModelProvider   ← this module defines the protocol
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class MessageRole(str, Enum):
    """Valid roles for a conversation message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Why the model stopped generating."""

    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Request-side models
# ─────────────────────────────────────────────────────────────────────────────


class ModelMessage(BaseModel):
    """A single message in a conversation history."""

    role: MessageRole
    content: str | None = None
    tool_calls: list["ToolCallRequest"] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    model_config = {"use_enum_values": True}


class ToolDefinition(BaseModel):
    """Description of a callable tool exposed to the model."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelRequest(BaseModel):
    """Everything needed to call a model provider for one generation turn."""

    messages: list[ModelMessage]
    tools: list[ToolDefinition] = Field(default_factory=list)
    model_name: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_seconds: float | None = None
    correlation_id: uuid.UUID = Field(default_factory=uuid.uuid4)


# ─────────────────────────────────────────────────────────────────────────────
# Response-side models
# ─────────────────────────────────────────────────────────────────────────────


class ToolCallRequest(BaseModel):
    """A single tool call emitted by the model."""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class UsageMetadata(BaseModel):
    """Token consumption and estimated cost for one model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ModelResponse(BaseModel):
    """The structured output of a single model generation call."""

    text_content: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: FinishReason = FinishReason.UNKNOWN
    model_id: str = ""
    usage: UsageMetadata = Field(default_factory=UsageMetadata)
    correlation_id: uuid.UUID | None = None
    raw_response: Any | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_final_answer(self) -> bool:
        return self.text_content is not None and not self.tool_calls


# ─────────────────────────────────────────────────────────────────────────────
# Typed error hierarchy
# ─────────────────────────────────────────────────────────────────────────────


class ModelError(Exception):
    """Base class for all model provider errors.

    Raw vendor SDK exceptions must NOT propagate past the provider adapter.
    """

    def __init__(
        self,
        message: str,
        model: str = "",
        correlation_id: uuid.UUID | None = None,
        raw_error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.correlation_id = correlation_id
        self.raw_error = raw_error

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(message={str(self)!r}, "
            f"model={self.model!r}, correlation_id={self.correlation_id})"
        )


class ModelTimeoutError(ModelError):
    """Provider request exceeded the configured timeout."""


class ModelAuthenticationError(ModelError):
    """Missing, invalid, or rejected API key / credentials."""


class ModelRateLimitError(ModelError):
    """Provider rate limit or quota exceeded."""


class ModelFormatError(ModelError):
    """Malformed provider response (e.g. invalid JSON in tool call arguments)."""


class ModelUnavailableError(ModelError):
    """Connection failure or provider 5xx server error."""


# ─────────────────────────────────────────────────────────────────────────────
# Protocol
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class ModelProvider(Protocol):
    """Provider-agnostic interface for LLM interaction.

    Implementations MUST:
    - Be async.
    - Return a ModelResponse on success.
    - Raise a ModelError subclass on failure (never leak raw SDK exceptions).
    """

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response for the given request."""
        ...


__all__ = [
    "MessageRole",
    "FinishReason",
    "ModelMessage",
    "ToolDefinition",
    "ModelRequest",
    "ToolCallRequest",
    "UsageMetadata",
    "ModelResponse",
    "ModelError",
    "ModelTimeoutError",
    "ModelAuthenticationError",
    "ModelRateLimitError",
    "ModelFormatError",
    "ModelUnavailableError",
    "ModelProvider",
]
