"""Retry and timeout utilities for tool calls and LLM requests.

Uses tenacity for configurable exponential backoff.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, TypeVar

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.config import settings

T = TypeVar("T")


class ToolTimeoutError(Exception):
    """Raised when a tool call exceeds its time budget."""


class MaxRetriesExceeded(Exception):
    """Raised when all retry attempts are exhausted."""


def with_retry(
    max_attempts: int | None = None,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator factory for async functions with exponential backoff.

    Args:
        max_attempts: Override for settings.max_retries.
        retry_on: Exception types that trigger a retry.
    """
    attempts = max_attempts or settings.max_retries

    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type(retry_on),
        reraise=True,
        before_sleep=lambda state: logger.warning(
            "retry",
            attempt=state.attempt_number,
            fn=state.fn.__name__ if state.fn else "unknown",
        ),
    )


async def with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout_seconds: float | None = None,
    label: str = "operation",
) -> T:
    """Run a coroutine with a timeout, raising ToolTimeoutError on expiry."""
    timeout = timeout_seconds or settings.tool_timeout_seconds
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise ToolTimeoutError(f"{label} timed out after {timeout}s") from exc
