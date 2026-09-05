"""Structured JSON logger using loguru.

Configures a consistent log format with trace/request IDs
so every log line is correlatable across the entire trial.
"""

from __future__ import annotations

import sys

from loguru import logger

from backend.config import settings

# Remove default handler and re-add with JSON serialization
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{message} | {extra}"
    ),
    serialize=False,  # Set True for pure JSON output in production
)


def get_logger(name: str = "agentforge") -> logger.__class__:
    """Return a contextualised logger with the given component name."""
    return logger.bind(component=name)
