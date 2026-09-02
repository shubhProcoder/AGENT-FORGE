"""Agent domain models."""

from __future__ import annotations

import uuid
from typing import Any
from pydantic import Field

from backend.domain.base import DomainEntity


class Agent(DomainEntity):
    """An AI agent defined by its model and system instructions."""
    project_id: uuid.UUID
    name: str
    description: str = ""
    model_name: str
    system_prompt: str
    allowed_tools: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
