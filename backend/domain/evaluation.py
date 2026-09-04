"""Evaluation domain models."""

from __future__ import annotations

import uuid
from typing import Any
from pydantic import Field

from backend.domain.base import DomainEntity


class Evaluation(DomainEntity):
    """How to test: ties an Agent to a Dataset with specific scoring rules."""
    project_id: uuid.UUID
    name: str
    description: str = ""
    agent_id: uuid.UUID
    dataset_id: uuid.UUID
    
    # Configuration
    scoring_rules: dict[str, Any] = Field(default_factory=dict)
    verifier_set: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
