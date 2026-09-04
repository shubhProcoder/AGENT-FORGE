"""Result and Verification domain models."""

from __future__ import annotations

import uuid
from typing import Any
from pydantic import Field

from backend.domain.base import DomainEntity


class VerificationResult(DomainEntity):
    """The outcome of running verifiers on a Trial, with public vs hidden test breakdown."""

    trial_id: uuid.UUID

    passed: bool
    overall_score: float

    # Separate public vs hidden test grading
    public_passed: bool = True
    public_score: float = 1.0
    hidden_passed: bool = True
    hidden_score: float = 1.0

    metrics: dict[str, float] = Field(default_factory=dict)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    hidden_failures: list[str] = Field(default_factory=list)


class EvaluationResult(DomainEntity):
    """The canonical outcome of evaluating a candidate solution on a task."""

    trial_id: uuid.UUID | None = None
    task_id: uuid.UUID | None = None

    passed: bool
    overall_score: float

    # Separate public vs hidden test grading
    public_passed: bool = True
    public_score: float = 1.0
    hidden_passed: bool = True
    hidden_score: float = 1.0

    failure_type: str | None = None
    failure_category: str | None = None

    public_failures: list[dict[str, Any]] = Field(default_factory=list)
    hidden_failures: list[dict[str, Any]] = Field(default_factory=list)

    metrics: dict[str, float] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    artifact_references: list[str] = Field(default_factory=list)

