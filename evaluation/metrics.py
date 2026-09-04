"""Evaluation metric models.

These are pure data containers — no LLM calls, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Single metric score."""

    name: str
    score: float  # 0.0 – 1.0
    details: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationOutcome:
    """Outcome of running all verifiers on one trial."""

    passed: bool
    failures: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def add_metric(self, name: str, score: float, details: str = "") -> None:
        self.metrics[name] = round(score, 4)
        if score < 1.0 and details:
            self.failures.append({"type": name, "detail": details})
        if score < 1.0:
            self.passed = False


@dataclass
class EvaluationScore:
    """Final aggregated score for a trial."""

    overall: float
    metric_scores: dict[str, float] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(self.overall, 4),
            "metrics": self.metric_scores,
            "failures": self.failures,
        }
