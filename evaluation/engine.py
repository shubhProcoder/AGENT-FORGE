"""Scoring engine — aggregates verifier results into an overall score.

Implements a weighted linear combination of metric scores.
Weights are configurable per evaluation definition.
"""

from __future__ import annotations

from typing import Any

from evaluation.metrics import EvaluationScore, VerificationOutcome

# ── Default weights (sum to 1.0) ─────────────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "functional_correctness": 0.35,
    "tool_correctness": 0.20,
    "state_correctness": 0.15,
    "safety": 0.10,
    "latency": 0.10,
    "cost": 0.05,
    "reliability": 0.05,
}


def compute_score(
    outcome: VerificationOutcome,
    weights: dict[str, float] | None = None,
) -> EvaluationScore:
    """Compute the overall evaluation score from verification metrics.

    Args:
        outcome: Result of running all verifiers.
        weights: Optional weight overrides (metric_name → float).

    Returns:
        EvaluationScore with the weighted overall and per-metric breakdown.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    # Normalise weights so they sum to 1.0
    total_weight = sum(w.values()) or 1.0
    normalised = {k: v / total_weight for k, v in w.items()}

    # Compute weighted sum — missing metrics count as 1.0 (benefit of doubt)
    overall = 0.0
    metric_scores: dict[str, float] = {}

    for metric_name, weight in normalised.items():
        score = outcome.metrics.get(metric_name, 1.0)
        metric_scores[metric_name] = round(score, 4)
        overall += weight * score

    return EvaluationScore(
        overall=round(overall, 4),
        metric_scores=metric_scores,
        failures=outcome.failures,
    )


def detect_regressions(
    current: EvaluationScore,
    previous: EvaluationScore,
    threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Compare two evaluation scores and flag metrics that regressed.

    A regression is detected when a metric drops by more than `threshold`.
    """
    regressions: list[dict[str, Any]] = []
    for metric, curr_val in current.metric_scores.items():
        prev_val = previous.metric_scores.get(metric)
        if prev_val is not None and (prev_val - curr_val) > threshold:
            regressions.append(
                {
                    "metric": metric,
                    "previous": round(prev_val, 4),
                    "current": round(curr_val, 4),
                    "delta": round(prev_val - curr_val, 4),
                }
            )
    return regressions
