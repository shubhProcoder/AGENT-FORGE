"""Dataset Versioning, Snapshotting, and Regression Detection."""

from __future__ import annotations

import copy
import time
import uuid
from typing import Any
from pydantic import BaseModel, Field

from backend.domain.dataset import Dataset, Task
from backend.domain.result import VerificationResult


class DatasetSnapshot(BaseModel):
    """Immutable snapshot of a dataset pinned to an evaluation run."""

    snapshot_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    dataset_id: uuid.UUID
    dataset_name: str
    version: str
    version_hash: str
    created_at: float = Field(default_factory=time.time)
    tasks: list[dict[str, Any]] = Field(default_factory=list)


def create_dataset_snapshot(dataset: Dataset) -> DatasetSnapshot:
    """Create an immutable snapshot of the dataset and its tasks."""
    version_hash = dataset.compute_version_hash()
    task_dicts = [task.model_dump() for task in dataset.tasks]

    return DatasetSnapshot(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        version=dataset.version,
        version_hash=version_hash,
        tasks=copy.deepcopy(task_dicts),
    )


class RegressionDetector:
    """Detects performance regressions between a baseline and candidate evaluation run."""

    @staticmethod
    def compare_results(
        baseline: VerificationResult,
        candidate: VerificationResult,
        tolerance: float = 0.05,
    ) -> dict[str, Any]:
        """Compare two verification results to identify overall and hidden test regressions."""
        overall_delta = candidate.overall_score - baseline.overall_score
        public_delta = candidate.public_score - baseline.public_score
        hidden_delta = candidate.hidden_score - baseline.hidden_score

        is_overall_regression = overall_delta < -tolerance
        is_hidden_regression = hidden_delta < -tolerance
        overfitting_detected = (public_delta >= 0) and (hidden_delta < -tolerance)

        regressions: list[dict[str, Any]] = []

        if is_overall_regression:
            regressions.append({
                "type": "overall_score_drop",
                "baseline": baseline.overall_score,
                "candidate": candidate.overall_score,
                "delta": overall_delta,
            })

        if is_hidden_regression:
            regressions.append({
                "type": "hidden_test_regression",
                "baseline": baseline.hidden_score,
                "candidate": candidate.hidden_score,
                "delta": hidden_delta,
                "new_hidden_failures": candidate.hidden_failures,
            })

        if baseline.passed and not candidate.passed:
            regressions.append({
                "type": "status_regression",
                "message": "Candidate failed evaluation while baseline passed.",
            })

        return {
            "has_regression": len(regressions) > 0,
            "is_hidden_regression": is_hidden_regression,
            "overfitting_detected": overfitting_detected,
            "overall_delta": round(overall_delta, 4),
            "public_delta": round(public_delta, 4),
            "hidden_delta": round(hidden_delta, 4),
            "regressions": regressions,
        }
