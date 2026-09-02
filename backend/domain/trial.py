"""Trial domain models."""

from __future__ import annotations

import uuid
from typing import Any

from backend.domain.base import DomainEntity


class Trial(DomainEntity):
    """One execution of an Agent on a specific Task."""

    run_id: uuid.UUID
    task_id: uuid.UUID

    # Trace
    model_calls: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    latency_ms: int = 0
    trace: Any | None = None

    # State tracking
    is_completed: bool = False
    error_message: str | None = None
