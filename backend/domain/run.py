"""Run domain models and state machines."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any
from pydantic import Field

from backend.domain.base import DomainEntity


class RunStatus(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    FINALIZATION_ERROR = "FINALIZATION_ERROR"


class Run(DomainEntity):
    """An execution of an Evaluation across its Dataset."""
    evaluation_id: uuid.UUID
    status: RunStatus = RunStatus.CREATED
    
    # Snapshots (implements the configuration snapshot invariant)
    snapshot_agent_version: str = ""
    snapshot_eval_version: str = ""
    snapshot_dataset_version: str = ""
    
    # Progress
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
