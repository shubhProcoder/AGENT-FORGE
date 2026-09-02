"""Environment domain models and state machines."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any
from pydantic import Field

from backend.domain.base import DomainEntity


class EnvironmentStatus(str, Enum):
    PROVISIONING = "PROVISIONING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    FROZEN = "FROZEN"
    DESTROYED = "DESTROYED"
    PROVISION_FAILED = "PROVISION_FAILED"
    CORRUPTED = "CORRUPTED"


class Environment(DomainEntity):
    """The Execution Plane context allocated for a Trial."""
    trial_id: uuid.UUID
    status: EnvironmentStatus = EnvironmentStatus.PROVISIONING
    
    # Connection details for the sandbox
    db_uri: str | None = None
    mcp_endpoint: str | None = None
    
    # Snapshots for diffing
    state_before: dict[str, Any] | None = None
    state_after: dict[str, Any] | None = None
