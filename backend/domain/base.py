"""Base domain models.

These are pure Pydantic models representing the core business entities.
They are completely decoupled from the database (SQLAlchemy) and the API (FastAPI).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field


def new_uuid() -> uuid.UUID:
    return uuid.uuid4()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class DomainEntity(BaseModel):
    """Base class for all domain entities."""
    id: uuid.UUID = Field(default_factory=new_uuid)
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)

    model_config = ConfigDict(from_attributes=True)
