"""Idempotency store managing at-most-once execution state."""

from __future__ import annotations

import asyncio
from enum import Enum
import time
from typing import Any
from pydantic import BaseModel, Field


class IdempotencyStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class IdempotencyConflictError(Exception):
    """Raised when an operation with the same idempotency key is concurrently in progress."""
    pass


class IdempotencyRecord(BaseModel):
    """Stored record of an idempotent request."""
    key: str
    status: IdempotencyStatus
    response_payload: Any = None
    resource_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class IdempotencyStore:
    """Thread-safe and async-safe store for managing idempotency keys."""

    def __init__(self) -> None:
        self._records: dict[str, IdempotencyRecord] = {}
        self._lock = asyncio.Lock()

    async def check_or_start(self, key: str) -> tuple[bool, Any]:
        """Check if an operation has already executed.
        
        Returns:
            (is_completed, cached_response_payload)
            - If (True, payload): caller should return cached payload immediately.
            - If (False, None): caller has claimed the key and should proceed with execution.
        Raises:
            IdempotencyConflictError: If request with same key is currently IN_PROGRESS.
        """
        async with self._lock:
            record = self._records.get(key)
            if record is not None:
                if record.status == IdempotencyStatus.COMPLETED:
                    return True, record.response_payload
                if record.status == IdempotencyStatus.IN_PROGRESS:
                    raise IdempotencyConflictError(
                        f"A request with idempotency key '{key}' is already in progress."
                    )
                # If FAILED, we allow a retry by reclaiming the key

            # Register as IN_PROGRESS
            now = time.time()
            self._records[key] = IdempotencyRecord(
                key=key,
                status=IdempotencyStatus.IN_PROGRESS,
                created_at=now,
                updated_at=now,
            )
            return False, None

    async def complete(self, key: str, response_payload: Any, resource_id: str | None = None) -> None:
        """Mark the operation as COMPLETED with its response payload."""
        async with self._lock:
            record = self._records.get(key)
            now = time.time()
            if record:
                record.status = IdempotencyStatus.COMPLETED
                record.response_payload = response_payload
                record.resource_id = resource_id
                record.updated_at = now
            else:
                self._records[key] = IdempotencyRecord(
                    key=key,
                    status=IdempotencyStatus.COMPLETED,
                    response_payload=response_payload,
                    resource_id=resource_id,
                    created_at=now,
                    updated_at=now,
                )

    async def fail(self, key: str, error_message: str | None = None) -> None:
        """Mark the operation as FAILED so retries can proceed."""
        async with self._lock:
            record = self._records.get(key)
            if record:
                record.status = IdempotencyStatus.FAILED
                record.updated_at = time.time()

    async def get_record(self, key: str) -> IdempotencyRecord | None:
        """Retrieve the record for a key."""
        async with self._lock:
            return self._records.get(key)

    async def clear(self) -> None:
        """Clear all stored idempotency records."""
        async with self._lock:
            self._records.clear()


# Default singleton instance
default_idempotency_store = IdempotencyStore()
