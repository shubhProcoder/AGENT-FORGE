"""Keyed Async Lock for resource-level mutual exclusion."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator


class KeyedAsyncLock:
    """Manages fine-grained asyncio Locks keyed by a resource string (e.g. order_id, user_id)."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncGenerator[None, None]:
        """Acquire a lock for the specified key."""
        lock = await self._get_lock(key)
        async with lock:
            yield

    async def clear(self) -> None:
        """Clear all active locks."""
        async with self._global_lock:
            self._locks.clear()


default_keyed_lock = KeyedAsyncLock()
