"""Redis connection helper.

Provides a lazy async Redis client used for:
  - caching
  - idempotency-key storage
  - lightweight job queues
  - rate limiting
"""

from __future__ import annotations

import redis.asyncio as aioredis

from backend.config import settings

_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Return (or create) a shared async Redis connection pool."""
    global _pool  # noqa: PLW0603
    if _pool is None:
        _pool = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=20,
        )
    return _pool


async def close_redis() -> None:
    """Gracefully close the Redis pool on shutdown."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        await _pool.aclose()
        _pool = None
