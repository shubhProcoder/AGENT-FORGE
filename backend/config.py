"""Application configuration via pydantic-settings.

Reads from environment variables or a .env file.
Every setting has a sensible default so the app boots locally without any env file.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for AgentForge."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "DEBUG"
    secret_key: str = "change-me-in-production"

    # ── Database ─────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+asyncpg://agentforge:agentforge_dev@localhost:5432/agentforge"
    )

    # ── Redis ────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── LLM ──────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ── Embedding Defaults ───────────────────────────────────────────────
    embedding_provider: str = "openai"  # 'openai' or 'fake'
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 100
    embedding_timeout_seconds: int = 30

    # ── Agent defaults ───────────────────────────────────────────────────
    agent_max_steps: int = 15
    agent_timeout_seconds: int = 120
    tool_timeout_seconds: int = 30
    max_retries: int = 3


settings = Settings()
