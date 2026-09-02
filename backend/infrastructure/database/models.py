"""SQLAlchemy ORM models for AgentForge.

Mapping the DomainEntities to the PostgreSQL database.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.infrastructure.database.database import Base

# ── Project ──────────────────────────────────────────────────────────────────

class ProjectModel(Base):
    __tablename__ = "projects"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ── Agent ────────────────────────────────────────────────────────────────────

class AgentModel(Base):
    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    allowed_tools: Mapped[list[str]] = mapped_column(JSONB, default=list)
    version: Mapped[str] = mapped_column(String(64), default="1.0.0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ── Dataset & Task ───────────────────────────────────────────────────────────

class DatasetModel(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    version: Mapped[str] = mapped_column(String(64), default="1.0.0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    tasks: Mapped[list["TaskModel"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class TaskModel(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    input_prompt: Mapped[str] = mapped_column(Text, default="")
    
    # Contract
    expected_state: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    allowed_tools: Mapped[list[str]] = mapped_column(JSONB, default=list)
    forbidden_tools: Mapped[list[str]] = mapped_column(JSONB, default=list)
    required_tools: Mapped[list[str]] = mapped_column(JSONB, default=list)
    invariants: Mapped[list[str]] = mapped_column(JSONB, default=list)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    dataset: Mapped["DatasetModel"] = relationship(back_populates="tasks")


# ── Evaluation ───────────────────────────────────────────────────────────────

class EvaluationModel(Base):
    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    
    agent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id"), nullable=False)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    
    scoring_rules: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    verifier_set: Mapped[list[str]] = mapped_column(JSONB, default=list)
    version: Mapped[str] = mapped_column(String(64), default="1.0.0")
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ── Run ──────────────────────────────────────────────────────────────────────

class RunModel(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("evaluations.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="CREATED")
    
    snapshot_agent_version: Mapped[str] = mapped_column(String(64), default="")
    snapshot_eval_version: Mapped[str] = mapped_column(String(64), default="")
    snapshot_dataset_version: Mapped[str] = mapped_column(String(64), default="")
    
    total_tasks: Mapped[int] = mapped_column(Integer, default=0)
    completed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    failed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ── Environment ──────────────────────────────────────────────────────────────

class EnvironmentModel(Base):
    __tablename__ = "environments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trial_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trials.id"), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(32), default="PROVISIONING")
    
    db_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)
    mcp_endpoint: Mapped[str | None] = mapped_column(String(512), nullable=True)
    
    state_before: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    state_after: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ── Trial ────────────────────────────────────────────────────────────────────

class TrialModel(Base):
    __tablename__ = "trials"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"), nullable=False)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    
    model_calls: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ── VerificationResult ───────────────────────────────────────────────────────

class VerificationResultModel(Base):
    __tablename__ = "verification_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trial_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trials.id"), unique=True, nullable=False)
    
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    
    metrics: Mapped[dict[str, float]] = mapped_column(JSONB, default=dict)
    failures: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
