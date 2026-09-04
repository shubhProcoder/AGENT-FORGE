"""Dataset and Task domain models with Hidden Test & Versioning support."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any
from pydantic import Field

from backend.domain.base import DomainEntity


class Task(DomainEntity):
    """An individual test case or scenario."""

    dataset_id: uuid.UUID
    name: str
    description: str = ""
    input_prompt: str
    version: str = "1.0.0"

    # Public contract (visible to agent)
    expected_state: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list)

    # Hidden contract (strictly hidden from agent context)
    is_hidden: bool = False
    hidden_invariants: list[str] = Field(default_factory=list)
    hidden_expected_state: dict[str, Any] = Field(default_factory=dict)

    def compute_content_hash(self) -> str:
        """Generate a deterministic hash of the task's contract."""
        data = {
            "prompt": self.input_prompt,
            "expected_state": self.expected_state,
            "allowed_tools": sorted(self.allowed_tools),
            "forbidden_tools": sorted(self.forbidden_tools),
            "invariants": sorted(self.invariants),
            "hidden_invariants": sorted(self.hidden_invariants),
            "version": self.version,
        }
        raw = json.dumps(data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get_public_view(self) -> dict[str, Any]:
        """View provided to the agent runtime — hides all hidden assertions."""
        return {
            "name": self.name,
            "description": self.description,
            "input_prompt": self.input_prompt,
            "allowed_tools": self.allowed_tools,
            "required_tools": self.required_tools,
            "invariants": self.invariants,
        }


class Dataset(DomainEntity):
    """A collection of tasks to be evaluated together with pinned versioning."""

    project_id: uuid.UUID
    name: str
    description: str = ""
    version: str = "1.0.0"
    tasks: list[Task] = Field(default_factory=list)

    def compute_version_hash(self) -> str:
        """Generate a composite hash across all contained task versions."""
        task_hashes = sorted(t.compute_content_hash() for t in self.tasks)
        composite = f"{self.name}:{self.version}:" + ",".join(task_hashes)
        return hashlib.sha256(composite.encode()).hexdigest()[:16]
