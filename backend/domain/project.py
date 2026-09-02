"""Project domain models."""

from __future__ import annotations

from pydantic import Field

from backend.domain.base import DomainEntity


class Project(DomainEntity):
    """A logical grouping of agents, datasets, and evaluations."""
    name: str
    description: str = ""
