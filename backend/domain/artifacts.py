"""Domain models for execution artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import uuid


@dataclass
class Artifact:
    """A reference to an artifact generated during execution or evaluation."""

    type: str  # e.g. "candidate_source", "junit_xml", "traceback", "diagnostic_log"
    path: str  # storage URI or filesystem path
    size_bytes: int = 0
    checksum: str = ""
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retention_policy: str = "transient"  # "transient", "retained", "permanent"

    @classmethod
    def from_content(
        cls,
        content: str | bytes,
        artifact_type: str,
        path: str,
        retention_policy: str = "transient",
    ) -> Artifact:
        """Create an Artifact instance computing size and checksum automatically."""
        raw = content.encode("utf-8") if isinstance(content, str) else content
        checksum = hashlib.sha256(raw).hexdigest()
        return cls(
            type=artifact_type,
            path=path,
            size_bytes=len(raw),
            checksum=checksum,
            retention_policy=retention_policy,
        )
