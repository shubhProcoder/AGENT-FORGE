"""Base tool abstraction.

Provides the ToolResult dataclass and a helper for attaching
OpenAI-compatible function schemas to tool functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Standardised response from any tool."""

    success: bool = True
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"success": self.success}
        if self.data:
            d["data"] = self.data
        if self.error:
            d["error"] = self.error
        return d


def tool_schema(
    name: str,
    description: str,
    parameters: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-compatible function-call schema dict.

    Attach this to a tool function via:
        fn.__tool_schema__ = tool_schema(...)
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required or [],
            },
        },
    }
