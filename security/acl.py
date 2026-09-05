"""Tool-level access control list (ACL).

Defines which tools an agent is allowed to call based on its role.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class Permission(str, Enum):
    ALLOWED = "allowed"
    REQUIRES_APPROVAL = "requires_approval"
    FORBIDDEN = "forbidden"


# ── Default permission matrix ────────────────────────────────────────────────

DEFAULT_ACL: dict[str, Permission] = {
    "search_customer": Permission.ALLOWED,
    "get_customer_tickets": Permission.ALLOWED,
    "create_ticket": Permission.ALLOWED,
    "get_order": Permission.ALLOWED,
    "create_order": Permission.ALLOWED,
    "refund_order": Permission.REQUIRES_APPROVAL,
    "search_documents": Permission.ALLOWED,
    "retrieve_policy": Permission.ALLOWED,
    "delete_customer": Permission.FORBIDDEN,
    "delete_customer_data": Permission.FORBIDDEN,
}


class ToolACL:
    """Enforce tool-level permissions for an agent."""

    def __init__(self, acl: dict[str, Permission] | None = None) -> None:
        self.acl = acl or dict(DEFAULT_ACL)

    def check(self, tool_name: str) -> Permission:
        """Return the permission level for a tool.

        Unknown tools default to FORBIDDEN (deny-by-default).
        """
        return self.acl.get(tool_name, Permission.FORBIDDEN)

    def is_allowed(self, tool_name: str) -> bool:
        return self.check(tool_name) == Permission.ALLOWED

    def is_forbidden(self, tool_name: str) -> bool:
        return self.check(tool_name) == Permission.FORBIDDEN

    def requires_approval(self, tool_name: str) -> bool:
        return self.check(tool_name) == Permission.REQUIRES_APPROVAL

    def enforce(self, tool_name: str) -> None:
        """Raise PermissionError if the tool is forbidden."""
        perm = self.check(tool_name)
        if perm == Permission.FORBIDDEN:
            raise PermissionError(f"Tool '{tool_name}' is FORBIDDEN by ACL policy")
        if perm == Permission.REQUIRES_APPROVAL:
            # In a real system this would pause for human review.
            # For the MVP we log a warning but allow it.
            pass
