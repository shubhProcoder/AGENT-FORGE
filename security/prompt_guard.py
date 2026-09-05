"""Simple prompt-injection guard.

Detects common injection patterns in user input and tool responses.
"""

from __future__ import annotations

import re

# ── Detection patterns ───────────────────────────────────────────────────────

INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*script\s*>", re.IGNORECASE),
    re.compile(r"```\s*system", re.IGNORECASE),
    re.compile(r"ADMIN\s+OVERRIDE", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?system\s+prompt", re.IGNORECASE),
]


def detect_prompt_injection(text: str) -> list[str]:
    """Return a list of matched injection patterns (empty if clean).

    This is a simple heuristic guard — not a production-grade defense.
    It serves as a demonstration of the security layer concept.
    """
    matches: list[str] = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def is_safe(text: str) -> bool:
    """Return True if no injection patterns are detected."""
    return len(detect_prompt_injection(text)) == 0
