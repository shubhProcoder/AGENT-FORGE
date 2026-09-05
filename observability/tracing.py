"""Tracing utilities — generate and propagate trace/request IDs."""

from __future__ import annotations

import uuid
from contextvars import ContextVar

# Context-local trace ID so any code can log it without passing it around
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_request_id: ContextVar[str] = ContextVar("request_id", default="")


def new_trace_id() -> str:
    """Generate and set a new trace ID."""
    tid = uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id.get()


def new_request_id() -> str:
    """Generate and set a new request ID."""
    rid = uuid.uuid4().hex[:8]
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id.get()
