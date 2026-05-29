"""
classify_call: maps a (callable, first-arg) pair from a sys.monitoring
CALL event to a SyncKind, or returns None if the call is not a
sync event PyFT cares about.

Lock / RLock / Semaphore / BoundedSemaphore / Condition.acquire are
handled by LockPatcher instead of via monitoring; they are skipped here
to avoid double-firing engine events.
"""

from __future__ import annotations

import threading
from enum import Enum, auto
from typing import Any, NamedTuple


class SyncKind(Enum):
    """The kind of synchronization event a CALL represents."""

    LOCK_ACQUIRE = auto()
    LOCK_RELEASE = auto()
    COND_WAIT = auto()
    COND_NOTIFY = auto()
    THREAD_START = auto()
    THREAD_JOIN = auto()
    EVENT_SET = auto()
    EVENT_WAIT = auto()
    BARRIER_WAIT = auto()


class SyncCall(NamedTuple):
    """A pending or completed synchronization call."""

    kind: SyncKind
    target: Any


_LOCK_TYPES = (threading.Lock().__class__, threading.RLock().__class__)
_SEMAPHORE_TYPES = (threading.Semaphore, threading.BoundedSemaphore)
_CONDITION_TYPE = threading.Condition
_EVENT_TYPE = threading.Event
_BARRIER_TYPE = threading.Barrier
_THREAD_TYPE = threading.Thread


def classify_call(callable_: Any, arg0: Any) -> SyncCall | None:  # noqa: ANN401
    """Return a SyncCall descriptor, or None if not a tracked sync event."""
    if not callable(callable_):
        return None

    fn_name = getattr(callable_, "__name__", "")

    if isinstance(arg0, _THREAD_TYPE):
        if fn_name == "start":
            return SyncCall(SyncKind.THREAD_START, arg0)
        if fn_name == "join":
            return SyncCall(SyncKind.THREAD_JOIN, arg0)
        return None

    # Lock / RLock / Semaphore are handled by LockPatcher.
    if isinstance(arg0, _LOCK_TYPES):
        return None

    if isinstance(arg0, _SEMAPHORE_TYPES):
        return None

    if isinstance(arg0, _CONDITION_TYPE):
        if fn_name in ("wait", "wait_for"):
            return SyncCall(SyncKind.COND_WAIT, arg0)
        if fn_name in ("notify", "notify_all"):
            return SyncCall(SyncKind.COND_NOTIFY, arg0)
        return None

    if isinstance(arg0, _EVENT_TYPE):
        if fn_name == "set":
            return SyncCall(SyncKind.EVENT_SET, arg0)
        if fn_name == "wait":
            return SyncCall(SyncKind.EVENT_WAIT, arg0)
        return None

    if isinstance(arg0, _BARRIER_TYPE):
        if fn_name == "wait":
            return SyncCall(SyncKind.BARRIER_WAIT, arg0)
        return None

    return None
