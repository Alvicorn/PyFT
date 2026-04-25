"""
Supported primitives (zero-patch):
  threading.Lock          acquire / release / __enter__ / __exit__
  threading.RLock         same
  threading.Semaphore     acquire / release
  threading.BoundedSemaphore  same
  threading.Condition     acquire / release / wait / wait_for / notify / notify_all
  threading.Event         set / clear / wait
  threading.Barrier       wait
  threading.Thread        start / join
"""

from __future__ import annotations

import logging
import threading
from enum import Enum, auto
from typing import Any, NamedTuple

log = logging.getLogger(__name__)


class SyncKind(Enum):
    LOCK_ACQUIRE = auto()  # Lock / RLock / Semaphore acquire
    LOCK_RELEASE = auto()  # Lock / RLock / Semaphore release
    COND_WAIT = auto()  # Condition.wait — releases + re-acquires
    COND_NOTIFY = auto()  # Condition.notify / notify_all
    THREAD_START = auto()  # Thread.start
    THREAD_JOIN = auto()  # Thread.join
    EVENT_SET = auto()  # Event.set — signals waiters
    EVENT_WAIT = auto()  # Event.wait — waits for set
    BARRIER_WAIT = auto()  # Barrier.wait


class SyncCall(NamedTuple):
    """A pending or completed synchronization call."""

    kind: SyncKind
    target: Any  # the Lock / Thread / Event etc.


_LOCK_TYPES = (
    threading.Lock().__class__,  # _thread.lock (C type)
    threading.RLock().__class__,  # _thread.RLock (C type)
)
_SEMAPHORE_TYPES = (threading.Semaphore, threading.BoundedSemaphore)
_CONDITION_TYPE = threading.Condition
_EVENT_TYPE = threading.Event
_BARRIER_TYPE = threading.Barrier
_THREAD_TYPE = threading.Thread

_ACQUIRE_NAMES = frozenset({"acquire", "__enter__"})
_RELEASE_NAMES = frozenset({"release", "__exit__"})


def classify_call(callable_: Any, arg0: Any) -> SyncCall | None:
    """
    Given the callable and its first argument (self) from a CALL event,
    return a SyncCall descriptor or None if this is not a sync event we
    care about.

    Called from the sys.monitoring CALL callback.
    """
    if not callable(callable_):
        return None

    fn_name = getattr(callable_, "__name__", "")

    if isinstance(arg0, _THREAD_TYPE):
        if fn_name == "start":
            return SyncCall(SyncKind.THREAD_START, arg0)
        if fn_name == "join":
            return SyncCall(SyncKind.THREAD_JOIN, arg0)
        return None

    if isinstance(arg0, _LOCK_TYPES):
        if fn_name in _ACQUIRE_NAMES:
            return SyncCall(SyncKind.LOCK_ACQUIRE, arg0)
        if fn_name in _RELEASE_NAMES:
            return SyncCall(SyncKind.LOCK_RELEASE, arg0)
        return None

    if isinstance(arg0, _SEMAPHORE_TYPES):
        if fn_name in _ACQUIRE_NAMES:
            return SyncCall(SyncKind.LOCK_ACQUIRE, arg0)
        if fn_name in _RELEASE_NAMES:
            return SyncCall(SyncKind.LOCK_RELEASE, arg0)
        return None

    if isinstance(arg0, _CONDITION_TYPE):
        if fn_name in ("wait", "wait_for"):
            return SyncCall(SyncKind.COND_WAIT, arg0)
        if fn_name in ("notify", "notify_all"):
            return SyncCall(SyncKind.COND_NOTIFY, arg0)
        if fn_name in _ACQUIRE_NAMES:
            return SyncCall(SyncKind.LOCK_ACQUIRE, arg0)
        if fn_name in _RELEASE_NAMES:
            return SyncCall(SyncKind.LOCK_RELEASE, arg0)
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
