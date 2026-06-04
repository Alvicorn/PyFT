from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import NamedTuple

_real_lock = threading.Lock


class RaceKind(Enum):
    """The three kinds of data race PyVFT distinguishes."""

    WRITE_WRITE = "write-write"
    READ_WRITE = "read-write"
    WRITE_READ = "write-read"


class AccessInfo(NamedTuple):
    """Snapshot of one memory access involved in a race report."""

    tid: int
    thread_name: str
    clock: int
    is_write: bool
    obj_repr: str
    attr: str
    stack: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RaceReport:
    """Immutable record of one detected data race."""

    kind: RaceKind
    access_a: AccessInfo
    access_b: AccessInfo
    obj_id: int
    sequence: int  # monotonic ordering


def _capture_stack(skip_frames: int = 3) -> tuple[str, ...]:
    frames = traceback.format_stack()
    relevant = frames[:-skip_frames] if len(frames) > skip_frames else frames
    return tuple(relevant)


def _safe_repr(obj: object, max_len: int = 60) -> str:
    try:
        r = repr(obj)
        return r if len(r) <= max_len else r[:max_len] + "…"
    except Exception:
        return f"<{type(obj).__name__} id={id(obj)}>"


class RaceLog:
    """
    Thread-safe append-only log of RaceReport objects with a de-duplication key.

    De-duplication key: ``(obj_id, attr, frozenset({tid_a, tid_b}), kind)``.

    Sequence numbers are assigned inside the de-dup critical section so
    suppressed duplicates do not leave gaps.
    """

    __slots__ = ("_lock", "_reports", "_seen", "_counter")

    def __init__(self) -> None:
        self._lock = _real_lock()
        self._reports: list[RaceReport] = []
        self._seen: set[tuple] = set()
        self._counter = 0

    def record(
        self, report_or_factory: RaceReport | Callable[[], RaceReport]
    ) -> bool:
        """
        Append a RaceReport. Return True if it was new.

        Accepts a RaceReport (its ``sequence`` is overwritten) or a
        zero-arg factory that produces one.
        """
        if callable(report_or_factory):
            report = report_or_factory()
        else:
            report = report_or_factory

        key = (
            report.obj_id,
            report.access_a.attr,
            frozenset([report.access_a.tid, report.access_b.tid]),
            report.kind,
        )
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            self._counter += 1
            stamped = replace(report, sequence=self._counter)
            self._reports.append(stamped)
            return True

    def all_reports(self) -> list[RaceReport]:
        with self._lock:
            return list(self._reports)

    def clear(self) -> None:
        with self._lock:
            self._reports.clear()
            self._seen.clear()
            self._counter = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._reports)
