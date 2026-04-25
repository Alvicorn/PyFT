from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from enum import Enum


class RaceKind(Enum):
    WRITE_WRITE = "write-write"
    READ_WRITE = "read-write"
    WRITE_READ = "write-read"


@dataclass(frozen=True, slots=True)
class AccessInfo:
    tid: int
    thread_name: str
    clock: int
    is_write: bool
    obj_repr: str
    attr: str
    stack: tuple[str, ...] = field(default_factory=tuple)  # traceback lines


@dataclass(frozen=True, slots=True)
class RaceReport:
    """
    Immutable record of one data race.
    The two accesses are `access_a` (the one being checked) and
    `access_b` (the prior conflicting access recovered from VarState).
    """

    kind: RaceKind
    access_a: AccessInfo  # current access (the one that triggered detection)
    access_b: AccessInfo  # prior conflicting access (from shadow state)
    obj_id: int
    sequence: int  # global monotonic counter for ordering


def _capture_stack(skip_frames: int = 3) -> tuple[str, ...]:
    """
    Capture the current call stack as a tuple of formatted strings,
    skipping internal pyft frames.
    """
    frames = traceback.format_stack()
    # Drop the last `skip_frames` which are PyFT internals
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
    Thread-safe append-only log of RaceReport objects.
    Deduplicates: the same (obj_id, attr, tid_a, tid_b) tuple is only
    recorded once to avoid flooding the report with the same race
    repeated thousands of times.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reports: list[RaceReport] = []
        self._seen: set[tuple] = set()
        self._counter = 0

    def record(self, report: RaceReport) -> bool:
        """
        Add a report. Returns True if it was new (not a duplicate).
        """
        key = (
            report.obj_id,
            report.access_a.attr,
            frozenset([report.access_a.tid, report.access_b.tid]),
            # report.kind,
        )
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            self._reports.append(report)
            return True

    def next_sequence(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

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
