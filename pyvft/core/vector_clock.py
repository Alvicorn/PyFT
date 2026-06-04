from __future__ import annotations

import threading
from typing import Dict, Iterator, Optional, Tuple

_real_lock = threading.Lock


class VectorClock:
    """
    Mapping of thread id to logical clock, with an internal mutex.

    Every read / mutate acquires ``self._lock``. Methods that need data
    from another VectorClock snapshot it first.

    Two VC locks are never held simultaneously, so symmetric calls
    like ``a.join(b)`` and ``b.join(a)`` cannot deadlock.
    """

    __slots__ = ("_clocks", "_lock")

    def __init__(self, clocks: Optional[Dict[int, int]] = None) -> None:
        self._clocks = dict(clocks) if clocks else {}
        self._lock = _real_lock()

    def get(self, tid: int) -> int:
        with self._lock:
            return self._clocks.get(tid, 0)

    def set(self, tid: int, clock: int) -> None:
        if clock < 0:
            raise ValueError("Logical clock can not be negative")
        with self._lock:
            self._clocks[tid] = clock

    def increment(self, tid: int) -> int:
        with self._lock:
            new = self._clocks.get(tid, 0) + 1
            self._clocks[tid] = new
            return new

    def copy(self) -> VectorClock:
        with self._lock:
            return VectorClock(self._clocks.copy())

    def items(self) -> Iterator[Tuple[int, int]]:
        with self._lock:
            snapshot = list(self._clocks.items())
        return iter(snapshot)

    def _snapshot_dict(self) -> Dict[int, int]:
        with self._lock:
            return self._clocks.copy()

    def epoch_happens_before(self, tid: int, clock: int) -> bool:
        with self._lock:
            return self._clocks.get(tid, 0) >= clock

    def vc_leq(self, other: VectorClock) -> bool:
        other_snap = other._snapshot_dict()
        with self._lock:
            for tid, clock in self._clocks.items():
                if other_snap.get(tid, 0) < clock:
                    return False
            return True

    def join(self, other: VectorClock) -> None:
        other_snap = other._snapshot_dict()
        with self._lock:
            for tid, clock in other_snap.items():
                if clock > self._clocks.get(tid, 0):
                    self._clocks[tid] = clock

    @staticmethod
    def joined(a: VectorClock, b: VectorClock) -> VectorClock:
        c = a.copy()
        c.join(b)
        return c

    @staticmethod
    def from_epoch(tid: int, clock: int) -> VectorClock:
        return VectorClock({tid: clock})

    def add_epoch(self, tid: int, clock: int) -> None:
        with self._lock:
            if clock > self._clocks.get(tid, 0):
                self._clocks[tid] = clock

    def __repr__(self) -> str:
        with self._lock:
            entries = ", ".join(
                f"t{tid}:{c}" for tid, c in sorted(self._clocks.items())
            )
        return f"VectorClock(clocks={entries})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        a = self._snapshot_dict()
        b = other._snapshot_dict()
        all_tids = set(a) | set(b)
        return all(a.get(t, 0) == b.get(t, 0) for t in all_tids)
