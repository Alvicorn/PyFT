from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple


class VectorClock:
    __slots__ = ("_clocks",)

    def __init__(self, clocks: Optional[Dict[int, int]] = None) -> None:
        self._clocks = clocks if clocks else {}

    def get(self, tid: int) -> int:
        return self._clocks.setdefault(tid, 0)

    def set(self, tid: int, clock: int) -> None:
        if clock < 0:
            raise ValueError("Logical clock can not be negative")
        self._clocks[tid] = clock

    def increment(self, tid: int) -> int:
        new = self.get(tid) + 1
        self._clocks[tid] = new
        return new

    def copy(self) -> VectorClock:
        return VectorClock(self._clocks.copy())

    def items(self) -> Iterator[Tuple[int, int]]:
        return iter(self._clocks.items())

    ### HAPPENS BEFORE ###

    def epoch_happens_before(self, tid: int, clock: int) -> bool:
        return self.get(tid) >= clock

    def vc_leq(self, other: VectorClock) -> bool:
        for tid, clock in self._clocks.items():
            if other.get(tid) < clock:
                return False
        return True

    ### MERGING ###

    def join(self, other: VectorClock) -> None:
        for tid, clock in other.items():
            if clock > self.get(tid):
                self._clocks[tid] = clock

    @staticmethod
    def joined(a: VectorClock, b: VectorClock) -> VectorClock:
        c = a.copy()
        c.join(b)
        return c

    ### EPOCH - VC CONVERSIONS ###

    @staticmethod
    def from_epoch(tid: int, clock: int) -> VectorClock:
        return VectorClock({tid: clock})

    def add_epoch(self, tid: int, clock: int) -> None:
        if clock > self.get(tid):
            self._clocks[tid] = clock

    ### HELPFUL DUNDER METHODS ###

    def __repr__(self) -> str:
        entries = ", ".join(f"t{tid}:{c}" for tid, c in sorted(self.items()))
        return f"VectorClock(clocks={entries})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        all_tids = set(self._clocks) | set(other._clocks)
        return all(self.get(t) == other.get(t) for t in all_tids)
