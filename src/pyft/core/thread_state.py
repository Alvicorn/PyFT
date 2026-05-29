from __future__ import annotations

import threading
from typing import Dict, List, Optional

from .vector_clock import VectorClock

_real_rlock = threading.RLock


class ThreadState:
    """Per-thread VerifiedFT state with a thread id, name, and vector clock."""

    __slots__ = ("tid", "vc", "name")

    def __init__(
        self,
        tid: int,
        initial_vc: Optional[VectorClock] = None,
        name: str = "",
    ) -> None:
        self.tid = tid
        self.name = name or f"Thread-{tid}"
        self.vc = VectorClock() if initial_vc is None else initial_vc.copy()

        if self.vc.get(tid) == 0:
            self.vc.set(tid, 1)

    def tick(self) -> int:
        return self.vc.increment(self.tid)

    def current_clock(self) -> int:
        return self.vc.get(self.tid)

    def absorb(self, other_vc: VectorClock) -> None:
        self.vc.join(other_vc)

    def snapshot(self) -> VectorClock:
        return self.vc.copy()

    def __repr__(self) -> str:
        return f"ThreadState(name={self.name}, vc={self.vc!r})"


class ThreadRegistry:
    """Thread-safe registry of ThreadState objects keyed by thread id."""

    def __init__(self) -> None:
        self._lock = _real_rlock()
        self._states: Dict[int, ThreadState] = {}

    def register(
        self,
        tid: int,
        initial_vc: Optional[VectorClock] = None,
        name: str = "",
    ) -> ThreadState:
        with self._lock:
            state = ThreadState(tid, initial_vc=initial_vc, name=name)
            self._states[tid] = state
            return state

    def get(self, tid: int) -> Optional[ThreadState]:
        with self._lock:
            return self._states.get(tid)

    def get_or_register(self, tid: int, name: str = "") -> ThreadState:
        with self._lock:
            if tid not in self._states:
                return self.register(tid, name=name)
            return self._states[tid]

    def remove(self, tid: int) -> Optional[ThreadState]:
        with self._lock:
            return self._states.pop(tid, None)

    def all_tids(self) -> List[int]:
        with self._lock:
            return list(self._states.keys())
