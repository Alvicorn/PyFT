from __future__ import annotations

import threading
from typing import Dict, NamedTuple

_NONE_TID = -1


class Epoch(NamedTuple):
    tid: int  # thread identify
    clock: int  # monotonically increasing pre-thread logical clock

    @staticmethod
    def bottom() -> Epoch:
        return Epoch(_NONE_TID, 0)

    @staticmethod
    def of(tid: int, clock: int) -> Epoch:
        return Epoch(tid, clock)

    def is_bottom(self) -> bool:
        return self.tid == _NONE_TID

    def __repr__(self) -> str:
        if self.is_bottom():
            return "Epoch(⊥)"
        return f"Epoch(tid={self.tid}, clock={self.clock})"


def current_epoch(thread_clocks: Dict[int, int]) -> Epoch:
    tid = threading.get_ident()
    return Epoch(tid, thread_clocks.get(tid, 0))
