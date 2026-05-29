from __future__ import annotations

import logging
import threading
from typing import NamedTuple, Tuple

from .epoch import _NONE_TID, Epoch
from .vector_clock import VectorClock

log = logging.getLogger(__name__)

_real_lock = threading.Lock


class ReadBottom:
    """Read state before any read has occurred."""

    pass


class ReadEpoch(NamedTuple):
    """Read state for a variable read by exactly one thread."""

    tid: int
    clock: int


class ReadVC(NamedTuple):
    """Read state for a variable read by two or more threads."""

    vc: VectorClock


ReadState = ReadBottom | ReadEpoch | ReadVC
READ_BOTTOM = ReadBottom()


class VarState:
    """Per-variable race-detection state: last writer, current readers, lock."""

    __slots__ = (
        "_lock",
        "write_epoch",
        "read_state",
        "first_tid",
        "is_shared",
    )

    def __init__(self) -> None:
        self._lock = _real_lock()
        self.write_epoch = Epoch.bottom()
        self.read_state: ReadState = READ_BOTTOM
        self.first_tid = _NONE_TID
        self.is_shared = False

    def _note_access(self, tid: int) -> bool:
        if self.first_tid == _NONE_TID:
            self.first_tid = tid
            return False
        if not self.is_shared and self.first_tid != tid:
            self.is_shared = True
            return True
        return False

    def check_read(
        self, tid: int, thread_vc: VectorClock
    ) -> Tuple[bool, Epoch]:
        """
        VerifiedFT read rule (Wilcox & Freund, PPoPP '18, Fig. 4):
          - No prior write (W_x = bottom)       -> safe
          - Last writer is this thread          -> safe (program order)
          - Last write happens-before this read -> safe (W_x <= C_t)
          - Otherwise                           -> write-read race

        Args:
            tid (int): thread id
            thread_vc (VectorClock): thread vector clock

        Returns:
            Tuple[bool, Epoch]: race_detected, prior_write_epoch
        """
        with self._lock:
            self._note_access(tid)

            we = self.write_epoch

            if we.is_bottom():
                self._update_read_state(tid, thread_vc)
                return False, we

            if we.tid == tid:
                self._update_read_state(tid, thread_vc)
                return False, we

            if thread_vc.epoch_happens_before(we.tid, we.clock):
                self._update_read_state(tid, thread_vc)
                return False, we

            return True, we

    def _update_read_state(self, tid: int, thread_vc: VectorClock) -> None:
        rs = self.read_state
        clock = thread_vc.get(tid)

        if isinstance(rs, ReadBottom):
            self.read_state = ReadEpoch(tid, clock)

        elif isinstance(rs, ReadEpoch):
            if rs.tid == tid:
                if clock > rs.clock:
                    self.read_state = ReadEpoch(tid, clock)
            else:
                vc = VectorClock.from_epoch(rs.tid, rs.clock)
                vc.add_epoch(tid, clock)
                self.read_state = ReadVC(vc)

        elif isinstance(rs, ReadVC):
            rs.vc.add_epoch(tid, clock)

    def check_write(
        self, tid: int, thread_vc: VectorClock
    ) -> Tuple[bool, bool, Epoch, ReadState]:
        """
        VerifiedFT write rule (Wilcox & Freund, PPoPP '18, Fig. 4):
          1. New write vs. last write:
               - no prior write              -> safe
               - same-thread last write      -> safe
               - last write HB this write    -> safe
               - otherwise                   -> write-write race
          2. New write vs. all reads:
               - ReadBottom                  -> safe
               - ReadEpoch(t, c): t == tid or (t@c) <= C_tid -> safe, else race
               - ReadVC(vc): every reader r (r != tid) must satisfy r <= C_tid
          3. Update shadow state:
               - W_x := tid@C_tid[tid]
               - reads cleared (new write supersedes them)
        Args:
            tid (int): thread id
            thread_vc (VectorClock): thread vector clock

        Returns:
            Tuple[bool, bool, Epoch, ReadState]: read_race, write_race, prior_write_epoch, prior_read_state
        """
        with self._lock:
            self._note_access(tid)

            clock = thread_vc.get(tid)

            prior_write_epoch = self.write_epoch
            prior_read_state = self.read_state

            read_race = self._check_write_vs_reads(tid, thread_vc)
            write_race = self._check_write_vs_write(tid, thread_vc)

            self.write_epoch = Epoch(tid, clock)
            self.read_state = ReadBottom()

            return read_race, write_race, prior_write_epoch, prior_read_state

    def _check_write_vs_write(self, tid: int, thread_vc: VectorClock) -> bool:
        we = self.write_epoch
        if we.is_bottom():
            return False
        if we.tid == tid:
            return False
        return not thread_vc.epoch_happens_before(we.tid, we.clock)

    def _check_write_vs_reads(self, tid: int, thread_vc: VectorClock) -> bool:
        rs = self.read_state

        if isinstance(rs, ReadBottom):
            return False

        if isinstance(rs, ReadEpoch):
            if rs.tid == tid:
                return False
            return not thread_vc.epoch_happens_before(rs.tid, rs.clock)

        if isinstance(rs, ReadVC):
            for r_tid, r_clock in rs.vc.items():
                if r_tid == tid:
                    continue
                if not thread_vc.epoch_happens_before(r_tid, r_clock):
                    return True
            return False

        return False  # pragma: no cover (unreachable)

    def __repr__(self) -> str:
        return (
            f"VarState(write={self.write_epoch!r}, "
            f"read={self.read_state!r}, "
            f"shared={self.is_shared}"
        )
