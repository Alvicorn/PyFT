from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Tuple

from .epoch import _NONE_TID, Epoch
from .vector_clock import VectorClock

log = logging.getLogger(__name__)

## READ STATES


@dataclass(frozen=True, slots=True)
class ReadBottom:
    """No read have occurred yet"""

    pass


@dataclass(slots=True)
class ReadEpoch:
    """Exactly one thread has read this variable"""

    tid: int
    clock: int


@dataclass(slots=True)
class ReadVC:
    """Two or more thread have read this variable; vector clock required"""

    vc: VectorClock


class VarState:
    __slots__ = (
        "_lock",
        "write_epoch",
        "read_state",
        "first_tid",
        "is_shared",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.write_epoch = Epoch.bottom()
        self.read_state = ReadBottom()
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

    ### VERIFIED-FT READ RULE ###

    def check_read(
        self, tid: int, thread_vc: VectorClock
    ) -> Tuple[bool, Epoch]:
        """
        VerifiedFT read rule (Wilcox & Freund, PPoPP '18, Fig. 4):
          - No prior write (W_x = bottom)         -> safe
          - Last writer is this thread            -> safe (program order)
          - Last write happens-before this read   -> safe (W_x <= C_t)
          - Otherwise                             -> write-read race

        Returns (race_detected, prior_write_epoch) where prior_write_epoch
        is captured atomically inside the lock so callers can report
        the correct prior-writer information without a TOCTOU race.
        """
        with self._lock:
            self._note_access(tid)

            we = self.write_epoch

            # no prior write
            if we.is_bottom():
                self._update_read_state(tid, thread_vc)
                return False, we

            # same thread as the last writer -> program order gives HB.
            if we.tid == tid:
                self._update_read_state(tid, thread_vc)
                return False, we

            # cross-thread -> last write must happen-before this read.
            if thread_vc.epoch_happens_before(we.tid, we.clock):
                self._update_read_state(tid, thread_vc)
                return False, we

            # Write-Read Race
            return True, we

    def _update_read_state(self, tid: int, thread_vc: VectorClock) -> None:
        rs = self.read_state
        clock = thread_vc.get(tid)

        if isinstance(rs, ReadBottom):  # first read by tid
            self.read_state = ReadEpoch(tid, clock)

        elif isinstance(rs, ReadEpoch):
            if rs.tid == tid:  # same tid reader
                if clock > rs.clock:
                    self.read_state = ReadEpoch(tid, clock)
            else:  # another unique tid reader -> upgrade to ReadVC
                vc = VectorClock.from_epoch(rs.tid, rs.clock)
                vc.add_epoch(tid, clock)
                self.read_state = ReadVC(vc)

        elif isinstance(rs, ReadVC):
            rs.vc.add_epoch(tid, clock)

    ### VERIFIED-FT WRITE RULE ###

    def check_write(
        self, tid: int, thread_vc: VectorClock
    ) -> Tuple[bool, bool, Epoch, object]:
        """
        VerifiedFT write rule (Wilcox & Freund, PPoPP '18, Fig. 4):
          1. Check (new write) vs. last write:
               - no prior write              -> safe
               - same-thread last write      -> safe
               - last write HB this write    -> safe
               - otherwise                   -> write-write race
          2. Check (new write) vs. all reads:
               - ReadBottom                  -> safe
               - ReadEpoch(t, c): t == tid or (t@c) <= C_tid -> safe, else race
               - ReadVC(vc): every reader r (r != tid) must satisfy r <= C_tid
          3. Update shadow state:
               - W_x := tid@C_tid[tid]
               - reads cleared (new write supersedes them)

        Returns (read_race, write_race, prior_write_epoch, prior_read_state)
        where both prior values are captured atomically inside the lock.
        This prevents the TOCTOU race where a caller snapshotting these
        fields outside the lock could see stale (bottom) values.
        """
        with self._lock:
            self._note_access(tid)

            clock = thread_vc.get(tid)

            # capture prior state inside the lock
            prior_write_epoch = self.write_epoch
            prior_read_state = self.read_state

            read_race = self._check_write_vs_reads(tid, thread_vc)
            write_race = self._check_write_vs_write(tid, thread_vc)

            self.write_epoch = Epoch(tid, clock)
            self.read_state = ReadBottom()  # new write supersedes prior reads

            return read_race, write_race, prior_write_epoch, prior_read_state

    def _check_write_vs_write(self, tid: int, thread_vc: VectorClock) -> bool:
        we = self.write_epoch
        if we.is_bottom():
            return False
        if we.tid == tid:
            return False

        # cross-thread: last write must HB the new write.
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
            # every concurrent reader (other than tid itself) must HB the write.
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
