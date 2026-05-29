from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Callable, NamedTuple, Tuple

from .epoch import _NONE_TID, Epoch
from .vector_clock import VectorClock

log = logging.getLogger(__name__)

_real_lock = threading.Lock


class VFTVersion(Enum):
    """Which VerifiedFT analysis variant to use.

    V1: idealized algorithm; stores a full VectorClock for the last
    write and a full VectorClock for the union of reads. Race checks
    use ``vc_leq`` on the full clocks.

    V2: optimized algorithm (FastTrack-style); stores a single
    ``Epoch`` for the last write and a compressed read-state machine
    ``ReadBottom | ReadEpoch | ReadVC``. Race checks reduce to a
    constant-time epoch comparison in the common cases.
    """

    V1 = "v1"
    V2 = "v2"


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


class VarStateV2:
    """Per-variable race-detection state (V2 — epoch-compressed FastTrack)."""

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

        Returns ``(race_detected, prior_write_epoch)``.
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

        Returns ``(read_race, write_race, prior_write_epoch, prior_read_state)``.
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
            f"VarStateV2(write={self.write_epoch!r}, "
            f"read={self.read_state!r}, "
            f"shared={self.is_shared}"
        )


class VarStateV1:
    """
    Per-variable race-detection state (V1 — idealized full-VC analyzer).

    Stores the last write's full VectorClock ``W_x`` and the pointwise
    max of all reads' VectorClocks ``R_x``. Race checks use
    ``VectorClock.vc_leq``.
    """

    __slots__ = ("_lock", "W_x", "R_x", "first_tid", "is_shared")

    def __init__(self) -> None:
        self._lock = _real_lock()
        self.W_x: VectorClock = VectorClock()
        self.R_x: VectorClock = VectorClock()
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

    def _violating_writer(self, thread_vc: VectorClock) -> Epoch:
        """Return an Epoch describing the writer whose clock violates
        ``W_x <= C_t``. The first violator found is returned; if there
        is none, ``Epoch.bottom()``.
        """
        w_snap = self.W_x._snapshot_dict()
        for wtid, wclock in w_snap.items():
            if wclock == 0:
                continue
            if thread_vc.get(wtid) < wclock:
                return Epoch(wtid, wclock)
        return Epoch.bottom()

    def _violating_reader(self, tid: int, thread_vc: VectorClock) -> Epoch:
        """Return the first reader whose clock violates ``R_x <= C_t``
        (skipping ``tid`` itself), or ``Epoch.bottom()`` if none.
        """
        r_snap = self.R_x._snapshot_dict()
        for rtid, rclock in r_snap.items():
            if rtid == tid:
                continue
            if rclock == 0:
                continue
            if thread_vc.get(rtid) < rclock:
                return Epoch(rtid, rclock)
        return Epoch.bottom()

    def check_read(
        self, tid: int, thread_vc: VectorClock
    ) -> Tuple[bool, Epoch]:
        """
        V1 read rule:
          race iff ``W_x !<= C_t``; otherwise safe and ``R_x[t] := C_t[t]``.

        Returns ``(race_detected, prior_write_epoch)`` — the
        ``prior_write_epoch`` is the violating writer when race is True,
        ``Epoch.bottom()`` otherwise.
        """
        with self._lock:
            self._note_access(tid)

            if self.W_x.vc_leq(thread_vc):
                clock = thread_vc.get(tid)
                self.R_x.add_epoch(tid, clock)
                return False, Epoch.bottom()

            return True, self._violating_writer(thread_vc)

    def check_write(
        self, tid: int, thread_vc: VectorClock
    ) -> Tuple[bool, bool, Epoch, ReadState]:
        """
        V1 write rule:
          write-race iff ``W_x !<= C_t``;
          read-race iff ``R_x !<= C_t``.

        On any outcome the shadow state is updated:
          - ``W_x := C_t``
          - ``R_x := bottom``

        Returns ``(read_race, write_race, prior_write_epoch,
        prior_read_state)`` shaped to match V2's contract so the engine
        can construct race reports uniformly.
        """
        with self._lock:
            self._note_access(tid)

            write_race = not self.W_x.vc_leq(thread_vc)
            read_race = not self.R_x.vc_leq(thread_vc)

            prior_write_epoch = (
                self._violating_writer(thread_vc)
                if write_race
                else Epoch.bottom()
            )
            prior_read_state: ReadState
            if read_race:
                offender = self._violating_reader(tid, thread_vc)
                if offender.is_bottom():
                    prior_read_state = READ_BOTTOM
                else:
                    prior_read_state = ReadEpoch(offender.tid, offender.clock)
            else:
                prior_read_state = READ_BOTTOM

            # Replace W_x with the new write's VC, clear R_x.
            self.W_x = thread_vc.copy()
            self.R_x = VectorClock()

            return read_race, write_race, prior_write_epoch, prior_read_state

    def __repr__(self) -> str:
        return (
            f"VarStateV1(W={self.W_x!r}, R={self.R_x!r}, "
            f"shared={self.is_shared})"
        )


def make_var_state(
    version: VFTVersion | str = VFTVersion.V2,
) -> Callable[[], VarStateV1 | VarStateV2]:
    """Return a zero-arg factory that builds VarStates of the chosen version."""
    if isinstance(version, str):
        version = VFTVersion(version)
    if version is VFTVersion.V1:
        return VarStateV1
    if version is VFTVersion.V2:
        return VarStateV2
    raise ValueError(f"unknown VFTVersion: {version!r}")
