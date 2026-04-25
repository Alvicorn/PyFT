"""
Engine: central event dispatcher for PyFT.

Events handled:
  read(obj, attr)           - a thread read obj.attr
  write(obj, attr)          - a thread wrote obj.attr
  lock_acquire(lock_id)     - a thread acquired a lock
  lock_release(lock_id)     - a thread released a lock
  thread_start(parent_tid, child_tid, child_name)
  thread_finish(tid)        - thread is about to exit

Thread-safety: fine-grained per-variable, per-lock, and per-registry locks.
No global engine lock is held during access checks.
"""

from __future__ import annotations

import logging
import threading

from ..core.thread_state import ThreadRegistry, ThreadState
from ..core.var_state import ReadBottom, ReadEpoch, ReadVC
from ..core.vector_clock import VectorClock
from .race_log import (
    AccessInfo,
    RaceKind,
    RaceLog,
    RaceReport,
    _capture_stack,
    _safe_repr,
)
from .shadow_map import ShadowMap

log = logging.getLogger(__name__)


def _cur_tid() -> int:
    """Stable per-thread identity: id() of the current Thread object.

    Unlike threading.get_ident(), this is never reused within a session
    because Thread objects outlive their OS thread.
    """
    return id(threading.current_thread())


class LockState:
    __slots__ = ("_lock", "release_vc")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.release_vc: VectorClock | None = None


class Engine:
    def __init__(self) -> None:
        self.thread_registry = ThreadRegistry()
        self.shadow_map = ShadowMap()
        self.race_log = RaceLog()

        self._lock_states: dict[int, LockState] = {}
        self._lock_states_mu = threading.Lock()

        # per-thread reentrancy guard (depth counter)
        self._active = threading.local()
        # per-thread guard against recursive race recording
        self._recording = threading.local()

        # register main thread at construction (fork HB from birth)
        main_tid = id(threading.main_thread())
        self.thread_registry.register(main_tid, name="MainThread")

        log.info("Engine initialized")

    ### Re-entrancy guard ###

    def _enter(self) -> bool:
        if getattr(self._active, "depth", 0) > 0:
            return False
        self._active.depth = 1
        return True

    def _exit(self) -> None:
        self._active.depth = 0

    ### Public memory access API ###

    def read(self, obj: object, attr: str) -> None:
        if not self._enter():
            return
        try:
            self._read_impl(obj, attr)
        finally:
            self._exit()

    def write(self, obj: object, attr: str) -> None:
        if not self._enter():
            return
        try:
            self._write_impl(obj, attr)
        finally:
            self._exit()

    ### VerifiedFT read rule ###

    def _read_impl(self, obj: object, attr: str) -> None:
        tid = _cur_tid()
        ts = self.thread_registry.get_or_register(tid)
        ts.tick()
        var = self.shadow_map.get_or_create(obj, attr)

        # check_read returns (race, prior_write_epoch) atomically from
        # inside VarState's lock — no TOCTOU between check and reporting.
        race, prior_write_epoch = var.check_read(tid, ts.snapshot())

        if race:
            self._report_ww_or_wr(
                kind=RaceKind.WRITE_READ,
                obj=obj,
                attr=attr,
                current_tid=tid,
                current_ts=ts,
                is_write=False,
                prior_epoch=prior_write_epoch,
            )

    ### VerifiedFT write rule ###

    def _write_impl(self, obj: object, attr: str) -> None:
        tid = _cur_tid()
        ts = self.thread_registry.get_or_register(tid)
        ts.tick()
        var = self.shadow_map.get_or_create(obj, attr)

        # check_write returns (read_race, write_race, prior_write_epoch,
        # prior_read_state).  All captured atomically inside VarState's lock.
        # This eliminates the TOCTOU where an external snapshot would see
        # stale (bottom) values when two threads race to write concurrently.
        read_race, write_race, prior_write_epoch, prior_read_state = (
            var.check_write(tid, ts.snapshot())
        )

        if write_race:
            self._report_ww_or_wr(
                kind=RaceKind.WRITE_WRITE,
                obj=obj,
                attr=attr,
                current_tid=tid,
                current_ts=ts,
                is_write=True,
                prior_epoch=prior_write_epoch,
            )
        if read_race:
            self._report_rw(
                obj=obj,
                attr=attr,
                current_tid=tid,
                current_ts=ts,
                prior_read_state=prior_read_state,
            )

    ### Synchronization events ###

    def lock_acquire(self, lock_id: int) -> None:
        # If we're already inside an engine operation, the lock being acquired
        # is an internal PyFT lock. Treating it as a user HB edge would cause
        # child threads to absorb each other's VCs through PyFT-internal locks,
        # masking real races.  Skip it.
        if not self._enter():
            return
        try:
            tid = _cur_tid()
            ts = self.thread_registry.get_or_register(tid)
            ls = self._get_lock_state(lock_id)
            with ls._lock:
                if ls.release_vc is not None:
                    ts.absorb(ls.release_vc)
        finally:
            self._exit()

    def lock_release(self, lock_id: int) -> None:
        if not self._enter():
            return
        try:
            tid = _cur_tid()
            ts = self.thread_registry.get_or_register(tid)
            ls = self._get_lock_state(lock_id)
            with ls._lock:
                ts.tick()
                ls.release_vc = ts.snapshot()
        finally:
            self._exit()

    def thread_start(
        self, parent_tid: int, child_tid: int, child_name: str = ""
    ) -> None:
        """
        Fork HB edge: child inherits parent's VC; parent ticks.

        Idempotent for child registration. It is safe to call from
        both AutoTracker (parent thread) and SyncMonitor (child thread)
        when both are installed simultaneously.
        """
        parent_ts = self.thread_registry.get_or_register(parent_tid)
        child_vc = parent_ts.snapshot()

        # only register child if not already present to prevent
        # double-registration when AutoTracker + SyncMonitor coexist
        if self.thread_registry.get(child_tid) is None:
            self.thread_registry.register(
                child_tid, initial_vc=child_vc, name=child_name
            )
        parent_ts.tick()

    def thread_finish(self, tid: int) -> None:
        ts = self.thread_registry.get(tid)
        if ts is not None:
            ts.tick()

    def thread_join(self, joiner_tid: int, joinee_tid: int) -> None:
        joiner_ts = self.thread_registry.get_or_register(joiner_tid)
        joinee_ts = self.thread_registry.get(joinee_tid)
        if joinee_ts is not None:
            joiner_ts.absorb(joinee_ts.vc)
        self.thread_registry.remove(joinee_tid)

    ### Internal helpers ###

    def _get_lock_state(self, lock_id: int) -> LockState:
        with self._lock_states_mu:
            if lock_id not in self._lock_states:
                self._lock_states[lock_id] = LockState()
            return self._lock_states[lock_id]

    def _report_ww_or_wr(
        self,
        kind: RaceKind,
        obj: object,
        attr: str,
        current_tid: int,
        current_ts: ThreadState,
        is_write: bool,
        prior_epoch,
    ) -> None:
        if prior_epoch is None or prior_epoch.is_bottom():
            return

        if getattr(self._recording, "active", False):
            return
        self._recording.active = True
        try:
            obj_repr = _safe_repr(obj)
            access_a = AccessInfo(
                tid=current_tid,
                thread_name=current_ts.name,
                clock=current_ts.current_clock(),
                is_write=is_write,
                obj_repr=obj_repr,
                attr=attr,
                stack=_capture_stack(skip_frames=4),
            )
            prior_ts = self.thread_registry.get(prior_epoch.tid)
            prior_name = (
                prior_ts.name if prior_ts else f"Thread-{prior_epoch.tid}"
            )
            access_b = AccessInfo(
                tid=prior_epoch.tid,
                thread_name=prior_name,
                clock=prior_epoch.clock,
                is_write=(kind == RaceKind.WRITE_WRITE),
                obj_repr=obj_repr,
                attr=attr,
                stack=(),
            )
            self.race_log.record(
                RaceReport(
                    kind=kind,
                    access_a=access_a,
                    access_b=access_b,
                    obj_id=id(obj),
                    sequence=self.race_log.next_sequence(),
                )
            )
        finally:
            self._recording.active = False

    def _report_rw(
        self,
        obj: object,
        attr: str,
        current_tid: int,
        current_ts: ThreadState,
        prior_read_state,
    ) -> None:
        """Record a read-write race using a snapshot of the prior read state."""
        if isinstance(prior_read_state, ReadBottom):
            return

        if getattr(self._recording, "active", False):
            return
        self._recording.active = True
        try:
            obj_repr = _safe_repr(obj)
            access_a = AccessInfo(
                tid=current_tid,
                thread_name=current_ts.name,
                clock=current_ts.current_clock(),
                is_write=True,
                obj_repr=obj_repr,
                attr=attr,
                stack=_capture_stack(skip_frames=4),
            )
            if isinstance(prior_read_state, ReadEpoch):
                r_tid, r_clock = prior_read_state.tid, prior_read_state.clock
            elif isinstance(prior_read_state, ReadVC):
                r_tid, r_clock = next(
                    iter(prior_read_state.vc.items()), (0, 0)
                )
            else:
                return

            r_ts = self.thread_registry.get(r_tid)
            r_name = r_ts.name if r_ts else f"Thread-{r_tid}"
            access_b = AccessInfo(
                tid=r_tid,
                thread_name=r_name,
                clock=r_clock,
                is_write=False,
                obj_repr=obj_repr,
                attr=attr,
                stack=(),
            )
            self.race_log.record(
                RaceReport(
                    kind=RaceKind.READ_WRITE,
                    access_a=access_a,
                    access_b=access_b,
                    obj_id=id(obj),
                    sequence=self.race_log.next_sequence(),
                )
            )
        finally:
            self._recording.active = False
