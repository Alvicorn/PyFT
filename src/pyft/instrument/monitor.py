"""
Register two event callbacks:

  CALL      -> on_call(code, offset, callable_, arg0)
               Identifies sync primitives via sync_observer.classify_call().
               For events that need RETURN correlation (acquire, thread.start,
               thread.join), push a SyncCall onto a per-thread deque.
               For events that fire-on-CALL (release, event.set, cond.notify),
               dispatch to the engine immediately.

  PY_RETURN -> on_return(code, offset, retval)
               Pops the pending SyncCall for this thread and dispatches to
               the engine, conditional on retval (acquire must return True).

Thread lifecycle:

    Thread.start():
    CALL   -> push pending THREAD_START (parent_tid captured here)
    RETURN -> pop, child.ident is now valid -> engine.thread_start(parent, child)

    Thread.join():
    CALL   -> push pending THREAD_JOIN
    RETURN -> pop, check not thread.is_alive() -> engine.thread_join(joiner, joinee)

    Condition.wait():
    CALL   -> push pending COND_WAIT (condition.acquire was already observed)
    RETURN -> pop -> engine.lock_acquire(condition_lock) (wait releases the lock then
                re-acquires it on return. The re-acquire is modeled as the relevant HB edge)

    Event.set():
    CALL   -> engine.event_set(event_id) immediately (no retval needed)

    Event.wait():
    CALL   -> push pending EVENT_WAIT
    RETURN -> pop, if retval (True=event was set) -> engine.event_wait(event_id)

    Barrier.wait():
    Barrier is modelled as a symmetric lock: all arrivals establish mutual HB.
    CALL   -> push pending BARRIER_WAIT
    RETURN -> pop → engine.barrier_release(barrier_id)
"""

from __future__ import annotations

import collections
import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict

from .sync_observer import SyncCall, SyncKind, classify_call

if TYPE_CHECKING:
    from ..detector.engine import Engine

log = logging.getLogger(__name__)


# using sys.monitoring.OPTIMIZER_ID (slot 5) rather than
# DEBUGGER_ID (slot 0) to avoid conflicting with actual debuggers.
# Users can override with PYFT_MONITOR_TOOL_ID env var if they
# need slot 5 for something else.
_DEFAULT_TOOL_ID = sys.monitoring.OPTIMIZER_ID


def _get_tool_id() -> int:
    try:
        return int(os.environ.get("PYFT_MONITOR_TOOL_ID", _DEFAULT_TOOL_ID))
    except ValueError:
        return _DEFAULT_TOOL_ID


_EVENTS = sys.monitoring.events

# CALL is used for all entries and PY_RETURN for correlation.
# PY_RETURN is cheaper than C_RETURN. it only fires for Python-implemented
# functions. Lock.acquire is C-implemented, so C_RETURN is also required.
_WANTED_EVENTS = (
    _EVENTS.CALL
    | _EVENTS.PY_RETURN
    | _EVENTS.C_RETURN
    | _EVENTS.C_RAISE
    | _EVENTS.PY_START  # thread entry point detection
)


class SyncMonitor:
    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        self._tool_id = _get_tool_id()
        self._installed = False

        # per-thread deque of pending SyncCall objects.
        # (CALL and RETURN always fire on the same thread).
        self._pending: threading.local = threading.local()

        self._pending_starts: Dict[int, int] = {}
        self._pending_starts_lock = threading.Lock()

        # Threading.Event id: set of thread ids that have called event.set()
        # used to establish HB: event.set() in T1 HB-before event.wait() in T2
        self._event_setters: dict[int, set[int]] = {}
        self._event_lock = threading.Lock()

        # barrier tracking: barrier_id -> list of participant tids
        self._barrier_participants: dict[int, list[int]] = {}
        self._barrier_lock = threading.Lock()
        log.info("Initialized SyncMonitor")

    def install(self) -> None:
        if self._installed:
            return

        tid = self._tool_id

        try:
            sys.monitoring.use_tool_id(tid, "pyft")
        except ValueError as e:
            log.error(f"SyncMonitor installation error: {e}")
            raise e

        sys.monitoring.register_callback(tid, _EVENTS.CALL, self._on_call)
        sys.monitoring.register_callback(
            tid, _EVENTS.PY_RETURN, self._on_return
        )
        sys.monitoring.register_callback(
            tid, _EVENTS.C_RETURN, self._on_c_return
        )
        sys.monitoring.register_callback(
            tid, _EVENTS.PY_START, self._on_py_start
        )

        sys.monitoring.set_events(tid, _WANTED_EVENTS)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        tid = self._tool_id
        sys.monitoring.set_events(tid, _EVENTS.NO_EVENTS)
        sys.monitoring.register_callback(tid, _EVENTS.CALL, None)
        sys.monitoring.register_callback(tid, _EVENTS.PY_RETURN, None)
        sys.monitoring.register_callback(tid, _EVENTS.C_RETURN, None)
        sys.monitoring.register_callback(tid, _EVENTS.PY_START, None)
        sys.monitoring.free_tool_id(tid)

        self._installed = False

    ### Reentrancy guard ###
    # The monitoring callbacks themselves may trigger further Python calls.
    # Guard with a per-thread flag (_in_callback) to prevent recursion.

    def _enter_callback(self) -> bool:
        """Returns True if we should proceed (not reentrant)."""
        if getattr(self._pending, "in_callback", False):
            return False
        self._pending.in_callback = True
        return True

    def _exit_callback(self) -> None:
        self._pending.in_callback = False

    def _get_pending_stack(self) -> collections.deque[SyncCall]:
        if not hasattr(self._pending, "stack"):
            self._pending.stack = collections.deque()
        return self._pending.stack

    ### CALLBACKS ###

    def _on_call(
        self, code: Any, instruction_offset: int, callable_: Any, arg0: Any
    ) -> Any:
        """
        Fired before any callable is invoked.
        sys.monitoring passes (code_object, offset, callable, arg0).
        """
        if not self._enter_callback():
            return sys.monitoring.DISABLE

        try:
            sc = classify_call(callable_, arg0)
            if sc is None:
                return

            stack = self._get_pending_stack()

            match sc.kind:
                # events that fire immediately (no return value needed)
                case SyncKind.LOCK_RELEASE:
                    self.engine.lock_release(id(sc.target))
                case SyncKind.EVENT_SET:
                    self._handle_event_set(sc.target)
                case SyncKind.COND_NOTIFY:
                    self.engine.lock_release(id(sc.target))

                # defer to _on_py_start
                case SyncKind.THREAD_START:
                    with self._pending_starts_lock:
                        parent_tid = id(threading.current_thread())
                        self._pending_starts[id(sc.target)] = parent_tid

                # events that return correlation - push onto stack
                case (
                    SyncKind.LOCK_ACQUIRE
                    | SyncKind.COND_WAIT
                    | SyncKind.THREAD_JOIN
                    | SyncKind.EVENT_WAIT
                    | SyncKind.BARRIER_WAIT
                ):
                    stack.append(sc)

        finally:
            self._exit_callback()

    def _on_return(
        self, code: Any, instruction_offset: int, retval: Any
    ) -> None:
        """Fired when a Python function returns."""
        self._handle_return(retval)

    def _on_c_return(self, *args) -> None:
        """
        Fired when a C function returns (e.g. Lock.acquire is C-level).
        Args could be:
            (code, offset, callable, arg0, retval)
            (code, offset, retval)

        """
        if not args:
            return

        retval = args[-1]
        self._handle_return(retval)

    def _handle_return(self, retval: Any) -> None:
        if not self._enter_callback():
            return

        try:
            stack = self._get_pending_stack()
            if not stack:
                return

            sc = stack[-1]
            match sc.kind:
                case SyncKind.LOCK_ACQUIRE:
                    stack.pop()

                    # only record HB edge if acquire actually succeeded
                    # if retval is True or retval is None:
                    if retval is not False:
                        self.engine.lock_acquire(id(sc.target))

                case SyncKind.THREAD_JOIN:
                    stack.pop()
                    thread = sc.target
                    if thread.ident is not None:
                        self.engine.thread_join(
                            joiner_tid=id(threading.current_thread()),
                            joinee_tid=id(thread),
                        )

                case SyncKind.COND_WAIT:
                    stack.pop()

                    # condition.wait() released and re-acquired the underlying lock.
                    # Model as: acquire on return (establishes HB from notifier)
                    # The underlying lock is sc.target._lock
                    underlying = getattr(sc.target, "_lock", sc.target)
                    self.engine.lock_acquire(id(underlying))

                case SyncKind.EVENT_WAIT:
                    stack.pop()
                    if retval is True:
                        self._handle_event_wait_return(sc.target)

                case SyncKind.BARRIER_WAIT:
                    stack.pop()
                    self._handle_barrier_wait_return(sc.target)

        finally:
            self._exit_callback()

    def _on_py_start(self, code: Any, instruction_offset: int) -> None:
        """
        Fired when a new Python frame starts executing.
        Use this to register threads that bypass Thread.start()
        (e.g. threads created by C extensions).
        """
        if not self._enter_callback():
            return
        try:
            current_thread = threading.current_thread()
            tid = id(current_thread)
            self.engine.thread_registry.get_or_register(tid)

            with self._pending_starts_lock:
                parent_tid = self._pending_starts.pop(tid, None)

            if parent_tid is not None:
                self.engine.thread_start(
                    parent_tid=parent_tid,
                    child_tid=tid,
                    child_name=current_thread.name,
                )

        finally:
            self._exit_callback()

    ### HELPERS ###

    def _handle_event_set(self, event: threading.Event) -> None:
        """
        Event.set() establishes: setter's current VC -> any future waiter.
        Modelled as a lock release: the event 'lock' is released by
        the setter thread, and waiters 'acquire' it.
        """
        eid = id(event)
        self.engine.lock_release(eid)
        with self._event_lock:
            if eid not in self._event_setters:
                self._event_setters[eid] = set()
            self._event_setters[eid].add(threading.get_ident())

    def _handle_event_wait_return(self, event: threading.Event) -> None:
        """
        Event.wait() returned True: waiter absorbs setter's VC.
        """
        eid = id(event)
        self.engine.lock_acquire(eid)

    def _handle_barrier_wait_return(self, barrier: threading.Barrier) -> None:
        """
        All threads that called barrier.wait() establish mutual HB.
        Modelled as: every participant releases a virtual lock,
        then every participant acquires all participants' virtual locks.

        Simplified model: use one shared 'barrier release VC' that all
        participants contribute to and absorb from, via a dedicated
        engine lock ID per barrier.wait() generation.

        Since we can't know when the barrier fires (vs when individual
        threads return), use on return, do a release then acquire
        on the same barrier_id. This means each thread absorbs every
        other thread's VC that arrived before it.
        """
        bid = id(barrier)
        self.engine.lock_release(bid)
        self.engine.lock_acquire(bid)
