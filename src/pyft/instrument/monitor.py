"""
SyncMonitor: sys.monitoring (PEP 669)-based observer for sync events.

Registered events:

  CALL      -> identifies sync primitives via ``classify_call``. For
               return-correlated events (acquire, thread.join, wait)
               pushes a SyncCall onto a per-thread deque. For
               fire-on-call events (release, event.set, cond.notify)
               dispatches to the engine immediately.

  PY_RETURN -> pops the pending SyncCall and dispatches conditionally
               on the return value (acquire must succeed).

  C_RETURN  -> same handling for C-level calls (Lock.acquire is C).

Note: in the current public install path PyFT uses synchronous
monkey-patches (``LockPatcher``, ``AutoTracker``) instead of this
observer, because the asynchronous monitoring callbacks don't preserve
the OS-level ordering of releases vs. acquires across threads under
free-threaded Python. SyncMonitor remains here for users who want
zero-patch observation and accept the resulting flakiness on lock-heavy
workloads.
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import types
from typing import TYPE_CHECKING, Any, Dict

from .sync_observer import SyncCall, SyncKind, classify_call

if TYPE_CHECKING:
    from ..detector.engine import Engine


# OPTIMIZER_ID (slot 5) chosen over DEBUGGER_ID to avoid conflicts with
# actual debuggers. Override via PYFT_MONITOR_TOOL_ID if needed.
_DEFAULT_TOOL_ID = sys.monitoring.OPTIMIZER_ID


def _get_tool_id() -> int:
    try:
        return int(os.environ.get("PYFT_MONITOR_TOOL_ID", _DEFAULT_TOOL_ID))
    except ValueError:
        return _DEFAULT_TOOL_ID


_EVENTS = sys.monitoring.events

# PY_START is omitted on purpose: it fires on every Python frame entry,
# which is extremely expensive and only useful for catching threads
# started outside threading.Thread.start (e.g. by C extensions).
_WANTED_EVENTS = (
    _EVENTS.CALL | _EVENTS.PY_RETURN | _EVENTS.C_RETURN | _EVENTS.C_RAISE
)


class SyncMonitor:
    """
    sys.monitoring-based observer that dispatches sync events to the
    engine. See module docstring for the trade-offs.
    """

    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        self._tool_id = _get_tool_id()
        self._installed = False

        self._pending: threading.local = threading.local()

        self._pending_starts: Dict[int, int] = {}
        self._pending_starts_lock = threading.Lock()

        self._event_setters: dict[int, set[int]] = {}
        self._event_lock = threading.Lock()

        self._barrier_participants: dict[int, list[int]] = {}
        self._barrier_lock = threading.Lock()

    def install(self) -> None:
        if self._installed:
            return

        tid = self._tool_id

        try:
            sys.monitoring.use_tool_id(tid, "pyft")
        except ValueError as e:
            raise e

        sys.monitoring.register_callback(tid, _EVENTS.CALL, self._on_call)
        sys.monitoring.register_callback(
            tid, _EVENTS.PY_RETURN, self._on_return
        )
        sys.monitoring.register_callback(
            tid, _EVENTS.C_RETURN, self._on_c_return
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
        sys.monitoring.free_tool_id(tid)

        self._installed = False

    def _enter_callback(self) -> bool:
        """
        Return True if not already inside a monitoring callback on
        this thread.
        """
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

    def _on_call(
        self,
        code: types.CodeType,
        instruction_offset: int,
        callable_: Any,  # noqa: ANN401
        arg0: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        # Return None (not sys.monitoring.DISABLE) on re-entry: DISABLE
        # permanently silences the call site, hiding subsequent user
        # acquires on the same source line.
        if not self._enter_callback():
            return None

        try:
            sc = classify_call(callable_, arg0)
            if sc is None:
                return

            stack = self._get_pending_stack()

            match sc.kind:
                case SyncKind.LOCK_RELEASE:
                    self.engine.lock_release(id(sc.target))
                case SyncKind.EVENT_SET:
                    self._handle_event_set(sc.target)
                case SyncKind.COND_NOTIFY:
                    self.engine.lock_release(id(sc.target))

                case SyncKind.THREAD_START:
                    with self._pending_starts_lock:
                        parent_tid = id(threading.current_thread())
                        self._pending_starts[id(sc.target)] = parent_tid

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
        self,
        code: types.CodeType,
        instruction_offset: int,
        retval: Any,  # noqa: ANN401
    ) -> None:
        self._handle_return(retval)

    def _on_c_return(self, *args: Any) -> None:  # noqa: ANN401
        if not args:
            return
        retval = args[-1]
        self._handle_return(retval)

    def _handle_return(self, retval: Any) -> None:  # noqa: ANN401
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
                    # Condition.wait released and re-acquired the
                    # underlying lock; model the re-acquire as the HB
                    # edge from the notifier.
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

    def _on_py_start(
        self, code: types.CodeType, instruction_offset: int
    ) -> None:
        """
        Optional callback to catch threads started outside
        ``threading.Thread.start``. Not registered by default — see the
        module docstring.
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

    def _handle_event_set(self, event: threading.Event) -> None:
        eid = id(event)
        self.engine.lock_release(eid)
        with self._event_lock:
            if eid not in self._event_setters:
                self._event_setters[eid] = set()
            self._event_setters[eid].add(threading.get_ident())

    def _handle_event_wait_return(self, event: threading.Event) -> None:
        eid = id(event)
        self.engine.lock_acquire(eid)

    def _handle_barrier_wait_return(self, barrier: threading.Barrier) -> None:
        # Each thread releases then acquires on the same barrier id, so
        # every participant absorbs every other's VC.
        bid = id(barrier)
        self.engine.lock_release(bid)
        self.engine.lock_acquire(bid)
