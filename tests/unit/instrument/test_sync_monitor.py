import sys
import threading
import time
from typing import Tuple
from unittest.mock import patch

import pytest

# Guard must come BEFORE importing SyncMonitor — monitor.py uses sys.monitoring
# at module level which doesn't exist on Python < 3.12.
if sys.version_info < (3, 12):
    pytest.skip(
        "SyncMonitor requires Python 3.12+ (sys.monitoring)",
        allow_module_level=True,
    )

from pyft.detector.engine import Engine
from pyft.instrument.monitor import SyncMonitor


def make_monitor() -> Tuple[Engine, SyncMonitor]:
    engine = Engine()
    monitor = SyncMonitor(engine)
    return engine, monitor


class TestInstallation:
    def test_install_and_uninstall(self) -> None:
        engine, monitor = make_monitor()
        monitor.install()
        assert monitor._installed

        monitor.install()  # second call should be no-op
        assert monitor._installed

        monitor.uninstall()
        assert not monitor._installed

    def test_uninstall_without_install_is_safe(self) -> None:
        engine, monitor = make_monitor()
        monitor.uninstall()  # should not raise

    def test_reinstall_after_uninstall(self) -> None:
        engine, monitor = make_monitor()
        for _ in range(5):
            monitor.install()
            assert monitor._installed

            monitor.uninstall()
            assert not monitor._installed

    def test_uses_optimizer_slot_by_default(self) -> None:
        engine, monitor = make_monitor()
        assert monitor._tool_id == sys.monitoring.OPTIMIZER_ID

    def test_install_value_error_caught_and_reraised(self) -> None:
        """Simulate use_tool_id raising ValueError."""
        engine, monitor = make_monitor()
        with patch("sys.monitoring.use_tool_id") as mock_use:
            mock_use.side_effect = ValueError("slot already in use")
            with pytest.raises(ValueError):
                monitor.install()
            assert not monitor._installed


class TestThreadObservation:
    def test_thread_join_observed(self) -> None:
        engine, monitor = make_monitor()
        monitor.install()
        try:
            joins = []
            engine.thread_join = lambda joiner_tid, joinee_tid: joins.append(
                (joiner_tid, joinee_tid)
            )

            t = threading.Thread(target=lambda: None)
            t.start()
            t.join()

            assert len(joins) == 1
            joiner, joinee = joins[0]
            assert joiner == id(threading.current_thread())
            assert joinee == id(t)
        finally:
            monitor.uninstall()

    def test_thread_join_with_timeout_success_observed(self) -> None:
        engine, monitor = make_monitor()
        monitor.install()
        try:
            joins = []
            engine.thread_join = lambda joiner_tid, joinee_tid: joins.append(
                (joiner_tid, joinee_tid)
            )

            t = threading.Thread(target=lambda: time.sleep(0.1))
            t.start()
            t.join(timeout=1.0)

            assert len(joins) == 1
            joiner, joinee = joins[0]
            assert joiner == id(threading.current_thread())
            assert joinee == id(t)
        finally:
            monitor.uninstall()

    def test_thread_join_with_timeout_expired_observed(self) -> None:
        engine, monitor = make_monitor()
        monitor.install()
        try:
            joins = []
            engine.thread_join = lambda joiner_tid, joinee_tid: joins.append(
                (joiner_tid, joinee_tid)
            )

            t = threading.Thread(target=lambda: time.sleep(3))
            t.start()
            t.join(timeout=1.0)

            assert len(joins) == 1
            joiner, joinee = joins[0]
            assert joiner == id(threading.current_thread())
            assert joinee == id(t)
        finally:
            monitor.uninstall()

    def test_no_join_no_join_event(self) -> None:
        """If we don't call join(), no thread_join event should fire."""
        engine, monitor = make_monitor()
        monitor.install()
        try:
            joins = []
            engine.thread_join = lambda joiner_tid, joinee_tid: joins.append(
                (joiner_tid, joinee_tid)
            )

            done = threading.Event()
            t = threading.Thread(target=lambda: done.set())
            t.start()
            done.wait()
            t.join()  # we do join here to clean up

            # Only one join (ours)
            assert len(joins) == 1
        finally:
            monitor.uninstall()


class TestEventObservation:
    def test_event_set_and_wait_observed(self) -> None:
        engine, monitor = make_monitor()
        monitor.install()
        try:
            releases = []
            acquires = []
            engine.lock_release = lambda lock_id: releases.append(lock_id)
            engine.lock_acquire = lambda lock_id: acquires.append(lock_id)

            ev = threading.Event()
            eid = id(ev)

            result_holder = []

            def waiter() -> None:
                ev.wait()
                result_holder.append(True)

            t = threading.Thread(target=waiter)
            t.start()
            time.sleep(0.01)
            ev.set()
            t.join()

            # event.set() should have triggered a release on event's id
            assert eid in releases
        finally:
            monitor.uninstall()


class TestBarrierObservation:
    def test_barrier_wait_triggers_release_and_acquire(self) -> None:
        engine, monitor = make_monitor()
        monitor.install()
        try:
            releases = []
            acquires = []
            engine.lock_release = lambda lock_id: releases.append(lock_id)
            engine.lock_acquire = lambda lock_id: acquires.append(lock_id)

            barrier = threading.Barrier(2)
            bid = id(barrier)

            def participant() -> None:
                barrier.wait()

            t1 = threading.Thread(target=participant)
            t2 = threading.Thread(target=participant)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # Each participant does one release and one acquire
            assert releases.count(bid) == 2
            assert acquires.count(bid) == 2
        finally:
            monitor.uninstall()


class TestMonitorNoMonkeyPatching:
    def test_lock_acquire_method_unchanged(self) -> None:
        """Core invariant: SyncMonitor must not modify threading.Lock.acquire."""
        original_acquire = threading.Lock.acquire
        engine, monitor = make_monitor()
        monitor.install()
        assert threading.Lock.acquire is original_acquire
        monitor.uninstall()
        assert threading.Lock.acquire is original_acquire

    def test_lock_release_method_unchanged(self) -> None:
        original_release = threading.Lock.release
        engine, monitor = make_monitor()
        monitor.install()
        assert threading.Lock.release is original_release
        monitor.uninstall()

    def test_thread_start_method_unchanged(self) -> None:
        original_start = threading.Thread.start
        engine, monitor = make_monitor()
        monitor.install()
        assert threading.Thread.start is original_start
        monitor.uninstall()

    def test_thread_join_method_unchanged(self) -> None:
        original_join = threading.Thread.join
        engine, monitor = make_monitor()
        monitor.install()
        assert threading.Thread.join is original_join
        monitor.uninstall()

    def test_rlock_unchanged(self) -> None:
        import _thread

        original_acquire = _thread.RLock.acquire
        engine, monitor = make_monitor()
        monitor.install()
        assert _thread.RLock.acquire is original_acquire
        monitor.uninstall()
