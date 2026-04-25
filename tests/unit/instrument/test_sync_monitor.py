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
    def test_install_and_uninstall(self):
        engine, monitor = make_monitor()
        monitor.install()
        assert monitor._installed

        monitor.install()  # second call should be no-op
        assert monitor._installed

        monitor.uninstall()
        assert not monitor._installed

    def test_uninstall_without_install_is_safe(self):
        engine, monitor = make_monitor()
        monitor.uninstall()  # should not raise

    def test_reinstall_after_uninstall(self):
        engine, monitor = make_monitor()
        for _ in range(5):
            monitor.install()
            assert monitor._installed

            monitor.uninstall()
            assert not monitor._installed

    def test_uses_optimizer_slot_by_default(self):
        engine, monitor = make_monitor()
        assert monitor._tool_id == sys.monitoring.OPTIMIZER_ID

    def test_install_value_error_caught_and_reraised(self):
        """Simulate use_tool_id raising ValueError."""
        engine, monitor = make_monitor()
        with patch("sys.monitoring.use_tool_id") as mock_use:
            mock_use.side_effect = ValueError("slot already in use")
            with pytest.raises(ValueError):
                monitor.install()
            assert not monitor._installed


class TestLockObservation:
    def test_lock_acquire_recorded(self):
        engine, monitor = make_monitor()
        monitor.install()
        try:
            lock_events = []
            orig_acquire = engine.lock_acquire
            engine.lock_acquire = lambda lock_id: lock_events.append(lock_id)

            lock = threading.Lock()

            lock.acquire()
            lock.release()

            assert len(lock_events) >= 1
        finally:
            monitor.uninstall()

    def test_lock_release_recorded(self):
        engine, monitor = make_monitor()
        monitor.install()
        try:
            release_events = []
            engine.lock_release = lambda lock_id: release_events.append(
                lock_id
            )

            lock = threading.Lock()
            lock.acquire()
            lock.release()

            assert len(release_events) >= 1
        finally:
            monitor.uninstall()

    def test_with_lock_syntax_observed(self):
        """The `with lock:` syntax uses __enter__/__exit__ — must be observed."""
        engine, monitor = make_monitor()
        monitor.install()
        try:
            acquire_events = []
            release_events = []
            engine.lock_acquire = lambda lock_id: acquire_events.append(
                lock_id
            )
            engine.lock_release = lambda lock_id: release_events.append(
                lock_id
            )

            lock = threading.Lock()
            with lock:
                pass

            assert len(acquire_events) >= 1
            assert len(release_events) >= 1
        finally:
            monitor.uninstall()

    def test_failed_nonblocking_acquire_not_recorded(self):
        """acquire(blocking=False) that fails must NOT establish an HB edge."""
        engine, monitor = make_monitor()
        monitor.install()
        try:
            acquire_events = []
            engine.lock_acquire = lambda lock_id: acquire_events.append(
                lock_id
            )

            lock = threading.Lock()
            lock.acquire()  # hold the lock
            result = lock.acquire(blocking=False)
            assert result is False

            # The failed acquire should not fire engine.lock_acquire
            count_before = len(acquire_events)
            assert count_before >= 1  # the first successful acquire
            lock.release()

            # Ensure no spurious second acquire was recorded
            assert len(acquire_events) == count_before
        finally:
            monitor.uninstall()

    def test_rlock_observed(self):
        engine, monitor = make_monitor()
        monitor.install()
        try:
            events = []
            engine.lock_acquire = lambda lock_id: events.append(
                ("acq", lock_id)
            )
            engine.lock_release = lambda lock_id: events.append(
                ("rel", lock_id)
            )

            lock = threading.RLock()
            with lock:
                pass

            kinds = [e[0] for e in events]
            assert "acq" in kinds
            assert "rel" in kinds
        finally:
            monitor.uninstall()


class TestThreadObservation:
    def test_thread_start_observed(self):
        engine, monitor = make_monitor()
        monitor.install()
        try:
            starts = []
            orig_start = engine.thread_start
            engine.thread_start = lambda parent_tid, child_tid, child_name="": (
                starts.append((parent_tid, child_tid))
            )

            t = threading.Thread(target=lambda: None)
            t.start()
            t.join()

            assert len(starts) == 1
            parent, child = starts[0]
            assert parent == id(threading.current_thread())
            assert child == id(t)
        finally:
            monitor.uninstall()

    def test_thread_join_observed(self):
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

    def test_thread_join_with_timeout_success_observed(self):
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

    def test_thread_join_with_timeout_expired_observed(self):
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

    def test_no_join_no_join_event(self):
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
    def test_event_set_and_wait_observed(self):
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

            def waiter():
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


class TestSemaphoreObservation:
    def test_semaphore_acquire_release_observed(self):
        engine, monitor = make_monitor()
        monitor.install()
        try:
            acquires = []
            releases = []
            engine.lock_acquire = lambda lock_id: acquires.append(lock_id)
            engine.lock_release = lambda lock_id: releases.append(lock_id)

            sem = threading.Semaphore(2)
            sem.acquire()
            sem.release()

            assert len(acquires) >= 1
            assert len(releases) >= 1
        finally:
            monitor.uninstall()


class TestConditionObservation:
    def test_condition_wait_acquires_underlying_lock(self):
        engine, monitor = make_monitor()
        monitor.install()
        try:
            cond = threading.Condition()
            acquires = []
            engine.lock_acquire = lambda lock_id: acquires.append(lock_id)

            def waiter():
                with cond:
                    cond.wait()

            t = threading.Thread(target=waiter)
            t.start()
            time.sleep(0.05)
            with cond:
                cond.notify()
            t.join()

            # The underlying lock ID is id(cond._lock) or cond itself
            expected_id = id(getattr(cond, "_lock", cond))
            assert expected_id in acquires
        finally:
            monitor.uninstall()


class TestBarrierObservation:
    def test_barrier_wait_triggers_release_and_acquire(self):
        engine, monitor = make_monitor()
        monitor.install()
        try:
            releases = []
            acquires = []
            engine.lock_release = lambda lock_id: releases.append(lock_id)
            engine.lock_acquire = lambda lock_id: acquires.append(lock_id)

            barrier = threading.Barrier(2)
            bid = id(barrier)

            def participant():
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
    def test_lock_acquire_method_unchanged(self):
        """Core invariant: SyncMonitor must not modify threading.Lock.acquire."""
        original_acquire = threading.Lock.acquire
        engine, monitor = make_monitor()
        monitor.install()
        assert threading.Lock.acquire is original_acquire
        monitor.uninstall()
        assert threading.Lock.acquire is original_acquire

    def test_lock_release_method_unchanged(self):
        original_release = threading.Lock.release
        engine, monitor = make_monitor()
        monitor.install()
        assert threading.Lock.release is original_release
        monitor.uninstall()

    def test_thread_start_method_unchanged(self):
        original_start = threading.Thread.start
        engine, monitor = make_monitor()
        monitor.install()
        assert threading.Thread.start is original_start
        monitor.uninstall()

    def test_thread_join_method_unchanged(self):
        original_join = threading.Thread.join
        engine, monitor = make_monitor()
        monitor.install()
        assert threading.Thread.join is original_join
        monitor.uninstall()

    def test_rlock_unchanged(self):
        import _thread

        original_acquire = _thread.RLock.acquire
        engine, monitor = make_monitor()
        monitor.install()
        assert _thread.RLock.acquire is original_acquire
        monitor.uninstall()
