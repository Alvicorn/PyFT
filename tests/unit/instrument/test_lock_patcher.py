"""
Tests for LockPatcher: monkey-patched lock factories that synchronously
dispatch engine.lock_acquire / engine.lock_release around the real OS
operation.
"""

from __future__ import annotations

import threading

import pytest

from pyft.detector.engine import Engine
from pyft.instrument.lock_patcher import (
    _ORIGINAL_BSEMAPHORE,
    _ORIGINAL_LOCK,
    _ORIGINAL_RLOCK,
    _ORIGINAL_SEMAPHORE,
    LockPatcher,
    _TrackedLock,
    _TrackedSemaphore,
)


def make_patcher() -> tuple[Engine, LockPatcher]:
    engine = Engine()
    return engine, LockPatcher(engine)


class TestInstall:
    def test_install_replaces_factories(self) -> None:
        engine, p = make_patcher()
        p.install()
        try:
            assert threading.Lock is not _ORIGINAL_LOCK
            assert threading.RLock is not _ORIGINAL_RLOCK
            assert threading.Semaphore is not _ORIGINAL_SEMAPHORE
            assert threading.BoundedSemaphore is not _ORIGINAL_BSEMAPHORE
        finally:
            p.uninstall()

    def test_uninstall_restores_factories(self) -> None:
        engine, p = make_patcher()
        p.install()
        p.uninstall()
        assert threading.Lock is _ORIGINAL_LOCK
        assert threading.RLock is _ORIGINAL_RLOCK
        assert threading.Semaphore is _ORIGINAL_SEMAPHORE
        assert threading.BoundedSemaphore is _ORIGINAL_BSEMAPHORE

    def test_double_install_is_idempotent(self) -> None:
        engine, p = make_patcher()
        p.install()
        factory_after_first = threading.Lock
        p.install()
        assert threading.Lock is factory_after_first
        p.uninstall()

    def test_uninstall_without_install_is_safe(self) -> None:
        engine, p = make_patcher()
        p.uninstall()  # must not raise


class TestTrackedLockBehavior:
    def test_acquire_calls_engine_lock_acquire(self) -> None:
        engine, p = make_patcher()
        p.install()
        try:
            calls = []
            engine.lock_acquire = lambda lid: calls.append(lid)
            lock = threading.Lock()
            lock.acquire()
            lock.release()
            assert calls == [lock._lock_id]
        finally:
            p.uninstall()

    def test_release_calls_engine_lock_release_before_os_release(self) -> None:
        """
        Verify the ordering invariant: engine.lock_release fires before
        the underlying OS release, by checking the engine event lands before
        the lock becomes available to the next acquirer.

        We can't monkey-patch _thread.lock methods (read-only), so we use
        a tracer: by the time engine.lock_release records, the OS lock
        must still be held.
        """
        engine, p = make_patcher()
        p.install()
        try:
            still_held = []

            def fake_release(lid: int) -> None:
                lock = current_lock[0]
                still_held.append(lock._real.locked())

            engine.lock_release = fake_release

            lock = threading.Lock()
            current_lock = [lock]
            lock.acquire()
            lock.release()

            assert still_held == [True], (
                f"engine.lock_release fired after OS release: {still_held}"
            )
        finally:
            p.uninstall()

    def test_with_lock_syntax(self) -> None:
        engine, p = make_patcher()
        p.install()
        try:
            events = []
            engine.lock_acquire = lambda lid: events.append(("acq", lid))
            engine.lock_release = lambda lid: events.append(("rel", lid))
            lock = threading.Lock()
            with lock:
                pass
            assert events == [("acq", lock._lock_id), ("rel", lock._lock_id)]
        finally:
            p.uninstall()

    def test_failed_nonblocking_acquire_does_not_fire_event(self) -> None:
        engine, p = make_patcher()
        p.install()
        try:
            calls = []
            engine.lock_acquire = lambda lid: calls.append(lid)
            lock = threading.Lock()
            lock.acquire()
            assert lock.acquire(blocking=False) is False
            assert len(calls) == 1  # only the first acquire fired
            lock.release()
        finally:
            p.uninstall()

    def test_locked_method_delegates(self) -> None:
        engine, p = make_patcher()
        p.install()
        try:
            lock = threading.Lock()
            assert lock.locked() is False
            lock.acquire()
            assert lock.locked() is True
            lock.release()
        finally:
            p.uninstall()

    def test_rlock_factory_returns_tracked(self) -> None:
        engine, p = make_patcher()
        p.install()
        try:
            lock = threading.RLock()
            assert isinstance(lock, _TrackedLock)
        finally:
            p.uninstall()


class TestTrackedSemaphore:
    def test_semaphore_factory_returns_tracked(self) -> None:
        engine, p = make_patcher()
        p.install()
        try:
            s = threading.Semaphore(2)
            assert isinstance(s, _TrackedSemaphore)
        finally:
            p.uninstall()

    def test_semaphore_acquire_release_fires_engine(self) -> None:
        engine, p = make_patcher()
        p.install()
        try:
            events = []
            engine.lock_acquire = lambda lid: events.append(("acq", lid))
            engine.lock_release = lambda lid: events.append(("rel", lid))
            s = threading.Semaphore(2)
            s.acquire()
            s.release()
            assert ("acq", s._lock_id) in events
            assert ("rel", s._lock_id) in events
        finally:
            p.uninstall()


class TestConditionInterop:
    """
    threading.Condition adopts ``_is_owned`` / ``_release_save`` /
    ``_acquire_restore`` from its underlying lock when present. The
    default ``Condition()`` uses an RLock, where the fallback
    ``_is_owned`` (acquire(False) + release) is unsound — it always
    reports "not owned" because RLock.acquire(False) succeeds on the
    owning thread. Without forwarding, ``cv.notify_all()`` inside
    ``with cv:`` raises ``RuntimeError: cannot notify on un-acquired
    lock``.
    """

    def test_default_condition_is_owned_reflects_with_block(self) -> None:
        engine, p = make_patcher()
        p.install()
        try:
            cv = threading.Condition()
            assert cv._is_owned() is False
            with cv:
                assert cv._is_owned() is True
            assert cv._is_owned() is False
        finally:
            p.uninstall()

    def test_default_condition_notify_all_does_not_raise(self) -> None:
        engine, p = make_patcher()
        p.install()
        try:
            cv = threading.Condition()
            with cv:
                cv.notify_all()  # must not raise
        finally:
            p.uninstall()

    def test_plain_lock_condition_falls_back_to_default_is_owned(self) -> None:
        # Condition with an explicit plain Lock should still work: the
        # tracker must NOT expose _is_owned for a plain Lock (whose C
        # type lacks _is_owned), so Condition falls back to its
        # acquire(False)-probe path.
        engine, p = make_patcher()
        p.install()
        try:
            lock = threading.Lock()
            assert not hasattr(lock, "_is_owned")
            cv = threading.Condition(lock)
            with cv:
                cv.notify_all()  # must not raise
        finally:
            p.uninstall()

    def test_wait_fires_release_and_acquire_engine_events(self) -> None:
        engine, p = make_patcher()
        p.install()
        try:
            events: list[tuple[str, int]] = []
            engine.lock_acquire = lambda lid: events.append(("acq", lid))
            engine.lock_release = lambda lid: events.append(("rel", lid))

            cv = threading.Condition()
            flag = [False]
            lock_id = cv._lock._lock_id

            def waiter() -> None:
                with cv:
                    cv.wait_for(lambda: flag[0])

            t = threading.Thread(target=waiter)
            t.start()
            # Spin briefly so the waiter reaches cv.wait_for before we notify
            import time

            time.sleep(0.05)
            with cv:
                flag[0] = True
                cv.notify_all()
            t.join(timeout=2.0)
            assert not t.is_alive()

            # The wait path must produce at least one release/acquire pair
            # on the cv's underlying lock (via _release_save /
            # _acquire_restore). Without forwarding, those engine events
            # would be missing entirely.
            wait_rels = sum(1 for e in events if e == ("rel", lock_id))
            wait_acqs = sum(1 for e in events if e == ("acq", lock_id))
            assert wait_rels >= 2  # waiter's with-exit + _release_save
            assert wait_acqs >= 2  # waiter's with-enter + _acquire_restore
        finally:
            p.uninstall()


@pytest.mark.timeout(10)
class TestLockPatcherProvidesHB:
    """
    End-to-end: with LockPatcher, two threads under a shared lock
    should produce zero races on the protected variable.
    """

    def test_two_writers_under_lock_no_race(self) -> None:
        from pyft.instrument.wrappers import AutoTracker, TrackedProxy

        engine = Engine()
        patcher = LockPatcher(engine)
        tracker = AutoTracker(engine)
        patcher.install()
        tracker.install()
        try:
            lock = threading.Lock()

            class C:
                def __init__(self) -> None:
                    self.v = 0

            shared = TrackedProxy(C(), engine)

            def worker() -> None:
                for _ in range(50):
                    with lock:
                        shared.v += 1

            ts = [threading.Thread(target=worker) for _ in range(2)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()

            assert engine.race_log.all_reports() == []
        finally:
            tracker.uninstall()
            patcher.uninstall()
