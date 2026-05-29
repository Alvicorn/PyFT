import threading
from typing import Callable

import pytest

from pyft.instrument.sync_observer import SyncKind, classify_call

# ── Helpers ──────────────────────────────────────────────────────────


def _bound(obj: object, method_name: str) -> tuple[Callable, object]:
    """Return the bound method and the object (simulating CALL arg0)."""
    method = getattr(obj, method_name)
    return method, obj


class TestLockClassification:
    """
    Lock / RLock / Semaphore are handled by LockPatcher (synchronous
    wrapping), not by sys.monitoring. classify_call must return None
    for them so we don't double-dispatch engine events.
    """

    def test_lock_acquire_not_classified(self) -> None:
        lock = threading.Lock()
        fn, self_ = _bound(lock, "acquire")
        assert classify_call(fn, self_) is None

    def test_lock_release_not_classified(self) -> None:
        lock = threading.Lock()
        fn, self_ = _bound(lock, "release")
        assert classify_call(fn, self_) is None

    def test_lock_enter_not_classified(self) -> None:
        lock = threading.Lock()
        fn, self_ = _bound(lock, "__enter__")
        assert classify_call(fn, self_) is None

    def test_lock_exit_not_classified(self) -> None:
        lock = threading.Lock()
        fn, self_ = _bound(lock, "__exit__")
        assert classify_call(fn, self_) is None

    def test_rlock_acquire_not_classified(self) -> None:
        lock = threading.RLock()
        fn, self_ = _bound(lock, "acquire")
        assert classify_call(fn, self_) is None

    def test_rlock_release_not_classified(self) -> None:
        lock = threading.RLock()
        fn, self_ = _bound(lock, "release")
        assert classify_call(fn, self_) is None


class TestSemaphoreClassification:
    """Same rationale as TestLockClassification — LockPatcher handles them."""

    def test_semaphore_acquire_not_classified(self) -> None:
        sem = threading.Semaphore(3)
        fn, self_ = _bound(sem, "acquire")
        assert classify_call(fn, self_) is None

    def test_semaphore_release_not_classified(self) -> None:
        sem = threading.Semaphore(3)
        fn, self_ = _bound(sem, "release")
        assert classify_call(fn, self_) is None

    def test_bounded_semaphore_acquire_not_classified(self) -> None:
        sem = threading.BoundedSemaphore(3)
        fn, self_ = _bound(sem, "acquire")
        assert classify_call(fn, self_) is None


class TestConditionClassification:
    def test_condition_wait(self) -> None:
        cond = threading.Condition()
        fn, self_ = _bound(cond, "wait")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.COND_WAIT

    def test_condition_wait_for(self) -> None:
        cond = threading.Condition()
        fn, self_ = _bound(cond, "wait_for")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.COND_WAIT

    def test_condition_notify(self) -> None:
        cond = threading.Condition()
        fn, self_ = _bound(cond, "notify")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.COND_NOTIFY

    def test_condition_notify_all(self) -> None:
        cond = threading.Condition()
        fn, self_ = _bound(cond, "notify_all")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.COND_NOTIFY

    def test_condition_acquire_not_classified(self) -> None:
        """Condition delegates acquire/release to its internal RLock,
        which LockPatcher already wraps."""
        cond = threading.Condition()
        fn, self_ = _bound(cond, "acquire")
        assert classify_call(fn, self_) is None

    def test_condition_release_not_classified(self) -> None:
        cond = threading.Condition()
        fn, self_ = _bound(cond, "release")
        assert classify_call(fn, self_) is None


class TestEventClassification:
    def test_event_set(self) -> None:
        ev = threading.Event()
        fn, self_ = _bound(ev, "set")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.EVENT_SET
        assert sc.target is ev

    def test_event_wait(self) -> None:
        ev = threading.Event()
        fn, self_ = _bound(ev, "wait")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.EVENT_WAIT

    def test_event_clear_not_classified(self) -> None:
        """Event.clear() has no HB significance — we ignore it."""
        ev = threading.Event()
        fn, self_ = _bound(ev, "clear")
        sc = classify_call(fn, self_)
        assert sc is None

    def test_event_is_set_not_classified(self) -> None:
        ev = threading.Event()
        fn, self_ = _bound(ev, "is_set")
        sc = classify_call(fn, self_)
        assert sc is None


class TestBarrierClassification:
    def test_barrier_wait(self) -> None:
        bar = threading.Barrier(2)
        fn, self_ = _bound(bar, "wait")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.BARRIER_WAIT
        assert sc.target is bar


class TestThreadClassification:
    def test_thread_start(self) -> None:
        t = threading.Thread(target=lambda: None)
        fn, self_ = _bound(t, "start")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.THREAD_START
        assert sc.target is t

    def test_thread_join(self) -> None:
        t = threading.Thread(target=lambda: None)
        fn, self_ = _bound(t, "join")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.THREAD_JOIN
        assert sc.target is t

    def test_thread_run_not_classified(self) -> None:
        """Thread.run() is not a sync event."""
        t = threading.Thread(target=lambda: None)
        fn, self_ = _bound(t, "run")
        sc = classify_call(fn, self_)
        assert sc is None

    def test_thread_is_alive_not_classified(self) -> None:
        t = threading.Thread(target=lambda: None)
        fn, self_ = _bound(t, "is_alive")
        sc = classify_call(fn, self_)
        assert sc is None


class TestNonSyncCallables:
    def test_plain_function_not_classified(self) -> None:
        def foo(x: int) -> int:
            return x

        sc = classify_call(foo, 42)
        assert sc is None

    def test_unrelated_object_not_classified(self) -> None:
        class MyClass:
            def acquire(self) -> None:
                pass

        obj = MyClass()
        fn, self_ = _bound(obj, "acquire")
        # Same method name but wrong type — should not be classified
        sc = classify_call(fn, self_)
        assert sc is None

    def test_none_callable_not_classified(self) -> None:
        sc = classify_call(None, None)
        assert sc is None

    def test_integer_not_classified(self) -> None:
        sc = classify_call(42, None)
        assert sc is None


class TestSyncCallNamedTuple:
    def test_sync_call_is_named_tuple(self) -> None:
        ev = threading.Event()
        fn, self_ = _bound(ev, "set")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.EVENT_SET
        assert sc.target is ev

    def test_sync_call_is_immutable(self) -> None:
        ev = threading.Event()
        fn, self_ = _bound(ev, "set")
        sc = classify_call(fn, self_)
        with pytest.raises((AttributeError, TypeError)):
            sc.kind = SyncKind.EVENT_WAIT  # type: ignore
