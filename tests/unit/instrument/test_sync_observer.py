import threading

import pytest

from pyft.instrument.sync_observer import SyncKind, classify_call

# ── Helpers ──────────────────────────────────────────────────────────


def _bound(obj: object, method_name: str):
    """Return the bound method and the object (simulating CALL arg0)."""
    method = getattr(obj, method_name)
    return method, obj


class TestLockClassification:
    def test_lock_acquire(self):
        lock = threading.Lock()
        fn, self_ = _bound(lock, "acquire")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.LOCK_ACQUIRE
        assert sc.target is lock

    def test_lock_release(self):
        lock = threading.Lock()
        fn, self_ = _bound(lock, "release")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.LOCK_RELEASE
        assert sc.target is lock

    def test_lock_enter(self):
        lock = threading.Lock()
        fn, self_ = _bound(lock, "__enter__")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.LOCK_ACQUIRE

    def test_lock_exit(self):
        lock = threading.Lock()
        fn, self_ = _bound(lock, "__exit__")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.LOCK_RELEASE

    def test_rlock_acquire(self):
        lock = threading.RLock()
        fn, self_ = _bound(lock, "acquire")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.LOCK_ACQUIRE

    def test_rlock_release(self):
        lock = threading.RLock()
        fn, self_ = _bound(lock, "release")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.LOCK_RELEASE


class TestSemaphoreClassification:
    def test_semaphore_acquire(self):
        sem = threading.Semaphore(3)
        fn, self_ = _bound(sem, "acquire")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.LOCK_ACQUIRE

    def test_semaphore_release(self):
        sem = threading.Semaphore(3)
        fn, self_ = _bound(sem, "release")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.LOCK_RELEASE

    def test_bounded_semaphore_acquire(self):
        sem = threading.BoundedSemaphore(3)
        fn, self_ = _bound(sem, "acquire")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.LOCK_ACQUIRE


class TestConditionClassification:
    def test_condition_wait(self):
        cond = threading.Condition()
        fn, self_ = _bound(cond, "wait")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.COND_WAIT

    def test_condition_wait_for(self):
        cond = threading.Condition()
        fn, self_ = _bound(cond, "wait_for")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.COND_WAIT

    def test_condition_notify(self):
        cond = threading.Condition()
        fn, self_ = _bound(cond, "notify")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.COND_NOTIFY

    def test_condition_notify_all(self):
        cond = threading.Condition()
        fn, self_ = _bound(cond, "notify_all")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.COND_NOTIFY

    def test_condition_acquire(self):
        cond = threading.Condition()
        fn, self_ = _bound(cond, "acquire")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.LOCK_ACQUIRE

    def test_condition_release(self):
        cond = threading.Condition()
        fn, self_ = _bound(cond, "release")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.LOCK_RELEASE


class TestEventClassification:
    def test_event_set(self):
        ev = threading.Event()
        fn, self_ = _bound(ev, "set")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.EVENT_SET
        assert sc.target is ev

    def test_event_wait(self):
        ev = threading.Event()
        fn, self_ = _bound(ev, "wait")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.EVENT_WAIT

    def test_event_clear_not_classified(self):
        """Event.clear() has no HB significance — we ignore it."""
        ev = threading.Event()
        fn, self_ = _bound(ev, "clear")
        sc = classify_call(fn, self_)
        assert sc is None

    def test_event_is_set_not_classified(self):
        ev = threading.Event()
        fn, self_ = _bound(ev, "is_set")
        sc = classify_call(fn, self_)
        assert sc is None


class TestBarrierClassification:
    def test_barrier_wait(self):
        bar = threading.Barrier(2)
        fn, self_ = _bound(bar, "wait")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.BARRIER_WAIT
        assert sc.target is bar


class TestThreadClassification:
    def test_thread_start(self):
        t = threading.Thread(target=lambda: None)
        fn, self_ = _bound(t, "start")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.THREAD_START
        assert sc.target is t

    def test_thread_join(self):
        t = threading.Thread(target=lambda: None)
        fn, self_ = _bound(t, "join")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.THREAD_JOIN
        assert sc.target is t

    def test_thread_run_not_classified(self):
        """Thread.run() is not a sync event."""
        t = threading.Thread(target=lambda: None)
        fn, self_ = _bound(t, "run")
        sc = classify_call(fn, self_)
        assert sc is None

    def test_thread_is_alive_not_classified(self):
        t = threading.Thread(target=lambda: None)
        fn, self_ = _bound(t, "is_alive")
        sc = classify_call(fn, self_)
        assert sc is None


class TestNonSyncCallables:
    def test_plain_function_not_classified(self):
        def foo(x):
            return x

        sc = classify_call(foo, 42)
        assert sc is None

    def test_unrelated_object_not_classified(self):
        class MyClass:
            def acquire(self):
                pass

        obj = MyClass()
        fn, self_ = _bound(obj, "acquire")
        # Same method name but wrong type — should not be classified
        sc = classify_call(fn, self_)
        assert sc is None

    def test_none_callable_not_classified(self):
        sc = classify_call(None, None)
        assert sc is None

    def test_integer_not_classified(self):
        sc = classify_call(42, None)
        assert sc is None


class TestSyncCallNamedTuple:
    def test_sync_call_is_named_tuple(self):
        lock = threading.Lock()
        fn, self_ = _bound(lock, "acquire")
        sc = classify_call(fn, self_)
        assert sc is not None
        assert sc.kind == SyncKind.LOCK_ACQUIRE
        assert sc.target is lock

    def test_sync_call_is_immutable(self):
        lock = threading.Lock()
        fn, self_ = _bound(lock, "acquire")
        sc = classify_call(fn, self_)
        with pytest.raises((AttributeError, TypeError)):
            sc.kind = SyncKind.LOCK_RELEASE  # type: ignore
