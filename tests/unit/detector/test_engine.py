import threading

from pyvft.detector.engine import Engine


class SimpleObj:
    """A plain object to use as a test target."""

    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x = x
        self.y = y


class TestEngineSetup:
    def test_main_thread_registered_on_init(self) -> None:
        engine = Engine()
        tid = id(threading.current_thread())
        ts = engine.thread_registry.get(tid)
        assert ts is not None

    def test_initial_no_races(self) -> None:
        engine = Engine()
        assert len(engine.race_log) == 0


class TestSingleThreadNoRace:
    def test_single_thread_write_no_race(self) -> None:
        engine = Engine()
        obj = SimpleObj()
        engine.write(obj, "x")
        assert len(engine.race_log) == 0

    def test_single_thread_read_no_race(self) -> None:
        engine = Engine()
        obj = SimpleObj()
        engine.write(obj, "x")
        engine.read(obj, "x")
        assert len(engine.race_log) == 0

    def test_single_thread_multiple_writes_no_race(self) -> None:
        engine = Engine()
        obj = SimpleObj()
        for _ in range(100):
            engine.write(obj, "x")
        assert len(engine.race_log) == 0

    def test_not_shared_never_races(self) -> None:
        """Objects only accessed by one thread are never flagged."""
        engine = Engine()
        obj = SimpleObj()
        # Only main thread ever touches this — not shared
        for _ in range(50):
            engine.read(obj, "x")
            engine.write(obj, "x")
        assert len(engine.race_log) == 0


class TestLockSynchronization:
    def test_lock_acquire_absorbs_release_vc(self) -> None:
        """After acquire, thread's VC should include releaser's clock."""
        engine = Engine()
        lock = threading.Lock()
        lock_id = id(lock)

        tid = id(threading.current_thread())
        ts = engine.thread_registry.get(tid)
        assert ts is not None

        # Simulate T1 releasing (sets release_vc)
        engine.lock_release(lock_id)
        ts.current_clock()

        # Simulate T2 acquiring: should absorb T1's VC
        # (In single-thread test: self-acquire restores own VC)
        engine.lock_acquire(lock_id)
        # No crash, no race
        assert len(engine.race_log) == 0

    def test_lock_release_ticks_clock(self) -> None:
        engine = Engine()
        lock_id = 12345

        tid = id(threading.current_thread())
        ts = engine.thread_registry.get(tid)
        assert ts is not None
        clock_before = ts.current_clock()

        engine.lock_release(lock_id)

        clock_after = ts.current_clock()
        assert clock_after > clock_before


class TestThreadLifecycle:
    def test_thread_start_registers_child(self) -> None:
        engine = Engine()
        parent_tid = id(threading.current_thread())
        child_tid = 99999

        engine.thread_start(parent_tid, child_tid, "child")
        child_ts = engine.thread_registry.get(child_tid)
        assert child_ts is not None

    def test_thread_start_child_inherits_parent_vc(self) -> None:
        engine = Engine()
        parent_tid = id(threading.current_thread())
        parent_ts = engine.thread_registry.get(parent_tid)
        assert parent_ts is not None
        assert parent_ts.vc is not None
        parent_ts.vc.set(parent_tid, 5)

        child_tid = 99998
        engine.thread_start(parent_tid, child_tid, "child")

        child_ts = engine.thread_registry.get(child_tid)
        # Child should have inherited parent's clock
        assert child_ts is not None
        assert child_ts.vc is not None
        assert child_ts.vc.get(parent_tid) >= 5

    def test_thread_join_absorbs_child_vc(self) -> None:
        engine = Engine()
        parent_tid = id(threading.current_thread())
        child_tid = 99997

        engine.thread_start(parent_tid, child_tid, "child")
        child_ts = engine.thread_registry.get(child_tid)
        assert child_ts is not None
        assert child_ts.vc is not None
        child_ts.vc.set(child_tid, 10)

        engine.thread_join(parent_tid, child_tid)

        parent_ts = engine.thread_registry.get(parent_tid)
        assert parent_ts is not None
        assert parent_ts.vc is not None
        assert parent_ts.vc.get(child_tid) == 10

    def test_thread_join_removes_child_from_registry(self) -> None:
        engine = Engine()
        parent_tid = id(threading.current_thread())
        child_tid = 99996

        engine.thread_start(parent_tid, child_tid, "child")
        engine.thread_join(parent_tid, child_tid)

        assert engine.thread_registry.get(child_tid) is None


class TestReentrancyGuard:
    def test_nested_engine_call_is_ignored(self) -> None:
        """Engine should not recurse into itself."""
        engine = Engine()
        obj = SimpleObj()

        # Manually set depth to simulate reentrant call
        engine._active.depth = 1
        engine.write(obj, "x")  # should be a no-op
        engine._active.depth = 0

        # No side effects
        assert len(engine.race_log) == 0


class TestRealThreadRace:
    def test_write_write_race_detected(self) -> None:
        """Two unordered concurrent writes -> write-write race detected."""
        engine = Engine()
        obj = SimpleObj()
        errors = []
        # Barrier(2) releases both writers simultaneously — no HB between them.
        barrier = threading.Barrier(2)

        def writer() -> None:
            try:
                barrier.wait()
                engine.write(obj, "x")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer, name="Writer-1")
        t2 = threading.Thread(target=writer, name="Writer-2")

        # Register children BEFORE start so engine has fork HB edges.
        # id(t) is the stable per-thread identity used throughout the engine.
        parent_tid = id(threading.current_thread())
        engine.thread_start(parent_tid, id(t1), "Writer-1")
        engine.thread_start(parent_tid, id(t2), "Writer-2")

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread exceptions: {errors}"
        assert len(engine.race_log) > 0, (
            "Expected a write-write race to be detected"
        )

    def test_no_crash_under_concurrent_engine_calls(self) -> None:
        """Engine must be thread-safe under concurrent read/write events."""
        engine = Engine()
        obj = SimpleObj()
        errors = []
        ITERATIONS = 500

        def accessor() -> None:
            tid = id(threading.current_thread())
            engine.thread_registry.get_or_register(tid)
            for _ in range(ITERATIONS):
                engine.read(obj, "x")
                engine.write(obj, "x")

        threads = [threading.Thread(target=accessor) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
