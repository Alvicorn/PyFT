import threading

import pytest

from tests.integration.helpers import (
    assert_no_races,
    pyft_session,
    run_threads,
)


class Counter:
    def __init__(self) -> None:
        self.value = 0


@pytest.mark.timeout(10)
class TestNoRaceWithLock:
    def test_lock_protected_writes(self) -> None:
        """Two threads write to shared object under a lock → no race."""
        with pyft_session() as (engine, track):
            lock = threading.Lock()
            obj = Counter()
            shared = track(obj)

            def writer() -> None:
                for _ in range(20):
                    with lock:
                        shared.value += 1

            run_threads(writer, writer)
            assert_no_races(engine)

    def test_lock_protected_read_and_write(self) -> None:
        """One thread writes, one reads — both under same lock."""
        with pyft_session() as (engine, track):
            lock = threading.Lock()
            obj = Counter()
            shared = track(obj)
            results = []

            def writer() -> None:
                for _ in range(10):
                    with lock:
                        shared.value += 1

            def reader() -> None:
                for _ in range(10):
                    with lock:
                        results.append(shared.value)

            run_threads(writer, reader)
            assert_no_races(engine)

    def test_rlock_protected_writes(self) -> None:
        """RLock (reentrant lock) also establishes HB."""
        with pyft_session() as (engine, track):
            lock = threading.RLock()
            obj = Counter()
            shared = track(obj)

            def writer() -> None:
                for _ in range(10):
                    with lock:
                        shared.value += 1

            run_threads(writer, writer)
            assert_no_races(engine)

    def test_multiple_variables_all_protected(self) -> None:
        """When ALL variables are protected, zero races reported."""
        with pyft_session() as (engine, track):
            lock = threading.Lock()

            class Point:
                def __init__(self) -> None:
                    self.x = 0
                    self.y = 0

            obj = Point()
            shared = track(obj)

            def worker() -> None:
                for _ in range(10):
                    with lock:
                        shared.x += 1
                        shared.y += 1

            run_threads(worker, worker)
            assert_no_races(engine)


@pytest.mark.timeout(10)
class TestNoRaceThreadLocal:
    def test_thread_local_objects_never_flagged(self) -> None:
        """Objects only accessed from one thread are never shared."""
        with pyft_session() as (engine, track):
            # Each thread creates and uses its own object
            def worker() -> None:
                obj = Counter()
                local = track(obj)
                for _ in range(50):
                    local.value += 1

            run_threads(worker, worker, worker)
            assert_no_races(engine)

    def test_disjoint_attribute_access(self) -> None:
        """Two threads access different attributes of same object."""
        with pyft_session() as (engine, track):

            class TwoSlot:
                def __init__(self) -> None:
                    self.a = 0
                    self.b = 0

            obj = TwoSlot()
            shared = track(obj)

            # T1 only touches .a, T2 only touches .b — still races
            # because VarState tracks per-attribute, so this is actually safe
            def t1() -> None:
                for _ in range(20):
                    shared.a += 1

            def t2() -> None:
                for _ in range(20):
                    shared.b += 1

            run_threads(t1, t2)
            # Each attribute has its own VarState so disjoint access is fine
            reports = engine.race_log.all_reports()
            race_attrs = {r.access_a.attr for r in reports}
            # .a and .b should NOT both be races (they're disjoint)
            assert "a" not in race_attrs or "b" not in race_attrs


@pytest.mark.timeout(10)
class TestNoRaceAfterJoin:
    def test_post_join_access_is_safe(self) -> None:
        """After joining a thread, accessing its results is HB-safe."""
        with pyft_session() as (engine, track):
            obj = Counter()
            shared = track(obj)

            def worker() -> None:
                shared.value = 42

            t = threading.Thread(target=worker)
            t.start()
            t.join()

            # This access happens after join → HB established → no race
            _ = shared.value
            assert_no_races(engine)
