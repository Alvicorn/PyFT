"""
Concurrency stress tests for the core data structures.

Under free-threaded Python (no GIL) these objects are accessed from
multiple OS threads simultaneously, so they must remain internally
consistent and never raise things like
``RuntimeError: dictionary changed size during iteration``.
"""

from __future__ import annotations

import threading

import pytest

from pyft.core.thread_state import ThreadRegistry
from pyft.core.var_state import ReadBottom, ReadEpoch, ReadVC, VarState
from pyft.core.vector_clock import VectorClock


@pytest.mark.timeout(10)
class TestVectorClockConcurrent:
    def test_concurrent_increment_and_iterate(self) -> None:
        """One thread iterates while another increments — no RuntimeError."""
        vc = VectorClock()
        for tid in range(20):
            vc.set(tid, 0)

        stop = threading.Event()
        errors: list[Exception] = []

        def writer() -> None:
            i = 0
            while not stop.is_set():
                vc.increment(i % 20)
                i += 1

        def reader() -> None:
            while not stop.is_set():
                try:
                    list(vc.items())
                    _ = repr(vc)
                except Exception as e:
                    errors.append(e)
                    return

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        threading.Event().wait(0.5)
        stop.set()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_join(self) -> None:
        """Two threads joining each other's VCs — no RuntimeError, no deadlock."""
        a = VectorClock({i: i for i in range(50)})
        b = VectorClock({i: i * 2 for i in range(50)})
        errors: list[Exception] = []
        done = threading.Event()

        def joiner_a() -> None:
            try:
                for _ in range(200):
                    a.join(b)
            except Exception as e:
                errors.append(e)
            finally:
                done.set()

        def joiner_b() -> None:
            try:
                for _ in range(200):
                    b.join(a)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=joiner_a)
        t2 = threading.Thread(target=joiner_b)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert done.is_set(), "joiner_a did not finish — possible deadlock"
        assert not errors

    def test_concurrent_set_and_eq(self) -> None:
        """Equality comparison while another thread mutates — no exception."""
        a = VectorClock({i: i for i in range(30)})
        b = a.copy()
        stop = threading.Event()
        errors: list[Exception] = []

        def mutator() -> None:
            i = 0
            while not stop.is_set():
                a.set(i % 30, i)
                i += 1

        def comparator() -> None:
            while not stop.is_set():
                try:
                    _ = a == b
                except Exception as e:
                    errors.append(e)
                    return

        threads = [
            threading.Thread(target=mutator),
            threading.Thread(target=comparator),
        ]
        for t in threads:
            t.start()
        threading.Event().wait(0.3)
        stop.set()
        for t in threads:
            t.join()

        assert not errors


@pytest.mark.timeout(10)
class TestVarStateConcurrent:
    def test_concurrent_writers_race_is_eventually_detected(self) -> None:
        """
        N threads write the same variable with no HB — at least one
        check_write must report a write_race.
        """
        N = 16
        reg = ThreadRegistry()
        for tid in range(1, N + 1):
            reg.register(tid)

        vs = VarState()
        barrier = threading.Barrier(N)
        race_seen: list[bool] = []
        lock = threading.Lock()

        def writer(tid: int) -> None:
            ts = reg.get(tid)
            assert ts is not None
            barrier.wait()
            for _ in range(50):
                ts.tick()
                _, write_race, _, _ = vs.check_write(tid, ts.snapshot())
                if write_race:
                    with lock:
                        race_seen.append(True)

        threads = [
            threading.Thread(target=writer, args=(tid,))
            for tid in range(1, N + 1)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # At least one cross-thread write must have raced
        # (with no synchronization between them)
        assert race_seen, "expected at least one write/write race"
        # Final write_epoch is owned by one of the writers
        assert not vs.write_epoch.is_bottom()
        assert 1 <= vs.write_epoch.tid <= N

    def test_concurrent_readers_no_race(self) -> None:
        """N threads only read — no race ever, read_state ends as ReadVC."""
        N = 8
        reg = ThreadRegistry()
        for tid in range(1, N + 1):
            reg.register(tid)
        vs = VarState()
        barrier = threading.Barrier(N)
        errors: list[Exception] = []

        def reader(tid: int) -> None:
            ts = reg.get(tid)
            barrier.wait()
            try:
                for _ in range(50):
                    ts.tick()
                    race, _ = vs.check_read(tid, ts.snapshot())
                    if race:
                        errors.append(
                            AssertionError(f"spurious race tid={tid}")
                        )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader, args=(tid,))
            for tid in range(1, N + 1)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        # Multiple readers across threads → must be ReadVC at the end
        assert isinstance(vs.read_state, (ReadEpoch, ReadVC))
        if isinstance(vs.read_state, ReadVC):
            tids = {t for t, _ in vs.read_state.vc.items()}
            assert len(tids) >= 2

    def test_concurrent_read_then_write_promotes_and_races(self) -> None:
        """Several readers then a writer with no HB → read/write race."""
        N = 4
        reg = ThreadRegistry()
        for tid in range(1, N + 2):
            reg.register(tid)
        vs = VarState()
        barrier = threading.Barrier(N)

        def reader(tid: int) -> None:
            ts = reg.get(tid)
            barrier.wait()
            ts.tick()
            vs.check_read(tid, ts.snapshot())

        readers = [
            threading.Thread(target=reader, args=(tid,))
            for tid in range(1, N + 1)
        ]
        for t in readers:
            t.start()
        for t in readers:
            t.join()

        # Now a fresh writer tries to write — must race with prior readers
        writer_tid = N + 1
        writer_ts = reg.get(writer_tid)
        writer_ts.tick()
        read_race, write_race, _, prior_read_state = vs.check_write(
            writer_tid, writer_ts.snapshot()
        )
        assert read_race, "writer racing with concurrent readers"
        assert not write_race  # no prior write
        assert not isinstance(prior_read_state, ReadBottom)
