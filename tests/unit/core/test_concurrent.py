"""
Concurrency stress tests for the core data structures.

Under free-threaded Python (no GIL) these objects are accessed from
multiple OS threads simultaneously, so they must remain internally
consistent and never raise things like
``RuntimeError: dictionary changed size during iteration``.

``TestVarStateConcurrent`` is parametrised over both VerifiedFT
analysers (``VarStateV1`` and ``VarStateV2``) — both must satisfy the
same observable invariants under load even though their shadow state
shapes differ.
"""

from __future__ import annotations

import threading

import pytest

from pyvft.core.thread_state import ThreadRegistry
from pyvft.core.var_state import (
    ReadBottom,
    ReadEpoch,
    ReadVC,
    VarStateV1,
    VarStateV2,
)
from pyvft.core.vector_clock import VectorClock

VAR_STATE_CLASSES = (VarStateV1, VarStateV2)
_IDS = ("V1", "V2")


def _writer_recorded(vs: object, tid: int) -> bool:
    """
    Did ``vs`` end up recording ``tid`` as a writer?

    For V2 this is ``write_epoch.tid == tid``; for V1 it's
    ``W_x.get(tid) > 0``.
    """
    if isinstance(vs, VarStateV2):
        return vs.write_epoch.tid == tid
    return vs.W_x.get(tid) > 0  # type: ignore[attr-defined]


def _any_writer_recorded(vs: object) -> bool:
    if isinstance(vs, VarStateV2):
        return not vs.write_epoch.is_bottom()
    return any(c > 0 for _, c in vs.W_x.items())  # type: ignore[attr-defined]


def _multiple_readers_recorded(vs: object) -> bool:
    """
    Both analyzers must end up with at least two distinct readers
    recorded after the all-readers stress test.
    """
    if isinstance(vs, VarStateV2):
        if isinstance(vs.read_state, ReadVC):
            return sum(1 for _, c in vs.read_state.vc.items() if c > 0) >= 2
        return False
    # V1: R_x is a VectorClock; count entries with c > 0.
    return (
        sum(1 for _, c in vs.R_x.items() if c > 0) >= 2  # type: ignore[attr-defined]
    )


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
@pytest.mark.parametrize("var_state_cls", VAR_STATE_CLASSES, ids=_IDS)
class TestVarStateConcurrent:
    def test_concurrent_writers_race_is_eventually_detected(
        self, var_state_cls: type
    ) -> None:
        """
        N threads write the same variable with no HB — at least one
        ``check_write`` must report a write_race, and the analyzer must
        have recorded at least one writer at the end.
        """
        N = 16
        reg = ThreadRegistry()
        for tid in range(1, N + 1):
            reg.register(tid)

        vs = var_state_cls()
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

        assert race_seen, "expected at least one write/write race"
        assert vs.is_shared
        assert _any_writer_recorded(vs)
        # At least one of the registered writer tids appears in the
        # analyzer's writer state.
        assert any(_writer_recorded(vs, tid) for tid in range(1, N + 1))

    def test_concurrent_readers_no_race(self, var_state_cls: type) -> None:
        """
        N threads only read — no race ever, and multiple readers
        recorded at the end.
        """
        N = 8
        reg = ThreadRegistry()
        for tid in range(1, N + 1):
            reg.register(tid)
        vs = var_state_cls()
        barrier = threading.Barrier(N)
        errors: list[Exception] = []

        def reader(tid: int) -> None:
            ts = reg.get(tid)
            assert ts is not None
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
        assert _multiple_readers_recorded(vs)

    def test_concurrent_read_then_write_promotes_and_races(
        self, var_state_cls: type
    ) -> None:
        """
        Several readers, then a writer with no HB → read/write race
        with a non-bottom ``prior_read_state``.
        """
        N = 4
        reg = ThreadRegistry()
        for tid in range(1, N + 2):
            reg.register(tid)
        vs = var_state_cls()
        barrier = threading.Barrier(N)

        def reader(tid: int) -> None:
            ts = reg.get(tid)
            barrier.wait()
            assert ts is not None
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

        writer_tid = N + 1
        writer_ts = reg.get(writer_tid)
        assert writer_ts is not None
        writer_ts.tick()
        read_race, write_race, _, prior_read_state = vs.check_write(
            writer_tid, writer_ts.snapshot()
        )
        assert read_race, "writer racing with concurrent readers"
        assert not write_race
        # Both analyzers surface a non-bottom prior reader on a real
        # read-race (V2: ReadEpoch / ReadVC; V1: synthesized ReadEpoch).
        assert not isinstance(prior_read_state, ReadBottom)
        assert isinstance(prior_read_state, (ReadEpoch, ReadVC))
