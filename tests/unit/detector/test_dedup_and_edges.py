"""
Detector tests for race deduplication, edge-case events, and concurrent
ShadowMap / Engine usage.
"""

from __future__ import annotations

import threading

import pytest

from pyft.core.var_state import VarStateV2
from pyft.detector.engine import Engine
from pyft.detector.race_log import AccessInfo, RaceKind, RaceLog, RaceReport
from pyft.detector.shadow_map import ShadowMap

VarState = VarStateV2


class _Obj:
    pass


def _mk_report(
    kind: RaceKind,
    tid_a: int,
    tid_b: int,
    obj_id: int = 1,
    attr: str = "x",
    seq: int = 0,
) -> RaceReport:
    a = AccessInfo(
        tid=tid_a,
        thread_name=f"T{tid_a}",
        clock=1,
        is_write=(kind != RaceKind.WRITE_READ),
        obj_repr="<obj>",
        attr=attr,
    )
    b = AccessInfo(
        tid=tid_b,
        thread_name=f"T{tid_b}",
        clock=1,
        is_write=(kind == RaceKind.WRITE_WRITE),
        obj_repr="<obj>",
        attr=attr,
    )
    return RaceReport(
        kind=kind, access_a=a, access_b=b, obj_id=obj_id, sequence=seq
    )


class TestDedup:
    def test_same_kind_same_tids_dedupes(self) -> None:
        log = RaceLog()
        assert log.record(_mk_report(RaceKind.WRITE_WRITE, 1, 2)) is True
        assert log.record(_mk_report(RaceKind.WRITE_WRITE, 1, 2)) is False
        assert len(log) == 1

    def test_different_kinds_same_tids_both_logged(self) -> None:
        """A WW race and a later RW race on same (obj, attr, tids) must
        both be reported — kind is part of the dedup key."""
        log = RaceLog()
        assert log.record(_mk_report(RaceKind.WRITE_WRITE, 1, 2)) is True
        assert log.record(_mk_report(RaceKind.READ_WRITE, 1, 2)) is True
        assert log.record(_mk_report(RaceKind.WRITE_READ, 1, 2)) is True
        assert len(log) == 3

    def test_dedup_symmetric_in_tids(self) -> None:
        """Tid order in the report should not affect dedup."""
        log = RaceLog()
        log.record(_mk_report(RaceKind.WRITE_WRITE, 1, 2))
        assert log.record(_mk_report(RaceKind.WRITE_WRITE, 2, 1)) is False

    def test_sequences_are_dense_under_dups(self) -> None:
        """A duplicate must not consume a sequence number."""
        log = RaceLog()
        log.record(_mk_report(RaceKind.WRITE_WRITE, 1, 2))
        log.record(_mk_report(RaceKind.WRITE_WRITE, 1, 2))  # dup
        log.record(_mk_report(RaceKind.WRITE_WRITE, 1, 3))
        seqs = [r.sequence for r in log.all_reports()]
        assert seqs == [1, 2]


class TestShadowMapEdges:
    def test_non_weakrefable_int_does_not_crash(self) -> None:
        sm = ShadowMap()
        vs = sm.get_or_create(42, "x")
        assert isinstance(vs, VarState)
        # Same lookup returns the same VarState.
        assert sm.get_or_create(42, "x") is vs

    def test_non_weakrefable_str_does_not_crash(self) -> None:
        sm = ShadowMap()
        assert sm.get_or_create("hello", "x") is not None

    def test_concurrent_same_obj_same_attr_stress(self) -> None:
        sm = ShadowMap()
        obj = _Obj()
        N = 32
        results: list = []
        barrier = threading.Barrier(N)
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            vs = sm.get_or_create(obj, "x")
            with lock:
                results.append(vs)

        threads = [threading.Thread(target=worker) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == N
        first = results[0]
        assert all(v is first for v in results)


class TestEngineEdges:
    def test_thread_join_of_unknown_joinee_does_not_crash(self) -> None:
        engine = Engine()
        # joinee_tid 99999 was never registered
        engine.thread_join(
            joiner_tid=id(threading.current_thread()), joinee_tid=99999
        )

    def test_thread_finish_of_unknown_does_not_crash(self) -> None:
        engine = Engine()
        engine.thread_finish(tid=99999)

    def test_lock_release_before_acquire_does_not_crash(self) -> None:
        engine = Engine()
        # release on a fresh lock_id with no prior acquire
        engine.lock_release(lock_id=7777)

    def test_lock_acquire_with_no_prior_release_is_noop(self) -> None:
        engine = Engine()
        # acquire on a fresh lock_id with no prior release: release_vc is
        # None, so no HB edge is established — must not crash.
        engine.lock_acquire(lock_id=9999)


@pytest.mark.timeout(10)
class TestEngineConcurrentReadWrite:
    def test_concurrent_writers_detect_race(self) -> None:
        engine = Engine()
        obj = _Obj()
        N = 8
        barrier = threading.Barrier(N)
        errors: list = []

        def writer() -> None:
            try:
                barrier.wait()
                for _ in range(100):
                    engine.write(obj, "x")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        reports = engine.race_log.all_reports()
        assert any(r.kind == RaceKind.WRITE_WRITE for r in reports), (
            f"expected at least one WW race, got: {[r.kind for r in reports]}"
        )

    def test_dedup_collapses_repeated_races(self) -> None:
        """Many iterations of the same race between the same threads should
        produce a small bounded number of reports."""
        engine = Engine()
        obj = _Obj()
        N = 4
        barrier = threading.Barrier(N)

        def writer() -> None:
            barrier.wait()
            for _ in range(500):
                engine.write(obj, "x")

        threads = [threading.Thread(target=writer) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        reports = engine.race_log.all_reports()
        # dedup key is (obj_id, attr, frozenset(tids), kind):
        # at most C(N, 2) distinct unordered pairs * 1 kind (WW) = 6 max
        assert 1 <= len(reports) <= N * (N - 1) // 2

    def test_concurrent_read_then_writes_produce_rw_race(self) -> None:
        engine = Engine()
        obj = _Obj()

        readers_done = threading.Event()

        def reader() -> None:
            for _ in range(50):
                engine.read(obj, "y")
            readers_done.set()

        def writer() -> None:
            readers_done.wait(timeout=5)
            for _ in range(50):
                engine.write(obj, "y")

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        reports = engine.race_log.all_reports()
        # writer racing with prior unsynchronised readers from another thread
        kinds = {r.kind for r in reports}
        assert kinds & {RaceKind.READ_WRITE, RaceKind.WRITE_READ}
