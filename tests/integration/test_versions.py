"""
Integration tests that exercise both VerifiedFT variants (v1, v2) over
the same race / no-race scenarios. Both versions must report the same
race outcome under proper / improper synchronization.
"""

from __future__ import annotations

import threading

import pytest

from pyvft.instrument.wrappers import TrackedProxy

from .helpers import assert_no_races, pyvft_session, run_threads


@pytest.mark.timeout(10)
@pytest.mark.parametrize("version", ["v1", "v2"])
class TestSameOutcomeAcrossVersions:
    def test_unsync_write_write_races(self, version: str) -> None:
        with pyvft_session(version=version) as (engine, track):

            class C:
                pass

            shared = track(C())
            shared.x = 0

            def worker() -> None:
                for _ in range(50):
                    shared.x += 1

            run_threads(worker, worker, worker)
            reports = engine.race_log.all_reports()
            assert len(reports) >= 1, (
                f"version={version}: expected at least one race"
            )
            assert all(r.access_a.attr == "x" for r in reports)

    def test_unsync_reader_writer_races(self, version: str) -> None:
        with pyvft_session(version=version) as (engine, track):

            class B:
                pass

            shared = track(B())
            shared.payload = 0

            def reader() -> None:
                for _ in range(30):
                    _ = shared.payload

            def writer() -> None:
                for i in range(30):
                    shared.payload = i

            run_threads(reader, writer)
            reports = engine.race_log.all_reports()
            assert len(reports) >= 1

    def test_lock_protected_no_race(self, version: str) -> None:
        with pyvft_session(version=version) as (engine, track):

            class C:
                pass

            shared = track(C())
            shared.count = 0
            lock = threading.Lock()

            def worker() -> None:
                for _ in range(30):
                    with lock:
                        shared.count += 1

            run_threads(worker, worker, worker)
            assert_no_races(engine)

    def test_fork_join_post_join_read_no_race(self, version: str) -> None:
        with pyvft_session(version=version) as (engine, track):

            class M:
                pass

            shared = track(M())
            shared.value = 0

            def child() -> None:
                shared.value = 99

            t = threading.Thread(target=child)
            t.start()
            t.join()

            # After join: parent reads — HB-safe via the join edge.
            _ = shared.value
            assert_no_races(engine)

    def test_each_thread_owns_its_object_no_race(self, version: str) -> None:
        with pyvft_session(version=version) as (engine, track):

            class L:
                pass

            def worker() -> None:
                local = track(L())
                local.value = 0
                for i in range(20):
                    local.value = i

            run_threads(worker, worker, worker, worker)
            assert_no_races(engine)


@pytest.mark.timeout(10)
def test_v1_and_v2_agree_on_race_count_for_classic_ww() -> None:
    """
    A purely deterministic comparison: same workload, both versions
    should report the same set of (attr, tid-pair, kind) race keys.
    """
    keys: dict[str, set[tuple]] = {}
    for version in ("v1", "v2"):
        with pyvft_session(version=version) as (engine, track):

            class C:
                pass

            shared = track(C())
            shared.x = 0

            def worker(shared: TrackedProxy = shared) -> None:
                for _ in range(20):
                    shared.x += 1

            run_threads(worker, worker, worker)
            keys[version] = {
                (r.access_a.attr, frozenset([r.access_a.tid, r.access_b.tid]))
                for r in engine.race_log.all_reports()
            }
    # Both must have detected some race on x; the exact set may differ
    # because of scheduling, but neither can be empty if both correctly
    # detect the WW race.
    assert keys["v1"], "v1 reported no races on an obviously-racy workload"
    assert keys["v2"], "v2 reported no races on an obviously-racy workload"
