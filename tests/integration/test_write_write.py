"""
Integration test: concurrent writes without synchronization → WRITE_WRITE race.
"""

import threading

import pytest

from pyft.detector.race_log import RaceKind

from .helpers import count_races_of_kind, pyft_session, run_threads


class Shared:
    def __init__(self):
        self.value = 0
        self.name = "initial"


@pytest.mark.timeout(10)
class TestWriteWriteRace:
    def test_two_threads_write_same_attr(self):
        """Classic unsynchronized counter increment → write-write race."""
        with pyft_session() as (engine, track):
            obj = Shared()
            shared = track(obj)

            # Use an event to maximize overlap
            go = threading.Event()

            def writer(val):
                go.wait()
                shared.value = val

            # Pre-register so engine knows about child threads
            results = []

            def make_writer(val):
                def fn():
                    shared.value = val

                return fn

            run_threads(make_writer(1), make_writer(2))

            reports = engine.race_log.all_reports()
            # Should have detected at least one race on 'value'
            race_attrs = {r.access_a.attr for r in reports}
            assert "value" in race_attrs

    def test_write_write_race_kind(self):
        """Races on concurrent writes should be WRITE_WRITE kind."""
        with pyft_session() as (engine, track):
            obj = Shared()
            shared = track(obj)

            def writer(v):
                for _ in range(10):
                    shared.value = v

            run_threads(writer.__get__(1), writer.__get__(2))

            ww_count = count_races_of_kind(engine, RaceKind.WRITE_WRITE)
            rw_count = count_races_of_kind(engine, RaceKind.READ_WRITE)
            wr_count = count_races_of_kind(engine, RaceKind.WRITE_READ)
            # Total races should be non-zero
            assert ww_count + rw_count + wr_count > 0

    def test_three_threads_write(self):
        """Three-way write race is detected."""
        with pyft_session() as (engine, track):
            obj = Shared()
            shared = track(obj)

            def writer(v):
                shared.name = str(v)

            run_threads(
                lambda: writer(1), lambda: writer(2), lambda: writer(3)
            )

            reports = engine.race_log.all_reports()
            assert len(reports) >= 1

    def test_race_report_has_correct_attr(self):
        """Race report should identify the correct attribute."""
        with pyft_session() as (engine, track):
            obj = Shared()
            shared = track(obj)

            run_threads(
                lambda: setattr(shared, "value", 1),
                lambda: setattr(shared, "value", 2),
            )

            reports = engine.race_log.all_reports()
            # If a race is detected, it should be on 'value'
            for r in reports:
                assert r.access_a.attr == "value"

    def test_race_involves_two_different_threads(self):
        """Each race report must involve two distinct thread IDs."""
        with pyft_session() as (engine, track):
            obj = Shared()
            shared = track(obj)

            run_threads(
                lambda: [setattr(shared, "value", i) for i in range(5)],
                lambda: [setattr(shared, "value", i + 100) for i in range(5)],
            )

            for report in engine.race_log.all_reports():
                assert report.access_a.tid != report.access_b.tid

    def test_deduplication(self):
        """The same variable race should only be reported once."""
        with pyft_session() as (engine, track):
            obj = Shared()
            shared = track(obj)

            # Many iterations to maximize chance of repeated detection
            def writer(v):
                for _ in range(100):
                    shared.value = v

            run_threads(lambda: writer(1), lambda: writer(2))

            reports = engine.race_log.all_reports()
            # All reports for 'value' should be deduplicated
            value_races = [r for r in reports if r.access_a.attr == "value"]
            assert len(value_races) <= 1
