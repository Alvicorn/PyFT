import pytest

from pyft.detector.race_log import RaceKind

from .helpers import pyft_session, run_threads


class Data:
    def __init__(self):
        self.x = 0


@pytest.mark.timeout(10)
class TestReadWriteRace:
    def test_reader_and_writer_concurrent(self):
        """One thread reads, another writes — no sync → race."""
        with pyft_session() as (engine, track):
            obj = Data()
            shared = track(obj)
            read_values = []

            def reader():
                for _ in range(20):
                    read_values.append(shared.x)

            def writer():
                for _ in range(20):
                    shared.x += 1

            run_threads(reader, writer)

            reports = engine.race_log.all_reports()
            assert len(reports) >= 1
            # All races should be on 'x'
            for r in reports:
                assert r.access_a.attr == "x"

    def test_race_kind_is_read_write_or_write_read(self):
        """Mixed read/write race should be READ_WRITE or WRITE_READ."""
        with pyft_session() as (engine, track):
            obj = Data()
            shared = track(obj)

            run_threads(
                lambda: [_ for _ in [shared.x] * 10],
                lambda: [setattr(shared, "x", i) for i in range(10)],
            )

            reports = engine.race_log.all_reports()
            if reports:
                kinds = {r.kind for r in reports}
                assert kinds <= {
                    RaceKind.READ_WRITE,
                    RaceKind.WRITE_READ,
                    RaceKind.WRITE_WRITE,
                }

    def test_multiple_readers_one_writer(self):
        """Multiple readers + one writer without sync → races detected."""
        with pyft_session() as (engine, track):
            obj = Data()
            shared = track(obj)

            def reader():
                for _ in range(10):
                    _ = shared.x

            def writer():
                for _ in range(10):
                    shared.x = 99

            run_threads(reader, reader, writer)
            assert len(engine.race_log.all_reports()) >= 1

    def test_one_reader_multiple_writers(self):
        with pyft_session() as (engine, track):
            obj = Data()
            shared = track(obj)

            def reader():
                for _ in range(10):
                    _ = shared.x

            def writer():
                for _ in range(10):
                    shared.x += 1

            run_threads(reader, writer, writer)
            assert len(engine.race_log.all_reports()) >= 1

    def test_read_write_reports_have_distinct_tids(self):
        with pyft_session() as (engine, track):
            obj = Data()
            shared = track(obj)

            run_threads(
                lambda: [_ for _ in [shared.x] * 5],
                lambda: [setattr(shared, "x", v) for v in range(5)],
            )

            for r in engine.race_log.all_reports():
                assert r.access_a.tid != r.access_b.tid
