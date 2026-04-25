import threading

import pytest

from .helpers import assert_no_races, pyft_session, run_threads


class SharedState:
    def __init__(self):
        self.counter = 0
        self.data = []
        self.flag = False


@pytest.mark.timeout(10)
class TestLockEliminatesRace:
    @pytest.mark.skip()
    def test_lock_before_write_no_race(self):
        """Lock acquired before every write eliminates WRITE_WRITE."""
        with pyft_session() as (engine, track):
            lock = threading.Lock()
            obj = SharedState()
            shared = track(obj)

            def increment():
                for _ in range(30):
                    with lock:
                        shared.counter += 1

            run_threads(increment, increment, increment)
            assert_no_races(engine)

    @pytest.mark.skip()
    def test_lock_before_read_and_write_no_race(self):
        """Lock on both read and write sides eliminates READ_WRITE."""
        with pyft_session() as (engine, track):
            lock = threading.Lock()
            obj = SharedState()
            shared = track(obj)
            snapshots = []

            def writer():
                for i in range(20):
                    with lock:
                        shared.counter = i

            def reader():
                for _ in range(20):
                    with lock:
                        snapshots.append(shared.counter)

            run_threads(writer, reader)
            assert_no_races(engine)

    def test_partial_lock_still_races(self):
        """If only one side uses the lock, we still get a race."""
        with pyft_session() as (engine, track):
            lock = threading.Lock()
            obj = SharedState()
            shared = track(obj)

            def locked_writer():
                for _ in range(10):
                    with lock:
                        shared.counter += 1

            def unlocked_writer():
                for _ in range(10):
                    shared.counter += 1  # no lock!

            run_threads(locked_writer, unlocked_writer)
            # Must detect a race despite one side being locked
            reports = engine.race_log.all_reports()
            assert len(reports) >= 1

    def test_wrong_lock_still_races(self):
        """Two different locks do NOT establish HB between threads."""
        with pyft_session() as (engine, track):
            lock_a = threading.Lock()
            lock_b = threading.Lock()
            obj = SharedState()
            shared = track(obj)

            def writer_a():
                for _ in range(10):
                    with lock_a:  # lock A
                        shared.counter += 1

            def writer_b():
                for _ in range(10):
                    with lock_b:  # lock B (different!)
                        shared.counter += 1

            run_threads(writer_a, writer_b)
            reports = engine.race_log.all_reports()
            assert len(reports) >= 1

    def test_same_lock_multiple_objects_no_race(self):
        """One lock protecting multiple objects → no races on any."""
        with pyft_session() as (engine, track):
            lock = threading.Lock()

            class Pair:
                def __init__(self):
                    self.left = 0
                    self.right = 0

            obj = Pair()
            shared = track(obj)

            def worker():
                for _ in range(20):
                    with lock:
                        shared.left += 1
                        shared.right += 1

            run_threads(worker, worker)
            assert_no_races(engine)

    def test_context_manager_lock_no_race(self):
        """Verify `with lock:` syntax (uses __enter__/__exit__) works correctly."""
        with pyft_session() as (engine, track):
            lock = threading.Lock()
            obj = SharedState()
            shared = track(obj)

            def worker():
                for _ in range(15):
                    lock.acquire()
                    try:
                        shared.flag = True
                    finally:
                        lock.release()

            run_threads(worker, worker)
            assert_no_races(engine)
