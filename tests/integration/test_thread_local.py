import threading

import pytest

from .helpers import assert_no_races, pyft_session, run_threads


class LocalCounter:
    def __init__(self) -> None:
        self.value = 0


@pytest.mark.timeout(10)
class TestThreadLocalNoRace:
    def test_each_thread_owns_its_object(self) -> None:
        """Each thread creates and uses its own object — no sharing."""
        with pyft_session() as (engine, track):

            def worker() -> None:
                obj = LocalCounter()
                local = track(obj)
                for _ in range(50):
                    local.value += 1
                assert local.value == 50

            run_threads(worker, worker, worker, worker)
            assert_no_races(engine)

    def test_thread_local_storage(self) -> None:
        """threading.local() values are per-thread — no sharing."""
        with pyft_session() as (engine, track):
            tls = threading.local()

            def worker() -> None:
                tls.obj = LocalCounter()
                tls.local = track(tls.obj)
                for _ in range(20):
                    tls.local.value += 1

            run_threads(worker, worker)
            assert_no_races(engine)

    def test_local_writes_do_not_pollute_shared_detection(self) -> None:
        """
        Thread-local writes before sharing should NOT count toward the
        HB history once the object becomes shared.
        """
        with pyft_session() as (engine, track):
            obj = LocalCounter()
            shared = track(obj)

            # Main thread writes first (object is still thread-local)
            shared.value = 10

            lock = threading.Lock()

            def worker() -> None:
                with lock:
                    # Now properly synchronized
                    shared.value += 1

            run_threads(worker, worker)
            assert_no_races(engine)

    def test_many_threads_none_sharing(self) -> None:
        """50 threads each with private objects → zero races."""
        with pyft_session() as (engine, track):

            def worker() -> None:
                obj = LocalCounter()
                local = track(obj)
                local.value = 0
                for i in range(10):
                    local.value = i

            threads = [threading.Thread(target=worker) for _ in range(50)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert_no_races(engine)

    def test_object_created_in_parent_accessed_only_by_child(self) -> None:
        """Object created by parent but only accessed by child → not shared."""
        with pyft_session() as (engine, track):
            obj = LocalCounter()
            shared = track(obj)

            def child() -> None:
                # Only child accesses this — parent never reads/writes after start
                for _ in range(20):
                    shared.value += 1

            t = threading.Thread(target=child)
            t.start()
            t.join()

            # After join, parent reads — this is HB-safe (post-join)
            _ = shared.value
            assert_no_races(engine)
