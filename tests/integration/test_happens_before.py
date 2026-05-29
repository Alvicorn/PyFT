import threading

import pytest

from .helpers import assert_no_races, pyft_session


class Payload:
    def __init__(self) -> None:
        self.result = None
        self.input = None


@pytest.mark.timeout(10)
class TestHappensBeforeForkJoin:
    def test_parent_write_before_fork_no_race(self) -> None:
        """
        Parent writes to obj BEFORE starting child thread.
        Child reads. Fork establishes HB → no race.
        """
        with pyft_session() as (engine, track):
            obj = Payload()
            shared = track(obj)

            # Parent writes before fork
            shared.input = 42

            results = []

            def child() -> None:
                # This read is HB-after the write above (fork edge)
                results.append(shared.input)

            t = threading.Thread(target=child)
            t.start()
            t.join()

            assert_no_races(engine)

    def test_child_write_before_join_no_race(self) -> None:
        """
        Child writes to obj BEFORE joining.
        Parent reads after join. Join establishes HB → no race.
        """
        with pyft_session() as (engine, track):
            obj = Payload()
            shared = track(obj)

            def child() -> None:
                shared.result = 99

            t = threading.Thread(target=child)
            t.start()
            t.join()  # join establishes HB: child's write → parent's read

            # Safe read post-join
            _ = shared.result
            assert_no_races(engine)

    def test_fork_join_chain_no_race(self) -> None:
        """
        Sequential thread chain: T1 → T2 → T3, each writes then joins.
        All transitions are HB-ordered → no races.
        """
        with pyft_session() as (engine, track):
            obj = Payload()
            shared = track(obj)

            def stage(val: int) -> None:
                shared.result = val

            for val in [1, 2, 3]:
                t = threading.Thread(target=stage, args=(val,))
                t.start()
                t.join()

            assert_no_races(engine)

    def test_parent_read_before_child_write_without_join_races(self) -> None:
        """
        Without join, parent reading after child started (but not joined)
        may race with child's write.
        """
        with pyft_session() as (engine, track):
            obj = Payload()
            shared = track(obj)
            # ready = threading.Barrier(2)
            # started = threading.Event()

            def child() -> None:
                # ready.wait()
                # started.set()
                # time.sleep(0.001)
                shared.result = 42  # write

            t = threading.Thread(target=child)
            t.start()

            # ready.wait()
            # started.wait()

            for _ in range(1000):
                _ = shared.result  # concurrent read (no join yet)

            t.join()

            # A race should be detected (read and write are concurrent)
            reports = engine.race_log.all_reports()
            assert len(reports) >= 1

    def test_multiple_children_all_joined_no_race(self) -> None:
        """
        Parent forks N children, each writes to a different attribute,
        then parent joins all and reads all → no races.
        """
        with pyft_session() as (engine, track):

            class MultiSlot:
                pass

            obj = MultiSlot()
            shared = track(obj)
            N = 5
            threads = []

            def child_write(i: int) -> None:
                setattr(shared, f"slot_{i}", i * 10)

            for i in range(N):
                t = threading.Thread(target=child_write, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # All reads after all joins → safe
            for i in range(N):
                _ = getattr(shared, f"slot_{i}")

            assert_no_races(engine)

    def test_fork_hb_means_child_sees_parent_history(self) -> None:
        """
        After fork, child's VC must include parent's clock, meaning
        child treats parent's prior writes as happened-before.
        """
        with pyft_session() as (engine, track):
            obj = Payload()
            shared = track(obj)

            # Three sequential writes by parent before fork
            shared.result = 1
            shared.result = 2
            shared.result = 3

            read_val = []

            def child() -> None:
                # Child reads — must see result=3 without race
                read_val.append(shared.result)

            t = threading.Thread(target=child)
            t.start()
            t.join()

            assert_no_races(engine)
