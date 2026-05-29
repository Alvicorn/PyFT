import threading

from pyft.core.thread_state import ThreadRegistry, ThreadState
from pyft.core.vector_clock import VectorClock


class TestThreadState:
    def test_initial_own_clock_is_one(self) -> None:
        ts = ThreadState(tid=1)
        assert ts.vc.get(1) == 1

    def test_initial_other_clocks_are_zero(self) -> None:
        ts = ThreadState(tid=1)
        assert ts.vc.get(2) == 0

    def test_inherits_parent_vc(self) -> None:
        parent_vc = VectorClock({1: 3, 2: 2})
        ts = ThreadState(tid=2, initial_vc=parent_vc)
        assert ts.vc.get(1) == 3
        # parent's clock for tid=2 was 2, ThreadState keeps it (already >= 1)
        assert ts.vc.get(2) == 2

    def test_inherits_parent_vc_bumps_own_clock_when_zero(self) -> None:
        parent_vc = VectorClock({1: 3})  # no entry for child's tid
        ts = ThreadState(tid=2, initial_vc=parent_vc)
        assert ts.vc.get(1) == 3
        # child sees vc[2] == 0 from inheritance; __init__ bumps it to 1
        assert ts.vc.get(2) == 1

    def test_inherited_vc_is_copy(self) -> None:
        parent_vc = VectorClock({1: 3})
        ts = ThreadState(tid=2, initial_vc=parent_vc)
        parent_vc.set(1, 99)
        assert ts.vc.get(1) == 3  # not affected

    def test_tick_increments_own_clock(self) -> None:
        ts = ThreadState(tid=1)
        old = ts.current_clock()
        new = ts.tick()
        assert new == old + 1
        assert ts.current_clock() == new

    def test_absorb_takes_pointwise_max(self) -> None:
        ts = ThreadState(tid=1)
        ts.vc.set(2, 3)
        other = VectorClock({2: 7, 3: 5})
        ts.absorb(other)
        assert ts.vc.get(2) == 7
        assert ts.vc.get(3) == 5

    def test_snapshot_is_copy(self) -> None:
        ts = ThreadState(tid=1)
        snap = ts.snapshot()
        ts.tick()
        assert snap.get(1) != ts.vc.get(1)

    def test_custom_name(self) -> None:
        ts = ThreadState(tid=1, name="Worker-A")
        assert ts.name == "Worker-A"

    def test_default_name_includes_tid(self) -> None:
        ts = ThreadState(tid=12345)
        assert "12345" in ts.name


class TestThreadRegistry:
    def test_register_and_get(self) -> None:
        reg = ThreadRegistry()
        ts = reg.register(1, name="T1")
        assert reg.get(1) is ts

    def test_get_unknown_returns_none(self) -> None:
        reg = ThreadRegistry()
        assert reg.get(999) is None

    def test_get_or_register_creates_if_missing(self) -> None:
        reg = ThreadRegistry()
        ts = reg.get_or_register(42, name="new")
        assert ts is not None
        assert reg.get(42) is ts

    def test_get_or_register_returns_existing(self) -> None:
        reg = ThreadRegistry()
        ts1 = reg.register(42)
        ts2 = reg.get_or_register(42)
        assert ts1 is ts2

    def test_remove(self) -> None:
        reg = ThreadRegistry()
        reg.register(1)
        removed = reg.remove(1)
        assert removed is not None
        assert reg.get(1) is None

    def test_remove_unknown_returns_none(self) -> None:
        reg = ThreadRegistry()
        assert reg.remove(999) is None

    def test_all_tids(self) -> None:
        reg = ThreadRegistry()
        reg.register(1)
        reg.register(2)
        reg.register(3)
        tids = reg.all_tids()
        assert set(tids) == {1, 2, 3}

    def test_thread_safety(self) -> None:
        """Register from many threads simultaneously."""
        reg = ThreadRegistry()
        errors = []

        def worker() -> None:
            tid = threading.get_ident()
            try:
                reg.get_or_register(tid)
                reg.get(tid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
