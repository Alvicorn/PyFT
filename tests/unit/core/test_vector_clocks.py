from pyft.core.vector_clock import VectorClock


class TestBasicAccessors:
    def test_empty_clock_returns_zero(self) -> None:
        vc = VectorClock()
        assert vc.get(1) == 0
        assert vc.get(999) == 0

    def test_set_and_get(self) -> None:
        vc = VectorClock()
        vc.set(1, 5)
        assert vc.get(1) == 5

    def test_increment_starts_at_one(self) -> None:
        vc = VectorClock()
        new_val = vc.increment(1)
        assert new_val == 1
        assert vc.get(1) == 1

    def test_increment_increases(self) -> None:
        vc = VectorClock({1: 3})
        new_val = vc.increment(1)
        assert new_val == 4
        assert vc.get(1) == 4

    def test_copy_is_independent(self) -> None:
        vc = VectorClock({1: 5, 2: 3})
        copy = vc.copy()
        copy.set(1, 99)
        assert vc.get(1) == 5  # original unchanged

    def test_items_iteration(self) -> None:
        vc = VectorClock({1: 2, 3: 4})
        items = dict(vc.items())
        assert items == {1: 2, 3: 4}


class TestHappensBefore:
    def test_epoch_hb_true_when_clock_sufficient(self) -> None:
        vc = VectorClock({1: 5})
        assert vc.epoch_happens_before(1, 5)  # exact match
        assert vc.epoch_happens_before(1, 3)  # older epoch

    def test_epoch_hb_false_when_clock_insufficient(self) -> None:
        vc = VectorClock({1: 3})
        assert not vc.epoch_happens_before(1, 5)

    def test_epoch_hb_false_for_unknown_thread(self) -> None:
        vc = VectorClock({1: 5})
        assert not vc.epoch_happens_before(
            2, 1
        )  # tid 2 not in vc → implicit 0 < 1

    def test_vc_leq_equal_clocks(self) -> None:
        a = VectorClock({1: 3, 2: 2})
        b = VectorClock({1: 3, 2: 2})
        assert a.vc_leq(b)
        assert b.vc_leq(a)

    def test_vc_leq_strict_less(self) -> None:
        a = VectorClock({1: 2, 2: 1})
        b = VectorClock({1: 3, 2: 2})
        assert a.vc_leq(b)
        assert not b.vc_leq(a)

    def test_vc_leq_concurrent(self) -> None:
        # a[1] > b[1] but a[2] < b[2] → concurrent, neither ≤ other
        a = VectorClock({1: 3, 2: 1})
        b = VectorClock({1: 1, 2: 3})
        assert not a.vc_leq(b)
        assert not b.vc_leq(a)

    def test_vc_leq_missing_entries_are_zero(self) -> None:
        a = VectorClock({1: 1})
        b = VectorClock({1: 2, 2: 1})
        # a[1]=1 <= b[1]=2; a[2]=0 (missing) <= b[2]=1
        assert a.vc_leq(b)


class TestMerging:
    def test_join_takes_pointwise_max(self) -> None:
        a = VectorClock({1: 5, 2: 2})
        b = VectorClock({1: 3, 2: 7, 3: 1})
        a.join(b)
        assert a.get(1) == 5
        assert a.get(2) == 7
        assert a.get(3) == 1

    def test_join_does_not_modify_other(self) -> None:
        a = VectorClock({1: 1})
        b = VectorClock({1: 10})
        a.join(b)
        assert b.get(1) == 10  # b unchanged

    def test_joined_static_method(self) -> None:
        a = VectorClock({1: 5})
        b = VectorClock({1: 3, 2: 4})
        result = VectorClock.joined(a, b)
        assert result.get(1) == 5
        assert result.get(2) == 4
        # originals unchanged
        assert a.get(2) == 0


class TestEpochConversions:
    def test_from_epoch(self) -> None:
        vc = VectorClock.from_epoch(7, 3)
        assert vc.get(7) == 3
        assert vc.get(1) == 0

    def test_add_epoch_updates_if_newer(self) -> None:
        vc = VectorClock({1: 3})
        vc.add_epoch(1, 5)
        assert vc.get(1) == 5

    def test_add_epoch_ignores_older(self) -> None:
        vc = VectorClock({1: 5})
        vc.add_epoch(1, 3)
        assert vc.get(1) == 5


class TestEquality:
    def test_equal_clocks(self) -> None:
        assert VectorClock({1: 2}) == VectorClock({1: 2})

    def test_missing_and_zero_are_equal(self) -> None:
        a = VectorClock({1: 0})
        b = VectorClock()
        assert a == b

    def test_not_equal(self) -> None:
        assert VectorClock({1: 1}) != VectorClock({1: 2})
