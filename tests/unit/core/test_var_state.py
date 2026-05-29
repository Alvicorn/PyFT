"""
Tests for the VerifiedFT per-variable analyser.

Two implementations live in ``pyft.core.var_state``:

  * ``VarStateV2`` — epoch-compressed FastTrack (default; matches the
    paper's optimized analyzer).
  * ``VarStateV1`` — idealized analyzer; full vector clocks for the
    last write and union of reads.

Both expose the same public surface
(``check_read`` / ``check_write`` / ``_note_access`` / ``is_shared`` /
``first_tid``), so the behavioural classes are parametrized over both
analyzers via ``var_state_cls``. The internal-state classes at the
bottom of the file are per-version because each implementation stores
its shadow state in its own shape.
"""

from __future__ import annotations

import pytest

from pyft.core.epoch import _NONE_TID, Epoch
from pyft.core.var_state import (
    ReadBottom,
    ReadEpoch,
    ReadVC,
    VarStateV1,
    VarStateV2,
)
from pyft.core.vector_clock import VectorClock

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

T1, T2, T3 = 101, 102, 103

VAR_STATE_CLASSES = (VarStateV1, VarStateV2)
_IDS = ("V1", "V2")


def vc(*pairs: tuple[int, int]) -> VectorClock:
    """Build a VectorClock from (tid, clock) pairs."""
    return VectorClock(dict(pairs))


def _mark_shared(vs: object, *tids: int) -> None:
    """
    Force ``_note_access`` to flip the variable into the shared
    state for the given tids. Used by tests that want to skip the
    first-access setup.
    """
    for t in tids:
        vs._note_access(t)  # type: ignore[attr-defined]


# ===========================================================================
# Behaviour shared by V1 and V2
# ===========================================================================


@pytest.mark.parametrize("var_state_cls", VAR_STATE_CLASSES, ids=_IDS)
class TestSharingDetection:
    """
    ``_note_access`` + ``first_tid`` + ``is_shared`` are identical
    in both analyzers.
    """

    def test_first_access_sets_first_tid(self, var_state_cls: type) -> None:
        vs = var_state_cls()
        became_shared = vs._note_access(T1)
        assert vs.first_tid == T1
        assert not became_shared
        assert not vs.is_shared

    def test_second_access_same_thread_not_shared(
        self, var_state_cls: type
    ) -> None:
        vs = var_state_cls()
        vs._note_access(T1)
        became_shared = vs._note_access(T1)
        assert not became_shared
        assert not vs.is_shared

    def test_second_access_different_thread_becomes_shared(
        self, var_state_cls: type
    ) -> None:
        vs = var_state_cls()
        vs._note_access(T1)
        became_shared = vs._note_access(T2)
        assert became_shared
        assert vs.is_shared

    def test_sharing_is_sticky(self, var_state_cls: type) -> None:
        vs = var_state_cls()
        vs._note_access(T1)
        vs._note_access(T2)
        became_shared = vs._note_access(T1)
        assert not became_shared
        assert vs.is_shared

    def test_initial_first_tid_is_none(self, var_state_cls: type) -> None:
        vs = var_state_cls()
        assert vs.first_tid == _NONE_TID
        assert not vs.is_shared


@pytest.mark.parametrize("var_state_cls", VAR_STATE_CLASSES, ids=_IDS)
class TestReadNoRaceBehaviour:
    """
    ``check_read`` returns ``(False, ...)`` whenever the access is
    safe under the VerifiedFT read rule.
    """

    def test_read_with_no_prior_write(self, var_state_cls: type) -> None:
        vs = var_state_cls()
        race, _ = vs.check_read(T2, vc((T2, 1)))
        assert not race

    def test_read_after_same_thread_write(self, var_state_cls: type) -> None:
        vs = var_state_cls()
        # T1 writes at clock 3, then T1 reads at a later clock.
        vs.check_write(T1, vc((T1, 3)))
        race, _ = vs.check_read(T1, vc((T1, 5)))
        assert not race

    def test_read_after_hb_write_from_other_thread(
        self, var_state_cls: type
    ) -> None:
        vs = var_state_cls()
        # T1 writes at clock 3.
        vs.check_write(T1, vc((T1, 3)))
        # T2 has seen T1's clock at 3 → write HB read → no race.
        race, _ = vs.check_read(T2, vc((T1, 3), (T2, 1)))
        assert not race

    def test_read_at_exact_boundary_is_not_race(
        self, var_state_cls: type
    ) -> None:
        vs = var_state_cls()
        vs.check_write(T1, vc((T1, 3)))
        race, _ = vs.check_read(T2, vc((T1, 3), (T2, 1)))
        assert not race

    def test_read_at_higher_clock_is_not_race(
        self, var_state_cls: type
    ) -> None:
        vs = var_state_cls()
        vs.check_write(T1, vc((T1, 2)))
        race, _ = vs.check_read(T2, vc((T1, 5), (T2, 1)))
        assert not race


@pytest.mark.parametrize("var_state_cls", VAR_STATE_CLASSES, ids=_IDS)
class TestReadRaceBehaviour:
    def test_concurrent_write_then_read_is_race(
        self, var_state_cls: type
    ) -> None:
        vs = var_state_cls()
        # T1 writes at clock 5.
        vs.check_write(T1, vc((T1, 5)))
        # T2 has only seen T1 up to clock 2 → race.
        race, _ = vs.check_read(T2, vc((T1, 2), (T2, 3)))
        assert race


@pytest.mark.parametrize("var_state_cls", VAR_STATE_CLASSES, ids=_IDS)
class TestWriteRaceBehaviour:
    def test_concurrent_write_write_is_race(self, var_state_cls: type) -> None:
        vs = var_state_cls()
        vs.check_write(T1, vc((T1, 5)))
        _, write_race, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1)))
        assert write_race

    def test_concurrent_read_then_write_is_read_race(
        self, var_state_cls: type
    ) -> None:
        vs = var_state_cls()
        # T1 reads at clock 5.
        vs.check_read(T1, vc((T1, 5)))
        # T2 writes without HB to T1 → read-race.
        read_race, _, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1)))
        assert read_race

    def test_own_thread_write_after_own_read_not_race(
        self, var_state_cls: type
    ) -> None:
        vs = var_state_cls()
        vs.check_read(T1, vc((T1, 3)))
        # Same thread writes — its own prior read can't race with it.
        read_race, write_race, _, _ = vs.check_write(T1, vc((T1, 5)))
        assert not read_race
        assert not write_race

    def test_hb_write_after_write_not_race(self, var_state_cls: type) -> None:
        vs = var_state_cls()
        vs.check_write(T1, vc((T1, 2)))
        # T2 has seen T1's clock at 2 → HB → no race.
        _, write_race, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1)))
        assert not write_race

    def test_hb_write_after_read_not_race(self, var_state_cls: type) -> None:
        vs = var_state_cls()
        vs.check_read(T1, vc((T1, 2)))
        # T2's VC sees T1's read clock → no read-race.
        read_race, _, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1)))
        assert not read_race


@pytest.mark.parametrize("var_state_cls", VAR_STATE_CLASSES, ids=_IDS)
class TestRaceWithLocks:
    def test_lock_orders_writes(self, var_state_cls: type) -> None:
        vs = var_state_cls()

        vc1 = VectorClock()
        vc2 = VectorClock()
        lock_vc = VectorClock()

        vc1.set(1, 1)
        vc2.set(2, 1)

        # T1 acquires.
        vc1.join(lock_vc)
        vs.check_write(1, vc1)
        # T1 releases.
        lock_vc = vc1.copy()
        # T2 acquires.
        vc2.join(lock_vc)
        _, write_race, _, _ = vs.check_write(2, vc2)
        assert not write_race

    def test_missing_lock_causes_race(self, var_state_cls: type) -> None:
        vs = var_state_cls()
        vc1 = VectorClock()
        vc2 = VectorClock()
        vc1.set(1, 1)
        vc2.set(2, 1)
        vs.check_write(1, vc1)
        # T2 did NOT acquire the lock.
        _, write_race, _, _ = vs.check_write(2, vc2)
        assert write_race


@pytest.mark.parametrize("var_state_cls", VAR_STATE_CLASSES, ids=_IDS)
class TestForkHBWrites:
    """
    Regression coverage for the 'became_shared -> race' short-circuit
    bug. Per VerifiedFT, HB must still be checked once the variable
    becomes shared.
    """

    def test_fork_ordered_writes_no_race(self, var_state_cls: type) -> None:
        vs = var_state_cls()
        _, write_race, _, _ = vs.check_write(T1, vc((T1, 2)))
        assert not write_race
        # T2 inherits T1's VC at fork; writes after HB.
        _, write_race, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1)))
        assert not write_race

    def test_unordered_writes_across_first_sharing_still_race(
        self, var_state_cls: type
    ) -> None:
        vs = var_state_cls()
        _, write_race, _, _ = vs.check_write(T1, vc((T1, 2)))
        assert not write_race
        # T2's VC has no view of T1 → race.
        _, write_race, _, _ = vs.check_write(T2, vc((T2, 1)))
        assert write_race

    def test_fork_ordered_write_after_read_no_race(
        self, var_state_cls: type
    ) -> None:
        vs = var_state_cls()
        race, _ = vs.check_read(T1, vc((T1, 2)))
        assert not race
        read_race, write_race, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1)))
        assert not read_race
        assert not write_race

    def test_fork_ordered_read_after_write_no_race(
        self, var_state_cls: type
    ) -> None:
        vs = var_state_cls()
        _, write_race, _, _ = vs.check_write(T1, vc((T1, 3)))
        assert not write_race
        race, _ = vs.check_read(T2, vc((T1, 3), (T2, 1)))
        assert not race


# ===========================================================================
# V2-only tests (epoch-compressed shadow state)
# ===========================================================================


class TestVarStateV2Internal:
    """
    Tests that inspect V2's ``write_epoch`` / ``read_state`` shadow
    fields — these have no direct counterpart in V1.
    """

    def test_initial_state(self) -> None:
        vs = VarStateV2()
        assert vs.write_epoch.is_bottom()
        assert isinstance(vs.read_state, ReadBottom)
        assert vs.first_tid == _NONE_TID
        assert not vs.is_shared

    def test_repr_does_not_crash(self) -> None:
        assert "VarStateV2" in repr(VarStateV2())

    def test_first_read_creates_read_epoch(self) -> None:
        vs = VarStateV2()
        vs.check_read(T1, vc((T1, 2)))
        assert isinstance(vs.read_state, ReadEpoch)
        assert vs.read_state.tid == T1

    def test_second_read_same_thread_stays_epoch(self) -> None:
        vs = VarStateV2()
        vs.check_read(T1, vc((T1, 2)))
        vs.check_read(T1, vc((T1, 4)))
        assert isinstance(vs.read_state, ReadEpoch)
        assert vs.read_state.clock == 4

    def test_second_read_different_thread_upgrades_to_vc(self) -> None:
        vs = VarStateV2()
        vs.check_read(T1, vc((T1, 2)))
        vs.check_read(T2, vc((T2, 3)))
        assert isinstance(vs.read_state, ReadVC)

    def test_third_reader_adds_to_vc(self) -> None:
        vs = VarStateV2()
        vs.check_read(T1, vc((T1, 1)))
        vs.check_read(T2, vc((T2, 1)))
        vs.check_read(T3, vc((T3, 1)))
        assert isinstance(vs.read_state, ReadVC)
        rc_vc = vs.read_state.vc
        assert rc_vc.get(T1) == 1
        assert rc_vc.get(T2) == 1
        assert rc_vc.get(T3) == 1

    def test_write_clears_read_state(self) -> None:
        vs = VarStateV2()
        _mark_shared(vs, T1, T2)
        vs.check_read(T1, vc((T1, 1)))
        vs.check_write(T2, vc((T1, 1), (T2, 1)))
        assert isinstance(vs.read_state, ReadBottom)

    def test_write_updates_write_epoch(self) -> None:
        vs = VarStateV2()
        _mark_shared(vs, T1, T2)
        vs.check_write(T2, vc((T2, 5)))
        assert vs.write_epoch == Epoch(T2, 5)

    def test_concurrent_read_write_with_read_vc(self) -> None:
        """
        V2-specific: an existing ReadVC with several readers must
        still produce a read-race when a non-HB write arrives.
        """
        vs = VarStateV2()
        vs.read_state = ReadVC(VectorClock({T1: 5, T3: 3}))
        _mark_shared(vs, T1, T2)
        # T2 has seen T3@3 but not T1@5 → race.
        read_race, _, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1), (T3, 3)))
        assert read_race


# ===========================================================================
# V1-only tests (full vector clocks)
# ===========================================================================


class TestVarStateV1Internal:
    """Tests that inspect V1's ``W_x`` / ``R_x`` shadow fields."""

    def test_initial_state(self) -> None:
        vs = VarStateV1()
        assert vs.W_x == VectorClock()
        assert vs.R_x == VectorClock()
        assert vs.first_tid == _NONE_TID
        assert not vs.is_shared

    def test_repr_does_not_crash(self) -> None:
        assert "VarStateV1" in repr(VarStateV1())

    def test_first_write_sets_W_x(self) -> None:
        vs = VarStateV1()
        rr, wr, prior_w, prior_r = vs.check_write(T1, vc((T1, 1)))
        assert not rr and not wr
        assert prior_w.is_bottom()
        assert isinstance(prior_r, ReadBottom)
        assert vs.W_x.get(T1) == 1

    def test_safe_read_updates_R_x(self) -> None:
        vs = VarStateV1()
        vs.check_read(T1, vc((T1, 4)))
        assert vs.R_x.get(T1) == 4
        vs.check_read(T2, vc((T2, 7)))
        assert vs.R_x.get(T2) == 7

    def test_raced_read_does_not_update_R_x(self) -> None:
        vs = VarStateV1()
        vs.check_write(T1, vc((T1, 5)))
        # T2 races against T1's write — R_x must not absorb T2's clock.
        race, _ = vs.check_read(T2, vc((T1, 2), (T2, 1)))
        assert race
        assert vs.R_x.get(T2) == 0

    def test_write_resets_R_x(self) -> None:
        vs = VarStateV1()
        vs.check_read(T1, vc((T1, 4)))
        vs.check_read(T2, vc((T1, 4), (T2, 5)))
        assert vs.R_x.get(T1) == 4 and vs.R_x.get(T2) == 5
        # A subsequent HB write resets R_x to bottom.
        vs.check_write(T1, vc((T1, 8), (T2, 5)))
        assert vs.R_x == VectorClock()

    def test_concurrent_read_and_write_with_two_readers(self) -> None:
        """
        V1-specific: two prior readers + a non-HB writer → V1
        synthesizes a ``ReadEpoch`` for the offending reader.
        """
        vs = VarStateV1()
        vs.check_read(T1, vc((T1, 4)))
        vs.check_read(T2, vc((T1, 4), (T2, 6)))
        # T3 writes concurrently with both prior readers.
        rr, _, _, prior_r = vs.check_write(T3, vc((T3, 1)))
        assert rr
        assert isinstance(prior_r, ReadEpoch)
        assert prior_r.tid in (T1, T2)

    def test_concurrent_write_read_extracts_violator(self) -> None:
        vs = VarStateV1()
        vs.check_write(T1, vc((T1, 5)))
        race, prior = vs.check_read(T2, vc((T1, 2), (T2, 3)))
        assert race
        assert prior == Epoch(T1, 5)
