from pyft.core.epoch import _NONE_TID, Epoch
from pyft.core.var_state import ReadBottom, ReadEpoch, ReadVC, VarState
from pyft.core.vector_clock import VectorClock

# Convenience thread IDs
T1, T2, T3 = 101, 102, 103


def vc(*pairs) -> VectorClock:
    """Helper: vc((t1,c1), (t2,c2)) → VectorClock"""
    return VectorClock(dict(pairs))


class TestInitialState:
    def test_fresh_var_state(self):
        vs = VarState()
        assert vs.write_epoch.is_bottom()
        assert isinstance(vs.read_state, ReadBottom)
        assert vs.first_tid == _NONE_TID
        assert not vs.is_shared

    def test_repr_does_not_crash(self):
        vs = VarState()
        assert "VarState" in repr(vs)


class TestSharingDetection:
    def test_first_access_sets_first_tid(self):
        vs = VarState()
        became_shared = vs._note_access(T1)
        assert vs.first_tid == T1
        assert not became_shared
        assert not vs.is_shared

    def test_second_access_same_thread_not_shared(self):
        vs = VarState()
        vs._note_access(T1)
        became_shared = vs._note_access(T1)
        assert not became_shared
        assert not vs.is_shared

    def test_second_access_different_thread_becomes_shared(self):
        vs = VarState()
        vs._note_access(T1)
        became_shared = vs._note_access(T2)
        assert became_shared
        assert vs.is_shared

    def test_sharing_is_sticky(self):
        vs = VarState()
        vs._note_access(T1)
        vs._note_access(T2)
        became_shared = vs._note_access(T1)  # 3rd access, same as first
        assert not became_shared  # transition already happened
        assert vs.is_shared  # still shared


class TestReadNoRace:
    def test_read_with_no_prior_write(self):
        vs = VarState()
        race, _ = vs.check_read(T2, vc((T2, 1)))
        assert not race

    def test_read_after_ordered_write_same_thread(self):
        """R2: same thread as writer — always safe."""
        vs = VarState()
        vs.write_epoch = Epoch(T1, 3)

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        # T1 reads its own write — no race
        race, _ = vs.check_read(T1, vc((T1, 5)))
        assert not race

    def test_read_after_hb_write_different_thread(self):
        """R3: write happened-before this read."""
        vs = VarState()
        vs.write_epoch = Epoch(T1, 3)

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        # T2's VC has T1's clock at 3 → write is in T2's past
        race, _ = vs.check_read(T2, vc((T1, 3), (T2, 1)))
        assert not race

    def test_read_after_hb_write_clock_exceeds(self):
        vs = VarState()
        vs.write_epoch = Epoch(T1, 2)

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        race, _ = vs.check_read(T2, vc((T1, 5), (T2, 1)))
        assert not race


class TestReadRace:
    def test_concurrent_write_read(self):
        """Write by T1 is concurrent with read by T2."""
        vs = VarState()
        vs.write_epoch = Epoch(T1, 5)

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        # T2 has NOT seen T1's clock at 5
        race, _ = vs.check_read(T2, vc((T1, 2), (T2, 3)))
        assert race

    def test_write_at_exact_boundary_is_not_race(self):
        vs = VarState()
        vs.write_epoch = Epoch(T1, 3)

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        race, _ = vs.check_read(T2, vc((T1, 3), (T2, 1)))
        assert not race


class TestReadStateProgression:
    def test_first_read_creates_read_epoch(self):
        vs = VarState()
        vs.check_read(T1, vc((T1, 2)))
        assert isinstance(vs.read_state, ReadEpoch)
        assert vs.read_state.tid == T1

    def test_second_read_same_thread_stays_epoch(self):
        vs = VarState()
        vs.check_read(T1, vc((T1, 2)))
        vs.check_read(T1, vc((T1, 4)))
        assert isinstance(vs.read_state, ReadEpoch)
        assert vs.read_state.clock == 4

    def test_second_read_different_thread_upgrades_to_vc(self):
        vs = VarState()
        vs.check_read(T1, vc((T1, 2)))
        vs.check_read(T2, vc((T2, 3)))
        assert isinstance(vs.read_state, ReadVC)

    def test_third_reader_adds_to_vc(self):
        vs = VarState()
        vs.check_read(T1, vc((T1, 1)))
        vs.check_read(T2, vc((T2, 1)))
        vs.check_read(T3, vc((T3, 1)))
        assert isinstance(vs.read_state, ReadVC)
        rc_vc = vs.read_state.vc
        assert rc_vc.get(T1) == 1
        assert rc_vc.get(T2) == 1
        assert rc_vc.get(T3) == 1


class TestWriteNoRace:
    def test_same_thread_no_race(self):
        vs = VarState()
        vc = VectorClock()
        vc.set(1, 1)

        vs.check_write(1, vc)
        vc.set(1, 2)

        read_race, write_race, _, _ = vs.check_write(1, vc)
        assert not read_race
        assert not write_race

    def test_write_with_no_prior_state(self):
        vs = VarState()

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        read_race, write_race, _, _ = vs.check_write(T1, vc((T1, 1)))
        assert not read_race
        assert not write_race

    def test_write_after_hb_write(self):
        vs = VarState()
        vs.write_epoch = Epoch(T1, 2)

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        read_race, write_race, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1)))
        assert not write_race

    def test_write_after_hb_read(self):
        vs = VarState()
        vs.read_state = ReadEpoch(T1, 2)

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        read_race, write_race, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1)))
        assert not read_race

    def test_write_clears_read_state(self):
        vs = VarState()

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        vs.check_read(T1, vc((T1, 1)))
        vs.check_write(T2, vc((T1, 1), (T2, 1)))
        assert isinstance(vs.read_state, ReadBottom)

    def test_write_updates_write_epoch(self):
        vs = VarState()

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        vs.check_write(T2, vc((T2, 5)))
        assert vs.write_epoch == Epoch(T2, 5)


class TestWriteRace:
    def test_concurrent_write_write(self):
        vs = VarState()
        vs.write_epoch = Epoch(T1, 5)

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        _, write_race, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1)))
        assert write_race

    def test_concurrent_read_write(self):
        vs = VarState()
        vs.read_state = ReadEpoch(T1, 5)

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        read_race, _, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1)))
        assert read_race

    def test_concurrent_read_write_with_read_vc(self):
        """Race detected when any reader in ReadVC is concurrent."""
        vs = VarState()
        rc_vc = VectorClock({T1: 5, T3: 3})
        from pyft.core.var_state import ReadVC

        vs.read_state = ReadVC(rc_vc)

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        # T2 has seen T3's clock at 3 but not T1's at 5
        read_race, _, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1), (T3, 3)))
        assert read_race

    def test_own_thread_write_after_own_read_not_race(self):
        vs = VarState()
        vs.read_state = ReadEpoch(T1, 3)

        # artificially mark T1 and T2 as exclusive
        vs._note_access(T1)
        vs._note_access(T2)

        read_race, _, _, _ = vs.check_write(T1, vc((T1, 5)))
        assert not read_race


class TestRaceWithLocks:
    def test_lock_orders_writes(self) -> None:
        vs = VarState()

        vc1 = VectorClock()
        vc2 = VectorClock()
        lock_vc = VectorClock()

        vc1.set(1, 1)
        vc2.set(2, 1)

        # T1 acquires
        vc1.join(lock_vc)

        vs.check_write(1, vc1)

        # T1 releases
        lock_vc = vc1.copy()

        # T2 acquires
        vc2.join(lock_vc)

        read_race, write_race, _, _ = vs.check_write(2, vc2)

        assert not write_race

    def test_missing_lock_causes_race(self):
        vs = VarState()

        vc1 = VectorClock()
        vc2 = VectorClock()

        vc1.set(1, 1)
        vc2.set(2, 1)

        vs.check_write(1, vc1)

        # T2 does NOT acquire lock
        read_race, write_race, _, _ = vs.check_write(2, vc2)

        assert write_race


class TestForkHBWrites:
    """
    Regression tests for the 'became_shared → race' short-circuit.

    The previous implementation short-circuited check_write() whenever the
    variable transitioned from thread-local to shared, reporting a write-write
    race even if the new write was happens-before the prior write (e.g. via
    fork). Per VerifiedFT, HB must still be checked in that case.

    These tests go through the real code path (no manual _note_access
    pre-manipulation) so they would have caught the bug.
    """

    def test_fork_ordered_writes_no_race(self):
        """T1 writes, then T2 writes with T2's VC including T1's clock → no race."""
        vs = VarState()

        # T1 writes at clock 2. First access; first_tid = T1; not yet shared.
        _, write_race, _, _ = vs.check_write(T1, vc((T1, 2)))
        assert not write_race

        # Simulate fork: T2 inherits T1's VC, so T2 sees (T1:2). T2 writes.
        # Under the old buggy short-circuit this reported a write-write race
        # because the transition "not shared → shared" fired before HB was
        # checked. It must NOT race because T1's write HB T2's write.
        _, write_race, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1)))
        assert not write_race

    def test_unordered_writes_across_first_sharing_still_race(self):
        """T1 writes, T2 writes WITHOUT HB → still a write-write race."""
        vs = VarState()

        _, write_race, _, _ = vs.check_write(T1, vc((T1, 2)))
        assert not write_race

        # T2's VC does NOT see T1's clock — no HB edge.
        _, write_race, _, _ = vs.check_write(T2, vc((T2, 1)))
        assert write_race

    def test_fork_ordered_write_after_read_no_race(self):
        """T1 reads, fork, T2 writes with HB → no read-write race."""
        vs = VarState()

        # T1 reads first (first access).
        race, _ = vs.check_read(T1, vc((T1, 2)))
        assert not race

        # T2 writes after absorbing T1's VC (fork edge). No race.
        read_race, write_race, _, _ = vs.check_write(T2, vc((T1, 2), (T2, 1)))
        assert not read_race
        assert not write_race

    def test_fork_ordered_read_after_write_no_race(self):
        """T1 writes, fork, T2 reads with HB → no write-read race."""
        vs = VarState()

        _, write_race, _, _ = vs.check_write(T1, vc((T1, 3)))
        assert not write_race

        # T2 reads after inheriting T1's VC.
        race, _ = vs.check_read(T2, vc((T1, 3), (T2, 1)))
        assert not race
