"""
tests/benchmarks/test_drb_suite.py

A small suite of DataRaceBench-inspired test programs, ported to Python
(https://github.com/llnl/dataracebench).Each benchmark drives the detector
through the full on_read/on_write API (simulating what the AST instrumentation
would produce) on shared Python objects with actual thread execution.

Naming convention (mirrors DataRaceBench):
  drb_<id>_<yes|no>_<description>
    yes = contains a race
    no  = race-free
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import threading
import unittest

from pyft.detector.engine import Engine


class _RaceLogFacade:
    """Adapter: exposes .count() / .reports() over the real RaceLog."""

    def __init__(self, log):
        self._log = log

    def count(self) -> int:
        return len(self._log)

    def reports(self):
        return self._log.all_reports()


class VerifiedFTDetector:
    """
    Thin shim bridging the test suite's tuple-key API to the real Engine.

    Tests call:
        det.on_read(key)   / det.on_write(key)
            where key = ("attr", id(obj), attr_name)
                      | ("list", id(lst), index)
                      | ("global", *parts)
        det.on_acquire(lock_obj) / det.on_release(lock_obj)

    Each unique key tuple maps to a stable sentinel object so the engine
    tracks happens-before per variable correctly across threads.

    Thread lifecycle (thread_start / thread_finish) is injected by
    patching threading.Thread.start so the engine assigns each Thread
    object a distinct id()-based tid — immune to OS thread-ID reuse.
    """

    def __init__(self) -> None:
        self._engine = Engine()
        self.race_log = _RaceLogFacade(self._engine.race_log)
        self._sentinels: dict = {}
        self._sentinels_mu = threading.Lock()
        self._orig_thread_start = threading.Thread.start
        self._install_thread_hooks()

    def _install_thread_hooks(self) -> None:
        engine = self._engine
        orig = self._orig_thread_start

        def patched_start(thread_self: threading.Thread) -> None:
            # Fork HB edge from PARENT before child starts.
            parent_tid = id(threading.current_thread())
            child_tid = id(thread_self)
            engine.thread_start(parent_tid, child_tid, thread_self.name)

            orig_run = thread_self.run

            def instrumented_run() -> None:
                try:
                    orig_run()
                finally:
                    engine.thread_finish(child_tid)

            thread_self.run = instrumented_run
            orig(thread_self)

        threading.Thread.start = patched_start  # type: ignore[method-assign]

    def teardown(self) -> None:
        threading.Thread.start = self._orig_thread_start  # type: ignore[method-assign]

    def _resolve(self, key: tuple):
        """Return (sentinel_obj, attr_str) for the given key tuple."""
        with self._sentinels_mu:
            if key not in self._sentinels:
                self._sentinels[key] = object()
        return self._sentinels[key], str(key[-1])

    def on_read(self, key) -> None:
        obj, attr = self._resolve(key)
        self._engine.read(obj, attr)

    def on_write(self, key) -> None:
        obj, attr = self._resolve(key)
        self._engine.write(obj, attr)

    def on_acquire(self, lock) -> None:
        self._engine.lock_acquire(id(lock))

    def on_release(self, lock) -> None:
        self._engine.lock_release(id(lock))


def make_det() -> VerifiedFTDetector:
    return VerifiedFTDetector()


class DRBTestCase(unittest.TestCase):
    def setUp(self):
        self.det = make_det()

    def _key(self, obj, attr):
        return ("attr", id(obj), attr)

    def _list_key(self, lst, index):
        return ("list", id(lst), index)

    def assertRace(self):
        self.assertGreater(
            self.det.race_log.count(),
            0,
            "Expected a race but none was detected.\n"
            + "(Hint: check thread scheduling — barrier may need adjustment.)",
        )

    def tearDown(self):
        self.det.teardown()

    def assertNoRace(self):
        self.assertEqual(
            self.det.race_log.count(),
            0,
            "Expected no race but got:\n"
            + "\n".join(str(r) for r in self.det.race_log.reports()),
        )


# ===========================================================================
# DRB 001 — yes — unsynchronised increment (read-modify-write)
# Classic race: two threads both read, increment, and write a counter.
# ===========================================================================


class TestDRB001_Yes_Increment(DRBTestCase):
    """
    Corresponds to DataRaceBench DRB001-antidep1-orig-yes.c
    Two threads increment a shared counter without synchronisation.
    """

    def test_unsync_increment(self):
        class Counter:
            value = 0

        c = Counter()
        barrier = threading.Barrier(2)
        key = self._key(c, "value")

        def inc():
            barrier.wait()
            self.det.on_read(key)
            self.det.on_write(key)

        threads = [threading.Thread(target=inc) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertRace()


# ===========================================================================
# DRB 002 — no — synchronised increment
# Same as above but with a lock.
# ===========================================================================


class TestDRB002_No_SyncIncrement(DRBTestCase):
    """
    Corresponds to DataRaceBench DRB002-antidep1-orig-no.c
    """

    def test_sync_increment(self):
        class Counter:
            value = 0

        c = Counter()
        lock = object()
        done1 = threading.Event()
        key = self._key(c, "value")

        def inc1():
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def inc2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_write(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=inc1)
        t2 = threading.Thread(target=inc2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 003 — yes — producer / consumer without synchronisation
# Producer writes; consumer reads without any HB relationship.
# ===========================================================================


class TestDRB003_Yes_ProducerConsumer(DRBTestCase):
    """
    Corresponds to DRB: producer writes data, consumer reads, no sync.
    """

    def test_unsync_producer_consumer(self):
        class Shared:
            data = None
            ready = False

        s = Shared()
        barrier = threading.Barrier(2)
        data_key = self._key(s, "data")
        ready_key = self._key(s, "ready")

        def producer():
            barrier.wait()
            self.det.on_write(data_key)
            self.det.on_write(ready_key)

        def consumer():
            barrier.wait()
            self.det.on_read(ready_key)
            self.det.on_read(data_key)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 004 — no — producer / consumer with lock handoff
# ===========================================================================


class TestDRB004_No_ProducerConsumer(DRBTestCase):
    """
    Same as DRB003 but producer releases a lock that consumer acquires.
    """

    def test_sync_producer_consumer(self):
        class Shared:
            data = None

        s = Shared()
        lock = object()
        published = threading.Event()
        data_key = self._key(s, "data")

        def producer():
            self.det.on_write(data_key)
            self.det.on_release(lock)
            published.set()

        def consumer():
            published.wait()
            self.det.on_acquire(lock)
            self.det.on_read(data_key)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 005 — no — two threads write different fields of same object
# Each field is a separate memory location — the races are per-field.
# ===========================================================================


class TestDRB005_No_IndependentFields(DRBTestCase):
    """
    DRB: two threads write *different* fields of the same object.
    Should NOT be a race (different keys).
    """

    def test_independent_fields(self):
        class Point:
            x = 0
            y = 0

        p = Point()
        barrier = threading.Barrier(2)
        key_x = self._key(p, "x")
        key_y = self._key(p, "y")

        def write_x():
            barrier.wait()
            self.det.on_write(key_x)

        def write_y():
            barrier.wait()
            self.det.on_write(key_y)

        t1 = threading.Thread(target=write_x)
        t2 = threading.Thread(target=write_y)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 006 — yes — lazy initialisation without double-checked locking
# Classic benign-looking race that is actually undefined behaviour.
# ===========================================================================


class TestDRB006_Yes_LazyInit(DRBTestCase):
    """
    Thread 1 checks-then-sets an 'initialised' flag without a lock.
    Thread 2 does the same concurrently.  Race on the flag.
    """

    def test_lazy_init_race(self):
        class Singleton:
            instance = None

        s = Singleton()
        barrier = threading.Barrier(2)
        key = self._key(s, "instance")

        def maybe_init():
            barrier.wait()
            self.det.on_read(key)
            self.det.on_write(key)

        t1 = threading.Thread(target=maybe_init)
        t2 = threading.Thread(target=maybe_init)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 007 — no — thread-local accumulator, merged under lock
# Each thread writes only its own local key; merge happens sequentially.
# ===========================================================================


class TestDRB007_No_ThreadLocalAccumulator(DRBTestCase):
    """
    Each thread accumulates into its own 'slot'; a reducer merges results
    under a lock.  No race.
    """

    def test_thread_local_accumulate(self):
        class Results:
            pass

        results = Results()
        handoff_a = object()
        handoff_b = object()
        total_key = ("global", "__test__", "total")

        done_a = threading.Event()
        done_b = threading.Event()

        def worker_a():
            self.det.on_write(self._key(results, "slot_a"))
            self.det.on_release(handoff_a)
            done_a.set()

        def worker_b():
            self.det.on_write(self._key(results, "slot_b"))
            self.det.on_release(handoff_b)
            done_b.set()

        t1 = threading.Thread(target=worker_a)
        t2 = threading.Thread(target=worker_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        done_a.wait()
        done_b.wait()
        self.det.on_acquire(handoff_a)
        self.det.on_acquire(handoff_b)
        self.det.on_read(self._key(results, "slot_a"))
        self.det.on_read(self._key(results, "slot_b"))
        self.det.on_write(total_key)

        self.assertNoRace()


# ===========================================================================
# DRB 008 — yes — missing release (simulated as write without lock release)
# A lock is acquired but never released, so the second thread's acquire
# never fires — the second write has no HB to the first.
# ===========================================================================


class TestDRB008_Yes_MissingRelease(DRBTestCase):
    """
    Thread 1 acquires a lock and writes, but forgets to release.
    Thread 2 proceeds to write anyway (bug: acquire path skipped).
    """

    def test_missing_release(self):
        lock = object()
        done1 = threading.Event()
        key = ("global", "__test__", "shared")

        def writer1():
            self.det.on_acquire(lock)
            self.det.on_write(key)
            done1.set()

        def writer2():
            done1.wait()
            self.det.on_write(key)

        t1 = threading.Thread(target=writer1)
        t2 = threading.Thread(target=writer2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 009 — yes — write-write race on same variable
# Two threads write the same integer without any synchronisation.
# ===========================================================================


class TestDRB009_Yes_WriteWriteRace(DRBTestCase):
    def test_write_write_race(self):
        class Flag:
            val = 0

        f = Flag()
        barrier = threading.Barrier(2)
        key = self._key(f, "val")

        def writer():
            barrier.wait()
            self.det.on_write(key)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=writer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 010 — no — write-write with lock
# ===========================================================================


class TestDRB010_No_WriteWriteLock(DRBTestCase):
    def test_write_write_locked(self):
        class Flag:
            val = 0

        f = Flag()
        lock = object()
        done1 = threading.Event()
        key = self._key(f, "val")

        def writer1():
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def writer2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=writer1)
        t2 = threading.Thread(target=writer2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 011 — yes — reduction missing (two threads accumulate without sync)
# ===========================================================================


class TestDRB011_Yes_ReductionMissing(DRBTestCase):
    def test_reduction_missing(self):
        class Accum:
            val = 0

        acc = Accum()
        barrier = threading.Barrier(2)
        key = self._key(acc, "val")

        def add():
            barrier.wait()
            self.det.on_read(key)
            self.det.on_write(key)

        t1 = threading.Thread(target=add)
        t2 = threading.Thread(target=add)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 012 — no — reduction with lock
# ===========================================================================


class TestDRB012_No_ReductionLock(DRBTestCase):
    def test_reduction_locked(self):
        class Accum:
            val = 0

        acc = Accum()
        lock = object()
        done1 = threading.Event()
        key = self._key(acc, "val")

        def add1():
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def add2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_write(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=add1)
        t2 = threading.Thread(target=add2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 013 — yes — missing barrier between write and read
# Thread 1 writes, thread 2 reads without any HB. Race.
# ===========================================================================


class TestDRB013_Yes_MissingBarrier(DRBTestCase):
    def test_missing_barrier(self):
        class Shared:
            data = None

        s = Shared()
        barrier = threading.Barrier(2)
        key = self._key(s, "data")

        def writer():
            barrier.wait()
            self.det.on_write(key)

        def reader():
            barrier.wait()
            self.det.on_read(key)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 014 — no — barrier established via lock handoff (simulated barrier)
# Lock release after write, acquire before read gives HB.
# ===========================================================================


class TestDRB014_No_BarrierWithLock(DRBTestCase):
    def test_barrier_with_lock(self):
        class Shared:
            data = None

        s = Shared()
        lock = object()
        published = threading.Event()
        key = self._key(s, "data")

        def writer():
            self.det.on_write(key)
            self.det.on_release(lock)
            published.set()

        def reader():
            published.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 015 — yes — two threads write same list element
# ===========================================================================


class TestDRB015_Yes_ArrayWriteSameIndex(DRBTestCase):
    def test_array_write_same_index(self):
        lst = [0, 0]
        barrier = threading.Barrier(2)
        key0 = self._list_key(lst, 0)

        def worker():
            barrier.wait()
            self.det.on_write(key0)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 016 — no — two threads write different list elements
# ===========================================================================


class TestDRB016_No_ArrayWriteDifferentIndex(DRBTestCase):
    def test_array_write_different_index(self):
        lst = [0, 0]
        barrier = threading.Barrier(2)
        key0 = self._list_key(lst, 0)
        key1 = self._list_key(lst, 1)

        def write0():
            barrier.wait()
            self.det.on_write(key0)

        def write1():
            barrier.wait()
            self.det.on_write(key1)

        t1 = threading.Thread(target=write0)
        t2 = threading.Thread(target=write1)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 017 — yes — indirect access through alias (two references to same object)
# ===========================================================================


class TestDRB017_Yes_IndirectAliasing(DRBTestCase):
    def test_indirect_aliasing(self):
        class Box:
            value = 0

        b = Box()
        alias = b  # another reference to the same object
        barrier = threading.Barrier(2)
        key = self._key(b, "value")  # same key as alias.value

        def writer1():
            barrier.wait()
            self.det.on_write(key)

        def writer2():
            barrier.wait()
            # using alias
            self.det.on_write(self._key(alias, "value"))

        t1 = threading.Thread(target=writer1)
        t2 = threading.Thread(target=writer2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 018 — no — indirect access with lock
# ===========================================================================


class TestDRB018_No_IndirectAliasingLock(DRBTestCase):
    def test_indirect_aliasing_locked(self):
        class Box:
            value = 0

        b = Box()
        alias = b
        lock = object()
        done1 = threading.Event()
        key = self._key(b, "value")

        def writer1():
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def writer2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(self._key(alias, "value"))
            self.det.on_release(lock)

        t1 = threading.Thread(target=writer1)
        t2 = threading.Thread(target=writer2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 019 — yes — double-checked locking without proper sync (variant)
# ===========================================================================


class TestDRB019_Yes_DoubleCheckedLocking(DRBTestCase):
    """
    Threads check a flag, then write without holding a lock.
    Both can enter the critical section concurrently.
    """

    def test_double_checked_race(self):
        class Init:
            done = False
            data = None

        init = Init()
        barrier = threading.Barrier(2)
        done_key = self._key(init, "done")
        data_key = self._key(init, "data")

        def init_once():
            barrier.wait()
            # Check without proper lock
            self.det.on_read(done_key)
            self.det.on_write(data_key)
            self.det.on_write(done_key)

        t1 = threading.Thread(target=init_once)
        t2 = threading.Thread(target=init_once)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # race on done_key (write/write) and also on data_key
        self.assertRace()


# ===========================================================================
# DRB 020 — no — double-checked locking with proper lock
# ===========================================================================


class TestDRB020_No_DoubleCheckedLocking(DRBTestCase):
    def test_double_checked_locked(self):
        class Init:
            done = False
            data = None

        init = Init()
        lock = object()
        done1 = threading.Event()
        done_key = self._key(init, "done")
        data_key = self._key(init, "data")

        def init1():
            self.det.on_read(done_key)
            self.det.on_acquire(lock)
            self.det.on_write(data_key)
            self.det.on_write(done_key)
            self.det.on_release(lock)
            self.det.on_release(done1)  # model event.set() as HB release
            done1.set()

        def init2():
            done1.wait()
            self.det.on_acquire(done1)  # model event.wait() as HB acquire
            self.det.on_read(done_key)
            # HB established via event handoff: no race on done_key

        t1 = threading.Thread(target=init1)
        t2 = threading.Thread(target=init2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 021 — yes — flag race after thread creation (missing join handoff)
# ===========================================================================


class TestDRB021_Yes_FlagAfterCreate(DRBTestCase):
    """
    Thread 1 writes a flag, thread 2 reads it without any sync.
    Even if thread 2 was created *after* the write, without a
    happens-before edge it's still a race.
    """

    def test_flag_after_create(self):
        class Shared:
            flag = False

        sh = Shared()
        started = threading.Event()
        key = self._key(sh, "flag")

        def writer():
            self.det.on_write(key)
            started.set()

        def reader():
            started.wait()
            self.det.on_read(key)

        # Launch writer first, then reader, but no sync edges
        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t1.join()  # join does not give HB to t2 in our detector model
        t2.start()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 022 — no — flag with lock handoff (join-like semantics)
# ===========================================================================


class TestDRB022_No_FlagWithHandoff(DRBTestCase):
    def test_flag_with_handoff(self):
        class Shared:
            flag = False

        sh = Shared()
        lock = object()
        published = threading.Event()
        key = self._key(sh, "flag")

        def writer():
            self.det.on_write(key)
            self.det.on_release(lock)
            published.set()

        def reader():
            published.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 023 — yes — race on a shared counter in a loop (unprotected)
# ===========================================================================


class TestDRB023_Yes_LoopCounterRace(DRBTestCase):
    def test_loop_counter_race(self):
        class Counter:
            cnt = 0

        c = Counter()
        barrier = threading.Barrier(2)
        key = self._key(c, "cnt")

        def loop():
            barrier.wait()
            for _ in range(3):
                self.det.on_read(key)
                self.det.on_write(key)

        t1 = threading.Thread(target=loop)
        t2 = threading.Thread(target=loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 024 — no — loop counter with lock
# ===========================================================================


class TestDRB024_No_LoopCounterLocked(DRBTestCase):
    def test_loop_counter_locked(self):
        class Counter:
            cnt = 0

        c = Counter()
        lock = object()
        done1 = threading.Event()
        key = self._key(c, "cnt")

        def loop1():
            for _ in range(3):
                self.det.on_acquire(lock)
                self.det.on_read(key)
                self.det.on_write(key)
                self.det.on_release(lock)
            done1.set()

        def loop2():
            done1.wait()
            for _ in range(3):
                self.det.on_acquire(lock)
                self.det.on_read(key)
                self.det.on_write(key)
                self.det.on_release(lock)

        t1 = threading.Thread(target=loop1)
        t2 = threading.Thread(target=loop2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 025 — yes — read-write race on different fields? Actually same field
# ===========================================================================


class TestDRB025_Yes_ReadWriteRaceSameField(DRBTestCase):
    def test_read_write_same_field(self):
        class Box:
            val = 0

        b = Box()
        barrier = threading.Barrier(2)
        key = self._key(b, "val")

        def writer():
            barrier.wait()
            self.det.on_write(key)

        def reader():
            barrier.wait()
            self.det.on_read(key)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 026 — no — multiple readers (no race)
# ===========================================================================


class TestDRB026_No_MultipleReaders(DRBTestCase):
    def test_multiple_readers(self):
        class Box:
            val = 42

        b = Box()
        barrier = threading.Barrier(2)
        key = self._key(b, "val")

        def reader():
            barrier.wait()
            self.det.on_read(key)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 027 — yes — write after read without sync (different threads)
# ===========================================================================


class TestDRB027_Yes_WriteAfterReadRace(DRBTestCase):
    def test_write_after_read_race(self):
        class Box:
            val = 0

        b = Box()
        key = self._key(b, "val")
        started = threading.Event()

        def read_then_signal():
            self.det.on_read(key)
            started.set()

        def write_after_signal():
            started.wait()
            self.det.on_write(key)

        t1 = threading.Thread(target=read_then_signal)
        t2 = threading.Thread(target=write_after_signal)
        t2.start()
        t1.start()
        t1.join()
        t2.join()

        # No HB: read and write are unordered -> race
        self.assertRace()


# ===========================================================================
# DRB 028 — no — write after read with lock HB
# ===========================================================================


class TestDRB028_No_WriteAfterReadLock(DRBTestCase):
    def test_write_after_read_locked(self):
        class Box:
            val = 0

        b = Box()
        lock = object()
        published = threading.Event()
        key = self._key(b, "val")

        def reader():
            self.det.on_read(key)
            self.det.on_release(lock)
            published.set()

        def writer():
            published.wait()
            self.det.on_acquire(lock)
            self.det.on_write(key)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=writer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 029 — yes — nested lock with release missing on inner (not a deadlock, a race)
# ===========================================================================


class TestDRB029_Yes_NestedLockReleaseMissing(DRBTestCase):
    """
    Thread 1 holds two locks, releases outer but forgets inner.
    Thread 2 acquires inner lock; it never sees HB from thread 1's release.
    """

    def test_nested_lock_missing_release(self):
        lock_outer = object()
        lock_inner = object()
        done1 = threading.Event()
        key = ("global", "shared")

        def worker1():
            self.det.on_acquire(lock_outer)
            self.det.on_acquire(lock_inner)
            self.det.on_write(key)
            self.det.on_release(lock_outer)  # BUG: inner not released
            done1.set()

        def worker2():
            done1.wait()
            self.det.on_acquire(lock_inner)
            self.det.on_write(key)
            # release not needed for detection

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 030 — no — nested lock correctly released
# ===========================================================================


class TestDRB030_No_NestedLockCorrect(DRBTestCase):
    def test_nested_lock_correct(self):
        lock_outer = object()
        lock_inner = object()
        done1 = threading.Event()
        key = ("global", "shared")

        def worker1():
            self.det.on_acquire(lock_outer)
            self.det.on_acquire(lock_inner)
            self.det.on_write(key)
            self.det.on_release(lock_inner)
            self.det.on_release(lock_outer)
            done1.set()

        def worker2():
            done1.wait()
            self.det.on_acquire(lock_outer)
            self.det.on_acquire(lock_inner)
            self.det.on_read(key)
            self.det.on_release(lock_inner)
            self.det.on_release(lock_outer)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 031 — yes — race on a list element via different slices
# ===========================================================================


class TestDRB031_Yes_ListSliceAliasing(DRBTestCase):
    """
    Two threads access the same underlying memory through different
    slice views (simulated by same index). Race.
    """

    def test_list_slice_alias_race(self):
        lst = [0, 0]
        key0 = self._list_key(lst, 0)
        barrier = threading.Barrier(2)

        def worker_a():
            barrier.wait()
            self.det.on_write(key0)

        def worker_b():
            barrier.wait()
            # another reference to same element
            self.det.on_write(self._list_key(lst, 0))

        t1 = threading.Thread(target=worker_a)
        t2 = threading.Thread(target=worker_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 032 — no — list element protected by lock
# ===========================================================================


class TestDRB032_No_ListElementLocked(DRBTestCase):
    def test_list_element_locked(self):
        lst = [0, 0]
        lock = object()
        done1 = threading.Event()
        key0 = self._list_key(lst, 0)

        def worker1():
            self.det.on_acquire(lock)
            self.det.on_write(key0)
            self.det.on_release(lock)
            done1.set()

        def worker2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(key0)
            self.det.on_release(lock)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 033 — yes — concurrent writes to two fields that are aliased
# ===========================================================================


class TestDRB033_Yes_AliasedFieldWrites(DRBTestCase):
    def test_aliased_field_writes(self):
        class Rec:
            a = 0
            b = 0

        r = Rec()
        barrier = threading.Barrier(2)
        key_a = self._key(r, "a")

        # Thread 1 writes to r.a, Thread 2 also writes to the same field via alias
        alias = r

        def write_a():
            barrier.wait()
            self.det.on_write(key_a)

        def write_alias_a():
            barrier.wait()
            self.det.on_write(self._key(alias, "a"))

        t1 = threading.Thread(target=write_a)
        t2 = threading.Thread(target=write_alias_a)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 034 — no — field writes protected by a single lock
# ===========================================================================


class TestDRB034_No_LockedAliasedFields(DRBTestCase):
    def test_locked_aliased_fields(self):
        class Rec:
            a = 0
            b = 0

        r = Rec()
        lock = object()
        done1 = threading.Event()
        key_a = self._key(r, "a")

        def worker1():
            self.det.on_acquire(lock)
            self.det.on_write(key_a)
            self.det.on_release(lock)
            done1.set()

        def worker2():
            done1.wait()
            alias = r
            self.det.on_acquire(lock)
            self.det.on_write(self._key(alias, "a"))
            self.det.on_release(lock)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 035 — yes — race inside a parallel loop on a stack variable passed by ref
# ===========================================================================


class TestDRB035_Yes_LoopVarRace(DRBTestCase):
    def test_loop_var_race(self):
        class Acc:
            val = 0

        a = Acc()
        barrier = threading.Barrier(2)
        key = self._key(a, "val")

        def updater():
            barrier.wait()
            for _ in range(5):
                self.det.on_read(key)
                self.det.on_write(key)

        t1 = threading.Thread(target=updater)
        t2 = threading.Thread(target=updater)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 036 — no — same loop with critical section
# ===========================================================================


class TestDRB036_No_LoopVarLocked(DRBTestCase):
    def test_loop_var_locked(self):
        class Acc:
            val = 0

        a = Acc()
        lock = object()
        done1 = threading.Event()
        key = self._key(a, "val")

        def updater1():
            for _ in range(3):
                self.det.on_acquire(lock)
                self.det.on_read(key)
                self.det.on_write(key)
                self.det.on_release(lock)
            done1.set()

        def updater2():
            done1.wait()
            for _ in range(3):
                self.det.on_acquire(lock)
                self.det.on_read(key)
                self.det.on_write(key)
                self.det.on_release(lock)

        t1 = threading.Thread(target=updater1)
        t2 = threading.Thread(target=updater2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 037 — yes — race due to missing lock on initialisation check
# ===========================================================================


class TestDRB037_Yes_InitCheckMissingLock(DRBTestCase):
    def test_init_check_missing_lock(self):
        class Data:
            ptr = None

        d = Data()
        barrier = threading.Barrier(2)
        key_ptr = self._key(d, "ptr")
        key_val = ("global", "val")

        def init():
            barrier.wait()
            self.det.on_read(key_ptr)
            self.det.on_write(key_val)  # allocate
            self.det.on_write(key_ptr)  # publish pointer

        t1 = threading.Thread(target=init)
        t2 = threading.Thread(target=init)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 038 — no — proper lazy init with lock
# ===========================================================================


class TestDRB038_No_LazyInitLocked(DRBTestCase):
    def test_lazy_init_locked(self):
        class Data:
            ptr = None

        d = Data()
        lock = object()
        done1 = threading.Event()
        key_ptr = self._key(d, "ptr")
        key_val = ("global", "val")

        def init1():
            self.det.on_read(key_ptr)
            self.det.on_acquire(lock)
            self.det.on_write(key_val)
            self.det.on_write(key_ptr)
            self.det.on_release(lock)
            self.det.on_release(done1)  # model event.set() as HB release
            done1.set()

        def init2():
            done1.wait()
            self.det.on_acquire(done1)  # model event.wait() as HB acquire
            # HB established via event handoff: no race on key_ptr
            self.det.on_read(key_ptr)

        t1 = threading.Thread(target=init1)
        t2 = threading.Thread(target=init2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 039 — yes — two threads update adjacent list elements but one index races
# ===========================================================================


class TestDRB039_Yes_AdjacentIndexRace(DRBTestCase):
    def test_adjacent_index_race(self):
        arr = [0, 0, 0]
        barrier = threading.Barrier(2)
        key1 = self._list_key(arr, 1)

        def worker_a():
            barrier.wait()
            self.det.on_write(key1)  # writes index 1

        def worker_b():
            barrier.wait()
            self.det.on_write(self._list_key(arr, 1))  # same index -> race

        t1 = threading.Thread(target=worker_a)
        t2 = threading.Thread(target=worker_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 040 — no — adjacent indices properly separated
# ===========================================================================


class TestDRB040_No_AdjacentIndexSafe(DRBTestCase):
    def test_adjacent_index_safe(self):
        arr = [0, 0, 0]
        barrier = threading.Barrier(2)
        key0 = self._list_key(arr, 0)
        key2 = self._list_key(arr, 2)

        def worker_a():
            barrier.wait()
            self.det.on_write(key0)

        def worker_b():
            barrier.wait()
            self.det.on_write(key2)

        t1 = threading.Thread(target=worker_a)
        t2 = threading.Thread(target=worker_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 041 — yes — race on a boolean flag used as a spinlock
# ===========================================================================


class TestDRB041_Yes_FlagSpinRace(DRBTestCase):
    def test_flag_spin_race(self):
        class Flag:
            ready = False

        f = Flag()
        barrier = threading.Barrier(2)
        key = self._key(f, "ready")

        def setter():
            barrier.wait()
            self.det.on_write(key)

        def spin_reader():
            barrier.wait()
            # read in a loop (simulate spin)
            for _ in range(3):
                self.det.on_read(key)

        t1 = threading.Thread(target=setter)
        t2 = threading.Thread(target=spin_reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 042 — no — flag set with lock handoff to end spin
# ===========================================================================


class TestDRB042_No_FlagHandoffSpin(DRBTestCase):
    def test_flag_handoff_spin(self):
        class Flag:
            ready = False

        f = Flag()
        lock = object()
        published = threading.Event()
        key = self._key(f, "ready")

        def setter():
            self.det.on_write(key)  # set ready
            self.det.on_release(lock)  # HB to acquirer
            published.set()

        def reader():
            published.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_read(key)  # second read still no race

        t1 = threading.Thread(target=setter)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 043 — yes — indirect write through a pointer inside a struct
# ===========================================================================


class TestDRB043_Yes_IndirectThroughStruct(DRBTestCase):
    def test_indirect_through_struct(self):
        class Inner:
            val = 0

        class Outer:
            ptr = None

        inner = Inner()
        o = Outer()
        o.ptr = inner
        barrier = threading.Barrier(2)
        inner_key = self._key(inner, "val")

        def worker():
            barrier.wait()
            self.det.on_write(
                inner_key
            )  # both threads write inner.val via o.ptr

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 044 — no — same indirection with lock
# ===========================================================================


class TestDRB044_No_IndirectThroughStructLocked(DRBTestCase):
    def test_indirect_through_struct_locked(self):
        class Inner:
            val = 0

        class Outer:
            ptr = None

        inner = Inner()
        o = Outer()
        o.ptr = inner
        lock = object()
        done1 = threading.Event()
        inner_key = self._key(inner, "val")

        def worker1():
            self.det.on_acquire(lock)
            self.det.on_write(inner_key)
            self.det.on_release(lock)
            done1.set()

        def worker2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(self._key(o.ptr, "val"))
            self.det.on_release(lock)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 045 — yes — reduction on a shared array element without sync
# ===========================================================================


class TestDRB045_Yes_ArrayReductionNoSync(DRBTestCase):
    def test_array_reduction_no_sync(self):
        arr = [0]
        barrier = threading.Barrier(2)
        key = self._list_key(arr, 0)

        def add():
            barrier.wait()
            self.det.on_read(key)
            self.det.on_write(key)

        t1 = threading.Thread(target=add)
        t2 = threading.Thread(target=add)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 046 — no — reduction with lock
# ===========================================================================


class TestDRB046_No_ArrayReductionLocked(DRBTestCase):
    def test_array_reduction_locked(self):
        arr = [0]
        lock = object()
        done1 = threading.Event()
        key = self._list_key(arr, 0)

        def add1():
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def add2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_write(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=add1)
        t2 = threading.Thread(target=add2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 047 — yes — two writes to a global through function parameter
# ===========================================================================


class TestDRB047_Yes_FunctionParamAlias(DRBTestCase):
    def test_function_param_alias(self):
        class Glob:
            x = 0

        g = Glob()
        barrier = threading.Barrier(2)
        key = self._key(g, "x")

        def write_via_ref(obj):
            barrier.wait()
            self.det.on_write(self._key(obj, "x"))

        t1 = threading.Thread(target=write_via_ref, args=(g,))
        t2 = threading.Thread(target=write_via_ref, args=(g,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 048 — no — same function but lock inside
# ===========================================================================


class TestDRB048_No_FunctionParamLocked(DRBTestCase):
    def test_function_param_locked(self):
        class Glob:
            x = 0

        g = Glob()
        lock = object()
        done1 = threading.Event()
        key = self._key(g, "x")

        def write(obj):
            self.det.on_acquire(lock)
            self.det.on_write(self._key(obj, "x"))
            self.det.on_release(lock)

        def worker1():
            write(g)
            done1.set()

        def worker2():
            done1.wait()
            write(g)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 049 — yes — race on a member inside a list of structs
# ===========================================================================


class TestDRB049_Yes_ListOfStructsRace(DRBTestCase):
    def test_list_of_structs_race(self):
        class Elem:
            val = 0

        items = [Elem() for _ in range(2)]
        barrier = threading.Barrier(2)
        key = self._key(items[0], "val")

        def worker():
            barrier.wait()
            self.det.on_write(key)  # both write to same element[0].val

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 050 — no — list of structs, locked per element
# ===========================================================================


class TestDRB050_No_ListOfStructsLocked(DRBTestCase):
    def test_list_of_structs_locked(self):
        class Elem:
            def __init__(self):
                self.val = 0
                self.lock = object()

        items = [Elem() for _ in range(2)]
        done1 = threading.Event()
        key = self._key(items[0], "val")
        lock0 = items[0].lock

        def worker1():
            self.det.on_acquire(lock0)
            self.det.on_write(key)
            self.det.on_release(lock0)
            done1.set()

        def worker2():
            done1.wait()
            self.det.on_acquire(lock0)
            self.det.on_write(key)
            self.det.on_release(lock0)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 051 — yes — read-write race on two fields with no sync
# ===========================================================================


class TestDRB051_Yes_TwoFieldsRace(DRBTestCase):
    def test_two_fields_race(self):
        class S:
            x = 0
            y = 0

        s = S()
        barrier = threading.Barrier(2)
        key_x = self._key(s, "x")
        key_y = self._key(s, "y")

        def writer():
            barrier.wait()
            self.det.on_write(key_x)
            self.det.on_write(key_y)

        def reader():
            barrier.wait()
            self.det.on_read(key_x)
            self.det.on_read(key_y)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # race on x (write/read) and y (write/read), but at least one race
        self.assertRace()


# ===========================================================================
# DRB 052 — no — two fields protected by a single lock
# ===========================================================================


class TestDRB052_No_TwoFieldsLocked(DRBTestCase):
    def test_two_fields_locked(self):
        class S:
            x = 0
            y = 0

        s = S()
        lock = object()
        done1 = threading.Event()
        key_x = self._key(s, "x")
        key_y = self._key(s, "y")

        def worker1():
            self.det.on_acquire(lock)
            self.det.on_write(key_x)
            self.det.on_write(key_y)
            self.det.on_release(lock)
            done1.set()

        def worker2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key_x)
            self.det.on_read(key_y)
            self.det.on_release(lock)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 053 — yes — condition variable used without mutex (simulated)
# ===========================================================================


class TestDRB053_Yes_CondVarNoMutex(DRBTestCase):
    def test_condvar_no_mutex(self):
        flag = object()  # treat as shared condition variable
        data_key = ("global", "data")
        started = threading.Event()

        def producer():
            self.det.on_write(data_key)
            # Signal without lock (no release)
            started.set()

        def consumer():
            started.wait()
            # Acquire missing: just read
            self.det.on_read(data_key)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 054 — no — correct condition variable handoff (release/acquire)
# ===========================================================================


class TestDRB054_No_CondVarCorrect(DRBTestCase):
    def test_condvar_correct(self):
        lock = object()
        data_key = ("global", "data")
        published = threading.Event()

        def producer():
            self.det.on_write(data_key)
            self.det.on_release(lock)  # signal with HB
            published.set()

        def consumer():
            published.wait()
            self.det.on_acquire(lock)  # receive signal
            self.det.on_read(data_key)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 055 — yes — multiple writers on a shared counter, no barrier
# ===========================================================================


class TestDRB055_Yes_MultiWriterCounter(DRBTestCase):
    def test_multi_writer_counter(self):
        class Counter:
            val = 0

        c = Counter()
        key = self._key(c, "val")
        writers = 3
        barrier = threading.Barrier(writers)

        def writer():
            barrier.wait()
            self.det.on_read(key)
            self.det.on_write(key)

        threads = [threading.Thread(target=writer) for _ in range(writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertRace()


# ===========================================================================
# DRB 056 — no — multiple writers each with their own lock handoff
# ===========================================================================


class TestDRB056_No_MultiWriterHandoff(DRBTestCase):
    def test_multi_writer_handoff(self):
        class Counter:
            val = 0

        c = Counter()
        locks = [object()]
        events = [threading.Event()]
        key = self._key(c, "val")

        def writer(lock, done):
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_write(key)
            self.det.on_release(lock)
            done.set()

        t1 = threading.Thread(target=writer, args=(locks[0], events[0]))
        t1.start()
        events[0].wait()

        t2 = threading.Thread(
            target=writer, args=(locks[0], threading.Event())
        )
        t2.start()
        t2.join()
        t1.join()

        self.assertNoRace()


# ===========================================================================
# DRB 057 — yes — race on a simd-like vector element (same index)
# ===========================================================================


class TestDRB057_Yes_VectorElementRace(DRBTestCase):
    def test_vector_element_race(self):
        vector = [0.0, 0.0]
        barrier = threading.Barrier(2)
        key = self._list_key(vector, 0)

        def write_first():
            barrier.wait()
            self.det.on_write(key)

        t1 = threading.Thread(target=write_first)
        t2 = threading.Thread(target=write_first)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 058 — no — vector element protected
# ===========================================================================


class TestDRB058_No_VectorElementLocked(DRBTestCase):
    def test_vector_element_locked(self):
        vector = [0.0, 0.0]
        lock = object()
        done1 = threading.Event()
        key = self._list_key(vector, 0)

        def worker1():
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def worker2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 059 — yes — missing synchronisation after realloc (simulated write)
# ===========================================================================


class TestDRB059_Yes_ReallocRace(DRBTestCase):
    def test_realloc_race(self):
        class Holder:
            ptr = None

        h = Holder()
        barrier = threading.Barrier(2)
        key = self._key(h, "ptr")

        def realloc():
            barrier.wait()
            self.det.on_write(key)  # realloc overwrites pointer

        t1 = threading.Thread(target=realloc)
        t2 = threading.Thread(target=realloc)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 060 — no — realloc protected by global lock
# ===========================================================================


class TestDRB060_No_ReallocLocked(DRBTestCase):
    def test_realloc_locked(self):
        class Holder:
            ptr = None

        h = Holder()
        lock = object()
        done1 = threading.Event()
        key = self._key(h, "ptr")

        def realloc1():
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def realloc2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=realloc1)
        t2 = threading.Thread(target=realloc2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 061 — yes — missing lock on a node in a linked list
# ===========================================================================


class TestDRB061_Yes_LinkedListNodeRace(DRBTestCase):
    def test_linked_list_node_race(self):
        class Node:
            def __init__(self, val):
                self.val = val
                self.next = None

        head = Node(1)
        barrier = threading.Barrier(2)
        val_key = self._key(head, "val")

        def writer():
            barrier.wait()
            self.det.on_write(val_key)  # both modify head.val

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=writer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 062 — no — linked list node with lock
# ===========================================================================


class TestDRB062_No_LinkedListNodeLocked(DRBTestCase):
    def test_linked_list_node_locked(self):
        class Node:
            def __init__(self, val):
                self.val = val
                self.next = None
                self.lock = object()

        head = Node(1)
        done1 = threading.Event()
        val_key = self._key(head, "val")
        lock = head.lock

        def worker1():
            self.det.on_acquire(lock)
            self.det.on_write(val_key)
            self.det.on_release(lock)
            done1.set()

        def worker2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(val_key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 063 — yes — read/write race using a getter function (no lock)
# ===========================================================================


class TestDRB063_Yes_GetterFunctionRace(DRBTestCase):
    def test_getter_function_race(self):
        class Container:
            value = 42

        obj = Container()
        barrier = threading.Barrier(2)
        key = self._key(obj, "value")

        def getter():
            barrier.wait()
            self.det.on_read(key)

        def setter():
            barrier.wait()
            self.det.on_write(key)

        t1 = threading.Thread(target=getter)
        t2 = threading.Thread(target=setter)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 064 — no — getter/setter with lock
# ===========================================================================


class TestDRB064_No_GetterSetterLocked(DRBTestCase):
    def test_getter_setter_locked(self):
        class Container:
            value = 42

        obj = Container()
        lock = object()
        done1 = threading.Event()
        key = self._key(obj, "value")

        def setter():
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def getter():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=setter)
        t2 = threading.Thread(target=getter)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 065 — yes — same variable used for loop index and reduction (race)
# ===========================================================================


class TestDRB065_Yes_LoopIndexShared(DRBTestCase):
    def test_loop_index_shared(self):
        class LoopCtx:
            i = 0

        ctx = LoopCtx()
        barrier = threading.Barrier(2)
        key_i = self._key(ctx, "i")

        def work():
            barrier.wait()
            self.det.on_read(key_i)  # read i
            self.det.on_write(key_i)  # i++

        t1 = threading.Thread(target=work)
        t2 = threading.Thread(target=work)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 066 — no — each thread uses private index
# ===========================================================================


class TestDRB066_No_ThreadPrivateIndex(DRBTestCase):
    def test_thread_private_index(self):
        # each thread writes to its own key
        class TData:
            pass

        d1 = TData()
        d2 = TData()
        barrier = threading.Barrier(2)
        key1 = self._key(d1, "i")
        key2 = self._key(d2, "i")

        def worker(obj, key):
            barrier.wait()
            self.det.on_write(key)

        t1 = threading.Thread(target=worker, args=(d1, key1))
        t2 = threading.Thread(target=worker, args=(d2, key2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 067 — yes — array of structures: write to same member, different instances
# Actually safe but test that adjacent writes to same field of different elements are ok; we'll intentionally share index
# ===========================================================================


class TestDRB067_Yes_ArrayOfStructSameMemberRace(DRBTestCase):
    def test_aos_same_member_race(self):
        class Particle:
            x = 0.0

        particles = [Particle(), Particle()]
        barrier = threading.Barrier(2)
        key = self._key(particles[0], "x")  # both write to particles[0].x

        def worker():
            barrier.wait()
            self.det.on_write(key)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 068 — no — array of struct, locked access
# ===========================================================================


class TestDRB068_No_ArrayOfStructLocked(DRBTestCase):
    def test_aos_locked(self):
        class Particle:
            x = 0.0

        particles = [Particle(), Particle()]
        lock = object()
        done1 = threading.Event()
        key = self._key(particles[0], "x")

        def worker1():
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def worker2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 069 — yes — unprotected string append (different keys)
# ===========================================================================


class TestDRB069_Yes_StringAppendRace(DRBTestCase):
    def test_string_append_race(self):
        class Buffer:
            data = ""

        buf = Buffer()
        barrier = threading.Barrier(2)
        key = self._key(buf, "data")

        def append():
            barrier.wait()
            self.det.on_read(key)
            self.det.on_write(key)

        t1 = threading.Thread(target=append)
        t2 = threading.Thread(target=append)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 070 — no — string append with lock
# ===========================================================================


class TestDRB070_No_StringAppendLocked(DRBTestCase):
    def test_string_append_locked(self):
        class Buffer:
            data = ""

        buf = Buffer()
        lock = object()
        done1 = threading.Event()
        key = self._key(buf, "data")

        def append1():
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def append2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_write(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=append1)
        t2 = threading.Thread(target=append2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 071 — yes — race on a boolean shared via a singleton pattern
# ===========================================================================


class TestDRB071_Yes_SingletonRace(DRBTestCase):
    def test_singleton_race(self):
        class Singleton:
            instance = None

        s = Singleton()
        barrier = threading.Barrier(2)
        key = self._key(s, "instance")

        def get():
            barrier.wait()
            self.det.on_read(key)
            self.det.on_write(key)

        t1 = threading.Thread(target=get)
        t2 = threading.Thread(target=get)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 072 — no — singleton with DCL and lock (model correct)
# ===========================================================================


class TestDRB072_No_SingletonDCL(DRBTestCase):
    def test_singleton_dcl(self):
        class Singleton:
            instance = None

        s = Singleton()
        lock = object()
        done1 = threading.Event()
        key = self._key(s, "instance")

        def get1():
            self.det.on_read(key)
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)
            self.det.on_release(done1)  # model event.set() as HB release
            done1.set()

        def get2():
            done1.wait()
            self.det.on_acquire(done1)  # model event.wait() as HB acquire
            self.det.on_read(key)  # HB via event handoff: no race

        t1 = threading.Thread(target=get1)
        t2 = threading.Thread(target=get2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 073 — yes — race in simd reduction across threads
# ===========================================================================


class TestDRB073_Yes_SimdReductionRace(DRBTestCase):
    def test_simd_reduction_race(self):
        arr = [0.0] * 8
        barrier = threading.Barrier(2)
        key0 = self._list_key(arr, 0)

        def partial_reduce():
            barrier.wait()
            self.det.on_read(key0)
            self.det.on_write(key0)

        t1 = threading.Thread(target=partial_reduce)
        t2 = threading.Thread(target=partial_reduce)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 074 — no — reduction with omp critical (model lock)
# ===========================================================================


class TestDRB074_No_SimdReductionLocked(DRBTestCase):
    def test_simd_reduction_locked(self):
        arr = [0.0] * 8
        lock = object()
        done1 = threading.Event()
        key0 = self._list_key(arr, 0)

        def partial_reduce1():
            self.det.on_acquire(lock)
            self.det.on_read(key0)
            self.det.on_write(key0)
            self.det.on_release(lock)
            done1.set()

        def partial_reduce2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key0)
            self.det.on_write(key0)
            self.det.on_release(lock)

        t1 = threading.Thread(target=partial_reduce1)
        t2 = threading.Thread(target=partial_reduce2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 075 — yes — two threads compute into the same local variable but pointer aliased
# ===========================================================================


class TestDRB075_Yes_AliasedLocal(DRBTestCase):
    def test_aliased_local(self):
        var = [0]
        # simulate passing int* to both threads
        barrier = threading.Barrier(2)
        key = self._list_key(var, 0)

        def work():
            barrier.wait()
            self.det.on_write(key)

        t1 = threading.Thread(target=work)
        t2 = threading.Thread(target=work)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 076 — no — local variable copy per thread
# ===========================================================================


class TestDRB076_No_ThreadLocalCopy(DRBTestCase):
    def test_thread_local_copy(self):
        barrier = threading.Barrier(2)
        key1 = ("local", id(threading.current_thread()), "val")
        key2 = ("local", id(threading.current_thread()), "val")

        # each thread has its own key; we create objects on the fly
        def worker():
            private_obj = object()
            barrier.wait()
            self.det.on_write(self._key(private_obj, "val"))

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 077 — yes — write to a file descriptor (proxy) without lock
# ===========================================================================


class TestDRB077_Yes_FileWriteRace(DRBTestCase):
    def test_file_write_race(self):
        class File:
            fd = object()

        f = File()
        barrier = threading.Barrier(2)
        key = self._key(f, "fd")

        def write():
            barrier.wait()
            self.det.on_write(key)

        t1 = threading.Thread(target=write)
        t2 = threading.Thread(target=write)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 078 — no — file write with lock
# ===========================================================================


class TestDRB078_No_FileWriteLocked(DRBTestCase):
    def test_file_write_locked(self):
        class File:
            fd = object()

        f = File()
        lock = object()
        done1 = threading.Event()
        key = self._key(f, "fd")

        def write1():
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def write2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=write1)
        t2 = threading.Thread(target=write2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 079 — yes — pointer swap without atomic
# ===========================================================================


class TestDRB079_Yes_PointerSwapRace(DRBTestCase):
    def test_pointer_swap_race(self):
        class Node:
            next = None

        n1 = Node()
        n2 = Node()
        barrier = threading.Barrier(2)
        key = self._key(n1, "next")

        def swapper():
            barrier.wait()
            # read then write next pointer
            self.det.on_read(key)
            self.det.on_write(key)

        t1 = threading.Thread(target=swapper)
        t2 = threading.Thread(target=swapper)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 080 — no — pointer swap with mutex
# ===========================================================================


class TestDRB080_No_PointerSwapLocked(DRBTestCase):
    def test_pointer_swap_locked(self):
        class Node:
            next = None

        n1 = Node()
        lock = object()
        done1 = threading.Event()
        key = self._key(n1, "next")

        def swapper1():
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def swapper2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_write(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=swapper1)
        t2 = threading.Thread(target=swapper2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 081 — yes — race on a hash table entry during resize
# ===========================================================================


class TestDRB081_Yes_HashTableRace(DRBTestCase):
    def test_hash_table_race(self):
        class Entry:
            key = 0
            value = None

        e = Entry()
        barrier = threading.Barrier(2)
        val_key = self._key(e, "value")

        def rehash():
            barrier.wait()
            self.det.on_write(val_key)

        t1 = threading.Thread(target=rehash)
        t2 = threading.Thread(target=rehash)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 082 — no — hash table entry locked
# ===========================================================================


class TestDRB082_No_HashTableLocked(DRBTestCase):
    def test_hash_table_locked(self):
        class Entry:
            key = 0
            value = None

        e = Entry()
        lock = object()
        done1 = threading.Event()
        val_key = self._key(e, "value")

        def writer1():
            self.det.on_acquire(lock)
            self.det.on_write(val_key)
            self.det.on_release(lock)
            done1.set()

        def writer2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(val_key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=writer1)
        t2 = threading.Thread(target=writer2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 083 — yes — race using a global flag to exit a loop
# ===========================================================================


class TestDRB083_Yes_GlobalFlagLoopRace(DRBTestCase):
    def test_global_flag_loop_race(self):
        class Control:
            stop = False

        ctrl = Control()
        barrier = threading.Barrier(2)
        key = self._key(ctrl, "stop")

        def looper():
            barrier.wait()
            for _ in range(3):
                self.det.on_read(key)  # checking stop

        def stopper():
            barrier.wait()
            self.det.on_write(key)  # set stop

        t1 = threading.Thread(target=looper)
        t2 = threading.Thread(target=stopper)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 084 — no — flag with lock handoff to stop loop
# ===========================================================================


class TestDRB084_No_GlobalFlagLocked(DRBTestCase):
    def test_global_flag_locked(self):
        class Control:
            stop = False

        ctrl = Control()
        lock = object()
        published = threading.Event()
        key = self._key(ctrl, "stop")

        def stopper():
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)
            published.set()

        def looper():
            published.wait()
            # just read after acquire to establish HB
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_read(key)  # loop check
            self.det.on_release(lock)

        t1 = threading.Thread(target=stopper)
        t2 = threading.Thread(target=looper)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 085 — yes — unprotected write to a shared object's member from task
# ===========================================================================


class TestDRB085_Yes_TaskMemberRace(DRBTestCase):
    def test_task_member_race(self):
        class Shared:
            res = 0

        s = Shared()
        barrier = threading.Barrier(2)
        key = self._key(s, "res")

        def task():
            barrier.wait()
            self.det.on_write(key)

        t1 = threading.Thread(target=task)
        t2 = threading.Thread(target=task)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 086 — no — task with private taskgroup lock
# ===========================================================================


class TestDRB086_No_TaskMemberLocked(DRBTestCase):
    def test_task_member_locked(self):
        class Shared:
            res = 0

        s = Shared()
        lock = object()
        done1 = threading.Event()
        key = self._key(s, "res")

        def task1():
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def task2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=task1)
        t2 = threading.Thread(target=task2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 087 — yes — two tasks writing to same list element in a parallel region
# ===========================================================================


class TestDRB087_Yes_ParallelRegionListRace(DRBTestCase):
    def test_parallel_region_list_race(self):
        lst = [0, 0, 0]
        barrier = threading.Barrier(2)
        key1 = self._list_key(lst, 1)

        def worker():
            barrier.wait()
            self.det.on_write(key1)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 088 — no — parallel region with critical
# ===========================================================================


class TestDRB088_No_ParallelRegionCritical(DRBTestCase):
    def test_parallel_region_critical(self):
        lst = [0, 0, 0]
        lock = object()
        done1 = threading.Event()
        key1 = self._list_key(lst, 1)

        def worker1():
            self.det.on_acquire(lock)
            self.det.on_write(key1)
            self.det.on_release(lock)
            done1.set()

        def worker2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(key1)
            self.det.on_release(lock)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 089 — yes — race due to missing barrier in work-sharing loop
# ===========================================================================


class TestDRB089_Yes_WorkSharingLoopRace(DRBTestCase):
    def test_work_sharing_race(self):
        class Shared:
            sum = 0

        s = Shared()
        barrier = threading.Barrier(2)
        key = self._key(s, "sum")

        def worker():
            barrier.wait()
            self.det.on_read(key)
            self.det.on_write(key)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 090 — no — work-sharing loop with reduction lock
# ===========================================================================


class TestDRB090_No_WorkSharingLocked(DRBTestCase):
    def test_work_sharing_locked(self):
        class Shared:
            sum = 0

        s = Shared()
        lock = object()
        done1 = threading.Event()
        key = self._key(s, "sum")

        def worker1():
            for _ in range(2):
                self.det.on_acquire(lock)
                self.det.on_read(key)
                self.det.on_write(key)
                self.det.on_release(lock)
            done1.set()

        def worker2():
            done1.wait()
            for _ in range(2):
                self.det.on_acquire(lock)
                self.det.on_read(key)
                self.det.on_write(key)
                self.det.on_release(lock)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 091 — yes — race on a shared cache line (two variables adjacent)
# ===========================================================================


class TestDRB091_Yes_AdjacentVariableRace(DRBTestCase):
    def test_adjacent_variable_race(self):
        class CacheLine:
            a = 0
            b = 0

        cl = CacheLine()
        barrier = threading.Barrier(2)
        key_a = self._key(cl, "a")

        def writer_a():
            barrier.wait()
            self.det.on_write(key_a)

        t1 = threading.Thread(target=writer_a)
        t2 = threading.Thread(target=writer_a)  # same field, not b
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 092 — no — adjacent variable but only one writer per field
# ===========================================================================


class TestDRB092_No_AdjacentVariableSafe(DRBTestCase):
    def test_adjacent_variable_safe(self):
        class CacheLine:
            a = 0
            b = 0

        cl = CacheLine()
        barrier = threading.Barrier(2)
        key_a = self._key(cl, "a")
        key_b = self._key(cl, "b")

        def writer_a():
            barrier.wait()
            self.det.on_write(key_a)

        def writer_b():
            barrier.wait()
            self.det.on_write(key_b)

        t1 = threading.Thread(target=writer_a)
        t2 = threading.Thread(target=writer_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 093 — yes — race on a pointer used for resource management
# ===========================================================================


class TestDRB093_Yes_ResourcePointerRace(DRBTestCase):
    def test_resource_pointer_race(self):
        class Resource:
            handle = None

        r = Resource()
        barrier = threading.Barrier(2)
        key = self._key(r, "handle")

        def alloc():
            barrier.wait()
            self.det.on_write(key)

        t1 = threading.Thread(target=alloc)
        t2 = threading.Thread(target=alloc)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 094 — no — resource pointer allocation with lock
# ===========================================================================


class TestDRB094_No_ResourcePointerLocked(DRBTestCase):
    def test_resource_pointer_locked(self):
        class Resource:
            handle = None

        r = Resource()
        lock = object()
        done1 = threading.Event()
        key = self._key(r, "handle")

        def alloc1():
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)
            done1.set()

        def alloc2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=alloc1)
        t2 = threading.Thread(target=alloc2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 095 — yes — race in parallel histogram fill
# ===========================================================================


class TestDRB095_Yes_HistogramRace(DRBTestCase):
    def test_histogram_race(self):
        bins = [0] * 4
        barrier = threading.Barrier(2)
        key_bin2 = self._list_key(bins, 2)

        def fill():
            barrier.wait()
            self.det.on_read(key_bin2)
            self.det.on_write(key_bin2)

        t1 = threading.Thread(target=fill)
        t2 = threading.Thread(target=fill)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 096 — no — histogram fill with atomic add (lock)
# ===========================================================================


class TestDRB096_No_HistogramLocked(DRBTestCase):
    def test_histogram_locked(self):
        bins = [0] * 4
        lock = object()
        done1 = threading.Event()
        key_bin2 = self._list_key(bins, 2)

        def fill1():
            self.det.on_acquire(lock)
            self.det.on_read(key_bin2)
            self.det.on_write(key_bin2)
            self.det.on_release(lock)
            done1.set()

        def fill2():
            done1.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key_bin2)
            self.det.on_write(key_bin2)
            self.det.on_release(lock)

        t1 = threading.Thread(target=fill1)
        t2 = threading.Thread(target=fill2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 097 — yes — race between a deallocate and a read (use-after-free)
# ===========================================================================


class TestDRB097_Yes_UseAfterFreeRace(DRBTestCase):
    def test_use_after_free_race(self):
        class Ptr:
            address = 123456

        ptr = Ptr()
        barrier = threading.Barrier(2)
        key = self._key(ptr, "address")

        def free():
            barrier.wait()
            self.det.on_write(key)  # free ptr

        def use():
            barrier.wait()
            self.det.on_read(key)  # use ptr

        t1 = threading.Thread(target=free)
        t2 = threading.Thread(target=use)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 098 — no — use-after-free prevented by lock
# ===========================================================================


class TestDRB098_No_UseAfterFreeLocked(DRBTestCase):
    def test_use_after_free_locked(self):
        class Ptr:
            address = 123456

        ptr = Ptr()
        lock = object()
        done = threading.Event()
        key = self._key(ptr, "address")

        def free():
            self.det.on_acquire(lock)
            self.det.on_write(key)
            self.det.on_release(lock)
            done.set()

        def use():
            done.wait()
            self.det.on_acquire(lock)
            self.det.on_read(key)
            self.det.on_release(lock)

        t1 = threading.Thread(target=free)
        t2 = threading.Thread(target=use)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertNoRace()


# ===========================================================================
# DRB 099 — yes — race on a flag set in a signal handler (simulated)
# ===========================================================================


class TestDRB099_Yes_SignalHandlerRace(DRBTestCase):
    """
    Simulated signal-handler race: one thread writes a flag asynchronously
    while another reads it in a polling loop — no synchronisation.
    """

    def test_signal_handler_race(self):
        class Flag:
            signalled = False

        f = Flag()
        barrier = threading.Barrier(2)
        key = self._key(f, "signalled")

        def handler():
            barrier.wait()
            self.det.on_write(key)  # async "signal" write

        def wait_loop():
            barrier.wait()
            self.det.on_read(key)  # polling read

        t1 = threading.Thread(target=handler)
        t2 = threading.Thread(target=wait_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertRace()


# ===========================================================================
# DRB 100 — no — flag set before thread start (proper publish)
# Writer sets the flag in the main thread before spawning the reader.
# The fork HB edge (thread_start) orders the write before the read.
# ===========================================================================


class TestDRB100_No_PublishBeforeStart(DRBTestCase):
    """
    Main thread writes a flag, then starts a reader thread.
    The fork HB edge (thread_start) orders write before read — no race.
    """

    def test_publish_before_start(self):
        class Flag:
            ready = False

        f = Flag()
        key = self._key(f, "ready")

        # Main thread writes before spawning the reader
        self.det.on_write(key)

        def reader():
            self.det.on_read(key)

        t = threading.Thread(target=reader)
        t.start()
        t.join()

        self.assertNoRace()


if __name__ == "__main__":
    import unittest

    unittest.main()
