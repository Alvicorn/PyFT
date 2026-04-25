from pyft.core.epoch import _NONE_TID, Epoch, current_epoch


class TestEpoch:
    def test_bottom(self) -> None:
        e = Epoch.bottom()
        assert e.tid == _NONE_TID
        assert e.clock == 0

        assert e == Epoch.bottom()
        assert e == Epoch(_NONE_TID, 0)

    def test_is_bottom(self) -> None:
        e = Epoch.bottom()
        assert e.is_bottom()
        assert not Epoch(1, 1).is_bottom()

    def test_of_factory(self) -> None:
        e = Epoch.of(10, 2)
        assert e.tid == 10
        assert e.clock == 2

    def test_equality(self) -> None:
        assert Epoch(1, 3) == Epoch(1, 3)
        assert Epoch(2, 3) != Epoch(1, 3)
        assert Epoch(1, 2) != Epoch(1, 3)

    def test_repr(self):
        assert "⊥" in repr(Epoch.bottom())

        r = repr(Epoch(5, 10))
        assert "5" in r
        assert "10" in r


class TestCurrentEpoch:
    def test_returns_calling_thread_epoch(self) -> None:
        import threading

        clocks = {threading.get_ident(): 3}
        e = current_epoch(clocks)

        assert e.tid == threading.get_ident()
        assert e.clock == 3

    def test_missing_thread_defaults_to_zero(self) -> None:
        e = current_epoch({})
        assert e.clock == 0

    def test_different_threads_have_different_epochs(self) -> None:
        import threading

        results = {}

        def record():
            tid = threading.get_ident()
            results[tid] = current_epoch({tid: 5})

        t = threading.Thread(target=record)
        t.start()
        t.join()

        main_tid = threading.get_ident()
        child_tid = next(k for k in results if k != main_tid)

        assert results[child_tid].tid == child_tid
        assert results[child_tid].clock == 5
