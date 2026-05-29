import threading

from pyft.detector.engine import Engine
from pyft.instrument.wrappers import (
    AutoTracker,
    TrackedProxy,
    make_tracked_class,
)


class Counter:
    def __init__(self, value: int = 0) -> None:
        self.value = value


class TestTrackedProxy:
    def test_proxy_forwards_getattr(self) -> None:
        engine = Engine()
        c = Counter(42)
        proxy = TrackedProxy(c, engine)
        assert proxy.value == 42

    def test_proxy_forwards_setattr(self) -> None:
        engine = Engine()
        c = Counter(0)
        proxy = TrackedProxy(c, engine)
        proxy.value = 99
        assert c.value == 99

    def test_read_calls_engine_read(self) -> None:
        engine = Engine()
        calls = []
        original_read = engine.read
        engine.read = lambda obj, attr: (
            calls.append(("read", attr)) or original_read(obj, attr)
        )

        c = Counter(1)
        proxy = TrackedProxy(c, engine)
        _ = proxy.value

        assert any(call[1] == "value" for call in calls if call[0] == "read")

    def test_write_calls_engine_write(self) -> None:
        engine = Engine()
        calls = []
        original_write = engine.write
        engine.write = lambda obj, attr: (
            calls.append(("write", attr)) or original_write(obj, attr)
        )

        c = Counter(1)
        proxy = TrackedProxy(c, engine)
        proxy.value = 5

        assert any(call[1] == "value" for call in calls if call[0] == "write")

    def test_internal_attrs_not_tracked(self) -> None:
        """__pyft_wrapped__ etc. should not trigger engine callbacks."""
        engine = Engine()
        reads = []
        engine.read = lambda obj, attr: reads.append(attr)

        c = Counter()
        proxy = TrackedProxy(c, engine)
        _ = proxy.__pyft_wrapped__

        assert "__pyft_wrapped__" not in reads

    def test_repr_includes_wrapped_repr(self) -> None:
        engine = Engine()
        c = Counter(7)
        proxy = TrackedProxy(c, engine)
        assert "TrackedProxy" in repr(proxy)

    def test_proxy_is_transparent_for_method_calls(self) -> None:
        engine = Engine()

        class Adder:
            def __init__(self) -> None:
                self.x = 0

            def add(self, n: int) -> None:
                self.x += n

        obj = Adder()
        proxy = TrackedProxy(obj, engine)
        proxy.add(5)
        assert obj.x == 5


class TestMakeTrackedClass:
    def test_tracked_class_calls_engine_on_setattr(self) -> None:
        engine = Engine()
        writes = []
        engine.write = lambda obj, attr: writes.append(attr)

        TrackedCounter = make_tracked_class(Counter, engine)
        c = TrackedCounter(0)
        c.value = 10

        assert "value" in writes

    def test_tracked_class_calls_engine_on_getattr(self) -> None:
        engine = Engine()
        reads = []
        original_read = engine.read
        engine.read = lambda obj, attr: (
            reads.append(attr) or original_read(obj, attr)
        )

        TrackedCounter = make_tracked_class(Counter, engine)
        c = TrackedCounter(5)
        _ = c.value

        assert "value" in reads

    def test_tracked_class_is_subclass(self) -> None:
        engine = Engine()
        TrackedCounter = make_tracked_class(Counter, engine)
        c = TrackedCounter()
        assert isinstance(c, Counter)

    def test_tracked_class_name_includes_original(self) -> None:
        engine = Engine()
        TrackedCounter = make_tracked_class(Counter, engine)
        assert "Counter" in TrackedCounter.__name__


class TestAutoTracker:
    def test_install_patches_thread(self) -> None:
        engine = Engine()
        tracker = AutoTracker(engine)
        original_start = threading.Thread.start
        tracker.install()
        assert threading.Thread.start is not original_start
        tracker.uninstall()
        assert threading.Thread.start is original_start

    def test_uninstall_restores_thread(self) -> None:
        engine = Engine()
        tracker = AutoTracker(engine)
        tracker.install()
        tracker.uninstall()
        # Should be fully restored — test that a normal thread still works
        result = []
        t = threading.Thread(target=lambda: result.append(1))
        t.start()
        t.join()
        assert result == [1]

    def test_thread_registered_on_start(self) -> None:
        engine = Engine()
        tracker = AutoTracker(engine)
        tracker.install()

        registered_tids = []

        def worker() -> None:
            # Engine uses id(threading.current_thread()) as the thread key
            tid = id(threading.current_thread())
            ts = engine.thread_registry.get(tid)
            if ts:
                registered_tids.append(tid)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        tracker.uninstall()
        assert len(registered_tids) == 1
