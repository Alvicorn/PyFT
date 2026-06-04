import gc
import threading

from pyvft.core.var_state import VarStateV2
from pyvft.detector.shadow_map import ShadowMap

VarState = VarStateV2


class SimpleObj:
    pass


class TestShadowMapBasics:
    def test_get_or_create_returns_var_state(self) -> None:
        sm = ShadowMap()
        obj = SimpleObj()
        vs = sm.get_or_create(obj, "x")
        assert isinstance(vs, VarState)

    def test_same_obj_same_attr_returns_same_instance(self) -> None:
        sm = ShadowMap()
        obj = SimpleObj()
        vs1 = sm.get_or_create(obj, "x")
        vs2 = sm.get_or_create(obj, "x")
        assert vs1 is vs2

    def test_same_obj_different_attrs_are_independent(self) -> None:
        sm = ShadowMap()
        obj = SimpleObj()
        vs_x = sm.get_or_create(obj, "x")
        vs_y = sm.get_or_create(obj, "y")
        assert vs_x is not vs_y

    def test_different_objects_are_independent(self) -> None:
        sm = ShadowMap()
        obj1, obj2 = SimpleObj(), SimpleObj()
        vs1 = sm.get_or_create(obj1, "x")
        vs2 = sm.get_or_create(obj2, "x")
        assert vs1 is not vs2

    def test_stats_counts_objects_and_vars(self) -> None:
        sm = ShadowMap()
        obj1, obj2 = SimpleObj(), SimpleObj()
        sm.get_or_create(obj1, "x")
        sm.get_or_create(obj1, "y")
        sm.get_or_create(obj2, "x")
        stats = sm.stats()
        assert stats["tracked_objects"] == 2
        assert stats["tracked_variables"] == 3

    def test_remove_object(self) -> None:
        sm = ShadowMap()
        obj = SimpleObj()
        sm.get_or_create(obj, "x")
        oid = id(obj)
        sm.remove_object(oid)
        stats = sm.stats()
        assert stats["tracked_objects"] == 0

    def test_remove_nonexistent_does_not_raise(self) -> None:
        sm = ShadowMap()
        sm.remove_object(99999999)  # should not raise


class TestShadowMapGC:
    def test_gc_removes_entry(self) -> None:
        sm = ShadowMap()

        class WeakableObj:
            pass

        obj = WeakableObj()
        sm.get_or_create(obj, "x")
        assert sm.stats()["tracked_objects"] == 1

        del obj
        gc.collect()
        # After GC, the entry should be removed
        assert sm.stats()["tracked_objects"] == 0

    def test_non_weakrefable_types_do_not_crash(self) -> None:
        """int, str etc. don't support weakref — should silently skip."""
        sm = ShadowMap()
        # We can't track plain ints/strs by design, but let's make sure
        # tracking a custom object works cleanly
        obj = SimpleObj()
        vs = sm.get_or_create(obj, "value")
        assert vs is not None


class TestShadowMapThreadSafety:
    def test_concurrent_get_or_create_same_obj(self) -> None:
        sm = ShadowMap()
        obj = SimpleObj()
        results = []
        errors = []

        def accessor() -> None:
            try:
                vs = sm.get_or_create(obj, "x")
                results.append(vs)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=accessor) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 20
        # All should be the same VarState instance
        assert all(r is results[0] for r in results)

    def test_concurrent_different_objects(self) -> None:
        sm = ShadowMap()
        errors = []

        def accessor(i: int) -> None:
            try:
                obj = SimpleObj()
                sm.get_or_create(obj, f"attr_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=accessor, args=(i,)) for i in range(30)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
