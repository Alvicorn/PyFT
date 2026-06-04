"""
Tests that the VFTVersion selection wires through to the actual
VarState instances created by the Engine's ShadowMap.
"""

import pytest

from pyvft.core.var_state import VarStateV1, VarStateV2, VFTVersion
from pyvft.detector.engine import Engine


class _Obj:
    pass


class TestEngineVersionSelection:
    def test_default_is_v2(self) -> None:
        engine = Engine()
        assert engine.version is VFTVersion.V2
        obj = _Obj()
        vs = engine.shadow_map.get_or_create(obj, "x")
        assert isinstance(vs, VarStateV2)

    def test_explicit_v2_string(self) -> None:
        engine = Engine(version="v2")
        assert engine.version is VFTVersion.V2
        vs = engine.shadow_map.get_or_create(_Obj(), "x")
        assert isinstance(vs, VarStateV2)

    def test_explicit_v2_enum(self) -> None:
        engine = Engine(version=VFTVersion.V2)
        vs = engine.shadow_map.get_or_create(_Obj(), "x")
        assert isinstance(vs, VarStateV2)

    def test_explicit_v1_string(self) -> None:
        engine = Engine(version="v1")
        assert engine.version is VFTVersion.V1
        vs = engine.shadow_map.get_or_create(_Obj(), "x")
        assert isinstance(vs, VarStateV1)

    def test_explicit_v1_enum(self) -> None:
        engine = Engine(version=VFTVersion.V1)
        vs = engine.shadow_map.get_or_create(_Obj(), "x")
        assert isinstance(vs, VarStateV1)

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError):
            Engine(version="v99")

    def test_both_versions_detect_a_simple_ww_race(self) -> None:
        """
        End-to-end: feed engine.write events from two threads with
        no HB and verify both versions report a race.
        """
        for v in ("v1", "v2"):
            engine = Engine(version=v)
            obj = _Obj()
            engine.write(obj, "x")
            # Simulate a different thread by directly registering a
            # second ThreadState — engine.write uses the current OS
            # thread, so we instead poke the shadow map directly.
            vs = engine.shadow_map.get_or_create(obj, "x")
            from pyvft.core.vector_clock import VectorClock

            # T2 writes without any HB to T1.
            _, write_race, _, _ = vs.check_write(2, VectorClock({2: 1}))
            assert write_race, f"expected WW race under version={v}"


class TestPyVFTInstallVersion:
    def test_install_with_v1(self) -> None:
        import pyvft

        pyvft.install(version="v1")
        try:
            engine = pyvft.get_engine()
            assert engine is not None
            assert engine.version is VFTVersion.V1
        finally:
            pyvft.uninstall()

    def test_install_with_v2_default(self) -> None:
        import pyvft

        pyvft.install()
        try:
            engine = pyvft.get_engine()
            assert engine is not None
            assert engine.version is VFTVersion.V2
        finally:
            pyvft.uninstall()

    def test_context_with_v1(self) -> None:
        import pyvft

        with pyvft.context(version="v1"):
            engine = pyvft.get_engine()
            assert engine is not None
            assert engine.version is VFTVersion.V1

    def test_detect_with_v1(self) -> None:
        import pyvft

        captured = {}

        @pyvft.detect(version="v1")
        def f() -> None:
            captured["engine"] = pyvft.get_engine()

        f()
        eng = captured["engine"]
        assert eng is not None
        assert eng.version is VFTVersion.V1

    def test_detect_bare_decorator_defaults_to_v2(self) -> None:
        import pyvft

        captured = {}

        @pyvft.detect
        def f() -> None:
            captured["engine"] = pyvft.get_engine()

        f()
        eng = captured["engine"]
        assert eng is not None
        assert eng.version is VFTVersion.V2
