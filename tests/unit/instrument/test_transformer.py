"""
Tests for the AST-rewriting AccessTracer.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path
from types import CodeType

from pyft.detector.engine import Engine
from pyft.instrument.import_hook import (
    _RUNTIME_MODULE_NAME,
    ImportHook,
    _make_runtime_module,
    transform_source,
)
from pyft.instrument.transformer import AccessTracer, should_skip_module


class TestShouldSkipModule:
    def test_pyft_skipped(self) -> None:
        assert should_skip_module("pyft")
        assert should_skip_module("pyft.core.var_state")

    def test_threading_skipped(self) -> None:
        assert should_skip_module("threading")
        assert should_skip_module("threading.foo")

    def test_user_module_not_skipped(self) -> None:
        assert not should_skip_module("myapp")
        assert not should_skip_module("myapp.module")

    def test_substring_match_not_enough(self) -> None:
        """'threadingfoo' should not be skipped (must be 'threading' or
        'threading.something')."""
        assert not should_skip_module("threadingfoo")


class TestTransformSource:
    def _compile(self, src: str) -> tuple[ast.Module, CodeType]:
        tree = transform_source(textwrap.dedent(src), "<test>")
        return tree, compile(tree, "<test>", "exec")

    def test_attribute_read_becomes_helper_call(self) -> None:
        tree, _ = self._compile(
            """
            def f(obj):
                return obj.attr
            """
        )
        # find the function body, then the return statement
        fn = tree.body[1]  # body[0] is the runtime import
        assert isinstance(fn, ast.FunctionDef)
        ret = fn.body[0]
        assert isinstance(ret, ast.Return)
        assert isinstance(ret.value, ast.Call)
        assert ret.value.func.id == "__pyft_get__"

    def test_attribute_write_becomes_helper_call(self) -> None:
        tree, _ = self._compile(
            """
            def f(obj):
                obj.x = 1
            """
        )
        fn = tree.body[1]
        stmt = fn.body[0]
        assert isinstance(stmt, ast.Expr)
        assert isinstance(stmt.value, ast.Call)
        assert stmt.value.func.id == "__pyft_set__"

    def test_subscript_read(self) -> None:
        tree, _ = self._compile(
            """
            def f(d, k):
                return d[k]
            """
        )
        fn = tree.body[1]
        ret = fn.body[0]
        assert isinstance(ret.value, ast.Call)
        assert ret.value.func.id == "__pyft_getitem__"

    def test_subscript_write(self) -> None:
        tree, _ = self._compile(
            """
            def f(d, k, v):
                d[k] = v
            """
        )
        fn = tree.body[1]
        stmt = fn.body[0]
        assert isinstance(stmt, ast.Expr)
        assert stmt.value.func.id == "__pyft_setitem__"

    def test_augmented_assignment_does_both(self) -> None:
        tree, _ = self._compile(
            """
            def f(obj):
                obj.count += 1
            """
        )
        fn = tree.body[1]
        stmt = fn.body[0]
        assert isinstance(stmt, ast.Expr)
        # outer is __pyft_set__, inner has __pyft_get__
        assert stmt.value.func.id == "__pyft_set__"
        # third arg is the BinOp whose left is __pyft_get__ call
        bin_op = stmt.value.args[3]
        assert isinstance(bin_op, ast.BinOp)
        assert isinstance(bin_op.left, ast.Call)
        assert bin_op.left.func.id == "__pyft_get__"

    def test_local_variable_not_transformed(self) -> None:
        tree, _ = self._compile(
            """
            def f():
                x = 1
                return x
            """
        )
        fn = tree.body[1]
        # neither the assign nor the return should be helper calls
        assign, ret = fn.body
        assert isinstance(assign, ast.Assign)
        assert isinstance(ret.value, ast.Name)
        assert ret.value.id == "x"


class TestRuntimeModule:
    def test_get_calls_engine_read_and_returns_value(self) -> None:
        engine = Engine()
        reads = []
        engine.read = lambda obj, attr: reads.append(attr)
        rt = _make_runtime_module(engine)

        class C:
            x = 7

        obj = C()
        result = rt._pyft_get(engine, obj, "x")
        assert result == 7
        assert reads == ["x"]

    def test_set_calls_engine_write_and_sets_attr(self) -> None:
        engine = Engine()
        writes = []
        engine.write = lambda obj, attr: writes.append(attr)
        rt = _make_runtime_module(engine)

        class C:
            pass

        obj = C()
        rt._pyft_set(engine, obj, "y", 99)
        assert obj.y == 99
        assert writes == ["y"]

    def test_get_swallows_engine_errors(self) -> None:
        """The runtime helpers must never break user code if the engine
        raises."""
        engine = Engine()

        def boom(*a: object) -> None:
            raise RuntimeError("engine failure")

        engine.read = boom
        rt = _make_runtime_module(engine)

        class C:
            v = 42

        # must still return the attribute even though engine.read raised
        assert rt._pyft_get(engine, C(), "v") == 42


class TestImportHookEndToEnd:
    """
    Drop a fixture .py file into sys.path, install the hook, import,
    and check the engine actually saw reads/writes.
    """

    def _setup(self, tmp_path: Path) -> Path:
        src = textwrap.dedent(
            """
            class Counter:
                def __init__(self):
                    self.value = 0

                def bump(self):
                    self.value += 1
                    return self.value
            """
        )
        fixture = tmp_path / "pyft_fixture_module.py"
        fixture.write_text(src)
        sys.path.insert(0, str(tmp_path))
        return fixture

    def _teardown(self, tmp_path: Path) -> None:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("pyft_fixture_module", None)
        sys.modules.pop(_RUNTIME_MODULE_NAME, None)

    def test_import_hook_traces_attribute_access(self, tmp_path: str) -> None:
        self._setup(tmp_path)
        try:
            engine = Engine()
            reads = []
            writes = []
            engine.read = lambda obj, attr: reads.append(attr)
            engine.write = lambda obj, attr: writes.append(attr)
            hook = ImportHook(engine)
            hook.install()
            try:
                import pyft_fixture_module as m

                c = m.Counter()  # __init__ does self.value = 0 → 1 write
                assert "value" in writes
                writes.clear()
                reads.clear()
                v = c.bump()  # reads value, writes value
                assert v == 1
                assert "value" in reads
                assert "value" in writes
            finally:
                hook.uninstall()
        finally:
            self._teardown(tmp_path)

    def test_skipped_modules_are_not_instrumented(self, tmp_path: str) -> None:
        """Re-importing threading after install must not break it."""
        engine = Engine()
        hook = ImportHook(engine)
        hook.install()
        try:
            import threading

            # construct + use a basic lock — would crash if instrumented
            # incorrectly
            ll = threading.Lock()
            with ll:
                pass
        finally:
            hook.uninstall()


class TestAccessTracerWrapper:
    def test_install_uninstall(self) -> None:
        engine = Engine()
        t = AccessTracer(engine)
        t.install()
        assert t._installed
        t.uninstall()
        assert not t._installed

    def test_double_install_is_idempotent(self) -> None:
        engine = Engine()
        t = AccessTracer(engine)
        t.install()
        t.install()  # no error
        t.uninstall()
