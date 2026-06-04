"""
Tests for the container-method AST rewriting + runtime helper.
"""

from __future__ import annotations

import ast
import textwrap
from typing import Any, Never

from pyvft.detector.engine import Engine
from pyvft.instrument.import_hook import (
    _CONTAINER_MUTATORS,
    _CONTAINER_READERS,
    _make_runtime_module,
    transform_source,
)


def _exec_module(source: str, engine: Engine) -> dict[str, Any]:
    """Compile + exec ``source`` against a fresh runtime module bound
    to ``engine``; return the module globals.
    """
    tree = transform_source(textwrap.dedent(source), "<test>")
    code = compile(tree, "<test>", "exec")
    # Install the runtime module under the canonical name so the
    # rewritten import resolves.
    import sys

    rt = _make_runtime_module(engine)
    sys.modules[rt.__name__] = rt
    try:
        ns: dict[str, Any] = {}
        exec(code, ns)
        return ns
    finally:
        sys.modules.pop(rt.__name__, None)


class TestAstRewrites:
    def test_append_call_is_rewritten(self) -> None:
        tree = transform_source(
            textwrap.dedent(
                """
                def f(lst):
                    lst.append(1)
                """
            ),
            "<test>",
        )
        fn = tree.body[1]  # body[0] = runtime import
        stmt = fn.body[0]
        assert isinstance(stmt, ast.Expr)
        call = stmt.value
        assert isinstance(call, ast.Call)
        # Outer func is the method helper alias.
        assert isinstance(call.func, ast.Name)
        assert call.func.id == "__pyvft_method__"
        # Args: engine ref, receiver, 'append', 1
        assert isinstance(call.args[2], ast.Constant)
        assert call.args[2].value == "append"

    def test_non_container_method_is_not_rewritten(self) -> None:
        tree = transform_source(
            textwrap.dedent(
                """
                def f(obj):
                    obj.frobnicate(1, 2)
                """
            ),
            "<test>",
        )
        fn = tree.body[1]
        stmt = fn.body[0]
        assert isinstance(stmt, ast.Expr)
        call = stmt.value
        assert isinstance(call, ast.Call)
        # The .frobnicate access still becomes _pyvft_get (Attribute in
        # Load context inside a Call), but the call itself isn't
        # routed through _pyvft_method.
        assert isinstance(call.func, ast.Call)
        assert call.func.func.id == "__pyvft_get__"
        assert call.func.args[2].value == "frobnicate"

    def test_keyword_arguments_preserved(self) -> None:
        tree = transform_source(
            textwrap.dedent(
                """
                def f(d):
                    d.update(other, foo=1)
                """
            ),
            "<test>",
        )
        fn = tree.body[1]
        stmt = fn.body[0]
        call = stmt.value
        assert call.func.id == "__pyvft_method__"
        assert len(call.keywords) == 1
        assert call.keywords[0].arg == "foo"

    def test_every_mutator_name_is_rewritten(self) -> None:
        for name in _CONTAINER_MUTATORS:
            tree = transform_source(f"def f(o):\n    o.{name}(1)\n", "<test>")
            fn = tree.body[1]
            call = fn.body[0].value
            assert call.func.id == "__pyvft_method__", name
            assert call.args[2].value == name

    def test_every_reader_name_is_rewritten(self) -> None:
        for name in _CONTAINER_READERS:
            tree = transform_source(f"def f(o):\n    o.{name}()\n", "<test>")
            fn = tree.body[1]
            call = fn.body[0].value
            assert call.func.id == "__pyvft_method__", name
            assert call.args[2].value == name


class TestRuntimeHelper:
    def test_mutator_fires_engine_write_and_actually_appends(self) -> None:
        engine = Engine()
        writes: list[tuple[object, str]] = []
        engine.write = lambda obj, attr: writes.append((id(obj), attr))  # type: ignore[method-assign]
        rt = _make_runtime_module(engine)
        lst: list[int] = []
        rt._pyvft_method(engine, lst, "append", 42)
        assert lst == [42]
        assert writes == [(id(lst), "__container__")]

    def test_reader_fires_engine_read(self) -> None:
        engine = Engine()
        reads: list[tuple[object, str]] = []
        engine.read = lambda obj, attr: reads.append((id(obj), attr))  # type: ignore[method-assign]
        rt = _make_runtime_module(engine)
        d = {"a": 1, "b": 2}
        result = rt._pyvft_method(engine, d, "get", "a")
        assert result == 1
        assert reads == [(id(d), "__container__")]

    def test_unrelated_method_is_dispatched_but_no_event(self) -> None:
        """A name not in mutator/reader sets shouldn't fire engine
        events but still calls the underlying method."""
        engine = Engine()
        seen_calls: list[str] = []
        engine.read = lambda *a: seen_calls.append("read")  # type: ignore[method-assign]
        engine.write = lambda *a: seen_calls.append("write")  # type: ignore[method-assign]
        rt = _make_runtime_module(engine)

        class C:
            def hello(self) -> str:
                return "hi"

        assert rt._pyvft_method(engine, C(), "hello") == "hi"
        assert seen_calls == []

    def test_helper_swallows_engine_errors(self) -> None:
        engine = Engine()

        def boom(*a: Any) -> Never:  # noqa: ANN401
            raise RuntimeError("boom")

        engine.write = boom  # type: ignore[method-assign]
        rt = _make_runtime_module(engine)
        lst: list[int] = []
        # Must still append even though engine.write raised.
        rt._pyvft_method(engine, lst, "append", 7)
        assert lst == [7]


class TestRewriteEndToEnd:
    def test_rewritten_module_records_write_on_append(self) -> None:
        engine = Engine()
        writes: list[tuple[int, str]] = []
        engine.write = lambda obj, attr: writes.append((id(obj), attr))  # type: ignore[method-assign]
        ns = _exec_module(
            """
            def run(lst):
                lst.append(1)
                lst.append(2)
                return len(lst)
            """,
            engine,
        )
        lst: list[int] = []
        result = ns["run"](lst)
        assert result == 2
        assert lst == [1, 2]
        # Two appends → two __container__ writes on the same object.
        assert writes == [
            (id(lst), "__container__"),
            (id(lst), "__container__"),
        ]
