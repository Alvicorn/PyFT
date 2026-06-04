"""
PEP 302 import hook that AST-rewrites every user module's source so
attribute reads, writes, subscripts, and container-method mutations
call into the engine:

  obj.attr              -> _pyvft_get(_pyvft_engine, obj, 'attr')
  obj.attr = v          -> _pyvft_set(_pyvft_engine, obj, 'attr', v)
  obj.attr += v         -> _pyvft_set(eng, obj, 'attr',
                                     _pyvft_get(eng, obj, 'attr') + v)
  obj[k]                -> _pyvft_getitem(eng, obj, k)
  obj[k] = v            -> _pyvft_setitem(eng, obj, k, v)
  obj.append(x)         -> _pyvft_method(eng, obj, 'append', x)  # mutator
  obj.get(k)            -> _pyvft_method(eng, obj, 'get', k)     # reader

Container mutators / readers are recognized by **method name** from a
curated set; calls of those names trigger a synthetic
``__container__`` write / read on the receiver so two threads
mutating the same list / dict / set / bytearray are detected as a
race.

The rewritten module imports a synthetic ``pyvft._pyvft_runtime`` module
that exposes the engine and the helper functions.
"""

from __future__ import annotations

import ast
import importlib.abc
import importlib.machinery
import sys
import types
from typing import TYPE_CHECKING, Any, Optional

from .transformer import should_skip_module

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..detector.engine import Engine


_RUNTIME_MODULE_NAME = "pyvft._pyvft_runtime"

# Method names treated as container mutations / reads. Matched by
# string only — any object with a method of these names is treated as
# a container for race-detection purposes (over-approximates, never
# under-reports).
_CONTAINER_MUTATORS: frozenset[str] = frozenset(
    {
        "append",
        "extend",
        "insert",
        "remove",
        "pop",
        "popitem",
        "clear",
        "sort",
        "reverse",
        "update",
        "setdefault",
        "add",
        "discard",
        "intersection_update",
        "difference_update",
        "symmetric_difference_update",
    }
)
_CONTAINER_READERS: frozenset[str] = frozenset(
    {
        "get",
        "keys",
        "values",
        "items",
        "index",
        "count",
        "copy",
        "union",
        "intersection",
        "difference",
        "issubset",
        "issuperset",
    }
)
_CONTAINER_METHODS: frozenset[str] = _CONTAINER_MUTATORS | _CONTAINER_READERS


def _make_runtime_module(engine: "Engine") -> types.ModuleType:
    """Build the in-memory runtime module bound to a specific engine."""
    import types as _types

    mod = _types.ModuleType(_RUNTIME_MODULE_NAME)

    def _pyvft_get(eng: "Engine", obj: Any, attr: str) -> Any:  # noqa: ANN401
        try:
            eng.read(obj, attr)
        except Exception:
            pass
        return getattr(obj, attr)

    def _pyvft_set(
        eng: "Engine",
        obj: Any,  # noqa: ANN401
        attr: str,
        value: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        try:
            eng.write(obj, attr)
        except Exception:
            pass
        setattr(obj, attr, value)
        return value

    def _pyvft_getitem(
        eng: "Engine",
        obj: Any,  # noqa: ANN401
        key: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        try:
            eng.read(obj, f"[{key!r}]")
        except Exception:
            pass
        return obj[key]

    def _pyvft_setitem(
        eng: "Engine",
        obj: Any,  # noqa: ANN401
        key: Any,  # noqa: ANN401
        value: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        try:
            eng.write(obj, f"[{key!r}]")
        except Exception:
            pass
        obj[key] = value
        return value

    def _pyvft_method(
        eng: "Engine",
        obj: Any,  # noqa: ANN401
        name: str,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        # Synthetic container event: mutators write the container,
        # readers read it. The actual call always runs.
        if name in _CONTAINER_MUTATORS:
            try:
                eng.write(obj, "__container__")
            except Exception:
                pass
        elif name in _CONTAINER_READERS:
            try:
                eng.read(obj, "__container__")
            except Exception:
                pass
        return getattr(obj, name)(*args, **kwargs)

    mod._pyvft_engine = engine  # type: ignore[attr-defined]
    mod._pyvft_get = _pyvft_get  # type: ignore[attr-defined]
    mod._pyvft_set = _pyvft_set  # type: ignore[attr-defined]
    mod._pyvft_getitem = _pyvft_getitem  # type: ignore[attr-defined]
    mod._pyvft_setitem = _pyvft_setitem  # type: ignore[attr-defined]
    mod._pyvft_method = _pyvft_method  # type: ignore[attr-defined]
    return mod


_RT_ENGINE = "__pyvft_engine__"
_RT_GET = "__pyvft_get__"
_RT_SET = "__pyvft_set__"
_RT_GETITEM = "__pyvft_getitem__"
_RT_SETITEM = "__pyvft_setitem__"
_RT_METHOD = "__pyvft_method__"


class _AccessTransformer(ast.NodeTransformer):
    """
    Rewrites Attribute / Subscript / Assign / AugAssign nodes to call
    the runtime helpers. Preserves source line numbers via
    ``ast.copy_location``.
    """

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            return self.generic_visit(node)
        self.generic_visit(node)
        call = ast.Call(
            func=ast.Name(id=_RT_GET, ctx=ast.Load()),
            args=[
                ast.Name(id=_RT_ENGINE, ctx=ast.Load()),
                node.value,
                ast.Constant(value=node.attr),
            ],
            keywords=[],
        )
        return ast.copy_location(call, node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            return self.generic_visit(node)
        self.generic_visit(node)
        call = ast.Call(
            func=ast.Name(id=_RT_GETITEM, ctx=ast.Load()),
            args=[
                ast.Name(id=_RT_ENGINE, ctx=ast.Load()),
                node.value,
                node.slice,
            ],
            keywords=[],
        )
        return ast.copy_location(call, node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node.value = self.visit(node.value)
        # Multi-target (a = b = ...) and tuple unpacking fall through.
        if len(node.targets) != 1:
            for t in node.targets:
                self._visit_store_target(t)
            return node
        target = node.targets[0]
        if isinstance(target, ast.Attribute):
            target.value = self.visit(target.value)
            expr = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id=_RT_SET, ctx=ast.Load()),
                    args=[
                        ast.Name(id=_RT_ENGINE, ctx=ast.Load()),
                        target.value,
                        ast.Constant(value=target.attr),
                        node.value,
                    ],
                    keywords=[],
                )
            )
            return ast.copy_location(expr, node)
        if isinstance(target, ast.Subscript):
            target.value = self.visit(target.value)
            target.slice = self.visit(target.slice)
            expr = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id=_RT_SETITEM, ctx=ast.Load()),
                    args=[
                        ast.Name(id=_RT_ENGINE, ctx=ast.Load()),
                        target.value,
                        target.slice,
                        node.value,
                    ],
                    keywords=[],
                )
            )
            return ast.copy_location(expr, node)
        self._visit_store_target(target)
        return node

    def _visit_store_target(self, t: ast.AST) -> None:
        if isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                self._visit_store_target(elt)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        node.value = self.visit(node.value)
        target = node.target
        if isinstance(target, ast.Attribute):
            target.value = self.visit(target.value)
            read = ast.Call(
                func=ast.Name(id=_RT_GET, ctx=ast.Load()),
                args=[
                    ast.Name(id=_RT_ENGINE, ctx=ast.Load()),
                    target.value,
                    ast.Constant(value=target.attr),
                ],
                keywords=[],
            )
            combined = ast.BinOp(left=read, op=node.op, right=node.value)
            expr = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id=_RT_SET, ctx=ast.Load()),
                    args=[
                        ast.Name(id=_RT_ENGINE, ctx=ast.Load()),
                        target.value,
                        ast.Constant(value=target.attr),
                        combined,
                    ],
                    keywords=[],
                )
            )
            return ast.copy_location(expr, node)
        if isinstance(target, ast.Subscript):
            target.value = self.visit(target.value)
            target.slice = self.visit(target.slice)
            read = ast.Call(
                func=ast.Name(id=_RT_GETITEM, ctx=ast.Load()),
                args=[
                    ast.Name(id=_RT_ENGINE, ctx=ast.Load()),
                    target.value,
                    target.slice,
                ],
                keywords=[],
            )
            combined = ast.BinOp(left=read, op=node.op, right=node.value)
            expr = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id=_RT_SETITEM, ctx=ast.Load()),
                    args=[
                        ast.Name(id=_RT_ENGINE, ctx=ast.Load()),
                        target.value,
                        target.slice,
                        combined,
                    ],
                    keywords=[],
                )
            )
            return ast.copy_location(expr, node)
        return node

    def visit_Delete(self, node: ast.Delete) -> ast.AST:
        # del obj.attr / del obj[k] are not instrumented.
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """
        Rewrite ``obj.<container_method>(*args, **kw)`` into
        ``_pyvft_method(eng, obj, '<name>', *args, **kw)``. Non-method
        calls and method calls whose name is not in the container set
        pass through unchanged (with their children still visited).

        We intercept BEFORE ``generic_visit`` so that the receiver
        ``obj`` is visited as an expression in its own right (any nested
        attribute access in it gets the usual ``_pyvft_get`` wrapping)
        instead of having ``visit_Attribute`` rewrite the
        ``obj.<method>`` lookup into a getattr call.
        """
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.ctx, ast.Load)
            and func.attr in _CONTAINER_METHODS
        ):
            receiver = self.visit(func.value)
            method_name = func.attr
            new_args = [self.visit(a) for a in node.args]
            new_keywords = [self.visit(k) for k in node.keywords]
            call = ast.Call(
                func=ast.Name(id=_RT_METHOD, ctx=ast.Load()),
                args=[
                    ast.Name(id=_RT_ENGINE, ctx=ast.Load()),
                    receiver,
                    ast.Constant(value=method_name),
                    *new_args,
                ],
                keywords=new_keywords,
            )
            return ast.copy_location(call, node)
        # All other calls: normal recursion.
        self.generic_visit(node)
        return node


def _prepend_runtime_import(tree: ast.Module) -> None:
    """
    Insert ``from pyvft._pyvft_runtime import ...`` after the module
    docstring and any ``from __future__`` imports.
    """
    imp = ast.ImportFrom(
        module=_RUNTIME_MODULE_NAME,
        names=[
            ast.alias(name="_pyvft_engine", asname=_RT_ENGINE),
            ast.alias(name="_pyvft_get", asname=_RT_GET),
            ast.alias(name="_pyvft_set", asname=_RT_SET),
            ast.alias(name="_pyvft_getitem", asname=_RT_GETITEM),
            ast.alias(name="_pyvft_setitem", asname=_RT_SETITEM),
            ast.alias(name="_pyvft_method", asname=_RT_METHOD),
        ],
        level=0,
    )
    ast.fix_missing_locations(imp)

    insert_at = 0
    for i, stmt in enumerate(tree.body):
        if (
            i == 0
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            insert_at = i + 1
            continue
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            insert_at = i + 1
            continue
        break
    tree.body.insert(insert_at, imp)


def transform_source(source: str, filename: str) -> ast.Module:
    """Parse ``source``, AST-rewrite it, and return the new ``ast.Module``."""
    tree = ast.parse(source, filename=filename)
    _AccessTransformer().visit(tree)
    _prepend_runtime_import(tree)
    ast.fix_missing_locations(tree)
    return tree


class _PyVFTLoader(importlib.abc.Loader):
    """Loader that wraps a real loader and AST-rewrites the module source
    in ``exec_module``.
    """

    def __init__(self, real_loader: importlib.abc.Loader) -> None:
        self._real = real_loader

    def create_module(
        self, spec: importlib.machinery.ModuleSpec
    ) -> types.ModuleType | None:  # type: ignore[override]
        if hasattr(self._real, "create_module"):
            return self._real.create_module(spec)
        return None

    def exec_module(self, module: types.ModuleType) -> None:  # type: ignore[override]
        name = module.__name__
        filename = getattr(module, "__file__", "<unknown>") or "<unknown>"
        source: Optional[str] = None
        if hasattr(self._real, "get_source"):
            try:
                source = self._real.get_source(name)
            except Exception:
                source = None
        if source is None:
            self._real.exec_module(module)
            return
        try:
            tree = transform_source(source, filename)
            code = compile(tree, filename, "exec")
        except Exception:
            self._real.exec_module(module)
            return
        exec(code, module.__dict__)


class _PyVFTFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder placed at the front of ``sys.meta_path``. For each
    not-skipped module name, delegates to the original finders to resolve
    the spec, then wraps its loader with ``_PyVFTLoader``.
    """

    def __init__(self) -> None:
        self._delegates: list[importlib.abc.MetaPathFinder] = []

    def set_delegates(
        self, finders: "Sequence[importlib.abc.MetaPathFinder]"
    ) -> None:
        self._delegates = list(finders)

    def find_spec(
        self,
        fullname: str,
        path: "Sequence[str] | None",
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if should_skip_module(fullname):
            return None
        for finder in self._delegates:
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                spec = None
            if spec is None:
                continue
            if spec.loader is None:
                return spec
            if not isinstance(
                spec.loader, importlib.machinery.SourceFileLoader
            ):
                return spec
            spec.loader = _PyVFTLoader(spec.loader)
            return spec
        return None


class ImportHook:
    """
    Manages installation of the AST-rewriting import hook and its
    runtime module in ``sys.modules``.
    """

    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        self._finder: Optional[_PyVFTFinder] = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        sys.modules[_RUNTIME_MODULE_NAME] = _make_runtime_module(self.engine)
        finder = _PyVFTFinder()
        finder.set_delegates(sys.meta_path)  # type: ignore[arg-type]
        sys.meta_path.insert(0, finder)
        self._finder = finder
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._finder in sys.meta_path:
            sys.meta_path.remove(self._finder)
        sys.modules.pop(_RUNTIME_MODULE_NAME, None)
        self._finder = None
        self._installed = False
