"""
AccessTracer: automatic attribute / subscript tracing for user code.

Strategy is AST rewriting at module import time; the real implementation
lives in ``import_hook.py``. Modules already imported when
``AccessTracer.install()`` is called are not retroactively
instrumented — only NEW imports go through the hook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detector.engine import Engine


_SKIP_PREFIXES = (
    "pyft",
    "threading",
    "_thread",
    "importlib",
    "dis",
    "types",
    "abc",
    "weakref",
    "traceback",
    "linecache",
    "tokenize",
    "ast",
    "inspect",
    "logging",
    "sys",
    "os",
    "io",
    "re",
    "collections",
    "functools",
    "itertools",
    "encodings",
    "codecs",
    "warnings",
    "contextlib",
)


def should_skip_module(module_name: str) -> bool:
    return any(
        module_name == p or module_name.startswith(p + ".")
        for p in _SKIP_PREFIXES
    )


class AccessTracer:
    """
    Installs the AST-rewriting import hook so newly-imported user
    modules have their attribute reads / writes / subscripts routed
    through the engine. Skips modules listed in ``_SKIP_PREFIXES``.
    """

    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        from .import_hook import ImportHook

        self._hook = ImportHook(engine)
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._hook.install()
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._hook.uninstall()
        self._installed = False
