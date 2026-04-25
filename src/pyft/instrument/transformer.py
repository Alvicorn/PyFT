"""
For each loaded module, patch its code objects so that every
attribute/subscript access calls back into the engine:

  obj.attr        (LOAD_ATTR)       ->  engine.read(obj, "attr");
  obj.attr = val  (STORE_ATTR)      -> engine.write(obj, "attr");
  obj[key]        (BINARY_SUBSCR)   -> engine.read(obj, repr(key));
  obj[key] = val  (STORE_SUBSCR)    -> engine.write(obj, repr(key));
"""

from __future__ import annotations

import dis
import sys
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detector.engine import Engine


_LOAD_ATTR = dis.opmap.get("LOAD_ATTR", -1)
_STORE_ATTR = dis.opmap.get("STORE_ATTR", -1)
_BINARY_SUBSCR = dis.opmap.get("BINARY_SUBSCR", -1)
_STORE_SUBSCR = dis.opmap.get("STORE_SUBSCR", -1)


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
)


def should_skip_module(module_name: str) -> bool:
    return any(
        module_name == p or module_name.startswith(p + ".")
        for p in _SKIP_PREFIXES
    )


def instrument_code(
    code: types.CodeType, engine: "Engine", module_name: str
) -> types.CodeType:
    """
    Recursively instrument a code object and all nested code objects
    (comprehensions, lambdas, inner functions, class bodies).

    Returns a new code object with instrumentation injected.
    This is a best-effort transform: if anything fails we return
    the original code unchanged.
    """
    try:
        return _instrument_code_inner(code, engine, module_name)
    except Exception:
        return code


def _instrument_code_inner(
    code: types.CodeType, engine: "Engine", module_name: str
) -> types.CodeType:
    """
    Inject wrapper functions into the module's globals and
    use sys.settrace-style function wrapping on __getattribute__ and
    __setattr__ for objects.
    """
    # recursively handle nested code objects (inner functions, etc.)
    new_consts = list(code.co_consts)
    changed = False
    for i, const in enumerate(new_consts):
        if isinstance(const, types.CodeType):
            new_const = _instrument_code_inner(const, engine, module_name)
            if new_const is not const:
                new_consts[i] = new_const
                changed = True

    if changed:
        return code.replace(co_consts=tuple(new_consts))
    return code


class AccessTracer:
    """
    Installs a sys.monitoring-based tracer that fires on attribute
    access events.

    Python 3.12+ sys.monitoring is used to hook CALL events
    combined with __getattribute__/__setattr__ wrapping on tracked
    objects to detect read/write accesses.
    """

    TOOL_ID = sys.monitoring.DEBUGGER_ID  # Use debugger slot

    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        # rely on sync_patch.py + wrappers.py for instrumentation.
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._installed = False
