from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..detector.engine import Engine

log = logging.getLogger(__name__)


_SKIP_ATTRS = frozenset(
    {
        "__class__",
        "__dict__",
        "__doc__",
        "__module__",
        "__weakref__",
        "__pyft_engine__",
        "__pyft_tracked__",
    }
)


class TrackedProxy:
    """
    Transparent proxy that calls engine.read()/write() on every
    attribute access to the wrapped object.

    Usage:
        obj = TrackedProxy(my_object, engine)
        obj.x        # triggers engine.read(my_object, "x")
        obj.x = 5    # triggers engine.write(my_object, "x")
    """

    __slots__ = ("__pyft_wrapped__", "__pyft_engine__")

    def __init__(self, wrapped: Any, engine: "Engine") -> None:
        object.__setattr__(self, "__pyft_wrapped__", wrapped)
        object.__setattr__(self, "__pyft_engine__", engine)

    def __getattribute__(self, name: str) -> Any:
        if name in ("__pyft_wrapped__", "__pyft_engine__"):
            return object.__getattribute__(self, name)
        wrapped = object.__getattribute__(self, "__pyft_wrapped__")
        engine = object.__getattribute__(self, "__pyft_engine__")
        if name not in _SKIP_ATTRS:
            log.debug(f"PROXY GET: tid={threading.get_ident()} name={name}")
            engine.read(wrapped, name)
        return getattr(wrapped, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("__pyft_wrapped__", "__pyft_engine__"):
            object.__setattr__(self, name, value)
            return
        wrapped = object.__getattribute__(self, "__pyft_wrapped__")
        engine = object.__getattribute__(self, "__pyft_engine__")
        if name not in _SKIP_ATTRS:
            log.debug(f"PROXY SET: tid={threading.get_ident()} name={name}")
            engine.write(wrapped, name)
        setattr(wrapped, name, value)

    def __repr__(self) -> str:
        wrapped = object.__getattribute__(self, "__pyft_wrapped__")
        return f"TrackedProxy({wrapped!r})"


def make_tracked_class(cls: type, engine: "Engine") -> type:
    """
    Return a subclass of `cls` that calls engine.read()/write()
    on every instance attribute access.

    Works for user-defined classes.
    Does not work for classes with
    __slots__ (we fall back to TrackedProxy in that case).
    """

    engine_ref = engine

    class TrackedMeta(type(cls)):
        pass

    class Tracked(cls, metaclass=TrackedMeta):
        __pyft_engine__ = engine_ref
        __pyft_base_class__ = cls

        def __getattribute__(self, name: str) -> Any:
            val = super().__getattribute__(name)
            if name not in _SKIP_ATTRS and not name.startswith("__pyft_"):
                try:
                    engine_ref.read(self, name)
                except Exception:
                    pass
            return val

        def __setattr__(self, name: str, value: Any) -> None:
            if name not in _SKIP_ATTRS and not name.startswith("__pyft_"):
                try:
                    engine_ref.write(self, name)
                except Exception:
                    pass
            super().__setattr__(name, value)

    Tracked.__name__ = f"Tracked[{cls.__name__}]"
    Tracked.__qualname__ = f"Tracked[{cls.__qualname__}]"
    return Tracked


class AutoTracker:
    """
    Wrap threading.Thread's run() method so that when a thread starts,
    walk the arguments it captured (via closure inspection) and register
    any mutable objects as potentially shared. Combined with the engine's
    strict "two threads touched it" rule, this gives precise detection.
    """

    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        self._original_start = threading.Thread.start
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        engine = self.engine
        original_start = threading.Thread.start
        log.info("AutoTracker Installed")

        def patched_start(thread_self: threading.Thread) -> None:
            # establish fork HB edge from PARENT thread before child starts.
            parent_tid = id(threading.current_thread())
            child_tid = id(thread_self)
            engine.thread_start(parent_tid, child_tid, thread_self.name)

            original_run = thread_self.run
            child_tid_ref = child_tid  # capture for closure

            def wrapped_run() -> None:
                try:
                    original_run()
                finally:
                    engine.thread_finish(child_tid_ref)

            thread_self.run = wrapped_run
            original_start(thread_self)

        threading.Thread.start = patched_start  # type: ignore[method-assign]
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        threading.Thread.start = self._original_start  # type: ignore[method-assign]
        self._installed = False
        log.info("AutoTracker Uninstalled")
