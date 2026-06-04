from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..detector.engine import Engine


_SKIP_ATTRS = frozenset(
    {
        "__class__",
        "__dict__",
        "__doc__",
        "__module__",
        "__weakref__",
        "__pyvft_engine__",
        "__pyvft_tracked__",
    }
)


class TrackedProxy:
    """
    Transparent proxy: forwards every attribute access to ``wrapped``
    after calling ``engine.read`` / ``engine.write``.
    """

    __slots__ = ("__pyvft_wrapped__", "__pyvft_engine__")

    def __init__(self, wrapped: Any, engine: "Engine") -> None:  # noqa: ANN401
        object.__setattr__(self, "__pyvft_wrapped__", wrapped)
        object.__setattr__(self, "__pyvft_engine__", engine)

    def __getattribute__(self, name: str) -> Any:  # noqa: ANN401
        if name in ("__pyvft_wrapped__", "__pyvft_engine__"):
            return object.__getattribute__(self, name)
        wrapped = object.__getattribute__(self, "__pyvft_wrapped__")
        engine = object.__getattribute__(self, "__pyvft_engine__")
        if name not in _SKIP_ATTRS:
            engine.read(wrapped, name)
        return getattr(wrapped, name)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        if name in ("__pyvft_wrapped__", "__pyvft_engine__"):
            object.__setattr__(self, name, value)
            return
        wrapped = object.__getattribute__(self, "__pyvft_wrapped__")
        engine = object.__getattribute__(self, "__pyvft_engine__")
        if name not in _SKIP_ATTRS:
            engine.write(wrapped, name)
        setattr(wrapped, name, value)

    def __repr__(self) -> str:
        wrapped = object.__getattribute__(self, "__pyvft_wrapped__")
        return f"TrackedProxy({wrapped!r})"


def make_tracked_class(cls: type, engine: "Engine") -> type:
    """
    Return a subclass of ``cls`` that calls ``engine.read`` /
    ``engine.write`` on every instance attribute access.

    Falls back to TrackedProxy for classes that define ``__slots__``.
    """

    engine_ref = engine

    class TrackedMeta(type(cls)):  # type: ignore[misc]
        pass

    class Tracked(cls, metaclass=TrackedMeta):  # type: ignore[misc,valid-type]
        __pyvft_engine__ = engine_ref

        def __getattribute__(self, name: str) -> Any:  # noqa: ANN401
            if name not in _SKIP_ATTRS and not name.startswith("__pyvft_"):
                engine_ref.read(self, name)
            return super().__getattribute__(name)

        def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
            if name not in _SKIP_ATTRS and not name.startswith("__pyvft_"):
                engine_ref.write(self, name)
            super().__setattr__(name, value)

    Tracked.__name__ = f"Tracked[{cls.__name__}]"
    Tracked.__qualname__ = f"Tracked[{cls.__qualname__}]"
    return Tracked


_ORIGINAL_THREAD_START = threading.Thread.start
_ORIGINAL_THREAD_JOIN = threading.Thread.join


class AutoTracker:
    """
    Monkey-patches ``Thread.start`` and ``Thread.join`` so the engine
    sees fork / finish / join events synchronously around the OS calls.
    """

    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        engine = self.engine

        def patched_start(thread_self: threading.Thread) -> None:
            parent_tid = id(threading.current_thread())
            child_tid = id(thread_self)
            engine.thread_start(parent_tid, child_tid, thread_self.name)

            original_run = thread_self.run

            def wrapped_run() -> None:
                try:
                    original_run()
                finally:
                    engine.thread_finish(child_tid)

            thread_self.run = wrapped_run  # type: ignore[method-assign]
            _ORIGINAL_THREAD_START(thread_self)

        def patched_join(
            thread_self: threading.Thread, timeout: float | None = None
        ) -> None:
            _ORIGINAL_THREAD_JOIN(thread_self, timeout)
            if not thread_self.is_alive():
                engine.thread_join(
                    joiner_tid=id(threading.current_thread()),
                    joinee_tid=id(thread_self),
                )

        threading.Thread.start = patched_start  # type: ignore[method-assign,assignment]
        threading.Thread.join = patched_join  # type: ignore[method-assign,assignment]
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        threading.Thread.start = _ORIGINAL_THREAD_START  # type: ignore[method-assign]
        threading.Thread.join = _ORIGINAL_THREAD_JOIN  # type: ignore[method-assign]
        self._installed = False
