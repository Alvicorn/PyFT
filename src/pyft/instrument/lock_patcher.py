"""
LockPatcher: monkey-patches threading synchronization primitives so the
engine sees acquire / release events synchronously around the OS call.

The engine event is dispatched in the same thread that performs the
operation: ``release`` calls ``engine.lock_release`` first, then the OS
release; ``acquire`` calls the OS acquire, then ``engine.lock_acquire``
once it succeeded. The next acquirer is guaranteed to observe the
release_vc written by the previous releaser.
"""

from __future__ import annotations

import threading
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..detector.engine import Engine


_ORIGINAL_LOCK = threading.Lock
_ORIGINAL_RLOCK = threading.RLock
_ORIGINAL_SEMAPHORE = threading.Semaphore
_ORIGINAL_BSEMAPHORE = threading.BoundedSemaphore
_ORIGINAL_EVENT_SET = threading.Event.set
_ORIGINAL_EVENT_WAIT = threading.Event.wait
_ORIGINAL_BARRIER_WAIT = threading.Barrier.wait


class _TrackedLock:
    """Wraps a real Lock / RLock; instruments acquire/release/__enter__/__exit__."""

    __slots__ = ("_real", "_engine", "_lock_id")

    def __init__(self, real: Any, engine: "Engine") -> None:  # noqa: ANN401
        self._real = real
        self._engine = engine
        self._lock_id = id(real)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        result = self._real.acquire(blocking, timeout)
        if result:
            self._engine.lock_acquire(self._lock_id)
        return result

    def release(self) -> None:
        self._engine.lock_release(self._lock_id)
        self._real.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.release()

    def locked(self) -> bool:
        return self._real.locked()

    def __repr__(self) -> str:
        return f"_TrackedLock({self._real!r})"


class _TrackedSemaphore:
    """Wraps a real Semaphore / BoundedSemaphore."""

    __slots__ = ("_real", "_engine", "_lock_id")

    def __init__(self, real: Any, engine: "Engine") -> None:  # noqa: ANN401
        self._real = real
        self._engine = engine
        self._lock_id = id(real)

    def acquire(
        self, blocking: bool = True, timeout: float | None = None
    ) -> bool:
        result = self._real.acquire(blocking, timeout)
        if result:
            self._engine.lock_acquire(self._lock_id)
        return result

    def release(self, n: int = 1) -> None:
        self._engine.lock_release(self._lock_id)
        self._real.release(n)

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.release()


class LockPatcher:
    """
    Installs / uninstalls monkey-patched lock factories.

    After ``install()``, ``threading.Lock``, ``RLock``, ``Semaphore``,
    ``BoundedSemaphore`` return wrappers that dispatch engine events.
    ``Event.set/wait`` and ``Barrier.wait`` are patched in place. Locks
    created before ``install()`` are not wrapped.
    """

    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        engine = self.engine

        def lock_factory(*args: Any, **kwargs: Any) -> _TrackedLock:  # noqa: ANN401
            return _TrackedLock(_ORIGINAL_LOCK(*args, **kwargs), engine)

        def rlock_factory(*args: Any, **kwargs: Any) -> _TrackedLock:  # noqa: ANN401
            return _TrackedLock(_ORIGINAL_RLOCK(*args, **kwargs), engine)

        def sema_factory(*args: Any, **kwargs: Any) -> _TrackedSemaphore:  # noqa: ANN401
            return _TrackedSemaphore(
                _ORIGINAL_SEMAPHORE(*args, **kwargs), engine
            )

        def bsema_factory(
            *args: Any,  # noqa: ANN401
            **kwargs: Any,  # noqa: ANN401
        ) -> _TrackedSemaphore:
            return _TrackedSemaphore(
                _ORIGINAL_BSEMAPHORE(*args, **kwargs), engine
            )

        def event_set_patched(self: threading.Event) -> None:
            engine.lock_release(id(self))
            return _ORIGINAL_EVENT_SET(self)

        def event_wait_patched(
            self: threading.Event, timeout: float | None = None
        ) -> bool:
            result = _ORIGINAL_EVENT_WAIT(self, timeout)
            if result is True:
                engine.lock_acquire(id(self))
            return result

        def barrier_wait_patched(
            self: threading.Barrier, timeout: float | None = None
        ) -> int:
            engine.lock_release(id(self))
            result = _ORIGINAL_BARRIER_WAIT(self, timeout)
            engine.lock_acquire(id(self))
            return result

        threading.Lock = lock_factory  # type: ignore[misc,assignment]
        threading.RLock = rlock_factory  # type: ignore[misc,assignment]
        threading.Semaphore = sema_factory  # type: ignore[misc,assignment]
        threading.BoundedSemaphore = bsema_factory  # type: ignore[misc,assignment]
        threading.Event.set = event_set_patched  # type: ignore[method-assign]
        threading.Event.wait = event_wait_patched  # type: ignore[method-assign]
        threading.Barrier.wait = barrier_wait_patched  # type: ignore[method-assign]
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        threading.Lock = _ORIGINAL_LOCK  # type: ignore[misc,assignment]
        threading.RLock = _ORIGINAL_RLOCK  # type: ignore[misc,assignment]
        threading.Semaphore = _ORIGINAL_SEMAPHORE  # type: ignore[misc,assignment]
        threading.BoundedSemaphore = _ORIGINAL_BSEMAPHORE  # type: ignore[misc,assignment]
        threading.Event.set = _ORIGINAL_EVENT_SET  # type: ignore[method-assign]
        threading.Event.wait = _ORIGINAL_EVENT_WAIT  # type: ignore[method-assign]
        threading.Barrier.wait = _ORIGINAL_BARRIER_WAIT  # type: ignore[method-assign]
        self._installed = False
