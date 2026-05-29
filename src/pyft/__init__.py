"""
PyFT — VerifiedFT precise dynamic race detector for free-threaded Python.

Public API:

  pyft.install(version="v2")     Start the detector
  pyft.uninstall()               Stop the detector and restore patched primitives
  pyft.report()                  Print the race report to stderr
  pyft.races()                   list[RaceReport] for programmatic inspection
  pyft.reset()                   Clear all recorded races (keeps detector running)
  pyft.context(version="v2")     Context manager: install on enter, report on exit
  @pyft.detect                   Decorator: install around a function, report on return
  @pyft.detect(version="v1")     Decorator with an explicit VerifiedFT variant
  pyft.get_engine()              Return the active Engine (for advanced inspection)

The ``version`` argument picks which VerifiedFT analysis variant to use:

  - "v1" — idealised analyser; stores a full VectorClock for the last
           write and for the union of reads. Race checks use vc_leq.
  - "v2" — optimised FastTrack-style analyser; stores a single Epoch for
           the last write and a compressed read-state machine. Default.

Typical usage:

  with pyft.context():
      import myapp                 # auto-traced by the import hook
      myapp.run_concurrent_code()
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import IO, TypeVar, overload

from .core.var_state import VFTVersion
from .detector.engine import Engine
from .detector.race_log import RaceReport
from .report.formatter import print_summary

try:
    from .instrument.lock_patcher import LockPatcher
    from .instrument.transformer import AccessTracer
    from .instrument.wrappers import AutoTracker

    _HAS_INSTRUMENT = True
except AttributeError:
    AccessTracer = None  # type: ignore[assignment,misc]
    AutoTracker = None  # type: ignore[assignment,misc]
    LockPatcher = None  # type: ignore[assignment,misc]
    _HAS_INSTRUMENT = False


_engine: Engine | None = None
_auto_tracker: AutoTracker | None = None
_access_tracer: AccessTracer | None = None
_lock_patcher: LockPatcher | None = None
_installed = False


def install(version: VFTVersion | str = VFTVersion.V2) -> None:
    """
    Start pyft. Monkey-patches threading.Lock, RLock, Semaphore,
    BoundedSemaphore, Event, Barrier, and Thread.start / Thread.join so
    the engine sees synchronization events synchronously with the
    operation, and installs an AST-rewriting import hook that
    instruments every newly-imported user module's attribute access.

    ``version`` selects which VerifiedFT analyser to use ("v1" or "v2").
    Re-installing with a different version is a no-op while the previous
    install is still active; call ``uninstall()`` first.
    """
    global _engine, _auto_tracker, _access_tracer, _lock_patcher, _installed
    if _installed:
        return

    _engine = Engine(version=version)
    _lock_patcher = LockPatcher(_engine)
    _auto_tracker = AutoTracker(_engine)
    _access_tracer = AccessTracer(_engine)

    _lock_patcher.install()
    _auto_tracker.install()
    _access_tracer.install()

    _installed = True


def uninstall() -> None:
    """
    Stop pyft and restore all patched primitives.
    Does NOT clear recorded races — call reset() for that.
    """
    global _installed
    if not _installed:
        return

    if _access_tracer:
        _access_tracer.uninstall()
    if _auto_tracker:
        _auto_tracker.uninstall()
    if _lock_patcher:
        _lock_patcher.uninstall()

    _installed = False


def report(file: IO[str] | None = None) -> None:
    """Print the race report to stderr (or `file` if given)."""
    if _engine is None:
        print("pyft: not installed", file=file)
        return
    print_summary(_engine.race_log, file=file)


def races() -> list[RaceReport]:
    """Return all recorded RaceReports for programmatic inspection."""
    if _engine is None:
        return []
    return _engine.race_log.all_reports()


def reset() -> None:
    """Clear all recorded races. Detector remains running."""
    if _engine is not None:
        _engine.race_log.clear()


def get_engine() -> Engine | None:
    """Return the active Engine instance (for advanced use)."""
    return _engine


@contextlib.contextmanager
def context(version: VFTVersion | str = VFTVersion.V2) -> Iterator[None]:
    """
    Context manager that installs pyft on enter and prints a race
    report + uninstalls on exit.

    Any module imported INSIDE the ``with`` block is automatically
    AST-rewritten so its attribute access is traced.

    Example:
        with pyft.context(version="v1"):
            import myapp
            myapp.run()
    """
    install(version=version)
    try:
        yield
    finally:
        uninstall()
        report()


_F = TypeVar("_F", bound=Callable[..., object])


@overload
def detect(fn: _F) -> _F: ...
@overload
def detect(
    fn: None = None, *, version: VFTVersion | str = VFTVersion.V2
) -> Callable[[_F], _F]: ...


def detect(fn=None, *, version: VFTVersion | str = VFTVersion.V2):  # type: ignore[no-untyped-def]
    """
    Decorator that runs a function under pyft and prints the race
    report when it returns. Usable bare (``@pyft.detect``) or with a
    ``version`` kwarg (``@pyft.detect(version="v1")``).
    """

    def _wrap(target: _F) -> _F:
        def wrapper(*args: object, **kwargs: object) -> object:
            install(version=version)
            try:
                return target(*args, **kwargs)
            finally:
                uninstall()
                report()

        wrapper.__name__ = target.__name__
        wrapper.__doc__ = target.__doc__
        return wrapper  # type: ignore[return-value]

    if fn is None:
        return _wrap
    return _wrap(fn)
