"""
PyFT — VerifiedFT precise dynamic race detector for free-threaded Python.

Public API:

  pyft.install()          Start the detector
  pyft.uninstall()        Stop the detector and restore patched primitives
  pyft.report()           Print the race report to stderr
  pyft.races()            Return list[RaceReport] for programmatic inspection
  pyft.reset()            Clear all recorded races (keeps detector running)
  pyft.context()          Context manager: install on enter, report+uninstall on exit
  @pyft.detect            Decorator: install around a function, report on return
  pyft.get_engine()       Return the active Engine (for advanced inspection)

Typical usage:

  with pyft.context():
      import myapp                 # auto-traced by the import hook
      myapp.run_concurrent_code()

Or as a decorator:

  @pyft.detect
  def test_my_concurrent_code():
      import workload
      workload.do_stuff()

How tracing works: ``install()`` registers an AST-rewriting import hook
plus monkey-patches for threading primitives. Any module imported AFTER
install runs through the hook, so its attribute access, subscripts, and
sync events feed the engine automatically. Modules imported BEFORE
install are not instrumented — structure your code so the workload lives
in a module that is imported inside the ``context()`` or ``@detect``
scope, or run the whole program with ``python -m pyft script.py``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import IO, TypeVar

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


def install() -> None:
    """
    Start pyft. Monkey-patches threading.Lock, RLock, Semaphore,
    BoundedSemaphore, Event, Barrier, and Thread.start / Thread.join so
    the engine sees synchronization events synchronously with the
    operation, and installs an AST-rewriting import hook that
    instruments every newly-imported user module's attribute access.
    """
    global _engine, _auto_tracker, _access_tracer, _lock_patcher, _installed
    if _installed:
        return

    _engine = Engine()
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
def context() -> Iterator[None]:
    """
    Context manager that installs pyft on enter and prints a race
    report + uninstalls on exit.

    Any module imported INSIDE the ``with`` block is automatically
    AST-rewritten so its attribute access is traced.

    Example:
        with pyft.context():
            import myapp
            myapp.run()
    """
    install()
    try:
        yield
    finally:
        uninstall()
        report()


_F = TypeVar("_F", bound=Callable[..., object])


def detect(fn: _F) -> _F:
    """
    Decorator that runs a function under pyft and prints the race
    report when it returns.

    Example:
        @pyft.detect
        def test_concurrent():
            import workload
            workload.do_stuff()
    """

    def wrapper(*args: object, **kwargs: object) -> object:
        install()
        try:
            return fn(*args, **kwargs)
        finally:
            uninstall()
            report()

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper  # type: ignore[return-value]
