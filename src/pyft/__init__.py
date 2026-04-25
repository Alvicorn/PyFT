"""
PyFT — VerifiedFT precise dynamic race detector for free-threaded Python.

Public API:

  pyft.install()          Start the detector (patches threading primitives)
  pyft.uninstall()        Stop the detector and restore original primitives
  pyft.track(obj)         Return a proxy that tracks accesses to obj
  pyft.report()           Print the race report to stderr
  pyft.races()            Return list[RaceReport] for programmatic inspection
  pyft.reset()            Clear all recorded races (keeps detector running)
  pyft.context()          Context manager: install on enter, report+uninstall on exit

Typical usage:

  import pyft

  with pyft.context():
      # ... your multi-threaded code ...
      shared = pyft.track(MyObject())
      # use shared.x, shared.y etc.

Or as a decorator:
  @pyft.detect
  def test_my_concurrent_code():
      ...

Note: for auto-detection of shared objects, wrap objects with pyft.track()
or use pyft.track_class() to instrument a class. Objects not passed through
track() are only flagged if they are accessed via a tracked container.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, TypeVar

from .detector.engine import Engine
from .detector.race_log import RaceReport
from .report.formatter import print_summary

try:
    from .instrument.monitor import SyncMonitor
    from .instrument.transformer import AccessTracer
    from .instrument.wrappers import (
        AutoTracker,
        TrackedProxy,
        make_tracked_class,
    )

    _HAS_MONITORING = True
except AttributeError:
    SyncMonitor = None  # type: ignore[assignment,misc]
    AccessTracer = None  # type: ignore[assignment,misc]
    AutoTracker = None  # type: ignore[assignment,misc]
    TrackedProxy = None  # type: ignore[assignment,misc]
    make_tracked_class = None  # type: ignore[assignment,misc]
    _HAS_MONITORING = False

T = TypeVar("T")


_engine: Engine | None = None
_sync_monitor: SyncMonitor | None = None
_auto_tracker: AutoTracker | None = None
_access_tracer: AccessTracer | None = None
_installed = False


def install() -> None:
    """
    Start pyft. Patches threading.Lock, RLock, and Thread lifecycle.
    """
    global _engine, _sync_patcher, _auto_tracker, _access_tracer, _installed
    if _installed:
        return

    _engine = Engine()
    _sync_monitor = SyncMonitor(_engine)
    _auto_tracker = AutoTracker(_engine)
    _access_tracer = AccessTracer(_engine)

    _sync_monitor.install()
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
    if _sync_monitor:
        _sync_monitor.uninstall()

    _installed = False


def track(obj: T) -> T:
    """
    Return a TrackedProxy wrapping `obj`. All attribute reads/writes
    on the returned proxy are monitored for races.

    The proxy is transparent: it forwards all attribute access to the
    underlying object.

    Example:
        shared = pyft.track(Counter())
        # Pass `shared` to threads; pyft will detect races on its attrs.
    """
    if _engine is None:
        raise RuntimeError("pyft.install() must be called before pyft.track()")
    return TrackedProxy(obj, _engine)  # type: ignore[return-value]


def track_class(cls: type) -> type:
    """
    Return a new class that is a subclass of `cls` with access tracking
    built in. All instances of the returned class are monitored.

    Example:
        @pyft.track_class
        class Counter:
            def __init__(self): self.value = 0
    """
    if _engine is None:
        raise RuntimeError(
            "pyft.install() must be called before pyft.track_class()"
        )
    return make_tracked_class(cls, _engine)


def report(file=None) -> None:
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
def context():
    """
    Context manager that installs pyft on enter and prints a race
    report + uninstalls on exit.

    Example:
        with pyft.context():
            run_concurrent_code()
    """
    install()
    try:
        yield
    finally:
        uninstall()
        report()


def detect(fn):
    """
    Decorator that runs a function under pyft and prints the race
    report when it returns.

    Example:
        @pyft.detect
        def test_concurrent():
            ...
    """

    def wrapper(*args, **kwargs):
        install()
        try:
            return fn(*args, **kwargs)
        finally:
            uninstall()
            report()

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper
