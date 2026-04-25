"""
Shared helpers for pyft integration tests.

Each test creates a fresh Engine + SyncPatcher + AutoTracker so tests
are fully isolated. We use a helper context manager `pyft_session()`
that installs all components, runs the body, then uninstalls.
"""

from __future__ import annotations

import contextlib
import threading

from pyft.detector.engine import Engine
from pyft.instrument.wrappers import AutoTracker, TrackedProxy

try:
    from pyft.instrument.monitor import SyncMonitor

    _HAS_MONITORING = True
except AttributeError, ImportError:
    SyncMonitor = None  # type: ignore[assignment]
    _HAS_MONITORING = False
from pyft.detector.race_log import RaceKind


@contextlib.contextmanager
def pyft_session():
    engine = Engine()
    monitor = SyncMonitor(engine)
    tracker = AutoTracker(engine)

    monitor.install()
    tracker.install()
    try:

        def track(obj):
            return TrackedProxy(obj, engine)

        yield engine, track
    finally:
        tracker.uninstall()
        monitor.uninstall()


def run_threads(*fns, timeout=5.0):
    errors = []

    def wrap(fn):
        def run():
            try:
                fn()
            except Exception as e:
                errors.append(e)

        return run

    threads = [threading.Thread(target=wrap(fn)) for fn in fns]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=timeout)

    if errors:
        raise errors[0]


def count_races_of_kind(engine: Engine, kind: RaceKind) -> int:
    return sum(1 for r in engine.race_log.all_reports() if r.kind == kind)


def assert_no_races(engine: Engine):
    reports = engine.race_log.all_reports()
    if reports:
        kinds = [r.kind.name for r in reports]
        attrs = [r.access_a.attr for r in reports]
        raise AssertionError(
            f"Expected no races but found {len(reports)}: "
            f"{list(zip(kinds, attrs))}"
        )


def assert_race_on(engine: Engine, attr: str):
    reports = engine.race_log.all_reports()
    attrs = [r.access_a.attr for r in reports]
    if attr not in attrs:
        raise AssertionError(
            f"Expected race on attr '{attr}' but found races on: {attrs}"
        )
