"""
Shared helpers for pyvft integration tests.

Each test creates a fresh Engine + SyncPatcher + AutoTracker so tests
are fully isolated. We use a helper context manager `pyvft_session()`
that installs all components, runs the body, then uninstalls.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator

from pyvft.core.var_state import VFTVersion
from pyvft.detector.engine import Engine
from pyvft.detector.race_log import RaceKind
from pyvft.instrument.lock_patcher import LockPatcher
from pyvft.instrument.wrappers import AutoTracker, TrackedProxy


@contextlib.contextmanager
def pyvft_session(
    version: VFTVersion | str = VFTVersion.V2,
) -> Iterator[tuple[Engine, Callable[[object], TrackedProxy]]]:
    """
    Install LockPatcher + AutoTracker. The explicit monkey-patches in
    LockPatcher and AutoTracker cover all synchronization primitives
    the integration tests exercise (Lock, RLock, Semaphore,
    Thread.start, Thread.join).

    ``version`` chooses the VerifiedFT analyzer variant ("v1" or "v2").
    """
    engine = Engine(version=version)
    tracker = AutoTracker(engine)
    lock_patcher = LockPatcher(engine)

    lock_patcher.install()
    tracker.install()
    try:

        def track(obj: object) -> TrackedProxy:
            return TrackedProxy(obj, engine)

        yield engine, track
    finally:
        tracker.uninstall()
        lock_patcher.uninstall()


def run_threads(*fns: Callable[[], None], timeout: float = 5.0) -> None:
    errors: list[BaseException] = []

    def wrap(fn: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
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


def assert_no_races(engine: Engine) -> None:
    reports = engine.race_log.all_reports()
    if reports:
        kinds = [r.kind.name for r in reports]
        attrs = [r.access_a.attr for r in reports]
        raise AssertionError(
            f"Expected no races but found {len(reports)}: "
            f"{list(zip(kinds, attrs, strict=True))}"
        )


def assert_race_on(engine: Engine, attr: str) -> None:
    reports = engine.race_log.all_reports()
    attrs = [r.access_a.attr for r in reports]
    if attr not in attrs:
        raise AssertionError(
            f"Expected race on attr '{attr}' but found races on: {attrs}"
        )
