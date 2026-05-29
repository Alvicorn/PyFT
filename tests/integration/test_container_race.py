"""
Integration tests for container-method tracing. The workload lives in
``_container_workload.py`` so the import hook installed by
``pyft_session()`` can rewrite its ``lst.append`` / ``d.update`` calls
before they execute.
"""

from __future__ import annotations

import importlib
import sys
import threading

import pytest

from pyft.detector.race_log import RaceKind

from .helpers import assert_no_races, pyft_session


@pytest.mark.timeout(15)
@pytest.mark.parametrize("version", ["v1", "v2"])
class TestContainerRaceAcrossVersions:
    def test_unsync_list_append_races(self, version: str) -> None:
        with pyft_session(version=version) as (engine, _track):
            # Install the AccessTracer-style import hook explicitly so
            # the workload module is AST-rewritten.
            from pyft.instrument.transformer import AccessTracer

            tracer = AccessTracer(engine)
            tracer.install()
            try:
                sys.modules.pop("tests.integration._container_workload", None)
                wl = importlib.import_module(
                    "tests.integration._container_workload"
                )
                shared: list[int] = []
                wl.append_unsync(shared, n=80, n_threads=3)
            finally:
                tracer.uninstall()

            reports = engine.race_log.all_reports()
            container_races = [
                r for r in reports if r.access_a.attr == "__container__"
            ]
            assert container_races, (
                f"version={version}: expected at least one "
                f"__container__ race; got {[r.access_a.attr for r in reports]}"
            )

    def test_lock_protected_list_append_no_race(self, version: str) -> None:
        with pyft_session(version=version) as (engine, _track):
            from pyft.instrument.transformer import AccessTracer

            tracer = AccessTracer(engine)
            tracer.install()
            try:
                sys.modules.pop("tests.integration._container_workload", None)
                wl = importlib.import_module(
                    "tests.integration._container_workload"
                )
                shared: list[int] = []
                lock = threading.Lock()
                wl.append_locked(shared, lock, n=30, n_threads=3)
            finally:
                tracer.uninstall()
            assert_no_races(engine)

    def test_dict_update_vs_iterate_races(self, version: str) -> None:
        with pyft_session(version=version) as (engine, _track):
            from pyft.instrument.transformer import AccessTracer

            tracer = AccessTracer(engine)
            tracer.install()
            try:
                sys.modules.pop("tests.integration._container_workload", None)
                wl = importlib.import_module(
                    "tests.integration._container_workload"
                )
                d: dict[str, int] = {}
                wl.dict_update_vs_iterate(d, n=80)
            finally:
                tracer.uninstall()
            reports = engine.race_log.all_reports()
            assert any(r.access_a.attr == "__container__" for r in reports), (
                f"version={version}: expected at least one __container__ race"
            )

    def test_per_thread_local_lists_no_race(self, version: str) -> None:
        with pyft_session(version=version) as (engine, _track):
            from pyft.instrument.transformer import AccessTracer

            tracer = AccessTracer(engine)
            tracer.install()
            try:
                sys.modules.pop("tests.integration._container_workload", None)
                wl = importlib.import_module(
                    "tests.integration._container_workload"
                )
                wl.per_thread_local_lists(n=40, n_threads=4)
            finally:
                tracer.uninstall()
            assert_no_races(engine)


@pytest.mark.timeout(10)
def test_container_race_kind_is_write_write() -> None:
    """End-to-end smoke: under v2, the classic shared-list race must
    produce at least one WRITE/WRITE race on __container__."""
    with pyft_session(version="v2") as (engine, _track):
        from pyft.instrument.transformer import AccessTracer

        tracer = AccessTracer(engine)
        tracer.install()
        try:
            sys.modules.pop("tests.integration._container_workload", None)
            wl = importlib.import_module(
                "tests.integration._container_workload"
            )
            shared: list[int] = []
            wl.append_unsync(shared, n=80, n_threads=3)
        finally:
            tracer.uninstall()

        reports = engine.race_log.all_reports()
        ww = [
            r
            for r in reports
            if r.kind == RaceKind.WRITE_WRITE
            and r.access_a.attr == "__container__"
        ]
        assert ww, "expected at least one WRITE/WRITE race on __container__"
