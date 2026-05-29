"""
Workload module for ``tests/integration/test_container_race.py``. Kept
separate so the AST import hook installed by ``pyft_session()`` can
AST-rewrite its method calls before any test code touches them.
"""

from __future__ import annotations

import threading
from typing import Any


def append_unsync(target: list[Any], n: int, n_threads: int = 3) -> None:
    """``n_threads`` workers each ``target.append`` ``n`` times — no sync."""

    def worker() -> None:
        for i in range(n):
            target.append(i)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def append_locked(
    target: list[Any], lock: threading.Lock, n: int, n_threads: int = 3
) -> None:
    """Same as ``append_unsync`` but every append is under ``lock``."""

    def worker() -> None:
        for i in range(n):
            with lock:
                target.append(i)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def dict_update_vs_iterate(d: dict[str, int], n: int = 50) -> None:
    """One thread updates ``d``, another iterates ``d.items()``."""

    stop = threading.Event()

    def updater() -> None:
        for i in range(n):
            d.update({f"k{i}": i})

    def iterator() -> None:
        for _ in range(n):
            list(d.items())

    t1 = threading.Thread(target=updater)
    t2 = threading.Thread(target=iterator)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    stop.set()


def per_thread_local_lists(n: int = 50, n_threads: int = 4) -> None:
    """Each thread builds its OWN list — no sharing."""

    def worker() -> None:
        local: list[int] = []
        for i in range(n):
            local.append(i)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
