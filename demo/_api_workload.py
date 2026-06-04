"""
Workload module for ``demo/api_usage.py``.

This module is intentionally separate so it can be imported INSIDE the
``pyvft.context()`` / ``@pyvft.detect`` scope — the AccessTracer import
hook then AST-rewrites it and every attribute access is traced.
"""

from __future__ import annotations

import threading


class Counter:
    def __init__(self) -> None:
        self.x = 0


class Account:
    def __init__(self) -> None:
        self.balance = 0


def race_three_writers(iterations: int = 100) -> Counter:
    """Three threads, no sync — expect a WRITE_WRITE race on ``x``."""
    c = Counter()

    def worker() -> None:
        for _ in range(iterations):
            c.x += 1

    ts = [threading.Thread(target=worker) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return c


def locked_deposits(iterations: int = 50) -> Account:
    """Four threads, all under a shared lock — expect 0 races."""
    acc = Account()
    lock = threading.Lock()

    def deposit() -> None:
        for _ in range(iterations):
            with lock:
                acc.balance += 1

    ts = [threading.Thread(target=deposit) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return acc
