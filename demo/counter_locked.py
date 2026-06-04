"""
Same counter as counter_race.py, but every access is guarded by a single
threading.Lock. PyVFT should report NO races.

Run with:
    uv run python -m pyvft demo/counter_locked.py

Expected: 0 races. The lock establishes a happens-before chain between
the two threads' accesses to ``count``.
"""

from __future__ import annotations

import threading


class Counter:
    def __init__(self) -> None:
        self.count = 0


def worker(c: Counter, lock: threading.Lock, n: int) -> None:
    for _ in range(n):
        with lock:
            c.count += 1


def main() -> None:
    c = Counter()
    lock = threading.Lock()
    t1 = threading.Thread(target=worker, args=(c, lock, 500))
    t2 = threading.Thread(target=worker, args=(c, lock, 500))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(f"final count: {c.count}  (expected 1000)")


if __name__ == "__main__":
    main()
