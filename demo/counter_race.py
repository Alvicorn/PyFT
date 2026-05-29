"""
Classic write/write race: two threads increment a shared counter with no
synchronization. PyFT should report a WRITE_WRITE race on ``count``.

Run with:
    uv run python -m pyft demo/counter_race.py

Expected: 1 race (WRITE_WRITE on 'count'). The auto-trace import hook
AST-rewrites this script so attribute access is intercepted automatically.
"""

from __future__ import annotations

import threading


class Counter:
    def __init__(self) -> None:
        self.count = 0


def worker(c: Counter, n: int) -> None:
    for _ in range(n):
        c.count += 1


def main() -> None:
    c = Counter()
    t1 = threading.Thread(target=worker, args=(c, 500))
    t2 = threading.Thread(target=worker, args=(c, 500))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(f"final count: {c.count}  (expected 1000, races likely lost some)")


if __name__ == "__main__":
    main()
