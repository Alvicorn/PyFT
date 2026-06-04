"""
Each thread creates and uses its OWN object — no sharing, no race.

PyVFT must report NO races: a variable is only flagged when at least two
distinct threads touch it (the "two threads accessed it" rule).

Run with:
    uv run python -m pyvft demo/thread_local_safe.py

Expected: 0 races.
"""

from __future__ import annotations

import threading


class LocalBuf:
    def __init__(self) -> None:
        self.value = 0


def worker() -> None:
    buf = LocalBuf()
    for i in range(1000):
        buf.value = i
    assert buf.value == 999


def main() -> None:
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("done — every thread used its own buffer")


if __name__ == "__main__":
    main()
