"""
One thread writes, one thread reads, no synchronization → race.

PyVFT should report a READ_WRITE or WRITE_READ race on ``payload``.

Run with:
    uv run python -m pyvft demo/reader_writer_race.py

Expected: at least one race report.
"""

from __future__ import annotations

import threading
from typing import Optional


class Box:
    def __init__(self) -> None:
        self.payload: Optional[int] = None


def writer(b: Box) -> None:
    for i in range(200):
        b.payload = i


def reader(b: Box, out: list[Optional[int]]) -> None:
    for _ in range(200):
        out.append(b.payload)


def main() -> None:
    b = Box()
    seen: list[Optional[int]] = []
    t1 = threading.Thread(target=writer, args=(b,))
    t2 = threading.Thread(target=reader, args=(b, seen))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(
        f"reader saw {len(seen)} samples, last = {seen[-1] if seen else None}"
    )


if __name__ == "__main__":
    main()
