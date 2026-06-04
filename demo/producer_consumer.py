"""
Producer/consumer using ``threading.Event``. The producer fills a buffer
and sets the event; the consumer waits and reads. Event.set/wait
establish happens-before, so PyVFT should report NO races.

Run with:
    uv run python -m pyvft demo/producer_consumer.py

Expected: 0 races.
"""

from __future__ import annotations

import threading


class Buffer:
    def __init__(self) -> None:
        self.data: list[int] = []


def producer(buf: Buffer, ready: threading.Event) -> None:
    buf.data = list(range(100))
    ready.set()  # HB: buf.data write → consumer's read


def consumer(buf: Buffer, ready: threading.Event, out: list[int]) -> None:
    ready.wait()  # HB acquires from producer's set
    out.append(sum(buf.data))


def main() -> None:
    buf = Buffer()
    ready = threading.Event()
    result: list = []
    t1 = threading.Thread(target=producer, args=(buf, ready))
    t2 = threading.Thread(target=consumer, args=(buf, ready, result))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(f"consumer computed sum = {result[0]}  (expected 4950)")


if __name__ == "__main__":
    main()
