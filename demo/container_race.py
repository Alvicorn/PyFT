"""
Two scenarios for built-in container race detection.

1. ``list_race``: three threads append to a shared ``list`` with no
   synchronization. PyVFT should report at least one ``WRITE_WRITE``
   race on the synthetic ``__container__`` attribute.

2. ``dict_race``: one thread mutates a shared ``dict`` via ``update``
   while another iterates ``items()`` and a third drops keys via
   ``pop``. The mutator + iterator combination should surface at
   least one race on ``__container__``.

Run with:
    uv run python -m pyvft demo/container_race.py

Expected: races reported on ``__container__`` for both the list and
the dict scenarios.
"""

from __future__ import annotations

import threading


def list_race() -> None:
    """Three threads append concurrently — classic shared-list race."""
    shared: list[int] = []

    def appender() -> None:
        for i in range(500):
            shared.append(i)

    ts = [threading.Thread(target=appender) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(
        f"list_race  final len: {len(shared)}  "
        f"(expected 1500, races likely lost some)"
    )


def dict_race() -> None:
    """Three threads: one updates, one iterates, one pops — all
    racing on the same shared dict."""
    shared: dict[str, int] = {f"k{i}": i for i in range(50)}

    def updater() -> None:
        for i in range(300):
            shared.update({f"u{i}": i})

    def iterator() -> None:
        for _ in range(300):
            # ``items()`` is in our reader set; iterating it while
            # another thread mutates is the classic dict-mutated-during-
            # iteration race.
            for _k, _v in list(shared.items()):
                pass

    def popper() -> None:
        for i in range(100):
            shared.pop(f"k{i}", None)

    ts = [
        threading.Thread(target=updater),
        threading.Thread(target=iterator),
        threading.Thread(target=popper),
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(
        f"dict_race  final len: {len(shared)}  "
        f"(non-deterministic; depends on interleaving)"
    )


def main() -> None:
    list_race()
    dict_race()


if __name__ == "__main__":
    main()
