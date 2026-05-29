"""
Parent writes BEFORE forking a child, then joins; parent reads AFTER
join. Both the fork edge and the join edge establish happens-before.

Run with:
    uv run python -m pyft demo/fork_join_safe.py

Expected: 0 races. Demonstrates that fork and join correctly propagate
HB through the engine's thread_start / thread_join events.
"""

from __future__ import annotations

import threading
from typing import Optional


class Mailbox:
    def __init__(self) -> None:
        self.message: Optional[str] = None
        self.reply: Optional[str] = None


def child(m: Mailbox) -> None:
    # safe: parent wrote `message` BEFORE start; fork HB makes it visible
    received = m.message
    m.reply = f"got: {received}"


def main() -> None:
    m = Mailbox()
    m.message = "hello child"  # written before fork

    t = threading.Thread(target=child, args=(m,))
    t.start()
    t.join()  # establishes HB: child's write → parent's read

    # safe: t.join() means child's writes happen-before this read
    print(f"parent saw reply: {m.reply!r}")


if __name__ == "__main__":
    main()
