"""
Demonstrates the explicit PyFT API: ``pyft.context()`` and
``@pyft.detect``. Use these when you don't want to run under
``python -m pyft`` and instead embed the detector around a specific
block of code.

The key idea: any module imported INSIDE the context / decorated
function is AST-rewritten by the AccessTracer import hook, so attribute
access in that module gets traced. The workload lives in
``demo/_workload.py`` for exactly that reason.

Run with (no ``-m pyft`` needed because the script calls
``pyft.context()`` / ``@pyft.detect`` itself):

    uv run python demo/api_usage.py

Expected output:
    - one WRITE_WRITE race from the @pyft.detect block
    - zero races from the lock-protected context() block
"""

from __future__ import annotations

import pyft

# --- 1. @pyft.detect: instrument a single function -------------------


@pyft.detect
def race_with_counter() -> None:
    # Import INSIDE the decorated function so the import hook (now
    # active) catches the workload module and AST-rewrites it.
    import _api_workload

    _api_workload.race_three_writers(iterations=100)


# --- 2. pyft.context(): manual scope ---------------------------------


def lock_protected_inside_context() -> None:
    with pyft.context():
        import _api_workload  # imported under the hook

        _api_workload.locked_deposits(iterations=50)


def main() -> None:
    print("=== 1) race_with_counter (expect 1 race) ===")
    race_with_counter()
    print()
    print("=== 2) lock_protected_inside_context (expect 0 races) ===")
    lock_protected_inside_context()


if __name__ == "__main__":
    main()
