"""
python -m pyft myscript.py [args...]

Runs myscript.py under the pyft race detector and prints a race
report to stderr when the script exits.

Examples:
  python -m pyft my_threaded_script.py
  python -m pyft -h
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyft",
        description="VerifiedFT data race detector for free-threaded Python",
    )
    parser.add_argument(
        "script", help="Python script to run under the race detector"
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the script",
    )
    parser.add_argument(
        "--no-colour",
        action="store_true",
        help="Disable ANSI colour in output",
    )

    args = parser.parse_args(argv)

    # Set up sys.argv for the target script
    sys.argv = [args.script] + args.script_args

    # Add script's directory to sys.path (same as running it directly)
    script_dir = os.path.dirname(os.path.abspath(args.script))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Install the detector
    import pyft

    pyft.install()

    exit_code = 0
    try:
        runpy.run_path(args.script, run_name="__main__")
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 0
    except Exception as e:
        print(f"\npyft: script raised an exception: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        pyft.uninstall()
        # Always print the report
        import sys as _sys

        from pyft import get_engine
        from pyft.report.formatter import format_summary

        engine = get_engine()
        if engine:
            use_colour = not args.no_colour
            print(
                format_summary(engine.race_log, use_colour=use_colour),
                file=_sys.stderr,
            )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
