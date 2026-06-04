"""
python -m pyvft myscript.py [args...]

Runs ``myscript.py`` under the PyVFT race detector and prints a race
report to stderr when the script exits. The entry script is
AST-transformed before exec, and every module it subsequently imports
goes through the AccessTracer import hook installed by ``pyvft.install``.

Examples:
  python -m pyvft my_threaded_script.py
  python -m pyvft -h
"""

from __future__ import annotations

import argparse
import os
import sys
import types


def _run_script_instrumented(script_path: str) -> None:
    """Read ``script_path``, AST-transform it, and exec it as ``__main__``."""
    from .instrument.import_hook import transform_source

    with open(script_path, "r", encoding="utf-8") as fh:
        source = fh.read()
    tree = transform_source(source, script_path)
    code = compile(tree, script_path, "exec")

    main_mod = types.ModuleType("__main__")
    main_mod.__file__ = script_path
    main_mod.__builtins__ = __builtins__  # type: ignore[attr-defined]
    sys.modules["__main__"] = main_mod
    exec(code, main_mod.__dict__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyvft",
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
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v2",
        help="VerifiedFT analyser variant: v1 (full vector clocks) or "
        "v2 (epoch-compressed FastTrack, default)",
    )

    args = parser.parse_args(argv)

    sys.argv = [args.script] + args.script_args

    script_dir = os.path.dirname(os.path.abspath(args.script))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import pyvft

    pyvft.install(version=args.version)

    exit_code = 0
    try:
        _run_script_instrumented(args.script)
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 0
    except Exception as e:
        print(f"\npyvft: script raised an exception: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        pyvft.uninstall()
        from pyvft import get_engine
        from pyvft.report.formatter import format_summary

        engine = get_engine()
        if engine:
            use_colour = not args.no_colour
            print(
                format_summary(engine.race_log, use_colour=use_colour),
                file=sys.stderr,
            )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
