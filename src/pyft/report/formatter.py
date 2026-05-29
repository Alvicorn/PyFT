from __future__ import annotations

import sys
from typing import IO

from ..detector.race_log import RaceLog, RaceReport
from .race import RaceKind

_WIDTH = 62
_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_DIM = "\033[2m"


def _colour(text: str, code: str, use_colour: bool) -> str:
    return f"{code}{text}{_RESET}" if use_colour else text


def _kind_label(kind: RaceKind) -> str:
    return {
        RaceKind.WRITE_WRITE: "WRITE / WRITE",
        RaceKind.READ_WRITE: "READ / WRITE",
        RaceKind.WRITE_READ: "WRITE / READ",
    }.get(kind, str(kind))


def format_report(report: RaceReport, use_colour: bool = True) -> str:
    lines: list[str] = []

    kind_str = _kind_label(report.kind)
    header = f"── Race #{report.sequence}: {kind_str} "
    header = header + "─" * max(0, _WIDTH - len(header))
    lines.append(_colour(header, _BOLD + _RED, use_colour))

    lines.append(f"  Object : {report.access_a.obj_repr}")
    lines.append(f"  Attr   : {report.access_a.attr}")
    lines.append("")

    for label, acc in [
        ("Access A", report.access_a),
        ("Access B", report.access_b),
    ]:
        rw = "WRITE" if acc.is_write else "READ"
        rw_col = _colour(rw, _BOLD + _YELLOW, use_colour)
        tid_str = _colour(
            f"{acc.thread_name} (tid={acc.tid}, clock={acc.clock})",
            _CYAN,
            use_colour,
        )
        lines.append(f"  {label}  [{rw_col}]  {tid_str}")

        if acc.stack:
            frames = [f.rstrip() for f in acc.stack if f.strip()]
            for frame in frames[-3:]:
                for sub in frame.splitlines():
                    lines.append(
                        f"    {_colour(sub.strip(), _DIM, use_colour)}"
                    )
        else:
            lines.append(
                _colour(
                    "    (stack not available for prior access)",
                    _DIM,
                    use_colour,
                )
            )
        lines.append("")

    footer = "─" * _WIDTH
    lines.append(_colour(footer, _DIM, use_colour))
    return "\n".join(lines)


def format_summary(log: RaceLog, use_colour: bool = True) -> str:
    reports = log.all_reports()
    lines: list[str] = []

    sep = "═" * _WIDTH
    if not reports:
        lines.append(_colour(sep, _BOLD, use_colour))
        lines.append(
            _colour("  pyft: NO DATA RACES DETECTED ", _BOLD, use_colour)
        )
        lines.append(_colour(sep, _BOLD, use_colour))
        return "\n".join(lines)

    n = len(reports)
    noun = "RACE" if n == 1 else "RACES"
    lines.append(_colour(sep, _BOLD + _RED, use_colour))
    lines.append(
        _colour(f"  pyft: {n} DATA {noun} DETECTED", _BOLD + _RED, use_colour)
    )
    lines.append(_colour(sep, _BOLD + _RED, use_colour))
    lines.append("")

    for report in reports:
        lines.append(format_report(report, use_colour=use_colour))
        lines.append("")

    return "\n".join(lines)


def print_summary(
    log: RaceLog, file: IO[str] | None = None, use_colour: bool | None = None
) -> None:
    if file is None:
        file = sys.stderr
    if use_colour is None:
        use_colour = hasattr(file, "isatty") and file.isatty()
    if use_colour is None:
        use_colour = False
    print(format_summary(log, use_colour=use_colour), file=file)
