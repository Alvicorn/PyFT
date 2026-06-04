"""
PyVFT benchmark driver.

For each (workload, config) cell:
  * 1 warm-up subprocess
  * N measured subprocesses  (default N=5).

Each subprocess runs the workload script under one of three configs:

  uninstr  python <workload>.py
  v1       python -m pyvft --version v1 <workload>.py
  v2       python -m pyvft --version v2 <workload>.py

While the subprocess is alive the driver samples its resident set size
at 50 ms via psutil to build a memory-over-time curve.

Headline timing is ``wall_s`` -- the full subprocess lifetime, including
Python startup, the PyVFT import hook + AST rewrite, execution, race
report formatting, and exit. AST-rewrite cost is therefore always
included in the measured overhead.

Run order within each workload is shuffled (seeded) so a thermal or
I/O trend can't bias one config.

Outputs:

  results.md
  plots/
    wall_time_per_benchmark.png
    peak_memory_per_benchmark.png
    memory_over_time/<workload>.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIGS = ("uninstr", "v1", "v2")
WARMUP_RUNS = 1
SAMPLE_INTERVAL_S = 0.050

ROOT = Path(__file__).resolve().parent
WORKLOADS_DIR = ROOT / "python-threading-benchmarks" / "workloads"


##########
# MODELS #
##########


@dataclass
class RunResult:
    config: str
    workload: str
    warmup: bool
    time_s: (
        float  # workload execution (from BENCH_RESULT); shown in live status
    )
    wall_s: float  # full subprocess lifetime (driver-measured)
    peak_mb_external: float  # driver-sampled max resident set size
    samples_n: int
    ok: bool
    stderr_tail: str = ""
    # (t, MB) RSS samples used by the memory-over-time plot.
    samples: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class CellSummary:
    wall_s_mean: float
    peak_mb_mean: float


@dataclass
class WorkloadSummary:
    cells: dict[str, CellSummary] = field(default_factory=dict)

    def overhead(self, cfg: str) -> float | None:
        if "uninstr" not in self.cells or cfg not in self.cells:
            return None
        base = self.cells["uninstr"].wall_s_mean
        if base <= 0 or not math.isfinite(base):
            return None
        cell = self.cells[cfg].wall_s_mean
        if not math.isfinite(cell):
            return None
        return (cell - base) / base

    def mem_ratio(self, cfg: str) -> float | None:
        if "uninstr" not in self.cells or cfg not in self.cells:
            return None
        base = self.cells["uninstr"].peak_mb_mean
        if base <= 0 or not math.isfinite(base):
            return None
        cell = self.cells[cfg].peak_mb_mean
        if not math.isfinite(cell):
            return None
        return cell / base


#############
# DISCOVERY #
#############


def discover_workloads() -> list[str]:
    names = []
    for p in sorted(WORKLOADS_DIR.glob("*.py")):
        if p.name in ("__init__.py", "common.py"):
            continue
        names.append(p.stem)
    return names


##########
# RUNNER #
##########


def parse_bench_result(stdout: str) -> dict[str, Any]:
    """Return the JSON payload of the last BENCH_RESULT line in stdout."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("BENCH_RESULT "):
            try:
                return json.loads(line[len("BENCH_RESULT ") :])
            except json.JSONDecodeError as e:
                raise ValueError(f"malformed BENCH_RESULT JSON: {e}") from e
    raise ValueError("no BENCH_RESULT line in workload output")


def build_cmd(workload: str, config: str) -> list[str]:
    script = str(WORKLOADS_DIR / f"{workload}.py")
    if config == "uninstr":
        return [sys.executable, script]
    if config in ("v1", "v2"):
        return [sys.executable, "-m", "pyvft", "--version", config, script]
    raise ValueError(f"unknown config: {config}")


def _sample_rss_loop(
    proc: subprocess.Popen[str],
    samples: list[tuple[float, float]],
    t0: float,
    interval_s: float,
    stop_event: threading.Event,
) -> None:
    """
    Poll resident set size until the subprocess exits or stop_event is set.
    """
    import psutil

    try:
        ps = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return
    while not stop_event.is_set():
        try:
            rss = ps.memory_info().rss
            for child in ps.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            samples.append((time.perf_counter() - t0, rss / (1024 * 1024)))
        except psutil.NoSuchProcess:
            return
        if proc.poll() is not None:
            return
        time.sleep(interval_s)


def run_one(
    workload: str, config: str, warmup: bool, smoke: bool, timeout_s: float
) -> RunResult:
    env = os.environ.copy()
    if smoke:
        env["BENCH_SMOKE"] = "1"
    cmd = build_cmd(workload, config)

    samples: list[tuple[float, float]] = []
    stop_event = threading.Event()
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    sampler = threading.Thread(
        target=_sample_rss_loop,
        args=(proc, samples, t0, SAMPLE_INTERVAL_S, stop_event),
        daemon=True,
    )
    sampler.start()

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "TIMEOUT and kill failed"
    wall = time.perf_counter() - t0
    stop_event.set()
    sampler.join(timeout=1.0)

    peak_ext = max((y for _, y in samples), default=0.0)

    if timed_out:
        return RunResult(
            config=config,
            workload=workload,
            warmup=warmup,
            time_s=float("nan"),
            wall_s=wall,
            peak_mb_external=peak_ext,
            samples_n=len(samples),
            ok=False,
            stderr_tail=f"TIMEOUT after {timeout_s:.0f}s",
            samples=samples,
        )
    if proc.returncode != 0:
        return RunResult(
            config=config,
            workload=workload,
            warmup=warmup,
            time_s=float("nan"),
            wall_s=wall,
            peak_mb_external=peak_ext,
            samples_n=len(samples),
            ok=False,
            stderr_tail=(stderr or "")[-400:],
            samples=samples,
        )
    try:
        payload = parse_bench_result(stdout)
    except ValueError as e:
        return RunResult(
            config=config,
            workload=workload,
            warmup=warmup,
            time_s=float("nan"),
            wall_s=wall,
            peak_mb_external=peak_ext,
            samples_n=len(samples),
            ok=False,
            stderr_tail=f"{e}; stderr tail: {(stderr or '')[-200:]}",
            samples=samples,
        )
    return RunResult(
        config=config,
        workload=workload,
        warmup=warmup,
        time_s=float(payload["time_s"]),
        wall_s=wall,
        peak_mb_external=peak_ext,
        samples_n=len(samples),
        ok=True,
        samples=samples,
    )


#############
# AGGREGATE #
#############


def aggregate(runs: list[RunResult]) -> dict[str, WorkloadSummary]:
    """
    Per-(workload, config) means. Warm-up runs excluded.
    """
    by_cell: dict[tuple[str, str], list[RunResult]] = {}
    for r in runs:
        if r.warmup:
            continue
        by_cell.setdefault((r.workload, r.config), []).append(r)
    summaries: dict[str, WorkloadSummary] = {}
    for (workload, config), rs in by_cell.items():
        ok = [r for r in rs if r.ok]
        if not ok:
            cell = CellSummary(
                wall_s_mean=float("nan"), peak_mb_mean=float("nan")
            )
        else:
            walls = [r.wall_s for r in ok]
            peaks = [r.peak_mb_external for r in ok]
            cell = CellSummary(
                wall_s_mean=statistics.fmean(walls),
                peak_mb_mean=statistics.fmean(peaks),
            )
        summaries.setdefault(workload, WorkloadSummary()).cells[config] = cell
    return summaries


def geomean_overhead(xs: list[float]) -> float:
    finite = [x for x in xs if math.isfinite(x) and x > -1.0]
    if not finite:
        return float("nan")
    s = sum(math.log(1.0 + x) for x in finite)
    return math.exp(s / len(finite)) - 1.0


def geomean_ratio(xs: list[float]) -> float:
    finite = [x for x in xs if math.isfinite(x) and x > 0]
    if not finite:
        return float("nan")
    s = sum(math.log(x) for x in finite)
    return math.exp(s / len(finite))


###############
# SYSTEM INFO #
###############


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def capture_system_info() -> dict[str, Any]:
    try:
        gil_on = sys._is_gil_enabled()  # type: ignore[attr-defined]
    except AttributeError:
        gil_on = True
    gil_tag = "GIL-enabled" if gil_on else "free-threaded"

    info: dict[str, Any] = {
        "python_version": f"{platform.python_version()} ({gil_tag})",
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pyvft_git_sha": _git_sha(ROOT.parent),
        "workloads_git_sha": _git_sha(WORKLOADS_DIR.parent),
        "num_threads_setting": "8 threads",
        "sample_interval_s": f"{SAMPLE_INTERVAL_S} s",
    }
    try:
        import psutil

        info["cpu_count_logical"] = f"{psutil.cpu_count(logical=True)} cores"
        info["cpu_count_physical"] = f"{psutil.cpu_count(logical=False)} cores"
        info["total_ram_gb"] = (
            f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB"
        )
    except Exception:
        pass
    return info


##########
# OUTPUT #
##########


def _fr(x: float | None) -> str:
    return "-" if x is None or not math.isfinite(x) else f"{x:.2f}x"


def _overhead_as_ratio(o: float | None) -> str:
    """
    Format a relative overhead as a slowdown ratio (e.g. ``2.57x``).
    """
    if o is None or not math.isfinite(o):
        return "-"
    return _fr(1.0 + o)


def format_summary_table(summaries: dict[str, WorkloadSummary]) -> str:
    cols = [
        "Workload",
        "uninstr wall (s)",
        "v1 wall (s)",
        "v2 wall (s)",
        "v1 wall ratio",
        "v2 wall ratio",
        "v1 peak resident set size ratio",
        "v2 peak resident set size ratio",
    ]
    rows = []
    for name in sorted(summaries):
        s = summaries[name]

        def t(c: str, s: WorkloadSummary = s) -> str:
            return f"{s.cells[c].wall_s_mean:.3f}" if c in s.cells else "-"

        rows.append(
            [
                name,
                t("uninstr"),
                t("v1"),
                t("v2"),
                _overhead_as_ratio(s.overhead("v1")),
                _overhead_as_ratio(s.overhead("v2")),
                _fr(s.mem_ratio("v1")),
                _fr(s.mem_ratio("v2")),
            ]
        )
    # geomean row
    g_v1 = geomean_overhead(
        [o for s in summaries.values() if (o := s.overhead("v1")) is not None]
    )
    g_v2 = geomean_overhead(
        [o for s in summaries.values() if (o := s.overhead("v2")) is not None]
    )
    gm_v1 = geomean_ratio(
        [
            mr
            for s in summaries.values()
            if (mr := s.mem_ratio("v1")) is not None
        ]
    )
    gm_v2 = geomean_ratio(
        [
            mr
            for s in summaries.values()
            if (mr := s.mem_ratio("v2")) is not None
        ]
    )
    rows.append(["-" * 8] * len(cols))
    rows.append(
        [
            "GEOMEAN",
            "-",
            "-",
            "-",
            f"{1 + g_v1:.2f}x" if math.isfinite(g_v1) else "-",
            f"{1 + g_v2:.2f}x" if math.isfinite(g_v2) else "-",
            f"{gm_v1:.2f}x" if math.isfinite(gm_v1) else "-",
            f"{gm_v2:.2f}x" if math.isfinite(gm_v2) else "-",
        ]
    )
    widths = [
        max(len(str(r[i])) for r in (rows + [cols])) for i in range(len(cols))
    ]
    out = []
    out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(cols)))
    out.append("  ".join("-" * widths[i] for i in range(len(cols))))
    for r in rows:
        out.append(
            "  ".join(str(r[i]).ljust(widths[i]) for i in range(len(cols)))
        )
    return "\n".join(out)


#########
# PLOTS #
#########


def _resample(
    samples: list[tuple[float, float]], t_grid: list[float]
) -> list[float]:
    """
    Linear interpolation samples onto t_grid.
    """
    if not samples:
        return [float("nan")] * len(t_grid)
    out: list[float] = []
    i = 0
    n = len(samples)
    for t in t_grid:
        while i + 1 < n and samples[i + 1][0] <= t:
            i += 1
        if i + 1 >= n:
            out.append(samples[-1][1])
            continue
        t0, y0 = samples[i]
        t1, y1 = samples[i + 1]
        if t <= t0:
            out.append(y0)
        elif t1 == t0:
            out.append(y0)
        else:
            frac = (t - t0) / (t1 - t0)
            out.append(y0 + (y1 - y0) * frac)
    return out


def render_plots(
    summaries: dict[str, WorkloadSummary], runs: list[RunResult], out_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(summaries)
    n = len(names)
    x = list(range(n))
    width = 0.27
    colours = ("#888888", "#d95f02", "#1b9e77")  # uninstr / v1 / v2

    def bars(ax: plt.Axes, attr: str) -> None:
        for i, c in enumerate(CONFIGS):
            ys = []
            for name in names:
                s = summaries[name]
                if c in s.cells and math.isfinite(getattr(s.cells[c], attr)):
                    ys.append(getattr(s.cells[c], attr))
                else:
                    ys.append(0.0)
            ax.bar(
                [xi + (i - 1) * width for xi in x],
                ys,
                width=width,
                label=c,
                color=colours[i],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    # 1. Wall time per benchmark (log y)
    fig, ax = plt.subplots(figsize=(13, 6))
    bars(ax, "wall_s_mean")
    ax.set_yscale("log")
    ax.set_ylabel("mean wall_s (log)")
    ax.set_title(
        "Wall time per benchmark (mean of measured runs; includes AST rewrite)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "wall_time_per_benchmark.png", dpi=140)
    plt.close(fig)

    # 2. Peak memory per benchmark
    fig, ax = plt.subplots(figsize=(13, 6))
    bars(ax, "peak_mb_mean")
    ax.set_yscale("log")
    ax.set_ylabel("peak resident set size (MB, log)")
    ax.set_title("Peak memory per benchmark (driver-sampled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "peak_memory_per_benchmark.png", dpi=140)
    plt.close(fig)

    # 3. Per-workload memory-over-time (sourced from RunResult.samples in memory)
    mot_dir = out_dir / "memory_over_time"
    mot_dir.mkdir(parents=True, exist_ok=True)
    by_cell_ts: dict[tuple[str, str], list[list[tuple[float, float]]]] = {}
    for r in runs:
        if r.warmup or not r.ok:
            continue
        by_cell_ts.setdefault((r.workload, r.config), []).append(r.samples)

    for name in names:
        fig, ax = plt.subplots(figsize=(10, 5))
        any_plotted = False
        for c, col in zip(CONFIGS, colours, strict=True):
            runs_ts = by_cell_ts.get((name, c), [])
            if not runs_ts:
                continue
            max_t = max((s[-1][0] for s in runs_ts if s), default=0.0)
            if max_t <= 0:
                continue
            steps = max(2, int(max_t / SAMPLE_INTERVAL_S) + 1)
            t_grid = [i * (max_t / (steps - 1)) for i in range(steps)]
            grid_samples: list[list[float]] = []
            for s in runs_ts:
                grid_samples.append(_resample(s, t_grid))
            # mean across runs (per grid point)
            mean = [
                statistics.fmean([g[i] for g in grid_samples])
                for i in range(steps)
            ]
            lo = [min(g[i] for g in grid_samples) for i in range(steps)]
            hi = [max(g[i] for g in grid_samples) for i in range(steps)]
            ax.plot(t_grid, mean, color=col, label=f"{c} (mean)")
            ax.fill_between(t_grid, lo, hi, color=col, alpha=0.18)
            any_plotted = True
        if not any_plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("seconds since subprocess start")
        ax.set_ylabel("resident set size (MB)")
        ax.set_title(f"Memory over time: {name}")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(mot_dir / f"{name}.png", dpi=140)
        plt.close(fig)


############
# MARKDOWN #
############


def write_markdown(
    out_path: Path,
    system_info: dict[str, Any],
    summaries: dict[str, WorkloadSummary],
    plots_dir: Path,
    results_dir: Path,
) -> None:
    plots_rel = os.path.relpath(plots_dir, results_dir).replace("\\", "/")

    def link(p: str) -> str:
        return f"{plots_rel}/{p}"

    lines: list[str] = []
    lines.append("# PyVFT benchmark results")
    lines.append("")
    lines.append("## System info")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("| --- | --- |")
    for k, v in system_info.items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")

    # geomean
    g_v1 = geomean_overhead(
        [o for s in summaries.values() if (o := s.overhead("v1")) is not None]
    )
    g_v2 = geomean_overhead(
        [o for s in summaries.values() if (o := s.overhead("v2")) is not None]
    )
    gm_v1 = geomean_ratio(
        [
            mr
            for s in summaries.values()
            if (mr := s.mem_ratio("v1")) is not None
        ]
    )
    gm_v2 = geomean_ratio(
        [
            mr
            for s in summaries.values()
            if (mr := s.mem_ratio("v2")) is not None
        ]
    )

    lines.append("## Geomean overhead (all workloads)")
    lines.append("")
    lines.append("| Metric | v1 | v2 |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Wall-time ratio | {_overhead_as_ratio(g_v1)} | {_overhead_as_ratio(g_v2)} |"
    )
    lines.append(f"| Peak-memory ratio | {_fr(gm_v1)} | {_fr(gm_v2)} |")
    lines.append("")

    # headline table
    lines.append("## Per-workload summary")
    lines.append("")
    lines.append(
        "| Workload | uninstr wall (s) | v1 wall (s) | v2 wall (s) | "
        "v1 wall ratio | v2 wall ratio | "
        "v1 peak resident set size ratio | v2 peak resident set size ratio |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in sorted(summaries):
        s = summaries[name]

        def t(c: str, s: WorkloadSummary = s) -> str:
            return f"{s.cells[c].wall_s_mean:.3f}" if c in s.cells else "-"

        lines.append(
            f"| {name} | {t('uninstr')} | {t('v1')} | {t('v2')} | "
            f"{_overhead_as_ratio(s.overhead('v1'))} | "
            f"{_overhead_as_ratio(s.overhead('v2'))} | "
            f"{_fr(s.mem_ratio('v1'))} | {_fr(s.mem_ratio('v2'))} |"
        )
    lines.append(
        f"| **GEOMEAN** | - | - | - | "
        f"**{_overhead_as_ratio(g_v1)}** | **{_overhead_as_ratio(g_v2)}** | "
        f"**{_fr(gm_v1)}** | **{_fr(gm_v2)}** |"
    )
    lines.append("")

    # summary plots
    lines.append("## Summary plots")
    lines.append("")
    for f, caption in [
        ("wall_time_per_benchmark.png", "Wall time per benchmark (log y)"),
        (
            "peak_memory_per_benchmark.png",
            "Peak resident set size per benchmark (log y)",
        ),
    ]:
        lines.append(f"### {caption}")
        lines.append("")
        lines.append(f"![{caption}]({link(f)})")
        lines.append("")

    # per-workload memory-over-time
    lines.append("## Memory-over-time per workload")
    lines.append("")
    for name in sorted(summaries):
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"![{name} memory over time]({link(f'memory_over_time/{name}.png')})"
        )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


########
# MAIN #
########


def build_schedule(
    workloads: list[str],
    configs: tuple[str, ...],
    runs: int,
    warmup: int,
    seed: int,
) -> list[tuple[str, str, int, bool]]:
    """
    For each workload, shuffle the (config, run_index) order with the seed.
    Warm-ups are scheduled first per workload, then the shuffled measured runs.
    """
    rng = random.Random(seed)
    schedule: list[tuple[str, str, int, bool]] = []
    for w in workloads:
        # warm-ups: one per config (in fixed order so the warm-up cost is the
        # same for all three regardless of seed)
        for c in configs:
            for i in range(warmup):
                schedule.append((w, c, i, True))
        measured = [(c, i) for c in configs for i in range(runs)]
        rng.shuffle(measured)
        for c, i in measured:
            schedule.append((w, c, i, False))
    return schedule


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PyVFT benchmark driver")
    p.add_argument(
        "--runs",
        type=int,
        default=5,
        help="measured runs per cell (warm-up is always 1 on top)",
    )
    p.add_argument(
        "--configs",
        default=",".join(CONFIGS),
        help="comma-separated configs from {uninstr,v1,v2}",
    )
    p.add_argument(
        "--benchmarks",
        default="",
        help="comma-separated workload names (default all)",
    )
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--plots-dir", default=str(ROOT / "plots"))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="set BENCH_SMOKE=1 in workloads (small problem sizes)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="per-subprocess timeout in seconds",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0xC0FFEE,
        help="run-order shuffle seed (for reproducibility)",
    )
    args = p.parse_args(argv)

    configs = tuple(c.strip() for c in args.configs.split(",") if c.strip())
    for c in configs:
        if c not in CONFIGS:
            p.error(f"unknown config: {c}")

    all_w = discover_workloads()
    if args.benchmarks.strip():
        wanted = [w.strip() for w in args.benchmarks.split(",") if w.strip()]
        unknown = [w for w in wanted if w not in all_w]
        if unknown:
            p.error(f"unknown benchmark(s): {unknown}; have: {all_w}")
        workloads = wanted
    else:
        workloads = all_w

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    system_info = capture_system_info()
    schedule = build_schedule(
        workloads, configs, args.runs, WARMUP_RUNS, args.seed
    )
    total = len(schedule)
    print("PyVFT benchmark sweep")
    print(f"  workloads: {len(workloads)} ({', '.join(workloads)})")
    print(f"  configs:   {', '.join(configs)}")
    print(
        f"  per cell:  {WARMUP_RUNS} warm-up + {args.runs} measured = "
        f"{WARMUP_RUNS + args.runs} subprocesses"
    )
    print(f"  total:     {total} subprocesses")
    print(f"  smoke:     {args.smoke}")
    print(f"  seed:      0x{args.seed:x}")
    print(f"  python:    {system_info['python_version']}")
    print()

    runs: list[RunResult] = []
    for idx, (w, c, i, warm) in enumerate(schedule, 1):
        t_cell = time.perf_counter()
        r = run_one(w, c, warm, smoke=args.smoke, timeout_s=args.timeout)
        runs.append(r)
        tag = "warm" if warm else "meas"
        status = "ok  " if r.ok else "FAIL"
        print(
            f"  [{idx:>3}/{total}] {w:<22} {c:<8} {tag} run {i:<2} "
            f"{status} wall={r.wall_s:6.2f}s time={r.time_s:6.2f}s "
            f"peak={r.peak_mb_external:6.1f}MB samples={r.samples_n:<4} "
            f"({time.perf_counter() - t_cell:5.1f}s)"
        )
        if not r.ok:
            print(f"     ! {r.stderr_tail.strip()[:200]}")

    summaries = aggregate(runs)

    print()
    print("=" * 80)
    print(format_summary_table(summaries))
    print("=" * 80)

    if not args.no_plots:
        try:
            render_plots(summaries, runs, plots_dir)
            print(f"Wrote: plots in {plots_dir}")
        except Exception as e:
            print(f"plot rendering failed: {e}", file=sys.stderr)

    write_markdown(
        results_dir / "results.md",
        system_info,
        summaries,
        plots_dir,
        results_dir,
    )
    print(f"Wrote: {results_dir / 'results.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
