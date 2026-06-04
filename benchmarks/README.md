# pyFT Benchmarking

Benchmark suite for `pyFT` using
[python-threading-benchmarks](https://github.com/Alvicorn/python-threading-benchmarks)
(checked out as a git submodule under `python-threading-benchmarks/`).

For each of the 24 workloads in the submodule (8 threads, set via the
submodule's `common.NUM_THREADS`), the driver runs:

- **1 warm-up** subprocess per `(workload, config)` cell
- **N measured** subprocesses per cell (default `N=5`, controlled by
  `--runs`).

across three configurations:

- `uninstr` — `python <workload>.py`
- `v1` — `python -m pyft --version v1 <workload>.py`
- `v2` — `python -m pyft --version v2 <workload>.py`

While each subprocess is alive, the driver samples its resident-set size
at 50 ms (`psutil.Process(pid).memory_info().rss`, plus children) to
build a memory-over-time curve.

**Headline timing is `wall_s`** — the full subprocess lifetime, including
Python startup, the PyFT import hook + AST rewrite, execution, race
report formatting, and exit. AST-rewrite cost is therefore always
included in the measured overhead.

Run order within each workload is shuffled with a seeded RNG so a
thermal/I-O trend can't bias one config; the seed is printed at start.

A geometric mean of overheads is reported across all selected workloads.

## Layout

```
run_pyft_benchmarks.py
python-threading-benchmarks
results/
plots/
```

## Requirements

- Free-threaded CPython 3.14+freethread
- `uv` for environment management.
  ```sh
  uv sync --group benchmark # install psutil and matplotlib
  ```

## Quick start

```sh
# From the root directory of the pyft project, using the parent PyFT venv:
uv sync --group benchmark

uv run python run_pyft_benchmarks.py

# Smoke-size run (workloads scale problem sizes ~10x down):
uv run --project .. python run_pyft_benchmarks.py --smoke
```

## Output

- `results/results.md`
- `results/plots/wall_time_per_benchmark.png`
- `results/plots/peak_memory_per_benchmark.png`
- `results/plots/memory_over_time/<workload>.png` (one per workload, showing
   uninstrumented/v1/v2 of mean RSS across runs with shaded min/max bands)

## Driver CLI

```
run_pyft_benchmarks.py [-h] [--runs N] [--configs LIST] [--benchmarks LIST]
                       [--results-dir DIR] [--plots-dir DIR]
                       [--no-plots] [--smoke] [--timeout S] [--seed N]
```

- `--runs` — measured runs per `(workload, config)` cell. Default 5.
  The driver always runs 1 warm-up on top.
- `--configs` — comma-separated subset of `uninstr,v1,v2`. Default all.
- `--benchmarks` — comma-separated subset of workload names. Default all.
- `--smoke` — set `BENCH_SMOKE=1` in the workload environment so each
  workload scales its problem size down ~10x.
- `--timeout` — per-subprocess timeout in seconds. Default 600.
- `--seed` — RNG seed for run-order shuffling. Default `0xC0FFEE`.
- `--no-plots` — skip plot rendering (writes json + csv + md only).

## Results
On a Windows 11 machine with a Ryzen 7, expect 8.13x slowdown when using v1 and
7.93x slowdown when using v2. Expect a similar memory overhead of 3.94x when using pyft.

For a more detailed set of results, see the [results/report.md](results/results.md)
