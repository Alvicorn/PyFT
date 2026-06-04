# PyFT — VerifiedFT for free-threaded Python

PyFT is a precise dynamic data race detector for free-threaded
CPython (PEP 703, 3.14+). It implements the **VerifiedFT** algorithm
(Wilcox & Freund, PPoPP '18), adapted to the runtime
semantics of Python's no-GIL build.

## Requirements

- **Python ≥ 3.14 free-threaded build** (the `-t` ABI). PyFT relies on
  the free-threaded interpreter for true threading. `pyft` will still run on
  the GIL build but won't surface races prevented by the GIL.
- [`uv`](https://docs.astral.sh/uv/) for environment management.
- No runtime dependencies.
- Dev dependencies:
  - `pre-commit`
  - `pytest`
  - `pytest-cov`
  - `pytest-timeout`

## Install / setup

```sh
uv python install 3.14+freethreaded
git clone --recurse-submodules https://github.com/Alvicorn/PyFT.git
git submodule init
git submodule update
cd pyft
uv sync
```

`uv sync` creates a `.venv/` with a free-threaded CPython 3.14 toolchain and installs PyFT in editable mode.

## Usage

### Run any script under the detector

```sh
uv run python -m pyft path/to/script.py [script args...]
```

This AST-rewrites the entry script and every module it subsequently
imports (excluding the stdlib and PyFT itself) so attribute reads /
writes / subscripts call into the race-detection engine. When the
script exits, PyFT prints a race report to stderr.

Try it on the included demos:

```sh
uv run python -m pyft demo/counter_race.py        # WW race expected
uv run python -m pyft demo/counter_locked.py      # no races expected
uv run python -m pyft demo/fork_join_safe.py      # no races expected
```

See [`demo/README.md`](demo/README.md) for the full demo index.

### Choosing a VerifiedFT variant

The Wilcox & Freund paper presents two analyses, both verified
race-precise. PyFT implements both:

| Version | Storage per variable | Race check | Use it when |
| --- | --- | --- | --- |
| `v1` | full `VectorClock` for the last write and the union of reads | element-wise `vc_leq` | you want the simpler reference analyser (matches the paper's idealised algorithm) |
| `v2` | single `Epoch` for the last write; compressed `ReadBottom → ReadEpoch → ReadVC` state machine for reads | constant-time epoch comparison in the common cases | default; matches the optimised FastTrack analyser, ~O(1) per access on the fast path |

Both versions detect the same set of data races on the same execution.
v2 is the default everywhere; pick v1 with the `--version` flag or the
`version=` kwarg:

```sh
uv run python -m pyft --version v1 demo/counter_race.py
```

```python
import pyft

with pyft.context(version="v1"):
    import workload
    workload.run()

@pyft.detect(version="v1")
def test_concurrent():
    import workload
    workload.do_stuff()

pyft.install(version="v1")
```

### Embed the detector around a region of your own code

```python
import pyft

with pyft.context():
    import workload          # auto-traced by the import hook
    workload.run()
```

Or as a decorator:

```python
@pyft.detect
def test_concurrent():
    import workload
    workload.do_stuff()
```

The detector is installed on enter and uninstalled on exit, and a race
report is printed automatically. Note that **modules imported BEFORE
`pyft.install()` (or before entering the context) are not retroactively
instrumented** — keep the workload in a module imported inside the
scope, or run the whole program under `python -m pyft`.

#### Public API

```text
pyft.install(version="v2")     install the detector (monkey-patches + import hook)
pyft.uninstall()               remove patches; race log is preserved
pyft.context(version="v2")     context manager: install on enter, report on exit
@pyft.detect                   decorator equivalent of pyft.context() around a function
@pyft.detect(version="v1")     decorator with an explicit analyser variant
pyft.report(file=...)          print the race report
pyft.races()                   list[RaceReport] for programmatic inspection
pyft.reset()                   clear the race log; detector keeps running
pyft.get_engine()              the active Engine instance (advanced)
```

## How it works

```
your code ──► AccessTracer (import hook, AST rewriting)
                      │
                      ▼
              engine.read / engine.write
                      │
                      ▼
              ShadowMap[(id(obj), attr)] ──► VarState ──► RaceLog
                                                ▲
                                                │
            engine.lock_{acquire,release}, thread_{start,finish,join}
                      ▲
                      │
       LockPatcher    │     AutoTracker
       (Lock/RLock/   │     (Thread.start / Thread.join)
        Semaphore/    │
        Event/Barrier)│
```

### Components

| Layer | Module | What it does |
| --- | --- | --- |
| **Core components** | `src/pyft/core/` | `Epoch`, `VectorClock`, `ThreadState`/`ThreadRegistry`, `VarStateV1` (full-VC analyser) and `VarStateV2` (epoch-compressed FastTrack); `VFTVersion` enum and `make_var_state` factory pick the variant. Free-threading-safe: each VC has its own lock, each VarState has its own lock, and the HB check + state update happen together inside that lock to eliminate TOCTOU. |
| **Detector** | `src/pyft/detector/` | `Engine` (event dispatch + per-thread reentrancy guard), `ShadowMap` (per-`(id(obj), attr)` VarState with weakref-based GC), `RaceLog` (atomic dedup by `(obj_id, attr, frozenset(tids), kind)`). |
| **Reporting** | `src/pyft/report/` | `RaceReport` formatting with stack traces for the triggering access, ANSI colour, deduplicated. |
| **Instrumentation** | `src/pyft/instrument/` | `LockPatcher` (synchronous wrappers around `threading.Lock`/`RLock`/`Semaphore`/`Event`/`Barrier`), `AutoTracker` (monkey-patches `Thread.start`/`Thread.join`), `AccessTracer` + `import_hook.py` (PEP 302 finder/loader that AST-rewrites every user module to call `engine.read`/`engine.write` around attribute and subscript ops). |

## Repository layout

```
src/pyft/
    __init__.py              public API: install, context, detect, report, races, reset
    __main__.py              "python -m pyft <script>" entry point
    core/                    epoch, vector clock, thread state, var state
    detector/                engine, shadow map, race log
    instrument/              lock_patcher, wrappers (AutoTracker), import_hook, transformer
    report/                  formatter
    utils/                   logging helpers
tests/
    unit/                    per-module unit tests
    integration/             end-to-end race-detection scenarios
demo/                        runnable example scripts
```

## Running the tests

```sh
# be sure to run the tests with free-threaded python
uv run --python 3.14t python -m pytest

# or more simply
uv run pytest tests/unit
uv run pytest tests/integration
```

## Known limitations

These are intentional design choices in the current implementation —
not bugs we plan to fix, but corners you should know about when
interpreting a race report.

- **Only modules imported *after* `pyft.install()` (or inside the
  `context()` / `@detect` body, or the `python -m pyft` entry script)
  are instrumented.** AST rewriting happens at import time, so
  anything already loaded passes through unmodified.
- **C extensions and source-less modules aren't instrumented.** The
  import hook needs `get_source()` to return Python text; compiled
  extensions, `.pyc`-only modules, and built-ins fall through to their
  original loader.
- **`del obj.attr` and `del obj[k]` aren't traced.** The AST
  transformer explicitly skips deletion to keep semantics safe.
- **Multi-target assignments (`a = b = ...`) and tuple-unpacking
  targets aren't rewritten on the target side.** Only the value side
  is.
- **Subscript keys are recorded via `repr(key)`.** `d[1]` and `d[1.0]`
  therefore share the same shadow-map slot.
- **Container-method tracing is done by *method name*, not by type.**
  Any user class with a method named `append`, `update`, `add`, etc.
  will also emit a synthetic `__container__` event. Conservative
  (never under-reports) but over-approximates.
- **Race reports include a stack trace only for the triggering access
  (Access A).** The prior conflicting access (Access B) shows
  `(stack not available for prior access)` because capturing stacks
  on every read / write would add substantial overhead.
- **Only `threading` primitives are modelled.** `asyncio` tasks,
  `concurrent.futures` async paths, and `multiprocessing` are not
  covered.

## Future work

Items we'd like to build but haven't yet — contributions welcome.

- **Benchmark suite** measuring runtime overhead
  `(checker_time − base_time) / base_time` and peak resident memory
  for `uninstrumented` vs `--version v1` vs `--version v2` on
  representative workloads.
- **Bytecode-level instrumentation fallback** so `.pyc`-only modules
  and (eventually) C extensions can be traced.
- **Race suppression / allowlist API** to silence known-safe
  attributes or modules so the report focuses on real bugs.
- **Structured race-report output for CI** — JSON, JUnit XML, SARIF
  alongside the human-readable stderr format.
- **GIL-build warning at install time** so users on a
  non-free-threaded interpreter aren't silently given an
  under-reported result.
- **Asyncio coverage** — model `asyncio.create_task` as a fork and
  `await task` as a join.
- **Multiprocessing coverage** — a worker-side stub that aggregates
  race reports back to the parent.
- **`--fail-on-race` CLI flag** so CI jobs can gate on a clean run.
- **Optional Access B stack traces** behind a `--full-stacks` flag,
  accepting the overhead when the user asks for it.
- **Type-aware container tracing** — promote the method-name match
  to an `isinstance` check against `list` / `dict` / `set` /
  `bytearray` so user classes with those method names aren't falsely
  conflated with built-in containers.
- **Mechanised correspondence to the paper** — generate v1 / v2 from
  the Coq proofs in the original VerifiedFT artefact, or carry a
  hand-written paper-to-code cross-reference document.
- **CI configuration** — GitHub Actions running the full suite on
  `cpython-3.14+freethreaded` on every push, plus a periodic
  stability run.

## References

- Cormac Flanagan and Stephen N. Freund. **FastTrack: Efficient and
  Precise Dynamic Race Detection.** PLDI 2009.
- James R. Wilcox, Cormac Flanagan, Stephen N. Freund. **VerifiedFT: A
  Verified, High-Performance Precise Dynamic Race Detector.** PPoPP 2018.
- [PEP 703](https://peps.python.org/pep-0703/) — Making the Global
  Interpreter Lock Optional in CPython.
