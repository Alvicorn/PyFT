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
git clone https://github.com/Alvicorn/PyFT.git
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
pyft.install()         install the detector (monkey-patches + import hook)
pyft.uninstall()       remove patches; race log is preserved
pyft.context()         context manager: install on enter, report+uninstall on exit
@pyft.detect           decorator equivalent of pyft.context() around a function
pyft.report(file=...)  print the race report
pyft.races()           list[RaceReport] for programmatic inspection
pyft.reset()           clear the race log; detector keeps running
pyft.get_engine()      the active Engine instance (advanced)
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
| **Core components** | `src/pyft/core/` | `Epoch`, `VectorClock`, `ThreadState`/`ThreadRegistry`, `VarState`. Direct encoding of the VerifiedFT rules. Free-threading-safe: each VC has its own lock, each VarState has its own lock, and the HB check + state update happen together inside that lock to eliminate TOCTOU. |
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

## References

- Cormac Flanagan and Stephen N. Freund. **FastTrack: Efficient and
  Precise Dynamic Race Detection.** PLDI 2009.
- James R. Wilcox, Cormac Flanagan, Stephen N. Freund. **VerifiedFT: A
  Verified, High-Performance Precise Dynamic Race Detector.** PPoPP 2018.
- [PEP 703](https://peps.python.org/pep-0703/) — Making the Global
  Interpreter Lock Optional in CPython.
- [PEP 669](https://peps.python.org/pep-0669/) — Low Impact Monitoring
  for CPython.
