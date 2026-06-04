# PyVFT — Dev Information

PyVFT is a pure-python precise dynamic data race detector for free-threaded
CPython (PEP 703, 3.14+). It implements the **VerifiedFT** algorithm
(Wilcox & Freund, PPoPP '18), adapted to the runtime
semantics of Python's no-GIL build.

## Requirements

- **Python 3.14 free-threaded build**. PyVFT relies on
  the free-threaded interpreter for true threading. `pyvft` will still run on
  the GIL build but won't surface races prevented by the GIL.
- [`uv`](https://docs.astral.sh/uv/) for environment management.
- No runtime dependencies.
- Dev dependencies:
  - `pre-commit`
  - `pytest`
  - `pytest-cov`
  - `pytest-timeout`
    ```sh
    uv sync --group dev
    ```

## Install

### From source (development)

```sh
uv python install 3.14+freethreaded
git clone --recurse-submodules https://github.com/Alvicorn/pyvft.git
git submodule init
git submodule update
cd pyvft
uv sync
```

`uv sync` creates a `.venv/` with a free-threaded CPython 3.14 toolchain and installs PyVFT in editable mode.


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
| **Core components** | `pyvft/core/` | `Epoch`, `VectorClock`, `ThreadState`/`ThreadRegistry`, `VarStateV1` (full-VC analyser) and `VarStateV2` (epoch-compressed FastTrack); `VFTVersion` enum and `make_var_state` factory pick the variant. Free-threading-safe: each VC has its own lock, each VarState has its own lock, and the HB check + state update happen together inside that lock to eliminate TOCTOU. |
| **Detector** | `pyvft/detector/` | `Engine` (event dispatch + per-thread reentrancy guard), `ShadowMap` (per-`(id(obj), attr)` VarState with weakref-based GC), `RaceLog` (atomic dedup by `(obj_id, attr, frozenset(tids), kind)`). |
| **Reporting** | `pyvft/report/` | `RaceReport` formatting with stack traces for the triggering access, ANSI colour, deduplicated. |
| **Instrumentation** | `pyvft/instrument/` | `LockPatcher` (synchronous wrappers around `threading.Lock`/`RLock`/`Semaphore`/`Event`/`Barrier`), `AutoTracker` (monkey-patches `Thread.start`/`Thread.join`), `AccessTracer` + `import_hook.py` (PEP 302 finder/loader that AST-rewrites every user module to call `engine.read`/`engine.write` around attribute and subscript ops). |

## Repository layout

```
pyvft/
    __init__.py              public API: install, context, detect, report, races, reset
    __main__.py              "python -m pyvft <script>" entry point
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

# or more simply and specifically
uv run pytest tests/unit
uv run pytest tests/integration
```

## Release

PyVFT is published to PyPI by [.github/workflows/publish.yaml](.github/workflows/publish.yaml).
The workflow runs on every `v*` tag push. The package version is derived from the
tag itself via `setuptools-scm` (configured in [pyproject.toml](pyproject.toml)), so the tag
is the single source of truth.

### Cutting a release

1. Make sure `main` is green and contains everything you want to ship.
2. Pick the next version (PEP 440 — e.g. `0.2.0`). The tag must be
   strictly greater than every version already on PyPI.
3. Tag the commit and push the tag:
   ```sh
   git switch main
   git pull
   git tag v<version-tag>
   git push origin v<version-tag>
   ```
4. Watch the `publish` workflow in GitHub Actions and recover as needed.

## Known limitations

These are intentional design choices in the current implementation.

- **Only modules imported *after* `pyvft.install()` (or inside the
  `context()` / `@detect` body, or the `python -m pyvft` entry script)
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
  and `concurrent.futures` async paths are not
  covered.

## Future work

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
- **`--fail-on-race` CLI flag** so CI jobs can gate on a clean run.
- **Optional Access B stack traces** behind a `--full-stacks` flag,
  accepting the overhead when the user asks for it.
- **Type-aware container tracing** — promote the method-name match
  to an `isinstance` check against `list` / `dict` / `set` /
  `bytearray` so user classes with those method names aren't falsely
  conflated with built-in containers.

## References

- Cormac Flanagan and Stephen N. Freund. **FastTrack: Efficient and
  Precise Dynamic Race Detection.** PLDI 2009.
- James R. Wilcox, Cormac Flanagan, Stephen N. Freund. **VerifiedFT: A
  Verified, High-Performance Precise Dynamic Race Detector.** PPoPP 2018.
- [PEP 703](https://peps.python.org/pep-0703/) — Making the Global
  Interpreter Lock Optional in CPython.
