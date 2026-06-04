# PyVFT — VerifiedFT for free-threaded Python

PyVFT is a pure-python precise dynamic data race detector for free-threaded
CPython (PEP 703, 3.14+). It implements the **VerifiedFT** algorithm
(Wilcox & Freund, PPoPP '18), adapted to the runtime
semantics of Python's no-GIL build.

Through benchmarking with [python-threading-benchmarks](https://github.com/Alvicorn/python-threading-benchmarks), expect a ~x slowdown and 3.9x memory overhead. See [benchmarks/results/results.md](/benchmarks/results/results.md) for more information.

## Requirements

- **Python 3.14 free-threaded build**. PyVFT relies on
  the free-threaded interpreter for true threading. `pyvft` will still run on
  the GIL build but won't surface races prevented by the GIL.

## Install

### From PyPI

```sh
pip install pyvft
# or, with uv:
uv add pyvft
```

## Usage

### Run any script under the detector

```sh
uv run python -m pyvft path/to/script.py [script args...]
```

This AST-rewrites the entry script and every module it subsequently
imports (excluding the stdlib and PyVFT itself) so attribute reads /
writes / subscripts call into the race-detection engine. When the
script exits, PyVFT prints a race report to stderr.

Try it on the included demos:

```sh
uv run python -m pyvft demo/counter_race.py        # WW race expected
uv run python -m pyvft demo/counter_locked.py      # no races expected
uv run python -m pyvft demo/fork_join_safe.py      # no races expected
```

See [`demo/README.md`](demo/README.md) for the full demo index.

### Choosing a VerifiedFT variant

The Wilcox & Freund paper presents two analyses, both verified
race-precise. PyVFT implements both:

| Version | Storage per variable | Race check | Use it when |
| --- | --- | --- | --- |
| `v1` | full `VectorClock` for the last write and the union of reads | element-wise `vc_leq` | you want the simpler reference analyser (matches the paper's idealised algorithm) |
| `v2` | single `Epoch` for the last write; compressed `ReadBottom → ReadEpoch → ReadVC` state machine for reads | constant-time epoch comparison in the common cases | default; matches the optimised FastTrack analyser, ~O(1) per access on the fast path |

Both versions detect the same set of data races on the same execution.
v2 is the default everywhere; pick v1 with the `--version` flag or the
`version=` kwarg:

```sh
uv run python -m pyvft --version v1 demo/counter_race.py
```

```python
import pyvft

with pyvft.context(version="v1"):
    import workload
    workload.run()

@pyvft.detect(version="v1")
def test_concurrent():
    import workload
    workload.do_stuff()

pyvft.install(version="v1")
```

### Embed the detector around a region of your own code

```python
import pyvft

with pyvft.context():
    import workload          # auto-traced by the import hook
    workload.run()
```

Or as a decorator:

```python
@pyvft.detect
def test_concurrent():
    import workload
    workload.do_stuff()
```

The detector is installed on enter and uninstalled on exit, and a race
report is printed automatically. Note that **modules imported BEFORE
`pyvft.install()` (or before entering the context) are not retroactively
instrumented**. Keep the workload in a module imported inside the
scope, or run the whole program under `python -m pyvft`.

#### Public API

```text
pyvft.install(version="v2")     install the detector (monkey-patches + import hook)
pyvft.uninstall()               remove patches; race log is preserved
pyvft.context(version="v2")     context manager: install on enter, report on exit
@pyvft.detect                   decorator equivalent of pyvft.context() around a function
@pyvft.detect(version="v1")     decorator with an explicit analyser variant
pyvft.report(file=...)          print the race report
pyvft.races()                   list[RaceReport] for programmatic inspection
pyvft.reset()                   clear the race log; detector keeps running
pyvft.get_engine()              the active Engine instance (advanced)
```

## References

- Cormac Flanagan and Stephen N. Freund. **FastTrack: Efficient and
  Precise Dynamic Race Detection.** PLDI 2009.
- James R. Wilcox, Cormac Flanagan, Stephen N. Freund. **VerifiedFT: A
  Verified, High-Performance Precise Dynamic Race Detector.** PPoPP 2018.
- [PEP 703](https://peps.python.org/pep-0703/) — Making the Global
  Interpreter Lock Optional in CPython.
