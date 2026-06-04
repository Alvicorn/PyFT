# PyVFT demos

Small, runnable examples of the PyVFT race detector. Each script is also
a smoke test — the expected race outcome is documented in the top-of-file
docstring.

## Running

Most demos rely on the auto-trace import hook, so run them under the
`pyvft` module:

```
uv run python -m pyvft demo/<script>.py
```

`demo/api_usage.py` is the exception — it calls `pyvft.install()` itself
and can be run with plain `python`:

```
uv run python demo/api_usage.py
```

## Demo index

| File | What it shows | Expected output |
| --- | --- | --- |
| `counter_race.py`     | Classic unsynchronized counter `+= 1` | ≥1 WRITE_WRITE race on `count` |
| `counter_locked.py`   | Same counter under `threading.Lock` | 0 races |
| `reader_writer_race.py` | One writer + one reader, no sync | ≥1 R/W or W/R race |
| `thread_local_safe.py` | Each thread owns its object | 0 races (two-thread-touch rule) |
| `fork_join_safe.py` | Parent writes before fork, reads after join | 0 races (fork + join HB) |
| `producer_consumer.py` | `threading.Event.set` / `wait` establishes HB | 0 races |
| `container_race.py` | Concurrent `list.append` + concurrent `dict.update` / `items` / `pop` | races on `__container__` for both the list and the dict |
| `api_usage.py` | Embedded use of `pyvft.context()` and `@pyvft.detect` | 1 race + 0 races |

## How auto-trace works

When you run `python -m pyvft myscript.py`, PyVFT installs three things:

1. **LockPatcher** — monkey-patches `threading.Lock`, `RLock`,
   `Semaphore`, `BoundedSemaphore`, `Event.set/wait`, and `Barrier.wait`
   so the engine sees synchronization events synchronously around the
   real OS operation. This gives deterministic event ordering under
   free-threaded Python.
2. **AutoTracker** — monkey-patches `Thread.start` and `Thread.join` so
   fork and join HB edges are recorded.
3. **AccessTracer** — installs a PEP 302 import hook that AST-rewrites
   every imported user module so attribute reads / writes / subscripts
   call into the engine. Stdlib / pyvft itself are skipped.

When the script exits, PyVFT prints a race report to stderr.
