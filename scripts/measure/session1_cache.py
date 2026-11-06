#!/usr/bin/env python3
"""Pre-run checklist 4.1: why has the A6000's compilation cache written nothing since 2026-10-15 12:24 (run 50)?

    .venv/bin/python -u scripts/measure/session1_cache.py runs/session1/cache

Read-only with respect to project data. The write test adds at most three tiny entries to the project cache.

**What it does, in order.** This driver imports neither JAX nor the project, so every probe below runs in a fresh
subprocess with exactly the environment the caller has, which under `scripts/a6000/session1.sh` is the scrubbed
environment `env.sh` gives the long run.

    census before   the cache directory: entries, `-atime` files, orphans, lock file, size against the cap,
                    newest write and newest read, and the autotune subdirectory (`session1_common.cache_census`)
    config probe    `import src`, then JAX's own cache settings as JAX holds them (every `jax.config` value whose
                    name contains "cache", read from the pinned JAX 0.11.0 rather than assumed), the JAX_ and XLA_
                    variables in effect, whether `filelock` imports, the device
    write test A    compile a tiny function whose cache key is NEW (a random constant baked into it), as
                    configured. JAX's own monitoring events say whether it missed; the census says whether an entry
                    was written
    read test       the SAME function in a second fresh process: JAX's events say whether it hit
    write test B    a second new function with `JAX_RAISE_PERSISTENT_CACHE_ERRORS=true`. JAX swallows cache write
                    errors by default (`jax_raise_persistent_cache_errors` is False), so a cache that silently stopped
                    writing shows nothing in A; in B the same failure raises, and its traceback is kept
    sweep order     a process started EXACTLY as a sweep process starts: it loads `scripts/measure/sweep.py`,
                    whose imports bring in jax before src. Reports whether the persistent cache is on in it

**Found while building this, 2026-11-06, on the laptop: JAX reads its `JAX_*` cache settings when it is imported,
and `sweep.py`, `replay_final.py`, `calibrate_s4.py` and `drift_diag.py`'s worker all import jax BEFORE src.** So in
those processes JAX holds no cache directory (`jax_compilation_cache_dir` None, min compile time back at JAX's 1 s
default) even though src sets the environment variable a moment later, and nothing is written. Only processes that
import src first, which the tests do through conftest, ever write. That predicts "nothing written since the last
time the tests ran" on the A6000, and the sweep-order probe measures it there rather than assuming it.

**What each outcome would mean, written before it is run:**

    sweep order cache OFF                the long run compiles every program from scratch in every process and
                                         never shares a compiled program; the import order is the cause

    A writes, read hits                  the cache works now. Whatever stopped it has cleared, or it only skips
                                         some programs: compare `newest_entry` with the census before
    A writes nothing, B raises           a write error JAX was hiding; the traceback names it. An orphan count
                                         above zero predicts this exact failure (see `cache_census`)
    A writes nothing, B does not raise   writing is gated by a setting rather than failing: read the config probe
                                         (min compile time, min entry size, max size, enable flag)
    the probe itself fails               the environment cannot import the project or JAX; the stderr is kept

Output: `summary.json` and one log per probe in the output folder.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import secrets
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import session1_common as s1  # noqa: E402  host-side only, imports no JAX

HIT, MISS = "/jax/compilation_cache/cache_hits", "/jax/compilation_cache/cache_misses"

#: The probe, run as `python -c PROBE <mode> <tag>` in a fresh process. Kept as data deliberately: it is the one
#: piece that imports JAX, and running it in a subprocess is what keeps every probe independent of the others.
PROBE = r"""
import json, os, sys
sys.path.insert(0, os.environ["S1_ROOT"])
mode, tag = sys.argv[1], sys.argv[2]
if mode == "sweep_order":
    # EXACTLY how a sweep process starts: load sweep.py, whose own imports bring in jax and then src. JAX reads its
    # JAX_* cache settings when it is imported, so whatever this order does to the cache, the long run gets too.
    import importlib.util as _u
    _s = _u.spec_from_file_location("sweep_order_probe", os.path.join(os.environ["S1_ROOT"], "scripts/measure/sweep.py"))
    _m = _u.module_from_spec(_s)
    sys.modules["sweep_order_probe"] = _m
    _s.loader.exec_module(_m)
else:
    import src  # noqa: F401  the project's settings BEFORE jax, so JAX sees them when it is imported
import jax
import jax.numpy as jnp
events = {}
jax.monitoring.register_event_listener(lambda name, **kw: events.__setitem__(name, events.get(name, 0) + 1))
import importlib.util
out = {
    "mode": mode,
    "config": {n: jax.config.values[n] for n in sorted(jax.config.values) if "cache" in n.lower()},
    "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("JAX_", "XLA_"))},
    "filelock_importable": importlib.util.find_spec("filelock") is not None,
    "jax_version": jax.__version__,
    "device": jax.devices()[0].device_kind,
    "platform": jax.devices()[0].platform,
}
if mode in ("compile", "sweep_order"):
    c = 1.0 + int(tag, 16) / 16.0**len(tag)  # a constant unique to this tag, so the cache key is new
    f = jax.jit(lambda x: x * c + 0.5)
    out["result"] = float(f(jnp.ones(7)).sum())
    from jax._src import compilation_cache as cc
    out["persistent_cache_enabled"] = bool(cc.is_persistent_cache_enabled())
    out["cache_dir_as_jax_holds_it"] = jax.config.values["jax_compilation_cache_dir"]
    out["env_cache_dir"] = os.environ.get("JAX_COMPILATION_CACHE_DIR")
out["events"] = events
print("S1_PROBE_JSON " + json.dumps(out, default=str))
"""


def run_probe(mode: str, tag: str, out_dir: pathlib.Path, name: str, extra_env: dict | None = None) -> dict:
    env = dict(os.environ, S1_ROOT=str(ROOT), **(extra_env or {}))
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", PROBE, mode, tag],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=900,
    )
    (out_dir / f"{name}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    rec = {"name": name, "exit": proc.returncode, "seconds": round(time.time() - t0, 1)}
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("S1_PROBE_JSON ")), None)
    if line is not None:
        rec.update(json.loads(line[len("S1_PROBE_JSON ") :]))
    if proc.returncode != 0:
        rec["stderr_tail"] = proc.stderr[-3000:]
    return rec


def production_cache(sweep_order: dict) -> dict:
    """Whether a process started the way the sweep starts has the persistent cache ON. Pure."""
    if sweep_order.get("exit") != 0 or "persistent_cache_enabled" not in sweep_order:
        return {
            "known": False,
            "enabled": None,
            "text": "the sweep-order probe failed; see probe_sweep_order.log",
        }
    on = bool(sweep_order["persistent_cache_enabled"])
    text = (
        "ON in a process started the way the sweep starts"
        if on
        else (
            "OFF in a process started the way the sweep starts: JAX was imported before src, so JAX read its "
            f"cache settings while they were still unset (JAX holds cache dir {sweep_order.get('cache_dir_as_jax_holds_it')!r} "
            f"although src then set {sweep_order.get('env_cache_dir')!r}). Every sweep, replay and calibration "
            "process therefore compiles from scratch and writes nothing; only processes that import src first "
            "(the tests, through conftest) ever write"
        )
    )
    return {"known": True, "enabled": on, "text": text}


def verdict(
    before: dict, after_a: dict, write_a: dict, read: dict, write_b: dict, sweep_order: dict | None = None
) -> dict:
    """Plain-words reading of the probes. Pure, so it is unit-tested on hand-built records."""
    prod = production_cache(sweep_order or {})
    probe_ok = write_a.get("exit") == 0 and "events" in write_a
    wrote = probe_ok and after_a.get("cache_entries", 0) > before.get("cache_entries", 0)
    missed = probe_ok and write_a["events"].get(MISS, 0) >= 1
    hit = read.get("exit") == 0 and read.get("events", {}).get(HIT, 0) >= 1
    strict_raised = write_b.get("exit") not in (0, None)
    causes = []
    if before.get("cache_without_atime", 0) > 0:
        causes.append(
            f"{before['cache_without_atime']} cache entries have no -atime file; every write reads all -atime files "
            "and fails on a missing one, silently by default"
        )
    frac = before.get("fraction_of_max")
    if frac is not None and frac >= 1.0:
        causes.append(f"the cache holds {frac:.0%} of its size cap, so every write must first evict")
    cfg = write_a.get("config", {})
    if cfg and not cfg.get("jax_enable_compilation_cache", True):
        causes.append("jax_enable_compilation_cache is False")
    if cfg and float(cfg.get("jax_persistent_cache_min_compile_time_secs", 0) or 0) > 0:
        causes.append(
            f"jax_persistent_cache_min_compile_time_secs is {cfg['jax_persistent_cache_min_compile_time_secs']}: "
            "faster compiles are never written"
        )
    if cfg and int(cfg.get("jax_persistent_cache_min_entry_size_bytes", 0) or 0) > 0:
        causes.append(
            f"jax_persistent_cache_min_entry_size_bytes is {cfg['jax_persistent_cache_min_entry_size_bytes']}"
        )
    if prod["known"] and not prod["enabled"]:
        causes.insert(0, "the long run's own processes never turn the cache on: " + prod["text"])
    if not probe_ok:
        status = "the probe itself failed: the environment could not import the project or JAX (see its log)"
    elif prod["known"] and not prod["enabled"] and wrote:
        status = (
            "CACHE WORKS, BUT THE SWEEP NEVER USES IT: a process that imports src first writes and reads, while a "
            "process started like the sweep has the cache off"
        )
    elif wrote and hit:
        status = "WORKS: a new program was written and a second process read it back"
    elif wrote and not hit:
        status = "PARTLY: a new program was written but a second process did not read it back"
    elif strict_raised:
        status = "BROKEN: writing fails, and JAX hides the error by default; write test B shows it"
    else:
        status = "NOT WRITING, without an error: writing is gated by a setting (see the config probe)"
    return {
        "status": status,
        "wrote_new_entry": bool(wrote),
        "compile_missed_as_expected": bool(missed),
        "second_process_hit": bool(hit),
        "strict_write_raised": bool(strict_raised),
        "strict_error_tail": (write_b.get("stderr_tail") or "")[-800:] if strict_raised else "",
        "production_cache": prod,
        "likely_causes": causes,
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = pathlib.Path(argv[0]) if argv else ROOT / "runs" / "session1" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = datetime.datetime.now() - datetime.timedelta(seconds=2)

    cfg_probe = run_probe("config", "0", out_dir, "probe_config")
    cache_dir = pathlib.Path(
        cfg_probe.get("config", {}).get("jax_compilation_cache_dir")
        or os.environ.get("JAX_COMPILATION_CACHE_DIR")
        or pathlib.Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")))
        / "jax-emergent-self-preservation"
    )
    max_size = cfg_probe.get("config", {}).get("jax_compilation_cache_max_size")
    before = s1.cache_census(cache_dir, since=t_start, max_size=max_size)
    print(f"cache directory {cache_dir}", flush=True)
    print(f"census before: {json.dumps(before)}", flush=True)

    tag_a, tag_b = secrets.token_hex(6), secrets.token_hex(6)
    write_a = run_probe("compile", tag_a, out_dir, "probe_write_a")
    after_a = s1.cache_census(cache_dir, since=t_start, max_size=max_size)
    read = run_probe("compile", tag_a, out_dir, "probe_read")
    write_b = run_probe(
        "compile", tag_b, out_dir, "probe_write_b", {"JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true"}
    )
    sweep_order = run_probe("sweep_order", secrets.token_hex(6), out_dir, "probe_sweep_order")
    after_all = s1.cache_census(cache_dir, since=t_start, max_size=max_size)

    v = verdict(before, after_a, write_a, read, write_b, sweep_order)
    summary = {
        "item": "4.1 compilation cache",
        "cache_dir": str(cache_dir),
        "census_before": before,
        "census_after_write_a": after_a,
        "census_after_all": after_all,
        "probes": {
            "config": cfg_probe,
            "write_a": write_a,
            "read": read,
            "write_b": write_b,
            "sweep_order": sweep_order,
        },
        "verdict": v,
        "host": s1.host_snapshot(),
        "at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\n4.1 VERDICT: {v['status']}", flush=True)
    print(f"  in the long run's own processes the cache is {v['production_cache']['text']}", flush=True)
    for c in v["likely_causes"]:
        print(f"  likely cause: {c}", flush=True)
    if v["strict_write_raised"]:
        print(f"  the hidden error, from write test B:\n{v['strict_error_tail']}", flush=True)
    print(f"written {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
