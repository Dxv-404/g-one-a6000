#!/usr/bin/env python3
"""Pre-run checklist 4.3: how many evolutionary runs can share the A6000 at once, MEASURED, not assumed.

    .venv/bin/python -u scripts/measure/session1_concurrency.py runs/session1/concurrency
    options: --generations 50  --max-n 16  --flags=--xla_gpu_autotune_level=0  --cache shared|fresh
             --smoke (2 generations, N up to 2, into a directory named for it)

**The code path is the long run's own.** Every worker runs `sweep.one_seed(sweep.params_for("treatment", "baseline",
None), seed, n_generations=G)` in its own process, in the environment the long run gets (`env.sh`'s scrub, then
`XLA_FLAGS` set to the flag under test, default `--xla_gpu_autotune_level=0` per run 50). Seeds start at 200, used
for nothing else (Phase A is 0 to 19, the long run 20 to 39, the S4 calibration 100 to 102).

**Steps.** (a) A solo reference per seed, one process at a time. (b) For N in 2, 3, 4, 6, 8, then 12 and 16 while
the stop rule allows, N workers launched together, one seed each.

**A start barrier makes the throughput honest.** Each worker compiles during its first generation, then signals
ready and waits; the driver releases all of them together once every one is ready. Without it, compiles finishing
at different times would let early workers run their timed loop partly alone, and the summed rate would overstate
what N concurrent runs achieve. The barrier lives in the `progress` hook, which `one_seed` hands only the
generation's wall time and fitness and which splits no key, so it cannot change a result; the bit-identity check
below would show it if it did.

**What is recorded for every N.** Bit-identity of each seed's `final_pop`, `fit_mean` and `fit_max` against its solo
reference, with the number of differing genomes; aggregate throughput, which is N times (G - 1) timed generations
over the window from the release to the last worker's end; each worker's own loop rate and compile time; GPU
utilisation, memory and temperature sampled during the round; host load and available memory; the compilation
cache's entry count before and after; each worker's cache hits and misses; and the BACKGROUND GPU load before and
after the round, with none of our processes running (`session1_common.round_clean`).

**The foreign workload.** A colleague's nnU-Net training job shares this GPU, about 9.4 GB, running intermittently,
and the owner's planning assumes it keeps running through the long run. So a round whose background before or after
differs from the session's first reading by more than 10 utilisation points or 1 GiB is NOT CLEAN: its bit-identity
verdict stands (load cannot change what is computed), but its throughput is excluded from the stop rule and from the
recommendation. Throughput here is measured BESIDE that job, so the projected long-run duration holds only if it
keeps running as it did during the measurement.

**The stop rule (`next_step`), checked after every round:** stop raising N on a failed worker (a crash, a missing
output or a barrier timeout), on any seed that is not bit-identical to its solo reference, when the next N would
exceed min(nproc, 16) or `--max-n`, when the projected GPU memory at the next N would leave less than 8 GiB free, or
when a CLEAN round gains less than 5 percent throughput over the previous clean round (exactly 5 percent continues).

**The recommendation (`recommend`), from rounds that pass all three rules:**

    bit-identity   every seed identical to its solo reference, and no worker failed
    clean          background before and after the round matches the session's first reading
    memory         the round's measured peak GPU memory, ours plus the background, leaves at least 8 GiB free of
                   the card (46 GiB on the A6000), headroom for the colleague's job to grow

Of those, the fastest N, then ONE STEP LOWER in the tested sequence for thermal and memory headroom. The projection
is the plan's generations (106 conditions x 20 seeds x 1,500) over the measured throughput at that N. If no round
passes all three, it says so plainly and recommends nothing.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import session1_common as s1  # noqa: E402  host-side only

_SPEC = importlib.util.spec_from_file_location(
    "drift_diag_s1c", ROOT / "scripts" / "measure" / "drift_diag.py"
)
dd = importlib.util.module_from_spec(_SPEC)
sys.modules["drift_diag_s1c"] = dd
_SPEC.loader.exec_module(dd)  # for `clean_env`, the environment scrub drift_diag's workers already use

FLAG = "--xla_gpu_autotune_level=0"
SEQUENCE = (2, 3, 4, 6, 8, 12, 16)
SEED_BASE = 200
N_CAP = 16
GAIN_FLOOR = 0.05  # stop when a clean round gains LESS than this over the previous clean round
GPU_HEADROOM_GIB = 8.0  # the recommendation and the stop rule both keep at least this much of the card free
HOST_HEADROOM_GIB = 4.0  # src.memory_headroom's floor for the sweep, D-054
PLAN_CONDITIONS, PLAN_SEEDS, PLAN_GENERATIONS = 106, 20, 1500  # D-084 (D-074's exclusions applied), D-073
BARRIER_TIMEOUT_S = 900.0
STATE_KEYS = ("final_pop", "fit_mean", "fit_max")


# ------------------------------------------------------------------------------------------------ worker ----
def worker(seed: int, generations: int, out: pathlib.Path, ready: pathlib.Path, go: pathlib.Path) -> None:
    """One evolutionary run through the long run's own code, timed, with the start barrier in its progress hook."""
    sys.path.insert(0, str(ROOT))
    # **Start EXACTLY as a sweep process starts: load sweep.py before anything else.** JAX reads its cache settings
    # when it is imported, and sweep.py's own imports bring in jax before src, so the order decides whether the
    # persistent cache is on (on 2026-11-06 it is not; see session1_cache.py). Importing jax or src here first
    # would give these workers a cache state the long run does not have. The state they actually ran with is
    # recorded below, so the summary reports it rather than assuming it.
    spec = importlib.util.spec_from_file_location("sweep_s1w", ROOT / "scripts" / "measure" / "sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    sys.modules["sweep_s1w"] = sweep
    spec.loader.exec_module(sweep)
    import jax
    from jax._src import compilation_cache as cc

    import src

    src.assert_runtime_took_effect()
    assert jax.devices()[0].platform == "gpu", "the concurrency check must run on the GPU"

    events: dict[str, int] = {}
    jax.monitoring.register_event_listener(
        lambda name, **kw: events.__setitem__(name, events.get(name, 0) + 1)
    )

    class Hook:
        """Sees the run and cannot touch it: no key, no device work, only host time."""

        def __init__(self):
            self.walls, self.go_time, self.loop_end_time, self.barrier_timed_out = [], None, None, False

        def generation(self, gen, wall, fit, acc_row):
            self.walls.append(float(wall))
            if gen == 0:
                ready.write_text(f"{time.time()}\n")
                t0 = time.time()
                while not go.exists():
                    if time.time() - t0 > BARRIER_TIMEOUT_S:
                        self.barrier_timed_out = True
                        break
                    time.sleep(0.02)
                self.go_time = time.time()

        def loop_end(self):
            self.loop_end_time = time.time()

    hook = Hook()
    p = sweep.params_for("treatment", "baseline", None)
    r = sweep.one_seed(p, seed, n_generations=generations, progress=hook)
    np.savez(
        out,
        seed=np.int32(seed),
        generations=np.int32(generations),
        final_pop=r["final_pop"],
        fit_mean=r["fit_mean"],
        fit_max=r["fit_max"],
        walls=np.array(hook.walls, np.float64),
        go_time=np.float64(hook.go_time if hook.go_time is not None else np.nan),
        loop_end_time=np.float64(hook.loop_end_time if hook.loop_end_time is not None else np.nan),
        barrier_timed_out=np.bool_(hook.barrier_timed_out),
        events=np.array(json.dumps(events)),
        xla_flags=np.array(os.environ.get("XLA_FLAGS", "")),
        cache_dir=np.array(str(jax.config.values.get("jax_compilation_cache_dir"))),
        persistent_cache_enabled=np.bool_(cc.is_persistent_cache_enabled()),
        device=np.array(jax.devices()[0].device_kind),
        jax_version=np.array(jax.__version__),
    )


# ------------------------------------------------------------------------------------ pure verdicts ----
def compare_state(ref: dict, got: dict) -> dict:
    """Bit-identity of one seed against its solo reference, on every array the long run keeps."""
    out = {k: bool(np.array_equal(np.asarray(ref[k]), np.asarray(got[k]))) for k in STATE_KEYS}
    a, b = np.asarray(ref["final_pop"]), np.asarray(got["final_pop"])
    out["genomes_differing"] = int(np.any(a != b, axis=1).sum()) if a.shape == b.shape else int(a.shape[0])
    out["identical"] = all(out[k] for k in STATE_KEYS)
    return out


def loop_rate(walls) -> float:
    """Generations per second over the timed loop: every generation after the first, which carries the compile."""
    w = np.asarray(walls, np.float64)[1:]
    return float(len(w) / w.sum()) if len(w) and w.sum() > 0 else float("nan")


def compile_seconds(walls) -> float:
    w = np.asarray(walls, np.float64)
    return float(w[0] - np.median(w[1:])) if len(w) > 1 else float("nan")


def aggregate_throughput(n_workers: int, generations: int, release: float, ends) -> float:
    """N x (G - 1) timed generations over the window from the release to the last worker's loop end."""
    window = max(ends) - release
    return float(n_workers * (generations - 1) / window) if window > 0 else float("nan")


def next_step(
    history: list[dict],
    n_next: int | None,
    cap: int,
    gpu_total_mib: float | None,
    host_available_gib: float | None,
    gpu_headroom_gib: float = GPU_HEADROOM_GIB,
) -> tuple[bool, str]:
    """(continue, reason) after the latest round in `history`, each a dict with n, failed, identical, clean,
    throughput, peak_gpu_mib, background_gpu_mib, host_mem_drop_gib. The first entry is the solo baseline (n 1)."""
    last = history[-1]
    if last["failed"]:
        return False, f"a worker failed at N = {last['n']}"
    if not last["identical"]:
        return False, f"not bit-identical at N = {last['n']}"
    if n_next is None or n_next > cap:
        return False, f"the next N would exceed the cap of {cap}"
    if gpu_total_mib and last.get("peak_gpu_mib") is not None and last.get("background_gpu_mib") is not None:
        ours = max(last["peak_gpu_mib"] - last["background_gpu_mib"], 0.0)
        projected = last["background_gpu_mib"] + ours / last["n"] * n_next
        if gpu_total_mib - projected < gpu_headroom_gib * 1024:
            return False, (
                f"at N = {n_next} projected GPU memory {projected / 1024:.1f} GiB would leave less than "
                f"{gpu_headroom_gib:.0f} GiB free"
            )
    if host_available_gib is not None and last.get("host_mem_drop_gib") is not None:
        need = last["host_mem_drop_gib"] / last["n"] * n_next
        if host_available_gib - need < HOST_HEADROOM_GIB:
            return (
                False,
                f"at N = {n_next} projected host memory would leave less than {HOST_HEADROOM_GIB:.0f} GiB",
            )
    if last["clean"]:
        prev = [h for h in history[:-1] if h["clean"] and not h["failed"] and h["identical"]]
        if prev:
            gain = last["throughput"] / prev[-1]["throughput"] - 1.0
            # Compared as a product, not as the ratio above: `1.05 / 1.0 - 1` is 0.050000000000000044 in floating
            # point, so a ratio test can never sit exactly on the boundary and `<` and `<=` become
            # indistinguishable (falsification found this). The product form makes exactly 5 percent testable.
            if last["throughput"] < prev[-1]["throughput"] * (1.0 + GAIN_FLOOR):
                return False, (
                    f"N = {last['n']} gained {gain:+.1%} over N = {prev[-1]['n']}, under {GAIN_FLOOR:.0%}"
                )
    return True, "continue"


def eligible(
    h: dict, gpu_total_mib: float | None, gpu_headroom_gib: float = GPU_HEADROOM_GIB
) -> tuple[bool, str]:
    if h["failed"]:
        return False, "a worker failed"
    if not h["identical"]:
        return False, "not bit-identical"
    if not h["clean"]:
        return False, "background changed (not clean)"
    if gpu_total_mib is None or h.get("peak_gpu_mib") is None:
        return False, "no GPU memory reading"
    free = (gpu_total_mib - h["peak_gpu_mib"]) / 1024
    if free < gpu_headroom_gib:
        return False, f"peak GPU memory left {free:.1f} GiB free, under {gpu_headroom_gib:.0f}"
    return True, "eligible"


def recommend(
    history: list[dict],
    gpu_total_mib: float | None,
    plan_generations: int | None = None,
    gpu_headroom_gib: float = GPU_HEADROOM_GIB,
) -> dict:
    """The fastest round passing bit-identity, cleanliness and memory headroom, then one step lower."""
    plan_generations = plan_generations or PLAN_CONDITIONS * PLAN_SEEDS * PLAN_GENERATIONS
    ok = [h for h in history if eligible(h, gpu_total_mib, gpu_headroom_gib)[0]]
    if not ok:
        return {
            "n": None,
            "text": (
                "NO RECOMMENDATION: no round passed all three rules (bit-identical, clean background, at least "
                f"{gpu_headroom_gib:.0f} GiB GPU memory left free), so no clean comparison was possible. The "
                "bit-identity verdicts above still stand; the throughput numbers do not support a choice of N."
            ),
        }
    best = max(ok, key=lambda h: h["throughput"])
    lower = [h for h in ok if h["n"] < best["n"]]
    pick = max(lower, key=lambda h: h["n"]) if lower else best
    days = plan_generations / pick["throughput"] / 86400.0
    return {
        "n": pick["n"],
        "fastest_n": best["n"],
        "throughput": pick["throughput"],
        "projected_days": days,
        "text": (
            f"run N = {pick['n']} at once ("
            + (
                f"fastest eligible was N = {best['n']}, one step lower for headroom"
                if pick is not best
                else f"N = {best['n']} is the fastest eligible and no lower N was eligible to step down to"
            )
            + "): "
            f"{pick['throughput']:.2f} generations per second, so about {days:.1f} days for the plan's "
            f"{plan_generations:,} generations, CONDITIONAL on the colleague's GPU job running as it did here"
        ),
    }


# ------------------------------------------------------------------------------------------- driver ----
def worker_env(flags: str, cache_dir: str | None) -> dict:
    return dd.clean_env(cache_dir, flags)


def run_round(seeds, generations, flags, cache_mode, round_dir: pathlib.Path, log) -> dict:
    """N workers together behind the start barrier. Never raises: a failure is recorded."""
    round_dir.mkdir(parents=True, exist_ok=True)
    go = round_dir / "GO"
    go.unlink(missing_ok=True)
    procs, outs, readies, tmps = [], {}, {}, []
    for s in seeds:
        cache = None
        if cache_mode == "fresh":
            t = tempfile.mkdtemp(prefix=f"s1cache-{s}-", dir=str(round_dir))
            tmps.append(t)
            cache = t
        outs[s] = round_dir / f"seed{s}.npz"
        readies[s] = round_dir / f"seed{s}.ready"
        outs[s].unlink(missing_ok=True)
        readies[s].unlink(missing_ok=True)
        logf = open(round_dir / f"seed{s}.log", "w")
        procs.append(
            (
                s,
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        __file__,
                        "--worker",
                        str(s),
                        str(generations),
                        str(outs[s]),
                        str(readies[s]),
                        str(go),
                    ],
                    env=worker_env(flags, cache),
                    cwd=str(ROOT),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                ),
                logf,
            )
        )
    t_launch = time.time()
    release = None
    with s1.Sampler(every=1.0) as smp:
        while release is None:
            if all(readies[s].exists() for s in seeds) or any(p.poll() is not None for _, p, _ in procs):
                go.write_text("go\n")
                release = time.time()
            elif time.time() - t_launch > BARRIER_TIMEOUT_S:
                go.write_text("go\n")
                release = time.time()
            else:
                time.sleep(0.05)
        deadline = time.time() + BARRIER_TIMEOUT_S + generations * 60
        for _s, p, _f in procs:
            try:
                p.wait(timeout=max(deadline - time.time(), 1))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
    for _, _, f in procs:
        f.close()
    gpu = smp.result()
    workers, failed = {}, False
    for s, p, _ in procs:
        rec = {"exit": p.returncode}
        if p.returncode != 0 or not outs[s].exists():
            failed = True
            rec["failed"] = True
            rec["log_tail"] = (round_dir / f"seed{s}.log").read_text()[-1500:]
        else:
            with np.load(outs[s]) as z:
                rec.update(
                    loop_rate=loop_rate(z["walls"]),
                    compile_seconds=compile_seconds(z["walls"]),
                    loop_end_time=float(z["loop_end_time"]),
                    barrier_timed_out=bool(z["barrier_timed_out"]),
                    events=json.loads(str(z["events"])),
                    persistent_cache_enabled=bool(z["persistent_cache_enabled"]),
                )
            if rec["barrier_timed_out"]:
                failed = True
        workers[s] = rec
    ends = [w["loop_end_time"] for w in workers.values() if "loop_end_time" in w]
    tp = aggregate_throughput(len(seeds), generations, release, ends) if ends and not failed else float("nan")
    log(
        f"    N = {len(seeds)}: {'FAILED' if failed else 'ran'}, aggregate {tp:.3f} gen/s, peak GPU "
        f"{gpu.get('mem_used_mib_max', float('nan')):.0f} MiB, util mean {gpu.get('util_mean', float('nan')):.0f}"
    )
    return {
        "workers": workers,
        "failed": failed,
        "throughput": tp,
        "gpu": gpu,
        "release": release,
        "launch_to_release_s": release - t_launch,
    }


def load_state(path: pathlib.Path) -> dict:
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in STATE_KEYS}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--worker":
        worker(
            int(argv[1]), int(argv[2]), pathlib.Path(argv[3]), pathlib.Path(argv[4]), pathlib.Path(argv[5])
        )
        return 0
    ap = argparse.ArgumentParser(description="4.3: the concurrency curve")
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--max-n", type=int, default=N_CAP)
    ap.add_argument("--flags", default=FLAG)
    ap.add_argument("--cache", choices=("shared", "fresh"), default="shared")
    ap.add_argument(
        "--gpu-headroom-gib",
        type=float,
        default=GPU_HEADROOM_GIB,
        help="GPU memory every recommended round must leave free (default 8, room for the colleague's job)",
    )
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    G, max_n, headroom = a.generations, a.max_n, a.gpu_headroom_gib
    if a.smoke:
        assert "smoke" in a.out_dir.name, "a smoke run writes only into a directory named for it"
        # A smoke run is not a measurement, and a laptop card cannot keep 8 GiB free, so the memory rule is off.
        G, max_n, headroom = 2, 2, 0.0
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    logp = out / "concurrency.log"

    def log(msg):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')}  {msg}"
        print(line, flush=True)
        with logp.open("a") as f:
            f.write(line + "\n")

    host = s1.host_snapshot()
    cap = min(host["nproc"] or 1, N_CAP, max_n)
    sequence = [n for n in SEQUENCE if n <= cap]
    cache_dir = (
        pathlib.Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")))
        / "jax-emergent-self-preservation"
    )
    log(
        f"4.3 concurrency: G = {G}, flags {a.flags!r}, cache {a.cache}, cap N = {cap} (nproc {host['nproc']}), "
        f"sequence {sequence}, host memory available {host['mem_available_gib']:.1f} GiB"
    )
    reference = s1.background_load()
    log(
        f"BACKGROUND GPU LOAD before anything of ours: {reference.get('util_mean', float('nan')):.0f} percent mean, "
        f"{reference.get('util_max', float('nan')):.0f} peak, {reference.get('mem_used_mib_mean', float('nan')):.0f} MiB "
        "used. This is the colleague's job and the desktop; every round is compared with it."
    )
    gpu_total = reference.get("mem_total_mib")

    refs: dict[int, dict] = {}
    solo_rates, solo_peaks, solo_records = [], [], []

    def ensure_refs(seeds):
        need = [s for s in seeds if s not in refs]
        if not need:
            return
        bg_b = s1.background_load()
        for s in need:
            r = run_round([s], G, a.flags, a.cache, out / "solo" / f"seed{s}", log)
            solo_records.append(
                {"seed": s, **{k: v for k, v in r.items() if k != "workers"}, "worker": r["workers"][s]}
            )
            if r["failed"]:
                refs[s] = None
                continue
            refs[s] = load_state(out / "solo" / f"seed{s}" / f"seed{s}.npz")
            solo_rates.append(r["workers"][s]["loop_rate"])
            solo_peaks.append(r["gpu"].get("mem_used_mib_max"))
        bg_a = s1.background_load()
        clean, why = s1.round_clean(bg_b, bg_a, reference)
        solo_records.append({"background_before": bg_b, "background_after": bg_a, "clean": clean, "why": why})
        return clean

    first_clean = ensure_refs([SEED_BASE + i for i in range(sequence[0] if sequence else 1)])
    solo_failed = any(refs.get(s) is None for s in refs)
    history = [
        {
            "n": 1,
            "failed": solo_failed,
            "identical": not solo_failed,
            "clean": bool(first_clean),
            "throughput": statistics.median(solo_rates) if solo_rates else float("nan"),
            "peak_gpu_mib": max((p for p in solo_peaks if p is not None), default=None),
            "background_gpu_mib": reference.get("mem_used_mib_mean"),
            "host_mem_drop_gib": None,
        }
    ]
    rounds = []
    cont, why = next_step(history, sequence[0] if sequence else None, cap, gpu_total, None, headroom)
    for i, n in enumerate(sequence):
        if not cont:
            break
        seeds = [SEED_BASE + j for j in range(n)]
        ensure_refs(seeds)
        if any(refs.get(s) is None for s in seeds):
            log(f"  a solo reference failed; N = {n} cannot be compared")
            history.append(
                {"n": n, "failed": True, "identical": False, "clean": False, "throughput": float("nan")}
            )
            break
        bg_b = s1.background_load()
        host_b = s1.host_snapshot()
        census_b = s1.cache_census(cache_dir)
        r = run_round(seeds, G, a.flags, a.cache, out / f"round_n{n:02d}", log)
        bg_a = s1.background_load()
        census_a = s1.cache_census(cache_dir)
        clean, cwhy = s1.round_clean(bg_b, bg_a, reference)
        ident = {}
        for s in seeds:
            f = out / f"round_n{n:02d}" / f"seed{s}.npz"
            ident[s] = (
                compare_state(refs[s], load_state(f)) if f.exists() else {"identical": False, "missing": True}
            )
        all_ident = all(v["identical"] for v in ident.values())
        min_avail = r["gpu"].get("host_mem_available_gib_min")
        h = {
            "n": n,
            "failed": r["failed"],
            "identical": all_ident and not r["failed"],
            "clean": clean,
            "throughput": r["throughput"],
            "peak_gpu_mib": r["gpu"].get("mem_used_mib_max"),
            "background_gpu_mib": (bg_b.get("mem_used_mib_mean") if bg_b.get("available") else None),
            "host_mem_drop_gib": (host_b["mem_available_gib"] - min_avail) if min_avail is not None else None,
        }
        history.append(h)
        rounds.append(
            {
                **h,
                "clean_why": cwhy,
                "background_before": bg_b,
                "background_after": bg_a,
                "identity": {str(k): v for k, v in ident.items()},
                "gpu_during": r["gpu"],
                "workers": {str(k): v for k, v in r["workers"].items()},
                "cache_entries_before": census_b.get("cache_entries"),
                "cache_entries_after": census_a.get("cache_entries"),
                "launch_to_release_s": r["launch_to_release_s"],
            }
        )
        log(
            f"  N = {n}: identical {all_ident}, clean {clean} ({cwhy}), throughput {r['throughput']:.3f} gen/s"
        )
        n_next = sequence[i + 1] if i + 1 < len(sequence) else None
        cont, why = next_step(
            history, n_next, cap, gpu_total, s1.host_snapshot()["mem_available_gib"], headroom
        )
        (out / "rounds.json").write_text(
            json.dumps({"history": history, "rounds": rounds, "solo": solo_records}, indent=1, default=str)
        )
    rec = recommend(history, gpu_total, gpu_headroom_gib=headroom)
    cache_states = sorted(
        {
            str(w.get("persistent_cache_enabled"))
            for rr in [*rounds, *[x for x in solo_records if "worker" in x]]
            for w in (rr["workers"].values() if "workers" in rr else [rr["worker"]])
        }
    )
    summary = {
        "item": "4.3 concurrency curve",
        "generations": G,
        "flags": a.flags,
        "cache": a.cache,
        "worker_persistent_cache_enabled": cache_states,
        "cap": cap,
        "reference_background": reference,
        "history": history,
        "stopped_because": why,
        "eligibility": {str(h["n"]): eligible(h, gpu_total, headroom)[1] for h in history},
        "recommendation": rec,
        "foreign_workload_note": (
            "Throughput was measured BESIDE a colleague's GPU job (nnU-Net, about 9.4 GB, "
            "intermittent). The long-run day estimate holds only if it keeps running as it did."
        ),
        "rules": {
            "gain_floor": GAIN_FLOOR,
            "gpu_headroom_gib": headroom,
            "host_headroom_gib": HOST_HEADROOM_GIB,
            "clean_util_margin": s1.CLEAN_UTIL_MARGIN,
            "clean_mem_margin_gib": s1.CLEAN_MEM_MARGIN_GIB,
        },
        "host": s1.host_snapshot(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log("")
    log("4.3 SUMMARY")
    for h in history:
        log(
            f"  N = {h['n']:>2}: identical {h['identical']}, clean {h['clean']}, throughput {h['throughput']:.3f} "
            f"gen/s, {eligible(h, gpu_total, headroom)[1]}"
        )
    log(f"  stopped because: {why}")
    log(
        f"  persistent compilation cache in the workers (started exactly like the sweep): {cache_states}. "
        "False means every worker compiled from scratch, as the long run's processes do today"
    )
    log(f"  {summary['foreign_workload_note']}")
    log(f"  RECOMMENDATION: {rec['text']}")
    log(f"written {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
