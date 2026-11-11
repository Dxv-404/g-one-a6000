#!/usr/bin/env python3
"""Pre-run checklist 4.3: how many evolutionary runs can share the A6000 at once, MEASURED, not assumed.

    .venv/bin/python -u scripts/measure/session1_concurrency.py runs/session1/concurrency
    options: --generations 50  --max-n 32  --flags=--xla_gpu_autotune_level=0  --cache shared|fresh
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

**Extended for the H100 (D-093).** Four things the A6000 version could not answer:

    cap        N goes to 32 (24 and 32 added), still stopping on a gain under 5 percent, because one small-network
               process does not fill an 80 GB card and a cap of 16 could hold the run below its best rate
    cards      `--gpus K`: the curve runs on card 0 with every worker pinned to it, as a long-run shard is. Then
               ONE round on every card at once, the recommended N per card, where each seed runs on EVERY card
               and must equal its solo reference bit for bit, so cards that disagree are found before the run.
               Memory is read per card. The shards share the host's cores and memory, which a one-card curve
               cannot see, so this round's throughput is the one the projection uses
    projection the plan's shape is read from sweep.py (conditions, those with a predator, seeds, generations, S4
               points), never transcribed. Each worker measures ONE in-run S4 at the long run's episode count
               after its timed loop, outside the throughput window, and its cost is converted to generations at
               that worker's own rate. The new-world null (Q-058) is projected as its own line
    heavy      solo timings of the costliest conditions (`HEAVY`) bound the estimate from above: the projection
               says what the run would take if every condition cost what the heaviest does
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
SEQUENCE = (2, 3, 4, 6, 8, 12, 16, 24, 32)
SEED_BASE = 200
N_CAP = 32
GAIN_FLOOR = 0.05  # stop when a clean round gains LESS than this over the previous clean round
GPU_HEADROOM_GIB = 8.0  # the recommendation and the stop rule both keep at least this much of the card free
HOST_HEADROOM_GIB = 4.0  # src.memory_headroom's floor for the sweep, D-054
#: The condition every curve round runs, and the costliest conditions of the plan, timed solo for the upper bound.
BASE_CONDITION = ("treatment", "baseline", None)
HEAVY = (("treatment", "n_pred", 6), ("treatment", "k", 12))
#: Q-058's default budget for the new-world null (the owner's to change); projected, never run here.
NULL_GENERATIONS = 300
BARRIER_TIMEOUT_S = 900.0
STATE_KEYS = ("final_pop", "fit_mean", "fit_max")


# ------------------------------------------------------------------------------------------------ worker ----
def worker(
    seed: int,
    generations: int,
    out: pathlib.Path,
    ready: pathlib.Path,
    go: pathlib.Path,
    condition: tuple = BASE_CONDITION,
) -> None:
    """One evolutionary run through the long run's own code, timed, with the start barrier in its progress hook.

    **One in-run S4 at the end**, at the long run's own episode count: `S4_AFTER` is set to the last generation in
    THIS process's copy of sweep.py, so `one_seed` measures exactly as the long run does. Its wall time is not in a
    generation's wall (one_seed measures after taking it), and the hook separates the two: `evo_end_time` closes the
    throughput window, `s4_seconds` is the measurement. Measurement keys never touch the evolution, so `final_pop`
    is the same with or without it, and the solo references and the rounds both include it."""
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
            self.prev, self.evo_end_time, self.s4_seconds = None, None, float("nan")

        def generation(self, gen, wall, fit, acc_row):
            now = time.time()
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
            elif gen == generations - 1 and self.prev is not None:
                self.evo_end_time, self.s4_seconds = split_last_generation(self.prev, float(wall), now)
            self.prev = time.time()

        def loop_end(self):
            self.loop_end_time = time.time()

    hook = Hook()
    arm, axis, level = condition
    p = sweep.params_for(arm, axis, level)
    sweep.S4_AFTER = (
        generations,
    )  # this process's copy only: one in-run S4, at the long run's episode count
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
        evo_end_time=np.float64(hook.evo_end_time if hook.evo_end_time is not None else np.nan),
        s4_seconds=np.float64(hook.s4_seconds),
        s4_episodes=np.int32(sweep.S4_EPISODES),
        has_predator=np.bool_(p.n_pred > 0),
        condition=np.array(json.dumps(list(condition))),
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


def split_last_generation(prev_call: float, wall: float, now: float) -> tuple[float, float]:
    """(evolution end, S4 seconds) of the last generation. `prev_call` is when the hook returned from the previous
    generation, `wall` the last generation's own wall (taken before the S4 pass), `now` when the hook is called
    after the S4 pass."""
    evo_end = prev_call + wall
    return evo_end, max(now - evo_end, 0.0)


def plan_shape(sweep) -> dict:
    """The long run's plan as sweep.py builds it: never transcribed, so it cannot drift from the run (D-093)."""
    conds = [(a, ax, lv) for a in sweep.ARMS for ax, lv in sweep.conditions(a)]
    return {
        "conditions": len(conds),
        "predator_conditions": sum(sweep.params_for(a, ax, lv).n_pred > 0 for a, ax, lv in conds),
        "null_conditions": sum(a == "null" for a, _, _ in conds),
        "seeds": len(sweep.seed_list()),
        "generations": sweep.N_GENERATIONS,
        "s4_points": len(sweep.S4_AFTER),
        "s4_episodes": sweep.S4_EPISODES,
    }


def s4_generation_equivalent(workers: dict) -> float:
    """Median over workers of one S4 measurement in generations at that worker's own loop rate."""
    v = [
        w["s4_seconds"] * w["loop_rate"]
        for w in workers.values()
        if w.get("has_predator")
        and np.isfinite(w.get("s4_seconds", np.nan))
        and np.isfinite(w.get("loop_rate", np.nan))
    ]
    return float(statistics.median(v)) if v else float("nan")


def project(
    throughput: float,
    shape: dict,
    s4_gen_equiv: float,
    heavy_ratio: float = 1.0,
    null_generations: int = NULL_GENERATIONS,
) -> dict:
    """Days for the long run at `throughput` generations per second, the in-run S4 included, and the null job."""
    evo = shape["conditions"] * shape["seeds"] * shape["generations"]
    s4 = shape["predator_conditions"] * shape["seeds"] * shape["s4_points"] * s4_gen_equiv
    total = evo + s4
    days = total / throughput / 86400.0
    return {
        "evo_generations": evo,
        "s4_generation_equivalents": s4,
        "total_generation_equivalents": total,
        "days": days,
        "heavy_ratio": max(heavy_ratio, 1.0),
        "days_if_every_condition_cost_the_heaviest": days * max(heavy_ratio, 1.0),
        "null_job_generations": shape["null_conditions"] * shape["seeds"] * null_generations,
        "null_job_days": shape["null_conditions"] * shape["seeds"] * null_generations / throughput / 86400.0,
        "null_generations": null_generations,
    }


def card_plan(n_per_card: int, n_cards: int) -> list[tuple[int, int]]:
    """(seed, card) per worker for the every-card round: each card runs `n_per_card` workers, and every seed runs
    on EVERY card, so its result on each card is compared with the same solo reference."""
    return [(SEED_BASE + i // n_cards, i % n_cards) for i in range(n_per_card * n_cards)]


def cards_eligible(
    r: dict, identical: bool, gpu_total_mib: float | None, gpu_headroom_gib: float
) -> tuple[bool, str]:
    if r["failed"]:
        return False, "a worker failed"
    if not identical:
        return False, "not bit-identical across cards"
    per = r["gpu"].get("per_card") or {}
    if not per or gpu_total_mib is None:
        return False, "no per-card GPU memory reading"
    for i, c in per.items():
        free = (gpu_total_mib - c.get("mem_used_mib_max", float("inf"))) / 1024
        if free < gpu_headroom_gib:
            return False, f"card {i} peak left {free:.1f} GiB free, under {gpu_headroom_gib:.0f}"
    return True, "eligible"


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
    """The fastest round passing bit-identity, cleanliness and memory headroom, then one step lower. The day count
    here is evolution only; `project` adds the in-run S4 and the null job."""
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
    days = plan_generations / pick["throughput"] / 86400.0 if plan_generations else None
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
            + f"): {pick['throughput']:.2f} generations per second"
            + (f", so about {days:.1f} days for {plan_generations:,} generations" if days is not None else "")
            + ", CONDITIONAL on the GPU's other load staying as it was during this measurement"
        ),
    }


# ------------------------------------------------------------------------------------------- driver ----
def worker_env(flags: str, cache_dir: str | None) -> dict:
    return dd.clean_env(cache_dir, flags)


def run_round(
    seeds,
    generations,
    flags,
    cache_mode,
    round_dir: pathlib.Path,
    log,
    *,
    condition: tuple = BASE_CONDITION,
    pin: str | None = None,
    cards: list | None = None,
) -> dict:
    """N workers together behind the start barrier. Never raises: a failure is recorded.

    `pin` sets CUDA_VISIBLE_DEVICES for every worker (the curve on card 0 of a several-card machine). `cards` gives
    each worker its own card, for the every-card round; workers are then keyed by position, since a seed repeats."""
    round_dir.mkdir(parents=True, exist_ok=True)
    go = round_dir / "GO"
    go.unlink(missing_ok=True)
    procs, outs, readies, tmps = [], {}, {}, []
    keyed = cards is not None
    for i, s0 in enumerate(seeds):
        s = i if keyed else s0
        stem = f"w{i:03d}_seed{s0}" if keyed else f"seed{s0}"
        cache = None
        if cache_mode == "fresh":
            t = tempfile.mkdtemp(prefix=f"s1cache-{s0}-", dir=str(round_dir))
            tmps.append(t)
            cache = t
        outs[s] = round_dir / f"{stem}.npz"
        readies[s] = round_dir / f"{stem}.ready"
        outs[s].unlink(missing_ok=True)
        readies[s].unlink(missing_ok=True)
        logf = open(round_dir / f"{stem}.log", "w")
        env = worker_env(flags, cache)
        device = str(cards[i]) if keyed else pin
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = device
        procs.append(
            (
                s,
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        __file__,
                        "--worker",
                        str(s0),
                        str(generations),
                        str(outs[s]),
                        str(readies[s]),
                        str(go),
                        json.dumps(list(condition)),
                    ],
                    env=env,
                    cwd=str(ROOT),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                ),
                logf,
            )
        )
    t_launch = time.time()
    release = None
    with s1.Sampler(every=1.0, all_cards=keyed) as smp:
        while release is None:
            if all(r.exists() for r in readies.values()) or any(p.poll() is not None for _, p, _ in procs):
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
            rec["log_tail"] = outs[s].with_suffix(".log").read_text()[-1500:]
        else:
            with np.load(outs[s]) as z:
                rec.update(
                    loop_rate=loop_rate(z["walls"]),
                    compile_seconds=compile_seconds(z["walls"]),
                    loop_end_time=float(z["loop_end_time"]),
                    evo_end_time=float(z["evo_end_time"]),
                    s4_seconds=float(z["s4_seconds"]),
                    has_predator=bool(z["has_predator"]),
                    seed=int(z["seed"]),
                    barrier_timed_out=bool(z["barrier_timed_out"]),
                    events=json.loads(str(z["events"])),
                    persistent_cache_enabled=bool(z["persistent_cache_enabled"]),
                )
            if rec["barrier_timed_out"]:
                failed = True
        workers[s] = rec
    # The window closes when the EVOLUTION ends, before each worker's S4 measurement (D-093)
    ends = [
        w["evo_end_time"] if np.isfinite(w.get("evo_end_time", np.nan)) else w["loop_end_time"]
        for w in workers.values()
        if "loop_end_time" in w
    ]
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
            int(argv[1]),
            int(argv[2]),
            pathlib.Path(argv[3]),
            pathlib.Path(argv[4]),
            pathlib.Path(argv[5]),
            tuple(json.loads(argv[6])) if len(argv) > 6 else BASE_CONDITION,
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
    ap.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="cards on this machine: the curve runs on card 0, then one round on every card at once (D-093)",
    )
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    n_cards = max(a.gpus, 1)
    pin = "0" if n_cards > 1 and not a.smoke else None
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
        "used. This is whatever else is on the GPU (the desktop, any other job); every round is compared with it."
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
            r = run_round([s], G, a.flags, a.cache, out / "solo" / f"seed{s}", log, pin=pin)
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
        r = run_round(seeds, G, a.flags, a.cache, out / f"round_n{n:02d}", log, pin=pin)
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

    # ---- the plan's shape, from sweep.py itself
    spec = importlib.util.spec_from_file_location("sweep_s1d", ROOT / "scripts" / "measure" / "sweep.py")
    sweep_mod = importlib.util.module_from_spec(spec)
    sys.modules["sweep_s1d"] = sweep_mod
    spec.loader.exec_module(sweep_mod)
    shape = plan_shape(sweep_mod)
    log(
        f"the plan (from sweep.py): {shape['conditions']} conditions, {shape['predator_conditions']} with a predator, "
        f"{shape['seeds']} seeds, {shape['generations']} generations, S4 at {shape['s4_points']} points over "
        f"{shape['s4_episodes']} episodes"
    )

    # ---- the heaviest conditions, solo, for the upper bound
    base_rate = statistics.median(solo_rates) if solo_rates else float("nan")
    heavy = {}
    for cond in HEAVY:
        hr = run_round(
            [SEED_BASE],
            G,
            a.flags,
            a.cache,
            out / "heavy" / "__".join(map(str, cond)),
            log,
            condition=cond,
            pin=pin,
        )
        w = hr["workers"][SEED_BASE]
        heavy["__".join(map(str, cond))] = {
            "failed": hr["failed"],
            "loop_rate": w.get("loop_rate"),
            "cost_ratio": (base_rate / w["loop_rate"]) if not hr["failed"] and w.get("loop_rate") else None,
        }
    ratios = [h["cost_ratio"] for h in heavy.values() if h["cost_ratio"] and np.isfinite(h["cost_ratio"])]
    heavy_ratio = max(ratios) if ratios else float("nan")
    log(
        f"heaviest conditions, solo, per-generation cost against the baseline: {json.dumps(heavy, default=str)}"
    )

    # ---- every card at once
    by_n = {rr["n"]: rr for rr in rounds}
    n_card = rec["n"]
    if a.smoke and n_card is None:
        n_card = max((h["n"] for h in history if h["identical"] and not h["failed"]), default=None)
    multi = None
    if n_cards > 1 and n_card is not None:
        plan_ = card_plan(n_card, n_cards)
        seeds_m = [s for s, _ in plan_]
        devices = [0 if a.smoke else c for _, c in plan_]  # a smoke run has one card and simulates the others
        ensure_refs(sorted(set(seeds_m)))
        rm = run_round(
            seeds_m, G, a.flags, a.cache, out / f"cards_k{n_cards}_n{n_card:02d}", log, cards=devices
        )
        ident_m = {}
        for i, s in enumerate(seeds_m):
            f = out / f"cards_k{n_cards}_n{n_card:02d}" / f"w{i:03d}_seed{s}.npz"
            ident_m[f"w{i:03d}_seed{s}_card{plan_[i][1]}"] = (
                compare_state(refs[s], load_state(f))
                if f.exists() and refs.get(s) is not None
                else {"identical": False}
            )
        all_m = all(v["identical"] for v in ident_m.values()) and not rm["failed"]
        ok_m, why_m = cards_eligible(rm, all_m, gpu_total, headroom)
        multi = {
            "cards": n_cards,
            "per_card": n_card,
            "workers": len(seeds_m),
            "identical": all_m,
            "eligible": ok_m,
            "why": why_m,
            "throughput": rm["throughput"],
            "gpu_per_card": rm["gpu"].get("per_card"),
            "host_load1_max": rm["gpu"].get("host_load1_max"),
            "identity": ident_m,
            "workers_detail": {str(k): v for k, v in rm["workers"].items()},
            "simulated": bool(a.smoke),
        }
        log(
            f"EVERY CARD: {n_card} per card x {n_cards} cards = {len(seeds_m)} workers, identical across cards "
            f"{all_m}, throughput {rm['throughput']:.3f} gen/s, {why_m}"
        )

    # ---- the projection: the in-run S4 included, the heaviest as an upper bound, the null job as its own line
    projection = None
    if rec["n"] is not None or (multi and multi["eligible"]):
        use_multi = bool(multi and multi["eligible"])
        tp = multi["throughput"] if use_multi else rec["throughput"]
        src_workers = multi["workers_detail"] if use_multi else by_n.get(rec["n"], {}).get("workers", {})
        if not src_workers:  # N 1 recommended: the solo references are its round
            src_workers = {str(x["seed"]): x["worker"] for x in solo_records if "worker" in x}
        s4_eq = s4_generation_equivalent(src_workers)
        projection = project(tp, shape, s4_eq, heavy_ratio if np.isfinite(heavy_ratio) else 1.0)
        projection.update(
            basis="every card at once" if use_multi else f"card 0, N = {rec['n']}",
            throughput=tp,
            s4_generation_equivalent=s4_eq,
            command=(
                f"bash scripts/a6000/run.sh longrun --shards={multi['workers']} --gpus={n_cards}"
                if use_multi
                else f"bash scripts/a6000/run.sh longrun --shards={rec['n']}"
            ),
        )
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
        "plan_shape": shape,
        "heavy": heavy,
        "every_card": multi,
        "projection": projection,
        "foreign_workload_note": (
            "Throughput was measured beside the BACKGROUND GPU LOAD logged at the start. The long-run day "
            "estimate holds only if the GPU's other load stays as it was."
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
    if multi is not None:
        log(f"  EVERY CARD: {multi['per_card']} per card on {multi['cards']} cards, {multi['why']}")
    if projection is not None:
        log(
            f"  PROJECTION ({projection['basis']}, {projection['throughput']:.2f} gen/s): {projection['days']:.1f} days "
            f"for the long run, of which the in-run S4 is {projection['s4_generation_equivalents']:,.0f} of "
            f"{projection['total_generation_equivalents']:,.0f} generation-equivalents; at most "
            f"{projection['days_if_every_condition_cost_the_heaviest']:.1f} days if every condition cost the heaviest "
            f"(x{projection['heavy_ratio']:.2f}); the new-world null at {projection['null_generations']} generations "
            f"{projection['null_job_days']:.1f} days more if run after it"
        )
        log(f"  LAUNCH: {projection['command']}")
    else:
        log("  PROJECTION: none, because nothing was eligible")
    log(f"written {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
