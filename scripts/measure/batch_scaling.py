#!/usr/bin/env python3
"""Does this workload get FASTER per environment as the batch grows? The answer decides the sweep's cost.

**The question, and why it decides a refactor rather than a setting.** The sweep is 327,000 sequential
generations at about 1.31 s each, and only the innermost 1,024 environments run concurrently: the 10 seeds
of a condition and the 109 conditions are both Python loops. Batching the seeds would make one generation do
ten seeds' work at once, but `make_generation_step` closes over `static` and `params` deliberately, for the
callable-identity stability that fixed DEV-029, so batching seeds is a refactor of the evolutionary loop
rather than a flag.

**A refactor of the most safety-critical code in the project is only worth proposing if the throughput
actually scales**, and that is a timing question with two possible answers and opposite consequences:

    time is FLAT as the batch grows          launch-bound. The GPU is waiting on dispatch, not computing,
                                             so ten seeds at once cost about what one costs and the
                                             refactor is worth a great deal
    time scales LINEARLY with the batch      compute-bound. The GPU is already saturated, batching moves
                                             work around without reducing it, and only more GPUs help

**This is a TIMING benchmark, not a run.** It varies `p_pop` to widen the outer `vmap` and measures wall
clock. No scientific parameter is changed for any registered measurement, nothing is saved as a result, and
`p_pop` stays frozen at 256 everywhere that matters. Widening the batch here is how the scaling question is
asked; it is not a proposal to change the population.

**Compile time is excluded from the per-generation figure** by running one warm-up generation first, per the
standing rule that the first call poisons a throughput measurement.

Usage: uv run python scripts/measure/batch_scaling.py [--reps N]
"""

from __future__ import annotations

import dataclasses
import gc
import sys
import time

import jax
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import DEFAULT
from src.agent import network as net
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius
from src.evo import genome, loop

MULTIPLIERS = (1, 2, 4, 8)
REPS = 3


def gpu_bytes() -> float:
    try:
        st = jax.devices()[0].memory_stats() or {}
        return float(st.get("bytes_in_use", 0)) / (1024**3)
    except Exception:
        return float("nan")


def time_one(mult: int, reps: int):
    """Wall clock per generation at `mult` times the standard population."""
    p = dataclasses.replace(DEFAULT, p_pop=DEFAULT.p_pop * mult)
    seed = 0
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
    static = E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )
    template = net.init_params(jax.random.key(seed), p)
    pop = loop.initial_population(jax.random.key(seed + 1), template, p)
    step = loop.make_generation_step(genome.unflattener(template), static, p)

    t0 = time.time()
    pop, fit = step(jax.random.key(99), pop)
    jax.block_until_ready(fit)
    compile_s = time.time() - t0

    times = []
    for r in range(reps):
        t1 = time.time()
        pop, fit = step(jax.random.key(100 + r), pop)
        jax.block_until_ready(fit)
        times.append(time.time() - t1)

    envs = p.p_pop * p.n_episodes
    mem = gpu_bytes()
    jax.clear_caches()
    gc.collect()
    return dict(
        mult=mult,
        p_pop=p.p_pop,
        envs=envs,
        compile_s=compile_s,
        per_gen=float(np.median(times)),
        spread=float(max(times) - min(times)),
        mem_gib=mem,
    )


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    print(src.memory_headroom(), flush=True)
    global REPS
    if "--reps" in sys.argv:
        REPS = int(sys.argv[sys.argv.index("--reps") + 1])

    rows = []
    for m in MULTIPLIERS:
        try:
            r = time_one(m, REPS)
        except Exception as exc:  # a batch that does not fit is a RESULT, not a crash
            print(f"  x{m}: FAILED, {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            break
        rows.append(r)
        print(
            f"  x{m}: p_pop {r['p_pop']:5d}  envs {r['envs']:6d}  compile {r['compile_s']:6.1f}s  "
            f"per-gen {r['per_gen']:6.3f}s  spread {r['spread']:.3f}  mem {r['mem_gib']:.2f} GiB",
            flush=True,
        )

    base = rows[0]
    print("\n" + "=" * 100, flush=True)
    print(
        "SCALING. Perfect batching would hold per-generation time FLAT as environments multiply.", flush=True
    )
    print("=" * 100, flush=True)
    print(
        f"  {'batch':>6s} {'envs':>7s} {'per-gen s':>10s} {'vs x1':>8s} {'env-ticks/s':>14s} "
        f"{'efficiency':>11s}",
        flush=True,
    )
    for r in rows:
        slow = r["per_gen"] / base["per_gen"]
        thr = r["envs"] * DEFAULT.t_horizon / r["per_gen"]
        eff = thr / (base["envs"] * DEFAULT.t_horizon / base["per_gen"])
        print(
            f"  x{r['mult']:<5d} {r['envs']:7d} {r['per_gen']:10.3f} {slow:8.2f} {thr:14,.0f} {eff:10.2f}x",
            flush=True,
        )

    if len(rows) < 2:
        print("\n  Only one batch size completed; no scaling conclusion available.", flush=True)
        return 0

    top = rows[-1]
    slow = top["per_gen"] / base["per_gen"]
    gain = top["mult"] / slow
    print("\n" + "=" * 100, flush=True)
    print(
        f"  At x{top['mult']} the batch is {top['mult']}x larger and takes {slow:.2f}x the time,", flush=True
    )
    print(f"  so throughput is {gain:.2f}x. **That is the factor batching the seeds would buy.**", flush=True)
    if gain >= 2.0:
        hours = 119.2 / gain
        print(
            f"\n  LAUNCH-BOUND. The sweep's 119.2 h would become roughly {hours:.0f} h on this same GPU,",
            flush=True,
        )
        print(
            "  from batching alone, before any faster hardware. The refactor is worth proposing.", flush=True
        )
    elif gain >= 1.2:
        print(f"\n  PARTIALLY launch-bound. Batching buys {gain:.2f}x, which is real but modest.", flush=True)
    else:
        print("\n  COMPUTE-BOUND. The GPU is already saturated and batching moves work without", flush=True)
        print("  reducing it. Only more GPUs, or fewer conditions, shorten the sweep.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
