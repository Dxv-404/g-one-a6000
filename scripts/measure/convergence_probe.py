#!/usr/bin/env python3
"""Convergence probe. Produced run 1 in `runs.md`. Build step 10.

**This file is here under an unresolved tension between two rules, and Q-039 asks the owner to settle it.**

CLAUDE.md section 11 says scaffolding does not live in the repository: a diagnostic script's RESULT is the
artifact, and the script is deleted once it has answered its question. The stated exception is a falsification
script that will be re-run when its gate changes.

`runs.md` says the opposite thing about a different property: "a result that is not in this registry cannot be
reproduced, and under the reporting standard it cannot be published either", and the registry exists for
"whoever has to reproduce a number in a paper eighteen months from now".

Run 1's numbers say `g_generations = 300` captures 45.9 percent of the 900-generation fitness gain. That is
evidence for changing a scientific parameter, so it is exactly the class of number the registry is for. Its
driver living in `/tmp` would make the run unreproducible the moment the machine restarts.

**Kept rather than deleted because deleting the only copy is the irreversible option and keeping it costs a
file.** If the owner decides scaffolding wins, delete this and record in `runs.md` that run 1's method is
reconstructible from `src/evo/loop.py` plus the SCRIPTED config, which is true but weaker than a script.

Usage: uv run python scripts/measure/convergence_probe.py <seed> <out.npz>
"""

# The original probe's reasoning, retained because it is why the measurement is trustworthy:
#
# **What is measured, and why it is not descriptors.** Step 11 is held pending the analysis inversion, so the
# descriptors and the classifier do not exist and must not be built. This probes FITNESS and the twenty-one
# raw ACCUMULATORS instead. That is not a substitute: descriptors are functions of accumulators, so
# accumulator level convergence implies descriptor level convergence whatever the inversion decides the
# descriptors are.
#
# **The accumulator pass is a separate clean evaluation of the current population**, not a copy of the GA's
# evaluation path, which would be the degrade-toward-the-sloppier-copy hazard.
#
# **Both the step and the accumulator pass are factories.** Building either inside the loop retains a
# compiled executable per call: 31.5 MB a generation, which is DEV-029 and which OOM-killed the first run of
# this probe at generation 475 of 900.

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.run_configs import SCRIPTED
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table
from src.evo import genome, loop

P = SCRIPTED.params
N_GENERATIONS = 900
SAMPLE_EVERY = 25

ACC_FIELDS = E.Accumulators._fields


def rss_mb():
    """Current resident set size. Not `ru_maxrss`, which is a high-water mark and cannot fall."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def make_accumulator_pass(unflatten, static):
    """Population mean of every accumulator, from a fresh evaluation of the CURRENT population.

    Deliberately its own episodes rather than reusing the generation's: the generation's evaluation is what
    selection acted on and is already consumed, and a probe that changed the GA's path to observe it would be
    measuring a different algorithm from the one being reported.

    **A factory for the same reason `make_generation_step` is one.** The first version of this defined the
    per-genome function inside the call, so it retained an executable at every sampling generation: 36 samples
    per seed at 31 MB each is about 1.1 GB, on top of the defect it was written to investigate. Caught by
    reading this file after fixing DEV-029 rather than by anything failing.
    """

    def one(vec, k):
        k_reset, k_draws = jax.random.split(k)
        _, acc = E.run_episode(k_reset, unflatten(vec), static, ent.PRNGSource(k_draws), P)
        return jnp.stack([jnp.mean(getattr(acc, f).astype(jnp.float32)) for f in ACC_FIELDS])

    batched = jax.jit(jax.vmap(one, in_axes=(0, 0)))

    def run(key, genomes):
        keys = jax.random.split(key, P.p_pop)
        return np.asarray(jnp.mean(batched(genomes, keys), axis=0))

    return run


def main() -> int:
    seed, out = int(sys.argv[1]), sys.argv[2]
    runtime = src.assert_runtime_took_effect()
    print(f"seed {seed}: {runtime}", flush=True)
    print(f"config hash {loop.config_hash(P)}", flush=True)

    ob = grid.make_obstacle_grid(jax.random.key(seed), P.h, P.w, P.obstacle_density)
    fs, ts, relax = resources.place_sources(jax.random.key(seed), ob, P)
    static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, P.r), food_src=fs, target_src=ts)

    template = net.init_params(jax.random.key(seed), P)
    unflatten = genome.unflattener(template)
    pop = loop.initial_population(jax.random.key(seed + 1), template, P)

    step = loop.make_generation_step(unflatten, static, P)
    acc_pass = make_accumulator_pass(unflatten, static)
    key = jax.random.key(seed + 2)
    fit_max = np.zeros(N_GENERATIONS, np.float32)
    fit_mean = np.zeros(N_GENERATIONS, np.float32)
    fit_nonzero = np.zeros(N_GENERATIONS, np.int32)
    sample_gens, sample_acc = [], []

    t0 = time.time()
    for gen in range(N_GENERATIONS):
        key, k_gen, k_acc = jax.random.split(key, 3)
        pop, fit = step(k_gen, pop)
        f = np.asarray(fit)
        fit_max[gen], fit_mean[gen], fit_nonzero[gen] = f.max(), f.mean(), int((f > 0).sum())

        if gen % SAMPLE_EVERY == 0 or gen == N_GENERATIONS - 1:
            sample_gens.append(gen)
            sample_acc.append(acc_pass(k_acc, pop))
            el = time.time() - t0
            print(
                f"  gen {gen:4d}  best {f.max():7.3f}  mean {f.mean():7.3f}  "
                f"nonzero {int((f > 0).sum()):3d}/{P.p_pop}  {el / (gen + 1):.2f}s/gen  "
                f"elapsed {el / 60:.1f}m  rss {rss_mb():.0f}MB",
                flush=True,
            )

    np.savez(
        out,
        seed=np.int32(seed),
        config_hash=np.array(loop.config_hash(P)),
        relax_level=np.int32(relax),
        fit_max=fit_max,
        fit_mean=fit_mean,
        fit_nonzero=fit_nonzero,
        sample_gens=np.array(sample_gens, np.int32),
        sample_acc=np.stack(sample_acc),
        acc_fields=np.array(ACC_FIELDS),
        wall_seconds=np.float32(time.time() - t0),
    )
    print(f"seed {seed} done in {(time.time() - t0) / 60:.1f} minutes -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
