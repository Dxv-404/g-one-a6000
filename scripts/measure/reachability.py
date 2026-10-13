#!/usr/bin/env python3
"""Battery D: what does the search actually reach in its first generations? Entry 0, measured at last.

**The bias, as recorded and never measured.** Generation zero is about 98 percent still under deterministic
argmax with zero-initialised biases. Freezing IS stillness, so the search STARTS at the one strategy the
project has repeatedly narrowed onto, while every other strategy requires evolving competent movement first.
The owner ranked this above every parameter bias precisely because nothing had ever measured it.

**The question, with its branches stated before the numbers:**

    stillness falls as foraging competence arrives, and no engaged-stillness signal precedes it
        the degenerate start is ESCAPED, not exploited: selection pays for movement first, and any
        later freezing would have to re-evolve stillness against that gradient. The bias is real but
        selection runs AGAINST it initially, which weakens it as an alternative explanation

    an engaged-stillness or survival signal appears BEFORE foraging competence
        the reachability bias is live: the search finds the strategy adjacent to its start before it
        finds the task. Any freezing in the sweep then has this as its first alternative explanation,
        with a measured mechanism rather than an argument

**Per-generation sampling, which the long runs never did.** Runs 1, 8, 11 and 12 sampled every 25
generations, so the first moments of the search, where the bias lives, were never observed. This samples
every generation for the first 60.

Usage: uv run python scripts/measure/reachability.py <out.npz> [--seeds N]
"""

from __future__ import annotations

import gc
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
from src.env.perception import build_vis_table, required_radius
from src.evo import genome, loop

N_SEEDS = 4
N_GENERATIONS = 60

ACC_FIELDS = E.Accumulators._fields
OUT_FIELDS = ("predated", "starved", "survived")


def make_accumulator_pass(unflatten, static, params):
    """A factory, per DEV-029."""

    def one(vec, k):
        k_reset, k_draws = jax.random.split(k)
        st, acc = E.run_episode(k_reset, unflatten(vec), static, ent.PRNGSource(k_draws), params)
        base = jnp.stack([jnp.mean(getattr(acc, f).astype(jnp.float32)) for f in ACC_FIELDS])
        out = jnp.stack(
            [
                jnp.mean(st.prey_died_predated.astype(jnp.float32)),
                jnp.mean(st.prey_died_starving.astype(jnp.float32)),
                jnp.mean(st.prey_alive.astype(jnp.float32)),
            ]
        )
        return jnp.concatenate([base, out])

    batched = jax.jit(jax.vmap(one, in_axes=(0, 0)))

    def run(key, genomes):
        keys = jax.random.split(key, params.p_pop)
        return np.asarray(jnp.mean(batched(genomes, keys), axis=0))

    return run


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()
    global N_SEEDS
    if "--seeds" in sys.argv:
        N_SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1])

    p = SCRIPTED.params
    fields = list(ACC_FIELDS) + list(OUT_FIELDS)
    i_moved = fields.index("moved_ticks")
    i_alive = fields.index("alive_ticks")
    i_se = fields.index("still_engaged")
    i_eng = fields.index("engaged_ticks")
    i_fe = fields.index("flee_engaged")
    i_pred = fields.index("predated")

    all_acc, all_fit = [], []
    for seed in range(N_SEEDS):
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(
            obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
        )
        template = net.init_params(jax.random.key(seed), p)
        unflatten = genome.unflattener(template)
        pop = loop.initial_population(jax.random.key(seed + 1), template, p)
        step = loop.make_generation_step(unflatten, static, p)
        acc_pass = make_accumulator_pass(unflatten, static, p)
        key = jax.random.key(seed + 2)

        acc_gen, fit_gen = [], []
        for gen in range(N_GENERATIONS):
            key, kg, ka = jax.random.split(key, 3)
            # Sample BEFORE the step at generation 0, so the initial population itself is observed.
            if gen == 0:
                acc_gen.append(acc_pass(ka, pop))
                fit_gen.append(0.0)
            pop, fit = step(kg, pop)
            key, ka = jax.random.split(key)
            acc_gen.append(acc_pass(ka, pop))
            fit_gen.append(float(np.asarray(fit).mean()))
        all_acc.append(np.stack(acc_gen))
        all_fit.append(np.array(fit_gen))
        print(f"  seed {seed} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
        jax.clear_caches()
        gc.collect()

    A = np.stack(all_acc)  # (seed, gen+1, field)
    F = np.stack(all_fit)
    m = A.mean(axis=0)

    print("\n" + "=" * 98, flush=True)
    print(
        "THE FIRST 60 GENERATIONS, population means over seeds. Generation 0 is the INITIAL population.",
        flush=True,
    )
    print("=" * 98, flush=True)
    print(
        f"{'gen':>4s} {'fitness':>8s} {'moved/alive':>12s} {'engaged':>8s} {'still_eng':>10s} "
        f"{'flee_eng':>9s} {'predated':>9s}",
        flush=True,
    )
    for g in range(0, A.shape[1], 3):
        mv = m[g, i_moved] / max(m[g, i_alive], 1e-9)
        print(
            f"{g:4d} {F.mean(axis=0)[g]:8.3f} {mv:12.3f} {m[g, i_eng]:8.3f} {m[g, i_se]:10.4f} "
            f"{m[g, i_fe]:9.4f} {m[g, i_pred]:9.3f}",
            flush=True,
        )

    print("\n" + "=" * 98, flush=True)
    print("THE ORDERING QUESTION, answered from the same table", flush=True)
    print("=" * 98, flush=True)
    mv0 = m[0, i_moved] / max(m[0, i_alive], 1e-9)
    mvN = m[-1, i_moved] / max(m[-1, i_alive], 1e-9)
    print(f"  movement fraction, generation 0 -> {N_GENERATIONS}:   {mv0:.3f} -> {mvN:.3f}", flush=True)
    se_max_gen = int(np.argmax(m[:, i_se]))
    print(f"  still_engaged peaks at generation {se_max_gen} with {m[se_max_gen, i_se]:.4f}", flush=True)
    half_fit = np.argmax(F.mean(axis=0) >= 0.5 * F.mean(axis=0)[-1]) if F.mean(axis=0)[-1] > 0 else -1
    print(f"  fitness reaches half its gen-{N_GENERATIONS} value at generation {half_fit}", flush=True)
    print(
        "\n  Reading: if still_engaged carries any early signal it peaks BEFORE fitness competence and",
        flush=True,
    )
    print("  decays as movement is selected for; the branch table in the docstring applies.", flush=True)

    np.savez(
        sys.argv[1],
        acc=A,
        fit=F,
        fields=np.array(fields),
        n_seeds=np.int32(N_SEEDS),
        wall_seconds=np.float32(time.time() - t0),
    )
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
