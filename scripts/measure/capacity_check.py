#!/usr/bin/env python3
"""Does network capacity change what evolves, or only how fast. `unswept.md` entry 4.

**The question is NOT whether the parameter budget fits.** That is already answered: 43,478 / 69,814 /
99,222 parameters at `hd` 32 / 48 / 64, all inside 100,000. The entry's magnitude column says "effect on
strategy unmeasured", and this measures it.

**Capacity is one of the two levers identified as pushing systematically toward simple strategies**, the
other being the search parameters, and `unswept.md` records that the bias is UNCOMPENSATED: the entire
characterisation sweep consists of environment parameters, so nothing in it pushes back on capacity in
either direction. That is a fact about the design rather than a suspicion, which is what makes this worth
three hours.

**What is measured, and why it is accumulators rather than descriptors.** Run 1's convergence probe measured
fitness plus the twenty-one raw accumulators, and this measures the same things so the two are comparable.
Descriptors are functions of accumulators, so accumulator-level differences imply descriptor-level ones
whatever the analysis settles on, and using the raw set avoids committing this measurement to a descriptor
definition that step 12 may still move.

**Beware the proxy, which `unswept.md` names for exactly this measurement.** Capacity measured by parameter
count rather than by what the network can represent is the tempting proxy and it is the wrong quantity: the
budget is already known to fit at every level. Fitness alone is nearly as bad, because a capacity effect that
shows only in fitness is a claim about learning SPEED, and the entry asks about strategy COMPLEXITY. So the
accumulators are the primary read and fitness is context.

**A null here is informative and must not be read as absence.** If 32, 48 and 64 produce indistinguishable
accumulator profiles, that is evidence the strategies being found are simple enough that a 1140-to-32
bottleneck does not bind, which is itself the concern `unswept.md` raises. Stating that in advance so a flat
result is not quietly discarded.

**Prediction, recorded before the run**, per D-041's rule, since a result this expensive should not be
interpreted after the fact:

    fitness       rises weakly with hd, if at all. Generation zero is about 98 percent still and the
                  search spends its early budget escaping that, which is not a capacity-limited problem
    accumulators  the honest expectation is NO clear separation at 900 generations, because run 1 shows
                  no anti-predator signal at all at this configuration, so there is little strategy
                  present for capacity to constrain
    the risk      that a difference appears and is read as a capacity effect when it is seed noise.
                  Three seeds cannot separate those, which is Q-030's problem arriving here

Usage: uv run python scripts/measure/capacity_check.py <out_dir>
"""

import dataclasses
import math
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import check_invariants, n_params
from configs.run_configs import SCRIPTED
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius
from src.evo import genome, loop

HIDDEN_SIZES = (32, 48, 64)
SEEDS = (0, 1, 2)
N_GENERATIONS = 900
SAMPLE_EVERY = 25

ACC_FIELDS = E.Accumulators._fields

#: Appended to every accumulator vector so the freeze criterion's currency is RECORDED rather than
#: proxied. `acc_fields` in the npz carries both, so a reader gets the names with the numbers.
OUTCOME_FIELDS = ("predated", "starved", "survived")


def rss_mb():
    """Current resident set size. Not `ru_maxrss`, which is a high-water mark and cannot fall."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def make_accumulator_pass(unflatten, static, params):
    """Population mean of every accumulator, from a fresh evaluation of the CURRENT population.

    A factory, per DEV-029: defining the per-genome function inside the call retains a compiled executable at
    every sampling generation.
    """

    def one(vec, k):
        k_reset, k_draws = jax.random.split(k)
        st, acc = E.run_episode(k_reset, unflatten(vec), static, ent.PRNGSource(k_draws), params)
        # **Tier-one OUTCOMES appended to the accumulator vector. Added 2026-08-26.**
        #
        # The revised freeze criterion's currency is the predated / starved / survived fractions. This
        # script recorded fitness and the 21 accumulators and nothing else, so runs 11 and 12 could be
        # classified only on a PROXY, `alive_ticks`, which is a duration rather than an outcome fraction.
        #
        # Three extra floats per genome. The omission cost a disclosed substitution in a freeze
        # classification, which is a great deal more than three floats.
        outcomes = jnp.stack(
            [
                jnp.mean(st.prey_died_predated.astype(jnp.float32)),
                jnp.mean(st.prey_died_starving.astype(jnp.float32)),
                jnp.mean(st.prey_alive.astype(jnp.float32)),
            ]
        )
        base = jnp.stack([jnp.mean(getattr(acc, f).astype(jnp.float32)) for f in ACC_FIELDS])
        return jnp.concatenate([base, outcomes])

    batched = jax.jit(jax.vmap(one, in_axes=(0, 0)))

    def run(key, genomes):
        keys = jax.random.split(key, params.p_pop)
        return np.asarray(jnp.mean(batched(genomes, keys), axis=0))

    return run


#: **Hold the expected mutation NORM constant instead of the per-weight scale. Owner's ruling, Q-048.**
#:
#: Run 8 varied `hd` with `sigma` fixed at 0.01 per weight, so the expected L2 norm of one mutation scaled as
#: `sqrt(n_params)` and hd 64 received a 1.51x larger total perturbation than hd 32. The run therefore varied
#: capacity AND effective mutation strength together, and the one signal that survived, seed variance growing
#: with capacity, is exactly what a larger step size produces. It could not be attributed.
#:
#:     sigma = SIGMA_REF * sqrt(N_REF / n_params)      hd 32  0.01267
#:                                                     hd 48  0.01000
#:                                                     hd 64  0.00839
#:
#: **This removes the confound; it does not answer the mutation-scale question**, which is Q-046 and run 11.
#: And it trades one objection for a smaller one: the arms now differ in per-weight step size instead of in
#: total perturbation. That is stated rather than hidden, because there is no normalisation that makes both
#: comparable and pretending otherwise would be the more serious error.
#: **Both anchors are DERIVED, not transcribed.** A first version hardcoded 0.01 and 69,814, which are
#: correct today and are exactly the kind of stated fact that drifts: if the owner changes the default
#: `sigma` after run 11, or `hd` moves, a literal anchor would silently stop meaning "the default".
SIGMA_REF = SCRIPTED.params.sigma
N_REF = n_params(SCRIPTED.params)

#: Set False to reproduce run 8 exactly, which is the only reason it is a flag rather than a fixed change.
SCALE_SIGMA = True


def sigma_for(n_params_at_hd: int) -> float:
    """Constant expected mutation norm across hidden sizes."""
    return SIGMA_REF * math.sqrt(N_REF / n_params_at_hd) if SCALE_SIGMA else SIGMA_REF


def one_run(hd: int, seed: int, out_dir: pathlib.Path) -> dict:
    base = dataclasses.replace(SCRIPTED.params, hd=hd)
    params = dataclasses.replace(base, sigma=sigma_for(n_params(base)))
    check_invariants(params)

    ob = grid.make_obstacle_grid(jax.random.key(seed), params.h, params.w, params.obstacle_density)
    fs, ts, relax = resources.place_sources(jax.random.key(seed), ob, params)
    static = E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(params)), food_src=fs, target_src=ts
    )

    template = net.init_params(jax.random.key(seed), params)
    unflatten = genome.unflattener(template)
    pop = loop.initial_population(jax.random.key(seed + 1), template, params)

    step = loop.make_generation_step(unflatten, static, params)
    acc_pass = make_accumulator_pass(unflatten, static, params)
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
                f"    gen {gen:4d}  best {f.max():7.3f}  mean {f.mean():7.3f}  "
                f"nonzero {int((f > 0).sum()):3d}/{params.p_pop}  {el / (gen + 1):.2f}s/gen  "
                f"elapsed {el / 60:.1f}m  rss {rss_mb():.0f}MB",
                flush=True,
            )

    wall = time.time() - t0
    path = out_dir / f"hd{hd}_seed{seed}.npz"
    np.savez(
        path,
        hd=np.int32(hd),
        seed=np.int32(seed),
        sigma=np.float64(params.sigma),
        sigma_scaled=np.bool_(SCALE_SIGMA),
        n_params=np.int32(n_params(params)),
        config_hash=np.array(loop.config_hash(params)),
        relax_level=np.int32(relax),
        fit_max=fit_max,
        fit_mean=fit_mean,
        fit_nonzero=fit_nonzero,
        sample_gens=np.array(sample_gens, np.int32),
        sample_acc=np.stack(sample_acc),
        acc_fields=np.array(ACC_FIELDS + OUTCOME_FIELDS),
        wall_seconds=np.float32(wall),
    )
    print(f"  hd {hd} seed {seed} done in {wall / 60:.1f} min -> {path}", flush=True)
    return {"hd": hd, "seed": seed, "final_max": float(fit_max[-1]), "final_mean": float(fit_mean[-1])}


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    out_dir = pathlib.Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("\nparameter budget at each level, from n_params, not transcribed:", flush=True)
    for hd in HIDDEN_SIZES:
        p = dataclasses.replace(SCRIPTED.params, hd=hd)
        check_invariants(p)
        npar = n_params(p)
        norm = math.sqrt(npar) * sigma_for(npar)
        print(
            f"  hd {hd:3d}  {npar:,} parameters  sigma {sigma_for(npar):.5f}  "
            f"expected mutation norm {norm:.3f}",
            flush=True,
        )

    rows = []
    # **Hidden size OUTERMOST.** It is the shape-changing parameter, so every seed at one hd shares a
    # compiled program and the sweep pays three compilations rather than nine. D-026.
    for hd in HIDDEN_SIZES:
        for seed in SEEDS:
            print(f"\n=== hd {hd}, seed {seed} ===", flush=True)
            rows.append(one_run(hd, seed, out_dir))
            done, total = len(rows), len(HIDDEN_SIZES) * len(SEEDS)
            el = time.time() - t0
            print(
                f"  progress {done}/{total}, elapsed {el / 60:.1f} min, "
                f"projected total {el / done * total / 60:.0f} min",
                flush=True,
            )

    print("\n" + "=" * 70, flush=True)
    print(f"{'hd':>5s} {'seed':>5s} {'final best':>11s} {'final mean':>11s}", flush=True)
    for r in rows:
        print(f"{r['hd']:5d} {r['seed']:5d} {r['final_max']:11.3f} {r['final_mean']:11.3f}", flush=True)
    print("=" * 70, flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
