#!/usr/bin/env python3
"""The mutation scale, swept. Q-046, and it blocks any reported null result.

**Why this runs before step 12.** `configs/default.py` calls `sigma` the "LOWEST CONFIDENCE VALUE IN THIS
FILE" and instructs "Sweep it first". It was not in the swept set, and nothing acted on the instruction. Its
own docstring names the failure mode: a badly set mutation scale looks exactly like "evolution does not work
in this environment".

**The asymmetry that makes this urgent rather than merely missing.** An unswept ENVIRONMENT parameter risks a
result conditional on an unexamined setting. An unswept SEARCH parameter risks the NULL. This design
deliberately makes a null reportable and interesting, and a null caused by a badly set mutation scale is
indistinguishable from the real one while being a completely different finding. Owner's ruling, 2026-08-25.

**Levels are fixed here, in the file, before the run.** Log-spaced, an order of magnitude either side of the
default, per the ruling. Choosing them after seeing a curve is how a threshold gets fitted to its case.

**The prediction, recorded before running**, per D-041, with a decision rule attached so the result cannot be
interpreted to taste:

    shape       unimodal in log-sigma. 0.001 clearly worst, being too slow to move 70k parameters in
                900 generations; 0.1 degrading as mutation outruns what elitism can protect
    peak        at 0.0316 rather than at the default. Generation zero is about 98 percent still and the
                search spends its early budget escaping a degenerate start, which favours a larger step
    the number  final mean fitness at the default was 1.269 to 1.559 across hidden sizes in run 8

    DECISION RULE, fixed in advance
      if the best level is within the seed range of the default   sigma is adequately set, a null is safe
      if the best level beats the default by more than 2x         the current setting materially threatens
                                                                  any null and must be changed before
                                                                  Phase A reports one

**Three seeds is not enough to separate close levels and is enough for the decision rule above**, which asks
only whether the default is grossly wrong. Run 8 measured the within-configuration seed range directly, 0.155
to 0.809, so the comparison is against a measured noise floor rather than an assumed one.

Usage: uv run python scripts/measure/sigma_sweep.py <out_dir>
"""

import dataclasses
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import check_invariants
from configs.run_configs import SCRIPTED
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius
from src.evo import genome, loop

#: Log-spaced, decade either side of the default. FIXED BEFORE THE RUN.
SIGMA_LEVELS = (0.001, 0.00316, 0.01, 0.0316, 0.1)
SEEDS = (0, 1, 2)
N_GENERATIONS = 900
SAMPLE_EVERY = 25

ACC_FIELDS = E.Accumulators._fields

#: Appended to every accumulator vector so the freeze criterion's currency is RECORDED rather than
#: proxied. `acc_fields` in the npz carries both, so a reader gets the names with the numbers.
OUTCOME_FIELDS = ("predated", "starved", "survived")

#: Measured in run 8, so the decision rule compares against a real noise floor rather than an assumed one.
RUN8_SEED_RANGE = 0.809


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def make_accumulator_pass(unflatten, static, params):
    """A factory, per DEV-029: building this inside the loop retains an executable per call."""

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


def one_run(sigma: float, seed: int, out_dir: pathlib.Path) -> dict:
    params = dataclasses.replace(SCRIPTED.params, sigma=sigma)
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
    path = out_dir / f"sigma{sigma}_seed{seed}.npz"
    np.savez(
        path,
        sigma=np.float64(sigma),
        seed=np.int32(seed),
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
    print(f"  sigma {sigma} seed {seed} done in {wall / 60:.1f} min -> {path}", flush=True)
    return {"sigma": sigma, "seed": seed, "final_mean": float(fit_mean[-1]), "final_max": float(fit_max[-1])}


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    out_dir = pathlib.Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"\nlevels fixed before the run: {SIGMA_LEVELS}", flush=True)
    print(f"default is {SCRIPTED.params.sigma}, seeds {SEEDS}, {N_GENERATIONS} generations", flush=True)
    for lv in SIGMA_LEVELS:
        check_invariants(dataclasses.replace(SCRIPTED.params, sigma=lv))
    print(f"pre-flight: all {len(SIGMA_LEVELS)} levels satisfy check_invariants\n", flush=True)

    rows = []
    for sigma in SIGMA_LEVELS:
        for seed in SEEDS:
            print(f"\n=== sigma {sigma}, seed {seed} ===", flush=True)
            rows.append(one_run(sigma, seed, out_dir))
            done, total = len(rows), len(SIGMA_LEVELS) * len(SEEDS)
            el = time.time() - t0
            print(
                f"  progress {done}/{total}, elapsed {el / 60:.1f} min, "
                f"projected {el / done * total / 60:.0f} min",
                flush=True,
            )

    print("\n" + "=" * 78, flush=True)
    print(f"{'sigma':>9s} {'final mean, 3 seeds':>28s} {'mean':>8s} {'range':>8s}", flush=True)
    by_sigma = {}
    for sigma in SIGMA_LEVELS:
        vals = [r["final_mean"] for r in rows if r["sigma"] == sigma]
        by_sigma[sigma] = vals
        print(
            f"{sigma:9.5f} {str([round(v, 3) for v in vals]):>28s} "
            f"{np.mean(vals):8.3f} {max(vals) - min(vals):8.3f}",
            flush=True,
        )

    means = {s: float(np.mean(v)) for s, v in by_sigma.items()}
    best_sigma = max(means, key=means.get)
    default = SCRIPTED.params.sigma
    ratio = means[best_sigma] / means[default] if means[default] > 0 else float("inf")
    print("=" * 78, flush=True)
    print(f"\n  best level      sigma {best_sigma} at mean {means[best_sigma]:.3f}", flush=True)
    print(f"  default         sigma {default} at mean {means[default]:.3f}", flush=True)
    print(f"  ratio           {ratio:.2f}x", flush=True)
    print(
        f"  seed range at the best level  {max(by_sigma[best_sigma]) - min(by_sigma[best_sigma]):.3f}",
        flush=True,
    )
    print(f"  run 8 largest within-config seed range  {RUN8_SEED_RANGE:.3f}", flush=True)

    gap = means[best_sigma] - means[default]
    print("\n  DECISION RULE, fixed before the run:", flush=True)
    if gap <= RUN8_SEED_RANGE:
        print("    the best level is WITHIN the measured seed range of the default.", flush=True)
        print("    sigma is adequately set and a null result is safe to report.", flush=True)
    elif ratio > 2.0:
        print("    the best level beats the default by MORE THAN 2x.", flush=True)
        print("    the current setting materially threatens any null and must be changed", flush=True)
        print("    before Phase A reports one. This is a scientific parameter change and", flush=True)
        print("    therefore the owner's.", flush=True)
    else:
        print("    the best level beats the default by more than the seed range but less", flush=True)
        print("    than 2x. Neither branch of the rule fires cleanly; report the curve and", flush=True)
        print("    let the owner decide. Do NOT pick a side by re-reading the numbers.", flush=True)

    np.savez(
        out_dir / "summary.npz",
        sigma_levels=np.array(SIGMA_LEVELS),
        means=np.array([means[s] for s in SIGMA_LEVELS]),
        all_finals=np.array([[v for v in by_sigma[s]] for s in SIGMA_LEVELS]),
        wall_seconds=np.float32(time.time() - t0),
    )
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
