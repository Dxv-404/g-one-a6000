#!/usr/bin/env python3
"""Measurement M7, the variance half: how much of a fitness gradient does selection actually have to act on?

**Not to be confused with `selection_gradient.py`, run 19**, which asks whether DYING costs fitness under a
fixed policy, the instrumental route. This asks whether the POPULATION carries a fitness spread that the
tournament can act on, generation by generation. Two quantities that both answer to "selection gradient",
which is the name-collision class CLAUDE.md records, so the two are named apart here.

**Why this is measured before the sweep rather than read off it.** Gate S3 asks whether fitness variance is
"meaningfully above zero", the spec gives no number, and run 23 found that at generation zero 92 to 95
percent of genomes score exactly zero and only 13 to 23 percent of tournaments are decided by fitness. Q-050
asks the owner to set the number. This script is the data that question is answered from: the same
quantities over enough generations to see where they go, on more than one seed.

**What is reported per generation**, on the evaluated population before selection:

    mean, max            the usual fitness curve
    var                  the spec's own quantity
    distinct             how many different values the tournament can distinguish
    zero_frac            the fraction of genomes that collected nothing
    decided              1 - sum_v (n_v/N)^t_sel: the fraction of tournaments decided by fitness rather than
                         by tie-breaking, which is the mechanism S3 guards. The test in
                         tests/scientific/test_s3_selection_gradient.py computes this identically

**The Spearman half of M7 is not here.** It needs per-genome descriptors, which the sweep's sampler produces
but the generation step does not expose, and it belongs with the sweep's analysis rather than with the
gradient question. Stated so the omission is a decision and not an oversight.

Usage: uv run python scripts/measure/fitness_distribution.py <out.npz> [--gens N] [--seeds N]
"""

from __future__ import annotations

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

N_GENS = 40
N_SEEDS = 3
COLUMNS = ("mean", "max", "var", "distinct", "zero_frac", "decided")


def decided_fraction(fit: np.ndarray, t_sel: int) -> float:
    _, counts = np.unique(fit, return_counts=True)
    return 1.0 - float(np.sum((counts / fit.size) ** t_sel))


def one_seed(p, seed: int, n_gens: int) -> tuple[np.ndarray, np.ndarray]:
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
    static = E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )
    template = net.init_params(jax.random.key(seed), p)
    pop = loop.initial_population(jax.random.key(seed + 1), template, p)
    step = loop.make_generation_step(genome.unflattener(template), static, p)
    key = jax.random.key(1000 + seed)
    rows, fits = [], []
    for _ in range(n_gens):
        key, k = jax.random.split(key)
        pop, fit = step(k, pop)
        f = np.asarray(fit, np.float64)
        fits.append(f)
        rows.append(
            [
                f.mean(),
                f.max(),
                f.var(),
                float(np.unique(f).size),
                float(np.mean(f == 0.0)),
                decided_fraction(f, p.t_sel),
            ]
        )
    jax.clear_caches()
    gc.collect()
    return np.array(rows), np.stack(fits)


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    print(src.memory_headroom(), flush=True)
    out = sys.argv[1]
    n_gens = int(sys.argv[sys.argv.index("--gens") + 1]) if "--gens" in sys.argv else N_GENS
    n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else N_SEEDS
    p = DEFAULT
    t0 = time.time()

    per_seed, fit_all = [], []
    for s in range(n_seeds):
        t1 = time.time()
        rows, fits = one_seed(p, s, n_gens)
        per_seed.append(rows)
        fit_all.append(fits)
        print(f"  seed {s}: {n_gens} generations in {time.time() - t1:5.1f}s", flush=True)
    per_seed = np.stack(per_seed)  # (seed, gen, col)
    fit_all = np.stack(fit_all)  # (seed, gen, p_pop)

    m = per_seed.mean(axis=0)
    print(f"\n  t_sel {p.t_sel}, p_pop {p.p_pop}, {n_seeds} seeds, mean over seeds\n", flush=True)
    print("  " + f"{'gen':>4s}" + "".join(f"{c:>11s}" for c in COLUMNS), flush=True)
    for g in range(n_gens):
        if g < 10 or g % 5 == 0 or g == n_gens - 1:
            print("  " + f"{g:4d}" + "".join(f"{v:11.4f}" for v in m[g]), flush=True)

    d = m[:, COLUMNS.index("decided")]
    print("\n" + "=" * 90, flush=True)
    print(
        f"  decided fraction: generation 0 {d[0]:.3f}, generation {n_gens - 1} {d[-1]:.3f}, "
        f"max {d.max():.3f} at generation {int(d.argmax())}",
        flush=True,
    )
    print(
        "  This is the number Q-050 asks the owner to set a floor for. The proposed floor in the gate is 0.5.",
        flush=True,
    )

    np.savez(
        out,
        per_seed=per_seed,
        fit=fit_all,
        columns=np.array(COLUMNS),
        n_gens=np.int32(n_gens),
        n_seeds=np.int32(n_seeds),
        t_sel=np.int32(p.t_sel),
        config_hash=np.array(loop.config_hash(p)),
        wall_seconds=np.float32(time.time() - t0),
    )
    print(f"\n  saved {out}, wall {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
