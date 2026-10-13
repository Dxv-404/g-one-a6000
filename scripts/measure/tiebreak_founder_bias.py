#!/usr/bin/env python3
"""Queue item A: is generation-zero tie-breaking a live founder bias in every evolutionary run?

**The suspicion, stated as a mechanism before it is measured.** Selection has two tie-breaking paths and
they are not the same shape:

    elitism     `jax.lax.top_k(fit, n_elite)[1]` returns the LOWEST indices among equal values, so if
                the top of the fitness distribution is tied, the elites are genomes 0..n_elite-1 by
                array position rather than by merit
    tournament  `jnp.argmax(fit[candidates], axis=1)` picks slot 0 among tied candidates, and slot 0 is
                a UNIFORMLY RANDOM genome, so ties here select uniformly and carry no index bias

So the tournament is exonerated by construction and only elitism can carry the defect. That asymmetry is
worth stating first because it decides what the measurement has to show: a founder effect would appear as
elites concentrated at low indices, NOT as a skewed parent-index distribution.

**And elites land at positions 0..n_elite-1 of the new population**, since `next_population` concatenates
elites before offspring. So a tie at the top is self-perpetuating: the same array slots win again next
generation if fitness ties again.

**The branches, stated before the numbers, per this project's standing rule.**

    top-of-distribution ties are common at generation 0
        the defect is LIVE. The first generations run on 8 founders chosen by array position, and
        every concluded run inherits whatever those 8 happened to be
    the top is essentially never tied
        the defect is REAL IN CODE and INERT IN PRACTICE. It is disclosed, and a deterministic
        tie-break by index remains defensible because it never fires

**Blast radius, which is why this runs before the diagnoses.** If live, it touches every evolutionary run
since step 10, including the sigma and capacity nulls. The owner's recorded expectation, before looking:
both nulls survive even if the bias is real, because both were comparisons made under the same bias. This
script does not test that expectation; it establishes whether the bias exists at all.

Usage: uv run python scripts/measure/tiebreak_founder_bias.py <out.npz> [--seeds N]
"""

from __future__ import annotations

import gc
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import check_invariants
from configs.run_configs import SCRIPTED
from src.agent import network as net
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius
from src.evo import genome, loop

N_SEEDS = 4


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()
    global N_SEEDS
    if "--seeds" in sys.argv:
        N_SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1])

    p = SCRIPTED.params
    check_invariants(p)
    print(
        f"config: p_pop={p.p_pop} n_elite={p.n_elite} t_sel={p.t_sel} k={p.k} "
        f"n_episodes={p.n_episodes} t_horizon={p.t_horizon}",
        flush=True,
    )

    rows = []
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

        # Generation 0's fitness, which is what selection at generation 0 actually sees.
        _, fit0 = step(jax.random.key(seed + 2), pop)
        f = np.asarray(fit0)

        # --- the two quantities that decide it -------------------------------------------------
        elite_idx = np.asarray(jax.lax.top_k(jnp.asarray(f), p.n_elite)[1])
        # How many genomes tie with the n_elite-th best? That is the size of the pool the
        # tie-break is choosing from, and it is the whole question.
        cutoff = np.sort(f)[::-1][p.n_elite - 1]
        n_at_cutoff = int((f == cutoff).sum())
        n_above = int((f > cutoff).sum())
        n_zero = int((f == 0.0).sum())

        # Parent indices from the tournament, measured rather than assumed uniform.
        parents = np.asarray(
            loop.tournament(jax.random.key(seed + 3), jnp.asarray(f), p.t_sel, p.p_pop - p.n_elite)
        )

        rows.append(
            dict(
                seed=seed,
                fit=f,
                elite_idx=elite_idx,
                parents=parents,
                n_at_cutoff=n_at_cutoff,
                n_above=n_above,
                n_zero=n_zero,
                distinct=int(len(np.unique(f))),
            )
        )
        print(
            f"  seed {seed}: distinct fitness values {rows[-1]['distinct']:4d}/{p.p_pop}, "
            f"zeros {n_zero:4d}, tied at the elite cutoff {n_at_cutoff:4d}, "
            f"strictly above cutoff {n_above:3d}",
            flush=True,
        )
        jax.clear_caches()
        gc.collect()

    print("\n" + "=" * 98, flush=True)
    print("THE ELITE TIE-BREAK: is the top of the distribution tied at generation 0?", flush=True)
    print("=" * 98, flush=True)
    total_contested = 0
    for r in rows:
        # An elite slot is CONTESTED if it was filled from a tie rather than won outright.
        contested = max(0, r["n_at_cutoff"] - (SCRIPTED.params.n_elite - r["n_above"]))
        total_contested += 1 if r["n_at_cutoff"] > (SCRIPTED.params.n_elite - r["n_above"]) else 0
        print(
            f"  seed {r['seed']}: elites {list(r['elite_idx'])}   "
            f"tie pool at cutoff {r['n_at_cutoff']}, slots to fill from it "
            f"{SCRIPTED.params.n_elite - r['n_above']}, surplus contenders {contested}",
            flush=True,
        )

    print("\n" + "=" * 98, flush=True)
    print("THE TOURNAMENT: parent-index distribution against uniform", flush=True)
    print("=" * 98, flush=True)
    p_pop = SCRIPTED.params.p_pop
    allp = np.concatenate([r["parents"] for r in rows])
    lo = int((allp < p_pop // 2).sum())
    hi = int((allp >= p_pop // 2).sum())
    print(
        f"  parents drawn: {len(allp)}   lower half {lo}   upper half {hi}   "
        f"expected {len(allp) / 2:.0f} each",
        flush=True,
    )
    print(f"  mean parent index {allp.mean():.1f}, uniform expectation {(p_pop - 1) / 2:.1f}", flush=True)
    # A chi-square-free reading: the first-index share, which is what a low-index bias would inflate.
    print(
        f"  share of parents in the first n_elite slots: {(allp < SCRIPTED.params.n_elite).mean():.4f}, "
        f"uniform expectation {SCRIPTED.params.n_elite / p_pop:.4f}",
        flush=True,
    )

    print("\n" + "=" * 98, flush=True)
    print("VERDICT, against the branches stated in the docstring", flush=True)
    print("=" * 98, flush=True)
    if total_contested == 0:
        print("  The elite cutoff was NOT tied in any seed: every elite won its slot outright.", flush=True)
        print("  The defect is REAL IN CODE and INERT AT GENERATION ZERO at this configuration.", flush=True)
    else:
        print(f"  The elite cutoff WAS tied in {total_contested} of {len(rows)} seeds.", flush=True)
        print("  The defect is LIVE: those elite slots were filled by array position.", flush=True)

    np.savez(
        sys.argv[1],
        fit=np.stack([r["fit"] for r in rows]),
        elite_idx=np.stack([r["elite_idx"] for r in rows]),
        parents=np.stack([r["parents"] for r in rows]),
        n_at_cutoff=np.array([r["n_at_cutoff"] for r in rows]),
        n_above=np.array([r["n_above"] for r in rows]),
        n_zero=np.array([r["n_zero"] for r in rows]),
        distinct=np.array([r["distinct"] for r in rows]),
        n_seeds=np.int32(N_SEEDS),
        wall_seconds=np.float32(time.time() - t0),
    )
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
