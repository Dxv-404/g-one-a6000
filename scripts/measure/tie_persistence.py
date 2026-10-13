"""Follow-up to queue item A: how long does the tie regime last, and does it ratchet?

Generation zero is not the question by itself. The elites are placed at positions 0..n_elite-1 by
`next_population`, and ties are broken by ascending index, so a tied elite occupying a low slot wins
again next generation. **That is a ratchet only if the tie regime PERSISTS.** If fitness diversifies
after a few generations, the mechanism fires a handful of times on genomes that were iid anyway.

Branches, stated before the numbers:

    ties vanish within a few generations
        the defect is real, bounded, and its consequence is that early elite identity is an
        artifact of array order rather than of merit. No lasting founder effect
    ties persist for tens of generations
        the ratchet runs: low array slots hold elite status across many generations, and the
        lineage that dominates a run is partly a function of initialisation ORDER
"""

import gc
import sys

import jax
import numpy as np

import src  # noqa: F401
from configs.run_configs import SCRIPTED
from src.agent import network as net
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius
from src.evo import genome, loop

N_SEEDS = 3
N_GENS = 25


def main():
    print(src.assert_runtime_took_effect(), flush=True)
    p = SCRIPTED.params
    ne = p.n_elite

    rows = []
    for seed in range(N_SEEDS):
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(
            obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
        )
        template = net.init_params(jax.random.key(seed), p)
        pop = loop.initial_population(jax.random.key(seed + 1), template, p)
        step = loop.make_generation_step(genome.unflattener(template), static, p)
        key = jax.random.key(seed + 2)

        per_gen = []
        for g in range(N_GENS):
            key, kg = jax.random.split(key)
            pop, fit = step(kg, pop)
            f = np.asarray(fit)
            cutoff = np.sort(f)[::-1][ne - 1]
            n_above = int((f > cutoff).sum())
            n_at = int((f == cutoff).sum())
            slots_from_tie = ne - n_above
            contested = slots_from_tie > 0 and n_at > slots_from_tie
            per_gen.append((g, int(len(np.unique(f))), n_above, n_at, slots_from_tie, contested))
        rows.append(per_gen)
        print(f"  seed {seed} done", flush=True)
        jax.clear_caches()
        gc.collect()

    print("\n" + "=" * 96, flush=True)
    print("TIE REGIME OVER THE FIRST GENERATIONS. contested = elite slots filled by array order", flush=True)
    print("=" * 96, flush=True)
    print(
        f"{'gen':>4s} {'distinct':>9s} {'above':>6s} {'tied':>6s} {'from tie':>9s} {'contested':>10s}",
        flush=True,
    )
    for g in range(N_GENS):
        d = np.mean([r[g][1] for r in rows])
        ab = np.mean([r[g][2] for r in rows])
        at = np.mean([r[g][3] for r in rows])
        sl = np.mean([r[g][4] for r in rows])
        c = sum(1 for r in rows if r[g][5])
        print(f"{g:4d} {d:9.1f} {ab:6.1f} {at:6.1f} {sl:9.1f} {c:6d}/{len(rows)}", flush=True)

    first_clean = None
    for g in range(N_GENS):
        if all(not r[g][5] for r in rows):
            first_clean = g
            break
    print("\n" + "=" * 96, flush=True)
    if first_clean is None:
        print(
            f"  CONTESTED IN EVERY GENERATION through {N_GENS - 1}: the ratchet regime persists.", flush=True
        )
    else:
        print(f"  First generation with NO contested elite slot in any seed: {first_clean}.", flush=True)
        still = sum(1 for r in rows for g in range(first_clean, N_GENS) if r[g][5])
        print(
            f"  Contested generations after that point: {still} of {len(rows) * (N_GENS - first_clean)}.",
            flush=True,
        )

    # A report is not a result until the artifact exists.
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/tie_persistence.npz"
    arr = np.array([[list(x) for x in r] for r in rows], dtype=np.float64)
    np.savez(
        out,
        per_gen=arr,
        columns=np.array(["gen", "distinct", "n_above", "n_at_cutoff", "slots_from_tie", "contested"]),
        n_seeds=np.int32(N_SEEDS),
        n_gens=np.int32(N_GENS),
    )
    print(f"\nsaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
