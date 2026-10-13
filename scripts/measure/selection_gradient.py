#!/usr/bin/env python3
"""Does dying cost fitness? The precondition of the entire hypothesis, measured for the first time.

**The problem statement is that survival instinct emerges INSTRUMENTALLY: staying alive pays because the
dead cannot forage.** That requires a measurable fitness gradient along survival, and nothing in eighteen
sessions has checked it exists. If a predated prey scores nearly as much as a survivor, there is nothing for
selection to climb and every null result downstream is guaranteed by construction rather than found.

Both branches, stated before the number:

    a large survivor-minus-predated score gap    the instrumental route exists and the experiment can
                                                 in principle produce its phenomenon
    a gap near zero or inside noise              the design cannot produce the phenomenon AT THIS
                                                 CONFIGURATION whatever evolution does, and the sweep
                                                 must not run until the configuration changes

**The instrument is the fixed forager**, deliberately: it holds behavior constant, so the gradient measured
is the WORLD's, not a policy's. An evolved lineage would change these numbers, and that change is the
experiment; this is its precondition.

**The layout confound, and how it is handled rather than dismissed.** A prey spawned in a bad corner may
both die early AND score low, inflating the raw gap without any causal path from survival to score. The
control is within-episode pairing: survivors and predated prey in the SAME episode share a layout, so their
paired difference removes it. Both numbers are reported; the paired one is the claim.

Usage: uv run python scripts/measure/selection_gradient.py <out.npz> [--seeds N]
"""

from __future__ import annotations

import dataclasses
import gc
import sys
import time

import jax
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import DEFAULT, check_invariants
from src.agent import baseline
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius

N_SEEDS = 96
T_HORIZON = 400


def policy_forager(params):
    def policy(weights, obs, hidden, step_index):
        del weights
        return jax.vmap(lambda o: baseline.forager(o, params, step_index))(obs), hidden

    return policy


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()
    global N_SEEDS
    if "--seeds" in sys.argv:
        N_SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1])

    p = dataclasses.replace(DEFAULT, t_horizon=T_HORIZON)
    check_invariants(p)
    policy = policy_forager(p)
    r_table = required_radius(p)

    score, alive_t, predated, starved, survived, episode = [], [], [], [], [], []
    for seed in range(N_SEEDS):
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, r_table), food_src=fs, target_src=ts)
        w = net.init_params(jax.random.key(seed), p)
        state, acc = E.run_episode(
            jax.random.key(1000 + seed), w, static, ent.PRNGSource(jax.random.key(seed)), p, policy=policy
        )
        score.append(np.asarray(state.prey_score, dtype=np.float64))
        alive_t.append(np.asarray(acc.alive_ticks, dtype=np.float64))
        predated.append(np.asarray(state.prey_died_predated))
        starved.append(np.asarray(state.prey_died_starving))
        survived.append(np.asarray(state.prey_alive))
        episode.append(np.full(p.k, seed))
        if (seed + 1) % 32 == 0:
            jax.clear_caches()
            gc.collect()

    s = np.concatenate(score)
    at = np.concatenate(alive_t)
    dp = np.concatenate(predated).astype(bool)
    ds = np.concatenate(starved).astype(bool)
    al = np.concatenate(survived).astype(bool)
    ep = np.concatenate(episode)

    print("\n" + "=" * 92, flush=True)
    print("FITNESS BY OUTCOME. Fitness is prey_score and nothing else, per invariant I1.", flush=True)
    print("=" * 92, flush=True)
    print(f"{'outcome':<12s} {'n':>6s} {'mean score':>11s} {'sd':>8s} {'mean alive ticks':>17s}", flush=True)
    for name, m in (("survived", al), ("predated", dp), ("starved", ds)):
        if m.sum():
            print(
                f"{name:<12s} {int(m.sum()):6d} {s[m].mean():11.3f} {s[m].std(ddof=1):8.3f} {at[m].mean():17.1f}",
                flush=True,
            )

    print("\n" + "=" * 92, flush=True)
    print("THE GRADIENT, raw and layout-controlled", flush=True)
    print("=" * 92, flush=True)

    gap_raw = s[al].mean() - s[dp].mean() if al.sum() and dp.sum() else float("nan")
    print(f"  raw survivor-minus-predated gap          {gap_raw:8.3f}", flush=True)

    # Within-episode pairing: episodes containing BOTH a survivor and a predated prey share a layout.
    paired = []
    for e in np.unique(ep):
        m = ep == e
        if (al & m).sum() and (dp & m).sum():
            paired.append(s[al & m].mean() - s[dp & m].mean())
    paired = np.array(paired)
    if len(paired):
        sem = paired.std(ddof=1) / np.sqrt(len(paired))
        print(
            f"  WITHIN-EPISODE paired gap                {paired.mean():8.3f} +/- {1.96 * sem:.3f} "
            f"across {len(paired)} episodes with both outcomes",
            flush=True,
        )
        print(
            f"  episodes where the gap is positive       {int((paired > 0).sum()):4d} of {len(paired)}",
            flush=True,
        )

    # The slope: how much score does a tick of life buy, among the predated, where death time varies.
    if dp.sum() > 10 and at[dp].std() > 0:
        slope = np.polyfit(at[dp], s[dp], 1)[0]
        r = float(np.corrcoef(at[dp], s[dp])[0, 1])
        print(f"\n  among the predated: score per 100 ticks  {slope * 100:8.3f}   corr {r:+.3f}", flush=True)
        print(f"  so dying 100 ticks earlier costs about {slope * 100:.2f} score at fixed policy", flush=True)

    print("\n" + "=" * 92, flush=True)
    print(
        "IS THE GRADIENT SELECTABLE? The gap against the fitness variation selection actually sees.",
        flush=True,
    )
    print("=" * 92, flush=True)
    per_seed_mean = np.array([s[ep == e].mean() for e in np.unique(ep)])
    seed_sd = per_seed_mean.std(ddof=1)
    print(f"  seed-to-seed SD of mean score            {seed_sd:8.3f}", flush=True)
    if len(paired):
        ratio = paired.mean() / seed_sd if seed_sd > 0 else float("inf")
        print(f"  paired gap / seed SD                     {ratio:8.2f}x", flush=True)
        print("\n  VERDICT, both branches were stated in the docstring:", flush=True)
        if paired.mean() > 0 and paired.mean() - 1.96 * sem > 0:
            print("    the paired gap is positive and its interval excludes zero.", flush=True)
            print("    THE INSTRUMENTAL ROUTE EXISTS AT THIS CONFIGURATION: dying costs fitness,", flush=True)
            print(
                "    so selection has a gradient toward survival without survival being rewarded.", flush=True
            )
        else:
            print("    the paired gap does not exclude zero. THE PRECONDITION FAILS at this", flush=True)
            print("    configuration and the sweep must not run until the configuration changes.", flush=True)

    np.savez(
        sys.argv[1],
        score=s,
        alive_ticks=at,
        predated=dp,
        starved=ds,
        survived=al,
        episode=ep,
        paired_gaps=paired,
        n_seeds=np.int32(N_SEEDS),
        wall_seconds=np.float32(time.time() - t0),
    )
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
