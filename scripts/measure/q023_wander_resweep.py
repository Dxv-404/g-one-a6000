#!/usr/bin/env python3
"""Q-023 part two, re-run across `p_wander`. Build step 11.

**P6 turns on this and the first sweep could not test it.** Every setting in the first sweep varied detection
or predator count; none varied whether the predator moves when it has no target. That is the one axis acting
directly on the mechanism that produces `near_ticks` 0.000 for a still prey: a predator that never moves
cannot approach one.

**The quantity is `near_ticks`.** It is geometric, ungated by detection, motion or any draw, so it is the
only threat measure in the frozen accumulator set not gated by the behaviour being measured. `visible_ticks`
and `engaged_ticks` are reported alongside because the comparison between them is informative, but the
tier-three question turns on `near`.

**Predictions on record before this runs.** The owner's: `p_wander` raises near ticks substantially and does
not reach fifty. Mine, from `docs/q023-prediction.md`: no setting reaches a usable third-tier denominator.
Both are falsifiable by this script and neither was written after seeing it.

**Two instruments, and the second is the point.** The repaired forager measures the world a moving policy
meets. The always-STAY policy measures whether stillness is still a perfect defence once the predator
searches, which is the specific thing `p_wander` was added to fix, and it cannot be seen with a policy that
moves on 88 percent of its ticks.

Usage: uv run python scripts/measure/q023_wander_resweep.py <out.npz>
"""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import DEFAULT
from src.agent import baseline
from src.agent import network as net
from src.env import dynamics, grid, resources
from src.env import entropy as ent
from src.env import env as E
from src.env.perception import build_vis_table, required_radius
from src.measure import outcomes as O

N_SEEDS = 16
WANDER = (0.0, 0.25, 0.5, 0.75, 1.0)

#: `p_wander` crossed with the axis that moved most in the first sweep, so the combination is measured rather
#: than assumed additive. `n_pred` 4 with `r_pred` 6 was the strongest setting there at 10.4 near ticks.
CROSSES = (
    ("", {}),
    (" n_pred=4", {"n_pred": 4}),
    (" n_pred=4 r_pred=6", {"n_pred": 4, "r_pred": 6}),
)

FIELDS = ("near_ticks", "visible_ticks", "engaged_ticks", "alive_ticks", "occluded_near", "moved_ticks")


def forager_policy(params):
    def policy(weights, obs, hidden, step_index):
        del weights
        return jax.vmap(lambda o: baseline.forager(o, params, step_index))(obs), hidden

    return policy


def stay_policy(params):
    del params

    def policy(weights, obs, hidden, step_index):
        del weights, step_index
        return jnp.full((obs.shape[0],), dynamics.ACTION_STAY, jnp.int32), hidden

    return policy


def run_setting(overrides: dict, policy_factory):
    p = DEFAULT.__class__(t_horizon=400, **overrides)
    pol = policy_factory(p)
    r_table = required_radius(p)

    acc_rows, out_rows = [], []
    for seed in range(N_SEEDS):
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, r_table), food_src=fs, target_src=ts)
        w = net.init_params(jax.random.key(seed), p)
        state, acc = E.run_episode(
            jax.random.key(1000 + seed), w, static, ent.PRNGSource(jax.random.key(seed)), p, policy=pol
        )
        acc_rows.append({f: np.asarray(getattr(acc, f), dtype=np.float64) for f in FIELDS})
        out_rows.append(
            (
                np.asarray(state.prey_died_predated),
                np.asarray(state.prey_died_starving),
                np.asarray(state.prey_alive),
            )
        )
    per_prey = {f: np.concatenate([r[f] for r in acc_rows]) for f in FIELDS}
    o = O.outcomes(*[np.concatenate([r[i] for r in out_rows]) for i in range(3)])
    return per_prey, o


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()
    payload = {}

    for label, policy_factory in (("forager", forager_policy), ("always-STAY", stay_policy)):
        print(f"\n=== {label} ===", flush=True)
        print(
            f"{'setting':30s} {'alive':>7} {'near':>8} {'visible':>8} {'engaged':>8} "
            f"{'predated':>9} {'starved':>8}",
            flush=True,
        )
        for cross_label, cross in CROSSES:
            for pw in WANDER:
                name = f"p_wander={pw}{cross_label}"
                per_prey, o = run_setting({**cross, "p_wander": pw}, policy_factory)
                print(
                    f"{name:30s} {per_prey['alive_ticks'].mean():7.1f} "
                    f"{per_prey['near_ticks'].mean():8.3f} {per_prey['visible_ticks'].mean():8.3f} "
                    f"{per_prey['engaged_ticks'].mean():8.3f} {o.predated:9.3f} {o.starved:8.3f}",
                    flush=True,
                )
                key = f"{label}__{name}".replace(" ", "_").replace("=", "").replace(".", "")
                for f in FIELDS:
                    payload[f"{key}__{f}"] = per_prey[f]
                payload[f"{key}__outcomes"] = np.array([o.predated, o.starved, o.survived, o.both, o.n])

    payload["wander"] = np.array(WANDER)
    payload["n_seeds"] = np.int32(N_SEEDS)
    payload["wall_seconds"] = np.float32(time.time() - t0)
    np.savez(sys.argv[1], **payload)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
