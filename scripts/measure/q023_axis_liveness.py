#!/usr/bin/env python3
"""Q-023 axis liveness. Two parts, per the restatement in `docs/q023-prediction.md`.

    part one   does each axis move OUTCOMES? Measured on tier one's outcome vector, which is counted
               over prey rather than over ticks, so an axis can be live here and dead in descriptors
    part two   does any setting give tier three a denominator? Measured on near_ticks, visible_ticks
               and engaged_ticks per prey, NOT on descriptors, which came back degenerate

**The conditioning variable is `near`, not `visible`.** `visible_ticks` counts the PREDATOR detecting the
prey, and the detection rule contains `subject_moved`, so it is a sample selected for movement: over the 69
percent of the sensing area outside `r_near`, a moving prey is detected every tick and a still one on a 0.25
draw. Engagement requires targeting, targeting requires reporting, so `engaged_ticks` inherits that selection
and `stillness_engaged` is stillness measured on a movement-selected sample. **Freezing excludes itself from
the descriptor built to detect it.**

`near` is geometric: the predator is within `r_pred`, regardless of detection, motion or any draw. It is the
only threat measure in the frozen accumulator set not gated by the behaviour being measured. All three are
reported because the comparison between them is now informative in its own right.

**The instrument is the repaired hand-coded forager, not an evolved population.** The question is whether the
AXIS moves the world, which a fixed policy answers with less compute and no confound from evolution's
response. Being predator-oblivious it cannot avoid, which is the right property: it measures the threat the
world delivers rather than the threat a policy fails to escape. An evolved lineage would move these numbers.

Usage: uv run python scripts/measure/q023_axis_liveness.py <out.npz>
"""

from __future__ import annotations

import sys
import time

import jax
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import DEFAULT
from src.agent import baseline
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius
from src.measure import outcomes as O

N_SEEDS = 16

#: The four axes the owner named, plus the combination. Baseline values are included in each axis so every
#: curve has its own control rather than being compared against a separately-run baseline.
SETTINGS: list[tuple[str, dict]] = [
    ("baseline", {}),
    ("n_pred=2", {"n_pred": 2}),
    ("n_pred=4", {"n_pred": 4}),
    ("r_pred=5", {"r_pred": 5}),
    ("r_pred=6", {"r_pred": 6}),
    ("p_detect=0.50", {"p_distant_detect": 0.5}),
    ("p_detect=1.00", {"p_distant_detect": 1.0}),
    ("cue=0.5", {"cue_dependence": 0.5}),
    ("cue=0.0", {"cue_dependence": 0.0}),
    ("n_pred=4 r_pred=6", {"n_pred": 4, "r_pred": 6}),
    ("n_pred=4 r_pred=6 p_detect=1.0", {"n_pred": 4, "r_pred": 6, "p_distant_detect": 1.0}),
]

FIELDS = ("near_ticks", "visible_ticks", "engaged_ticks", "alive_ticks", "occluded_near", "moved_ticks")


def make_policy(params):
    def policy(weights, obs, hidden, step_index):
        del weights
        return jax.vmap(lambda o: baseline.forager(o, params, step_index))(obs), hidden

    return policy


def run_setting(name: str, overrides: dict):
    p = DEFAULT.__class__(t_horizon=400, **overrides)
    policy = make_policy(p)

    # **`required_radius`, not `p.r`.** DEV-030: one table serves the prey at `r` and the predator at
    # `r_pred`, so it must cover the larger, and querying beyond the build radius now fails loudly.
    r_table = required_radius(p)

    acc_rows, out_rows = [], []
    for seed in range(N_SEEDS):
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, r_table), food_src=fs, target_src=ts)
        w = net.init_params(jax.random.key(seed), p)
        state, acc = E.run_episode(
            jax.random.key(1000 + seed), w, static, ent.PRNGSource(jax.random.key(seed)), p, policy=policy
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
    o = O.outcomes(
        np.concatenate([r[0] for r in out_rows]),
        np.concatenate([r[1] for r in out_rows]),
        np.concatenate([r[2] for r in out_rows]),
    )
    return per_prey, o


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    out_path = sys.argv[1]
    t0 = time.time()

    results = {}
    print(
        f"{'setting':32s} {'near':>8} {'visible':>8} {'engaged':>8} {'occl_near':>10} "
        f"{'predated':>9} {'starved':>8} {'alive':>7}",
        flush=True,
    )
    for name, overrides in SETTINGS:
        per_prey, o = run_setting(name, overrides)
        results[name] = (per_prey, o)
        print(
            f"{name:32s} {per_prey['near_ticks'].mean():8.3f} {per_prey['visible_ticks'].mean():8.3f} "
            f"{per_prey['engaged_ticks'].mean():8.3f} {per_prey['occluded_near'].mean():10.3f} "
            f"{o.predated:9.3f} {o.starved:8.3f} {o.survived:7.3f}",
            flush=True,
        )

    payload = {}
    for name, (per_prey, o) in results.items():
        key = name.replace(" ", "_").replace("=", "").replace(".", "")
        for f in FIELDS:
            payload[f"{key}__{f}"] = per_prey[f]
        payload[f"{key}__outcomes"] = np.array([o.predated, o.starved, o.survived, o.both, o.n])
    payload["settings"] = np.array([n for n, _ in SETTINGS])
    payload["n_seeds"] = np.int32(N_SEEDS)
    payload["wall_seconds"] = np.float32(time.time() - t0)
    np.savez(out_path, **payload)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
