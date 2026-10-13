#!/usr/bin/env python3
"""The cell neither sweep reached: `p_wander` crossed with `cue_dependence`. Build step 11.

**Neither parameter alone can price freezing, and the two previous sweeps each missed for the opposite
reason.**

```
p_wander 0, cue anything     the predator never approaches a still prey, so cue has nothing to act on
p_wander > 0, cue 1.0        the predator finds it and damage is DAMAGE * moved, so a still prey
                             takes exactly zero. Found, engaged, and immune
p_wander > 0, cue < 1.0      the case that has never been run
```

Measured on the previous sweep: always-STAY at `p_wander` 1.0 with `n_pred` 4 reaches 41.0 near ticks, 35.7
visible and 6.07 engaged, and is predated on 0.000 of prey. It is found; it is not killed.

**A prediction, recorded here before the run because the file is committed before it executes.** At
`cue_dependence` below 1.0 with `p_wander` above 0, a still prey should finally be predated, and its
`near_ticks` should FALL rather than rise, because the 41.0 figure is partly an artefact of never dying: an
immortal prey accumulates threat ticks for the whole episode. If predation stays at 0.000 here, something
other than the damage rule is protecting still prey and neither parameter is the explanation.

Usage: uv run python scripts/measure/q023_wander_cue_cross.py <out.npz>
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
WANDER = (0.0, 0.5, 1.0)
CUE = (1.0, 0.5, 0.0)
FIELDS = ("near_ticks", "visible_ticks", "engaged_ticks", "alive_ticks", "still_engaged", "moved_ticks")


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


def run(overrides, policy_factory):
    p = DEFAULT.__class__(t_horizon=400, n_pred=4, **overrides)
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
    return per_prey, O.outcomes(*[np.concatenate([r[i] for r in out_rows]) for i in range(3)])


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()
    payload = {}
    for label, factory in (("always-STAY", stay_policy), ("forager", forager_policy)):
        print(f"\n=== {label}, n_pred=4 ===", flush=True)
        print(
            f"{'p_wander':>9} {'cue':>5} {'alive':>7} {'near':>8} {'visible':>8} {'engaged':>8} "
            f"{'still_eng':>10} {'predated':>9} {'starved':>8}",
            flush=True,
        )
        for pw in WANDER:
            for cue in CUE:
                per, o = run({"p_wander": pw, "cue_dependence": cue}, factory)
                print(
                    f"{pw:9.2f} {cue:5.2f} {per['alive_ticks'].mean():7.1f} "
                    f"{per['near_ticks'].mean():8.3f} {per['visible_ticks'].mean():8.3f} "
                    f"{per['engaged_ticks'].mean():8.3f} {per['still_engaged'].mean():10.3f} "
                    f"{o.predated:9.3f} {o.starved:8.3f}",
                    flush=True,
                )
                key = f"{label}__pw{pw}__cue{cue}".replace(".", "")
                for f in FIELDS:
                    payload[f"{key}__{f}"] = per[f]
                payload[f"{key}__outcomes"] = np.array([o.predated, o.starved, o.survived, o.both, o.n])

    payload["wander"] = np.array(WANDER)
    payload["cue"] = np.array(CUE)
    payload["wall_seconds"] = np.float32(time.time() - t0)
    np.savez(sys.argv[1], **payload)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
