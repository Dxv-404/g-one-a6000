#!/usr/bin/env python3
"""Battery B and C: can the measurement stack detect the five strategies when they are KNOWN to be present?

**The calibration every measurement science performs and this project never has.** Thirty-three evolutionary
runs produced flee_engaged = 0.0000 and nothing distinguishes "the behavior does not emerge" from "the
pipeline cannot see it". Planting each strategy by hand and running it through the FULL stack separates
those, and converts any eventual null from uninterpretable into publishable.

**What is reported is detection POWER per strategy, not detection yes/no.** The taxonomy audit found the
instrument is not equally sensitive to its own five hypotheses: freezing has four descriptors and distancing
one. If the sweep later reports strategy X and not strategy Y, that comparison inherits whatever asymmetry
this battery measures, so the asymmetry has to be on the record first.

**The firewall, same as the S2 forager.** These policies are measurement instruments: functions from
observation to action, with no path to weights, no shared representation with the lineage, and nothing an
evolutionary loop could initialise from. They exist to calibrate the instrument and never touch any lineage.

**Battery C is the forager itself**: it has no anti-predator behavior by construction, so it must NOT
classify as any strategy. A classifier that assigns the negative control a label is finding structure in
noise, and every positive result would be suspect.

**Two configurations, because detection power depends on the condition.** At the default, engagement runs at
0.19 ticks per prey, so engaged-conditioned descriptors have almost no denominator whatever the behavior.
The battery runs the default AND a high-pressure condition the sweep also visits, so the power table shows
where in the sweep's own space the instrument can and cannot see.

**Thresholds here are CALIBRATION-ONLY.** `classify` requires a floor and separation that the design derives
from null data at step 13; those do not exist yet and must not be invented here. The battery reports the raw
z-matrix so any later threshold can be applied to it, and uses a labelled calibration pair only to render
the confusion table readable.

**The hider is approximate and labelled.** True hiding needs to know what the PREDATOR can see, which is not
in the egocentric observation. The planted hider moves away preferring destinations adjacent to obstacles,
which is the best local proxy. If its detection is weak, that is itself a finding: hiding may not be
plantable, and possibly not evolvable, from local information alone.

Usage: uv run python scripts/measure/planted_policies.py <out.npz> [--seeds N]
"""

from __future__ import annotations

import dataclasses
import gc
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import DEFAULT, check_invariants
from src.agent import baseline
from src.agent import network as net
from src.env import dynamics, grid, resources
from src.env import entropy as ent
from src.env import env as E
from src.env.observation import CH_OBSTACLE, CH_PREDATOR, CH_PREY
from src.env.perception import build_vis_table, required_radius
from src.measure import classifier as C
from src.measure import descriptors as D
from src.measure.classifier import NAMED_STRATEGIES  # noqa: F401  (re-exported for readers)

N_SEEDS = 32
T_HORIZON = 400

#: Calibration-only rendering thresholds for the confusion table. NOT the pre-registered step 13 values,
#: which are derived from null data and do not exist yet. The raw z-matrix is saved so any threshold can be
#: applied later without re-running anything.
CAL_FLOOR = 2.0
CAL_SEPARATION = 1.0

#: The engaged-descriptor denominator floor, calibration-only for the same reason. 1.0 admits any prey with
#: at least one engaged tick, which maximises defined values; the mean denominators are printed beside every
#: result so the reader sees how much data each number stands on.
CAL_DENOM_FLOOR = 1.0

CONFIGS = {
    "default": {},
    "high_pressure": {"p_wander": 1.0, "n_pred": 4},
}

ACC_FIELDS = E.Accumulators._fields


def _patch(obs, params):
    s, c = params.s_patch, params.c_channels
    return obs[: s * s * c].reshape(s, s, c)


def _nearest(mask, s):
    """(found, dy, dx) of the nearest true cell to the centre, Chebyshev, centre excluded."""
    centre = s // 2
    yy, xx = jnp.meshgrid(jnp.arange(s), jnp.arange(s), indexing="ij")
    dy, dx = yy - centre, xx - centre
    dist = jnp.maximum(jnp.abs(dy), jnp.abs(dx))
    scored = jnp.where(mask & (dist > 0), dist, jnp.int32(s * s))
    flat = jnp.argmin(scored)
    return jnp.any(mask & (dist > 0)), dy.reshape(-1)[flat], dx.reshape(-1)[flat]


def _obstacle_adjacency(obstacle):
    """3x3 dilation: is any neighbouring cell an obstacle. Static Python loop, fine under jit."""
    p = jnp.pad(obstacle.astype(jnp.float32), 1)
    s = obstacle.shape[0]
    out = jnp.zeros_like(obstacle, dtype=jnp.float32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            out = jnp.maximum(out, p[1 + di : 1 + di + s, 1 + dj : 1 + dj + s])
    return out


def _steer(patch, params, t_dy, t_dx, toward, prefer_cover):
    """The free adjacent cell that best increases (or decreases) Chebyshev distance to the target cell.

    Returns STAY when no adjacent cell is free, which cannot happen on this map's densities but must not
    fail silently if it ever does.
    """
    s = params.s_patch
    centre = s // 2
    obstacle = patch[:, :, CH_OBSTACLE] > 0
    dirs = baseline.DIRECTIONS
    dest_y = centre + dirs[:, 0]
    dest_x = centre + dirs[:, 1]
    free = ~obstacle[dest_y, dest_x]
    nd = jnp.maximum(jnp.abs(dest_y - (centre + t_dy)), jnp.abs(dest_x - (centre + t_dx))).astype(jnp.float32)
    score = jnp.where(toward, -nd, nd)
    if prefer_cover:
        score = score + 1.5 * _obstacle_adjacency(obstacle)[dest_y, dest_x]
    score = jnp.where(free, score, -1e9)
    best = jnp.argmax(score).astype(jnp.int32)
    return jnp.where(jnp.any(free), best, jnp.int32(dynamics.ACTION_STAY))


def _make(kind):
    """One planted policy. `kind` is static, so each compiles separately and cleanly."""

    def planted(obs, params, step_index):
        base = baseline.forager(obs, params, step_index)
        patch = _patch(obs, params)
        pred_mask = patch[:, :, CH_PREDATOR] > 0
        seen, p_dy, p_dx = _nearest(pred_mask, params.s_patch)

        if kind == "forager":
            return base
        if kind == "freezer":
            return jnp.where(seen, jnp.int32(dynamics.ACTION_STAY), base)
        if kind == "fleer":
            # Trigger at 3 rather than 2: at 2 the fleer responded on so few of its near ticks that its
            # own-descriptor signal was mostly the forager base rate (run 22 diagnostic). Kept below the
            # distancer's respond-whenever-seen so the two stay distinct instruments.
            near = seen & (jnp.maximum(jnp.abs(p_dy), jnp.abs(p_dx)) <= 3)
            return jnp.where(near, _steer(patch, params, p_dy, p_dx, False, False), base)
        if kind == "distancer":
            return jnp.where(seen, _steer(patch, params, p_dy, p_dx, False, False), base)
        if kind == "hider":
            return jnp.where(seen, _steer(patch, params, p_dy, p_dx, False, True), base)
        if kind == "idler3":
            return jnp.where(step_index % 3 == 0, jnp.int32(dynamics.ACTION_STAY), base)
        if kind == "idler7":
            return jnp.where(step_index % 7 == 0, jnp.int32(dynamics.ACTION_STAY), base)
        if kind == "grouper":
            prey_mask = patch[:, :, CH_PREY] > 0
            ally, a_dy, a_dx = _nearest(prey_mask, params.s_patch)
            move = jnp.where(
                ally,
                _steer(patch, params, a_dy, a_dx, True, False),
                _steer(patch, params, p_dy, p_dx, False, False),
            )
            return jnp.where(seen, move, base)
        raise ValueError(kind)

    def policy(weights, obs, hidden, step_index):
        del weights
        return jax.vmap(lambda o: planted(o, params_ref[0], step_index))(obs), hidden

    # `params_ref` is closed over and set per configuration, so one policy object serves both configs
    # without retracing surprises: the params object changes, the closure cell is repointed, and jit sees
    # a fresh trace because run() rebuilds the policy per config anyway.
    params_ref = [None]
    policy.params_ref = params_ref
    return policy


PLANTED = {
    "freezer": "freezing",
    "fleer": "fleeing",
    "hider": "hiding",
    "distancer": "distancing",
    "grouper": "grouping",
}

#: The calibration null is a FAMILY of predator-blind policies, not the forager alone. Run 22 found the
#: forager-only null DEGENERATE on the stillness descriptors: a policy that never emits stay gives every
#: prey exactly zero, the null sd is zero, and z is correctly refused, so a large real freezer signal
#: (0.284 against 0.000) was unreadable. The idlers stay on a fixed cadence REGARDLESS of the predator, so
#: they put honest variance into the stillness null without any anti-predator response, and their cadence
#: stillness cancels in the conditional forms (still on both sides of the near/far split). They double as
#: negative controls: a predator-blind idler must not classify as freezing.
NULL_POLICIES = ("forager", "idler3", "idler7")
POLICIES = ["forager", "idler3", "idler7", *PLANTED]


def run_policy(kind: str, overrides: dict) -> dict:
    p = dataclasses.replace(DEFAULT, t_horizon=T_HORIZON, **overrides)
    check_invariants(p)
    policy = _make(kind)
    policy.params_ref[0] = p
    r_table = required_radius(p)

    rows = []
    for seed in range(N_SEEDS):
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, r_table), food_src=fs, target_src=ts)
        w = net.init_params(jax.random.key(seed), p)
        _, acc = E.run_episode(
            jax.random.key(1000 + seed), w, static, ent.PRNGSource(jax.random.key(seed)), p, policy=policy
        )
        rows.append({f: np.asarray(getattr(acc, f), dtype=np.float64) for f in ACC_FIELDS})
    jax.clear_caches()
    gc.collect()
    return {f: np.concatenate([r[f] for r in rows]) for f in ACC_FIELDS}


def per_prey_descriptors(acc: dict) -> dict[str, np.ndarray]:
    n = len(acc["alive_ticks"])
    out: dict[str, list] = {}
    for i in range(n):
        row = {f: acc[f][i] for f in acc}
        desc = D.descriptors(row, denominator_floor=CAL_DENOM_FLOOR)
        for name, d in desc.items():
            out.setdefault(name, []).append(d.value if d.defined else np.nan)
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()
    global N_SEEDS
    if "--seeds" in sys.argv:
        N_SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1])

    payload: dict = {}
    for cfg_name, overrides in CONFIGS.items():
        print(f"\n{'=' * 96}\nCONFIGURATION: {cfg_name}  {overrides}\n{'=' * 96}", flush=True)

        accs = {}
        for kind in POLICIES:
            t1 = time.time()
            accs[kind] = run_policy(kind, overrides)
            eng = accs[kind]["engaged_ticks"].mean()
            print(
                f"  ran {kind:<10s} {N_SEEDS} seeds in {(time.time() - t1) / 60:4.1f} min   "
                f"mean engaged {eng:7.3f}  near {accs[kind]['near_ticks'].mean():7.3f}  "
                f"visible {accs[kind]['visible_ticks'].mean():7.3f}  "
                f"moved {accs[kind]['moved_ticks'].mean():7.1f}",
                flush=True,
            )

        desc = {kind: per_prey_descriptors(accs[kind]) for kind in POLICIES}
        # The pooled predator-blind null family; see NULL_POLICIES for why the forager alone is degenerate.
        nulls = {
            name: C.null_from(np.concatenate([desc[k][name] for k in NULL_POLICIES]))
            for name in desc["forager"]
        }

        # ---- detection power: z of each policy's population MEAN on each strategy descriptor ----------
        print(
            "\n  DETECTION POWER, z of policy mean against the pooled predator-blind null. Rows are policies.",
            flush=True,
        )
        strategies = list(C.NAMED_STRATEGIES)
        print(
            f"  {'policy':<10s} " + " ".join(f"{s[:9]:>10s}" for s in strategies) + "   own-strategy z",
            flush=True,
        )
        for kind in POLICIES:
            zs = []
            for strat in strategies:
                dname = C.NAMED_STRATEGIES[strat]
                vals = desc[kind][dname]
                vals = vals[np.isfinite(vals)]
                z = C.z_score(float(vals.mean()), nulls[dname]) if len(vals) else float("nan")
                zs.append(z)
                payload[f"{cfg_name}__{kind}__{strat}__z"] = np.float64(z)
                payload[f"{cfg_name}__{kind}__{strat}__n_defined"] = np.int32(len(vals))
            own = zs[strategies.index(PLANTED[kind])] if kind in PLANTED else float("nan")
            print(
                f"  {kind:<10s} "
                + " ".join(f"{z:10.2f}" for z in zs)
                + (f"   {own:8.2f}" if kind in PLANTED else "   control"),
                flush=True,
            )

        # ---- per-prey classification with CALIBRATION thresholds -> confusion table ------------------
        print(
            f"\n  CONFUSION TABLE at calibration thresholds floor={CAL_FLOOR} sep={CAL_SEPARATION}.",
            flush=True,
        )
        print(
            "  NOT the pre-registered thresholds, which are derived at step 13. The z-matrix above is",
            flush=True,
        )
        print("  the durable result; this renders it legible.", flush=True)
        labels_all = list(C.NAMED_STRATEGIES) + [C.UNCLASSIFIED, C.INSUFFICIENT]
        print(f"  {'planted':<10s} " + " ".join(f"{lb[:9]:>10s}" for lb in labels_all), flush=True)
        for kind in POLICIES:
            n = len(desc[kind]["stillness_engaged"])
            labels = []
            for i in range(n):
                values = {dn: desc[kind][dn][i] for dn in desc[kind]}
                labels.append(C.classify(values, nulls, floor=CAL_FLOOR, separation=CAL_SEPARATION).label)
            prev = C.prevalence(labels)
            for lb, v in prev.items():
                payload[f"{cfg_name}__{kind}__prev__{lb}"] = np.float64(v)
            print(f"  {kind:<10s} " + " ".join(f"{prev.get(lb, 0.0):10.3f}" for lb in labels_all), flush=True)

        # denominators, so power is read against the data each number stands on. `near_ticks` is the one
        # the re-based descriptors divide by (D-046), so it is recorded for every policy.
        payload[f"{cfg_name}__mean_engaged__forager"] = np.float64(accs["forager"]["engaged_ticks"].mean())
        payload[f"{cfg_name}__mean_engaged__freezer"] = np.float64(accs["freezer"]["engaged_ticks"].mean())
        for kind in POLICIES:
            payload[f"{cfg_name}__mean_near__{kind}"] = np.float64(accs[kind]["near_ticks"].mean())

    payload["n_seeds"] = np.int32(N_SEEDS)
    payload["cal_floor"] = np.float64(CAL_FLOOR)
    payload["cal_separation"] = np.float64(CAL_SEPARATION)
    payload["cal_denom_floor"] = np.float64(CAL_DENOM_FLOOR)
    payload["wall_seconds"] = np.float32(time.time() - t0)
    np.savez(sys.argv[1], **payload)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
