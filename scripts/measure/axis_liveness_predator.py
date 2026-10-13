#!/usr/bin/env python3
"""The last two axes, with the instrument defects from the previous two passes removed.

**Two passes have now produced verdicts that were facts about the instrument rather than the code**, so this
one states the mechanism it needs and asserts the mechanism ran before reading any verdict.

    p_distant_detect   pass 2 ran it against an always-STAY prey and got near_ticks EXACTLY ZERO. The
                       default predator has p_wander = 0.0, so it never moves; spawn separation is 5
                       against r_pred 4, so a motionless predator and a motionless prey never meet.
                       The detection rule was never reached because nothing was ever in range.
                       Fixed here by p_wander = 1.0, and near_ticks > 0 is ASSERTED before the verdict.

    s_burst_pred       pass 2 got 0.8 percent with near, visible, engaged, alive and moved identical to
                       three decimals across a 4x change in predator sprint uptime. Three sufficient
                       causes, and the flat response alone cannot choose between them:

                           1  the predator never sprints           instrument or configuration
                           2  stamina is not written back          a DEV-014 regression
                           3  uptime does not change capture       a real null, worth reporting

                       So this measures the predator directly: does it sprint, does its stamina vary
                       within an episode, and does the sprint RATE respond to s_burst_pred. Only if the
                       mechanism demonstrably runs is a flat outcome a finding about the world.

**This is the count-the-sufficient-causes rule applied to a measurement instead of to an assertion.** A flat
response is a symptom with several causes, exactly like "scores near zero" at S9 and "a different seed gives
a different run" at S7. The fix is the same one: instrument the mechanism, not the outcome.

Usage: uv run python scripts/measure/axis_liveness_predator.py <out.npz>
"""

from __future__ import annotations

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
from src.env.perception import build_vis_table, required_radius

N_SEEDS = 8
T_HORIZON = 400

ACC_FIELDS = (
    "near_ticks",
    "visible_ticks",
    "engaged_ticks",
    "alive_ticks",
    "moved_ticks",
    "occluded_near",
    "accidental_still",
    "still_unthreatened",
)


def policy_still(params):
    del params

    def policy(weights, obs, hidden, step_index):
        del weights, step_index
        return jnp.full((obs.shape[0],), dynamics.ACTION_STAY, jnp.int32), hidden

    return policy


def policy_sprinting(params):
    def policy(weights, obs, hidden, step_index):
        del weights
        a = jax.vmap(lambda o: baseline.forager(o, params, step_index))(obs)
        basic = (a >= dynamics.ACTION_MOVE_FIRST) & (a <= dynamics.ACTION_MOVE_LAST)
        return jnp.where(basic, a + dynamics.ACTION_SPRINT_FIRST, a), hidden

    return policy


def run(policy_maker, overrides: dict) -> dict:
    p = DEFAULT.__class__(t_horizon=T_HORIZON, **overrides)
    check_invariants(p)
    policy = policy_maker(p)
    r_table = required_radius(p)

    rows, died_p, died_s, alive = [], [], [], []
    for seed in range(N_SEEDS):
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, r_table), food_src=fs, target_src=ts)
        w = net.init_params(jax.random.key(seed), p)
        state, acc = E.run_episode(
            jax.random.key(1000 + seed), w, static, ent.PRNGSource(jax.random.key(seed)), p, policy=policy
        )
        rows.append({f: np.asarray(getattr(acc, f), dtype=np.float64) for f in ACC_FIELDS})
        died_p.append(np.asarray(state.prey_died_predated))
        died_s.append(np.asarray(state.prey_died_starving))
        alive.append(np.asarray(state.prey_alive))

    out = {f: float(np.concatenate([r[f] for r in rows]).mean()) for f in ACC_FIELDS}
    out["predated"] = float(np.concatenate(died_p).mean())
    out["starved"] = float(np.concatenate(died_s).mean())
    out["survived"] = float(np.concatenate(alive).mean())
    return out


def probe_predator(s_burst_pred: float, ticks: int = 200) -> dict:
    """Watch the predator itself: does it sprint, and does its stamina move within an episode.

    **Reads `state.pred_stamina` and `state.pred_still` directly rather than inferring from prey outcomes.**
    DEV-014 was predator stamina never being written back, and the symptom was a swept axis that did
    nothing. Inferring from outcomes is how it survived eight build steps; reading the pool is what found it.
    """
    p = DEFAULT.__class__(t_horizon=T_HORIZON, p_wander=1.0, n_pred=4, s_burst_pred=s_burst_pred)
    check_invariants(p)
    policy = policy_sprinting(p)
    ob = grid.make_obstacle_grid(jax.random.key(0), p.h, p.w, p.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(0), ob, p)
    static = E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )
    w = net.init_params(jax.random.key(0), p)
    state = E.reset(ent.PRNGSource(jax.random.key(1000)), static, p)
    entropy = ent.PRNGSource(jax.random.key(0))
    acc = E.zero_accumulators(p.k)

    stamina_seen, moved_cells, targeted = [], 0, 0
    prev = np.asarray(state.pred_pos).copy()
    for t in range(ticks):
        state, acc, _rec = E.tick(state, acc, w, static, entropy, jnp.int32(t), p, policy)
        st = np.asarray(state.pred_stamina, dtype=np.float64)
        stamina_seen.append(st.copy())
        now = np.asarray(state.pred_pos)
        # Chebyshev step length per predator. 2 means a sprint actually executed.
        step = np.abs(now - prev).max(axis=1)
        moved_cells += int((step >= 2).sum())
        targeted += int((np.asarray(state.pred_target) >= 0).sum())
        prev = now.copy()

    s = np.array(stamina_seen)  # (ticks, n_pred)
    return {
        "sprint_steps": float(moved_cells) / (ticks * p.n_pred),
        "targeted_share": float(targeted) / (ticks * p.n_pred),
        "stamina_min": float(s.min()),
        "stamina_max": float(s.max()),
        "stamina_mean": float(s.mean()),
        "stamina_distinct": float(len(np.unique(np.round(s, 3)))),
    }


def spread(rows: list[dict]) -> tuple[str, float]:
    best_q, best_v = "none", 0.0
    for q in rows[0]:
        vals = [r[q] for r in rows]
        lo, hi = min(vals), max(vals)
        denom = max(abs(hi), abs(lo))
        rel = 0.0 if denom == 0.0 else (hi - lo) / denom
        if rel > best_v:
            best_q, best_v = q, rel
    return best_q, best_v


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()
    payload: dict = {}

    # ---------------------------------------------------------------- 1. the predator mechanism ---------
    print("\n=== does the predator sprint at all, and does its stamina move ===", flush=True)
    print(
        f"{'s_burst_pred':>13s} {'sprint steps':>13s} {'targeted':>9s} {'stam min':>9s} "
        f"{'stam max':>9s} {'stam mean':>10s} {'distinct':>9s}",
        flush=True,
    )
    probes = {}
    for lv in (4.0, 8.0, 16.0):
        r = probe_predator(lv)
        probes[lv] = r
        for q, v in r.items():
            payload[f"probe_{lv}__{q}"] = np.float64(v)
        print(
            f"{lv:13.1f} {r['sprint_steps']:13.3f} {r['targeted_share']:9.3f} {r['stamina_min']:9.2f} "
            f"{r['stamina_max']:9.2f} {r['stamina_mean']:10.2f} {r['stamina_distinct']:9.0f}",
            flush=True,
        )

    varies = probes[8.0]["stamina_distinct"] > 1
    sprints = probes[8.0]["sprint_steps"] > 0.0
    print(f"\n  predator stamina takes more than one value : {varies}", flush=True)
    print(f"  predator executes at least one sprint step : {sprints}", flush=True)
    if not sprints:
        print(
            "  ==> CAUSE 1. The predator never sprints, so s_burst_pred has no mechanism to act", flush=True
        )
    elif not varies:
        print("  ==> CAUSE 2. Stamina is constant, which is DEV-014's signature returning", flush=True)
    else:
        print("  ==> mechanism runs. A flat outcome would be CAUSE 3, a real null", flush=True)

    # ---------------------------------------------------------------- 2. p_distant_detect, properly -----
    print("\n=== p_distant_detect, still prey and a WANDERING predator ===", flush=True)
    print(
        "pass 2 used p_wander 0.0, so near_ticks was exactly zero and the axis was unreachable.", flush=True
    )
    hdr = f"{'level':>6s} " + " ".join(f"{f[:9]:>9s}" for f in ACC_FIELDS) + f" {'predat':>7s} {'starv':>7s}"
    print(hdr, flush=True)
    rows = []
    for lv in (0.0, 0.5, 1.0):
        r = run(policy_still, {"p_distant_detect": lv, "p_wander": 1.0, "n_pred": 4})
        rows.append(r)
        for q, v in r.items():
            payload[f"pdd_{lv}__{q}"] = np.float64(v)
        print(
            f"{lv:6.2f} "
            + " ".join(f"{r[f]:9.3f}" for f in ACC_FIELDS)
            + f" {r['predated']:7.3f} {r['starved']:7.3f}",
            flush=True,
        )
    assert rows[1]["near_ticks"] > 0.0, (
        f"near_ticks is {rows[1]['near_ticks']}, so the predator never came within r_pred and the "
        "detection rule was never reached. The verdict below would be a fact about the geometry."
    )
    print("  precondition holds: near_ticks > 0, so the detection rule was actually reached", flush=True)
    q, v = spread(rows)
    verdict = "DEAD" if v == 0.0 else ("faint" if v < 0.02 else "live")
    print(f"  -> p_distant_detect: {verdict}, largest move {v * 100:.1f}% on {q}", flush=True)

    # ---------------------------------------------------------------- 3. s_burst_pred, wider read -------
    print("\n=== s_burst_pred, sprinting prey, wandering predators ===", flush=True)
    print(hdr, flush=True)
    rows2 = []
    for lv in (4.0, 8.0, 16.0):
        r = run(policy_sprinting, {"s_burst_pred": lv, "p_wander": 1.0, "n_pred": 4})
        rows2.append(r)
        for qq, vv in r.items():
            payload[f"sbp_{lv}__{qq}"] = np.float64(vv)
        print(
            f"{lv:6.1f} "
            + " ".join(f"{r[f]:9.3f}" for f in ACC_FIELDS)
            + f" {r['predated']:7.3f} {r['starved']:7.3f}",
            flush=True,
        )
    q2, v2 = spread(rows2)
    verdict2 = "DEAD" if v2 == 0.0 else ("faint" if v2 < 0.02 else "live")
    print(f"  -> s_burst_pred: {verdict2}, largest move {v2 * 100:.1f}% on {q2}", flush=True)

    payload["wall_seconds"] = np.float32(time.time() - t0)
    np.savez(sys.argv[1], **payload)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
