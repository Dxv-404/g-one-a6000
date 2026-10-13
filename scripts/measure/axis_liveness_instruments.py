#!/usr/bin/env python3
"""The four axes the forager could not test, re-run against instruments that CAN move them.

**The first pass reported three axes DEAD and one faint, and all four verdicts were artifacts of the
instrument rather than facts about the code.** That is the failure Q-023 exists to prevent, reproduced by
the tool built to prevent it, so it is worth stating exactly how it happened.

    e_sprint          charged only when the prey SPRINTS
    s_burst_prey      drawn only when the prey SPRINTS
    s_burst_pred      matters only during sustained PURSUIT
    p_distant_detect  the draw is reached only when the prey is STILL

`src/agent/baseline.py` says so in its own docstring: "this policy does not sprint". It also moves on about
88 percent of its alive ticks, and the two-regime rule detects a distant entity on motion OR a draw, so the
draw is short-circuited almost always. And the default predator has `p_wander = 0.0`, so it moves only when
it has a target, and engagement runs at 0.188 ticks per episode.

**So four of thirteen axes were structurally unmeasurable and three of them reported 0.0 percent.** A
verdict of DEAD and a verdict of NOT ASKED are different claims that print identically.

The rule this implies, which is the transferable part: **an axis is only testable by an instrument that
exercises the mechanism the axis prices.** A single instrument cannot cover a parameter set that prices
several different behaviours, and choosing one instrument for all of them silently converts "this policy
never does X" into "X does not matter".

    forager           environment geometry, resources, predator count and reach
    sprinting         exertion: e_sprint, s_burst_prey
    still             concealment: p_distant_detect, where motion must NOT short-circuit the draw
    fleeing + wander  pursuit: s_burst_pred, which needs the predator actually chasing

**`accidental_still` is read here and was missing from the first pass**, which is its own small lesson. It
counts ticks where the policy emitted a move and the agent did not move anyway, so it is the direct
observable for stamina exhaustion. Measuring exertion axes without it was measuring the effect while
omitting the quantity the effect is defined on.

Usage: uv run python scripts/measure/axis_liveness_instruments.py <out.npz>
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

#: Wider than the first pass. `accidental_still` is the exhaustion observable and its omission is why the
#: exertion axes had nothing to move even in principle.
ACC_FIELDS = (
    "near_ticks",
    "visible_ticks",
    "engaged_ticks",
    "alive_ticks",
    "moved_ticks",
    "occluded_near",
    "accidental_still",
    "still_unthreatened",
    "unthreatened_ticks",
    "far_ticks",
)


def policy_forager(params):
    """The step 9 baseline, unchanged. Never sprints."""

    def policy(weights, obs, hidden, step_index):
        del weights
        return jax.vmap(lambda o: baseline.forager(o, params, step_index))(obs), hidden

    return policy


def policy_sprinting(params):
    """The same forager, with every basic move promoted to the sprint of the same direction.

    Sprint actions are `ACTION_SPRINT_FIRST + d` for the same direction index `d` as `ACTION_MOVE_FIRST + d`,
    so the promotion is an offset and nothing about the direction choice changes. PICK and STAY are left
    alone: promoting a PICK would change what the policy DOES rather than how hard it does it, and the point
    of this instrument is to hold the behaviour fixed and vary only the exertion.

    Blocked sprints are masked to STAY by `mask_actions` and land in `accidental_still`, which is exactly the
    signal the exertion axes are supposed to move.
    """

    def policy(weights, obs, hidden, step_index):
        del weights
        a = jax.vmap(lambda o: baseline.forager(o, params, step_index))(obs)
        basic = (a >= dynamics.ACTION_MOVE_FIRST) & (a <= dynamics.ACTION_MOVE_LAST)
        return jnp.where(basic, a + dynamics.ACTION_SPRINT_FIRST, a), hidden

    return policy


def policy_still(params):
    """Always STAY, so motion never short-circuits the distant-detection draw.

    This is the ONLY instrument under which `p_distant_detect` can be read at all: the rule is motion OR a
    draw, and a policy that moves makes the draw unreachable.
    """
    del params

    def policy(weights, obs, hidden, step_index):
        del weights, step_index
        return jnp.full((obs.shape[0],), dynamics.ACTION_STAY, jnp.int32), hidden

    return policy


INSTRUMENTS = {"forager": policy_forager, "sprinting": policy_sprinting, "still": policy_still}

#: Each axis, the instrument that can move it, why, and the levels. Levels are unchanged from the first pass
#: where they were valid, so the two runs are comparable on the axes they share.
PLAN: list[tuple[str, str, tuple, dict, str]] = [
    (
        "e_sprint",
        "sprinting",
        (0.0, 1.0, 4.0),
        {},
        "sprint energy draw. Charged per sprint, so a non-sprinting policy cannot see it.",
    ),
    (
        "s_burst_prey",
        "sprinting",
        (10.0, 20.0, 80.0),
        {},
        "prey burst cost. Drawn per sprint. Exhaustion shows in accidental_still.",
    ),
    (
        "p_distant_detect",
        "still",
        (0.0, 0.5, 1.0),
        {},
        "concealment. The draw is reached only when the prey did not move.",
    ),
    (
        "s_burst_pred",
        "sprinting",
        (4.0, 8.0, 16.0),
        {"p_wander": 1.0, "n_pred": 4},
        "pursuit cost. Needs the predator actually chasing: p_wander 1.0 and 4 predators, "
        "against fleeing prey. At the default the predator is static and engagement is 0.19 ticks.",
    ),
]


def run_setting(instrument: str, overrides: dict) -> dict:
    p = DEFAULT.__class__(t_horizon=T_HORIZON, **overrides)
    check_invariants(p)
    policy = INSTRUMENTS[instrument](p)
    r_table = required_radius(p)

    acc_rows, died_p, died_s, alive, score = [], [], [], [], []
    for seed in range(N_SEEDS):
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, r_table), food_src=fs, target_src=ts)
        w = net.init_params(jax.random.key(seed), p)
        state, acc = E.run_episode(
            jax.random.key(1000 + seed), w, static, ent.PRNGSource(jax.random.key(seed)), p, policy=policy
        )
        acc_rows.append({f: np.asarray(getattr(acc, f), dtype=np.float64) for f in ACC_FIELDS})
        died_p.append(np.asarray(state.prey_died_predated))
        died_s.append(np.asarray(state.prey_died_starving))
        alive.append(np.asarray(state.prey_alive))
        score.append(np.asarray(state.prey_score, dtype=np.float64))

    out = {f: float(np.concatenate([r[f] for r in acc_rows]).mean()) for f in ACC_FIELDS}
    out["score"] = float(np.concatenate(score).mean())
    out["predated"] = float(np.concatenate(died_p).mean())
    out["starved"] = float(np.concatenate(died_s).mean())
    out["survived"] = float(np.concatenate(alive).mean())
    return out


def spread(rows: list[dict]) -> tuple[str, float, str]:
    best_q, best_v = "none", 0.0
    for q in rows[0]:
        vals = [r[q] for r in rows]
        lo, hi = min(vals), max(vals)
        denom = max(abs(hi), abs(lo))
        rel = 0.0 if denom == 0.0 else (hi - lo) / denom
        if rel > best_v:
            best_q, best_v = q, rel
    interior = "ok"
    if best_v > 0.0:
        eq_lo = all(rows[1][q] == rows[0][q] for q in rows[0])
        eq_hi = all(rows[1][q] == rows[2][q] for q in rows[0])
        if eq_lo or eq_hi:
            interior = "COLLAPSED to " + ("low" if eq_lo else "high")
    return best_q, best_v, interior


def audit_instruments() -> dict:
    """**Assert the instrument does what its name says, before trusting any verdict it produces.**

    The first pass trusted a docstring. This measures.

    **Read from `TickRecord.action_emitted`, inside the real loop**, rather than by calling the policy on a
    reconstructed observation. That matters for the same reason D-031 split emitted from executed: the
    emitted action is what the policy chose, and a sprint that is masked to STAY for want of stamina is
    still evidence that the instrument sprints. Reconstructing the observation outside `tick` would be a
    second copy of the observation pipeline, which is the duplication failure in CLAUDE.md's list, and would
    make the audit agree with a parallel implementation instead of with the mechanism.

    Also reports `executed_sprint_share`, and the GAP between the two is itself the exertion signal: it is
    the fraction of intended sprints that stamina refused.
    """
    p = DEFAULT.__class__(t_horizon=T_HORIZON)
    check_invariants(p)
    ob = grid.make_obstacle_grid(jax.random.key(0), p.h, p.w, p.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(0), ob, p)
    static = E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )
    w = net.init_params(jax.random.key(0), p)

    report = {}
    for name, maker in INSTRUMENTS.items():
        policy = maker(p)
        state = E.reset(ent.PRNGSource(jax.random.key(1000)), static, p)
        entropy = ent.PRNGSource(jax.random.key(0))
        acc = E.zero_accumulators(p.k)
        emit_sprint = emit_move = emit_stay = exec_sprint = total = 0
        for t in range(60):
            alive = np.asarray(state.prey_alive)
            state, acc, rec = E.tick(state, acc, w, static, entropy, jnp.int32(t), p, policy)
            em = np.asarray(rec.action_emitted)
            ex = np.asarray(rec.action_executed)
            emit_sprint += int(((em >= 8) & (em <= 15) & alive).sum())
            emit_move += int(((em >= 0) & (em <= 7) & alive).sum())
            emit_stay += int(((em == 16) & alive).sum())
            exec_sprint += int(((ex >= 8) & (ex <= 15) & alive).sum())
            total += int(alive.sum())
        n = max(total, 1)
        report[name] = {
            "sprint_share": emit_sprint / n,
            "basic_move_share": emit_move / n,
            "stay_share": emit_stay / n,
            "executed_sprint_share": exec_sprint / n,
        }
    return report


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()

    print("\n=== instrument audit: does each instrument do what its name says ===", flush=True)
    audit = audit_instruments()
    print(
        f"{'instrument':14s} {'emit sprint':>12s} {'emit move':>10s} {'emit stay':>10s} {'exec sprint':>12s}",
        flush=True,
    )
    for name, r in audit.items():
        print(
            f"{name:14s} {r['sprint_share']:12.3f} {r['basic_move_share']:10.3f} "
            f"{r['stay_share']:10.3f} {r['executed_sprint_share']:12.3f}",
            flush=True,
        )
    assert audit["sprinting"]["sprint_share"] > 0.5, (
        f"the sprinting instrument emits {audit['sprinting']['sprint_share']:.3f} sprints. It cannot move an "
        "exertion axis, so any DEAD verdict from it would be an artifact exactly like the first pass."
    )
    assert audit["forager"]["sprint_share"] == 0.0, "the plain forager should emit no sprints at all"
    assert audit["still"]["stay_share"] == 1.0, "the still instrument must never move"
    print(
        "  instrument audit passed: the sprinting policy sprints, the still policy never moves\n", flush=True
    )

    print("=== pre-flight ===", flush=True)
    bad = []
    for axis, _inst, levels, extra, _why in PLAN:
        for lv in levels:
            try:
                check_invariants(DEFAULT.__class__(t_horizon=T_HORIZON, **{axis: lv}, **extra))
            except AssertionError as e:
                bad.append(f"  {axis}={lv} {extra}: {str(e).splitlines()[0][:100]}")
    if bad:
        print("PRE-FLIGHT FAILED:", flush=True)
        print("\n".join(bad), flush=True)
        return 1
    print("  all settings satisfy check_invariants\n", flush=True)

    hdr = f"{'axis':18s} {'instr':>10s} {'level':>8s} " + " ".join(f"{f[:9]:>9s}" for f in ACC_FIELDS[:7])
    hdr += f" {'score':>7s} {'predat':>7s} {'starv':>7s}"

    results, verdicts = {}, []
    for axis, instrument, levels, extra, why in PLAN:
        print(f"\n{axis}  via `{instrument}`  {extra if extra else ''}", flush=True)
        print(f"  {why}", flush=True)
        print(hdr, flush=True)
        rows = []
        for lv in levels:
            r = run_setting(instrument, {axis: lv, **extra})
            rows.append(r)
            results[f"{axis}={lv}"] = r
            print(
                f"{axis:18s} {instrument:>10s} {str(lv):>8s} "
                + " ".join(f"{r[f]:9.3f}" for f in ACC_FIELDS[:7])
                + f" {r['score']:7.3f} {r['predated']:7.3f} {r['starved']:7.3f}",
                flush=True,
            )
        q, v, interior = spread(rows)
        verdicts.append((axis, instrument, q, v, interior))
        print(f"  -> largest move {v * 100:.1f}% on {q}, interior {interior}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print(
        f"{'axis':18s} {'instrument':>11s} {'verdict':>8s} {'largest move':>14s}  {'on':<18s} interior",
        flush=True,
    )
    for axis, instrument, q, v, interior in sorted(verdicts, key=lambda t: t[3]):
        verdict = "DEAD" if v == 0.0 else ("faint" if v < 0.02 else "live")
        print(f"{axis:18s} {instrument:>11s} {verdict:>8s} {v * 100:13.1f}%  {q:<18s} {interior}", flush=True)
    print("=" * 100, flush=True)

    payload = {f"{k}__{q}": np.float64(v) for k, row in results.items() for q, v in row.items()}
    for name, r in audit.items():
        for q, v in r.items():
            payload[f"audit__{name}__{q}"] = np.float64(v)
    payload["n_seeds"] = np.int32(N_SEEDS)
    payload["t_horizon"] = np.int32(T_HORIZON)
    payload["wall_seconds"] = np.float32(time.time() - t0)
    np.savez(sys.argv[1], **payload)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
