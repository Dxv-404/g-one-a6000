#!/usr/bin/env python3
"""Liveness across EVERY swept axis, at both ends AND in the middle. Q-023, widened after DEV-031.

**The question is deliberately crude: does moving this parameter change anything at all?** Not whether the
response has the right sign or size, which is characterisation's job. A dead axis produces a flat response,
and a flat response reads as "this parameter does not matter", which is a finding rather than an absence.
Three swept axes have now been found dead or partly dead: `s_burst_pred` at DEV-014, `r_pred` at DEV-030 and
`cue_dependence` at DEV-031. Each existed only at non-default values, and every gate runs at the default.

**Why a midpoint, when the instruction was both ends.** DEV-031 is the reason and it is worth stating,
because the endpoint pair is the natural design and it is exactly what missed it. `cue_dependence` was
checked at 1.0 and 0.0 and looked alive: near_ticks 36.250 against 6.141, a large and real response. The
defect sat strictly between them. The graded damage is exactly 0 at cue 1.0 and exactly DAMAGE at cue 0.0, so
the boolean reduction that discarded the grading was precisely correct at both ends and wrong everywhere
else. A five-level axis had two behaviours and an endpoint test could not see it.

    an endpoint pair establishes that the axis REACHES the simulation
    a midpoint establishes that it reaches it CONTINUOUSLY

Those are different claims and only the first is what "both ends" tests. The midpoint costs one extra run per
axis and is the only thing that would have caught the defect that motivated this pass.

**The error direction, which is what justifies choosing wide levels.** A false "live" is not available here:
if any measured quantity moved, the parameter demonstrably reaches the simulation, whatever the levels were.
The only reachable error is a false "dead", from levels too close together to produce a visible response. So
the levels are chosen wide, spanning most of each parameter's plausible range, and the asymmetry decides it
rather than a judgement about what a realistic sweep would use. These levels set no published number: this is
a diagnostic, and characterisation chooses its own.

**The instrument is the repaired hand-coded forager, not an untrained network.** Q-023 measured this and it
is not a preference: against an untrained policy 48 of 48 prey starve, movement runs at about 2 percent of
ticks, and every predator-side axis reads dead whether or not it is. A prey that never moves never forages
and is barely worth hunting. The forager is predator-oblivious, which is the right property for this
question: it measures the threat the world delivers rather than the threat a policy fails to escape.

**Axes that are not yet parameters cannot be tested and are reported as such, not as passes.** The roster is
read from `configs.provenance.SWEPT` at runtime rather than transcribed, so this script cannot go stale
against it, and `PROPOSED_AXES` members with no `EnvParams` field are listed separately. That separation is
the whole point of the two frozensets: a proposed axis counted as coverage is worse than an uncovered one.

Usage: uv run python scripts/measure/axis_liveness_full.py <out.npz>
"""

from __future__ import annotations

import dataclasses
import sys
import time

import jax
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import DEFAULT, check_invariants, make_relax_ladder
from configs.provenance import PROPOSED_AXES, SWEPT
from src.agent import baseline
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius
from src.measure import outcomes as O

#: Eight, not sixteen. The check is "did anything move", and the smallest response found so far across any
#: axis is far above the seed-to-seed spread at eight. Recorded rather than chosen silently: if an axis comes
#: back dead with a response comparable to the spread, it is re-run at more seeds before being believed.
N_SEEDS = 8

#: **400, and it is not a free choice.** A first version used 200 to keep forty settings affordable, and
#: `check_invariants` rejected it at the DEFAULT configuration:
#:
#:     feeds = (e_base * t_horizon - e_max) / e_food = (0.5 * 200 - 100) / 20 = 0.0
#:
#: At 200 ticks the total energy drain exactly equals the starting energy, so an agent never needs to eat and
#: there is NO foraging pressure at all. The band requires 2 to 10 feeds and 400 gives exactly 5. A liveness
#: measurement at 200 would have been taken in a world with no forage economy, which is not the environment
#: the experiment is about, and nothing in the numbers would have said so.
T_HORIZON = 400

#: Levels per axis: (low, mid, high). The default is run once as a shared control rather than per axis.
#:
#: Chosen wide, per the error-direction argument in the module docstring. Each is inside what
#: `check_invariants` accepts, which is asserted below rather than assumed: several are coupled, and a level
#: that silently violates a derivation would produce a response for the wrong reason.
#: **Every level below is bounded by an invariant, not by taste, and six of the first draft's levels were
#: rejected by `check_invariants` before anything ran.** They are recorded with the binding constraint
#: because "chosen wide" is only defensible up to the point where wide becomes invalid, and the boundary is
#: where the interesting reading sits.
LEVELS: dict[str, tuple] = {
    # -- predator-side ------------------------------------------------------------------------------------
    # Unconstrained by check_invariants. The full range is available.
    "cue_dependence": (0.0, 0.5, 1.0),
    "p_wander": (0.0, 0.5, 1.0),
    "p_distant_detect": (0.0, 0.5, 1.0),
    "n_pred": (1, 2, 4),
    # **2, 3, 4 here is a MISTAKE that was left in place rather than quietly corrected, and the correction
    # is recorded because the reasoning failure is more useful than the fix.**
    #
    # The comment that used to sit here said r_pred could not exceed 4, because `min(d_spawn_relax)` is 5
    # and must strictly exceed r_pred. That is wrong. `d_spawn_relax` is a PROPERTY derived from r_pred,
    # `floor = r_pred + 1`, so the assertion is a tautology and can never bind. r_pred is VALID to 6,
    # bounded by `r_pred <= r + 2`, a different assertion entirely. Measured after the fact:
    #
    #     r_pred 4 -> (10, 8, 6, 5)   VALID        r_pred 6 -> (14, 11, 9, 7)   VALID
    #     r_pred 5 -> (12, 10, 8, 6)  VALID        r_pred 7 -> (16, 13, 10, 8)  invalid, on r <= r + 2
    #
    # So this axis was tested over a third of its range for no reason, and DEV-030's gate and run 2, which
    # used r_pred 5 and 6, were INSIDE the permitted region rather than outside it as the log first said.
    #
    # A bound asserted from reasoning and never executed, which is the failure this whole pass was written
    # to catch in the code. `scripts/measure/spawn_feasibility.py` covers 4, 5 and 6 properly.
    "r_pred": (2, 3, 4),
    # Uptime is `s_regen / s_burst`, so SMALL means fast pursuit. Bounded above by `uptime_pred >
    # uptime_prey`, which is `s_burst_pred < s_burst_prey` at 20. 32 was rejected; 16 is the widest usable.
    "s_burst_pred": (4.0, 8.0, 16.0),
    # -- shared -------------------------------------------------------------------------------------------
    # Same constraint from the other side: must exceed `s_burst_pred` at 8. 5 was rejected.
    "s_burst_prey": (10.0, 20.0, 80.0),
    # Bounded by the encounter rate `k * (2r+1)^2 / (h * w * (1 - density))`, which must stay in (0.2, 1.0).
    # Density moves the denominator only, and 0.0 to 0.35 keeps it in band at the default k.
    "obstacle_density": (0.0, 0.15, 0.35),
    # Squeezed from BOTH sides, which is why it is the narrowest axis here. k = 2 gives an encounter rate of
    # 0.119, under the 0.2 floor. k = 16 gives a food demand of 80 against a supply of 72, so agents starve
    # regardless of behaviour, which removes the selection gradient rather than sharpening it.
    "k": (4, 8, 12),
    # Unconstrained by check_invariants.
    "e_sprint": (0.0, 1.0, 4.0),
    # Bounded below by `supply > demand`: at k = 8 demand is 40 and each source supplies 12, so 2 sources
    # give 24 and were rejected. 4 is the lowest valid value.
    "n_food_src": (4, 6, 12),
    # Does not enter the supply constraint, so the full range is available. The asymmetry with n_food_src is
    # real rather than an oversight: only food sources feed the energy economy.
    "n_target_src": (2, 6, 12),
    # The ladder is swept whole, via its construction rule, not by moving the head alone: a head-only edit
    # breaks the descending assertion at low values and unbalances the steps at high ones.
    "d_src_relax": (make_relax_ladder(2), make_relax_ladder(8), make_relax_ladder(16)),
}

#: Every quantity the verdict is read from. Broad on purpose: an axis that moves only one of these is live,
#: and narrowing the vector is how a live axis gets called dead.
ACC_FIELDS = ("near_ticks", "visible_ticks", "engaged_ticks", "alive_ticks", "moved_ticks", "occluded_near")


def make_policy(params):
    def policy(weights, obs, hidden, step_index):
        del weights
        return jax.vmap(lambda o: baseline.forager(o, params, step_index))(obs), hidden

    return policy


def run_setting(overrides: dict) -> dict:
    p = DEFAULT.__class__(t_horizon=T_HORIZON, **overrides)
    check_invariants(p)  # a level that breaks a derivation must fail here, not produce a response
    policy = make_policy(p)
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

    o = O.outcomes(np.concatenate(died_p), np.concatenate(died_s), np.concatenate(alive))
    out = {f: float(np.concatenate([r[f] for r in acc_rows]).mean()) for f in ACC_FIELDS}
    out["score"] = float(np.concatenate(score).mean())
    out["predated"] = float(o.predated)
    out["starved"] = float(o.starved)
    out["survived"] = float(o.survived)
    return out


def spread(rows: list[dict]) -> tuple[str, float, str]:
    """Largest relative movement of any quantity across the three levels, and which quantity it was.

    Relative to the largest of the three, so a quantity that goes to zero reads as a full move rather than as
    a division by zero. A quantity flat at zero everywhere contributes nothing, which is correct: it is not
    evidence of liveness.
    """
    best_q, best_v = "none", 0.0
    for q in rows[0]:
        vals = [r[q] for r in rows]
        lo, hi = min(vals), max(vals)
        denom = max(abs(hi), abs(lo))
        rel = 0.0 if denom == 0.0 else (hi - lo) / denom
        if rel > best_v:
            best_q, best_v = q, rel

    # Interior collapse: the midpoint sitting exactly on an endpoint across EVERY quantity is the DEV-031
    # signature, and it is invisible to an endpoint pair.
    interior = "ok"
    if best_v > 0.0:
        eq_lo = all(rows[1][q] == rows[0][q] for q in rows[0])
        eq_hi = all(rows[1][q] == rows[2][q] for q in rows[0])
        if eq_lo or eq_hi:
            interior = "COLLAPSED to " + ("low" if eq_lo else "high")
    return best_q, best_v, interior


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()

    fields = {f.name for f in dataclasses.fields(DEFAULT)}
    testable = sorted(a for a in SWEPT if a in fields)
    untestable = sorted((SWEPT | PROPOSED_AXES) - fields)
    missing_levels = [a for a in testable if a not in LEVELS]
    assert not missing_levels, f"swept axis with no levels defined: {missing_levels}"
    extra_levels = [a for a in LEVELS if a not in testable]
    assert not extra_levels, f"levels defined for a non-swept name: {extra_levels}"

    print("\nroster read from configs.provenance, not transcribed", flush=True)
    print(f"  swept axes that are EnvParams fields : {len(testable)}  {testable}", flush=True)
    print(f"  named but not parameters, untestable : {len(untestable)}  {untestable}", flush=True)
    print(f"  settings to run                      : {1 + 3 * len(testable)}\n", flush=True)

    # **Pre-flight: validate every setting BEFORE running any of them.**
    #
    # A first version validated inside `run_setting`, so an invalid level would have surfaced after however
    # many valid settings preceded it, forty minutes into a run. Six of the first draft's levels were
    # invalid, which would have meant six discover-fix-restart cycles at an hour each.
    #
    # The general shape is worth naming: when a batch of N jobs each has a cheap precondition and an
    # expensive body, check every precondition first. The cost is N cheap checks and the saving is the
    # partial run, and this is the same reasoning as reading a step's done criterion at step START.
    bad = []
    for axis in testable:
        for lv in LEVELS[axis]:
            try:
                check_invariants(DEFAULT.__class__(t_horizon=T_HORIZON, **{axis: lv}))
            except AssertionError as e:
                bad.append(f"  {axis}={lv}: {str(e).splitlines()[0][:110]}")
    try:
        check_invariants(DEFAULT.__class__(t_horizon=T_HORIZON))
    except AssertionError as e:
        bad.append(f"  the DEFAULT control itself: {str(e).splitlines()[0][:110]}")
    if bad:
        print("PRE-FLIGHT FAILED. These settings break a derivation and were not run:", flush=True)
        print("\n".join(bad), flush=True)
        return 1
    print(f"pre-flight: all {1 + 3 * len(testable)} settings satisfy check_invariants\n", flush=True)

    results: dict[str, dict] = {}
    print("running the shared default control", flush=True)
    results["default"] = run_setting({})

    hdr = f"{'axis':20s} {'level':>16s} " + " ".join(f"{f[:9]:>9s}" for f in ACC_FIELDS)
    hdr += f" {'score':>7s} {'predat':>7s} {'starv':>7s}"
    print("\n" + hdr, flush=True)
    d = results["default"]
    print(
        f"{'(default)':20s} {'-':>16s} "
        + " ".join(f"{d[f]:9.3f}" for f in ACC_FIELDS)
        + f" {d['score']:7.3f} {d['predated']:7.3f} {d['starved']:7.3f}",
        flush=True,
    )

    verdicts = []
    for axis in testable:
        rows = []
        for lv in LEVELS[axis]:
            r = run_setting({axis: lv})
            rows.append(r)
            results[f"{axis}={lv}"] = r
            print(
                f"{axis:20s} {str(lv)[:16]:>16s} "
                + " ".join(f"{r[f]:9.3f}" for f in ACC_FIELDS)
                + f" {r['score']:7.3f} {r['predated']:7.3f} {r['starved']:7.3f}",
                flush=True,
            )
        q, v, interior = spread(rows)
        verdicts.append((axis, q, v, interior))
        print(f"{'':20s} {'-> ':>16s} largest move {v * 100:6.1f}% on {q}, interior {interior}\n", flush=True)

    print("\n" + "=" * 100, flush=True)
    print(f"{'axis':20s} {'verdict':>8s} {'largest move':>14s}  {'on':<16s} interior", flush=True)
    for axis, q, v, interior in sorted(verdicts, key=lambda t: t[2]):
        verdict = "DEAD" if v == 0.0 else ("faint" if v < 0.02 else "live")
        print(f"{axis:20s} {verdict:>8s} {v * 100:13.1f}%  {q:<16s} {interior}", flush=True)
    print("=" * 100, flush=True)
    print(f"\nuntestable, not parameters: {untestable}", flush=True)

    payload = {f"{k}__{q}": np.float64(v) for k, row in results.items() for q, v in row.items()}
    payload["axes_testable"] = np.array(testable)
    payload["axes_untestable"] = np.array(untestable)
    payload["n_seeds"] = np.int32(N_SEEDS)
    payload["t_horizon"] = np.int32(T_HORIZON)
    payload["wall_seconds"] = np.float32(time.time() - t0)
    np.savez(sys.argv[1], **payload)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
