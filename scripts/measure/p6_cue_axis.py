#!/usr/bin/env python3
"""P6 across the FULL cue axis, plus the measurement that separates instrument from configuration.

**Why a re-run rather than a re-read.** P6 was answered at `cue_dependence` 1.0 and 0.0, which is exactly the
pair DEV-031 made uninformative: the graded damage is precisely 0 at one end and precisely DAMAGE at the
other, so the boolean reduction that discarded the grading was correct at both and wrong at every value
between. The axis had five levels and two behaviours. It now has five, so P6 is asked across all of them.

**P6, as stated in `docs/q023-prediction.md`:** tier three needs about fifty visible ticks per prey to have a
usable denominator, on the reasoning that a rate estimated from fewer than about thirty events has a standard
error above 20 percent of itself. Predicted: no single axis reaches it, and the combination does not either.

**The confound this run exists to remove.** Reading runs 5 and 7 together gives:

    forager, every single-axis setting     best 8.703 visible ticks per prey, P6 HOLDS, 5.7x short
    still prey, p_wander 1.0, n_pred 4          58.812, P6 EXCEEDED

but that second configuration changes THREE things at once against the forager baseline: the policy is
always-STAY rather than foraging, the predator wanders, and there are four of them. "Tier three gets a
denominator if the prey freezes" is the tempting reading and the data does not license it, because the
predator settings alone could account for all of it.

So the cross below runs both instruments at both predator configurations. Four cells, and the difference
between the two policy rows at fixed predator settings is the part attributable to stillness.

**A prediction, recorded before the run**, per D-041, since this is the interpretation that would be reached
for afterwards:

    the predator settings carry most of it. `near_ticks` at the default forager is 4.016 and rises to
    6.406 at n_pred 4 alone, while the still-prey cell shows 46.6. A 7x gap that n_pred alone moves
    1.6x is unlikely to be policy-free.

    but stillness should carry a real share, because a still prey does not walk out of the predator's
    radius. Predicted: the forager at p_wander 1.0 and n_pred 4 lands between 8 and 25 visible ticks
    per prey, so P6 still HOLDS for a foraging policy and the still-prey figure is not reproducible
    by configuration alone.

    if the forager cell lands above 50, P6 is falsified for a realistic policy and tier three is
    reachable by configuration, which is a materially different finding and a better one.

Usage: uv run python scripts/measure/p6_cue_axis.py <out.npz>
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
P6_THRESHOLD = 50.0

#: The full axis. Five levels, which is what the owner's sweep design specifies and what the endpoint pair
#: could not see.
CUE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)

ACC_FIELDS = (
    "near_ticks",
    "visible_ticks",
    "engaged_ticks",
    "alive_ticks",
    "moved_ticks",
    "occluded_near",
    "still_engaged",
    "accidental_still",
)


def policy_forager(params):
    def policy(weights, obs, hidden, step_index):
        del weights
        return jax.vmap(lambda o: baseline.forager(o, params, step_index))(obs), hidden

    return policy


def policy_still(params):
    del params

    def policy(weights, obs, hidden, step_index):
        del weights, step_index
        return jnp.full((obs.shape[0],), dynamics.ACTION_STAY, jnp.int32), hidden

    return policy


INSTRUMENTS = {"forager": policy_forager, "still": policy_still}


def run(instrument: str, overrides: dict) -> dict:
    p = DEFAULT.__class__(t_horizon=T_HORIZON, **overrides)
    check_invariants(p)
    policy = INSTRUMENTS[instrument](p)
    r_table = required_radius(p)

    rows, alive = [], []
    for seed in range(N_SEEDS):
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, r_table), food_src=fs, target_src=ts)
        w = net.init_params(jax.random.key(seed), p)
        state, acc = E.run_episode(
            jax.random.key(1000 + seed), w, static, ent.PRNGSource(jax.random.key(seed)), p, policy=policy
        )
        rows.append({f: np.asarray(getattr(acc, f), dtype=np.float64) for f in ACC_FIELDS})
        alive.append(np.asarray(state.prey_alive))

    out = {f: float(np.concatenate([r[f] for r in rows]).mean()) for f in ACC_FIELDS}
    out["survived"] = float(np.concatenate(alive).mean())
    return out


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()
    payload: dict = {}
    hdr = f"{'cell':<44s} " + " ".join(f"{f[:9]:>9s}" for f in ACC_FIELDS)

    # ------------------------------------------------------------ 1. the full cue axis -----------------
    print("\n=== P6 and P4 across the FULL cue axis, forager, default predator ===", flush=True)
    print(hdr, flush=True)
    cue_rows = []
    for lv in CUE_LEVELS:
        r = run("forager", {"cue_dependence": lv})
        cue_rows.append(r)
        for q, v in r.items():
            payload[f"cue_{lv}__{q}"] = np.float64(v)
        print(f"{f'cue = {lv}':<44s} " + " ".join(f"{r[f]:9.3f}" for f in ACC_FIELDS), flush=True)

    vis = [r["visible_ticks"] for r in cue_rows]
    eng = [r["engaged_ticks"] for r in cue_rows]
    print(
        f"\n  visible spread {(max(vis) - min(vis)) / max(vis) * 100:.1f}%, "
        f"engaged spread {(max(eng) - min(eng)) / max(eng) * 100:.1f}%",
        flush=True,
    )
    print(
        f"  P4, cue moves engaged more than visible: {'HOLDS' if (max(eng) - min(eng)) / max(eng) > (max(vis) - min(vis)) / max(vis) else 'FALSIFIED'}",
        flush=True,
    )
    print(
        f"  best visible on this axis {max(vis):.3f} against P6's {P6_THRESHOLD}: "
        f"{'REACHES' if max(vis) >= P6_THRESHOLD else f'{P6_THRESHOLD / max(vis):.1f}x short'}",
        flush=True,
    )
    print(f"  monotone in cue: {vis == sorted(vis) or vis == sorted(vis, reverse=True)}", flush=True)

    # ------------------------------------------------------------ 2. the 2x2 that separates ------------
    print("\n=== the confound: instrument against predator configuration ===", flush=True)
    print(
        "run 7's 58.8 changed policy, p_wander and n_pred all at once. This holds two fixed at a time.",
        flush=True,
    )
    print(hdr, flush=True)
    cells = {}
    for instrument in ("forager", "still"):
        for label, extra in (
            ("default pred", {}),
            ("p_wander 1.0, n_pred 4", {"p_wander": 1.0, "n_pred": 4}),
        ):
            r = run(instrument, extra)
            cells[(instrument, label)] = r
            for q, v in r.items():
                payload[f"cell_{instrument}_{label.replace(' ', '_').replace(',', '')}__{q}"] = np.float64(v)
            print(
                f"{f'{instrument}, {label}':<44s} " + " ".join(f"{r[f]:9.3f}" for f in ACC_FIELDS), flush=True
            )

    def ratio(b: float, a: float) -> str:
        """Guarded, because `still, default pred` is EXACTLY zero and that is a result rather than an error.

        A first version divided unguarded and raised, having already produced every number it needed. The
        zero is the finding: a motionless predator and a motionless prey never meet, so `near_ticks` is 0.000
        and no detection rule is ever reached.
        """
        return "undefined, denominator 0" if a == 0.0 else f"{b / a:.2f}x"

    print("\n  visible ticks per prey, the P6 quantity:", flush=True)
    print(f"    {'':<24s} {'default pred':>14s} {'wander + 4 pred':>17s}   ratio", flush=True)
    for instrument in ("forager", "still"):
        a = cells[(instrument, "default pred")]["visible_ticks"]
        b = cells[(instrument, "p_wander 1.0, n_pred 4")]["visible_ticks"]
        print(f"    {instrument:<24s} {a:14.3f} {b:17.3f}   {ratio(b, a)}", flush=True)

    f_hi = cells[("forager", "p_wander 1.0, n_pred 4")]
    s_hi = cells[("still", "p_wander 1.0, n_pred 4")]
    f_lo = cells[("forager", "default pred")]

    # **The survival confound, which raw visible_ticks hides.** A still prey lives 257 ticks and a foraging
    # one dies at 78.7 under heavy predation, so a raw ratio conflates "more visible per tick" with "alive
    # far longer". These are different mechanisms and only the first is about detectability.
    print("\n  DECOMPOSING the stillness advantage, at fixed predator settings:", flush=True)
    per_tick = (s_hi["visible_ticks"] / s_hi["alive_ticks"]) / (f_hi["visible_ticks"] / f_hi["alive_ticks"])
    survival = s_hi["alive_ticks"] / f_hi["alive_ticks"]
    total = s_hi["visible_ticks"] / f_hi["visible_ticks"]
    print(
        f"    visible per ALIVE tick   forager {f_hi['visible_ticks'] / f_hi['alive_ticks']:.4f}  "
        f"still {s_hi['visible_ticks'] / s_hi['alive_ticks']:.4f}   {per_tick:.2f}x",
        flush=True,
    )
    print(
        f"    alive ticks              forager {f_hi['alive_ticks']:.1f}  "
        f"still {s_hi['alive_ticks']:.1f}   {survival:.2f}x",
        flush=True,
    )
    print(
        f"    product                  {per_tick * survival:.2f}x against the raw ratio {total:.2f}x",
        flush=True,
    )
    print(
        "    so the stillness advantage is PART detectability and PART simply living longer, and the",
        flush=True,
    )
    print(
        "    second is the larger term. Reporting the raw ratio as a detectability effect would be wrong.",
        flush=True,
    )

    print(
        f"\n  attributable to PREDATOR SETTINGS, holding the forager fixed : "
        f"{ratio(f_hi['visible_ticks'], f_lo['visible_ticks'])}",
        flush=True,
    )
    fv = f_hi["visible_ticks"]
    p6_forager = (
        "FALSIFIED, tier three is reachable by configuration"
        if fv >= P6_THRESHOLD
        else f"HOLDS, {P6_THRESHOLD / fv:.1f}x short"
    )
    sv = s_hi["visible_ticks"]
    p6_still = "FALSIFIED" if sv >= P6_THRESHOLD else f"HOLDS, {P6_THRESHOLD / sv:.2f}x short"
    print(f"\n  P6 for a FORAGING policy at the strongest predator setting: {p6_forager}", flush=True)
    print(f"  P6 for a STILL prey at the same setting:                     {p6_still}", flush=True)
    print("\n  If P6 holds for the forager and fails for the still prey, tier three's denominator depends on")
    print("  the behaviour under study being present, so the analysis cannot distinguish 'no freezing' from")
    print("  'too little data to tell'. That is a structural limit, not a tuning problem.")

    # ------------------------------------------------------------ 3. cue where it can actually bite ----
    #
    # **The cue axis against a STILL prey, which is the only instrument that can test it properly.**
    # `damage = DAMAGE * (1 - cue * (1 - moved))`, so for a MOVING prey the cue term vanishes entirely. The
    # forager moves on about 88 percent of its ticks and `still_engaged` is EXACTLY 0.000 at every cue level,
    # so the axis was measured through a mechanism that almost never fires. Against a still prey the damage
    # per contact is 34 / 25.5 / 17 / 8.5 / 0 across the five levels, which is as much leverage as the axis
    # can have.
    print("\n=== the cue axis against a STILL prey, where the cue term actually applies ===", flush=True)
    print(
        "forager rows had still_engaged EXACTLY 0.000 at every level, so cue had almost no mechanism.",
        flush=True,
    )
    print(hdr, flush=True)
    still_cue = []
    for lv in CUE_LEVELS:
        r = run("still", {"cue_dependence": lv, "p_wander": 1.0, "n_pred": 4})
        still_cue.append(r)
        for q, v in r.items():
            payload[f"stillcue_{lv}__{q}"] = np.float64(v)
        print(f"{f'still, cue = {lv}':<44s} " + " ".join(f"{r[f]:9.3f}" for f in ACC_FIELDS), flush=True)

    distinct = len({tuple(round(r[f], 6) for f in ACC_FIELDS) for r in still_cue})
    print(f"\n  distinct behaviours across the five levels: {distinct} of {len(CUE_LEVELS)}", flush=True)
    surv = [r["survived"] for r in still_cue]
    print(f"  survival across the axis: {[round(v, 3) for v in surv]}", flush=True)
    print(f"  monotone in cue: {surv == sorted(surv)}", flush=True)
    print(
        "\n  Five distinct rows here means the axis is genuinely five-valued and the forager collapses were"
    )
    print("  small counts. Fewer means something still merges levels, which would be DEV-031's shape at a")
    print("  finer grain than the three-level liveness test could see.")

    payload["wall_seconds"] = np.float32(time.time() - t0)
    np.savez(sys.argv[1], **payload)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
