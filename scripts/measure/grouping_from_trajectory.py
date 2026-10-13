#!/usr/bin/env python3
"""Is grouping's payoff MEMBERSHIP or POSITIONING? Computed from the trajectory dump, no environment change.

**Owner's ruling, 2026-08-27: try the cheaper path first.** The proposed fix was a new accumulator, distance
summed over visible unengaged ticks. The trajectory dump already carries per-tick positions for prey and
predators, D-034 having made its schema renderer-complete, so the same quantity is computable directly with
no environment change, no oracle change, and no prediction-before-change ceremony.

**Its weakness is power and that is checked rather than assumed.** The dump is a small, differently-selected
sample, so the sample size is reported first and every verdict is stated against it.

### What went wrong with the accumulator-based attempt, since this exists to avoid repeating it

`dist_sum_visible / visible_ticks` was contaminated by the outcome it predicted:

    mean distance, PREDATED and seen    0.4804      89.7 percent sit at <= 1.05
    mean distance, SURVIVED and seen    1.1616

Capture takes several adjacent ticks; those are visible ticks at distance 1; `visible_ticks` is small; so the
capture window is essentially the whole mean. `corr(distance, predated)` was close to measuring whether the
prey was caught.

**And the confound ran TOWARD the printed conclusion, which is the dangerous direction.** The density
confound ran away from its finding and so strengthened it. This one would have manufactured the result. The
general rule, which the owner asked to be recorded: establish a confound's DIRECTION before deciding whether
it matters. Same direction as the finding leaves the finding unsupported; opposite direction makes it a lower
bound.

### The measure used here, and why it cannot be contaminated the same way

Position is the mean Chebyshev distance to the nearest LIVING predator, over ticks where the prey is alive
AND **strictly outside contact range**. A capture tick is a contact tick by definition, so excluding them
removes the mechanism that produced the contamination.

Two exclusions are reported side by side, because the second is the stricter reading and the pair says
whether the answer depends on where the line is drawn:

    contact-excluded    distance > 1, so no tick of any capture sequence is counted
    endgame-excluded    additionally drops each prey's final 15 alive ticks, which removes the APPROACH
                        as well as the contact

### The question, and both branches stated before the number

    density predicts risk ACROSS bands, position does not predict WITHIN them
      -> MEMBERSHIP. Grouping is a free lunch and nearest targeting has not made it a skill
    position predicts risk WITHIN bands
      -> POSITIONING, which is what the rule was chosen to create

**Nearest targeting does not remove the dilution.** In a group of 8 you are chosen 1 time in 8 under random
targeting, and when you are nearest under nearest targeting, which absent any positioning skill is also about
1 time in 8. It makes the dilution depend on position rather than be automatic. That claim was asserted as a
property of the rule for several sessions and is corrected at `configs/provenance.py`.

Usage: uv run python scripts/measure/grouping_from_trajectory.py <out.npz> [--seeds N]
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

N_SEEDS = 48
T_HORIZON = 400
ENDGAME_TICKS = 15


def policy_forager(params):
    def policy(weights, obs, hidden, step_index):
        del weights
        return jax.vmap(lambda o: baseline.forager(o, params, step_index))(obs), hidden

    return policy


def one_episode(seed: int, p) -> dict:
    """Per-prey density, two position measures, and the outcome, from the recorded trajectory."""
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
    static = E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )
    w = net.init_params(jax.random.key(seed), p)
    state, _acc, traj = E.run_episode(
        jax.random.key(1000 + seed),
        w,
        static,
        ent.PRNGSource(jax.random.key(seed)),
        p,
        record=True,
        policy=policy_forager(p),
    )

    prey = np.asarray(traj.prey_pos)  # (T, k, 2)
    alive = np.asarray(traj.prey_alive)  # (T, k)
    pred = np.asarray(traj.pred_pos)  # (T, n_pred, 2)

    # Chebyshev distance from every prey to the nearest predator, per tick.
    diff = np.abs(prey[:, :, None, :] - pred[:, None, :, :])  # (T, k, n_pred, 2)
    dist = diff.max(axis=-1).min(axis=-1).astype(np.float64)  # (T, k)

    # Living neighbours within r_group, per tick. The grouping exposure, computed here rather than taken
    # from `density_sum` so both quantities come from the same source and cannot disagree.
    pd = np.abs(prey[:, :, None, :] - prey[:, None, :, :]).max(axis=-1)  # (T, k, k)
    near = (pd <= p.r_group) & alive[:, None, :] & alive[:, :, None]
    np.einsum("tii->ti", near)[...] = False  # a prey is not its own neighbour
    density = near.sum(axis=-1).astype(np.float64)  # (T, k)

    n_alive = alive.sum(axis=0).astype(np.float64)  # (k,)
    mean_density = np.where(n_alive > 0, (density * alive).sum(axis=0) / np.maximum(n_alive, 1), np.nan)

    contact_ok = alive & (dist > 1.0)
    n_contact = contact_ok.sum(axis=0)
    pos_contact = np.where(n_contact > 0, (dist * contact_ok).sum(axis=0) / np.maximum(n_contact, 1), np.nan)

    # Drop each prey's final ENDGAME_TICKS alive ticks, which removes the approach as well as the contact.
    last = np.where(alive.any(axis=0), alive.shape[0] - 1 - alive[::-1].argmax(axis=0), -1)
    tix = np.arange(alive.shape[0])[:, None]
    endgame_ok = contact_ok & (tix < (last[None, :] - ENDGAME_TICKS))
    n_end = endgame_ok.sum(axis=0)
    pos_endgame = np.where(n_end > 0, (dist * endgame_ok).sum(axis=0) / np.maximum(n_end, 1), np.nan)

    return {
        "density": mean_density,
        "pos_contact": pos_contact,
        "pos_endgame": pos_endgame,
        "predated": np.asarray(state.prey_died_predated).astype(np.float64),
        "n_contact": n_contact.astype(np.float64),
        "n_endgame": n_end.astype(np.float64),
    }


def banded(d, pos, y, label):
    """corr(position, predated) within density terciles. Both branches were stated in the docstring."""
    ok = np.isfinite(pos) & np.isfinite(d)
    print(f"\n  {label}: {int(ok.sum())} of {len(d)} prey have a defined position", flush=True)
    if ok.sum() < 24:
        print("    too few to band. Not a verdict.", flush=True)
        return None
    dd, pp, yy = d[ok], pos[ok], y[ok]
    edges = [dd.min() - 1e-9, np.quantile(dd, 1 / 3), np.quantile(dd, 2 / 3), dd.max() + 1e-9]
    print(f"    {'band':<20s} {'n':>5s} {'predated':>9s} {'corr(pos, predated)':>21s}", flush=True)
    rs = []
    for i in range(3):
        m = (dd > edges[i]) & (dd <= edges[i + 1])
        n = int(m.sum())
        if n >= 12 and pp[m].std() > 0 and yy[m].std() > 0:
            r = float(np.corrcoef(pp[m], yy[m])[0, 1])
            rs.append(r)
            shown = f"{r:+.4f}"
        else:
            shown = "degenerate"
        rate = float(yy[m].mean()) if n else float("nan")
        print(
            f"    {f'{edges[i]:.2f} to {edges[i + 1]:.2f}':<20s} {n:5d} {rate:9.4f} {shown:>21s}", flush=True
        )
    if not rs:
        return None
    mean_r = float(np.mean(rs))
    signs = "consistent" if all(np.sign(x) == np.sign(rs[0]) for x in rs) else "INCONSISTENT across bands"
    print(f"    mean within-band corr {mean_r:+.4f}, signs {signs}", flush=True)
    return mean_r, signs


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()
    global N_SEEDS
    if "--seeds" in sys.argv:
        N_SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1])

    p = dataclasses.replace(DEFAULT, t_horizon=T_HORIZON)
    check_invariants(p)
    print(f"\n{N_SEEDS} recorded episodes, k = {p.k}, so {N_SEEDS * p.k} prey.", flush=True)
    print(
        "The dump is a small, differently-selected sample; the count is reported before any verdict.\n",
        flush=True,
    )

    rows = []
    for seed in range(N_SEEDS):
        rows.append(one_episode(seed, p))
        if (seed + 1) % 16 == 0:
            jax.clear_caches()
            gc.collect()
            print(f"  {seed + 1}/{N_SEEDS} episodes", flush=True)

    cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    d, y = cat["density"], cat["predated"]

    print("\n" + "=" * 92, flush=True)
    print("THE OVERALL RELATION, for comparison with the accumulator-based measurement", flush=True)
    print("=" * 92, flush=True)
    r_all = float(np.corrcoef(d, y)[0, 1]) if d.std() > 0 else float("nan")
    print(f"  n = {len(d)} prey, corr(density, predated) = {r_all:+.4f}", flush=True)
    lo, hi = np.quantile(d, 0.25), np.quantile(d, 0.75)
    print(f"  predation, LOWEST density quartile:  {y[d <= lo].mean():.4f}", flush=True)
    print(f"  predation, HIGHEST density quartile: {y[d >= hi].mean():.4f}", flush=True)

    print("\n" + "=" * 92, flush=True)
    print("IS THE NEW POSITION MEASURE STILL CONTAMINATED? Checked before it is used.", flush=True)
    print("=" * 92, flush=True)
    for name in ("pos_contact", "pos_endgame"):
        v = cat[name]
        ok = np.isfinite(v)
        pm, sm = ok & (y > 0), ok & (y == 0)
        print(
            f"  {name:<12s} predated {v[pm].mean():6.3f} (n={int(pm.sum()):4d})   "
            f"survived {v[sm].mean():6.3f} (n={int(sm.sum()):4d})   "
            f"at <=1.05: {float((v[pm] <= 1.05).mean()):.3f}",
            flush=True,
        )
    print(
        "\n  The accumulator measure gave 0.480 against 1.162 with 89.7 percent at the floor. A measure",
        flush=True,
    )
    print("  that still pins predated prey at the adjacency floor has not fixed anything.", flush=True)

    print("\n" + "=" * 92, flush=True)
    print("MEMBERSHIP OR POSITIONING", flush=True)
    print("=" * 92, flush=True)
    a = banded(d, cat["pos_contact"], y, "contact-excluded")
    b = banded(d, cat["pos_endgame"], y, f"endgame-excluded, final {ENDGAME_TICKS} alive ticks dropped")

    print("\n  VERDICT", flush=True)
    for label, res in (("contact-excluded", a), ("endgame-excluded", b)):
        if res is None:
            print(f"    {label}: not computable at this n. Not a verdict.", flush=True)
            continue
        mean_r, signs = res
        if signs.startswith("INCONSISTENT"):
            print(f"    {label}: signs disagree across bands. NOT a verdict at this n.", flush=True)
        elif abs(mean_r) < 0.10 <= abs(r_all):
            print(
                f"    {label}: MEMBERSHIP. Density predicts risk, position within a band does not.",
                flush=True,
            )
        elif abs(mean_r) >= 0.10:
            print(f"    {label}: POSITIONING contributes, mean within-band corr {mean_r:+.4f}.", flush=True)
        else:
            print(f"    {label}: neither clear. Not a verdict.", flush=True)

    payload = {k: v for k, v in cat.items()}
    payload["n_seeds"] = np.int32(N_SEEDS)
    payload["corr_overall"] = np.float64(r_all)
    payload["wall_seconds"] = np.float32(time.time() - t0)
    np.savez(sys.argv[1], **payload)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
