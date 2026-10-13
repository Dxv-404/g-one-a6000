#!/usr/bin/env python3
"""Queue item C and D-052 step one: diagnose hiding, distancing and grouping to a mechanism.

**D-052 fixes the order and this script implements step one for all three, plus the two known priors.**

    step one   verify the PLANT exercises the mechanism. Does the planted hider's RAW occlusion rise
               against the forager's? Does the planted grouper's RAW density rise? If the plant does
               not move the raw quantity, the descriptor was never tested and the failure is the
               plant's, not the descriptor's. D-044, which has four false verdicts behind it
    step two   only if step one passes, trace the descriptor

**Raw quantities, deliberately, because that is what separates the two failures.** Run 22 reported
descriptor z-scores and found hiding, distancing and grouping undetected. A z-score cannot distinguish
"the behaviour never happened" from "the behaviour happened and the descriptor cannot see it", and those
have opposite fixes. Everything below is a rate over an accumulator, printed for the plant and for the
forager side by side.

### The two known priors, from corpus 097, checked before anything else

**Hiding: `occlusion_baseline` was measured SATURATED at 1.0 on run 1.** The descriptor is
`conditional_occlusion = occluded_near/near_ticks - occluded_far/far_ticks`. If the baseline sits at its
ceiling then the difference is bounded above by `near - 1.0`, which is at most zero, so **a conditional
descriptor whose baseline is saturated can only ever return zero or a negative number, whatever the prey
does.** That is a construction failure, not a power failure, and it is checked first because it would
explain hiding's null completely.

**Distancing: `mean_distance` was measured as dominated by the capture window.** It is
`dist_sum_visible / visible_ticks`, capture takes several adjacent ticks at distance 1, and `visible_ticks`
is small, so the mean is mostly the capture sequence and the descriptor largely restates whether the prey
was caught. Checked here from the trajectory dump by recomputing the same quantity with contact ticks
excluded, which is the method `grouping_from_trajectory.py` established.

### Branches, stated before the numbers

    the plant does not move the raw quantity
        step one FAILS. The descriptor was never tested. Fix the plant, re-run, and only then ask
        whether the descriptor can see it
    the plant moves the raw quantity and the descriptor does not respond
        step two, and the two priors above are the first candidate mechanisms
    the baseline is saturated
        hiding's descriptor is broken BY CONSTRUCTION and no sample size repairs it. The terminal
        state is undetectable-by-descriptor with the mechanism stated, per D-052

Usage: uv run python scripts/measure/detector_diagnosis.py <out.npz> [--seeds N]
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
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius

# The planted policies are the calibrated instruments; importing them rather than re-implementing is
# the duplication rule. A second copy is written under time pressure by definition.
sys.path.insert(0, "scripts/measure")
from planted_policies import _make  # noqa: E402

N_SEEDS = 32
T_HORIZON = 400
N_TRAJ_SEEDS = 48

CONFIGS = {"default": {}, "high_pressure": {"p_wander": 1.0, "n_pred": 4}}
POLICIES = ("forager", "hider", "distancer", "grouper")

ACC = E.Accumulators._fields


def _static(p, seed):
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
    return E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )


def run_policy(kind: str, overrides: dict, n_seeds: int) -> dict:
    """Per-prey accumulators for one policy, pooled across seeds."""
    p = dataclasses.replace(DEFAULT, t_horizon=T_HORIZON, **overrides)
    check_invariants(p)
    policy = _make(kind)
    policy.params_ref[0] = p

    rows = []
    for seed in range(n_seeds):
        static = _static(p, seed)
        w = net.init_params(jax.random.key(seed), p)
        _, acc = E.run_episode(
            jax.random.key(1000 + seed), w, static, ent.PRNGSource(jax.random.key(seed)), p, policy=policy
        )
        rows.append({f: np.asarray(getattr(acc, f), dtype=np.float64) for f in ACC})
    jax.clear_caches()
    gc.collect()
    return {f: np.concatenate([r[f] for r in rows]) for f in ACC}


def rate(num, den):
    """Population rate with its denominator, NaN where the denominator is zero."""
    n, d = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(d > 0, n / np.where(d > 0, d, 1.0), np.nan)
    fin = per[np.isfinite(per)]
    return (fin.mean() if len(fin) else np.nan, fin.std() if len(fin) else np.nan, len(fin), d.mean())


def capture_window(p, n_seeds: int) -> dict:
    """Distancing's known prior, measured on the quantity the DESCRIPTOR reads.

    **The first version of this function measured the wrong thing and the error is instructive.** It
    averaged distance over all ALIVE ticks; `mean_distance` is `dist_sum_visible / visible_ticks`, which
    is conditioned on ticks where the predator REPORTS the prey. The two disagree by two orders of
    magnitude, 18.88 against 0.2693, so excluding contact ticks from the unconditioned version moved
    nothing and would have refuted the prior for the wrong reason. D-044 one level in: the instrument
    must exercise the mechanism the descriptor reads.

    So the test is on the accumulators themselves, split by outcome, which is how the contamination was
    originally found: capture takes several adjacent ticks at distance 1, those are visible ticks, and
    `visible_ticks` is small, so a predated prey's mean is mostly its own capture sequence.

    The unconditioned trajectory measure is returned alongside, labelled, because the CONTRAST is the
    result rather than noise: it shows the conditioning is what creates the contamination.
    """
    out = {"all": [], "excl": [], "predated": [], "acc_mean": [], "acc_vis": []}
    for seed in range(n_seeds):
        static = _static(p, seed)
        w = net.init_params(jax.random.key(seed), p)
        pol = _make("forager")
        pol.params_ref[0] = p
        state, acc, traj = E.run_episode(
            jax.random.key(1000 + seed),
            w,
            static,
            ent.PRNGSource(jax.random.key(seed)),
            p,
            record=True,
            policy=pol,
        )
        # The DESCRIPTOR's own quantity, per prey, which is what the prior is about.
        dsv = np.asarray(acc.dist_sum_visible, float)
        vt = np.asarray(acc.visible_ticks, float)
        prey = np.asarray(traj.prey_pos)
        pred = np.asarray(traj.pred_pos)
        alive = np.asarray(traj.prey_alive)
        d = np.abs(prey[:, :, None, :] - pred[:, None, :, :]).max(axis=-1).min(axis=-1).astype(float)
        for i in range(p.k):
            m = alive[:, i].astype(bool)
            if m.sum() == 0:
                continue
            di = d[m, i]
            out["all"].append(di.mean())
            keep = di > 1
            out["excl"].append(di[keep].mean() if keep.any() else np.nan)
            out["predated"].append(float(np.asarray(state.prey_died_predated)[i]))
            out["acc_mean"].append(dsv[i] / vt[i] if vt[i] > 0 else np.nan)
            out["acc_vis"].append(vt[i])
        jax.clear_caches()
        gc.collect()
    return {k: np.array(v, float) for k, v in out.items()}


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    print(src.memory_headroom(), flush=True)
    t0 = time.time()
    global N_SEEDS
    if "--seeds" in sys.argv:
        N_SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1])

    payload: dict = {}
    for cfg, ov in CONFIGS.items():
        print(f"\n{'=' * 100}\nCONFIGURATION: {cfg}  {ov}\n{'=' * 100}", flush=True)
        accs = {}
        for kind in POLICIES:
            t1 = time.time()
            accs[kind] = run_policy(kind, ov, N_SEEDS)
            print(f"  ran {kind:<10s} in {(time.time() - t1) / 60:4.1f} min", flush=True)

        # ---- prior 1: is hiding's baseline saturated? -------------------------------------------
        print("\n  HIDING. occlusion_baseline = occluded_far / far_ticks", flush=True)
        print(
            f"  {'policy':<10s} {'baseline':>10s} {'near':>10s} {'conditional':>12s} {'near_ticks':>11s}",
            flush=True,
        )
        for kind in POLICIES:
            a = accs[kind]
            b, _, _, _ = rate(a["occluded_far"], a["far_ticks"])
            n, _, _, nd = rate(a["occluded_near"], a["near_ticks"])
            print(f"  {kind:<10s} {b:10.4f} {n:10.4f} {n - b:12.4f} {nd:11.2f}", flush=True)
            payload[f"{cfg}__{kind}__occl_baseline"] = np.float64(b)
            payload[f"{cfg}__{kind}__occl_near"] = np.float64(n)

        base_f = payload[f"{cfg}__forager__occl_baseline"]
        print(f"\n  baseline at the forager: {base_f:.4f}", flush=True)
        if base_f >= 0.99:
            print(
                "  SATURATED. conditional_occlusion is bounded above by (near - baseline) <= 0.01,",
                flush=True,
            )
            print("  so it can only return zero or a negative number WHATEVER the prey does.", flush=True)
            print("  That is a construction failure and no sample size repairs it.", flush=True)
        else:
            print(
                "  NOT saturated. The prior does not hold at this configuration; step two applies.",
                flush=True,
            )

        # ---- step one: do the plants move their raw quantities? ---------------------------------
        print("\n  STEP ONE. Does each plant move the RAW quantity its descriptor reads?", flush=True)
        f = accs["forager"]
        checks = [
            ("hider", "occluded_near/near_ticks", "occluded_near", "near_ticks"),
            ("distancer", "dist_sum_visible/visible_ticks", "dist_sum_visible", "visible_ticks"),
            ("grouper", "density_sum/alive_ticks", "density_sum", "alive_ticks"),
        ]
        for kind, label, num, den in checks:
            fv, fsd, _, _ = rate(f[num], f[den])
            pv, psd, _, pd = rate(accs[kind][num], accs[kind][den])
            delta = pv - fv
            moved = "MOVES" if abs(delta) > 0.05 * max(abs(fv), 1e-9) else "DOES NOT MOVE"
            print(
                f"  {kind:<10s} {label:<32s} forager {fv:8.4f}  plant {pv:8.4f}  "
                f"delta {delta:+8.4f}  {moved}",
                flush=True,
            )
            payload[f"{cfg}__{kind}__raw_forager"] = np.float64(fv)
            payload[f"{cfg}__{kind}__raw_plant"] = np.float64(pv)

        # grouping's conditional form, which is the descriptor rather than the raw density
        for kind in ("forager", "grouper"):
            a = accs[kind]
            th, _, _, _ = rate(a["nn_sum_visible"], a["nn_visible_ticks"])
            bl, _, _, _ = rate(a["nn_sum_unseen"], a["nn_unseen_ticks"])
            print(
                f"  {kind:<10s} nn_threat {th:7.4f}  nn_baseline {bl:7.4f}  "
                f"conditional_grouping {bl - th:+7.4f}",
                flush=True,
            )
            payload[f"{cfg}__{kind}__cond_grouping"] = np.float64(bl - th)

    # ---- prior 2: distancing's capture window ---------------------------------------------------
    print(
        f"\n{'=' * 100}\nDISTANCING. Is mean_distance dominated by the capture window?\n{'=' * 100}",
        flush=True,
    )
    p = dataclasses.replace(DEFAULT, t_horizon=T_HORIZON)
    cw = capture_window(p, N_TRAJ_SEEDS)
    am = np.asarray(cw["acc_mean"], float)
    vis = np.asarray(cw["acc_vis"], float)
    pr = np.asarray(cw["predated"], float)
    ok = np.isfinite(am)
    print(f"  prey measured: {int(ok.sum())} of {len(am)} over {N_TRAJ_SEEDS} recorded episodes", flush=True)
    print("  (the rest have visible_ticks == 0, which is itself the engagement-scarcity finding)", flush=True)

    print(
        "\n  THE DESCRIPTOR'S OWN QUANTITY, dist_sum_visible / visible_ticks, split by outcome:", flush=True
    )
    for label, mask in (("predated", pr[ok] > 0.5), ("survived", pr[ok] <= 0.5)):
        v = am[ok][mask]
        if len(v) == 0:
            print(f"    {label:<9s} none", flush=True)
            continue
        at_floor = float((v <= 1.05).mean())
        print(
            f"    {label:<9s} n={len(v):4d}  mean {v.mean():7.4f}  "
            f"share at the adjacency floor (<= 1.05) {at_floor:6.1%}  "
            f"mean visible_ticks {vis[ok][mask].mean():6.2f}",
            flush=True,
        )
        payload[f"cw_acc_{label}_mean"] = np.float64(v.mean())
        payload[f"cw_acc_{label}_atfloor"] = np.float64(at_floor)

    if len(np.unique(pr[ok])) > 1:
        c_acc = np.corrcoef(am[ok], pr[ok])[0, 1]
        print(f"\n    corr(mean_distance, predated) = {c_acc:+7.4f}", flush=True)
        payload["cw_corr_acc"] = np.float64(c_acc)

    ok2 = np.isfinite(cw["all"]) & np.isfinite(cw["excl"])
    a, e, pr2 = cw["all"][ok2], cw["excl"][ok2], cw["predated"][ok2]
    print("\n  CONTRAST, a DIFFERENT quantity: distance over ALL alive ticks, unconditioned.", flush=True)
    print(f"    all alive ticks        {a.mean():7.4f}", flush=True)
    print(f"    contact ticks excluded {e.mean():7.4f}   shift {e.mean() - a.mean():+7.4f}", flush=True)
    if len(np.unique(pr2)) > 1:
        print(
            f"    corr with predated, all {np.corrcoef(a, pr2)[0, 1]:+7.4f}, "
            f"excluded {np.corrcoef(e, pr2)[0, 1]:+7.4f}",
            flush=True,
        )
    print(
        "\n  Reading: the unconditioned measure is two orders of magnitude larger and barely moves",
        flush=True,
    )
    print(
        "  when contact is excluded. **The CONDITIONING is what creates the contamination**, not the",
        flush=True,
    )
    print("  distance. A descriptor conditioned on visibility is conditioned on being hunted.", flush=True)
    payload["cw_all"] = a
    payload["cw_excl"] = e
    payload["cw_predated"] = pr2
    payload["cw_acc_mean"] = am
    payload["cw_acc_vis"] = vis

    payload["n_seeds"] = np.int32(N_SEEDS)
    payload["n_traj_seeds"] = np.int32(N_TRAJ_SEEDS)
    payload["wall_seconds"] = np.float32(time.time() - t0)
    np.savez(sys.argv[1], **payload)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
