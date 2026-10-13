#!/usr/bin/env python3
"""Queue item F: calibrate tier 1 and tier 2 against planted behaviour. Neither ever has been.

**Tier 1 is the CONFIRMATORY claim and has never been shown to detect anything.** The whole result rests
on a survival differential between lineages, measured over 8,192 prey outcomes per generation. Tier 3's
descriptors were calibrated at runs 20, 22 and 25 and three of five failed; tier 1 has had no equivalent
test at all. **If the outcome vector cannot separate policies that differ enormously in behaviour, the
primary claim is unpowered and no amount of sweep fixes it.**

**Tier 2 has never been shown to recover a mode it was given.** Its job is finding behavioural modes
without naming them first. Planting a MIXTURE of known policies and asking whether the clusters correspond
to the plants is the only way to know whether a cluster it reports means anything.

### What is measured, and the branches stated before the numbers

**Tier 1.** Each planted policy's outcome vector against the forager's, with the seed-to-seed spread of the
forager itself as the noise floor. That floor is the honest comparator: a difference smaller than the
variation between two forager runs is not a detection.

    every plant separates            tier 1 is powered, and its advantage over tier 3 is real:
                                     8,192 outcomes against a descriptor's handful of near ticks
    some plants separate             tier 1 is powered for SOME behaviours, and which ones it is blind
                                     to is a disclosure the confirmatory claim has to carry
    nothing separates                the confirmatory claim is unpowered. That is a stop-the-sweep
                                     finding of the same class as run 20

**Tier 2.** A population mixing planted policies in known proportion, run through the real admission and
clustering path, scored by whether the clusters align with the plants.

    clusters align with the plants   tier 2 can recover a mode it was given
    clusters cut across them         a cluster tier 2 reports is not evidence of a behavioural mode,
                                     and tier 2's output is descriptive of the feature space rather
                                     than of behaviour

**Purity is reported rather than a single index**, because a single number would hide the interesting case:
clusters that split one plant in two are a different failure from clusters that merge two plants, and
purity plus the confusion table distinguishes them.

Usage: uv run python scripts/measure/tier_calibration.py <out.npz> [--seeds N]
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
from src.measure import descriptors as D
from src.measure import discovery as DISC
from src.measure import outcomes as OUT

sys.path.insert(0, "scripts/measure")
from planted_policies import _make  # noqa: E402

N_SEEDS = 16
T_HORIZON = 400
POLICIES = ("forager", "freezer", "fleer", "hider", "distancer", "grouper")
DENOM_FLOOR = 1.0


def _static(p, seed):
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
    return E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )


def run_policy(kind: str, p, n_seeds: int):
    """Per-seed outcome vectors and pooled per-prey accumulators for one planted policy."""
    pol = _make(kind)
    pol.params_ref[0] = p
    per_seed, acc_rows = [], []
    for seed in range(n_seeds):
        static = _static(p, seed)
        w = net.init_params(jax.random.key(seed), p)
        state, acc = E.run_episode(
            jax.random.key(1000 + seed), w, static, ent.PRNGSource(jax.random.key(seed)), p, policy=pol
        )
        per_seed.append(OUT.from_state(state).as_vector())
        acc_rows.append({f: np.asarray(getattr(acc, f), float) for f in E.Accumulators._fields})
        jax.clear_caches()
        gc.collect()
    pooled = {f: np.concatenate([r[f] for r in acc_rows]) for f in E.Accumulators._fields}
    return np.stack(per_seed), pooled


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    print(src.memory_headroom(), flush=True)
    t0 = time.time()
    global N_SEEDS
    if "--seeds" in sys.argv:
        N_SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1])

    p = dataclasses.replace(DEFAULT, t_horizon=T_HORIZON)
    check_invariants(p)

    per_seed, pooled = {}, {}
    for kind in POLICIES:
        t1 = time.time()
        per_seed[kind], pooled[kind] = run_policy(kind, p, N_SEEDS)
        print(f"  ran {kind:<10s} in {(time.time() - t1) / 60:4.1f} min", flush=True)

    # ---------------------------------------------------------------- TIER 1
    print("\n" + "=" * 100, flush=True)
    print(
        "TIER 1, THE CONFIRMATORY CLAIM. Outcome vectors, and the forager's own spread is the floor.",
        flush=True,
    )
    print("=" * 100, flush=True)
    f = per_seed["forager"]
    # **The floor is the control's own spread, bounded below by the measurement's granularity.**
    # Two failures are avoided together and they pull opposite ways. A seed spread of exactly zero
    # would divide to infinity or, as the first version did, be special-cased to a ratio of zero,
    # scoring the most obviously real case as no evidence. And a fraction over n_seeds * k prey moves
    # in steps of 1/(n_seeds*k), so a one-prey difference must not read as a huge multiple of a tiny
    # spread. `max(sd, granularity)` is the honest floor.
    granularity = 1.0 / (N_SEEDS * p.k)
    floor = np.maximum(f.std(axis=0), granularity)
    print(
        f"  forager mean      predated {f.mean(0)[0]:.4f}  starved {f.mean(0)[1]:.4f}  "
        f"survived {f.mean(0)[2]:.4f}   n_seeds {N_SEEDS}",
        flush=True,
    )
    print(
        f"  forager seed SD   predated {floor[0]:.4f}  starved {floor[1]:.4f}  "
        f"survived {floor[2]:.4f}   <- floor = max(seed SD, {granularity:.4f} granularity)",
        flush=True,
    )
    print(
        f"\n  {'policy':<10s} {'d_predated':>11s} {'d_starved':>10s} {'d_survived':>11s}   "
        f"{'largest paired |t|':>19s}  verdict",
        flush=True,
    )

    # **PAIRED, per D-045.** Every policy ran on the same seeds with the same layouts, asserted by
    # `test_every_arm_shares_the_layout_at_a_given_condition_and_seed`. The seed-to-seed spread is
    # mostly a property of the layout, and pairing removes it: the unpaired reading of this same data
    # scores the distancer at 0.42 and the paired reading at t = 2.47. The first version of this
    # script compared group means against the UNPAIRED spread, which is the reading D-045 forbids,
    # and printed "TIER 1 SEPARATES NOTHING" on the strength of it.
    #
    # Critical |t| at 5 percent two-sided, n-1 degrees of freedom, from scipy rather than a table
    # lookup or an interpolation. An earlier version hard-coded 2.131 for 16 seeds with an invented
    # fallback for every other count, which is a magic number filling a branch: at 2 seeds the true
    # value is 12.706 and that fallback gave 4.36. scipy is already a dependency of
    # `discovery.cluster`, so the exact value costs nothing.
    from scipy.stats import t as _t

    t_crit = float(_t.ppf(0.975, N_SEEDS - 1))
    separated = []
    for kind in POLICIES:
        if kind == "forager":
            continue
        dpair = per_seed[kind] - f  # (seed, 3), the paired differences
        m = dpair.mean(0)
        sd_p = dpair.std(0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = np.where(sd_p > 0, m / (sd_p / np.sqrt(N_SEEDS)), np.where(np.abs(m) > 0, np.inf, 0.0))
        tmax = float(np.max(np.abs(tstat)))
        ok = tmax >= t_crit
        separated.append((kind, bool(ok), tmax))
        print(
            f"  {kind:<10s} {m[0]:+11.4f} {m[1]:+10.4f} {m[2]:+11.4f}   {tmax:19.2f}  "
            f"{'SEPARATES' if ok else 'not separated'}",
            flush=True,
        )

    n_sep = sum(1 for _, ok, _ in separated if ok)
    print(
        f"\n  {n_sep} of {len(separated)} planted policies separate from the forager, PAIRED, "
        f"at |t| >= {t_crit:.3f}.",
        flush=True,
    )
    if n_sep == len(separated):
        print("  TIER 1 IS POWERED for every planted behaviour.", flush=True)
    elif n_sep == 0:
        print(
            "  TIER 1 SEPARATES NOTHING. The confirmatory claim is unpowered; stop-the-sweep class.",
            flush=True,
        )
    else:
        blind = [k for k, ok, _ in separated if not ok]
        print(
            f"  TIER 1 IS PARTIALLY POWERED. Blind to: {blind}. That is a disclosure the claim carries.",
            flush=True,
        )

    # ---------------------------------------------------------------- TIER 2
    print("\n" + "=" * 100, flush=True)
    print(
        "TIER 2. Plant a MIXTURE, run the real admission and clustering, ask if clusters match plants.",
        flush=True,
    )
    print("=" * 100, flush=True)

    # A mixed population: every prey row from every policy, labelled by which policy produced it.
    mix = [k for k in POLICIES if k != "forager"] + ["forager"]
    accs = {f_: np.concatenate([pooled[k][f_] for k in mix]) for f_ in E.Accumulators._fields}
    labels_true = np.concatenate([np.full(len(pooled[k]["alive_ticks"]), i) for i, k in enumerate(mix)])
    print(f"  mixed population: {len(labels_true)} prey rows across {len(mix)} planted policies", flush=True)

    desc = D.population(accs, denominator_floor=DENOM_FLOOR)
    dv = {name: desc[name] for name in D.NAMED_STRATEGIES.values() if name in desc}
    rows = DISC.admit(accs, dv)
    accepted = [r.name for r in rows if r.admitted]
    print(f"  admitted features: {accepted}", flush=True)
    for r in rows:
        print(
            f"    {r.name:<20s} admitted={str(r.admitted):<5s} r2_taxonomy={r.r2_taxonomy:.4f}  {r.reason}",
            flush=True,
        )

    if len(accepted) < 2:
        print(
            "\n  FEWER THAN TWO FEATURES ADMITTED. Clustering is not defined; tier 2 cannot run here.",
            flush=True,
        )
        names, X, labels, k_best, sil = [], None, None, 0, float("nan")
    else:
        names, X = DISC.accepted_matrix(rows, accs)
        keep = np.all(np.isfinite(X), axis=1)
        X, lt = X[keep], labels_true[keep]
        labels, k_best, sil = DISC.cluster(X, key_seed=0)
        print(
            f"\n  clustered {len(X)} complete rows on {len(names)} features -> k={k_best}, "
            f"silhouette {sil:.4f}",
            flush=True,
        )

        print("\n  CONFUSION, rows are planted policies, columns are clusters:", flush=True)
        header = "  " + " " * 12 + "".join(f"{c:>8d}" for c in range(k_best))
        print(header, flush=True)
        purity_num = 0
        for i, k in enumerate(mix):
            row = [int(((lt == i) & (labels == c)).sum()) for c in range(k_best)]
            print("  " + f"{k:<12s}" + "".join(f"{v:8d}" for v in row), flush=True)
        for c in range(k_best):
            in_c = labels == c
            if in_c.sum():
                purity_num += max(int(((lt == i) & in_c).sum()) for i in range(len(mix)))
        purity = purity_num / len(lt)
        print(
            f"\n  cluster purity {purity:.4f}   against {1.0 / len(mix):.4f} for random assignment",
            flush=True,
        )
        if purity >= 0.5:
            print("  TIER 2 RECOVERS the planted modes above chance.", flush=True)
        else:
            print(
                "  TIER 2 DOES NOT RECOVER the planted modes: clusters cut across the plants, so a",
                flush=True,
            )
            print(
                "  cluster it reports is descriptive of the feature space rather than of behaviour.",
                flush=True,
            )

    np.savez(
        sys.argv[1],
        per_seed=np.stack([per_seed[k] for k in POLICIES]),
        policies=np.array(POLICIES),
        floor=floor,
        separated=np.array([[k, str(ok), f"{r:.4f}"] for k, ok, r in separated]),
        accepted=np.array(accepted),
        # Saved so tier 2's clustering can be re-examined without a 17 minute re-run. Their absence
        # is why the first run's 761 against 7 split could not be checked.
        tier2_X=X if X is not None else np.zeros((0, 0)),
        tier2_labels=labels if labels is not None else np.zeros(0, np.int32),
        tier2_labels_true=lt if X is not None else np.zeros(0, np.int32),
        tier2_feature_names=np.array(names),
        tier2_k=np.int32(k_best),
        tier2_silhouette=np.float64(sil),
        n_seeds=np.int32(N_SEEDS),
        wall_seconds=np.float32(time.time() - t0),
    )
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
