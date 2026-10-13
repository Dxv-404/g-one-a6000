#!/usr/bin/env python3
"""Spawn feasibility at ladder heads 10, 14 and 18, and the `r_pred` range a wrong claim truncated. Q-028.

**Heads 14 and 18 are `r_pred` 6 and 8**, because the ladder is derived rather than literal:

    floor = r_pred + 1          head = 2 * r_pred + 2

so `r_pred` 4 / 6 / 8 give heads 10 / 14 / 18. Asking about a head is asking about a sensing radius.

**What `check_invariants` says about each, which is not what a first pass assumed.**

    r_pred 4   head 10   VALID
    r_pred 5   head 12   VALID
    r_pred 6   head 14   VALID
    r_pred 8   head 18   REJECTED, and NOT on spawn feasibility

The rejection at 8 is `r_pred <= r + 2`, whose own docstring gives the reason: expected prey-in-view runs
0.48 at r_pred 4, 0.99 at 6 and 1.70 at 8, so at 8 the predator always has a target, distancing is excluded
by construction, and `p_wander` dies entirely because it is conditional on having no target. **The ceiling is
where the axis stops producing distinguishable conditions and starts killing another axis.**

So head 18 is measured here for its GEOMETRY only, by constructing the ladder directly, and the result is a
fact about whether a 40 by 40 grid can seat a separation of 18 rather than a proposal to sweep there.

**A wrong claim this corrects.** An earlier comment in `axis_liveness_full.py`, repeated into the log, into
Q-023's answer and into a commit message, said `r_pred` could not exceed 4 because `min(d_spawn_relax)` is 5
and must exceed it. `d_spawn_relax` is derived, `floor = r_pred + 1`, so that assertion is a tautology and
can never bind. The bound was reasoned and never executed. `r_pred` 5 and 6 are covered here.

**What is measured.** `spawn_relax_level` is recorded per episode at reset: the index into the ladder of the
separation actually achieved, or -1 if EVERY level failed. Level 0 means the nominal separation was met.
A configuration that relaxes often sits outside the valid envelope and is not comparable to one that does
not, which is why M4 reports the rate per configuration.

Usage: uv run python scripts/measure/spawn_feasibility.py <out.npz>
"""

from __future__ import annotations

import dataclasses
import sys
import time

import jax
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import DEFAULT, check_invariants
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius

#: Many, because this is a rate over a discrete outcome and the interesting values are near zero. 256 resets
#: per cell gives a standard error of about 3 percent at p = 0.5 and much less near 0 or 1.
N_RESETS = 256

#: `r_pred` 8 is included deliberately even though `check_invariants` rejects it, because the QUESTION is
#: about spawn geometry at head 18 and the rejection is on an unrelated bound. Flagged in the output rather
#: than silently run, so the row cannot be mistaken for a permitted configuration.
RADII = (4, 5, 6, 8)

#: Grid sizes. 40 is the design value; the smaller ones are where the provenance table records reset failure
#: rising, so they say whether head 14 or 18 is feasible at all or only feasible at 40.
GRIDS = (40, 32, 24)


def measure(r_pred: int, side: int, k: int) -> dict:
    """Relaxation level distribution over `N_RESETS` independent resets."""
    p = dataclasses.replace(DEFAULT, r_pred=r_pred, h=side, w=side, k=k)
    permitted = True
    try:
        check_invariants(p)
    except AssertionError:
        permitted = False

    r_table = required_radius(p)
    levels = []
    for seed in range(N_RESETS):
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, r_table), food_src=fs, target_src=ts)
        state = E.reset(ent.PRNGSource(jax.random.key(10_000 + seed)), static, p)
        levels.append(int(state.spawn_relax_level))

    lv = np.array(levels)
    n_levels = len(p.d_spawn_relax)
    return {
        "r_pred": r_pred,
        "side": side,
        "k": k,
        "head": p.d_spawn_relax[0],
        "ladder": p.d_spawn_relax,
        "permitted": permitted,
        "nominal_rate": float((lv == 0).mean()),
        "relaxed_rate": float((lv > 0).mean()),
        "failed_rate": float((lv < 0).mean()),
        "mean_level": float(lv[lv >= 0].mean()) if (lv >= 0).any() else float("nan"),
        "hist": [int((lv == i).sum()) for i in range(n_levels)] + [int((lv < 0).sum())],
    }


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()
    payload: dict = {}

    print(f"\n{N_RESETS} independent resets per cell. Level 0 means the NOMINAL separation was achieved.")
    print(
        "A negative level means every rung failed, which is the silent-failure case the ladder exists for.\n"
    )
    print(
        f"{'r_pred':>7s} {'head':>5s} {'grid':>5s} {'k':>3s} {'ladder':>20s} {'permitted':>10s} "
        f"{'nominal':>8s} {'relaxed':>8s} {'failed':>7s} {'mean lv':>8s}",
        flush=True,
    )

    rows = []
    for side in GRIDS:
        for r_pred in RADII:
            k = DEFAULT.k
            r = measure(r_pred, side, k)
            rows.append(r)
            key = f"r{r_pred}_g{side}"
            for q in ("nominal_rate", "relaxed_rate", "failed_rate", "mean_level"):
                payload[f"{key}__{q}"] = np.float64(r[q])
            payload[f"{key}__hist"] = np.array(r["hist"], np.int32)
            print(
                f"{r['r_pred']:7d} {r['head']:5d} {r['side']:5d} {r['k']:3d} {str(r['ladder']):>20s} "
                f"{('yes' if r['permitted'] else 'NO'):>10s} {r['nominal_rate']:8.3f} "
                f"{r['relaxed_rate']:8.3f} {r['failed_rate']:7.3f} {r['mean_level']:8.2f}",
                flush=True,
            )
        print(flush=True)

    print("=" * 92)
    at40 = {r["r_pred"]: r for r in rows if r["side"] == 40}
    print("At the design grid of 40 by 40:")
    for rp in RADII:
        r = at40[rp]
        verdict = (
            "never relaxes"
            if r["relaxed_rate"] == 0.0 and r["failed_rate"] == 0.0
            else f"relaxes {r['relaxed_rate'] * 100:.1f}%, fails {r['failed_rate'] * 100:.1f}%"
        )
        note = "" if r["permitted"] else "   (NOT permitted by check_invariants: r_pred <= r + 2)"
        print(f"  head {r['head']:2d}, r_pred {rp}: {verdict}{note}")
    print("\nSpawn feasibility is a GEOMETRIC question and is answered above. Whether an r_pred is worth")
    print("sweeping is a different question, bounded at 6 by prey-in-view reaching 0.99. Q-028.")
    print("=" * 92)

    payload["wall_seconds"] = np.float32(time.time() - t0)
    np.savez(sys.argv[1], **payload)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
