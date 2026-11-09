#!/usr/bin/env python3
"""S4 over the long run, from its own in-run measurements (D-087), with D-083's rule at every measured point.

The long run measures every genome of every arm with a predator over 16 episodes after generations 100, 300, 500,
750, 1,000, 1,250 and 1,500 (`sweep.S4_AFTER`). This reads those arrays from every condition checkpoint, in every
shard folder, and reports per condition and point:

    coverage            share of genomes with at least n_eng_min decided engagements
    median learnability of the defined genomes
    passes              D-083's rule at that point: coverage >= 0.5 AND median >= 0.16, every seed
    pooled              the pooled per-seed value, labelled NOT learnability (Q-047), beside it

through `src.measure.learnability.s4`, the one implementation the calibration and run 52 used. It adds no verdict of
its own: whether learnability stays in band is read off the trajectory, and any rule for "sustained" beyond D-083's
per-point rule is the owner's to fix before the long run's results are read.

It reads nothing but the S4 arrays: no descriptor, no tier statistic, so it can be run on the long run's output
before the thresholds are frozen without reading any treatment behaviour.

    uv run python scripts/measure/longrun_s4.py runs/longrun [--out runs/longrun-s4.json]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.default import DEFAULT  # noqa: E402
from src.measure import learnability  # noqa: E402

#: D-083. The same constants the calibration applies, read from the same places.
COVERAGE_FLOOR = 0.5
N_ENG_MIN = DEFAULT.n_eng_min
FLOOR = DEFAULT.learnability_floor


def point_records(eng: np.ndarray) -> list[dict]:
    """eng (seeds, G, E, F) at one point -> one S4 record per seed."""
    return [learnability.s4(eng[s][None], N_ENG_MIN, FLOOR) for s in range(eng.shape[0])]


def passes(rec: dict) -> bool:
    med = rec["median_learnability"]
    return bool(rec["coverage"] >= COVERAGE_FLOOR and not math.isnan(med) and med >= FLOOR)


def condition_summary(path: pathlib.Path) -> dict | None:
    """None for a condition without in-run S4 (the null arm, whose world has no predator)."""
    with np.load(path, allow_pickle=True) as d:
        after = [int(a) for a in d["s4_after"]]
        if not after:
            return None
        eng = np.asarray(d["s4_eng"])  # (seeds, points, G, E, F)
        seeds = [int(s) for s in d["seeds"]]
        arm, axis, level = str(d["arm"]), str(d["axis"]), str(d["level"])
    points = {}
    for k, a in enumerate(after):
        recs = point_records(eng[:, k])
        points[a] = {
            "passes_every_seed": all(passes(r) for r in recs),
            "seeds_passing": sum(passes(r) for r in recs),
            "coverage_median": float(np.median([r["coverage"] for r in recs])),
            "median_learnability_median": float(np.nanmedian([r["median_learnability"] for r in recs]))
            if any(not math.isnan(r["median_learnability"]) for r in recs)
            else math.nan,
            "pooled_NOT_learnability_median": float(
                np.nanmedian([r["pooled_NOT_learnability_median"] for r in recs])
            ),
        }
    return {"arm": arm, "axis": axis, "level": level, "seeds": seeds, "points": points}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("longrun_dir", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args(argv)
    files = sorted(p for p in a.longrun_dir.rglob("*.npz") if p.parent.name != "partial")
    if not files:
        raise SystemExit(f"no condition checkpoints under {a.longrun_dir}")
    report = {"rule": {"coverage_floor": COVERAGE_FLOOR, "learnability_floor": FLOOR, "n_eng_min": N_ENG_MIN}}
    conds = {}
    for f in files:
        s = condition_summary(f)
        if s is not None:
            conds[f.stem] = s
    report["conditions"] = conds
    for tag, s in sorted(conds.items()):
        line = "  ".join(
            f"{a}:{'P' if v['passes_every_seed'] else '-'}{v['median_learnability_median']:.2f}"
            for a, v in s["points"].items()
        )
        print(f"{tag:<44s} {line}")
    if a.out:
        a.out.write_text(json.dumps(report, indent=1, default=float))
        print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
