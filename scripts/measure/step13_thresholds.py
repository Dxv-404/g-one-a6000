"""Step 13: derive the classifier's floor and separation margin from NULL data only, and freeze them. Run 46.

Reads the null arm of the characterisation sweep (`runs/a6000-handoff/extract`, runs 43 and 44) and writes
`configs/step13_frozen.json`, which every later analysis reads and nothing else may set. D-072.

**What is being thresholded.** The map's grain is the condition-lineage (D-050), and the treatment arm stores
population rows per seed, not per individual, so the classifier runs on a condition's z-profile: for each named
descriptor, a paired statistic over the seeds two arms share,

    z_d = mean(diff) / (sd(diff) / sqrt(n)),    diff = treatment seed value - null seed value

over the seeds where both values are defined. The classifier then takes the largest z and requires it to clear
the floor and to beat the runner-up by the separation margin (`src/measure/classifier.py`, unchanged).

**How the thresholds come from null data alone.** In each null condition, the 20 seeds are split at random into
two halves of ten, the halves are paired at random, and the same statistic is computed as though one half were
the treatment. Nothing is happening in such a comparison, so the distribution of its best z, and of the gap
between its best and second-best z, is what chance alone produces. The owner's ruling (2026-11-04, "standard")
sets both thresholds at the 95th percentile of those distributions.

**Two properties of the derivation, stated so they are not discovered later.**
    The pseudo-comparison pairs ten seeds that share no layout, where the real one pairs twenty that do. Its
    statistic has nine degrees of freedom against the real one's nineteen, so its tails are heavier and the
    thresholds come out higher than a t(19) reference would give. That errs toward missing a weak effect, not
    toward reporting a false one.
    The thresholds are global, pooled over all 29 null conditions, not per condition, because a per-condition
    95th percentile from 2,000 splits of only 20 seeds would mostly measure which seeds happened to be drawn.

The descriptor values use the same pooled denominator floor as S5 and run 45 (D-071), unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.measure import classifier as C  # noqa: E402
from src.measure import descriptors as D  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("sweep_mod", ROOT / "scripts" / "measure" / "sweep.py")
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)

EXTRACT = ROOT / "runs" / "a6000-handoff" / "extract"
OUT = ROOT / "configs" / "step13_frozen.json"

#: Owner's ruling, 2026-11-04: the standard bar, a 95th percentile.
QUANTILE = 0.95
#: Unchanged from S5 and run 45, D-071.
DENOM_FLOOR = 1.0
#: Pseudo-comparisons per null condition. An implementation constant: enough that the pooled quantiles are
#: stable to the second decimal, which `stability` below reports rather than asserts.
N_SPLITS = 2000
#: A pair statistic needs at least this many seeds defined in both halves.
MIN_PAIRS = 3
SEED = 20261104


def paired_z(a: np.ndarray, b: np.ndarray) -> float:
    """The paired statistic every comparison uses: mean difference over its standard error. NaN if fewer than
    MIN_PAIRS pairs are defined or the differences do not vary."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_PAIRS:
        return float("nan")
    d = a[ok] - b[ok]
    sd = d.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(d.mean() / (sd / math.sqrt(d.size)))


def best_and_margin(zs: dict[str, float]) -> tuple[float, float]:
    """The classifier's own best z and margin, through `classify` with thresholds that accept everything, so
    this derivation and the classifier cannot disagree about what "best" and "margin" mean."""
    vals = {d: z for d, z in zs.items()}
    nulls = {d: C.Null(0.0, 1.0) for d in zs}
    res = C.classify(vals, nulls, floor=-math.inf, separation=-math.inf)
    return res.z, res.margin


def null_seed_values(path: pathlib.Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    if str(d["arm"]) != "null":
        raise PermissionError(f"step 13 derives from null data only; {path.name} is {d['arm']}")
    cols = [str(c) for c in d["acc_fields"]]
    rows = np.asarray(d["naive_acc"], np.float64) * float(sweep.POOLED_PREY_EPISODES)
    pop = D.population({c: rows[:, i] for i, c in enumerate(cols)}, denominator_floor=DENOM_FLOOR)
    return {
        name: np.where(pop[name + "__defined"], pop[name], np.nan) for name in D.NAMED_STRATEGIES.values()
    }


def main() -> None:
    files = sorted(EXTRACT.glob("null__*.npz"))
    assert len(files) == 29, f"expected 29 null conditions, found {len(files)}"
    rng = np.random.default_rng(SEED)
    named = list(D.NAMED_STRATEGIES.values())
    best, margin, per_cond = [], [], {}
    for p in files:
        v = null_seed_values(p)
        n = len(next(iter(v.values())))
        cb, cm = [], []
        for _ in range(N_SPLITS):
            perm = rng.permutation(n)
            a_idx, b_idx = perm[: n // 2], perm[n // 2 :]
            zs = {name: paired_z(v[name][a_idx], v[name][b_idx]) for name in named}
            zs = {k: z for k, z in zs.items() if math.isfinite(z)}
            if len(zs) < 2:
                continue
            b, m = best_and_margin(zs)
            cb.append(b)
            cm.append(m)
        per_cond[p.stem] = {
            "splits": len(cb),
            "q95_best": float(np.quantile(cb, QUANTILE)),
            "q95_margin": float(np.quantile(cm, QUANTILE)),
        }
        best += cb
        margin += cm
    best, margin = np.asarray(best), np.asarray(margin)
    floor = float(np.quantile(best, QUANTILE))
    separation = float(np.quantile(margin, QUANTILE))
    fp = float(np.mean((best >= floor) & (margin >= separation)))
    halves = [float(np.quantile(best[i::2], QUANTILE)) for i in (0, 1)]

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    frozen = {
        "decision": "D-072",
        "run": 46,
        "frozen_on": "2026-11-04",
        "derived_from": "null arm only, runs 43 and 44",
        "statistic": "paired z per named descriptor over shared seeds: mean(diff) / (sd(diff) / sqrt(n))",
        "quantile": QUANTILE,
        "classifier_floor": round(floor, 4),
        "separation_margin": round(separation, 4),
        "denominator_floor": DENOM_FLOOR,
        "min_pairs": MIN_PAIRS,
        "null_false_label_rate_at_both": round(fp, 4),
        "pseudo_comparisons": int(best.size),
        "stability_floor_two_halves": [round(h, 4) for h in halves],
        "script_commit": commit,
        "extract_index_sha256": hashlib.sha256((EXTRACT / "index.json").read_bytes()).hexdigest(),
        "per_condition": per_cond,
    }
    OUT.write_text(json.dumps(frozen, indent=1) + "\n")
    print(f"floor {floor:.4f}  separation {separation:.4f}  over {best.size} null pseudo-comparisons")
    print(f"null label rate with both applied {fp:.4f}; floor from two halves {halves}")
    q = sorted(per_cond.items(), key=lambda kv: kv[1]["q95_best"])
    print(
        "per-condition 95th pct of best z, lowest and highest:",
        q[0][0],
        round(q[0][1]["q95_best"], 3),
        "|",
        q[-1][0],
        round(q[-1][1]["q95_best"], 3),
    )
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
