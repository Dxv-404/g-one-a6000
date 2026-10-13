#!/usr/bin/env python3
"""Q-051, reading 1: measure the seed pairing between the treatment and null baselines, and re-run S6's
power with the paired noise observed rather than assumed.

**Why this is a measurement and not a change to gate S6.** S6 as built resamples two independent arms from
the null, because at the time only the null existed. That assumes the arms' seed noise is uncorrelated, and
the design pairs seeds on shared layouts, so the gate's 0.568 at ten seeds is a floor. Run 33 is the
treatment baseline on the same ten seeds. The per-seed treatment-minus-null differences carry the paired
noise directly, and resampling THEM is the procedure the analysis actually uses. The gate is not edited
here; this reports what the gate would say under the paired variance, and the owner rules on Q-051 with both
numbers in front of them.

**What is paired with what.** For each seed: the treatment arm's own population-mean descriptor at its
last sampled generation, the evolved lineage in its own predator world; and the null arm's `naive_acc`
descriptor, the predator-naive lineage's final population evaluated in that same predator world on that same
layout. The layout sharing is asserted by `test_every_arm_shares_the_layout_at_a_given_condition_and_seed`.

**The three numbers per descriptor.** rho, the seed correlation between arms; the ratio of the observed
difference sd to root-two times the null sd, which is what independence would give (below one means pairing
helped); and, for freezing, the power curve at the candidate seed counts under the paired noise, beside the
gate's independent-noise curve, from the same effect size in the same units.

Usage: uv run python scripts/measure/pairing_power.py <out.npz>
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from src.measure import descriptors as D
from src.measure import power as P

ROOT = pathlib.Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("sweep_mod", ROOT / "scripts" / "measure" / "sweep.py")
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)

NULL = ROOT / "runs" / "sweep" / "null__baseline__None.npz"
TREAT = ROOT / "runs" / "sweep" / "treatment__baseline__None.npz"
DENOM_FLOOR = 1.0  # the same floor tests/scientific/test_s5_s6_null.py and the calibration scripts use
CANDIDATE_SEEDS = (5, 10, 15, 20, 30, 40)


def per_seed(values: np.ndarray, cols: list[str]) -> dict[str, np.ndarray]:
    """Named-strategy descriptors per seed from (seeds, columns) population-mean accumulators."""
    out = {name: [] for name in D.NAMED_STRATEGIES.values()}
    defined = {name: [] for name in D.NAMED_STRATEGIES.values()}
    for s in range(values.shape[0]):
        row = {c: float(values[s, i]) for i, c in enumerate(cols)}
        desc = D.descriptors(row, denominator_floor=DENOM_FLOOR)
        for name in out:
            out[name].append(desc[name].value)
            defined[name].append(desc[name].defined)
    return {
        **{k: np.array(v, np.float64) for k, v in out.items()},
        **{k + "__defined": np.array(v, bool) for k, v in defined.items()},
    }


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    out_path = sys.argv[1]
    for p in (NULL, TREAT):
        assert p.exists(), f"{p} is absent"
        sweep.assert_checkpoint_schema(p)
    with np.load(NULL) as d:
        cols = [str(c) for c in d["acc_fields"]]
        null_vals = per_seed(d["naive_acc"], cols)
        n_seeds = int(d["n_seeds"])
    with np.load(TREAT) as d:
        assert [str(c) for c in d["acc_fields"]] == cols
        assert int(d["n_seeds"]) == n_seeds
        gens = d["sample_gens"]
        treat_vals = per_seed(d["sample_acc"][:, -1, :], cols)
    print(
        f"  {n_seeds} seeds paired; treatment read at its last sampled generation, {int(gens[-1])}\n",
        flush=True,
    )

    print(
        f"  {'strategy':<10s} {'n':>3s} {'rho':>7s} {'sd_null':>8s} {'sd_diff':>8s} {'sd_diff / (root2 sd_null)':>26s}",
        flush=True,
    )
    saved = {}
    for strategy, name in sorted(D.NAMED_STRATEGIES.items()):
        both = null_vals[name + "__defined"] & treat_vals[name + "__defined"]
        nv, tv = null_vals[name][both], treat_vals[name][both]
        if both.sum() < 3:
            print(f"  {strategy:<10s} {int(both.sum()):3d}   too few paired seeds", flush=True)
            continue
        diff = tv - nv
        rho = float(np.corrcoef(nv, tv)[0, 1])
        sd_null, sd_diff = float(nv.std(ddof=1)), float(diff.std(ddof=1))
        ratio = sd_diff / (np.sqrt(2.0) * sd_null) if sd_null > 0 else float("nan")
        print(
            f"  {strategy:<10s} {int(both.sum()):3d} {rho:+7.3f} {sd_null:8.4f} {sd_diff:8.4f} {ratio:26.3f}",
            flush=True,
        )
        saved[name] = dict(null=nv, treat=tv, diff=diff, rho=rho)

    # the gate's descriptor, both curves from the same shift in the same units
    name = D.NAMED_STRATEGIES["freezing"]
    nv, diff = saved[name]["null"], saved[name]["diff"]
    shift = P.SMALLEST_EFFECT_D * float(nv.std(ddof=1))
    print(
        f"\n  S6 under both noise models, d = {P.SMALLEST_EFFECT_D}, shift {shift:.4f} in {name}", flush=True
    )
    print(f"  {'seeds':>6s} {'independent (the gate)':>24s} {'paired (observed)':>18s}", flush=True)
    indep = P.power_curve(
        nv, CANDIDATE_SEEDS, d=P.SMALLEST_EFFECT_D, n_sims=1000, n_boot=500, rng=np.random.default_rng(32)
    )
    paired = {
        n: P.detection_rate_from_differences(
            diff, shift, n, n_sims=1000, n_boot=500, rng=np.random.default_rng(33)
        )
        for n in CANDIDATE_SEEDS
    }
    for n in CANDIDATE_SEEDS:
        mark = "  <- planned" if n == sweep.N_SEEDS else ""
        print(f"  {n:6d} {indep[n]:24.3f} {paired[n]:18.3f}{mark}", flush=True)
    first_i = next((n for n in CANDIDATE_SEEDS if indep[n] >= P.REQUIRED_POWER), None)
    first_p = next((n for n in CANDIDATE_SEEDS if paired[n] >= P.REQUIRED_POWER), None)
    print(
        f"\n  smallest candidate reaching {P.REQUIRED_POWER:.0%}: independent {first_i}, paired {first_p}",
        flush=True,
    )

    np.savez(
        out_path,
        seeds=np.int32(n_seeds),
        columns=np.array(cols),
        candidate_seeds=np.array(CANDIDATE_SEEDS, np.int32),
        indep_curve=np.array([indep[n] for n in CANDIDATE_SEEDS]),
        paired_curve=np.array([paired[n] for n in CANDIDATE_SEEDS]),
        shift=np.float64(shift),
        **{f"{k}__{f}": v for k, d_ in saved.items() for f, v in d_.items()},
    )
    print(f"  saved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
