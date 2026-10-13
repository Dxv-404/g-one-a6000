"""Gates S5 and S6, built against the observed null. Build step 12.

**Both read a registered run rather than computing one**, because the null is ten predator-naive
evolutionary runs of 300 generations evaluated against the predator, about 75 minutes, which no test can
contain. Run 32 produced it through the sweep runner as condition one of the null arm, into the sweep's own
directory, so the sweep reuses it. These are step-completion gates in the sense CLAUDE.md section 3 gives
S2 and S6: marked slow, run explicitly, and they FAIL rather than skip when the run is absent, because a
gate that skips is a gate that passes.

**S5, from the spec:** "`naive_eval` descriptors are stable across seeds with a characterisable spread." The
canonical sets the floor on seeds at "explicitly more than five", so each named-strategy descriptor must be
defined on more than five of the ten seeds, and its spread must be finite and non-degenerate. The last is
load-bearing for S6: the null's standard deviation is the UNIT of the smallest effect worth claiming, so a
null with no spread makes that effect zero. "Stable" is read as defined and non-degenerate on the seeds
it is defined on; the spread itself is reported as an interquartile mean with its interval rather than held
to a threshold the spec does not state.

**S6, from the spec:** simulation-based power at the planned seed count must reach 80 percent for a shift
of one null standard deviation in conditional stillness. The planned count is the runner's `N_SEEDS`, read
from the runner so the two cannot disagree. The curve across candidate counts is printed beside the verdict,
because the number the owner needs if this fails is how many seeds WOULD reach 80 percent.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from src.measure import descriptors as D
from src.measure import power as P

ROOT = pathlib.Path(__file__).resolve().parents[2]
NULL_CHECKPOINT = ROOT / "runs" / "sweep" / "null__baseline__None.npz"

_SPEC = importlib.util.spec_from_file_location("sweep_mod", ROOT / "scripts" / "measure" / "sweep.py")
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)

#: The same floor the calibration scripts use on population-mean accumulators.
DENOM_FLOOR = 1.0
#: Canonical, Statistics: the seed count is "explicitly more than five".
MIN_DEFINED_SEEDS = 6
#: Candidate seed counts for the power curve. The planned count is included by construction below.
CANDIDATE_SEEDS = (5, 10, 15, 20, 30, 40)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def null_by_seed() -> dict[str, np.ndarray]:
    """Per-seed named-strategy descriptors of the null: the population-mean accumulators of each seed's
    final predator-naive population, evaluated in the predator world, through the real descriptor code."""
    assert NULL_CHECKPOINT.exists(), (
        f"{NULL_CHECKPOINT} is absent. S5 and S6 read run 32, the null baseline through the sweep runner: "
        "uv run python scripts/measure/sweep.py runs/sweep --arms null --limit 1 --verified-commit <sha>"
    )
    sweep.assert_checkpoint_schema(NULL_CHECKPOINT)
    with np.load(NULL_CHECKPOINT) as d:
        assert str(d["arm"]) == "null"
        cols = [str(c) for c in d["acc_fields"]]
        naive = d["naive_acc"]  # (seeds, columns)
    assert naive.shape[0] == sweep.N_SEEDS, (
        f"the null carries {naive.shape[0]} seeds, the runner plans {sweep.N_SEEDS}"
    )
    out: dict[str, list] = {name: [] for name in D.NAMED_STRATEGIES.values()}
    defined: dict[str, list] = {name: [] for name in D.NAMED_STRATEGIES.values()}
    for s in range(naive.shape[0]):
        row = {c: float(naive[s, i]) for i, c in enumerate(cols)}
        desc = D.descriptors(row, denominator_floor=DENOM_FLOOR)
        for name in out:
            out[name].append(desc[name].value)
            defined[name].append(desc[name].defined)
    return {
        **{k: np.array(v, np.float64) for k, v in out.items()},
        **{k + "__defined": np.array(v, bool) for k, v in defined.items()},
    }


# ------------------------------------------------------------------------------------------ S5 --
@pytest.mark.parametrize("strategy,name", sorted(D.NAMED_STRATEGIES.items()))
def test_s5_the_null_descriptor_is_defined_on_more_than_five_seeds(null_by_seed, strategy, name):
    n = int(null_by_seed[name + "__defined"].sum())
    assert n >= MIN_DEFINED_SEEDS, f"{strategy} ({name}) is defined on {n} of {sweep.N_SEEDS} null seeds"


@pytest.mark.parametrize("strategy,name", sorted(D.NAMED_STRATEGIES.items()))
def test_s5_the_null_descriptor_has_a_finite_non_degenerate_spread(null_by_seed, strategy, name):
    v = null_by_seed[name][null_by_seed[name + "__defined"]]
    assert np.all(np.isfinite(v)), f"{strategy}: non-finite null values {v}"
    sd = float(v.std(ddof=1))
    assert sd > 0.0, f"{strategy}: the null has no spread across seeds, so one null sd is not an effect"
    lo, hi = P.bootstrap_interval(v, np.random.default_rng(0))
    print(
        f"\n  S5 {strategy:<10s} {name:<28s} n {v.size:2d}  IQM {P.iqm(v):+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  sd {sd:.4f}"
    )
    assert np.isfinite(lo) and np.isfinite(hi)


# ------------------------------------------------------------------------------------------ S6 --
def test_s6_the_planned_seed_count_reaches_eighty_percent_power(null_by_seed):
    """The gate. Conditional stillness near, the freezing detector, is the descriptor S6 names."""
    name = D.NAMED_STRATEGIES["freezing"]
    v = null_by_seed[name][null_by_seed[name + "__defined"]]
    counts = tuple(sorted(set(CANDIDATE_SEEDS) | {sweep.N_SEEDS}))
    curve = P.power_curve(
        v, counts, d=P.SMALLEST_EFFECT_D, n_sims=1000, n_boot=500, rng=np.random.default_rng(32)
    )
    print(
        f"\n  S6 power for a d = {P.SMALLEST_EFFECT_D} shift in {name}, null sd {v.std(ddof=1):.4f} over {v.size} seeds"
    )
    print("  seeds   detection")
    for n in counts:
        mark = "  <- planned" if n == sweep.N_SEEDS else ""
        print(f"  {n:5d}   {curve[n]:9.3f}{mark}")
    enough = [n for n in counts if curve[n] >= P.REQUIRED_POWER]
    print(
        f"  smallest candidate reaching {P.REQUIRED_POWER:.0%}: {enough[0] if enough else 'none of the candidates'}"
    )
    assert curve[sweep.N_SEEDS] >= P.REQUIRED_POWER, (
        f"power at the planned {sweep.N_SEEDS} seeds is {curve[sweep.N_SEEDS]:.3f}, below "
        f"{P.REQUIRED_POWER}. Smallest candidate reaching it: {enough[0] if enough else 'none'}"
    )


def test_s6_the_gate_can_fail():
    """A null with a wide spread relative to nothing: at two seeds no procedure reaches 80 percent."""
    wide = np.random.default_rng(1).normal(0.0, 1.0, 10)
    assert P.detection_rate(wide, 2, n_sims=200, n_boot=200, rng=np.random.default_rng(2)) < P.REQUIRED_POWER
