"""Simulation-based power, gate S6. Build step 12.

**What the spec asks for, quoted so the implementation can be checked against it.** "Resample from the
observed `naive_eval` null, inject an effect of the smallest size worth claiming, and measure detection rate
across candidate seed counts under the planned interquartile-mean and bootstrap procedure. The planned seed
count must reach 80 percent detection. The smallest effect worth claiming is defined before Phase B as a
shift in conditional stillness equal to one standard deviation of the `naive_eval` null, which is a Cohen's
d of 1.0."

**The planned procedure, from the canonical's Statistics section and D-045.** Results are interquartile
means with ninety-five percent bootstrap confidence intervals over seeds, and any comparison over arms that
share seeds is stated as a paired test. So one simulated experiment is: a treatment sample and a null sample
of `n_seeds` runs each, the per-seed difference, the interquartile mean of those differences, and a
percentile bootstrap interval on that mean. **Detection is the interval excluding zero.**

**Three things this simulation assumes, stated because each biases the result and in a known direction.**

    the treatment's seed-to-seed spread equals the null's      no treatment data exists yet; nothing else is
                                                               available. Neutral
    the two arms' seed noise is INDEPENDENT                    the real design pairs seeds, which shares layouts
                                                               and can only REMOVE noise from the difference.
                                                               So this understates power: the rate reported is
                                                               a FLOOR on the real one
    the bootstrap is over seeds in one stratum                 Agarwal et al. stratify over tasks; the sweep
                                                               will stratify over conditions when it aggregates.
                                                               One condition is one stratum, so the stratified
                                                               and plain percentile bootstraps coincide here

The interquartile mean is `scipy.stats.trim_mean(x, 0.25)`, the same definition rliable uses, and one
implementation serves the gate, the measurement script and the eventual analysis, per the rule against
duplicated verification logic.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import trim_mean

#: Cohen's d of the smallest effect worth claiming. Spec, S6: one standard deviation of the observed null.
SMALLEST_EFFECT_D = 1.0
#: Detection rate the planned seed count must reach. Spec, S6.
REQUIRED_POWER = 0.80
#: Confidence level of the reported intervals. Canonical, Statistics.
CONFIDENCE = 0.95


def iqm(x: np.ndarray) -> float:
    """Interquartile mean: the mean of the middle fifty percent. Agarwal et al. 2021, via trim_mean."""
    return float(trim_mean(np.asarray(x, np.float64), 0.25))


def iqm_rows(m: np.ndarray) -> np.ndarray:
    """`iqm` of every row of a 2-d array at once. The same cut trim_mean makes, int(0.25 n) from each
    end of the sorted row, and asserted equal to `iqm` row by row in the module's tests."""
    m = np.asarray(m, np.float64)
    n = m.shape[1]
    k = int(0.25 * n)
    return np.sort(m, axis=1)[:, k : n - k].mean(axis=1)


def bootstrap_interval(x: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float]:
    """Percentile bootstrap interval of the interquartile mean, over the runs in `x`."""
    x = np.asarray(x, np.float64)
    n = x.size
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = iqm_rows(x[idx])
    lo = (1.0 - CONFIDENCE) / 2.0
    return float(np.quantile(stats, lo)), float(np.quantile(stats, 1.0 - lo))


def detection_rate(
    null: np.ndarray,
    n_seeds: int,
    *,
    d: float = SMALLEST_EFFECT_D,
    n_sims: int = 500,
    n_boot: int = 500,
    rng: np.random.Generator,
) -> float:
    """Fraction of simulated experiments at `n_seeds` runs per arm in which an injected effect of size `d`
    standard deviations of `null` is detected: the bootstrap interval of the interquartile mean of the
    per-seed treatment-minus-null differences lies entirely above zero.

    `null` is the observed per-seed statistic under `naive_eval`. Its standard deviation is the unit of the
    effect, so a degenerate null with zero spread makes the effect zero and this function raises rather than
    reporting a rate that would mean nothing.
    """
    null = np.asarray(null, np.float64)
    assert null.ndim == 1 and null.size >= 2, "the null must be a vector of at least two per-seed values"
    sd = float(null.std(ddof=1))
    # The range, not the deviation: a constant vector's std is about 1e-17 under pairwise summation, not
    # zero, and `sd <= 0.0` let one through. `ptp == 0` is exact. Caught by the module's can-it-fail test.
    if not np.isfinite(sd) or float(np.ptp(null)) == 0.0:
        raise ValueError(f"the null has no spread (sd = {sd}), so an effect of {d} sd is not an effect")
    shift = d * sd
    hits = 0
    for _ in range(n_sims):
        treat = null[rng.integers(0, null.size, n_seeds)] + shift
        ctrl = null[rng.integers(0, null.size, n_seeds)]
        lo, _ = bootstrap_interval(treat - ctrl, rng, n_boot)
        hits += int(lo > 0.0)
    return hits / n_sims


def power_curve(
    null: np.ndarray,
    seed_counts: tuple[int, ...],
    *,
    d: float = SMALLEST_EFFECT_D,
    rng: np.random.Generator,
    **kw,
) -> dict[int, float]:
    """`detection_rate` at each candidate seed count, from one generator so the curve is reproducible."""
    return {n: detection_rate(null, n, d=d, rng=rng, **kw) for n in seed_counts}


def detection_rate_from_differences(
    diffs: np.ndarray,
    shift: float,
    n_seeds: int,
    *,
    n_sims: int = 500,
    n_boot: int = 500,
    rng: np.random.Generator,
) -> float:
    """Power when the per-seed NOISE of the difference is observed rather than assumed.

    `detection_rate` resamples two independent arms from the null, which assumes the arms' seed noise is
    uncorrelated and is therefore a floor on the paired design's power. Once a treatment arm exists on the
    same seeds, the observed per-seed differences carry the paired noise directly: their spread is the
    spread of the statistic the analysis actually tests. This resamples that spread, centred so any real
    effect in the observed differences is removed, and injects `shift` in the statistic's own units.

    `shift` is passed rather than `d`, because the effect is still defined as d null standard deviations,
    and the null's sd is not a property of the differences. The caller computes `shift = d * sd_null`.
    """
    diffs = np.asarray(diffs, np.float64)
    assert diffs.ndim == 1 and diffs.size >= 2, "need at least two paired differences"
    if float(np.ptp(diffs)) == 0.0:
        raise ValueError("the paired differences have no spread, so their noise cannot be resampled")
    noise = diffs - diffs.mean()
    hits = 0
    for _ in range(n_sims):
        sample = noise[rng.integers(0, noise.size, n_seeds)] + shift
        lo, _ = bootstrap_interval(sample, rng, n_boot)
        hits += int(lo > 0.0)
    return hits / n_sims
