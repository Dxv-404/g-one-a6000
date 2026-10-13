"""Tier three's classifier: which named strategy, if any, does an individual show. Build step 11.

Governed by `docs/analysis-structure.md`. The form is unchanged from the earlier specification: **standardise
each descriptor against its own null, take the largest, subject to a floor and a separation margin.**

**Both thresholds are required arguments with no defaults.** The specification derives them at step 13 from
null data only and freezes them before treatment runs. A default would let a caller who has not derived them
get a number anyway, and it would then travel into results as though it had been chosen. Same structural move
as the denominator floor, and as invariant I1's signature.

**Standardising against the null is what makes five different descriptors comparable.** `conditional_stillness`
is a difference of rates and `mean_distance` is a distance in cells, so their raw magnitudes mean nothing
against each other. A z-score against each descriptor's own null distribution puts them on one axis, and the
axis is "how unusual is this against a lineage that never met a predator".

**Unclassified is a real answer and appears often.** A run where most individuals are unclassified is a
finding about the world rather than a failure of the classifier, and on run 1's numbers two of the five
descriptors have a zero numerator and a denominator of 0.262, so they cannot be classified at all.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from src.measure.descriptors import NAMED_STRATEGIES

#: Returned when no strategy clears the floor and the margin, or when too few descriptors are defined to
#: compare. A string rather than None so it survives into a table as a category rather than a hole.
UNCLASSIFIED = "unclassified"

#: Returned when fewer than two descriptors are defined, so a separation margin cannot be computed at all.
#: **Distinct from `unclassified` on purpose.** "Nothing stood out" and "there was not enough data to ask"
#: are different findings, and collapsing them would report missing data as an absence of behaviour, which
#: is the same error as reporting learnability as zero rather than undefined.
INSUFFICIENT = "insufficient-data"


class Classification(NamedTuple):
    label: str
    z: float
    runner_up: str
    margin: float
    n_defined: int


class Null(NamedTuple):
    """The null distribution for one descriptor, from the predator-naive lineage.

    `sd` of zero means the descriptor did not vary in the null at all, which makes a z-score undefined rather
    than infinite. That is treated as insufficient data rather than as an infinitely strong signal, because
    an undefined comparison is not evidence.
    """

    mean: float
    sd: float


def z_score(value: float, null: Null) -> float:
    """Standardised against the null. NaN when the value is undefined or the null did not vary."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if not np.isfinite(null.sd) or null.sd <= 0.0:
        return float("nan")
    return (float(value) - null.mean) / null.sd


def classify(
    values: dict[str, float],
    nulls: dict[str, Null],
    *,
    floor: float,
    separation: float,
) -> Classification:
    """Classify one individual.

    `values` maps DESCRIPTOR names to values; `nulls` maps the same names to their null distributions. Only
    the descriptors named in `NAMED_STRATEGIES` are candidates, so an unconditional diagnostic like
    `exposure_rate` cannot become a strategy label by being present in the dict.

    Returns `INSUFFICIENT` rather than `UNCLASSIFIED` when fewer than two candidates have a finite z-score,
    because a margin between fewer than two things is not a small margin, it is not a margin.
    """
    scored: list[tuple[str, float]] = []
    for strategy, descriptor in NAMED_STRATEGIES.items():
        if descriptor not in values or descriptor not in nulls:
            continue
        z = z_score(values[descriptor], nulls[descriptor])
        if math.isfinite(z):
            scored.append((strategy, z))

    if len(scored) < 2:
        only = scored[0][0] if scored else ""
        return Classification(
            INSUFFICIENT, scored[0][1] if scored else float("nan"), only, float("nan"), len(scored)
        )

    scored.sort(key=lambda t: t[1], reverse=True)
    (best, best_z), (second, second_z) = scored[0], scored[1]
    margin = best_z - second_z

    if best_z < floor or margin < separation:
        return Classification(UNCLASSIFIED, best_z, second, margin, len(scored))
    return Classification(best, best_z, second, margin, len(scored))


def prevalence(labels) -> dict[str, float]:
    """Fraction of the population carrying each label. **The primary result is a population-prevalence
    statistic**, which is why the GA evolves a population of distinct individuals rather than a search
    distribution, and this is the function that expresses it.

    Every named strategy appears in the output even at zero, so a strategy that never occurred is visible as
    a zero rather than as a missing key. A missing key reads as "not measured".
    """
    labels = list(labels)
    n = len(labels)
    if n == 0:
        raise ValueError("no labels to summarise")
    out = {s: 0.0 for s in NAMED_STRATEGIES}
    out[UNCLASSIFIED] = 0.0
    out[INSUFFICIENT] = 0.0
    for label in labels:
        out[label] = out.get(label, 0.0) + 1.0
    return {k: v / n for k, v in out.items()}


def null_from(population_values: np.ndarray) -> Null:
    """Fit a null from the predator-naive lineage's values for one descriptor.

    NaNs are dropped, since an undefined descriptor contributes no information about the null, and the count
    that survives is what the caller should check before trusting it. Fewer than two finite values gives an
    sd of zero, which `z_score` then treats as undefined rather than as certainty.
    """
    v = np.asarray(population_values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return Null(float(v[0]) if v.size == 1 else float("nan"), 0.0)
    return Null(float(v.mean()), float(v.std(ddof=1)))
