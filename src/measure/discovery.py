"""Tier two: unsupervised discovery. Build step 11.

Governed by `docs/analysis-structure.md`. **What behavioural modes exist, without naming them first?**

**Feature admission is measured, not argued, and it has TWO stages.** Stage one regresses each candidate on
the five named-strategy descriptors: a feature the taxonomy can already predict contributes nothing, so the
tier's power is the residual. Stage two regresses it on the features already accepted: two candidates that
are transformations of each other are both orthogonal to the taxonomy and would both pass stage one, and a
duplicated dimension double-weights its axis in every distance computation.

Stage two was added after `occlusion_rate` turned out to be exactly `1 - exposure_rate` and stage one could
not have caught it, because `exposure_rate` is not among the five predictors. **Algebra caught that instance;
algebra only catches the instances someone notices.**

**Two structural facts about the accumulator set constrain what tier two can even ask, and both were found
while building this rather than stated anywhere.**

*There is no unconditional stillness accumulator.* The frozen set carries `still_engaged` and
`still_unthreatened` but no `still_alive`, and the set cannot be extended after step 7 because nothing
unaccumulated can be recovered. `still_unthreatened / unthreatened_ticks` stands in, and it is a close stand
in rather than an exact one: measured, `unthreatened_ticks` is 220.9 of 222.8 alive ticks, so it covers 99.2
percent of the agent's life. The 0.8 percent it omits is exactly the threatened part, which is the part tier
three is for.

*`occluded_near + occluded_far == unthreatened_ticks` is an identity.* The two partition near against far
under "not visible", so their sum is every non-visible living tick. **Any overall-occlusion feature is
therefore exactly `1 - exposure_rate`**, and admitting both would admit one feature twice while making them
look like two independent pieces of evidence. Verified on run 1: 1.1408 plus 219.7896 is 220.930, and
`unthreatened_ticks` is 220.930.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from configs.analysis import K_RANGE, N_RESTARTS, R2_ADMISSION, R2_REDUNDANCY
from src.measure.descriptors import NAMED_STRATEGIES


class Candidate(NamedTuple):
    """One candidate feature: how to compute it, and why it is a candidate rather than a descriptor."""

    name: str
    numerator: str
    denominator: str
    note: str


#: The unconditional candidate features, over alive ticks or their closest available stand-in.
#:
#: **Deliberately short, because the accumulator set is what it is.** Every entry is a rate over a
#: denominator above two hundred, which is the whole point of tier two: quantities with samples.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate("activity_rate", "moved_ticks", "alive_ticks", "how much the agent moves at all"),
    Candidate(
        "stillness_rate",
        "still_unthreatened",
        "unthreatened_ticks",
        "the closest available unconditional stillness; there is no still_alive accumulator",
    ),
    Candidate("local_density", "density_sum", "alive_ticks", "living neighbours within r_group"),
    Candidate("exposure_rate", "visible_ticks", "alive_ticks", "how much threat the agent experiences"),
    Candidate("proximity_rate", "near_ticks", "alive_ticks", "ticks with the predator inside r_pred"),
    Candidate(
        "neighbour_distance",
        "nn_sum_unseen",
        "nn_unseen_ticks",
        "mean distance to the nearest living prey when unthreatened",
    ),
)

#: Excluded by construction rather than by admission, and the reason is different from a low R-squared.
#: `occlusion_rate` would be `(occluded_near + occluded_far) / alive_ticks`, which is exactly
#: `1 - exposure_rate` by the identity above. It is not a weak feature; it is the same feature.
EXCLUDED_BY_IDENTITY: dict[str, str] = {
    "occlusion_rate": "identically 1 - exposure_rate, since occluded_near + occluded_far == unthreatened_ticks"
}


def features(accs) -> dict[str, np.ndarray]:
    """Compute every candidate feature for a population. NaN where the denominator is zero.

    No denominator floor here, unlike the descriptors: every candidate's denominator is `alive_ticks` or
    `unthreatened_ticks`, both above two hundred per prey, so the floor would never bind. A zero denominator
    still means a prey that never lived a tick, which is a defect rather than a small sample.
    """

    def g(name):
        v = getattr(accs, name) if hasattr(accs, name) else accs[name]
        return np.asarray(v, dtype=np.float64)

    out = {}
    for c in CANDIDATES:
        num, den = g(c.numerator), g(c.denominator)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[c.name] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def r_squared(candidate: np.ndarray, predictors: np.ndarray) -> float:
    """R-squared of `candidate` regressed on `predictors`, with an intercept.

    Rows where anything is not finite are dropped, because a descriptor undefined for lack of data must not
    be silently treated as zero. **The surviving row count is what the caller has to check**: on this
    project's numbers two of the five named descriptors are undefined for most individuals, so a regression
    on all five can be left with very few rows and an R-squared that means nothing.

    Returns NaN when fewer rows survive than there are coefficients to fit, rather than a perfect fit, which
    is what least squares returns for an underdetermined system and which would read as "fully explained".
    """
    y = np.asarray(candidate, dtype=np.float64)
    x = np.asarray(predictors, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y, x = y[ok], x[ok]

    n_coef = x.shape[1] + 1
    if y.size <= n_coef:
        return float("nan")

    design = np.column_stack([np.ones(y.size), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


class Admission(NamedTuple):
    """One candidate's admission decision.

    `r2_taxonomy` is stage one, against the five named descriptors. `r2_accepted` is stage two, against the
    features already admitted. They answer different questions and a candidate must clear both.

    **`r2_accepted` is populated for rejected candidates too**, so the admission order's effect is visible in
    the table rather than merely fixed by pre-registration. NaN means there was nothing accepted yet to
    compare against, which is different from having been compared and found independent.
    """

    name: str
    r2_taxonomy: float
    r2_accepted: float
    n_rows: int
    admitted: bool
    reason: str


def admit(
    accs,
    descriptor_values: dict[str, np.ndarray],
    *,
    threshold: float = R2_ADMISSION,
    redundancy: float = R2_REDUNDANCY,
):
    """Two-stage admission. A candidate must clear BOTH to enter the feature set.

    ```
    stage one   candidate against the five named descriptors   does it add anything the taxonomy lacks
    stage two   candidate against the ALREADY ACCEPTED set     is it already in there
    ```

    **Stage one alone was insufficient, and the case that showed it is instructive.** `occlusion_rate` is
    exactly `1 - exposure_rate`, by the identity `occluded_near + occluded_far == unthreatened_ticks`. Both
    are orthogonal to the taxonomy, so both clear stage one, and admitting both would double-weight that axis
    in every distance computation and bend the clusters toward whatever it measures. That case was caught by
    algebra here and excluded by construction, but algebra only catches the instances someone notices.

    **Order matters and is part of the pre-registration.** Stage two compares against what has already been
    accepted, so with two equivalent candidates the LATER one is rejected. The order of `CANDIDATES` is
    therefore a committed choice rather than an incidental one.

    A candidate whose R-squared cannot be computed is NOT admitted at either stage. Admitting it would mean
    admitting a feature on the grounds that the check failed.
    """
    feats = features(accs)
    predictor_names = list(NAMED_STRATEGIES.values())
    missing = [n for n in predictor_names if n not in descriptor_values]
    if missing:
        raise KeyError(f"missing descriptor predictors: {missing}")

    x = np.column_stack([np.asarray(descriptor_values[n], dtype=np.float64) for n in predictor_names])

    rows: list[Admission] = []
    accepted: list[np.ndarray] = []
    for c in CANDIDATES:
        y = feats[c.name]
        ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        n_rows = int(ok.sum())

        r2_tax = r_squared(y, x)

        # **Computed for every candidate, including ones stage one rejects.** Fixing the admission order
        # makes the result reproducible; it does not make it neutral. A different order gives a different
        # feature set, and reporting each rejected candidate's R-squared against the accepted set AT THE
        # MOMENT OF REJECTION is what makes the order's cost visible rather than merely fixed. The numbers
        # are computed anyway.
        r2_acc = float("nan") if not accepted else r_squared(y, np.column_stack(accepted))

        if not np.isfinite(r2_tax):
            rows.append(Admission(c.name, r2_tax, r2_acc, n_rows, False, "R-squared not computable"))
            continue
        if r2_tax >= threshold:
            rows.append(Admission(c.name, r2_tax, r2_acc, n_rows, False, "inside the named taxonomy's span"))
            continue
        if np.isfinite(r2_acc) and r2_acc >= redundancy:
            rows.append(
                Admission(c.name, r2_tax, r2_acc, n_rows, False, "already spanned by the accepted features")
            )
            continue

        accepted.append(y)
        rows.append(Admission(c.name, r2_tax, r2_acc, n_rows, True, "carries residual information"))
    return rows


def accepted_matrix(rows, accs) -> tuple[list[str], np.ndarray]:
    """The admitted features as a matrix, in admission order, with their names.

    Kept separate from `admit` so a caller cannot accidentally cluster on the rejected ones, which is the
    kind of slip that produces a result nobody can reproduce from the admission table.
    """
    feats = features(accs)
    names = [r.name for r in rows if r.admitted]
    if not names:
        return names, np.zeros((0, 0))
    return names, np.column_stack([feats[n] for n in names])


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette, in numpy. `scikit-learn` is not a dependency and adding one mid-phase would
    re-resolve the lockfile, which D-021 forbids for the sake of gate E1's determinism baseline."""
    uniq = np.unique(labels)
    if uniq.size < 2:
        return float("nan")
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    s = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        own = labels == labels[i]
        own_count = int(own.sum()) - 1
        if own_count <= 0:
            s[i] = 0.0
            continue
        a = d[i, own].sum() / own_count
        b = min(d[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(s.mean())


def cluster(x: np.ndarray, key_seed: int, *, k_range=K_RANGE, restarts: int = N_RESTARTS):
    """k-means over standardised features, k chosen by mean silhouette. Returns (labels, k, silhouette).

    **Standardised first**, because the candidates are rates in [0, 1] and distances in cells, and unscaled
    Euclidean distance would let the distance feature dominate purely by having a larger range.

    Deterministic given `key_seed`, because gates E1 and S7 both require that a run reproduces, and a
    clustering that moved between runs would break reproduction at the analysis stage rather than the
    simulation stage, which is harder to notice.
    """
    from scipy.cluster.vq import kmeans2

    x = np.asarray(x, dtype=np.float64)
    ok = np.all(np.isfinite(x), axis=1)
    x = x[ok]
    if x.shape[0] < max(k_range) + 1:
        return np.zeros(0, dtype=int), 0, float("nan")

    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (x - mu) / sd

    best = (None, 0, -np.inf)
    for k in k_range:
        for r in range(restarts):
            centroids, labels = kmeans2(z, k, minit="++", seed=key_seed * 1000 + r)
            if np.unique(labels).size < k:
                continue
            score = _silhouette(z, labels)
            if np.isfinite(score) and score > best[2]:
                best = (labels, k, score)
    if best[0] is None:
        return np.zeros(0, dtype=int), 0, float("nan")
    return best[0], best[1], best[2]
