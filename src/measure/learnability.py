"""Learnability per genome, as Q-047 and D-076 define it. The single implementation.

Run 52's analysis (`scripts/measure/replay_analysis.py`) and the S4 calibration (`scripts/measure/calibrate_s4.py`,
D-083) both call `s4` here rather than each carrying a copy, because duplicated verification logic in this
repository has twice degraded toward the sloppier copy (CLAUDE.md section 11).

**Pure numpy, and it imports nothing from the behaviour side of this package** (descriptors, classifier,
discovery). The calibration must stay blind to behaviour, and `tests/engineering/test_calibrate_s4.py` asserts
that importing the calibration driver loads none of those modules. Adding such an import here would fail it.

Definitions, fixed at D-076 before any of this ran:

    decided       escaped + captured + died_other, over every episode of the genome; censored is excluded
    p             escaped / decided, DEFINED only where decided >= n_eng_min
    learnability  p (1 - p), in band at the floor or above (0.16 is p in [0.2, 0.8])
    coverage      the share of genomes where p is defined

Undefined is reported as undefined (NaN medians, coverage), never as zero, because zero reads as collapse.
"""

from __future__ import annotations

import math

import numpy as np

#: Column order of an engagement record, as `scripts/measure/replay_final.py::ENGAGEMENT_FIELDS` writes it.
#: Asserted equal to that tuple in `test_calibrate_s4.py`, so the two cannot drift apart silently.
ENG_FIELDS = (
    "engagements",
    "escaped",
    "captured",
    "died_other",
    "censored",
    "engaged_ticks",
    "predated_outside_engagement",
)
_I = {f: i for i, f in enumerate(ENG_FIELDS)}


def decided_and_escaped(eng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S, G, E, F) engagement records -> (decided, escaped), each (S, G), summed over the E episodes."""
    eng = np.asarray(eng, np.float64)
    assert eng.ndim == 4 and eng.shape[-1] == len(ENG_FIELDS), (
        f"expected (S, G, E, {len(ENG_FIELDS)}), got {eng.shape}"
    )
    den = (eng[..., _I["escaped"]] + eng[..., _I["captured"]] + eng[..., _I["died_other"]]).sum(axis=2)
    esc = eng[..., _I["escaped"]].sum(axis=2)
    return den, esc


def s4(eng: np.ndarray, n_eng_min: int, floor: float) -> dict:
    """D-076's S4 quantities over every genome of every seed in `eng` (S, G, E, F).

    The pooled value is reported beside the per-genome ones and labelled as the statistic Q-047 ruled is NOT
    learnability: it pools escapes across genomes with different behaviour."""
    den, esc = decided_and_escaped(eng)
    ok = den >= n_eng_min
    p = esc[ok] / den[ok]
    learn = p * (1 - p)
    pool_den = den.sum(axis=1)
    pool_p = np.where(pool_den > 0, esc.sum(axis=1) / np.maximum(pool_den, 1), np.nan)
    return {
        "coverage": float(ok.mean()),
        "defined": int(ok.sum()),
        "share_in_band_of_defined": float(np.mean(learn >= floor)) if learn.size else math.nan,
        "share_in_band_of_all": float(np.sum(learn >= floor) / ok.size),
        "median_learnability": float(np.median(learn)) if learn.size else math.nan,
        "median_p": float(np.median(p)) if p.size else math.nan,
        "pooled_NOT_learnability_median": float(np.nanmedian(pool_p * (1 - pool_p)))
        if np.isfinite(pool_p).any()
        else math.nan,
    }
