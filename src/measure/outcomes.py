"""Tier one: the category-free outcome test. Build step 11.

Governed by `docs/analysis-structure.md` 1.0.0.0.

**Does predation selection change what happens to the population, and does that change constitute
self-preservation?** This tier names no strategy. It is the question the project actually asks, and it
answers with or without any behaviour being identifiable.

**The statistic is the full outcome vector, not a single rate, because the components trade against each
other.** A lineage that hides more is predated less and starved more. Reporting predation alone would call
that success.

**Predated and starved are independent flags and both can be true**, which the specification does not spell
out and which changes the arithmetic: the three fractions do NOT sum to one. `EnvState` records them
separately on purpose, because deriving a single cause under a precedence rule under-counts predation among
starving agents, and those are exactly the ones most likely to be caught. So the overlap is reported rather
than resolved away, and the identity that does hold is asserted instead.

Sample is 8,192 prey outcomes per generation at target scale, so this is the only tier with power to spare.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Outcomes(NamedTuple):
    """Fractions over the prey that were evaluated. **These do not sum to one**; see `both` and the identity.

    `n` travels with them for the same reason a `Descriptor` carries its denominator: a fraction without its
    sample size can be quoted in a table with the sample size stripped.
    """

    predated: float
    starved: float
    survived: float
    both: float
    n: int

    def as_vector(self) -> np.ndarray:
        """The comparison vector, which is the first three. `both` is diagnostic rather than part of the
        outcome the tier compares, and `n` is not an outcome at all."""
        return np.array([self.predated, self.starved, self.survived], dtype=np.float64)


def outcomes(died_predated, died_starving, alive_at_end) -> Outcomes:
    """The outcome vector for one population.

    Takes three boolean arrays and nothing else, which keeps this a pure function of what happened rather
    than of any state object. The identity below is checked here rather than in a test, because it is a
    property of the arithmetic and a caller passing mismatched arrays should find out at the call site.
    """
    pred = np.asarray(died_predated).astype(bool).ravel()
    starv = np.asarray(died_starving).astype(bool).ravel()
    alive = np.asarray(alive_at_end).astype(bool).ravel()
    if not (pred.shape == starv.shape == alive.shape):
        raise ValueError(f"mismatched shapes: {pred.shape}, {starv.shape}, {alive.shape}")
    n = pred.size
    if n == 0:
        raise ValueError("no prey to score")

    # **The accounting identity.** Every prey either survived or died of at least one recorded cause. A prey
    # that is dead with no cause is a defect in `resolve_deaths`, and a prey that is alive with a cause is a
    # defect in the write-once rule. Neither is visible in any fraction.
    accounted = alive | pred | starv
    if not bool(np.all(accounted)):
        missing = int(np.sum(~accounted))
        raise ValueError(
            f"{missing} of {n} prey are neither alive nor recorded as having died of any cause. Every prey "
            "must be accounted for, and a death with no cause means the write-once cause recording did not "
            "fire"
        )
    if bool(np.any(alive & (pred | starv))):
        bad = int(np.sum(alive & (pred | starv)))
        raise ValueError(f"{bad} prey are alive at the horizon and also carry a death cause")

    return Outcomes(
        predated=float(pred.mean()),
        starved=float(starv.mean()),
        survived=float(alive.mean()),
        both=float((pred & starv).mean()),
        n=n,
    )


def from_state(state) -> Outcomes:
    """Convenience over an `EnvState`, so callers do not have to remember the three field names."""
    return outcomes(state.prey_died_predated, state.prey_died_starving, state.prey_alive)


def difference(evolved: Outcomes, naive: Outcomes) -> dict[str, float]:
    """Evolved minus naive, component by component. **The comparison the tier is built on.**

    The evolved and predator-naive lineages differ only in whether predation shaped them, so any difference
    here is attributable to predation selection. That is the conditioning a per-tick filter cannot do, and it
    is measured over 8,192 outcomes rather than over the 0.262 engaged ticks a conditional descriptor gets.

    Returned as a dict rather than an `Outcomes`, because a difference of fractions is not a set of fractions
    and typing it as one would invite it being read as a population.
    """
    d = evolved.as_vector() - naive.as_vector()
    return {
        "d_predated": float(d[0]),
        "d_starved": float(d[1]),
        "d_survived": float(d[2]),
        "n_evolved": evolved.n,
        "n_naive": naive.n,
    }
