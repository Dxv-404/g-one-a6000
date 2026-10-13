"""The genome vector: weights flattened to one float array per individual. Build step 10.

**Invariant I3 lives in this file's signatures, in the same way invariant I1 lives in `fitness`.**

    Weight-only inheritance. Hidden state resets to exactly zero at every episode start and is never
    inherited. Only weights cross generations.

`to_vector` takes the weight PyTree and nothing else. It is not given hidden state, an environment, an
episode, or a score, so no activation-derived quantity can enter the genome: there is no argument through
which one could arrive. That is the structural half of I3, and gate S8 checks the behavioural half, which is
that a fixed genome produces identical behaviour in episode one and episode two.

**Why this matters as much as I1 does.** The project's claim is about evolved instinct, not inherited
learning. If any activation carried across an episode boundary or across a generation, the result would be
about lifetime learning accumulated through a population, which is a different and much less interesting
claim, and CLAUDE.md notes that a violation produces no visible symptom whatsoever. Nothing in a learning
curve looks wrong when this breaks.

`ravel_pytree` rather than a hand-written flattener: it round-trips by construction, and a hand-written one
would need its own test for the property `ravel_pytree` already guarantees.
"""

from __future__ import annotations

from collections.abc import Callable

from jax import Array
from jax.flatten_util import ravel_pytree


def to_vector(weights: dict) -> Array:
    """Flatten a weight PyTree to one float vector. **Takes weights and nothing else, by design.**

    Widening this signature to accept hidden state, an episode, or anything activation-derived is invariant
    I3 broken, and it is lifted only by a decision in `decisions.md` with `promote: yes`.
    """
    return ravel_pytree(weights)[0]


def unflattener(template: dict) -> Callable[[Array], dict]:
    """The inverse for a given PyTree structure.

    The template supplies structure only, never values: `ravel_pytree` reads shapes and dtypes to build the
    unflattening function, and every number in the result comes from the vector. So passing a live genome as
    a template cannot leak that genome's values into another one.
    """
    return ravel_pytree(template)[1]


def n_genes(weights: dict) -> int:
    """Length of the genome vector. Derived from the weights rather than transcribed, because a transcribed
    parameter count is exactly what DEV-010 removed."""
    return int(to_vector(weights).size)
