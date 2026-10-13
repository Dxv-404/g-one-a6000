"""Invariant I1, made structural. Build step 10.

**This is the most constrained function in the repository and its signature is the constraint.**

    def fitness(prey_score: Array) -> Array

Energy, stamina, health, elapsed steps and predator distance are not in scope and cannot reach it, because
they are not passed to it. That is the whole mechanism: **a function that is not given the state cannot read
the state.** It is not a matter of discipline, a review checklist, or a call-graph analysis that three
documents described and none implemented.

**Why this matters more than any other line here.** The experiment asks whether self-preservation emerges
without being rewarded. If fitness could see survival time, energy, health, or distance from the predator,
then any self-protective behaviour that appeared would have been selected for directly, and the result would
answer nothing. Every anti-predator behaviour this project hopes to observe must be **instrumental**: worth
doing only because a dead agent collects no more targets.

**Widening this signature is the single most damaging change available in this repository, and it will look
harmless when it is proposed.** "Add a small survival term so it learns faster", "shape the reward away from
the predator", "penalise death". Each is on the refusal list in CLAUDE.md section 1, each destroys I1, and
each is lifted only by an explicit human decision recorded in `decisions.md` with `promote: yes`.

Gate S1 asserts the annotation, asserts the parameter count, asserts that nothing in the body reads a state
field, and asserts behaviourally that an agent surviving the full horizon with zero collected targets scores
exactly zero.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

#: Axes of `prey_score` that are reduced away. A genome is evaluated on `n_episodes` randomised episodes and
#: every prey in an environment is a clone of that genome, so both axes are repeated measurements of the same
#: individual and both are averaged. See D-038; the build spec states the episode average and is silent on
#: the prey axis.
_SAMPLE_AXES = (-2, -1)


def fitness(prey_score: Array) -> Array:
    """Mean collected-target count. **The only input is the score, by design.**

    `prey_score` has shape `(..., n_episodes, k)`: whatever leading axis the caller vmaps over, then the
    episodes a genome was evaluated on, then the prey within an environment. The return drops the last two.

    **The score is already "at the individual's own death or at the horizon"** because `prey_score` stops
    changing when an agent dies: `resolve_deaths` masks its contributions, and nothing adds to a dead agent's
    total. So no death-time bookkeeping is needed here, and more importantly none is WANTED here, since
    reading a death step would mean reading state.

    Dead agents are averaged in at whatever they collected before dying, which is the point. An agent that
    dies early scores less because it stopped collecting, not because dying was penalised.
    """
    return jnp.mean(prey_score, axis=_SAMPLE_AXES)
