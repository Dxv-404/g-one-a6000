"""Hand-coded policies for gates S2 and S9. Build step 9.

    forager         a competent hand-coded forager. Gate S2's validating baseline
    random_policy   uniform random action selection. Gate S9's floor

**These are measurement instruments and must never touch the evolved lineage.** CLAUDE.md section 1 refuses
"pretrained weights, imitation from a scripted policy, behaviour cloning, or any human-derived data", and a
hand-coded forager is exactly the human-derived policy that refusal names. The protection here is structural
rather than a matter of discipline:

**Both functions map an observation to an action and cannot produce weights.** There is no path from either
to `init_params`, no shared representation, and nothing an evolutionary loop could initialise from. A
function that returns an integer action cannot seed a genome, in the same way that a fitness function not
given the state cannot read the state. `test_s2_s9_baselines.py::test_the_baseline_module_cannot_produce_weights` asserts the module exposes
no weight-shaped output.

**The forager ignores the predator entirely, and that is deliberate.** Gate S2 asks whether the environment
can express and reward COMPETENCE AT THE TASK, and the task is collecting the forage target. A baseline that
also evaded the predator would be measuring survival strategy, which is the thing the experiment exists to
discover rather than to demonstrate. So this policy reads the food, target and obstacle channels and never
reads channel 3.

That also keeps the baseline honest as a floor: if a forager that completely ignores predation still beats
random by a wide margin, the margin is attributable to foraging competence alone.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from src.env import dynamics

#: Patch channel indices, from `src/env/observation.py`. Only these three are read.
CH_OBSTACLE = 0
CH_FOOD = 1
CH_TARGET = 2

#: The eight move directions, in the same order as `env.DIRECTIONS`, which is the order the action indices
#: use. Duplicated deliberately rather than imported: importing from `src.env.env` would make the agent
#: depend on the environment module. **The ordering is asserted equal in
#: `tests/engineering/test_baseline.py`**, which is what makes the duplication safe. That assertion did
#: not exist until DEV-025: this comment claimed it did, and the claim was the only thing holding the
#: duplication up.
DIRECTIONS = jnp.array([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], jnp.int32)


def random_policy(key: Array, n_actions: int) -> Array:
    """Uniform over ALL actions, including the ones that are masked unavailable.

    **Uniform over the emitted action, not over the executed one.** `mask_actions` converts blocked actions
    to `STAY`, so a policy uniform over emitted actions produces a distribution over executed actions that is
    biased toward staying. That is the correct floor: it is what a policy with no information does in THIS
    environment, and correcting for the mask would be giving the floor knowledge the agents do not have.

    This is deliberately NOT the same distribution as an untrained network. Measured, an untrained network
    under deterministic argmax emits a near-constant action and moves on about 2 percent of ticks, so
    generation zero may start BELOW this floor. Q-025.
    """
    return jax.random.randint(key, (), 0, n_actions).astype(jnp.int32)


def _patch(obs: Array, s_patch: int, n_channels: int) -> Array:
    """The egocentric patch, reshaped out of the flat observation."""
    return obs[: s_patch * s_patch * n_channels].reshape(s_patch, s_patch, n_channels)


#: Ticks an unobstructed explorer commits to a heading before rotating. **A gate threshold**: it sits in a
#: measurement instrument whose score is a reported margin, so it is chain-traced like one rather than left
#: as a bare literal. See `gate-thresholds.md`.
#:
#: Derived, not picked: hold the heading long enough to translate the patch by its own width, so successive
#: exploration legs survey disjoint ground. That is `s_patch - 1`, which is 8.
#:
#: **Not chosen by maximising the score, and the sweep is recorded so that can be checked.** Total across 8
#: seeds: hold 1 gives 58.2, 2 gives 69.0, 4 gives 57.9, 8 gives 64.6, 16 gives 37.4, 32 gives 40.9, and a
#: hold exceeding the horizon gives 29.0. The derived value is not the maximum. Everything from 1 to 8 sits
#: within about one standard error of the mean, so the constant is not load-bearing inside its principled
#: range, and only values well above the patch width degrade the instrument.
EXPLORE_HOLD = 8


def forager(obs: Array, params, step_index: Array) -> Array:
    """A competent hand-coded forager. Reads food, target and obstacles; never reads the predator channel.

    The rule, in priority order:

        1. standing on a forage target or food   -> PICK
        2. a target visible in the patch          -> step toward the nearest one
        3. food visible in the patch              -> step toward the nearest one
        4. nothing visible                        -> step along the current exploration heading

    and then, over all of 2 to 4, **the emitted direction is replaced by the first free direction whenever
    the desired one is blocked**. That last clause is the whole of DEV-024 and is why it is stated as
    applying to every case rather than only to case 4.

    **Targets outrank food** because `prey_score` counts collected targets and food only refills energy. A
    baseline that preferred food would be competent at staying alive and incompetent at the task fitness
    actually measures, which would understate the margin S2 is looking for.

    **The policy must be deterministic given its inputs**, so that the baseline's score does not depend on
    its own draws and the comparison against the random floor is not partly a comparison of two random
    policies. The first version delivered determinism with a CONSTANT heading, which is degenerate against a
    solid grid: an agent with an obstacle or a boundary to its east emitted east into it for all 400 ticks.
    Determinism was the requirement and constancy was the implementation, and only the requirement had ever
    been justified. See DEV-024.

    **No emitted direction is ever one the patch reports blocked, and that is structural rather than
    approximate.** Out-of-bounds cells read as obstacles in `CH_OBSTACLE`, so the grid boundary is covered
    without a separate bounds check, and a single step's legality in `grid.legal_move` is exactly
    "destination in bounds and not an obstacle". Sprints have an intermediate-cell rule and this policy does
    not sprint. Measured after the change: 0.0 percent of emitted moves blocked, against 89.9 before.
    """
    # `c_channels` is a derived property on EnvParams; recomputing the sum here would be a second copy that
    # could disagree with it.
    s, c = params.s_patch, params.c_channels
    patch = _patch(obs, s, c)
    centre = s // 2

    on_target = patch[centre, centre, CH_TARGET] > 0
    on_food = patch[centre, centre, CH_FOOD] > 0

    # Chebyshev distance from the centre to every patch cell, with the centre itself pushed out of contention.
    yy, xx = jnp.meshgrid(jnp.arange(s), jnp.arange(s), indexing="ij")
    dy, dx = yy - centre, xx - centre
    dist = jnp.maximum(jnp.abs(dy), jnp.abs(dx))
    far = jnp.int32(s * s)
    reachable = ~(patch[:, :, CH_OBSTACLE] > 0) & (dist > 0)

    def nearest(channel: int) -> tuple[Array, Array, Array]:
        present = (patch[:, :, channel] > 0) & reachable
        scored = jnp.where(present, dist, far)
        flat = jnp.argmin(scored)
        return jnp.any(present), dy.reshape(-1)[flat], dx.reshape(-1)[flat]

    has_target, t_dy, t_dx = nearest(CH_TARGET)
    has_food, f_dy, f_dx = nearest(CH_FOOD)

    # Which of the eight adjacent cells can actually be entered.
    free = patch[centre + DIRECTIONS[:, 0], centre + DIRECTIONS[:, 1], CH_OBSTACLE] == 0

    # The exploration heading, and the escape hatch for a blocked desired direction. Scanning from a
    # tick-derived offset rather than always from direction zero stops a blocked agent from resolving to the
    # same substitute every tick, which is the constant-heading failure in another costume.
    order = ((step_index // EXPLORE_HOLD) % 8 + jnp.arange(8)) % 8
    # If every direction is blocked the agent is entombed and the choice cannot matter.
    first_free = order[jnp.argmax(free[order])]

    step_dy = jnp.where(has_target, t_dy, jnp.where(has_food, f_dy, DIRECTIONS[first_free, 0]))
    step_dx = jnp.where(has_target, t_dx, jnp.where(has_food, f_dx, DIRECTIONS[first_free, 1]))

    want = jnp.stack([jnp.sign(step_dy), jnp.sign(step_dx)])
    match = jnp.all(DIRECTIONS == want[None, :], axis=-1)
    desired = jnp.argmax(match).astype(jnp.int32)

    return jnp.where(
        on_target | on_food,
        jnp.int32(dynamics.ACTION_PICK),
        jnp.where(free[desired], desired, first_free),
    )
