"""Properties of the hand-coded baseline itself, as opposed to the scientific gates that use it.

**This file exists because two of its assertions were claimed and absent.** `src/agent/baseline.py` justified
duplicating `DIRECTIONS` by saying the ordering "is asserted equal in the tests", and named a file that did
not exist. The orderings did in fact match, so nothing was broken; what was missing was anything that would
notice if they stopped matching. Third and fourth instances of a stated fact with no consumer. See DEV-025.

The jam property is the one that matters. It is a **property test that consults behaviour**, not a comparison
against a second description, because the defect it guards was invisible to every score-level assertion:
gates S2 and S9 both passed while the forager spent 89.9 percent of its ticks emitting a move into a wall.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import DEFAULT
from src.agent import baseline
from src.agent import network as net
from src.env import env as E
from src.env import grid, resources
from src.env.entropy import PRNGSource
from src.env.perception import build_vis_table

#: Smaller than the S2 world and deliberately so: a tighter grid puts every agent nearer a boundary, which
#: is the condition the jam needed. This makes the test faster AND stricter, which is the rare direction.
P = DEFAULT.__class__(h=20, w=20, t_horizon=100)


def _policy(weights, obs, hidden, step_index):
    del weights
    return jax.vmap(lambda o: baseline.forager(o, P, step_index))(obs), hidden


def _episode(seed: int):
    ob = grid.make_obstacle_grid(jax.random.key(seed), P.h, P.w, P.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, P)
    static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, P.r), food_src=fs, target_src=ts)
    w = net.init_params(jax.random.key(seed), P)
    return E.run_episode(
        jax.random.key(1000 + seed),
        w,
        static,
        PRNGSource(jax.random.key(seed)),
        P,
        record=True,
        policy=_policy,
    )


def test_the_direction_table_matches_the_environment():
    """**The claim that made duplicating `DIRECTIONS` safe, which until now did not exist.**

    `baseline.DIRECTIONS` is a copy of `grid.DIRECTIONS`, kept so the agent package does not import the
    environment. The copy is only safe while the orderings agree, because `forager` returns an INDEX into it
    and that index is used directly as an action. A reordering on either side would silently turn every
    movement decision into a different movement.
    """
    b = [tuple(int(v) for v in d) for d in np.asarray(baseline.DIRECTIONS)]
    g = [tuple(int(v) for v in d) for d in np.asarray(grid.DIRECTIONS)]
    assert b == g, (
        f"baseline.DIRECTIONS {b} does not match grid.DIRECTIONS {g}. The forager returns an index into its "
        "own copy and that index is used as an action, so a disagreement silently remaps every move."
    )


def test_the_exploration_hold_is_the_derived_value():
    """`EXPLORE_HOLD` is documented as derived from the patch width. This is what makes that a derivation
    rather than an assertion that it was derived, and it is what fails if `s_patch` moves and the constant
    does not."""
    assert baseline.EXPLORE_HOLD == P.s_patch - 1, (
        f"EXPLORE_HOLD is {baseline.EXPLORE_HOLD} but the patch width is {P.s_patch}, so the heading no "
        "longer translates the patch by its own width and successive exploration legs overlap."
    )


#: Call duration measured with `--durations=0` at build step 9: 8.03 s running the file alone, 5.06 s inside
#: the full engineering suite, where the compilation cache is already warm for these shapes. Both are over
#: the five-second line so the mark needs no judgement, which is the same spread Q-019 met at 7.85 against
#: 5.4 and is why the measurement condition is recorded rather than just the number.
#:
#: The property needs whole episodes rather than single ticks because the defect is a jam that only shows
#: over time: a single tick emitting a blocked move looks like an ordinary collision.
@pytest.mark.slow
def test_the_forager_never_emits_a_move_into_an_obstacle():
    """**The gate for DEV-024, and it asserts the mechanism rather than the score.**

    The first forager stepped east whenever nothing was visible, and stepped toward a visible item without
    checking the step was legal. Both jam permanently: the agent does not move, so the patch is identical
    next tick, so it emits the same blocked direction for the rest of the episode. S2 and S9 both passed
    throughout, because a jammed forager still outscores a random one by a wide margin.

    A single blocked emitted move is a defect here, not a tolerance. The policy has the obstacle channel and
    out-of-bounds reads as an obstacle, so it can always see which of the eight neighbours are enterable, and
    a single step's legality is exactly that. There is no case where it must guess.
    """
    blocked_total = 0
    for seed in range(3):
        _, _, traj = _episode(seed)
        emitted = np.asarray(traj.action_emitted)
        moved = np.asarray(traj.prey_moved).astype(bool)
        alive = np.asarray(traj.prey_alive).astype(bool)
        is_walk = emitted <= 7
        blocked_total += int(np.sum(is_walk & ~moved & alive))

    assert blocked_total == 0, (
        f"the forager emitted a walk that did not move the agent on {blocked_total} alive ticks. Every such "
        "tick is an agent pressed against an obstacle it can see, and because the policy is deterministic "
        "and the observation does not change, it will press against it for the rest of the episode."
    )


def test_the_forager_explores_rather_than_settling():
    """The jam's visible consequence, kept as a separate assertion because it fails for causes the blocked
    check would not catch, such as a heading that rotates without ever leaving a pocket.

    Ten of sixty-four prey never left their spawn cell under the original policy.
    """
    _, _, traj = _episode(0)
    pos = np.asarray(traj.prey_pos)
    alive = np.asarray(traj.prey_alive).astype(bool)
    visited = [len({tuple(p) for p in pos[alive[:, i], i]}) for i in range(P.k) if alive[:, i].any()]
    assert min(visited) > 1, (
        f"a prey visited {min(visited)} distinct cells across the episode, so it never moved at all. "
        f"per prey: {visited}"
    )


def test_the_exploration_heading_rotates():
    """**Added because a falsification break survived**, and the survivor analysis is worth keeping.

    Restoring the original constant east heading no longer jams the agent, because the blocked-direction
    guard added in DEV-024 substitutes a free direction whatever the heading was. So the break stopped being
    the defect it was written for. But it is not harmless: measured over 8 seeds at full scale it takes the
    mean score from 64.62 to 41.00 and mean cells visited from 161.7 to 74.2, with blocked ticks staying at
    0.0 percent throughout. A real regression that every existing assertion was blind to.

    That is two of the three survivor causes at once: the CLAIM was no longer true, and the gate set also
    genuinely could not see what the break still did.

    **Asserted on the mechanism at unit level rather than as a coverage floor over an episode**, because a
    coverage floor would need a number nobody can derive, and an unjustified threshold is a check whose
    strictness nobody chose. Here there is nothing to choose: on an empty patch the emitted direction is the
    exploration heading and nothing else, so it either varies with the tick or it does not.

    **The tick range is `t_horizon` and deliberately NOT `8 * EXPLORE_HOLD`, which is what it said first.**
    A loop bound derived from the constant under test scales with the break: setting `EXPLORE_HOLD` to 100000
    in falsification turned 64 iterations into 800000 and hung the run for twelve minutes, twice, and the
    second time it was diagnosed only because the falsifier printed unbuffered progress. The property that
    matters is anyway the stronger one: the heading completes a rotation WITHIN AN EPISODE, which is what
    makes exploration cover ground in the time the agent has.
    """
    empty = jnp.zeros((P.obs_dim,))
    emitted = jax.vmap(lambda t: baseline.forager(empty, P, t))(jnp.arange(P.t_horizon))
    seen = sorted({int(v) for v in emitted})
    assert len(seen) == 8, (
        f"within a single episode the forager emitted {seen} on an empty patch, {len(seen)} distinct "
        "directions rather than 8. The exploration heading is not rotating, so the policy explores in a "
        "biased direction and covers roughly half the ground."
    )
