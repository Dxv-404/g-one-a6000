"""Gate E11: no predator spawns within the achieved separation, and the level used is recorded.

Without a spawn separation a predator can appear adjacent to prey and the episode opens with a capture, which
biases every measurement downstream. The guard is a relaxation ladder rather than a single distance, because
a categorical draw over an all-zero mask does not raise under static shapes: it returns index zero, which may
be an obstacle or a cell touching prey, and the episode looks normal afterwards.

**This gate has two halves and the second is the one that rots quietly.** Checking the separation is easy and
would pass on an implementation that always relaxed to the loosest level. What makes the relaxation rate
meaningful is that the level is recorded honestly, so measurement M4 can report it and a configuration that
relaxes often can be excluded from comparison.

The tests below are ordered by what they defend against:

    separation holds            the property itself
    the ladder is exercised     coverage. A test that only ever sees level 0 never runs the relaxation path
    -1 is distinguishable       polarity. `argmax` over an all-False array returns 0, so a total failure of
                                the ladder would otherwise record as "achieved the nominal separation"
"""

import jax
import jax.numpy as jnp
import pytest

from configs.default import DEFAULT
from src.env import env as E
from src.env import resources
from src.env.entropy import PRNGSource
from src.env.perception import build_vis_table

P = DEFAULT.__class__(h=20, w=20, k=6, t_horizon=40, n_food_src=3, n_target_src=3)


def _static(params, seed=0, density=0.0):
    key = jax.random.key(seed)
    obstacles = jnp.zeros((params.h, params.w), bool)
    food_src, target_src, _ = resources.place_sources(key, obstacles, params)
    return E.StaticEnv(
        obstacles=obstacles,
        vis_table=build_vis_table(obstacles, params.r),
        food_src=food_src,
        target_src=target_src,
    )


def _min_separation(state):
    """Chebyshev distance from the predator to the nearest prey."""
    d = jnp.max(jnp.abs(state.pred_pos[:, None, :] - state.prey_pos[None, :, :]), axis=-1)
    return int(jnp.min(d))


@pytest.fixture(scope="module")
def static():
    return _static(P)


def test_no_predator_spawns_inside_the_achieved_separation(static):
    """The property E11 exists for, checked against the level actually recorded rather than the nominal one.

    Comparing against `d_spawn_relax[0]` would fail legitimately whenever the ladder relaxed, so the
    assertion has to read the recorded level. That is what makes the recording load-bearing rather than
    decorative: if it were wrong, this test would be checking the wrong threshold.
    """
    for seed in range(40):
        state = E.reset(PRNGSource(jax.random.key(seed)), static, P)
        level = int(state.spawn_relax_level)
        assert -1 <= level < len(P.d_spawn_relax), f"seed {seed} recorded an out-of-range level {level}"
        if level < 0:
            continue  # the ladder failed entirely; the next test is the one that owns this case
        required = P.d_spawn_relax[level]
        got = _min_separation(state)
        assert got >= required, (
            f"seed {seed} recorded relaxation level {level}, meaning a separation of {required}, but the "
            f"predator spawned {got} cells from the nearest prey"
        )


def test_the_recorded_level_is_the_tightest_feasible_one(static):
    """Recording *a* feasible level is not enough; it must be the first one on the ladder that fits.

    An implementation that always jumped to the loosest level would pass the separation test above on every
    seed and would silently report the environment as relaxing constantly, which is the signal M4 uses to
    decide a configuration is outside the valid envelope.
    """
    for seed in range(20):
        state = E.reset(PRNGSource(jax.random.key(seed)), static, P)
        level = int(state.spawn_relax_level)
        if level <= 0:
            continue
        tighter = P.d_spawn_relax[level - 1]
        free = ~static.obstacles.reshape(-1)
        h, w = static.obstacles.shape
        cells = jnp.stack(jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij"), axis=-1).reshape(-1, 2)
        d_to_prey = jnp.min(
            jnp.max(jnp.abs(cells[:, None, :] - state.prey_pos[None, :, :]), axis=-1), axis=-1
        )
        assert not bool(jnp.any(free & (d_to_prey >= tighter))), (
            f"seed {seed} relaxed to level {level} while level {level - 1}, separation {tighter}, still had "
            "a feasible cell. The ladder is skipping levels, so the recorded relaxation rate overstates how "
            "constrained the configuration really is"
        )


def test_the_relaxation_ladder_is_actually_exercised():
    """**Coverage.** A gate that only ever observes level 0 has never run the relaxation path.

    This is the third time in this project that a green assertion turned out to cover code that never
    executed, so the condition is constructed rather than hoped for. A crowded small grid forces relaxation:
    the ladder's head is 10, and on a 12 by 12 grid with 10 prey no cell is 10 away from all of them.
    """
    tight = DEFAULT.__class__(h=12, w=12, k=10, t_horizon=20, n_food_src=2, n_target_src=2)
    static = _static(tight)
    levels = {int(E.reset(PRNGSource(jax.random.key(s)), static, tight).spawn_relax_level) for s in range(30)}
    assert levels - {0, -1}, (
        f"across 30 seeds on a deliberately crowded grid the recorded levels were {sorted(levels)}. The "
        "relaxation path never ran, so every assertion about it in this file is inert."
    )


def test_a_fully_infeasible_ladder_records_minus_one_not_level_zero():
    """**Polarity, and the sharpest trap in this gate.**

    `jnp.argmax` over an all-False array returns 0. Without an explicit guard, a reset where NO level on the
    ladder was feasible records level 0, which reads as "achieved the nominal separation of 10" and is the
    most favourable possible interpretation of the worst possible case. M4's relaxation rate would then
    under-report exactly the configurations it exists to flag, and nothing anywhere would look wrong.

    A positive assertion cannot catch this, because level 0 is a legitimate value. Only asserting the
    negative, that this specific state is NOT recorded as 0, distinguishes them.
    """
    impossible = DEFAULT.__class__(h=5, w=5, k=8, t_horizon=20, n_food_src=1, n_target_src=1)
    reachable = max(impossible.h, impossible.w) - 1
    assert min(impossible.d_spawn_relax) > reachable, (
        f"this fixture is only meaningful if EVERY rung of the ladder exceeds the largest Chebyshev "
        f"distance the grid admits, {reachable}. The loosest rung is {min(impossible.d_spawn_relax)}, so "
        "infeasibility would depend on where the prey happened to land and this test would flake rather "
        "than fail. A change to d_spawn_relax could also make it silently inert."
    )
    static = _static(impossible)
    recorded = [
        int(E.reset(PRNGSource(jax.random.key(s)), static, impossible).spawn_relax_level) for s in range(15)
    ]
    assert all(v == -1 for v in recorded), (
        f"on a 5 by 5 grid with 8 prey, where no level of the ladder can be satisfied, reset recorded "
        f"{sorted(set(recorded))} instead of -1. Level 0 here would mean the run reports the nominal "
        "separation was achieved when the predator may have spawned adjacent to prey."
    )
