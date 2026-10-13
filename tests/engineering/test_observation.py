"""The egocentric observation: channel semantics, bounds, and what visibility does and does not gate.

**The load-bearing assertion here is that visibility gates entity channels but not terrain**, and that
channel 6 carries the visibility mask itself. Without channel 6, "no predator there" and "cannot see there"
are the same vector of zeros, and an agent could not learn to treat them differently. That distinction is
what makes occlusion-seeking a strategy the network can express at all.

Coverage is asserted: a test that never places a predator in the patch cannot detect a predator-channel bug.
"""

import jax
import jax.numpy as jnp
import pytest

from configs.default import DEFAULT as P
from src.env import observation as obs_mod
from src.env.observation import build_observation
from src.env.perception import build_vis_table

H = W = 16
K = 3
CENTRE = (8, 8)


def _scene(obstacles=None, food=None, target=None, prey=None, pred=(2, 2), self_posture=0):
    """Build the arguments for one observation. Everything defaults to empty so each test sets one thing."""
    obstacles = jnp.zeros((H, W), bool) if obstacles is None else obstacles
    food_grid = jnp.zeros((H, W), jnp.int32) if food is None else food
    target_grid = jnp.zeros((H, W), jnp.int32) if target is None else target
    prey_pos = jnp.array(prey if prey is not None else [[8, 8], [0, 0], [15, 15]], jnp.int32)
    return dict(
        obstacles=obstacles,
        food_grid=food_grid,
        target_grid=target_grid,
        vis_table=build_vis_table(obstacles, P.r),
        self_pos=jnp.array(CENTRE, jnp.int32),
        self_posture=jnp.int32(self_posture),
        self_pools=jnp.array([P.e_max, P.s_max, P.h_max], jnp.float32),
        prey_pos=prey_pos,
        prey_alive=jnp.ones(prey_pos.shape[0], bool),
        prey_moved=jnp.zeros(prey_pos.shape[0], bool),
        prey_posture=jnp.zeros(prey_pos.shape[0], jnp.int32),
        pred_pos=jnp.array(pred, jnp.int32),
        pred_moved=jnp.asarray(False),
        pred_appearance=jnp.full((P.a_appearance,), 0.5, jnp.float32),
        self_index=jnp.int32(0),
        params=P,
    )


def _patch(o):
    """The observation's patch part, reshaped to (S, S, C) so a cell can be addressed by offset."""
    return o[: P.s_patch * P.s_patch * P.c_channels].reshape(P.s_patch, P.s_patch, P.c_channels)


def _cell(o, dr, dc):
    return _patch(o)[dr + P.r, dc + P.r]


def test_observation_has_exactly_obs_dim_entries():
    o = build_observation(**_scene())
    assert o.shape == (P.obs_dim,) == (1140,)
    assert o.dtype == jnp.float32


def test_scalars_are_the_normalised_pools_then_own_posture_one_hot():
    s = _scene(self_posture=2)
    s["self_pools"] = jnp.array([P.e_max / 2, P.s_max, 0.0], jnp.float32)
    o = build_observation(**s)
    scalars = o[-P.N_SCALARS :]
    assert float(scalars[0]) == pytest.approx(0.5)
    assert float(scalars[1]) == pytest.approx(1.0)
    assert float(scalars[2]) == pytest.approx(0.0)
    assert [float(x) for x in scalars[3:]] == [0.0, 0.0, 1.0], "own posture one-hot is wrong"


def test_out_of_bounds_reads_as_obstacle_and_nothing_else():
    """The boundary behaves like a wall, so a missing bounds check fails closed."""
    s = _scene()
    s["self_pos"] = jnp.array([0, 0], jnp.int32)
    o = build_observation(**s)
    off = _cell(o, -P.r, -P.r)  # well outside the grid
    assert float(off[obs_mod.CH_OBSTACLE]) == 1.0, "out of bounds must read as obstacle"
    for ch in (obs_mod.CH_FOOD, obs_mod.CH_TARGET, obs_mod.CH_PREY, obs_mod.CH_PREDATOR):
        assert float(off[ch]) == 0.0, f"channel {ch} should be zero out of bounds"


def test_terrain_is_reported_even_where_sight_is_blocked():
    """Obstacles are static and an agent has been there. Terrain is not gated on visibility."""
    obstacles = jnp.zeros((H, W), bool).at[8, 9].set(True).at[8, 10].set(True)
    o = build_observation(**_scene(obstacles=obstacles))
    assert float(_cell(o, 0, 2)[obs_mod.CH_OBSTACLE]) == 1.0, "an occluded obstacle should still be reported"
    assert float(_cell(o, 0, 2)[obs_mod.CH_VISIBLE]) == 0.0, "that cell should not be marked visible"


def test_an_entity_behind_a_wall_is_not_reported_but_the_cell_is_still_described():
    """The distinction the experiment needs: absent and unseen must not be the same vector.

    Channel 6 is what separates them. Without it both are all-zero and occlusion-seeking is not expressible.
    """
    obstacles = jnp.zeros((H, W), bool).at[8, 9].set(True)
    s = _scene(obstacles=obstacles, pred=(8, 10))
    o = build_observation(**s)
    behind = _cell(o, 0, 2)
    assert float(behind[obs_mod.CH_PREDATOR]) == 0.0, "a predator behind a wall was reported"
    assert float(behind[obs_mod.CH_VISIBLE]) == 0.0, "the occluded cell must be marked not visible"

    empty_visible = _cell(o, -2, 0)
    assert float(empty_visible[obs_mod.CH_PREDATOR]) == 0.0
    assert float(empty_visible[obs_mod.CH_VISIBLE]) == 1.0, (
        "an empty visible cell must differ from an occluded one, or absence and blindness are identical"
    )


def test_a_visible_predator_is_reported_with_its_appearance():
    s = _scene(pred=(8, 10))
    o = build_observation(**s)
    c = _cell(o, 0, 2)
    assert float(c[obs_mod.CH_PREDATOR]) == 1.0, "a visible predator was not reported"
    for k in range(P.a_appearance):
        assert float(c[obs_mod.CH_APPEARANCE_FIRST + k]) == pytest.approx(0.5), (
            "the appearance vector should be written alongside the predator"
        )


def test_other_prey_are_reported_and_the_observer_is_not():
    s = _scene(prey=[[8, 8], [8, 10], [0, 0]])
    o = build_observation(**s)
    assert float(_cell(o, 0, 2)[obs_mod.CH_PREY]) == 1.0, "another prey in view was not reported"
    assert float(_cell(o, 0, 0)[obs_mod.CH_PREY]) == 0.0, (
        "the observer wrote itself into its own patch; its posture already reaches the network as a scalar"
    )


def test_a_dead_prey_is_not_reported():
    s = _scene(prey=[[8, 8], [8, 10], [0, 0]])
    s["prey_alive"] = jnp.array([True, False, True])
    o = build_observation(**s)
    assert float(_cell(o, 0, 2)[obs_mod.CH_PREY]) == 0.0, "a dead prey was reported as present"


def test_the_moved_channel_tracks_the_entity_not_the_cell():
    s = _scene(prey=[[8, 8], [8, 10], [0, 0]])
    s["prey_moved"] = jnp.array([False, True, False])
    o = build_observation(**s)
    assert float(_cell(o, 0, 2)[obs_mod.CH_MOVED]) == 1.0, "a moved prey did not set the moved channel"
    assert float(_cell(o, 0, 1)[obs_mod.CH_MOVED]) == 0.0, "an empty cell has nothing that moved"


def test_posture_channels_are_one_hot_per_occupied_cell():
    s = _scene(prey=[[8, 8], [8, 10], [0, 0]])
    s["prey_posture"] = jnp.array([0, 2, 0], jnp.int32)
    o = build_observation(**s)
    c = _cell(o, 0, 2)
    one_hot = [float(c[obs_mod.CH_POSTURE_FIRST + k]) for k in range(3)]
    assert one_hot == [0.0, 0.0, 1.0], f"prone prey should set channel 9, got {one_hot}"


def test_food_and_target_are_separate_channels():
    food = jnp.zeros((H, W), jnp.int32).at[8, 9].set(1)
    target = jnp.zeros((H, W), jnp.int32).at[8, 7].set(1)
    o = build_observation(**_scene(food=food, target=target))
    assert float(_cell(o, 0, 1)[obs_mod.CH_FOOD]) == 1.0
    assert float(_cell(o, 0, 1)[obs_mod.CH_TARGET]) == 0.0
    assert float(_cell(o, 0, -1)[obs_mod.CH_TARGET]) == 1.0
    assert float(_cell(o, 0, -1)[obs_mod.CH_FOOD]) == 0.0


def test_coverage_every_channel_class_was_actually_exercised():
    """Power check. The assertions above are only meaningful if each channel can be non-zero at all."""
    obstacles = jnp.zeros((H, W), bool).at[8, 6].set(True)
    food = jnp.zeros((H, W), jnp.int32).at[8, 9].set(1)
    target = jnp.zeros((H, W), jnp.int32).at[8, 7].set(1)
    s = _scene(obstacles=obstacles, food=food, target=target, prey=[[8, 8], [8, 10], [7, 8]], pred=(9, 8))
    s["prey_moved"] = jnp.array([False, True, False])
    s["prey_posture"] = jnp.array([0, 1, 2], jnp.int32)
    o = build_observation(**s)
    patch = _patch(o)

    for ch, name in (
        (obs_mod.CH_OBSTACLE, "obstacle"),
        (obs_mod.CH_FOOD, "food"),
        (obs_mod.CH_TARGET, "target"),
        (obs_mod.CH_PREDATOR, "predator"),
        (obs_mod.CH_PREY, "prey"),
        (obs_mod.CH_MOVED, "moved"),
        (obs_mod.CH_VISIBLE, "visible"),
    ):
        assert float(jnp.max(patch[:, :, ch])) > 0.0, f"channel {name} was never set, so it is untested"

    posture_seen = float(jnp.max(patch[:, :, obs_mod.CH_POSTURE_FIRST : obs_mod.CH_POSTURE_FIRST + 3]))
    assert posture_seen > 0.0, "no posture channel was ever set"
    appearance_seen = float(jnp.max(patch[:, :, obs_mod.CH_APPEARANCE_FIRST :]))
    assert appearance_seen > 0.0, "the appearance vector was never written"


def test_the_observation_is_vmappable_over_agents():
    """Single-agent by design; the batch comes from vmap. Asserted so a shape error surfaces here."""
    s = _scene(prey=[[8, 8], [8, 10], [7, 8]])
    over_agents = jax.vmap(
        lambda pos, idx: build_observation(**{**s, "self_pos": pos, "self_index": idx}),
        in_axes=(0, 0),
    )
    out = over_agents(s["prey_pos"], jnp.arange(s["prey_pos"].shape[0]))
    assert out.shape == (3, P.obs_dim)
