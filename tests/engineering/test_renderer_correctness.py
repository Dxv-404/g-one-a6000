"""The renderer's correctness test. This is the point of the renderer, not an addendum to it.

**An unverified renderer is worse than no renderer**, because a wrong one is trusted: obstacles drawn
transposed, or agent positions off by one, would make you confident about a world that is not the world being
simulated, and it would corroborate whatever you already believed. So every assertion here is against the
rendered array, never against how the picture looks.

### Every quantity below was measured before it was asserted on

D-022: falsifying by breaking the code and checking the gate goes red assumes the gate can observe what is
broken. The renderer is the artifact most likely to escape that, because the picture looks right either way.
So each candidate was rendered across a pair of states differing in exactly one way, and confirmed to differ,
before any assertion was written:

```
obstacle (2,5) vs (5,2)                    differs, 288 px
render(obstacles) vs render(obstacles.T)   differs, 288 px
prey (1,7) vs (7,1)                        differs, 128 px
prey alive vs dead, same cell              differs,  64 px
predator over prey, same cell              differs,  64 px
food present vs absent                     differs,  36 px
block(2,5) vs block(2,6)                   differs, off-by-one detectable
SYMMETRIC layout under transpose           IDENTICAL
```

**That last line is why `test_the_fixture_is_not_transpose_symmetric` exists.** A layout symmetric about the
diagonal renders identically to its own transpose, so a transposition falsification run against it would pass
and the whole transpose check would be inert while looking thorough. The fixture's asymmetry is a
precondition of the test rather than an incidental property of it, so it is asserted rather than assumed.
"""

import numpy as np
import pytest

from src.render.frame import CELL_PX, CHANNEL, cell_block, render_state

H = W = 9

# Deliberately asymmetric about the diagonal. See the module docstring.
OBSTACLE_RC = (2, 5)
PREY_RC = (1, 7)
PRED_RC = (6, 3)
FOOD_RC = (8, 0)


@pytest.fixture(scope="module")
def state():
    obstacles = np.zeros((H, W), dtype=bool)
    obstacles[OBSTACLE_RC] = True
    food = np.zeros((H, W), dtype=np.int8)
    food[FOOD_RC] = 1
    return dict(
        obstacles=obstacles,
        food_grid=food,
        prey_pos=np.array([PREY_RC], dtype=np.int32),
        prey_alive=np.array([True]),
        pred_pos=np.array([PRED_RC], dtype=np.int32),
    )


def test_the_fixture_is_not_transpose_symmetric(state):
    """A precondition of the transpose test, asserted rather than assumed.

    Measured: a layout symmetric about the diagonal renders identically to its transpose. If this fixture
    were symmetric, `test_a_transposed_render_is_detected` would pass against a transposed renderer and the
    check would be inert while looking thorough.
    """
    obstacles = state["obstacles"]
    assert not np.array_equal(obstacles, obstacles.T), "the fixture layout is transpose-symmetric"
    for name, rc in (("obstacle", OBSTACLE_RC), ("prey", PREY_RC), ("predator", PRED_RC), ("food", FOOD_RC)):
        assert rc[0] != rc[1], f"{name} sits on the diagonal at {rc}, so its transpose is itself"


def test_shape_is_cells_times_cell_size(state):
    img = render_state(**state)
    assert img.shape == (H * CELL_PX, W * CELL_PX, 3)
    assert img.dtype == np.float32
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


def test_an_obstacle_lands_on_its_own_pixel_block(state):
    """The core correspondence: grid (r, c) maps to the block at row r, column c."""
    img = render_state(**state)
    block = cell_block(img, *OBSTACLE_RC)
    assert np.allclose(block, CHANNEL["obstacle"]), "the obstacle block is not the obstacle colour"


def test_the_transposed_cell_is_not_painted(state):
    """The assertion that actually catches a transpose, stated as the negative.

    Checking only that (2,5) is an obstacle would pass on a renderer that painted both (2,5) and (5,2).
    Asserting that (5,2) is untouched is what makes the correspondence one-to-one.
    """
    img = render_state(**state)
    r, c = OBSTACLE_RC
    assert np.allclose(cell_block(img, c, r), CHANNEL["background"]), (
        f"cell {(c, r)} is painted, so rows and columns are swapped somewhere"
    )


def test_a_transposed_render_is_detected(state):
    """Render the transposed world and confirm it produces a different image.

    This is the falsification built into the suite rather than run by hand once: if the renderer ever becomes
    transpose-invariant, this goes red without anyone having to remember to check.
    """
    img = render_state(**state)
    flipped = dict(state)
    flipped["obstacles"] = state["obstacles"].T
    flipped["food_grid"] = state["food_grid"].T
    flipped["prey_pos"] = state["prey_pos"][:, ::-1]
    flipped["pred_pos"] = state["pred_pos"][:, ::-1]
    assert not np.array_equal(img, render_state(**flipped)), (
        "a transposed world rendered identically, so the renderer cannot distinguish rows from columns"
    )


def test_an_off_by_one_neighbour_is_untouched(state):
    """The neighbouring block must be background, which is what catches an off-by-one in the block arithmetic."""
    img = render_state(**state)
    r, c = OBSTACLE_RC
    assert np.allclose(cell_block(img, r, c + 1), CHANNEL["background"])
    assert np.allclose(cell_block(img, r + 1, c), CHANNEL["background"])


def test_block_boundaries_are_exact(state):
    """Every pixel of the obstacle block is obstacle, and the pixels either side of it are not.

    A block drawn one pixel too wide would satisfy a centre-pixel check and fail this.
    """
    img = render_state(**state)
    r, c = OBSTACLE_RC
    assert np.allclose(
        img[r * CELL_PX : (r + 1) * CELL_PX, c * CELL_PX : (c + 1) * CELL_PX], CHANNEL["obstacle"]
    )
    assert not np.allclose(img[r * CELL_PX, c * CELL_PX - 1], CHANNEL["obstacle"])
    assert not np.allclose(img[r * CELL_PX, (c + 1) * CELL_PX], CHANNEL["obstacle"])


def test_prey_and_predator_land_on_their_own_cells(state):
    img = render_state(**state)
    mid = CELL_PX // 2
    assert np.allclose(cell_block(img, *PREY_RC)[mid, mid], CHANNEL["prey"])
    assert np.allclose(cell_block(img, *PRED_RC)[mid, mid], CHANNEL["predator"])


def test_a_dead_prey_is_drawn_differently(state):
    """Dead agents keep their array slots, so the render must distinguish them or it misrepresents the state."""
    alive = render_state(**state)
    dead_state = dict(state, prey_alive=np.array([False]))
    dead = render_state(**dead_state)
    mid = CELL_PX // 2
    assert np.allclose(cell_block(alive, *PREY_RC)[mid, mid], CHANNEL["prey"])
    assert np.allclose(cell_block(dead, *PREY_RC)[mid, mid], CHANNEL["prey_dead"])
    assert not np.array_equal(alive, dead)


def test_the_predator_is_drawn_over_prey_on_a_shared_cell(state):
    """Draw order is load-bearing: a capture must be visible rather than hidden under the prey."""
    shared = dict(state, prey_pos=np.array([PRED_RC], dtype=np.int32))
    img = render_state(**shared)
    mid = CELL_PX // 2
    assert np.allclose(cell_block(img, *PRED_RC)[mid, mid], CHANNEL["predator"]), (
        "prey was drawn over the predator, so a capture would be invisible"
    )


def test_a_standing_resource_is_drawn(state):
    img = render_state(**state)
    mid = CELL_PX // 2
    assert np.allclose(cell_block(img, *FOOD_RC)[mid, mid], CHANNEL["food"])
    empty = render_state(**dict(state, food_grid=np.zeros((H, W), dtype=np.int8)))
    assert not np.allclose(cell_block(empty, *FOOD_RC)[mid, mid], CHANNEL["food"])


def test_obstacles_alone_is_enough_to_render():
    """Usable from build step 1 onward. A renderer needing a complete state cannot check an incomplete one."""
    obstacles = np.zeros((H, W), dtype=bool)
    obstacles[OBSTACLE_RC] = True
    img = render_state(obstacles)
    assert img.shape == (H * CELL_PX, W * CELL_PX, 3)
    assert np.allclose(cell_block(img, *OBSTACLE_RC), CHANNEL["obstacle"])


def test_the_visibility_overlay_respects_obstacles():
    """The overlay exists to make an occlusion bug obvious, so it must not tint through a wall.

    Built as a real check rather than decoration: a wall with visible cells behind it is wrong in a way no
    numeric gate was looking for, which is the whole argument for having a renderer.
    """
    import jax

    from src.env.grid import make_obstacle_grid
    from src.env.perception import build_vis_table

    # Kept as a JAX array for `build_vis_table`, which indexes it with traced indices inside a vmap. A NumPy
    # array here raises TracerArrayConversionError rather than silently doing the wrong thing, which is the
    # good failure mode. The renderer takes either.
    obstacles_j = make_obstacle_grid(jax.random.key(0), 15, 15, 0.2)
    vis = np.array(build_vis_table(obstacles_j, 4))
    obstacles = np.array(obstacles_j)
    prey = np.array([[7, 7]], dtype=np.int32)
    img = render_state(obstacles, prey_pos=prey, focus=0, vis_table=vis, r=4)

    for rr in range(15):
        for cc in range(15):
            if obstacles[rr, cc]:
                assert np.allclose(cell_block(img, rr, cc), CHANNEL["obstacle"]), (
                    f"obstacle at {(rr, cc)} was tinted by the visibility overlay"
                )


def test_coverage_the_render_actually_painted_something(state):
    """Power check. Every assertion above is vacuous against an all-background image."""
    img = render_state(**state)
    background = np.all(np.isclose(img, CHANNEL["background"]), axis=-1)
    painted = int(np.sum(~background))
    assert painted > 200, f"only {painted} pixels differ from background; the render is nearly empty"
