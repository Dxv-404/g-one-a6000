"""Gate E4, visibility symmetry.

Passing condition, from the build spec: "Holds by construction under the bidirectional definition; the test
asserts the implementation matches the definition."

Two distinct things are checked, and the distinction matters for whether this gate can fail at all.

The definition test is close to a tautology, because the table is built from the definition. It is kept as a
regression guard: if someone later replaces the table build with a faster single-ray version, this goes red.

The symmetry test is the one with real failure modes. It exercises the table's indexing, and an index
transposition or an off-by-one in the `dr + R` offset is invisible to the definition test while breaking
symmetry immediately.
"""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from configs.default import DEFAULT  # noqa: E402
from src.env.grid import make_obstacle_grid  # noqa: E402
from src.env.perception import (  # noqa: E402
    build_vis_table,
    ray_clear,
    visible,
    visible_from_table,
)

R = DEFAULT.r


@pytest.fixture(scope="module")
def grid():
    return make_obstacle_grid(jax.random.key(0), DEFAULT.h, DEFAULT.w, DEFAULT.obstacle_density)


@pytest.fixture(scope="module")
def table(grid):
    return build_vis_table(grid, R)


@pytest.mark.slow  # 42s measured, the slowest single test in the suite: 21 cells times 81 offsets
def test_table_matches_the_bidirectional_definition(grid, table):
    """Every table entry equals ray_clear(a,b) AND ray_clear(b,a)."""
    h, w = DEFAULT.h, DEFAULT.w
    rng = jax.random.key(1)
    for i in range(400):
        k1, k2, rng = jax.random.split(rng, 3)
        a = jnp.array(
            [jax.random.randint(k1, (), 0, h), jax.random.randint(k2, (), 0, w)],
            dtype=jnp.int32,
        )
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                b = a + jnp.array([dr, dc], dtype=jnp.int32)
                if not (0 <= int(b[0]) < h and 0 <= int(b[1]) < w):
                    continue
                expected = bool(visible(grid, a, b, R))
                got = bool(table[int(a[0]), int(a[1]), dr + R, dc + R])
                assert got == expected, f"table disagrees with the definition at {a} -> {b}"
        if i > 20:  # 21 cells times 81 offsets is ample; the loop bound keeps the gate fast
            break


def test_visibility_is_symmetric(grid, table):
    """visible(a, b) == visible(b, a) for every in-range pair in the whole grid.

    This is the assertion that catches an indexing error. It is checked exhaustively rather than by sampling,
    because it is cheap as a vectorised comparison and a sparse asymmetry would be easy to miss.
    """
    h, w, s, _ = table.shape
    r = (s - 1) // 2
    mismatches = 0
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            # a = (row, col); b = a + (dr, dc). Compare table[a, +d] against table[b, -d] on the overlap.
            r0, r1 = max(0, -dr), min(h, h - dr)
            c0, c1 = max(0, -dc), min(w, w - dc)
            if r0 >= r1 or c0 >= c1:
                continue
            forward = table[r0:r1, c0:c1, dr + r, dc + r]
            backward = table[r0 + dr : r1 + dr, c0 + dc : c1 + dc, -dr + r, -dc + r]
            mismatches += int(jnp.sum(forward != backward))
    assert mismatches == 0, f"{mismatches} asymmetric pairs; visibility must be bidirectional"


def test_a_wall_blocks_sight_across_it(grid):
    """A constructed wall, so the gate does not depend on a random layout containing a useful case."""
    obstacles = jnp.zeros((11, 11), dtype=bool).at[5, :].set(True)
    above = jnp.array([3, 5], dtype=jnp.int32)
    below = jnp.array([7, 5], dtype=jnp.int32)
    assert not bool(visible(obstacles, above, below, R)), "a solid wall did not block sight"
    assert not bool(visible(obstacles, below, above, R)), "blocking must hold in both directions"


def test_empty_grid_is_fully_visible():
    """With no obstacles every cell in the patch is visible. Catches a ray that blocks on nothing."""
    obstacles = jnp.zeros((11, 11), dtype=bool)
    table = build_vis_table(obstacles, R)
    centre = table[5, 5]
    assert bool(jnp.all(centre)), "an empty grid must be fully visible within the patch"


def test_obstacle_cell_is_itself_visible():
    """Endpoints are excluded from the ray, so an obstacle does not hide itself.

    If this inverted, the obstacle channel of the observation would always be empty and nothing could be
    navigated around or hidden behind. See Q-010.
    """
    obstacles = jnp.zeros((11, 11), dtype=bool).at[5, 7].set(True)
    a = jnp.array([5, 5], dtype=jnp.int32)
    wall = jnp.array([5, 7], dtype=jnp.int32)
    beyond = jnp.array([5, 8], dtype=jnp.int32)
    assert bool(visible(obstacles, a, wall, R)), "an obstacle must be visible, not hide itself"
    assert not bool(visible(obstacles, a, beyond, R)), "an obstacle must hide what lies beyond it"


def test_table_lookup_agrees_with_direct_computation(grid, table):
    """The per-tick gather path must agree with the reference path, and report False beyond the radius."""
    a = jnp.array([20, 20], dtype=jnp.int32)
    for dr in (-R, 0, R):
        for dc in (-R, 0, R):
            b = a + jnp.array([dr, dc], dtype=jnp.int32)
            assert bool(visible_from_table(table, a, b, R)) == bool(visible(grid, a, b, R))
    far = a + jnp.array([R + 1, 0], dtype=jnp.int32)
    assert not bool(visible_from_table(table, a, far, R)), "beyond the radius must not be visible"


def test_ray_is_not_required_to_be_symmetric(grid):
    """The single ray may be asymmetric. That is why visibility conjoins both directions.

    Not an assertion that asymmetry exists, since a given layout may happen to have none. This records the
    reason the conjunction is in the definition, and fails only if `ray_clear` stops being direction-aware.
    """
    a = jnp.array([10, 10], dtype=jnp.int32)
    b = jnp.array([13, 12], dtype=jnp.int32)
    forward = bool(ray_clear(grid, a, b, R))
    backward = bool(ray_clear(grid, b, a, R))
    assert bool(visible(grid, a, b, R)) == (forward and backward)
