"""Static grid: obstacle layout, bounds, solidity, and movement legality.

Everything here is a pure function of the obstacle grid, which never changes during a run. That is what
makes the visibility table in `perception.py` precomputable, and it is why this module holds no state.

Single-environment throughout. The batch axis is obtained by `vmap`, never hand-written, because the NumPy
reference oracle is single-environment and hand-batching would make the differential harness awkward.

Build step 1. Gates E5, obstacle solidity, and E4 via `perception.py`.
"""

import jax
import jax.numpy as jnp
from jax import Array

# Eight-direction movement, as (d_row, d_col), clockwise from north.
#
# This order fixes action indices 0 to 7 for a single-cell move and 8 to 15 for a sprint, per the build
# spec's action set. Any permutation would be scientifically equivalent, since it is a relabelling of output
# units, but it must be identical in the vectorised implementation and in the NumPy oracle or gate E2
# compares different action semantics. See D-009.
DIRECTIONS = jnp.array(
    [
        (-1, 0),  # 0  N
        (-1, 1),  # 1  NE
        (0, 1),  # 2  E
        (1, 1),  # 3  SE
        (1, 0),  # 4  S
        (1, -1),  # 5  SW
        (0, -1),  # 6  W
        (-1, -1),  # 7  NW
    ],
    dtype=jnp.int32,
)

N_DIRECTIONS = 8


def make_obstacle_grid(key: Array, h: int, w: int, density: float) -> Array:
    """Place exactly `round(h * w * density)` obstacles, sampled without replacement.

    Exactly, not per-cell Bernoulli. The grid size is a derived parameter: 40 by 40 at density 0.15 gives
    exactly 1360 free cells, which is what sets the encounter rate to 0.48 expected prey in view. A Bernoulli
    draw would give 1360 only on average, so the free-cell count, and therefore the encounter rate, would
    vary from episode to episode and the derivation would hold only in expectation.

    Returns (h, w) bool, True where an obstacle stands.
    """
    n_cells = h * w
    n_obstacles = int(round(n_cells * density))
    # Permute cell indices and take a prefix. This is sampling without replacement, and unlike a rejection
    # loop it has a static shape, so it compiles.
    order = jax.random.permutation(key, n_cells)
    chosen = order[:n_obstacles]
    flat = jnp.zeros(n_cells, dtype=bool).at[chosen].set(True)
    return flat.reshape(h, w)


def in_bounds(pos: Array, h: int, w: int) -> Array:
    """True where `pos`, shape (..., 2) as (row, col), lies inside the grid."""
    return (pos[..., 0] >= 0) & (pos[..., 0] < h) & (pos[..., 1] >= 0) & (pos[..., 1] < w)


def is_obstacle(obstacles: Array, pos: Array) -> Array:
    """True where `pos` holds an obstacle. Out-of-bounds positions report True.

    Reporting out-of-bounds as solid is deliberate and makes the grid boundary behave like a wall, so a
    caller that forgets an `in_bounds` check fails closed rather than open. The clamp is required because
    JAX silently clamps out-of-range indices rather than raising, so without it an out-of-bounds read would
    return whatever sits at the clamped edge cell.
    """
    h, w = obstacles.shape
    r = jnp.clip(pos[..., 0], 0, h - 1)
    c = jnp.clip(pos[..., 1], 0, w - 1)
    return jnp.where(in_bounds(pos, h, w), obstacles[r, c], True)


def legal_move(obstacles: Array, pos: Array, direction: Array, sprint: Array) -> Array:
    """Whether moving from `pos` along `direction` is legal, at one or two cells.

    Three conditions, from the build spec's update equations:

        legal = in_bounds(proposed) AND NOT obstacle(proposed) AND path_clear(pos, proposed)

    `path_clear` is what stops a sprint tunnelling through a wall. A sprint moves two cells, so its
    intermediate cell is `pos + direction`, and that cell must itself be free.

    For a diagonal sprint the intermediate cell is itself diagonal. The build spec specifies this explicitly
    rather than leaving it to the implementation: a diagonal sprint whose intermediate diagonal cell is an
    obstacle is illegal **even when both orthogonal neighbours are free**, so the sprint does not clip
    corners. The alternative convention makes obstacles substantially more permeable to diagonal movement,
    which would quietly weaken the occlusion dynamics the experiment depends on.

    `direction` is a (..., 2) offset, `sprint` a (...) bool. Returns (...) bool.
    """
    h, w = obstacles.shape
    step = jnp.where(sprint, 2, 1)[..., None]
    intermediate = pos + direction
    destination = pos + direction * step

    dest_ok = in_bounds(destination, h, w) & ~is_obstacle(obstacles, destination)
    # Only a sprint has an intermediate cell distinct from its destination.
    inter_ok = in_bounds(intermediate, h, w) & ~is_obstacle(obstacles, intermediate)
    return dest_ok & jnp.where(sprint, inter_ok, True)


def resolve_move(obstacles: Array, pos: Array, direction: Array, sprint: Array) -> Array:
    """Apply a move if legal, otherwise stay. Returns the new position, shape (..., 2).

    Blocked movement falls through to staying put rather than sliding along the obstacle, which keeps
    `prey_moved` an honest record of whether the cell actually changed. The build spec sets `prey_moved`
    from whether the position changed, not from whether a move was attempted, and gate E5 depends on that
    distinction.
    """
    step = jnp.where(sprint, 2, 1)[..., None]
    proposed = pos + direction * step
    legal = legal_move(obstacles, pos, direction, sprint)
    return jnp.where(legal[..., None], proposed, pos)


def free_cell_count(obstacles: Array) -> Array:
    """Number of non-obstacle cells. The encounter-rate derivation is stated against this."""
    return jnp.sum(~obstacles)
