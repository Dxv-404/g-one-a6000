"""Gate E5, obstacle solidity.

Passing condition, from the build spec: "No agent occupies an obstacle cell, no sprint traverses one, and no
sprint's intermediate cell is an obstacle. Ray behaviour is covered by E4 against the precomputed table."

The corner-clipping case is the one worth stating twice. `path_clear` matters because a sprint moves two
cells: without it, anyone with stamina passes through obstacles and the occlusion dynamics quietly stop
existing. That failure is silent, since the simulation still runs and produces plausible numbers.
"""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from configs.default import DEFAULT  # noqa: E402
from src.env.grid import (  # noqa: E402
    DIRECTIONS,
    free_cell_count,
    is_obstacle,
    legal_move,
    make_obstacle_grid,
    resolve_move,
)


@pytest.fixture(scope="module")
def grid():
    return make_obstacle_grid(jax.random.key(0), DEFAULT.h, DEFAULT.w, DEFAULT.obstacle_density)


def test_obstacle_count_is_exact(grid):
    """The free-cell count is a derived quantity, so it must be exact rather than exact in expectation.

    40 by 40 at density 0.15 gives exactly 1360 free cells, which is what sets the encounter rate to 0.48
    expected prey in view. A per-cell Bernoulli draw would hit 1360 only on average and the derivation would
    hold only in expectation.
    """
    expected_free = DEFAULT.h * DEFAULT.w - round(DEFAULT.h * DEFAULT.w * DEFAULT.obstacle_density)
    assert int(free_cell_count(grid)) == expected_free == 1360


def test_a_legal_move_never_has_an_obstacle_destination(grid):
    """The core solidity property, asserted over every cell in the grid without exception.

    This is stated over all start cells, including obstacle cells, because it is a property of `legal_move`
    itself and does not depend on where the agent came from. It is the strong form: whatever the start, if
    the move is permitted then the destination is free.
    """
    h, w = DEFAULT.h, DEFAULT.w
    rows, cols = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    positions = jnp.stack([rows, cols], axis=-1).reshape(-1, 2)

    for sprint in (False, True):
        for d in range(len(DIRECTIONS)):
            direction = jnp.broadcast_to(DIRECTIONS[d], positions.shape)
            flag = jnp.full(positions.shape[0], sprint)
            legal = legal_move(grid, positions, direction, flag)
            destination = positions + direction * jnp.where(flag, 2, 1)[:, None]
            assert not bool(jnp.any(legal & is_obstacle(grid, destination))), (
                f"a legal move had an obstacle destination: direction {d}, sprint={sprint}"
            )


def test_an_agent_on_a_free_cell_never_lands_on_an_obstacle(grid):
    """Exhaustive over every reachable start cell, every direction, walking and sprinting.

    Start cells are restricted to non-obstacle cells because that is the reachable state space: reset samples
    prey positions from non-obstacle cells, and by induction from this test an agent never subsequently
    enters one.

    The restriction is not a weakening, and the distinction is worth being explicit about. Run over all 1600
    cells this fails on exactly 240, which are precisely the obstacle cells themselves: an agent standing on
    an obstacle whose move is refused stays put, and so "lands" on the obstacle it was already on. That is an
    artefact of an unreachable start state, not a solidity failure. The property that holds without any
    restriction is asserted directly in the test above.
    """
    h, w = DEFAULT.h, DEFAULT.w
    rows, cols = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    positions = jnp.stack([rows, cols], axis=-1).reshape(-1, 2)
    free = positions[~is_obstacle(grid, positions)]
    assert free.shape[0] == 1360, "the reachable start set should be the free-cell count"

    for sprint in (False, True):
        for d in range(len(DIRECTIONS)):
            direction = jnp.broadcast_to(DIRECTIONS[d], free.shape)
            flag = jnp.full(free.shape[0], sprint)
            new = resolve_move(grid, free, direction, flag)
            assert not bool(jnp.any(is_obstacle(grid, new))), (
                f"a move from a free cell landed on an obstacle: direction {d}, sprint={sprint}"
            )


def test_no_move_ever_leaves_the_grid(grid):
    h, w = DEFAULT.h, DEFAULT.w
    rows, cols = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    positions = jnp.stack([rows, cols], axis=-1).reshape(-1, 2)

    for sprint in (False, True):
        for d in range(len(DIRECTIONS)):
            direction = jnp.broadcast_to(DIRECTIONS[d], positions.shape)
            flag = jnp.full(positions.shape[0], sprint)
            new = resolve_move(grid, positions, direction, flag)
            assert bool(jnp.all((new[:, 0] >= 0) & (new[:, 0] < h))), "row left the grid"
            assert bool(jnp.all((new[:, 1] >= 0) & (new[:, 1] < w))), "col left the grid"


def test_sprint_cannot_tunnel_through_a_wall():
    """The destination is free but the intermediate cell is solid. The sprint must be refused."""
    obstacles = jnp.zeros((11, 11), dtype=bool).at[5, 6].set(True)
    pos = jnp.array([5, 5], dtype=jnp.int32)
    east = jnp.array([0, 1], dtype=jnp.int32)

    assert not bool(legal_move(obstacles, pos, east, jnp.array(True))), (
        "a sprint tunnelled through an obstacle; occlusion dynamics would stop existing"
    )
    # The walk into the wall is refused too, for the different reason that its destination is solid.
    assert not bool(legal_move(obstacles, pos, east, jnp.array(False)))
    # With the wall removed both are legal, so the refusal above is about the wall, not about sprinting.
    clear = jnp.zeros((11, 11), dtype=bool)
    assert bool(legal_move(clear, pos, east, jnp.array(True)))


def test_diagonal_sprint_does_not_clip_a_corner():
    """The build spec's explicit case, and the one an implementation is most likely to get wrong.

    A diagonal sprint whose intermediate diagonal cell is an obstacle is illegal even when both orthogonal
    neighbours are free. The alternative convention makes obstacles substantially more permeable to diagonal
    movement.
    """
    obstacles = jnp.zeros((11, 11), dtype=bool).at[6, 6].set(True)  # the diagonal intermediate cell
    pos = jnp.array([5, 5], dtype=jnp.int32)
    southeast = jnp.array([1, 1], dtype=jnp.int32)

    # Both orthogonal neighbours of the intermediate cell are free, so a corner-clipping rule would allow it.
    assert not bool(is_obstacle(obstacles, jnp.array([6, 5])))
    assert not bool(is_obstacle(obstacles, jnp.array([5, 6])))

    assert not bool(legal_move(obstacles, pos, southeast, jnp.array(True))), (
        "a diagonal sprint clipped a corner through its diagonal intermediate cell"
    )


def test_blocked_move_stays_put_rather_than_sliding():
    """A refused move must leave the position unchanged, so `prey_moved` stays an honest record.

    Sliding along the obstacle would report movement on a tick the agent did not choose to move, and capture
    is evaluated against movement from the same tick.
    """
    obstacles = jnp.zeros((11, 11), dtype=bool).at[5, 6].set(True)
    pos = jnp.array([5, 5], dtype=jnp.int32)
    east = jnp.array([0, 1], dtype=jnp.int32)
    new = resolve_move(obstacles, pos, east, jnp.array(False))
    assert bool(jnp.all(new == pos)), "a blocked move changed the position"


def test_out_of_bounds_reads_as_solid():
    """The grid boundary behaves like a wall, so a missing bounds check fails closed rather than open."""
    obstacles = jnp.zeros((11, 11), dtype=bool)
    assert bool(is_obstacle(obstacles, jnp.array([-1, 5])))
    assert bool(is_obstacle(obstacles, jnp.array([11, 5])))
    assert not bool(is_obstacle(obstacles, jnp.array([5, 5])))
