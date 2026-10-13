"""Gate E3, resource conservation.

Passing condition, from the build spec: "Items consumed plus items standing plus items regrown balances every
tick, per source and globally."

The identity, stated so it is checkable:

    standing(t) = standing(t-1) - consumed(t) + regrown(t)

**Per source is well defined even when source patches overlap**, which they do at the default layout: seeding
gives per-source local counts of [5, 9, 6, 5, 5, 7] against a cap of 5, because a patch counts its
neighbour's items too. The identity is stated over the patch region rather than over item ownership. An item
inside two patches is counted by both and each patch's own balance still holds exactly, because items never
move once placed. Ownership would have needed a source-id grid the state does not carry.

### Why these tests consume

The first version of this gate seeded every patch to its cap and then never consumed anything. Regrowth
triggers only when `local < cap`, so it never fired: measured directly, **zero placements across 120 ticks**.
Conservation then held trivially as 0 = 0, and three of five deliberate breaks passed. A gate that cannot
fail is worse than no gate, so every test here consumes, and `test_regrowth_actually_fires` exists purely as
a canary: if regrowth stops happening for any reason, that test fails first and names the cause, rather than
the rest of the file passing vacuously.
"""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

# Every test here drives 150-tick trajectories and runs 12 to 17 seconds, measured. Marked slow as a whole
# module so the pre-commit hook can deselect it: a hook that takes minutes is bypassed within a day, which is
# worse than no hook. The gate still runs in full on every suite run.
pytestmark = pytest.mark.slow

from configs.default import DEFAULT  # noqa: E402
from src.env.entropy import PRNGSource  # noqa: E402
from src.env.grid import make_obstacle_grid  # noqa: E402
from src.env.resources import (  # noqa: E402
    local_count,
    place_sources,
    regrow,
    resolve_pick,
    seed_grid,
)

P = DEFAULT
N_TICKS = 150


@pytest.fixture(scope="module")
def world():
    obstacles = make_obstacle_grid(jax.random.key(0), P.h, P.w, P.obstacle_density)
    food_src, target_src, relax = place_sources(jax.random.key(1), obstacles, P)
    food = seed_grid(jax.random.key(2), food_src, obstacles, P.cap_food, P.r_src)
    target = seed_grid(jax.random.key(3), target_src, obstacles, P.cap_target, P.r_src)
    return obstacles, food_src, target_src, food, target, relax


def _consume_one(grid, tick):
    """Remove one standing item, deterministically. Stands in for a prey pick.

    Consumption is what makes regrowth fire at all, since the trigger is `local < cap` and seeding leaves
    every patch at its cap.
    """
    occupied = jnp.argwhere(grid > 0)
    if occupied.shape[0] == 0:
        return grid, 0
    cell = occupied[tick % occupied.shape[0]]
    return grid.at[cell[0], cell[1]].set(0), 1


def _run(obstacles, sources, grid, n_ticks, key_seed, on_tick=None, p_regrow=None):
    """Drive consumption and regrowth for n_ticks, yielding per-tick bookkeeping.

    `p_regrow` defaults to the configured rate. A test may raise it to increase the number of placements and
    therefore its own statistical power, which is not the same as changing a parameter: the rate is an
    argument to `regrow`, and the property under test is that a placement lands on an empty cell, which must
    hold at any rate.
    """
    rate = P.p_food if p_regrow is None else p_regrow
    source = PRNGSource(key=jax.random.key(key_seed))
    history = []
    for step in range(n_ticks):
        before_local = jnp.array([local_count(grid, s, P.r_src) for s in sources])
        before = int(jnp.sum(grid))

        grid, consumed = _consume_one(grid, step)
        mid = int(jnp.sum(grid))
        assert mid == before - consumed, "the test's own consumption did not balance"

        grid, placed = regrow(
            source,
            grid,
            sources,
            obstacles,
            P.cap_food,
            P.r_src,
            rate,
            step,
            "regrow.food.trigger",
            "regrow.food.cell",
        )
        after = int(jnp.sum(grid))
        after_local = jnp.array([local_count(grid, s, P.r_src) for s in sources])
        history.append((before, consumed, int(jnp.sum(placed)), after, before_local, after_local, placed))
        if on_tick:
            on_tick(grid, step)
    return grid, history


def test_regrowth_actually_fires(world):
    """Canary. If this fails, every other conservation test in this file is passing vacuously.

    It exists because the first version of this gate never triggered regrowth at all, and three deliberate
    breaks passed as a result.
    """
    obstacles, food_src, _, food, _, _ = world
    _, history = _run(obstacles, food_src, food, N_TICKS, 10)
    total_placed = sum(h[2] for h in history)
    total_consumed = sum(h[1] for h in history)
    assert total_consumed > 0, "the test consumed nothing, so regrowth could never be triggered"
    assert total_placed > 0, (
        f"regrowth never fired across {N_TICKS} ticks despite {total_consumed} items consumed. "
        "Every conservation assertion in this file is vacuous until this passes."
    )


def test_every_placement_lands_on_an_empty_cell(world):
    """High-rate variant, written because the default rate leaves this property under-tested.

    At the configured rate of 0.03 per source per tick, with most patches sitting at or above their cap
    because of overlap, only a handful of placements occur across a run. The chance that any one of them
    would have hit an occupied cell is roughly one in ten, so an implementation that had dropped the
    emptiness mask escaped detection by luck rather than by being correct. Measured: the break passed.

    Driving the rate to 1.0 makes every below-cap source place on every tick while consumption keeps patches
    just below their cap, so occupied cells are always present as collision candidates. The property being
    tested is unchanged and must hold at any rate: standing rises by exactly the number placed.

    The explicit floor on placements is the point. A canary asserting merely "more than zero" is what let the
    first version of this file look exercised while carrying almost no power.
    """
    obstacles, food_src, _, food, _, _ = world
    _, history = _run(obstacles, food_src, food, N_TICKS, 20, p_regrow=1.0)

    total_placed = sum(h[2] for h in history)
    assert total_placed >= 100, (
        f"only {total_placed} placements at rate 1.0 across {N_TICKS} ticks. This test has too little power "
        "to detect a placement landing on an occupied cell."
    )
    for step, (before, consumed, placed, after, _, _, _) in enumerate(history):
        assert after == before - consumed + placed, (
            f"tick {step}: {placed} placed but standing went {before} -> {after} after {consumed} consumed. "
            "A placement landed on a cell that already held an item, so it vanished."
        )


def test_conserves_globally_every_tick(world):
    """standing(t) = standing(t-1) - consumed(t) + regrown(t), checked on each tick, not at the end.

    Checking only endpoints would let a placement lost on one tick be masked by an extra on another, which is
    exactly what two overlapping patches choosing the same cell would produce.
    """
    obstacles, food_src, _, food, _, _ = world
    _, history = _run(obstacles, food_src, food, N_TICKS, 11)
    for step, (before, consumed, placed, after, _, _, _) in enumerate(history):
        assert after == before - consumed + placed, (
            f"tick {step}: {before} - {consumed} + {placed} should be {before - consumed + placed}, got {after}. "
            "A placement was lost or an item was created."
        )


def test_conserves_per_source_every_tick(world):
    """The same identity per patch region. Overlap is expected and does not break it."""
    obstacles, food_src, _, food, _, _ = world
    _, history = _run(obstacles, food_src, food, N_TICKS, 12)
    for step, (_, consumed, placed, _, before_local, after_local, _) in enumerate(history):
        delta = after_local - before_local
        # A patch can lose at most the items consumed and gain at most the items placed anywhere, since an
        # item in an overlap region is counted by every patch containing it.
        assert bool(jnp.all(delta >= -consumed)), f"tick {step}: a patch lost more than was consumed"
        assert bool(jnp.all(delta <= placed)), f"tick {step}: a patch gained more than was placed"


def test_a_placement_happens_only_when_that_patch_is_below_cap(world):
    """The exact regrowth rule, asserted tightly rather than through a loose bound.

    The trigger is `local < cap` evaluated per source at the start of the tick. A loose bound such as
    "standing stays under cap times the number of sources" is satisfied by an implementation that ignores the
    cap entirely, which is how the first version of this gate let that break pass.
    """
    obstacles, food_src, _, food, _, _ = world
    _, history = _run(obstacles, food_src, food, N_TICKS, 13)
    checked = 0
    for step, (_, _, _, _, before_local, _, placed) in enumerate(history):
        for s in range(len(food_src)):
            if int(placed[s]) == 1:
                checked += 1
                assert int(before_local[s]) < P.cap_food, (
                    f"tick {step}: source {s} placed an item while holding {int(before_local[s])} "
                    f"against a cap of {P.cap_food}. The cap is not regulating regrowth."
                )
    assert checked > 0, "no placement was observed, so the cap rule was never exercised"


def test_standing_never_exceeds_cap_when_patches_are_disjoint(world):
    """With non-overlapping sources the bound is exact, so it can be asserted strictly.

    At the default layout patches overlap and a local count legitimately exceeds the cap by counting a
    neighbour's items. Constructing a disjoint layout removes that confound and lets the real ceiling be
    tested.
    """
    obstacles, _, _, _, _, _ = world
    # Spaced by more than 2 * r_src so no two patches share a cell.
    spacing = 2 * P.r_src + 2
    sources = jnp.array(
        [[P.r_src + 1 + (i // 3) * spacing, P.r_src + 1 + (i % 3) * spacing] for i in range(6)],
        dtype=jnp.int32,
    )
    grid = seed_grid(jax.random.key(4), sources, obstacles, P.cap_food, P.r_src)

    def check(g, step):
        counts = jnp.array([local_count(g, s, P.r_src) for s in sources])
        assert int(jnp.max(counts)) <= P.cap_food, (
            f"tick {step}: a disjoint patch holds {int(jnp.max(counts))} against a cap of {P.cap_food}"
        )

    _run(obstacles, sources, grid, N_TICKS, 14, on_tick=check)


def test_regrowth_never_places_on_an_obstacle(world):
    """An item on an obstacle would be unreachable and would break the supply-versus-demand derivation."""
    obstacles, food_src, _, food, target, _ = world
    assert not bool(jnp.any((food > 0) & obstacles)), "seeding placed food on an obstacle"
    assert not bool(jnp.any((target > 0) & obstacles)), "seeding placed a forage target on an obstacle"

    def check(g, step):
        assert not bool(jnp.any((g > 0) & obstacles)), f"tick {step}: regrowth placed an item on an obstacle"

    _run(obstacles, food_src, food, N_TICKS, 15, on_tick=check)


def test_a_cell_never_holds_more_than_one_item(world):
    """The grid is int8 and a cell holds at most one item of each resource."""
    obstacles, food_src, _, food, _, _ = world

    def check(g, step):
        assert int(jnp.max(g)) <= 1, f"tick {step}: a cell holds {int(jnp.max(g))} items"

    _run(obstacles, food_src, food, N_TICKS, 16, on_tick=check)


# --------------------------------------------------------------------------- pick


def test_a_pick_consumes_exactly_one_item(world):
    _, _, _, food, target, _ = world
    prey_pos = jnp.argwhere(target > 0)[:4].astype(jnp.int32)
    alive = jnp.ones(prey_pos.shape[0], dtype=bool)
    picking = jnp.ones(prey_pos.shape[0], dtype=bool)

    before = int(jnp.sum(target))
    _, target_after, gained, _ = resolve_pick(food, target, prey_pos, alive, picking)

    assert int(jnp.sum(gained)) == 4, "each prey stood on a target and picked, so all four should score"
    assert int(jnp.sum(target_after)) == before - int(jnp.sum(gained))


def test_contended_pick_is_taken_once_by_the_lowest_index(world):
    """Two prey on one cell both picking. The item must be consumed once, not twice.

    This is the case that silently inflates score if resolution is parallel rather than ordered.
    """
    _, _, _, food, target, _ = world
    cell = jnp.argwhere(target > 0)[0]
    prey_pos = jnp.stack([cell, cell, cell]).astype(jnp.int32)

    before = int(jnp.sum(target))
    _, target_after, gained, _ = resolve_pick(
        food, target, prey_pos, jnp.ones(3, dtype=bool), jnp.ones(3, dtype=bool)
    )

    assert int(jnp.sum(gained)) == 1, f"three prey on one item scored {int(jnp.sum(gained))} times"
    assert [int(g) for g in gained] == [1, 0, 0], "contention must resolve to the lowest prey index"
    assert int(jnp.sum(target_after)) == before - 1


def test_a_dead_or_non_picking_prey_consumes_nothing(world):
    """Dead agents keep their array slots and still have actions computed, and must be masked to zero."""
    _, _, _, food, target, _ = world
    cell = jnp.argwhere(target > 0)[0]
    prey_pos = jnp.stack([cell, cell]).astype(jnp.int32)
    before = int(jnp.sum(target))

    _, after_dead, gained_dead, _ = resolve_pick(
        food, target, prey_pos, jnp.array([False, False]), jnp.array([True, True])
    )
    assert int(jnp.sum(gained_dead)) == 0 and int(jnp.sum(after_dead)) == before

    _, after_idle, gained_idle, _ = resolve_pick(
        food, target, prey_pos, jnp.array([True, True]), jnp.array([False, False])
    )
    assert int(jnp.sum(gained_idle)) == 0 and int(jnp.sum(after_idle)) == before
