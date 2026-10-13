"""Resource sources, source-regulated regrowth, and pick resolution.

Two resources with separate sources, separate caps, and separate rates: the forage target, which is what
fitness counts, and food, which is only fuel. They are tunable independently because they do different jobs,
and the split is what makes survival instrumental rather than rewarded.

Regrowth is source-regulated rather than uniform. The cap is what makes it regulated: regrowth is bounded by
standing local density, so a source cannot be farmed indefinitely, scarcity is real, and good foraging
locations are spatially predictable and can therefore become ambush points.

**Source layout is a swept parameter, not a fixed choice.** `D_SRC` is the minimum distance from any target
source to the nearest food source. At zero the two coincide and an agent can camp on one spot that supplies
both, which degenerates energy into a second timer. Large values force travel, and travel is where predation
happens. That prices camping against foraging, and a parameter that prices one strategy against another has
no neutral value, so it is swept and the result is reported as a map from conditions to which strategy
emerges. See node 4.23.0.0.

Build step 3. Gate E3, resource conservation.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax import lax

from src.env.entropy import Entropy, draw_uniform

# Sentinel for "no valid cell". Selection functions return this rather than silently falling back to index
# zero, which is the failure mode the spawn relaxation ladder exists to prevent: a categorical draw over an
# all-zero mask does not raise under static shapes, it returns index zero, which may be an obstacle.
NO_CELL = -1


def _chebyshev_to_nearest(points: Array, h: int, w: int) -> Array:
    """Chebyshev distance from every cell to the nearest of `points`. Returns (h*w,) int32.

    Chebyshev rather than Euclidean or Manhattan, because movement is eight-directional: the Chebyshev
    distance is exactly the number of moves needed to cross the gap, so it is the travel-cost metric this
    world actually runs on. It is also what `D_SPAWN` already uses, so the two separations are measured the
    same way. See D-014.
    """
    rows, cols = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    rows = rows.reshape(-1)
    cols = cols.reshape(-1)

    def to_one(p):
        return jnp.maximum(jnp.abs(rows - p[0]), jnp.abs(cols - p[1]))

    return jnp.min(jax.vmap(to_one)(points), axis=0)


def _top_k_masked(key: Array, mask: Array, k: int) -> Array:
    """Sample `k` distinct indices uniformly from where `mask` is True. Returns (k,) int32.

    Gumbel top-k rather than a rejection loop, because rejection needs an unbounded loop and does not compile
    under static shapes. Adding Gumbel noise to a uniform log-weight and taking the top k is exactly sampling
    without replacement, and masked-out entries are pushed to negative infinity so they can never be chosen
    while any valid cell remains.
    """
    noise = jax.random.gumbel(key, mask.shape)
    scores = jnp.where(mask, noise, -jnp.inf)
    return jnp.argsort(-scores)[:k].astype(jnp.int32)


def _patch_mask(centre: Array, r: int, h: int, w: int) -> Array:
    """Cells within Chebyshev radius `r` of `centre`, flattened. Returns (h*w,) bool."""
    rows, cols = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    d = jnp.maximum(jnp.abs(rows.reshape(-1) - centre[0]), jnp.abs(cols.reshape(-1) - centre[1]))
    return d <= r


def place_sources(key: Array, obstacles: Array, params) -> tuple[Array, Array, Array]:
    """Place food sources, then target sources subject to `D_SRC`. Construction-time, not per tick.

    Food first, then targets constrained to be at least `d` from every food source, with `d` taken from
    `D_SRC_RELAX` and the first feasible level winning. Feasibility here means at least `n_target_src` valid
    cells remain, not merely one: the draw takes that many distinct cells, so a mask with fewer is as
    unsatisfiable as an empty one and fails in the same silent way.

    Returns (food_src, target_src, relax_level) with shapes (n_food_src, 2), (n_target_src, 2), and a scalar
    index into `D_SRC_RELAX`.

    **The relaxation level must be recorded.** A configuration that relaxes often is outside the valid
    envelope and its results are not comparable to one that does not, exactly as for the spawn ladder. Sources
    are static for a run rather than redrawn per episode, so the level is a property of the run and belongs in
    the run registry rather than in the per-episode accumulators. See D-015.

    No entropy tag is consumed. Source layout is closed over at construction, like the obstacle grid, so the
    differential harness injects the resulting arrays into both implementations rather than each generating
    its own. That is why D-001's registry has `spawn.*` and `regrow.*` entries and nothing for layout.
    """
    h, w = obstacles.shape
    free = (~obstacles).reshape(-1)
    k_food, k_target = jax.random.split(key)

    food_flat = _top_k_masked(k_food, free, params.n_food_src)
    food_src = jnp.stack([food_flat // w, food_flat % w], axis=-1)

    dist = _chebyshev_to_nearest(food_src, h, w)

    # Evaluate every ladder level, then select. All levels are computed and chosen with jnp.where rather than
    # branched on, because a data-dependent branch does not compile and lax.cond would run both arms anyway.
    levels = jnp.asarray(params.d_src_relax, dtype=jnp.int32)

    def feasible(d):
        return jnp.sum(free & (dist >= d)) >= params.n_target_src

    ok = jax.vmap(feasible)(levels)
    # argmax on a boolean array gives the first True. If no level is feasible it gives 0, which would be the
    # strictest level and would then silently under-fill; the E3 gate asserts feasibility at the loosest level
    # so that case cannot pass unnoticed.
    relax_level = jnp.argmax(ok).astype(jnp.int32)
    chosen_d = levels[relax_level]

    valid = free & (dist >= chosen_d)
    target_flat = _top_k_masked(k_target, valid, params.n_target_src)
    target_src = jnp.stack([target_flat // w, target_flat % w], axis=-1)

    return food_src, target_src, relax_level


def seed_grid(key: Array, sources: Array, obstacles: Array, cap: int, r_src: int) -> Array:
    """Fill each source patch to its standing cap. Returns (h, w) int8, one item per cell at most.

    Sources are seeded one after another with `lax.scan` rather than all at once, because patches can overlap
    when two sources sit closer than twice `r_src`. Placing in parallel would let two sources choose the same
    cell in the same pass, and since a cell holds at most one item the second placement would vanish, leaving
    the standing count quietly below the cap.
    """
    h, w = obstacles.shape
    free = (~obstacles).reshape(-1)

    def one_source(grid_flat, inputs):
        centre, sub = inputs
        available = free & _patch_mask(centre, r_src, h, w) & (grid_flat == 0)
        chosen = _top_k_masked(sub, available, cap)
        # Only place where a genuinely valid cell was returned. `_top_k_masked` returns the top `cap` entries
        # regardless, so entries beyond the number of available cells are arbitrary and must be masked out.
        n_available = jnp.sum(available)
        keep = jnp.arange(cap) < n_available
        grid_flat = grid_flat.at[chosen].set(jnp.where(keep, 1, grid_flat[chosen]))
        return grid_flat, None

    subkeys = jax.random.split(key, sources.shape[0])
    grid_flat, _ = lax.scan(one_source, jnp.zeros(h * w, dtype=jnp.int8), (sources, subkeys))
    return grid_flat.reshape(h, w)


def regrow(
    entropy: Entropy,
    grid: Array,
    sources: Array,
    obstacles: Array,
    cap: int,
    r_src: int,
    p_regrow: float,
    step,
    tag_trigger: str,
    tag_cell: str,
) -> tuple[Array, Array]:
    """One tick of source-regulated regrowth. Returns (new_grid, n_placed_per_source).

    Per the build spec:

        for each source s:
            local = sum of grid within radius r_src of s
            if local < cap:
                with probability p_regrow per tick,
                place one item at a uniformly sampled free cell within r_src of s

    Sources are processed sequentially for the same reason `seed_grid` is: overlapping patches could otherwise
    both place into the same cell on the same tick, and one placement would silently vanish, which would break
    the conservation identity gate E3 checks.

    The cell is chosen by rank rather than by sampling a coordinate pair and rejecting invalid draws, because
    rejection needs an unbounded loop. One uniform selects the n-th currently-valid cell, which is exactly
    uniform over valid cells and consumes one draw per source per tick. See D-016.
    """
    h, w = obstacles.shape
    free = (~obstacles).reshape(-1)
    n_src = sources.shape[0]

    triggers = draw_uniform(entropy, (n_src,), tag_trigger, step)
    cell_draws = draw_uniform(entropy, (n_src,), tag_cell, step)

    def one_source(grid_flat, inputs):
        centre, u_trigger, u_cell = inputs
        patch = _patch_mask(centre, r_src, h, w)
        local = jnp.sum(jnp.where(patch, grid_flat, 0))
        available = free & patch & (grid_flat == 0)
        n_available = jnp.sum(available)

        want = (local < cap) & (u_trigger < p_regrow) & (n_available > 0)

        # Rank selection: take the k-th available cell, k uniform over the available count.
        rank = jnp.floor(u_cell * n_available).astype(jnp.int32)
        rank = jnp.clip(rank, 0, jnp.maximum(n_available - 1, 0))
        position = jnp.cumsum(available) - 1  # index among available cells, for available entries
        target = jnp.argmax(available & (position == rank))

        grid_flat = grid_flat.at[target].set(jnp.where(want, 1, grid_flat[target]))
        return grid_flat, want.astype(jnp.int32)

    grid_flat, placed = lax.scan(one_source, grid.reshape(-1), (sources, triggers, cell_draws))
    return grid_flat.reshape(h, w), placed


def resolve_pick(
    food_grid: Array,
    target_grid: Array,
    prey_pos: Array,
    prey_alive: Array,
    picking: Array,
) -> tuple[Array, Array, Array, Array]:
    """Resolve the pick action. Returns (food_grid, target_grid, gained_score, ate_food).

    A picked target increments `prey_score` immediately and a picked food adds energy immediately. There is no
    inventory: nothing is ever held, which is why the build spec deleted the carried-item reference rather
    than adding a state field for it.

    Both resources on the same cell are both taken, since they are separate grids and the action is one pick
    of whatever is present.

    **Contention is resolved by lowest prey index.** Two living prey can occupy the same cell and both pick on
    the same tick, and an item must not be consumed twice. Prey are resolved in index order via `lax.scan`, so
    the outcome is deterministic rather than dependent on evaluation order. Index order is arbitrary but it is
    fixed and recorded; a random tiebreak would need a draw site and a tag for an event that is rare and whose
    resolution carries no strategic content. See D-017.
    """
    h, w = food_grid.shape
    flat_pos = prey_pos[:, 0] * w + prey_pos[:, 1]
    active = prey_alive & picking

    def one_prey(carry, inputs):
        food_flat, target_flat = carry
        cell, take = inputs

        has_food = (food_flat[cell] > 0) & take
        has_target = (target_flat[cell] > 0) & take

        food_flat = food_flat.at[cell].set(jnp.where(has_food, 0, food_flat[cell]))
        target_flat = target_flat.at[cell].set(jnp.where(has_target, 0, target_flat[cell]))
        return (food_flat, target_flat), (has_target.astype(jnp.int32), has_food)

    (food_flat, target_flat), (gained_score, ate_food) = lax.scan(
        one_prey, (food_grid.reshape(-1), target_grid.reshape(-1)), (flat_pos, active)
    )
    return food_flat.reshape(h, w), target_flat.reshape(h, w), gained_score, ate_food


def local_count(grid: Array, centre: Array, r_src: int) -> Array:
    """Standing items within `r_src` of a source. The quantity the cap regulates and E3 balances per source."""
    h, w = grid.shape
    return jnp.sum(jnp.where(_patch_mask(centre, r_src, h, w), grid.reshape(-1), 0))
