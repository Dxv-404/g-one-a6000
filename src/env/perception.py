"""Bidirectional line of sight, and the precomputed visibility table.

Two design commitments from the build spec, both load-bearing.

**Visibility is bidirectional by construction.**

    visible(a, b) = ray_clear(a, b) AND ray_clear(b, a)

A single Bresenham ray is not symmetric in general: the cells visited walking from a to b need not match
those visited walking from b to a, because the tie-breaking at half-integer crossings depends on the
direction of travel. Defining visibility as the conjunction of both directions removes the problem entirely,
which is what makes gate E4 a tautology rather than a test a correct implementation can fail. It is cheap
here because rays inside the sensing patch are at most four cells long.

**Visibility is precomputed, not ray-cast per tick.**

Obstacles are static for the life of a run, so `visible(a, b)` is a pure function of the grid. Ray casting a
9 by 9 patch bidirectionally would be 81 cells times two rays per agent per tick, roughly 5.3 million
ray-steps per tick at working scale, and nothing in the throughput budget accounted for it. The table costs
130 KB once and turns per-tick perception into a gather.

    vis_table  (H, W, S, S)  bool,  S = 2R + 1
    at H = W = 40, R = 4:  1600 * 81 = 129,600 bytes

The table is shared across all environments because the obstacle layout is shared. If layouts are ever
randomised per environment it becomes (E, H, W, S, S), which is 133 MB at E of 1024: affordable but no
longer free, and that tradeoff should be measured before adopting per-environment layouts.

Build step 1. Gate E4.
"""

import jax
import jax.numpy as jnp
from jax import Array


def _interior_cells(a: Array, b: Array, r_max: int) -> tuple[Array, Array]:
    """The cells strictly between a and b, as a fixed-size array plus an activity mask.

    The ray rule, stated exactly, because the build spec says "Bresenham" without pinning a variant and gate
    E2 requires the vectorised implementation and the NumPy oracle to agree **exactly** on discrete state.
    An underspecified ray would let two individually correct implementations disagree, and the natural
    response to that failure would be to weaken the gate. So the rule is fixed here and the oracle will
    consume this same definition. See D-010.

    With n = max(|dy|, |dx|) the number of steps, the cell at step i is

        offset_k = sign(d_k) * round_half_away_from_zero(i * |d_k| / n)

    computed in integers as (2*i*|d_k| + n) // (2*n). Steps i = 1 .. n-1 are interior; i = 0 and i = n are
    the endpoints and are excluded.

    **Endpoints are excluded deliberately**, which means an obstacle standing at b does not block sight of
    b itself. Obstacles must be visible or the obstacle channel in the observation would always be empty and
    no agent could navigate around, let alone hide behind, anything. It also means an obstacle blocks sight
    of what lies beyond it rather than of itself. See Q-010: the build spec does not state this and it is a
    choice that could change a measured descriptor, so it is flagged rather than assumed silently.

    Returns (cells, active), shapes (r_max, 2) int32 and (r_max,) bool. Every returned cell lies inside the
    bounding box of a and b, so if both endpoints are in bounds every interior cell is too.
    """
    d = b - a
    n = jnp.max(jnp.abs(d))
    safe_n = jnp.maximum(n, 1)  # n is zero only when a == b, where no step is active anyway

    i = jnp.arange(1, r_max + 1, dtype=jnp.int32)  # (r_max,)
    magnitude = (2 * i[:, None] * jnp.abs(d)[None, :] + safe_n) // (2 * safe_n)
    cells = a[None, :] + jnp.sign(d)[None, :] * magnitude  # (r_max, 2)
    active = i < n  # (r_max,)
    return cells, active


def ray_clear(obstacles: Array, a: Array, b: Array, r_max: int) -> Array:
    """True when no obstacle lies strictly between a and b. Not symmetric in general, by design."""
    cells, active = _interior_cells(a, b, r_max)
    blocked = obstacles[cells[:, 0], cells[:, 1]] & active
    return ~jnp.any(blocked)


def visible(obstacles: Array, a: Array, b: Array, r_max: int) -> Array:
    """Bidirectional line of sight. Symmetric by construction, which is what gate E4 asserts."""
    return ray_clear(obstacles, a, b, r_max) & ray_clear(obstacles, b, a, r_max)


def required_radius(params) -> int:
    """The radius a table must be built at to serve every consumer in one environment.

    The prey perceive at `r` and the predator at `r_pred`, and both read the SAME table, so it has to cover
    the larger. `check_invariants` permits `r_pred` up to `r + 2`, so these genuinely differ across the
    sweep even though they coincide at the default.
    """
    return max(int(params.r), int(params.r_pred))


def build_vis_table(obstacles: Array, r: int) -> Array:
    """Precompute visibility from every cell to every cell in its egocentric patch.

    Returns (H, W, S, S) bool with S = 2R + 1, indexed as

        vis_table[row, col, dr + R, dc + R] = visible((row, col), (row + dr, col + dc))

    Patch cells falling outside the grid are False, so an off-grid cell is never reported visible and callers
    need no separate bounds test when gathering.

    Built once at construction. The double `vmap` is over cells and over patch offsets; the underlying
    function stays single-pair, which keeps it identical in shape to what the NumPy oracle will do.
    """
    h, w = obstacles.shape
    s = 2 * r + 1

    rows, cols = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    centres = jnp.stack([rows, cols], axis=-1).reshape(-1, 2)  # (H*W, 2)

    d_rows, d_cols = jnp.meshgrid(jnp.arange(-r, r + 1), jnp.arange(-r, r + 1), indexing="ij")
    offsets = jnp.stack([d_rows, d_cols], axis=-1).reshape(-1, 2)  # (S*S, 2)

    def one_pair(centre: Array, offset: Array) -> Array:
        target = centre + offset
        inside = (target[0] >= 0) & (target[0] < h) & (target[1] >= 0) & (target[1] < w)
        # Clamp before the ray so an out-of-bounds target cannot index outside the grid. The result is
        # discarded by `inside` regardless; the clamp exists because JAX silently clamps rather than raising,
        # and relying on that implicitly would be a trap for anyone reading this later.
        safe = jnp.stack([jnp.clip(target[0], 0, h - 1), jnp.clip(target[1], 0, w - 1)])
        return inside & visible(obstacles, centre, safe, r)

    over_offsets = jax.vmap(one_pair, in_axes=(None, 0))
    over_cells = jax.vmap(over_offsets, in_axes=(0, None))
    table = over_cells(centres, offsets)  # (H*W, S*S)
    return table.reshape(h, w, s, s)


def table_radius(vis_table: Array) -> int:
    """The radius the table was built at, derived from its shape rather than passed alongside it.

    A radius passed as an argument beside the table it indexes is a stated fact with no consumer: nothing
    breaks when the two stop agreeing. Deriving it from the shape makes disagreement impossible.
    """
    return (vis_table.shape[-1] - 1) // 2


def visible_from_table(vis_table: Array, a: Array, b: Array, r: int) -> Array:
    """Look up visibility for a pair within sensing range `r`. The per-tick path.

    Pairs further apart than `r` report False, since an entity beyond the sensing radius is not perceived at
    all.

    **The query radius `r` and the table's BUILD radius are different things, and conflating them was
    DEV-030.** The table is indexed at its own build radius; `r` only decides how far the caller is willing
    to see. The previous version used `r` for both, so a table built at 4 queried at 6 computed offsets up to
    12 into an axis of size 9. JAX clamps an out-of-bounds gather rather than raising, so it silently read
    the wrong cell: measured, 37.8 percent of pairs at Chebyshev distance 5 or 6 disagreed with the truth,
    and the errors were in the direction of reporting visibility THROUGH walls.

    That configuration is reachable: `r_pred` is a swept axis, `check_invariants` permits `r_pred <= r + 2`,
    and the environment builds one table for both the prey's perception at `r` and the predator's at
    `r_pred`.

    The assertion is a Python one on a static shape, so it costs nothing at run time and fails at trace time
    rather than producing a wrong number.
    """
    R = table_radius(vis_table)
    assert R >= r, (
        f"visibility table was built at radius {R} but is being queried at radius {r}. Build it at "
        f"max(r, r_pred); querying beyond the build radius silently reads the wrong cell rather than raising."
    )
    d = b - a
    within = jnp.max(jnp.abs(d)) <= r
    dr = jnp.clip(d[0] + R, 0, 2 * R)
    dc = jnp.clip(d[1] + R, 0, 2 * R)
    return within & vis_table[a[0], a[1], dr, dc]


def two_regime_report(
    vis_table: Array,
    observer: Array,
    subject: Array,
    subject_moved: Array,
    distant_draw: Array,
    r: int,
    r_near: int,
    p_distant_detect: float,
) -> Array:
    """Whether an observer reports a subject this tick, under the build spec's two-regime rule.

        near   = chebyshev(observer, subject) <= R_NEAR
        report = visible AND (near OR subject_moved OR bernoulli(P_DISTANT_DETECT))

    **The near regime is what makes the target behaviour thanatosis rather than crypsis, and it is the single
    most important line in this function.** An entity inside `r_near` with clear line of sight is reported
    whether or not it moved, so holding still never makes a prey disappear from an adjacent predator. Motion
    gates *distant* detection only. Without that, a still prey would become invisible, freezing would be
    concealment, and a stillness result would be crypsis wearing thanatosis's name.

    `distant_draw` is a pre-drawn uniform on [0, 1) from the tagged entropy source under
    `perceive.distant_detect`, passed in rather than drawn here so the draw site stays at the caller and
    replay alignment is the caller's single responsibility. See D-001.

    Visibility itself is symmetric and motion-independent: this function gates *reporting*, never the
    underlying line of sight, which is why `vis_table` is consulted unchanged.
    """
    vis = visible_from_table(vis_table, observer, subject, r)
    near = jnp.max(jnp.abs(subject - observer)) <= r_near
    detected_far = subject_moved | (distant_draw < p_distant_detect)
    return vis & (near | detected_far)
