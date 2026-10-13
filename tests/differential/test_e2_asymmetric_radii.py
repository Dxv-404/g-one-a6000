"""The differential at `r_pred != r`. Build step 11, added with DEV-030.

**This configuration had never been run and that is why the defect survived.** The environment builds one
visibility table and consults it at two radii: the prey's `r` and the predator's `r_pred`. They coincide at
the default, so every existing test compared a table against itself.

**The oracle does not share the defect**, which is the whole reason this gate works. `src/reference/` has no
table: it recomputes visibility per pair with its own Bresenham. So this is a genuine differential rather
than two copies of one belief, and it would have caught DEV-030 the first time it ran.

That is the difference from the alive-mask change, where the same modification had to be applied to both
sides and the differential was inert for it. Here the oracle needed no change at all.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import DEFAULT
from src.env import grid
from src.env.perception import build_vis_table, required_radius, table_radius, visible_from_table
from src.reference import naive_env as oracle

#: `check_invariants` permits `r_pred <= r + 2`, so 4, 5 and 6 are all legal against `r = 4` and all three
#: are configurations the M4 sweep will visit.
RADII = (4, 5, 6)


@pytest.mark.slow  # 15s, 14s and 14s measured at r_pred 4, 5 and 6
@pytest.mark.parametrize("r_pred", RADII)
def test_the_fast_lookup_matches_the_oracle_at_every_legal_r_pred(r_pred):
    """The comparison the defect hid from. A table built at `required_radius` must give the oracle's answer
    at every query radius the configuration admits."""
    p = DEFAULT.__class__(r_pred=r_pred)
    assert p.r_pred <= p.r + 2, "fixture error: this r_pred is outside what check_invariants permits"

    ob = grid.make_obstacle_grid(jax.random.key(0), p.h, p.w, p.obstacle_density)
    table = build_vis_table(ob, required_radius(p))
    assert table_radius(table) == max(p.r, p.r_pred)

    ob_np = np.asarray(ob)
    rng = np.random.default_rng(0)
    checked = 0
    far_pairs = 0
    for _ in range(1500):
        a = np.array([rng.integers(8, p.h - 8), rng.integers(8, p.w - 8)])
        d = np.array([rng.integers(-r_pred, r_pred + 1), rng.integers(-r_pred, r_pred + 1)])
        b = a + d
        fast = bool(visible_from_table(table, jnp.asarray(a), jnp.asarray(b), r_pred))
        slow = bool(oracle.visible(ob_np, tuple(a), tuple(b)))
        slow = slow and int(np.max(np.abs(d))) <= r_pred
        assert fast == slow, (
            f"at r_pred {r_pred}, {a.tolist()} to {b.tolist()}: fast says {fast}, oracle says {slow}"
        )
        checked += 1
        if int(np.max(np.abs(d))) > DEFAULT.r:
            far_pairs += 1

    # Coverage: at r_pred above r, the test is only meaningful if it actually queried beyond the prey's
    # radius, which is exactly the region the defect lived in.
    if r_pred > DEFAULT.r:
        assert far_pairs > 50, (
            f"only {far_pairs} pairs beyond r={DEFAULT.r} were checked at r_pred={r_pred}; the region the "
            "defect lived in was barely exercised"
        )
    assert checked > 500


def test_querying_beyond_the_build_radius_fails_loudly():
    """**The defect was silent, and that is what made it dangerous.** JAX clamps an out-of-bounds gather
    rather than raising, so a table built at 4 queried at 6 returned a wrong answer with no signal at all.

    Now it is a Python assertion on a static shape: it fires at trace time and costs nothing at run time.
    """
    ob = grid.make_obstacle_grid(jax.random.key(0), DEFAULT.h, DEFAULT.w, DEFAULT.obstacle_density)
    small = build_vis_table(ob, 4)
    with pytest.raises(AssertionError, match="built at radius 4 but is being queried at radius 6"):
        visible_from_table(small, jnp.array([10, 10]), jnp.array([13, 13]), 6)


def test_the_defect_reproduces_on_the_old_indexing():
    """A regression witness. The old code used the QUERY radius for the index offset; this reproduces that
    arithmetic directly and shows it disagrees with the correct lookup, so the test cannot quietly become
    vacuous if someone reverts the fix and the assertion with it."""
    ob = grid.make_obstacle_grid(jax.random.key(0), DEFAULT.h, DEFAULT.w, DEFAULT.obstacle_density)
    small = build_vis_table(ob, 4)
    big = build_vis_table(ob, 6)

    rng = np.random.default_rng(1)
    disagreements = 0
    for _ in range(400):
        a = np.array([rng.integers(8, DEFAULT.h - 8), rng.integers(8, DEFAULT.w - 8)])
        d = np.array([rng.integers(-6, 7), rng.integers(-6, 7)])
        if int(np.max(np.abs(d))) not in (5, 6):
            continue
        b = a + d
        # The old arithmetic: offset by the QUERY radius into a table built at 4.
        old = bool(
            (np.max(np.abs(d)) <= 6)
            & np.asarray(small)[a[0], a[1], np.clip(d[0] + 6, 0, 8), np.clip(d[1] + 6, 0, 8)]
        )
        new = bool(visible_from_table(big, jnp.asarray(a), jnp.asarray(b), 6))
        disagreements += old != new

    assert disagreements > 20, (
        f"the old indexing disagreed with the correct lookup on only {disagreements} pairs. If this is near "
        "zero the witness has stopped witnessing and DEV-030 could be reintroduced unnoticed."
    )
