"""Gate E2, differential equivalence against the NumPy reference oracle.

Passing condition, from the build spec: with stochasticity injected identically into both implementations,
the vectorised implementation reproduces the NumPy reference exactly on discrete state and within `E2_ATOL`
and `E2_RTOL` on float state, over a suite of seeds and configurations. Tolerances live in the config, never
hardcoded here, and loosening them requires a `decisions.md` entry.

**Scope at build step 2.** Only the static grid and visibility exist, so only those are compared. Everything
currently compared is discrete, so every comparison is exact and no tolerance is consumed yet; the tolerances
are imported and asserted to exist so that the day float state appears, the mechanism is already in place
rather than being invented under pressure.

**Stochasticity is injected, not generated.** JAX threefry and NumPy PCG64 produce different streams from the
same seed, so both implementations receive the identical obstacle grid rather than each building its own. The
obstacle layout is static and closed over at construction, which is why it is injected as a whole array
rather than through the tagged entropy source; the entropy source covers per-tick draws, which arrive from
build step 3 onward.
"""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from configs.default import DEFAULT, EnvParams  # noqa: E402
from src.env import grid as fast_grid  # noqa: E402
from src.env import perception as fast_perception  # noqa: E402
from src.reference import naive_env as oracle  # noqa: E402

# The suite of configurations. Deliberately not only the default: a vectorisation bug can hide behind a
# square grid, an even radius, or a density that happens to produce a convenient layout. Small grids keep the
# oracle's quadruple loop tractable, since it is O(H*W*S*S) with a Python-level ray inside.
CONFIGS = [
    dict(h=17, w=17, r=4, density=0.15),  # default shape, reduced size
    dict(h=13, w=21, r=4, density=0.15),  # non-square, catches a row/col transposition
    dict(h=15, w=15, r=3, density=0.30),  # odd radius, dense: many blocked rays
    dict(h=11, w=11, r=2, density=0.05),  # small radius, sparse: almost everything visible
    dict(h=19, w=13, r=4, density=0.45),  # non-square and very dense
]

SEEDS = [0, 1, 2]


def _shared_obstacle_grid(seed, h, w, density):
    """Build once in JAX, hand the identical layout to both implementations."""
    g = fast_grid.make_obstacle_grid(jax.random.key(seed), h, w, density)
    return g, np.array(g)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c['h']}x{c['w']}r{c['r']}d{c['density']}")
@pytest.mark.parametrize("seed", SEEDS)
def test_visibility_table_matches_the_oracle(cfg, seed):
    """The comparison that matters most at this step.

    `build_vis_table` is a double `vmap` over cells and patch offsets. An axis error there produces a table
    that is wrong in a structured, plausible-looking way: still boolean, still roughly the right density of
    visible cells, and still symmetric if the error is symmetric. The looped oracle cannot make that class of
    mistake.
    """
    g_jax, g_np = _shared_obstacle_grid(seed, cfg["h"], cfg["w"], cfg["density"])

    fast = np.array(fast_perception.build_vis_table(g_jax, cfg["r"]))
    slow = oracle.build_vis_table(g_np, cfg["r"])

    assert fast.shape == slow.shape, f"shape mismatch: {fast.shape} against {slow.shape}"
    n_diff = int(np.sum(fast != slow))
    if n_diff:
        where = np.argwhere(fast != slow)[:5]
        raise AssertionError(
            f"{n_diff} of {fast.size} visibility entries disagree; first few at {where.tolist()}"
        )


@pytest.mark.slow  # 38s and 35s measured on the 17x17 and 13x21 grids: every ray on every cell pair
@pytest.mark.parametrize("cfg", CONFIGS[:3], ids=lambda c: f"{c['h']}x{c['w']}r{c['r']}d{c['density']}")
def test_single_ray_matches_the_oracle(cfg):
    """Compare the ray itself, not only the symmetrised result.

    Worth separating, because `visible` conjoins two rays and a bug that inverts both directions identically
    would survive a comparison of the conjunction alone while changing which single rays are clear.
    """
    g_jax, g_np = _shared_obstacle_grid(0, cfg["h"], cfg["w"], cfg["density"])
    r = cfg["r"]
    mismatches = 0
    for row in range(0, cfg["h"], 2):
        for col in range(0, cfg["w"], 2):
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    tr, tc = row + dr, col + dc
                    if not (0 <= tr < cfg["h"] and 0 <= tc < cfg["w"]):
                        continue
                    a = jnp.array([row, col], dtype=jnp.int32)
                    b = jnp.array([tr, tc], dtype=jnp.int32)
                    fast = bool(fast_perception.ray_clear(g_jax, a, b, r))
                    slow = oracle.ray_clear(g_np, (row, col), (tr, tc))
                    mismatches += fast != slow
    assert mismatches == 0, f"{mismatches} single rays disagree between implementations"


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c['h']}x{c['w']}r{c['r']}d{c['density']}")
@pytest.mark.parametrize("seed", SEEDS)
def test_movement_resolution_matches_the_oracle(cfg, seed):
    """Every cell, every direction, walking and sprinting, compared exactly.

    Positions are discrete, so this is exact equality. It covers the corner-clipping rule, the sprint
    intermediate cell, the boundary, and the stay-put-when-blocked behaviour in one sweep, over layouts dense
    enough that all of those actually occur.
    """
    g_jax, g_np = _shared_obstacle_grid(seed, cfg["h"], cfg["w"], cfg["density"])
    h, w = cfg["h"], cfg["w"]

    rows, cols = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    positions = jnp.stack([rows, cols], axis=-1).reshape(-1, 2)

    for sprint in (False, True):
        for d_index, d in enumerate(oracle.DIRECTIONS):
            direction = jnp.broadcast_to(jnp.array(d, dtype=jnp.int32), positions.shape)
            flags = jnp.full(positions.shape[0], sprint)
            fast = np.array(fast_grid.resolve_move(g_jax, positions, direction, flags))

            slow = np.array(
                [oracle.resolve_move(g_np, (int(p[0]), int(p[1])), d, sprint) for p in np.array(positions)]
            )

            assert np.array_equal(fast, slow), (
                f"movement disagrees at direction {d_index} {d}, sprint={sprint}, seed={seed}: "
                f"{int(np.sum(np.any(fast != slow, axis=1)))} of {len(fast)} positions differ"
            )


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c['h']}x{c['w']}r{c['r']}d{c['density']}")
def test_legality_matches_the_oracle(cfg):
    """`legal_move` compared directly, since `resolve_move` can mask a legality bug.

    A move judged illegal and a move judged legal but landing back on the origin are indistinguishable in the
    resolved position whenever the direction happens to be blocked, so comparing only positions could let a
    legality inversion through.
    """
    g_jax, g_np = _shared_obstacle_grid(0, cfg["h"], cfg["w"], cfg["density"])
    h, w = cfg["h"], cfg["w"]
    rows, cols = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    positions = jnp.stack([rows, cols], axis=-1).reshape(-1, 2)

    for sprint in (False, True):
        for d in oracle.DIRECTIONS:
            direction = jnp.broadcast_to(jnp.array(d, dtype=jnp.int32), positions.shape)
            flags = jnp.full(positions.shape[0], sprint)
            fast = np.array(fast_grid.legal_move(g_jax, positions, direction, flags))
            slow = np.array(
                [oracle.legal_move(g_np, (int(p[0]), int(p[1])), d, sprint) for p in np.array(positions)]
            )
            assert np.array_equal(fast, slow), f"legality disagrees at direction {d}, sprint={sprint}"


def test_direction_orderings_are_identical():
    """D-009. A mismatch here makes every other comparison compare different action semantics.

    This is checked explicitly rather than left implicit in the movement tests, because if the orderings were
    a rotation of each other the movement comparison would fail with a confusing per-direction message rather
    than naming the actual cause.
    """
    fast = [tuple(int(x) for x in d) for d in np.array(fast_grid.DIRECTIONS)]
    assert fast == list(oracle.DIRECTIONS), (
        "direction orderings differ between the vectorised implementation and the oracle. "
        "Gate E2 would compare different action semantics. See D-009."
    )


def test_free_cell_counts_agree():
    """The exact-count obstacle placement of D-011, seen from both sides."""
    for cfg in CONFIGS:
        g_jax, g_np = _shared_obstacle_grid(0, cfg["h"], cfg["w"], cfg["density"])
        expected = cfg["h"] * cfg["w"] - round(cfg["h"] * cfg["w"] * cfg["density"])
        assert int(fast_grid.free_cell_count(g_jax)) == expected
        assert oracle.free_cell_count(g_np) == expected


def test_tolerances_come_from_the_config_not_from_this_file():
    """Gate E2 requires tolerances to live in the config so that loosening one is visible in a diff.

    Nothing compared at build step 2 is float, so no tolerance is consumed yet. This asserts the values exist
    and are sane, so the mechanism is in place before the first float comparison rather than being invented
    at the moment a test is failing, which is exactly when loosening one looks reasonable.
    """
    assert DEFAULT.e2_atol == 1e-6
    assert DEFAULT.e2_rtol == 1e-5
    assert 0 < DEFAULT.e2_atol < DEFAULT.e2_rtol, (
        "a mixed tolerance is used because pools legitimately reach exactly zero, where relative tolerance "
        "is undefined, so the absolute term must be the tighter of the two"
    )
    assert isinstance(EnvParams().e2_atol, float)
