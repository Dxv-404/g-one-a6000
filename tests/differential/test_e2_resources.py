"""Gate E2 over resources: regrowth and pick, and the first real exercise of replay mode.

The entropy source was built at build step 2 and had no consumer, because step 2 had no stochastic dynamics.
Regrowth is the first. **The expected failure here is tag alignment, not regrowth logic**, and it is worth
finding now while four tags exist rather than after steps 5 and 8 add more. A replay mechanism that is wrong
and untested is a worse liability than an untested subsystem.

What is injected and what is compared. The obstacle grid, the source positions, and the seeded starting grids
are all constructed once and handed to both implementations, because they are closed over at construction and
because JAX threefry and NumPy PCG64 cannot agree on a layout from the same seed. What is compared is the
per-tick dynamics, driven by identical pre-generated draws indexed by tag and tick.

Everything compared here is discrete, so every comparison is exact and no tolerance is consumed.

### Coverage is asserted, not assumed

Standing practice after gate E3 turned out to be inert: for any gate with a stochastic path, record how many
times the path was exercised and treat a low count as a failing condition rather than a passing one. An
agreement test over a path that never executes agrees trivially. `test_coverage_is_sufficient` fails if too
few placements or picks occurred, so a future change that quietly stops exercising regrowth fails here rather
than passing green.
"""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from configs.default import DEFAULT, make_relax_ladder  # noqa: E402
from src.env import resources as fast  # noqa: E402
from src.env.entropy import ReplaySource  # noqa: E402
from src.env.grid import make_obstacle_grid  # noqa: E402
from src.reference import naive_env as oracle  # noqa: E402

P = DEFAULT
N_TICKS = 60

# Small grids keep the oracle's per-cell Python loops tractable. Densities and source counts vary so that
# patches overlap in some configurations and are disjoint in others, since overlap is what makes the
# sequential ordering of sources observable.
CONFIGS = [
    dict(h=20, w=20, n_src=4, cap=5, r_src=3, density=0.15, p=0.30),
    dict(h=17, w=23, n_src=6, cap=5, r_src=3, density=0.15, p=0.50),  # non-square
    dict(h=15, w=15, n_src=6, cap=4, r_src=2, density=0.30, p=1.00),  # dense, always places
    dict(h=19, w=19, n_src=3, cap=6, r_src=4, density=0.05, p=0.20),  # sparse, wide patches
]


def _build(cfg, seed):
    """Construct one world and hand identical arrays to both implementations."""
    obstacles = make_obstacle_grid(jax.random.key(seed), cfg["h"], cfg["w"], cfg["density"])
    free = jnp.argwhere(~obstacles)
    idx = jax.random.permutation(jax.random.key(seed + 500), free.shape[0])[: cfg["n_src"]]
    sources = free[idx].astype(jnp.int32)
    grid = fast.seed_grid(jax.random.key(seed + 900), sources, obstacles, cfg["cap"], cfg["r_src"])
    return obstacles, sources, grid


def _replay(cfg, seed):
    """Pre-generate the draws once, then expose them to both implementations.

    Both sides index the same arrays by tag and tick. If the two disagree on that indexing, the values differ
    and the comparison fails, which is the point of the exercise.
    """
    k1, k2 = jax.random.split(jax.random.key(seed + 77))
    draws = {
        "regrow.food.trigger": jax.random.uniform(k1, (N_TICKS, cfg["n_src"])),
        "regrow.food.cell": jax.random.uniform(k2, (N_TICKS, cfg["n_src"])),
    }
    return ReplaySource(draws=draws), oracle.ReplayReader({k: np.array(v) for k, v in draws.items()})


def _consume(grid_j, grid_n, tick):
    """Remove one standing item from both grids identically, so regrowth keeps firing.

    Without consumption every patch sits at its cap and regrowth never triggers, which is precisely how gate
    E3 came to test nothing.
    """
    occupied = np.argwhere(np.array(grid_j) > 0)
    if occupied.shape[0] == 0:
        return grid_j, grid_n, 0
    r, c = occupied[tick % occupied.shape[0]]
    return grid_j.at[int(r), int(c)].set(0), _np_set(grid_n, int(r), int(c)), 1


def _np_set(grid, r, c):
    out = grid.copy()
    out[r, c] = 0
    return out


@pytest.mark.slow  # 6.5s at the slowest parameterisation, several others just over the line
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c['h']}x{c['w']}n{c['n_src']}p{c['p']}")
@pytest.mark.parametrize("seed", [0, 1])
def test_regrowth_matches_the_oracle_tick_by_tick(cfg, seed):
    """Compare the grid after every tick, not only at the end.

    Tick by tick because a divergence that appears and then coincidentally re-converges would be invisible in
    an endpoint comparison, and because the first tick that differs names the cause far more precisely than
    the hundredth.
    """
    obstacles, sources, grid_j = _build(cfg, seed)
    replay_j, replay_n = _replay(cfg, seed)
    grid_n = np.array(grid_j)
    obstacles_n = np.array(obstacles)
    sources_n = np.array(sources)

    assert np.array_equal(np.array(grid_j), grid_n), "the two implementations did not start identical"

    for step in range(N_TICKS):
        grid_j, grid_n, _ = _consume(grid_j, grid_n, step)

        grid_j, placed_j = fast.regrow(
            replay_j,
            grid_j,
            sources,
            obstacles,
            cfg["cap"],
            cfg["r_src"],
            cfg["p"],
            step,
            "regrow.food.trigger",
            "regrow.food.cell",
        )
        grid_n, placed_n = oracle.regrow(
            replay_n,
            grid_n,
            sources_n,
            obstacles_n,
            cfg["cap"],
            cfg["r_src"],
            cfg["p"],
            step,
            "regrow.food.trigger",
            "regrow.food.cell",
        )

        assert np.array_equal(np.array(placed_j), placed_n), (
            f"tick {step}: placement decisions differ, {np.array(placed_j)} against {placed_n}. "
            "The two implementations disagree on which sources triggered, which points at the trigger draw "
            "rather than at cell selection."
        )
        diff = np.argwhere(np.array(grid_j) != grid_n)
        assert diff.shape[0] == 0, (
            f"tick {step}: grids differ at {diff[:5].tolist()}. The same sources placed, so the two disagree "
            "on which cell the rank selected. Check that both count available cells in row-major order."
        )


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c['h']}x{c['w']}n{c['n_src']}p{c['p']}")
def test_pick_matches_the_oracle(cfg):
    """Includes contention, which is where an ordering disagreement shows up."""
    obstacles, sources, grid_j = _build(cfg, 0)
    grid_n = np.array(grid_j)
    occupied = np.argwhere(grid_n > 0)
    assert occupied.shape[0] >= 3, "the fixture must have standing items for this to test anything"

    # Deliberately include duplicates so contention is exercised rather than hoped for.
    positions = np.array(
        [occupied[0], occupied[0], occupied[1], occupied[2], occupied[2], occupied[2]], dtype=np.int32
    )
    alive = np.array([True, True, True, False, True, True])
    picking = np.array([True, True, False, True, True, True])

    food_j, target_j, gained_j, ate_j = fast.resolve_pick(
        grid_j, grid_j, jnp.asarray(positions), jnp.asarray(alive), jnp.asarray(picking)
    )
    food_n, target_n, gained_n, ate_n = oracle.resolve_pick(grid_n, grid_n, positions, alive, picking)

    assert np.array_equal(np.array(gained_j), gained_n), (
        f"score differs: {np.array(gained_j)} against {gained_n}. Contention must resolve to the lowest "
        "living, picking prey index in both."
    )
    assert np.array_equal(np.array(ate_j), ate_n), "food consumption differs"
    assert np.array_equal(np.array(target_j), target_n), "target grid differs after picking"
    assert np.array_equal(np.array(food_j), food_n), "food grid differs after picking"


def test_replay_supplies_identical_draws_to_both():
    """The premise everything else rests on, asserted directly rather than inferred from agreement.

    If this failed while the dynamics tests passed, the dynamics would be agreeing for the wrong reason.
    """
    cfg = CONFIGS[0]
    replay_j, replay_n = _replay(cfg, 0)
    for tag in ("regrow.food.trigger", "regrow.food.cell"):
        for step in (0, 1, N_TICKS - 1):
            from src.env.entropy import draw_uniform

            a = np.array(draw_uniform(replay_j, (cfg["n_src"],), tag, step))
            b = replay_n.uniform((cfg["n_src"],), tag, step)
            assert np.array_equal(a, b), (
                f"replay disagrees for {tag} at tick {step}. The two implementations index the same "
                "pre-generated array differently, so every comparison downstream is meaningless."
            )


def test_coverage_is_sufficient():
    """Standing practice after E3: assert how often the stochastic path ran, not merely that it ran.

    An agreement test over a path that never executes agrees trivially. `total > 0` reads as a power check
    and is not one, which is exactly how the first version of gate E3 looked exercised while carrying almost
    no power.
    """
    cfg = CONFIGS[1]
    obstacles, sources, grid_j = _build(cfg, 0)
    replay_j, _ = _replay(cfg, 0)
    total_placed = 0
    for step in range(N_TICKS):
        grid_j, _, _ = _consume(grid_j, np.array(grid_j), step)
        grid_j, placed = fast.regrow(
            replay_j,
            grid_j,
            sources,
            obstacles,
            cfg["cap"],
            cfg["r_src"],
            cfg["p"],
            step,
            "regrow.food.trigger",
            "regrow.food.cell",
        )
        total_placed += int(jnp.sum(placed))

    assert total_placed >= N_TICKS // 3, (
        f"only {total_placed} placements across {N_TICKS} ticks. The differential comparison is running over "
        "a path that barely executes, so agreement between the two implementations means little."
    )


def test_ladder_construction_is_shared_ground():
    """The sweep rule is one construction, so both sides and every caller use the same ladder.

    Not a differential comparison, but it belongs here: a sweep that builds ladders differently per caller
    would compare configurations that are not the ones recorded.
    """
    assert make_relax_ladder(P.d_src) == P.d_src_relax
    assert make_relax_ladder(12) == (12, 9, 6, 3, 0)

    # Heads from 1 upward. Head 0 is excluded deliberately and asserted separately below: it produces
    # (0, 0, 0, 0, 0), a ladder with one distinct level and therefore no fallback, which `make_relax_ladder`
    # refuses. This loop originally ran from 0 and contradicted that refusal, which is the defect the full
    # suite caught after an engineering-only run had hidden it for a session.
    for head in range(1, 41):
        ladder = make_relax_ladder(head)
        assert list(ladder) == sorted(ladder, reverse=True), f"head {head} produced a non-descending ladder"
        assert ladder[0] == head and ladder[-1] == 0

    with pytest.raises(AssertionError, match="distinct level"):
        make_relax_ladder(0)
