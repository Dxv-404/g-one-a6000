"""Gate E2 over the predator: the vectorised implementation against the NumPy oracle.

**This is the extension that keeps E2's coverage from falling as the environment grows.** The oracle covered
grid, visibility and resources; the predator arrives at build step 5, and shipping it without a matching
oracle would lower the fraction of the environment E2 actually witnesses while its name and reputation stayed
constant. See CLAUDE.md section 10.

What is injected and what is compared. Obstacles, prey positions, movement flags and the per-tick detection
draws are constructed once and handed to both implementations, because JAX threefry and NumPy PCG64 cannot
agree on a layout from the same seed. The vectorised side consults a precomputed visibility table; the oracle
recomputes each ray with its own Fraction arithmetic. That divergence in method is deliberate and is what
makes agreement meaningful.

**Run at both cue endpoints.** A disagreement that appears only at CUE=0.0 would be invisible in a suite that
only ever ran the default.

### Coverage is asserted, not assumed

An agreement test over a path that never executes agrees trivially. Gate E3 was inert for exactly this reason.
`test_coverage_is_sufficient` fails if too few ticks produced a target, an adjacency, real damage, or a
disengagement, so a future change that quietly stops exercising pursuit fails here rather than passing green.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import DEFAULT
from src.env import predator as fast
from src.env.grid import make_obstacle_grid
from src.env.perception import build_vis_table
from src.reference import naive_env as oracle

N_TICKS = 60
K = 4

CONFIGS = [
    dict(h=16, w=16, density=0.10, seed=0),
    dict(h=14, w=18, density=0.20, seed=1),  # non-square
    dict(h=20, w=20, density=0.05, seed=2),  # sparse
    dict(h=12, w=12, density=0.30, seed=3),  # cluttered, exercises blocked pursuit
]
CUES = [1.0, 0.0]


def _scenario(cfg, cue):
    """Build one shared scenario. Everything stochastic is drawn once here and injected into both sides."""
    params = dataclasses.replace(DEFAULT, cue_dependence=cue)
    rng = np.random.default_rng(cfg["seed"])

    key = jnp.asarray(np.array([0, cfg["seed"]], dtype=np.uint32))
    obstacles_j = make_obstacle_grid(key, cfg["h"], cfg["w"], cfg["density"])
    obstacles_n = np.asarray(obstacles_j)

    free = np.argwhere(~obstacles_n)
    picks = rng.choice(len(free), size=K + 1, replace=False)
    prey_pos = free[picks[:K]].astype(np.int32)
    pred_pos = free[picks[K]].astype(np.int32)

    # Prey teleport each tick rather than acting, because this gate tests the PREDATOR. Driving them with a
    # policy would couple the comparison to code that does not exist until step 6.
    prey_paths = np.zeros((N_TICKS, K, 2), dtype=np.int32)
    prey_moved = rng.random((N_TICKS, K)) < 0.5
    pos = prey_pos.copy()
    for t in range(N_TICKS):
        step = rng.integers(-1, 2, size=(K, 2))
        cand = np.clip(pos + step * prey_moved[t][:, None], [0, 0], [cfg["h"] - 1, cfg["w"] - 1])
        blocked = obstacles_n[cand[:, 0], cand[:, 1]]
        pos = np.where(blocked[:, None], pos, cand)
        prey_paths[t] = pos

    return dict(
        params=params,
        obstacles_j=obstacles_j,
        obstacles_n=obstacles_n,
        vis_table=build_vis_table(obstacles_j, params.r_pred),
        pred_pos=pred_pos,
        prey_paths=prey_paths,
        prey_moved=prey_moved,
        prey_alive=np.ones(K, dtype=bool),
        draws=rng.random((N_TICKS, K)),
        # Drawn once here and handed to both sides, like every other stochastic input. One per predator per
        # tick; this scenario runs a single predator, so the axis is 1.
        wander_trigger=rng.random((N_TICKS, 1)),
        wander_dir=rng.random((N_TICKS, 1)),
    )


def _run_both(s):
    """Drive both implementations tick by tick from identical state. Returns the per-tick records."""
    p = s["params"]
    jf = dict(
        pos=jnp.asarray(s["pred_pos"]),
        target=jnp.int32(fast.NO_TARGET),
        still=jnp.asarray(0.0, jnp.float32),
        engage=jnp.int32(0),
    )
    nf = dict(pos=s["pred_pos"].copy(), target=oracle.NO_TARGET, still=0.0, engage=0)

    rows = []
    for t in range(N_TICKS):
        out_j = fast.step_predator(
            s["obstacles_j"],
            s["vis_table"],
            jf["pos"],
            jnp.asarray(100.0),
            jf["target"],
            jf["still"],
            jf["engage"],
            jnp.asarray(s["prey_paths"][t]),
            jnp.asarray(s["prey_alive"]),
            jnp.asarray(s["prey_moved"][t]),
            jnp.asarray(s["draws"][t]),
            # **The SAME wander draws to both sides.** A differential comparing two implementations handed
            # different randomness compares nothing, and the wander is the newest place that could happen.
            jnp.asarray(s["wander_trigger"][t][0]),
            jnp.asarray(s["wander_dir"][t][0]),
            p,
        )
        out_n = oracle.step_predator(
            s["obstacles_n"],
            nf["pos"],
            100.0,
            nf["target"],
            nf["still"],
            nf["engage"],
            s["prey_paths"][t],
            s["prey_alive"],
            s["prey_moved"][t],
            s["draws"][t],
            s["wander_trigger"][t][0],
            s["wander_dir"][t][0],
            p,
        )
        rows.append((out_j, out_n))
        jf = dict(
            pos=out_j["pred_pos"],
            target=out_j["target"],
            still=out_j["still"],
            engage=out_j["engage"],
        )
        nf = dict(
            pos=out_n["pred_pos"],
            target=out_n["target"],
            still=out_n["still"],
            engage=out_n["engage"],
        )
    return rows


@pytest.fixture(scope="module")
def all_runs():
    return {(i, cue): _run_both(_scenario(cfg, cue)) for i, cfg in enumerate(CONFIGS) for cue in CUES}


@pytest.mark.parametrize("cue", CUES)
@pytest.mark.parametrize("cfg_i", range(len(CONFIGS)))
def test_predator_matches_the_oracle_tick_by_tick(all_runs, cfg_i, cue):
    """Every discrete field exactly, damage within tolerance. Compared per tick so the first divergence
    is reported at the tick it happened rather than at the end of the episode."""
    for t, (j, n) in enumerate(all_runs[(cfg_i, cue)]):
        where = f"config {cfg_i}, cue {cue}, tick {t}"
        assert [int(x) for x in j["pred_pos"]] == [int(x) for x in n["pred_pos"]], f"pred_pos: {where}"
        assert int(j["target"]) == int(n["target"]), f"target: {where}"
        assert int(j["engage"]) == int(n["engage"]), f"engage: {where}"
        assert bool(j["sprinted"]) == bool(n["sprinted"]), f"sprinted: {where}"
        assert bool(j["adjacent"]) == bool(n["adjacent"]), f"adjacent: {where}"
        assert [bool(x) for x in j["reported"]] == [bool(x) for x in n["reported"]], f"reported: {where}"
        assert float(j["still"]) == pytest.approx(float(n["still"]), abs=DEFAULT.e2_atol), f"still: {where}"
        np.testing.assert_allclose(
            np.asarray(j["damage_per_prey"]),
            n["damage_per_prey"],
            atol=DEFAULT.e2_atol,
            rtol=DEFAULT.e2_rtol,
            err_msg=f"damage: {where}",
        )


@pytest.mark.parametrize("cue", CUES)
def test_coverage_is_sufficient(all_runs, cue):
    """The paths this gate claims to compare must actually execute.

    Without this, a scenario where the predator never acquires a target would agree perfectly and prove
    nothing. E3 shipped in exactly that state.
    """
    targeted = adjacent = damaged = 0
    for cfg_i in range(len(CONFIGS)):
        for j, _ in all_runs[(cfg_i, cue)]:
            targeted += int(int(j["target"]) != fast.NO_TARGET)
            adjacent += int(bool(j["adjacent"]))
            damaged += int(float(np.sum(np.asarray(j["damage_per_prey"]))) > 0.0)

    total = len(CONFIGS) * N_TICKS
    assert targeted >= total * 0.2, (
        f"only {targeted} of {total} ticks had a target at cue {cue}; pursuit is barely exercised"
    )
    assert adjacent >= 5, f"only {adjacent} adjacency events at cue {cue}; capture is barely exercised"
    assert damaged >= 3, f"only {damaged} damaging ticks at cue {cue}; the damage path is barely exercised"


def test_the_two_cue_endpoints_actually_differ(all_runs):
    """A power check on the parameterisation itself.

    If CUE=0.0 and CUE=1.0 produced identical trajectories, every test above would pass while
    `cue_dependence` did nothing at all. That is the construction class from CLAUDE.md section 11: the
    comparison cannot see the condition it claims to vary.
    """
    differed = False
    for cfg_i in range(len(CONFIGS)):
        one = all_runs[(cfg_i, 1.0)]
        zero = all_runs[(cfg_i, 0.0)]
        for (j1, _), (j0, _) in zip(one, zero, strict=True):
            if float(np.sum(np.asarray(j1["damage_per_prey"]))) != float(
                np.sum(np.asarray(j0["damage_per_prey"]))
            ):
                differed = True
                break
        if differed:
            break
    assert differed, (
        "cue_dependence 0.0 and 1.0 produced identical damage across every tick of every config. "
        "The parameter is inert, or no still prey was ever adjacent to the predator"
    )
