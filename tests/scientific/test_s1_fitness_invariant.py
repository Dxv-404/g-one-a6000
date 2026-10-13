"""Gate S1. Invariant I1: fitness is `prey_score` and nothing else, enforced by signature. Build step 10.

**This is the keystone gate of the project.** If fitness can see survival time, energy, health, or distance
from the predator, then any self-protective behaviour that emerges was selected for directly and the central
claim collapses. Everything else in the repository is downstream of this one function staying narrow.

**Four assertions, deliberately at different levels, because a single level is escapable.**

    signature      the function takes one array and no state object. Structural
    imports        the module cannot reach the environment at all. Structural
    body           nothing in the compiled code names a state field. Textual, and the weakest
    behaviour      a survivor that collected nothing scores exactly zero. Empirical

The behavioural one is last and is the one that would survive a clever violation. A widened signature that
still returned the right number for a survivor would pass the first three and fail the fourth only if the
extra input actually changed the answer, which is precisely when it matters.

**CLAUDE.md section 14 lists this gate as previously not existing**, and describes I1 in three places as
structurally impossible to violate. Until this file, that was a promise about code that did not exist.
"""

import inspect
import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import DEFAULT
from src.agent import network as net
from src.env import dynamics, grid, resources
from src.env import env as E
from src.env.entropy import PRNGSource
from src.env.perception import build_vis_table
from src.evo import fitness as F

#: Short horizon on purpose: prey must SURVIVE for the keystone assertion to mean anything, and over a full
#: 400 ticks a policy that never eats starves. 15 ticks is comfortably inside the energy budget.
P = DEFAULT.__class__(t_horizon=15)

#: Names that would indicate fitness reading the body or the clock. Not exhaustive and not meant to be; the
#: signature and import checks are the structural guarantees and this is the cheap textual backstop.
STATE_NAMES = (
    "energy",
    "stamina",
    "health",
    "alive",
    "death",
    "step",
    "tick",
    "horizon",
    "pred",
    "predator",
    "dist",
    "posture",
    "moved",
    "state",
    "EnvState",
)


def _stay_policy(weights, obs, hidden, step_index):
    """Emit STAY every tick. Collects nothing, because collecting requires PICK."""
    del weights, step_index
    return jnp.full((obs.shape[0],), dynamics.ACTION_STAY, jnp.int32), hidden


# ------------------------------------------------------------------- 1. the signature is the invariant --
def test_fitness_takes_a_score_array_and_nothing_else():
    """**Invariant I1 as a structural property rather than a rule.**

    A function that is not given the state cannot read the state. This asserts the parameter count, so a
    second parameter of any kind fails here whatever it is called and whatever its default.
    """
    sig = inspect.signature(F.fitness)
    params = list(sig.parameters.values())
    assert len(params) == 1, (
        f"fitness takes {len(params)} parameters: {[p.name for p in params]}. Invariant I1 is enforced by "
        "this signature, and a second parameter is the single most damaging change available in this "
        "repository. It is lifted only by a decision recorded in decisions.md with promote: yes."
    )
    (p0,) = params
    assert p0.kind in (p0.POSITIONAL_ONLY, p0.POSITIONAL_OR_KEYWORD), (
        f"fitness's parameter is {p0.kind}, so it could absorb extra arguments"
    )
    assert p0.default is inspect.Parameter.empty, "fitness's parameter has a default, so it can be omitted"

    hints = typing.get_type_hints(F.fitness)
    assert hints.get(p0.name) is jax.Array, f"{p0.name} is annotated {hints.get(p0.name)!r}, not Array"
    assert hints.get("return") is jax.Array, f"fitness returns {hints.get('return')!r}, not Array"


def test_the_fitness_module_cannot_reach_the_environment():
    """The second structural guarantee. A narrow signature is escapable by importing the state type or a
    helper that reads it, so the module's imports are asserted rather than its parameter list alone.

    Same argument that makes `src/reference/` provably independent of `src/env/`."""
    src = inspect.getsource(F)
    for forbidden in ("src.env", "from src import env", "EnvState", "Accumulators"):
        assert forbidden not in src.replace("`src/env/`", "").replace("src/env/", ""), (
            f"src/evo/fitness.py references {forbidden!r}. Fitness must not be able to reach environment "
            "state even indirectly."
        )


def test_nothing_in_the_body_names_a_state_field():
    """The textual backstop, and the weakest of the four. Named as weak so nobody mistakes it for the
    guarantee: it inspects the compiled code's names, which a determined violation would simply rename."""
    names = set(F.fitness.__code__.co_names) | set(F.fitness.__code__.co_varnames)
    hits = sorted(n for n in names if any(s in n.lower() for s in STATE_NAMES))
    assert not hits, f"fitness's code references {hits}, which look like state rather than score"


# ------------------------------------------------------- 2. the keystone, asserted on behaviour --
def test_a_survivor_that_collected_nothing_scores_exactly_zero():
    """**The assertion CLAUDE.md names as S1's keystone.**

    An agent that survives the full horizon and collects nothing must score exactly zero. Not nearly zero:
    any survival term, however small, shows up here as a positive number, and "however small" is exactly how
    such a term would be introduced.

    Run rather than constructed, because a synthetic all-zero array would pass against a fitness function
    that adds a survival bonus keyed on a state field it was never given. This drives a real episode.
    """
    ob = grid.make_obstacle_grid(jax.random.key(0), P.h, P.w, P.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(0), ob, P)
    static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, P.r), food_src=fs, target_src=ts)
    w = net.init_params(jax.random.key(0), P)
    state, _ = E.run_episode(
        jax.random.key(1), w, static, PRNGSource(jax.random.key(0)), P, policy=_stay_policy
    )

    alive = np.asarray(state.prey_alive).astype(bool)
    score = np.asarray(state.prey_score)

    # Coverage, before the assertion that depends on it. Without survivors this test is inert, and an inert
    # keystone is worse than no keystone.
    assert alive.any(), (
        f"no prey survived {P.t_horizon} ticks, so this test asserts nothing about survivors. Shorten the "
        "horizon or check that the stay policy is not being killed by something new."
    )
    survivors_scored_nothing = score[alive] == 0
    assert survivors_scored_nothing.all(), (
        f"a surviving prey scored {score[alive].max()} without ever emitting PICK; the score path has "
        "changed and this test's premise no longer holds"
    )

    f = F.fitness(jnp.asarray(score)[None, None, :])
    print(
        f"\n  S1: {int(alive.sum())} of {P.k} prey survived {P.t_horizon} ticks, fitness {float(f[0]):.10f}"
    )
    assert float(f[0]) == 0.0, (
        f"agents that survived the full horizon and collected nothing scored {float(f[0])!r}, not exactly "
        "zero. Something other than collected targets is reaching fitness, which is invariant I1 broken and "
        "the central claim of the project with it."
    )


def test_fitness_moves_only_when_the_score_moves():
    """The complement: fitness must respond to score, or the gate above is satisfied by a function that
    returns zero unconditionally. Two ways to be wrong and this covers the other one."""
    base = jnp.zeros((1, P.n_episodes if hasattr(P, "n_episodes") else 4, P.k))
    assert float(F.fitness(base)[0]) == 0.0
    bumped = base.at[0, 0, 0].set(1.0)
    assert float(F.fitness(bumped)[0]) > 0.0, (
        "raising one prey's collected count did not raise fitness, so fitness is not reading the score"
    )


@pytest.mark.parametrize("shape", [(1, 4, 8), (16, 4, 8), (256, 4, 8)])
def test_each_genome_is_scored_on_its_own_episodes_only(shape):
    """**Genome isolation, and the fixture is the whole point of this test.**

    A first version filled the array with `jnp.ones` and asserted the result was all ones. That passes
    against a fitness function that averages across the GENOME axis as well as the sample axes, because on
    uniform data a global mean and a per-genome mean are the same number. The break survived falsification
    and the gate could not have seen it. Third cause: the fixture hid it.

    So every genome gets a distinct value, and each one's fitness is checked against its own mean. Selection
    on a fitness that mixed genomes together would be selection on noise, and nothing downstream would
    report it: the shapes would be right and the numbers would be plausible.

    **The general rule, from CLAUDE.md: before comparing, verify that every compared quantity actually
    varies.** Fixtures routinely use defaults, defaults are frequently uniform, and uniform is the value that
    makes a cross-axis reduction invisible.
    """
    g = shape[0]
    per_genome = jnp.arange(1, g + 1, dtype=jnp.float32)
    x = jnp.broadcast_to(per_genome[:, None, None], shape)
    out = F.fitness(x)

    assert out.shape == shape[:-2], f"fitness({shape}) returned shape {out.shape}, expected {shape[:-2]}"
    assert np.allclose(np.asarray(out), np.asarray(per_genome)), (
        "a genome's fitness does not equal the mean of its own episodes. If the values are all equal to the "
        "population mean, fitness is reducing the genome axis too, and every genome would receive the same "
        "score however it behaved."
    )


def test_one_genomes_score_cannot_change_anothers():
    """The same property stated as an intervention rather than as an identity, because they fail to
    different bugs: an identity catches a wrong reduction, an intervention catches a leak through a shared
    buffer or an in-place update."""
    x = jnp.zeros((4, 4, 8))
    before = np.asarray(F.fitness(x))
    after = np.asarray(F.fitness(x.at[0].set(10.0)))
    assert after[0] != before[0], "raising genome 0's score did not change genome 0's fitness"
    assert np.array_equal(after[1:], before[1:]), (
        f"raising genome 0's score changed the other genomes' fitness, from {before[1:]} to {after[1:]}"
    )
