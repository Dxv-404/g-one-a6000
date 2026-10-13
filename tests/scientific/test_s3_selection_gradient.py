"""Gate S3, selection gradient. Build step 12.

**The spec's line.** "Fitness variance across the population is meaningfully above zero and not degenerate.
Selection has nothing to act on and nothing will evolve, however healthy learnability looks." And the
paragraph under the table: S3 is the quiet killer, because a world can look healthy on every other measure
while fitness variance sits near zero, and evolution then produces nothing for reasons invisible unless
specifically checked.

**Two claims in that line, and they are handled differently here.** "Not degenerate" is unambiguous: the
fitness vector is not constant, in any generation. That is asserted and binds. "Meaningfully above zero" has
no number in the spec, and per CLAUDE.md an ambiguous specification is an open question rather than a guess,
so the third test below carries a PROPOSED threshold, reports, and does not bind until Q-050 is ruled.

**The quantity the proposed threshold is stated in is the mechanism, not the symptom.** Variance is a
symptom with several sufficient causes; the mechanism selection runs on is the tournament. A size-`t_sel`
tournament whose candidates all share one fitness value is decided by tie-breaking rather than by fitness,
and with replacement the probability of that is

    P(tied) = sum over distinct values v of (n_v / N) ** t_sel

so `1 - P(tied)` is the fraction of parent selections that fitness actually decides. That is what "selection
has something to act on" means operationally, and it is what run 23 measured at generation zero: 13 to 23
percent decided, with 235 to 244 of 256 genomes scoring exactly zero. The threshold question is what fraction
counts as meaningful, and it is the owner's.

**The fixture runs the real loop at the real scale.** Ten generations of the standard configuration, one
seed, which is about 15 seconds of compile and 12 of run. It is a coverage floor on generations, not a claim
about the 300 the design runs: the gate checks that the gradient exists where the loop starts, which is where
the quiet killer would act.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from configs.default import DEFAULT
from src.agent import network as net
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius
from src.evo import genome, loop

#: Coverage floor. Enough generations to see the gradient move, not a claim about the design's 300.
N_GENERATIONS = 10
SEED = 0

#: Strictness threshold, PROPOSED and NOT YET BINDING. The fraction of tournaments decided by fitness rather
#: than by tie-breaking, at the last fixture generation. Q-050 asks the owner for the number; until it is
#: ruled the test that uses it is marked xfail so it reports without blocking step 12. gate-thresholds.md.
S3_MIN_DECIDED_FRACTION = 0.5


def decided_fraction(fit: np.ndarray, t_sel: int) -> float:
    """Fraction of size-`t_sel` tournaments, sampled with replacement, whose candidates do not all tie."""
    _, counts = np.unique(fit, return_counts=True)
    return 1.0 - float(np.sum((counts / fit.size) ** t_sel))


@pytest.fixture(scope="module")
def fitness_by_generation() -> np.ndarray:
    """(N_GENERATIONS, p_pop) fitness of the evaluated population, generation by generation."""
    p = DEFAULT
    ob = grid.make_obstacle_grid(jax.random.key(SEED), p.h, p.w, p.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(SEED), ob, p)
    static = E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )
    template = net.init_params(jax.random.key(SEED), p)
    pop = loop.initial_population(jax.random.key(SEED + 1), template, p)
    step = loop.make_generation_step(genome.unflattener(template), static, p)
    key = jax.random.key(100)
    fits = []
    for _ in range(N_GENERATIONS):
        key, k = jax.random.split(key)
        pop, fit = step(k, pop)
        fits.append(np.asarray(fit, np.float64))
    return np.stack(fits)


def test_the_fitness_vector_has_the_population_shape(fitness_by_generation):
    """Determined. Guards the fixture rather than the science: a wrong axis would make every test below
    a statement about the wrong thing."""
    assert fitness_by_generation.shape == (N_GENERATIONS, DEFAULT.p_pop)


def test_fitness_variance_is_above_zero_in_every_generation(fitness_by_generation):
    """The degenerate case, exactly. A constant fitness vector gives selection nothing to act on."""
    var = fitness_by_generation.var(axis=1)
    assert np.all(var > 0.0), f"fitness variance is exactly zero in generations {np.flatnonzero(var == 0.0)}"


def test_more_than_one_fitness_value_exists_in_every_generation(fitness_by_generation):
    """The same claim from the other side. Variance above zero is a property of floats; distinct values
    above one is a property of the tournament's inputs, and both are cheap."""
    distinct = np.array([np.unique(f).size for f in fitness_by_generation])
    assert np.all(distinct > 1), f"a single fitness value in generations {np.flatnonzero(distinct <= 1)}"


@pytest.mark.xfail(
    strict=False,
    reason="Q-050: the spec says 'meaningfully above zero' and gives no number. Reports, does not bind",
)
def test_most_tournaments_are_decided_by_fitness_by_the_last_generation(fitness_by_generation):
    """The PROPOSED meaningful-gradient criterion. See the module docstring and Q-050."""
    fracs = [decided_fraction(f, DEFAULT.t_sel) for f in fitness_by_generation]
    zeros = [int(np.sum(f == 0.0)) for f in fitness_by_generation]
    distinct = [int(np.unique(f).size) for f in fitness_by_generation]
    print("\n  gen  decided  distinct  zero_of_" + str(DEFAULT.p_pop))
    for g, (fr, d, z) in enumerate(zip(fracs, distinct, zeros, strict=True)):
        print(f"  {g:3d}  {fr:7.3f}  {d:8d}  {z:8d}")
    assert fracs[-1] >= S3_MIN_DECIDED_FRACTION, (
        f"only {fracs[-1]:.3f} of tournaments are decided by fitness at generation {N_GENERATIONS - 1}, "
        f"against a proposed floor of {S3_MIN_DECIDED_FRACTION}"
    )


def test_the_decided_fraction_can_fail():
    """Can-it-fail for the criterion itself, so a broken formula cannot pass the gate vacuously."""
    assert decided_fraction(np.zeros(256), 3) == 0.0
    assert decided_fraction(np.arange(256.0), 3) > 0.99
    # two values, half and half: P(tied) = 2 * 0.5 ** 3 = 0.25
    half = np.concatenate([np.zeros(128), np.ones(128)])
    assert abs(decided_fraction(half, 3) - 0.75) < 1e-12
