"""The genetic algorithm's mechanics. Build step 10.

**S7 cannot see any of this.** A GA with inverted selection, unmutated offspring, or mutilated elites
reproduces from its config and seed exactly as faithfully as a correct one, so every assertion here is
outside S7's reach by construction. Same argument as gate S9's random floor: the outcome has more than one
sufficient cause, so the mechanism has to be asserted directly.

**Synthetic fitness throughout, and that is what makes these fast and sharp.** Nothing here runs an episode.
Fitness is supplied as an array with a known ordering, so the question "did selection prefer the fitter
genome" has an exact answer rather than a statistical one.

The scheme these check, from the build spec: tournament selection at `t_sel`, elitism preserving the top
`n_elite` unchanged, Gaussian mutation at `sigma`, no crossover.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import DEFAULT
from src.evo import loop

P = DEFAULT.__class__(p_pop=16, n_elite=2, t_sel=3, sigma=0.01)

#: Genome i is the constant row i, so a genome's identity is readable off any one of its genes and a
#: crossover would show up as a row that is not constant.
GENOMES = jnp.arange(P.p_pop, dtype=jnp.float32)[:, None] * jnp.ones((1, 5), jnp.float32)

#: Fitness equal to the index, so "fitter" and "higher index" are the same statement.
FITNESS = jnp.arange(P.p_pop, dtype=jnp.float32)


def test_the_elites_are_the_fittest_and_are_copied_unmutated():
    """**"Preserving the top n_elite unchanged" means unchanged**, and a mutated elite is not an elite. This
    is also the property that makes best-fitness monotone across generations, which is what anyone reading a
    learning curve assumes without checking."""
    nxt = np.asarray(loop.next_population(jax.random.key(0), GENOMES, FITNESS, P))
    elites = nxt[: P.n_elite]

    identities = sorted(int(round(float(r[0]))) for r in elites)
    assert identities == [P.p_pop - 2, P.p_pop - 1], (
        f"the elite block holds genomes {identities}, not the two fittest {[P.p_pop - 2, P.p_pop - 1]}"
    )
    for row in elites:
        assert len(set(np.round(row, 6))) == 1, f"an elite row is not constant, so it was altered: {row}"


def test_selection_prefers_fitter_genomes():
    """A size-3 tournament's expected winner is the maximum of three uniform draws, which for indices 0..15
    is about 11.25. Asserted against uniform rather than against that figure, because the exact expectation
    depends on the tie rule and the point is the direction and its magnitude."""
    parents = np.asarray(loop.tournament(jax.random.key(1), FITNESS, P.t_sel, 8192))
    uniform_mean = (P.p_pop - 1) / 2
    print(f"\n  mean parent index {parents.mean():.2f}, uniform would be {uniform_mean:.2f}")
    assert parents.mean() > uniform_mean + 2.0, (
        f"tournament selection produced a mean parent index of {parents.mean():.2f} against a uniform "
        f"{uniform_mean:.2f}. Selection is not preferring fitter genomes, so the population drifts."
    )
    assert parents.max() == P.p_pop - 1, "the fittest genome was never selected"
    assert parents.min() < P.p_pop - 1, "only one genome was ever selected; selection has collapsed"


def test_a_larger_tournament_selects_harder():
    """**Responsiveness, so `t_sel` is a live parameter rather than an inert one.** A tournament that
    ignored its size would pass the test above and be wrong, which is the construction failure in the
    green-and-meaningless taxonomy."""
    means = [
        float(np.mean(np.asarray(loop.tournament(jax.random.key(2), FITNESS, t, 8192)))) for t in (1, 2, 5)
    ]
    print(f"  mean parent index at t_sel 1, 2, 5: {[round(m, 2) for m in means]}")
    assert means[0] < means[1] < means[2], (
        f"selection pressure does not increase with tournament size: {means}. t_sel is not being used."
    )


def test_offspring_are_mutated_and_at_the_right_scale():
    """Mutation must happen and must be sigma-scaled. A mutation of zero makes the GA a copier, and one of
    the wrong scale changes the search without changing any recorded parameter."""
    nxt = np.asarray(loop.next_population(jax.random.key(3), GENOMES, FITNESS, P))
    offspring = nxt[P.n_elite :]
    residual = offspring - np.round(offspring)
    sd = float(np.std(residual))
    print(f"  offspring residual sd {sd:.5f} against sigma {P.sigma}")
    assert sd > 0.0, "offspring are identical to their parents; no mutation is being applied"
    assert 0.5 * P.sigma < sd < 2.0 * P.sigma, (
        f"offspring residual sd is {sd:.5f} but sigma is {P.sigma}. The mutation scale in use is not the "
        "one recorded in the config, so the search is not the one the provenance describes."
    )


def test_there_is_no_crossover():
    """The spec says no crossover, matching the deep-neuroevolution convention. Each genome is a constant
    row here, so a recombined offspring is one whose genes came from two different parents and is therefore
    not constant once the mutation is rounded away."""
    nxt = np.asarray(loop.next_population(jax.random.key(4), GENOMES, FITNESS, P))
    for row in nxt[P.n_elite :]:
        origins = set(np.round(row).astype(int).tolist())
        assert len(origins) == 1, (
            f"an offspring's genes trace to more than one parent {sorted(origins)}, which is crossover. The "
            "spec specifies none."
        )


def test_the_population_shape_is_preserved():
    """A generation that changed the population size would break the vmapped evaluation's shape contract and
    the memory budget with it."""
    nxt = loop.next_population(jax.random.key(5), GENOMES, FITNESS, P)
    assert nxt.shape == GENOMES.shape, f"population shape changed from {GENOMES.shape} to {nxt.shape}"
    assert P.n_elite + (P.p_pop - P.n_elite) == P.p_pop, "elite and offspring counts do not partition"


def test_the_config_hash_does_not_use_pythons_salted_hash():
    """Cheap and local. The behavioural version lives in S7, which compares across processes; this one fails
    at the moment someone writes it rather than one subprocess later.

    **Asserted on the code object rather than on the source text.** A first version searched the source for
    `hash(` and failed against the correct implementation, because the DOCSTRING explains why Python's `hash`
    is unsuitable and therefore contains the string. `co_names` holds the identifiers the compiled function
    actually references, so a mention in prose cannot reach it. Gate S1's own docstring names the textual
    form as the weakest of its four levels, which is exactly how this went wrong.
    """
    names = set(loop.config_hash.__code__.co_names)
    assert "hashlib" in names, f"config_hash does not reference hashlib; it references {sorted(names)}"
    assert "hash" not in names, (
        "config_hash calls Python's built-in hash, which is salted per process, so the same config would "
        "fingerprint differently in two runs and S7 would fail for a reason unrelated to the science"
    )


def test_every_keyed_constructor_responds_to_its_key():
    """**Added because a falsification break survived**, and the reason it survived is worth keeping.

    The break made `initial_population` build from a constant key instead of the one it was given. Gate S7's
    "a different seed gives a different run" stayed green, because `_s7_run` derives five things from the
    seed: the obstacle grid, the resource sources, the weight template, the initial population, and the
    generation keys. **Any one of the five changing satisfies that assertion**, so it cannot isolate a
    component that silently ignores its key.

    That is the symptom-with-many-sufficient-causes class, the same one that let a constant policy pass as
    gate S9's random floor. The only thing that separates them is asserting the mechanism: this function,
    two keys, different output.

    Cheap to state and it generalises: any constructor taking a key should be checked here as it is added,
    because "the run came out different" will never localise which one stopped listening.
    """
    from configs.default import DEFAULT

    q = DEFAULT.__class__(p_pop=4)
    a = np.asarray(loop.initial_population(jax.random.key(0), None, q))
    b = np.asarray(loop.initial_population(jax.random.key(1), None, q))

    assert a.shape == b.shape == (q.p_pop, a.shape[1])
    assert not np.array_equal(a, b), (
        "initial_population returned the same population for two different keys, so it is not using the key "
        "it was given. A run would then start from a fixed generation zero whatever seed it was launched "
        "with, and every seed in the matrix would share its starting point."
    )

    # And within one call the genomes must differ from each other, or the population is p_pop copies of one
    # network and generation zero carries no diversity for selection to act on.
    assert not np.array_equal(a[0], a[1]), (
        "two genomes in the initial population are identical; the per-genome keys are not being split"
    )


def test_the_generation_step_does_not_grow_host_memory():
    """**The gate for DEV-029, and it asserts the mechanism because every outcome-level check passed.**

    The leaking version completed generations, produced correct fitness, reproduced from its config and seed,
    and passed every assertion in this file and in S7. It grew host memory by 31.5 MB per generation and was
    OOM-killed at about generation 475 of 900, having reached 11.5 GB. Nothing observable at the level of "did
    the run work" could distinguish it, which is the many-sufficient-causes class again.

    **The metric has to be current RSS, not `ru_maxrss`.** The first attempt at this measurement used
    `resource.getrusage`, which reports a HIGH-WATER MARK: monotonic by construction, so it cannot tell a leak
    from a peak that happened once, and it made a correct implementation look identical to a leaking one.
    Verify the metric can move in both directions before asserting on it.

    Threshold reasoning: the leak was 31.5 MB per generation and the fix measures 0.00. Anything above 1 MB
    per generation over 25 generations is far outside measurement noise and far below the defect, so the
    bound does not need to be tight to be decisive.
    """

    def rss_mb():
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
        pytest.skip("VmRSS unavailable on this platform")

    from configs.default import DEFAULT
    from src.agent import network as net
    from src.env import env as E
    from src.env import grid, resources
    from src.env.perception import build_vis_table
    from src.evo import genome

    q = DEFAULT.__class__(t_horizon=30, p_pop=8, n_episodes=1, n_elite=2, k=4, h=20, w=20)
    ob = grid.make_obstacle_grid(jax.random.key(0), q.h, q.w, q.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(0), ob, q)
    static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, q.r), food_src=fs, target_src=ts)
    template = net.init_params(jax.random.key(0), q)

    step = loop.make_generation_step(genome.unflattener(template), static, q)
    pop = loop.initial_population(jax.random.key(1), template, q)
    key = jax.random.key(2)

    # Warm up first: the first calls compile, and compilation legitimately allocates.
    for _ in range(5):
        key, k = jax.random.split(key)
        pop, _ = step(k, pop)
    jax.block_until_ready(pop)
    before = rss_mb()

    n = 25
    for _ in range(n):
        key, k = jax.random.split(key)
        pop, _ = step(k, pop)
    jax.block_until_ready(pop)
    growth = rss_mb() - before

    print(f"\n  host memory growth over {n} generations: {growth:.1f} MB ({growth / n:.2f} MB/gen)")
    assert growth / n < 1.0, (
        f"the generation step grew host memory by {growth / n:.2f} MB per generation. At 300 generations "
        f"that is {growth / n * 300:.0f} MB and at 900 it is {growth / n * 900:.0f} MB. The known cause is a "
        "fresh callable handed to vmap each generation, which misses JAX's identity-keyed trace cache and "
        "retains an executable every time. Build the step once with make_generation_step."
    )


def test_the_generation_step_is_compiled_once_and_not_per_generation():
    """**The reliable detector for DEV-029, and the memory assertion above is not one.**

    Falsification exposed this: reintroducing the defect left the memory assertion green. Measured on the
    same small configuration on the day, the defect grew host memory by 0.03 MB per generation, not the 31.5
    that was measured when DEV-029 was first found. That earlier figure was real, and the 11.5 GB OOM kill is
    in `dmesg`, but it does not reproduce now and the reason is not established. **What does reproduce, every
    time and by a factor of sixty, is the TIME signature.**

        built once                 first 6.646 s, steady 0.022 s, ratio 0.0034
        rebuilt per generation     first 6.282 s, steady 1.302 s, ratio 0.2073

    **A ratio rather than an absolute time, so it is machine-independent.** If the step is compiled once, the
    first generation pays for compilation and the rest are nearly free. If it is rebuilt, every generation
    pays and steady state approaches the first. The threshold of 0.05 sits about fifteen times above the
    correct value and four times below the defect, so it is not resolving a close call.

    This is why the memory gate is kept as well rather than replaced: it caught the real production failure,
    and this catches the cause. Neither alone is sufficient, which is the point.
    """
    from configs.default import DEFAULT
    from src.agent import network as net
    from src.env import env as E
    from src.env import grid, resources
    from src.env.perception import build_vis_table
    from src.evo import genome

    q = DEFAULT.__class__(t_horizon=30, p_pop=8, n_episodes=1, n_elite=2, k=4, h=20, w=20)
    ob = grid.make_obstacle_grid(jax.random.key(0), q.h, q.w, q.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(0), ob, q)
    static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, q.r), food_src=fs, target_src=ts)
    template = net.init_params(jax.random.key(0), q)

    step = loop.make_generation_step(genome.unflattener(template), static, q)
    pop = loop.initial_population(jax.random.key(1), template, q)
    key = jax.random.key(2)

    key, k = jax.random.split(key)
    t = time.time()
    pop, _ = step(k, pop)
    jax.block_until_ready(pop)
    first = time.time() - t

    times = []
    for _ in range(9):
        key, k = jax.random.split(key)
        t = time.time()
        pop, _ = step(k, pop)
        jax.block_until_ready(pop)
        times.append(time.time() - t)
    steady = sorted(times)[len(times) // 2]

    ratio = steady / first
    print(f"\n  first generation {first:.3f}s, steady median {steady:.3f}s, ratio {ratio:.4f}")
    assert ratio < 0.05, (
        f"steady-state generations cost {ratio:.1%} of the first generation. The first pays for compilation "
        "and the rest should be nearly free, so this means the step is being rebuilt and recompiled every "
        "generation. That is DEV-029: build the step once with make_generation_step and do not construct "
        "the vmapped or jitted callable inside the per-generation call."
    )
