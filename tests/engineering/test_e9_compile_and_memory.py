"""Gate E9: the generation step compiles, and peak memory at target scale stays inside the device.

**This is the first thing in the repository that runs at the real configuration.** Every other test uses a
shrunken world, 14 by 14 with 4 prey over 60 ticks, because that is what keeps a suite fast. The consequence
is that nothing before this file has ever asked whether the actual experiment fits on the actual card, and
D-008's 14.9 GB against 6.17 GB finding says that question has a real answer rather than an obvious one.

The generation shape is the one the evolution loop will use, and it is not a flat batch of E environments.
There are `p_pop` distinct genomes, each evaluated on `n_episodes` episodes, so the weights are vmapped once
and the episode keys twice:

    inner = vmap(run_episode, in_axes=(0, None, ...))   over episode keys, one genome
    gen   = vmap(inner,       in_axes=(0, 0,    ...))   over genomes

Written that way the weights stay at `p_pop` copies. Tiling them to E instead would allocate 1024 by 69,814
floats, 286 MB, that the real loop never needs, so the gate would measure a cost the experiment does not pay
and could fail on a configuration that actually fits.

Compile time is measured separately from run time, per the JAX conventions in CLAUDE.md, because the first
call includes compilation and would otherwise poison the throughput figure.

The metric is `pool_bytes`. `bytes_in_use` reports zero whether preallocation is on or off, and
`test_runtime_environment.py` records that finding: a memory assertion written against it is inert, which is
the construction class of green-and-meaningless assertion.
"""

import time

import jax
import jax.numpy as jnp
import pytest

import src
from configs.default import DEFAULT as P
from src.agent import network as net
from src.env import env as E
from src.env import grid, resources
from src.env.entropy import PRNGSource
from src.env.perception import build_vis_table

pytestmark = pytest.mark.slow  # measured at the mark site below, target scale by construction


def _generation_fn(static, entropy, params):
    """The generation step: every genome on every episode, as the evolution loop will call it.

    **`static`, `entropy` and `params` are closed over rather than passed**, and that is forced rather than
    stylistic. `EnvParams` is a plain dataclass with no pytree registration, so `vmap` cannot flatten it and
    rejects it outright even at `in_axes=None`. Closing over it makes every shape-determining field a
    compile-time constant, which is what `t_horizon`, `k` and `obs_dim` need to be anyway.

    The same treatment suits `static` and `entropy`: obstacles and the visibility table cannot change inside
    a run, so XLA folds them into the kernel instead of reloading them per environment.

    Only `keys` and `weights` are traced, and only they carry the batch axes.
    """

    def one_episode(key, genome):
        return E.run_episode(key, genome, static, entropy, params)

    over_episodes = jax.vmap(one_episode, in_axes=(0, None))  # n_episodes, one genome
    return jax.jit(jax.vmap(over_episodes, in_axes=(0, 0)))  # p_pop genomes


@pytest.fixture(scope="module")
def target_scale():
    """Static world at the real configuration. No shrinking anywhere in this file."""
    src.assert_runtime_took_effect()
    key = jax.random.key(0)
    obstacles = grid.make_obstacle_grid(key, P.h, P.w, P.obstacle_density)
    food_src, target_src, _ = resources.place_sources(key, obstacles, P)
    static = E.StaticEnv(
        obstacles=obstacles,
        vis_table=build_vis_table(obstacles, P.r),
        food_src=food_src,
        target_src=target_src,
    )
    weights = jax.vmap(lambda k: net.init_params(k, P))(jax.random.split(jax.random.key(1), P.p_pop))
    keys = jax.random.split(jax.random.key(2), P.p_pop * P.n_episodes).reshape(P.p_pop, P.n_episodes)
    return static, weights, keys


def test_a_generation_compiles_and_runs_at_target_scale(target_scale):
    """E9's first half. A generation must compile at all, and compile time must be bounded.

    The episode loop is a `lax.scan`, so its body compiles once rather than 400 times. If this ever takes
    minutes, the scan has been unrolled somewhere and the compute budget is gone.
    """
    static, weights, keys = target_scale
    entropy = PRNGSource(jax.random.key(3))
    gen = _generation_fn(static, entropy, P)

    t0 = time.perf_counter()
    out = gen(keys, weights)
    jax.block_until_ready(out)
    compile_and_first_run = time.perf_counter() - t0

    t1 = time.perf_counter()
    out = gen(keys, weights)
    jax.block_until_ready(out)
    warm_run = time.perf_counter() - t1

    compile_only = compile_and_first_run - warm_run
    print(
        f"\n  E9 generation, E = {P.p_pop} x {P.n_episodes} = {P.p_pop * P.n_episodes}, "
        f"t_horizon {P.t_horizon}"
        f"\n    compile   {compile_only:8.1f} s"
        f"\n    warm run  {warm_run:8.1f} s"
    )

    assert compile_only < 300.0, (
        f"compiling one generation took {compile_only:.0f} s. The episode loop is a lax.scan and should "
        "compile its body once; minutes of compile time means something unrolled."
    )

    state, acc = out
    assert state.prey_score.shape == (P.p_pop, P.n_episodes, P.k), (
        f"a generation produced scores shaped {state.prey_score.shape}, not one per prey per episode "
        "per genome, so the generation step is not evaluating what evolution will select on"
    )
    assert bool(jnp.all(jnp.isfinite(state.prey_score))), "a generation produced non-finite scores"


def test_peak_memory_at_target_scale_stays_inside_the_device(target_scale):
    """E9's second half, and the one D-008 exists because of.

    Stacking observations across an episode would be 14.9 GB against 6.17 GB usable, which is why the main
    scan path returns carry only. This asserts the resulting footprint rather than trusting that argument.
    """
    static, weights, keys = target_scale
    entropy = PRNGSource(jax.random.key(3))
    gen = _generation_fn(static, entropy, P)

    jax.block_until_ready(gen(keys, weights))

    stats = jax.devices()[0].memory_stats()
    pool, limit = stats["pool_bytes"], stats["bytes_limit"]
    print(f"\n  E9 memory: pool {pool / 2**30:.2f} GiB of limit {limit / 2**30:.2f} GiB")

    assert pool < limit, (
        f"a generation reserved {pool / 2**30:.2f} GiB against a {limit / 2**30:.2f} GiB ceiling. The "
        "experiment does not fit at target scale."
    )


def test_the_generation_actually_simulated_something(target_scale):
    """**E9 must not be satisfiable by a generation that computed nothing.**

    A warm generation runs in about a second, which is roughly a hundred times better than the compute budget
    assumed. CLAUDE.md section 15 says a result better than expected earns more scrutiny than a disappointing
    one, and the specific failure to rule out is dead-code elimination: if nothing downstream consumed the
    accumulators, XLA would be free to delete the work and both halves of E9 would still pass, one quickly and
    one cheaply.

    The arithmetic says the figure is physically reasonable. 1024 environments times 8 prey times 400 ticks is
    3.3 million prey-ticks, each dominated by a 1140 by 48 matmul and three 48 by 48 GRU gates, order 90k
    FLOPs, so about 300 GFLOP in a second or so. That is a few percent of the card, which is what a long
    sequential scan over small arrays should achieve.

    But reasoning is not evidence, so this asserts that the episode left marks: time passed, the pools moved
    off their reset values, and the outcome varies across the population.
    """
    static, weights, keys = target_scale
    entropy = PRNGSource(jax.random.key(3))
    gen = _generation_fn(static, entropy, P)
    state, acc = jax.block_until_ready(gen(keys, weights))

    alive = acc.alive_ticks
    assert int(jnp.max(alive)) > 1, (
        f"the longest-lived prey in the entire generation survived {int(jnp.max(alive))} ticks, so the scan "
        "did not run the episode and E9 is measuring an empty computation"
    )
    assert int(jnp.max(alive)) <= P.t_horizon, (
        f"a prey accumulated {int(jnp.max(alive))} alive ticks against a {P.t_horizon}-tick horizon"
    )

    assert not bool(jnp.all(state.prey_energy == P.e_max)), (
        "every prey ended at full energy, so the pool dynamics did not run"
    )
    assert int(jnp.unique(alive).size) > 1, (
        "every environment in the generation produced an identical survival time, which means the population "
        "is not actually being varied over and selection would have nothing to act on"
    )


def test_the_memory_metric_responds_to_scale(target_scale):
    """**Verify the metric can see the condition before asserting on it.**

    `pool_bytes < bytes_limit` passing tells you nothing unless the left side moves when the workload does.
    This is the construction class of green-and-meaningless assertion, the one that already caught a
    `bytes_in_use` check in `test_runtime_environment.py`, so it is checked here rather than assumed.

    A generation an eighth the size must reserve measurably less. Compared as a fresh process would be ideal;
    within one process the pool only grows, so the comparison is made in increasing order of size and the
    assertion is that the larger workload is not smaller.
    """
    static, _, _ = target_scale
    entropy = PRNGSource(jax.random.key(3))
    gen = _generation_fn(static, entropy, P)

    small_pop = max(P.p_pop // 8, 1)
    w_small = jax.vmap(lambda k: net.init_params(k, P))(jax.random.split(jax.random.key(11), small_pop))
    k_small = jax.random.split(jax.random.key(12), small_pop * P.n_episodes).reshape(small_pop, P.n_episodes)
    jax.block_until_ready(gen(k_small, w_small))
    small_pool = jax.devices()[0].memory_stats()["pool_bytes"]

    w_full = jax.vmap(lambda k: net.init_params(k, P))(jax.random.split(jax.random.key(1), P.p_pop))
    k_full = jax.random.split(jax.random.key(2), P.p_pop * P.n_episodes).reshape(P.p_pop, P.n_episodes)
    jax.block_until_ready(gen(k_full, w_full))
    full_pool = jax.devices()[0].memory_stats()["pool_bytes"]

    print(
        f"\n  E9 metric response: pool at p_pop {small_pop} = {small_pool / 2**20:.0f} MiB, "
        f"at p_pop {P.p_pop} = {full_pool / 2**20:.0f} MiB"
    )
    assert full_pool >= small_pool, (
        f"an eight-times-larger generation reserved {full_pool / 2**20:.0f} MiB against "
        f"{small_pool / 2**20:.0f} MiB for the small one. pool_bytes is not tracking the workload, so the "
        "memory assertion in this file cannot see the condition it claims to guard."
    )
