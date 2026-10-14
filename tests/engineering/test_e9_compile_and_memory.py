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

The metric is `peak_bytes_in_use`, since D-067 moved the allocator to cuda_async. Until then it was BFC's
`pool_bytes`, which cuda_async does not report. Measured on the A6000, `peak_bytes_in_use` went from 116 MiB
to 5172 MiB between a 16 MB and a 3 GB workload, so it tracks the workload, and
`test_the_memory_metric_responds_to_scale` re-checks that here rather than assuming it.

The ceiling is 75 percent of the card, which is what BFC's `bytes_limit` reported whether preallocation was on
or off (D-008), so the gate asks the same question it always asked. Under cuda_async with preallocation off,
`bytes_limit` is 0 and is no ceiling at all, so the card's total is read from the driver instead.
"""

import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

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

# XLA's default memory fraction, the ceiling BFC's bytes_limit reported (D-008) and E9 always asserted against.
CEILING_FRACTION = 0.75


def _device_ceiling_bytes():
    """75 percent of device 0's physical memory, from the CUDA driver, which needs no context."""
    cu = ctypes.CDLL("libcuda.so.1")
    assert cu.cuInit(0) == 0, "cuInit failed, so the card's memory cannot be read"
    dev, total = ctypes.c_int(), ctypes.c_size_t()
    assert cu.cuDeviceGet(ctypes.byref(dev), 0) == 0
    assert cu.cuDeviceTotalMem_v2(ctypes.byref(total), dev) == 0
    return int(CEILING_FRACTION * total.value)


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
    return _target_world()


def _target_world():
    """The fixture's body, callable from the fresh process the metric-response test needs."""
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

    peak, limit = jax.devices()[0].memory_stats()["peak_bytes_in_use"], _device_ceiling_bytes()
    print(f"\n  E9 memory: peak {peak / 2**30:.2f} GiB of ceiling {limit / 2**30:.2f} GiB")

    assert peak < limit, (
        f"a generation peaked at {peak / 2**30:.2f} GiB against a {limit / 2**30:.2f} GiB ceiling. The "
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


# Run in a fresh process by the test below. Small generation first, then full, so the high-water mark can
# only show a response if the metric really tracks the workload.
_RESPONSE_SCRIPT = """
import json, jax
from configs.default import DEFAULT as P
from src.agent import network as net
from src.env.entropy import PRNGSource
from tests.engineering.test_e9_compile_and_memory import _generation_fn, _target_world

static, _, _ = _target_world()
gen = _generation_fn(static, PRNGSource(jax.random.key(3)), P)
peaks = {}
for pop in (max(P.p_pop // 8, 1), P.p_pop):
    w = jax.vmap(lambda k: net.init_params(k, P))(jax.random.split(jax.random.key(11), pop))
    k = jax.random.split(jax.random.key(12), pop * P.n_episodes).reshape(pop, P.n_episodes)
    jax.block_until_ready(gen(k, w))
    peaks[pop] = jax.devices()[0].memory_stats()["peak_bytes_in_use"]
print("PEAKS " + json.dumps(peaks))
"""


def test_the_memory_metric_responds_to_scale():
    """**Verify the metric can see the condition before asserting on it.**

    `peak_bytes_in_use < ceiling` passing tells you nothing unless the left side moves when the workload does.
    This is the construction class of green-and-meaningless assertion, the one that already caught a
    `bytes_in_use` check in `test_runtime_environment.py`, so it is checked here rather than assumed.

    **It runs in a fresh process, and that is the fix for this test having been inert.** The metric is a
    high-water mark, and the tests above run the full generation first, so inside this process an
    eight-times-smaller generation reads the same peak as the full one and `full >= small` passes whatever
    the metric does. Found at D-067 when the response printed 1316 MiB for both. The same was true of
    `pool_bytes` before it, which also only grows. In a fresh process the small generation runs first, and the
    assertion is strict.
    """
    out = subprocess.run(
        [sys.executable, "-c", _RESPONSE_SCRIPT],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [line for line in out.stdout.splitlines() if line.startswith("PEAKS ")]
    assert out.returncode == 0 and lines, f"the fresh-process measurement failed:\n{out.stderr[-2000:]}"
    peaks = {int(k): v for k, v in json.loads(lines[-1][len("PEAKS ") :]).items()}
    (small_pop, small_peak), (full_pop, full_peak) = sorted(peaks.items())

    print(
        f"\n  E9 metric response, fresh process: peak at p_pop {small_pop} = {small_peak / 2**20:.0f} MiB, "
        f"at p_pop {full_pop} = {full_peak / 2**20:.0f} MiB"
    )
    assert full_peak > small_peak, (
        f"an eight-times-larger generation peaked at {full_peak / 2**20:.0f} MiB against "
        f"{small_peak / 2**20:.0f} MiB for the small one, in a process where it ran second. "
        "peak_bytes_in_use is not tracking the workload, so the memory assertion in this file cannot see the "
        "condition it claims to guard."
    )
