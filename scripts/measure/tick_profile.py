#!/usr/bin/env python3
"""Where does a tick's time go? The answer decides whether a 2 to 3x speed-up is available before the sweep.

**The hypothesis, stated before the numbers so the numbers can refute it.** D-057 records that the sweep is
compute-bound at 1.5 s per generation, and that the network arithmetic is roughly 3 percent of the GPU's
FP32 peak. So the time is not in the matrix multiplies. The leading candidate is the observation builder:
`build_observation` scatters each entity into the egocentric patch through Python loops that unroll at
trace time, nine `.at[].max()` writes per entity, nine entities, built once per prey, which is on the order
of 650 single-element scatters per environment-tick. If that is where the time goes, replacing the unrolled
writes with one vectorised scatter is a multi-x win that changes the sweep's cost. If it is not, the
hypothesis is dropped and the sweep is planned at D-057's figure.

    observation share >= 50 percent    CONFIRMED. Vectorising the scatter is worth doing before the sweep
    25 to 50 percent                   PARTIAL. Worth doing, but it will not halve the run on its own
    under 25 percent                   REFUTED. The time is elsewhere and the hypothesis is withdrawn

**Method: time the components in isolation at the sweep's exact batch shape, inside the same `lax.scan`.**
Three jitted programs, each vmapped over 1,024 environments and scanned over 400 ticks, so dispatch and
fusion behave as they do in the real loop rather than as they would in a Python loop:

    full episode        `run_episode` unchanged: the real cost, and the denominator
    observation only    `build_observation` for all k prey per tick, positions advanced each tick so the
                        gathers cannot be hoisted, the result summed into the carry so it cannot be eliminated
    network only        `network_policy` per tick with the hidden state carried, which is the real recurrence

**Two caveats that keep the fractions honest.** XLA fuses across components in the full program and cannot
in the isolated ones, so an isolated component is an UPPER bound on its share, and the shares need not sum
to one. And the isolated observation program still carries a state and a scan, so a small fixed cost is
attributed to it that the full program also pays. Both bias the observation share UP, which means a REFUTED
verdict is trustworthy and a CONFIRMED one is a reason to optimise and re-measure rather than a promise.

**A structural check that does not depend on timing.** The compiled full episode's HLO is scanned for
scatter operations. The count in the loop body is what the unrolled loops actually produced, and it is
reported beside the estimate above so the estimate can be corrected rather than trusted.

Usage: uv run python scripts/measure/tick_profile.py [--reps N]
"""

from __future__ import annotations

import re
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import DEFAULT
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.observation import build_observation
from src.env.perception import build_vis_table, required_radius

N_ENVS = DEFAULT.p_pop * DEFAULT.n_episodes  # 1,024 at the standard configuration, the sweep's batch
REPS = 3


def setup(p):
    seed = 0
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
    static = E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )
    w = net.init_params(jax.random.key(seed), p)
    keys = jax.random.split(jax.random.key(1), N_ENVS)
    return static, w, keys


def program_full(static, w, p):
    def one(key):
        st, _ = E.run_episode(key, w, static, ent.PRNGSource(key), p)
        return jnp.sum(st.prey_score)

    return jax.jit(jax.vmap(one))


def program_obs(static, w, p):
    k = p.k

    def one(key):
        st = E.reset(ent.PRNGSource(key), static, p)
        pools = jnp.stack([st.prey_energy, st.prey_stamina, st.prey_health], axis=-1)

        def obs_for(pos_all, i):
            return build_observation(
                obstacles=static.obstacles,
                food_grid=st.food_grid,
                target_grid=st.target_grid,
                vis_table=static.vis_table,
                self_pos=pos_all[i],
                self_posture=st.prey_posture[i],
                self_pools=pools[i],
                prey_pos=pos_all,
                prey_alive=st.prey_alive,
                prey_moved=st.prey_moved,
                prey_posture=st.prey_posture,
                pred_pos=st.pred_pos,
                pred_moved=jnp.zeros((p.n_pred,), bool),
                pred_appearance=st.pred_appearance,
                self_index=i,
                params=p,
            )

        def body(carry, _):
            total, pos = carry
            obs = jax.vmap(obs_for, in_axes=(None, 0))(pos, jnp.arange(k))
            # advance every position by one cell, wrapping, so each tick gathers from new addresses
            pos = (pos + 1) % jnp.array([p.h, p.w], pos.dtype)
            return (total + jnp.sum(obs), pos), None

        (total, _), _ = jax.lax.scan(body, (jnp.float32(0.0), st.prey_pos), None, length=p.t_horizon)
        return total

    return jax.jit(jax.vmap(one))


def program_net(static, w, p):
    def one(key):
        st = E.reset(ent.PRNGSource(key), static, p)
        obs = jax.random.normal(key, (p.k, p.obs_dim))

        def body(carry, t):
            total, h = carry
            action, h = E.network_policy(w, obs, h, t)
            return (total + jnp.sum(action), h), None

        (total, _), _ = jax.lax.scan(
            body, (jnp.int32(0), st.prey_hidden), jnp.arange(p.t_horizon, dtype=jnp.int32)
        )
        return total

    return jax.jit(jax.vmap(one))


def timed(fn, keys, reps):
    t0 = time.time()
    out = fn(keys)
    jax.block_until_ready(out)
    compile_s = time.time() - t0
    ts = []
    for _ in range(reps):
        t1 = time.time()
        out = fn(keys)
        jax.block_until_ready(out)
        ts.append(time.time() - t1)
    return compile_s, float(np.median(ts)), float(max(ts) - min(ts))


def scatter_count(fn, keys) -> tuple[int, int]:
    """(scatter ops, gather ops) in the compiled HLO of the full program, counted by instruction name."""
    txt = fn.lower(keys).compile().as_text()
    return len(re.findall(r"\bscatter\b", txt)), len(re.findall(r"\bgather\b", txt))


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    print(src.memory_headroom(), flush=True)
    global REPS
    if "--reps" in sys.argv:
        REPS = int(sys.argv[sys.argv.index("--reps") + 1])

    p = DEFAULT
    static, w, keys = setup(p)
    print(
        f"batch {N_ENVS} envs x {p.k} prey x {p.t_horizon} ticks, {REPS} reps, median reported\n", flush=True
    )

    rows = {}
    for name, builder in (
        ("full episode", program_full),
        ("observation only", program_obs),
        ("network only", program_net),
    ):
        fn = builder(static, w, p)
        c, med, spread = timed(fn, keys, REPS)
        rows[name] = med
        print(
            f"  {name:<18s} compile {c:6.1f}s   per generation {med:7.3f}s   spread {spread:.3f}", flush=True
        )
        if name == "full episode":
            n_sc, n_ga = scatter_count(fn, keys)
            print(f"  {'':<18s} compiled HLO: {n_sc} scatter, {n_ga} gather instructions", flush=True)
        jax.clear_caches()

    full, obs, netw = rows["full episode"], rows["observation only"], rows["network only"]
    rest = full - obs - netw
    print("\n" + "=" * 90, flush=True)
    print("SHARES OF THE FULL EPISODE. Isolated components are UPPER bounds; see the docstring.", flush=True)
    print("=" * 90, flush=True)
    print(f"  observation build   {obs:7.3f}s   {100 * obs / full:5.1f} percent", flush=True)
    print(f"  network forward     {netw:7.3f}s   {100 * netw / full:5.1f} percent", flush=True)
    print(
        f"  everything else     {rest:7.3f}s   {100 * rest / full:5.1f} percent   (predator, movement, pools, resources, accumulators; by subtraction, can be negative if fusion is large)",
        flush=True,
    )

    share = obs / full
    print(flush=True)
    if share >= 0.5:
        print(
            f"  CONFIRMED. The observation build is {100 * share:.0f} percent of the tick. Vectorising its scatter is",
            flush=True,
        )
        print(
            "  the one software change that can shorten the sweep, and it is worth doing before the sweep runs.",
            flush=True,
        )
    elif share >= 0.25:
        print(
            f"  PARTIAL. The observation build is {100 * share:.0f} percent. Worth optimising, but it cannot halve the run alone.",
            flush=True,
        )
    else:
        print(
            f"  REFUTED. The observation build is {100 * share:.0f} percent of the tick. The hypothesis in D-057 is withdrawn",
            flush=True,
        )
        print("  and the sweep is planned at D-057's figure unless something else is found.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
