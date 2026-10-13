"""A short evolutionary run, reconstructed from a config name and a seed alone. Run as a subprocess by S7.

**A module, not a string.** Code held as data is invisible to every tool that looks at code; see DEV-023,
where a signature change missed a caller living inside a Python string in gate S10.

**The point of the interface is that it takes a config NAME and a seed and nothing else.** If the run needed
anything else to reproduce, that thing would have to be passed here, and its absence from this signature is
what makes S7's claim checkable rather than asserted.

Usage: python -m tests.scientific._s7_run <config-name> <seed> <out.npz>
"""

from __future__ import annotations

import sys

import jax
import numpy as np

from configs.default import DEFAULT
from src.agent import network as net
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table
from src.evo import genome, loop

#: Named configurations, so the subprocess is given a NAME rather than a pile of keyword arguments. A run
#: that could only be reconstructed by re-typing its parameters would not satisfy S7's premise.
CONFIGS = {
    "s7-small": dict(t_horizon=60, p_pop=16, n_episodes=2, n_elite=2, t_sel=3),
    "s7-alt": dict(t_horizon=60, p_pop=16, n_episodes=2, n_elite=4, t_sel=3),
}

N_GENERATIONS = 3


def build(name: str):
    return DEFAULT.__class__(**CONFIGS[name])


def run(name: str, seed: int) -> dict[str, np.ndarray]:
    """Everything below derives from `name` and `seed`. Nothing is read from the environment or the clock."""
    params = build(name)

    # The world is part of the configuration, not of the episode, so it is keyed on the run seed.
    ob = grid.make_obstacle_grid(jax.random.key(seed), params.h, params.w, params.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, params)
    static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, params.r), food_src=fs, target_src=ts)

    template = net.init_params(jax.random.key(seed), params)
    unflatten = genome.unflattener(template)

    pop = loop.initial_population(jax.random.key(seed + 1), template, params)
    key = jax.random.key(seed + 2)

    # Built once, outside the loop. See DEV-029: building it inside leaks an executable per generation.
    step = loop.make_generation_step(unflatten, static, params)

    history = []
    for _ in range(N_GENERATIONS):
        key, k = jax.random.split(key)
        pop, fit = step(k, pop)
        history.append(np.asarray(fit))

    return {
        "config_hash": np.array(loop.config_hash(params)),
        "final_population": np.asarray(pop),
        "fitness_history": np.stack(history),
    }


def main() -> int:
    name, seed, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    np.savez(path, **run(name, seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
