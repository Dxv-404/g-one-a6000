"""One fixed episode, dumped to an npz. Run as a subprocess by gate E1. Build step 10.

**A module rather than a string, deliberately.** `tests/scientific/test_s10_clean_room.py` holds an entire
episode script as a Python string, and when `resolve_deaths` changed signature no source search could find
that caller: the subprocess raised a TypeError which surfaced two steps away as "a short episode did not
complete". Code held as data is invisible to every tool that looks at code. See DEV-023.

Not named `test_*`, so pytest does not collect it.

Usage: python -m tests.engineering._e1_replay <seed> <out.npz>
"""

from __future__ import annotations

import sys

import jax
import numpy as np

from configs.default import DEFAULT
from src.agent import network as net
from src.env import env as E
from src.env import grid, resources
from src.env.entropy import PRNGSource
from src.env.perception import build_vis_table

#: Long enough that deaths, regrowth and predator engagement all occur, short enough to run twice per gate.
PARAMS = DEFAULT.__class__(t_horizon=60)

#: Exactly the quantities E1's definition names as having to reproduce EXACTLY. Integers and booleans, so
#: "exactly" is meaningful rather than a tolerance in disguise.
DISCRETE = ("prey_pos", "prey_alive", "prey_score", "prey_death_cause", "prey_death_step", "pred_pos")

#: The float pools, which reproduce within a stated tolerance rather than exactly. Cross-device bitwise
#: equality is not achievable for float GPU work and E1 does not ask for it.
FLOAT = ("prey_energy", "prey_stamina", "prey_health")


def episode(seed: int) -> dict[str, np.ndarray]:
    """The network policy, not a hand-coded one, because deterministic argmax over float logits is exactly
    where a nondeterministic reduction would show up as a changed ACTION rather than a changed float."""
    p = PARAMS
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
    static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, p.r), food_src=fs, target_src=ts)
    weights = net.init_params(jax.random.key(seed), p)

    state, acc, traj = E.run_episode(
        jax.random.key(1000 + seed), weights, static, PRNGSource(jax.random.key(seed)), p, record=True
    )

    out: dict[str, np.ndarray] = {}
    for name in DISCRETE:
        out["state." + name] = np.asarray(getattr(state, name))
    for name in FLOAT:
        out["float." + name] = np.asarray(getattr(state, name))
    for name in ("prey_pos", "prey_alive", "prey_moved", "action_emitted", "action_executed", "pred_pos"):
        out["traj." + name] = np.asarray(getattr(traj, name))
    for name in acc._fields:
        out["acc." + name] = np.asarray(getattr(acc, name))
    return out


def main() -> int:
    seed, path = int(sys.argv[1]), sys.argv[2]
    np.savez(path, **episode(seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
