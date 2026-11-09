"""The long run's world: the parameters the S4 calibration chose (D-083, Q-057), applied to every arm.

**Chosen 2026-11-09, by run 58 (D-092).** D-083's rule applied to stage 3 (D-091) re-measured at 64 episodes
(Q-061) chose `cue_dependence` 0.8 at `p_wander` 0.5, `h_regen` and `damage` at their defaults. Before that `WORLD`
was None, and `scripts/measure/sweep.py` refuses to start a long run (anything but `--smoke` or `--plan-only`)
while it is, because a long run in an unchosen world is the Phase A world again, the one S4 rejected (run 52,
Q-056). The choice is re-checked on fresh seeds 103 to 105 by `calibrate_s4.py --frozen` (A6000 session 2).

**How it is applied, and why not by editing `configs/default.py`.** Every gate was written and verified against
`DEFAULT`, so moving `DEFAULT` would silently change what every gate tests. The frozen world is applied on top of
each arm's run configuration instead, by `apply`, which replaces a field only where the arm agrees with the
scripted control. So each arm keeps the one field that defines it: the null's `n_pred = 0`, the third arm's
`damage = 0`. `tests/engineering/test_longrun_runner.py` asserts that every arm differs from the frozen scripted
world in exactly its own field.

`WORLD` is the chosen setting exactly as the summary names it, and `SOURCE` the run id and the summary's path.
`tests/engineering/test_calibrate_s4.py` asserts that SOURCE's run is registered in runs.md, that WORLD is a stage 3
grid setting, and that the long run's treatment world equals the calibration's world for that setting.
"""

from __future__ import annotations

import dataclasses

#: The chosen setting. None until D-083's rule chose, on 2026-11-09.
WORLD: dict | None = {"h_regen": 0.25, "p_wander": 0.5, "damage": 34.0, "cue_dependence": 0.8}
#: Where the choice came from: the registered run and the summary file.
SOURCE: str | None = "run 58, runs/calibration-s4-episodes/summary_episodes.json (Q-061, D-092)"

#: The only fields the calibration may move: Q-057's three, and `cue_dependence` since stage 3 (Q-060). Anything
#: else in WORLD is refused.
MOVABLE = frozenset({"h_regen", "p_wander", "damage", "cue_dependence"})


def check(world: dict | None = None) -> None:
    world = WORLD if world is None else world
    if world is None:
        return
    extra = set(world) - MOVABLE
    if extra:
        raise ValueError(f"the frozen world moves {sorted(extra)}, which neither Q-057 nor Q-060 allows")


def apply(base, scripted, world: dict | None = None):
    """`base` with the frozen world applied wherever `base` agrees with the scripted control `scripted`."""
    world = WORLD if world is None else world
    if world is None:
        return base
    check(world)
    changes = {k: v for k, v in world.items() if getattr(base, k) == getattr(scripted, k)}
    return dataclasses.replace(base, **changes) if changes else base
