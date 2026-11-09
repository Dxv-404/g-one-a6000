"""Engagements per genome: the per-tick pass, the counting, and the S4 measurement. The single implementation.

Moved here unchanged on 2026-11-06 from `scripts/measure/replay_final.py` (run 48's driver) so that the replay,
the S4 calibration (`scripts/measure/calibrate_s4.py`, D-083) and the long run's in-run S4 measurement
(`scripts/measure/sweep.py`) all count engagements with ONE piece of code. Two copies of a counting rule drift
apart, and the record of this repository says they drift toward the sloppier one.

**It imports nothing from the behaviour side of this package** (descriptors, classifier, discovery), because the
calibration that uses it must stay blind to behaviour; `tests/engineering/test_calibrate_s4.py` asserts what the
calibration actually loads.

### Definitions, fixed at D-076 before any of it ran

An engagement is a maximal run of ticks on which the prey is engaged, read from the environment's OWN
`engaged_ticks` counter tick by tick, so the definition cannot drift from the one every descriptor uses. It ends
in exactly one of four ways:

    captured     the prey died with the predation flag set on a tick inside the engagement
    died_other   the prey died inside the engagement without the predation flag (starvation)
    censored     the engagement was still running on the last tick and the prey was alive
    escaped      the engagement ended and the prey was alive on the following tick

### Measurement keys

`measure_keys` derives every key from the seed and the generation count only, through a fixed tag, never from an
evolution's key chain (which starts at `key(seed + 2)` and splits from there). So measuring a population cannot
perturb the run it was taken from, and two measurements of one population agree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.env import entropy as ent
from src.env import env as E
from src.measure.learnability import ENG_FIELDS

#: Column order of an engagement record. Defined once, in `learnability`, which must stay import-light.
ENGAGEMENT_FIELDS = ENG_FIELDS
#: Per prey per episode, in the order `make_engagement_pass` stacks them.
PREY_FIELDS = ("alive_ticks", "death_step", "died_predated", "died_starving", "alive_at_end", "score")
#: Folded into the seed's key for measurement, so measurement keys never meet an evolution's chain.
MEASURE_TAG = 0x53344341

_ACC_FIELDS = E.Accumulators._fields


def make_engagement_pass(unflatten, params):
    """One episode per genome through the SAME `reset` and `tick` evolution uses, emitting per tick the running
    `engaged_ticks` and the alive mask, so engagement runs are read from the environment's own definition.

    Returns, per genome: the accumulator row plus the three tier-one outcomes (what run 48 cross-checks against
    the sweep), the per-tick running `engaged_ticks` (T, k), the prey facts (6, k) and the spawn relaxation level.

    **The world is an ARGUMENT, not a closed-over constant**, so one compiled program serves every seed of a job."""

    def one(vec, k, static):
        k_reset, k_draws = jax.random.split(k)
        weights = unflatten(vec)
        entropy = ent.PRNGSource(k_draws)
        state = E.reset(ent.PRNGSource(k_reset), static, params)
        acc = E.zero_accumulators(params.k)

        def body(carry, _):
            st, ac, t = carry
            st, ac, _rec = E.tick(st, ac, weights, static, entropy, t, params)
            return (st, ac, t + 1), (ac.engaged_ticks, st.prey_alive)

        (st, ac, _), (eng_running, _alive_after) = jax.lax.scan(
            body, (state, acc, jnp.int32(0)), None, length=params.t_horizon
        )
        row = jnp.concatenate(
            [
                jnp.stack([jnp.mean(getattr(ac, f).astype(jnp.float32)) for f in _ACC_FIELDS]),
                jnp.stack(
                    [
                        jnp.mean(st.prey_died_predated.astype(jnp.float32)),
                        jnp.mean(st.prey_died_starving.astype(jnp.float32)),
                        jnp.mean(st.prey_alive.astype(jnp.float32)),
                    ]
                ),
            ]
        )
        prey = jnp.stack(
            [
                ac.alive_ticks.astype(jnp.int32),
                st.prey_death_step,
                st.prey_died_predated.astype(jnp.int32),
                st.prey_died_starving.astype(jnp.int32),
                st.prey_alive.astype(jnp.int32),
                st.prey_score.astype(jnp.int32),
            ]
        )
        return row, eng_running, prey, st.spawn_relax_level

    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


def engagement_counts(eng_running: np.ndarray, prey: np.ndarray) -> np.ndarray:
    """(G, T, k) running engaged_ticks and (G, 6, k) prey facts -> (G, len(ENGAGEMENT_FIELDS)) counts.

    Pure numpy, so it is unit-tested on hand-built arrays."""
    eng_running = np.asarray(eng_running, np.int64)
    G, T, k = eng_running.shape
    inc = np.diff(np.concatenate([np.zeros((G, 1, k), np.int64), eng_running], axis=1), axis=1)
    assert np.isin(inc, (0, 1)).all(), "engaged_ticks moved by something other than 0 or 1 in a tick"
    e = inc.astype(np.int8)
    d = np.diff(
        np.concatenate([np.zeros((G, 1, k), np.int8), e, np.zeros((G, 1, k), np.int8)], axis=1), axis=1
    )
    sg, st, sp = np.nonzero(d == 1)
    eg, et, ep = np.nonzero(d == -1)
    so = np.lexsort((st, sp, sg))
    eo = np.lexsort((et, ep, eg))
    sg, st, sp = sg[so], st[so], sp[so]
    eg, et, ep = eg[eo], et[eo] - 1, ep[eo]
    assert np.array_equal(sg, eg) and np.array_equal(sp, ep), "engagement starts and ends do not pair"

    death = prey[:, 1, :]
    predated = prey[:, 2, :].astype(bool)
    alive_end = prey[:, 4, :].astype(bool)
    ds = death[sg, sp]
    died_in = (ds >= st) & (ds <= et) & (ds >= 0)
    captured = died_in & predated[sg, sp]
    died_other = died_in & ~predated[sg, sp]
    censored = ~died_in & (et == T - 1) & alive_end[sg, sp]
    escaped = ~died_in & ~censored

    out = np.zeros((G, len(ENGAGEMENT_FIELDS)), np.int64)
    for j, mask in enumerate((np.ones_like(captured), escaped, captured, died_other, censored)):
        np.add.at(out[:, j], sg, mask.astype(np.int64))
    np.add.at(out[:, 5], sg, (et - st + 1))
    in_any = np.zeros((G, k), bool)
    in_any[sg[captured], sp[captured]] = True
    out[:, 6] = (predated & ~in_any).sum(axis=1)
    # **No identity is asserted here, and that is deliberate.** The four outcomes partition the engagements and
    # the run lengths sum to the counter's last value BY CONSTRUCTION: both are computed from the same counter,
    # so an assertion of either can never fail, and a check that cannot fail manufactures confidence. The
    # independent check is run 48's `sweep_mismatch`, against the sweep's own accumulator from a separate program.
    return out


def measure_keys(seed: int, after: int, n: int):
    return jax.random.split(
        jax.random.fold_in(jax.random.fold_in(jax.random.key(seed), MEASURE_TAG), after), n
    )


def measure(eng_pass, pop, static, seed: int, after: int, episodes: int):
    """Every genome over `episodes` episodes -> (engagement (G, E, F) int32, prey (G, E, 6, k) int16)."""
    keys = measure_keys(seed, after, pop.shape[0])
    fold = jax.vmap(jax.random.fold_in, in_axes=(0, None))
    eng, prey = [], []
    for e in range(episodes):
        _row, running, pr, _relax = eng_pass(pop, fold(keys, e), static)
        pr = np.asarray(pr)
        eng.append(engagement_counts(np.asarray(running), pr))
        prey.append(pr.astype(np.int16))
    return np.stack(eng, axis=1).astype(np.int32), np.stack(prey, axis=1)
