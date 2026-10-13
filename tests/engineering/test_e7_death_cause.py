"""Gate E7, death cause write-once.

"Every death carries exactly one cause, recorded on the death tick, never overwritten."

Coverage is asserted, not assumed. **A death cause never exercised because nothing died during the test is
exactly the defect gate E3 had, wearing different clothes.** So the trajectory test asserts that both causes
were actually produced, and that some agents died and some survived, before it asserts anything about how they
were recorded.
"""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from configs.default import DEFAULT as P  # noqa: E402
from src.env.dynamics import (  # noqa: E402
    CAUSE_ALIVE,
    CAUSE_PREDATION,
    CAUSE_STARVATION,
    resolve_deaths,
    update_energy,
    update_health,
)

K = 24
N_TICKS = 400


def _run(seed=0):
    """Drive to death by both routes and record the full history of cause and step.

    **Coverage is arranged by construction, not left to chance.** The first version fed all agents sparsely
    and damaged them all, expecting both causes to appear. Zero starvation deaths resulted, and the coverage
    assertion caught it: three contacts kill in roughly 75 ticks at a 4 percent damage rate, while energy takes
    200 ticks to drain from full, so predation always won the race and the starvation path was never
    exercised. Randomness producing all the outcomes you need is a hope, not a design.

    Three explicit regimes instead, so each outcome is guaranteed:

        starvers   never fed, never damaged   energy empties at tick 200, then bleeds to death near 257
        prey       fed generously, damaged    dies of predation with energy still in hand
        survivors  fed generously, undamaged  alive at the horizon
    """
    key = jax.random.key(seed)
    energy = jnp.full((K,), P.e_max, jnp.float32)
    health = jnp.full((K,), P.h_max, jnp.float32)
    alive = jnp.ones((K,), dtype=bool)
    cause = jnp.zeros((K,), jnp.int32)
    dstep = jnp.full((K,), -1, jnp.int32)

    group = jnp.arange(K) % 3  # 0 starvers, 1 prey, 2 survivors
    feed_p = jnp.where(group == 0, 0.0, 0.40)
    damage_p = jnp.where(group == 1, 0.04, 0.0)

    history = []
    starved = jnp.zeros_like(alive)
    predated = jnp.zeros_like(alive)
    for step in range(N_TICKS):
        key, k_f, k_d = jax.random.split(key, 3)
        ate = jax.random.uniform(k_f, (K,)) < feed_p
        damaged = (jax.random.uniform(k_d, (K,)) < damage_p) & alive

        energy = update_energy(energy, jnp.zeros((K,), bool), ate, P)
        # DEV-031: `update_health` takes the magnitude; `resolve_deaths` still takes the flag,
        # because it asks whether predation damage occurred rather than how much.
        health = update_health(health, damaged.astype(jnp.float32) * P.damage, energy, P)
        alive, cause, dstep, starved, predated = resolve_deaths(
            alive, health, energy, damaged, cause, dstep, starved, predated, step
        )
        history.append((alive, cause, dstep))

    return alive, cause, dstep, history


@pytest.fixture(scope="module")
def outcome():
    return _run()


def test_coverage_both_causes_and_survivors_occur(outcome):
    """The power check, asserted before anything else in this file means anything."""
    alive, cause, _, _ = outcome
    n_starve = int(jnp.sum(cause == CAUSE_STARVATION))
    n_pred = int(jnp.sum(cause == CAUSE_PREDATION))
    n_alive = int(jnp.sum(alive))

    assert n_starve >= 3, f"only {n_starve} starvation deaths; the starvation path is barely exercised"
    assert n_pred >= 3, f"only {n_pred} predation deaths; the predation path is barely exercised"
    assert n_alive >= 1, "every agent died, so the surviving-agent case is untested"


def test_every_dead_agent_has_exactly_one_cause(outcome):
    alive, cause, _, _ = outcome
    dead = ~alive
    assert bool(jnp.all(cause[dead] != CAUSE_ALIVE)), "a dead agent carries no recorded cause"
    assert bool(jnp.all((cause[dead] == CAUSE_STARVATION) | (cause[dead] == CAUSE_PREDATION))), (
        "a dead agent carries a cause outside the defined set"
    )


def test_every_living_agent_has_no_cause_and_no_death_step(outcome):
    alive, cause, dstep, _ = outcome
    assert bool(jnp.all(cause[alive] == CAUSE_ALIVE)), "a living agent has a death cause recorded"
    assert bool(jnp.all(dstep[alive] == -1)), "a living agent has a death step recorded"


@pytest.mark.slow  # 7s measured, walks the full 400-tick history
def test_cause_is_never_overwritten(outcome):
    """The write-once property. Once set, neither cause nor step may change on any later tick.

    This is the assertion that catches the failure with no visible symptom: an agent whose cause is rewritten
    on a later tick still looks dead and still has a valid cause, so nothing else in the pipeline notices.
    """
    _, _, _, history = outcome
    prev_alive, prev_cause, prev_step = history[0]
    for tick in range(1, len(history)):
        alive, cause, dstep = history[tick]
        already_dead = ~prev_alive
        assert bool(jnp.all(cause[already_dead] == prev_cause[already_dead])), (
            f"tick {tick}: a cause changed for an agent that was already dead"
        )
        assert bool(jnp.all(dstep[already_dead] == prev_step[already_dead])), (
            f"tick {tick}: a death step changed for an agent that was already dead"
        )
        assert bool(jnp.all(~alive[already_dead])), f"tick {tick}: a dead agent came back alive"
        prev_alive, prev_cause, prev_step = alive, cause, dstep


@pytest.mark.slow  # 12s measured, walks the full 400-tick history
def test_death_step_is_the_tick_health_reached_zero(outcome):
    """Recorded on the death tick, not later. Checked against the first tick each agent was not alive."""
    _, _, dstep, history = outcome
    for k in range(K):
        first_dead = None
        for tick, (alive, _, _) in enumerate(history):
            if not bool(alive[k]):
                first_dead = tick
                break
        if first_dead is None:
            assert int(dstep[k]) == -1, f"agent {k} survived but has death step {int(dstep[k])}"
        else:
            assert int(dstep[k]) == first_dead, (
                f"agent {k} first dead at tick {first_dead} but death step records {int(dstep[k])}"
            )


def test_starvation_takes_precedence_when_both_conditions_hold():
    """Attribution when an agent reaches zero health on a tick where energy is also zero.

    The build spec's rule is `cause = where(energy' <= 0, STARVATION, PREDATION)`, so starvation wins. Asserted
    directly rather than left to arise by chance, because it is the one case where the two causes compete and
    the precedence is a specification choice rather than a consequence.
    """
    alive = jnp.array([True])
    health = jnp.array([0.0], jnp.float32)
    energy_zero = jnp.array([0.0], jnp.float32)
    energy_left = jnp.array([50.0], jnp.float32)

    zero_flag = jnp.zeros((1,), bool)
    _, cause_a, _, _, _ = resolve_deaths(
        alive,
        health,
        energy_zero,
        zero_flag,
        jnp.array([0], jnp.int32),
        jnp.array([-1], jnp.int32),
        zero_flag,
        zero_flag,
        7,
    )
    assert int(cause_a[0]) == CAUSE_STARVATION, "zero energy at death must record starvation"

    _, cause_b, step_b, _, _ = resolve_deaths(
        alive,
        health,
        energy_left,
        jnp.ones((1,), bool),
        jnp.array([0], jnp.int32),
        jnp.array([-1], jnp.int32),
        zero_flag,
        zero_flag,
        7,
    )
    assert int(cause_b[0]) == CAUSE_PREDATION, "energy remaining at death must record predation"
    assert int(step_b[0]) == 7, "the death step must be the tick death was resolved on"


def test_a_dead_agent_keeps_its_array_slot():
    """Dead agents are masked, never removed. Correct under static shapes, and it must stay that way."""
    alive = jnp.array([True, True, True])
    health = jnp.array([0.0, 50.0, 0.0], jnp.float32)
    energy = jnp.array([50.0, 50.0, 0.0], jnp.float32)
    alive_after, cause, dstep, _, _ = resolve_deaths(
        alive,
        health,
        energy,
        jnp.zeros(3, bool),
        jnp.zeros(3, jnp.int32),
        jnp.full(3, -1, jnp.int32),
        jnp.zeros(3, bool),
        jnp.zeros(3, bool),
        3,
    )
    assert alive_after.shape == (3,) and cause.shape == (3,) and dstep.shape == (3,)
    assert [bool(x) for x in alive_after] == [False, True, False]
    assert [int(c) for c in cause] == [CAUSE_PREDATION, CAUSE_ALIVE, CAUSE_STARVATION]


def test_both_death_flags_can_be_true_at_once():
    """**The case the precedence rule discarded, and the reason the two flags exist.**

    An agent that is starving AND takes a lethal hit is recorded as starvation by
    `where(energy <= 0, STARVATION, PREDATION)`. That under-counts predation among exactly the weakened
    agents most likely to be caught, which is a bias in the direction that matters: it makes predation look
    less lethal precisely where it is most lethal.

    The flags are independent, so this state is representable rather than resolved by a tie-break. **There is
    no tie left to break**, which is enforcement by construction rather than a better rule.
    """
    alive = jnp.array([True])
    health = jnp.array([0.0], jnp.float32)
    starving_energy = jnp.array([0.0], jnp.float32)
    was_hit = jnp.array([True])
    z = jnp.zeros((1,), bool)

    _, cause, _, starved, predated = resolve_deaths(
        alive,
        health,
        starving_energy,
        was_hit,
        jnp.array([0], jnp.int32),
        jnp.array([-1], jnp.int32),
        z,
        z,
        5,
    )

    assert bool(starved[0]) and bool(predated[0]), (
        "an agent that starved AND was struck must record both. Recording one discards the fact that "
        "predation killed a weakened agent, which is the population the predator most affects."
    )
    assert int(cause[0]) == CAUSE_STARVATION, (
        "the derived single cause keeps the old precedence, so existing consumers are unaffected"
    )


def test_the_flags_are_write_once_like_the_cause():
    """A flag set on the tick of death is never revisited, even though the slot keeps updating."""
    z = jnp.zeros((1,), bool)
    dead, cause, dstep, starved, predated = resolve_deaths(
        jnp.array([True]),
        jnp.array([0.0], jnp.float32),
        jnp.array([50.0], jnp.float32),
        jnp.array([True]),
        jnp.array([0], jnp.int32),
        jnp.array([-1], jnp.int32),
        z,
        z,
        2,
    )
    assert bool(predated[0]) and not bool(starved[0])

    # A later tick where the agent is already dead and now also has zero energy must not add a flag.
    _, _, _, starved2, predated2 = resolve_deaths(
        dead,
        jnp.array([0.0], jnp.float32),
        jnp.array([0.0], jnp.float32),
        jnp.array([True]),
        cause,
        dstep,
        starved,
        predated,
        9,
    )
    assert not bool(starved2[0]), "a flag was written after death, so the record is not write-once"
    assert bool(predated2[0]), "the flag set at death was lost"
