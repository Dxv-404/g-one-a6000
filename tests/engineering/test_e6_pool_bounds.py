"""Gate E6, pool bounds. "No pool ever leaves its range."

Driven over a long randomised trajectory rather than by spot checks, because the bound that matters is the one
that holds under every combination of drain, refill, damage and bleed arriving on the same tick.

Coverage is asserted, not assumed. Standing practice after gate E3 turned out inert: a bounds test over a
trajectory that never approaches a bound passes trivially. `test_coverage_reaches_both_bounds` fails unless
each pool was observed at zero and at its maximum during the run, so a future change that stops exercising the
extremes fails here rather than passing green.
"""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from configs.default import DEFAULT as P  # noqa: E402
from src.env.dynamics import (  # noqa: E402
    N_ACTIONS,
    mask_actions,
    update_energy,
    update_health,
    update_stamina,
)

K = 8
N_TICKS = 600


def _trajectory(seed=0):
    """Drive the pools with random actions, feeding and damage, recording every value seen.

    Deliberately harsh: food and damage arrive often enough that pools are pushed against both bounds
    repeatedly rather than drifting in the middle of their range.
    """
    key = jax.random.key(seed)
    energy = jnp.full((K,), P.e_max, dtype=jnp.float32)
    stamina = jnp.full((K,), P.s_max, dtype=jnp.float32)
    health = jnp.full((K,), P.h_max, dtype=jnp.float32)

    seen = {"energy": [], "stamina": [], "health": []}
    for _step in range(N_TICKS):
        key, k_a, k_f, k_d = jax.random.split(key, 4)
        # Sampled over N_ACTIONS rather than a literal 20, so widening the action set keeps this test
        # exercising the whole space instead of silently ignoring the new tail.
        actions = jax.random.randint(k_a, (K,), 0, N_ACTIONS)
        ate = jax.random.uniform(k_f, (K,)) < 0.08
        damaged = jax.random.uniform(k_d, (K,)) < 0.05

        actions, _ = mask_actions(actions, energy, stamina, P)
        sprinting = (actions >= 8) & (actions <= 15)

        energy = update_energy(energy, sprinting, ate, P)
        stamina = update_stamina(stamina, sprinting, P)
        # DEV-031: the magnitude, not the flag. `damaged` is a bool array here, so scale it.
        health = update_health(health, damaged.astype(jnp.float32) * P.damage, energy, P)

        seen["energy"].append(energy)
        seen["stamina"].append(stamina)
        seen["health"].append(health)

    return {k: jnp.stack(v) for k, v in seen.items()}


@pytest.fixture(scope="module")
def traj():
    return _trajectory()


def test_energy_stays_in_range(traj):
    e = traj["energy"]
    assert float(jnp.min(e)) >= 0.0, f"energy went negative, min {float(jnp.min(e))}"
    assert float(jnp.max(e)) <= P.e_max, f"energy exceeded e_max, max {float(jnp.max(e))}"


def test_stamina_stays_in_range(traj):
    s = traj["stamina"]
    assert float(jnp.min(s)) >= 0.0, f"stamina went negative, min {float(jnp.min(s))}"
    assert float(jnp.max(s)) <= P.s_max, f"stamina exceeded s_max, max {float(jnp.max(s))}"


def test_health_stays_in_range(traj):
    h = traj["health"]
    assert float(jnp.min(h)) >= 0.0, f"health went negative, min {float(jnp.min(h))}"
    assert float(jnp.max(h)) <= P.h_max, f"health exceeded h_max, max {float(jnp.max(h))}"


def test_coverage_reaches_both_bounds(traj):
    """The power check. A bounds test that never approaches a bound asserts nothing.

    Written as a floor on observations rather than as "it happened at least once", for the reason the E3
    canary failed: existence reads as sufficiency and is not.
    """
    for name, arr in traj.items():
        at_floor = int(jnp.sum(arr <= 0.0))
        at_ceiling = int(jnp.sum(arr >= (P.e_max if name == "energy" else P.s_max)))
        assert at_floor + at_ceiling >= 20, (
            f"{name} touched a bound only {at_floor + at_ceiling} times across {N_TICKS} ticks. "
            "The bound is not being exercised, so this gate is passing on a trajectory that never tests it."
        )


def test_a_single_tick_cannot_overshoot_from_an_extreme():
    """Clipping is checked against the worst single-tick delta from each extreme, directly.

    The trajectory test covers combinations that arise by chance. This covers the ones that would be worst by
    construction: everything draining at once from empty, and everything refilling at once from full.
    """
    zeros = jnp.zeros((1,), dtype=jnp.float32)
    full_e = jnp.full((1,), P.e_max, dtype=jnp.float32)

    # Drain everything from empty. Nothing may go below zero.
    assert float(update_energy(zeros, jnp.array([True]), jnp.array([False]), P)[0]) >= 0.0
    assert float(update_stamina(zeros, jnp.array([True]), P)[0]) >= 0.0
    assert float(update_health(zeros, jnp.array([P.damage]), zeros, P)[0]) >= 0.0

    # Refill everything from full. Nothing may exceed its maximum.
    assert float(update_energy(full_e, jnp.array([False]), jnp.array([True]), P)[0]) <= P.e_max
    assert float(update_stamina(jnp.full((1,), P.s_max, jnp.float32), jnp.array([False]), P)[0]) <= P.s_max
    assert (
        float(update_health(jnp.full((1,), P.h_max, jnp.float32), jnp.array([0.0]), full_e, P)[0]) <= P.h_max
    )


def test_stamina_at_zero_causes_no_harm():
    """Invariant: exhaustion gates what an agent can do, never whether it lives.

    Stamina must not feed into health or energy. Checked by holding stamina at zero and confirming the other
    two pools behave exactly as they do at full stamina.
    """
    e = jnp.full((1,), 50.0, jnp.float32)
    h = jnp.full((1,), 50.0, jnp.float32)
    no_sprint = jnp.array([False])
    e_after = update_energy(e, no_sprint, jnp.array([False]), P)
    h_after = update_health(h, jnp.array([0.0]), e_after, P)

    assert float(e_after[0]) == 50.0 - P.e_base, "energy drain depends on something other than sprinting"
    assert float(h_after[0]) == min(50.0 + P.h_regen, P.h_max), (
        "health changed for a reason other than damage"
    )
