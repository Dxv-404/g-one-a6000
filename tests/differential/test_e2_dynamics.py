"""Gate E2 over the step 4 dynamics: masks, the three pools, and write-once death.

**This closes the oracle debt that has been the whole of E2's gap since build step 4.** The oracle covered
grid, visibility, resources and then the predator, while `mask_actions`, `update_energy`, `update_stamina`,
`update_health`, `resolve_deaths` and `update_posture` had no counterpart at all, so E2 reported the same
green over a system missing six functions. See CLAUDE.md section 10.

The oracle side is written from the build specification's update equations rather than transcribed from
`src/env/dynamics.py`. That independence is the only thing that makes agreement mean anything: a reference
copied from the implementation is an equality test against itself.

### Coverage is asserted, not assumed

An agreement test over a path that never executes agrees trivially, which is the state gate E3 shipped in.
`test_coverage_is_sufficient` fails unless every branch that matters actually fired: each mask reason, each
death cause, both pool clipping bounds, and each posture.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import DEFAULT as P
from src.env import dynamics as fast
from src.reference import naive_env as oracle

K = 24
N_TICKS = 250


def _drive(seed: int):
    """Drive both implementations tick by tick from identical state, and record every intermediate.

    Three regimes by construction rather than by chance, so both death causes and survivors all occur:
    starvers are never fed, prey are fed and damaged, survivors are fed and undamaged. Gate E7 learned this
    the hard way when a uniform regime produced zero starvation deaths.
    """
    rng = np.random.default_rng(seed)

    energy = np.full(K, P.e_max, np.float32)
    stamina = np.full(K, P.s_max, np.float32)
    health = np.full(K, P.h_max, np.float32)
    posture = np.zeros(K, np.int32)
    alive = np.ones(K, bool)
    cause = np.zeros(K, np.int32)
    dstep = np.full(K, -1, np.int32)
    starved = np.zeros(K, bool)
    predated = np.zeros(K, bool)

    group = np.arange(K) % 3
    feed_p = np.where(group == 0, 0.0, 0.40)
    damage_p = np.where(group == 1, 0.05, 0.0)

    seen = dict(mask_reasons=set(), causes=set(), postures=set(), hit_zero=set(), hit_max=set())

    for step in range(N_TICKS):
        actions = rng.integers(0, fast.N_ACTIONS, size=K).astype(np.int32)
        ate = rng.random(K) < feed_p
        damaged = (rng.random(K) < damage_p) & alive

        m_j, b_j = fast.mask_actions(jnp.asarray(actions), jnp.asarray(energy), jnp.asarray(stamina), P)
        m_n, b_n = oracle.mask_actions(actions, energy, stamina, P)
        assert [int(x) for x in m_j] == [int(x) for x in m_n], f"masked actions, tick {step}"
        assert [bool(x) for x in b_j] == [bool(x) for x in b_n], f"mask events, tick {step}"

        for k in range(K):
            a = int(actions[k])
            if bool(b_n[k]):
                if oracle.is_reserved(a):
                    seen["mask_reasons"].add("reserved")
                elif oracle.is_burst(a):
                    seen["mask_reasons"].add("burst")
                else:
                    seen["mask_reasons"].add("complex")

        masked = np.asarray(m_n)
        sprinting = np.array([oracle.is_burst(a) for a in masked], dtype=bool)

        e_j = fast.update_energy(jnp.asarray(energy), jnp.asarray(sprinting), jnp.asarray(ate), P)
        e_n = oracle.update_energy(energy, sprinting, ate, P)
        np.testing.assert_allclose(
            np.asarray(e_j), e_n, atol=P.e2_atol, rtol=P.e2_rtol, err_msg=f"energy, tick {step}"
        )

        s_j = fast.update_stamina(jnp.asarray(stamina), jnp.asarray(sprinting), P)
        s_n = oracle.update_stamina(stamina, sprinting, P)
        np.testing.assert_allclose(
            np.asarray(s_j), s_n, atol=P.e2_atol, rtol=P.e2_rtol, err_msg=f"stamina, tick {step}"
        )

        # **Graded magnitudes, not a boolean scaled by DAMAGE. DEV-031.**
        # Comparing the two sides on 0-or-DAMAGE inputs would agree whatever the fix did, because that is
        # exactly the case where the old boolean reduction was exact. The differential is inert for this
        # change anyway, since it lands identically on both sides, but the fixture should at least carry
        # values that could tell the two apart.
        dmg = np.where(damaged, P.damage * (1.0 - 0.5 * (step % 3)), 0.0).astype(np.float32)
        h_j = fast.update_health(jnp.asarray(health), jnp.asarray(dmg), jnp.asarray(e_n), P)
        h_n = oracle.update_health(health, dmg, e_n, P)
        np.testing.assert_allclose(
            np.asarray(h_j), h_n, atol=P.e2_atol, rtol=P.e2_rtol, err_msg=f"health, tick {step}"
        )

        p_j = fast.update_posture(jnp.asarray(posture), jnp.asarray(masked))
        p_n = oracle.update_posture(posture, masked)
        assert [int(x) for x in p_j] == [int(x) for x in p_n], f"posture, tick {step}"

        a_j, c_j, d_j, starve_j, pred_j = fast.resolve_deaths(
            jnp.asarray(alive),
            jnp.asarray(h_n),
            jnp.asarray(e_n),
            jnp.asarray(damaged),
            jnp.asarray(cause),
            jnp.asarray(dstep),
            jnp.asarray(starved),
            jnp.asarray(predated),
            step,
        )
        a_n, c_n, d_n, starve_n, pred_n = oracle.resolve_deaths(
            alive, h_n, e_n, damaged, cause, dstep, starved, predated, step
        )
        assert [bool(x) for x in starve_j] == [bool(x) for x in starve_n], f"died_starving, tick {step}"
        assert [bool(x) for x in pred_j] == [bool(x) for x in pred_n], f"died_predated, tick {step}"
        starved, predated = np.asarray(starve_n), np.asarray(pred_n)
        assert [bool(x) for x in a_j] == [bool(x) for x in a_n], f"alive, tick {step}"
        assert [int(x) for x in c_j] == [int(x) for x in c_n], f"death cause, tick {step}"
        assert [int(x) for x in d_j] == [int(x) for x in d_n], f"death step, tick {step}"

        seen["causes"].update(int(x) for x in c_n)
        seen["postures"].update(int(x) for x in p_n)
        for arr, name in ((e_n, "energy"), (s_n, "stamina"), (h_n, "health")):
            if float(np.min(arr)) <= 0.0:
                seen["hit_zero"].add(name)
        if float(np.max(s_n)) >= P.s_max:
            seen["hit_max"].add("stamina")
        if float(np.max(e_n)) >= P.e_max:
            seen["hit_max"].add("energy")

        energy, stamina, health, posture = e_n, s_n, h_n, p_n
        alive, cause, dstep = a_n, c_n, d_n

    return seen


@pytest.fixture(scope="module")
def run():
    return _drive(seed=0)


@pytest.mark.slow  # 45s measured at seed 2: every pool compared tick by tick
@pytest.mark.parametrize("seed", [1, 2])
def test_dynamics_match_the_oracle_tick_by_tick(seed):
    """Every field compared every tick, so the first divergence is reported where it happened.

    The assertions live inside `_drive` because the two implementations are stepped in lockstep and the
    oracle's output is what carries state forward: comparing only at the end would let an early divergence
    be masked by clipping.

    **Seed 0 is deliberately absent from this list and is not untested.** The module-scoped `run` fixture
    drives it, and every comparison lives inside `_drive`, so the fixture performs the same assertions. An
    earlier version parametrised over 0, 1 and 2 as well as building the fixture from seed 0, which drove
    that seed twice and cost a redundant 165 seconds in a suite that had just grown by four minutes.
    """
    _drive(seed)


def test_coverage_is_sufficient(run):
    """The branches this gate claims to compare must actually execute."""
    assert run["mask_reasons"] == {"burst", "complex", "reserved"}, (
        f"only these mask reasons fired: {run['mask_reasons']}. A reason that never fires is not compared"
    )
    assert oracle.CAUSE_STARVATION in run["causes"], "no starvation death occurred, so that path is untested"
    assert oracle.CAUSE_PREDATION in run["causes"], "no predation death occurred, so that path is untested"
    assert run["postures"] == {0, 1, 2}, f"not every posture occurred: {run['postures']}"
    assert "energy" in run["hit_zero"], "energy never reached its lower clip, so clipping is untested"
    assert "stamina" in run["hit_zero"], "stamina never reached its lower clip"
    assert "stamina" in run["hit_max"], "stamina never reached its upper clip"


def test_the_oracle_is_not_a_copy_of_the_implementation():
    """A structural guard on the thing that makes this gate meaningful.

    The oracle must not import from `src.env`. If it ever does, the two implementations have collapsed into
    one and every assertion above becomes an equality test against itself. Cheap to check and impossible to
    notice by reading a green result.
    """
    import inspect

    source = inspect.getsource(oracle)
    assert "from src.env" not in source and "import src.env" not in source, (
        "the oracle imports from src.env, so it is no longer independent of the code it checks"
    )
