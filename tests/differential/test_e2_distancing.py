"""Gate E2 for D-065: the distancing and flight-initiation accumulators against the oracle, over random ticks.

**Why a second E2 file rather than more ticks in `test_e2_env.py`.** That gate's 60-tick episode was measured
to exercise almost none of D-065: one perceived tick, no occluded tick, no death, no unfled or censored
encounter. An agreement over paths that never ran is the inert-gate failure this project has met repeatedly,
so this file drives the paths directly: hundreds of single ticks from RANDOM states, with random obstacles,
positions, pools, encounter states and near-death prey, each run through both implementations and compared
on every accumulator and every state field.

**The coverage is asserted, not hoped for.** After the comparison, every branch D-065 adds must have occurred
somewhere in the sample: each distance band, occlusion, contact, the far baseline, a flight, an encounter
closed by lost perception, one closed by death, one left open and unfled, a blocked or masked emitted move.
If a future change makes any of them unreachable here, the coverage test fails rather than the agreement
quietly becoming vacuous.

The two implementations reach "what the prey perceived" by different routes: the implementation reads the
predator channel of the observation the policy received, the oracle asks the geometry (radius and line of
sight). So perception is compared as well as used.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import DEFAULT
from src.agent import network as net
from src.env import env as E
from src.env import resources
from src.env.entropy import PRNGSource, make_replay_source
from src.env.perception import build_vis_table
from src.reference import naive_env as oracle
from tests.differential.test_e2_env import tolerance_for

P = DEFAULT.__class__(h=14, w=14, k=6, t_horizon=1, n_food_src=3, n_target_src=3)
N_CASES = 120  # per predator count; the coverage test below proves this is enough to reach every branch

#: **Not compared HERE, and why.** `prey_hidden` differs between the two sides by float32 against float64
#: network arithmetic, and `test_e2_env.py`'s tolerance for it was fitted under that file's weight
#: perturbation of 0.3 and a zero starting hidden state. This file perturbs at 1.0 and starts from a random
#: hidden state, so that the emitted actions vary, and measured gaps up to 1.5e-2 there: a tolerance fitted
#: to one condition and applied to another. The hidden state is that file's subject and `test_e2_agent.py`'s;
#: what it could corrupt here is the ACTION, and the actions are compared exactly below instead.
NOT_COMPARED_HERE = frozenset({"prey_hidden"})

#: **The implementation's tick runs at full float32 matmul precision in this file, and why.** On this GPU
#: JAX's default runs a BATCHED float32 matmul in reduced precision (TF32): measured against a float64
#: reference on the same inputs, the batched network's logits were off by 7.5e-2 at the default and 9.0e-5
#: at "highest", while one prey at a time was 2.6e-5 either way. Seed 1048 had two logits 6.8e-3 apart, so
#: the default flipped its argmax and the oracle, in float64, did not. That is the network's arithmetic,
#: which `test_e2_agent.py` and the episode gate own; this file tests D-065's logic, so it removes the
#: rounding from the comparison rather than loosening anything. The default precision itself is recorded
#: as an open question for the owner, not changed here.
TICK_PRECISION = "highest"

NEW_FIELDS = tuple(f for f in E.Accumulators._fields if f.startswith(("dg_", "fid_")))
NEW_FIELDS += ("pred_present_ticks", "dist_sum_alive")


def _sites(n_pred: int) -> dict:
    return {
        "perceive.distant_detect": (1, P.k),
        "regrow.food.trigger": (1, P.n_food_src),
        "regrow.food.cell": (1, P.n_food_src),
        "regrow.target.trigger": (1, P.n_target_src),
        "regrow.target.cell": (1, P.n_target_src),
        "pred.wander.trigger": (1, n_pred),
        "pred.wander.dir": (1, n_pred),
    }


def _random_case(seed: int, n_pred: int):
    """One random world and one random, internally consistent starting state."""
    rng = np.random.default_rng(seed)
    pp = replace(P, n_pred=n_pred)
    obstacles_np = rng.random((P.h, P.w)) < 0.18
    free = np.argwhere(~obstacles_np)
    obstacles = jnp.asarray(obstacles_np)
    food_src, target_src, _ = resources.place_sources(jax.random.key(seed), obstacles, P)
    static = E.StaticEnv(
        obstacles=obstacles,
        vis_table=build_vis_table(obstacles, P.r),
        food_src=food_src,
        target_src=target_src,
    )
    state = E.reset(PRNGSource(jax.random.key(seed + 1)), static, pp)

    picks = free[rng.choice(len(free), P.k, replace=False)]
    # Predators are placed NEAR a random prey, at a random distance from adjacency to beyond perception, so
    # every band, contact, occlusion and the far baseline all occur across the sample.
    pred = []
    for _ in range(n_pred):
        anchor = picks[rng.integers(P.k)]
        d = int(rng.integers(0, P.r + 3))
        near = free[np.max(np.abs(free - anchor), axis=1) == d]
        pred.append(near[rng.integers(len(near))] if len(near) else free[rng.integers(len(free))])
    pred_pos = np.array(pred, np.int32).reshape(n_pred, 2)

    alive = rng.random(P.k) < 0.9
    energy = rng.choice([0.0, 0.5 * P.e_max, P.e_max], P.k).astype(np.float32)
    stamina = rng.choice([0.0, 0.5 * P.s_min_burst, P.s_max], P.k).astype(np.float32)
    health = np.where(rng.random(P.k) < 0.35, 1e-3, P.h_max).astype(np.float32)  # some will die this tick
    enc_open = rng.random(P.k) < 0.5
    enc_fled = enc_open & (rng.random(P.k) < 0.4)
    enc_age = np.where(enc_open, rng.integers(0, 9, P.k), 0).astype(np.int32)
    # A consistent state: only a living prey can be in an encounter, and with no predator in the world no
    # encounter can ever have opened.
    enc_open &= alive & (n_pred > 0)
    enc_fled &= enc_open
    enc_age = np.where(enc_open, enc_age, 0).astype(np.int32)

    state = state._replace(
        prey_pos=jnp.asarray(picks, jnp.int32),
        pred_pos=jnp.asarray(pred_pos),
        prey_alive=jnp.asarray(alive),
        prey_energy=jnp.asarray(energy),
        prey_stamina=jnp.asarray(stamina),
        prey_health=jnp.asarray(health),
        prey_hidden=jnp.asarray(rng.normal(0, 1, (P.k, P.hd)), jnp.float32),
        prey_enc_open=jnp.asarray(enc_open),
        prey_enc_fled=jnp.asarray(enc_fled),
        prey_enc_age=jnp.asarray(enc_age),
    )
    weights = net.init_params(jax.random.key(seed + 2), P)
    pert = jax.random.split(jax.random.key(seed + 3), len(weights))
    weights = {
        k: v + 1.0 * jax.random.normal(pert[i], v.shape, v.dtype)
        for i, (k, v) in enumerate(sorted(weights.items()))
    }
    return static, state, weights, pp


def _one_tick_both(seed: int, n_pred: int):
    static, state_j, weights_j, pp = _random_case(seed, n_pred)
    ent_j = make_replay_source(jax.random.key(seed + 4), _sites(n_pred))
    ent_n = oracle.ReplayReader({k: np.asarray(v) for k, v in ent_j.draws.items()})
    state_n = {f: np.asarray(getattr(state_j, f)) for f in state_j._fields}
    with jax.default_matmul_precision(TICK_PRECISION):
        new_j, acc_j, rec_j = E.tick(
            state_j, E.zero_accumulators(P.k), weights_j, static, ent_j, jnp.int32(0), pp
        )
    new_n, acc_n, rec_n = oracle.tick(
        state_n,
        oracle.zero_accumulators(P.k),
        {k: np.asarray(v, np.float64) for k, v in weights_j.items()},
        np.asarray(static.obstacles),
        np.asarray(static.food_src),
        np.asarray(static.target_src),
        ent_n,
        0,
        P,
    )
    for name in ("action_emitted", "action_executed"):
        np.testing.assert_array_equal(
            np.asarray(getattr(rec_j, name)), rec_n[name], err_msg=f"{name} seed {seed}"
        )
    return state_j, new_j, acc_j, new_n, acc_n


@pytest.fixture(scope="module", params=[1, 2], ids=["n_pred1", "n_pred2"])
def cases(request):
    n_pred = request.param
    return n_pred, [_one_tick_both(1000 * n_pred + s, n_pred) for s in range(N_CASES)]


@pytest.mark.slow
def test_every_accumulator_and_state_field_agrees_on_random_ticks(cases):
    n_pred, results = cases
    for s, (_, new_j, acc_j, new_n, acc_n) in enumerate(results):
        where = f"n_pred {n_pred} case {s}"
        assert set(acc_n) == set(acc_j._fields)
        for name in acc_j._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(acc_j, name)), acc_n[name], err_msg=f"{name}: {where}"
            )
        for name in new_j._fields:
            if name in NOT_COMPARED_HERE:
                continue
            a, b = np.asarray(getattr(new_j, name)), np.asarray(new_n[name])
            if a.dtype.kind == "f":
                atol, rtol = tolerance_for(name, n_pred)
                np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=f"{name}: {where}")
            else:
                np.testing.assert_array_equal(a, b, err_msg=f"{name}: {where}")


@pytest.mark.slow
def test_coverage_every_d065_branch_occurred_in_the_sample(cases):
    """The agreement above is only evidence for the branches it exercised. Each one is required here."""
    _, results = cases
    seen = dict.fromkeys(
        (
            "band_a",
            "band_b",
            "occluded",
            "far",
            "contact",
            "flight",
            "closed_by_lost_perception",
            "closed_by_death",
            "left_open_unfled",
            "emit_differs_from_exec",
            "negative_change",
            "positive_excess",
            "negative_excess",
        ),
        0,
    )
    for before, after, acc, _, _ in results:
        a = {f: np.asarray(getattr(acc, f)) for f in acc._fields}
        died = np.asarray(before.prey_alive) & ~np.asarray(after.prey_alive)
        was_open = np.asarray(before.prey_enc_open)
        seen["band_a"] += int((a["dg_a_ticks"] > 0).sum())
        seen["band_b"] += int((a["dg_b_ticks"] > 0).sum())
        seen["occluded"] += int((a["dg_occl_ticks"] > 0).sum())
        seen["far"] += int((a["dg_far_ticks"] > 0).sum())
        seen["contact"] += int((a["dg_contact_ticks"] > 0).sum())
        seen["flight"] += int((a["fid_fled"] > 0).sum())
        seen["closed_by_lost_perception"] += int(((a["fid_unfled_closed"] > 0) & was_open & ~died).sum())
        seen["closed_by_death"] += int(
            ((a["fid_unfled_closed"] > 0) & (a["fid_encounters"] > 0) & died).sum()
        )
        seen["left_open_unfled"] += int(
            (np.asarray(after.prey_enc_open) & ~np.asarray(after.prey_enc_fled)).sum()
        )
        seen["emit_differs_from_exec"] += int((a["dg_perc_emit_sum"] != a["dg_perc_exec_sum"]).sum())
        seen["negative_change"] += int((a["dg_perc_emit_sum"] < 0).sum())
        seen["positive_excess"] += int((a["dg_perc_excess_sum"] > 0).sum())
        seen["negative_excess"] += int((a["dg_perc_excess_sum"] < 0).sum())
    missing = [k for k, v in seen.items() if v == 0]
    assert not missing, f"branches never exercised by the random sample: {missing}; counts {seen}"


def test_a_predator_free_world_accumulates_nothing_on_either_side():
    """n_pred = 0, both sides: every D-065 field stays exactly zero, and no encounter ever opens."""
    for s in range(6):
        _, new_j, acc_j, new_n, acc_n = _one_tick_both(77 + s, 0)
        for name in NEW_FIELDS:
            assert not np.any(np.asarray(getattr(acc_j, name))), f"{name} fired with no predator"
            assert not np.any(acc_n[name]), f"oracle {name} fired with no predator"
        assert not np.any(np.asarray(new_j.prey_enc_open))
