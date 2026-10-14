"""D-065's distancing and flight-initiation accumulators, against hand-worked expectations. Property tests.

**Why these exist beside the differential.** `tests/differential/test_e2_distancing.py` proves the
implementation and the oracle agree. Agreement cannot catch a blind spot both authors share, which is
DEV-014's lesson, so these tests consult the RULES instead: each scenario places a prey and a predator at
known cells, scripts the prey's action, and asserts the exact value worked out by hand from the
definitions in `src/env/env.py`. The numbers in the assertions were computed on paper, not read off a run.

The geometry, used throughout: the prey starts at (6, 6) on a 12 by 12 open grid, Chebyshev distance, the
observation radius `r` is 4 and `r_near` is 2. Direction indices: 0 N, 1 NE, 2 E, 3 SE, 4 S, 5 SW, 6 W,
7 NW; sprints are the same plus 8; 16 is stay. Prey 1 is dead in every scenario so only prey 0 acts.

**The statistical test at the end is the bias test.** A predator-blind random walker must score zero on
the chance-corrected measure, and the RAW measure must show the geometric bias that made the correction
necessary. If the raw bias ever disappears the correction is guarding nothing; if the corrected value
drifts from zero the correction is wrong.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import DEFAULT
from src.agent import baseline
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.observation import CH_PREDATOR
from src.env.perception import build_vis_table
from src.reference import naive_env as oracle

# No wander and no distant detection, so a predator beyond r_near stays where it is while the prey is still.
P = DEFAULT.__class__(
    h=12, w=12, k=2, t_horizon=8, n_food_src=2, n_target_src=2, p_wander=0.0, p_distant_detect=0.0
)
PREY = (6, 6)
N, NE, E_, SE, S, SW, W, NW = range(8)
STAY = 16


def _world(obstacle_cells=()):
    ob = np.zeros((P.h, P.w), bool)
    for c in obstacle_cells:
        ob[c] = True
    obstacles = jnp.asarray(ob)
    fs, ts, _ = resources.place_sources(jax.random.key(0), obstacles, P)
    return E.StaticEnv(
        obstacles=obstacles, vis_table=build_vis_table(obstacles, P.r), food_src=fs, target_src=ts
    )


def _state(static, pred_cells, *, stamina=None, energy=None, health=None, params=P):
    st = E.reset(ent.PRNGSource(jax.random.key(1)), static, params)
    return st._replace(
        prey_pos=jnp.array([PREY, (0, 0)], jnp.int32),
        prey_alive=jnp.array([True, False]),
        pred_pos=jnp.array(pred_cells, jnp.int32).reshape(len(pred_cells), 2),
        prey_stamina=st.prey_stamina if stamina is None else st.prey_stamina.at[0].set(stamina),
        prey_energy=st.prey_energy if energy is None else st.prey_energy.at[0].set(energy),
        prey_health=st.prey_health if health is None else st.prey_health.at[0].set(health),
    )


def _scripted(actions):
    table = jnp.asarray(actions, jnp.int32)

    def policy(weights, obs, hidden, step_index):
        return jnp.full((obs.shape[0],), table[step_index], jnp.int32), hidden

    return policy


def _run(static, state, actions, params=P):
    acc = E.zero_accumulators(params.k)
    w = net.init_params(jax.random.key(2), params)
    pol = _scripted(actions)
    states = [state]
    for t in range(len(actions)):
        state, acc, _ = E.tick(
            state, acc, w, static, ent.PRNGSource(jax.random.key(3)), jnp.int32(t), params, pol
        )
        states.append(state)
    return {f: float(np.asarray(getattr(acc, f))[0]) for f in acc._fields}, states


def _one(pred_cells, action, obstacles=(), **pools):
    static = _world(obstacles)
    acc, _ = _run(static, _state(static, pred_cells, **pools), [action])
    return acc


def _perceives(static, pred_cell) -> bool:
    """The precondition, from the oracle's geometry: radius r and line of sight both ways."""
    d = max(abs(PREY[0] - pred_cell[0]), abs(PREY[1] - pred_cell[1]))
    return d <= P.r and oracle.visible(np.asarray(static.obstacles), PREY, tuple(pred_cell))


# ------------------------------------------------------------------- single ticks, worked by hand --


@pytest.mark.slow  # 5.34s measured warm, 46s cold: the first test here pays the eager tick's compile
def test_a_move_directly_away_scores_one_and_is_neutral_by_chance():
    """Predator two cells south. N reaches distance 3: +1. Over the eight moves from (6,6) the changes are
    N +1, NE +1, E 0, SE -1, S -1, SW -1, W 0, NW +1, mean 0, so the excess is also +1. The best available
    is a sprint north, +2."""
    a = _one([(8, 6)], N)
    assert (a["dg_perc_ticks"], a["dg_a_ticks"], a["dg_b_ticks"], a["dg_perc_moves"]) == (1, 1, 0, 1)
    assert (a["dg_perc_emit_sum"], a["dg_perc_exec_sum"], a["dg_perc_max_sum"]) == (1, 1, 2)
    assert a["dg_perc_excess_sum"] == 1.0
    assert (a["dg_contact_ticks"], a["dg_occl_ticks"], a["dg_far_ticks"], a["pred_present_ticks"]) == (
        0,
        0,
        0,
        1,
    )


def test_a_sprint_away_counts_two_cells_and_its_chance_is_at_sprint_reach():
    """Sprint N reaches (4,6), distance 4: +2. At reach two the eight changes are +2, +2, 0, 0, -2, 0, 0, +2,
    mean +0.5, so the excess is +1.5."""
    a = _one([(8, 6)], 8 + N)
    assert (a["dg_perc_emit_sum"], a["dg_perc_exec_sum"]) == (2, 2)
    assert a["dg_perc_excess_sum"] == 1.5


def test_a_move_toward_the_predator_is_negative():
    a = _one([(8, 6)], S)
    assert (a["dg_perc_emit_sum"], a["dg_perc_exec_sum"], a["dg_perc_excess_sum"]) == (-1, -1, -1.0)


def test_staying_is_zero_and_is_not_a_move():
    a = _one([(8, 6)], STAY)
    assert (a["dg_perc_ticks"], a["dg_perc_moves"]) == (1, 0)
    assert (a["dg_perc_emit_sum"], a["dg_perc_exec_sum"], a["dg_perc_excess_sum"]) == (0, 0, 0.0)
    assert a["dg_perc_max_sum"] == 2, "staying does not change what was available"


def test_a_blocked_move_counts_the_intent_and_the_executed_change_separately():
    """A wall at (5,6). N is refused, so the prey stays: intent +1, executed 0. D-031's rule, and the reason
    both are kept. The sprint north is blocked at its first cell, but the diagonal sprints are not: NE through
    (5,7) to (4,8) is distance 4, so the best available is still +2."""
    a = _one([(8, 6)], N, obstacles=[(5, 6)])
    assert (a["dg_perc_emit_sum"], a["dg_perc_exec_sum"], a["dg_perc_max_sum"]) == (1, 0, 2)


def test_a_masked_sprint_counts_the_intent_and_the_best_available_respects_the_mask():
    """Stamina zero: a sprint is masked to stay. Intent +2, executed 0, and the best AVAILABLE is a single move,
    +1, because no sprint was available this tick."""
    a = _one([(8, 6)], 8 + N, stamina=0.0)
    assert (a["dg_perc_emit_sum"], a["dg_perc_exec_sum"], a["dg_perc_max_sum"]) == (2, 0, 1)


def test_the_chance_correction_removes_the_diagonal_bias():
    """**The bias itself.** Predator diagonally at (8,8), distance 2. From (6,6) the eight moves reach
    N (5,6) 3, NE (5,7) 3, E (6,7) 2, SE (7,7) 1, S (7,6) 2, SW (7,5) 3, W (6,5) 3, NW (5,5) 3, so the
    changes are +1, +1, 0, -1, 0, +1, +1, +1, mean +0.5. A predator-blind walker gains half a cell per move
    here, and the correction subtracts it: E scores raw 0 and excess -0.5, N raw +1 and excess +0.5."""
    east = _one([(8, 8)], E_)
    north = _one([(8, 8)], N)
    assert (east["dg_perc_emit_sum"], east["dg_perc_excess_sum"]) == (0, -0.5)
    assert (north["dg_perc_emit_sum"], north["dg_perc_excess_sum"]) == (1, 0.5)


def test_band_b_is_beyond_r_near_and_within_sight():
    a = _one([(10, 6)], N)  # distance 4
    assert (a["dg_perc_ticks"], a["dg_a_ticks"], a["dg_b_ticks"]) == (1, 0, 1)
    assert a["dg_b_excess_sum"] == 1.0  # straight south: chance is zero


def test_an_occluded_predator_is_not_perceived_and_is_counted_as_occluded():
    wall = [(8, 4), (8, 5), (8, 6), (8, 7), (8, 8)]
    static = _world(wall)
    assert not _perceives(static, (9, 6)), "precondition: the wall must block the line of sight"
    a, states = _run(static, _state(static, [(9, 6)]), [N])
    obs_pred = _observation_predator_cells(static, states[0])
    assert obs_pred == 0, "precondition: the prey's own observation must show no predator"
    assert (a["dg_perc_ticks"], a["dg_occl_ticks"], a["dg_occl_emit_sum"]) == (0, 1, 1)
    assert a["fid_encounters"] == 0, "an unseen predator must not open an encounter"


def test_a_predator_beyond_sight_is_the_far_baseline():
    a = _one([(6, 11)], E_)  # distance 5 > r; E moves toward it
    assert (a["dg_far_ticks"], a["dg_far_emit_sum"], a["dg_perc_ticks"]) == (1, -1, 0)


def test_adjacency_is_contact_and_still_opens_an_encounter():
    a = _one([(7, 7)], STAY)
    assert (a["dg_contact_ticks"], a["dg_perc_ticks"], a["fid_encounters"], a["fid_onset_sum"]) == (
        1,
        0,
        1,
        1,
    )


def test_the_partition_holds_on_every_hand_scenario():
    """contact + perceived + occluded + far = ticks with a predator present, one category per tick."""
    for cells, act in [([(8, 6)], N), ([(10, 6)], STAY), ([(6, 11)], W), ([(7, 7)], S), ([(8, 8)], E_)]:
        a = _one(cells, act)
        parts = a["dg_contact_ticks"] + a["dg_perc_ticks"] + a["dg_occl_ticks"] + a["dg_far_ticks"]
        assert parts == a["pred_present_ticks"] == 1


# ------------------------------------------------------------------- the encounter, across ticks --


def test_flight_initiation_distance_and_latency():
    """Predator three cells east, beyond r_near, so with no wander and no distant detection it does not move
    while the prey is still. Stay, stay, then W, which reaches distance 4: the flight. Onset at distance 3 on
    tick 0, flight at distance 3 on tick 2, so FID 3 and latency 2."""
    static = _world()
    a, states = _run(static, _state(static, [(6, 9)]), [STAY, STAY, W])
    assert np.array_equal(np.asarray(states[2].pred_pos), np.asarray(states[0].pred_pos)), (
        "precondition: the predator must not move while the prey is still"
    )
    assert (a["fid_encounters"], a["fid_onset_sum"], a["fid_fled"]) == (1, 3, 1)
    assert (a["fid_dist_sum"], a["fid_latency_sum"], a["fid_unfled_closed"]) == (3, 2, 0)


def test_an_encounter_lost_from_sight_without_flight_is_closed_unfled():
    """A wall from the top edge down to row 5 at column 8. From (6,6) the predator at (6,10) is in sight along
    row 6; one step N to (5,6) keeps the distance at 4, so it is not a flight, and puts the wall in the way."""
    wall = [(r, 8) for r in range(0, 6)]
    static = _world(wall)
    assert _perceives(static, (6, 10)), "precondition: visible from the start cell"
    assert not oracle.visible(np.asarray(static.obstacles), (5, 6), (6, 10)), (
        "precondition: hidden from the cell one step north"
    )
    a, _ = _run(static, _state(static, [(6, 10)]), [N, STAY])
    assert (a["fid_encounters"], a["fid_fled"], a["fid_unfled_closed"]) == (1, 0, 1)


def test_an_encounter_ended_by_death_without_flight_is_closed_unfled():
    """Energy zero and health almost gone: the starvation bleed kills the prey this tick."""
    static = _world()
    a, states = _run(static, _state(static, [(6, 9)], energy=0.0, health=1e-3), [STAY])
    assert not bool(states[1].prey_alive[0]), "precondition: the prey must die this tick"
    assert (a["fid_encounters"], a["fid_fled"], a["fid_unfled_closed"]) == (1, 0, 1)
    assert not bool(states[1].prey_enc_open[0]), "a dead prey's encounter must be closed"


def test_an_encounter_still_open_at_the_end_is_censored():
    static = _world()
    a, states = _run(static, _state(static, [(6, 9)]), [STAY, STAY, STAY])
    censored = a["fid_encounters"] - a["fid_fled"] - a["fid_unfled_closed"]
    assert censored == 1 and bool(states[-1].prey_enc_open[0]) and not bool(states[-1].prey_enc_fled[0])
    assert int(states[-1].prey_enc_age[0]) == 3


def test_an_encounter_is_counted_once_however_long_it_lasts():
    static = _world()
    a, _ = _run(static, _state(static, [(6, 9)]), [STAY] * 6)
    assert a["fid_encounters"] == 1


def test_the_encounter_state_starts_closed_every_episode():
    """Invariant I3's shape for the new state: nothing carries across a reset."""
    st = E.reset(ent.PRNGSource(jax.random.key(4)), _world(), P)
    assert not np.any(np.asarray(st.prey_enc_open)) and not np.any(np.asarray(st.prey_enc_fled))
    assert not np.any(np.asarray(st.prey_enc_age))


def test_the_encounter_state_never_reaches_the_policy():
    """The observation is a function of positions, pools and grids; the encounter fields must not change it."""
    static = _world()
    st = _state(static, [(6, 9)])
    flipped = st._replace(
        prey_enc_open=~st.prey_enc_open, prey_enc_fled=~st.prey_enc_fled, prey_enc_age=st.prey_enc_age + 5
    )
    assert _observation_vector(static, st).tolist() == _observation_vector(static, flipped).tolist()


def _observation_vector(static, st):
    from src.env.observation import build_observation

    i = 0
    return np.asarray(
        build_observation(
            static.obstacles,
            st.food_grid,
            st.target_grid,
            static.vis_table,
            st.prey_pos[i],
            st.prey_posture[i],
            jnp.stack([st.prey_energy[i], st.prey_stamina[i], st.prey_health[i]]),
            st.prey_pos,
            st.prey_alive,
            st.prey_moved,
            st.prey_posture,
            st.pred_pos,
            jnp.zeros((st.pred_pos.shape[0],), bool),
            st.pred_appearance,
            i,
            P,
        )
    )


def _observation_predator_cells(static, st) -> int:
    v = _observation_vector(static, st)
    n = P.s_patch * P.s_patch
    return int((v[: n * P.c_channels].reshape(n, P.c_channels)[:, CH_PREDATOR] > 0).sum())


# ------------------------------------------------------------- identities over real episodes --

Q = DEFAULT.__class__(h=20, w=20, k=8, t_horizon=150, n_pred=2, p_wander=1.0)


def _random_walk_policy(key):
    def policy(weights, obs, hidden, step_index):
        keys = jax.random.split(jax.random.fold_in(key, step_index), obs.shape[0])
        # Uniform over the eight moves and eight sprints: every tick is directional, so the test has the
        # most ticks on which the chance correction can be wrong.
        return jax.vmap(lambda k: baseline.random_policy(k, 16))(keys), hidden

    return policy


def _episodes(params, n_worlds=6, n_eps=24, policy_key=7):
    """Every world and every episode in ONE compiled call: the worlds are stacked and vmapped like the keys,
    so the fixture compiles once rather than once per world."""
    w = net.init_params(jax.random.key(5), params)
    pol = _random_walk_policy(jax.random.key(policy_key))
    statics = []
    for s in range(n_worlds):
        ob = grid.make_obstacle_grid(jax.random.key(100 + s), params.h, params.w, params.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(200 + s), ob, params)
        statics.append(
            E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, params.r), food_src=fs, target_src=ts)
        )
    stacked = jax.tree.map(lambda *x: jnp.stack(x), *statics)
    keys = jax.random.split(jax.random.key(300), n_worlds * n_eps).reshape(n_worlds, n_eps)
    one = lambda st, k: E.run_episode(k, w, st, ent.PRNGSource(k), params, policy=pol)[1]  # noqa: E731
    acc = jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, 0)))(stacked, keys)
    return {f: np.asarray(getattr(acc, f), np.float64).reshape(-1) for f in acc._fields}


@pytest.fixture(scope="module")
def walk():
    return _episodes(Q)


def test_identities_hold_for_every_prey_in_real_episodes(walk):
    a = walk
    parts = a["dg_contact_ticks"] + a["dg_perc_ticks"] + a["dg_occl_ticks"] + a["dg_far_ticks"]
    np.testing.assert_array_equal(parts, a["pred_present_ticks"], err_msg="the D-065 partition does not hold")
    np.testing.assert_array_equal(a["dg_a_ticks"] + a["dg_b_ticks"], a["dg_perc_ticks"])
    np.testing.assert_allclose(
        a["dg_a_excess_sum"] + a["dg_b_excess_sum"], a["dg_perc_excess_sum"], atol=1e-4
    )
    assert np.all(a["pred_present_ticks"] <= a["alive_ticks"])
    assert np.all(a["dg_perc_moves"] <= a["dg_perc_ticks"])
    assert np.all(a["dg_perc_exec_sum"] <= a["dg_perc_max_sum"] + 1e-4), (
        "an executed change beat the best available"
    )
    censored = a["fid_encounters"] - a["fid_fled"] - a["fid_unfled_closed"]
    assert np.all((censored == 0) | (censored == 1)), "more than one encounter open at the horizon"
    assert np.all(a["fid_latency_sum"] <= Q.t_horizon), "latencies of disjoint encounters exceed the horizon"
    assert np.all(a["fid_dist_sum"] <= Q.r * a["fid_fled"]), "a flight was initiated beyond sight"
    assert np.all(a["fid_onset_sum"] <= Q.r * a["fid_encounters"]), "an encounter opened beyond sight"


def test_the_identities_were_exercised_not_vacuous(walk):
    a = walk
    for name in ("dg_a_ticks", "dg_b_ticks", "dg_occl_ticks", "dg_far_ticks", "dg_contact_ticks", "fid_fled"):
        assert a[name].sum() > 0, f"{name} never occurred, so the identities above say nothing about it"
    assert (a["fid_unfled_closed"] > 0).any(), "no encounter ever closed unfled"
    # **Not required here: an encounter still open and unfled at the horizon.** A random walker moves away
    # within a tick or two of almost every sighting, so in one measured sample of 1152 prey none was left
    # open, and in another one was. Requiring it would make this test pass or fail on the sample's luck.
    # That path is covered deterministically by `test_an_encounter_still_open_at_the_end_is_censored` and
    # by the randomized differential's coverage assertion, `left_open_unfled`.


def test_a_predator_blind_walker_shows_the_raw_bias_and_no_corrected_bias(walk):
    """**The bias test.** A random walker knows nothing of the predator. Its chance-corrected change must be
    zero within sampling error, and its RAW change must be detectably positive, which is the geometric bias
    that made the correction necessary. Bootstrapped over prey, since the prey is the sampling unit."""
    a = walk
    has = a["dg_perc_ticks"] > 0
    ticks, raw, exc = a["dg_perc_ticks"][has], a["dg_perc_emit_sum"][has], a["dg_perc_excess_sum"][has]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ticks), (4000, len(ticks)))
    raw_b = raw[idx].sum(1) / ticks[idx].sum(1)
    exc_b = exc[idx].sum(1) / ticks[idx].sum(1)
    lo_raw = np.quantile(raw_b, 0.0005)
    lo_exc, hi_exc = np.quantile(exc_b, [0.0005, 0.9995])
    assert ticks.sum() > 2000, f"too few perceived ticks for the test to have power: {ticks.sum()}"
    assert lo_raw > 0, f"the raw bias was expected and is absent: 99.9% interval lower bound {lo_raw:.4f}"
    assert lo_exc < 0 < hi_exc, (
        f"the corrected measure is biased for a blind walker: [{lo_exc:.4f}, {hi_exc:.4f}]"
    )
