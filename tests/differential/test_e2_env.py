"""Gate E2 over the episode loop: the twelve-step tick order, compared tick by tick.

**What this gate adds that the per-subsystem differentials do not is the ORDER.** Every piece already has its
own E2 coverage. A tick order that resolves capture before movement, or reads a stale `prey_moved`, composes
correct pieces into a simulation that runs perfectly and answers a different question. Invariant I5 is exactly
that ordering, and it is not visible from any single subsystem.

The initial state is built once and injected into both implementations, because `reset` uses `jax.random`
draws that NumPy cannot reproduce. That is the same convention as every other differential here: what is
compared is the per-tick dynamics, driven by identical pre-generated draws indexed by tag and tick.

**Replay runs inside a compiled scan on the JAX side and inside a plain Python loop on the oracle side.** If
the tick index is threaded wrongly through the scan carry, the two disagree, which is the failure the tagged
entropy design exists to prevent and which nothing before step 7 could have detected.
"""

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

P = DEFAULT.__class__(h=14, w=14, k=4, t_horizon=60, n_food_src=3, n_target_src=3)
N_TICKS = P.t_horizon

SITES = {
    "perceive.distant_detect": (N_TICKS, P.k),
    "regrow.food.trigger": (N_TICKS, P.n_food_src),
    "regrow.food.cell": (N_TICKS, P.n_food_src),
    "regrow.target.trigger": (N_TICKS, P.n_target_src),
    "regrow.target.cell": (N_TICKS, P.n_target_src),
}


def sites_for(n_pred: int) -> dict:
    """`SITES` plus the wander draws, whose shape follows the arm's predator count.

    **A fixed predator axis was wrong and the reshape said so immediately.** `env.py` draws
    `(params.n_pred,)`, so a replay source holding four columns cannot serve an arm running one. The draw
    COUNT still does not depend on the state of the world within an arm, which is what DEV-022 requires; it
    depends on the arm's shape, which is static.
    """
    return {
        **SITES,
        # Exactly `n_pred`, zero included. A `max(n_pred, 1)` here gave a wander axis of one against a
        # predator axis of zero and vmap said so at once: the empty predator axis has to stay empty
        # everywhere, which is the property that makes `n_pred = 0` a no-op by construction rather than a
        # special case.
        "pred.wander.trigger": (N_TICKS, n_pred),
        "pred.wander.dir": (N_TICKS, n_pred),
    }


def build_scenario(n_pred: int):
    """One shared world and one shared starting state, handed to both implementations.

    **A module-level function rather than only a fixture, so a diagnostic can construct the same scenario by
    CALLING this rather than by reproducing it.** Duplicated setup degrades toward the sloppier copy, and the
    copy is always the one written under time pressure; the tolerance below was set from a measurement that
    would have had to rebuild this by hand.

    **Parametrised over the predator count, because one predator was the only case E2 had ever witnessed.**
    `no_predator_evolve` requires `n_pred = 0` and the proposed sweep axis requires more than one, and the
    oracle was written for exactly one until step 8. Both DEV-016 fixes, the draw without replacement and the
    ladder's one-cell feasibility test, live in that code and had no independent implementation to disagree
    with them. That is the shape DEV-014 had.

    Zero is included deliberately rather than as a curiosity: it is the condition the null depends on, so an
    unwitnessed difference there would move every emergence threshold.

    **`reset` reads `params.n_pred` TWICE**, once for the draw shape and once for the ladder's feasibility
    test, so it needs a matching params object. An earlier version of this fixture reset with the shared `P`
    at `n_pred = 1` and then replaced the predator arrays, which produced a state whose recorded
    `spawn_relax_level` described a one-predator world that the scenario did not contain. Nothing noticed
    until that field started being compared, which is DEV-015 paying for itself a second time.

    `tick` does take the count from the array shapes, so it is still handed the shared `P`.
    """
    from dataclasses import replace as _replace

    PP = _replace(P, n_pred=n_pred)
    key = jax.random.key(3)
    obstacles = jnp.zeros((P.h, P.w), bool).at[4, 4].set(True).at[9, 9].set(True).at[4, 9].set(True)
    food_src, target_src, _ = resources.place_sources(key, obstacles, P)
    static = E.StaticEnv(
        obstacles=obstacles,
        vis_table=build_vis_table(obstacles, P.r),
        food_src=food_src,
        target_src=target_src,
    )
    # Place the predator beside a prey. The natural reset produces no predation at all in a short episode,
    # measured at 0 targeted ticks in 30, which made the capture path inert and let a stale prey_moved break
    # survive falsification. The comparison must exercise what it claims to compare.
    state = E.reset(PRNGSource(jax.random.key(5)), static, PP)
    _t = state.prey_pos[0]
    # Predators sit at increasing offsets from prey 0, so two predators are two THREATS rather than a stack
    # on one cell, which is the distinction DEV-016 exists for. At zero this is an empty array of the right
    # shape, which is what makes every reduction over the predator axis a no-op by construction.
    # **Placed inside `r_near`, where the two-regime rule reports regardless of motion or the detection
    # draw.** They previously sat at `r_pred`, outside `r_near`, so target acquisition depended on the prey
    # happening to move or on a 25 percent draw. An untrained policy drifts steadily in one direction, so the
    # prey walked out of range and the predator, never having acquired a target, never pursued. That held only
    # while `reset` happened to produce compatible geometry, and DEV-022 changed the positions.
    #
    # Two distinct cells both at Chebyshev distance `r_near`: one along the column, one along the row. The
    # direction is chosen to stay inside the grid, because clipping at an edge would silently shorten the
    # offset and could place a predator ON a prey.
    _dc = P.r_near if int(_t[1]) + P.r_near < P.w else -P.r_near
    _dr = P.r_near if int(_t[0]) + P.r_near < P.h else -P.r_near
    _offsets = [(0, _dc), (_dr, 0)]
    _placed = (
        jnp.stack(
            [jnp.array([_t[0] + _offsets[j][0], _t[1] + _offsets[j][1]], jnp.int32) for j in range(n_pred)]
        )
        if n_pred
        else jnp.zeros((0, 2), jnp.int32)
    )
    # Only the POSITIONS are overridden. Everything else comes from `reset` at the right shape, so
    # `spawn_relax_level` remains the level reset actually computed for THIS predator count. The ladder's
    # feasibility depends on prey positions and the obstacle grid, not on where the predator is put, so
    # moving the predator afterwards does not invalidate it.
    state = state._replace(pred_pos=_placed)

    # **The precondition, asserted rather than assumed.** A fixture whose validity depends on a property must
    # check that property. Outside this window every predator-conditioned quantity reads zero and the
    # comparison agrees over paths that never ran, which is the failure the coverage assertions exist for.
    if n_pred:
        _d = jnp.max(jnp.abs(state.pred_pos[:, None, :] - state.prey_pos[None, :, :]), axis=-1)
        assert 1 < int(jnp.min(_d)) <= P.r_near, (
            f"constructed predators sit {int(jnp.min(_d))} from the nearest prey; they must be within "
            f"r_near {P.r_near} so reporting is deterministic, and beyond adjacency so engagement accumulates"
        )
        assert len({(int(a), int(b)) for a, b in state.pred_pos}) == n_pred, (
            "constructed predators share a cell, so this scenario does not exercise distinct threats"
        )
    weights = net.init_params(jax.random.key(6), P)

    # Perturb every leaf so no bias path is inert. Falsification found that comparing at initialisation
    # leaves the bias paths dead, since init_params zeroes them.
    pert = jax.random.split(jax.random.key(77), len(weights))
    weights = {
        k: v + 0.3 * jax.random.normal(pert[i], v.shape, v.dtype)
        for i, (k, v) in enumerate(sorted(weights.items()))
    }
    return static, state, weights, PP


@pytest.fixture(scope="module", params=[0, 1, 2], ids=["n_pred0", "n_pred1", "n_pred2"])
def scenario(request):
    """Thin wrapper. The construction lives in `build_scenario` so it has exactly one implementation."""
    return build_scenario(request.param)


def _to_numpy_state(s: E.EnvState) -> dict:
    return {f: np.asarray(getattr(s, f)) for f in s._fields}


# ---------------------------------------------------------------- DEV-015, D-036 --------
# **Every `EnvState` field is compared by iteration, and the exclusions are named in TWO KINDS.**
#
# The previous form was a written-out list of nine assertions covering twenty fields, so eleven were silently
# uncompared and nothing said so. That is how a difference in `prey_hidden` sat unnoticed: it was never
# looked at.
#
#   UNCOMPARABLE   not comparable in principle. A genuine decision, recorded with its reason
#   UNMODELLED     the oracle does not implement it. DEBT, and asserted as an exact set below so it can
#                  only shrink deliberately
#
# Collapsing the two would launder debt into a documented exclusion, and a documented gap stops being read
# as work.
# **Empty, as of the oracle growing a `reset_derived`.** `spawn_relax_level` was the single entry: the
# oracle implemented no reset, so it never produced a relaxation level and the field could not be compared.
# It is now recomputed independently from the injected positions and the obstacle grid, and the two agree.
#
# Keeping the set rather than deleting it is deliberate. It is the ratchet from D-036: a field the oracle
# does not model fails the exact-set assertion below until someone models it or adds it here on purpose, so
# debt can only reappear by decision.
UNMODELLED: frozenset[str] = frozenset()

# Per-field float tolerance. Absent entries use `P.e2_atol` and `P.e2_rtol`.
#
# **`prey_hidden` needs its own, and the reason is a deliberate design choice rather than a defect.** The
# oracle runs the network in float64 and the implementation in float32, which `test_e2_agent.py` already
# accommodates at 1e-4 for a single forward pass on synthetic input. Inside the full environment the
# OBSERVATION is also computed through both precisions, so the gap is larger. Measured across 30 ticks:
#
#     tick 0   1.09e-03      already above e2_atol before any accumulation
#     tick 6   3.63e-03      peak
#     tick 29  2.15e-03      saturated, not growing
#
# **Re-measured across the predator counts the gate actually runs, after the fixture was parametrised.** The
# original 5e-3 came from the 3.63e-3 peak above, measured at n_pred = 1 alone, and n_pred = 2 then exceeded
# it at 5.58e-3. **A tolerance set from a measurement that does not span the gate's own parameter range is
# the unasserted-precondition problem in a different costume**, and the fix is to measure the range rather
# than to raise the number until the test passes.
#
#     n_pred 0   peak 3.611e-03    actions identical
#     n_pred 1   peak 4.541e-03    actions identical
#     n_pred 2   peak 5.584e-03    actions identical
#
# The gap GROWS with predator count, because more predators mean more computation feeding the observation and
# so a larger float32-against-float64 divergence. It does not grow within an episode.
#
# **So the tolerance is DERIVED from the predator count rather than being a literal**, because a literal set
# at one point on an axis breaks when that axis sweeps. That is the coupling class, and it is the same shape
# as `E_MIN` against predator speed and the spawn floor against `r_pred`. A comment saying "re-measure if the
# fixture goes higher" depends on someone reading it at the moment the fixture goes higher, which is exactly
# what deriving removes.
#
# The literal 8e-3 was already known to fail at the proposed axis's midpoint:
#
#     projected n_pred=4    7.56e-3   within 8e-3
#     projected n_pred=6    9.53e-3   EXCEEDS
#     projected n_pred=8   11.50e-3   EXCEEDS
#
# It SATURATES rather than diverging, which is what contractive tanh and sigmoid gates should do, and the
# emitted actions were identical on every tick. Reduced-precision matmul was ruled out by measurement:
# `precision=None` and `precision="highest"` both give 7.3e-06 on a 1140-term dot, and NumPy float32 gives
# 1.5e-05, so the matmul path is not the cause.
#
# **The tolerance is set from the measured peak with margin, not chosen to make the test pass**, and it is
# tight enough that genuine divergence would still fail: an implementation difference would grow past this
# within a few ticks rather than settling under it.
# **The load-bearing evidence that this gap is benign is NOT that it saturates.** A structural difference
# could saturate too. What cannot survive a structural difference is the action sequence, and the emitted and
# executed actions are compared as EXACT integer lists above, not within any tolerance. If those ever differ,
# this tolerance stops mattering and the comparison has failed regardless of what the hidden state does.
#: Fitted to the three measured peaks: 3.611, 4.541 and 5.584 in units of 1e-3 at n_pred 0, 1 and 2. The
#: slope is 0.99e-3 per predator and the intercept 3.61e-3, and 1.38 is the margin factor the original
#: tolerance used, carried forward rather than re-chosen.
HIDDEN_INTERCEPT, HIDDEN_SLOPE, HIDDEN_MARGIN = 3.61e-3, 0.99e-3, 1.38


def tolerance_for(name: str, n_pred: int) -> tuple[float, float]:
    """Per-field float tolerance. Everything not named here uses the suite-wide values.

    `prey_hidden` is the only field needing its own, and it needs one that MOVES with the predator count.
    """
    if name != "prey_hidden":
        return (P.e2_atol, P.e2_rtol)
    tol = (HIDDEN_INTERCEPT + HIDDEN_SLOPE * n_pred) * HIDDEN_MARGIN
    return (tol, tol)


@pytest.mark.slow  # 46s and 41s measured across n_pred 0 and 1: a full episode compared tick by tick against the oracle
def test_the_episode_matches_the_oracle_tick_by_tick(scenario):
    """Every field, every tick, so a divergence is reported where it starts rather than at the horizon."""
    static, state_j, weights_j, PP = scenario
    weights_n = {k: np.asarray(v, np.float64) for k, v in weights_j.items()}

    ent_j = make_replay_source(jax.random.key(21), sites_for(PP.n_pred))
    # The oracle uses its own NumPy ReplayReader rather than the JAX ReplaySource, so the two replay paths
    # are independent implementations of the same indexing convention. Same draws, separate readers.
    ent_n = oracle.ReplayReader({k: np.asarray(v) for k, v in ent_j.draws.items()})

    # **The oracle DERIVES its starting state rather than copying the implementation's.** This is the fork
    # moving upstream, per DEV-021: anything copied is shared driver state and invisible to the comparison,
    # however complete the field coverage. Positions are still injected, because `reset` calls `jax.random`
    # directly and a NumPy oracle cannot reproduce threefry, which is DEV-022. So the fork moves by the size
    # of the derived quantities and not by the positions, and `spawn_relax_level` in particular is now
    # independently computed on both sides instead of being uncomparable.
    state_n = oracle.reset_derived(
        np.asarray(state_j.prey_pos),
        np.asarray(state_j.pred_pos),
        np.asarray(static.obstacles),
        np.asarray(state_j.food_grid),
        np.asarray(state_j.target_grid),
        np.asarray(state_j.pred_appearance),
        PP,
    )
    acc_j = E.zero_accumulators(P.k)
    acc_n = oracle.zero_accumulators(P.k)

    for t in range(N_TICKS):
        state_j, acc_j, rec_j = E.tick(state_j, acc_j, weights_j, static, ent_j, jnp.int32(t), PP)
        state_n, acc_n, rec_n = oracle.tick(
            state_n,
            acc_n,
            weights_n,
            np.asarray(static.obstacles),
            np.asarray(static.food_src),
            np.asarray(static.target_src),
            ent_n,
            t,
            P,
        )

        where = f"tick {t}"

        # **The whole record, and its field sets asserted equal, per queue item E.**
        #
        # This was two hand-written assertions on the two action fields. That is the same whitelist
        # shape the state comparison below guards against with `UNMODELLED` and the accumulator
        # comparison guards against with set equality: a loop over what one side happens to carry
        # LOOKS exhaustive while skipping whatever the other side lacks. The oracle carried four of
        # the record's nine fields, so `prey_pos`, `prey_alive`, `prey_posture`, `prey_moved` and
        # `pred_pos` were never compared at all.
        #
        # **`prey_pos` and `pred_pos` are what the trajectory tier consumes as ground truth**, which
        # is what makes this worth more than tidiness: run 18's grouping measurement and run 25's
        # capture-window measurement both read those positions, and until now nothing had checked
        # them against an independent implementation.
        assert set(rec_j._fields) == set(rec_n), (
            f"record field sets differ. only in implementation: {sorted(set(rec_j._fields) - set(rec_n))}; "
            f"only in oracle: {sorted(set(rec_n) - set(rec_j._fields))}. A field on one side only is a "
            "field nothing compares."
        )
        for name in rec_j._fields:
            a = np.asarray(getattr(rec_j, name))
            b = np.asarray(rec_n[name])
            if a.dtype.kind == "f":
                np.testing.assert_allclose(a, b, atol=1e-6, rtol=0, err_msg=f"record {name}: {where}")
            else:
                np.testing.assert_array_equal(a, b, err_msg=f"record {name}: {where}")
        for name in state_j._fields:
            if name in UNMODELLED:
                continue
            a = np.asarray(getattr(state_j, name))
            b = np.asarray(state_n[name])
            atol, rtol = tolerance_for(name, int(state_j.pred_pos.shape[0]))
            if a.dtype.kind == "f":
                np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=f"{name}: {where}")
            else:
                np.testing.assert_array_equal(a, b, err_msg=f"{name}: {where}")

    # **The comparison iterates the ORACLE's keys, so a field the oracle does not model is silently
    # skipped rather than compared.** That is DEV-015's whitelist failure in a different disguise: the set
    # looks exhaustive because it is a loop. Asserting the two field sets are equal is what makes adding an
    # accumulator to one side fail until it is added to the other.
    # D-036: the debt set is asserted to EQUAL an explicit list, not bounded by a count. Adding a field the
    # oracle does not model fails here until someone models it or adds it deliberately, so the debt can only
    # shrink on purpose.
    missing = {f for f in state_j._fields if f not in state_n}
    assert missing == set(UNMODELLED), (
        f"the set of state fields the oracle does not model has changed. now missing: {sorted(missing)}; "
        f"declared UNMODELLED: {sorted(UNMODELLED)}. Model it, or add it to UNMODELLED deliberately."
    )

    assert set(acc_n) == set(acc_j._fields), (
        f"accumulator sets differ. only in implementation: {sorted(set(acc_j._fields) - set(acc_n))}; "
        f"only in oracle: {sorted(set(acc_n) - set(acc_j._fields))}"
    )
    for name in acc_n:
        np.testing.assert_array_equal(
            np.asarray(getattr(acc_j, name)), acc_n[name], err_msg=f"accumulator {name}"
        )


@pytest.mark.slow  # 44s, 41s and 38s measured across n_pred 0 to 2: the same episode, re-walked for coverage
def test_coverage_the_episode_actually_exercised_its_paths(scenario):
    """An agreement test over paths that never execute agrees trivially. Gate E3 shipped in that state.

    **At `n_pred = 0` the predator-side assertions are inverted rather than skipped.** Skipping would leave
    the predator-free case with no coverage claim at all, which is how a condition ends up unwitnessed. The
    honest assertion there is the negative one: nothing predation-related may fire, and every reduction over
    the empty predator axis must produce exactly zero. That is a stronger statement than the positive form,
    because a stray non-zero would mean the empty axis is not the no-op the design relies on.
    """
    static, state_j, weights_j, PP = scenario
    ent = make_replay_source(jax.random.key(21), sites_for(PP.n_pred))
    state = state_j
    acc = E.zero_accumulators(P.k)
    n_pred = int(state_j.pred_pos.shape[0])
    saw_move = saw_report = saw_mask_differ = False

    for t in range(N_TICKS):
        state, acc, rec = E.tick(state, acc, weights_j, static, ent, jnp.int32(t), PP)
        saw_move |= bool(jnp.any(state.prey_moved))
        saw_mask_differ |= bool(jnp.any(rec.action_emitted != rec.action_executed))
        saw_report |= bool(jnp.any(state.pred_target != -1))

    assert saw_move, "no prey ever moved, so the movement path is untested"
    assert saw_mask_differ, (
        "emitted and executed never differed across the episode, so the mask path is untested and the "
        "distinction D-031 depends on is not being exercised"
    )
    assert int(jnp.sum(acc.alive_ticks)) > 0, "no agent was ever alive"

    if n_pred == 0:
        # The predator-free world, asserted negatively. Every one of these is a reduction over an axis of
        # length zero, and the whole reason `no_predator_evolve` needs no masking is that they collapse.
        #
        # **This assertion is incomplete on its own and is one half of a pair. Do not simplify either half.**
        # `engaged_ticks == 0` would pass just as well if the field had been DELETED, because absent and zero
        # are indistinguishable to a sum over a missing name. What rules that out is the field-set equality
        # check in `test_the_episode_matches_the_oracle_tick_by_tick`, which fails if either side is missing a
        # field at all.
        #
        #   this assertion          the field exists and is correctly zero
        #   field-set equality      the field exists on both sides
        #
        # Neither is sufficient alone; together they are. Someone weakening one would silently weaken the
        # other, and nothing in either would say so, which is why the dependency is written here.
        assert not saw_report, "a target was acquired with no predators present"
        for name in ("engaged_ticks", "visible_ticks", "near_ticks", "still_engaged", "accidental_still"):
            assert int(jnp.sum(getattr(acc, name))) == 0, (
                f"{name} is non-zero with no predators present, so the empty predator axis is not the no-op "
                "the predator-naive lineage depends on"
            )
        assert int(jnp.sum(acc.alive_ticks)) == P.k * N_TICKS, (
            "a prey died with no predator present, which starvation cannot do inside this short comparison"
        )
        return

    assert saw_report, "the predator never acquired a target, so the pursuit path is untested"
    # Capture, not engagement. See Q-021: `engaged` requires closing and adjacency simultaneously and may
    # be unsatisfiable in practice, so it cannot serve as a coverage floor.
    assert int(jnp.sum(acc.near_ticks)) > 0, (
        "no predator ever came within r_pred, so capture, occlusion and the distance descriptors are all "
        "measured over paths that did not execute"
    )


def test_invariant_i5_capture_reads_movement_from_the_same_tick(scenario):
    """The ordering assertion, and it is what this gate exists for beyond the per-subsystem ones.

    A predator adjacent to a prey that moved this tick must deal damage this tick. If capture read a stale
    `prey_moved` from the previous tick, damage would lag by one and every feign would be mistimed, while
    every subsystem stayed individually correct.
    """
    from src.env import predator as pred_mod

    obstacles = jnp.zeros((12, 12), bool)
    vis = build_vis_table(obstacles, P.r_pred)

    moved_now = pred_mod.step_predator(
        obstacles,
        vis,
        jnp.array([6, 6]),
        jnp.asarray(100.0),
        jnp.int32(0),
        jnp.asarray(0.0, jnp.float32),
        jnp.int32(0),
        jnp.array([[6, 7]]),
        jnp.array([True]),
        jnp.array([True]),
        jnp.array([0.0]),
        jnp.asarray(1.0),  # wander_trigger: never fires, so this stays a test of capture timing only
        jnp.asarray(0.0),  # wander_dir
        P,
    )
    still_now = pred_mod.step_predator(
        obstacles,
        vis,
        jnp.array([6, 6]),
        jnp.asarray(100.0),
        jnp.int32(0),
        jnp.asarray(0.0, jnp.float32),
        jnp.int32(0),
        jnp.array([[6, 7]]),
        jnp.array([True]),
        jnp.array([False]),
        jnp.array([0.0]),
        jnp.asarray(1.0),  # wander_trigger: never fires, so this stays a test of capture timing only
        jnp.asarray(0.0),  # wander_dir
        P,
    )
    assert float(moved_now["damage_per_prey"][0]) > 0.0, "a prey that moved took no damage"
    assert float(still_now["damage_per_prey"][0]) == 0.0, (
        "a still prey took damage at cue_dependence 1.0, so capture is not reading this tick's movement"
    )
