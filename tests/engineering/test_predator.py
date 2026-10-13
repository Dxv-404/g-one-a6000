"""The scripted predator: cue dependence, capture, pursuit, and disengagement.

**The assertion that matters most in this file is that `cue_dependence` 1.0 reproduces the original
motion-gating rule exactly.** Every gate written before step 5 assumed that rule. If the parameterisation
does not reproduce it, the parameterisation is wrong rather than the old rule being wrong, and the whole
suite's relationship to the specification quietly breaks.

**Falsification is at both endpoints, never only the default.** A break caught at CUE=1.0 and passing at
CUE=0.0 means the test is exercising the cue path rather than the damage path, or the reverse. Tests here
are parameterised over both endpoints wherever the property should hold at both, and assert the
*difference* between endpoints wherever it should not.
"""

import jax
import jax.numpy as jnp
import pytest

from configs.default import DEFAULT as P
from src.env.predator import (
    NO_TARGET,
    chebyshev,
    resolve_capture,
    select_target,
    step_predator,
    update_engagement,
)

CUE_ENDPOINTS = [0.0, 1.0]


def cfg(**kw):
    """A config copy with overrides. `dataclasses.replace` keeps the derived properties consistent."""
    import dataclasses

    return dataclasses.replace(P, **kw)


# --------------------------------------------------------------- the equivalence check --


def _old_rule_damage(adjacent, moved):
    """The original hardcoded rule, written out independently rather than derived from the new one.

    Deliberately a separate implementation. Deriving it from the parameterised form would make this a
    tautology: it would compare the new rule against itself with CUE substituted, and pass no matter what.
    """
    return P.damage if (adjacent and moved) else 0.0


def _old_rule_still(adjacent, moved, still):
    """The original counter: reset on motion while adjacent, increment while adjacent and still, else hold."""
    if adjacent and moved:
        return 0
    if adjacent:
        return still + 1
    return still


@pytest.mark.parametrize("adjacent", [True, False])
@pytest.mark.parametrize("moved", [True, False])
@pytest.mark.parametrize("still_in", [0.0, 1.0, 4.0])
def test_cue_one_reproduces_the_original_rule_exactly(adjacent, moved, still_in):
    """CUE=1.0 must equal the hardcoded rule on damage and on the counter, over the full truth table.

    This is the check the revised specification names as the one that matters most, and it is exhaustive
    rather than sampled because the input space is twelve cases.
    """
    params = cfg(cue_dependence=1.0)
    damage, still_out, adj = resolve_capture(
        pred_pos=jnp.array([5, 5]),
        target_pos=jnp.array([5, 6]) if adjacent else jnp.array([5, 9]),
        target_moved=jnp.asarray(moved),
        still=jnp.asarray(still_in, jnp.float32),
        has_target=jnp.asarray(True),
        params=params,
    )

    assert bool(adj) == adjacent, "adjacency itself disagrees, so the rest of the comparison is meaningless"
    assert float(damage) == pytest.approx(_old_rule_damage(adjacent, moved)), (
        f"damage differs from the original rule at adjacent={adjacent}, moved={moved}"
    )
    assert float(still_out) == pytest.approx(_old_rule_still(adjacent, moved, still_in)), (
        f"the disengage counter differs at adjacent={adjacent}, moved={moved}, still={still_in}"
    )


def _disengage_tick(params, max_ticks=40):
    """Drive `step_predator` with a still, adjacent prey and return the tick pursuit actually broke.

    **This observes the predator's own decision rather than recomputing the threshold.** The first version of
    these tests applied `still > disengage_steps` in the test body, which is the duplication class from
    CLAUDE.md section 11: the check validated a parallel copy of the rule, so flipping the real comparison
    from `>` to `>=` changed the predator's behaviour and left every assertion passing. The falsification run
    caught it, and this helper is the structural fix, one implementation consulted by both.

    The prey never moves and sits adjacent, so it stays inside `r_near` and is reported every tick under the
    two-regime rule. Nothing but the disengage threshold can end the pursuit.
    """
    obstacles = jnp.zeros((12, 12), bool)
    vis = jnp.ones((12, 12, 2 * P.r_pred + 1, 2 * P.r_pred + 1), bool)
    state = dict(
        pos=jnp.array([6, 6]),
        target=jnp.int32(0),
        still=jnp.asarray(0.0, jnp.float32),
        engage=jnp.int32(0),
    )
    for tick in range(1, max_ticks + 1):
        out = step_predator(
            obstacles,
            vis,
            state["pos"],
            jnp.asarray(100.0),
            state["target"],
            state["still"],
            state["engage"],
            prey_pos=jnp.array([[6, 7]]),
            prey_alive=jnp.array([True]),
            prey_moved=jnp.array([False]),
            distant_draws=jnp.array([0.99]),
            wander_trigger=jnp.asarray(1.0),
            wander_dir=jnp.asarray(0.0),
            params=params,
        )
        state = dict(
            pos=out["pred_pos"],
            target=out["target"],
            still=out["still"],
            engage=out["engage"],
        )
        if int(out["target"]) == NO_TARGET:
            return tick
    return None


def test_cue_one_disengages_on_the_same_tick_as_the_original():
    """Timing, not just per-tick values, and read from the predator rather than recomputed.

    The original rule breaks pursuit when the counter *exceeds* disengage_steps, so with a threshold of 5 the
    sixth consecutive still tick is the one that ends it. An off-by-one here would shift every feign's cost
    by a tick, which is small enough to go unnoticed and large enough to move a descriptor.
    """
    params = cfg(cue_dependence=1.0)
    assert _disengage_tick(params) == params.disengage_steps + 1, (
        f"pursuit must break on tick {params.disengage_steps + 1}, the first tick the counter exceeds "
        f"disengage_steps of {params.disengage_steps}"
    )


# ------------------------------------------------------------------- the cue endpoints --


def test_cue_zero_makes_stillness_confer_nothing():
    """The motion-blind endpoint. A still prey takes full damage and pursuit never breaks.

    This is the endpoint that proves stillness is not a law of the environment. If this test ever fails by
    a still prey taking reduced damage at CUE=0.0, something has hardcoded a payoff for stillness, which is
    invariant I2.
    """
    params = cfg(cue_dependence=0.0)
    damage, still, _ = resolve_capture(
        jnp.array([5, 5]),
        jnp.array([5, 6]),
        jnp.asarray(False),
        jnp.asarray(0.0, jnp.float32),
        jnp.asarray(True),
        params,
    )
    assert float(damage) == pytest.approx(params.damage), (
        "at cue_dependence 0.0 a still prey must take full damage; stillness confers nothing"
    )
    assert float(still) == pytest.approx(0.0), (
        "at cue_dependence 0.0 the counter must never accrue, so pursuit never breaks"
    )


@pytest.mark.parametrize("cue,expected_damage", [(1.0, 0.0), (0.5, 17.0), (0.0, 34.0)])
def test_still_prey_damage_interpolates_continuously(cue, expected_damage):
    """The worked table from the revised specification, asserted directly."""
    params = cfg(cue_dependence=cue)
    damage, _, _ = resolve_capture(
        jnp.array([5, 5]),
        jnp.array([5, 6]),
        jnp.asarray(False),
        jnp.asarray(0.0, jnp.float32),
        jnp.asarray(True),
        params,
    )
    assert float(damage) == pytest.approx(expected_damage), (
        f"still prey at cue {cue} should take {expected_damage}"
    )


@pytest.mark.parametrize("cue", [1.0, 0.5, 0.25, 0.0])
def test_a_moving_prey_always_takes_full_damage_at_every_cue(cue):
    """Cue dependence modulates the still case only. The moving case is the invariant end of the rule.

    A parameterisation that also scaled the moving case would change catch difficulty with the cue, which is
    precisely the confound the two-dimensional sweep exists to avoid.
    """
    params = cfg(cue_dependence=cue)
    damage, _, _ = resolve_capture(
        jnp.array([5, 5]),
        jnp.array([5, 6]),
        jnp.asarray(True),
        jnp.asarray(0.0, jnp.float32),
        jnp.asarray(True),
        params,
    )
    assert float(damage) == pytest.approx(params.damage), (
        f"a moving prey must take full damage at every cue; got {float(damage)} at cue {cue}"
    )


@pytest.mark.parametrize("cue", CUE_ENDPOINTS)
@pytest.mark.parametrize("moved", [True, False])
def test_a_distant_prey_is_never_damaged_at_any_cue(cue, moved):
    """The adjacency gate, asserted at both endpoints. See DEV-009.

    The revised specification's damage line omits adjacency. Read literally it would damage every prey on
    the grid. This is the test that enforces the retained gate, and it runs at both endpoints because at
    CUE=1.0 a still distant prey is zero for the wrong reason: the cue term already zeroes it, so only the
    moving case at CUE=1.0 and both cases at CUE=0.0 actually exercise adjacency.
    """
    params = cfg(cue_dependence=cue)
    damage, _, adjacent = resolve_capture(
        jnp.array([5, 5]),
        jnp.array([5, 12]),
        jnp.asarray(moved),
        jnp.asarray(0.0, jnp.float32),
        jnp.asarray(True),
        params,
    )
    assert not bool(adjacent), "the fixture is wrong: these positions are not distant"
    assert float(damage) == 0.0, (
        f"a prey seven cells away took {float(damage)} damage at cue {cue}, moved={moved}. The adjacency "
        "gate is missing, which is the literal reading of the revised spec that DEV-009 rejects"
    )


@pytest.mark.parametrize("cue,ticks", [(1.0, 6), (0.5, 11), (0.25, 21)])
def test_disengage_timing_scales_with_the_inverse_cue(cue, ticks):
    """The specification's disengage table: 1.0 breaks in 5 ticks of stillness, 0.5 in 10, 0.25 in 20.

    Read from the predator's own decision via `_disengage_tick`, not by reapplying the threshold here. See
    that helper for why: the earlier version duplicated the comparison and could not see it change.
    """
    assert _disengage_tick(cfg(cue_dependence=cue)) == ticks, (
        f"at cue {cue} pursuit should break on tick {ticks}"
    )


def test_cue_zero_never_breaks_pursuit():
    """The motion-blind endpoint has no disengagement at all, so a feign can never end a pursuit."""
    assert _disengage_tick(cfg(cue_dependence=0.0), max_ticks=60) is None, (
        "at cue_dependence 0.0 the counter never accrues, so pursuit must never break"
    )


# ------------------------------------------------------------------- target selection --


def test_target_persists_while_reported_even_if_another_is_nearer():
    """Persistence beats proximity, which is what makes pursuit a commitment rather than a reassignment."""
    reported = jnp.array([True, True])
    distances = jnp.array([5.0, 1.0])
    assert int(select_target(jnp.int32(0), reported, distances)) == 0, (
        "the predator switched to a nearer prey while its current target was still reported"
    )


def test_target_falls_back_to_nearest_when_the_current_one_is_lost():
    reported = jnp.array([False, True, True])
    distances = jnp.array([1.0, 6.0, 3.0])
    assert int(select_target(jnp.int32(0), reported, distances)) == 2, (
        "on losing its target the predator must take the nearest REPORTED prey, not the nearest prey"
    )


def test_no_target_when_nothing_is_reported():
    reported = jnp.array([False, False])
    distances = jnp.array([1.0, 2.0])
    assert int(select_target(jnp.int32(1), reported, distances)) == NO_TARGET


def test_a_predator_with_no_target_acquires_by_the_fallback():
    """Selection from NO_TARGET must go through the nearest-reported fallback.

    This test previously claimed to protect the clamp in `select_target` against -1 wrapping to the last
    prey. It does not, and falsification proved it: removing the clamp changes nothing, because
    `current != NO_TARGET` short-circuits the conjunction before the index is used. The claim was wrong, not
    the test, so the test keeps its assertions and drops the claim.
    """
    reported = jnp.array([True, False])
    distances = jnp.array([9.0, 1.0])
    assert int(select_target(jnp.int32(NO_TARGET), reported, distances)) == 0, (
        "with no current target the fallback should select the only reported prey"
    )
    reported_none = jnp.array([False, False])
    assert int(select_target(jnp.int32(NO_TARGET), reported_none, distances)) == NO_TARGET


# --------------------------------------------------------------------- engagement --


def test_engagement_accumulates_only_while_closing_on_the_same_target():
    e = update_engagement(jnp.int32(0), jnp.asarray(5.0), jnp.asarray(4.0), jnp.int32(1), jnp.int32(1))
    assert int(e) == 1, "closing on the same target must increment engagement"

    e = update_engagement(jnp.int32(3), jnp.asarray(4.0), jnp.asarray(5.0), jnp.int32(1), jnp.int32(1))
    assert int(e) == 0, "retreating must reset engagement"

    e = update_engagement(jnp.int32(3), jnp.asarray(5.0), jnp.asarray(4.0), jnp.int32(2), jnp.int32(1))
    assert int(e) == 0, "a retarget must not inherit the previous pursuit's engagement"


# ------------------------------------------------------------------------- pursuit --


def test_the_predator_holds_position_with_no_target():
    """No search behaviour. The spec gives none, and inventing one changes the control condition."""
    obstacles = jnp.zeros((12, 12), bool)
    vis = jnp.ones((12, 12, 2 * P.r_pred + 1, 2 * P.r_pred + 1), bool)
    out = step_predator(
        obstacles,
        vis,
        pred_pos=jnp.array([6, 6]),
        pred_stamina=jnp.asarray(100.0),
        target=jnp.int32(NO_TARGET),
        still=jnp.asarray(0.0, jnp.float32),
        engage=jnp.int32(0),
        prey_pos=jnp.array([[0, 0]]),
        prey_alive=jnp.array([False]),
        prey_moved=jnp.array([False]),
        distant_draws=jnp.array([0.99]),
        wander_trigger=jnp.asarray(1.0),
        wander_dir=jnp.asarray(0.0),
        params=P,
    )
    assert [int(x) for x in out["pred_pos"]] == [6, 6]
    assert int(out["target"]) == NO_TARGET
    assert float(out["damage_per_prey"][0]) == 0.0


def test_a_dead_prey_is_never_reported_or_targeted():
    obstacles = jnp.zeros((12, 12), bool)
    vis = jnp.ones((12, 12, 2 * P.r_pred + 1, 2 * P.r_pred + 1), bool)
    out = step_predator(
        obstacles,
        vis,
        pred_pos=jnp.array([6, 6]),
        pred_stamina=jnp.asarray(100.0),
        target=jnp.int32(NO_TARGET),
        still=jnp.asarray(0.0, jnp.float32),
        engage=jnp.int32(0),
        prey_pos=jnp.array([[6, 7], [6, 5]]),
        prey_alive=jnp.array([False, True]),
        prey_moved=jnp.array([True, True]),
        distant_draws=jnp.array([0.0, 0.0]),
        wander_trigger=jnp.asarray(1.0),
        wander_dir=jnp.asarray(0.0),
        params=P,
    )
    assert not bool(out["reported"][0]), "a dead prey was reported"
    assert int(out["target"]) == 1, "the predator targeted a dead prey"


def test_sprint_is_suppressed_at_adjacency():
    """A sprint from distance one overshoots. Asserted because the symptom is a predator that never lands."""
    obstacles = jnp.zeros((12, 12), bool)
    vis = jnp.ones((12, 12, 2 * P.r_pred + 1, 2 * P.r_pred + 1), bool)
    out = step_predator(
        obstacles,
        vis,
        pred_pos=jnp.array([6, 6]),
        pred_stamina=jnp.asarray(100.0),
        target=jnp.int32(0),
        still=jnp.asarray(0.0, jnp.float32),
        engage=jnp.int32(0),
        prey_pos=jnp.array([[6, 7]]),
        prey_alive=jnp.array([True]),
        prey_moved=jnp.array([True]),
        distant_draws=jnp.array([0.0]),
        wander_trigger=jnp.asarray(1.0),
        wander_dir=jnp.asarray(0.0),
        params=P,
    )
    assert not bool(out["sprinted"]), "the predator sprinted from adjacency and would overshoot"


def test_sprint_requires_stamina():
    """Stamina gates the sprint, and the fixture's prey must sit inside r_pred or nothing is targeted at all.

    The first version placed the prey at distance 6 against an r_pred of 4. It failed, correctly: an
    unreported prey is not a target, and an untargeted predator does not sprint. The precondition is asserted
    below so the separation cannot drift out of range again and quietly turn this into a test of nothing.
    """
    obstacles = jnp.zeros((12, 12), bool)
    vis = jnp.ones((12, 12, 2 * P.r_pred + 1, 2 * P.r_pred + 1), bool)
    pred_col, prey_col = 2, 6
    assert 1 < prey_col - pred_col <= P.r_pred, (
        f"fixture error: separation {prey_col - pred_col} must exceed 1 so a sprint is allowed, and be "
        f"within r_pred of {P.r_pred} so the prey is reported at all"
    )
    kw = dict(
        pred_pos=jnp.array([6, pred_col]),
        target=jnp.int32(0),
        still=jnp.asarray(0.0, jnp.float32),
        engage=jnp.int32(0),
        prey_pos=jnp.array([[6, prey_col]]),
        prey_alive=jnp.array([True]),
        prey_moved=jnp.array([True]),
        distant_draws=jnp.array([0.0]),
        wander_trigger=jnp.asarray(1.0),
        wander_dir=jnp.asarray(0.0),
        params=P,
    )
    rich = step_predator(obstacles, vis, pred_stamina=jnp.asarray(100.0), **kw)
    poor = step_predator(obstacles, vis, pred_stamina=jnp.asarray(0.0), **kw)
    assert bool(rich["sprinted"]), "a predator with full stamina at range should sprint"
    assert not bool(poor["sprinted"]), "a predator at zero stamina sprinted anyway"
    assert int(rich["pred_pos"][1]) - pred_col == 2, "a sprint should cover two cells"
    assert int(poor["pred_pos"][1]) - pred_col == 1, "a walk should cover one cell"


def test_the_predator_cannot_move_through_an_obstacle():
    """Movement goes through grid.resolve_move, so obstacle solidity applies to the predator too."""
    obstacles = jnp.zeros((12, 12), bool).at[6, 7].set(True)
    vis = jnp.ones((12, 12, 2 * P.r_pred + 1, 2 * P.r_pred + 1), bool)
    out = step_predator(
        obstacles,
        vis,
        pred_pos=jnp.array([6, 6]),
        pred_stamina=jnp.asarray(0.0),
        target=jnp.int32(0),
        still=jnp.asarray(0.0, jnp.float32),
        engage=jnp.int32(0),
        prey_pos=jnp.array([[6, 9]]),
        prey_alive=jnp.array([True]),
        prey_moved=jnp.array([True]),
        distant_draws=jnp.array([0.0]),
        wander_trigger=jnp.asarray(1.0),
        wander_dir=jnp.asarray(0.0),
        params=P,
    )
    assert [int(x) for x in out["pred_pos"]] == [6, 6], "the predator walked into an obstacle"


def test_chebyshev_counts_a_diagonal_as_one():
    assert int(chebyshev(jnp.array([0, 0]), jnp.array([1, 1]))) == 1
    assert int(chebyshev(jnp.array([0, 0]), jnp.array([3, 1]))) == 3


# ---------------------------------------------------------------------- Q-021 engagement --
def _trace(distances, params):
    """Run a distance sequence through the counter as one uninterrupted pursuit."""
    engage = jnp.int32(0)
    tgt = jnp.int32(0)
    out = []
    for i in range(1, len(distances)):
        engage = update_engagement(engage, jnp.int32(distances[i - 1]), jnp.int32(distances[i]), tgt, tgt)
        out.append(int(engage))
    return out


def test_engagement_is_reachable_at_BOTH_predator_speeds():
    """**The check whose absence let the old definition survive every test.**

    Q-021: the old rule incremented once per closing TICK while the predator closes two cells per tick when
    sprinting, so it arrived adjacent having accumulated fewer counts than cells crossed and never reached
    the threshold. A fixture exercising only a walking predator cannot tell the two rules apart, and
    sprinting is the predator's designed advantage, so the sprinting case is the one that matters.
    """
    walking = _trace([4, 3, 2, 1, 1, 1], P)
    sprinting = _trace([4, 2, 1, 1, 1, 1], P)

    assert max(walking) >= P.e_min, f"a walking approach never reached e_min {P.e_min}: {walking}"
    assert max(sprinting) >= P.e_min, (
        f"a SPRINTING approach never reached e_min {P.e_min}: {sprinting}. This is the exact failure Q-021 "
        "records, and it is invisible to any fixture that only walks the predator."
    )


def test_the_counter_accumulates_DISTANCE_CLOSED_not_ticks_elapsed():
    """**Falsification found this gap and it is worth stating precisely why the previous test missed it.**

    Q-021 fixed TWO independent faults: the counter measured closing ticks rather than distance closed, and it
    reset on arrival instead of holding. **Fixing EITHER one alone makes engagement reachable**, because the
    adjacency credit will carry a sprinting approach over the threshold on its own, just one tick later.

    So a test that asserts REACHABILITY cannot distinguish the two, and reverting the distance accumulation
    survived falsification while every other break was caught. The distance fault is about RATE, so the
    assertion has to be about rate.
    """
    # One tick that crosses two cells must contribute two, not one.
    two = int(update_engagement(jnp.int32(0), jnp.int32(4), jnp.int32(2), jnp.int32(0), jnp.int32(0)))
    assert two == 2, (
        f"closing two cells in one tick contributed {two}. The counter is measuring TICKS, which is the fault "
        "Q-021 exists to fix: a sprinting predator crosses two cells per tick and would arrive adjacent "
        "having accumulated fewer counts than cells crossed."
    )
    three = int(update_engagement(jnp.int32(0), jnp.int32(5), jnp.int32(2), jnp.int32(0), jnp.int32(0)))
    assert three == 3, f"closing three cells contributed {three}"

    # And the consequence that motivated it: closing faster must reach the threshold sooner.
    walking = _trace([6, 5, 4, 3, 2, 1, 1], P)
    sprinting = _trace([6, 4, 2, 1, 1, 1, 1], P)
    first_walk = next(i for i, v in enumerate(walking) if v >= P.e_min)
    first_sprint = next(i for i, v in enumerate(sprinting) if v >= P.e_min)
    assert first_sprint < first_walk, (
        f"a sprinting approach reached the threshold at tick {first_sprint + 1} and a walking one at tick "
        f"{first_walk + 1}. Closing twice as fast must count twice as fast, or the counter is measuring "
        "elapsed time rather than deliberate approach."
    )


def test_engagement_persists_while_adjacent_rather_than_firing_once_on_arrival():
    """`stillness_engaged` measures behaviour DURING an engagement, not on the tick one begins.

    Under the old rule the counter stopped once adjacent, because distance can no longer decrease, so
    `engaged` was an arrival event true for a single tick. Holding contact is continued pursuit.
    """
    held = _trace([4, 3, 2, 1, 1, 1, 1], P)
    over_threshold = sum(1 for v in held if v >= P.e_min)
    assert over_threshold >= 3, (
        f"engagement was at or above threshold on only {over_threshold} ticks of a sustained adjacency: "
        f"{held}. A descriptor measured over that window would have almost no data."
    )
    assert held == sorted(held), f"the counter fell during an unbroken pursuit: {held}"


def test_a_pause_does_not_reset_engagement_but_a_retreat_does():
    """**The only reset path besides losing the target, and nothing had ever exercised it.**

    A predator that closes, pauses, then closes is still approaching deliberately. Pausing is not evidence of
    coincidence; retreating is.
    """
    paused = _trace([4, 3, 3, 2, 1], P)
    assert 0 not in paused, f"a hold reset the counter: {paused}"

    retreated = _trace([4, 3, 2, 1, 2, 3], P)
    assert retreated[3] == 0, (
        f"the counter survived the predator retreating: {retreated}. Distance increasing is now the only "
        "reset path other than losing the target, so if this passes without the reset the counter can never "
        "fall and every pursuit accumulates forever."
    )

    rebuilt = _trace([5, 4, 3, 4, 3, 2], P)
    assert rebuilt[2] == 0 and rebuilt[-1] > 0, (
        f"a retreat should reset and a renewed approach should rebuild: {rebuilt}"
    )


def test_changing_target_resets_engagement():
    """A retarget must not inherit the previous pursuit's accumulated engagement."""
    engage = jnp.int32(9)
    out = update_engagement(engage, jnp.int32(3), jnp.int32(2), jnp.int32(1), jnp.int32(0))
    assert int(out) == 0, "engagement carried across a change of target"

    none = update_engagement(
        jnp.int32(9),
        jnp.int32(3),
        jnp.int32(2),
        jnp.int32(NO_TARGET),
        jnp.int32(NO_TARGET),
    )
    assert int(none) == 0, "engagement accumulated with no target"


# ---------------------------------------------------------------------------- the wander --
#
# **These consult behaviour rather than comparing implementations, and that is deliberate.** The wander was
# added to `src/env/` and to the oracle within minutes of each other, from the same sentence of the
# specification, so gate E2 over it is close to comparing two copies of one belief. A differential detects
# divergent mistakes, not shared ones. What follows measures what the mechanism ACHIEVES.


def _wander_steps(p_wander, n=4000, seed=0):
    """Move a targetless predator on an empty grid `n` times and return where it went each tick.

    **Vmapped, not looped.** A first version called `step_predator` 4000 times in an eager Python loop and
    timed out: CLAUDE.md measures eager dispatch at about 1.3 ms per operation, so a few thousand ticks of a
    thirty-operation function is minutes rather than seconds. The rule is stated for the episode loop and
    applies just as much to a test fixture.
    """
    import numpy as np

    params = cfg(p_wander=p_wander)
    obstacles = jnp.zeros((21, 21), bool)
    vis = jnp.ones((21, 21, 2 * params.r_pred + 1, 2 * params.r_pred + 1), bool)
    rng = np.random.default_rng(seed)
    start = jnp.array([10, 10], jnp.int32)

    def one(trigger, direction):
        out = step_predator(
            obstacles,
            vis,
            start,
            jnp.asarray(100.0),
            jnp.int32(NO_TARGET),
            jnp.asarray(0.0, jnp.float32),
            jnp.int32(0),
            prey_pos=jnp.zeros((1, 2), jnp.int32),  # dead, so it is never reported and never a target
            prey_alive=jnp.array([False]),
            prey_moved=jnp.array([False]),
            distant_draws=jnp.array([1.0]),
            wander_trigger=trigger,
            wander_dir=direction,
            params=params,
        )
        return out["pred_pos"] - start

    triggers = jnp.asarray(rng.random(n), jnp.float32)
    dirs = jnp.asarray(rng.random(n), jnp.float32)
    return np.asarray(jax.jit(jax.vmap(one))(triggers, dirs))


def test_the_wander_fires_at_its_stated_rate():
    """**The rate is the parameter's meaning, so it is what gets asserted.** A wander that fired at half its
    stated probability would still look like a wandering predator and would price search at half what the
    provenance record says."""
    import numpy as np

    for p_wander in (0.25, 0.5, 1.0):
        deltas = _wander_steps(p_wander)
        fired = float(np.mean(np.any(deltas != 0, axis=1)))
        # 4000 draws, so the standard error on a proportion is under 0.008; 0.04 is five of those.
        assert abs(fired - p_wander) < 0.04, (
            f"p_wander {p_wander} fired on {fired:.3f} of ticks. The rate is the parameter's meaning."
        )


def test_the_wander_visits_every_direction():
    """A wander biased toward some directions is a search pattern, not a random walk, and a search pattern is
    the unspecified change to the control condition that kept a wander out of the design originally."""
    import numpy as np

    deltas = _wander_steps(1.0)
    moved = deltas[np.any(deltas != 0, axis=1)]
    seen = {tuple(d) for d in moved}
    assert len(seen) == 8, f"the wander visited {len(seen)} distinct directions, not 8: {sorted(seen)}"

    counts = np.array([np.sum(np.all(moved == np.array(d), axis=1)) for d in sorted(seen)])
    share = counts / counts.sum()
    assert share.min() > 0.08 and share.max() < 0.17, (
        f"direction shares {np.round(share, 3).tolist()} are not close to uniform at one eighth each"
    )


def test_p_wander_zero_reproduces_holding_position():
    """The default, and the reason it is the default: every gate in this repository was written against a
    predator that holds position without a target."""
    import numpy as np

    deltas = _wander_steps(0.0, n=500)
    assert not np.any(deltas != 0), "at p_wander 0.0 a targetless predator must never move"


def test_the_wander_never_overrides_pursuit():
    """Pursuit takes priority. A predator that wandered away from a target it can see would be a different
    predator, and the parameter is about SEARCH."""
    obstacles = jnp.zeros((12, 12), bool)
    vis = jnp.ones((12, 12, 2 * P.r_pred + 1, 2 * P.r_pred + 1), bool)
    params = cfg(p_wander=1.0)
    out = step_predator(
        obstacles,
        vis,
        jnp.array([6, 2], jnp.int32),
        jnp.asarray(100.0),
        jnp.int32(NO_TARGET),
        jnp.asarray(0.0, jnp.float32),
        jnp.int32(0),
        prey_pos=jnp.array([[6, 5]], jnp.int32),
        prey_alive=jnp.array([True]),
        prey_moved=jnp.array([True]),
        distant_draws=jnp.array([0.0]),
        wander_trigger=jnp.asarray(0.0),  # would fire, if pursuit did not take priority
        wander_dir=jnp.asarray(0.5),
        params=params,
    )
    assert int(out["target"]) == 0, "the prey is reported, so the predator should have a target"
    assert int(out["pred_pos"][1]) > 2, "with a target the predator must close, not wander"
