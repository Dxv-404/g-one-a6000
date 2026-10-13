"""Gate E8, mask correctness. Protects invariant I4, so this one gets full falsification.

"No burst below `S_MIN_BURST`, no complex action at zero energy, and no stay ever caused by a movement mask,
since basic movement is never gated."

**Why I4 matters, in the terms CLAUDE.md now uses.** If any action could be involuntarily masked, the action
log would record availability-filtered choices rather than policy outputs, and no behaviour would be
attributable to policy. That is why the third clause is the important one: the first two say the mask does its
job, the third says the mask does nothing else. A mask that quietly converted a blocked move into a stay would
put an environment-authored action into the log under the agent's name.

Exhaustive rather than sampled where it can be. There are only `N_ACTIONS` actions, twenty-two since D-030
reserved `attack` and `drop`, so every action can be tested against every relevant pool state directly instead
of hoping a random trajectory covers the corners. Loops are written over `N_ACTIONS` rather than a literal, so
widening the action set widens the coverage instead of silently leaving the new tail untested.
"""

import jax.numpy as jnp  # noqa: E402

from configs.default import DEFAULT as P  # noqa: E402
from src.env.dynamics import (  # noqa: E402
    ACTION_CROUCH,
    ACTION_LIE_DOWN,
    ACTION_PICK,
    ACTION_STAY,
    N_ACTIONS,
    RESERVED_ACTIONS,
    is_basic_move,
    is_burst,
    is_complex,
    is_reserved,
    mask_actions,
)

ALL_ACTIONS = jnp.arange(N_ACTIONS, dtype=jnp.int32)


def _mask_all(energy_value, stamina_value):
    """Every action, at one pool state. Returns (masked_actions, blocked_flags)."""
    energy = jnp.full((N_ACTIONS,), energy_value, jnp.float32)
    stamina = jnp.full((N_ACTIONS,), stamina_value, jnp.float32)
    return mask_actions(ALL_ACTIONS, energy, stamina, P)


def test_no_burst_survives_below_the_burst_threshold():
    """Exhaustive over every action, at stamina one step below the threshold."""
    masked, blocked = _mask_all(P.e_max, P.s_min_burst - 0.001)
    for a in range(N_ACTIONS):
        if bool(is_burst(jnp.int32(a))):
            assert int(masked[a]) == ACTION_STAY, f"burst action {a} survived below s_min_burst"
            assert bool(blocked[a]), f"burst action {a} was blocked without recording the event"


def test_every_burst_survives_at_the_threshold_exactly():
    """The boundary is `stamina < s_min_burst`, so exactly at the threshold a burst is legal.

    Asserted because an off-by-one here would silently make bursts one step more expensive than specified, and
    `s_min_burst` equals `s_burst_prey` precisely so that a burst requires a full burst's worth banked.
    """
    masked, blocked = _mask_all(P.e_max, P.s_min_burst)
    for a in range(N_ACTIONS):
        if bool(is_burst(jnp.int32(a))):
            assert int(masked[a]) == a, f"burst action {a} was blocked at exactly s_min_burst"
            assert not bool(blocked[a])


def test_no_complex_action_survives_at_zero_energy():
    masked, blocked = _mask_all(0.0, P.s_max)
    for a in range(N_ACTIONS):
        if bool(is_complex(jnp.int32(a))):
            assert int(masked[a]) == ACTION_STAY, f"complex action {a} survived at zero energy"
            assert bool(blocked[a])


def test_basic_movement_is_never_masked_anywhere_in_the_pool_space():
    """Invariant I4, asserted over the whole grid of pool states rather than at a few points.

    This is the clause that says the mask does nothing beyond its job. If a basic move is ever converted to
    stay, the action log stops being a record of policy output.
    """
    for energy in (0.0, 0.001, 0.5, P.e_max / 2, P.e_max):
        for stamina in (0.0, 0.001, P.s_min_burst - 0.001, P.s_min_burst, P.s_max):
            masked, blocked = _mask_all(energy, stamina)
            for a in range(N_ACTIONS):
                if bool(is_basic_move(jnp.int32(a))):
                    assert int(masked[a]) == a, (
                        f"basic move {a} was masked at energy {energy}, stamina {stamina}. "
                        "Invariant I4: basic movement is never gated."
                    )
                    assert not bool(blocked[a]), f"basic move {a} recorded a mask event"


def test_no_stay_is_ever_caused_by_masking_a_basic_move():
    """The gate's third clause, stated the way the build spec states it.

    Distinct from the test above because it asserts about the *reason* a stay appears: every stay in the masked
    output either was a stay already, or came from a burst, a complex action, or a reserved action being
    blocked. None came from a basic move.
    """
    for energy in (0.0, 0.5, P.e_max):
        for stamina in (0.0, P.s_min_burst - 0.001, P.s_max):
            masked, blocked = _mask_all(energy, stamina)
            for a in range(N_ACTIONS):
                if int(masked[a]) == ACTION_STAY and a != ACTION_STAY:
                    # The invariant itself, stated negatively. A whitelist of allowed causes needs editing
                    # every time a cause is added, and it was the clause that had to change when attack and
                    # drop were reserved. This form does not, and it is what I4 actually requires.
                    assert not bool(is_basic_move(jnp.int32(a))), (
                        f"basic move {a} became stay, so a movement mask produced it. This breaks I4."
                    )
                    # And the cause must still be one the design knows about, so an unexplained new mask
                    # source fails here rather than being absorbed silently.
                    assert (
                        bool(is_burst(jnp.int32(a)))
                        or bool(is_complex(jnp.int32(a)))
                        or bool(is_reserved(jnp.int32(a)))
                    ), f"action {a} became stay for no reason the design recognises"


def test_stay_crouch_and_lie_down_are_never_masked():
    """The canonical design leaves basic movement, stay, crouch and lie-down available at zero energy."""
    for energy in (0.0, P.e_max):
        for stamina in (0.0, P.s_max):
            masked, blocked = _mask_all(energy, stamina)
            for a in (ACTION_STAY, ACTION_CROUCH, ACTION_LIE_DOWN):
                assert int(masked[a]) == a, f"action {a} was masked at energy {energy}, stamina {stamina}"
                assert not bool(blocked[a])


def test_pick_is_blocked_at_zero_energy_and_this_makes_starvation_irreversible():
    """The literal reading of the complex-action list, with its consequence asserted rather than implied.

    `pick` counts as item manipulation, so it is blocked at zero energy. Because `pick` is also how food is
    eaten in this increment, an agent that reaches zero energy cannot eat and therefore cannot recover: it
    bleeds at `starve_rate` for `h_max / starve_rate` ticks and dies.

    Asserted here so the behaviour is visible in a named test rather than emerging from the interaction of two
    modules. If the intended reading is the other one, this test says so loudly. See Q-013.
    """
    masked, blocked = _mask_all(0.0, P.s_max)
    assert int(masked[ACTION_PICK]) == ACTION_STAY, "pick survived at zero energy"
    assert bool(blocked[ACTION_PICK])

    ticks_to_die = P.h_max / P.starve_rate
    assert ticks_to_die == 50.0, (
        f"starvation decline is {ticks_to_die} ticks. The irreversibility above is only tolerable because the "
        "decline is long enough to be an observable cue rather than a sudden collapse."
    )


def test_only_reserved_actions_are_masked_when_both_pools_are_full():
    """The null case, now that two actions are unconditionally unavailable.

    Previously this asserted that nothing at all is masked at full pools. Reserving `attack` and `drop` makes
    that false by design, so the assertion is narrowed to exactly the reserved pair rather than dropped: if
    anything else is blocked at full pools, a mask condition is inverted somewhere.
    """
    masked, blocked = _mask_all(P.e_max, P.s_max)
    for a in range(N_ACTIONS):
        if bool(is_reserved(jnp.int32(a))):
            assert int(masked[a]) == ACTION_STAY, f"reserved action {a} survived at full pools"
            assert bool(blocked[a]), f"reserved action {a} was masked without recording the event"
        else:
            assert int(masked[a]) == a, f"action {a} was masked with both pools full"
            assert not bool(blocked[a]), f"action {a} recorded a mask event with both pools full"


def test_reserved_actions_are_masked_at_every_pool_combination():
    """`attack` and `drop` are unavailable in increment 1 unconditionally, not as a function of state.

    Asserted across the pool space because unconditional is the property that matters: a reserved action that
    happened to be available at some energy would be an action the agent could evolve to use, and increment 1
    is supposed to have no access to either.
    """
    for energy in (0.0, 0.5, P.e_max):
        for stamina in (0.0, P.s_min_burst - 0.001, P.s_max):
            masked, blocked = _mask_all(energy, stamina)
            for a in RESERVED_ACTIONS:
                assert int(masked[a]) == ACTION_STAY, (
                    f"reserved action {a} survived at energy {energy}, stamina {stamina}"
                )
                assert bool(blocked[a]), f"reserved action {a} blocked without a recorded event"


def test_coverage_every_action_class_was_actually_exercised():
    """Power check. The exhaustive loops above are only meaningful if each class is non-empty.

    The total is stated as an accounting identity in which every term is attributable, rather than as a
    literal count. An unexplained delta then fails whether the action set grew or shrank, and a legitimate
    change like reserving attack and drop is expressed by adding its term rather than by editing a number.
    """
    n_burst = int(jnp.sum(is_burst(ALL_ACTIONS)))
    n_move = int(jnp.sum(is_basic_move(ALL_ACTIONS)))
    n_complex = int(jnp.sum(is_complex(ALL_ACTIONS)))
    n_reserved = int(jnp.sum(is_reserved(ALL_ACTIONS)))

    assert n_move == 8, f"expected 8 basic moves, found {n_move}"
    assert n_burst == 8, f"expected 8 burst actions, found {n_burst}"
    assert n_complex == 9, f"expected 9 complex actions, 8 sprints plus pick, found {n_complex}"
    assert n_reserved == 2, f"expected 2 reserved actions, attack and drop, found {n_reserved}"

    n_singletons = 4  # stay, crouch, lie_down, pick
    assert n_move + n_burst + n_singletons + n_reserved == N_ACTIONS, (
        f"the action set does not add up: {n_move} moves + {n_burst} bursts + {n_singletons} singletons "
        f"+ {n_reserved} reserved is {n_move + n_burst + n_singletons + n_reserved}, but N_ACTIONS is "
        f"{N_ACTIONS}. Every action must belong to exactly one accounted class."
    )


def test_mask_events_are_recorded_for_every_block():
    """Gate E8 needs the events to check its third clause, so a block without an event is a defect."""
    for energy in (0.0, P.e_max):
        for stamina in (0.0, P.s_max):
            masked, blocked = _mask_all(energy, stamina)
            changed = masked != ALL_ACTIONS
            assert bool(jnp.all(changed == blocked)), (
                f"at energy {energy}, stamina {stamina}: the set of changed actions does not match the set of "
                "recorded mask events"
            )
