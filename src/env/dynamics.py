"""Action masking, the three body pools, and write-once death causes.

The three pools are never collapsed into one bar, because collapsing any two rebuilds a survival drive or a
survival reward through the back door.

    pool      job                        drains on              refilled by     at zero
    energy    hunger, survival pressure  every tick             eating food     starvation death
    stamina   exertion budget            burst actions only     passive rest    bursts blocked, no harm
    health    damage buffer              predator damage only   slow recovery   death by predation

All three are inputs to the network and none is ever a term in fitness. They are state variables, not
proxies: the decisive test is whether an agent could raise its fitness by manipulating the variable directly,
and for these the answer is no, because fitness cannot see them. They are world dynamics, closer to a wall
than to a reward.

**Energy drains a flat amount every tick regardless of action.** This is deliberate and confirmed rather than
inherited: any differential between moving and staying is a direct incentive on one of the behaviours being
measured, so equal cost is the only neutral setting. A movement surcharge has a real fidelity argument behind
it and must still not be added. Sprint draws an additional amount, which is a burst cost rather than a
movement cost.

**Invariant I4: basic movement never costs stamina and is never gated.** Only bursts draw stamina and only
bursts are blocked at zero. The reason is attribution rather than fairness: if any action could be
involuntarily masked, the action log would record availability-filtered choices rather than policy outputs,
and no behaviour would be attributable to policy.

Build step 4. Gates E6 pool bounds, E7 death cause, E8 mask correctness.
"""

import jax.numpy as jnp
from jax import Array

# Action indices, fixed by the build spec's action set. 0 to 7 move, 8 to 15 sprint, then the singletons.
ACTION_MOVE_FIRST = 0
ACTION_MOVE_LAST = 7
ACTION_SPRINT_FIRST = 8
ACTION_SPRINT_LAST = 15
ACTION_STAY = 16
ACTION_CROUCH = 17
ACTION_LIE_DOWN = 18
ACTION_PICK = 19
# Reserved in increment 1, masked unavailable, evolved against later. See D-030 and CLAUDE.md section 14.
# `attack` arrives at increment 2 with the coevolved predator's explicit strike decision, nodes 4.6.0.0 and
# 4.21.0.0. `drop` follows from the single held slot: an agent holding a weapon that wants a block needs it.
# They are reserved NOW because adding an action later changes `n_actions` and voids every evolved weight,
# and 98 parameters is nothing against that.
ACTION_ATTACK = 20
ACTION_DROP = 21
N_ACTIONS = 22

# The actions that exist in the action space but are unavailable in this increment. Masked to stay, with the
# mask event recorded, which is what makes their exclusion auditable at step 11 rather than invisible.
RESERVED_ACTIONS = (ACTION_ATTACK, ACTION_DROP)

# Death causes. Zero is alive, so an unset entry cannot be mistaken for a recorded cause.
CAUSE_ALIVE = 0
CAUSE_STARVATION = 1
CAUSE_PREDATION = 2

# Posture codes, matching the observation's three one-hot channels.
POSTURE_UPRIGHT = 0
POSTURE_CROUCHED = 1
POSTURE_PRONE = 2


def is_burst(actions: Array) -> Array:
    """Burst actions are the eight sprints. These are the only actions that draw stamina."""
    return (actions >= ACTION_SPRINT_FIRST) & (actions <= ACTION_SPRINT_LAST)


def is_basic_move(actions: Array) -> Array:
    """The eight single-cell moves. Never gated, never charged. Invariant I4."""
    return (actions >= ACTION_MOVE_FIRST) & (actions <= ACTION_MOVE_LAST)


def is_complex(actions: Array) -> Array:
    """Actions blocked at zero energy: the sprints, and item manipulation.

    `pick` is counted as item manipulation, which is the literal reading of the canonical design's list of
    complex actions, "sprint, jump, attack, and item manipulation". It has a consequence worth stating rather
    than discovering: because `pick` is also how food is eaten in this increment, blocking it at zero energy
    makes starvation irreversible. An agent that reaches zero energy cannot eat, so it bleeds at `starve_rate`
    for `h_max / starve_rate` ticks and dies.

    That may well be intended, since the design says a starving agent "visibly degrades rather than simply
    dropping dead" and wants energy management to be an emergent problem with an evolvable cue for an
    opponent. But the design also separates object interaction from eating in the full action set, and this
    increment collapses both into one action, so the reading is not forced. Raised as Q-013 rather than
    settled here, and asserted either way by gate E8 so the choice is visible rather than implicit.
    """
    return is_burst(actions) | (actions == ACTION_PICK)


def is_reserved(actions: Array) -> Array:
    """The two actions that exist in the action space but are unavailable in increment 1.

    `attack` and `drop` are reserved so that `n_actions` is 22 from the start. Adding either later would
    change the output layer's shape and void every evolved weight, which is the same reason crouch, lie-down
    and the appearance vector already exist unused. See D-030.

    They are **not** complex actions and not bursts. Being unavailable is a property of this increment, not a
    property of the action, so it is a separate predicate rather than a clause bolted onto `is_complex`. At
    increment 2 this function goes away rather than being edited.
    """
    return (actions == ACTION_ATTACK) | (actions == ACTION_DROP)


def mask_actions(actions: Array, energy: Array, stamina: Array, params) -> tuple[Array, Array]:
    """Block illegal actions, falling through to stay. Returns (masked_actions, mask_event).

    Tick order step three. Three rules, and only three:

        bursts           blocked when stamina < S_MIN_BURST
        complex actions  blocked when energy <= 0
        reserved         always blocked in increment 1: attack and drop

    A blocked action becomes stay and the event is recorded, because gate E8 asserts that no stay was ever
    produced by masking a basic move, and that assertion needs the events to check.

    **The reserved rule has a consequence for step 11 that must not be discovered there.** A masked action
    becomes `stay`, so a policy that outputs `attack` is indistinguishable in the action log from one that
    chose to stay still. Since `stillness_engaged` counts stay, reserved actions would inflate the very
    descriptor the experiment turns on. The mask event is what makes this recoverable: **descriptors must
    exclude ticks where the stay was produced by masking, not merely count stays.** Recorded in D-030 and in
    `session.md`, because by step 11 the reason will not be obvious from the code.

    Unavailability is unconditional and therefore never a function of the agent's state, which is what keeps
    it outside invariant I4's concern: I4 forbids masking that could involuntarily remove a *basic move*, and
    no basic move is reserved.

    **Basic movement is never masked here.** A move that an obstacle refuses still resolves as a move action
    whose position happens not to change; that is movement resolution in `grid.resolve_move`, not action
    masking, and the distinction is load-bearing. Collapsing the two would either gate basic movement, which
    breaks I4, or make obstacles intangible.
    """
    burst = is_burst(actions)
    complex_ = is_complex(actions)

    blocked_burst = burst & (stamina < params.s_min_burst)
    blocked_energy = complex_ & (energy <= 0.0)
    blocked_reserved = is_reserved(actions)
    blocked = blocked_burst | blocked_energy | blocked_reserved

    masked = jnp.where(blocked, ACTION_STAY, actions)
    return masked, blocked


def update_energy(energy: Array, sprinting: Array, ate_food: Array, params) -> Array:
    """Flat baseline drain every tick, an additional draw for sprinting, refilled by food.

    The baseline is charged whether or not the agent moved, so idling is never free and stillness is never
    costless. Clipped into range, which gate E6 asserts.
    """
    delta = -params.e_base - params.e_sprint * sprinting.astype(energy.dtype)
    delta = delta + params.e_food * ate_food.astype(energy.dtype)
    return jnp.clip(energy + delta, 0.0, params.e_max)


def update_stamina(stamina: Array, bursting: Array, params, burst_cost: float | None = None) -> Array:
    """Drains on bursts only, regenerates passively otherwise. At zero it blocks bursts and never harms.

    `burst_cost` defaults to the prey cost. The predator passes `s_burst_pred`, so **one rule serves both
    bodies and they differ only in price**. DEV-014 is why that matters: the predator's stamina was never
    written back at all, so it sat at `s_max` forever, `stamina >= s_min_burst` was always true, and the
    predator sprinted on every tick where distance exceeded one. Its designed 50 percent sprint uptime was
    100 percent, the exhaustion window node 4.16.0.0 depends on did not exist, and `s_burst_pred`, one of the
    eight swept pricing parameters, was inert.
    """
    cost = params.s_burst_prey if burst_cost is None else burst_cost
    bursting_f = bursting.astype(stamina.dtype)
    delta = -cost * bursting_f + params.s_regen * (1.0 - bursting_f)
    return jnp.clip(stamina + delta, 0.0, params.s_max)


def update_health(health: Array, damage: Array, energy_after: Array, params) -> Array:
    """Damage from capture contact, slow passive recovery, and the starvation bleed.

    **`damage` is the MAGNITUDE, not a boolean. DEV-031.** The build spec wrote this equation as
    `health - DAMAGE * damaged` with `damaged` a boolean, while `resolve_capture` computes a graded damage
    `DAMAGE * (1 - CUE * (1 - moved))`. Both implementations were faithful to both equations, so the graded
    value was computed, reduced to its sign, and discarded, and `cue_dependence` could only switch damage
    between zero and full. A five-level swept axis had two behaviours.

    **It survived because the default is the one value where the reduction is exact**: at `cue_dependence`
    1.0 the graded damage is exactly 0 or exactly DAMAGE. Every gate runs at the default.

    Taking the magnitude here means `params.damage` no longer appears in this function at all, which is the
    point: the amount of damage is decided where the capture is resolved, and this applies what it was given.

    The bleed is keyed on energy **after** its own update, per the build spec's ordering: starvation is
    evaluated against the energy the agent ends the tick with, not the energy it started with. Health is the
    only pool that drains from damage and it never drains on a clock except when starving, which is what
    makes capture take several ticks and gives any behavioural response a window to act in.
    """
    starving = (energy_after <= 0.0).astype(health.dtype)
    delta = -damage.astype(health.dtype) + params.h_regen - params.starve_rate * starving
    return jnp.clip(health + delta, 0.0, params.h_max)


def resolve_deaths(
    alive: Array,
    health_after: Array,
    energy_after: Array,
    damaged: Array,
    death_cause: Array,
    death_step: Array,
    died_starving: Array,
    died_predated: Array,
    step,
) -> tuple[Array, Array, Array, Array, Array]:
    """Write-once death recording. Returns (alive, death_cause, death_step).

    Per the build spec:

        newly_dead = alive AND (health' <= 0)
        cause      = where(energy' <= 0, STARVATION, PREDATION)

    Write-once is the property gate E7 checks: a cause is recorded on the tick of death and never overwritten,
    and every death carries exactly one. It is enforced by conditioning the write on `alive`, so an
    already-dead agent cannot have its cause rewritten on a later tick even though it still occupies its array
    slot and still has its pools updated.

    **Two independent flags are recorded, and the single cause is derived from them.** The precedence rule
    alone, `where(energy <= 0, STARVATION, PREDATION)`, records an agent that is starving AND takes a lethal
    hit as starvation, which **under-counts predation among exactly the weakened agents most likely to be
    caught**. The two conditions are not alternatives and a tie-break between them discards information.

    So `died_starving` and `died_predated` are set independently and can both be true. `death_cause` keeps the
    old precedence and is retained for convenience, so every existing consumer and gate E7 are unaffected.
    **This is enforcement by construction rather than a better tie-break: there is no tie left to break.**

    All four records are write-once, enforced by conditioning on `newly_dead`, which is true on exactly one
    tick per agent. An already-dead agent keeps its array slot and its pools keep updating, and none of that
    can rewrite what killed it.
    """
    newly_dead = alive & (health_after <= 0.0)
    starving = energy_after <= 0.0
    cause = jnp.where(starving, CAUSE_STARVATION, CAUSE_PREDATION)

    death_cause = jnp.where(newly_dead, cause, death_cause)
    death_step = jnp.where(newly_dead, jnp.asarray(step, death_step.dtype), death_step)
    died_starving = died_starving | (newly_dead & starving)
    died_predated = died_predated | (newly_dead & damaged)
    alive = alive & ~newly_dead
    return alive, death_cause, death_step, died_starving, died_predated


def update_posture(posture: Array, actions: Array) -> Array:
    """Crouch and lie-down set posture; anything else returns the agent upright.

    Inert in this increment: the scripted predator does not read posture. They exist because increment two
    evolves a response to them and because adding actions later would change `n_actions` and void every
    evolved weight. CLAUDE.md section 14 lists this among the things that look like bugs and are not.
    """
    posture = jnp.where(actions == ACTION_CROUCH, POSTURE_CROUCHED, POSTURE_UPRIGHT)
    return jnp.where(actions == ACTION_LIE_DOWN, POSTURE_PRONE, posture)
