"""The scripted predator: perception, target selection, pursuit, capture, and disengagement.

The control condition and the calibration reference, not a placeholder. It has stamina but no energy, cannot
starve, and persists for the full horizon, because it is the instrument applying selection pressure rather
than the subject being studied.

### Cue dependence, and why it is a parameter rather than two implementations

The build spec's original rule hardcoded motion gating: a still prey took no damage and broke pursuit after
`disengage_steps` ticks. The revised specification replaces the two predator variants D5 and D6 with the two
endpoints of one continuous parameter, `cue_dependence`:

    damage  = adjacent * DAMAGE * (1 - CUE * (1 - moved))
    still'  = 0 if moved else still + CUE          while adjacent

    CUE=1.0   still prey takes 0 damage, pursuit breaks in 5 ticks    the original rule, exactly
    CUE=0.5   still prey takes 17 damage, pursuit breaks in 10 ticks
    CUE=0.0   still prey takes full damage, pursuit never breaks

**`cue_dependence` prices freezing against every other anti-predator strategy, so it has no neutral value and
is a swept axis rather than a design choice.** See node 4.23.0.0 and the parameter's docstring in
`configs/default.py`. It defaults to 1.0, which reproduces the original behaviour exactly and is what keeps
every gate written against that rule valid.

**This is a property of this predator, not a law of the world.** Nothing here states that a still prey cannot
be captured; at `cue_dependence` 0.0 stillness confers nothing whatsoever. That distinction is invariant I2
and the first entry on CLAUDE.md's refusal list, and the parameterisation is what makes it explicit rather
than implicit.

### Invariant I5

`prey_moved` is an argument, never recomputed here. The tick order resolves prey movement at step 4 and
capture at step 7, and capture reads the flag from step 4 of the *same* tick. That is what makes stillness a
live decision taken under the predator's current position rather than a state read from a stale flag. Do not
reorder, and do not be tempted to derive movement from positions inside this module: by the time the predator
moves, the prey's previous position is gone.
"""

import jax
import jax.numpy as jnp
from jax import Array

from src.env import dynamics

# `target` uses -1 for "no target". Kept as a named constant because it appears in four places and an
# off-by-one here would silently retarget prey zero.
NO_TARGET: int = -1


def chebyshev(a: Array, b: Array) -> Array:
    """Chebyshev distance, the metric for eight-connected movement. One diagonal step costs one."""
    return jnp.max(jnp.abs(a - b), axis=-1)


def select_target(
    current: Array,
    reported: Array,
    distances: Array,
) -> Array:
    """Target persistence with nearest-visible fallback.

        target' = current target if still reported
                  else argmin chebyshev over reported prey
                  else NO_TARGET

    Persistence comes first, which is what makes pursuit a commitment rather than a per-tick reassignment.
    Without it the predator would switch to whichever prey drifted nearest each tick, engagements would never
    accumulate, and `disengage_steps` would have nothing to count against.

    `reported` is the two-regime perception result per prey, already ANDed with alive. Ties in the fallback go
    to the lowest index, matching the pick-contention convention in `resources.resolve_pick`, so that
    contention is resolved the same way everywhere.
    """
    valid = jnp.any(reported)

    # Keep the current target only if it is still reported.
    #
    # The clamp is DEFENSIVE, not load-bearing, and the comment here previously claimed otherwise. It said
    # the clamp stopped -1 wrapping to the last prey and silently keeping a target that was never selected.
    # Falsification removed the clamp and every gate still passed, which was correct: `current != NO_TARGET`
    # short-circuits the whole conjunction, so `reported[-1]` is read and discarded and the result cannot
    # differ. Verified directly rather than argued. The clamp stays because an out-of-bounds read that
    # happens to be harmless today is not worth keeping, but it earns no test and no claim.
    safe_current = jnp.clip(current, 0, reported.shape[0] - 1)
    keeps = (current != NO_TARGET) & reported[safe_current]

    # argmin over reported prey only. Unreported prey are pushed to +inf so they can never win.
    masked = jnp.where(reported, distances, jnp.inf)
    nearest = jnp.argmin(masked).astype(current.dtype)

    fallback = jnp.where(valid, nearest, jnp.int32(NO_TARGET).astype(current.dtype))
    return jnp.where(keeps, current, fallback)


def pursue(
    obstacles: Array,
    pred_pos: Array,
    target_pos: Array,
    stamina: Array,
    has_target: Array,
    wander_trigger: Array,
    wander_dir: Array,
    params,
) -> tuple[Array, Array]:
    """One step of pursuit. Returns the new position and whether the predator sprinted.

        step_dir = sign(target_pos - pred_pos)
        sprint   = (distance > 1) AND (stamina >= S_MIN_BURST)
        pred_pos'= move(pred_pos, step_dir, 2 if sprint else 1)

    Sprint is suppressed at distance one, because a sprint from adjacency overshoots the target rather than
    reaching it. Movement goes through `grid.resolve_move`, the same path prey use, so obstacle solidity and
    the no-corner-clipping rule apply identically to both. A predator that could clip corners would be
    strictly faster through cluttered terrain than the prey it hunts, which is a speed advantage the design
    never granted it.

    **With no target the predator takes a random step with probability `p_wander`, and holds position
    otherwise.** At `p_wander` 0.0 it always holds position, which is what this function did before and what
    every gate was written against.

    The wander exists because holding position has a measured consequence that no rule states: a prey that
    never moves is never approached, so it is never detected, never targeted and never damaged, at any
    `cue_dependence`. Freezing is then free rather than priced, which is the capacity-not-incentive principle
    broken by a composition of individually deliberate behaviours. See Q-042.

    **A pure random walk, not a search heuristic.** The original objection to a wander was that inventing
    search behaviour would be an unspecified change to the control condition. A uniform random step does not
    raise it: it adds no information about where prey are.

    **The wander never sprints.** Sprinting is pursuit, it costs stamina from the same budget the chase
    needs, and a predator that sprinted at random would arrive at the chase already tired for reasons
    unrelated to the chase.
    """
    from src.env.grid import DIRECTIONS, resolve_move

    delta = target_pos - pred_pos
    step_dir = jnp.sign(delta).astype(pred_pos.dtype)
    distance = chebyshev(pred_pos, target_pos)

    sprint = has_target & (distance > 1) & (stamina >= params.s_min_burst)

    # Draws arrive from the caller, per D-001: the draw site stays in one place and replay alignment has a
    # single owner. Both are consumed every tick whether or not the predator has a target, so the draw COUNT
    # never depends on the state of the world, which is DEV-022's rule.
    wander_index = jnp.clip((wander_dir * DIRECTIONS.shape[0]).astype(jnp.int32), 0, DIRECTIONS.shape[0] - 1)
    wandering = (~has_target) & (wander_trigger < params.p_wander)
    wander_step = DIRECTIONS[wander_index].astype(pred_pos.dtype)

    direction = jnp.where(has_target, step_dir, jnp.where(wandering, wander_step, jnp.zeros_like(step_dir)))

    moved_pos = resolve_move(obstacles, pred_pos, direction, sprint)
    return jnp.where(has_target | wandering, moved_pos, pred_pos), sprint


def resolve_capture(
    pred_pos: Array,
    target_pos: Array,
    target_moved: Array,
    still: Array,
    has_target: Array,
    params,
) -> tuple[Array, Array, Array]:
    """Capture contact and the disengagement counter. Returns damage, the new counter, and adjacency.

        adjacent = chebyshev(pred_pos', target_pos') <= 1
        still'   = 0 if moved else still + CUE           while adjacent, otherwise unchanged
        damage   = adjacent * DAMAGE * (1 - CUE * (1 - moved))

    **`target_moved` is `prey_moved` from step 4 of this tick.** Invariant I5. See the module docstring.

    **On the adjacency factor in `damage`.** The revised specification writes the damage line as
    `DAMAGE * (1 - CUE * (1 - moved))` without the adjacency term that its own "before" line carried. Taken
    literally that would damage every prey on the grid at once. Adjacency is retained here, because it is the
    only reading under which `cue_dependence` 1.0 reproduces the original rule, which the specification names
    as the check that matters most. Recorded in `deviations.md` DEV-009 as `amend-spec`.

    `still` is float rather than int, which is the revision's only type change. At CUE=1.0 it takes integral
    values and behaves exactly as the integer counter did; at CUE=0.25 it accrues in quarters, so pursuit
    breaks after 20 ticks of stillness rather than 5. The threshold itself never moves.
    """
    adjacent = has_target & (chebyshev(pred_pos, target_pos) <= 1)
    moved_f = target_moved.astype(still.dtype)
    cue = jnp.asarray(params.cue_dependence, still.dtype)

    damage = adjacent.astype(still.dtype) * params.damage * (1.0 - cue * (1.0 - moved_f))

    accrued = jnp.where(target_moved, 0.0, still + cue)
    still_next = jnp.where(adjacent, accrued, still)

    return damage, still_next, adjacent


def update_engagement(
    engage: Array,
    prev_distance: Array,
    distance: Array,
    target: Array,
    prev_target: Array,
) -> Array:
    """The engagement counter. **Q-021 rewrote this; the old rule was unsatisfiable in practice.**

        if not targeting k:      engage = 0
        if distance DECREASED:   engage += (prev_distance - distance)
        elif adjacent:           engage += 1
        elif distance UNCHANGED: engage unchanged
        else (INCREASED):        engage = 0

    The old rule was `engage + 1 if distance decreased, else 0`, and it had two independent faults.

    **It counted closing TICKS while the predator closes DISTANCE.** A sprinting predator crosses two cells
    per tick, so it arrived adjacent having accumulated fewer counts than cells crossed and never reached the
    threshold. Accumulating distance closed is what the counter was always trying to measure: has this
    predator deliberately approached me. Three cells of approach is the evidence; how many ticks it took is
    not the question.

    **And it reset on arrival, which made `engaged` an ARRIVAL EVENT rather than a STATE.** Once adjacent the
    distance cannot decrease, so the counter stopped and reset on the next tick. `stillness_engaged` is meant
    to measure behaviour DURING an engagement; under the old rule it measured behaviour on the single tick a
    predator arrived, which would have been near-meaningless even had it fired. Maintaining adjacency is
    continued pursuit, so it counts.

    **A hold does not reset; an increase does.** A predator that closes, pauses, closes is still approaching
    deliberately. Pausing is not evidence of coincidence; retreating is. That makes distance INCREASE the only
    reset path other than losing the target, which is why it has its own test.

    Requiring the target to be unchanged still stops a retarget from inheriting the previous pursuit's
    accumulated engagement.

    Measured consequence of the old rule: `engaged` was zero under every arrangement tried, which made
    `stillness_engaged` zero over zero and `accidental_stillness` and `flee_engaged` identically zero. Five
    descriptors across two strategy classes. See Q-021 and DEV-019.
    """
    same_target = (target == prev_target) & (target != NO_TARGET)
    decreased = distance < prev_distance
    adjacent = distance <= 1
    unchanged = distance == prev_distance

    # Priority order is the specification's, and `adjacent` deliberately sits ABOVE `unchanged` so that
    # holding contact accumulates rather than merely failing to reset.
    stepped = jnp.where(
        decreased,
        engage + (prev_distance - distance).astype(engage.dtype),
        jnp.where(adjacent, engage + 1, jnp.where(unchanged, engage, jnp.zeros_like(engage))),
    )
    return jnp.where(same_target, stepped, jnp.zeros_like(engage))


def step_predator(
    obstacles: Array,
    vis_table: Array,
    pred_pos: Array,
    pred_stamina: Array,
    target: Array,
    still: Array,
    engage: Array,
    prey_pos: Array,
    prey_alive: Array,
    prey_moved: Array,
    distant_draws: Array,
    wander_trigger: Array,
    wander_dir: Array,
    params,
) -> dict:
    """One predator tick: perceive, select, pursue, strike, disengage. Tick order steps 6 and 7.

    Returns a dict rather than a tuple because eight outputs positionally is how a caller silently swaps two
    of them. `damage_per_prey` is (K,) so the caller can apply it through `dynamics.update_health` without
    this module knowing anything about pools.

    `distant_draws` is (K,), one uniform per prey from `perceive.distant_detect` for this tick, drawn by the
    caller so that the draw site stays in one place and replay alignment has a single owner.
    """
    from src.env.perception import two_regime_report

    reported = jax.vmap(two_regime_report, in_axes=(None, None, 0, 0, 0, None, None, None))(
        vis_table,
        pred_pos,
        prey_pos,
        prey_moved,
        distant_draws,
        params.r_pred,
        params.r_near,
        params.p_distant_detect,
    )
    reported = reported & prey_alive

    distances = jax.vmap(chebyshev, in_axes=(None, 0))(pred_pos, prey_pos)
    prev_target = target
    prev_distance = jnp.where(
        target != NO_TARGET,
        distances[jnp.clip(target, 0, prey_pos.shape[0] - 1)],
        jnp.inf,
    )

    target = select_target(target, reported, distances)
    has_target = target != NO_TARGET
    safe_target = jnp.clip(target, 0, prey_pos.shape[0] - 1)
    target_pos = prey_pos[safe_target]

    pred_pos, sprinted = pursue(
        obstacles, pred_pos, target_pos, pred_stamina, has_target, wander_trigger, wander_dir, params
    )

    # Capture reads prey_moved from step 4 of this tick. Invariant I5.
    damage, still, adjacent = resolve_capture(
        pred_pos, target_pos, prey_moved[safe_target], still, has_target, params
    )

    distance = chebyshev(pred_pos, target_pos)
    engage = update_engagement(engage, prev_distance, distance, target, prev_target)

    # Disengage after the counter passes the threshold. Strictly greater than, per the build spec, so
    # disengage_steps of 5 means the sixth consecutive still tick breaks pursuit.
    broke_off = still > params.disengage_steps
    target = jnp.where(broke_off, jnp.int32(NO_TARGET).astype(target.dtype), target)
    still = jnp.where(broke_off, jnp.zeros_like(still), still)
    engage = jnp.where(broke_off, jnp.zeros_like(engage), engage)

    damage_per_prey = (
        jnp.zeros(prey_pos.shape[0], damage.dtype).at[safe_target].add(jnp.where(has_target, damage, 0.0))
    )

    # **The predator pays for its sprints.** DEV-014: this was computed nowhere and `env.py` wrote the old
    # value straight back, so stamina never moved off `s_max`. Nothing could have caught it: `pred_stamina`
    # appears in no module but this one and `env.py`, and the oracle did not model it, so E2 compared nothing
    # and agreed. That is the duplication class inverted, a quantity with no second implementation to
    # disagree with it.
    stamina = dynamics.update_stamina(pred_stamina, sprinted, params, burst_cost=params.s_burst_pred)

    return {
        "pred_pos": pred_pos,
        "target": target,
        "still": still,
        "engage": engage,
        "sprinted": sprinted,
        "stamina": stamina,
        "adjacent": adjacent,
        "damage_per_prey": damage_per_prey,
        "reported": reported,
    }
