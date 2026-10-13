"""The episode: state, reset, one tick, and the scanned loop.

Build step 7. This is where the twelve-step tick order becomes code, and it is the piece the build
specification calls the mechanical heart of the experiment.

### The tick order is fixed and step 7 reads step 4

    1  build observations            7  resolve capture, against prey_moved from step 4
    2  policy forward, argmax        8  update pools
    3  mask illegal actions          9  starvation bleed where energy is zero
    4  resolve prey movement        10  death checks, write-once cause
    5  resolve prey pick            11  resource regrowth
    6  predator perceive and move   12  accumulate descriptors, sample dump if recording

**Invariant I5 lives in the gap between 4 and 7.** Capture is evaluated against `prey_moved` as set at step
4 of the *same* tick, which is what makes stillness a live decision taken under the predator's current
position rather than a state read from a stale flag. Do not reorder, and do not recompute movement at step 7.

### Two actions are recorded, not one

`action_emitted` is what the network chose; `action_executed` is what the mask resolved it to. Physics uses
the executed action and `moved`; descriptors key on the emitted one. D-031.

Without this, a masked sprint is indistinguishable from a chosen stay, and since stamina drains during
chases, masked-sprint-to-stay is commoner in engaged ticks than baseline ticks by construction. That inflates
`conditional_stillness` itself, most for the agents that sprint most, so a pure fleeing strategy would produce
the signature of freezing. The artifact would manufacture the primary result from its own comparator.

### Memory: no scan output carries a T axis

D-008. Stacked observations across an episode would be 14.9 GB against 6.17 GB usable, 2.4 times over. So the
main path scans with **carry only and no `ys`**, and every descriptor is an online accumulator updated in the
carry.

The sampled trajectory dump is the deliberate exception and it is not a contradiction: it runs over a handful
of environments chosen before the run, not over all of them, which is exactly why it fits in megabytes while
the general case does not. `record=True` is a static flag, so the two paths compile separately and the main
path has no per-tick output at all.

### The sampled dump carries what the RENDERER needs

D-034. `food_grid` and `target_grid` are saved per tick and no descriptor consumes either. You cannot draw a
trajectory you did not save, and a schema chosen by asking what the statistics need is a schema that cannot be
rendered. About 3.3 KB per tick, 5.3 MB across four sampled episodes, free in advance and impossible to
retrofit.

### Replay mode runs inside the scan here for the first time

The tagged entropy source was built at step 2 and exercised for regrowth at step 3, neither inside a compiled
loop. Inside `lax.scan` the tick index comes from the carry rather than a Python counter, so the threading has
to be right or replay misaligns and gate E2 compares two different worlds, which is the exact failure the
tagged design exists to prevent. `tick` is carried explicitly and passed to every draw site.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from src.agent import network as net
from src.env import dynamics, predator, resources
from src.env import entropy as ent
from src.env.grid import resolve_move
from src.env.observation import build_observation

# Eight-direction offsets, indexed by action. Move actions are 0 to 7 and sprints 8 to 15, sharing the same
# direction table, which is why the sprint index is taken modulo eight.
DIRECTIONS = jnp.array([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], jnp.int32)


class StaticEnv(NamedTuple):
    """Closed over at construction and never traced. Obstacles never change during a run."""

    obstacles: Array
    vis_table: Array
    food_src: Array
    target_src: Array


class EnvState(NamedTuple):
    """One environment. Fixed shapes throughout; dead agents are masked, never removed."""

    prey_pos: Array
    prey_alive: Array
    prey_energy: Array
    prey_stamina: Array
    prey_health: Array
    prey_posture: Array
    prey_hidden: Array
    prey_score: Array
    prey_moved: Array
    prey_death_cause: Array
    prey_death_step: Array
    # Independent, and both can be true. `prey_death_cause` derives from them under the old precedence and is
    # kept for convenience. Recording only the derived cause under-counts predation among starving agents,
    # which are exactly the ones most likely to be caught. See the docstring on `resolve_deaths`.
    prey_died_starving: Array
    prey_died_predated: Array

    pred_pos: Array
    pred_stamina: Array
    pred_target: Array
    pred_engage: Array
    pred_still: Array
    pred_appearance: Array

    food_grid: Array
    target_grid: Array

    # Index into `params.d_spawn_relax` of the separation actually achieved at reset, or -1 if EVERY level
    # was infeasible and the fallback to any free cell was taken. Gate E11 checks both the separation and
    # that this is recorded, and measurement M4 reports the relaxation rate per configuration: a run that
    # relaxes often sits outside the valid envelope and is not comparable to one that does not.
    spawn_relax_level: Array


class Accumulators(NamedTuple):
    """Online sums. Every one is a scalar per prey, updated per tick, so nothing carries a T axis.

    The build spec's rule: any statistic the confirmatory analysis depends on must be an online accumulator,
    because the sampled trajectories are far too few to carry a population statistic.

    `accidental_still` is the one that is easy to omit and is the reason emitted-versus-executed was chosen
    over exclusion. It counts ticks where the policy emitted something other than stay and the agent did not
    move anyway, which is the exhaustion-to-freezing pathway made observable. D-031, and it was missing from
    the build spec's descriptor block until step 7.
    """

    engaged_ticks: Array
    still_engaged: Array
    accidental_still: Array
    unthreatened_ticks: Array
    still_unthreatened: Array
    alive_ticks: Array
    moved_ticks: Array

    # ---- the other four strategy classes ------------------------------------------------
    # **These were absent until step 8 and their absence was the defect, not their arrival.** The build spec
    # says the accumulator set is fixed by this loop and nothing unaccumulated can be recovered, and the set
    # above serves freezing completely and the other four classes not at all. The specification carried the
    # corrected descriptor balance, four formulas for freezing, three fleeing, three occlusion, four grouping,
    # one distance maintenance; the code was built to the shape the specification had BEFORE that correction.
    # See DEV-019.
    #
    # The sampled trajectory is not a substitute. `record=True` is a small, differently-selected dump, so
    # relying on it would measure freezing on every agent in every episode and the other four on a handful,
    # which is instrumentation with different statistical power per class. That is worse than none, because
    # the numbers look comparable and are not.
    visible_ticks: Array  # pred_visible AND alive. Denominator for mean_distance and nn_distance_threat
    flee_engaged: Array
    visible_unengaged_ticks: Array
    flee_visible_unengaged: Array
    near_ticks: Array  # a predator within r_pred, whether or not it can see
    occluded_near: Array
    far_ticks: Array  # kept explicit rather than derived as alive - near, so the identity is checkable
    occluded_far: Array
    dist_sum_visible: Array  # float. Chebyshev to the NEAREST predator, summed over visible ticks
    nn_sum_visible: Array  # float
    nn_visible_ticks: Array  # separate denominator: nearest-neighbour distance is undefined with no neighbour
    nn_sum_unseen: Array
    nn_unseen_ticks: Array
    density_sum: Array  # float. Living OTHER prey within r_group, summed over alive ticks

    # ---- near-conditioned response, the re-based conditioning ---------------------------
    # Run 20 measured the engagement-conditioned descriptors as SELF-DEFEATING for exactly the behaviours
    # they exist to measure: successful freezing prevents and ends engagement, erasing its own denominator,
    # and a natural fleer responds to visibility rather than the engagement flag, so the conditional
    # baseline subtracted its signal away. Five planted strategies, none detected. `near` is geometric,
    # nearest predator within r_pred, and no prey behaviour can erase it. The far-side fields are the
    # baseline arm of the conditional forms, geometric for the same reason. Owner ruling 2026-08-29,
    # D-046; the prediction-before-change record is in logs.md, per D-041, because this change lands
    # identically in the oracle and the differential is inert for it.
    still_near: Array
    accidental_still_near: Array  # D-031's chosen-versus-accidental split, under the new conditioning
    flee_near: Array
    still_far: Array
    flee_far: Array

    # ---- the EMITTED-direction flee event, D-048 ----------------------------------------
    # `flee_near` keys on the realized distance increase, and against a pursuing predator at matched speed
    # fleeing cannot increase distance: the run 22 diagnostic measured the planted fleer BELOW the forager
    # on it. The prey-attributable event is the emitted action, a move or sprint whose destination, from
    # the prey's start-of-tick position, strictly increases Chebyshev distance to the nearest predator's
    # start-of-tick position, which is the state the prey could perceive when it chose. Intent rather than
    # outcome, D-031's lesson a third time. Same near/far partition as the other near fields.
    away_near: Array
    away_far: Array


def zero_accumulators(k: int) -> Accumulators:
    """Counts are int32 and running sums are float32. Both start at exactly zero."""
    z = jnp.zeros((k,), jnp.int32)
    f = jnp.zeros((k,), jnp.float32)
    return Accumulators(
        engaged_ticks=z,
        still_engaged=z,
        accidental_still=z,
        unthreatened_ticks=z,
        still_unthreatened=z,
        alive_ticks=z,
        moved_ticks=z,
        visible_ticks=z,
        flee_engaged=z,
        visible_unengaged_ticks=z,
        flee_visible_unengaged=z,
        near_ticks=z,
        occluded_near=z,
        far_ticks=z,
        occluded_far=z,
        dist_sum_visible=f,
        nn_sum_visible=f,
        nn_visible_ticks=z,
        nn_sum_unseen=f,
        nn_unseen_ticks=z,
        density_sum=f,
        still_near=z,
        accidental_still_near=z,
        flee_near=z,
        still_far=z,
        flee_far=z,
        away_near=z,
        away_far=z,
    )


class TickRecord(NamedTuple):
    """One tick of a sampled trajectory. **Renderer-complete, per D-034.**

    `food_grid` and `target_grid` are here because the renderer needs them and no descriptor does. The
    obstacle grid is static and lives in `StaticEnv` rather than being repeated 400 times.
    """

    prey_pos: Array
    prey_alive: Array
    prey_posture: Array
    prey_moved: Array
    pred_pos: Array
    food_grid: Array
    target_grid: Array
    action_emitted: Array
    action_executed: Array


def _choose_masked(u: Array, mask: Array, n: int) -> Array:
    """`n` DISTINCT cells drawn uniformly from `mask`, using a fixed-size vector of uniforms.

    **Gumbel top-k.** Adding a Gumbel variate to each candidate's log-weight and taking the top `n` is
    equivalent to sampling `n` items without replacement proportional to those weights. Weights here are
    uniform over the mask, so the log-weight is constant and this reduces to taking the top `n` Gumbels among
    masked cells.

    **The point is that the DRAW COUNT is fixed.** `u` has one entry per cell whatever the mask contains, so
    the number of uniforms consumed does not depend on how many ladder rungs were tried, how many cells are
    feasible, or anything else the world does. A per-attempt draw would shift every subsequent tag index and
    reintroduce replay misalignment as a consequence of the fix that exists to prevent it.

    Excluded cells get -inf rather than a small score, so they can never be selected even if `n` exceeds the
    number of feasible cells; in that case the selection is arbitrary among excluded cells and the caller has
    already recorded a relaxation failure.
    """
    # `1 - u` avoids log(0) at u == 1; the clip guards the u == 0 end. Both are exact-endpoint cases that a
    # uniform draw can produce and that would otherwise give inf.
    gumbel = -jnp.log(-jnp.log(jnp.clip(u, 1e-20, 1.0 - 1e-7)))
    scores = jnp.where(mask, gumbel, -jnp.inf)
    return jax.lax.top_k(scores, n)[1]


def reset(entropy, static: StaticEnv, params) -> EnvState:
    """Spawn prey, then a predator at the first feasible separation on the relaxation ladder.

    The ladder exists because a categorical draw over an all-zero mask does not raise under static shapes; it
    returns index zero, which may be an obstacle or a cell adjacent to prey, producing a corrupted episode
    that looks normal. All levels are evaluated and selected with `where`, so it compiles.
    """
    h, w = static.obstacles.shape
    free = ~static.obstacles.reshape(-1)

    # **All three spawn draws go through the tagged entropy source.** DEV-022: these were `jax.random.choice`
    # and `jax.random.uniform` called directly, which CLAUDE.md forbids at a draw site, and the three tags
    # `spawn.prey`, `spawn.pred` and `spawn.appearance` were registered with no consumer. The consequence was
    # not merely conformance: threefry and PCG64 give different streams from the same seed, so the NumPy
    # oracle could not implement `reset` at all, which is why the differential's fork sat after it and
    # everything reset produced was invisible shared driver state. DEV-021 and DEV-022 were one finding.
    #
    # Every draw below is a FIXED-SIZE vector taken once. Nothing about the ladder, the mask, or the number
    # of feasible cells changes how many uniforms are consumed.
    u_prey = ent.draw_uniform(entropy, (h * w,), "spawn.prey", 0)
    u_pred = ent.draw_uniform(entropy, (h * w,), "spawn.pred", 0)

    idx = _choose_masked(u_prey, free, params.k)
    prey_pos = jnp.stack([idx // w, idx % w], axis=-1).astype(jnp.int32)

    cells = jnp.stack(jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij"), axis=-1).reshape(-1, 2)
    d_to_prey = jnp.min(jnp.max(jnp.abs(cells[:, None, :] - prey_pos[None, :, :]), axis=-1), axis=-1)

    def level_mask(d):
        return free & (d_to_prey >= d)

    masks = jnp.stack([level_mask(d) for d in params.d_spawn_relax])
    # **Feasible means at least `n_pred` cells, not merely one.** The draw below takes that many DISTINCT
    # cells, so a mask with fewer is as unsatisfiable as an empty one and fails the same silent way. This is
    # the reasoning `resources.place_sources` already records for source placement; the spawn ladder was
    # written with `any` because `n_pred` had only ever been 1, where the two tests coincide. DEV-016.
    nonempty = jnp.sum(masks, axis=-1) >= params.n_pred
    any_feasible = jnp.any(nonempty)
    level = jnp.argmax(nonempty)  # first feasible, argmax takes the lowest index
    mask = masks[level]
    # Fall back to any free cell only if every level is empty, which the ladder is designed to avoid.
    mask = jnp.where(any_feasible, mask, free)
    # **-1 is a distinct outcome, not level 0.** `argmax` over an all-False array returns 0, so without this
    # a total failure of the ladder would be recorded as "achieved the nominal separation", which is the most
    # favourable possible reading of the worst case. M4's relaxation rate would then under-report exactly the
    # configurations it exists to flag.
    spawn_relax_level = jnp.where(any_feasible, level, -1).astype(jnp.int32)

    # `replace=False`, matching the prey draw above. DEV-016: this was drawn WITH replacement, so at
    # `n_pred > 1` predators could stack on one cell and eight predators would not be eight threats. The
    # asymmetry was invisible while `n_pred` was 1 and increment 1 has never run another value.
    # `u_pred` was drawn ABOVE, before the ladder was evaluated, and is reused here whichever rung won. That
    # is what keeps the draw count independent of the ladder outcome.
    p_idx = _choose_masked(u_pred, mask, params.n_pred)
    pred_pos = jnp.stack([p_idx // w, p_idx % w], axis=-1).astype(jnp.int32)

    appearance = ent.draw_uniform(
        entropy, (params.n_pred, params.a_appearance), "spawn.appearance", 0
    ).astype(jnp.float32)

    # Resource layout is closed over at construction rather than drawn per episode, and consumes no tag. See
    # `resources.place_sources`, which records why: the differential injects the resulting arrays into both
    # implementations rather than each generating its own.
    k_food, k_target = jax.random.split(jax.random.fold_in(jax.random.key(0), 0), 2)
    food = resources.seed_grid(k_food, static.food_src, static.obstacles, params.cap_food, params.r_src)
    target = resources.seed_grid(
        k_target, static.target_src, static.obstacles, params.cap_target, params.r_src
    )

    k = params.k
    return EnvState(
        prey_pos=prey_pos,
        prey_alive=jnp.ones((k,), bool),
        prey_energy=jnp.full((k,), params.e_max, jnp.float32),
        prey_stamina=jnp.full((k,), params.s_max, jnp.float32),
        prey_health=jnp.full((k,), params.h_max, jnp.float32),
        prey_posture=jnp.zeros((k,), jnp.int32),
        # Hidden state is EXACTLY zero at every episode start and is never inherited. Invariant I3.
        prey_hidden=jnp.zeros((k, params.hd), jnp.float32),
        prey_score=jnp.zeros((k,), jnp.int32),
        prey_moved=jnp.zeros((k,), bool),
        prey_death_cause=jnp.zeros((k,), jnp.int32),
        prey_death_step=jnp.full((k,), -1, jnp.int32),
        prey_died_starving=jnp.zeros((k,), bool),
        prey_died_predated=jnp.zeros((k,), bool),
        pred_pos=pred_pos,
        pred_stamina=jnp.full((params.n_pred,), params.s_max, jnp.float32),
        pred_target=jnp.full((params.n_pred,), predator.NO_TARGET, jnp.int32),
        pred_engage=jnp.zeros((params.n_pred,), jnp.int32),
        pred_still=jnp.zeros((params.n_pred,), jnp.float32),
        pred_appearance=appearance,
        food_grid=food,
        target_grid=target,
        spawn_relax_level=spawn_relax_level,
    )


def network_policy(weights, obs: Array, hidden: Array, step_index) -> tuple[Array, Array]:
    """The evolved policy: forward the network over the prey batch and take a deterministic argmax.

    **This is the default and the only policy that carries weights.** The seam exists because the four run
    configurations declare a `policy` field and one of them is `random`, so the loop has to admit a policy
    that is not a network. Gate S9's floor and gate S2's hand-coded forager are the other two, both in
    `src/agent/baseline.py`.

    `step_index` is accepted and unused here. It exists for policies that need a fresh draw per tick and
    cannot carry their own counter, because a Python counter inside a scan increments at trace time only.

    **The seam creates no path from a hand-coded policy into the evolved lineage.** A policy consumes weights
    and returns actions; nothing here returns weights, so there is no direction in which a scripted policy
    could seed a genome. That holds CLAUDE.md section 1's refusal structurally rather than by discipline, in
    the same way invariant I1 holds by a function signature.
    """
    del step_index  # the network is a function of the observation and its own carried state, not of time
    logits, hidden = jax.vmap(net.forward, in_axes=(None, 0, 0))(weights, obs, hidden)
    return jax.vmap(net.select_action)(logits), hidden


def _direction_of(action: Array) -> Array:
    """Move and sprint share one direction table; sprints are the same eight offsets at index + 8."""
    is_dir = action <= dynamics.ACTION_SPRINT_LAST
    return jnp.where(is_dir, DIRECTIONS[jnp.mod(action, 8)], jnp.zeros((2,), jnp.int32))


def tick(
    state: EnvState,
    acc: Accumulators,
    weights: dict,
    static: StaticEnv,
    entropy,
    step_index,
    params,
    policy=network_policy,
) -> tuple[EnvState, Accumulators, TickRecord]:
    """One tick, in the build specification's twelve-step order. Single environment; batch by `vmap`.

    Returns the new state, the updated accumulators, and a record of this tick. The record is discarded by
    the caller unless it is recording, and because the main scan returns no `ys` at all, discarding it costs
    nothing and creates no per-tick array. D-008.
    """
    k = params.k

    # 1. observations for every prey. Dead agents still have theirs computed and masked, which is correct
    #    under static shapes and is listed in CLAUDE.md section 14 as a thing that looks wrong and is not.
    distant_draws = ent.draw_uniform(entropy, (k,), "perceive.distant_detect", step_index)
    # One draw per predator per tick for each, consumed whether or not the predator has a target, so the
    # draw count never depends on the state of the world. DEV-022.
    # **Shape from the STATE, not from `params.n_pred`.** The predator axis comes from the state arrays
    # everywhere else in this function, and taking it from params instead couples the draw to a field that a
    # caller can pass inconsistently. The differential did exactly that, ticking with the module default
    # while building state from the arm's params, and this draw was the first thing to read the field and
    # notice.
    n_pred = state.pred_pos.shape[0]
    wander_trigger = ent.draw_uniform(entropy, (n_pred,), "pred.wander.trigger", step_index)
    wander_dir = ent.draw_uniform(entropy, (n_pred,), "pred.wander.dir", step_index)

    def obs_for(i):
        return build_observation(
            obstacles=static.obstacles,
            food_grid=state.food_grid,
            target_grid=state.target_grid,
            vis_table=static.vis_table,
            self_pos=state.prey_pos[i],
            self_posture=state.prey_posture[i],
            self_pools=jnp.stack([state.prey_energy[i], state.prey_stamina[i], state.prey_health[i]]),
            prey_pos=state.prey_pos,
            prey_alive=state.prey_alive,
            prey_moved=state.prey_moved,
            prey_posture=state.prey_posture,
            pred_pos=state.pred_pos,
            pred_moved=jnp.zeros((params.n_pred,), bool),
            pred_appearance=state.pred_appearance,
            self_index=i,
            params=params,
        )

    obs = jax.vmap(obs_for)(jnp.arange(k))

    # 2. policy forward. This is action_emitted: what the POLICY chose, before any masking.
    # **`step_index` is passed to the policy, and a stateless policy needs it.** A policy that tried to keep
    # its own tick counter in Python would increment once during TRACING and then emit the same action every
    # tick under `lax.scan`. That is not hypothetical: the first random-floor adapter did exactly that, and
    # gate S9 passed anyway, because a CONSTANT policy also scores zero and also dies. The gate was satisfied
    # by the opposite of what it claimed to measure.
    action_emitted, hidden = policy(weights, obs, state.prey_hidden, step_index)

    # 3. mask. action_executed is what the environment allowed. D-031: both are kept.
    action_executed, _mask_event = dynamics.mask_actions(
        action_emitted, state.prey_energy, state.prey_stamina, params
    )

    # 4. movement. prey_moved is set from whether the cell actually changed, not from intent.
    directions = jax.vmap(_direction_of)(action_executed)
    sprinting = dynamics.is_burst(action_executed)
    proposed = jax.vmap(resolve_move, in_axes=(None, 0, 0, 0))(
        static.obstacles, state.prey_pos, directions, sprinting
    )
    prey_pos = jnp.where(state.prey_alive[:, None], proposed, state.prey_pos)
    prey_moved = jnp.any(prey_pos != state.prey_pos, axis=-1) & state.prey_alive

    # 5. pick, which scores and feeds immediately. No inventory.
    picking = (action_executed == dynamics.ACTION_PICK) & state.prey_alive
    food_grid, target_grid, gained, ate = resources.resolve_pick(
        state.food_grid, state.target_grid, prey_pos, state.prey_alive, picking
    )
    prey_score = state.prey_score + gained

    # 6. predator perceive, select, move.
    # **Mapped over the predator axis, not indexed at [0].** DEV-013: the previous form hardcoded exactly
    # one predator, so `n_pred = 0` raised `IndexError` and `no_predator_evolve` could not run at all, while
    # `configs/default.py` claimed the shapes were "written generally". That claim had no consumer and was
    # false the whole time, because no value other than 1 had ever been executed.
    #
    # With `n_pred = 0` every array below has a leading axis of length zero and every reduction over it
    # returns the correct empty result by construction. **That is why the predator-naive lineage does not
    # need a mask**: there is nothing present to mask, so no path can be missed. A masked predator would
    # define "no predator" by the completeness of the masking, and any missed path would contaminate the
    # null silently rather than erroring, shifting every emergence threshold in the same direction.
    #
    # `params` is closed over rather than passed: it is a plain dataclass with no pytree registration, so
    # `vmap` cannot flatten it even at `in_axes=None`. Same reason as in gate E9.
    def _one_predator(pos, stamina, target, still, engage, w_trigger, w_dir):
        return predator.step_predator(
            static.obstacles,
            static.vis_table,
            pos,
            stamina,
            target,
            still,
            engage,
            prey_pos,
            state.prey_alive,
            prey_moved,  # 7 reads step 4 of THIS tick. Invariant I5.
            distant_draws,
            w_trigger,
            w_dir,
            params,
        )

    pred = jax.vmap(_one_predator)(
        state.pred_pos,
        state.pred_stamina,
        state.pred_target,
        state.pred_still,
        state.pred_engage,
        wander_trigger,
        wander_dir,
    )

    # 8, 9. pools. Health takes damage, regenerates, and bleeds where energy ended the tick at zero.
    energy = dynamics.update_energy(state.prey_energy, sprinting, ate, params)
    stamina = dynamics.update_stamina(state.prey_stamina, sprinting, params)
    # **Sum the MAGNITUDE over predators, and keep the boolean only for the death cause. DEV-031.**
    # `damaged` is the right input to `resolve_deaths`, which asks whether predation damage occurred rather
    # than how much. It was the wrong input to `update_health`, which needs the amount: reducing to a sign
    # discarded the graded damage `resolve_capture` computes and left `cue_dependence` binary.
    #
    # Summed rather than maxed, so two predators reaching the same prey do twice the damage, which is what
    # `damage_per_prey` being a per-predator array means. Reduces to all-zero at n_pred = 0.
    damage_total = jnp.sum(pred["damage_per_prey"], axis=0)
    damaged = damage_total > 0.0
    health = dynamics.update_health(state.prey_health, damage_total, energy, params)
    posture = dynamics.update_posture(state.prey_posture, action_executed)

    # 10. death, write-once.
    alive, cause, death_step, died_starving, died_predated = dynamics.resolve_deaths(
        state.prey_alive,
        health,
        energy,
        damaged,
        state.prey_death_cause,
        state.prey_death_step,
        state.prey_died_starving,
        state.prey_died_predated,
        step_index,
    )

    # 11. regrowth, both grids, each with its own tagged draws.
    food_grid, _ = resources.regrow(
        entropy,
        food_grid,
        static.food_src,
        static.obstacles,
        params.cap_food,
        params.r_src,
        params.p_food,
        step_index,
        "regrow.food.trigger",
        "regrow.food.cell",
    )
    target_grid, _ = resources.regrow(
        entropy,
        target_grid,
        static.target_src,
        static.obstacles,
        params.cap_target,
        params.r_src,
        params.p_target,
        step_index,
        "regrow.target.trigger",
        "regrow.target.cell",
    )

    # 12. accumulate. Online sums only; nothing here grows with T.
    # Per predator, then reduced: a prey is engaged if ANY predator is engaged with it, and is visible if
    # ANY predator reports it. Both collapse to all-False at n_pred = 0, which is the correct meaning of a
    # predator-free world rather than a special case written for it.
    engaged_by = (
        (pred["target"][:, None] == jnp.arange(k))
        & pred["adjacent"][:, None]
        & (pred["engage"][:, None] >= params.e_min)
    )
    # **Behavioural accumulators use ALIVE AT THE START OF THE TICK, not alive after deaths resolve.**
    # The tick order is action, movement, capture, deaths, so a prey that dies this tick DID act, and that
    # action is a genuine choice made under maximal threat. Excluding it biases every measurement toward
    # less-threatened ticks, and on the threat-conditioned denominators, which run to about two ticks per
    # prey, it removes roughly half the data. See `docs/analysis-structure.md`.
    #
    # `alive` below remains the post-death mask and is what `new_state` stores. Only the accumulators move.
    acting = state.prey_alive

    engaged = jnp.any(engaged_by, axis=0) & acting
    pred_visible = jnp.any(pred["reported"], axis=0)
    unthreatened = (~pred_visible) & acting

    emitted_stay = action_emitted == dynamics.ACTION_STAY

    # ---- quantities the other four strategy classes are computed from ---------------------
    # `FAR` stands in for "no predator" and "no living neighbour". It exceeds any achievable Chebyshev
    # distance on the grid, so a comparison against a real radius is always False and `jnp.min` over an EMPTY
    # predator axis returns it rather than raising. That is what makes n_pred = 0 a no-op here too.
    far_sentinel = jnp.int32(static.obstacles.shape[0] + static.obstacles.shape[1])

    def _nearest_pred(prey_p, pred_p):
        d = jnp.max(jnp.abs(pred_p[:, None, :] - prey_p[None, :, :]), axis=-1)
        return jnp.min(d, axis=0, initial=far_sentinel)

    # Both positions are in scope, so "did the distance increase" needs no new state field: `state.*` is the
    # previous tick and the locals are this one.
    d_prev = _nearest_pred(state.prey_pos, state.pred_pos)
    d_now = _nearest_pred(prey_pos, pred["pred_pos"])
    dist_increased = d_now > d_prev

    near = d_now <= params.r_pred
    visible_alive = pred_visible & acting
    unengaged_visible = visible_alive & ~engaged

    # The emitted-direction flee event, D-048. Intent from the start-of-tick state, so pursuit cannot
    # erase it: destination is start position plus the emitted direction, unclamped, compared against the
    # nearest predator's start-of-tick position. `d_prev` above is already that start-to-start distance.
    emit_directional = dynamics.is_basic_move(action_emitted) | dynamics.is_burst(action_emitted)
    emit_dir = (
        jnp.where(
            action_emitted <= dynamics.ACTION_MOVE_LAST,
            action_emitted,
            action_emitted - dynamics.ACTION_SPRINT_FIRST,
        )
        % 8
    )
    emit_dest = state.prey_pos + DIRECTIONS[emit_dir]
    d_dest = _nearest_pred(emit_dest, state.pred_pos)
    away = emit_directional & (d_dest > d_prev)

    # Nearest LIVING OTHER prey, and the local density, both from the post-move positions.
    pair = jnp.max(jnp.abs(prey_pos[:, None, :] - prey_pos[None, :, :]), axis=-1)
    other_alive = acting[None, :] & ~jnp.eye(k, dtype=bool)
    nn = jnp.min(jnp.where(other_alive, pair, far_sentinel), axis=1, initial=far_sentinel)
    has_neighbour = jnp.any(other_alive, axis=1) & acting
    density = jnp.sum(other_alive & (pair <= params.r_group), axis=1)

    acc = Accumulators(
        engaged_ticks=acc.engaged_ticks + engaged,
        still_engaged=acc.still_engaged + (emitted_stay & engaged),
        # The exhaustion-to-freezing pathway: the policy wanted something else and the agent did not move.
        accidental_still=acc.accidental_still + (~emitted_stay & ~prey_moved & engaged),
        unthreatened_ticks=acc.unthreatened_ticks + unthreatened,
        still_unthreatened=acc.still_unthreatened + (emitted_stay & unthreatened),
        alive_ticks=acc.alive_ticks + acting,
        moved_ticks=acc.moved_ticks + prey_moved,
        # fleeing
        visible_ticks=acc.visible_ticks + visible_alive,
        flee_engaged=acc.flee_engaged + (dist_increased & engaged),
        visible_unengaged_ticks=acc.visible_unengaged_ticks + unengaged_visible,
        flee_visible_unengaged=acc.flee_visible_unengaged + (dist_increased & unengaged_visible),
        # occlusion seeking
        near_ticks=acc.near_ticks + (near & acting),
        occluded_near=acc.occluded_near + (near & ~pred_visible & acting),
        far_ticks=acc.far_ticks + (~near & acting),
        occluded_far=acc.occluded_far + (~near & ~pred_visible & acting),
        # distance maintenance
        dist_sum_visible=acc.dist_sum_visible + jnp.where(visible_alive, d_now, 0).astype(jnp.float32),
        # grouping
        nn_sum_visible=acc.nn_sum_visible
        + jnp.where(has_neighbour & visible_alive, nn, 0).astype(jnp.float32),
        nn_visible_ticks=acc.nn_visible_ticks + (has_neighbour & visible_alive),
        nn_sum_unseen=acc.nn_sum_unseen + jnp.where(has_neighbour & unthreatened, nn, 0).astype(jnp.float32),
        nn_unseen_ticks=acc.nn_unseen_ticks + (has_neighbour & unthreatened),
        density_sum=acc.density_sum + jnp.where(acting, density, 0).astype(jnp.float32),
        # near-conditioned response, D-046: the same events as the engaged forms, conditioned on geometry
        # the behaviour cannot erase. The denominators are near_ticks and far_ticks above.
        still_near=acc.still_near + (emitted_stay & near & acting),
        accidental_still_near=acc.accidental_still_near + (~emitted_stay & ~prey_moved & near & acting),
        flee_near=acc.flee_near + (dist_increased & near & acting),
        still_far=acc.still_far + (emitted_stay & ~near & acting),
        flee_far=acc.flee_far + (dist_increased & ~near & acting),
        away_near=acc.away_near + (away & near & acting),
        away_far=acc.away_far + (away & ~near & acting),
    )

    new_state = EnvState(
        prey_pos=prey_pos,
        prey_alive=alive,
        prey_energy=energy,
        prey_stamina=stamina,
        prey_health=health,
        prey_posture=posture,
        prey_hidden=hidden,
        prey_score=prey_score,
        prey_moved=prey_moved,
        prey_death_cause=cause,
        prey_death_step=death_step,
        prey_died_starving=died_starving,
        prey_died_predated=died_predated,
        pred_pos=pred["pred_pos"],
        pred_stamina=pred["stamina"],  # DEV-014: was state.pred_stamina, so it never changed
        pred_target=pred["target"],
        pred_engage=pred["engage"],
        pred_still=pred["still"],
        pred_appearance=state.pred_appearance,
        food_grid=food_grid,
        target_grid=target_grid,
        spawn_relax_level=state.spawn_relax_level,  # fixed at reset, carried unchanged
    )

    record = TickRecord(
        prey_pos=prey_pos,
        prey_alive=alive,
        prey_posture=posture,
        prey_moved=prey_moved,
        pred_pos=new_state.pred_pos,
        food_grid=food_grid,  # renderer needs it, no descriptor does. D-034
        target_grid=target_grid,  # same
        action_emitted=action_emitted,
        action_executed=action_executed,
    )
    return new_state, acc, record


def run_episode(
    key: Array,
    weights: dict,
    static: StaticEnv,
    entropy,
    params,
    record: bool = False,
    policy=network_policy,
):
    """Scan the episode. Returns (final_state, accumulators) or, when recording, plus the trajectory.

    **`record` is a static Python bool, so the two paths compile separately.** With it false the scan returns
    no `ys` at all, so nothing carries a T axis and D-008 holds. With it true the caller is running the small
    fixed set of sampled environments and the per-tick dump is the point.

    `tick_index` is carried rather than closed over, because inside a scan the tick comes from the carry and
    replay indexes draws by tag and tick. Getting that wrong misaligns replay and makes gate E2 compare two
    different worlds, which is the failure the tagged entropy design exists to prevent.
    """
    # **Replay must reproduce the layout, or the fork stays after reset.** In replay mode the shared source
    # supplies the spawn draws; in production each episode needs its OWN layout, so the per-episode key
    # becomes the source. The branch is on a Python type and is therefore static under jit.
    reset_source = entropy if isinstance(entropy, ent.ReplaySource) else ent.PRNGSource(key)
    state = reset(reset_source, static, params)
    acc = zero_accumulators(params.k)

    def body(carry, _):
        st, ac, t = carry
        st, ac, rec = tick(st, ac, weights, static, entropy, t, params, policy)
        out = rec if record else None
        return (st, ac, t + 1), out

    (state, acc, _), ys = jax.lax.scan(body, (state, acc, jnp.int32(0)), None, length=params.t_horizon)
    return (state, acc, ys) if record else (state, acc)
