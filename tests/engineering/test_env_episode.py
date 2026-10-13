"""The episode loop: tick order, invariant I5, the two action records, and the memory constraint.

**The assertion that matters most is that no scan output carries a T axis in the main path.** D-008: stacked
observations across an episode would be 14.9 GB against 6.17 GB usable. The scan runs with carry only, so the
constraint is structural rather than a matter of discipline, and `test_the_main_path_returns_no_per_tick_array`
fails if a `ys` output is ever added.

Second is that replay mode works **inside** the scan. The tagged entropy source was built at step 2 and
exercised for regrowth at step 3, neither inside a compiled loop, and inside `lax.scan` the tick index comes
from the carry rather than a Python counter. A threading mistake there misaligns replay silently.
"""

import jax
import jax.numpy as jnp
import pytest

from configs.default import DEFAULT
from src.agent import network as net
from src.env import dynamics, predator, resources
from src.env import env as E
from src.env.entropy import PRNGSource, make_replay_source
from src.env.perception import build_vis_table

P = DEFAULT.__class__(h=14, w=14, k=4, t_horizon=60, n_food_src=3, n_target_src=3)


@pytest.fixture(scope="module")
def static():
    key = jax.random.key(0)
    obstacles = jnp.zeros((P.h, P.w), bool).at[5, 5].set(True).at[6, 6].set(True)
    food_src, target_src, _ = resources.place_sources(key, obstacles, P)
    return E.StaticEnv(
        obstacles=obstacles,
        vis_table=build_vis_table(obstacles, P.r),
        food_src=food_src,
        target_src=target_src,
    )


@pytest.fixture(scope="module")
def weights():
    return net.init_params(jax.random.key(1), P)


def _entropy():
    """A replay source sized for the whole episode, so every draw site is pre-generated and indexable."""
    return make_replay_source(
        jax.random.key(7),
        {
            # The three spawn draws are taken once at step 0, so the leading axis is 1. Their SIZE does not
            # depend on the ladder outcome, which is what keeps replay aligned. DEV-022.
            "spawn.prey": (1, P.h * P.w),
            "spawn.pred": (1, P.h * P.w),
            "spawn.appearance": (1, P.n_pred * P.a_appearance),
            "perceive.distant_detect": (P.t_horizon, P.k),
            # One per predator per tick, drawn whether or not it has a target. DEV-022.
            "pred.wander.trigger": (P.t_horizon, P.n_pred),
            "pred.wander.dir": (P.t_horizon, P.n_pred),
            "regrow.food.trigger": (P.t_horizon, P.n_food_src),
            "regrow.food.cell": (P.t_horizon, P.n_food_src),
            "regrow.target.trigger": (P.t_horizon, P.n_target_src),
            "regrow.target.cell": (P.t_horizon, P.n_target_src),
        },
    )


def _engaged_state(static, seed=5):
    """A start state where predation is GUARANTEED, by placing the predator next to a prey.

    **Falsification found that the natural reset never produces predation in a short episode.** Measured: the
    predator acquired a target on 0 of 30 ticks and never got closer than 8 cells, so capture, the engagement
    accumulators and the distant-detection draw were all inert. Four breaks survived because of it, including
    a stale `prey_moved` violating invariant I5.

    That is gate E3's inertness for the third time, now at the scenario level rather than the branch or value
    level: the paths existed, the scenario never reached them. Constructing the situation is the fix, and the
    coverage assertions below are what stop it recurring silently.
    """
    st = E.reset(PRNGSource(jax.random.key(seed)), static, P)
    target = st.prey_pos[0]
    # Not adjacent, because `engage` only accumulates while the distance DECREASES, so a predator that starts
    # adjacent never builds engagement. It has to approach.
    #
    # **The offset is `r_near`, and that makes detection DETERMINISTIC rather than likely.**
    #
    # The two-regime rule is `report = visible AND (near OR moved OR draw < p_distant_detect)`. The predator
    # previously started at exactly `r_pred`, outside `r_near`, so reporting depended on the prey happening to
    # move or on a 25 percent draw. An untrained policy emits a near-constant action and drifts steadily in
    # one direction, so prey 0 walked out of sensing range while the predator, never having acquired a target,
    # never pursued. Deadlock, and every predator-conditioned accumulator read zero for the whole episode.
    #
    # It held only because the old `reset` happened to produce compatible geometry. DEV-022 changed the
    # positions and it stopped holding, which is a fixture resting on a threshold it never asserted.
    #
    # Inside `r_near` the near clause fires on its own, so detection no longer depends on the draw or on the
    # prey's behaviour. Still strictly beyond adjacency, so `engage` can accumulate as the predator closes.
    offset = P.r_near
    near = jnp.array([target[0], jnp.clip(target[1] + offset, 0, P.w - 1)], jnp.int32)
    st = st._replace(pred_pos=near[None, :])

    # **The precondition, asserted rather than assumed.** A fixture whose validity depends on a property must
    # check that property, because fixtures get edited by people who do not know what they are load-bearing
    # for. `clip` at the grid edge can silently shorten the offset, which would make the predator adjacent and
    # quietly disable engagement accumulation.
    d = int(jnp.min(jnp.max(jnp.abs(st.pred_pos[:, None, :] - st.prey_pos[None, :, :]), axis=-1)))
    assert 1 < d <= P.r_near, (
        f"the constructed predator sits {d} from the nearest prey. It must be within r_near {P.r_near}, where "
        "the two-regime rule reports it regardless of motion or the detection draw, and strictly beyond "
        "adjacency so engagement can accumulate. Outside that window every predator-conditioned accumulator "
        "reads zero and every assertion built on them passes trivially."
    )
    return st


# ------------------------------------------------------------------------ reset --


def test_reset_places_everyone_on_free_cells(static):
    s = E.reset(PRNGSource(jax.random.key(2)), static, P)
    for k in range(P.k):
        assert not bool(static.obstacles[s.prey_pos[k][0], s.prey_pos[k][1]]), "prey spawned in a wall"
    assert not bool(static.obstacles[s.pred_pos[0][0], s.pred_pos[0][1]]), "predator spawned in a wall"


def test_reset_separates_the_predator_from_every_prey(static):
    """The spawn ladder exists so no episode opens with a capture, which would bias every measurement."""
    s = E.reset(PRNGSource(jax.random.key(3)), static, P)
    d = jnp.min(jnp.max(jnp.abs(s.prey_pos - s.pred_pos[0]), axis=-1))
    assert int(d) >= min(P.d_spawn_relax), (
        f"predator spawned {int(d)} from the nearest prey, below the ladder floor {min(P.d_spawn_relax)}"
    )
    assert int(d) > P.r_pred, "the episode opens with the predator already able to see prey"


def test_hidden_state_is_exactly_zero_at_reset(static):
    """Invariant I3. Not small, exactly zero, and asserted here as well as in the network tests."""
    s = E.reset(PRNGSource(jax.random.key(4)), static, P)
    assert bool(jnp.all(s.prey_hidden == 0.0))


def test_reset_fills_pools_and_clears_death_records(static):
    s = E.reset(PRNGSource(jax.random.key(5)), static, P)
    assert bool(jnp.all(s.prey_energy == P.e_max))
    assert bool(jnp.all(s.prey_health == P.h_max))
    assert bool(jnp.all(s.prey_alive))
    assert bool(jnp.all(s.prey_death_cause == 0))
    assert bool(jnp.all(s.prey_death_step == -1))


# ---------------------------------------------------------------- the memory constraint --


def test_the_main_path_returns_no_per_tick_array(static, weights):
    """**D-008, asserted structurally.** Any scan output with a T axis is a defect until proven otherwise.

    The main path returns only the final state and the accumulators. If a `ys` output is ever added, every
    array here gains a leading `t_horizon` axis and this fails, which is the point: the constraint is checked
    rather than remembered.
    """
    out = E.run_episode(jax.random.key(6), weights, static, _entropy(), P)
    assert len(out) == 2, "the main path should return exactly (state, accumulators)"
    state, acc = out

    for name, leaf in zip(state._fields, jax.tree_util.tree_leaves(state), strict=False):
        assert leaf.ndim == 0 or leaf.shape[0] != P.t_horizon, (
            f"state field {name} has shape {leaf.shape}, whose leading axis is t_horizon. That is a T axis "
            "in a scan output, which D-008 forbids"
        )
    for name, leaf in zip(acc._fields, jax.tree_util.tree_leaves(acc), strict=False):
        assert leaf.shape == (P.k,), f"accumulator {name} has shape {leaf.shape}, expected per-prey scalars"


def test_recording_produces_a_renderer_complete_trajectory(static, weights):
    """D-034. The dump must carry what the renderer needs, not only what the descriptors need."""
    state, acc, traj = E.run_episode(jax.random.key(6), weights, static, _entropy(), P, record=True)

    assert traj.food_grid.shape == (P.t_horizon, P.h, P.w), (
        "the food grid must be saved per tick; no descriptor consumes it and the renderer cannot draw "
        "without it"
    )
    assert traj.target_grid.shape == (P.t_horizon, P.h, P.w)
    assert traj.prey_pos.shape == (P.t_horizon, P.k, 2)
    assert traj.action_emitted.shape == (P.t_horizon, P.k)
    assert traj.action_executed.shape == (P.t_horizon, P.k)


def test_the_renderer_can_actually_draw_a_recorded_tick(static, weights):
    """The claim in D-034 made executable rather than asserted.

    A schema is renderer-complete only if the renderer accepts it. This calls the real renderer on a real
    recorded tick, which is the only check that cannot pass while the schema is subtly short.
    """
    from src.render.frame import render_state

    _, _, traj = E.run_episode(jax.random.key(6), weights, static, _entropy(), P, record=True)
    t = 5
    img = render_state(
        obstacles=static.obstacles,
        food_grid=traj.food_grid[t],
        target_grid=traj.target_grid[t],
        prey_pos=traj.prey_pos[t],
        prey_alive=traj.prey_alive[t],
        pred_pos=traj.pred_pos[t],
        r_src=P.r_src,
    )
    assert img.shape[0] == P.h * 12 and img.shape[1] == P.w * 12
    assert float(img.max()) > 0.0


# ------------------------------------------------------------- emitted versus executed --


def test_both_actions_are_recorded_and_can_differ(static, weights):
    """D-031. Reserved actions are always masked, so emitted and executed must differ somewhere."""
    _, _, traj = E.run_episode(jax.random.key(8), weights, static, _entropy(), P, record=True)

    emitted_reserved = (traj.action_emitted >= 20).sum()
    if int(emitted_reserved) > 0:
        differ = (traj.action_emitted != traj.action_executed).sum()
        assert int(differ) > 0, (
            "a reserved action was emitted but emitted and executed never differ, so the mask is not being "
            "recorded separately and every descriptor keyed on stay would be inflated"
        )
    assert bool(jnp.all(traj.action_executed < P.n_actions))
    assert bool(jnp.all(traj.action_emitted < P.n_actions))


# ----------------------------------------------------------------------- replay in scan --


def test_replay_mode_is_deterministic_inside_the_scan(static, weights):
    """**The step 7 risk no earlier rule covered.**

    The entropy source has never run inside a compiled loop. Inside `lax.scan` the tick index comes from the
    carry, so a threading mistake misaligns replay and gate E2 would compare two different worlds while both
    stayed internally consistent.

    Two runs from the same replay source must be identical in every field.
    """
    a_state, a_acc = E.run_episode(jax.random.key(9), weights, static, _entropy(), P)
    b_state, b_acc = E.run_episode(jax.random.key(9), weights, static, _entropy(), P)

    for x, y in zip(jax.tree_util.tree_leaves(a_state), jax.tree_util.tree_leaves(b_state), strict=True):
        assert bool(jnp.all(x == y)), "replay is not deterministic inside the scan"
    for x, y in zip(jax.tree_util.tree_leaves(a_acc), jax.tree_util.tree_leaves(b_acc), strict=True):
        assert bool(jnp.all(x == y))


def test_the_distant_detection_draw_actually_reaches_the_scan():
    """Power check on the determinism test above, and the first version of it was inert.

    **That first version compared two whole episodes under different replay streams and they came out
    identical, which is gate E3's failure wearing new clothes.** `seed_grid` fills every source patch to its
    cap, regrowth only fires when `local < cap`, and a random policy picks almost nothing in a short episode,
    so no regrow draw ever changed anything. The predator likewise never got a distant sighting over 25 ticks
    on a small grid, so `perceive.distant_detect` never mattered either. Two different streams, identical
    worlds, and a determinism test that would have passed over a source nothing consumed.

    This version constructs the situation instead of hoping for it: a prey placed beyond `r_near` but inside
    `r_pred`, standing still, with clear line of sight. Under that arrangement the two-regime rule consults
    the draw and nothing else can decide the outcome, so an all-detect stream and a never-detect stream must
    disagree about whether the prey is reported.
    """
    obstacles = jnp.zeros((14, 14), bool)
    vis = build_vis_table(obstacles, P.r_pred)

    pred_pos = jnp.array([7, 7], jnp.int32)
    prey_pos = jnp.array([[7, 11]], jnp.int32)  # distance 4: beyond r_near of 2, inside r_pred of 4
    assert P.r_near < 4 <= P.r_pred, "fixture error: this separation is not in the distant regime"

    def reported_with(draw_value):
        out = predator.step_predator(
            obstacles,
            vis,
            pred_pos,
            jnp.asarray(100.0),
            jnp.int32(predator.NO_TARGET),
            jnp.asarray(0.0, jnp.float32),
            jnp.int32(0),
            prey_pos,
            jnp.array([True]),
            jnp.array([False]),
            jnp.array([draw_value], jnp.float32),
            jnp.asarray(1.0),  # wander_trigger: never wanders, so this stays a test of detection only
            jnp.asarray(0.0),  # wander_dir
            P,
        )
        return bool(out["reported"][0])

    assert reported_with(0.0), "an always-firing draw should report a still distant prey"
    assert not reported_with(1.0), "a never-firing draw should not report a still distant prey"


# -------------------------------------------------------------------- accumulators --


# Accumulators that are RUNNING SUMS rather than tick counts, so the horizon is not their bound. Named
# explicitly and negatively: everything NOT in this set is checked against the strict count bound, so a new
# accumulator defaults to the tighter assertion rather than escaping it. That is the maintenance form of the
# polarity rule, and this test is why it matters: it was written when every accumulator was a count, and it
# broke the moment the set grew a second kind. Adding the new fields to a positive whitelist would have to be
# repeated every time; this way only a genuinely new KIND needs an edit.
SUM_ACCUMULATORS = frozenset({"dist_sum_visible", "nn_sum_visible", "nn_sum_unseen", "density_sum"})


def test_accumulators_are_bounded_by_the_horizon(static, weights):
    """Counts cannot exceed the horizon. Sums cannot exceed the horizon times their per-tick maximum.

    The distinction is real rather than bookkeeping: a count above the horizon means a tick was counted twice,
    and a sum above its own bound means a distance was accumulated that the grid cannot contain.
    """
    _, acc = E.run_episode(jax.random.key(12), weights, static, _entropy(), P)
    span = max(P.h, P.w)
    for name in acc._fields:
        leaf = getattr(acc, name)
        assert bool(jnp.all(leaf >= 0)), f"{name} went negative"
        if name in SUM_ACCUMULATORS:
            # Per tick: a distance cannot exceed the grid span, and density cannot exceed the other prey.
            per_tick = (P.k - 1) if name == "density_sum" else span
            assert bool(jnp.all(leaf <= P.t_horizon * per_tick)), (
                f"{name} exceeded {P.t_horizon} ticks times its per-tick maximum of {per_tick}"
            )
        else:
            assert bool(jnp.all(leaf <= P.t_horizon)), f"{name} exceeded the horizon"

    assert SUM_ACCUMULATORS <= set(acc._fields), (
        f"SUM_ACCUMULATORS names fields that no longer exist: {SUM_ACCUMULATORS - set(acc._fields)}. "
        "A stale exclusion silently weakens the count bound for nothing."
    )

    assert bool(jnp.all(acc.still_engaged <= acc.engaged_ticks)), (
        "more still engaged ticks than engaged ticks"
    )
    assert bool(jnp.all(acc.accidental_still <= acc.engaged_ticks))
    assert bool(jnp.all(acc.still_unthreatened <= acc.unthreatened_ticks))


def test_accidental_stillness_is_present_in_the_carry(static, weights):
    """**Existence and shape only. The name says so, because the body cannot say more yet.**

    The descriptor that makes the exhaustion-to-freezing pathway observable was absent from the build spec's
    descriptor block until step 7, and the accumulator set is fixed by this loop, so anything not accumulated
    here cannot be recovered afterwards. D-031 and D-034.

    This test was previously called `..._is_accumulated_at_all` while asserting only `hasattr` and `.shape`.
    That is why the `drop_accidental` falsification break survived: the test that should have caught it made a
    claim in its name that its body never checked. **A test name is a stated fact with no consumer**, so it
    drifts exactly as a line count in prose drifts, and nothing fails when it stops being true.

    Asserting actual accumulation is blocked on **Q-021**, not on effort: `accidental_still` is gated on
    `engaged`, `engaged` measures zero under every arrangement tried, so the quantity is identically zero and
    a positive assertion would be asserting a state the current definitions do not admit.
    """
    _, acc = E.run_episode(jax.random.key(13), weights, static, _entropy(), P)
    assert hasattr(acc, "accidental_still")
    assert acc.accidental_still.shape == (P.k,)


def test_score_only_ever_increases(static, weights):
    """`prey_score` is the sole fitness input. Invariant I1 depends on it being a clean count.

    **Checked tick by tick, because the end-state form of this test asserted `>= 0`**, which is trivially
    true of a counter that starts at zero and has no decrement anywhere. That would pass against an
    implementation that reset the score, double-counted, or awarded on the wrong event, so it constrained
    nothing. Monotonicity is the property the name claims and it is directly observable per tick.
    """
    state = E.reset(PRNGSource(jax.random.key(14)), static, P)
    assert bool(jnp.all(state.prey_score == 0))

    acc = E.zero_accumulators(P.k)
    ent = _entropy()
    previous = state.prey_score
    for t in range(P.t_horizon):
        state, acc, _ = E.tick(state, acc, weights, static, ent, jnp.int32(t), P)
        assert bool(jnp.all(state.prey_score >= previous)), f"prey_score decreased at tick {t}"
        previous = state.prey_score


def test_the_episode_actually_exercises_predation(static, weights):
    """**The coverage assertion whose absence let four falsification breaks survive.**

    Without predation, capture never runs, `engaged_ticks` stays zero, `accidental_still` cannot be non-zero,
    and the distant-detection draw never decides anything. Every assertion about those paths then passes over
    a path that did not execute.
    """
    state = _engaged_state(static)
    acc = E.zero_accumulators(P.k)
    ent = _entropy()
    saw_target = saw_damage = 0
    for t in range(P.t_horizon):
        state, acc, _ = E.tick(state, acc, weights, static, ent, jnp.int32(t), P)
        saw_target += int(state.pred_target[0] != -1)
        saw_damage += int(bool(jnp.any(state.prey_health < P.h_max)))

    assert saw_target > 0, "the predator never acquired a target, so the capture path is untested"
    assert saw_damage > 0, "no prey was ever damaged, so capture never resolved"

    # `engaged_ticks` is deliberately NOT asserted positive. Q-021: `engage` accumulates only while the
    # distance decreases and `engaged` also requires adjacency, so once adjacent the counter resets and the
    # conjunction is satisfied only on a tick that arrives at adjacency after e_min consecutive closing
    # steps. A sprinting predator closes two cells per tick and skips that window. Measured zero across
    # every arrangement tried. Asserting it here would be asserting a property the definition may not admit,
    # so the finding is recorded as an open question rather than forced into a fixture.


def test_the_recorded_grids_carry_real_content_not_just_the_right_shape(static, weights):
    """Shape-only assertions let a break that zeroed the recorded food grid survive falsification.

    D-034 is about what the renderer can draw, and an all-zero grid of the right shape draws an empty world
    while satisfying every shape check. The record must match the state it was taken from.
    """
    state, acc, traj = E.run_episode(jax.random.key(6), weights, static, _entropy(), P, record=True)
    assert bool(jnp.any(traj.food_grid > 0)), "every recorded food grid is empty, so the dump is not real"
    assert bool(jnp.any(traj.target_grid > 0)), "every recorded target grid is empty"
    assert bool(jnp.all(traj.food_grid[-1] == state.food_grid)), (
        "the last recorded food grid does not match the final state, so the record is not what happened"
    )
    assert bool(jnp.all(traj.target_grid[-1] == state.target_grid))


def test_the_other_four_strategy_classes_are_actually_accumulated(static, weights):
    """**The coverage assertion for DEV-019, and the reason it is written as a constructed scenario.**

    The accumulator set served freezing completely and fleeing, occlusion, distance maintenance and grouping
    not at all, against a build spec that says the set is fixed by this loop and nothing unaccumulated can be
    recovered. Fourteen accumulators were added. Adding them is not the same as exercising them: under the
    natural reset the predator never approaches, so every predator-conditioned counter reads zero and the new
    fields would be as inert as their absence was.

    Measured under the natural reset before this test was written: `visible_ticks`, `near_ticks`,
    `occluded_near`, `dist_sum_visible` and both flee counters were all zero. Under a constructed engagement
    thirteen of fourteen move.
    """
    state = _engaged_state(static)
    acc = E.zero_accumulators(P.k)
    ent = _entropy()
    for i in range(P.t_horizon):
        state, acc, _ = E.tick(state, acc, weights, static, ent, jnp.int32(i), P)

    for name in (
        "visible_ticks",
        "near_ticks",
        "far_ticks",
        "occluded_far",
        "visible_unengaged_ticks",
        "dist_sum_visible",
        "nn_visible_ticks",
        "nn_sum_visible",
        "nn_unseen_ticks",
        "nn_sum_unseen",
        "density_sum",
    ):
        assert float(jnp.sum(getattr(acc, name))) > 0.0, (
            f"{name} is zero across a constructed engagement, so the quantity it exists to measure was "
            "never observed and any descriptor built on it would report zero rather than nothing"
        )

    # **The coverage condition that was withheld while Q-021 was open, now asserted.** Every stillness and
    # fleeing descriptor divides by this, and while the old engagement rule stood it was identically zero, so
    # asserting it would have been asserting a state the definitions did not admit.
    assert int(jnp.sum(acc.engaged_ticks)) > 0, (
        "engaged_ticks is zero across a constructed engagement, so stillness_engaged, conditional_stillness, "
        "accidental_stillness, flee_engaged and conditional_flee are all measured over nothing"
    )

    # `flee_engaged` is still not asserted positive, and the reason has CHANGED, which is the point. It is no
    # longer definitional: engagement now fires, so a zero means this particular prey never increased its
    # distance on an engaged tick. The fixture's prey is an untrained network that moves on about 2 percent
    # of ticks and does not flee. That is a behavioural zero and it belongs to Q-025, not Q-021.

    # **The near-conditioned stillness fields, D-046, exercised with a policy that CAN move them.** The
    # network fixture emits one constant non-stay action, so `still_near` and `still_far` are behavioural
    # zeros under it, exactly like the flee fields above; asserting them positive there failed when this
    # was first written, which is the exercised check doing its job. A stay-emitting policy is the
    # instrument that prices the mechanism (D-044): under it, stillness must accumulate on BOTH sides of
    # the geometric split, because the fixture holds the predator near some prey and far from others.
    def stay_policy(w, obs, hidden, step_index):
        del w
        return jnp.full((obs.shape[0],), dynamics.ACTION_STAY, jnp.int32), hidden

    state2 = _engaged_state(static)
    acc2 = E.zero_accumulators(P.k)
    for i in range(20):
        state2, acc2, _ = E.tick(state2, acc2, weights, static, ent, jnp.int32(i), P, stay_policy)
    assert int(jnp.sum(acc2.still_near)) > 0, (
        "a policy emitting STAY every tick with the predator inside r_pred must accumulate still_near; "
        "if it does not, the near-conditioned stillness the re-based descriptors divide by is inert"
    )
    assert int(jnp.sum(acc2.still_far)) > 0, (
        "the same policy must accumulate still_far for the prey the predator is not near, or the "
        "baseline arm of conditional_stillness_near is inert"
    )


def test_the_accumulator_accounting_identities_hold(static, weights):
    """Two partitions that must add up, asserted as identities rather than as values.

    A miscount in one arm of a partition is invisible in that arm alone and obvious in the sum. Both of these
    would break silently under an off-by-one in the visibility or radius test, and a per-class descriptor
    computed from a broken denominator would look like a finding rather than a fault.
    """
    state = _engaged_state(static)
    acc = E.zero_accumulators(P.k)
    ent = _entropy()
    for i in range(P.t_horizon):
        state, acc, _ = E.tick(state, acc, weights, static, ent, jnp.int32(i), P)

    assert bool(jnp.all(acc.near_ticks + acc.far_ticks == acc.alive_ticks)), (
        "near + far must equal alive: every living tick has the predator either inside r_pred or outside it"
    )
    # **Third identity, added with the alive-mask change.** `prey_moved` has ALWAYS used alive-at-start-of-
    # tick while `alive_ticks` used alive-after-deaths-resolve, so a prey that moved on its death tick was
    # counted in the numerator of `activity_rate` and not in its denominator. Nothing asserted the relation,
    # so the mismatch was invisible. The alive-mask change makes both masks the same; this is what stops them
    # diverging again.
    assert bool(jnp.all(acc.moved_ticks <= acc.alive_ticks)), (
        "moved_ticks exceeds alive_ticks for some prey, so the two are computed against different alive "
        "masks. activity_rate = moved / alive is then a ratio of quantities counted over different tick "
        "sets, which is exactly the defect the alive-mask change removed"
    )

    assert bool(jnp.all(acc.visible_ticks + acc.unthreatened_ticks == acc.alive_ticks)), (
        "visible + unthreatened must equal alive: every living tick is either reported or not"
    )
    # **A relation BETWEEN partitions, which the two sums above cannot catch.**
    #
    # Two different arguments make it true and they give DIFFERENT diagnoses on failure, so the mechanism
    # matters rather than just the conclusion:
    #
    #   via adjacency   engagement requires adjacency, adjacency is inside the sensing radius
    #   via TARGETING   `select_target` both selects and persists ONLY on a currently reported prey, so
    #                   `target == k` already implies `reported[k]` this tick
    #
    # **The targeting route is what the code actually does**, and it is the stronger one: it holds at any
    # distance, where the adjacency argument only covers the adjacent case. So a failure here means TARGETING
    # and REPORTING have drifted apart, not that adjacency and the radius have. Both feed descriptor
    # denominators, and an invariant is worth mostly what it tells you when it breaks.
    assert bool(jnp.all(acc.engaged_ticks <= acc.visible_ticks)), (
        "engaged exceeded visible, so a prey was engaged on a tick where no predator reported it"
    )

    assert bool(jnp.all(acc.occluded_near <= acc.near_ticks))
    assert bool(jnp.all(acc.flee_visible_unengaged <= acc.visible_unengaged_ticks))

    # **The near-conditioned fields, D-046.** Each is a subset of its own denominator, and chosen and
    # accidental stillness are disjoint by construction (emitted stay versus emitted-something-else), so
    # their SUM is also bounded by the denominator, which a pair of subset checks alone would not catch.
    assert bool(jnp.all(acc.still_near + acc.accidental_still_near <= acc.near_ticks)), (
        "chosen and accidental stillness near the predator must be disjoint subsets of near_ticks"
    )
    assert bool(jnp.all(acc.flee_near <= acc.near_ticks))
    assert bool(jnp.all(acc.still_far <= acc.far_ticks))
    assert bool(jnp.all(acc.flee_far <= acc.far_ticks))
    # D-048: an away event requires an emitted move or sprint, and a stay is not directional, so away and
    # chosen stillness are disjoint subsets of the same denominator on each side of the split.
    assert bool(jnp.all(acc.away_near + acc.still_near <= acc.near_ticks)), (
        "away_near and still_near must be disjoint subsets of near_ticks: an emitted stay is not a move"
    )
    assert bool(jnp.all(acc.away_far + acc.still_far <= acc.far_ticks))
    assert bool(jnp.all(acc.nn_visible_ticks <= acc.visible_ticks))
