"""Starting configuration for increment one.

Every value below is a sweep centre, not a claim of correctness. Phase A exists to find the settings that
put learnability in band; this module exists so that no value is ever chosen silently at a call site.

Each field carries a provenance class in its comment:

    [D]  Derived      follows from a stated requirement by arithmetic. Changing it breaks that requirement.
    [S]  Semi-derived partially constrained; the specific number still has slack.
    [C]  Conventional standard practice, or a scale choice made for readability.
    [A]  Arbitrary    a plausible starting point. No argument behind the specific number.

**The authoritative parameter count is `n_params()` in this file, not a number written down anywhere.**
An earlier version of this docstring pointed at docs/parameter-provenance.md, which contains no counts, and
meanwhile four documents had transcribed the figure and three had gone stale after D-030 changed n_actions.
See DEV-010. A transcribed number has no consumer that can disagree with it, so nothing catches it drifting;
that is the same defect the derived properties below exist to prevent, one level out.

Before changing anything marked [D], recompute the constraint it serves. Several parameters are coupled,
and changing one alone can silently remove foraging pressure or the predator's only advantage. The coupling
table is at the bottom of this file and is asserted by `check_invariants()`.

See docs/parameter-provenance.md for the full derivations.
"""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class EnvParams:
    # ---------------------------------------------------------------- grid --
    h: int = 40
    """[D] Encounter rate. For a predator to have a prey in view about half the time:
    K * (2R+1)^2 / free_cells = 0.5, giving free_cells near 1360. A 40 by 40 grid at 15 percent
    obstacles is exactly 1360, and yields 0.48 expected prey in view."""
    w: int = 40  # [D] square, same derivation as h
    obstacle_density: float = 0.15  # [A] enough occlusion to matter without becoming a maze

    n_food_src: int = 6  # [A] enough for spatial choice, few enough to be worth returning to
    n_target_src: int = 6  # [A] as above
    r_src: int = 3  # [A] a 7 by 7 patch per source
    cap_food: int = 5  # [S] 5 in a 49-cell source area is about 10 percent local density
    cap_target: int = 5  # [S] as above
    p_food: float = 0.03
    """[D] Supply against demand. Eight prey needing five feeds each is 40 per episode; six sources at
    0.03 over 400 ticks offer 72 potential spawns. Supply exceeds demand but not greatly, so scarcity
    is real."""
    p_target: float = 0.03  # [D] same derivation

    # ---------------------------------------------------------- perception --
    r: int = 4
    """[D] Parameter budget. At c=14 channels, r=5 with hd=64 gives 134,932 parameters, over the
    100,000 limit. r, c, and hd are coupled and cannot be chosen independently."""
    r_pred: int = 4
    """[A] Equal to r by DESIGN CHOICE, not by requirement. An earlier version justified this with
    mutual discovery, which does not entail it: mutual discovery requires only that neither agent
    starts knowing the other's position, and a predator with a longer radius would still have to
    discover prey through perception. The real reasons are that the predator's advantage is
    deliberately concentrated in one lever, stamina, so difficulty tunes along a single understood
    axis; and that once the predator becomes a network its radius sets its observation tensor and
    therefore its parameter count. This remains a sweep knob rather than a fixed equality."""
    r_near: int = 2  # [S] must be >= 1 (adjacency) and < r; 2 is adjacency plus one ring
    p_distant_detect: float = 0.25
    """[A] LOW CONFIDENCE. Controls how much stillness is worth at range, so it is one of the strongest
    levers on the difficulty band. Sweep this early."""

    # ------------------------------------------------------------- episode --
    t_horizon: int = 400  # [D] ten grid traversals, at ~40 ticks to cross a 40-cell grid
    k: int = 8  # [A] compute and variance tradeoff; also sets predation load at 1/k
    n_pred: int = 1  # [F] fixed this increment; shapes written generally
    n_episodes: int = 4
    """[S] Fitness noise falls as 1/sqrt(n), so four halves the noise of one. Revisit against observed
    between-episode variance in Phase A rather than keeping this by default."""

    @property
    def d_spawn_relax(self) -> tuple:
        """[D] **The FLOOR is derived from `r_pred`; the HEAD is fixed. Owner's ruling, 2026-08-28.**

            floor = r_pred + 1          strictly above the radius, so no episode opens with mutual sight
            head  = 10                  fixed, and no longer a function of r_pred

        **What changed and why.** The head was `2 * r_pred + 2`, so sweeping `r_pred` moved the spawn
        separation with it. That made the axis measure two things at once: a larger sensing radius AND a
        larger starting distance. The two act in opposite directions on encounter rate, so the axis was
        partly self-cancelling and its effect was not attributable to sensing.

        **The floor stays derived because it guards a different property.** `min(d_spawn_relax) > r_pred` is
        what prevents an episode opening with mutual sight, and that has to hold at every radius. Fixing the
        floor as well would break the guarantee the moment `r_pred` exceeded it.

        **The default is unchanged, and that is the point rather than a coincidence.** At `r_pred = 4` the
        head was already 10, so `(10, 8, 6, 5)` is reproduced exactly and every gate, which runs at the
        default, sees no behavioural change at all. Only non-default radii move:

            r_pred 2   was (6, 5, 4, 3)      now (10, 7, 5, 3)
            r_pred 4   was (10, 8, 6, 5)     now (10, 8, 6, 5)     unchanged
            r_pred 6   was (14, 11, 9, 7)    now (10, 9, 8, 7)

        The intermediate rungs are evenly spaced and truncated, the same construction rule
        `make_relax_ladder` uses.

        **It was a literal, and `r_pred` is a proposed sweep axis, so the guarantee would have broken
        silently the first time the radius moved.** That is the coupling class: a derivation invalidated by
        sweeping a different parameter. Asserting the relation rather than the value is what makes it hold at
        every level instead of at the one it was written for. See DEV-018 and Q-028.

        A categorical draw over an empty mask does not raise under static shapes; it returns index zero,
        which may be an obstacle or a cell adjacent to prey. Reset takes the first level with at least
        `n_pred` free cells, and the level used is recorded per episode so M4 can report the relaxation rate.
        """
        floor = self.r_pred + 1
        head = 10
        assert head > floor, (
            f"spawn ladder head {head} does not exceed its floor {floor} at r_pred {self.r_pred}. "
            "With a fixed head this fails once r_pred reaches 9, which is already outside the permitted "
            "range of r_pred <= r + 2, but it fails LOUDLY rather than producing a non-descending ladder."
        )
        return tuple(int(head - i * (head - floor) / 3) for i in range(4))

    # -------------------------------------------------------------- energy --
    e_max: float = 100.0  # [C] all pools on 0-100 for readability, no other significance
    e_base: float = 0.5
    """[D] From the feeds-per-episode requirement: (e_base * t_horizon - e_max) / e_food = 5.
    Also means an unfed agent dies at exactly the halfway point of the episode."""
    e_sprint: float = 1.0  # [A] makes sprinting triple total drain, 0.5 -> 1.5. Sweep
    e_food: float = 20.0  # [C] e_max / 5, so five items fill an empty agent
    starve_rate: float = 2.0
    """[D] h_max / starve_rate gives 50 ticks of visible decline before death, long enough to be an
    observable cue rather than a sudden collapse."""

    # ------------------------------------------------------------- stamina --
    s_max: float = 100.0  # [C] scale choice
    s_regen: float = 4.0  # [A] sets absolute recovery timescale; only the ratio matters
    s_burst_prey: float = 20.0  # [D] sprint uptime = sprint_uptime(s_regen, s_burst_prey) = 16.7 percent
    s_burst_pred: float = 8.0
    """[D] 33.3 percent uptime, so the predator sprints 2.0 times as often as prey. This ratio is the
    speed lever the design places on the predator, and it is the predator's only advantage.

    **The figures were 50 percent and 2.5x until Q-022, and both were wrong for the same reason.**
    The design derived these parameters under "regen applies every tick and a burst is an extra
    cost", giving `regen / burst`. `dynamics.update_stamina` implements
    `-cost * bursting + regen * (1 - bursting)`, so regen applies only on non-burst ticks and the
    steady state is `regen / (burst + regen)`. Measured over 2000 ticks of continuous sprint demand:
    0.337 against the implemented rule's predicted 0.333.

    Ruled option one, `docs/correspondence/093`: amend the description, not the rule. The
    implemented rule is self-consistent, identical for both bodies, and no measured number is wrong.
    Changing the rule instead would invalidate the E1 determinism baseline and every measured
    descriptor, which the freeze exists to forbid absent a structural defect. **The consequence,
    recorded plainly rather than buried: the predator's sprint advantage is 2.0x, not the 2.5x the
    design intended.** Both burst costs are swept axes, so the intended-advantage region is visited
    by the sweep rather than lost."""
    s_min_burst: float = 20.0
    """[D] Equal to s_burst_prey, so a burst requires a full burst's worth banked and cannot be
    attempted on a partial reserve."""

    # -------------------------------------------------------------- health --
    h_max: float = 100.0  # [C] scale choice
    damage: float = 34.0  # [D] ceil(h_max / 3): three contacts kill. One mistake survivable
    h_regen: float = 0.25  # [A] full heal takes roughly one episode. Sweep
    disengage_steps: int = 5
    """[S] Must exceed e_min so a feign outlasts the engagement signal. Sets the feign cost at 2.5
    energy, an eighth of one food item: cheap enough to be viable, expensive enough to matter."""
    e_min: int = 4
    """[D] Random-walk false positive rate. Three consecutive distance-decreasing steps under random
    motion has probability near (3/8)^3, about 5 percent."""
    p_wander: float = 0.0
    """[A] Probability that a predator with NO reported prey takes a random step this tick.

    **Swept at 0, 0.25, 0.5, 0.75, 1.0.** At 0.0 the predator holds position, reproducing the behaviour every
    gate was written against exactly, which is why 0.0 is the default for the same reason `cue_dependence`
    defaults to 1.0.

    **It exists because the measured consequence of a static predator is that freezing is free.** Q-042: a
    prey that never moves is never predated at any `cue_dependence`, including 0.0, and its `near_ticks` is
    exactly 0.000. That is a proximity failure rather than a detection one, and every mechanism
    `cue_dependence` touches is downstream of targeting, so it cannot price freezing-as-avoidance at all.

    **0.0 stays in the sweep as a reportable point rather than being hidden.** The sweep should contain the
    world where stillness confers immunity and report it as such; what it must not do is contain only that
    world.

    Pure random walk, not a search heuristic. The original objection to a wander was that inventing search
    behaviour would be an unspecified change to the control condition. A uniform random step does not raise
    it, because it carries no information about where prey are."""
    cue_dependence: float = 1.0
    """[A] How much the predator's strike and pursuit depend on prey motion.

    1.0 reproduces the motion-cued predator exactly: a still prey takes no damage and breaks pursuit in
    disengage_steps ticks. 0.0 is motion-blind: stillness confers nothing and pursuit never breaks.
    Intermediate values interpolate continuously.

    This PRICES freezing against the other anti-predator strategies and therefore has no neutral value, so
    it is a swept axis rather than a design choice. See node 4.23.0.0. Default 1.0 for increment 1, which is
    what keeps every gate written against the original motion-gating rule valid; step 12 sweeps it.

    It is a property of the SCRIPTED PREDATOR, not a law of the environment. Nothing here states that a still
    prey cannot be captured, and the advantage of freezing is a consequence of this predator's behaviour
    rather than being built into the world. See CLAUDE.md section 1 and invariant I2.

    **This docstring used to continue "at cue_dependence 0.0 stillness confers nothing at all, which is the
    point of having the parameter". Measured, that is false, and the correction is left in place rather than
    quietly dropped.** At `p_wander` 0.0 a prey that never moves is never predated at ANY value of this
    parameter: the predator holds position without a target, a still prey never enters its sensing radius,
    and `near_ticks` is exactly 0.000. Everything this parameter touches, damage and pursuit-breaking, is
    downstream of targeting.

    **So it prices freezing-AFTER-detection and cannot price freezing-as-avoidance.** `p_wander` is the
    parameter that reaches the case this one cannot. See Q-042."""

    # ------------------------------------------------------------- network --
    hd: int = 48  # [D] parameter budget, coupled to r and c_channels
    a_appearance: int = 4  # [A] needs >= 2 (predator dim + orthogonal control); 4 gives room
    n_actions: int = 22
    """[D] 8 move + 8 sprint + stay + crouch + lie_down + pick + attack + drop.

    **`attack` and `drop` are reserved, masked unavailable in increment 1, and must not be removed.** They
    are here because this is the last cheap moment: adding an action after weights exist changes the output
    layer's shape and voids every evolved weight and the parameter budget. Reserving them costs 98 parameters,
    `hd * 2 + 2`, which is 0.14 percent of the budget.

    `attack` is needed at increment 2 rather than increment 4, because the coevolved predator's explicit
    strike decision is core rather than part of the weapon layer. Nodes 4.6.0.0 and 4.21.0.0. `drop` follows
    from the single held slot: an agent holding a weapon that wants to carry a block needs it.

    Same trade as crouch, lie-down and the appearance vector, which are also present and unused for later
    increments. See D-030 and CLAUDE.md section 14."""

    N_BASE_CHANNELS: int = 7  # [D] obstacle, food, target, predator, prey, moved, visible
    N_POSTURE_CHANNELS: int = 3  # [D] upright, crouched, prone one-hot
    N_SCALARS: int = 6  # [D] 3 normalised pools + 3-way own-posture one-hot

    @property
    def s_patch(self) -> int:
        """[D] Derived. Patch side, S = 2r + 1. The observation is an egocentric S by S patch."""
        return 2 * self.r + 1

    @property
    def obs_dim(self) -> int:
        """[D] Derived, not stored, and this is the value step 6 fixes for the whole project.

        `obs_dim = S * S * C + n_scalars`, where the scalars are three normalised pools and a three-way
        own-posture one-hot.

        Derived for the same reason `c_channels` is. It was previously recomputed inline inside `n_params`,
        so the network and the observation builder would each have had their own copy of the arithmetic and
        nothing would have caught them disagreeing. Changing `r`, `a_appearance` or the channel counts now
        moves every consumer together or fails loudly.

        **Changing this after step 6 weights exist voids every evolved weight**, exactly as `n_actions` does.
        See D-030 for the equivalent decision on the output side.
        """
        return self.s_patch * self.s_patch * self.c_channels + self.N_SCALARS

    @property
    def c_channels(self) -> int:
        """[D] Derived, not stored. An earlier version made this a free field alongside
        a_appearance, so changing the appearance width silently left the channel count wrong and
        the parameter budget was computed against a lie. Derive it."""
        return self.N_BASE_CHANNELS + self.N_POSTURE_CHANNELS + self.a_appearance

    # ----------------------------------------------------------- evolution --
    p_pop: int = 256  # [C] neuroevolution populations are typically 100-1000
    g_generations: int = 300
    """[A] Compute budget guess. Must exceed time-to-convergence, which is unknown until Phase A
    measures it. Phase A should set this from measurement rather than keeping the default."""
    t_sel: int = 3  # [C] tournament sizes of 2 to 7 are standard
    n_elite: int = 8  # [C] 3 percent of population; 1-10 percent is standard
    sigma: float = 0.01
    """[A] LOWEST CONFIDENCE VALUE IN THIS FILE. Mutation scale interacts with parameter count and
    landscape roughness, and 70k parameters is small enough that this could be off by a sizeable factor
    either way. Sweep it first: a badly set mutation scale looks exactly like 'evolution does not work
    in this environment', which risks misdiagnosing the whole approach."""

    # ------------------------------------------------------- swept, priced --
    # These seven price one strategy against another. No value of any is neutral,
    # so none is a design choice; each is an axis of the M4 sweep and the result is
    # a map from conditions to which strategy emerges. See node 4.23.0.0.
    #   s_burst_prey / s_burst_pred  price fleeing and pursuit
    #   predator speed via sprint uptime ratio   prices fleeing
    #   p_distant_detect             prices concealment
    #   predator cue-dependence      prices freezing
    #   obstacle_density             prices hiding
    #   k (prey per environment)     prices grouping
    #   d_src_relax[0]               prices camping against travel
    d_src_relax: tuple = (8, 6, 4, 2, 0)
    """[A] Minimum distance from any target source to the nearest food source. The ladder head is the
    nominal separation and `d_src` below is derived from it, not stored, for the reason D-003 deleted
    d_spawn: two fields holding one number will eventually disagree, and an assertion that they agree
    is weaker than making disagreement impossible.

    Zero means co-located, in which case an agent can camp on one spot supplying both resources and
    energy degenerates into a second timer. Large means energy forces travel, and travel is where
    predation happens. Swept, and it prices camping against travel.

    Same failure mode as d_spawn: a large separation on a small grid can be unsatisfiable, and a
    categorical draw over an empty mask returns index zero silently rather than raising. First
    non-empty level wins and the level used is recorded per episode. Measured feasibility: 40x40 never
    relaxes, 24x24 at 99.3 percent, 20x20 at 85.3, 16x16 at 40.7. Grid area binds, not source count or
    obstacle density."""

    @property
    def d_src(self) -> int:
        """[D] Derived. The nominal separation is the head of the relaxation ladder."""
        return self.d_src_relax[0]

    r_group: int = 4
    """[A] Radius for the local_density grouping descriptor. An ANALYSIS parameter, and free.

    An earlier docstring gave equality with r as the reason for the value, which reads as a derivation
    and is not one. Setting it equal to r means density is measured over what an agent could perceive
    and therefore could have responded to, which is a defensible default rather than a requirement.

    Tight clustering and loose aggregation are different behaviours at different scales, so this is
    arguably a sweep axis in its own right and is left free rather than tied to r."""

    # ---------------------------------------------------------- measurement --
    n_eng_min: int = 30
    """[A] Minimum engagements before learnability is defined. Thirty gives a standard error on p of at
    most 0.09, tolerable against a band of width 0.6. Below this, learnability is reported as UNDEFINED,
    never as zero: zero would read as collapse, which is a fail state, and the run would be abandoned
    for the wrong reason."""
    learnability_floor: float = 0.16
    """[D] Analytic. p(1-p) = 0.16 has roots at exactly p = 0.2 and p = 0.8, so this floor is the band
    p in [0.2, 0.8]. Maximum is 0.25 at p = 0.5. Learnability is also the variance of a Bernoulli
    variable with parameter p, which is why it is the right quantity: it measures how much outcome
    variation exists for selection to act on."""
    e2_atol: float = 1e-6  # [C] mixed tolerance; pools reach exactly zero, where rtol is undefined
    e2_rtol: float = 1e-5  # [C] as above
    smallest_effect_d: float = 1.0  # [S] one sd of the naive_eval null. Cohen's d of 1.0
    power_target: float = 0.80  # [C] standard across the empirical sciences


# --------------------------------------------------------------------------
# Coupled parameters. Changing one side without the other silently breaks a
# derivation while leaving the code running and the results plausible.
# --------------------------------------------------------------------------


def make_relax_ladder(head: int, n_levels: int = 5) -> tuple:
    """Build a relaxation ladder from its head by proportional scaling. One construction rule, not many.

    Sweeping a separation means sweeping the whole ladder, not just its head, and varying the head alone
    breaks the ladder at low values and unbalances it at high ones:

        head 4   head-only  (4, 6, 4, 2, 0)   invalid, not descending, the ordering assertion catches it
        head 12  head-only  (12, 6, 4, 2, 0)  valid, but the 12-to-6 step is now the largest by far
        head 12  scaled     (12, 9, 6, 3, 0)  proportional, uniform steps

    Proportional scaling keeps the steps uniform at every head and keeps the descending assertion satisfiable
    across the whole sweep range rather than only near the default. The floor is pinned at zero because a
    ladder that cannot reach its most permissive level has no last resort.

    **Floor division, not `round`.** Python's `round` is banker's rounding, half to even, so `round(1.5)` is
    2 and `round(2.5)` is also 2. That is the same trap D-013 found in the ray rule, where using float
    rounding in the oracle disagreed with the specification in twelve configurations. Floor division is exact
    integer arithmetic with no tie-breaking to get wrong.

        levels[i] = head * (n_levels - 1 - i) // (n_levels - 1)

    Reproduces the default exactly: `make_relax_ladder(8)` is `(8, 6, 4, 2, 0)`.

    At small heads consecutive levels can repeat, for example `make_relax_ladder(2)` giving
    `(2, 1, 1, 0, 0)`. That is harmless rather than a defect, since the first feasible level wins and the
    ordering assertion is non-strict, but it does mean a low-head configuration has fewer distinct fallbacks
    than the level count suggests. Recorded rather than deduplicated, because removing entries would change
    the tuple length and therefore the meaning of a recorded relaxation index.
    """
    assert head >= 0, "a negative separation is meaningless"
    assert n_levels >= 2, "a ladder needs at least a head and a floor"
    ladder = tuple(head * (n_levels - 1 - i) // (n_levels - 1) for i in range(n_levels))

    # Distinct levels degrade faster than the level count suggests: head 2 gives 3 of 5, head 1 gives 2, and
    # head 0 gives 1, which is a ladder with no fallback at all. Fail loudly at that point rather than
    # returning something that looks like a ladder and cannot relax.
    assert len(set(ladder)) >= 2, (
        f"head {head} with {n_levels} levels produces {ladder}, which has only "
        f"{len(set(ladder))} distinct level. A ladder with one level has no fallback, so a reset that cannot "
        "satisfy it has nothing to relax to and will draw over an empty mask."
    )
    return ladder


COUPLINGS = {
    "parameter budget": ("r", "c_channels", "hd", "n_actions"),
    "foraging pressure": ("e_base", "t_horizon", "e_food", "e_max"),
    "predator advantage": ("s_regen", "s_burst_prey", "s_burst_pred"),
    "contacts to kill": ("h_max", "damage"),
    "feign viability": ("e_min", "disengage_steps", "e_base"),
    "encounter rate": ("k", "r", "h", "w", "obstacle_density"),
    "device memory": ("p_pop", "n_episodes", "k", "t_horizon"),
    "supply vs demand": ("n_food_src", "p_food", "cap_food", "k", "t_horizon"),
}


def n_params(p: EnvParams) -> int:
    """Total network parameters. Asserted against the 100,000 design limit.

    Note the GRU term assumes one bias vector per gate. PyTorch-style implementations carry two,
    b_ih and b_hh, adding 3 * hd. Check which convention the chosen library uses and record it in
    decisions.md.
    """
    obs_dim = p.obs_dim
    p_input = obs_dim * p.hd + p.hd
    p_gru = 3 * (p.hd * p.hd + p.hd * p.hd + p.hd)
    p_output = p.hd * p.n_actions + p.n_actions
    return p_input + p_gru + p_output


def sprint_uptime(regen: float, burst: float) -> float:
    """Steady-state fraction of ticks spent bursting, as `dynamics.update_stamina` implements it.

    **The single expression of this rule.** `update_stamina` computes
    `-cost * bursting + regen * (1 - bursting)`: regen applies only on NON-burst ticks, so a
    sustainable cycle spends `burst` recovering for every `regen` spent bursting and the steady
    state is `regen / (burst + regen)`.

    It lives here, called by `check_invariants` and by the summary printer, because the alternative
    is what this project already ran for a month: the formula inline in the check and restated in
    two docstrings, three copies free to drift from the code and from each other. They did, in the
    same direction, and nothing disagreed. Q-022, and D-051.

    **What this does not do, which is the limitation the entry diagnosed.** It expresses the
    implemented rule; it does not verify it. A consistency check on the config never consults the
    simulator, which is why the discrepancy survived every green run of `check_invariants`. The
    verification is behavioural: 2000 ticks of continuous sprint demand measured 0.337 against this
    expression's 0.333 for the predator.
    """
    return regen / (burst + regen)


def check_invariants(p: EnvParams) -> None:
    """Fail loudly if a derived relationship has been broken by a sweep.

    This is not a style check. Each assertion guards a derivation that, if broken silently, changes
    what the experiment is measuring while leaving it running normally.
    """
    assert n_params(p) < 100_000, (
        f"parameter budget breached: {n_params(p):,}. r, c_channels, and hd are coupled."
    )

    free = p.h * p.w * (1 - p.obstacle_density)
    in_view = p.k * (2 * p.r + 1) ** 2 / free
    assert 0.2 < in_view < 1.0, (
        f"encounter rate {in_view:.2f} outside the productive range. "
        "Either the predator almost never sees prey, or the grid is crowded."
    )

    feeds = (p.e_base * p.t_horizon - p.e_max) / p.e_food
    assert 2.0 <= feeds <= 10.0, (
        f"foraging pressure broken: {feeds:.1f} feeds per episode. "
        "Too few and agents can ignore food; too many and they cannot do anything else."
    )

    assert p.r_pred <= p.r + 2, (
        "predator sensing radius is far above the prey's. Equal radii are the default design choice, not "
        "a requirement of mutual discovery, but a large gap concentrates the predator's advantage in "
        "vision rather than stamina, which is a different experiment from the one specified. "
        "**And there is a measured reason alongside that design one, which is the checkable half:** "
        "expected prey-in-view runs 0.48 at r_pred 4, the value the grid was sized to produce, 0.99 at 6, "
        "and 1.70 at 8. At 8 the predator always has a target, so distancing is excluded by construction "
        "and p_wander dies entirely, being conditional on having no target. The ceiling is where the axis "
        "stops producing distinguishable conditions and starts killing another axis. Note that 6 sits at "
        "the boundary at 0.99, so the axis has no headroom above it rather than an arbitrary stop."
    )

    uptime_prey = sprint_uptime(p.s_regen, p.s_burst_prey)
    uptime_pred = sprint_uptime(p.s_regen, p.s_burst_pred)
    assert uptime_pred > uptime_prey, (
        "the predator has lost its only advantage. Sprint uptime must exceed the prey's."
    )

    assert p.disengage_steps > p.e_min, (
        "a feign must outlast the engagement signal, or stillness cannot break pursuit."
    )

    assert min(p.d_spawn_relax) > p.r_pred, (
        "the most relaxed spawn separation is inside the predator's sensing radius, so an episode "
        "could open with mutual sight."
    )

    supply = p.n_food_src * p.p_food * p.t_horizon
    demand = p.k * (p.e_base * p.t_horizon - p.e_max) / p.e_food
    assert supply > demand, (
        f"food supply {supply:.0f} below demand {demand:.0f}. Agents starve regardless of behaviour, "
        "which removes the selection gradient rather than sharpening it."
    )

    assert 0.0 < p.learnability_floor < 0.25, (
        "learnability is bounded above by 0.25 at p = 0.5. A floor at or above it can never be met."
    )

    assert p.r_near < p.r, (
        f"r_near {p.r_near} must be inside the sensing radius {p.r}. A near radius at or beyond the "
        "sensing radius makes the two-regime detection rule collapse into always-detect."
    )

    # **`damage = 0` is the third arm and this assertion cannot be applied to it.** Second instance found by
    # step 12's pre-flight, after `n_pred >= 1`: a parameter invariant written for the treatment arm, applied
    # to a configuration that legitimately differs. Here it divided by zero rather than merely failing.
    #
    # "Contacts to kill" is not a quantity at zero damage. It is infinite, and that is the arm's definition
    # rather than a broken configuration: `harmless_evolve` evolves against a predator that senses, targets,
    # pursues and makes contact, and removes no health.
    #
    # **Guarded rather than relaxed.** The band still applies at every nonzero damage, so the treatment arm
    # is checked exactly as before, and zero is admitted as the one value that makes the ratio undefined
    # rather than out of range.
    if p.damage > 0.0:
        contacts = p.h_max / p.damage
        assert 2.0 <= contacts <= 5.0, (
            f"contacts to kill is {contacts:.1f}. Below 2 a single mistake is fatal and there is no window "
            "for feigning to act; above 5 capture stops being a real threat. damage and h_max are coupled."
        )
    else:
        assert p.damage == 0.0, f"damage is {p.damage}, which is negative. Contact cannot heal."

    decline = p.h_max / p.starve_rate
    assert 20.0 <= decline <= 0.25 * p.t_horizon, (
        f"starvation decline is {decline:.0f} ticks against a {p.t_horizon}-tick episode. Too fast and "
        "starvation is indistinguishable from instant death; too slow and it never bites within an "
        "episode. starve_rate and h_max are coupled."
    )

    assert p.s_min_burst <= p.s_max, "a burst can never be afforded"
    assert p.k >= 1, "at least one prey is required"

    # **`n_pred >= 1` was here and it is WRONG at this level, found by step 12's pre-flight.**
    #
    # `no_predator_evolve` sets `n_pred = 0`, deliberately and since step 8: DEV-013 chose a zero-length
    # predator axis over a capture mask precisely so that every reduction over it is a no-op by construction
    # with no path that can be missed. So a first-class run configuration has always violated this assertion.
    #
    # **It never fired because nothing had ever applied THIS function to that configuration.**
    # `configs/run_configs.py` has its own `check_invariants()` taking no arguments, and the gate calls that
    # one. The two share a name and check different things, which is how the gap survived from step 8 to
    # step 12 and was found by a pre-flight rather than by a test.
    #
    # The requirement is real and belongs where it is true: on the configurations that need a predator.
    # `configs/run_configs.check_invariants` now asserts it per configuration, which is stronger, because it
    # also pins WHICH configuration is the zero-predator one instead of merely permitting zero everywhere.
    assert p.n_pred >= 0, "a negative predator count is not a configuration"

    assert 0.0 <= p.cue_dependence <= 1.0, (
        f"cue_dependence is {p.cue_dependence}, outside [0, 1]. Above 1 a still prey would take negative "
        "damage, healing it on contact, and the disengage counter would run faster than the tick clock. "
        "Below 0 a still prey would take more damage than a moving one, which inverts the rule rather than "
        "interpolating it."
    )

    assert p.a_appearance >= 2, (
        "the overgeneralisation test needs at least a predator-appearance dimension and one orthogonal "
        "control dimension."
    )

    assert list(p.d_src_relax) == sorted(p.d_src_relax, reverse=True), (
        "d_src_relax must be strictly descending, because reset takes the first feasible level and a "
        "non-monotone ladder would relax further than necessary or not at all."
    )

    assert min(p.d_src_relax) >= 0, "a negative separation is meaningless"

    assert len(set(p.d_src_relax)) >= 2, (
        f"d_src_relax is {p.d_src_relax}, which has only {len(set(p.d_src_relax))} distinct level. A ladder "
        "with one level has no fallback, so a configuration that cannot satisfy it draws over an empty mask "
        "and silently returns index zero. Asserted here as well as in make_relax_ladder, because a ladder can "
        "also be written by hand."
    )

    # A source patch spans 2*r_src + 1 cells. Once d_src exceeds twice the source radius the food and target
    # patches cannot overlap at all, which is the point at which camping on one spot stops being available.
    # This is not a failure, it is the meaning of the parameter, so it is recorded rather than asserted
    # against: the sweep deliberately spans both regimes.
    assert p.d_src <= max(p.h, p.w), (
        f"d_src of {p.d_src} exceeds the grid dimension, so no target source can ever satisfy it and every "
        "layout would relax to a lower level, making the nominal value meaningless."
    )

    assert p.r_group >= 1, "the grouping descriptor needs a radius of at least one cell"

    # The coupling table is load-bearing documentation, so verify it has not gone stale.
    valid = {f.name for f in fields(p)} | {"c_channels"}
    for group, names in COUPLINGS.items():
        unknown = set(names) - valid
        assert not unknown, (
            f"COUPLINGS group '{group}' names fields that do not exist: {unknown}. "
            "A stale coupling table is worse than none, because it is trusted."
        )


DEFAULT = EnvParams()

if __name__ == "__main__":
    check_invariants(DEFAULT)
    print(f"parameters:      {n_params(DEFAULT):,} / 100,000")
    free = DEFAULT.h * DEFAULT.w * (1 - DEFAULT.obstacle_density)
    print(f"free cells:      {free:.0f}")
    print(f"prey in view:    {DEFAULT.k * (2 * DEFAULT.r + 1) ** 2 / free:.2f}")
    print(f"feeds/episode:   {(DEFAULT.e_base * DEFAULT.t_horizon - DEFAULT.e_max) / DEFAULT.e_food:.1f}")
    print(
        f"sprint uptime:   prey {sprint_uptime(DEFAULT.s_regen, DEFAULT.s_burst_prey):.1%}, "
        f"pred {sprint_uptime(DEFAULT.s_regen, DEFAULT.s_burst_pred):.1%}, "
        f"advantage {sprint_uptime(DEFAULT.s_regen, DEFAULT.s_burst_pred) / sprint_uptime(DEFAULT.s_regen, DEFAULT.s_burst_prey):.2f}x"
    )
    print(f"contacts to kill:{DEFAULT.h_max / DEFAULT.damage:.1f}")
    print(f"envs (E):        {DEFAULT.p_pop * DEFAULT.n_episodes}")
    print("all invariants hold")
