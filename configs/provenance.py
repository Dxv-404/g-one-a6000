"""Machine-readable provenance, and the chain each parameter's justification terminates in.

**This exists because "it is derived" has been treated as settling the question, and it does not.** A derived
value can be exactly as arbitrary as a magic number, with the arbitrariness moved one level up and concealed
by the arithmetic. Tracing the chains makes that visible: two values in this design have justifications and
no root, and a justification check alone cannot see them.

    grid 40x40   <- encounter rate 0.48   <- k, which is SWEPT               terminates
    s_burst_*    <- uptime targets        <- SWEPT                           terminates
    E_MIN 4      <- false positive 6.2%   <- a statistical convention        terminates, weakly
    DAMAGE 34    <- "three contacts kill" <- nothing                         FAILS
    disengage 5  <- "must exceed e_min"   <- e_min, own root unrecorded      FAILS

**The standard: every derivation terminates in something SWEPT, MEASURED, or EXTERNALLY CONSTRAINED.** Not in
another choice. A chain that ends in a preference has moved the preference rather than removed it.

**`derives_from` means DETERMINED BY, not merely related to, and the distinction is load-bearing.** `damage`
is `ceil(h_max / 3)` and `disengage_steps` must exceed `e_min`. Listing those as inputs would let the walk
reach an external root through them and terminate falsely, concealing that the divisor and the value are the
real choices. A constraint bounds a value; it does not derive it. Both were caught by the gate on its first
run, walking past their own arbitrariness to an ancestor's root.

The two known failures are declared in `UNROOTED` rather than hidden, and that set is asserted to be exactly
what it says. **A new parameter without a root fails the gate**, which is the ratchet form used for the
differential's `UNMODELLED` set: debt can only grow by decision.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A chain may terminate in one of these. Anything else is another choice wearing a derivation.
ROOT_KINDS = ("swept", "measured", "external", "scope")

#: Parameter classes, as tagged in `configs/default.py`.
#:   D derived   S semi-derived   A arbitrary   C conventional   F fixed for this increment
CLASSES = ("D", "S", "A", "C", "F")

#: Swept axes that exist as `EnvParams` fields today. Axes that are named but are not yet parameters go in
#: `PROPOSED_AXES` instead of here, and keeping the two apart is what stops a proposed axis being counted as
#: coverage.
#:
#: **The count that used to be stated here was wrong and had nothing to catch it.** The comment read "the
#: revised specification names fourteen; `p_wander`, search persistence and prey solidity are not yet
#: parameters". `p_wander` became an `EnvParams` field at step 11 and was added to this set without being
#: removed from `PROPOSED_AXES`, so it sat in both while the prose beside it said it was in neither. The
#: membership has a consumer, the design audit's chain walk, so it could not drift. The count did not.
#: DEV-011's shape exactly, and the reason the count is not restated with a corrected number is that a
#: corrected number would have exactly the same problem. See Q-044, which asks the owner what fourteen
#: counted, since 13 here plus 2 proposed is 15 and guessing which pair merges would be inventing a fact.
SWEPT: frozenset[str] = frozenset(
    {
        "cue_dependence",
        "p_wander",
        "s_burst_pred",
        "s_burst_prey",
        "p_distant_detect",
        "n_pred",
        "r_pred",
        "obstacle_density",
        "k",
        "d_src_relax",
        "e_sprint",
        "n_food_src",
        "n_target_src",
        # **Added by owner ruling 2026-08-25, Q-046.** It is a SEARCH parameter and every other member here
        # is an environment one, which is exactly why it was missing: the M4 sweep's scope is the
        # environment, so nothing outside it was ever a candidate however important.
        #
        # The asymmetry that made it urgent: an unswept environment parameter risks a result conditional on
        # an unexamined setting, and an unswept search parameter risks the NULL. This design deliberately
        # makes a null reportable, and a null caused by a badly set mutation scale is indistinguishable from
        # the real one while being a completely different finding.
        "sigma",
    }
)

#: Named as swept axes but NOT yet `EnvParams` fields, so they cannot be measured, cannot be checked for
#: liveness, and must never be counted as coverage. **Asserted disjoint from the parameter set** by
#: `test_a_proposed_axis_is_not_already_a_parameter`, which is what makes "not yet a parameter" a checkable
#: claim rather than a comment. `p_wander` was removed on 2026-08-25 having been a field since step 11.
PROPOSED_AXES: frozenset[str] = frozenset({"search_persistence", "prey_solid"})

#: **Levers that are mechanisms rather than parameters**, so they have no value and no ratio. They cannot be
#: swept by moving a dial; changing them means choosing a different rule. Recorded here so a strategy's payoff
#: can name one without the check demanding it be a parameter.
#:
#: `nearest_target_selection` is grouping's payoff and is the live instance.
#:
#: **The justification here was WRONG and is corrected in place rather than rewritten, because the error is
#: the useful part.** It read: "under RANDOM target selection, hiding in a crowd would cut risk to 1/K for
#: free, so grouping would pay automatically. Under NEAREST it costs positioning."
#:
#: The second sentence does not follow from the first.
#:
#:     random targeting    in a group of 8 you are chosen 1 time in 8
#:     nearest targeting   in a group of 8 you are chosen when you are nearest, which absent any
#:                         positioning skill is ALSO about 1 time in 8
#:
#: **Nearest targeting does not remove the dilution. It makes the dilution depend on position rather than be
#: automatic.** Whether that becomes a skill is an empirical question about whether position actually varies
#: and actually predicts risk, and it was being asserted as a property of the rule.
#:
#: Run 14 measures it: the correlation of local density with predation, decomposed by whether position
#: predicts risk WITHIN a density band. Flat within bands means the payoff is membership and the rule has not
#: made grouping a skill. See `unswept.md`.
MECHANISMS: frozenset[str] = frozenset({"nearest_target_selection"})


@dataclass(frozen=True)
class Entry:
    """One parameter's provenance. `derives_from` names inputs; `root` names where the chain ends."""

    cls: str
    derives_from: tuple[str, ...] = ()
    root: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        assert self.cls in CLASSES, f"unknown class {self.cls!r}"
        assert self.root is None or self.root in ROOT_KINDS, f"unknown root {self.root!r}"


#: **Parameters whose chain does NOT terminate, declared rather than hidden.**
#:
#: Both have justifications, which is why a justification check cannot see them. What they lack is a root
#: that is swept, measured, or externally constrained. Both are on `unswept.md`.
UNROOTED: frozenset[str] = frozenset(
    {
        "damage",
        "disengage_steps",
        # **Added by the gate's own coverage audit, which asked how many chains end in `scope`.**
        # `scope` is a DECLARATION rather than a checkable claim, so any arbitrary choice can be labelled it
        # and pass. Auditing the ten that ended there found six naming a preference rather than a bounded
        # claim. `r_near` is the sharpest: it was labelled `scope` in the same session it was recorded on
        # unswept.md as a freezing payoff lever with a measured 5.4x range, so the gate would have converted
        # a known finding into a false clearance.
        "r_near",
        "h_regen",
        "starve_rate",
        "cap_food",
        "cap_target",
        "r_src",
        # **Fourth instance of the constraint-is-not-a-derivation trap**, found when the unswept list's arity
        # and the redundancy table's row count disagreed by one. `p_food`'s justification cites the same
        # "five feeds each" figure and has NOT been re-examined; it is a follow-up rather than a claim.
        "e_base",
    }
)

#: A `scope` terminator must name a BOUNDED claim, one someone could disagree with, rather than an unbounded
#: "this is out of scope". Checked by requiring one of these markers in the note, which is crude and is the
#: point: without it, `scope` accepts anything.
BOUNDED_SCOPE_MARKERS = ("increment", "not a world parameter", "analysis")

PROVENANCE: dict[str, Entry] = {
    # ---- world geometry -----------------------------------------------------------------
    "h": Entry("D", ("k", "r"), "swept", "encounter rate near 0.48; k is swept so the chain terminates"),
    "w": Entry("D", ("h",), "swept", "square, same derivation as h"),
    "obstacle_density": Entry("A", (), "swept", "swept axis; prices hiding"),
    "n_food_src": Entry("A", (), "swept", "swept as source count"),
    "n_target_src": Entry("A", (), "swept", "as above"),
    "r_src": Entry(
        "A",
        (),
        None,
        "a 7 by 7 patch names no consequence anyone could disagree with, which is the test a scope claim "
        "has to pass. Sets source concentration, so it prices camping against travel. UNROOTED",
    ),
    "cap_food": Entry(
        "S",
        (),
        None,
        "about 10 percent local density is a chosen number. Sets how rich a source is, which prices camping "
        "against travel, the same lever as regrowth rate and source count. UNROOTED",
    ),
    "cap_target": Entry("S", ("cap_food",), None, "as cap_food. UNROOTED"),
    "p_food": Entry(
        "D",
        ("k", "e_base", "t_horizon", "e_food", "n_food_src"),
        "swept",
        "supply against demand; k and n_food_src are both swept",
    ),
    "p_target": Entry("D", ("p_food",), "swept", "same derivation"),
    "d_src_relax": Entry("A", (), "swept", "head is the swept d_src axis"),
    # ---- perception ---------------------------------------------------------------------
    "r": Entry("D", ("hd",), "external", "parameter budget of 100k, a hardware and compute constraint"),
    "r_pred": Entry("A", (), "swept", "swept axis; equal radii were a design choice, not a requirement"),
    "r_near": Entry(
        "S",
        (),
        None,
        "the invariant bounds it to 1..3 and does not choose 2. **It prices freezing directly**: inside "
        "r_near detection is guaranteed regardless of motion, so stillness confers nothing there. Measured "
        "at 11.1 / 30.9 / 60.5 percent of the predator's sensing area exempt at 1 / 2 / 3, a 5.4x range. "
        "Labelled `scope` in the same session it was put on unswept.md, which is the contradiction the "
        "terminator audit caught. UNROOTED",
    ),
    "p_distant_detect": Entry("A", (), "swept", "swept axis; prices concealment"),
    # ---- episode ------------------------------------------------------------------------
    "t_horizon": Entry("D", ("h",), "swept", "ten grid traversals; h derives from swept k"),
    "k": Entry("A", (), "swept", "swept axis; prices grouping"),
    "n_pred": Entry("F", (), "swept", "now a swept axis; was fixed with no stated reason"),
    "n_episodes": Entry("S", (), "external", "fitness noise falls as 1/sqrt(n); a statistical relationship"),
    # ---- pools --------------------------------------------------------------------------
    "e_max": Entry("C", (), "external", "scale convention, 0 to 100"),
    "e_base": Entry(
        "D",
        (),
        None,
        "from (e_base * t_horizon - e_max) / e_food = 5. **The inputs are not listed, because the 5 is the "
        "choice**: given t_horizon, e_max and e_food, the equation determines e_base only once someone picks "
        "five feeds per episode. Structurally identical to damage = ceil(h_max / 3), where the scale is a "
        "convention and the divisor is the decision. Listing the inputs let the walk reach t_horizon's swept "
        "root and terminate falsely. FOURTH instance of that trap, and it was caught by reconciling the "
        "arity of two records rather than by review. UNROOTED",
    ),
    "e_sprint": Entry("A", (), "swept", "swept axis"),
    "e_food": Entry("C", ("e_max",), "external", "e_max / 5, a scale convention"),
    "starve_rate": Entry(
        "D",
        (),
        None,
        "50 ticks of visible decline is a LEGIBILITY target, which is a preference. `h_max` is not listed "
        "as an input for the same reason as `damage`: it sets the scale and not the rate. UNROOTED",
    ),
    "s_max": Entry("C", (), "external", "scale convention"),
    "s_regen": Entry(
        "A",
        (),
        "swept",
        "sets the ABSOLUTE timescale, and only the RATIO to burst cost determines uptime. Both "
        "s_burst_prey and s_burst_pred are swept, so the ratio is swept and a change here is absorbed. "
        "That is an argument someone can disagree with, which is what makes it a root rather than a label",
    ),
    "s_burst_prey": Entry(
        "D",
        ("s_regen",),
        "swept",
        "from the swept prey uptime target, under the IMPLEMENTED rule "
        "uptime = regen / (burst + regen), which is 16.7 percent at the defaults. Q-022 found this "
        "derivation had been made under regen / burst, a rule the code does not implement, and the "
        "correction is recorded here rather than only in the docstring because a derivation stated "
        "in one place drifts",
    ),
    "s_burst_pred": Entry(
        "D",
        ("s_regen",),
        "swept",
        "from the swept predator uptime target, same implemented rule, 33.3 percent at the "
        "defaults. **The consequence, stated plainly: the predator's sprint advantage is 2.0x and "
        "not the 2.5x the design intended.** Both burst costs are swept, so the intended-advantage "
        "region is visited by the sweep rather than lost. Q-022, ruled option one",
    ),
    "s_min_burst": Entry("D", ("s_burst_prey",), "swept", "derived equal to the burst cost, DEV-017"),
    "h_max": Entry("C", (), "external", "scale convention"),
    "damage": Entry(
        "D",
        (),
        None,
        "ceil(h_max / 3). **`h_max` is deliberately NOT listed as an input**, because the value is not "
        "DETERMINED by it: h_max is a scale convention and the divisor 3, 'three contacts kill', is the "
        "actual choice. Listing it would let the chain walk past the arbitrariness to h_max's external root "
        "and terminate falsely, which is exactly the concealment this module exists to prevent. UNROOTED",
    ),
    "h_regen": Entry(
        "A",
        (),
        None,
        "its own justification says flagged for sweeping and not yet swept, which is an admission rather "
        "than a scope claim. Prices tolerating contact against avoiding it, alongside `damage`. UNROOTED",
    ),
    # ---- predator -----------------------------------------------------------------------
    "disengage_steps": Entry(
        "S",
        (),
        None,
        "must EXCEED e_min, which is a constraint on the value and not a derivation of it. `e_min` is not "
        "listed as an input for the same reason as `damage`: an ordering constraint bounds a value without "
        "determining it, and treating it as a derivation would inherit e_min's external root. UNROOTED",
    ),
    "e_min": Entry(
        "D", (), "external", "random-walk false positive near 5 percent, a statistical convention"
    ),
    "cue_dependence": Entry("A", (), "swept", "swept axis; prices freezing"),
    "p_wander": Entry(
        "A",
        (),
        "swept",
        "swept axis; whether the predator searches at all. At 0.0 it holds position and a prey that "
        "never moves is never approached, so freezing is free rather than priced. Q-042",
    ),
    # ---- agent --------------------------------------------------------------------------
    "hd": Entry("D", ("r",), "external", "parameter budget of 100k"),
    "a_appearance": Entry("A", (), "scope", "needs at least 2; increment 3 consumes it"),
    "n_actions": Entry("D", (), "scope", "enumeration of the action set fixed for this increment"),
    # ---- search -------------------------------------------------------------------------
    "p_pop": Entry("C", (), "external", "neuroevolution convention, 100 to 1000"),
    # **The note NAMES its run, and a test checks that the run exists.** This root was `measured` with
    # the note "the probe measures it", which is true: run 1 is the convergence probe. `sigma` carried
    # the same kind of root with the same kind of note and NOTHING had measured it. The two were
    # indistinguishable to the audit, because it checks the KIND of terminator and not its fulfilment.
    "g_generations": Entry(
        "A", (), "measured", "must exceed time-to-convergence; measured by run 1, the convergence probe"
    ),
    "t_sel": Entry("C", (), "external", "tournament sizes 2 to 7 are standard"),
    "n_elite": Entry("C", ("p_pop",), "external", "1 to 10 percent of population is standard"),
    # **Root corrected from `measured` to `swept`, 2026-08-25.** It read `measured` with the note "to be
    # swept first", and nothing had measured it. `measured` is a checkable terminator, so the chain walk
    # accepted a promissory note as a completed one: the gate checks the KIND of terminator and cannot
    # check that the measurement exists. Now genuinely swept, run 11.
    "sigma": Entry("A", (), "swept", "swept axis; prices which strategies are reachable. Q-046, run 11"),
    # ---- analysis -----------------------------------------------------------------------
    "r_group": Entry("A", (), "scope", "an analysis radius, free and not a world parameter"),
    "n_eng_min": Entry("A", (), "external", "standard error on a proportion at n=30"),
    "learnability_floor": Entry("D", (), "external", "analytic: p(1-p)=0.16 has roots at 0.2 and 0.8"),
    "e2_atol": Entry("C", (), "external", "float tolerance convention"),
    "e2_rtol": Entry("C", (), "external", "as above"),
    "smallest_effect_d": Entry("S", (), "external", "Cohen's d of 1.0, a statistical convention"),
    "power_target": Entry("C", (), "external", "0.80 is standard across the empirical sciences"),
}


def chain(name: str) -> list[str]:
    """The derivation chain from `name` to its root, as a list of parameter names."""
    seen, out, cur = set(), [], name
    while cur is not None and cur in PROVENANCE and cur not in seen:
        seen.add(cur)
        out.append(cur)
        froms = PROVENANCE[cur].derives_from
        cur = froms[0] if froms else None
    return out


def root_of(name: str) -> str | None:
    """Walk to the end of the chain and return the root kind recorded there."""
    for step in chain(name):
        if PROVENANCE[step].root is not None:
            return PROVENANCE[step].root
    return None


if __name__ == "__main__":
    print(f"{'parameter':<22}{'cls':>4}  {'root':<10} chain")
    for n, e in PROVENANCE.items():
        c = " <- ".join(chain(n))
        print(f"{n:<22}{e.cls:>4}  {str(root_of(n) or 'NONE'):<10} {c}")
    print(f"\nunrooted: {sorted(UNROOTED)}")
