"""The design audit, as a gate that runs rather than a pass that is remembered.

**Why this is a test and not a checklist.** Four audit passes each found something and the finding rate has
not fallen, because nothing prevents new instances entering while old ones are being found. This project has
exactly one thing that has worked against that: enforcement rather than recollection. Ruff, the pre-commit
hook, the falsifiers, the differential, a failed anchor. What has never worked is a rule written down and
then relied upon, D-027 six times over.

So every parameter added during steps 9 to 13 is checked as it is added, rather than found in a sweep weeks
later. A parameter caught at entry costs a line; the same parameter caught at step 13 costs a re-derivation
and possibly a re-run.

**And it is what the write-up needs.** "We were careful" is unfalsifiable and every project claims it. "Every
value terminates in something swept, measured, or externally constrained, here is the check, and here is the
printed chain for each parameter" is a demonstration. That is the difference between claiming integrity and
showing it.

The five checks, and the fifth is the one that caught two findings a justification check could not see.

**What this gate does NOT cover, stated here so a passing run is not read as a clean design.**

It covers parameters and declared mechanisms. It does not cover STRUCTURAL findings, which have no parameter
and no magnitude:

    DEV-021   a differential sees only downstream of the fork
    DEV-022   reset bypasses the entropy source, so it is unreplayable
    DEV-020   a comparison that iterates one side's keys is a whitelist with no tell

**And it does not cover INTERACTING DERIVATIONS, which is a different gap and a live one.** This gate asks
whether a value has a root. It does not ask whether two values with good roots jointly determine something
neither of them mentions.

The worked instance is Q-032. `e_min` was derived from a random-walk false-positive rate. `damage` was
derived from wanting three contacts to kill. Both derivations are sound, neither names the other, and
together with `r_pred` and the predator's sprint speed they make the engaged-tick denominator structurally
near-empty whatever the prey does: 26 captures produced 5 accumulated engaged ticks across eight full
episodes. **Every chain in this file terminates correctly and the gate is green.**

There is no mechanical form for this check yet and one is deliberately not being invented here. A first
instinct is to flag parameters that appear together in `MECHANISMS`, but Q-032's pair does not: they appear
under "feign viability" and "contacts to kill" respectively, which is exactly why nobody noticed. Recording
the gap is the honest state.

That is consistent with the two-track freeze criterion rather than a gap here: **this gate answers the
pricing track completely and the structural track not at all.** A structural finding blocks a freeze
regardless of magnitude, and nothing in this file can tell you whether one is outstanding.
"""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from configs.default import EnvParams
from configs.provenance import (
    BOUNDED_SCOPE_MARKERS,
    CLASSES,
    MECHANISMS,
    PROPOSED_AXES,
    PROVENANCE,
    SWEPT,
    UNROOTED,
    chain,
    root_of,
)

#: Class constants rather than tunable parameters. They describe the observation's SHAPE, which is fixed for
#: the project because changing it voids every evolved weight, so they are not design choices to justify.
NOT_PARAMETERS = frozenset({"N_BASE_CHANNELS", "N_POSTURE_CHANNELS", "N_SCALARS"})

PARAM_NAMES = frozenset(f.name for f in fields(EnvParams)) - NOT_PARAMETERS


# ---------------------------------------------------------------------- 1. every parameter is classified --
def test_every_parameter_carries_a_class_and_appears_in_the_provenance():
    """**The ratchet.** A parameter added without provenance fails here, at the moment it is added.

    Asserted as set EQUALITY rather than as containment, so an entry for a parameter that no longer exists
    also fails. A stale entry is how a provenance record starts describing a design that has moved.
    """
    recorded = frozenset(PROVENANCE)
    assert recorded == PARAM_NAMES, (
        f"parameters with no provenance entry: {sorted(PARAM_NAMES - recorded)}; "
        f"entries for parameters that no longer exist: {sorted(recorded - PARAM_NAMES)}"
    )
    for name, entry in PROVENANCE.items():
        assert entry.cls in CLASSES, f"{name}: class {entry.cls!r} is not one of {CLASSES}"


def test_the_recorded_class_matches_the_tag_in_the_config():
    """Two sources of the same fact must agree, or one of them is decoration.

    `configs/default.py` tags each parameter `[D]`, `[A]` and so on in its comment or docstring. This asserts
    the machine-readable copy matches, so the two cannot drift apart silently.
    """
    src = Path("configs/default.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    tags: dict[str, str] = {}
    for i, ln in enumerate(lines):
        m = re.match(r"^    ([a-z_][a-z0-9_]*)\s*:\s*[^=]+=\s*(.+?)\s*(?:#\s*(.*))?$", ln)
        if not m or m.group(1) not in PARAM_NAMES:
            continue
        name, comment = m.group(1), m.group(3)
        tag = None
        if comment and (t := re.match(r"\[([A-Z])\]", comment)):
            tag = t.group(1)
        elif i + 1 < len(lines) and (t := re.match(r'"""\[([A-Z])\]', lines[i + 1].strip())):
            tag = t.group(1)
        if tag:
            tags[name] = tag

    disagree = {n: (t, PROVENANCE[n].cls) for n, t in tags.items() if PROVENANCE[n].cls != t}
    assert not disagree, f"class tag disagrees between config and provenance: {disagree}"
    assert len(tags) >= 45, f"only {len(tags)} tags parsed from the config; the parser has probably broken"


# ------------------------------------------------------------------------- 2. derived names its inputs --
def test_derived_parameters_name_their_inputs():
    """A derivation that names no inputs is an assertion that it was derived, not a derivation."""
    # `UNROOTED` entries are exempt BY DEFINITION rather than by convenience: the reason they are unrooted is
    # that nothing determines them, so demanding they name a determining input would be demanding they cease
    # to be the finding. They are caught by the chain-termination check instead, which is where they belong.
    missing = [
        n
        for n, e in PROVENANCE.items()
        if e.cls == "D" and not e.derives_from and e.root not in ("external", "scope") and n not in UNROOTED
    ]
    assert not missing, (
        f"{missing} are marked derived, name no inputs, and do not terminate in an external or scope root. "
        "Either they derive from something, or they are not derived."
    )


def test_every_named_input_is_itself_a_parameter():
    """A chain that points at something outside the parameter set cannot be walked, so it cannot be checked."""
    for name, entry in PROVENANCE.items():
        for src in entry.derives_from:
            assert src in PROVENANCE, f"{name} derives from {src!r}, which is not a recorded parameter"


def test_no_derivation_chain_contains_a_cycle():
    """`chain` stops on revisiting a node, so a cycle would silently truncate rather than hang."""
    for name in PROVENANCE:
        walked = chain(name)
        assert len(walked) == len(set(walked)), f"{name}: cycle in {walked}"


# ------------------------------------------------------------- 3. the literal phrase that flagged seven --
def test_no_justification_says_a_mechanic_exists_so_that_a_behaviour_becomes_possible():
    """A literal string search, and it is deliberately crude.

    **It is the exact phrasing that flagged seven mechanics in the original neutrality audit**, which is why
    it is worth its own check despite catching only one form of words.

    **Its limitation prints with its result rather than living somewhere else**: this catches the PHRASE, not
    the reasoning, and a justification rewritten in different words passes. Zero matches means the phrase is
    gone. It does not mean the thinking is. The accumulator asymmetry in DEV-019 is the demonstration: no
    decision anywhere said "instrument freezing more heavily", and the instrument did.
    """
    patterns = [
        r"so that \w+ becomes possible",
        r"so that \w+ is possible",
        r"to make \w+ possible",
        r"so \w+ can (?:emerge|appear|pay)",
    ]
    hits = []
    for name, entry in PROVENANCE.items():
        for pat in patterns:
            if re.search(pat, entry.note, re.I):
                hits.append((name, entry.note))
    assert not hits, f"justifications naming a hoped-for behaviour: {hits}"


# ------------------------------------------------------------------------ 4. chains terminate somewhere --
def test_every_chain_terminates_in_something_swept_measured_or_external():
    """**The check that "it is derived" was hiding.**

    A derived value can be exactly as arbitrary as a magic number with the arbitrariness moved one level up.
    The chain has to END somewhere that is not another choice.

    The known failures are declared in `UNROOTED` and asserted as an EXACT SET, so a NEW parameter without a
    root fails here rather than joining a growing list quietly. Same ratchet as the differential's
    `UNMODELLED`.
    """
    unrooted = {n for n in PROVENANCE if root_of(n) is None}
    assert unrooted == set(UNROOTED), (
        f"parameters whose derivation chain terminates in nothing: {sorted(unrooted)}. "
        f"Declared as known: {sorted(UNROOTED)}. "
        "A chain ending in a preference has moved the preference, not removed it."
    )


def test_the_known_unrooted_parameters_are_on_the_unswept_list():
    """A declared gap must be tracked somewhere a human reads, or declaring it is how it gets forgotten."""
    listed = Path("unswept.md").read_text(encoding="utf-8")
    for name in UNROOTED:
        assert name in listed, f"{name} is declared unrooted and does not appear in unswept.md"


# ------------------------------------------------- 5. availability AND payoff, per strategy. Claude's --
#: For each strategy: what makes it POSSIBLE, and what determines what it PAYS. The split is the check, not
#: the table: sweeping availability while fixing payoff is the failure it exists to catch, and it caught
#: `disengage_steps` and `r_near` in a pass built to find exactly that.
STRATEGY_LEVERS: dict[str, dict[str, tuple[str, ...]]] = {
    "freezing": {"availability": ("cue_dependence",), "payoff": ("disengage_steps", "r_near")},
    "fleeing": {"availability": ("e_sprint",), "payoff": ("s_burst_prey", "s_burst_pred")},
    "hiding": {"availability": ("obstacle_density",), "payoff": ("obstacle_density", "p_distant_detect")},
    "distancing": {"availability": ("r_pred",), "payoff": ("r_pred", "h")},
    # **`r_group` was here and it was wrong.** Its own justification calls it an ANALYSIS parameter, the
    # radius the `local_density` descriptor measures over, and it is free: it changes what is REPORTED, not
    # what grouping pays in the world. Grouping's payoff is the target-selection RULE, which is categorical
    # and not a parameter at all. The gate caught the miscategorisation on its first run.
    "grouping": {"availability": ("k",), "payoff": ("nearest_target_selection",)},
}


@pytest.mark.parametrize("strategy", sorted(STRATEGY_LEVERS))
def test_both_availability_and_payoff_are_accounted_for(strategy):
    """**Every lever named must be a real parameter, and its sweep status must be recorded either way.**

    This does not assert that both are swept. It asserts they are both NAMED and their status known, because
    the failure being guarded against is a payoff nobody listed. `disengage_steps` and `r_near` were exactly
    that: unswept, unlisted, and invisible to a pass asking only "does this mechanic price a strategy".
    """
    levers = STRATEGY_LEVERS[strategy]
    for kind in ("availability", "payoff"):
        assert levers[kind], f"{strategy}: no {kind} lever named"
        for lever in levers[kind]:
            assert lever in PROVENANCE or lever in MECHANISMS, (
                f"{strategy}: {kind} lever {lever!r} is neither a recorded parameter nor a declared mechanism"
            )


def test_every_unswept_payoff_lever_is_declared_on_the_unswept_list():
    """The gate's own output feeds the running list, rather than a human transcribing between them."""
    listed = Path("unswept.md").read_text(encoding="utf-8")
    for strategy, levers in STRATEGY_LEVERS.items():
        for lever in levers["payoff"]:
            if lever not in SWEPT:
                # Mechanisms have no ratio, so they are listed on unswept.md with a stated disposition
                # rather than a magnitude. The check is the same: it must be written down somewhere.
                assert lever in listed, (
                    f"{strategy}'s payoff lever {lever!r} is not swept and is not on unswept.md. An unswept "
                    "payoff that nobody has listed is the exact gap this check exists for."
                )


def test_a_proposed_axis_is_not_already_a_parameter():
    """`PROPOSED_AXES` means NOT YET A PARAMETER, so membership must be disjoint from the parameter set.

    **Without this the set is a comment.** `p_wander` sat in `PROPOSED_AXES` and in `SWEPT` and in
    `EnvParams` simultaneously for a full build step, while the prose beside the two frozensets asserted it
    was not a parameter. Nothing failed, because nothing read the claim: the design audit walks `SWEPT` for
    chain termination and never looks at `PROPOSED_AXES` at all.

    That is the stated-fact-with-no-consumer failure in its purest form, and the fix is the one CLAUDE.md
    prescribes: give the fact something that breaks when it stops being true, rather than correcting it and
    leaving it unwatched. A proposed axis that lands as a parameter now fails here until it is moved.

    **The direction matters and only one direction is asserted.** A parameter absent from both sets is not
    this test's business, and `SWEPT` membership is not required of a proposed axis. The single claim is
    that "proposed" and "exists" cannot both be true.
    """
    param_names = {f.name for f in fields(EnvParams)}
    landed = sorted(PROPOSED_AXES & param_names)
    assert not landed, (
        f"{landed} are in PROPOSED_AXES, whose stated meaning is 'not yet a parameter', and are "
        "EnvParams fields. Move each to SWEPT and remove it here. A proposed axis counted as coverage is "
        "worse than an uncovered one, because it looks like coverage."
    )


# NOTE: a second check belongs here, asserting every proposed axis is described somewhere a human reads.
# It is NOT added yet because it would fail: `prey_solid` is named nowhere in the repository outside the
# frozenset itself, and the fix is a description of what it prices, which is the owner's to write rather
# than mine to infer from the name. Q-044.


def test_a_measured_root_names_a_run_that_exists():
    """**A `measured` terminator must REFERENCE a run, and that run must be in the registry.**

    This closes a gap the audit had from the start, found when `sigma` was swept. `sigma`'s root read
    `measured` with the note "lowest-confidence value in the file; to be swept first" and **nothing had
    measured it**. The chain walk accepted it and counted it toward the checkable fraction, because the walk
    checks the KIND of terminator and cannot check that the thing it names has happened.

    **It is the inverse of the stated-fact-with-no-consumer family and belongs beside it.** There, a fact has
    nothing that breaks when it stops being true. Here, a fact HAS a consumer, the chain walk, and the
    consumer never checked whether the fact was true. A promissory note passing an audit because the audit
    checks form rather than fulfilment.

    `swept` and `external` do not need this. `swept` is checkable against the `SWEPT` set, which
    `test_every_unswept_payoff_lever_is_declared_on_the_unswept_list` already exercises, and `external`
    names a constraint outside the project's control. `measured` is the only kind whose truth depends on
    work this project must actually have done.
    """
    registry = Path("runs.md").read_text(encoding="utf-8")
    run_ids = set(re.findall(r"^\|\s*(\d+)\s*\|", registry, re.M))
    assert run_ids, "runs.md has no parseable run rows, so this check would pass vacuously"

    offenders = {}
    for name, entry in PROVENANCE.items():
        if root_of(name) != "measured":
            continue
        cited = re.findall(r"\brun (\d+)\b", entry.note)
        if not cited:
            offenders[name] = "names no run"
        elif not set(cited) <= run_ids:
            offenders[name] = f"names run(s) {sorted(set(cited) - run_ids)} absent from runs.md"
    assert not offenders, (
        f"`measured` roots that do not reference an existing run: {offenders}. A `measured` terminator is a "
        "claim that a measurement HAPPENED. Name the run, or change the root to what it actually is."
    )


def test_the_measured_root_check_can_fail():
    """The parser must reject a note with no run and a note citing a run that does not exist.

    Demonstrated in-test rather than in a log entry, because a log entry is not re-run.
    """
    registry = Path("runs.md").read_text(encoding="utf-8")
    run_ids = set(re.findall(r"^\|\s*(\d+)\s*\|", registry, re.M))

    no_run = re.findall(r"\brun (\d+)\b", "lowest-confidence value in the file; to be swept first")
    assert not no_run, "the parser found a run reference in a note that has none"

    absent = set(re.findall(r"\brun (\d+)\b", "measured by run 9999"))
    assert absent and not absent <= run_ids, "the parser did not reject a citation to a nonexistent run"


# ------------------------------------------------------ 6. the gate's own coverage. Owner's questions --
def test_a_scope_terminator_names_a_bounded_claim():
    """**`scope` is the only terminator that is a declaration rather than a checkable claim**, so without a
    constraint it accepts anything and the chain walk has moved the arbitrariness rather than removed it.

    Asking how many chains ended in `scope` found ten, of which six named a preference: a legibility target,
    "about 10 percent", "a layout choice". `r_near` was the sharpest, labelled `scope` in the same session it
    was recorded on `unswept.md` as a freezing payoff lever with a measured 5.4x range. **The gate passed it
    because `scope` accepted anything**, which would have converted a known finding into a false clearance.

    The marker test is crude on purpose. A bounded claim names what is out of scope and why, so someone can
    disagree with it; an unbounded one just says "out of scope".
    """
    offenders = {
        n: e.note
        for n, e in PROVENANCE.items()
        if root_of(n) == "scope" and not any(m in e.note.lower() for m in BOUNDED_SCOPE_MARKERS)
    }
    assert not offenders, (
        f"`scope` terminators naming no bounded claim: {offenders}. A scope terminator must say WHAT is out "
        "of scope and why, or it is a preference wearing a label."
    )


#: The terminator distribution, PINNED rather than bounded. Every other guard in this file is an exact set:
#: `UNROOTED`, the config modules, the parameter set. A floor with slack was the one guard here that permitted
#: erosion instead of requiring a decision, and a 70 percent floor at 78 percent actual would have allowed
#: four parameters to drift back into declarations without anything failing.
#:
#: Improving these numbers is a deliberate edit. So is worsening them, which is the point.
#: `checkable` went 38 to 39 when `p_wander` was added at step 11. A swept parameter terminates in
#: SWEPT, so it is checkable, and the deliberate edit is the point of pinning this rather than deriving
#: it: both directions require someone to say why. Q-042.
EXPECTED_DISTRIBUTION = {"checkable": 39, "scope": 3, "unrooted": 9}


def test_the_terminator_distribution_matches_exactly():
    """**The number that says how strong this gate actually is**, pinned rather than floored.

    If most chains ended in `scope` the walk would be laundering rather than checking. Auditing that question
    once already found six of ten `scope` terminators naming a preference, so the distribution is a live
    measure rather than a formality.
    """
    roots = [root_of(n) for n in PROVENANCE]
    actual = {
        "checkable": sum(1 for r in roots if r in ("swept", "measured", "external")),
        "scope": sum(1 for r in roots if r == "scope"),
        "unrooted": sum(1 for r in roots if r is None),
    }
    assert actual == EXPECTED_DISTRIBUTION, (
        f"terminator distribution changed: {actual} against the pinned {EXPECTED_DISTRIBUTION}. "
        "Both directions require a deliberate edit with a reason, which is the standard every other guard "
        "in this file holds to."
    )
    assert sum(actual.values()) == len(PROVENANCE), "the three categories must partition the parameters"


#: Config modules that define tunable parameters and therefore need provenance. **Asserted as an exact set**,
#: so a NEW config module fails this until someone gives it provenance or declares it exempt.
#:
#: The owner's question: population size, generation count and the mutation regime live here today, but the
#: descriptors and classifier parameters arrive at step 11 and may land in a new module. Without this, the
#: gate would silently under-cover as the project grows, and its passing would mean less each step without
#: saying so. That is the same shape as every other finding here: complete over what it knows about, and
#: nothing tells it what it does not know about.
CONFIG_MODULES_WITH_PARAMETERS = frozenset({"default.py", "analysis.py"})

#: `analysis.py` arrived at step 11 and this gate caught it, which is what the note above predicted. It is
#: covered rather than exempt: it holds values that can change a reported result, so it is held to the class
#: tag discipline below. It is NOT in `PROVENANCE`, because that table is over `EnvParams` fields and an
#: analysis parameter is a different kind of thing: two runs differing only in a value there produce the same
#: trajectories and a different reading of them. Its discipline is pre-registration, per
#: `docs/analysis-structure.md`, with the git commit as the evidence.

#: Config modules that define no tunable parameters, with the reason each is exempt.
CONFIG_MODULES_EXEMPT = {
    "provenance.py": "is the provenance record itself",
    "run_configs.py": "declares which world each arm runs in; it introduces no new tunable values",
    "__init__.py": "package marker",
}


def test_every_config_module_is_either_covered_or_declared_exempt():
    """The gate runs over `configs/provenance.py`. Nothing currently tells it when a new config module
    appears carrying parameters it has never seen."""
    present = {p.name for p in Path("configs").glob("*.py")}
    known = CONFIG_MODULES_WITH_PARAMETERS | set(CONFIG_MODULES_EXEMPT)
    assert present == known, (
        f"config modules with no provenance decision: {sorted(present - known)}; "
        f"declared modules that no longer exist: {sorted(known - present)}. "
        "A new config module must either be covered by the provenance record or declared exempt with a "
        "reason, or this gate under-covers silently as the project grows."
    )


def test_every_analysis_parameter_carries_a_class_tag():
    """`configs/analysis.py` is covered by the ratchet above but is not in `PROVENANCE`, so nothing else
    holds it to a standard. This does.

    Every module-level constant there must carry a `[D]`, `[C]`, `[A]`, `[S]` or `[F]` tag in the comment
    above it, exactly as `configs/default.py` does. Without it, a value could enter the analysis with no
    stated provenance class at all, which is the failure the whole audit exists to prevent, one module over.
    """
    src = Path("configs/analysis.py").read_text(encoding="utf-8")
    lines = src.splitlines()

    untagged = []
    for i, line in enumerate(lines):
        m = re.match(r"^([A-Z][A-Z0-9_]*)\s*:", line)
        if not m:
            continue
        name = m.group(1)
        # Walk back over the contiguous comment block above the assignment.
        tagged = False
        j = i - 1
        while j >= 0 and lines[j].lstrip().startswith("#"):
            if re.search(r"\[[DCASF]\]", lines[j]):
                tagged = True
                break
            j -= 1
        if not tagged:
            untagged.append(name)

    assert not untagged, (
        f"analysis parameters with no provenance class tag: {untagged}. Every value that can change a "
        "reported result carries its class, in this module exactly as in configs/default.py."
    )

    # Coverage, so the parser cannot silently match nothing and pass.
    names = [m.group(1) for m in (re.match(r"^([A-Z][A-Z0-9_]*)\s*:", ln) for ln in lines) if m]
    assert len(names) >= 4, f"only {len(names)} analysis parameters parsed; the parser has probably broken"
