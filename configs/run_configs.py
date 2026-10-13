"""The five run configurations, all first-class in increment one.

Build step 8's second half. The done criterion from the handoff is "E11 passes; all four run configurations
instantiate, including the `no_predator_evolve` and `naive_eval` pair".

**They live in `configs/` rather than `src/evo/` because they are declarations rather than machinery.** Each
says what world a lineage lives in, what policy it uses, and whether it evolves. The loop that consumes them
arrives at step 10.

    scripted             the control condition and the calibration reference
    no_predator_evolve   a predator-naive lineage. n_pred zero, everything else identical
    naive_eval           THE NULL. Those weights evaluated against the scripted predator, no evolution
    random_policy        the floor for gate S9. Uniform random actions, no evolution
    harmless_evolve      THE THIRD ARM. A lineage evolved against a predator that cannot damage it

**Why the null is an evaluation rather than a lineage.** The obvious null, measuring descriptors in a
predator-free world, is undefined for most of the descriptor set: `stillness_engaged` divides by a count of
engaged ticks, which is exactly zero with no predator, and the same holds for the flee, occlusion and
distance descriptors. More than half the measures would be zero over zero. The correct null is a
predator-naive population evaluated in a predator-PRESENT world, which isolates the contribution of predation
selection rather than the contribution of the predator merely existing.

**The pair imposes a sequencing dependency that the schedule must respect.** `naive_eval` consumes final
weights from a completed `no_predator_evolve` run, so the null cannot be characterised until a full
predator-free evolutionary run has finished. That is declared here as `weights_from` rather than left to be
discovered at step twelve.

**Known limitation, stated here rather than in a results discussion.** `naive_eval` takes weights into a
world containing an input those agents never saw under selection, so a difference between the evolved lineage
and the null has two possible causes, predation selection and distribution shift, and nothing in this PAIR
separates them.

**`harmless_evolve` is the third arm that bounds it, added by owner ruling 2026-08-28.** It evolves against a
predator that is present, visible, and pursues, but cannot damage. So its lineage HAS seen the predator
channel under selection, which removes distribution shift as an explanation for the difference between it and
the scripted arm.

**Its own confound is stated here, in the specification, rather than in a discussion afterwards.** Against a
harmless predator, fleeing costs stamina and buys nothing, so a responding lineage is selected AGAINST. The
arm is therefore selected against responding, which means the evolved-minus-harmless difference OVERSTATES
the selection effect rather than isolating it. **It bounds the distribution-shift explanation; it does not
decompose the magnitude.** That is what it is for, and the paper says so in advance rather than being asked.

**`damage = 0`, a parameter value, not a capture mask.** A mask would define the condition by the
completeness of the masking, and one missed path would leave the arm having evolved against a predator that
could hurt it. DEV-013 rejected exactly that option for `no_predator_evolve`, and the same reasoning applies
here: a parameter set to zero has no path that can be missed.
"""

from dataclasses import dataclass, replace

from configs.default import DEFAULT, EnvParams

#: Policies a configuration can run. `evolved` means weights from a genetic algorithm; `random` means uniform
#: random action selection, which is the S9 floor and is deliberately not the same distribution as an
#: untrained network. An untrained network under deterministic argmax emits a near-constant action and moves
#: on about 2 percent of ticks, measured, so generation zero may start BELOW random. See Q-025.
POLICIES = ("evolved", "random")


@dataclass(frozen=True)
class RunConfig:
    """One run configuration. Frozen, because a configuration that can be mutated after a run started is a
    configuration that cannot be recorded in `runs.md` and trusted."""

    name: str
    params: EnvParams
    policy: str
    evolves: bool
    #: Name of the configuration whose final weights this one consumes, or None. Only `naive_eval` sets it.
    weights_from: str | None = None

    def __post_init__(self) -> None:
        assert self.policy in POLICIES, f"{self.name}: unknown policy {self.policy!r}"
        if self.weights_from is not None:
            assert not self.evolves, (
                f"{self.name} both evolves and inherits weights. An evaluation configuration does neither "
                "its own selection nor anything that would change the weights it was given."
            )


SCRIPTED = RunConfig(
    name="scripted",
    params=DEFAULT,
    policy="evolved",
    evolves=True,
)

NO_PREDATOR_EVOLVE = RunConfig(
    name="no_predator_evolve",
    # `n_pred = 0` and nothing else. Not a masked predator: with an empty predator axis every reduction over
    # it is a no-op by construction, so there is no path that can be missed. A mask would define the
    # condition by the completeness of the masking, and a single missed path would leave the predator-naive
    # lineage having evolved with a predator. DEV-013 rejected that option for this reason.
    params=replace(DEFAULT, n_pred=0),
    policy="evolved",
    evolves=True,
)

NAIVE_EVAL = RunConfig(
    name="naive_eval",
    # The environment is identical to `scripted`. That is the whole point: the lineage changes, the world
    # does not, so any difference is attributable to what the lineage was selected under.
    params=DEFAULT,
    policy="evolved",
    evolves=False,
    weights_from="no_predator_evolve",
)

RANDOM_POLICY = RunConfig(
    name="random_policy",
    params=DEFAULT,
    policy="random",
    evolves=False,
)

HARMLESS_EVOLVE = RunConfig(
    name="harmless_evolve",
    # `damage = 0` and nothing else. The predator spawns, senses, targets, pursues and makes contact; the
    # contact simply removes no health. Every path a real predator exercises is exercised here, which is the
    # property a capture mask would not have.
    params=replace(DEFAULT, damage=0.0),
    policy="evolved",
    evolves=True,
)

ALL: tuple[RunConfig, ...] = (
    SCRIPTED,
    NO_PREDATOR_EVOLVE,
    NAIVE_EVAL,
    RANDOM_POLICY,
    HARMLESS_EVOLVE,
)

BY_NAME: dict[str, RunConfig] = {c.name: c for c in ALL}


def check_invariants() -> None:
    """Relationships between the five that must hold, asserted rather than described.

    Each of these encodes a design requirement that would otherwise live only in prose, and prose invariants
    are not checked. `configs/default.py` carries the same discipline for parameter relationships.
    """
    assert len(BY_NAME) == len(ALL), "two configurations share a name"

    # `no_predator_evolve` differs from `scripted` in EXACTLY one field. Anything else and the predator-naive
    # lineage is not a controlled comparison, it is a different experiment.
    diffs = [
        f
        for f in vars(SCRIPTED.params)
        if getattr(SCRIPTED.params, f) != getattr(NO_PREDATOR_EVOLVE.params, f)
    ]
    assert diffs == ["n_pred"], (
        f"no_predator_evolve differs from scripted in {diffs}, not only n_pred. Every other difference is an "
        "uncontrolled variable in the comparison the null depends on."
    )
    assert NO_PREDATOR_EVOLVE.params.n_pred == 0, "the predator-naive lineage must have no predator"

    # The null's world is the treatment's world.
    assert NAIVE_EVAL.params == SCRIPTED.params, (
        "naive_eval's environment must be identical to scripted's. If the worlds differ, a difference "
        "between them is not attributable to predation selection."
    )

    # The pair, and its direction.
    assert NAIVE_EVAL.weights_from == NO_PREDATOR_EVOLVE.name, (
        "naive_eval must consume no_predator_evolve's weights. The pair is not optional: measurement M5 and "
        "every Phase B threshold depend on both."
    )
    assert NO_PREDATOR_EVOLVE.weights_from is None, "the lineage must not inherit weights from anywhere"

    # **Exactly three configurations run a genetic algorithm; the other two evaluate.** The third arm was
    # added on 2026-08-28 and this exact set moved with it, which is the assertion working: pinned as a set
    # rather than a count, so an arm appearing or vanishing is a deliberate edit and not a silent one.
    evolving = {c.name for c in ALL if c.evolves}
    assert evolving == {"scripted", "no_predator_evolve", "harmless_evolve"}, (
        f"configurations that evolve are {sorted(evolving)}. An evaluation arm that evolves would change the "
        "weights it exists to evaluate."
    )

    # **`harmless_evolve` differs from `scripted` in EXACTLY one field**, the same standard
    # `no_predator_evolve` is held to. Anything else and the arm is a different experiment rather than a
    # controlled comparison, and it would bound nothing.
    harmless_diffs = [
        f for f in vars(SCRIPTED.params) if getattr(SCRIPTED.params, f) != getattr(HARMLESS_EVOLVE.params, f)
    ]
    assert harmless_diffs == ["damage"], (
        f"harmless_evolve differs from scripted in {harmless_diffs}, not just ['damage']. The arm bounds the "
        "distribution-shift explanation only if the world is otherwise identical."
    )
    assert HARMLESS_EVOLVE.params.damage == 0.0, (
        "harmless_evolve must set damage to exactly zero. A small nonzero damage would make it a weak-predator "
        "arm, which answers a different question and would be reported as this one."
    )

    # **Which configurations need a predator, asserted here rather than in the parameter invariants.**
    #
    # `configs/default.check_invariants` used to assert `n_pred >= 1` for every parameter set, which
    # `no_predator_evolve` has violated since step 8. It never fired because nothing applied that function to
    # this configuration; the gate calls THIS `check_invariants`, which shares its name and checks other
    # things. Step 12's sweep pre-flight was the first thing to run both against the same params.
    #
    # Stated per configuration is stronger than permitting zero everywhere: it pins which arm is the
    # zero-predator one, so a second arm losing its predator by accident fails here.
    needs_predator = {c.name for c in ALL if c.params.n_pred >= 1}
    assert needs_predator == {"scripted", "naive_eval", "random_policy", "harmless_evolve"}, (
        f"configurations with a predator are {sorted(needs_predator)}. Exactly one arm, "
        "no_predator_evolve, runs without one, and it is the lineage the null evaluates."
    )
    assert NO_PREDATOR_EVOLVE.params.n_pred == 0, (
        "no_predator_evolve must have exactly zero predators. DEV-013 chose a zero-length axis over a "
        "capture mask so that no path can be missed; a nonzero count reintroduces the thing the mask "
        "would have had to catch."
    )

    assert RANDOM_POLICY.policy == "random", "the S9 floor must not use evolved weights"
    assert sum(c.policy == "random" for c in ALL) == 1, "exactly one configuration is the random floor"


if __name__ == "__main__":
    check_invariants()
    print(f"{'configuration':<22}{'n_pred':>7}{'policy':>10}{'evolves':>9}  weights_from")
    for c in ALL:
        print(f"{c.name:<22}{c.params.n_pred:>7}{c.policy:>10}{str(c.evolves):>9}  {c.weights_from or '-'}")
    print("\ninvariants pass")
