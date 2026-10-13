"""Pre-registered analysis parameters. Build step 11, tier two.

**These are pre-registered, not derived.** `docs/analysis-structure.md` commits tier two's feature list,
clustering method and R-squared admission threshold before any Phase B run, with the git history as the
evidence. That is a different discipline from the step 13 quantities, which are derived from null data and
frozen: those have no defaults anywhere and must be supplied by the caller. These have values, and the commit
that fixed them is the record.

**They are separate from `configs/default.py` on purpose.** That module is `EnvParams`, and the design audit
asserts set equality between its fields and the provenance table. An analysis parameter is not an environment
parameter: two runs differing only in a value here produce the same trajectories and a different reading of
them. Mixing them would make the audit's ratchet meaningless.

**Every value here can change a reported result**, so each carries its provenance class in the same scheme
`configs/default.py` uses: [D] derived, [C] conventional, [A] arbitrary and therefore a sweep centre.
"""

from __future__ import annotations

#: [C] Above this R-squared, a candidate feature is judged to lie inside the named taxonomy's span and is
#: NOT admitted to tier two. The tier's power is the residual: a feature the five named descriptors can
#: already predict adds nothing that tier three would not find.
#:
#: **Conventional, and the direction of the risk is worth stating.** Too low admits features that merely
#: restate the taxonomy, so tier two rediscovers tier three and reports it as independent confirmation. Too
#: high admits nothing and the tier is empty.
#:
#: **Its root is weak and stating that is the point.** 0.95 below corresponds to a standard collinearity
#: rule; **0.80 corresponds to no convention the owner or I can name.** It is a chosen number meaning "admit
#: a candidate if at least a fifth of its variance is new". A weak root stated is worth more than a strong
#: one implied, and the alternative here is a number that reads as derived because it sits next to one that
#: is.
R2_ADMISSION: float = 0.80

#: [C] Above this R-squared against the ALREADY ACCEPTED features, a candidate is judged redundant and is
#: not admitted. This is stage two of admission and it asks a different question from `R2_ADMISSION`.
#:
#: **Stage one alone was insufficient and the reason is worth stating.** It regresses each candidate on the
#: five NAMED DESCRIPTORS. `exposure_rate` is not one of the five, so two candidates that are exact
#: transformations of each other but orthogonal to the taxonomy both score low and both are admitted. In
#: clustering that is worse than a weak feature: a duplicated dimension double-weights its axis in every
#: distance computation, so the clusters bend toward whatever it measures.
#:
#: Higher than `R2_ADMISSION` on purpose. Stage one asks "does this add anything at all", where a moderate
#: overlap is a real reason to reject. Stage two asks "is this already in the set", where only near-collinear
#: duplication matters and merely correlated features are fine and expected.
#:
#: **Rooted in the variance-inflation convention.** VIF is `1 / (1 - R2)`, so an R-squared of 0.95 against
#: the accepted set is a VIF of 20, which is the standard line above which a predictor is treated as
#: collinear rather than merely correlated. That is a real convention rather than a preference, which is why
#: this threshold is rooted and `R2_ADMISSION` is not.
R2_REDUNDANCY: float = 0.95

#: [D] Actions with no fitness consequence in increment 1, excluded from every tier two feature.
#:
#: Crouch and lie-down do nothing because the scripted predator does not read posture. Attack and drop are
#: masked unavailable throughout. All four therefore drift freely under selection, so a cluster found in them
#: is structure in drift reported as an unnamed behavioural mode. **That is a false-positive generator inside
#: a check built to catch false negatives**, which is the worst place to put one.
#:
#: Derived from `dynamics.ACTION_CROUCH`, `ACTION_LIE_DOWN` and `RESERVED_ACTIONS` rather than transcribed,
#: so the list cannot drift from the action set it describes. Asserted in the tests.
INERT_ACTION_NAMES: tuple[str, ...] = ("crouch", "lie_down", "attack", "drop")

#: [A] Candidate range for the number of clusters. Swept, with the selection rule below, rather than fixed:
#: the design does not predict how many behavioural modes exist, and committing to one would answer tier
#: two's question by assumption.
K_RANGE: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)

#: [C] How k is chosen within `K_RANGE`. Mean silhouette, computed over the admitted features.
#:
#: Chosen because it needs no ground truth and no assumption about cluster shape beyond compactness, and
#: because it is computable from `scipy` and `numpy` alone: `scikit-learn` is not a dependency and adding one
#: mid-phase would re-resolve the lockfile, which D-021 forbids.
K_SELECTION: str = "mean-silhouette"

#: [C] Restarts for the clustering, since k-means is sensitive to initialisation. The best objective across
#: restarts is taken. An implementation constant in every respect except that too few would make the result
#: depend on the seed, which E1 and S7 would then both be right to complain about.
N_RESTARTS: int = 20


# --------------------------------------------------------------------------------------------------------
# The aggregation level for engagement-conditioned quantities. Q-047, owner's ruling 2026-08-25.
# --------------------------------------------------------------------------------------------------------
#
# **The build spec's M4 wording, "per generation", admits two readings that differ by `p_pop` and give
# opposite verdicts.** Measured engagement at the default configuration, runs 5 and 7:
#
#     per prey-episode                                   0.19 to 0.26     UNDEFINED against n_eng_min 30
#     per genome, k * n_episodes = 32 prey-episodes      6.0 to 8.4       UNDEFINED, about 4x short
#     per generation, pooled over p_pop = 256            1,536 to 2,146   DEFINED, about 51x over
#
# **The ruling is not "pick a reading". It is that the floor applies at the level the quantity is
# REPORTED**, and the two levels genuinely answer different questions:
#
#     the strategy map needs the POPULATION-level quantity, and has it comfortably
#     learnability needs PER-GENOME variance, and does not have it
#
# Learnability is defined as p(1-p) over escape outcomes, and its purpose is to measure how much outcome
# variation exists FOR SELECTION TO ACT ON. Selection acts per genome. A rate pooled across a population is
# a different statistic: 5 percent of genomes escaping reliably while 95 percent never engage gives the same
# pooled p as a uniform population, and only the second is a fact about selectable variation.

#: [D] The level at which `n_eng_min` is applied, per quantity class. Not one value, because the ruling is
#: that the floor follows the reporting level rather than being fixed globally.
LEARNABILITY_AGGREGATION: dict[str, str] = {
    "population_descriptors": "per_generation",
    "learnability": "per_genome",
}

#: [D] Engagement-conditioned descriptors reported at the per-genome level are UNDEFINED at the default
#: configuration, and this records that as a commitment rather than leaving it to be discovered at analysis
#: time. **Learnability for those descriptors is reported as undefined, never as a number computed on about
#: eight engaged ticks.**
#:
#: **This is Q-032 arriving one level down rather than a new problem.** Engagement is rare, so everything
#: conditioned on it is starved, and learnability needs it at the finest grain available. It is a real
#: limitation of the current configuration and not a defect: DEV-014, DEV-030 and DEV-031 were defects, and
#: Q-023 established that the axes now work.
LEARNABILITY_UNDEFINED_AT_DEFAULT: bool = True


# --------------------------------------------------------------------------------------------------------
# How the characterisation sweep compares arms. D-045, pre-registered before step 12.
# --------------------------------------------------------------------------------------------------------
#
# **Every arm of the sweep runs the same seeds**, so every comparison is paired and must be stated as one.
# Comparing group means against a pooled seed range discards information the run has already paid for,
# because that spread is largely a property of the seed rather than of the arm.
#
# The instance that produced this rule is run 11. Its decision rule was fixed in advance, which was right,
# and it was the weaker of two available comparisons, which was not. Paired, both levels above the default
# beat it on 3 of 3 seeds; the pooled-range rule could not see that at all.
#
# **The pre-registered rule was HONOURED and `sigma` was not moved.** The paired analysis was devised after
# seeing the data, and overriding a pre-registered rule with a better analysis constructed afterwards is
# exactly what pre-registration prevents. It does not stop being that because the better analysis is
# genuinely better. This constant is that lesson applied FORWARD, where it costs nothing.

#: [D] Comparisons across sweep arms are paired on seed. Not a preference: the arms share seeds by
#: construction, so the unpaired reading answers a question nobody asked.
PAIRED_COMPARISON: bool = True

#: [C] The paired test. A sign test over per-seed differences, chosen because it assumes nothing about the
#: distribution of a fitness difference and this project has no basis for assuming one.
PAIRED_TEST: str = "sign-test"

#: [D] **The power limit, recorded WITH the rule so the rule is not over-read.** Pairing does not manufacture
#: power. At three seeds a unanimous 3 of 3 is p = 0.125 one-sided, which is the smallest p the design can
#: produce, so no paired result at three seeds can reach 0.05 however it is read.
#:
#: Pairing moves a comparison from uninformative to suggestive, not from uninformative to conclusive. Q-030's
#: seed-count problem is untouched by it and is not addressed here.
PAIRED_MIN_P_AT_THREE_SEEDS: float = 0.125
