"""Tiers one and three of the analysis: outcomes and descriptors. Build step 11.

Governed by `docs/analysis-structure.md`. These are engineering gates: they check the arithmetic is what the
specification says, not that any scientific claim holds.

**The properties worth asserting here are the ones that would otherwise be assumed**, and this session has
supplied several examples of assumed properties being false: a denominator of 1.888 where 300 was assumed, a
comparator saturated at 1.0, three fractions that do not sum to one because two of the flags are independent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.env.env import Accumulators
from src.measure import classifier as C
from src.measure import descriptors as D
from src.measure import discovery as DISC
from src.measure import outcomes as O

FLOOR = 5.0


def _acc(**overrides) -> dict:
    """A single prey's accumulators, zero everywhere unless named. Zero is the honest default here because a
    prey that never met a predator really does have zero engaged ticks."""
    row = {name: 0.0 for name in Accumulators._fields}
    row.update(overrides)
    return row


# ------------------------------------------------------------------ the floor is not optional --
def test_the_denominator_floor_has_no_default():
    """**Structural, in the same way invariant I1's signature is.**

    The specification derives this floor at step 13 from null data only, frozen before treatment runs. A
    default would let a caller who has not decided it silently get one, and the number would then travel into
    results as though it had been chosen.
    """
    with pytest.raises(TypeError):
        D.descriptors(_acc())  # no denominator_floor


def test_a_descriptor_below_the_floor_is_undefined_and_not_zero():
    """**NaN, never zero.** Zero reads as "the behaviour was absent"; the truth is "there was nothing to
    measure". CLAUDE.md names this for learnability, where a zero would read as collapse and the run would be
    abandoned for the wrong reason. It applies to every conditioned quantity for the same reason.

    The numbers here are run 1's: an engagement denominator of 0.262 ticks per prey.
    """
    d = D.descriptors(_acc(engaged_ticks=0.262, still_engaged=0.0), denominator_floor=FLOOR)
    se = d["stillness_engaged"]
    assert not se.defined
    assert math.isnan(se.value), f"expected NaN below the floor, got {se.value}"
    assert se.denominator == pytest.approx(0.262), "the denominator must survive even when undefined"


def test_a_defined_descriptor_carries_its_denominator():
    d = D.descriptors(_acc(engaged_ticks=20.0, still_engaged=5.0), denominator_floor=FLOOR)
    se = d["stillness_engaged"]
    assert se.defined and se.value == pytest.approx(0.25) and se.denominator == pytest.approx(20.0)


def test_a_conditional_descriptor_needs_both_arms():
    """A difference is only as defined as its worse-sampled arm. Run 1's arms differ by two orders of
    magnitude, 221 unthreatened ticks against 0.262 engaged, so this is the normal case rather than an edge
    one."""
    d = D.descriptors(
        _acc(engaged_ticks=0.262, still_engaged=0.0, unthreatened_ticks=221.0, still_unthreatened=3.0),
        denominator_floor=FLOOR,
    )
    assert d["stillness_baseline"].defined, "the well-sampled arm should be defined"
    assert not d["conditional_stillness"].defined, "the difference must not be defined when an arm is not"
    assert d["conditional_stillness"].denominator == pytest.approx(0.262), (
        "the reported denominator must be the SMALLER arm; reporting the larger would describe the "
        "difference as better sampled than it is"
    )


def test_grouping_has_the_opposite_sign_to_the_others():
    """**Grouping means moving CLOSER under threat**, so a smaller threatened distance must give a LARGER
    descriptor. Every other conditional is threatened minus baseline; this one is baseline minus threatened,
    and asserting it is what stops the sign being flipped by someone tidying the code into one helper.
    """
    d = D.descriptors(
        _acc(nn_visible_ticks=10.0, nn_sum_visible=20.0, nn_unseen_ticks=10.0, nn_sum_unseen=60.0),
        denominator_floor=FLOOR,
    )
    assert d["nn_distance_threat"].value == pytest.approx(2.0)
    assert d["nn_distance_baseline"].value == pytest.approx(6.0)
    assert d["conditional_grouping"].value == pytest.approx(4.0), (
        "closer under threat must give a POSITIVE grouping descriptor"
    )


def test_every_named_strategy_maps_to_a_descriptor_that_exists():
    """A ratchet. A named strategy pointing at a descriptor that does not exist would fail at analysis time,
    on a run, rather than here."""
    d = D.descriptors(_acc(), denominator_floor=FLOOR)
    missing = [s for s, name in D.NAMED_STRATEGIES.items() if name not in d]
    assert not missing, f"named strategies with no descriptor: {missing}"
    assert len(D.NAMED_STRATEGIES) == 5, "the design names five strategies and no more"


def test_the_saturated_comparator_is_reported_rather_than_hidden():
    """Run 1 measured `occlusion_baseline` at 1.0000: essentially every far tick is occluded. That is a
    finding about the world, so the descriptor must still return it with its denominator rather than
    suppressing or special-casing it. Q-040."""
    d = D.descriptors(_acc(far_ticks=220.0, occluded_far=220.0), denominator_floor=FLOOR)
    assert d["occlusion_baseline"].defined
    assert d["occlusion_baseline"].value == pytest.approx(1.0)


def test_population_returns_values_denominators_and_flags():
    accs = {name: np.zeros(3) for name in Accumulators._fields}
    accs["engaged_ticks"] = np.array([0.0, 10.0, 20.0])
    accs["still_engaged"] = np.array([0.0, 5.0, 5.0])
    accs["alive_ticks"] = np.array([100.0, 100.0, 100.0])
    out = D.population(accs, denominator_floor=FLOOR)
    assert out["stillness_engaged"].shape == (3,)
    assert list(out["stillness_engaged__defined"]) == [False, True, True]
    assert out["stillness_engaged"][1] == pytest.approx(0.5)
    assert math.isnan(out["stillness_engaged"][0])


# ------------------------------------------------------------------------------ tier one --
def test_the_outcome_fractions_do_not_have_to_sum_to_one():
    """**Predated and starved are independent flags and both can be true.** `EnvState` records them
    separately because deriving one cause under a precedence rule under-counts predation among starving
    agents, which are exactly the ones most likely to be caught. Anyone assuming a partition here would get
    the arithmetic wrong."""
    o = O.outcomes(
        died_predated=[True, True, False, False],
        died_starving=[True, False, True, False],
        alive_at_end=[False, False, False, True],
    )
    assert o.predated == pytest.approx(0.5)
    assert o.starved == pytest.approx(0.5)
    assert o.survived == pytest.approx(0.25)
    assert o.both == pytest.approx(0.25)
    assert o.predated + o.starved + o.survived == pytest.approx(1.25), (
        "the three fractions sum above one when a prey died of both causes, which is the point"
    )
    assert o.n == 4


def test_a_prey_with_no_outcome_is_an_error():
    """A dead prey with no recorded cause means the write-once cause recording did not fire, and it is
    invisible in every fraction."""
    with pytest.raises(ValueError, match="neither alive nor recorded"):
        O.outcomes([False], [False], [False])


def test_a_prey_that_is_alive_and_dead_is_an_error():
    with pytest.raises(ValueError, match="alive at the horizon and also carry"):
        O.outcomes([True], [False], [True])


def test_the_lineage_difference_is_componentwise():
    a = O.outcomes([True, False], [False, False], [False, True])
    b = O.outcomes([False, False], [True, False], [False, True])
    d = O.difference(a, b)
    assert d["d_predated"] == pytest.approx(0.5)
    assert d["d_starved"] == pytest.approx(-0.5)
    assert d["d_survived"] == pytest.approx(0.0)
    assert d["n_evolved"] == 2 and d["n_naive"] == 2


# ----------------------------------------------------------------- tier three, the classifier --
NULLS = {name: C.Null(0.0, 1.0) for name in D.NAMED_STRATEGIES.values()}


def test_the_classifier_thresholds_have_no_defaults():
    """Same structural argument as the denominator floor. Both are derived at step 13 from null data only and
    frozen before treatment runs, so a caller who has not derived them must not be able to get a number."""
    with pytest.raises(TypeError):
        C.classify({}, NULLS)


def test_the_largest_z_wins_when_it_clears_floor_and_margin():
    values = {name: 0.0 for name in D.NAMED_STRATEGIES.values()}
    values["conditional_stillness_near"] = 4.0
    values["conditional_away_near"] = 1.0
    r = C.classify(values, NULLS, floor=2.0, separation=1.0)
    assert r.label == "freezing"
    assert r.z == pytest.approx(4.0)
    assert r.runner_up == "fleeing" and r.margin == pytest.approx(3.0)


def test_a_winner_below_the_floor_is_unclassified():
    """**Unclassified is a real answer.** A descriptor that is merely the largest of five unremarkable ones
    is not evidence of a strategy."""
    values = {name: 0.0 for name in D.NAMED_STRATEGIES.values()}
    values["conditional_stillness_near"] = 1.0
    r = C.classify(values, NULLS, floor=2.0, separation=0.5)
    assert r.label == C.UNCLASSIFIED and r.z == pytest.approx(1.0)


def test_two_strategies_too_close_together_are_unclassified():
    """The separation margin exists because a classifier that always names something turns a tie into a
    finding."""
    values = {name: 0.0 for name in D.NAMED_STRATEGIES.values()}
    values["conditional_stillness_near"] = 4.0
    values["conditional_away_near"] = 3.9
    r = C.classify(values, NULLS, floor=2.0, separation=0.5)
    assert r.label == C.UNCLASSIFIED
    assert r.margin == pytest.approx(0.1, abs=1e-9)


def test_too_few_defined_descriptors_is_distinct_from_unclassified():
    """**"Nothing stood out" and "there was not enough data to ask" are different findings.** Collapsing them
    would report missing data as an absence of behaviour, which is the error CLAUDE.md names for learnability
    reported as zero rather than undefined.

    This is the normal case on run 1's numbers, where two of the five descriptors have a denominator of
    0.262 and are undefined.
    """
    values = {name: float("nan") for name in D.NAMED_STRATEGIES.values()}
    values["conditional_stillness_near"] = 4.0
    r = C.classify(values, NULLS, floor=2.0, separation=0.5)
    assert r.label == C.INSUFFICIENT, "one defined descriptor cannot produce a margin"
    assert r.n_defined == 1


def test_a_null_that_did_not_vary_gives_no_z_score():
    """An sd of zero would make every z infinite. Infinity is not certainty; it is an undefined comparison,
    and treating it as the strongest possible signal would let a degenerate null manufacture a result."""
    assert math.isnan(C.z_score(5.0, C.Null(0.0, 0.0)))
    assert math.isnan(C.z_score(float("nan"), C.Null(0.0, 1.0)))


def test_only_named_strategies_can_win():
    """`exposure_rate` is specified but not primary, and it is not a strategy. A descriptor present in the
    values dict but absent from `NAMED_STRATEGIES` must not be able to become a label."""
    values = {name: 0.0 for name in D.NAMED_STRATEGIES.values()}
    values["exposure_rate"] = 99.0
    nulls = dict(NULLS)
    nulls["exposure_rate"] = C.Null(0.0, 1.0)
    r = C.classify(values, nulls, floor=2.0, separation=0.5)
    assert r.label != "exposure_rate"
    assert r.label == C.UNCLASSIFIED, "the named descriptors were all at zero, so nothing should win"


def test_prevalence_reports_every_strategy_including_zero():
    """A strategy that never occurred must be a zero, not a missing key. A missing key reads as "not
    measured", and the primary result is a population-prevalence statistic."""
    p = C.prevalence(["freezing", "freezing", C.UNCLASSIFIED, "hiding"])
    assert p["freezing"] == pytest.approx(0.5)
    assert p["hiding"] == pytest.approx(0.25)
    assert p["fleeing"] == 0.0, "a strategy that did not occur must be present as zero"
    assert set(D.NAMED_STRATEGIES) <= set(p)
    assert sum(p.values()) == pytest.approx(1.0)


def test_the_null_is_fitted_from_finite_values_only():
    n = C.null_from(np.array([1.0, 2.0, 3.0, float("nan")]))
    assert n.mean == pytest.approx(2.0)
    assert n.sd == pytest.approx(1.0)
    assert C.null_from(np.array([float("nan"), float("nan")])).sd == 0.0


# -------------------------------------------------------------- tier two, unsupervised discovery --
def test_the_inert_actions_are_derived_from_the_action_set():
    """**Derived, not transcribed.** The four inert actions must track `dynamics`, or the exclusion list
    drifts from the action set it describes and starts excluding the wrong things. DEV-010's class."""
    from configs.analysis import INERT_ACTION_NAMES
    from src.env import dynamics

    assert set(INERT_ACTION_NAMES) == {"crouch", "lie_down", "attack", "drop"}
    assert dynamics.RESERVED_ACTIONS == (dynamics.ACTION_ATTACK, dynamics.ACTION_DROP), (
        "the reserved actions moved; the inert list must move with them"
    )
    assert dynamics.ACTION_CROUCH != dynamics.ACTION_LIE_DOWN


def test_the_occlusion_identity_holds_so_the_feature_is_excluded_by_construction():
    """**`occluded_near + occluded_far == unthreatened_ticks`.** The two partition near against far under
    "not visible", so an overall-occlusion feature is exactly `1 - exposure_rate`. Admitting both would admit
    one feature twice while making them look like two independent pieces of evidence.

    Asserted on real accumulator arithmetic rather than trusted, because the whole exclusion rests on it.
    """
    accs = {name: np.zeros(2) for name in Accumulators._fields}
    accs["alive_ticks"] = np.array([100.0, 100.0])
    accs["unthreatened_ticks"] = np.array([98.0, 90.0])
    accs["visible_ticks"] = np.array([2.0, 10.0])
    accs["occluded_near"] = np.array([1.0, 4.0])
    accs["occluded_far"] = np.array([97.0, 86.0])

    assert np.allclose(accs["occluded_near"] + accs["occluded_far"], accs["unthreatened_ticks"])
    f = DISC.features(accs)
    implied = (accs["occluded_near"] + accs["occluded_far"]) / accs["alive_ticks"]
    assert np.allclose(implied, 1.0 - f["exposure_rate"]), (
        "overall occlusion is not identically 1 - exposure_rate, so the exclusion's premise is wrong"
    )
    assert "occlusion_rate" in DISC.EXCLUDED_BY_IDENTITY
    assert "occlusion_rate" not in f


def test_no_candidate_feature_uses_a_threat_conditioned_denominator():
    """Tier two exists because threat-conditioned quantities have no samples. A candidate with a denominator
    of 0.262 would import exactly the problem the tier was built to avoid."""
    forbidden = {
        "engaged_ticks",
        "visible_ticks",
        "visible_unengaged_ticks",
        "near_ticks",
        "nn_visible_ticks",
    }
    offenders = [c.name for c in DISC.CANDIDATES if c.denominator in forbidden]
    assert not offenders, f"candidates conditioned on a rare denominator: {offenders}"


def test_a_feature_the_taxonomy_predicts_is_not_admitted():
    """**Feature admission is measured, not argued.** A candidate that is an exact linear function of a named
    descriptor has R-squared 1 and must be rejected: it restates the taxonomy, so tier two would rediscover
    tier three and report it as independent confirmation."""
    n = 60
    rng = np.random.default_rng(0)
    base = rng.normal(size=n)
    desc = {name: rng.normal(size=n) for name in D.NAMED_STRATEGIES.values()}
    desc["conditional_stillness_near"] = base

    accs = {name: np.full(n, 100.0) for name in Accumulators._fields}
    accs["alive_ticks"] = np.full(n, 100.0)
    accs["unthreatened_ticks"] = np.full(n, 100.0)
    accs["moved_ticks"] = 50.0 + 10.0 * base  # an exact affine function of a named descriptor
    accs["nn_unseen_ticks"] = np.full(n, 100.0)

    rows = {r.name: r for r in DISC.admit(accs, desc)}
    assert rows["activity_rate"].r2_taxonomy > 0.99
    assert not rows["activity_rate"].admitted
    assert rows["activity_rate"].reason == "inside the named taxonomy's span"


def test_a_feature_the_taxonomy_cannot_predict_is_admitted():
    n = 60
    rng = np.random.default_rng(1)
    desc = {name: rng.normal(size=n) for name in D.NAMED_STRATEGIES.values()}
    accs = {name: np.full(n, 100.0) for name in Accumulators._fields}
    accs["alive_ticks"] = np.full(n, 100.0)
    accs["unthreatened_ticks"] = np.full(n, 100.0)
    accs["nn_unseen_ticks"] = np.full(n, 100.0)
    accs["moved_ticks"] = rng.normal(50.0, 10.0, size=n)  # unrelated to any descriptor

    rows = {r.name: r for r in DISC.admit(accs, desc)}
    assert rows["activity_rate"].r2_taxonomy < 0.5
    assert rows["activity_rate"].admitted


def test_an_uncomputable_r_squared_does_not_admit():
    """**A candidate is not admitted because the check failed.** Underdetermined least squares returns a
    perfect fit, which would read as "fully explained"; returning NaN and refusing admission is the honest
    handling of too few rows."""
    n = 4  # fewer rows than the six coefficients a 5-predictor fit needs
    desc = {name: np.arange(n, dtype=float) for name in D.NAMED_STRATEGIES.values()}
    accs = {name: np.full(n, 100.0) for name in Accumulators._fields}
    accs["alive_ticks"] = np.full(n, 100.0)
    accs["unthreatened_ticks"] = np.full(n, 100.0)
    accs["nn_unseen_ticks"] = np.full(n, 100.0)
    accs["moved_ticks"] = np.arange(n, dtype=float)

    rows = {r.name: r for r in DISC.admit(accs, desc)}
    assert math.isnan(rows["activity_rate"].r2_taxonomy)
    assert not rows["activity_rate"].admitted
    assert rows["activity_rate"].reason == "R-squared not computable"


def test_r_squared_drops_rows_where_a_descriptor_is_undefined():
    """An undefined descriptor must not be silently read as zero. The surviving row count is reported so the
    caller can see how much of the population the regression actually used, which matters here because two
    of the five named descriptors are undefined for most individuals on this project's numbers."""
    n = 40
    rng = np.random.default_rng(2)
    y = rng.normal(size=n)
    x = rng.normal(size=(n, 2))
    x[:20, 0] = np.nan
    assert math.isfinite(DISC.r_squared(y, x))
    x[:, 0] = np.nan
    assert math.isnan(DISC.r_squared(y, x)), "no rows survive, so there is nothing to fit"


def test_clustering_is_deterministic_given_a_seed():
    """Gates E1 and S7 both require a run to reproduce. A clustering that moved between runs would break
    reproduction at the ANALYSIS stage rather than the simulation stage, which is harder to notice because
    every simulation gate would still be green."""
    rng = np.random.default_rng(3)
    x = np.vstack([rng.normal(0, 0.3, size=(40, 2)), rng.normal(4, 0.3, size=(40, 2))])
    a = DISC.cluster(x, key_seed=7)
    b = DISC.cluster(x, key_seed=7)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert a[1] == 2, f"two well-separated blobs should give k=2, got {a[1]}"
    assert a[2] > 0.5, f"silhouette on separated blobs should be high, got {a[2]}"


def test_a_candidate_already_spanned_by_the_accepted_set_is_rejected():
    """**Stage two, and stage one cannot do its job.**

    Two candidates that are exact transformations of each other, both orthogonal to the taxonomy, both clear
    stage one. Admitting both double-weights that axis in every distance computation, so the clusters bend
    toward whatever it measures. That is worse than a weak feature, because a weak feature only dilutes.

    Constructed so `local_density` is an exact affine function of `activity_rate` and neither is predictable
    from the named descriptors, which is precisely the configuration stage one is blind to.
    """
    n = 80
    rng = np.random.default_rng(11)
    desc = {name: rng.normal(size=n) for name in D.NAMED_STRATEGIES.values()}

    signal = rng.normal(size=n)
    accs = {name: np.full(n, 100.0) for name in Accumulators._fields}
    accs["alive_ticks"] = np.full(n, 100.0)
    accs["unthreatened_ticks"] = np.full(n, 100.0)
    accs["nn_unseen_ticks"] = np.full(n, 100.0)
    accs["moved_ticks"] = 50.0 + 10.0 * signal
    accs["density_sum"] = 300.0 - 20.0 * signal  # exactly affine in activity_rate

    rows = {r.name: r for r in DISC.admit(accs, desc)}
    assert rows["activity_rate"].admitted, "the first of the pair should be admitted"
    assert rows["activity_rate"].r2_taxonomy < 0.5, "and it is orthogonal to the taxonomy"

    assert rows["local_density"].r2_taxonomy < 0.5, (
        "the duplicate is ALSO orthogonal to the taxonomy, which is why stage one cannot catch it"
    )
    assert rows["local_density"].r2_accepted > 0.99
    assert not rows["local_density"].admitted
    assert rows["local_density"].reason == "already spanned by the accepted features"


def test_stage_two_does_not_reject_merely_correlated_features():
    """Stage two targets near-collinear duplication, not correlation. A feature sharing some variance with an
    accepted one still carries its own, and rejecting it would strip the tier of exactly the residual power
    it exists for. The threshold is higher than stage one's for this reason."""
    n = 200
    rng = np.random.default_rng(12)
    desc = {name: rng.normal(size=n) for name in D.NAMED_STRATEGIES.values()}
    shared = rng.normal(size=n)

    accs = {name: np.full(n, 100.0) for name in Accumulators._fields}
    accs["alive_ticks"] = np.full(n, 100.0)
    accs["unthreatened_ticks"] = np.full(n, 100.0)
    accs["nn_unseen_ticks"] = np.full(n, 100.0)
    accs["moved_ticks"] = 50.0 + 10.0 * shared
    accs["density_sum"] = 300.0 + 20.0 * shared + 30.0 * rng.normal(size=n)  # correlated, not duplicated

    rows = {r.name: r for r in DISC.admit(accs, desc)}
    assert rows["activity_rate"].admitted
    assert 0.0 < rows["local_density"].r2_accepted < 0.95
    assert rows["local_density"].admitted, "a correlated feature still carries its own variance"


def test_the_first_candidate_has_nothing_to_be_redundant_against():
    n = 60
    rng = np.random.default_rng(13)
    desc = {name: rng.normal(size=n) for name in D.NAMED_STRATEGIES.values()}
    accs = {name: np.full(n, 100.0) for name in Accumulators._fields}
    accs["alive_ticks"] = np.full(n, 100.0)
    accs["unthreatened_ticks"] = np.full(n, 100.0)
    accs["nn_unseen_ticks"] = np.full(n, 100.0)
    accs["moved_ticks"] = rng.normal(50.0, 10.0, size=n)

    rows = {r.name: r for r in DISC.admit(accs, desc)}
    assert math.isnan(rows["activity_rate"].r2_accepted), (
        "an empty comparison has no answer; reporting 0.0 would claim it was checked and found independent"
    )
    assert rows["activity_rate"].admitted


def test_the_accepted_matrix_holds_only_admitted_features_in_order():
    """A caller clustering on the rejected features would produce a result nobody could reproduce from the
    admission table, which is the kind of slip that is invisible in the output."""
    n = 80
    rng = np.random.default_rng(14)
    desc = {name: rng.normal(size=n) for name in D.NAMED_STRATEGIES.values()}
    signal = rng.normal(size=n)
    accs = {name: np.full(n, 100.0) for name in Accumulators._fields}
    accs["alive_ticks"] = np.full(n, 100.0)
    accs["unthreatened_ticks"] = np.full(n, 100.0)
    accs["nn_unseen_ticks"] = np.full(n, 100.0)
    accs["moved_ticks"] = 50.0 + 10.0 * signal
    accs["density_sum"] = 300.0 - 20.0 * signal

    rows = DISC.admit(accs, desc)
    names, matrix = DISC.accepted_matrix(rows, accs)
    admitted = [r.name for r in rows if r.admitted]
    assert names == admitted, "the matrix must follow the admission decisions and their order"
    assert matrix.shape == (n, len(admitted))
    assert "local_density" not in names
