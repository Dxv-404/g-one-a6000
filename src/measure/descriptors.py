"""The five named strategies, as descriptors over accumulators. Build step 11, tier three.

Governed by `docs/analysis-structure.md` 1.0.0.0. **Tier three is hypotheses tested against what tiers one
and two found, not the frame that decides what can be seen.** A named class that never forms a cluster was
not present; a cluster matching no named class is a finding that requires looking at trajectories.

**Every threat-conditioned descriptor carries its denominator.** This is the specification's requirement and
it is the whole reason this module returns a `Descriptor` rather than a float: measured on run 1, the
engagement denominator is 0.262 ticks per prey and the visible denominator is 1.888. **A value computed over
0.262 ticks is not a value, and nothing in the number itself says so.** Returning a bare float would let it
travel into a table, a plot and a claim with the sample size stripped, which is precisely how a stated fact
loses its qualifier.

**The denominator floor is NOT set here and has no default.** The specification derives it at step 13 from
null data only, frozen before treatment runs. `denominator_floor` is therefore a required keyword argument:
a caller that has not decided it cannot silently get one. That is the same structural move as invariant I1's
signature, applied to a threshold rather than to state.

**These are pure functions over accumulators.** They never touch the environment, a trajectory, or a score,
so a descriptor cannot come to depend on anything the accumulator set does not already carry.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Descriptor(NamedTuple):
    """A descriptor value that cannot be read without its sample size.

    `defined` is false when the denominator is below the floor. **The value is then NaN rather than zero**,
    because zero reads as "the behaviour was absent" and the truth is "there was nothing to measure". That
    distinction is the one CLAUDE.md names for learnability, and it applies to every conditioned quantity for
    the same reason: a run abandoned because a descriptor read zero would be abandoned for the wrong reason.
    """

    value: float
    denominator: float
    defined: bool


UNDEFINED = Descriptor(float("nan"), 0.0, False)


def _rate(num, den, floor: float) -> Descriptor:
    num, den = float(num), float(den)
    if den < floor or den <= 0.0:
        return Descriptor(float("nan"), den, False)
    return Descriptor(num / den, den, True)


def _difference(threatened: Descriptor, baseline: Descriptor) -> Descriptor:
    """A conditional descriptor is a difference, and it is defined only if BOTH arms are.

    The reported denominator is the SMALLER of the two, because that is what limits the difference. Reporting
    the larger would describe a quantity as better sampled than it is, and on this project's numbers the two
    arms differ by two orders of magnitude: `unthreatened_ticks` is 221 and `engaged_ticks` is 0.262.
    """
    if not (threatened.defined and baseline.defined):
        return Descriptor(float("nan"), min(threatened.denominator, baseline.denominator), False)
    return Descriptor(
        threatened.value - baseline.value, min(threatened.denominator, baseline.denominator), True
    )


def _mean(total, count, floor: float) -> Descriptor:
    """A mean over a counted set, which differs from `_rate` only in that the numerator is a sum of values
    rather than a count of events. Kept separate so the two cannot be confused at a call site."""
    return _rate(total, count, floor)


def descriptors(acc, *, denominator_floor: float) -> dict[str, Descriptor]:
    """Every descriptor in the build spec's block, plus `exposure_rate`, from one prey's accumulators.

    `acc` is anything with the `Accumulators` field names: the NamedTuple, a dict, or a row of arrays. Values
    are read with `getattr` falling back to indexing, so a per-prey slice of the vectorised accumulators works
    without a conversion step.

    **`exposure_rate` is specified but NOT primary**, per the analysis structure, and the reason is recorded
    rather than left implicit: it was proposed as a measure of avoidance and measurement showed it tracks
    ACTIVITY. Across 900 generations it rose 370 percent while activity rose 860, and the activity-normalised
    version's apparent fall is an artefact of the generation-zero point where activity is 0.031. It is
    reported because it is informative about the world, not because it measures avoidance.
    """
    f = denominator_floor

    def g(name):
        try:
            return getattr(acc, name)
        except AttributeError:
            return acc[name]

    # ---- freezing ---------------------------------------------------------------------------
    stillness_engaged = _rate(g("still_engaged"), g("engaged_ticks"), f)
    stillness_baseline = _rate(g("still_unthreatened"), g("unthreatened_ticks"), f)
    accidental_stillness = _rate(g("accidental_still"), g("engaged_ticks"), f)

    # ---- fleeing ----------------------------------------------------------------------------
    flee_engaged = _rate(g("flee_engaged"), g("engaged_ticks"), f)
    flee_baseline = _rate(g("flee_visible_unengaged"), g("visible_unengaged_ticks"), f)

    # ---- the near-conditioned forms, D-046, and why they are the PRIMARY conditioned forms --
    # Run 20 planted all five strategies and the engagement-conditioned forms above detected none. The
    # mechanism is self-defeat: successful freezing prevents and ends engagement, so `stillness_engaged`
    # loses its own denominator, and a natural fleer responds to VISIBILITY rather than the engagement
    # flag, so `conditional_flee`'s baseline subtracts exactly the signal it is looking for. `near_ticks`
    # is geometric, nearest predator within r_pred whether or not either can see the other, and no prey
    # behaviour can erase it. The far side is the baseline arm, geometric for the same reason. The
    # engagement forms above are still reported: they are informative about the capture window, they are
    # just not the detection instrument.
    stillness_near = _rate(g("still_near"), g("near_ticks"), f)
    stillness_far = _rate(g("still_far"), g("far_ticks"), f)
    accidental_stillness_near = _rate(g("accidental_still_near"), g("near_ticks"), f)
    flee_near = _rate(g("flee_near"), g("near_ticks"), f)
    flee_far = _rate(g("flee_far"), g("far_ticks"), f)

    # The EMITTED-direction flee forms, D-048, and why fleeing's detection uses them. `flee_near` keys on
    # the realized distance increase, and against a pursuing predator at matched speed fleeing cannot
    # increase distance: the run 22 diagnostic measured the planted fleer BELOW the forager on flee_near.
    # `away_near` keys on the emitted move's direction, which is prey-attributable and cannot be erased by
    # the predator's motion. The realized forms are still reported: their gap against the away forms is
    # itself informative about pursuit.
    away_near = _rate(g("away_near"), g("near_ticks"), f)
    away_far = _rate(g("away_far"), g("far_ticks"), f)

    # ---- occlusion seeking ------------------------------------------------------------------
    # Measured on run 1, `occlusion_baseline` saturates at 1.0000: essentially every far tick is occluded, so
    # the comparator is a constant and the conditional form carries no information the raw term does not.
    # Reported anyway, with its denominator, because a saturated comparator is a finding about the world and
    # suppressing it here would hide it. Q-040.
    occlusion_near = _rate(g("occluded_near"), g("near_ticks"), f)
    occlusion_baseline = _rate(g("occluded_far"), g("far_ticks"), f)

    # ---- distance maintenance ---------------------------------------------------------------
    # **`mean_distance` is RETIRED as distancing's evidence and kept only as a report, D-065.** Its
    # denominator, `visible_ticks`, counts the PREDATOR's detection of the prey, which being hunted causes:
    # conditioning on it is collider conditioning, and predated prey read 0.45 with 90.6 percent of their
    # visible ticks adjacent. It measured pursuit.
    mean_distance = _mean(g("dist_sum_visible"), g("visible_ticks"), f)

    # ---- distancing as the prey's OWN distance change, D-065 ----------------------------------
    # Per perceived tick, how far the prey's emitted move changed its distance from the predator held where
    # it was, MINUS what a direction chosen at random at the same reach would have changed it: so a
    # predator-blind policy scores zero in expectation whatever the geometry, and only a direction chosen
    # BECAUSE of the predator moves it. The far baseline is the same quantity with the predator out of
    # sight, and the occluded control the same distances with the line of sight blocked, which is the
    # cleanest contrast (it differs from the perceived ticks only in perception) and the least sampled.
    dgain_perceived = _mean(g("dg_perc_excess_sum"), g("dg_perc_ticks"), f)
    dgain_far = _mean(g("dg_far_excess_sum"), g("dg_far_ticks"), f)
    dgain_occluded = _mean(g("dg_occl_excess_sum"), g("dg_occl_ticks"), f)
    dgain_band_a = _mean(g("dg_a_excess_sum"), g("dg_a_ticks"), f)
    dgain_band_b = _mean(g("dg_b_excess_sum"), g("dg_b_ticks"), f)
    # Reports, not evidence: the raw change carries the geometric bias, the executed change the walls and
    # the mask, and efficiency is executed gain over the best gain available.
    dgain_perceived_raw = _mean(g("dg_perc_emit_sum"), g("dg_perc_ticks"), f)
    dgain_executed = _mean(g("dg_perc_exec_sum"), g("dg_perc_ticks"), f)
    dgain_efficiency = _rate(g("dg_perc_exec_sum"), g("dg_perc_max_sum"), f)
    # The OUTCOME form: distance after each tick a predator exists. Reported beside capture, never as
    # evidence of behaviour, because surviving longer and being far are confounded through survival.
    mean_distance_alive = _mean(g("dist_sum_alive"), g("pred_present_ticks"), f)

    # ---- the flight-initiation-distance analogue, D-065 ---------------------------------------
    flight_rate = _rate(g("fid_fled"), g("fid_encounters"), f)
    flight_initiation_distance = _mean(g("fid_dist_sum"), g("fid_fled"), f)
    flight_latency = _mean(g("fid_latency_sum"), g("fid_fled"), f)
    onset_distance = _mean(g("fid_onset_sum"), g("fid_encounters"), f)

    # ---- grouping ---------------------------------------------------------------------------
    # The conditional form is baseline MINUS threat, the opposite order to the others, because grouping means
    # moving CLOSER under threat and a smaller distance must give a larger descriptor. Written out rather
    # than folded into `_difference` so the sign is visible at the point it is decided.
    nn_threat = _mean(g("nn_sum_visible"), g("nn_visible_ticks"), f)
    nn_baseline = _mean(g("nn_sum_unseen"), g("nn_unseen_ticks"), f)
    conditional_grouping = _difference(nn_baseline, nn_threat)

    # ---- unconditional ----------------------------------------------------------------------
    # No floor: these are over `alive_ticks`, which is about 223 per prey. Passing the floor anyway keeps one
    # code path, and a floor that never binds on a denominator of 223 costs nothing.
    local_density = _mean(g("density_sum"), g("alive_ticks"), f)
    activity_rate = _rate(g("moved_ticks"), g("alive_ticks"), f)
    exposure_rate = _rate(g("visible_ticks"), g("alive_ticks"), f)

    return {
        "stillness_engaged": stillness_engaged,
        "stillness_baseline": stillness_baseline,
        "conditional_stillness": _difference(stillness_engaged, stillness_baseline),
        "accidental_stillness": accidental_stillness,
        "flee_engaged": flee_engaged,
        "flee_baseline": flee_baseline,
        "conditional_flee": _difference(flee_engaged, flee_baseline),
        "stillness_near": stillness_near,
        "stillness_far": stillness_far,
        "conditional_stillness_near": _difference(stillness_near, stillness_far),
        "accidental_stillness_near": accidental_stillness_near,
        "flee_near": flee_near,
        "flee_far": flee_far,
        "conditional_flee_near": _difference(flee_near, flee_far),
        "away_near": away_near,
        "away_far": away_far,
        "conditional_away_near": _difference(away_near, away_far),
        "occlusion_near": occlusion_near,
        "occlusion_baseline": occlusion_baseline,
        "conditional_occlusion": _difference(occlusion_near, occlusion_baseline),
        "mean_distance": mean_distance,
        "dgain_perceived": dgain_perceived,
        "dgain_far": dgain_far,
        "dgain_occluded": dgain_occluded,
        "conditional_dgain": _difference(dgain_perceived, dgain_far),
        "dgain_sight_contrast": _difference(dgain_perceived, dgain_occluded),
        "dgain_band_a": dgain_band_a,
        "dgain_band_b": dgain_band_b,
        "dgain_perceived_raw": dgain_perceived_raw,
        "dgain_executed": dgain_executed,
        "dgain_efficiency": dgain_efficiency,
        "mean_distance_alive": mean_distance_alive,
        "flight_rate": flight_rate,
        "flight_initiation_distance": flight_initiation_distance,
        "flight_latency": flight_latency,
        "onset_distance": onset_distance,
        "nn_distance_threat": nn_threat,
        "nn_distance_baseline": nn_baseline,
        "conditional_grouping": conditional_grouping,
        "local_density": local_density,
        "activity_rate": activity_rate,
        "exposure_rate": exposure_rate,
    }


#: The descriptors tier three treats as evidence for each named strategy. Grouping's entry is the
#: conditional form, and hiding's is the conditional occlusion, so every entry is a difference against the
#: agent's own baseline where one exists.
#:
#: **Freezing and fleeing point at the NEAR-conditioned forms, D-046 and D-048.** The
#: engagement-conditioned forms failed calibration outright at run 20 (planted freezing undefined, planted
#: fleeing anti-detected) for the self-defeat mechanism documented above; the near forms condition on
#: geometry the behaviour cannot erase. Fleeing's is the EMITTED-direction form, because the realized form
#: is erased by pursuit (run 22 diagnostic). Hiding's conditional occlusion was already near/far based,
#: and grouping keeps its form until the battery says otherwise.
#:
#: **Distancing points at `conditional_dgain`, D-065, chosen by the planted battery (run 39).** The prey's
#: own chance-corrected distance change on perceived ticks, minus the same on far ticks. Over 32 worlds per
#: policy it detected the planted distancer at z 2.39 (default) and 2.27 (high pressure), against
#: `mean_distance`'s 1.11 and 1.24, which it replaces as evidence.
#:
#: **And it carries a residual bias that no accumulator removes, stated here so nobody reads its zero as
#: "no avoidance".** Predator-blind policies with a persistent heading score NEGATIVE on it (forager -0.37,
#: idlers -0.28 and -0.20 at the default): a prey that keeps its heading tends to bring itself into view of
#: a predator BY heading toward it, so conditioning on perception selects ticks where it already was. Any
#: perception-conditioned movement measure inherits this; the occluded contrast,
#: `dgain_sight_contrast`, reduces it most and is reported as the within-prey control, but is defined for
#: about half the prey. **The control that removes it is the design's own: every lineage is compared with
#: the paired predator-naive lineage, which has the same selection.** Never read this descriptor against
#: zero.
NAMED_STRATEGIES: dict[str, str] = {
    "freezing": "conditional_stillness_near",
    "fleeing": "conditional_away_near",
    "hiding": "conditional_occlusion",
    "distancing": "conditional_dgain",
    "grouping": "conditional_grouping",
}


def population(accs, *, denominator_floor: float) -> dict[str, np.ndarray]:
    """Descriptors for a whole population: arrays of values, denominators and defined flags per name.

    Returns three arrays per descriptor rather than an array of `Descriptor`, because every downstream
    consumer wants the values as a vector and the denominators as a mask, and packing them together would
    make both awkward.
    """
    n = len(
        np.asarray(
            getattr(accs, "alive_ticks", None) if hasattr(accs, "alive_ticks") else accs["alive_ticks"]
        )
    )
    # **Every field the accumulators carry, not a hand-kept list.** The list this replaced named 28 fields
    # and would have silently dropped every D-065 field, so a descriptor reading one would have raised far
    # from the cause, or worse, read a default. Reading the fields that exist cannot fall behind them.
    names = list(accs._fields) if hasattr(accs, "_fields") else list(accs)
    rows = []
    for i in range(n):
        row = {}
        for name in names:
            v = getattr(accs, name) if hasattr(accs, name) else accs[name]
            row[name] = np.asarray(v)[i]
        rows.append(descriptors(row, denominator_floor=denominator_floor))

    names = list(rows[0])
    out: dict[str, np.ndarray] = {}
    for name in names:
        out[name] = np.array([r[name].value for r in rows], dtype=np.float64)
        out[name + "__denominator"] = np.array([r[name].denominator for r in rows], dtype=np.float64)
        out[name + "__defined"] = np.array([r[name].defined for r in rows], dtype=bool)
    return out
