"""The S4 calibration driver, D-083 and Q-057: the grid as ruled, the pass rule, the stage 2 guard, blindness to
behaviour, and that its evolution is the sweep's evolution bit for bit."""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import DEFAULT
from src.agent import network as net
from src.evo import genome
from src.measure import learnability

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "calibrate_s4", ROOT / "scripts" / "measure" / "calibrate_s4.py"
)
cal = importlib.util.module_from_spec(_spec)
sys.modules["calibrate_s4"] = cal
_spec.loader.exec_module(cal)


def rec(seed, after, coverage, median):
    return {"seed": seed, "after": after, "coverage": coverage, "median_learnability": median}


def full(coverage=0.6, median=0.2, override=None):
    """Every (seed, generation) record of one setting, all passing unless overridden per (seed, after)."""
    out = []
    for s in cal.SEEDS:
        for a in cal.MEASURE_AFTER:
            c, m = (override or {}).get((s, a), (coverage, median))
            out.append(rec(s, a, c, m))
    return out


# ---------------------------------------------------------------------------------------- the grid ----
def test_the_grid_is_the_one_ruled():
    one, two = cal.settings(1), cal.settings(2)
    assert len(one) == len(two) == 9
    assert {s["damage"] for s in one} == {DEFAULT.damage} and {s["damage"] for s in two} == {50.0}
    assert {s["h_regen"] for s in one} == {0.25, 0.1, 0.0} and {s["p_wander"] for s in one} == {0.0, 0.5, 1.0}
    assert cal.SEEDS == (100, 101, 102) and cal.MEASURE_AFTER == (100, 300) and cal.EPISODES == 16
    assert not set(cal.SEEDS) & set(range(40)), (
        "calibration seeds must be disjoint from Phase A and the long run"
    )


def test_every_setting_moves_only_the_ruled_parameters():
    base = dataclasses.asdict(cal.sweep.params_for("treatment", "baseline", None))
    for s in cal.settings(1) + cal.settings(2):
        moved = {k for k, v in dataclasses.asdict(cal.params_for(s)).items() if v != base[k]}
        assert moved <= set(cal.MOVED), moved
    # The default point of the stage 1 grid IS the current world, so the grid contains the status quo.
    assert (
        cal.n_moved({"h_regen": DEFAULT.h_regen, "p_wander": DEFAULT.p_wander, "damage": DEFAULT.damage}) == 0
    )
    assert any(cal.n_moved(s) == 0 for s in cal.settings(1))


# ----------------------------------------------------------------------------------- the pass rule ----
def test_a_setting_passes_only_with_coverage_and_median_in_every_seed_at_both_generations():
    assert cal.setting_passes(full()) is True
    assert (
        cal.setting_passes(full(override={(101, 100): (0.49, 0.2)})) is False
    )  # coverage short in one record
    assert (
        cal.setting_passes(full(override={(102, 300): (0.9, 0.159)})) is False
    )  # median short in one record
    assert (
        cal.setting_passes(full(override={(100, 300): (0.9, math.nan)})) is False
    )  # undefined is not a pass
    assert cal.setting_passes(full(coverage=0.5, median=0.16)) is True  # both floors are inclusive
    assert cal.setting_passes(full()[:-1]) is None  # a missing record is undecided, not a fail


def test_the_choice_is_the_highest_median_after_300_and_a_tie_goes_to_fewer_moves():
    grid = cal.settings(1)
    t = {cal.tag(s): {"setting": s, "records": full(median=0.17)} for s in grid}
    # Make one setting fail, give one the best generation 300 median, and check the choice follows.
    t[cal.tag(grid[0])]["records"] = full(override={(100, 100): (0.1, 0.2)})
    best = grid[5]
    t[cal.tag(best)]["records"] = full(median=0.17, override={(s, 300): (0.6, 0.24) for s in cal.SEEDS})
    assert cal.choose(t) == cal.tag(best)
    # A tie on score goes to the setting moving fewer parameters from the default. **The fixture must make the
    # two tie-breaks disagree**: the grid lists low-move settings first, so with every setting tied, "fewer moves"
    # and "earlier in the grid" pick the same one and a broken fewer-moves rule would pass unseen. Here only an
    # EARLIER two-move setting and a LATER one-move setting pass, with equal scores.
    early_two, late_one = grid[4], grid[6]
    assert cal.n_moved(early_two) == 2 and cal.n_moved(late_one) == 1  # the fixture's own precondition
    t2 = {cal.tag(s): {"setting": s, "records": full(median=0.1)} for s in grid}
    for s in (early_two, late_one):
        t2[cal.tag(s)]["records"] = full(median=0.2)
    assert cal.choose(t2) == cal.tag(late_one)
    # Nothing passes: nothing is chosen.
    t3 = {cal.tag(s): {"setting": s, "records": full(median=0.1)} for s in grid}
    assert cal.choose(t3) is None
    # Incomplete: nothing is chosen, even if what exists passes.
    t4 = {cal.tag(s): {"setting": s, "records": full(median=0.2)} for s in grid}
    t4[cal.tag(grid[3])] = {"setting": grid[3], "records": full()[:2]}
    assert cal.choose(t4) is None


def test_stage_2_runs_only_after_stage_1_completes_with_nothing_passing():
    assert not cal.stage2_allowed(None)[0]
    assert not cal.stage2_allowed({"status": "incomplete", "chosen": None})[0]
    assert not cal.stage2_allowed({"status": "complete", "chosen": "h_regen0.1__p_wander1__damage34"})[0]
    assert cal.stage2_allowed({"status": "complete", "chosen": None})[0]


# ------------------------------------------------------------------------------- the shared arithmetic ----
def test_the_learnability_columns_are_the_replays_columns():
    assert learnability.ENG_FIELDS == cal.rf.ENGAGEMENT_FIELDS


def test_exactly_n_eng_min_decided_engagements_is_defined_and_one_fewer_is_not():
    """The floor is inclusive (D-076, "at least"). Tested AT the boundary, the one value where >= and > differ."""
    eng = np.zeros((1, 2, 1, len(learnability.ENG_FIELDS)), np.int32)
    eng[0, 0, 0, 1:4] = (DEFAULT.n_eng_min - 10, 10, 0)  # exactly n_eng_min decided
    eng[0, 1, 0, 1:4] = (DEFAULT.n_eng_min - 11, 10, 0)  # one fewer
    eng[0, :, 0, 4] = 50  # censored engagements never count toward the floor
    r = learnability.s4(eng, DEFAULT.n_eng_min, DEFAULT.learnability_floor)
    assert r["defined"] == 1 and r["coverage"] == 0.5


def test_summarise_reports_s4_with_coverage_and_outcomes():
    G, E, k = 4, 2, 8
    eng = np.zeros((G, E, len(learnability.ENG_FIELDS)), np.int32)
    eng[0, :, :4] = (20, 10, 10, 0)  # 40 decided over two episodes, p 0.5, learnability 0.25
    eng[1, :, :4] = (20, 19, 1, 0)  # 40 decided, p 0.95
    prey = np.zeros((G, E, 6, k), np.int16)
    prey[0, 0, 2, :4] = 1
    r = cal.summarise(eng, prey)
    assert r["coverage"] == 0.5 and r["defined"] == 2
    assert np.isclose(r["median_learnability"], (0.25 + 0.95 * 0.05) / 2)
    assert np.isclose(r["predated"], 4 / (G * E * k))
    assert np.isclose(r["engagements_per_prey_episode"], 80 / (G * E * k))


# ----------------------------------------------------------------------------- blind to behaviour ----
def test_the_calibration_loads_no_behaviour_measure():
    """D-083: calibration on a proxy criterion only. Checked against what the import actually loads, not against
    the source text, so a transitive import through any helper is caught too."""
    probe = (
        "import importlib.util, sys, pathlib\n"
        f"p = pathlib.Path({str(ROOT / 'scripts' / 'measure' / 'calibrate_s4.py')!r})\n"
        "s = importlib.util.spec_from_file_location('cal_probe', p)\n"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        "print(sorted(n for n in sys.modules if n.startswith('src.measure')))\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True)
    loaded = out.stdout.strip().splitlines()[-1]
    assert "src.measure.learnability" in loaded, loaded  # the probe saw the module it must see
    for forbidden in ("descriptors", "classifier", "discovery"):
        assert f"src.measure.{forbidden}" not in loaded, (
            f"calibration imports src.measure.{forbidden}: {loaded}"
        )


# ---------------------------------------------------------------------------- the sweep's evolution ----
@pytest.mark.slow
def test_the_calibrations_evolution_is_the_sweeps_and_its_measurement_repeats():
    """The copied loop is a GUARDED copy: after three generations its population equals `sweep.one_seed`'s bit for
    bit, so a population here is the one the sweep's runner would produce. And measuring a population twice gives
    the same counts, because the measurement keys depend only on the seed and the generation."""
    s = cal.settings(1)[4]
    p = cal.params_for(s)
    ours = cal.evolve(p, 100, 3, (3,), 2, log=lambda m: None)
    theirs = cal.sweep.one_seed(p, 100, n_generations=3)
    assert np.array_equal(ours["pop"], theirs["final_pop"])
    assert np.array_equal(ours["fit_mean"], theirs["fit_mean"])
    static, _ = cal.build_world(p, 100)
    unflatten = genome.unflattener(net.init_params(jax.random.key(100), p))
    eng_pass = cal.rf.make_engagement_pass(unflatten, p)
    again = cal.measure(eng_pass, jnp.asarray(ours["pop"]), static, 100, 3, 2)
    assert np.array_equal(again[0], ours["measured"][3][0]) and np.array_equal(
        again[1], ours["measured"][3][1]
    )
