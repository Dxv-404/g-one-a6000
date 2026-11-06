"""Run 52's arithmetic on hand-built arrays, D-082. The data itself is read only by the script."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "replay_analysis", ROOT / "scripts" / "measure" / "replay_analysis.py"
)
ra = importlib.util.module_from_spec(_spec)
sys.modules["replay_analysis"] = ra
_spec.loader.exec_module(ra)

S, G, E, K = 2, 3, 2, 4


def arrays():
    prey = np.zeros((S, G, E, 6, K), np.int16)
    eng = np.zeros((S, G, E, 7), np.int32)
    return prey, eng


def test_per_seed_quantities_are_the_means_d082_names():
    prey, eng = arrays()
    prey[0, 0, 0, ra.PREY["died_predated"], 0] = 1  # one of 24 prey-episodes in seed 0
    prey[1, :, :, ra.PREY["died_starving"], :] = 1  # all of seed 1
    prey[0, :, :, ra.PREY["alive_ticks"], :] = 100
    eng[0, 0, 0, :5] = (3, 2, 1, 0, 0)  # 3 engagements: 2 escaped, 1 captured
    eng[0, 1, 1, :5] = (2, 0, 1, 0, 1)  # 2: 1 captured, 1 censored, which is excluded
    q = ra.per_seed(prey, eng)
    assert np.isclose(q["predated"][0], 1 / (G * E * K)) and q["predated"][1] == 0
    assert q["starved"][1] == 1 and q["starved"][0] == 0
    assert q["time_alive"][0] == 100 and q["time_alive"][1] == 0
    assert np.isclose(q["escape_rate"][0], 2 / 4)
    assert np.isnan(q["escape_rate"][1])
    assert np.isclose(q["engagements"][0], 5 / (G * E * K))


def test_a_genomes_escape_rate_is_undefined_below_n_eng_min():
    prey, eng = arrays()
    eng[0, 0, 0, :4] = (40, 30, 10, 0)  # 40 engagements, defined: p 0.75
    eng[0, 1, 0, :4] = (29, 29, 0, 0)  # 29 decided, undefined
    g = ra.per_genome(prey, eng)
    assert np.isclose(g["escape_rate"][0, 0], 0.75)
    assert np.isnan(g["escape_rate"][0, 1])


def test_s4_counts_coverage_and_band_per_genome():
    prey, eng = arrays()
    eng[0, 0, 0, :4] = (40, 20, 20, 0)  # p 0.5, learnability 0.25, in band
    eng[0, 1, 0, :4] = (40, 38, 2, 0)  # p 0.95, learnability 0.0475, out
    r = ra.s4(eng)
    assert r["defined"] == 2 and np.isclose(r["coverage"], 2 / (S * G))
    assert np.isclose(r["share_in_band_of_defined"], 0.5)
    assert np.isclose(r["share_in_band_of_all"], 1 / (S * G))


def test_exceedance_is_five_percent_for_identical_populations_and_more_for_a_shifted_one():
    rng = np.random.default_rng(0)
    null = rng.normal(size=(3, 4000))
    above, below = ra.exceedance(null.copy(), null)
    assert np.all(np.abs(above - 0.05) < 0.002) and np.all(np.abs(below - 0.05) < 0.002)
    above, below = ra.exceedance(null + 1.0, null)
    assert np.all(above > 0.2) and np.all(below < 0.01)
    few = np.full((1, 50), np.nan)
    few[0, :5] = 1.0
    assert np.isnan(ra.exceedance(few, null[:1])[0][0])
