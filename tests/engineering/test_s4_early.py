"""D-089's diagnosis verdicts, fixed before the diagnosis ran: EARLY IN BAND needs D-083's per-point rule at
generation 0, 10 or 25 in at least 2 of 3 seeds; the answer is A if any setting is early in band, else B."""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("s4_early", ROOT / "scripts" / "measure" / "s4_early.py")
early = importlib.util.module_from_spec(_spec)
sys.modules["s4_early"] = early
_spec.loader.exec_module(early)


def recs(passing_at: dict[int, int]):
    """passing_at[after] = how many of the three seeds pass there; the rest fail."""
    out = []
    for a in early.AFTER:
        n = passing_at.get(a, 0)
        for i, seed in enumerate(early.SEEDS):
            ok = i < n
            out.append(
                {
                    "seed": seed,
                    "after": a,
                    "coverage": 0.9 if ok else 0.9,
                    "median_learnability": 0.2 if ok else 0.05,
                }
            )
    return out


def test_two_of_three_seeds_in_band_early_is_early_in_band():
    assert early.early_verdict(recs({10: 2})) == "EARLY IN BAND"
    assert early.early_verdict(recs({0: 3})) == "EARLY IN BAND"


def test_one_seed_or_only_late_is_not():
    assert early.early_verdict(recs({0: 1, 10: 1, 25: 1})) == "NEVER IN BAND EARLY"
    assert early.early_verdict(recs({50: 3, 100: 3})) == "NEVER IN BAND EARLY", "50 and 100 are not early"


def test_undefined_learnability_never_counts_as_in_band():
    r = recs({})
    for x in r:
        if x["after"] == 0:
            x["median_learnability"] = math.nan
    assert early.early_verdict(r) == "NEVER IN BAND EARLY"


def test_the_answer_is_a_if_any_setting_is_early_in_band():
    assert early.answer({"x": "NEVER IN BAND EARLY", "y": "EARLY IN BAND"}) == "A"
    assert early.answer({"x": "NEVER IN BAND EARLY", "y": "NEVER IN BAND EARLY"}) == "B"


def test_the_settings_and_points_are_the_registered_ones():
    tags = [early.cal.tag(s) for s in early.SETTINGS]
    assert tags == [
        "h_regen0.25__p_wander0__damage34",
        "h_regen0.1__p_wander0.5__damage34",
        "h_regen0__p_wander0__damage34",
        "h_regen0__p_wander1__damage50",
    ]
    assert early.AFTER == (0, 10, 25, 50, 100) and early.EARLY == (0, 10, 25)
    assert early.SEEDS == (100, 101, 102)
