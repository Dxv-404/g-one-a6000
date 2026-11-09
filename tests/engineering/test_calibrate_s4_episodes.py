"""Q-061's episode-count rule and its re-measurement driver: each branch of the rule, the margin arithmetic, the
directory and episode guards, and blindness to behaviour like the calibration it extends."""

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from src.measure import learnability

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "calibrate_s4_episodes", ROOT / "scripts" / "measure" / "calibrate_s4_episodes.py"
)
ce = importlib.util.module_from_spec(_spec)
sys.modules["calibrate_s4_episodes"] = ce
_spec.loader.exec_module(ce)


def by(passing48, chosen48, m48, passing64=None, chosen64=None, m64=None):
    passing64 = passing48 if passing64 is None else passing64
    chosen64 = chosen48 if chosen64 is None else chosen64
    m64 = m48 if m64 is None else m64
    return {
        48: {"passing": passing48, "chosen": chosen48, "margin": m48},
        64: {"passing": passing64, "chosen": chosen64, "margin": m64},
    }


# ------------------------------------------------------------------------------------- the rule ----
def test_the_rule_reads_the_stated_constants():
    assert ce.CANDIDATES == (48, 64) and ce.MARGIN_TARGET == 2.0


def test_a_different_choice_at_64_means_48_was_not_stable():
    assert ce.pick_episodes(by(["a", "b"], "a", {"a": 3.0, "b": 3.0}, chosen64="b")) == 64


def test_a_different_passing_set_at_64_means_48_was_not_stable():
    assert ce.pick_episodes(by(["a"], "a", {"a": 3.0, "b": 1.0}, passing64=["a", "b"])) == 64


def test_the_same_outcome_and_enough_margin_at_48_picks_48():
    assert ce.pick_episodes(by(["a"], "a", {"a": 2.0}, m64={"a": 2.7})) == 48


def test_the_same_outcome_and_margin_only_at_64_picks_64():
    assert ce.pick_episodes(by(["a"], "a", {"a": 1.65}, m64={"a": 2.2})) == 64


def test_the_same_outcome_and_margin_at_neither_picks_the_larger():
    assert ce.pick_episodes(by(["a"], "a", {"a": 1.2}, m64={"a": 1.6})) == 64


def test_nothing_passing_at_either_count_leaves_nothing_to_choose():
    assert ce.pick_episodes(by([], None, {"a": 3.0})) is None


# ----------------------------------------------------------------------------------- the margin ----
def test_the_margin_is_the_worst_median_genome_over_n_eng_min():
    """Constructed so the median genome's decided count is known per (seed, after): 60 and 45, so 1.5 at 30."""
    f = {n: i for i, n in enumerate(learnability.ENG_FIELDS)}

    def eng(decided_per_genome):
        e = np.zeros((len(decided_per_genome), 1, len(learnability.ENG_FIELDS)), np.int32)
        e[:, 0, f["escaped"]] = np.asarray(decided_per_genome) // 2
        e[:, 0, f["captured"]] = np.asarray(decided_per_genome) - np.asarray(decided_per_genome) // 2
        e[:, 0, f["censored"]] = 99  # excluded from decided, so it must not move the margin
        return e

    eng_by = {(100, 100): eng([10, 60, 90]), (100, 300): eng([0, 45, 200])}
    assert ce.worst_margin(eng_by, n_eng_min=30) == pytest.approx(1.5)


# ----------------------------------------------------------------------------------- the guards ----
def test_it_refuses_the_stages_own_directory(tmp_path):
    with pytest.raises(SystemExit, match="directory named for it"):
        ce.main([str(tmp_path / "calibration-s4"), "--episodes", "64"])


def test_it_refuses_a_count_that_does_not_reach_both_candidates(tmp_path):
    with pytest.raises(SystemExit, match="must reach 64"):
        ce.main([str(tmp_path / "calibration-s4-episodes"), "--episodes", "48"])


def test_it_loads_no_behaviour_measure():
    """Same probe as the calibration's: what the import actually loads, so a transitive import is caught."""
    probe = (
        "import importlib.util, sys, pathlib\n"
        f"p = pathlib.Path({str(ROOT / 'scripts' / 'measure' / 'calibrate_s4_episodes.py')!r})\n"
        "s = importlib.util.spec_from_file_location('ce_probe', p)\n"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        "print(sorted(n for n in sys.modules if n.startswith('src.measure')))\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True)
    loaded = out.stdout.strip().splitlines()[-1]
    assert "src.measure.learnability" in loaded, loaded
    for forbidden in ("descriptors", "classifier", "discovery"):
        assert f"src.measure.{forbidden}" not in loaded, loaded
