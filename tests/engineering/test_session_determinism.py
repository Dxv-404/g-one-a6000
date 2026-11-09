"""The final-program determinism check (D-090): every fresh process must equal the first, and a failure or a single
run concludes nothing. Uses session 1's comparison, so the two checks cannot mean different things by identical."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "session_determinism", ROOT / "scripts" / "measure" / "session_determinism.py"
)
det = importlib.util.module_from_spec(_spec)
sys.modules["session_determinism"] = det
_spec.loader.exec_module(det)


def state(v=0.0):
    return {"final_pop": np.full((4, 3), v, np.float32), "fit_mean": np.full(5, v), "fit_max": np.full(5, v)}


def ok_round():
    return {"failed": False}


def test_identical_runs_pass():
    ok, text = det.verdict([ok_round()] * 3, [state(), state(), state()])
    assert ok and "identical" in text


def test_one_differing_genome_fails_and_is_counted():
    s = state()
    s["final_pop"][2, 1] = 1.0
    ok, text = det.verdict([ok_round()] * 3, [state(), state(), s])
    assert not ok and "(2, 1)" in text


def test_a_failed_run_concludes_nothing():
    ok, text = det.verdict([ok_round(), {"failed": True}], [state(), None])
    assert not ok and "failed" in text


def test_a_single_run_compares_nothing():
    ok, text = det.verdict([ok_round()], [state()])
    assert not ok and "fewer than two" in text


def test_the_check_uses_session_ones_comparison_and_a_seed_used_nowhere_else():
    assert det.conc.compare_state.__module__ == det.conc.__name__
    assert det.SEED not in set(range(40)) | {100, 101, 102} | set(range(200, 216))
