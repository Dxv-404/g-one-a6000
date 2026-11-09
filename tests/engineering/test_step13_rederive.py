"""Q-058: the thresholds re-derived in the frozen world, from a sweep folder, into a NEW file, written once."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "step13", ROOT / "scripts" / "measure" / "step13_thresholds.py"
)
s13 = importlib.util.module_from_spec(_spec)
sys.modules["step13"] = s13
_spec.loader.exec_module(s13)


def _null_folder(tmp_path, n_cond=29, n_seeds=20):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cols = list(s13.sweep.EXPECTED_SCHEMA)
    for i in range(n_cond):
        rows = rng.uniform(0.05, 1.0, size=(n_seeds, len(cols))).astype(np.float32)
        np.savez(
            tmp_path / f"null__axis{i}__x.npz",
            arm=np.array("null"),
            acc_fields=np.array(cols),
            naive_acc=rows,
        )
    return tmp_path


def test_the_rederivation_writes_a_new_file_once_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(s13, "N_SPLITS", 40)
    src = _null_folder(tmp_path / "phase-a-null")
    out = tmp_path / "step13_frozen_longrun.json"
    s13.main(["--source", str(src), "--out", str(out), "--run", "56", "--decision", "Q-058"])
    frozen = json.loads(out.read_text())
    assert frozen["run"] == 56 and frozen["decision"] == "Q-058" and frozen["source"] == str(src)
    assert np.isfinite(frozen["classifier_floor"]) and np.isfinite(frozen["separation_margin"])
    assert len(frozen["extract_index_sha256"]) == 64
    with pytest.raises(SystemExit, match="written once"):
        s13.main(["--source", str(src), "--out", str(out)])


def test_run_46s_frozen_file_is_refused_as_a_target():
    """The default target is run 46's file, which exists and is read by runs 47 and 52."""
    assert s13.OUT.exists()
    with pytest.raises(SystemExit, match="written once"):
        s13.main([])


def test_a_non_null_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(s13, "N_SPLITS", 40)
    src = _null_folder(tmp_path / "phase-a-null", n_cond=28)
    cols = list(s13.sweep.EXPECTED_SCHEMA)
    np.savez(
        src / "null__bad__x.npz",
        arm=np.array("treatment"),
        acc_fields=np.array(cols),
        naive_acc=np.ones((20, len(cols))),
    )
    with pytest.raises(PermissionError, match="null data only"):
        s13.main(["--source", str(src), "--out", str(tmp_path / "x.json")])
