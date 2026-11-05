"""The drift diagnostic's driver logic, Q-055. The GPU comparison itself runs only on the A6000."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("drift_diag", ROOT / "scripts" / "measure" / "drift_diag.py")
dd = importlib.util.module_from_spec(_spec)
sys.modules["drift_diag"] = dd
_spec.loader.exec_module(dd)


def test_the_worker_environment_drops_every_inherited_jax_and_xla_variable(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--inherited")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    monkeypatch.setenv("HOME_PROBE", "kept")
    env = dd.clean_env(None, "")
    assert not any(k.startswith(("JAX_", "XLA_")) for k in env)
    assert "PYTHONPATH" not in env and env["HOME_PROBE"] == "kept"
    env = dd.clean_env("/tmp/c", "--xla_gpu_autotune_level=0")
    assert env["XLA_FLAGS"] == "--xla_gpu_autotune_level=0" and env["JAX_COMPILATION_CACHE_DIR"] == "/tmp/c"
    assert "JAX_ENABLE_X64" not in env


def test_targets_come_from_the_replay_record_with_the_named_seed_first(tmp_path):
    for job, _, _ in dd.PICK:
        rec = [
            {"seed": s, "exact": s not in (3, 9, 11, 13, 14, 0 if "baseline" in job else 99)}
            for s in range(20)
        ]
        np.savez(tmp_path / f"{job}.npz", reproduction=np.array(json.dumps(rec)))
    t = dd.choose_targets(tmp_path)
    by_job = {}
    for job, s, drifted in t:
        by_job.setdefault(job, []).append((s, drifted))
    assert by_job["own__treatment__r_pred__3"] == [(14, True), (0, False)]
    assert by_job["own__null__baseline__None"] == [(0, True), (1, False)]
    assert by_job["own__third__obstacle_density__0.0"][0] == (13, True)
    assert all(d == (s in (3, 9, 11, 13, 14) or (s == 0 and "baseline" in j)) for j, s, d in t)


def test_every_job_it_names_is_in_the_replay_plan():
    spec = importlib.util.spec_from_file_location("rf_plan", ROOT / "scripts" / "measure" / "replay_final.py")
    rf = importlib.util.module_from_spec(spec)
    sys.modules["rf_plan"] = rf
    spec.loader.exec_module(rf)
    names = {j.name for j in rf.plan()}
    assert {j for j, _, _ in dd.PICK} <= names
    assert set(dd.PREFER) <= {j for j, _, _ in dd.PICK}
