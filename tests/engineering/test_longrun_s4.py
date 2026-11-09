"""The long run's S4 reader: D-083's rule at every in-run point, through the one learnability implementation."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("longrun_s4", ROOT / "scripts" / "measure" / "longrun_s4.py")
ls4 = importlib.util.module_from_spec(_spec)
sys.modules["longrun_s4"] = ls4
_spec.loader.exec_module(ls4)

F = 7  # engagement fields


def _ckpt(path, eng, after=(100, 300), arm="treatment"):
    np.savez(
        path,
        arm=np.array(arm),
        axis=np.array("baseline"),
        level=np.array("None"),
        seeds=np.array([20, 21], np.int32),
        s4_after=np.array(after, np.int32),
        s4_eng=eng,
    )


def _eng(decided, escaped, n_seeds=2, n_points=2, G=4, E=2):
    """Every genome with `decided` decided engagements of which `escaped` escaped, split over E episodes."""
    eng = np.zeros((n_seeds, n_points, G, E, F), np.int32)
    eng[..., 1] = escaped // E
    eng[..., 2] = (decided - escaped) // E
    eng[..., 0] = decided // E
    return eng


def test_in_band_with_full_coverage_passes_and_out_of_band_does_not(tmp_path):
    _ckpt(tmp_path / "good.npz", _eng(40, 20))  # p 0.5, learnability 0.25
    _ckpt(tmp_path / "bad.npz", _eng(40, 38))  # p 0.95
    good = ls4.condition_summary(tmp_path / "good.npz")
    bad = ls4.condition_summary(tmp_path / "bad.npz")
    assert all(v["passes_every_seed"] for v in good["points"].values())
    assert np.isclose(good["points"][100]["median_learnability_median"], 0.25)
    assert not any(v["passes_every_seed"] for v in bad["points"].values())


def test_too_few_engagements_is_undefined_not_zero(tmp_path):
    _ckpt(tmp_path / "thin.npz", _eng(10, 5))  # 10 decided, under n_eng_min
    s = ls4.condition_summary(tmp_path / "thin.npz")
    v = s["points"][100]
    assert (
        v["coverage_median"] == 0.0
        and np.isnan(v["median_learnability_median"])
        and not v["passes_every_seed"]
    )


def test_the_null_arm_has_no_in_run_s4_and_is_skipped(tmp_path):
    _ckpt(tmp_path / "null.npz", np.zeros((2, 0, 4, 2, F), np.int32), after=(), arm="null")
    assert ls4.condition_summary(tmp_path / "null.npz") is None


def test_the_reader_uses_the_shared_implementation_and_the_calibrations_rule():
    import importlib

    learnability = importlib.import_module("src.measure.learnability")
    assert ls4.learnability is learnability
    assert ls4.COVERAGE_FLOOR == 0.5 and ls4.FLOOR == 0.16 and ls4.N_ENG_MIN == 30


def test_main_reads_every_shard_and_skips_partials(tmp_path, capsys):
    for shard in ("shard0of2", "shard1of2"):
        (tmp_path / shard / "partial").mkdir(parents=True)
        _ckpt(tmp_path / shard / f"treatment__{shard}.npz", _eng(40, 20))
        _ckpt(tmp_path / shard / "partial" / "treatment__x.seed20.npz", _eng(40, 20))
    out = tmp_path / "s4.json"
    assert ls4.main([str(tmp_path), "--out", str(out)]) == 0
    import json

    conds = json.loads(out.read_text())["conditions"]
    assert sorted(conds) == ["treatment__shard0of2", "treatment__shard1of2"]


def test_an_in_band_median_over_too_few_measurable_genomes_fails(tmp_path):
    """The coverage half of D-083's rule, isolated: one genome of four is measurable and perfectly in band, so the
    median is 0.25 but coverage is 0.25. It must fail, which a rule reading only the median would pass."""
    eng = _eng(10, 5)  # every genome under n_eng_min
    eng[:, :, 0, :, 0] = 20  # genome 0: 40 decided over two episodes
    eng[:, :, 0, :, 1] = 10
    eng[:, :, 0, :, 2] = 10
    _ckpt(tmp_path / "sparse.npz", eng)
    v = ls4.condition_summary(tmp_path / "sparse.npz")["points"][100]
    assert np.isclose(v["coverage_median"], 0.25) and np.isclose(v["median_learnability_median"], 0.25)
    assert not v["passes_every_seed"]
