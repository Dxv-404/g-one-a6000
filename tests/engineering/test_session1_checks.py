"""The A6000 session 1 checks (pre-run checklist 4.1 to 4.3): every verdict on hand-built inputs, boundaries included,
and laptop smoke runs of the drivers. Nothing here runs on the A6000."""

from __future__ import annotations

import datetime
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MEASURE = ROOT / "scripts" / "measure"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, MEASURE / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


s1 = _load("session1_common", "session1_common.py")
cache = _load("session1_cache_t", "session1_cache.py")
det = _load("session1_determinism_t", "session1_determinism.py")
conc = _load("session1_concurrency_t", "session1_concurrency.py")

GIB = 1024.0


def bg(util, mem_mib, available=True):
    return {"available": available, "util_mean": util, "mem_used_mib_mean": mem_mib}


# ------------------------------------------------------------------------------------- nvidia-smi parsing ----
def test_nvidia_smi_line_is_parsed_and_unreadable_fields_are_skipped():
    assert s1.parse_nvidia_smi("55, 9395, 46068, 67\n") == {
        "util": 55.0,
        "mem_used_mib": 9395.0,
        "mem_total_mib": 46068.0,
        "temp_c": 67.0,
    }
    assert s1.parse_nvidia_smi("[N/A], 9395, 46068, 67\n") is None
    assert s1.parse_nvidia_smi("garbage\n0, 1, 2, 3\n")["mem_total_mib"] == 2.0
    assert s1.summarise_samples([]) == {"available": False, "n": 0}
    s = s1.summarise_samples(
        [s1.parse_nvidia_smi("10, 100, 1000, 50"), s1.parse_nvidia_smi("30, 300, 1000, 60")]
    )
    assert (
        s["util_mean"] == 20
        and s["util_max"] == 30
        and s["mem_used_mib_max"] == 300
        and s["temp_c_max"] == 60
    )


# --------------------------------------------------------------------------------------- cleanliness ----
def test_a_background_reading_is_clean_up_to_and_including_each_margin():
    ref = bg(50.0, 9400.0)
    assert s1.is_clean(bg(60.0, 9400.0), ref)[0]  # exactly 10 points
    assert not s1.is_clean(bg(60.01, 9400.0), ref)[0]
    assert s1.is_clean(bg(50.0, 9400.0 + 1 * GIB), ref)[0]  # exactly 1 GiB
    assert not s1.is_clean(bg(50.0, 9400.0 + 1 * GIB + 1), ref)[0]
    assert not s1.is_clean(bg(50.0, 9400.0, available=False), ref)[0]  # unknown is not clean


def test_a_round_is_clean_only_if_the_background_before_AND_after_match():
    ref = bg(50.0, 9400.0)
    assert s1.round_clean(bg(52, 9400), bg(48, 9400), ref)[0]
    ok, why = s1.round_clean(bg(52, 9400), bg(0, 9400), ref)  # the job stopped during the round
    assert not ok and why.startswith("after:")
    ok, why = s1.round_clean(bg(0, 9400), bg(50, 9400), ref)
    assert not ok and why.startswith("before:")


# ------------------------------------------------------------------------------------ the cache census ----
def test_the_census_names_cache_entries_without_an_atime_file(tmp_path):
    for k in ("a", "b"):
        (tmp_path / f"jit_f-{k}-cache").write_bytes(b"x" * 10)
        (tmp_path / f"jit_f-{k}-atime").write_bytes(b"t")
    (tmp_path / "jit_f-orphan-cache").write_bytes(b"x" * 5)  # the entry that stops every later write
    (tmp_path / "jit_f-lonely-atime").write_bytes(b"t")
    (tmp_path / ".lockfile").write_bytes(b"")
    (tmp_path / s1.AUTOTUNE_DIR).mkdir()
    (tmp_path / s1.AUTOTUNE_DIR / "fusion").write_bytes(b"y")
    old = datetime.datetime(2026, 10, 15).timestamp()
    os.utime(tmp_path / "jit_f-a-cache", (old, old))
    c = s1.cache_census(tmp_path, since=datetime.datetime(2026, 11, 1), max_size=20)
    assert c["cache_entries"] == 3 and c["atime_files"] == 3
    assert c["cache_without_atime"] == 1 and c["cache_without_atime_names"] == ["jit_f-orphan"]
    assert c["atime_without_cache"] == 1 and c["lockfile_present"] and c["autotune_files"] == 1
    assert c["cache_bytes"] == 25 and c["fraction_of_max"] == 1.25
    assert c["entries_written_since"] == 2  # the old one is not counted
    assert s1.cache_census(tmp_path / "absent") == {"dir": str(tmp_path / "absent"), "exists": False}


# --------------------------------------------------------------------------------------- 4.1 verdict ----
def _probe(exit_=0, events=None, cfg=None, tail=""):
    return {"exit": exit_, "events": events or {}, "config": cfg or {}, "stderr_tail": tail}


def test_the_cache_verdict_reads_each_outcome_in_plain_words():
    b, a_more, a_same = {"cache_entries": 5}, {"cache_entries": 6}, {"cache_entries": 5}
    miss, hit = _probe(events={cache.MISS: 1}), _probe(events={cache.HIT: 1})
    assert cache.verdict(b, a_more, miss, hit, _probe())["status"].startswith("WORKS")
    assert cache.verdict(b, a_more, miss, _probe(), _probe())["status"].startswith("PARTLY")
    v = cache.verdict(
        {**b, "cache_without_atime": 3}, a_same, miss, _probe(), _probe(exit_=1, tail="FileNotFoundError")
    )
    assert v["status"].startswith("BROKEN") and "FileNotFoundError" in v["strict_error_tail"]
    assert any("no -atime" in c for c in v["likely_causes"])
    gated = _probe(events={cache.MISS: 1}, cfg={"jax_persistent_cache_min_compile_time_secs": 1.0})
    v = cache.verdict(b, a_same, gated, _probe(), _probe())
    assert v["status"].startswith("NOT WRITING") and any("min_compile_time" in c for c in v["likely_causes"])
    assert cache.verdict(b, a_same, _probe(exit_=1), _probe(), _probe())["status"].startswith(
        "the probe itself"
    )


def test_a_cache_the_sweep_never_turns_on_is_named_as_the_cause():
    """The sweep-order probe: a cache that works for a process importing src first, and is OFF in a process started
    like the sweep, must lead the verdict, because that is the state the long run is in."""
    b, a_more = {"cache_entries": 5}, {"cache_entries": 6}
    miss, hit = _probe(events={cache.MISS: 1}), _probe(events={cache.HIT: 1})
    off = {
        "exit": 0,
        "persistent_cache_enabled": False,
        "cache_dir_as_jax_holds_it": None,
        "env_cache_dir": "/c",
    }
    v = cache.verdict(b, a_more, miss, hit, _probe(), off)
    assert v["status"].startswith("CACHE WORKS, BUT THE SWEEP NEVER USES IT")
    assert v["production_cache"]["enabled"] is False and v["likely_causes"][0].startswith(
        "the long run's own"
    )
    on = {**off, "persistent_cache_enabled": True}
    v = cache.verdict(b, a_more, miss, hit, _probe(), on)
    assert (
        v["status"].startswith("WORKS")
        and v["production_cache"]["enabled"] is True
        and not v["likely_causes"]
    )
    assert cache.production_cache({"exit": 1})["known"] is False


# --------------------------------------------------------------------------------------- 4.2 verdict ----
def test_the_determinism_settings_keep_drift_diags_prefixes():
    rows = det.settings(10, 3)
    assert len(rows) == 13 and rows[0] == ("noauto_01", True, det.FLAG) and rows[9][0] == "noauto_10"
    assert [r[0] for r in rows[10:]] == ["fresh_1", "fresh_2", "fresh_3"] and all(
        r[2] == "" for r in rows[10:]
    )
    assert all(r[1] for r in rows)  # every process compiles into its own empty cache
    with pytest.raises(AssertionError):
        det.settings(1, 3)


def _summary(n, m, flagged_agree=True, contrast_agree=False, failed=()):
    status = {f"noauto_{i:02d}": {"exit": 1 if f"noauto_{i:02d}" in failed else 0} for i in range(1, n + 1)}
    status |= {f"fresh_{i}": {"exit": 0} for i in range(1, m + 1)}
    cell = {"equals_stored": True, "equals_replay": True}
    row = {k: cell for k in status if status[k]["exit"] == 0}
    row |= {
        "drifted_in_replay": True,
        "noauto_processes_agree": flagged_agree,
        "fresh_processes_agree": contrast_agree,
    }
    return {"status": status, "results": {"job seed 0": row, "job seed 1": dict(row)}}


def test_the_determinism_verdict_passes_only_when_every_planned_process_ran_and_agreed():
    v = det.verdict(_summary(10, 3), 10, 3)
    assert (
        v["status"].startswith("PASSES")
        and v["flag_pins_program"] is True
        and "is what makes" in v["contrast"]
    )
    v = det.verdict(_summary(10, 3, flagged_agree=False), 10, 3)
    assert v["status"].startswith("FAILS") and v["flag_pins_program"] is False
    v = det.verdict(_summary(10, 3, failed=("noauto_03",)), 10, 3)  # a failure is not counted as agreement
    assert v["status"].startswith("INCOMPLETE") and v["flagged_ran"] == 9 and v["flag_pins_program"] is None
    v = det.verdict(_summary(10, 3, contrast_agree=True), 10, 3)
    assert "does not show the flag is necessary" in v["contrast"]
    assert det.verdict({"status": {}, "results": {}}, 10, 3)["status"].startswith("NOT DECIDED")


# --------------------------------------------------------------------------------- 4.3 pure verdicts ----
def _state(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "final_pop": rng.normal(size=(5, 7)).astype(np.float32),
        "fit_mean": np.arange(4, dtype=np.float32),
        "fit_max": np.arange(4, dtype=np.float32) + 1,
    }


def test_bit_identity_covers_every_kept_array_and_counts_differing_genomes():
    ref = _state()
    assert conc.compare_state(ref, {k: v.copy() for k, v in ref.items()})["identical"]
    one = {k: v.copy() for k, v in ref.items()}
    one["final_pop"][3, 2] = np.nextafter(one["final_pop"][3, 2], np.float32(1e9))  # one ulp, one genome
    r = conc.compare_state(ref, one)
    assert not r["identical"] and r["genomes_differing"] == 1
    fmax = {k: v.copy() for k, v in ref.items()}
    fmax["fit_max"][2] += 1
    r = conc.compare_state(ref, fmax)
    assert not r["identical"] and r["final_pop"] and r["fit_mean"] and not r["fit_max"]


def test_rates_exclude_the_compile_and_the_window_starts_at_the_release():
    walls = [10.0, 0.5, 0.5, 0.5, 0.5]
    assert conc.loop_rate(walls) == 2.0 and conc.compile_seconds(walls) == 9.5
    assert conc.aggregate_throughput(4, 5, 100.0, [102.0, 104.0, 103.0]) == 4 * 4 / 4.0


def _h(n, tp, clean=True, failed=False, identical=True, peak=20000.0, bgmem=9400.0, drop=None):
    return {
        "n": n,
        "throughput": tp,
        "clean": clean,
        "failed": failed,
        "identical": identical,
        "peak_gpu_mib": peak,
        "background_gpu_mib": bgmem,
        "host_mem_drop_gib": drop,
    }


TOTAL = 46068.0


def test_the_stop_rule_continues_at_exactly_five_percent_and_stops_below_it():
    assert conc.next_step([_h(1, 1.0), _h(2, 1.05)], 3, 16, TOTAL, 60.0)[0]
    stop, why = conc.next_step([_h(1, 1.0), _h(2, 1.0499)], 3, 16, TOTAL, 60.0)
    assert not stop and "under" in why


def test_the_stop_rule_stops_on_failure_non_identity_cap_and_memory():
    """Every case asserts its REASON, not just that it stopped: several stop causes can fire on one round, and a
    case that stops for the wrong one would pass while the rule it names is broken."""
    stop, why = conc.next_step([_h(1, 1.0), _h(2, 2.0, failed=True)], 3, 16, TOTAL, 60.0)
    assert not stop and "failed" in why
    stop, why = conc.next_step([_h(1, 1.0), _h(2, 2.0, identical=False)], 3, 16, TOTAL, 60.0)
    assert not stop and "not bit-identical" in why
    stop, why = conc.next_step([_h(1, 1.0), _h(16, 9.0)], None, 16, TOTAL, 60.0)
    assert not stop and "cap" in why
    stop, why = conc.next_step([_h(1, 1.0), _h(12, 9.0)], 16, 12, TOTAL, 60.0)
    assert not stop and "cap" in why
    # ours 2 GiB a process at N 4; at N 16 projected 9400 MiB + 32 GiB leaves under 8 GiB of 46,068 MiB
    stop, why = conc.next_step([_h(1, 1.0), _h(4, 3.0, peak=9400 + 4 * 2048)], 16, 16, TOTAL, 60.0)
    assert not stop and "projected GPU memory" in why
    # host: 3 GiB a process at N 4, so N 16 needs 48 of 15 available. The GPU footprint is kept tiny so the GPU
    # rule cannot be what stops it
    stop, why = conc.next_step([_h(1, 1.0), _h(4, 3.0, peak=9400 + 400, drop=12.0)], 16, 16, TOTAL, 15.0)
    assert not stop and "projected host memory" in why
    # and the same round with ample host memory continues, so the host rule is what stopped it above
    assert conc.next_step([_h(1, 1.0), _h(4, 3.0, peak=9400 + 400, drop=12.0)], 16, 16, TOTAL, 500.0)[0]


def test_a_round_that_is_not_clean_cannot_stop_the_curve_and_gain_is_against_the_last_clean_round():
    # N 3 is not clean and slower; it neither stops the curve nor becomes the comparison point for N 4
    hist = [_h(1, 1.0), _h(2, 1.8), _h(3, 0.5, clean=False)]
    assert conc.next_step(hist, 4, 16, TOTAL, 60.0)[0]
    hist.append(_h(4, 1.85))  # +2.8 percent over N 2, the last CLEAN round
    stop, why = conc.next_step(hist, 6, 16, TOTAL, 60.0)
    assert not stop and "over N = 2" in why


def test_the_recommendation_is_one_step_below_the_fastest_eligible_round():
    hist = [_h(1, 1.0), _h(2, 1.9), _h(3, 2.6), _h(4, 3.0), _h(6, 2.9)]
    r = conc.recommend(hist, TOTAL, plan_generations=86400 * 3)
    assert r["fastest_n"] == 4 and r["n"] == 3 and r["projected_days"] == pytest.approx(3 / 2.6)
    assert "CONDITIONAL" in r["text"]
    # N 4 now leaves under 8 GiB free, so it is not eligible: the fastest eligible is N 6, and the step down must
    # SKIP the ineligible N 4 and land on N 3
    hist[3] = _h(4, 3.0, peak=TOTAL - 8 * 1024 + 1)
    r = conc.recommend(hist, TOTAL)
    assert r["fastest_n"] == 6 and r["n"] == 3
    # exactly 8 GiB free is eligible
    assert conc.eligible(_h(4, 3.0, peak=TOTAL - 8 * 1024), TOTAL)[0]
    # an unclean or non-identical round is never eligible
    assert not conc.eligible(_h(2, 9.0, clean=False), TOTAL)[0]
    assert not conc.eligible(_h(2, 9.0, identical=False), TOTAL)[0]
    # only N 1 eligible: recommend N 1, not a step below it, and do not claim a step down happened
    r = conc.recommend([_h(1, 1.0), _h(2, 5.0, clean=False)], TOTAL)
    assert r["n"] == 1 and "no lower N was eligible" in r["text"] and "one step lower" not in r["text"]


def test_with_no_eligible_round_the_recommendation_says_so_plainly():
    r = conc.recommend([_h(1, 1.0, clean=False), _h(2, 2.0, clean=False)], TOTAL)
    assert r["n"] is None and "no clean comparison was possible" in r["text"]


def test_the_plan_size_is_the_ruled_one():
    assert conc.PLAN_CONDITIONS * conc.PLAN_SEEDS * conc.PLAN_GENERATIONS == 106 * 20 * 1500
    assert conc.SEED_BASE >= 200 and conc.SEQUENCE[0] == 2 and conc.N_CAP == 16
    assert conc.GPU_HEADROOM_GIB == 8.0 and conc.GAIN_FLOOR == 0.05


# -------------------------------------------------------------------------------------------- smoke ----
def test_the_determinism_plan_reads_its_targets_from_a_replay_folder():
    replay = ROOT / "runs" / "replay48" / "replay"
    if not replay.is_dir():
        pytest.skip("no local replay folder")
    out = subprocess.run(
        [
            sys.executable,
            str(MEASURE / "session1_determinism.py"),
            "absent-sweep",
            str(replay),
            "/tmp/unused",
            "--plan-only",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    assert "13 fresh processes (10 with '--xla_gpu_autotune_level=0', 3 without)" in out.stdout


@pytest.mark.slow
def test_the_concurrency_driver_runs_end_to_end_at_smoke_size(tmp_path):
    """Laptop only, 2 generations, N up to 2. Checks the output schema and that both workers of the N 2 round
    reproduce their solo references, which is what the bit-identity verdict reads on the A6000."""
    out = tmp_path / "smoke-concurrency"
    r = subprocess.run(
        [sys.executable, str(MEASURE / "session1_concurrency.py"), str(out), "--smoke"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=1800,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    s = json.loads((out / "summary.json").read_text())
    assert [h["n"] for h in s["history"]] == [1, 2]
    assert s["history"][1]["identical"] is True and s["history"][1]["failed"] is False
    assert "recommendation" in s and "BESIDE" in s["foreign_workload_note"]
    rounds = json.loads((out / "rounds.json").read_text())["rounds"]
    assert set(rounds[0]["identity"]) == {"200", "201"}
    assert rounds[0]["background_before"]["available"] and rounds[0]["background_after"]["available"]
