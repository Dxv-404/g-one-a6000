"""The sweep's progress reporting and health checks, D-062.

**Two different claims, tested two different ways.**

The first is that reporting CANNOT change the result. It is tested on the real `one_seed` on the GPU: the
same seed run twice, once with the monitor at full verbosity listening to every JAX trace and once with no
monitor, must agree bitwise on every array the checkpoint stores. That is an identity, so it has no tolerance
to choose. The same run asserts the RECOMPILE check does not fire on the real loop, because a check that
fires on every healthy run is one people learn to ignore.

The second is that each check fires when it should and stays quiet when it should. Those are tested with
stand-in readings, because the conditions they guard (a leak, a full disk, a hung GPU) cannot be produced
on demand. Every warning test has its quiet twin, so a check that always fires fails as surely as one that
never does.
"""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import time

import jax
import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "sweep_mod_progress", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "measure" / "sweep.py"
)
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)
sm = sweep.sm


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _monitor(tmp_path, *, plan=None, level="normal", stream=None, **kw):
    plan = plan if plan is not None else [("treatment", "treatment__baseline__None", 2)]
    defaults = dict(
        gpu=lambda: {
            "available": True,
            "name": "test",
            "driver": "0",
            "temp_c": 50.0,
            "sm_mhz": 1500.0,
            "power_w": 100.0,
            "mem_used_mb": 1.0,
            "mem_total_mb": 2.0,
            "util_pct": 50.0,
            "reasons": 0,
            "others": [],
        },
        rss=lambda: 1000.0,
        host_gb=lambda: 50.0,
        disk_gb=lambda _p: 500.0,
        device_mb=lambda: 100.0,
    )
    defaults.update(kw)
    return sm.Monitor(level, tmp_path, plan, 2, 30, 25, stream=stream or io.StringIO(), **defaults)


def _codes(mon) -> list[str]:
    return [w.split()[2] for w in mon.recent_warnings]


def _steady_seed(mon, seed=0, seconds=0.5, n=None, fit=None):
    mon.seed_start(seed)
    for g in range(n if n is not None else mon.n_generations):
        mon.generation(g, seconds, fit if fit is not None else np.array([0.0, 0.1, 0.2]))
    mon.loop_end()


# ---------------------------------------------------------------------------------- the interface --
def test_progress_levels_and_the_default_are_as_documented():
    assert sm.LEVELS == ("quiet", "normal", "verbose")
    assert sweep.parse_args(["out"]).progress == "normal"
    with pytest.raises(SystemExit):
        sweep.parse_args(["out", "--progress", "chatty"])


def test_exit_codes_tell_a_restartable_crash_from_a_stop_a_person_must_handle():
    assert sm.exit_code_for(RuntimeError("a CUDA error")) == sm.EXIT_CRASH
    assert sm.exit_code_for(SystemExit("SCHEMA MISMATCH")) == sm.EXIT_FATAL
    assert sm.exit_code_for(AssertionError("layout differs")) == sm.EXIT_FATAL
    assert sm.exit_code_for(KeyboardInterrupt()) == sm.EXIT_INTERRUPT
    assert sm.exit_code_for(SystemExit(0)) == sm.EXIT_DONE
    assert sm.exit_code_for(sm.DiskStop("full")) == sm.EXIT_DISK
    assert sm.exit_code_for(sm.MemoryStop("low")) == sm.EXIT_MEMORY
    assert sm.exit_code_for(sm.FatalStop("nan")) == sm.EXIT_FATAL
    assert sm.exit_code_for(MemoryError()) == sm.EXIT_MEMORY
    # The wrapper restarts exactly the restartable codes, so they must be distinct from the stopping ones.
    restartable = {sm.EXIT_CRASH, sm.EXIT_HANG, sm.EXIT_MEMORY}
    stopping = {sm.EXIT_DONE, sm.EXIT_FATAL, sm.EXIT_DISK, sm.EXIT_INTERRUPT}
    assert not restartable & stopping


# --------------------------------------------------------------------------------- the accounting --
def test_the_work_accounting_balances_at_every_generation(tmp_path):
    """done + remaining == total, throughout. An identity, so a miscounted resume fails it either way."""
    plan = [("treatment", "a", 2), ("null", "b", 1), ("treatment", "c", 0)]
    mon = _monitor(tmp_path, plan=plan)
    total = len(plan) * mon.n_seeds * mon.n_generations
    assert mon.total_gens == total
    assert mon.done_gens == total - (2 + 1) * mon.n_generations, "a finished condition counts as done"
    for arm, left in (("treatment", 2), ("null", 1)):
        mon.arm = arm
        for s in range(left):
            mon.seed_start(s)
            for g in range(mon.n_generations):
                mon.generation(g, 0.5, np.array([0.1]))
                assert mon.done_gens + sum(mon.remaining_by_arm.values()) == total
    assert mon.done_gens == total
    assert mon.eta_seconds() == 0


def test_the_estimate_uses_each_arms_own_measured_rate(tmp_path):
    mon = _monitor(tmp_path, plan=[("treatment", "a", 1), ("null", "b", 1)])
    mon.rate_by_arm = {"treatment": 2.0, "null": 1.0}
    expected = sum(left * mon.rate_by_arm[arm] for arm, left in mon.remaining_by_arm.items())
    assert mon.eta_seconds() == pytest.approx(expected)


# ------------------------------------------------------------------------------------ the output --
def test_the_live_line_reaches_a_terminal_and_never_the_log_file(tmp_path):
    term = _Tty()
    mon = _monitor(tmp_path, stream=term)
    mon.condition_start(1, "treatment", "treatment__baseline__None", 0)
    _steady_seed(mon)
    assert "\r" in term.getvalue(), "a real terminal gets the in-place line"
    log = (tmp_path / "sweep.log").read_text()
    assert "\r" not in log and "\x1b" not in log, "the log file must stay readable as text"


def test_a_pipe_gets_no_live_line(tmp_path):
    pipe = io.StringIO()
    mon = _monitor(tmp_path, stream=pipe)
    _steady_seed(mon)
    assert "\r" not in pipe.getvalue()


def test_each_level_prints_what_it_promises(tmp_path):
    counts = {}
    for level in sm.LEVELS:
        out = io.StringIO()
        mon = _monitor(tmp_path / level, level=level, stream=out)
        mon.seed_start(0)
        for g in range(mon.n_generations):
            row = np.array([0.1, 0.2, 0.7]) if g % mon.sample_every == 0 else None
            mon.generation(g, 0.5, np.array([0.1]), row)
        counts[level] = out.getvalue().count("    gen ")
    sampled = len([g for g in range(30) if g % 25 == 0])
    assert counts["quiet"] == 0
    assert counts["normal"] == sampled, "normal prints the sampled generations"
    assert counts["verbose"] == 30, "verbose prints every generation"


def test_the_status_file_is_valid_json_and_says_how_the_run_ended(tmp_path):
    mon = _monitor(tmp_path)
    mon.run_start(["header"])
    _steady_seed(mon)
    mon.run_end(sm.EXIT_DONE)
    d = json.loads((tmp_path / "status.json").read_text())
    assert d["state"] == "complete" and d["exit_code"] == sm.EXIT_DONE
    mon.run_end(sm.EXIT_DISK, "full")
    d = json.loads((tmp_path / "status.json").read_text())
    assert d["state"] == "stopped" and d["exit_code"] == sm.EXIT_DISK


def test_a_warning_reaches_the_terminal_and_warnings_log_whatever_the_level(tmp_path):
    out = io.StringIO()
    mon = _monitor(tmp_path, level="quiet", stream=out)
    mon.warn("TEST", "something out of place")
    assert "WARNING TEST" in out.getvalue()
    assert "WARNING TEST" in (tmp_path / "warnings.log").read_text()


# -------------------------------------------------------------------------------------- the checks --
def test_a_trace_mid_seed_is_named_and_traces_while_compiling_or_after_the_loop_are_not(tmp_path):
    mon = _monitor(tmp_path)
    mon.seed_start(0)
    mon.on_trace()  # the seed's own compilation
    for g in range(sm.WARMUP_GENERATIONS - 1):
        mon.generation(g, 0.5, np.array([0.1]))
    mon.on_trace()  # still warming up
    assert "RECOMPILE" not in _codes(mon)
    mon.generation(sm.WARMUP_GENERATIONS - 1, 0.5, np.array([0.1]))
    mon.on_trace()
    assert _codes(mon).count("RECOMPILE") == 1
    mon.loop_end()
    mon.on_trace()  # the null arm's naive_eval pass compiles a new program, legitimately
    assert _codes(mon).count("RECOMPILE") == 1


def test_a_generation_far_slower_than_its_seed_is_named_and_jitter_is_not(tmp_path):
    mon = _monitor(tmp_path)
    mon.seed_start(0)
    for g in range(10):
        mon.generation(g, 0.5, np.array([0.1]))
    assert "SLOW-GENERATION" not in _codes(mon)
    spike = max(sm.SPIKE_FACTOR * 0.5, sm.SPIKE_MIN_SECONDS) * 2
    mon.generation(10, spike, np.array([0.1]))
    assert "SLOW-GENERATION" in _codes(mon)


def test_a_seed_slower_than_the_first_of_its_condition_is_named(tmp_path):
    mon = _monitor(tmp_path)
    _steady_seed(mon, 0, seconds=0.5)
    mon.seed_end(0, 0.1, 0.2)
    _steady_seed(mon, 1, seconds=0.5)
    mon.seed_end(1, 0.1, 0.2)
    assert "SLOWDOWN" not in _codes(mon)
    _steady_seed(mon, 2, seconds=0.5 * sm.DRIFT_FACTOR * 1.5)
    mon.seed_end(2, 0.1, 0.2)
    assert "SLOWDOWN" in _codes(mon)


def test_host_memory_growing_within_a_seed_is_named_and_a_flat_reading_is_not(tmp_path):
    reading = {"mb": 1000.0}
    mon = _monitor(tmp_path, rss=lambda: reading["mb"])
    for s in range(2):
        _steady_seed(mon, s)
        mon.seed_end(s, 0.1, 0.2)
    assert "MEMORY-SEED" not in _codes(mon)
    _steady_seed(mon, 2)
    reading["mb"] += sm.RSS_SEED_GROWTH_MB * 2
    mon.seed_end(2, 0.1, 0.2)
    assert "MEMORY-SEED" in _codes(mon)


def test_device_memory_growth_across_seeds_is_named(tmp_path):
    reading = {"mb": 100.0}
    mon = _monitor(tmp_path, device_mb=lambda: reading["mb"])
    _steady_seed(mon, 0)
    mon.seed_end(0, 0.1, 0.2)
    reading["mb"] += sm.GPU_GROWTH_MB * 2
    _steady_seed(mon, 1)
    mon.seed_end(1, 0.1, 0.2)
    assert "GPU-MEMORY" in _codes(mon)


def test_host_memory_growth_across_conditions_is_named(tmp_path):
    reading = {"mb": 1000.0}
    mon = _monitor(tmp_path, rss=lambda: reading["mb"])
    mon.condition_end("a", 60.0, 0.1, 0.01)
    reading["mb"] += sm.RSS_RUN_GROWTH_MB * 2
    mon.condition_end("b", 60.0, 0.1, 0.01)
    assert "MEMORY-RUN" in _codes(mon)


def test_non_finite_fitness_stops_the_run_as_fatal(tmp_path):
    mon = _monitor(tmp_path)
    mon.seed_start(0)
    with pytest.raises(sm.FatalStop) as e:
        mon.generation(0, 0.5, np.array([0.1, np.nan]))
    assert sm.exit_code_for(e.value) == sm.EXIT_FATAL


def test_a_population_scoring_zero_for_long_is_named_but_does_not_stop(tmp_path):
    mon = _monitor(tmp_path)
    mon.n_generations = sm.COLLAPSE_GENERATIONS + 5
    _steady_seed(mon, fit=np.zeros(4))
    assert "ZERO-POPULATION" in _codes(mon)
    quiet = _monitor(tmp_path / "q")
    _steady_seed(quiet)
    assert "ZERO-POPULATION" not in _codes(quiet)


def test_a_full_disk_stops_cleanly_with_the_disk_code(tmp_path):
    mon = _monitor(tmp_path, disk_gb=lambda _p: sm.DISK_STOP_GB / 2)
    with pytest.raises(sm.DiskStop) as e:
        mon.condition_start(1, "treatment", "t", 0)
    assert sm.exit_code_for(e.value) == sm.EXIT_DISK
    low = _monitor(tmp_path / "w", disk_gb=lambda _p: (sm.DISK_STOP_GB + sm.DISK_WARN_GB) / 2)
    low.condition_start(1, "treatment", "t", 0)
    assert "DISK-LOW" in _codes(low)


def test_exhausted_host_memory_stops_with_the_restartable_memory_code(tmp_path):
    mon = _monitor(tmp_path, host_gb=lambda: sm.HOST_AVAILABLE_STOP_GB / 2)
    with pytest.raises(sm.MemoryStop) as e:
        mon.condition_start(1, "treatment", "t", 0)
    assert sm.exit_code_for(e.value) == sm.EXIT_MEMORY


def test_the_gpu_checks_name_heat_throttling_sharing_and_an_unreadable_gpu(tmp_path):
    hot = {
        "available": True,
        "name": "x",
        "driver": "0",
        "temp_c": sm.TEMP_WARN_C + 1,
        "sm_mhz": 1.0,
        "power_w": 1.0,
        "mem_used_mb": 1.0,
        "mem_total_mb": 1.0,
        "util_pct": 1.0,
        "reasons": 0x40,
        "others": [["4242", "python"]],
    }
    mon = _monitor(tmp_path, gpu=lambda: hot)
    mon.condition_start(1, "treatment", "t", 0)
    assert {"GPU-HOT", "GPU-THROTTLE", "GPU-SHARED"} <= set(_codes(mon))
    capped = dict(hot, temp_c=50.0, reasons=0x4, others=[])  # a power cap under load is normal
    calm = _monitor(tmp_path / "c", gpu=lambda: capped)
    calm.condition_start(1, "treatment", "t", 0)
    assert not {"GPU-HOT", "GPU-THROTTLE", "GPU-SHARED"} & set(_codes(calm))
    blind = _monitor(tmp_path / "b", gpu=lambda: {"available": False})
    blind.condition_start(1, "treatment", "t", 0)
    assert "GPU-UNREADABLE" in _codes(blind)


def test_the_watchdog_exits_with_the_hang_code_when_nothing_moves(tmp_path):
    mon = _monitor(tmp_path)
    mon.state = "running"
    mon.last_progress = mon.clock() - 10.0
    exited = []
    sm.start_watchdog(mon, warn_after=1.0, exit_after=5.0, poll=0.05, exit_fn=exited.append)
    deadline = time.time() + 5.0
    while not exited and time.time() < deadline:
        time.sleep(0.05)
    assert exited == [sm.EXIT_HANG]
    assert json.loads((tmp_path / "status.json").read_text())["exit_code"] == sm.EXIT_HANG


# ----------------------------------------------------------------------------- the runner's guards --
def test_smoke_output_and_sweep_output_never_share_a_directory(tmp_path):
    runs_sweep = tmp_path / "runs" / "sweep"
    with pytest.raises(sm.FatalStop):
        sweep.guard_smoke(runs_sweep, smoke=True)
    real = tmp_path / "real"
    real.mkdir()
    (real / "treatment__baseline__None.npz").write_bytes(b"")
    with pytest.raises(sm.FatalStop):
        sweep.guard_smoke(real, smoke=True)
    scratch = tmp_path / "scratch"
    sweep.guard_smoke(scratch, smoke=True)
    with pytest.raises(sm.FatalStop):
        sweep.guard_smoke(scratch, smoke=False)
    sweep.guard_smoke(tmp_path / "fresh", smoke=False)  # a clean directory is fine for the sweep


def test_a_second_sweep_in_the_same_directory_is_refused(tmp_path):
    first = sm.acquire_run_lock(tmp_path)
    try:
        with pytest.raises(sm.FatalStop):
            sm.acquire_run_lock(tmp_path)
    finally:
        first.close()
    sm.acquire_run_lock(tmp_path).close()  # released when the holder goes


# ------------------------------------------------------------------------ the identity, on the GPU --
@pytest.mark.slow  # --durations: 86.8 s call on the RTX 4060, 2026-10-13. Two real seeds of 27 generations
def test_reporting_changes_no_bit_of_a_real_seed_and_the_healthy_loop_raises_no_recompile(tmp_path):
    """The null arm, so the naive_eval pass after the loop is exercised: it compiles a new program, which the
    monitor must not mistake for DEV-029. 27 generations reach a sampled generation past the first (25) and
    the final one, so every reporting path runs."""
    n_gen = 27
    p = sweep.params_for("null", "baseline", None)
    eval_p = sweep.BY_NAME["naive_eval"].params

    plain = sweep.one_seed(p, 0, eval_params=eval_p, n_generations=n_gen)

    mon = _monitor(tmp_path, level="verbose", stream=io.StringIO(), plan=[("null", "t", 1)])
    jax.monitoring.register_event_duration_secs_listener(
        lambda event, duration, **kw: mon.on_trace() if event == sweep.TRACE_EVENT else None
    )
    mon.seed_start(0)
    before = mon.done_gens
    watched = sweep.one_seed(p, 0, eval_params=eval_p, n_generations=n_gen, progress=mon)

    assert set(plain) == set(watched)
    for k in plain:
        a, b = np.asarray(plain[k]), np.asarray(watched[k])
        assert a.shape == b.shape and a.dtype == b.dtype, k
        assert np.array_equal(a, b), f"reporting changed `{k}`"
    # The comparison is only worth something if the compared values vary.
    assert np.unique(plain["final_pop"]).size > 1 and plain["acc"].shape[0] == len([0, 25, n_gen - 1])
    assert mon.done_gens - before == n_gen, "the monitor saw every generation exactly once"
    assert "RECOMPILE" not in _codes(mon), "the real loop must not trip the recompile check"
