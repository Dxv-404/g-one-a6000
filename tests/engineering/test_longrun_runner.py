"""The long run's runner (checklist 3.1 to 3.6): budget, seeds, the frozen world, in-run S4, per-genome final rows,
sharding, and the guards that refuse a long run in the wrong state."""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "measure"))
import sweep  # noqa: E402
import sweep_monitor as sm  # noqa: E402

from configs import frozen_world  # noqa: E402
from src.agent import network as net  # noqa: E402
from src.evo import genome  # noqa: E402
from src.measure import engagement  # noqa: E402

WORLD = {"h_regen": 0.1, "p_wander": 0.5, "damage": 50.0}  # a stage 2 grid point, moving all three fields


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(frozen_world, "WORLD", dict(WORLD))
    monkeypatch.setattr(frozen_world, "SOURCE", "test")
    return WORLD


# ------------------------------------------------------------------------------------- budget and seeds ----
def test_the_budget_and_seeds_are_the_ruled_ones():
    assert sweep.N_GENERATIONS == 1500  # D-073
    assert sweep.N_SEEDS == 20 and sweep.seed_list() == list(range(20, 40))  # D-070, D-084
    used_elsewhere = (
        set(range(20)) | {100, 101, 102} | set(range(200, 216))
    )  # Phase A, calibration, session 1
    assert not set(sweep.seed_list()) & used_elsewhere


def test_s4_is_measured_at_rising_points_ending_at_the_budget():
    pts = sweep.S4_AFTER
    assert list(pts) == sorted(set(pts)) and pts[-1] == sweep.N_GENERATIONS and pts[0] > 0
    assert sweep.S4_EPISODES == 64  # D-083's measurement budget since Q-061 (D-092)


# ------------------------------------------------------------------------------------- the frozen world ----
def test_every_arm_differs_from_the_frozen_world_only_in_its_own_field(frozen):
    scripted = sweep.arm_base("scripted")
    for k, v in WORLD.items():
        assert getattr(scripted, k) == v
    own = {"treatment": set(), "null": {"n_pred"}, "third": {"damage"}}
    for arm, field in own.items():
        p = sweep.params_for(arm, "baseline", None)
        moved = {k for k, v in dataclasses.asdict(p).items() if v != getattr(scripted, k)}
        assert moved == field, (arm, moved)
    assert sweep.params_for("third", "baseline", None).damage == 0.0, (
        "the harmless arm keeps its defining field"
    )
    assert sweep.arm_base("naive_eval") == scripted, "the null is evaluated in the frozen treatment world"


def test_without_a_frozen_world_every_arm_is_its_run_configuration(monkeypatch):
    monkeypatch.setattr(frozen_world, "WORLD", None)  # the state before D-092; the shipped world is tested in
    # test_calibrate_s4.py, against the run that chose it
    for name in ("scripted", "no_predator_evolve", "harmless_evolve", "naive_eval"):
        assert sweep.arm_base(name) == sweep.BY_NAME[name].params


def test_the_plan_keeps_its_shape_and_its_baseline_under_the_frozen_world(frozen):
    counts = {a: len(sweep.conditions(a)) for a in ("treatment", "null", "third")}
    assert counts == {"treatment": 53, "null": 29, "third": 27}
    wander = [lv for ax, lv in sweep.conditions("treatment") if ax == "p_wander"]
    assert WORLD["p_wander"] not in wander, "the frozen value is the baseline, not a level"
    assert 0.0 in wander, "the old default becomes a level"


def test_a_long_run_refuses_an_unfrozen_or_out_of_grid_world(monkeypatch):
    sweep.require_frozen_world(smoke=False)  # the shipped world (D-092) is accepted
    monkeypatch.setattr(frozen_world, "WORLD", None)
    with pytest.raises(sm.FatalStop, match="not frozen"):
        sweep.require_frozen_world(smoke=False)
    sweep.require_frozen_world(smoke=True)  # smoke tests may run in any world
    monkeypatch.setattr(frozen_world, "WORLD", {"p_wander": 0.3})
    with pytest.raises(sm.FatalStop, match="not one of the axis levels"):
        sweep.require_frozen_world(smoke=False)
    # cue_dependence became movable at Q-060, so a field no ruling allows stands in for it
    monkeypatch.setattr(frozen_world, "WORLD", {"obstacle_density": 0.08})
    with pytest.raises(ValueError, match="Q-060"):
        sweep.require_frozen_world(smoke=False)


def test_the_old_replay_refuses_once_the_world_is_frozen(frozen):
    spec = importlib.util.spec_from_file_location(
        "rf_guard", ROOT / "scripts" / "measure" / "replay_final.py"
    )
    rf = importlib.util.module_from_spec(spec)
    sys.modules["rf_guard"] = rf
    spec.loader.exec_module(rf)
    with pytest.raises(SystemExit, match="Phase A"):
        rf.main(["runs/sweep", "runs/replay", "--plan-only"])


# ------------------------------------------------------------------------------------------- sharding ----
def test_shards_partition_the_plan_exactly_for_every_n():
    plan = [(a, ax, lv) for a in ("treatment", "null", "third") for ax, lv in sweep.conditions(a)]
    for n in range(1, 17):
        parts = [sweep.shard_plan(plan, (i, n)) for i in range(n)]
        flat = [c for part in parts for c in part]
        assert sorted(map(str, flat)) == sorted(map(str, plan)), n
        assert max(map(len, parts)) - min(map(len, parts)) <= 1, n


def test_a_shard_is_parsed_strictly():
    assert sweep.parse_shard(None) is None and sweep.parse_shard("2/4") == (2, 4)
    for bad in ("4/4", "-1/2", "x", "1/0", "1/2/3"):
        with pytest.raises(sm.FatalStop):
            sweep.parse_shard(bad)


def test_shard_layouts_are_never_mixed(tmp_path):
    (tmp_path / "shard0of4").mkdir()
    sweep.guard_shards(tmp_path, (1, 4))  # same N: fine
    with pytest.raises(sm.FatalStop, match="different N"):
        sweep.guard_shards(tmp_path, (0, 3))
    with pytest.raises(sm.FatalStop, match="sharded output"):
        sweep.guard_shards(tmp_path, None)
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "c.npz").write_bytes(b"")
    with pytest.raises(sm.FatalStop, match="unsharded"):
        sweep.guard_shards(flat, (0, 2))
    assert (
        sweep.shard_dir(tmp_path, (1, 4)) == tmp_path / "shard1of4"
        and sweep.shard_dir(tmp_path, None) == tmp_path
    )


# ------------------------------------------------------------------------------------------ the guards ----
def test_a_written_condition_without_the_planned_s4_points_stops_the_run(tmp_path, monkeypatch):
    def one_seed(p, seed, eval_params=None):
        cols = len(sweep.EXPECTED_SCHEMA)
        return {
            "fit_mean": np.ones(sweep.N_GENERATIONS, np.float32),
            "fit_max": np.ones(sweep.N_GENERATIONS, np.float32),
            "gens": np.array([0], np.int32),
            "acc": np.zeros((1, cols), np.float32),
            "relax": 0,
            "final_pop": np.zeros((4, 3), np.float32),
            "final_rows": np.zeros((4, cols), np.float32),
            "s4_after": np.array([100], np.int32),  # stopped early: not the plan
            "s4_eng": np.zeros((1, 4, 2, 7), np.int32),
            "s4_prey": np.zeros((1, 4, 2, 6), np.int32),
        }

    monkeypatch.setattr(sweep, "one_seed", one_seed)
    monkeypatch.setattr(sweep, "N_SEEDS", 2)
    with pytest.raises(sm.FatalStop, match="not the planned"):
        sweep.run_condition("treatment", "baseline", None, tmp_path)


def test_a_partial_without_the_in_run_s4_is_refused(tmp_path):
    part = tmp_path / "partial"
    part.mkdir()
    p = part / "treatment__baseline__None.seed20.npz"
    np.savez(p, acc_fields=np.array(sweep.EXPECTED_SCHEMA), arm=np.array("treatment"), final_pop=np.zeros(1))
    with pytest.raises(SystemExit, match="partial seed without the in-run S4"):
        sweep.assert_checkpoint_schema(p)


# -------------------------------------------------------------------------------- the real thing, GPU ----
@pytest.mark.slow
def test_in_run_s4_is_the_calibrations_measurement_and_cannot_change_the_run(monkeypatch):
    """On the real `one_seed`, 3 generations: (1) the stored in-run S4 equals `engagement.measure` run afresh on
    the stored final population with the same keys, so the runner measures with the one implementation; (2) the
    population is bit-identical to a run with no S4 at all, so measuring cannot perturb evolution; (3) the stored
    final sample is exactly the mean of the stored per-genome rows; (4) the null arm measures nothing."""
    monkeypatch.setattr(sweep, "S4_EPISODES", 2)
    p = sweep.params_for("treatment", "baseline", None)
    monkeypatch.setattr(sweep, "S4_AFTER", (3,))
    with_s4 = sweep.one_seed(p, 20, n_generations=3)
    monkeypatch.setattr(sweep, "S4_AFTER", ())
    without = sweep.one_seed(p, 20, n_generations=3)

    assert list(with_s4["s4_after"]) == [3] and with_s4["s4_eng"].shape == (1, p.p_pop, 2, 7)
    static = sweep.E.StaticEnv(**_static(p, 20))
    unflatten = genome.unflattener(net.init_params(jax.random.key(20), p))
    eng, prey = engagement.measure(
        engagement.make_engagement_pass(unflatten, p), jnp.asarray(with_s4["final_pop"]), static, 20, 3, 2
    )
    assert np.array_equal(eng, with_s4["s4_eng"][0])
    assert np.array_equal(prey.astype(np.int32).sum(axis=-1), with_s4["s4_prey"][0])

    assert np.array_equal(with_s4["final_pop"], without["final_pop"]), "measuring changed the evolution"
    assert np.array_equal(with_s4["fit_mean"], without["fit_mean"])
    assert without["s4_eng"].shape[0] == 0

    assert np.array_equal(with_s4["final_rows"].mean(axis=0), with_s4["acc"][-1])
    assert with_s4["final_rows"].shape == (p.p_pop, len(sweep.EXPECTED_SCHEMA))

    monkeypatch.setattr(sweep, "S4_AFTER", (3,))
    null = sweep.one_seed(sweep.params_for("null", "baseline", None), 20, n_generations=3)
    assert null["s4_eng"].shape[0] == 0 and null["final_rows"].shape[0] == p.p_pop


def _static(p, seed):
    """The world `one_seed` builds for this seed, by the same three calls."""
    ob = sweep.grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, _ = sweep.resources.place_sources(jax.random.key(seed), ob, p)
    return {
        "obstacles": ob,
        "vis_table": sweep.build_vis_table(ob, sweep.required_radius(p)),
        "food_src": fs,
        "target_src": ts,
    }


# ---------------------------------------------------------------------------------- Phase A re-runs ----
def test_a_phase_a_rerun_must_say_so_and_avoid_the_long_runs_seeds(tmp_path):
    sweep.guard_phase_a(tmp_path / "anything", None, None)  # a normal run is not affected
    sweep.guard_phase_a(tmp_path / "phase-a-null", 0, 300)
    with pytest.raises(sm.FatalStop, match="Phase A re-runs only"):
        sweep.guard_phase_a(tmp_path / "longrun", 0, 300)
    with pytest.raises(sm.FatalStop, match="overlap"):
        sweep.guard_phase_a(tmp_path / "phase-a-null", 10, None)
    with pytest.raises(sm.FatalStop, match="at least 1"):
        sweep.guard_phase_a(tmp_path / "phase-a-null", 0, 0)


def test_the_phase_a_options_set_the_plans_seeds_and_budget(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sweep, "SEED_OFFSET", sweep.SEED_OFFSET)  # restored after, whatever main sets
    monkeypatch.setattr(sweep, "N_GENERATIONS", sweep.N_GENERATIONS)
    # main refuses to start below 4 GB of free host memory (D-054). That check is not what this test is about and
    # made it fail inside the full hook run beside the calibration, so it is held open here; it is tested on its own.
    monkeypatch.setattr(sweep.src, "memory_headroom", lambda **kw: {"ok": True})
    rc = sweep.main(
        [
            str(tmp_path / "phase-a-null"),
            "--arms",
            "null",
            "--seed-offset",
            "0",
            "--generations",
            "300",
            "--plan-only",
        ]
    )
    out = capsys.readouterr().out
    assert rc == sm.EXIT_DONE, out[-600:]
    assert "seeds 0 to 19, 300 generations" in out
    assert "null" in out and "29" in out


# ------------------------------------------------------------------------- crash durability, DEV-041 ----
def test_a_checkpoint_is_forced_to_disk_before_it_is_renamed(tmp_path, monkeypatch):
    """The order is the point: contents synced, then renamed, then the directory synced. A rename that reaches the
    disk before the contents is what left a 0-byte calibration file after the 2026-11-07 crash."""
    events = []
    real_fsync, real_replace = sweep.os.fsync, pathlib.Path.replace
    monkeypatch.setattr(sweep.os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(
        pathlib.Path, "replace", lambda self, t: (events.append("rename"), real_replace(self, t))[1]
    )
    tmp = tmp_path / ".c.npz"
    np.savez(tmp, x=np.arange(3))
    sweep.durable_replace(tmp, tmp_path / "c.npz")
    assert events == ["fsync", "rename", "fsync"]
    assert sweep.readable(tmp_path / "c.npz") and not tmp.exists()


def test_an_unreadable_condition_is_set_aside_and_redone(tmp_path, monkeypatch):
    calls = []

    def one_seed(p, seed, eval_params=None):
        calls.append(seed)
        cols = len(sweep.EXPECTED_SCHEMA)
        return {
            "fit_mean": np.ones(sweep.N_GENERATIONS, np.float32),
            "fit_max": np.ones(sweep.N_GENERATIONS, np.float32),
            "gens": np.array([0], np.int32),
            "acc": np.zeros((1, cols), np.float32),
            "relax": 0,
            "final_pop": np.zeros((4, 3), np.float32),
            "final_rows": np.zeros((4, cols), np.float32),
            "s4_after": np.array(sweep.S4_AFTER, np.int32),
            "s4_eng": np.zeros((len(sweep.S4_AFTER), 4, 2, 7), np.int32),
            "s4_prey": np.zeros((len(sweep.S4_AFTER), 4, 2, 6), np.int32),
        }

    monkeypatch.setattr(sweep, "one_seed", one_seed)
    monkeypatch.setattr(sweep, "N_SEEDS", 2)
    (tmp_path / "treatment__baseline__None.npz").write_bytes(b"")  # what the crash left
    part = tmp_path / "partial"
    part.mkdir()
    (part / f"treatment__baseline__None.seed{sweep.SEED_OFFSET}.npz").write_bytes(b"PK\x03\x04 truncated")
    res = sweep.run_condition("treatment", "baseline", None, tmp_path)
    assert res is not None and calls == sweep.seed_list(2), "both seeds redone, the condition rewritten"
    assert sweep.readable(tmp_path / "treatment__baseline__None.npz")
    assert (tmp_path / "treatment__baseline__None.npz.unreadable").exists(), (
        "the damaged file is kept to inspect"
    )


def test_a_readable_checkpoint_with_a_wrong_schema_still_stops_the_run(tmp_path):
    """Setting aside is for DAMAGE only. A complete file from other code or another schema must still stop."""
    p = tmp_path / "treatment__baseline__None.npz"
    np.savez(p, acc_fields=np.array(["only_one"]), arm=np.array("treatment"), final_pop=np.zeros(1))
    assert sweep.readable(p)
    with pytest.raises(SystemExit, match="SCHEMA MISMATCH"):
        sweep.run_condition("treatment", "baseline", None, tmp_path)


# ------------------------------------------------------------------------ the axis-level rule, Q-060 ----
def test_a_frozen_value_between_levels_replaces_the_nearest_interior_level(monkeypatch):
    lv = sweep.LEVELS["cue_dependence"]
    assert sweep.axis_levels("cue_dependence", {"cue_dependence": 0.85}) == (0.0, 0.25, 0.5, 0.85, 1.0)
    assert sweep.axis_levels("cue_dependence", {"cue_dependence": 0.6}) == (0.0, 0.25, 0.6, 0.75, 1.0)
    # Values whose NEAREST level is an end: the fixture must be able to tell an end-preserving rule from one
    # that replaces the end (0.85 and 0.6 are nearer interior levels, so both rules agree on them)
    assert sweep.axis_levels("cue_dependence", {"cue_dependence": 0.95}) == (0.0, 0.25, 0.5, 0.95, 1.0)
    assert sweep.axis_levels("cue_dependence", {"cue_dependence": 0.05}) == (0.0, 0.05, 0.5, 0.75, 1.0)
    assert sweep.axis_levels("cue_dependence", {"cue_dependence": 0.75}) == lv, (
        "an existing level changes nothing"
    )
    assert sweep.axis_levels("cue_dependence", {"p_wander": 0.5}) == lv, (
        "another axis's value changes nothing"
    )
    monkeypatch.setattr(frozen_world, "WORLD", {"cue_dependence": 0.85, "p_wander": 0.5})
    counts = {a: len(sweep.conditions(a)) for a in ("treatment", "null", "third")}
    assert counts == {"treatment": 53, "null": 29, "third": 27}, "the map keeps its shape"
    cue = [x for ax, x in sweep.conditions("treatment") if ax == "cue_dependence"]
    assert 0.85 not in cue and {0.0, 1.0} <= set(cue), "the frozen value is the baseline; both ends stay"
    sweep.require_frozen_world(smoke=False)  # an interior value is accepted


def test_only_cue_dependence_may_be_frozen_between_levels():
    """Q-060's replacement rule is scoped to the axis it was proposed for. Every other axis keeps the strict rule."""
    assert sweep.REPLACEABLE == {"cue_dependence"}
    assert sweep.axis_levels("p_wander", {"p_wander": 0.3}) == sweep.LEVELS["p_wander"]
