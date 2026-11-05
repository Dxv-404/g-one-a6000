"""The replay of the sweep's final populations, run 48: its plan, its engagement counter, and its fidelity.

The fast tests pin the plan and the engagement counter on hand-built cases, one per way an engagement can end,
and confirm each accounting identity fires when the data breaks it. The slow tests build REAL checkpoints with
the sweep's own `run_condition` at one seed and two generations, replay them, and require the stored rows to be
reproduced bit for bit, then corrupt one stored number and require the replay to refuse. That is the
mechanism-coupled check: the key reconstruction is judged against the program that produced the keys, not
against the replay's own idea of it.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "replay_final", ROOT / "scripts" / "measure" / "replay_final.py"
)
rf = importlib.util.module_from_spec(_SPEC)
sys.modules["replay_final"] = rf  # a dataclass in the module looks itself up here while it is being defined
_SPEC.loader.exec_module(rf)
sweep = rf.sweep


# ---------------------------------------------------------------------------------------------- the plan --
def test_the_plan_counts_follow_from_the_sweep_and_the_exclusions():
    jobs = rf.plan()
    own = [j for j in jobs if j.kind == "own"]
    by_arm = {a: sum(j.arm == a for j in own) for a in ("treatment", "third", "null")}
    # 53 treatment minus the two predator radii above the prey's; 27 third arm minus three repeats of its
    # baseline (DEV-035) minus the radius-6 end; 29 null.
    assert by_arm == {"treatment": 51, "third": 23, "null": 29}
    assert sum(j.kind == "real" for j in jobs) == 23
    assert sum(j.kind == "nullp" for j in jobs) == 22
    assert len({j.name for j in jobs}) == len(jobs), "job names collide"


def test_no_job_lets_the_predator_out_see_the_prey():
    for j in rf.plan():
        if j.axis == "r_pred":
            assert int(j.level) <= rf.DEFAULT.r, j.name
    assert rf.excluded("r_pred", rf.DEFAULT.r + 1) and not rf.excluded("r_pred", rf.DEFAULT.r)


def test_the_null_is_evaluated_at_every_predator_setting_the_treatment_ran():
    treat = {
        (a, lv)
        for a, lv in sweep.conditions("treatment")
        if a in sweep.PREDATOR_ONLY and not rf.excluded(a, lv)
    }
    nullp = {(j.axis, j.level) for j in rf.plan() if j.kind == "nullp"}
    assert nullp == treat
    third_pred = {(j.axis, j.level) for j in rf.plan() if j.kind == "real" and j.axis in sweep.PREDATOR_ONLY}
    assert third_pred <= nullp, "a third-arm predator setting would have no null to compare against"


def test_the_predator_world_is_exactly_the_scripted_predators_world():
    scripted = sweep.BY_NAME["scripted"].params
    for j in rf.plan():
        if j.world != "predator":
            continue
        want = scripted if j.axis == "baseline" else dataclasses.replace(scripted, **{j.axis: j.level})
        assert rf.world_params(j) == want, j.name


def test_the_own_world_is_where_the_stored_numbers_came_from():
    for j in rf.plan():
        if j.kind != "own":
            continue
        if j.arm == "null":
            base = sweep.BY_NAME["naive_eval"].params
            want = base if j.axis == "baseline" else dataclasses.replace(base, **{j.axis: j.level})
        else:
            want = sweep.params_for(j.arm, j.axis, j.level)
        assert rf.world_params(j) == want, j.name
        assert j.key_kind == ("ke" if j.arm == "null" else "ka"), j.name


def test_the_third_arm_own_world_cannot_damage_and_its_real_world_can():
    own = rf.world_params(next(j for j in rf.plan() if j.kind == "own" and j.arm == "third"))
    real = rf.world_params(next(j for j in rf.plan() if j.kind == "real"))
    assert own.damage == 0.0 and real.damage > 0.0


# ------------------------------------------------------------------------------- the engagement counter --
def _case():
    """One genome, ten ticks, five prey, one per way an engagement can end, written as engaged/not per tick."""
    T, k = 10, 5
    e = np.zeros((1, T, k), np.int64)
    e[0, 2:5, 0] = 1  # prey 0: engaged 2 to 4, alive after: ESCAPED
    e[0, 7:10, 0] = 1  #          and 7 to 9, still engaged on the last tick and alive: CENSORED
    e[0, 3:6, 1] = 1  # prey 1: engaged 3 to 5, dies predated on tick 5: CAPTURED
    e[0, 1:3, 2] = 1  # prey 2: engaged 1 to 2, dies starving on tick 2: DIED_OTHER
    # prey 3: never engaged, dies predated on tick 6: PREDATED OUTSIDE ANY ENGAGEMENT
    # prey 4: never engaged, survives
    running = np.cumsum(e, axis=1)
    prey = np.zeros((1, 6, k), np.int64)
    prey[0, 1] = [-1, 5, 2, 6, -1]  # death step
    prey[0, 2] = [0, 1, 0, 1, 0]  # died_predated
    prey[0, 3] = [0, 0, 1, 0, 0]  # died_starving
    prey[0, 4] = [1, 0, 0, 0, 1]  # alive at end
    return running, prey


def test_each_way_an_engagement_ends_is_counted_once():
    running, prey = _case()
    out = dict(zip(rf.ENGAGEMENT_FIELDS, rf.engagement_counts(running, prey)[0].tolist(), strict=True))
    # Four engagements: prey 0 has two, prey 1 and prey 2 one each. A first draft of this test said five, a
    # hand count that was wrong while the code was right, which the per-outcome counts summing to four exposed.
    assert out == {
        "engagements": 4,
        "escaped": 1,
        "captured": 1,
        "died_other": 1,
        "censored": 1,
        "engaged_ticks": 3 + 3 + 3 + 2,
        "predated_outside_engagement": 1,
    }


def test_a_counter_that_jumps_by_two_is_refused():
    running, prey = _case()
    running[0, 5:, 4] += 2
    with pytest.raises(AssertionError, match="other than 0 or 1"):
        rf.engagement_counts(running, prey)


def test_a_counter_that_goes_backwards_is_refused():
    """Prey 4 is never engaged, so lowering its last value is a genuine step backwards. Lowering an engaged
    prey's last value would only shorten its engagement, which is valid data, and a first draft did exactly
    that and was rightly not refused."""
    running, prey = _case()
    running[0, -1, 4] -= 1
    with pytest.raises(AssertionError, match="other than 0 or 1"):
        rf.engagement_counts(running, prey)


def test_the_independent_check_against_the_sweeps_accumulator_can_fail():
    """The check that is NOT definitional: engaged ticks counted here against the sweep's own accumulator."""
    running, prey = _case()
    counts = rf.engagement_counts(running, prey)
    cols = list(sweep.EXPECTED_SCHEMA)
    k = running.shape[2]
    rows = np.zeros((1, len(cols)), np.float32)
    rows[0, cols.index("engaged_ticks")] = counts[0, rf.ENGAGEMENT_FIELDS.index("engaged_ticks")] / k
    assert rf.sweep_mismatch(counts, rows, cols, k) == 0
    rows[0, cols.index("engaged_ticks")] += 1.0 / k  # one tick the sweep saw and this pass did not
    assert rf.sweep_mismatch(counts, rows, cols, k) == 1


def test_a_trajectory_that_ends_differently_from_the_counted_episode_is_marked():
    G, T_, k = 6, 5, 3
    prey = np.zeros((G, 6, k), np.int32)
    prey[:, 4, :] = 1  # every prey alive at the end of the counted episode
    prey[4, 4, 2] = 0  # except genome 4's third prey
    idx = np.array([1, 4, 5])
    alive = np.ones((idx.size, T_, k), bool)
    alive[1, -1, 2] = False
    assert not rf.traj_mismatch(alive, prey, idx).any()
    alive[2, -1, 0] = False  # genome 5's trajectory loses a prey the counted episode kept
    assert rf.traj_mismatch(alive, prey, idx).tolist() == [False, False, True]
    alive[1, -1, 2] = True  # and genome 4's keeps one the counted episode lost
    assert rf.traj_mismatch(alive, prey, idx).tolist() == [False, True, True]


def test_engagements_in_different_genomes_and_prey_are_kept_apart():
    running, prey = _case()
    two_r, two_p = np.concatenate([running, running[:, :, ::-1]]), np.concatenate([prey, prey[:, :, ::-1]])
    a = rf.engagement_counts(two_r, two_p)
    assert np.array_equal(a[0], a[1]), "reordering prey within a genome changed its counts"


# ---------------------------------------------------------------------------------------------- keys -----
def test_the_key_chain_is_the_sweeps_and_depends_on_the_generation_count():
    ka2, ke2 = rf.seed_keys(0, 2)
    ka3, ke3 = rf.seed_keys(0, 3)
    import jax

    assert not np.array_equal(jax.random.key_data(ka2), jax.random.key_data(ka3))
    assert not np.array_equal(jax.random.key_data(ke2), jax.random.key_data(ke3))
    assert not np.array_equal(jax.random.key_data(ka2), jax.random.key_data(ke2))


# ------------------------------------------------------------------------- the end to end, on the GPU ----
@pytest.fixture(scope="module")
def tiny_sweep(tmp_path_factory):
    """Real checkpoints from the sweep's own runner: one seed, two generations, full scale."""
    out = tmp_path_factory.mktemp("sweep")
    saved = (sweep.N_SEEDS, sweep.N_GENERATIONS, sweep.SAMPLE_EVERY)
    sweep.N_SEEDS, sweep.N_GENERATIONS, sweep.SAMPLE_EVERY = 1, 2, 1
    try:
        for arm in ("treatment", "null", "third"):
            sweep.run_condition(arm, "baseline", None, out)
    finally:
        sweep.N_SEEDS, sweep.N_GENERATIONS, sweep.SAMPLE_EVERY = saved
    return out


JOBS = "^(own__treatment__baseline|own__null__baseline|own__third__baseline|real__third__baseline|nullp__null__n_pred__2)"


@pytest.mark.slow
def test_the_replay_reproduces_the_sweep_bit_for_bit_and_measures_the_rest(tiny_sweep, tmp_path):
    out = tmp_path / "replay"
    assert rf.main([str(tiny_sweep), str(out), "--jobs", JOBS, "--episodes", "2", "--traj-genomes", "4"]) == 0
    files = {p.stem: p for p in out.glob("*__*.npz")}
    assert set(files) == {
        "own__treatment__baseline__None",
        "own__null__baseline__None",
        "own__third__baseline__None",
        "real__third__baseline__None",
        "nullp__null__n_pred__2",
    }
    for name, path in files.items():
        with np.load(path) as d:
            rows, eng, prey = d["rows"], d["engagement"], d["prey"]
            assert rows.shape == (1, 256, len(sweep.EXPECTED_SCHEMA)), name
            assert eng.shape == (1, 256, 2, len(rf.ENGAGEMENT_FIELDS)), name
            assert prey.shape[:3] == (1, 256, 2), name
            assert d["traj_prey_pos"].shape[:3] == (1, 4, d["traj_prey_pos"].shape[2])
            assert d["traj_diverged"].shape == (1, 4) and not d["traj_diverged"].any(), name
            repro = json.loads(str(d["reproduction"]))
            cross = json.loads(str(d["cross_check"]))
            if name.startswith("own__"):
                assert repro[0]["exact"], f"{name} did not reproduce the stored row bit for bit"
            assert cross[0]["max_abs_diff"] < 1e-4, f"{name}: engagement pass disagrees with the sweep's pass"
            cols = list(sweep.EXPECTED_SCHEMA)
            # the counter's engaged ticks over episode 0 equal the sweep pass's engaged_ticks, per genome
            ep0 = eng[0, :, 0, rf.ENGAGEMENT_FIELDS.index("engaged_ticks")]
            assert np.allclose(ep0 / sweep.DEFAULT.k, rows[0, :, cols.index("engaged_ticks")], atol=1e-5), (
                name
            )
    with np.load(files["own__third__baseline__None"]) as d:
        assert float(d["rows"][..., list(sweep.EXPECTED_SCHEMA).index("predated")].max()) == 0.0
    with np.load(files["nullp__null__n_pred__2"]) as d:
        assert str(d["world"]) == "predator" and str(d["key_kind"]) == "ke"


@pytest.mark.slow
def test_the_replay_refuses_a_stored_row_it_cannot_reproduce(tiny_sweep, tmp_path):
    """Falsification of the reproduction guard: one stored number changed, and the replay must stop."""
    bad = tmp_path / "sweep"
    bad.mkdir()
    src = tiny_sweep / "treatment__baseline__None.npz"
    with np.load(src) as d:
        data = {k: d[k] for k in d.files}
    data["sample_acc"] = data["sample_acc"].copy()
    data["sample_acc"][0, -1, 0] += 1e-3
    np.savez(bad / src.name, **data)
    with pytest.raises(SystemExit, match="bit for bit"):
        rf.main(
            [
                str(bad),
                str(tmp_path / "out"),
                "--jobs",
                "^own__treatment__baseline",
                "--episodes",
                "1",
                "--traj-genomes",
                "0",
            ]
        )
