"""Queue item B: the resume schema guard, and the arm-pairing assertion. One test each.

**Neither of these is satisfied by having fixed an instance.** Six stale checkpoints were archived on
2026-08-30; archiving fixed six files and prevented nothing. D-052's own logic, gate on states rather than
on instances, applies to its neighbour: the NEXT schema change recreates the hazard unless a checkpoint's
field set is asserted against the current schema at load time.

**The pairing test is the other half and it guards a claim rather than a file.** D-045 ruled that every
pre-registered comparison over arms sharing seeds is stated as a PAIRED test, because the seed-to-seed
spread is largely a property of the seed rather than of the arm. That ruling is only sound if the arms
actually do share layouts at a given condition. Nothing asserted it, and the arms differ in `params`, so
the sharing is a consequence of which fields `place_sources` happens to read rather than a guaranteed
property.
"""

from __future__ import annotations

import importlib.util
import pathlib

import jax
import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "sweep_mod", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "measure" / "sweep.py"
)
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)

from src.env import grid, resources  # noqa: E402


# ------------------------------------------------------------------ the resume schema guard --
def _write_checkpoint(path: pathlib.Path, fields, arm: str = "treatment") -> None:
    """A minimal checkpoint carrying only what the guard reads: the schema, the arm, the final population."""
    np.savez(
        path,
        acc_fields=np.array(list(fields)),
        arm=np.array(arm),
        final_pop=np.zeros((1, 1), np.float32),
        sample_acc=np.zeros((2, 2, len(fields))),
    )


def test_the_expected_schema_is_derived_from_the_accumulator_set(tmp_path):
    """**Derived, not transcribed.** A literal column list here would drift from `Accumulators` exactly as
    the sprint-uptime formula drifted from `update_stamina`, and D-051 is that lesson."""
    from src.env.env import Accumulators

    assert sweep.EXPECTED_SCHEMA[: len(Accumulators._fields)] == Accumulators._fields
    assert sweep.EXPECTED_SCHEMA[len(Accumulators._fields) :] == sweep.OUTCOME_FIELDS
    assert len(sweep.EXPECTED_SCHEMA) == len(Accumulators._fields) + 3


def test_a_current_checkpoint_passes_the_guard(tmp_path):
    """The coverage half. A guard that rejects everything would also reject stale files."""
    p = tmp_path / "treatment__baseline__None.npz"
    _write_checkpoint(p, sweep.EXPECTED_SCHEMA)
    sweep.assert_checkpoint_schema(p)  # must not raise


def test_a_stale_checkpoint_is_refused_by_the_guard(tmp_path):
    """**The real instance, reconstructed.** The six archived checkpoints stored the 21 accumulators of
    2026-08-28 plus three outcome columns, 24 in all, and lacked the seven near-conditioned and away fields
    that D-046 and D-048 added. Resuming would have carried them into the final dataset."""
    stale = [f for f in sweep.EXPECTED_SCHEMA if not (f.endswith("_near") or f.endswith("_far"))]
    assert len(stale) < len(sweep.EXPECTED_SCHEMA), "the fixture must actually be missing fields"

    p = tmp_path / "treatment__cue_dependence__0.0.npz"
    _write_checkpoint(p, stale)
    with pytest.raises(SystemExit) as e:
        sweep.assert_checkpoint_schema(p)
    msg = str(e.value)
    assert "SCHEMA MISMATCH" in msg
    assert "still_near" in msg, "the message must name what is missing, or the operator cannot act on it"


def test_a_checkpoint_with_no_schema_field_is_refused_rather_than_assumed(tmp_path):
    """A checkpoint predating the guard cannot be checked, and unskippable is the safe reading. Silently
    accepting it would reintroduce the exact hazard for every file written before today."""
    p = tmp_path / "treatment__baseline__None.npz"
    np.savez(p, something_else=np.zeros(3))
    with pytest.raises(SystemExit) as e:
        sweep.assert_checkpoint_schema(p)
    assert "no `acc_fields`" in str(e.value)


# --------------------------------------------------------------------- the pairing assertion --
def _layout(arm: str, axis: str, level, seed: int):
    """The obstacle grid and source placements one arm actually gets at one condition and seed."""
    p = sweep.params_for(arm, axis, level)
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, relax = resources.place_sources(jax.random.key(seed), ob, p)
    return np.asarray(ob), np.asarray(fs), np.asarray(ts), int(relax)


@pytest.mark.parametrize("axis,level", [("baseline", None), ("k", 12), ("s_burst_prey", 40.0)])
def test_every_arm_shares_the_layout_at_a_given_condition_and_seed(axis, level):
    """**D-045's paired comparisons are only sound if the arms are actually paired.**

    The three arms differ in `params`: the null sets `n_pred = 0` and the third arm sets `damage = 0`.
    Neither field is read by `make_obstacle_grid` or `place_sources`, so the layouts coincide. **That is a
    consequence of which fields those functions happen to read, not a guarantee**, and a future arm
    differing in `obstacle_density` or `n_food_src` would silently unpair every comparison while every
    gate stayed green.

    Asserted across three conditions rather than one, including a `k` level, because `k` is the field most
    likely to be mistaken for a layout parameter.
    """
    seeds = (0, 1)
    for seed in seeds:
        base = _layout("treatment", axis, level, seed)
        for arm in ("null", "third"):
            other = _layout(arm, axis, level, seed)
            assert np.array_equal(base[0], other[0]), (
                f"{arm} sees a different obstacle grid from treatment at {axis}={level}, seed {seed}. "
                "The arms are not paired and every cross-arm comparison is confounded by layout."
            )
            assert np.array_equal(base[1], other[1]), f"{arm}: food sources differ at {axis}={level}"
            assert np.array_equal(base[2], other[2]), f"{arm}: target sources differ at {axis}={level}"
            assert base[3] == other[3], f"{arm}: spawn relaxation level differs at {axis}={level}"


def test_the_pairing_test_can_fail():
    """**Coverage for the test above, which would pass vacuously if the layouts were trivially equal.**

    A gate that cannot fail manufactures confidence, and this one is at risk of exactly that: if
    `make_obstacle_grid` ignored its parameters the assertion would hold for every arm forever. Perturbing
    a field the layout DOES read must produce a different grid.
    """
    a = _layout("treatment", "baseline", None, 0)
    b = _layout("treatment", "obstacle_density", 0.35, 0)
    assert not np.array_equal(a[0], b[0]), (
        "changing obstacle_density did not change the obstacle grid, so the pairing assertion above is "
        "comparing a constant against itself and proves nothing"
    )


# ------------------------------------------------------------------ D-059: final population, naive_eval, per-seed partials --
#
# **These run the REAL `run_condition` with a stand-in for the evolutionary run.** The mechanism under test is
# the runner's bookkeeping: what it writes, what it reuses on resume, and which world it evaluates the null
# in. None of that needs a GPU, and stubbing `one_seed` is what lets the crash-and-resume path be exercised
# deterministically, which a real 75-minute condition never could be in a test.


def _stub_one_seed(calls: list):
    """Returns the runner's per-seed contract with recognisable values, and records every call."""

    def one_seed(p, seed: int, eval_params=None) -> dict:
        calls.append((seed, eval_params))
        n_cols = len(sweep.EXPECTED_SCHEMA)
        r = {
            "fit_mean": np.full(sweep.N_GENERATIONS, float(seed), np.float32),
            "fit_max": np.full(sweep.N_GENERATIONS, float(seed), np.float32),
            "gens": np.array([0, sweep.N_GENERATIONS - 1], np.int32),
            "acc": np.zeros((2, n_cols), np.float32),
            "relax": 0,
            "final_pop": np.full((4, 3), float(seed), np.float32),
        }
        if eval_params is not None:
            r["naive_acc_per_genome"] = np.full((4, n_cols), float(seed), np.float32)
            r["naive_acc"] = np.full(n_cols, float(seed), np.float32)
        return r

    return one_seed


@pytest.fixture
def stubbed(monkeypatch):
    calls: list = []
    monkeypatch.setattr(sweep, "one_seed", _stub_one_seed(calls))
    monkeypatch.setattr(sweep, "N_SEEDS", 3)
    return calls


def test_a_condition_keeps_the_final_population_of_every_seed(tmp_path, stubbed):
    sweep.run_condition("treatment", "baseline", None, tmp_path)
    (path,) = tmp_path.glob("*.npz")
    with np.load(path) as d:
        assert d["final_pop"].shape == (3, 4, 3)
        assert [float(x) for x in d["final_pop"][:, 0, 0]] == [0.0, 1.0, 2.0]
        assert "naive_acc" not in d.files, "only the null arm carries a naive_eval pass"
    assert not list((tmp_path / "partial").glob("*.npz")), "partials are removed once the condition is whole"


def test_only_the_null_arm_is_given_an_evaluation_world(tmp_path, stubbed):
    for arm in ("treatment", "third"):
        sweep.run_condition(arm, "baseline", None, tmp_path)
    assert all(ep is None for _, ep in stubbed)


def test_the_null_arm_evaluates_in_the_predator_world_at_the_same_axis_level(tmp_path, stubbed):
    """The pair: predator-naive weights, predator-present world, one replacement applied to both."""
    axis, level = sweep.conditions("null")[1]  # the first swept condition the null arm has, derived
    sweep.run_condition("null", axis, level, tmp_path)
    evolved = sweep.params_for("null", axis, level)
    for _, ep in stubbed:
        assert ep is not None
        assert evolved.n_pred == 0 and ep.n_pred > 0, (
            "the lineage evolved without a predator, evaluated with one"
        )
        assert getattr(ep, axis) == level, "the evaluation world carries this condition's replacement"
        assert ep == sweep.params_for("treatment", axis, level), "and is otherwise the treatment world"
    (path,) = tmp_path.glob("*.npz")
    with np.load(path) as d:
        assert d["naive_acc"].shape == (3, len(sweep.EXPECTED_SCHEMA))
        assert d["naive_acc_per_genome"].shape == (3, 4, len(sweep.EXPECTED_SCHEMA))


def _crashing_after_seed_zero(real):
    def one_seed(p, seed: int, eval_params=None):
        if seed == 1:
            raise RuntimeError("simulated crash after seed 0 finished")
        return real(p, seed, eval_params)

    return one_seed


def test_a_crash_mid_condition_costs_only_the_unfinished_seeds(tmp_path, monkeypatch):
    calls: list = []
    real = _stub_one_seed(calls)
    monkeypatch.setattr(sweep, "N_SEEDS", 3)
    monkeypatch.setattr(sweep, "one_seed", _crashing_after_seed_zero(real))
    with pytest.raises(RuntimeError):
        sweep.run_condition("treatment", "baseline", None, tmp_path)
    assert [s for s, _ in calls] == [0]
    assert len(list((tmp_path / "partial").glob("*.seed0.npz"))) == 1, "seed 0's partial survives the crash"
    assert not list(tmp_path.glob("*.npz")), "no condition file was written for a half-finished condition"

    calls.clear()
    monkeypatch.setattr(sweep, "one_seed", real)
    sweep.run_condition("treatment", "baseline", None, tmp_path)
    assert [s for s, _ in calls] == [1, 2], "seed 0 came from its partial and was not recomputed"
    (path,) = tmp_path.glob("*.npz")
    with np.load(path) as d:
        assert [float(x) for x in d["final_pop"][:, 0, 0]] == [0.0, 1.0, 2.0], (
            "the resumed seed's data is intact"
        )
    assert not list((tmp_path / "partial").glob("*.npz"))


def test_the_resume_test_can_fail(tmp_path, monkeypatch):
    """Couples the test above to the mechanism: remove the partial and seed 0 IS recomputed. Without this,
    a runner that ignored partials and happened to produce the same file would pass the test above."""
    calls: list = []
    real = _stub_one_seed(calls)
    monkeypatch.setattr(sweep, "N_SEEDS", 3)
    monkeypatch.setattr(sweep, "one_seed", _crashing_after_seed_zero(real))
    with pytest.raises(RuntimeError):
        sweep.run_condition("treatment", "baseline", None, tmp_path)
    for pp in (tmp_path / "partial").glob("*.npz"):
        pp.unlink()
    calls.clear()
    monkeypatch.setattr(sweep, "one_seed", real)
    sweep.run_condition("treatment", "baseline", None, tmp_path)
    assert [s for s, _ in calls] == [0, 1, 2]


def test_a_stale_partial_is_refused_rather_than_resumed(tmp_path, stubbed):
    """The schema guard applies to partials too, or a crash straddling a schema change would splice an old
    seed into a new condition."""
    part = tmp_path / "partial"
    part.mkdir()
    stale = [f for f in sweep.EXPECTED_SCHEMA if not (f.endswith("_near") or f.endswith("_far"))]
    np.savez(
        part / "treatment__baseline__None.seed0.npz",
        arm=np.array("treatment"),
        acc_fields=np.array(stale),
        final_pop=np.zeros((4, 3), np.float32),
        relax=np.int32(0),
    )
    with pytest.raises(SystemExit, match="SCHEMA MISMATCH"):
        sweep.run_condition("treatment", "baseline", None, tmp_path)


def test_the_guard_requires_the_final_population(tmp_path):
    p = tmp_path / "c.npz"
    np.savez(p, acc_fields=np.array(sweep.EXPECTED_SCHEMA), arm=np.array("treatment"))
    with pytest.raises(SystemExit, match="final_pop"):
        sweep.assert_checkpoint_schema(p)


def test_the_guard_requires_the_null_arm_to_carry_its_naive_eval(tmp_path):
    n = tmp_path / "n.npz"
    np.savez(n, acc_fields=np.array(sweep.EXPECTED_SCHEMA), arm=np.array("null"), final_pop=np.zeros(1))
    with pytest.raises(SystemExit, match="naive_acc"):
        sweep.assert_checkpoint_schema(n)
    t = tmp_path / "t.npz"
    np.savez(t, acc_fields=np.array(sweep.EXPECTED_SCHEMA), arm=np.array("treatment"), final_pop=np.zeros(1))
    sweep.assert_checkpoint_schema(t)  # a non-null arm without one is whole


@pytest.mark.slow
def test_the_real_null_pass_sees_a_predator_the_lineage_never_did(tmp_path, monkeypatch):
    """The stubbed tests above check the runner's bookkeeping. This runs the REAL `one_seed` on the null
    arm, two generations at full scale, through `run_condition`, and asserts the mechanism the pair rests
    on: the lineage's own accumulators come from a world with no predator, and the naive_eval pass's come
    from one with a predator. Marked slow: two generations plus two compiles is about a minute, per the
    marking rule in CLAUDE.md section 10."""
    from src.env.env import Accumulators

    monkeypatch.setattr(sweep, "N_SEEDS", 1)
    monkeypatch.setattr(sweep, "N_GENERATIONS", 2)
    monkeypatch.setattr(sweep, "SAMPLE_EVERY", 1)
    res = sweep.run_condition("null", "baseline", None, tmp_path)
    assert res is not None
    (path,) = tmp_path.glob("*.npz")
    sweep.assert_checkpoint_schema(path)

    cols = list(sweep.EXPECTED_SCHEMA)
    near = cols.index("near_ticks")
    predated = cols.index("predated")
    with np.load(path) as d:
        own = d["sample_acc"][0]  # (samples, columns), the lineage in its own world
        null = d["naive_acc"][0]  # (columns,), the same final population with the predator present
        per_genome = d["naive_acc_per_genome"][0]
        pop = d["final_pop"][0]
    assert pop.shape[0] == sweep.DEFAULT.p_pop and pop.dtype == np.float32
    assert per_genome.shape == (sweep.DEFAULT.p_pop, len(cols))
    assert np.allclose(per_genome.mean(axis=0), null, atol=1e-6), "the stored mean is the mean of the rows"

    # the mechanism, both directions
    assert float(own[:, near].max()) == 0.0, "the null lineage's own world has no predator to be near"
    assert float(own[:, predated].max()) == 0.0
    assert float(null[near]) > 0.0, "the naive_eval pass put the same weights in a world with a predator"
    assert "near_ticks" in Accumulators._fields  # the column the assertion above reads exists by that name
