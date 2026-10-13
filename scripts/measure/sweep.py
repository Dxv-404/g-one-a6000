#!/usr/bin/env python3
"""The M4 characterisation sweep. Step 12.

**Three properties matter more than the iteration, and each is a lesson this project paid for.**

    checkpoint per condition   a thermal stall or an OOM costs ONE condition, not five days. Every
                               condition writes its own npz and a completed condition is skipped on
                               restart, so the job is resumable rather than restartable
    clear caches per condition DEV-029's class alive outside the loop it was found in. RSS grew about
                               30 MB per seed in the freeze pricing runs, and a 1,090-run job would
                               reach tens of gigabytes without this
    register as runs           each condition becomes a row in `runs.md`, written by this script rather
                               than by hand, against a commit whose full differential is verified

**The commit must be verified BEFORE the sweep starts, and this refuses to run otherwise.** The binding
at `tests/scientific/test_differential_binding.py` would fail afterwards, which is too late for a job
measured in days. Checked here as a precondition rather than discovered as a test failure.

### The plan, derived rather than transcribed

Thirteen swept axes at five levels each. A condition is one axis at one non-default level, plus one shared
baseline. The counts fall out of the levels below rather than being asserted:

    treatment   13 axes x 4 non-default levels, plus the baseline          53
    null        the same MINUS the predator-only axes, which n_pred = 0    29
                makes inert. Q-023 named this saving and said it has to
                be designed in rather than discovered
    third arm   baseline plus both ends of every axis                      27
                                                                         ---
                                                                          109 conditions, 1,090 runs

**The specification said 50 treatment conditions and its own null figure of 29 is inconsistent with that:**
29 is treatment minus the 24 predator-only conditions, which requires treatment to be 53. The third arm's 27
is 1 + 2 x 13 and corroborates thirteen axes. 53 is used.

### What is recorded per condition, and why the timing is part of the data

Fitness trajectory, every accumulator plus the three tier-one outcomes at every sample generation, and
**wall clock**. The GPU reached 83 C under sustained load and the test suite slowed 3.5x across consecutive
runs this month. A five-day job on a laptop GPU is not five days of cold-start throughput, so per-condition
wall clock goes in the data where the throttling is visible, rather than being inferred afterwards from a
total that came in late.

### Watching it run, D-062

Progress is reported at every scale of the run: a line per condition and per seed, a line every
`SAMPLE_EVERY` generations with the tier-one outcomes, and a live status line on a real terminal. Anything out
of place is named in a WARNING line and kept in `warnings.log`. `status.json` in the output directory is
rewritten at least once a minute, so progress can be read without the terminal. The checks, their
thresholds and the exit codes live in `sweep_monitor.py`, which reads only what this script already has on
the host and is asserted not to change a single bit of the result.

Usage:
    uv run python scripts/measure/sweep.py <out_dir> --verified-commit <sha> [--arms treatment,null,third]
        [--limit N] [--progress quiet|normal|verbose] [--log-file PATH]
    uv run python scripts/measure/sweep.py <out_dir> --plan-only
    uv run python scripts/measure/sweep.py <scratch_dir> --smoke --verified-commit <sha>
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import pathlib
import re
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import DEFAULT, check_invariants, make_relax_ladder
from configs.run_configs import BY_NAME
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius
from src.evo import genome, loop

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sweep_monitor as sm  # noqa: E402  host-side reporting only, D-062

#: Ten, and the precision is REPORTED rather than assumed. Owner's ruling: doubling to twenty doubles the
#: whole plan for a factor of 1.41 in precision, and the primary claim routes through tier-one outcomes at
#: 8,192 prey per generation rather than through cross-seed means. The standard error of each condition's
#: mean is written into its npz so a reader gets the precision with the number.
#:
#: **If a condition lands near a decision boundary, add seeds for THAT condition** rather than re-running the
#: sweep. The freeze work established why: a required-n computed from a pilot chases both a moving effect and
#: a moving bar, so a global re-run at a larger n is the expensive way to answer a local question.
N_SEEDS = 10

#: 300, with the trajectory reported rather than an endpoint. Owner's ruling: neither 300 nor 900 converges,
#: run 1 measured 300 at 45.9 percent of the 900-generation gain, and choosing 900 buys a longer unfinished
#: process for three times the compute. The reporting commitment is what makes the cheaper budget defensible,
#: and it was made for an independent reason: generation zero is 98 percent still, so an endpoint-against-
#: start reading would be meaningless whatever the budget.
N_GENERATIONS = 300
SAMPLE_EVERY = 25

ACC_FIELDS = E.Accumulators._fields
OUTCOME_FIELDS = ("predated", "starved", "survived")

LEVELS: dict[str, tuple] = {
    "cue_dependence": (0.0, 0.25, 0.5, 0.75, 1.0),
    "p_wander": (0.0, 0.25, 0.5, 0.75, 1.0),
    "p_distant_detect": (0.0, 0.25, 0.5, 0.75, 1.0),
    "n_pred": (1, 2, 3, 4, 6),
    "r_pred": (2, 3, 4, 5, 6),
    "s_burst_pred": (4.0, 6.0, 8.0, 12.0, 16.0),
    "s_burst_prey": (10.0, 15.0, 20.0, 40.0, 80.0),
    "obstacle_density": (0.0, 0.08, 0.15, 0.25, 0.35),
    "k": (4, 6, 8, 10, 12),
    "e_sprint": (0.0, 0.5, 1.0, 2.0, 4.0),
    "n_food_src": (4, 5, 6, 9, 12),
    "n_target_src": (2, 4, 6, 9, 12),
    "d_src_relax": tuple(make_relax_ladder(h) for h in (2, 4, 8, 12, 16)),
}

#: Inert at `n_pred = 0`, so the null arm does not re-run them. Roughly a third of the sweep, free.
PREDATOR_ONLY = frozenset(
    {"cue_dependence", "p_wander", "p_distant_detect", "n_pred", "r_pred", "s_burst_pred"}
)

ARMS = {"treatment": "scripted", "null": "no_predator_evolve", "third": "harmless_evolve"}

#: `--smoke`: the whole runner path, checkpoints, resume, the null's naive_eval pass, logging and the status
#: file, at a size that finishes in minutes. Its output can never be mistaken for the sweep's: it refuses the
#: sweep's directory, marks its own, and a real run refuses any marked directory. D-062.
SMOKE_SEEDS = 2
SMOKE_GENERATIONS = 3
SMOKE_PLAN = (("treatment", "baseline", None), ("null", "baseline", None))
SMOKE_MARKER = "SMOKE_RUN"

#: The jax.monitoring event emitted once per trace. Measured in the pinned JAX 0.11.0: a cached call emits
#: nothing and a fresh callable emits this, which is exactly DEV-029's cause made observable.
TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"


def conditions(arm: str) -> list[tuple[str, object]]:
    """(axis, level) pairs for one arm, with `("baseline", None)` first. Derived, never transcribed."""
    out: list[tuple[str, object]] = [("baseline", None)]
    for axis, levels in LEVELS.items():
        if arm == "null" and axis in PREDATOR_ONLY:
            continue
        picked = [lv for lv in levels if lv != getattr(DEFAULT, axis)]
        if arm == "third":
            picked = [levels[0], levels[-1]]  # both ends only
        out.extend((axis, lv) for lv in picked)
    return out


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def make_accumulator_pass(unflatten, static, params):
    """A factory, per DEV-029. Returns every accumulator plus the three tier-one outcomes."""

    def one(vec, k):
        k_reset, k_draws = jax.random.split(k)
        st, acc = E.run_episode(k_reset, unflatten(vec), static, ent.PRNGSource(k_draws), params)
        base = jnp.stack([jnp.mean(getattr(acc, f).astype(jnp.float32)) for f in ACC_FIELDS])
        out = jnp.stack(
            [
                jnp.mean(st.prey_died_predated.astype(jnp.float32)),
                jnp.mean(st.prey_died_starving.astype(jnp.float32)),
                jnp.mean(st.prey_alive.astype(jnp.float32)),
            ]
        )
        return jnp.concatenate([base, out])

    batched = jax.jit(jax.vmap(one, in_axes=(0, 0)))

    def run_all(key, genomes):
        """(p_pop, n_columns): one row per genome. What the naive_eval pass stores, so S6 can resample."""
        keys = jax.random.split(key, params.p_pop)
        return np.asarray(batched(genomes, keys))

    def run(key, genomes):
        """(n_columns,): the population mean. What the per-generation samples store, unchanged."""
        return run_all(key, genomes).mean(axis=0)

    run.all = run_all

    return run


def params_for(arm: str, axis: str, level) -> object:
    base = BY_NAME[ARMS[arm]].params
    p = base if axis == "baseline" else dataclasses.replace(base, **{axis: level})
    check_invariants(p)
    return p


def one_seed(p, seed: int, eval_params=None, *, n_generations: int | None = None, progress=None) -> dict:
    """One evolutionary run. With `eval_params`, the final population is also evaluated in THAT world.

    **`progress` sees the run and cannot touch it.** After each generation it is handed the generation's wall
    time, the fitness vector this loop already copies to the host, and the accumulator row when one was
    sampled. No key is split for it and nothing runs on the GPU for it, so the key sequence, and therefore
    every number below, is the same with or without it. `test_sweep_progress.py` asserts that bitwise.
    `n_generations` exists for `--smoke` and the tests; the sweep always runs `N_GENERATIONS`.

    **The naive_eval pass lives here, not in a later job, and that is a decision.** `naive_eval` is the
    experiment's null: the predator-naive lineage's final weights evaluated against the scripted predator.
    `configs/run_configs.py` declares the dependency as `weights_from`, and the first version of this
    runner discarded the final population, so the null could never have been produced from its output.
    Evaluating here, while the population is in memory and the layout is the seed's, removes the
    sequencing dependency in practice and stores the null beside the arm it belongs to. D-059.

    The evaluation world is built from `eval_params` by the same three calls that build this arm's, so
    the layout is the one `test_every_arm_shares_the_layout_at_a_given_condition_and_seed` asserts is
    shared. The per-genome rows are stored as well as the mean, because S6 resamples the observed null.
    """
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, relax = resources.place_sources(jax.random.key(seed), ob, p)
    static = E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )
    template = net.init_params(jax.random.key(seed), p)
    unflatten = genome.unflattener(template)
    pop = loop.initial_population(jax.random.key(seed + 1), template, p)
    step = loop.make_generation_step(unflatten, static, p)
    acc_pass = make_accumulator_pass(unflatten, static, p)
    key = jax.random.key(seed + 2)

    n_gen = N_GENERATIONS if n_generations is None else n_generations
    fit_mean = np.zeros(n_gen, np.float32)
    fit_max = np.zeros(n_gen, np.float32)
    gens, accs = [], []
    for gen in range(n_gen):
        t_gen = time.perf_counter()
        key, kg, ka = jax.random.split(key, 3)
        pop, fit = step(kg, pop)
        f = np.asarray(fit)
        fit_mean[gen], fit_max[gen] = f.mean(), f.max()
        acc_row = None
        if gen % SAMPLE_EVERY == 0 or gen == n_gen - 1:
            gens.append(gen)
            acc_row = acc_pass(ka, pop)
            accs.append(acc_row)
        if progress is not None:
            progress.generation(gen, time.perf_counter() - t_gen, f, acc_row)
    if progress is not None:
        progress.loop_end()
    out = {
        "fit_mean": fit_mean,
        "fit_max": fit_max,
        "gens": np.array(gens, np.int32),
        "acc": np.stack(accs),
        "relax": int(relax),
        # **The final population is kept.** 71.5 MB per seed at 69,814 parameters, which is what makes
        # a final-generation quantity re-evaluable under a changed accumulator set without re-evolving,
        # and what the naive_eval pass consumes. D-059.
        "final_pop": np.asarray(pop, np.float32),
    }
    if eval_params is not None:
        ob_e = grid.make_obstacle_grid(
            jax.random.key(seed), eval_params.h, eval_params.w, eval_params.obstacle_density
        )
        fs_e, ts_e, _ = resources.place_sources(jax.random.key(seed), ob_e, eval_params)
        static_e = E.StaticEnv(
            obstacles=ob_e,
            vis_table=build_vis_table(ob_e, required_radius(eval_params)),
            food_src=fs_e,
            target_src=ts_e,
        )
        # Same layout by construction; asserted rather than trusted, because a silent divergence here
        # would evaluate the null in a different world from the arm it is compared against.
        assert bool(jnp.array_equal(ob_e, static.obstacles)), "naive_eval layout differs from the arm's"
        key, ke = jax.random.split(key)
        per_genome = make_accumulator_pass(unflatten, static_e, eval_params).all(ke, pop)
        out["naive_acc_per_genome"] = per_genome.astype(np.float32)
        out["naive_acc"] = per_genome.mean(axis=0).astype(np.float32)
    return out


#: The schema every checkpoint must carry, derived from the accumulator set rather than transcribed, so
#: it cannot drift from it. `Accumulators._fields` plus the three tier-one outcome columns.
EXPECTED_SCHEMA = tuple(ACC_FIELDS) + tuple(OUTCOME_FIELDS)


def assert_checkpoint_schema(path: pathlib.Path) -> None:
    """**The resume guard.** A completed condition is skipped on existence; this asserts it is skippable.

    **The archive is not a guard, which is the whole reason this exists.** Six checkpoints written before
    D-046 and D-048 stored 24 columns against a current 31, and the resume path skipped them on existence
    alone, so they would have entered the final dataset permanently lacking every near-conditioned and away
    accumulator. Archiving those six fixed six instances. **The next schema change recreates the class**,
    and D-052's own logic, gate on states rather than on instances, applies here.

    Raises rather than warns. A warning printed at hour three of a five-day job is not read.
    """
    with np.load(path, allow_pickle=True) as d:
        if "acc_fields" not in d:
            raise SystemExit(
                f"{path.name} carries no `acc_fields` and predates the schema guard. Its column set "
                f"cannot be checked, so it cannot be resumed. Archive it and re-run the condition."
            )
        stored = tuple(str(x) for x in d["acc_fields"])
        keys = set(d.files)
        arm = str(d["arm"]) if "arm" in d else None
    if stored != EXPECTED_SCHEMA:
        missing = [f for f in EXPECTED_SCHEMA if f not in stored]
        extra = [f for f in stored if f not in EXPECTED_SCHEMA]
        raise SystemExit(
            f"SCHEMA MISMATCH in {path.name}: it stores {len(stored)} columns, the current schema has "
            f"{len(EXPECTED_SCHEMA)}.\n"
            f"  missing from the checkpoint: {missing}\n"
            f"  present but no longer expected: {extra}\n"
            "Resuming would carry a condition into the final dataset lacking fields the analysis reads. "
            "Archive the stale checkpoints and re-run those conditions. See D-053."
        )
    # D-059: a checkpoint without the final population cannot produce the null and cannot be
    # re-evaluated, and a null-arm checkpoint without its naive_eval pass IS the missing null.
    if "final_pop" not in keys:
        raise SystemExit(
            f"{path.name} carries no `final_pop`. It predates D-059, so the null cannot be produced from it "
            "and no final-generation quantity can be re-evaluated. Archive it and re-run the condition."
        )
    if arm == "null" and "naive_acc" not in keys:
        raise SystemExit(
            f"{path.name} is a null-arm checkpoint with no `naive_acc`. The experiment's null is missing "
            "from it. Archive it and re-run the condition."
        )


_PARTIAL_KEYS = ("fit_mean", "fit_max", "gens", "acc", "final_pop", "naive_acc", "naive_acc_per_genome")


def _save_partial(path: pathlib.Path, arm: str, r: dict) -> None:
    """One seed, written atomically with the schema so the guard can read it."""
    tmp = path.with_name("." + path.name)
    np.savez(
        tmp,
        arm=np.array(arm),
        acc_fields=np.array(ACC_FIELDS + OUTCOME_FIELDS),
        relax=np.int32(r["relax"]),
        **{k: r[k] for k in _PARTIAL_KEYS if k in r},
    )
    tmp.replace(path)


def _load_partial(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        r = {k: np.asarray(d[k]) for k in _PARTIAL_KEYS if k in d.files}
        r["relax"] = int(d["relax"])
    return r


def condition_tag(arm: str, axis: str, level) -> str:
    return f"{arm}__{axis}__{str(level).replace(' ', '').replace(',', '-')}"


def verify_written_checkpoint(path: pathlib.Path) -> None:
    """Read back what was just written. A checkpoint is trusted on resume, so it is checked once, here.

    The schema guard is the same one the resume path runs. Finiteness is checked on the arrays every analysis
    reads first: a NaN there is a defect, and a defect written into a checkpoint that resume then skips would
    reach the final dataset without anything noticing.
    """
    assert_checkpoint_schema(path)
    with np.load(path, allow_pickle=True) as d:
        for k in ("fit_mean", "fit_max", "final_mean", "sample_acc"):
            if not np.all(np.isfinite(d[k])):
                raise sm.FatalStop(
                    f"{path.name}: `{k}` holds non-finite values. The checkpoint is kept for "
                    "inspection; the run stops because the condition would repeat the result."
                )


def run_condition(
    arm: str,
    axis: str,
    level,
    out_dir: pathlib.Path,
    *,
    mon=None,
    index: int = 0,
    n_seeds: int | None = None,
    n_generations: int | None = None,
) -> dict | None:
    """One condition, all seeds. Returns None if it was already complete, which is what makes this resumable.

    The seed and generation counts are read from the module at CALL time unless given, so the sweep always
    runs `N_SEEDS` by `N_GENERATIONS` and the tests can still shrink them. Keyword arguments reach `one_seed`
    only when set, so a stand-in with the original signature still works.
    """
    n_seeds = N_SEEDS if n_seeds is None else n_seeds
    n_gen = N_GENERATIONS if n_generations is None else n_generations
    extra = {}
    if mon is not None:
        extra["progress"] = mon
    if n_generations is not None:
        extra["n_generations"] = n_gen
    tag = condition_tag(arm, axis, level)
    path = out_dir / f"{tag}.npz"
    if path.exists():
        assert_checkpoint_schema(path)
        return None

    p = params_for(arm, axis, level)
    # The null arm's final populations are evaluated in the predator world at the SAME axis level, which is
    # `naive_eval`'s declared configuration with this condition's one replacement applied. D-059.
    eval_p = None
    if arm == "null":
        base = BY_NAME["naive_eval"].params
        eval_p = base if axis == "baseline" else dataclasses.replace(base, **{axis: level})
        check_invariants(eval_p)
    t0 = time.time()
    # **Per-seed partials.** A condition is ten seeds and about 75 minutes at the sustained rate, and the
    # host has crashed mid-tier twice. A seed that finished is written at once and reused on resume, so a
    # crash costs at most one seed. Partials carry the schema and pass the same guard as a condition.
    part_dir = out_dir / "partial"
    part_dir.mkdir(parents=True, exist_ok=True)
    if mon is not None:
        done_before = sum((part_dir / f"{tag}.seed{s}.npz").exists() for s in range(n_seeds))
        mon.condition_start(index, arm, tag, done_before)
    seeds = []
    for s in range(n_seeds):
        pp = part_dir / f"{tag}.seed{s}.npz"
        if pp.exists():
            assert_checkpoint_schema(pp)
            seeds.append(_load_partial(pp))
            if mon is not None:
                mon.seed_skipped(s)
            continue
        if mon is not None:
            mon.seed_start(s)
        r = one_seed(p, s, eval_params=eval_p, **extra)
        _save_partial(pp, arm, r)
        seeds.append(r)
        # After the partial is on disk, so a stop raised by a health check never costs the finished seed.
        if mon is not None:
            mon.seed_end(s, float(r["fit_mean"][-1]), float(r["fit_max"][-1]))

    finals = np.array([s["fit_mean"][-1] for s in seeds], np.float64)
    acc = np.stack([s["acc"] for s in seeds])  # (seed, sample, field)
    wall = time.time() - t0

    # **Written atomically, and the FIRST version of this was broken in a way worth recording.**
    #
    # A checkpoint half-written by a kill is worse than none, because the resume logic skips it on existence
    # and the sweep would carry a truncated condition to the end. So the write goes to a temporary name and
    # is renamed, which is atomic within a filesystem.
    #
    # The temporary name was `path.with_suffix(".npz.partial")`. **`np.savez` APPENDS `.npz` to any filename
    # that does not already end in it**, so it wrote `...npz.partial.npz` while the rename looked for
    # `...npz.partial`, and the very first condition died with FileNotFoundError. The mechanism added to
    # guarantee checkpoints produced none.
    #
    # A leading dot fixes both halves: the name still ends in `.npz` so numpy writes exactly where it is
    # told, and a hidden file is not matched by a `*.npz` glob, so a stray temporary from a kill cannot be
    # miscounted as a completed condition.
    tmp = path.with_name("." + path.name)
    np.savez(
        tmp,
        arm=np.array(arm),
        axis=np.array(axis),
        level=np.array(str(level)),
        config_hash=np.array(loop.config_hash(p)),
        n_seeds=np.int32(n_seeds),
        n_generations=np.int32(n_gen),
        fit_mean=np.stack([s["fit_mean"] for s in seeds]),
        fit_max=np.stack([s["fit_max"] for s in seeds]),
        final_mean=finals,
        final_sem=np.float64(finals.std(ddof=1) / np.sqrt(n_seeds)),
        sample_gens=seeds[0]["gens"],
        sample_acc=acc,
        acc_fields=np.array(ACC_FIELDS + OUTCOME_FIELDS),
        relax_levels=np.array([s["relax"] for s in seeds], np.int32),
        wall_seconds=np.float32(wall),
        final_pop=np.stack([s["final_pop"] for s in seeds]),
        **(
            {
                "naive_acc": np.stack([s["naive_acc"] for s in seeds]),
                "naive_acc_per_genome": np.stack([s["naive_acc_per_genome"] for s in seeds]),
            }
            if eval_p is not None
            else {}
        ),
    )
    tmp.replace(path)
    # Verified BEFORE the partials go, so a bad checkpoint leaves its seeds on disk to inspect.
    verify_written_checkpoint(path)
    for s in range(n_seeds):
        (part_dir / f"{tag}.seed{s}.npz").unlink(missing_ok=True)

    # DEV-029's class, outside the loop it was found in.
    jax.clear_caches()
    gc.collect()
    res = {
        "tag": tag,
        "wall": wall,
        "final_mean": float(finals.mean()),
        "sem": float(finals.std(ddof=1) / np.sqrt(n_seeds)),
    }
    if mon is not None:
        mon.condition_end(tag, wall, res["final_mean"], res["sem"])
    return res


def assert_commit_verified(sha: str) -> None:
    """**Refuse to start if the commit has no recorded full-differential pass.**

    The binding asserts this AFTER the fact, which is too late for a job measured in days: every condition
    would register against an unverified commit and the whole sweep would need re-registering. Checked as a
    precondition instead.
    """
    text = pathlib.Path("differential-verified.md").read_text(encoding="utf-8")
    body = text.split("## Verifications", 1)
    assert len(body) == 2, "differential-verified.md has no Verifications section"
    rows = {
        m.group(1): m.group(3)
        for m in re.finditer(r"^\|\s*([0-9a-f]{7,40})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", body[1], re.M)
    }
    hit = next((r for c, r in rows.items() if c.startswith(sha) or sha.startswith(c)), None)
    assert hit is not None, (
        f"commit {sha} has no row in differential-verified.md. Run `uv run pytest tests/differential -q` at "
        "that commit and record the result BEFORE starting a sweep, or every condition registers against an "
        "unverified commit."
    )
    assert hit.lower().startswith("pass"), f"commit {sha} is recorded as {hit!r}, not a pass"


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="sweep.py",
        description="The M4 characterisation sweep. Resumable: run the same command again to continue.",
    )
    ap.add_argument("out_dir", type=pathlib.Path, help="where checkpoints, logs and status.json go")
    ap.add_argument("--verified-commit", help="a commit with a recorded full-differential pass")
    ap.add_argument("--arms", default="treatment,null,third", help="comma-separated, default all three")
    ap.add_argument(
        "--limit",
        type=int,
        help="only the first N conditions of the plan. Used to run each arm's baseline alone, into the sweep's "
        "own directory, so the sweep later skips it on resume rather than repeating it",
    )
    ap.add_argument(
        "--plan-only", action="store_true", help="print the plan and check every condition, run nothing"
    )
    ap.add_argument(
        "--progress",
        choices=sm.LEVELS,
        default="normal",
        help="quiet: a line per condition and every warning. normal (default): also a line per seed, a line "
        f"every {SAMPLE_EVERY} generations and a live status line. verbose: also a line every generation",
    )
    ap.add_argument(
        "--log-file", type=pathlib.Path, help="default <out_dir>/sweep.log; appended, never truncated"
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=f"the whole runner path at {SMOKE_SEEDS} seeds x {SMOKE_GENERATIONS} generations on two conditions, "
        "into a scratch directory. Never into the sweep's directory",
    )
    return ap.parse_args(argv)


def guard_smoke(out_dir: pathlib.Path, smoke: bool) -> None:
    """Keep smoke output and sweep output apart in both directions. D-062."""
    marker = out_dir / SMOKE_MARKER
    if smoke:
        if out_dir.resolve().name == "sweep" and out_dir.resolve().parent.name == "runs":
            raise sm.FatalStop("--smoke refuses runs/sweep. Give it a scratch directory such as runs/smoke.")
        if out_dir.exists() and any(out_dir.glob("*.npz")) and not marker.exists():
            raise sm.FatalStop(f"--smoke refuses {out_dir}: it holds checkpoints that are not smoke output.")
        out_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text("smoke-test output, never sweep data\n", encoding="utf-8")
    elif marker.exists():
        raise sm.FatalStop(
            f"{out_dir} holds smoke-test output ({SMOKE_MARKER}). The sweep will not write there."
        )


def _device_mb() -> float:
    try:
        return jax.devices()[0].memory_stats()["bytes_in_use"] / 2**20
    except Exception:  # noqa: BLE001  a reading, never a reason to stop
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = args.out_dir
    mon = None
    lock = None
    try:
        print(src.assert_runtime_took_effect(), flush=True)
        # D-054: this is a job measured in DAYS, and the host bugchecked on 2026-08-31 two minutes after a
        # ten-minute tier finished. Refusing here rather than warning, because a run started with no headroom
        # is exactly the case that entry exists for.
        try:
            print(src.memory_headroom(min_available_gb=4.0, raise_on_low=True), flush=True)
        except SystemExit as exc:
            raise sm.FatalStop(f"{exc} Close other programs, then run the same command again.") from None

        arms = args.arms.split(",")
        n_seeds, n_gen = (SMOKE_SEEDS, SMOKE_GENERATIONS) if args.smoke else (N_SEEDS, N_GENERATIONS)
        plan = list(SMOKE_PLAN) if args.smoke else [(a, ax, lv) for a in arms for ax, lv in conditions(a)]
        if args.limit is not None:
            plan = plan[: args.limit]
        print(f"\n{'arm':<11s} {'conditions':>11s}", flush=True)
        for a in sorted({a for a, _ax, _lv in plan}, key=list(ARMS).index):
            print(f"{a:<11s} {sum(1 for x in plan if x[0] == a):11d}", flush=True)
        print(
            f"{'TOTAL':<11s} {len(plan):11d} conditions, {len(plan) * n_seeds} evolutionary runs", flush=True
        )
        if not args.smoke:
            print(
                f"for scale: {len(plan) * n_seeds * n_gen * 1.496 / 3600:.1f} hours on the RTX 4060 laptop at "
                "D-057's 1.496 s per generation. This machine's own rate is measured as the run goes.",
                flush=True,
            )

        print("\npre-flight: every condition against check_invariants", flush=True)
        for a, ax, lv in plan:
            params_for(a, ax, lv)
        print(f"  all {len(plan)} conditions satisfy check_invariants", flush=True)
        if args.plan_only:
            print("\n--plan-only, nothing run", flush=True)
            return sm.EXIT_DONE

        if not args.verified_commit:
            raise sm.FatalStop("pass --verified-commit <sha>; the sweep refuses to start without it")
        sha = args.verified_commit
        assert_commit_verified(sha)
        print(f"  commit {sha} has a recorded full-differential pass\n", flush=True)

        guard_smoke(out_dir, args.smoke)
        lock = sm.acquire_run_lock(out_dir)

        # What is left, per condition, from the checkpoints already on disk. The resume logic is unchanged;
        # this only tells the monitor how much work remains.
        status_plan = []
        for a, ax, lv in plan:
            tag = condition_tag(a, ax, lv)
            if (out_dir / f"{tag}.npz").exists():
                assert_checkpoint_schema(out_dir / f"{tag}.npz")
                left = 0
            else:
                left = n_seeds - sum(
                    (out_dir / "partial" / f"{tag}.seed{s}.npz").exists() for s in range(n_seeds)
                )
            status_plan.append((a, tag, left))

        log_path = args.log_file or out_dir / "sweep.log"
        mon = sm.Monitor(
            args.progress,
            out_dir,
            status_plan,
            n_seeds,
            n_gen,
            SAMPLE_EVERY,
            log_path=log_path,
            device_mb=_device_mb,
        )
        jax.monitoring.register_event_duration_secs_listener(
            lambda event, duration, **kw: mon.on_trace() if event == TRACE_EVENT else None
        )
        sm.start_watchdog(mon)
        g = sm.gpu_snapshot()
        gpu_desc = (
            f"{g['name']}, driver {g['driver']}, {g['mem_total_mb'] / 1024:.0f} GB"
            if g.get("available")
            else "nvidia-smi unreadable"
        )
        mon.run_start(
            [
                "=" * 100,
                f"sweep started{' (SMOKE TEST, not sweep data)' if args.smoke else ''}  pid {__import__('os').getpid()}",
                f"  gpu       {gpu_desc}  |  jax {jax.__version__} on {jax.devices()[0].device_kind}",
                f"  commit    {sha}   output {out_dir.resolve()}",
                f"  log       {pathlib.Path(log_path).resolve()}   status {(out_dir / 'status.json').resolve()}",
                f"  progress  {args.progress}   disk free {sm.disk_free_gb(out_dir):.0f} GB   host memory available "
                f"{sm.host_available_gb():.1f} GB",
            ]
        )

        manifest = out_dir / "manifest.jsonl"
        t0 = time.time()
        done = skipped = 0
        for i, (a, ax, lv) in enumerate(plan, 1):
            kw = {"n_seeds": n_seeds, "n_generations": n_gen} if args.smoke else {}
            res = run_condition(a, ax, lv, out_dir, mon=mon, index=i, **kw)
            if res is None:
                skipped += 1
                continue
            done += 1
            with manifest.open("a") as f:
                f.write(json.dumps({**res, "arm": a, "axis": ax, "level": str(lv), "sha": sha}) + "\n")

        mon.line(f"{done} conditions run, {skipped} already complete, {(time.time() - t0) / 3600:.1f} hours")
        mon.line(f"manifest -> {manifest}")
        if not args.smoke:
            mon.line(
                "Register these in runs.md from the manifest. The sweep does not write the registry itself:"
            )
            mon.line(
                "runs.md is append-only and hand-audited, and a job that appends 109 rows unattended would"
            )
            mon.line(
                "make a mangled row indistinguishable from a real one. The manifest is the input to that."
            )
        mon.run_end(sm.EXIT_DONE)
        return sm.EXIT_DONE
    except BaseException as exc:  # noqa: BLE001  every ending is reported with a code the wrapper can act on
        code = sm.exit_code_for(exc)
        reason = getattr(exc, "message", None) or str(exc) or type(exc).__name__
        if code == sm.EXIT_CRASH:
            traceback.print_exc()
        if mon is not None:
            mon.run_end(code, reason)
        else:
            print(f"\nstopped before the run began, exit code {code}: {reason}", flush=True)
        return code
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
