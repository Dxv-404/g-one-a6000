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
import os
import pathlib
import re
import sys
import time
import traceback
import zipfile

import jax
import jax.numpy as jnp
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs import frozen_world
from configs.default import DEFAULT, check_invariants, make_relax_ladder
from configs.run_configs import BY_NAME
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius
from src.evo import genome, loop
from src.measure import engagement

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sweep_monitor as sm  # noqa: E402  host-side reporting only, D-062

#: **Twenty, by owner ruling on Q-051, 2026-10-14, D-070.** The A6000's own baselines measured the freezing
#: descriptor's cross-arm seed correlation at -0.62, where the laptop's runs 32 and 33 had +0.55, so pairing
#: ADDS noise here: power to detect a one-null-sd shift is 0.616 at ten seeds, 0.785 at fifteen and 0.858 at
#: twenty under the paired model, 0.676, 0.848 and 0.935 under the independent one. Twenty is the smallest
#: count that clears S6's 0.80 under both. The earlier ruling, kept below as the record of what was believed:
#:
#: Ten, and the precision is REPORTED rather than assumed. Owner's ruling: doubling to twenty doubles the
#: whole plan for a factor of 1.41 in precision, and the primary claim routes through tier-one outcomes at
#: 8,192 prey per generation rather than through cross-seed means. The standard error of each condition's
#: mean is written into its npz so a reader gets the precision with the number.
#:
#: **If a condition lands near a decision boundary, add seeds for THAT condition** rather than re-running the
#: sweep. The freeze work established why: a required-n computed from a pilot chases both a moving effect and
#: a moving bar, so a global re-run at a larger n is the expensive way to answer a local question.
N_SEEDS = 20

#: **Seeds 20 to 39 for the long run (D-084)**, disjoint from Phase A (0 to 19, runs 42 to 44), the S4
#: calibration (100 to 102) and session 1's test seeds (200 upward). A seed's VALUE drives its layout, initial
#: population and key chain; partial checkpoints are named by value and each condition stores its `seeds`.
SEED_OFFSET = 20


def seed_list(n_seeds: int | None = None) -> list[int]:
    n = N_SEEDS if n_seeds is None else n_seeds
    return list(range(SEED_OFFSET, SEED_OFFSET + n))


#: What one stored accumulator row pools. Each per-genome row is the mean over one episode's `k` prey
#: (`make_accumulator_pass`), and a population row is the mean of `p_pop` of those, so a row stands for
#: `p_pop * k` prey-episodes: 2048 at the defaults.
POOLED_PREY_EPISODES = DEFAULT.p_pop * DEFAULT.k


def pooled(mean_row: np.ndarray) -> np.ndarray:
    """A population-mean accumulator row as the pooled totals it averages. D-071, owner ruling 2026-10-14.

    Every descriptor is a ratio of two accumulators, so scaling both by the same count leaves every value
    unchanged; what changes is the denominator floor, which then applies to the ticks the seed actually
    observed rather than to a per-prey average. Applied to a mean row, a floor of 1.0 demanded one perceived
    tick per prey-episode, and the A6000's null perceives the predator on 0.13 to 0.80 per prey-episode, about
    270 to 1630 ticks per seed, so distancing read undefined on every seed while the data were plentiful.
    """
    return np.asarray(mean_row, np.float64) * POOLED_PREY_EPISODES


#: **1,500, fixed in advance, by owner ruling (D-073, confirmed at D-081).** A budget, not a plateau: run 51
#: could not say whether 1,500 reaches 67 or 97 percent of the eventual gain, and every run's full fitness
#: trajectory is reported (D-040). The scripted control never runs longer (D-086). Phase A ran 300.
N_GENERATIONS = 1500
SAMPLE_EVERY = 25

#: **S4 measured IN the run (D-083, checklist 3.3)**, so it never again needs a replay (Q-053). After each of
#: these generation counts, every genome of an arm whose world has a predator is measured over
#: `S4_EPISODES` episodes by `src.measure.engagement.measure`, the code the calibration uses, with keys that
#: never touch the evolution's chain. Seven points rather than all 61 samples, and seven points still show
#: whether learnability is SUSTAINED, which the ignition criterion asks. **64 episodes since Q-061** (run 58,
#: D-092), the calibration's budget, so the long run's S4 is the rule the world was chosen by. Measured on the
#: laptop at about 30 s a measurement, roughly 20 generations, so the seven points add about 9 percent of a
#: condition, against about 3 at 16.
S4_AFTER = (100, 300, 500, 750, 1000, 1250, 1500)
S4_EPISODES = 64

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


#: Axes whose frozen value may sit between levels (Q-060 proposed it for `cue_dependence` alone).
REPLACEABLE = frozenset({"cue_dependence"})


def axis_levels(axis: str, world: dict | None = None) -> tuple:
    """An axis's levels with the frozen world's value on it (Q-060). ONLY for the axes in `REPLACEABLE`: a frozen
    value there that is not already a level REPLACES the nearest interior level, so the axis keeps both ends, five
    levels and the map its shape. Every other axis is returned unchanged, so a frozen value off its levels is
    refused by `require_frozen_world`, as before Q-060."""
    world = frozen_world.WORLD if world is None else world
    levels = LEVELS[axis]
    if axis not in REPLACEABLE or not world or axis not in world:
        return levels
    v = world[axis]
    if v in levels or not (min(levels) < v < max(levels)):
        return levels
    interior = levels[1:-1]
    nearest = min(interior, key=lambda lv: (abs(lv - v), lv))
    return tuple(sorted(v if lv == nearest else lv for lv in levels))


def conditions(arm: str) -> list[tuple[str, object]]:
    """(axis, level) pairs for one arm, with `("baseline", None)` first. Derived, never transcribed.

    **Never a predator radius above the prey's (D-074, the refusal list).** Derived from the prey radius of the long
    run's own world, not transcribed. D-084 states the long run omits these conditions; until 2026-11-09 this
    function did not, so the plan held treatment r_pred 5 and 6 and third r_pred 6 (DEV-043)."""
    out: list[tuple[str, object]] = [("baseline", None)]
    prey_r = arm_base("scripted").r
    for axis in LEVELS:
        levels = axis_levels(axis)
        if arm == "null" and axis in PREDATOR_ONLY:
            continue
        # Around the long run's own baseline: if the frozen world moved this axis, its old default becomes a
        # level and the frozen value is the baseline, so the map never repeats its own baseline (D-083).
        picked = [lv for lv in levels if lv != getattr(arm_base("scripted"), axis)]
        if arm == "third":
            picked = [levels[0], levels[-1]]  # both ends only
        if axis == "r_pred":
            picked = [lv for lv in picked if lv <= prey_r]  # D-074: dropped, never replaced by another level
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


def arm_base(run_config: str):
    """A run configuration with the long run's frozen world applied (configs/frozen_world.py, D-083). Each arm
    keeps the field that defines it, because `apply` changes a field only where the arm agrees with the
    scripted control."""
    return frozen_world.apply(BY_NAME[run_config].params, BY_NAME["scripted"].params)


def params_for(arm: str, axis: str, level) -> object:
    base = arm_base(ARMS[arm])
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
    n_gen_planned = N_GENERATIONS if n_generations is None else n_generations
    eng_pass = engagement.make_engagement_pass(unflatten, p) if p.n_pred > 0 else None
    s4_after = [a for a in S4_AFTER if a <= n_gen_planned] if eng_pass is not None else []
    if s4_after:
        # Compiled here, before the loop and before the monitor arms its recompile guard, by one discarded
        # measurement of the initial population. Its keys are measurement keys, so the run cannot see it.
        engagement.measure(eng_pass, pop, static, seed, 0, 1)
    s4_eng, s4_prey, final_rows = [], [], None

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
            if gen == n_gen - 1:
                # Per-genome rows at the final generation, for every arm (checklist 3.4). `acc_pass` IS the mean
                # of these rows, so the stored sample is bit-identical to what it was.
                final_rows = acc_pass.all(ka, pop)
                acc_row = final_rows.mean(axis=0)
            else:
                acc_row = acc_pass(ka, pop)
            accs.append(acc_row)
        wall = (
            time.perf_counter() - t_gen
        )  # before the S4 pass, so a measurement never reads as a slow generation
        if gen + 1 in s4_after:
            eng, prey = engagement.measure(eng_pass, pop, static, seed, gen + 1, S4_EPISODES)
            s4_eng.append(eng)
            s4_prey.append(prey.astype(np.int32).sum(axis=-1))  # (G, E, 6): per genome and episode, over prey
        if progress is not None:
            progress.generation(gen, wall, f, acc_row)
    if progress is not None:
        progress.loop_end()
    out = {
        "fit_mean": fit_mean,
        "fit_max": fit_max,
        "gens": np.array(gens, np.int32),
        "acc": np.stack(accs),
        "relax": int(relax),
        # **The final population is kept.** 71.5 MB per seed at Phase A's 69,814 parameters (D-085 adds 3,888), which is what makes
        # a final-generation quantity re-evaluable under a changed accumulator set without re-evolving,
        # and what the naive_eval pass consumes. D-059.
        "final_pop": np.asarray(pop, np.float32),
        "final_rows": np.asarray(final_rows, np.float32),
        "s4_after": np.array(s4_after[: len(s4_eng)], np.int32),
        "s4_eng": np.stack(s4_eng)
        if s4_eng
        else np.zeros((0, p.p_pop, S4_EPISODES, len(engagement.ENGAGEMENT_FIELDS)), np.int32),
        "s4_prey": np.stack(s4_prey)
        if s4_prey
        else np.zeros((0, p.p_pop, S4_EPISODES, len(engagement.PREY_FIELDS)), np.int32),
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
    if path.parent.name != "partial":
        missing_long = [k for k in LONGRUN_KEYS if k not in keys]
        if missing_long:
            raise SystemExit(
                f"{path.name} carries no {missing_long}. It predates the long run's runner (in-run S4, per-genome "
                "final rows, the seed values, the frozen world), so the analysis cannot read it. Archive it and "
                "re-run the condition."
            )
    elif any(k not in keys for k in ("final_rows", "s4_after", "s4_eng", "s4_prey")):
        raise SystemExit(
            f"{path.name} is a partial seed without the in-run S4 or the final rows. It predates the long run's "
            "runner. Delete it; the seed will be recomputed."
        )
    if arm == "null" and "naive_acc" not in keys:
        raise SystemExit(
            f"{path.name} is a null-arm checkpoint with no `naive_acc`. The experiment's null is missing "
            "from it. Archive it and re-run the condition."
        )


def assert_same_code(path: pathlib.Path, code_commit: str | None) -> None:
    """**The code-version guard, D-066. Schema equality is not code equality.**

    D-064 changed selection without changing a single column, so a baseline evolved under the old elitism
    would have passed `assert_checkpoint_schema` and been reused by a sweep running the new one: two versions
    of the experiment in one dataset, with nothing in any file to say so. A checkpoint is therefore reused only
    if the code that wrote it is the code now running, named by the VERIFIED commit, which the derived
    repository ships as the newest verified commit byte-identical to its runnable code. A checkpoint from
    before this guard carries no name, so it is refused rather than trusted.

    `code_commit` None skips the check, for the bookkeeping tests that drive `run_condition` directly; the
    sweep's `main` always passes it.
    """
    if code_commit is None:
        return
    with np.load(path, allow_pickle=True) as d:
        stored = str(d["code_commit"]) if "code_commit" in d.files else ""
    if stored != code_commit:
        raise SystemExit(
            f"CODE MISMATCH in {path.name}: it was written by "
            f"{('code ' + stored) if stored else 'code from before the code-version guard'}, and the code "
            f"running now is {code_commit}. Reusing it would mix two versions of the experiment in one "
            "dataset. Delete every output of the old code (the whole output folder, and results/BASELINES_DONE) "
            "and run again from the baselines. D-066."
        )


_PARTIAL_KEYS = (
    "fit_mean",
    "fit_max",
    "gens",
    "acc",
    "final_pop",
    "naive_acc",
    "naive_acc_per_genome",
    "final_rows",
    "s4_after",
    "s4_eng",
    "s4_prey",
)
#: Every condition checkpoint carries these since the long run's runner (checklist 3.3, 3.4); one without them
#: predates it and is refused, as a checkpoint without `final_pop` is.
LONGRUN_KEYS = ("final_rows", "s4_after", "s4_eng", "s4_prey", "seeds", "world")


def _save_partial(path: pathlib.Path, arm: str, r: dict, code_commit: str | None = None) -> None:
    """One seed, written atomically with the schema and the code version so both guards can read it."""
    tmp = path.with_name("." + path.name)
    np.savez(
        tmp,
        arm=np.array(arm),
        code_commit=np.array(code_commit or ""),
        acc_fields=np.array(ACC_FIELDS + OUTCOME_FIELDS),
        relax=np.int32(r["relax"]),
        **{k: r[k] for k in _PARTIAL_KEYS if k in r},
    )
    durable_replace(tmp, path)


def _load_partial(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        r = {k: np.asarray(d[k]) for k in _PARTIAL_KEYS if k in d.files}
        r["relax"] = int(d["relax"])
    return r


def durable_replace(tmp: pathlib.Path, path: pathlib.Path) -> None:
    """Rename a finished temporary file over `path` so that a MACHINE crash cannot leave a named but empty file.

    The rename alone protects against a killed process, not a crashed host: after the Windows bugcheck of
    2026-11-07 a calibration file survived as its name with 0 bytes, because the rename reached the disk and
    the contents had not (DEV-041). So the contents are forced to disk first, then the rename, then the
    directory entry."""
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def readable(path: pathlib.Path) -> bool:
    """True if `path` is a complete npz: non-empty, a valid zip, every member intact. Says nothing about its
    schema, which `assert_checkpoint_schema` checks and which stops the run when wrong."""
    try:
        if path.stat().st_size == 0:
            return False
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except (OSError, zipfile.BadZipFile, EOFError):
        return False


def set_aside(path: pathlib.Path) -> pathlib.Path:
    """Move an unreadable checkpoint out of the way, kept for inspection, so the work is redone rather than the
    run crashing on it at every restart. The move is reported by the caller."""
    dest = path.with_name(path.name + ".unreadable")
    n = 1
    while dest.exists():
        dest = path.with_name(f"{path.name}.unreadable{n}")
        n += 1
    path.replace(dest)
    return dest


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
        # The planned points that fit this condition's budget: all of S4_AFTER in the long run, none in a
        # three-generation smoke test. A condition missing any of them stops the run.
        if str(d["arm"]) != "null":
            want = tuple(a for a in S4_AFTER if a <= int(d["n_generations"]))
            got = tuple(int(x) for x in d["s4_after"])
            if got != want:
                raise sm.FatalStop(
                    f"{path.name}: S4 was measured after {got}, not the planned {want}. The condition would "
                    "enter the dataset without the learnability trajectory the plan promises."
                )
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
    code_commit: str | None = None,
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
    if path.exists() and not readable(path):
        moved = set_aside(path)
        print(
            f"  {path.name} is unreadable (a crash during its write); set aside as {moved.name}, redoing it",
            flush=True,
        )
    if path.exists():
        assert_checkpoint_schema(path)
        assert_same_code(path, code_commit)
        return None

    p = params_for(arm, axis, level)
    # The null arm's final populations are evaluated in the predator world at the SAME axis level, which is
    # `naive_eval`'s declared configuration with this condition's one replacement applied. D-059.
    eval_p = None
    if arm == "null":
        base = arm_base("naive_eval")
        eval_p = base if axis == "baseline" else dataclasses.replace(base, **{axis: level})
        check_invariants(eval_p)
    t0 = time.time()
    # **Per-seed partials.** A condition is ten seeds and about 75 minutes at the sustained rate, and the
    # host has crashed mid-tier twice. A seed that finished is written at once and reused on resume, so a
    # crash costs at most one seed. Partials carry the schema and pass the same guard as a condition.
    part_dir = out_dir / "partial"
    part_dir.mkdir(parents=True, exist_ok=True)
    if mon is not None:
        done_before = sum((part_dir / f"{tag}.seed{s}.npz").exists() for s in seed_list(n_seeds))
        mon.condition_start(index, arm, tag, done_before)
    seeds = []
    # The monitor numbers seeds by position (1 to n) for its progress line; the files and the run use the seed's
    # VALUE, which is SEED_OFFSET plus the position.
    for i, s in enumerate(seed_list(n_seeds)):
        pp = part_dir / f"{tag}.seed{s}.npz"
        if pp.exists() and not readable(pp):
            moved = set_aside(pp)
            print(
                f"  {pp.name} is unreadable (a crash during its write); set aside as {moved.name}, redoing it",
                flush=True,
            )
        if pp.exists():
            assert_checkpoint_schema(pp)
            assert_same_code(pp, code_commit)
            seeds.append(_load_partial(pp))
            if mon is not None:
                mon.seed_skipped(i)
            continue
        if mon is not None:
            mon.seed_start(i)
        r = one_seed(p, s, eval_params=eval_p, **extra)
        _save_partial(pp, arm, r, code_commit)
        seeds.append(r)
        # After the partial is on disk, so a stop raised by a health check never costs the finished seed.
        if mon is not None:
            mon.seed_end(i, float(r["fit_mean"][-1]), float(r["fit_max"][-1]))

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
        code_commit=np.array(code_commit or ""),
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
        seeds=np.array(seed_list(n_seeds), np.int32),
        final_rows=np.stack([s["final_rows"] for s in seeds]),
        s4_after=seeds[0]["s4_after"],
        s4_eng=np.stack([s["s4_eng"] for s in seeds]),
        s4_prey=np.stack([s["s4_prey"] for s in seeds]),
        world=np.array(json.dumps(frozen_world.WORLD)),
        **(
            {
                "naive_acc": np.stack([s["naive_acc"] for s in seeds]),
                "naive_acc_per_genome": np.stack([s["naive_acc_per_genome"] for s in seeds]),
            }
            if eval_p is not None
            else {}
        ),
    )
    durable_replace(tmp, path)
    # Verified BEFORE the partials go, so a bad checkpoint leaves its seeds on disk to inspect.
    verify_written_checkpoint(path)
    for s in seed_list(n_seeds):
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
    ap.add_argument(
        "--seed-offset",
        type=int,
        help="Phase A re-runs only (Q-058): seeds from this value instead of SEED_OFFSET. Refused unless the "
        "output directory is named for Phase A and the seeds avoid the long run's",
    )
    ap.add_argument(
        "--generations",
        type=int,
        help="Phase A re-runs only (Q-058): a budget other than N_GENERATIONS, under the same refusal",
    )
    ap.add_argument(
        "--shard",
        help="I/N: run only every N-th condition of the plan starting at I, into <out_dir>/shardIofN. N processes with I = 0 to N-1 run the whole plan side by side; how many at once is measured, not assumed (session 1, 4.3)",
    )
    return ap.parse_args(argv)


def parse_shard(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    m = re.fullmatch(r"(\d+)/(\d+)", text.strip())
    if not m or not (0 <= int(m.group(1)) < int(m.group(2))):
        raise sm.FatalStop(f"--shard {text!r}: expected I/N with 0 <= I < N, for example 0/4")
    return int(m.group(1)), int(m.group(2))


def shard_plan(plan: list, shard: tuple[int, int] | None) -> list:
    """Round robin, so each shard gets every arm and a mix of costly and cheap conditions. The shards of one N
    partition the plan exactly, which `test_longrun_runner.py` asserts."""
    if shard is None:
        return plan
    i, n = shard
    return [c for j, c in enumerate(plan) if j % n == i]


def shard_dir(out_dir: pathlib.Path, shard: tuple[int, int] | None) -> pathlib.Path:
    return out_dir if shard is None else out_dir / f"shard{shard[0]}of{shard[1]}"


def guard_shards(out_dir: pathlib.Path, shard: tuple[int, int] | None) -> None:
    """Refuse to mix layouts. A resume with a different N would move conditions between shards, and a sharded and
    an unsharded run in one folder would each redo the other's conditions."""
    existing = sorted(d.name for d in out_dir.glob("shard*of*") if d.is_dir()) if out_dir.exists() else []
    if shard is None:
        if existing:
            raise sm.FatalStop(
                f"{out_dir} holds sharded output ({existing}); run it with the same --shard I/N"
            )
        return
    other = [d for d in existing if not d.endswith(f"of{shard[1]}")]
    if other:
        raise sm.FatalStop(
            f"{out_dir} holds shards of a different N ({other}). Resuming with N = {shard[1]} would move conditions "
            "between shards. Use the N it was started with."
        )
    if out_dir.exists() and any(out_dir.glob("*.npz")):
        raise sm.FatalStop(f"{out_dir} holds unsharded checkpoints; a sharded run will not write beside them")


def guard_phase_a(out_dir: pathlib.Path, seed_offset: int | None, generations: int | None) -> None:
    """A run with other seeds or another budget is a Phase A re-run (Q-058), never part of the long run. It
    must say so in its directory name, and its seeds may never be the long run's, because the canonical's
    calibration and confirmatory phases use disjoint seeds."""
    if seed_offset is None and generations is None:
        return
    if "phase-a" not in out_dir.name:
        raise sm.FatalStop(
            f"--seed-offset and --generations are for Phase A re-runs only; {out_dir} is not named for one "
            "(for example runs/phase-a-null)."
        )
    if seed_offset is not None:
        theirs = set(range(SEED_OFFSET, SEED_OFFSET + N_SEEDS))
        ours = set(range(seed_offset, seed_offset + N_SEEDS))
        if ours & theirs:
            raise sm.FatalStop(
                f"seeds {min(ours)} to {max(ours)} overlap the long run's {min(theirs)} to {max(theirs)}"
            )
    if generations is not None and generations < 1:
        raise sm.FatalStop(f"--generations {generations}: must be at least 1")


def require_frozen_world(smoke: bool) -> None:
    """D-083: the long run runs in the world the S4 calibration chose. Smoke tests may run in any world."""
    if smoke:
        return
    if frozen_world.WORLD is None:
        raise sm.FatalStop(
            "the long run's world is not frozen: configs/frozen_world.py WORLD is None. The S4 calibration "
            "(D-083) chooses it; until then this would run the Phase A world that S4 rejected (run 52)."
        )
    frozen_world.check()
    for axis, value in frozen_world.WORLD.items():
        if axis in LEVELS and value not in axis_levels(axis):
            raise sm.FatalStop(
                f"the frozen {axis} = {value} is not one of the axis levels {LEVELS[axis]}, so the axis would run "
                "five non-baseline levels instead of four and the plan would change shape."
            )


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
    global SEED_OFFSET, N_GENERATIONS
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = args.out_dir
    shard = None
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

        guard_phase_a(out_dir, args.seed_offset, args.generations)
        if args.seed_offset is not None:
            SEED_OFFSET = args.seed_offset
        if args.generations is not None:
            N_GENERATIONS = args.generations
        arms = args.arms.split(",")
        n_seeds, n_gen = (SMOKE_SEEDS, SMOKE_GENERATIONS) if args.smoke else (N_SEEDS, N_GENERATIONS)
        plan = list(SMOKE_PLAN) if args.smoke else [(a, ax, lv) for a in arms for ax, lv in conditions(a)]
        if args.limit is not None:
            plan = plan[: args.limit]
        shard = parse_shard(args.shard)
        if shard is not None:
            full = len(plan)
            plan = shard_plan(plan, shard)
            print(f"shard {shard[0]} of {shard[1]}: {len(plan)} of the plan's {full} conditions", flush=True)
        print(f"seeds {seed_list(n_seeds)[0]} to {seed_list(n_seeds)[-1]}, {n_gen} generations", flush=True)
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
        require_frozen_world(args.smoke)
        print(f"  world: {frozen_world.WORLD} from {frozen_world.SOURCE}", flush=True)

        if not args.verified_commit:
            raise sm.FatalStop("pass --verified-commit <sha>; the sweep refuses to start without it")
        sha = args.verified_commit
        assert_commit_verified(sha)
        print(f"  commit {sha} has a recorded full-differential pass\n", flush=True)

        guard_shards(out_dir, shard)
        out_dir = shard_dir(out_dir, shard)
        guard_smoke(out_dir, args.smoke)
        lock = sm.acquire_run_lock(out_dir)

        # What is left, per condition, from the checkpoints already on disk. The resume logic is unchanged;
        # this only tells the monitor how much work remains.
        status_plan = []
        for a, ax, lv in plan:
            tag = condition_tag(a, ax, lv)
            if (out_dir / f"{tag}.npz").exists():
                assert_checkpoint_schema(out_dir / f"{tag}.npz")
                assert_same_code(out_dir / f"{tag}.npz", sha)
                left = 0
            else:
                left = n_seeds - sum(
                    (out_dir / "partial" / f"{tag}.seed{s}.npz").exists() for s in seed_list(n_seeds)
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
            res = run_condition(a, ax, lv, out_dir, mon=mon, index=i, code_commit=sha, **kw)
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
