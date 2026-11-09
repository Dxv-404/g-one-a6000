#!/usr/bin/env python3
"""The S4 calibration, D-083 and Q-057. Registered as a run when it executes.

**What it decides.** Which world the long run uses. S4 failed in every configuration the Phase A sweep ran (run
52, Q-056): the median genome never meets the predator, and the genomes that do escape 93 to 97 percent of the
time. The owner ruled to recalibrate first, judged by S4 alone, over a grid fixed before any of it ran.

**The grid (Q-057).** Stage 1 moves `h_regen` over {0.25, 0.1, 0.0} and `p_wander` over {0.0, 0.5, 1.0} with
`damage` at its default 34. Stage 2, which this script refuses to run unless stage 1 is complete and nothing in it
passed, is the same nine settings at `damage` 50. Every other parameter is the treatment arm's scripted world,
unchanged. Seeds 100, 101, 102, used for nothing else.

**Per setting and seed.** Evolve for 300 generations by the sweep's own construction: the same layout, initial
population and key chain as `sweep.one_seed` (asserted bit for bit in the slow test), so a population here is the
population the sweep's runner would produce for that configuration and seed on the same device. After 100 and
after 300 generations, every genome is measured over `EPISODES` episodes (16 in stages 1 to 3, 64 since Q-061) through `replay_final`'s engagement pass and
`engagement_counts`, the code run 48 used, and S4 is computed by `src.measure.learnability.s4`, the code run 52
used. The measurement keys are derived from the seed and the generation count only, never from the evolution's
key chain, so measuring cannot perturb the run and two measurements of one population agree.

**The pass rule (D-083), applied by `setting_passes` and `choose`, not by a reader:**

    pass     coverage >= 0.5 AND median learnability of the defined genomes >= 0.16, in EVERY seed at BOTH
             generations
    choice   among passing settings, the highest median learnability after 300 generations, averaged over
             seeds; a tie goes to the setting that moves fewer parameters from the default
    stop     nothing passes: the summary says so and nothing is chosen. The grid is not widened

**Blind to behaviour, by construction.** Nothing here computes a descriptor, a tier-three statistic or a
trajectory, and `test_calibrate_s4.py` asserts that importing this module loads none of
`src.measure.descriptors`, `classifier`, `discovery`. Calibration on a proxy criterion is what keeps it from
being tuning toward a result.

Usage, from the repository root:
    uv run python scripts/measure/calibrate_s4.py runs/calibration-s4 --stage 1
    uv run python scripts/measure/calibrate_s4.py runs/calibration-s4 --stage 1 --plan-only
    uv run python scripts/measure/calibrate_s4.py runs/calibration-s4-smoke --smoke
Resumable: a finished (setting, seed) file is skipped after its layout is checked. The summary is rewritten
after every file, and reads "incomplete" until every file of the stage exists.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib.util
import json
import math
import pathlib
import subprocess
import sys
import time

import jax
import numpy as np

import src  # runtime control, before any JAX operation, as in sweep.py

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.default import DEFAULT, check_invariants  # noqa: E402
from src.agent import network as net  # noqa: E402
from src.env import env as E  # noqa: E402
from src.env import grid, resources  # noqa: E402
from src.env.perception import build_vis_table, required_radius  # noqa: E402
from src.evo import genome, loop  # noqa: E402
from src.measure import engagement, learnability  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "replay_final_cal", ROOT / "scripts" / "measure" / "replay_final.py"
)
rf = importlib.util.module_from_spec(_SPEC)
sys.modules["replay_final_cal"] = rf  # its dataclass resolves annotations through sys.modules
_SPEC.loader.exec_module(rf)
sweep = rf.sweep

#: Bumped whenever the per-file layout changes, so resume never mixes layouts.
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------- the plan, D-083 and Q-057 --
SEEDS = (100, 101, 102)
#: The re-check of the frozen choice (checklist 4.5) uses FRESH seeds. By stage 3 about forty settings have been
#: compared on seeds 100 to 102, so the best of them is partly the luckiest on those seeds; confirming it on seeds
#: it was not selected on is what separates the two (Q-060). Disjoint from every other use.
RECHECK_SEEDS = (103, 104, 105)
N_GENERATIONS = 300
MEASURE_AFTER = (100, 300)
#: D-083's measurement budget, 64 since Q-061 (run 58, D-092): at 16 a genome had about 18 decided engagements
#: against n_eng_min 30, so coverage failed as a sample size. Stages 1 to 3 (runs 53, 55, 57) and run 56 were
#: measured at STAGE_EPISODES, and the drivers that reproduce them read that constant rather than this one.
EPISODES = 64
STAGE_EPISODES = 16
H_REGEN = (0.25, 0.1, 0.0)
P_WANDER = (0.0, 0.5, 1.0)
DAMAGE = {1: DEFAULT.damage, 2: 50.0}
#: Stage 3 (Q-060, after run 56 answered B): the predator's escape rule, centred on the measured transition.
#: For a still prey adjacent to the predator, `cue_dependence` cd sets the damage per tick, 34 x (1 - cd), and how
#: fast pursuit breaks (stillness accrues cd per tick, pursuit breaks past 5): at 0.95 a freeze survives about
#: ten bouts, at 0.85 about three, at 0.75 two, at 0.6 one. Run 52 measured escape 0.82 at 1.0 and 0.16 at 0.75, so
#: the band lies inside this range and 0.95 would behave almost exactly like 1.0. Moving prey are unaffected.
CUE_DEPENDENCE = (0.6, 0.7, 0.75, 0.8, 0.85)
P_WANDER_3 = (0.5, 1.0)
STAGES = (1, 2, 3)
MOVED = ("h_regen", "p_wander", "damage", "cue_dependence")
#: D-083. A STRICTNESS threshold and a judgement ("most genomes"), not derived. Inventoried as such.
COVERAGE_FLOOR = 0.5
N_ENG_MIN = DEFAULT.n_eng_min
FLOOR = DEFAULT.learnability_floor


def settings(stage: int) -> list[dict]:
    if stage == 3:
        return [
            {"h_regen": DEFAULT.h_regen, "p_wander": w, "damage": DEFAULT.damage, "cue_dependence": c}
            for c in CUE_DEPENDENCE
            for w in P_WANDER_3
        ]
    return [{"h_regen": h, "p_wander": w, "damage": DAMAGE[stage]} for h in H_REGEN for w in P_WANDER]


def tag(setting: dict) -> str:
    return "__".join(f"{k}{setting[k]:g}" for k in MOVED if k in setting)


def n_moved(setting: dict) -> int:
    """Parameters this setting moves away from the project default."""
    return sum(setting[k] != getattr(DEFAULT, k) for k in MOVED if k in setting)


def params_for(setting: dict):
    """The treatment arm's scripted world with exactly the moved parameters replaced.

    **Built on the run configuration, never on the frozen world.** `sweep.params_for` applies configs/frozen_world.py,
    so once WORLD is set it would leak the frozen `cue_dependence` into every stage 1 and 2 setting, which do not
    name it, and runs 53, 55 and 56 would stop reproducing without any error. The calibration's world must not
    depend on its own output (D-092)."""
    base = sweep.BY_NAME[sweep.ARMS["treatment"]].params
    p = dataclasses.replace(base, **setting)
    check_invariants(p)
    return p


# ------------------------------------------------------------------------------------- the pass rule ----
def record_passes(rec: dict) -> bool:
    med = rec["median_learnability"]
    return bool(rec["coverage"] >= COVERAGE_FLOOR and not math.isnan(med) and med >= FLOOR)


def setting_passes(recs: list[dict], seeds=SEEDS, measure_after=MEASURE_AFTER) -> bool | None:
    """True or False once every (seed, generation) record exists; None while any is missing."""
    have = {(r["seed"], r["after"]) for r in recs}
    need = {(s, a) for s in seeds for a in measure_after}
    if not need <= have:
        return None
    return all(record_passes(r) for r in recs if (r["seed"], r["after"]) in need)


def score(recs: list[dict], last=MEASURE_AFTER[-1]) -> float:
    return float(np.mean([r["median_learnability"] for r in recs if r["after"] == last]))


def choose(table: dict[str, dict], seeds=SEEDS, measure_after=MEASURE_AFTER) -> str | None:
    """`table[tag] = {"setting": ..., "records": [...]}`. The frozen choice, or None if nothing passes or the
    stage is incomplete. Highest score; ties to fewer moved parameters, then to the earlier grid position."""
    verdicts = {t: setting_passes(v["records"], seeds, measure_after) for t, v in table.items()}
    if any(v is None for v in verdicts.values()):
        return None
    passing = [t for t, v in verdicts.items() if v]
    if not passing:
        return None
    order = {t: i for i, t in enumerate(table)}
    last = measure_after[-1]
    return max(
        passing, key=lambda t: (score(table[t]["records"], last), -n_moved(table[t]["setting"]), -order[t])
    )


def frozen_setting(world: dict | None) -> tuple[int, str]:
    """(stage, tag) of the grid setting equal to `configs/frozen_world.py`'s WORLD, so the A6000's S4 re-check
    reads the frozen choice rather than a typed tag. Refuses an unfrozen world or one outside the grid."""
    if world is None:
        raise SystemExit(
            "--frozen: configs/frozen_world.py WORLD is None; nothing has been frozen yet (D-083)"
        )
    for stage in STAGES:
        for s in settings(stage):
            if set(world) == set(s) and all(world[k] == s[k] for k in s):
                return stage, tag(s)
    raise SystemExit(f"--frozen: the frozen world {world} is not a setting of the calibration grid")


def stage3_allowed(stage1_summary: dict | None, stage2_summary: dict | None) -> tuple[bool, str]:
    """Stage 3 runs only when stages 1 and 2 are complete and chose nothing (Q-060)."""
    for n, s in ((1, stage1_summary), (2, stage2_summary)):
        if s is None or s.get("status") != "complete":
            return False, f"stage {n} is not complete"
        if s.get("chosen") is not None:
            return False, f"stage {n} chose {s['chosen']}; stage 3 runs only if nothing passed before it"
    return True, "stages 1 and 2 complete and nothing passed"


def stage2_allowed(stage1_summary: dict | None) -> tuple[bool, str]:
    if stage1_summary is None:
        return False, "stage 1 has no summary yet"
    if stage1_summary.get("status") != "complete":
        return False, "stage 1 is not complete"
    if stage1_summary.get("chosen") is not None:
        return (
            False,
            f"stage 1 passed ({stage1_summary['chosen']}); D-083 runs stage 2 only if nothing passed",
        )
    return True, "stage 1 complete and nothing passed"


# --------------------------------------------------------------------------------- evolve and measure ----
def build_world(p, seed: int):
    """The sweep's construction for this seed, exactly as `sweep.one_seed` builds it."""
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, relax = resources.place_sources(jax.random.key(seed), ob, p)
    static = E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )
    return static, int(relax)


def evolve(p, seed: int, n_gen: int, measure_after, episodes: int, log=print):
    """`sweep.one_seed`'s evolution, with the population measured after each count in `measure_after`."""
    static, relax = build_world(p, seed)
    template = net.init_params(jax.random.key(seed), p)
    unflatten = genome.unflattener(template)
    pop = loop.initial_population(jax.random.key(seed + 1), template, p)
    step = loop.make_generation_step(unflatten, static, p)
    eng_pass = engagement.make_engagement_pass(unflatten, p)
    key = jax.random.key(seed + 2)
    fit_mean = np.zeros(n_gen, np.float32)
    fit_max = np.zeros(n_gen, np.float32)
    measured = {}
    t0 = time.time()
    if 0 in measure_after:
        # The initial population, before any selection (D-089's diagnosis). Measurement keys never touch the
        # evolution's chain, so measuring here leaves every later generation exactly as it would have been.
        measured[0] = engagement.measure(eng_pass, pop, static, seed, 0, episodes)
    for gen in range(n_gen):
        key, kg, _ka = jax.random.split(
            key, 3
        )  # the sweep's three-way split; _ka feeds only its sampling pass
        pop, fit = step(kg, pop)
        f = np.asarray(fit)
        fit_mean[gen], fit_max[gen] = f.mean(), f.max()
        done = gen + 1
        if done in measure_after:
            tm = time.time()
            measured[done] = engagement.measure(eng_pass, pop, static, seed, done, episodes)
            log(f"    measured after {done}: {time.time() - tm:.1f} s")
        if done % 25 == 0 or done == n_gen:
            log(
                f"    gen {done}/{n_gen}  fit mean {fit_mean[gen]:.3f}  max {fit_max[gen]:.3f}  {time.time() - t0:.0f} s"
            )
    return {
        "pop": np.asarray(pop, np.float32),
        "fit_mean": fit_mean,
        "fit_max": fit_max,
        "relax": relax,
        "measured": measured,
    }


def summarise(eng: np.ndarray, prey: np.ndarray) -> dict:
    r = learnability.s4(eng[None], N_ENG_MIN, FLOOR)
    r["predated"] = float(prey[:, :, 2, :].mean())
    r["starved"] = float(prey[:, :, 3, :].mean())
    r["engagements_per_prey_episode"] = float(
        eng[..., 0].sum() / (prey.shape[0] * prey.shape[1] * prey.shape[3])
    )
    return r


# ----------------------------------------------------------------------------------------- the files ----
def git_state() -> tuple[str, bool]:
    def run(*a):
        return subprocess.run(
            ["git", *a], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain"))


def file_ok(path: pathlib.Path, after, episodes: int) -> bool:
    """A finished file of THIS budget. A file measured over a different episode count is not resumed or summarised,
    so a summary never reports one budget over data of another (D-092)."""
    try:
        with np.load(path) as z:
            return (
                int(z["schema_version"]) == SCHEMA_VERSION
                and all(f"eng_{a}" in z for a in after)
                and int(z["episodes"]) == episodes
            )
    except Exception:
        return False


def records_from(path: pathlib.Path, after) -> list[dict]:
    with np.load(path) as z:
        seed = int(z["seed"])
        out = []
        for a in after:
            r = summarise(z[f"eng_{a}"], z[f"prey_{a}"])
            r.update(seed=seed, after=int(a), fit_mean=float(z["fit_mean"][a - 1]))
            out.append(r)
        return out


def write_summary(out_dir: pathlib.Path, stage: int, grid_settings, seeds, after, episodes: int) -> dict:
    table = {}
    for s in grid_settings:
        recs = []
        for seed in seeds:
            f = out_dir / f"{tag(s)}__seed{seed}.npz"
            if file_ok(f, after, episodes):
                recs.extend(records_from(f, after))
        table[tag(s)] = {"setting": s, "records": recs}
    complete = all(setting_passes(v["records"], seeds, after) is not None for v in table.values())
    summary = {
        "run": "S4 calibration",
        "decision": "D-083",
        "knobs": "Q-057",
        "stage": stage,
        "rule": {
            "coverage_floor": COVERAGE_FLOOR,
            "learnability_floor": FLOOR,
            "n_eng_min": N_ENG_MIN,
            "seeds": list(seeds),
            "measure_after": list(after),
            "episodes": episodes,
        },
        "status": "complete" if complete else "incomplete",
        "settings": {
            t: {
                **v,
                "passes": setting_passes(v["records"], seeds, after),
                "score": score(v["records"], after[-1]) if v["records"] else None,
            }
            for t, v in table.items()
        },
        "chosen": choose(table, seeds, after) if complete else None,
    }
    tmp = out_dir / f".summary_stage{stage}.json"
    tmp.write_text(json.dumps(summary, indent=1, default=float))
    tmp.replace(out_dir / f"summary_stage{stage}.json")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--stage", type=int, choices=STAGES, default=1)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 generations, 2 episodes, one setting, one seed")
    ap.add_argument(
        "--frozen",
        action="store_true",
        help="re-check the setting configs/frozen_world.py holds (implies --only and its stage)",
    )
    ap.add_argument(
        "--only",
        help="re-check ONE setting by its tag, into a directory named for a recheck (checklist 4.5: the frozen "
        "setting on the A6000 with the widened network, D-085). Same seeds, budget and pass rule",
    )
    a = ap.parse_args(argv)
    if a.frozen:
        from configs import frozen_world

        a.stage, a.only = frozen_setting(frozen_world.WORLD)
    src.assert_runtime_took_effect()

    grid_settings = settings(a.stage)
    seeds, n_gen, after, episodes = SEEDS, N_GENERATIONS, MEASURE_AFTER, EPISODES
    if a.smoke:
        assert "smoke" in a.out_dir.name, "a smoke run writes only into a directory named for it"
        grid_settings, seeds, n_gen, after, episodes = grid_settings[:1], seeds[:1], 2, (1, 2), 2
    if a.only is not None:
        picked = [s for s in grid_settings if tag(s) == a.only]
        if not picked:
            raise SystemExit(
                f"--only {a.only}: not a stage {a.stage} setting. The settings are {[tag(s) for s in grid_settings]}"
            )
        if "recheck" not in a.out_dir.name:
            raise SystemExit(
                "--only writes into a directory named for a recheck (for example runs/s4-recheck), never into "
                "the calibration's own directory, whose summary must stay the calibration's."
            )
        grid_settings = picked
        seeds = RECHECK_SEEDS  # never the seeds the choice was made on
    for s in grid_settings:
        params_for(s)  # every setting passes check_invariants before anything runs

    print(
        f"S4 calibration stage {a.stage}: {len(grid_settings)} settings x {len(seeds)} seeds {tuple(seeds)}, {n_gen} generations, "
        f"measured after {after} over {episodes} episodes, device {jax.devices()[0].device_kind}"
    )
    for s in grid_settings:
        print(f"  {tag(s)}  moves {n_moved(s)} from default")
    if a.plan_only:
        return 0

    a.out_dir.mkdir(parents=True, exist_ok=True)

    def _summary(n):
        f = a.out_dir / f"summary_stage{n}.json"
        return json.loads(f.read_text()) if f.exists() else None

    if a.stage == 2 and not a.smoke and a.only is None:
        ok, why = stage2_allowed(_summary(1))
        if not ok:
            raise SystemExit(f"stage 2 refused: {why}")
    if a.stage == 3 and not a.smoke and a.only is None:
        ok, why = stage3_allowed(_summary(1), _summary(2))
        if not ok:
            raise SystemExit(f"stage 3 refused: {why}")

    commit, dirty = git_state()
    for s in grid_settings:
        p = params_for(s)
        for seed in seeds:
            path = a.out_dir / f"{tag(s)}__seed{seed}.npz"
            if file_ok(path, after, episodes):
                print(f"  {path.name}: done, skipped")
                continue
            print(f"  {path.name}: running", flush=True)
            t0 = time.time()
            r = evolve(p, seed, n_gen, after, episodes, log=lambda m: print(m, flush=True))
            out = {
                "schema_version": np.int32(SCHEMA_VERSION),
                "seed": np.int32(seed),
                "stage": np.int32(a.stage),
                "setting": np.array(json.dumps(s)),
                "config_hash": np.array(loop.config_hash(p)),
                "fit_mean": r["fit_mean"],
                "fit_max": r["fit_max"],
                "relax": np.int32(r["relax"]),
                "engagement_fields": np.array(rf.ENGAGEMENT_FIELDS),
                "prey_fields": np.array(rf.PREY_FIELDS),
                "episodes": np.int32(episodes),
                "code_commit": np.array(commit),
                "dirty_tree": np.bool_(dirty),
                "device": np.array(jax.devices()[0].device_kind),
                "jax_version": np.array(jax.__version__),
                "wall_seconds": np.float32(time.time() - t0),
            }
            for k, (eng, prey) in r["measured"].items():
                out[f"eng_{k}"], out[f"prey_{k}"] = eng, prey
            tmp = path.with_name("." + path.name)
            np.savez_compressed(tmp, **out)
            sweep.durable_replace(tmp, path)  # DEV-041
            for rec in records_from(path, after):
                print(
                    f"    after {rec['after']}: coverage {rec['coverage']:.3f}  median L {rec['median_learnability']:.3f}"
                    f"  median p {rec['median_p']:.3f}  predated {rec['predated']:.3f}  {'PASS' if record_passes(rec) else 'fail'}",
                    flush=True,
                )
            del r
            jax.clear_caches()  # once per run, as the sweep does per condition. DEV-029
            gc.collect()
            summary = write_summary(a.out_dir, a.stage, grid_settings, seeds, after, episodes)
    summary = write_summary(a.out_dir, a.stage, grid_settings, seeds, after, episodes)
    print(f"stage {a.stage} {summary['status']}; chosen: {summary['chosen']}")
    for t, v in summary["settings"].items():
        print(f"  {t}: passes {v['passes']}  score {v['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
