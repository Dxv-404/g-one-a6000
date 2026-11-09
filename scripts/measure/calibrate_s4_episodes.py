"""Re-measure an S4 calibration stage at more episodes, and choose the episode count by a rule stated first. Q-061.

    uv run python scripts/measure/calibrate_s4_episodes.py runs/calibration-s4-episodes --stage 3 --episodes 64

Stage 3 (run 57) failed D-083 on coverage alone: 16 episodes give a genome about 18 decided engagements against
`n_eng_min` 30. This re-runs the stage's whole grid, every setting and seed, with `--episodes` measurement episodes
through `calibrate_s4`'s own `evolve`, `summarise`, `setting_passes`, `score` and `choose`, so nothing here is a
second implementation of the rule.

**The first 16 episodes ARE stage 3's measurement.** `engagement.measure` folds episode e into per-genome keys, so
episode e is the same draw at any episode count. Every file is checked bit for bit against stage 3's saved counts
before it is written, which checks that both the evolution and the measurement reproduced. `--prefix-check` adds
the same check against an earlier, longer measurement (the 48-episode diagnostic).

**The episode count is chosen by `pick_episodes`, written before the 64-episode numbers existed** (Q-061):

    1  apply D-083 to the WHOLE grid at 48 and at 64 episodes (prefixes of the one measurement)
    2  if the passing set or the chosen setting differs, 48 was not stable: 64
    3  otherwise the smallest of (48, 64) at which the chosen setting's worst median-genome margin, the median
       decided count over n_eng_min, minimised over seeds and checkpoints, is at least 2.0: coverage then
       survives the engagement rate HALVING late in the long run, which 300 generations cannot measure
    4  if neither reaches 2.0: 64, the larger

Writes only into a directory whose name contains "episodes", so the stage's own directory and summary stay the
16-episode record they are. Blind to behaviour like `calibrate_s4`: it loads nothing from the behaviour side.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import jax  # noqa: E402

import src  # noqa: E402
from src.measure import learnability  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "calibrate_s4", ROOT / "scripts" / "measure" / "calibrate_s4.py"
)
cal = sys.modules.get("calibrate_s4")
if cal is None:
    cal = importlib.util.module_from_spec(_spec)
    sys.modules["calibrate_s4"] = cal
    _spec.loader.exec_module(cal)

#: The two counts Q-061 compares, and the margin rule's target: coverage survives a halving of the engagement rate.
CANDIDATES = (48, 64)
MARGIN_TARGET = 2.0
#: Prefixes reported in the summary. 16 is stage 3 itself, so its verdicts must equal run 57's.
PREFIXES = (16, 32, 48, 64)


def worst_margin(eng_by: dict, n_eng_min: int = cal.N_ENG_MIN) -> float:
    """min over (seed, after) of the MEDIAN genome's decided count over n_eng_min. `eng_by[(seed, after)]` is (G, E, F)."""
    out = math.inf
    for eng in eng_by.values():
        den, _ = learnability.decided_and_escaped(np.asarray(eng)[None])
        out = min(out, float(np.median(den[0])) / n_eng_min)
    return out


def pick_episodes(by_e: dict) -> int | None:
    """Q-061's rule. `by_e[E] = {"passing": set of tags, "chosen": tag or None, "margin": {tag: worst margin}}`.

    None when nothing passes at any candidate: then there is no frozen choice to make the count for."""
    lo, hi = CANDIDATES
    a, b = by_e[lo], by_e[hi]
    if a["chosen"] is None and b["chosen"] is None:
        return None
    if set(a["passing"]) != set(b["passing"]) or a["chosen"] != b["chosen"]:
        return hi
    for e in CANDIDATES:
        if by_e[e]["margin"][by_e[e]["chosen"]] >= MARGIN_TARGET:
            return e
    return hi


def table_at(files: dict, e: int) -> tuple[dict, dict]:
    """D-083's table at the first `e` episodes, and each setting's worst margin there."""
    table, margins = {}, {}
    for t, (setting, per_seed) in files.items():
        recs, eng_by = [], {}
        for seed, z in per_seed.items():
            for a in cal.MEASURE_AFTER:
                eng, prey = z[f"eng_{a}"][:, :e], z[f"prey_{a}"][:, :e]
                r = cal.summarise(eng, prey)
                r.update(seed=seed, after=int(a), fit_mean=float(z["fit_mean"][a - 1]))
                recs.append(r)
                eng_by[(seed, a)] = eng
        table[t] = {"setting": setting, "records": recs}
        margins[t] = worst_margin(eng_by)
    return table, margins


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--source", type=pathlib.Path, default=ROOT / "runs" / "calibration-s4")
    ap.add_argument("--prefix-check", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)
    if "episodes" not in a.out_dir.name:
        raise SystemExit(
            "writes only into a directory named for it (for example runs/calibration-s4-episodes)"
        )
    if a.episodes < max(CANDIDATES) or a.episodes < cal.STAGE_EPISODES:
        raise SystemExit(
            f"--episodes must reach {max(CANDIDATES)}, so both candidates are prefixes of one measurement"
        )
    src.assert_runtime_took_effect()

    grid = cal.settings(a.stage)
    for s in grid:
        cal.params_for(s)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = cal.git_state()
    print(
        f"S4 stage {a.stage} re-measured at {a.episodes} episodes: {len(grid)} settings x {cal.SEEDS}, from "
        f"{commit[:7]} dirty {dirty}, device {jax.devices()[0].device_kind}",
        flush=True,
    )
    for s in grid:
        p = cal.params_for(s)
        for seed in cal.SEEDS:
            path = a.out_dir / f"{cal.tag(s)}__seed{seed}.npz"
            if path.exists():
                print(f"  {path.name}: done, skipped", flush=True)
                continue
            print(f"  {path.name}: running", flush=True)
            t0 = time.time()
            r = cal.evolve(
                p, seed, cal.N_GENERATIONS, cal.MEASURE_AFTER, a.episodes, log=lambda m: print(m, flush=True)
            )
            with np.load(a.source / path.name) as old:
                assert int(old["episodes"]) == cal.STAGE_EPISODES
                for af in cal.MEASURE_AFTER:
                    eng, prey = r["measured"][af]
                    same = np.array_equal(eng[:, : cal.STAGE_EPISODES], old[f"eng_{af}"]) and np.array_equal(
                        prey[:, : cal.STAGE_EPISODES], old[f"prey_{af}"]
                    )
                    print(
                        f"    after {af}: first {cal.STAGE_EPISODES} episodes equal stage {a.stage}'s: {same}",
                        flush=True,
                    )
                    assert same, (
                        "the evolution or the measurement did not reproduce; nothing here is comparable"
                    )
            if a.prefix_check is not None and (a.prefix_check / path.name).exists():
                with np.load(a.prefix_check / path.name) as pre:
                    n = int(pre["episodes"])
                    for af in cal.MEASURE_AFTER:
                        same = np.array_equal(r["measured"][af][0][:, :n], pre[f"eng_{af}"])
                        print(
                            f"    after {af}: first {n} episodes equal {a.prefix_check.name}'s: {same}",
                            flush=True,
                        )
                        assert same, "the earlier longer measurement did not reproduce"
            out = {
                "schema_version": np.int32(cal.SCHEMA_VERSION),
                "seed": np.int32(seed),
                "stage": np.int32(a.stage),
                "setting": np.array(json.dumps(s)),
                "fit_mean": r["fit_mean"],
                "fit_max": r["fit_max"],
                "relax": np.int32(r["relax"]),
                "episodes": np.int32(a.episodes),
                "code_commit": np.array(commit),
                "dirty_tree": np.bool_(dirty),
                "device": np.array(jax.devices()[0].device_kind),
                "jax_version": np.array(jax.__version__),
                "wall_seconds": np.float32(time.time() - t0),
            }
            for af, (eng, prey) in r["measured"].items():
                out[f"eng_{af}"], out[f"prey_{af}"] = eng, prey
            tmp = path.with_name("." + path.name)
            np.savez_compressed(tmp, **out)
            cal.sweep.durable_replace(tmp, path)  # DEV-041, as calibrate_s4 does
            del r
            jax.clear_caches()  # once per run, as calibrate_s4 and the sweep do. DEV-029
            gc.collect()

    files = {}
    for s in grid:
        per_seed = {seed: dict(np.load(a.out_dir / f"{cal.tag(s)}__seed{seed}.npz")) for seed in cal.SEEDS}
        files[cal.tag(s)] = (s, per_seed)
    by_e, report = {}, {}
    for e in [x for x in PREFIXES if x <= a.episodes]:
        table, margins = table_at(files, e)
        verdicts = {t: cal.setting_passes(v["records"]) for t, v in table.items()}
        chosen = cal.choose(table)
        by_e[e] = {
            "passing": sorted(t for t, ok in verdicts.items() if ok),
            "chosen": chosen,
            "margin": margins,
        }
        report[e] = {
            "passing": by_e[e]["passing"],
            "chosen": chosen,
            "margin": margins,
            "score": {t: cal.score(v["records"]) for t, v in table.items()},
            "settings": table,
        }
        print(f"  E {e}: passing {by_e[e]['passing']}  chosen {chosen}", flush=True)
    pick = pick_episodes(by_e)
    summary = {
        "run": "S4 calibration, re-measured at more episodes",
        "decision": "Q-061",
        "stage": a.stage,
        "episodes_measured": a.episodes,
        "rule": {"candidates": list(CANDIDATES), "margin_target": MARGIN_TARGET, "n_eng_min": cal.N_ENG_MIN},
        "episodes_chosen": pick,
        "chosen_setting": by_e[pick]["chosen"] if pick else None,
        "by_episodes": report,
    }
    tmp = a.out_dir / ".summary_episodes.json"
    tmp.write_text(json.dumps(summary, indent=1, default=float))
    tmp.replace(a.out_dir / "summary_episodes.json")
    print(f"episodes chosen: {pick}; setting: {summary['chosen_setting']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
