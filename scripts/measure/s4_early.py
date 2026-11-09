#!/usr/bin/env python3
"""Is S4's signal there early and used up by evolution, or never there? The diagnosis Q-059 ruled first. D-089.

Runs 53 and 55 found no setting with learnability in band after 100 and 300 generations, and the deadliest settings
evolved the HIGHEST escape rates. Two readings predict that, and they need opposite responses:

    A  the world gives selection a usable signal at the start, and evolution uses it up as prey learn to escape
       (then D-083's rule, which asks a FIXED predator to keep learnability in band, asks the impossible)
    B  the signal is never there (then the predator's escape rule has to move: stage 3)

So this measures S4, blind to behaviour, at generations 0, 10, 25, 50 and 100 on four settings already run, with
the calibration's own evolution and measurement code and the calibration's seeds.

**Fixed before it ran (D-089).** The four settings and why each:

    h_regen0.25__p_wander0__damage34    the current world
    h_regen0.1__p_wander0.5__damage34   stage 1's best with coverage (0.74)
    h_regen0__p_wander0__damage34       stage 1's best median learnability (0.107)
    h_regen0__p_wander1__damage50       the deadliest, with the highest evolved escape rate

The verdict, per setting: EARLY IN BAND if D-083's per-point rule (coverage >= 0.5 and median learnability >= 0.16)
holds at generation 0, 10 or 25 in at least 2 of 3 seeds; otherwise NEVER IN BAND EARLY. Answer A if any setting is
EARLY IN BAND, B if none is. What to do with the answer is the owner's.

**A reproduction check comes free.** The measurement after 100 generations uses the same evolution, keys and code as
run 53 or 55 on the same device, so it must equal their stored `eng_100` bit for bit; it is compared per seed and
reported, and a mismatch is a finding about the code or the device, not about the world.

    uv run python scripts/measure/s4_early.py runs/s4-early
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

import src

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_SPEC = importlib.util.spec_from_file_location(
    "calibrate_s4_early", ROOT / "scripts" / "measure" / "calibrate_s4.py"
)
cal = importlib.util.module_from_spec(_SPEC)
sys.modules["calibrate_s4_early"] = cal
_SPEC.loader.exec_module(cal)

SETTINGS = (
    {"h_regen": 0.25, "p_wander": 0.0, "damage": 34.0},
    {"h_regen": 0.1, "p_wander": 0.5, "damage": 34.0},
    {"h_regen": 0.0, "p_wander": 0.0, "damage": 34.0},
    {"h_regen": 0.0, "p_wander": 1.0, "damage": 50.0},
)
SEEDS = cal.SEEDS
AFTER = (0, 10, 25, 50, 100)
EARLY = (0, 10, 25)
EPISODES = cal.STAGE_EPISODES  # run 56 was measured at 16
CALIBRATION = ROOT / "runs" / "calibration-s4"


def early_verdict(records: list[dict]) -> str:
    """records: one per (seed, after), with coverage and median_learnability. D-089's rule."""
    for a in EARLY:
        passing = sum(cal.record_passes(r) for r in records if r["after"] == a)
        if passing >= 2:
            return "EARLY IN BAND"
    return "NEVER IN BAND EARLY"


def answer(verdicts: dict[str, str]) -> str:
    return "A" if any(v == "EARLY IN BAND" for v in verdicts.values()) else "B"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=pathlib.Path)
    a = ap.parse_args(argv)
    src.assert_runtime_took_effect()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = cal.git_state()
    summary = {"decision": "D-089", "question": "Q-059", "after": list(AFTER), "settings": {}}
    for s in SETTINGS:
        tag = cal.tag(s)
        p = cal.params_for(s)
        recs, repro = [], []
        for seed in SEEDS:
            path = a.out_dir / f"{tag}__seed{seed}.npz"
            if not cal.sweep.readable(path):
                print(f"  {path.name}: running", flush=True)
                t0 = time.time()
                r = cal.evolve(p, seed, max(AFTER), AFTER, EPISODES, log=lambda m: print(m, flush=True))
                out = {
                    "seed": np.int32(seed),
                    "setting": np.array(json.dumps(s)),
                    "code_commit": np.array(commit),
                    "dirty_tree": np.bool_(dirty),
                    "wall_seconds": np.float32(time.time() - t0),
                }
                for k, (eng, prey) in r["measured"].items():
                    out[f"eng_{k}"], out[f"prey_{k}"] = eng, prey
                tmp = path.with_name("." + path.name)
                np.savez_compressed(tmp, **out)
                cal.sweep.durable_replace(tmp, path)
            with np.load(path) as z:
                for k in AFTER:
                    rec = cal.summarise(z[f"eng_{k}"], z[f"prey_{k}"])
                    rec.update(seed=seed, after=k)
                    recs.append(rec)
                stored = CALIBRATION / f"{tag}__seed{seed}.npz"
                if stored.exists():
                    with np.load(stored) as c:
                        repro.append(
                            {
                                "seed": seed,
                                "eng_100_identical": bool(np.array_equal(c["eng_100"], z["eng_100"])),
                            }
                        )
        v = early_verdict(recs)
        summary["settings"][tag] = {"verdict": v, "records": recs, "reproduction": repro}
        print(f"{tag}: {v}; reproduction {repro}", flush=True)
        for k in AFTER:
            rs = [r for r in recs if r["after"] == k]
            print(
                f"    after {k:3d}: coverage {np.median([r['coverage'] for r in rs]):.2f}  "
                f"median L {np.nanmedian([r['median_learnability'] for r in rs]):.3f}  "
                f"median p {np.nanmedian([r['median_p'] for r in rs]):.3f}  "
                f"pooled NOT L {np.nanmedian([r['pooled_NOT_learnability_median'] for r in rs]):.3f}  "
                f"seeds passing {sum(cal.record_passes(r) for r in rs)}/3",
                flush=True,
            )
    summary["answer"] = answer({t: v["verdict"] for t, v in summary["settings"].items()})
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"ANSWER {summary['answer']}  (A: in band early, used up by evolution; B: never in band early)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
