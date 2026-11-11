#!/usr/bin/env python3
"""Is the long run's final program the same program in every fresh process? Checklist 4.2 on a machine without
Phase A's data, such as the H100 (D-090).

Session 1's 4.2 replayed Phase A checkpoints, which exist only on the A6000. This asks the same question of the
program that will actually run: one seed of the long run's own world (`sweep.params_for`, so the frozen world when
there is one) is evolved for a few generations in `--runs` processes one after another, each with an EMPTY
compilation cache, so every process compiles everything itself, which is the condition DEV-038's drift came from.
Every run must equal the first bit for bit: final population, mean and best fitness. The unflagged contrast this
used to run beside it is no longer possible: src/__init__.py pins `--xla_gpu_autotune_level=0` in every process
(DEV-045), so `--contrast` defaults to 0 and a request for it is reported as not run.

It reuses session 1's worker and comparison (`session1_concurrency.run_round`, `compare_state`), so the two checks
cannot differ in what "identical" means.

What each outcome means, written before it runs:

    every flagged run identical         the program is one program on this machine; the long run may start
    a flagged run differs               two compilations of one program disagree here: do not start; send the folder
    a run failed                        reported with its log; nothing is concluded from the runs that finished

    .venv/bin/python -u scripts/measure/session_determinism.py runs/sessionX/determinism [--runs 6] [--contrast 2]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "session1_concurrency_det", ROOT / "scripts" / "measure" / "session1_concurrency.py"
)
conc = importlib.util.module_from_spec(_SPEC)
sys.modules["session1_concurrency_det"] = conc
_SPEC.loader.exec_module(conc)

SEED = 230  # a test seed: not Phase A's 0 to 19, the long run's 20 to 39, the calibration's 100 to 102 or 200 upward
GENERATIONS = 30


def verdict(rounds: list[dict], states: list[dict | None]) -> tuple[bool, str]:
    """rounds: run_round results in order; states: the final state of each, None if it failed."""
    failed = [i for i, (r, s) in enumerate(zip(rounds, states, strict=True)) if r["failed"] or s is None]
    if failed:
        return False, f"runs {failed} failed; nothing is concluded from the others"
    if len(states) < 2:
        return False, "fewer than two runs, so nothing was compared"
    diffs = []
    for i, s in enumerate(states[1:], 1):
        c = conc.compare_state(states[0], s)
        if not c["identical"]:
            diffs.append((i, c["genomes_differing"]))
    if diffs:
        return False, f"runs differ from the first: {diffs} (run, genomes differing)"
    return True, f"all {len(states)} runs identical bit for bit"


def series(n, flags, out, label, log) -> tuple[list, list]:
    rounds, states = [], []
    for k in range(n):
        d = out / f"{label}{k + 1:02d}"
        r = conc.run_round([SEED], GENERATIONS, flags, "fresh", d, log)
        rounds.append(r)
        f = d / f"seed{SEED}.npz"
        states.append(conc.load_state(f) if f.exists() and not r["failed"] else None)
    return rounds, states


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--contrast", type=int, default=0)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(m, flush=True)

    log(f"determinism: seed {SEED}, {GENERATIONS} generations, {a.runs} fresh processes with {conc.FLAG!r}")
    fr, fs = series(a.runs, conc.FLAG, a.out_dir, "flagged", log)
    ok, text = verdict(fr, fs)
    log(f"VERDICT: {'PASSES' if ok else 'FAILS'}: {text}")
    c_ok, c_text = (None, "not run")
    if a.contrast:
        c_text = "not run: src/__init__.py pins the flag in every process, so an unflagged run is impossible (DEV-045)"
        log(f"contrast: {c_text}")
    (a.out_dir / "verdict.json").write_text(
        json.dumps(
            {"passes": ok, "text": text, "contrast_identical": c_ok, "contrast_text": c_text}, indent=1
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
