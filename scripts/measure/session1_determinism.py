#!/usr/bin/env python3
"""Pre-run checklist 4.2: does turning off XLA's autotuning give ONE compiled program across MANY fresh processes?

    .venv/bin/python -u scripts/measure/session1_determinism.py runs/sweep runs/replay runs/session1/determinism
    .venv/bin/python -u scripts/measure/session1_determinism.py runs/sweep runs/replay OUT --processes 10 --contrast 3
    .venv/bin/python -u scripts/measure/session1_determinism.py runs/sweep runs/replay OUT --plan-only

**Why.** Run 50 (DEV-038) found that two compilations of one program in two processes can differ on near-tied
argmax actions, and that `--xla_gpu_autotune_level=0` gave the same result in 2 of 2 fresh processes. Two is not
evidence for a 27-day run whose conditions each compile afresh. This runs the flag in N fresh processes (default
10) and, as the contrast, 3 fresh processes WITHOUT it, so the result can show the flag is what makes them agree.

**It reuses `drift_diag.py` rather than copying it.** The worker, the clean environment, the targets (chosen from the
replay's own reproduction record) and the comparison are drift_diag's; this script only replaces its list of
settings and then reads its summary. Each process compiles into its own empty cache, which is the hardest case: no
process can borrow another's compiled program.

**What each outcome would mean, written before it is run:**

    all N flagged processes agree, contrast processes disagree   the flag is what pins the program; adopt it
    all N flagged processes agree, contrast processes also agree compiles were stable this time either way; the
                                                                 flag is still needed (run 50 saw them differ), and
                                                                 this run neither confirms nor refutes its necessity
    any two flagged processes disagree                           the flag does NOT pin the program on this machine.
                                                                 The long run cannot rely on it; stop and report
    a process failed                                             counted as not run, never as agreement; the verdict
                                                                 says how many of N actually ran

Inputs, read only: the sweep's checkpoints (final populations) and the replay's files. Output: one npz per process,
drift_diag's `summary.json`, and `verdict.json` here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FLAG = "--xla_gpu_autotune_level=0"

_SPEC = importlib.util.spec_from_file_location(
    "drift_diag_s1", ROOT / "scripts" / "measure" / "drift_diag.py"
)
dd = importlib.util.module_from_spec(_SPEC)
sys.modules["drift_diag_s1"] = dd
_SPEC.loader.exec_module(dd)


def settings(n_flagged: int, n_contrast: int, flag: str = FLAG) -> tuple:
    """drift_diag's (name, fresh_cache, XLA_FLAGS) rows. Names keep drift_diag's prefixes, because its comparison
    groups processes by prefix ("noauto", "fresh") to decide whether they agree."""
    assert n_flagged >= 2, "agreement needs at least two flagged processes"
    w = len(str(n_flagged))
    rows = [(f"noauto_{i:0{w}d}", True, flag) for i in range(1, n_flagged + 1)]
    rows += [(f"fresh_{i}", True, "") for i in range(1, n_contrast + 1)]
    return tuple(rows)


def verdict(summary: dict, n_flagged: int, n_contrast: int) -> dict:
    """Plain-words reading of drift_diag's summary. Pure, so it is unit-tested on hand-built summaries.

    A process that failed is counted as NOT RUN, never as agreeing: agreement among the processes that ran says
    nothing about the ones that did not."""
    status = summary.get("status", {})
    ran_flagged = sorted(k for k, v in status.items() if k.startswith("noauto") and v.get("exit") == 0)
    ran_contrast = sorted(k for k, v in status.items() if k.startswith("fresh") and v.get("exit") == 0)
    results = summary.get("results", {})
    per_target, flagged_agree_all, contrast_agree_all = {}, True, True
    for target, row in results.items():
        fa = row.get("noauto_processes_agree")
        ca = row.get("fresh_processes_agree")
        flagged_cells = [row[k] for k in ran_flagged if k in row]
        per_target[target] = {
            "flagged_processes_agree": fa,
            "contrast_processes_agree": ca,
            "flagged_equal_stored": sum(c["equals_stored"] for c in flagged_cells),
            "flagged_equal_replay": sum(c["equals_replay"] for c in flagged_cells),
            "flagged_ran": len(flagged_cells),
            "drifted_in_replay": row.get("drifted_in_replay"),
        }
        flagged_agree_all &= fa is True
        contrast_agree_all &= ca is True
    complete = len(ran_flagged) == n_flagged and bool(results)
    if not results or len(ran_flagged) < 2:
        status_text = f"NOT DECIDED: only {len(ran_flagged)} of {n_flagged} flagged processes ran"
        pins = None
    elif not flagged_agree_all:
        status_text = (
            f"FAILS: flagged processes disagree on at least one target ({len(ran_flagged)} of {n_flagged} ran). "
            "The flag does not pin the compiled program on this machine; the long run cannot rely on it"
        )
        pins = False
    elif not complete:
        status_text = (
            f"INCOMPLETE: the {len(ran_flagged)} flagged processes that ran agree on every target, but "
            f"{n_flagged - len(ran_flagged)} did not run, so this is not the {n_flagged}-process result"
        )
        pins = None
    else:
        status_text = f"PASSES: all {n_flagged} flagged processes agree bit for bit on every target"
        pins = True
    if len(ran_contrast) < 2:
        contrast_text = f"contrast not decided: {len(ran_contrast)} of {n_contrast} unflagged processes ran"
    elif contrast_agree_all:
        contrast_text = (
            "the unflagged processes ALSO agreed this time, so this run does not show the flag is necessary "
            "(run 50 did); it shows only that the flag is sufficient here"
        )
    else:
        contrast_text = "the unflagged processes disagreed, so the flag is what makes the flagged ones agree"
    return {
        "status": status_text,
        "flag_pins_program": pins,
        "contrast": contrast_text,
        "flagged_ran": len(ran_flagged),
        "flagged_planned": n_flagged,
        "contrast_ran": len(ran_contrast),
        "contrast_planned": n_contrast,
        "per_target": per_target,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="4.2: one compiled program across many fresh processes")
    ap.add_argument("sweep_dir", type=pathlib.Path)
    ap.add_argument("replay_dir", type=pathlib.Path)
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--processes", type=int, default=10, help="fresh processes WITH the flag (default 10)")
    ap.add_argument("--contrast", type=int, default=3, help="fresh processes WITHOUT it (default 3)")
    ap.add_argument("--flag", default=FLAG)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    rows = settings(a.processes, a.contrast, a.flag)
    if not a.replay_dir.is_dir():
        print(f"STOPPED: no replay folder at {a.replay_dir}; the targets are read from it.", flush=True)
        return 2
    targets = dd.choose_targets(a.replay_dir)
    print(
        f"4.2: {len(rows)} fresh processes ({a.processes} with {a.flag!r}, {a.contrast} without), "
        f"{len(targets)} targets: {targets}",
        flush=True,
    )
    if a.plan_only:
        return 0
    if not a.sweep_dir.is_dir():
        print(
            f"STOPPED: no sweep folder at {a.sweep_dir}; the final populations are read from it.", flush=True
        )
        return 2
    dd.SETTINGS = rows  # drift_diag reads its settings at call time; everything else is its own code
    dd.main([str(a.sweep_dir), str(a.replay_dir), str(a.out_dir)])
    summary = json.loads((a.out_dir / "summary.json").read_text())
    v = verdict(summary, a.processes, a.contrast)
    (a.out_dir / "verdict.json").write_text(json.dumps(v, indent=1))
    print(f"\n4.2 VERDICT: {v['status']}\n  contrast: {v['contrast']}", flush=True)
    print(f"written {a.out_dir / 'verdict.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
