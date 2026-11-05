"""Why do 240 of 2,060 stored seed evaluations not reproduce? Q-055 path 1, part B. DEV-038.

    .venv/bin/python -u scripts/measure/drift_diag.py runs/sweep runs/replay runs/drift-diag

Re-evaluates a few seeds, chosen FROM THE REPLAY'S OWN RECORD (some that drifted, some that reproduced), with the
sweep's own accumulator pass, the identical program the sweep and the replay ran. Each evaluation happens in a
fresh process under one GPU setting, so the only thing that differs between settings is how the program was
compiled:

    shared        the project's compilation cache, as the replay ran
    fresh_1..3    an empty cache each: the program is recompiled from scratch, three times, in three processes
    noauto_1..2   an empty cache and XLA's timing-based kernel selection turned off (--xla_gpu_autotune_level=0)
    detops_1..2   an empty cache and XLA's deterministic-ops mode (--xla_gpu_deterministic_ops=true)

What each outcome would mean, written before it is run:

    fresh_1..3 disagree with each other     recompiling alone changes results: compile-time choice is the cause
    fresh_1..3 agree, differ from stored    recompiling is repeatable here; the sweep's compile differed
    noauto or detops agree across processes a candidate setting for the 1500-generation run, to be confirmed
                                            by a bit-for-bit pilot, not adopted on this alone
    a flag is refused by XLA                recorded as unsupported, never guessed around

Nothing in runs/sweep or runs/replay is written. Output: one file per setting and summary.json in the out dir.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = (
    pathlib.Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")))
    / "jax-emergent-self-preservation"
)

#: (job, how many of its drifting seeds, how many of its exact seeds). Seeds are read from the replay's record.
PICK = (
    ("own__null__baseline__None", 1, 1),
    ("own__third__obstacle_density__0.0", 2, 1),
    ("own__treatment__r_pred__3", 1, 1),
    ("own__treatment__n_target_src__9", 1, 0),
    ("own__null__n_food_src__4", 1, 0),
)
#: Seeds named in DEV-038 and run 48 as outcome-level or first-diagnosed drifts; taken first when they drifted.
PREFER = {
    "own__null__baseline__None": 0,
    "own__treatment__r_pred__3": 14,
    "own__treatment__n_target_src__9": 9,
    "own__null__n_food_src__4": 11,
    "own__third__obstacle_density__0.0": 13,
}
SETTINGS = (
    ("shared", False, ""),
    ("fresh_1", True, ""),
    ("fresh_2", True, ""),
    ("fresh_3", True, ""),
    ("noauto_1", True, "--xla_gpu_autotune_level=0"),
    ("noauto_2", True, "--xla_gpu_autotune_level=0"),
    ("detops_1", True, "--xla_gpu_deterministic_ops=true"),
    ("detops_2", True, "--xla_gpu_deterministic_ops=true"),
)


def choose_targets(replay_dir: pathlib.Path) -> list[tuple[str, int, bool]]:
    """(job, seed, drifted_in_replay) from the replay's own reproduction record, never from memory."""
    out = []
    for job, n_drift, n_exact in PICK:
        with np.load(replay_dir / f"{job}.npz") as d:
            rec = json.loads(str(d["reproduction"]))
        drift = [r["seed"] for r in rec if not r["exact"]]
        exact = [r["seed"] for r in rec if r["exact"]]
        pref = PREFER.get(job)
        if pref in drift:
            drift = [pref] + [s for s in drift if s != pref]
        out += [(job, s, True) for s in drift[:n_drift]] + [(job, s, False) for s in exact[:n_exact]]
    return out


def clean_env(fresh_cache: str | None, xla_flags: str) -> dict:
    """The inherited JAX_ and XLA_ variables and Python path variables removed, as env.sh does, then only what
    this setting asks for. src/__init__.py sets the project's own runtime variables in the worker."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("JAX_", "XLA_")) and k not in ("PYTHONPATH", "PYTHONHOME")
    }
    if fresh_cache is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = fresh_cache
    if xla_flags:
        env["XLA_FLAGS"] = xla_flags
    return env


def cache_census() -> dict:
    if not CACHE.is_dir():
        return {"dir": str(CACHE), "exists": False}
    times = [p.stat().st_mtime for p in CACHE.rglob("*") if p.is_file()]
    iso = lambda t: datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds")  # noqa: E731
    sweep_end = datetime.datetime(2026, 10, 21, 2, 0).timestamp()
    replay_start = datetime.datetime(2026, 11, 4, 19, 0).timestamp()
    return {
        "dir": str(CACHE),
        "files": len(times),
        "oldest": iso(min(times)) if times else None,
        "newest": iso(max(times)) if times else None,
        "written_by_2026_10_21": sum(t < sweep_end for t in times),
        "written_since_replay_start": sum(t >= replay_start for t in times),
    }


# --------------------------------------------------------------------------------------------- worker -----
def worker(setting: str, sweep_dir: pathlib.Path, targets_json: str, out: pathlib.Path) -> None:
    sys.path.insert(0, str(ROOT))
    import importlib.util

    import jax

    import src

    spec = importlib.util.spec_from_file_location("rf", ROOT / "scripts" / "measure" / "replay_final.py")
    rf = importlib.util.module_from_spec(spec)
    sys.modules["rf"] = rf
    spec.loader.exec_module(rf)
    src.assert_runtime_took_effect()
    assert jax.devices()[0].platform == "gpu", "the diagnostic must run on the GPU the sweep ran on"

    jobs = {j.name: j for j in rf.plan()}
    targets = json.loads(targets_json)
    rows_out, loaded, ckpt = [], None, None
    for name, s, _ in targets:
        job = jobs[name]
        if job.source_tag != loaded:
            with np.load(sweep_dir / f"{job.source_tag}.npz", allow_pickle=True) as d:
                ckpt = {"final_pop": d["final_pop"], "n_generations": int(d["n_generations"])}
            loaded = job.source_tag
        p_arm = rf.sweep.params_for(job.arm, job.axis, job.level)
        eval_p = rf.world_params(job)
        template = rf.net.init_params(jax.random.key(s), p_arm)
        unflatten = rf.genome.unflattener(template)
        static = rf.build_static(p_arm, s, eval_p)
        ka, ke = rf.seed_keys(s, ckpt["n_generations"])
        key = ka if job.key_kind == "ka" else ke
        t0 = time.time()
        rows = np.asarray(
            rf.sweep.make_accumulator_pass(unflatten, static, eval_p).all(
                key, jax.numpy.asarray(ckpt["final_pop"][s])
            )
        )
        rows_out.append(rows.astype(np.float32))
        print(f"  [{setting}] {name} seed {s}: {time.time() - t0:.1f} s", flush=True)
        jax.clear_caches()
    np.savez_compressed(
        out,
        rows=np.stack(rows_out),
        xla_flags=np.array(os.environ.get("XLA_FLAGS", "")),
        cache_dir=np.array(os.environ.get("JAX_COMPILATION_CACHE_DIR", "")),
        jax_version=np.array(jax.__version__),
        device=np.array(jax.devices()[0].device_kind),
    )


# --------------------------------------------------------------------------------------------- driver -----
def compare(targets, sweep_dir, replay_dir, out_dir) -> dict:
    results = {}
    files = {name: out_dir / f"{name}.npz" for name, _, _ in SETTINGS}
    got = {k: np.load(p)["rows"] for k, p in files.items() if p.exists()}
    stored_cache = {}
    for i, (job, s, drifted) in enumerate(targets):
        with np.load(replay_dir / f"{job}.npz") as d:
            replay_rows = d["rows"][s]
            arm, src = str(d["arm"]), str(d["source_tag"])
        if src not in stored_cache:
            with np.load(sweep_dir / f"{src}.npz", allow_pickle=True) as d:
                stored_cache[src] = {
                    k: d[k] for k in ("naive_acc", "naive_acc_per_genome", "sample_acc") if k in d.files
                }
        st = stored_cache[src]
        stored_mean = st["naive_acc"][s] if arm == "null" else st["sample_acc"][s, -1]
        row = {"drifted_in_replay": drifted}
        for k, rows in got.items():
            r = rows[i]
            mean = r.mean(axis=0).astype(np.float32)
            cell = {
                "equals_stored": bool(np.array_equal(mean, stored_mean)),
                "equals_replay": bool(np.array_equal(r, replay_rows)),
                "genomes_differing_from_replay": int(np.any(r != replay_rows, axis=1).sum()),
            }
            if arm == "null":
                cell["genomes_differing_from_stored"] = int(
                    np.any(r != st["naive_acc_per_genome"][s], axis=1).sum()
                )
            row[k] = cell
        for a in ("fresh", "noauto", "detops"):
            ks = [k for k in got if k.startswith(a)]
            if len(ks) > 1:
                row[f"{a}_processes_agree"] = all(np.array_equal(got[ks[0]][i], got[k][i]) for k in ks[1:])
        results[f"{job} seed {s}"] = row
    return results


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--worker":
        worker(argv[1], pathlib.Path(argv[2]), argv[3], pathlib.Path(argv[4]))
        return 0
    sweep_dir, replay_dir, out_dir = (pathlib.Path(a) for a in argv[:3])
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = choose_targets(replay_dir)
    census_before = cache_census()
    print("targets:", targets, flush=True)
    print("compilation cache before:", census_before, flush=True)
    status = {}
    for name, fresh, flags in SETTINGS:
        with tempfile.TemporaryDirectory(prefix=f"diag-{name}-", dir=str(out_dir)) as tmp:
            env = clean_env(tmp if fresh else None, flags)
            print(f"\n== {name}: fresh cache {fresh}, XLA_FLAGS '{flags}'", flush=True)
            t0 = time.time()
            proc = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "--worker",
                    name,
                    str(sweep_dir),
                    json.dumps(targets),
                    str(out_dir / f"{name}.npz"),
                ],
                env=env,
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            print(proc.stdout, end="", flush=True)
            status[name] = {"exit": proc.returncode, "seconds": round(time.time() - t0, 1)}
            if proc.returncode != 0:
                status[name]["stderr_tail"] = proc.stderr[-1500:]
                print(f"  {name} FAILED, exit {proc.returncode}:\n{proc.stderr[-1500:]}", flush=True)
    results = compare(targets, sweep_dir, replay_dir, out_dir)
    summary = {
        "targets": targets,
        "settings": [list(s) for s in SETTINGS],
        "status": status,
        "cache_before": census_before,
        "cache_after": cache_census(),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n================ summary: S = equals the sweep's stored value, R = equals the replay")
    names = [n for n, _, _ in SETTINGS if status[n]["exit"] == 0]
    print(f"{'target':<46s} {'drift':<6s} " + " ".join(f"{n:>9s}" for n in names))
    for t, row in results.items():
        cells = []
        for n in names:
            c = row[n]
            cells.append(f"{('S' if c['equals_stored'] else '-') + ('R' if c['equals_replay'] else '-'):>9s}")
        print(f"{t:<46s} {'yes' if row['drifted_in_replay'] else 'no':<6s} " + " ".join(cells))
    for t, row in results.items():
        agree = {k: v for k, v in row.items() if k.endswith("_processes_agree")}
        print(f"  {t}: {agree}")
    print(f"\nwritten {out_dir / 'summary.json'}. Send the whole {out_dir} folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
