"""Phase A characterisation of the sweep's null arm, measurement M5, plus fitness and spawn health. Run 45.

Reads `runs/a6000-handoff/extract` (runs 42 to 44) and writes `runs/phase-a/null_characterisation.json`.

**Phase A observes only.** Build spec, Phase A measurements: it "attaches no pass or fail, sets no thresholds,
and computes no target-behaviour detector". The canonical's calibration rule is the same one from the other
side: the target-behaviour detectors are not computed during calibration, so there is no signal to tune
toward. So this script computes descriptors on the NULL ARM ONLY, the predator-naive lineage evaluated
against the predator, which is what M5 names. **No descriptor and no outcome column is read from the
treatment or third arm**, and `_null_rows` refuses any other arm in the code rather than by convention.
From the other two arms it reads fitness (`fit_mean`, which is `prey_score`, the selection criterion and not
a behaviour) and the spawn relaxation level, which are the calibration proxies the canonical names.

**Nothing here is a threshold.** The floor and separation margin and the denominator floor are derived at
step 13 by a rule the owner has not yet stated (Q-054), and this script does not choose one. The descriptor
values are computed at the floor the S5 gate already uses, `DENOM_FLOOR = 1.0` on pooled ticks (D-071), and
the per-genome defined fraction is reported so the floor's consequence is visible before anyone sets it.

Two levels, because the classifier and the population map read different ones:
    seed level     the population-mean row pooled to p_pop * k prey-episodes, as S5 reads it
    genome level   each genome's row pooled to its own k prey-episodes, which is what tier three's
                   classifier standardises an individual against

DEV-035: conditions are deduplicated by config hash, so a configuration is counted once.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.default import DEFAULT  # noqa: E402
from src.measure import descriptors as D  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("sweep_mod", ROOT / "scripts" / "measure" / "sweep.py")
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)

EXTRACT = ROOT / "runs" / "a6000-handoff" / "extract"
OUT = ROOT / "runs" / "phase-a" / "null_characterisation.json"

#: The floor S5 applies, unchanged. Not a step 13 threshold; see the module docstring.
DENOM_FLOOR = 1.0
#: The only arm whose descriptors Phase A may compute.
DESCRIPTOR_ARMS = frozenset({"null"})
#: What one per-genome row averages: one episode's k prey (`sweep.POOLED_PREY_EPISODES` docstring).
GENOME_PREY_EPISODES = DEFAULT.k
OUTCOMES = ("predated", "starved", "survived")


def _null_rows(d) -> tuple[np.ndarray, np.ndarray]:
    """The naive-eval rows of a null-arm file. Raises on any other arm, which is the Phase A guard."""
    arm = str(d["arm"])
    if arm not in DESCRIPTOR_ARMS:
        raise PermissionError(f"Phase A computes descriptors on {sorted(DESCRIPTOR_ARMS)} only, not {arm!r}")
    return np.asarray(d["naive_acc"], np.float64), np.asarray(d["naive_acc_per_genome"], np.float64)


def _summary(v: np.ndarray, defined: np.ndarray) -> dict:
    x = v[defined]
    out = {"n": int(v.size), "n_defined": int(defined.sum())}
    if x.size >= 2:
        q = np.percentile(x, [5, 25, 50, 75, 95])
        out.update(
            mean=float(x.mean()),
            sd=float(x.std(ddof=1)),
            min=float(x.min()),
            max=float(x.max()),
            p5=float(q[0]),
            p25=float(q[1]),
            median=float(q[2]),
            p75=float(q[3]),
            p95=float(q[4]),
        )
    return out


def _describe(rows: np.ndarray, cols: list[str], scale: float) -> dict[str, dict]:
    """Every descriptor over a stack of mean rows, each pooled by `scale` before the floor applies."""
    accs = {c: rows[:, i] * scale for i, c in enumerate(cols)}
    pop = D.population(accs, denominator_floor=DENOM_FLOOR)
    names = [n for n in pop if not n.endswith("__denominator") and not n.endswith("__defined")]
    out = {}
    for n in names:
        s = _summary(pop[n], pop[n + "__defined"])
        s["denominator_median"] = float(np.median(pop[n + "__denominator"]))
        out[n] = s
    return out


def main() -> None:
    files = sorted(EXTRACT.glob("*.npz"))
    if len(files) != 109:
        raise SystemExit(f"expected 109 condition files in {EXTRACT}, found {len(files)}")
    commit0 = str(np.load(files[0], allow_pickle=True)["code_commit"])

    seen: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    null_out: dict[str, dict] = {}
    health: dict[str, dict] = {}
    for p in files:
        d = np.load(p, allow_pickle=True)
        tag, arm, h = p.stem, str(d["arm"]), str(d["config_hash"])
        if str(d["code_commit"]) != commit0:
            raise SystemExit(f"{tag} was produced by a different code commit")
        if h in seen:
            duplicates[tag] = seen[h]
            continue
        seen[h] = tag

        fit = np.asarray(d["fit_mean"], np.float64)
        health[tag] = {
            "arm": arm,
            "axis": str(d["axis"]),
            "level": str(d["level"]),
            "config_hash": h,
            "fit_final_mean": float(fit[:, -1].mean()),
            "fit_final_sd": float(fit[:, -1].std(ddof=1)),
            "fit_gen25_mean": float(fit[:, 25].mean()),
            "relax_levels": sorted({int(x) for x in d["relax_levels"]}),
        }
        if arm not in DESCRIPTOR_ARMS:
            continue

        seed_rows, genome_rows = _null_rows(d)
        cols = [str(c) for c in d["acc_fields"]]
        g = genome_rows.reshape(-1, genome_rows.shape[-1])
        null_out[tag] = {
            "axis": str(d["axis"]),
            "level": str(d["level"]),
            "config_hash": h,
            "outcomes_seed": {
                o: _summary(seed_rows[:, cols.index(o)], np.ones(len(seed_rows), bool)) for o in OUTCOMES
            },
            "seed_level": _describe(seed_rows, cols, float(sweep.POOLED_PREY_EPISODES)),
            "genome_level": _describe(g, cols, float(GENOME_PREY_EPISODES)),
        }

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    index = (EXTRACT / "index.json").read_bytes()
    result = {
        "run": 45,
        "phase": "A",
        "measurement": "M5 plus fitness and spawn health",
        "script_commit": commit,
        "data_code_commit": commit0,
        "extract_index_sha256": hashlib.sha256(index).hexdigest(),
        "denominator_floor": DENOM_FLOOR,
        "named_strategies": D.NAMED_STRATEGIES,
        "distinct_configurations": len(seen),
        "duplicates_counted_once": duplicates,
        "null": null_out,
        "health": health,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))

    print(f"commit {commit[:7]}  data {commit0[:7]}  distinct configurations {len(seen)}")
    print(f"duplicates counted once: {duplicates}")
    print(f"null conditions characterised: {len(null_out)}")
    named = list(D.NAMED_STRATEGIES.values())
    print("\nseed level, named descriptors " + " | ".join(named) + ": defined of 20, mean, sd")
    for tag, r in null_out.items():
        cells = []
        for n in named:
            s = r["seed_level"][n]
            cells.append(
                f"{s['n_defined']:>2d} {s.get('mean', float('nan')):+.3f} {s.get('sd', float('nan')):.3f}"
            )
        print(f"  {tag:<36s} " + " | ".join(cells))
    print("\ngenome level, fraction of genomes defined, same order")
    for tag, r in null_out.items():
        cells = [f"{r['genome_level'][n]['n_defined'] / r['genome_level'][n]['n']:.3f}" for n in named]
        print(f"  {tag:<36s} " + "  ".join(cells))
    print("\nnull outcomes per seed: predated, starved, survived (mean over 20 seeds)")
    for tag, r in null_out.items():
        o = r["outcomes_seed"]
        print(f"  {tag:<36s} " + "  ".join(f"{o[k]['mean']:.3f}" for k in OUTCOMES))
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
