"""The exploratory strategy map from the characterisation sweep: tier one and tier three. Run 47. D-072, D-075.

**Refuses to run unless the step 13 thresholds are frozen first.** `configs/step13_frozen.json` must be committed,
unmodified in the working tree, and its commit must be an ancestor of HEAD. That is the pre-registration made
structural: no treatment or third-arm value is read by this script until the thresholds that judge it exist in
history.

Every procedure is the one D-072 fixed before any result was seen:

    tier one     per matched condition and per outcome, the paired per-seed difference arm minus null at the
                 last sampled generation: interquartile mean, 95 percent bootstrap interval (10,000 resamples),
                 two-sided exact sign test with ties dropped, Holm across conditions within each outcome
    tier three   the paired z-profile of the five named descriptors, classified with the frozen floor and
                 margin; fleeing and distancing report together as directed avoidance (D-050)
    novelty      third arm against null, and treatment against third arm, by the same procedures

Excluded: the conditions where the predator out-sees the prey (D-074). Pending: treatment and third-arm
conditions on predator-only axes, which have no null evaluated in the same world (DEV-036). Everything here is
exploratory (D-075): the seeds also produced the null the thresholds came from.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.measure import classifier as C  # noqa: E402
from src.measure import descriptors as D  # noqa: E402
from src.measure import power  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load("sweep_mod", "scripts/measure/sweep.py")
s13 = _load("s13_mod", "scripts/measure/step13_thresholds.py")

EXTRACT = ROOT / "runs" / "a6000-handoff" / "extract"
FROZEN = ROOT / "configs" / "step13_frozen.json"
OUT = ROOT / "runs" / "phase-a" / "map.json"
OUTCOMES = ("predated", "starved", "survived")
EXCLUDED = {"treatment__r_pred__5", "treatment__r_pred__6", "third__r_pred__6"}  # D-074
N_BOOT = 10_000
SEED = 20261104
FAMILY = {"fleeing": "directed avoidance", "distancing": "directed avoidance"}  # D-050


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def require_frozen() -> dict:
    """The pre-registration guard, in code."""
    rel = str(FROZEN.relative_to(ROOT))
    if git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"{rel} differs from HEAD; the thresholds must be committed before any arm is read")
    last = git("log", "-1", "--format=%H", "--", rel).stdout.strip()
    if not last:
        raise SystemExit(f"{rel} has no commit; freeze and commit the thresholds first")
    if git("merge-base", "--is-ancestor", last, "HEAD").returncode != 0:
        raise SystemExit("the freezing commit is not an ancestor of HEAD")
    pushed = git("branch", "-r", "--contains", last).stdout.strip()
    fz = json.loads(FROZEN.read_text())
    fz["_freeze_commit"], fz["_freeze_pushed"] = last, bool(pushed)
    return fz


def sign_test(d: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact sign test, ties dropped. Returns (positive, negative, p)."""
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return pos, neg, min(1.0, p)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def arm_rows(d) -> tuple[np.ndarray, list[str]]:
    """Seed rows at the last sampled generation for an evolving arm, or the naive_eval rows for the null."""
    cols = [str(c) for c in d["acc_fields"]]
    if str(d["arm"]) == "null":
        return np.asarray(d["naive_acc"], np.float64), cols
    gens = np.asarray(d["sample_gens"])
    assert int(gens[-1]) == int(d["n_generations"]) - 1, "last sample is not the last generation"
    return np.asarray(d["sample_acc"], np.float64)[:, -1, :], cols


def named_values(rows: np.ndarray, cols: list[str]) -> tuple[dict, dict]:
    pooled = rows * float(sweep.POOLED_PREY_EPISODES)
    pop = D.population({c: pooled[:, i] for i, c in enumerate(cols)}, denominator_floor=s13.DENOM_FLOOR)
    vals = {n: np.where(pop[n + "__defined"], pop[n], np.nan) for n in D.NAMED_STRATEGIES.values()}
    dens = {n: pop[n + "__denominator"] for n in D.NAMED_STRATEGIES.values()}
    return vals, dens


def compare(a, b, rng, floor, sep) -> dict:
    """Arm a against arm b, paired by seed, by D-072's procedures."""
    ra, ca = arm_rows(a)
    rb, cb = arm_rows(b)
    assert ca == cb, "accumulator schemas differ"
    out = {"tier1": {}, "tier3": {}}
    for o in OUTCOMES:
        diff = ra[:, ca.index(o)] - rb[:, cb.index(o)]
        lo, hi = power.bootstrap_interval(diff, rng, n_boot=N_BOOT)
        pos, neg, p = sign_test(diff)
        out["tier1"][o] = {
            "arm": float(ra[:, ca.index(o)].mean()),
            "ref": float(rb[:, cb.index(o)].mean()),
            "iqm_diff": power.iqm(diff),
            "ci95": [lo, hi],
            "pos": pos,
            "neg": neg,
            "p": p,
        }
    va, da = named_values(ra, ca)
    vb, db = named_values(rb, cb)
    zs, detail = {}, {}
    for strat, name in D.NAMED_STRATEGIES.items():
        z = s13.paired_z(va[name], vb[name])
        pairs = int((np.isfinite(va[name]) & np.isfinite(vb[name])).sum())
        detail[strat] = {
            "z": z,
            "pairs": pairs,
            "den_arm_median": float(np.median(da[name])),
            "den_ref_median": float(np.median(db[name])),
        }
        if math.isfinite(z):
            zs[name] = z
    res = C.classify(zs, {n: C.Null(0.0, 1.0) for n in zs}, floor=floor, separation=sep)
    label = res.label
    if label in D.NAMED_STRATEGIES:
        label = FAMILY.get(label, label)
    out["tier3"] = {
        "profile": detail,
        "label": label,
        "best_z": res.z,
        "runner_up": res.runner_up,
        "margin": res.margin,
        "n_defined": res.n_defined,
    }
    return out


def main() -> None:
    fz = require_frozen()
    floor, sep = fz["classifier_floor"], fz["separation_margin"]
    files = {p.stem: p for p in EXTRACT.glob("*.npz")}
    assert len(files) == 109
    load = {k: np.load(p, allow_pickle=True) for k, p in files.items()}
    hashes = {k: str(v["config_hash"]) for k, v in load.items()}
    rng = np.random.default_rng(SEED)

    def cond(tag):
        return tag.split("__", 1)[1]

    seen, results, pending = {}, {}, []
    for tag in sorted(files):
        arm = tag.split("__", 1)[0]
        if arm == "null" or tag in EXCLUDED:
            continue
        if hashes[tag] in seen:  # DEV-035: a configuration is counted once
            continue
        seen[hashes[tag]] = tag
        null_tag = f"null__{cond(tag)}"
        if null_tag not in files:
            pending.append(tag)
            continue
        results[f"{tag}  vs  {null_tag}"] = {
            "family": f"{arm} vs null",
            **compare(load[tag], load[null_tag], rng, floor, sep),
        }
        if arm == "third":
            tr = f"treatment__{cond(tag)}"
            if tr in files and tr not in EXCLUDED:
                results[f"{tr}  vs  {tag}"] = {
                    "family": "treatment vs third",
                    **compare(load[tr], load[tag], rng, floor, sep),
                }

    for fam in ("treatment vs null", "third vs null", "treatment vs third"):
        keys = [k for k, v in results.items() if v["family"] == fam]
        for o in OUTCOMES:
            adj = holm([results[k]["tier1"][o]["p"] for k in keys])
            for k, a in zip(keys, adj, strict=True):
                results[k]["tier1"][o]["p_holm"] = a

    commit = git("rev-parse", "HEAD").stdout.strip()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "run": 47,
                "decision": "D-072",
                "exploratory": True,
                "script_commit": commit,
                "thresholds": {
                    "floor": floor,
                    "separation": sep,
                    "freeze_commit": fz["_freeze_commit"],
                    "freeze_pushed": fz["_freeze_pushed"],
                },
                "extract_index_sha256": hashlib.sha256((EXTRACT / "index.json").read_bytes()).hexdigest(),
                "excluded_D074": sorted(EXCLUDED),
                "pending_DEV036": pending,
                "comparisons": results,
            },
            indent=1,
        )
    )

    print(
        f"thresholds floor {floor} sep {sep}, frozen at {fz['_freeze_commit'][:7]} pushed={fz['_freeze_pushed']}"
    )
    print(f"comparisons {len(results)}, pending {len(pending)}, excluded {len(EXCLUDED)}")
    for fam in ("treatment vs null", "third vs null", "treatment vs third"):
        print(f"\n===== {fam}: outcome IQM difference [95% CI] sign p (Holm), then tier three")
        for k, v in results.items():
            if v["family"] != fam:
                continue
            cells = []
            for o in OUTCOMES:
                t = v["tier1"][o]
                cells.append(
                    f"{o[:4]} {t['iqm_diff']:+.3f} [{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}] "
                    f"{t['pos']}/{t['neg']} p={t['p']:.3g}({t['p_holm']:.2g})"
                )
            t3 = v["tier3"]
            prof = " ".join(f"{s[:4]}{d['z']:+.1f}" for s, d in t3["profile"].items())
            print(f"{k.split('  vs')[0]:<38s} " + " | ".join(cells) + f" || {t3['label']} [{prof}]")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
