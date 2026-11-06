"""Run 52. The analysis of run 48, exactly as D-082 fixed it before any of it was computed.

    uv run python scripts/measure/replay_analysis.py

**Every result is exploratory (D-075) and comes from run 48, which failed D-077's pre-registered drift rule; the
cause is demonstrated (run 50) and no run 47 conclusion moves on it (run 49). D-081.**

Families, each paired by seed within ONE evaluation world:

    F1  treatment own  vs null own      the scripted world, conditions both arms ran
    F2  treatment own  vs nullp         the predator-only settings (DEV-036)
    F3  third real     vs null          own null where it ran that condition, nullp otherwise (DEV-037)
    F4  treatment own  vs third real    both in the scripted world

Per-seed quantities from the replay's 4-episode `prey` and `engagement` arrays: predated, starved, time_alive,
escape_rate (pooled escaped / (escaped + captured + died_other)), engagements per prey-episode. Tier one is
D-072's procedure, tier three is `phase_a_map.compare` on the one-episode accumulator rows. Individuals: share of
treatment genomes beyond the same seed's null 5th and 95th percentiles, against 0.05. S4 per D-076.

Reads runs/replay48/replay. Writes runs/replay-analysis/analysis.json.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REP = ROOT / "runs" / "replay48" / "replay"
OUT = ROOT / "runs" / "replay-analysis" / "analysis.json"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


pam = _load("pam_ra", "scripts/measure/phase_a_map.py")
power = pam.power
from configs.default import DEFAULT  # noqa: E402
from src.measure import learnability  # noqa: E402

N_ENG_MIN = DEFAULT.n_eng_min
BAND = DEFAULT.learnability_floor
QUANTITIES = ("predated", "starved", "time_alive", "escape_rate", "engagements")
PREY = {
    "alive_ticks": 0,
    "death_step": 1,
    "died_predated": 2,
    "died_starving": 3,
    "alive_at_end": 4,
    "score": 5,
}
ENG = {"engagements": 0, "escaped": 1, "captured": 2, "died_other": 3, "censored": 4}
MIN_DEFINED_PER_SEED = 10  # an implementation guard: a percentile from fewer genomes is not computed


# ------------------------------------------------------------------------------------------ quantities ----
def per_seed(prey: np.ndarray, eng: np.ndarray) -> dict[str, np.ndarray]:
    """prey (S, G, E, 6, k), eng (S, G, E, F) -> each quantity per seed, shape (S,)."""
    prey = prey.astype(np.float64)
    eng = eng.astype(np.float64)
    k = prey.shape[-1]
    den = eng[..., ENG["escaped"]] + eng[..., ENG["captured"]] + eng[..., ENG["died_other"]]
    den_s = den.sum(axis=(1, 2))
    return {
        "predated": prey[:, :, :, PREY["died_predated"], :].mean(axis=(1, 2, 3)),
        "starved": prey[:, :, :, PREY["died_starving"], :].mean(axis=(1, 2, 3)),
        "time_alive": prey[:, :, :, PREY["alive_ticks"], :].mean(axis=(1, 2, 3)),
        "escape_rate": np.where(
            den_s > 0, eng[..., ENG["escaped"]].sum(axis=(1, 2)) / np.maximum(den_s, 1), np.nan
        ),
        "engagements": eng[..., ENG["engagements"]].sum(axis=(1, 2)) / (eng.shape[1] * eng.shape[2] * k),
    }


def per_genome(prey: np.ndarray, eng: np.ndarray) -> dict[str, np.ndarray]:
    """Per genome over its episodes and prey, shape (S, G). Escape rate NaN where its denominator is below
    n_eng_min (D-076, Q-047)."""
    eng = eng.astype(np.float64)
    den = (eng[..., ENG["escaped"]] + eng[..., ENG["captured"]] + eng[..., ENG["died_other"]]).sum(axis=2)
    esc = eng[..., ENG["escaped"]].sum(axis=2)
    p = np.where(den >= N_ENG_MIN, esc / np.maximum(den, 1), np.nan)
    return {
        "escape_rate": p,
        "time_alive": prey[:, :, :, PREY["alive_ticks"], :].astype(np.float64).mean(axis=(2, 3)),
        "denominator": den,
    }


def tier1(diff: np.ndarray, rng) -> dict:
    d = np.asarray(diff, np.float64)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return {
            "n": int(d.size),
            "iqm_diff": math.nan,
            "ci95": [math.nan, math.nan],
            "pos": 0,
            "neg": 0,
            "p": 1.0,
        }
    lo, hi = power.bootstrap_interval(d, rng, n_boot=pam.N_BOOT)
    pos, neg, p = pam.sign_test(d)
    return {"n": int(d.size), "iqm_diff": power.iqm(d), "ci95": [lo, hi], "pos": pos, "neg": neg, "p": p}


def exceedance(treat: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per seed, the share of treatment genomes above the null's 95th percentile and below its 5th, from the
    same seed. NaN where either arm has fewer than MIN_DEFINED_PER_SEED defined genomes."""
    above, below = np.full(treat.shape[0], np.nan), np.full(treat.shape[0], np.nan)
    for s in range(treat.shape[0]):
        t, n = treat[s][np.isfinite(treat[s])], null[s][np.isfinite(null[s])]
        if t.size < MIN_DEFINED_PER_SEED or n.size < MIN_DEFINED_PER_SEED:
            continue
        above[s] = float(np.mean(t > np.percentile(n, 95)))
        below[s] = float(np.mean(t < np.percentile(n, 5)))
    return above, below


def s4(eng: np.ndarray) -> dict:
    """D-076: per genome p = escaped / (escaped + captured + died_other), censored excluded, defined where that
    denominator is at least n_eng_min; learnability p(1 - p), in band at the floor or above."""
    # The arithmetic lives in `src.measure.learnability`, the one implementation the S4 calibration also calls
    # (D-083). Moved there unchanged on 2026-11-06; run 52's numbers were produced by the inline copy at 1fb6b7a.
    return learnability.s4(eng, N_ENG_MIN, BAND)


# ------------------------------------------------------------------------------------------------ main ----
def shim(d) -> dict:
    """A run 48 job in the form `phase_a_map.compare` reads: population means per seed as `naive_acc`."""
    return {
        "acc_fields": d["acc_fields"],
        "arm": "null",
        "naive_acc": d["rows"].astype(np.float64).mean(axis=1),
    }


def main() -> None:
    fz = pam.require_frozen()
    floor, sep = fz["classifier_floor"], fz["separation_margin"]
    jobs = {p.stem: p for p in sorted(REP.glob("*__*.npz"))}
    assert len(jobs) == 148, len(jobs)
    cache = {}

    def get(name):
        if name not in cache:
            with np.load(jobs[name]) as d:
                cache[name] = {k: d[k] for k in ("prey", "engagement", "rows", "acc_fields")}
        return cache[name]

    def cond(name):
        return name.split("__", 2)[2]

    families = {"F1": [], "F2": [], "F3": [], "F4": []}
    for n in jobs:
        if n.startswith("own__treatment__") and f"own__null__{cond(n)}" in jobs:
            families["F1"].append((n, f"own__null__{cond(n)}"))
        if n.startswith("nullp__null__") and f"own__treatment__{cond(n)}" in jobs:
            families["F2"].append((f"own__treatment__{cond(n)}", n))
        if n.startswith("real__third__"):
            ref = next((r for r in (f"own__null__{cond(n)}", f"nullp__null__{cond(n)}") if r in jobs), None)
            if ref:
                families["F3"].append((n, ref))
            if f"own__treatment__{cond(n)}" in jobs:
                families["F4"].append((f"own__treatment__{cond(n)}", n))

    rng = np.random.default_rng(pam.SEED)
    rng3 = np.random.default_rng(pam.SEED + 1)
    out = {
        "run": 52,
        "decision": "D-082",
        "status": "EXPLORATORY (D-075); run 48 failed D-077, cause demonstrated run 50, no conclusion moved run 49 (D-081)",
        "thresholds": {"floor": floor, "separation": sep},
        "families": {},
        "s4": {},
    }
    for fam, pairs in families.items():
        res = {}
        for a, b in pairs:
            A, B = get(a), get(b)
            qa, qb = per_seed(A["prey"], A["engagement"]), per_seed(B["prey"], B["engagement"])
            r = {
                "arm_means": {q: float(np.nanmean(qa[q])) for q in QUANTITIES},
                "ref_means": {q: float(np.nanmean(qb[q])) for q in QUANTITIES},
                "tier1": {q: tier1(qa[q] - qb[q], rng) for q in QUANTITIES},
            }
            t3 = pam.compare(shim(A), shim(B), rng3, floor, sep)["tier3"]
            r["tier3"] = {
                "label": t3["label"],
                "best_z": t3["best_z"],
                "z": {s: v["z"] for s, v in t3["profile"].items()},
            }
            if fam in ("F1", "F2"):
                ga, gb = per_genome(A["prey"], A["engagement"]), per_genome(B["prey"], B["engagement"])
                r["individuals"] = {}
                for q in ("escape_rate", "time_alive"):
                    above, below = exceedance(ga[q], gb[q])
                    r["individuals"][q] = {
                        "above95": tier1(above - 0.05, rng),
                        "below5": tier1(below - 0.05, rng),
                        "above95_median": float(np.nanmedian(above))
                        if np.isfinite(above).any()
                        else math.nan,
                        "below5_median": float(np.nanmedian(below)) if np.isfinite(below).any() else math.nan,
                    }
            res[f"{a}  vs  {b}"] = r
        keys = list(res)
        for q in QUANTITIES:
            for k, adj in zip(keys, pam.holm([res[k]["tier1"][q]["p"] for k in keys]), strict=True):
                res[k]["tier1"][q]["p_holm"] = adj
        if fam in ("F1", "F2"):
            for q in ("escape_rate", "time_alive"):
                for tail in ("above95", "below5"):
                    ps = [res[k]["individuals"][q][tail]["p"] for k in keys]
                    for k, adj in zip(keys, pam.holm(ps), strict=True):
                        res[k]["individuals"][q][tail]["p_holm"] = adj
        out["families"][fam] = res

    for n in jobs:
        if n.startswith(("own__treatment__", "own__null__", "nullp__", "real__third__", "own__third__")):
            out["s4"][n] = s4(get(n)["engagement"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    report(out)
    print(f"\nwritten {OUT}")


def report(out: dict) -> None:
    names = {
        "F1": "treatment vs null (own worlds)",
        "F2": "treatment vs null at predator-only settings",
        "F3": "harmless-predator lineage vs null, real predator",
        "F4": "treatment vs harmless-predator lineage, real predator",
    }
    for fam, res in out["families"].items():
        print(f"\n===== {fam} {names[fam]}: {len(res)} conditions")
        for q in QUANTITIES:
            vals = [v["tier1"][q] for v in res.values()]
            hi = sum(1 for t in vals if t["iqm_diff"] > 0)
            lo = sum(1 for t in vals if t["iqm_diff"] < 0)
            sig = [
                (k.split("  vs")[0], round(v["tier1"][q]["iqm_diff"], 4))
                for k, v in res.items()
                if v["tier1"][q]["p_holm"] < 0.05
            ]
            med = np.nanmedian([t["iqm_diff"] for t in vals])
            print(
                f"  {q:<12s} arm higher in {hi}, lower in {lo}; median IQM diff {med:+.4f}; Holm-significant {len(sig)}: {sig[:8]}"
            )
        labels = [
            (k.split("  vs")[0], v["tier3"]["label"], round(v["tier3"]["best_z"], 2))
            for k, v in res.items()
            if v["tier3"]["label"] != "unclassified"
        ]
        bz = max(
            abs(v["tier3"]["best_z"])
            for v in res.values()
            if v["tier3"]["best_z"] is not None and math.isfinite(v["tier3"]["best_z"])
        )
        print(f"  tier three: labelled {len(labels)} {labels}; largest |best z| {bz:.2f}")
        if fam in ("F1", "F2"):
            for q in ("escape_rate", "time_alive"):
                for tail in ("above95", "below5"):
                    sig = [
                        (k.split("  vs")[0], v["individuals"][q][f"{tail}_median"])
                        for k, v in res.items()
                        if v["individuals"][q][tail].get("p_holm", 1) < 0.05
                    ]
                    meds = [v["individuals"][q][f"{tail}_median"] for v in res.values()]
                    print(
                        f"  individuals {q:<11s} {tail:<7s}: median share {np.nanmedian(meds):.3f} (0.05 if alike); Holm-significant {len(sig)}: {sig[:6]}"
                    )
    print("\n===== S4 (D-076), per job: coverage, share of defined genomes in band, median learnability")
    for kind in ("own__treatment__", "own__null__", "nullp__", "real__third__", "own__third__"):
        rows = {k: v for k, v in out["s4"].items() if k.startswith(kind)}
        cov = [v["coverage"] for v in rows.values()]
        inb = [v["share_in_band_of_defined"] for v in rows.values()]
        ml = [v["median_learnability"] for v in rows.values()]
        best = max(rows.items(), key=lambda kv: (kv[1]["share_in_band_of_all"]))
        print(
            f"  {kind:<18s} {len(rows)} jobs: coverage median {np.nanmedian(cov):.3f} (range {np.nanmin(cov):.3f}-{np.nanmax(cov):.3f}); "
            f"in band of defined median {np.nanmedian(inb):.3f} (max {np.nanmax(inb):.3f}); median learnability {np.nanmedian(ml):.3f}; "
            f"most genomes in band: {best[0]} {best[1]['share_in_band_of_all']:.3f} of all"
        )


if __name__ == "__main__":
    main()
