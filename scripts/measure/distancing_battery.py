"""Run 39, D-065: which distancing descriptor should be the named one? The planted battery over the candidates.

For each configuration and each candidate: the z of every policy's mean against the pooled predator-blind
null (forager, idler3, idler7), and the blind policies' own means with their standard errors, which is the
bias check: a blind policy must sit at zero on a chance-corrected measure.
"""

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "measure"))

import planted_policies as PP  # noqa: E402  after the path is set

import src  # noqa: E402
from src.measure import classifier as C  # noqa: E402

CANDIDATES = (
    "dgain_perceived",
    "conditional_dgain",
    "dgain_sight_contrast",
    "dgain_perceived_raw",
    "mean_distance",
    "conditional_away_near",
    "flight_rate",
    "flight_initiation_distance",
)

print(src.assert_runtime_took_effect(), flush=True)
t0 = time.time()
out = {}
for cfg, overrides in PP.CONFIGS.items():
    accs = {kind: PP.run_policy(kind, overrides) for kind in PP.POLICIES}
    desc = {kind: PP.per_prey_descriptors(accs[kind]) for kind in PP.POLICIES}
    print(f"\n=== {cfg} {overrides}   ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"  {'candidate':<27s}" + "".join(f"{k[:9]:>10s}" for k in PP.POLICIES), flush=True)
    for name in CANDIDATES:
        null = C.null_from(np.concatenate([desc[k][name] for k in PP.NULL_POLICIES]))
        zs = []
        for kind in PP.POLICIES:
            v = desc[kind][name]
            v = v[np.isfinite(v)]
            z = C.z_score(float(v.mean()), null) if len(v) else float("nan")
            zs.append(z)
            out[f"{cfg}__{name}__{kind}__z"] = z
        print(f"  {name:<27s}" + "".join(f"{z:10.2f}" for z in zs), flush=True)
    print("\n  blind-policy means (mean +/- se, n defined) on the corrected candidates", flush=True)
    for name in ("dgain_perceived", "conditional_dgain", "dgain_sight_contrast", "dgain_perceived_raw"):
        cells = []
        for kind in PP.NULL_POLICIES:
            v = desc[kind][name]
            v = v[np.isfinite(v)]
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
            cells.append(f"{kind}: {v.mean():+.4f} +/- {se:.4f} (n={len(v)})")
        print(f"  {name:<22s} " + "   ".join(cells), flush=True)
    for name in ("dgain_perceived", "dgain_sight_contrast", "flight_rate"):
        d = desc["distancer"][name]
        print(f"  distancer {name}: defined for {np.isfinite(d).sum()} of {len(d)} prey", flush=True)
np.savez(ROOT / "runs" / "distancing_battery.npz", **{k: np.float64(v) for k, v in out.items()})
print(f"\ndone in {(time.time() - t0) / 60:.1f} min", flush=True)
