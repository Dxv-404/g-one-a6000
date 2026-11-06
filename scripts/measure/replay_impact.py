"""Run 49. Q-055 path 1, part A: does any run 47 conclusion move when the replayed values replace the stored ones?

Builds a copy of the extract with naive_acc[s] (null) and sample_acc[s, -1] (evolving arms) replaced by the
replay's population means, runs phase_a_map.main() UNCHANGED against it, and diffs the two map.json files.
Read-only on the repository and on the original extract.

    uv run python scripts/measure/replay_impact.py

Needs runs/replay48 (run 48, extracted) and runs/a6000-handoff/extract. Writes runs/replay48/map-replayed.json.
"""

import importlib.util
import json
import math
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EX = ROOT / "runs/a6000-handoff/extract"
REP = ROOT / "runs/replay48/replay"
SUB = ROOT / "runs/replay48/extract-replayed"
OUT = ROOT / "runs/replay48/map-replayed.json"

if SUB.exists():
    shutil.rmtree(SUB)
SUB.mkdir(parents=True)
shutil.copy(EX / "index.json", SUB / "index.json")
replaced = 0
for p in sorted(EX.glob("*.npz")):
    d = dict(np.load(p, allow_pickle=True))
    r = REP / f"own__{p.stem}.npz"
    if r.exists():
        with np.load(r) as rd:
            rows = rd["rows"]
            means = rows.mean(axis=1).astype(np.float32)  # (S, C), the same reduction the replay compared
            assert [str(c) for c in rd["acc_fields"]] == [str(c) for c in d["acc_fields"]]
        if str(d["arm"]) == "null":
            assert d["naive_acc"].shape == means.shape
            d["naive_acc"] = means
        else:
            sa = d["sample_acc"].copy()
            assert sa[:, -1, :].shape == means.shape
            sa[:, -1, :] = means
            d["sample_acc"] = sa
        replaced += 1
    np.savez(SUB / p.name, **d)
print(f"conditions with replayed values: {replaced} of {len(list(EX.glob('*.npz')))}")

spec = importlib.util.spec_from_file_location("pam", ROOT / "scripts/measure/phase_a_map.py")
pam = importlib.util.module_from_spec(spec)
sys.modules["pam"] = pam
spec.loader.exec_module(pam)
pam.EXTRACT, pam.OUT = SUB, OUT
pam.main()

a = json.loads((ROOT / "runs/phase-a/map.json").read_text())["comparisons"]
b = json.loads(OUT.read_text())["comparisons"]
assert set(a) == set(b)


def walk(x, y, path, out):
    if isinstance(x, dict):
        for k in x:
            walk(x[k], y[k], path + [k], out)
    elif isinstance(x, list):
        for i, (u, v) in enumerate(zip(x, y, strict=True)):
            walk(u, v, path + [str(i)], out)
    elif isinstance(x, float) and isinstance(y, float):
        if not (x == y or (math.isnan(x) and math.isnan(y))):
            out.append((path, x, y))
    elif x != y:
        out.append((path, x, y))


print("\n================ stored (run 47) against replayed, every comparison")
for fam in ("treatment vs null", "third vs null", "treatment vs third"):
    keys = [k for k in a if a[k]["family"] == fam]
    labels_changed = [k for k in keys if a[k]["tier3"]["label"] != b[k]["tier3"]["label"]]
    sig = lambda v, o: v["tier1"][o]["p_holm"] < 0.05  # noqa: E731
    sig_changed = [(k, o) for k in keys for o in pam.OUTCOMES if sig(a[k], o) != sig(b[k], o)]
    raw_changed = [
        (k, o)
        for k in keys
        for o in pam.OUTCOMES
        if (a[k]["tier1"][o]["p"] < 0.05) != (b[k]["tier1"][o]["p"] < 0.05)
    ]
    sign_changed = [
        (k, o)
        for k in keys
        for o in pam.OUTCOMES
        if (a[k]["tier1"][o]["pos"], a[k]["tier1"][o]["neg"])
        != (b[k]["tier1"][o]["pos"], b[k]["tier1"][o]["neg"])
    ]
    diffs = []
    for k in keys:
        walk(a[k], b[k], [k], diffs)
    num = [(p, x, y) for p, x, y in diffs if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    maxz = max((abs(x - y) for p, x, y in num if p[-1] == "z" or p[-1] == "best_z"), default=0.0)
    maxiqm = max((abs(x - y) for p, x, y in num if p[-1] == "iqm_diff"), default=0.0)
    maxp = max((abs(x - y) for p, x, y in num if p[-1] in ("p", "p_holm")), default=0.0)
    print(f"\n--- {fam}: {len(keys)} comparisons, {len({tuple(p[:1]) for p, _, _ in diffs})} with any change")
    print(f"    tier 3 labels changed: {len(labels_changed)} {labels_changed}")
    print(f"    Holm-significant at 0.05 changed: {len(sig_changed)} {sig_changed}")
    print(f"    raw p < 0.05 changed: {len(raw_changed)} {raw_changed}")
    print(f"    sign-test counts changed: {len(sign_changed)} {sign_changed[:6]}")
    print(f"    largest change: z {maxz:.3g}, IQM difference {maxiqm:.3g}, p {maxp:.3g}")
    nb = [k for k in keys if a[k]["tier1"]["predated"]["p"] < 0.05 or b[k]["tier1"]["predated"]["p"] < 0.05]
    for k in nb:
        print(
            f"    predated {k.split('  vs')[0]}: p {a[k]['tier1']['predated']['p']:.4g} -> {b[k]['tier1']['predated']['p']:.4g}, "
            f"IQM {a[k]['tier1']['predated']['iqm_diff']:+.5f} -> {b[k]['tier1']['predated']['iqm_diff']:+.5f}"
        )
k0 = "treatment__baseline__None  vs  null__baseline__None"
print(
    "\nbaseline:",
    json.dumps({o: [a[k0]["tier1"][o]["p"], b[k0]["tier1"][o]["p"]] for o in pam.OUTCOMES}),
    a[k0]["tier3"]["label"],
    "->",
    b[k0]["tier3"]["label"],
)
