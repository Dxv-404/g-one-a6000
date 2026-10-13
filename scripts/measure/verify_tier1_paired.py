"""Re-analyse tier 1 from the SAVED per-seed outcomes. No re-simulation.

**Two defects in the reported verdict, both mine, and both inflate the floor.**

1. **The comparison is UNPAIRED and the design is paired.** Every policy ran on seeds 0..15 with
   identical layouts, which `test_every_arm_shares_the_layout_at_a_given_condition_and_seed` asserts.
   D-045 ruled that any comparison over arms sharing seeds is stated as a PAIRED test, because the
   seed-to-seed spread is mostly a property of the seed. I compared group means against the unpaired
   spread, which is the exact reading D-045 forbids, in a script written after it.

2. **The floor is a standard DEVIATION where the question needs a standard ERROR.** The reported
   difference is a difference of MEANS, so its uncertainty is SD/sqrt(n), not SD. Using SD overstates
   the floor by a factor of 4 at 16 seeds.

Together those two make the printed ratios roughly an order of magnitude too small, and the verdict
"TIER 1 SEPARATES NOTHING, stop-the-sweep class" is an artifact of the instrument rather than a
finding about the tier.

Recomputed here three ways, so the reader can see what each choice does rather than take one number.
"""

import numpy as np

d = np.load("/home/dev/project/runs/tier_calibration.npz", allow_pickle=True)
per_seed = d["per_seed"]  # (policy, seed, 3)
policies = [str(x) for x in d["policies"]]
n_seeds = int(d["n_seeds"])
comps = ["predated", "starved", "survived"]

f_idx = policies.index("forager")
F = per_seed[f_idx]  # (seed, 3)

print(f"policies {policies}, seeds {n_seeds}")
print(f"forager per-seed SD      {F.std(0, ddof=1).round(4)}")
print(f"forager SE of the mean   {(F.std(0, ddof=1) / np.sqrt(n_seeds)).round(4)}")
print()
print("=" * 104)
print("THREE READINGS OF THE SAME DATA. Only the third is the one D-045 requires.")
print("=" * 104)
hdr = f"{'policy':<11s}"
for c in comps:
    hdr += f"{'d_' + c:>12s}"
hdr += f"{'unpaired/SD':>13s}{'unpaired/SE':>13s}{'PAIRED t':>10s}{'paired p<0.05':>15s}"
print(hdr)

results = {}
for i, name in enumerate(policies):
    if name == "forager":
        continue
    P = per_seed[i]
    diff = P.mean(0) - F.mean(0)

    sd = F.std(0, ddof=1)
    se = sd / np.sqrt(n_seeds)
    r_sd = float(np.max(np.abs(diff) / np.maximum(sd, 1e-12)))
    r_se = float(np.max(np.abs(diff) / np.maximum(se, 1e-12)))

    # the paired reading: per-seed differences, which removes the layout variance entirely
    dpair = P - F  # (seed, 3)
    m = dpair.mean(0)
    s = dpair.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(s > 0, m / (s / np.sqrt(n_seeds)), np.where(np.abs(m) > 0, np.inf, 0.0))
    tmax = float(np.max(np.abs(t)))
    # two-sided 5 percent for 15 degrees of freedom is 2.131
    sig = tmax >= 2.131
    results[name] = dict(diff=diff, paired_t=t, tmax=tmax, sig=sig)

    row = f"{name:<11s}"
    for j in range(3):
        row += f"{diff[j]:+12.4f}"
    row += f"{r_sd:13.2f}{r_se:13.2f}{tmax:10.2f}{'YES' if sig else 'no':>15s}"
    print(row)

print()
print("=" * 104)
print("PER-COMPONENT PAIRED t, which names WHICH outcome each behaviour moves")
print("=" * 104)
print(f"{'policy':<11s}" + "".join(f"{c:>14s}" for c in comps))
for name, r in results.items():
    print(f"{name:<11s}" + "".join(f"{r['paired_t'][j]:14.2f}" for j in range(3)))

n_sig = sum(1 for r in results.values() if r["sig"])
print()
print(f"paired verdict: {n_sig} of {len(results)} planted policies differ from the forager at p<0.05")
print("(two-sided, 15 degrees of freedom, critical |t| = 2.131)")
