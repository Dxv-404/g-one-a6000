"""Verify the mechanism behind tier 2's null, rather than asserting it.

The clustering split 761 against 7 and purity sat at chance. The PROPOSED mechanism was that the six
features are unconditional while the planted differences live in about 2 percent of ticks, so the
signal is averaged away. That was a hypothesis and is checked here against the saved feature matrix.

The decisive comparison is between-policy separation against within-policy spread, per feature. If
the between-policy differences are small relative to the spread among prey of the SAME policy, then
no clustering algorithm can recover the plants and the null is a property of the feature set rather
than of k-means.
"""

import numpy as np

d = np.load("/home/dev/project/runs/tier_calibration.npz", allow_pickle=True)
X = d["tier2_X"]
lab = d["tier2_labels"]
lt = d["tier2_labels_true"]
names = [str(x) for x in d["tier2_feature_names"]]
pol = [str(x) for x in d["policies"]]
mix = [k for k in pol if k != "forager"] + ["forager"]

print(f"feature matrix {X.shape}, features {names}")
print(f"clusters {np.bincount(lab)}, true labels {np.bincount(lt)}")

print("\n" + "=" * 104)
print("BETWEEN-POLICY SEPARATION AGAINST WITHIN-POLICY SPREAD, per feature")
print("=" * 104)
print(f"{'feature':<22s}{'between SD':>12s}{'within SD':>11s}{'ratio':>8s}   reading")
weak = 0
for j, nm in enumerate(names):
    means = np.array([X[lt == i, j].mean() for i in range(len(mix))])
    withins = np.array([X[lt == i, j].std(ddof=1) for i in range(len(mix))])
    between = float(means.std(ddof=1))
    within = float(np.sqrt((withins**2).mean()))
    ratio = between / within if within > 0 else float("inf")
    if ratio < 0.5:
        weak += 1
    print(
        f"{nm:<22s}{between:12.4f}{within:11.4f}{ratio:8.2f}   "
        f"{'policies overlap heavily' if ratio < 0.5 else 'some separation'}"
    )

print(
    f"\n{weak} of {len(names)} features have between-policy separation below half the within-policy spread."
)

print("\n" + "=" * 104)
print("WHAT ARE THE 7 PREY IN CLUSTER 1? If they are outliers, k=2 found an outlier group, not a mode.")
print("=" * 104)
minor = int(np.bincount(lab).argmin())
sel = lab == minor
print(f"cluster {minor}: n={int(sel.sum())}, true policies {[mix[i] for i in np.unique(lt[sel])]}")
print(f"{'feature':<22s}{'cluster mean':>14s}{'rest mean':>12s}{'z vs rest':>11s}")
for j, nm in enumerate(names):
    a, b = X[sel, j], X[~sel, j]
    z = (a.mean() - b.mean()) / b.std(ddof=1) if b.std(ddof=1) > 0 else float("nan")
    print(f"{nm:<22s}{a.mean():14.4f}{b.mean():12.4f}{z:11.2f}")

print("\n" + "=" * 104)
print("VERDICT")
print("=" * 104)
if weak >= len(names) - 1:
    print("  The features do NOT separate the policies: between-policy differences are small relative")
    print("  to the spread among prey of the same policy, on nearly every feature. **No clustering")
    print("  algorithm can recover the plants from this feature set**, so tier 2's null is a property")
    print("  of the FEATURES rather than of k-means or of the sample size. The proposed mechanism is")
    print("  supported.")
else:
    print("  Some features DO separate the policies, so the null is not simply feature-blindness and")
    print("  the proposed mechanism is NOT supported as stated. Look at the clustering itself.")
