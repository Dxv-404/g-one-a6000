#!/usr/bin/env python3
"""Three pricing measurements for the revised freeze criterion, in the criterion's own currency.

**The currency is TIER-ONE OUTCOMES**, the predated / starved / survived fractions over 8,192 prey, because
the descriptors are degenerate: `flee_engaged` is 0.0000 in fifteen of fifteen sweep runs and `still_engaged`
maxes at 0.001. An unswept parameter cannot be measured against a descriptor effect that does not exist.
Owner's revision, 2026-08-26.

**Threshold: one seed-to-seed standard deviation of the outcome**, measured here alongside the effect rather
than assumed, so the comparison is against this configuration's own noise.

What is measured, and one of the three is not what was asked for
    grid size        `h` and `w`. Bounded by the encounter-rate invariant: at k = 8 the productive range
                     runs out just under 28, so 28 / 32 / 40 rather than the 20 the concern names
    starvation rate  `e_base`, the flat per-tick energy drain, which is `unswept.md` entry 5. Bounded on
                     both sides: `feeds` must stay in [2, 10] AND supply must exceed demand, giving
                     0.35 / 0.5 / 0.65 for feeds of 2 / 5 / 8
    grouping payoff  **NOT the requested comparison, and the substitution is deliberate.** The request was
                     nearest-prey targeting against random targeting. `select_target` uses `argmin` with no
                     switch, so that comparison needs a second rule implemented in `src/env/predator.py`
                     AND in `src/reference/naive_env.py` or gate E2 breaks, plus a mechanism flag with
                     provenance. That is a design change, not a cheap measurement, and doing it quietly
                     under this heading would be the wrong call.

                     What IS measured is the substantive half: under the CURRENT nearest rule, does per-prey
                     predation risk rise with local density? If it is flat or falls, grouping is already a
                     free lunch and the rule is not doing the work it was chosen to do. If it rises,
                     grouping costs positioning, which is what nearest targeting was for.

**The instrument is the hand-coded forager**, per D-044, because all three of these price the WORLD rather
than a policy's response to it, and a fixed policy measures the world with less compute and no confound from
evolution. Stated as a limit: an evolved lineage would move these numbers.

Usage: uv run python scripts/measure/freeze_pricing.py <out.npz>
"""

from __future__ import annotations

import dataclasses
import gc
import math
import sys
import time

import jax
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import DEFAULT, check_invariants
from src.agent import baseline
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius
from src.measure import outcomes as O

#: **96, and the seed count is the whole story of this measurement. Three earlier values are superseded.**
#:
#: 8, a pilot. 24, after 64 was OOM-killed at 11.4 GB. 96, after the owner answered the blocking question.
#:
#: **The threshold TRANSFERS, and the decomposition proves it.** The observed seed SD contains true
#: layout-to-layout variation plus binomial noise from `k` prey: `observed^2 = true^2 + p(1-p)/k`.
#:
#:     observed 0.315    binomial term 0.173    layout 0.263
#:
#: At infinite prey per seed the bar falls only from 0.315 to 0.263. **Layout variation does not shrink with
#: prey count**, so the bar is real at any sample size. The 8,192 figure in the criterion justified the
#: CURRENCY, that outcomes have power at all, rather than the threshold.
#:
#: **So the constraint is precision on the EFFECT, and it is far cheaper than 1,024 seeds.** The interval on
#: a difference of two group means is `1.96 * sd * sqrt(2 / n)`:
#:
#:     8 seeds    +/- 0.309     interval spans the bar
#:    24 seeds    +/- 0.178     interval spans the bar
#:    64 seeds    +/- 0.109     interval spans the bar
#:    96 seeds    +/- 0.089     clear of the bar
#:
#: About 93 seeds is where a 0.224 effect separates from a 0.315 bar. 96 at roughly fifty minutes an axis,
#: against the ten hours a literal reading of 8,192 prey demanded.
#:
#: **This is also why the 24-seed verdicts were withdrawn.** Grid size read 0.224 with an interval of
#: [0.046, 0.402] across a bar of 0.315: UNCLASSIFIED, not below. A point estimate under a bar its own
#: interval crosses is not a verdict, and the instability across 0.375 / 0.164 / 0.224 is what that width
#: predicts.
N_SEEDS = 96
T_HORIZON = 400

AXES: dict[str, tuple] = {
    "h": (28, 32, 40),
    "e_base": (0.35, 0.5, 0.65),
}


def rss_mb() -> float:
    """Current RSS, printed per configuration so growth is visible rather than inferred from a corpse.

    Not `ru_maxrss`, which is a high-water mark and cannot fall, so it would show a leak that had already
    been released and hide one that was still growing.
    """
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def policy_forager(params):
    def policy(weights, obs, hidden, step_index):
        del weights
        return jax.vmap(lambda o: baseline.forager(o, params, step_index))(obs), hidden

    return policy


def run(overrides: dict) -> dict:
    p = dataclasses.replace(DEFAULT, t_horizon=T_HORIZON, **overrides)
    check_invariants(p)
    policy = policy_forager(p)
    r_table = required_radius(p)

    per_seed, dens, pred_flag, pdist, seen = [], [], [], [], []
    for seed in range(N_SEEDS):
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, r_table), food_src=fs, target_src=ts)
        w = net.init_params(jax.random.key(seed), p)
        state, acc = E.run_episode(
            jax.random.key(1000 + seed), w, static, ent.PRNGSource(jax.random.key(seed)), p, policy=policy
        )
        dp = np.asarray(state.prey_died_predated)
        ds = np.asarray(state.prey_died_starving)
        al = np.asarray(state.prey_alive)
        per_seed.append(O.outcomes(dp, ds, al))
        # Mean living neighbours within r_group per alive tick, per prey. The grouping exposure.
        alive_t = np.asarray(acc.alive_ticks, dtype=np.float64)
        dens.append(np.asarray(acc.density_sum, dtype=np.float64) / np.maximum(alive_t, 1.0))
        pred_flag.append(dp.astype(np.float64))
        # **POSITION, which is what separates membership from skill.** Mean Chebyshev distance to the
        # nearest predator over the ticks that prey was visible. Undefined where it was never seen, and
        # those prey are excluded rather than imputed, with the count reported.
        vis_t = np.asarray(acc.visible_ticks, dtype=np.float64)
        pdist.append(np.asarray(acc.dist_sum_visible, dtype=np.float64) / np.maximum(vis_t, 1.0))
        seen.append(vis_t > 0)

        # **Clear the JAX caches INSIDE the loop, not only per configuration.**
        #
        # RSS ran 2.3 GB at 24 seeds and 5.2 GB at 96, which is about 30 MB per seed. That is DEV-029's
        # signature almost exactly, where the generation loop retained a compiled executable per generation
        # at 31.5 MB each. Something here is retained per seed and `gc.collect()` alone does not release it,
        # because the reference is held by JAX's compilation cache rather than by Python.
        #
        # At 268 seeds the same rate reaches roughly 14 GB, and 11.4 GB was already enough to be OOM-killed
        # once. Clearing every 32 seeds costs a recompilation, about fifteen seconds, eight times across a
        # long run. That is minutes against a run measured in hours.
        #
        # **Still a mitigation rather than a diagnosis**, and labelled as one. The likely cause is that
        # `PRNGSource` or `StaticEnv` is fresh per seed and triggers a retrace, but that has not been
        # established and no claim is made about it.
        if (seed + 1) % 32 == 0:
            jax.clear_caches()
            gc.collect()

    # **Bound the growth rather than diagnose it.** 64 seeds was OOM-killed at 11.4 GB; the per-seed
    # `StaticEnv` and its traced episode retain something. Clearing the caches costs a recompilation per
    # configuration, which is seconds against a run measured in minutes.
    jax.clear_caches()
    gc.collect()

    return {
        "predated": float(np.mean([o.predated for o in per_seed])),
        "starved": float(np.mean([o.starved for o in per_seed])),
        "survived": float(np.mean([o.survived for o in per_seed])),
        "predated_sd": float(np.std([o.predated for o in per_seed], ddof=1)),
        "starved_sd": float(np.std([o.starved for o in per_seed], ddof=1)),
        "survived_sd": float(np.std([o.survived for o in per_seed], ddof=1)),
        "density": np.concatenate(dens),
        "was_predated": np.concatenate(pred_flag),
        "pred_distance": np.concatenate(pdist),
        "was_seen": np.concatenate(seen),
    }


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    t0 = time.time()
    payload: dict = {}

    # **Axis and seed count are arguments, because the required count is PER OUTCOME and differs by axis.**
    # Grid size binds on `survived` and needs 268; `e_base` binds on `starved` and needs 201. A single
    # N_SEEDS constant would run both at the larger and waste an hour, or at the smaller and settle neither.
    global N_SEEDS
    args = sys.argv[2:]
    if "--seeds" in args:
        N_SEEDS = int(args[args.index("--seeds") + 1])
    selected = AXES
    if "--axis" in args:
        name = args[args.index("--axis") + 1]
        assert name in AXES, f"unknown axis {name!r}; known: {sorted(AXES)}"
        selected = {name: AXES[name]}
    do_grouping = "--no-grouping" not in args
    print(
        f"\naxes {sorted(selected)}, {N_SEEDS} seeds, grouping block {'on' if do_grouping else 'off'}",
        flush=True,
    )

    print("\npre-flight", flush=True)
    for axis, levels in selected.items():
        for lv in levels:
            ov = {"h": lv, "w": lv} if axis == "h" else {axis: lv}
            check_invariants(dataclasses.replace(DEFAULT, t_horizon=T_HORIZON, **ov))
    print("  every level satisfies check_invariants\n", flush=True)

    print("=" * 96, flush=True)
    print("TIER-ONE OUTCOMES, the criterion's currency. Threshold is ONE seed SD of the outcome.", flush=True)
    print("=" * 96, flush=True)
    print(
        f"{'axis':>10s} {'level':>7s} {'predated':>10s} {'starved':>9s} {'survived':>10s}   seed SDs",
        flush=True,
    )

    verdicts = []
    for axis, levels in selected.items():
        rows = []
        for lv in levels:
            ov = {"h": lv, "w": lv} if axis == "h" else {axis: lv}
            r = run(ov)
            rows.append(r)
            key = f"{axis}_{lv}"
            for q in ("predated", "starved", "survived", "predated_sd", "starved_sd", "survived_sd"):
                payload[f"{key}__{q}"] = np.float64(r[q])
            payload[f"{key}__density"] = r["density"]
            payload[f"{key}__was_predated"] = r["was_predated"]
            print(
                f"{axis:>10s} {str(lv):>7s} {r['predated']:10.4f} {r['starved']:9.4f} {r['survived']:10.4f}"
                f"   {r['predated_sd']:.4f} / {r['starved_sd']:.4f} / {r['survived_sd']:.4f}"
                f"   rss {rss_mb():.0f}MB",
                flush=True,
            )

        # **The effect carries a CONFIDENCE INTERVAL, and a point estimate below the bar is not a verdict.**
        #
        # Owner's correction, 2026-08-27. A previous version compared the point estimate against the bar and
        # reported "below". At 24 seeds grid size read 0.224 with a 95 percent interval of plus or minus
        # 0.178, so the interval ran from 0.046 to 0.402 across a bar of 0.315. That is UNCLASSIFIED, not
        # below, and the instability seen across three sample sizes, 0.375 / 0.164 / 0.224, is exactly what
        # an interval that wide predicts.
        #
        # The interval is on a difference of two group means, so `1.96 * sd * sqrt(2 / n_seeds)`.
        print(f"\n  {axis}: effect against one seed SD, WITH its interval", flush=True)
        status = "below"
        for q in ("predated", "starved", "survived"):
            vals = [r[q] for r in rows]
            effect = max(vals) - min(vals)
            sd = max(r[f"{q}_sd"] for r in rows)
            half = 1.96 * sd * math.sqrt(2.0 / N_SEEDS)
            lo, hi = effect - half, effect + half
            if hi < sd:
                verdict = "below"
            elif lo > sd:
                verdict = "ABOVE THRESHOLD"
                status = "above"
            else:
                verdict = "UNCLASSIFIED, interval spans the bar"
                status = "unclassified" if status != "above" else status
            ratio = effect / sd if sd > 0 else float("inf")
            payload[f"{axis}__{q}_effect"] = np.float64(effect)
            payload[f"{axis}__{q}_sd"] = np.float64(sd)
            payload[f"{axis}__{q}_ci_half"] = np.float64(half)
            payload[f"{axis}__{q}_ratio_to_bar"] = np.float64(ratio)
            print(
                f"    {q:<9s} effect {effect:.4f} +/- {half:.4f}  [{lo:.4f}, {hi:.4f}]  "
                f"bar {sd:.4f}  {ratio * 100:5.1f}% of bar   {verdict}",
                flush=True,
            )
        verdicts.append((axis, status))
        summary = {
            "below": "below threshold on every outcome. CLASSIFIED",
            "above": "above threshold. Must be swept or disclosed with this magnitude",
            "unclassified": "UNCLASSIFIED. At least one interval spans its bar; more seeds needed",
        }[status]
        print(f"    -> {axis}: {summary}", flush=True)

        # **The ratio to bar is printed WITH the verdict, not derivable from it. Owner's ruling, 2026-08-28.**
        #
        # "Below threshold" spans 4.9 to 80.8 percent of the bar across the four items classified so far, and
        # the same three words describe all of them. A write-up that carries only the verdict flattens a
        # sixteenfold range into one claim, and "below threshold" is not "no detectable effect": a quarter
        # more effect and neither grid size nor `e_base` would classify.
        worst = max(
            (float(payload[f"{axis}__{q}_ratio_to_bar"]) for q in ("predated", "starved", "survived")),
        )
        band = "negligible" if worst < 0.25 else "modest" if worst < 0.5 else "SUBSTANTIAL, near the bar"
        print(f"       closest outcome reaches {worst * 100:.1f}% of its bar: {band}\n", flush=True)

    # ------------------------------------------------- grouping: membership against skill ---------------
    #
    # **The previous version of this block could not answer its own question, and the owner said why.**
    #
    # It reported that risk falls with local density and read that as grouping being protective. It is, but
    # the correlation is IDENTICAL under two readings that mean opposite things for the design:
    #
    #     membership   you are diluted by being in a group of 8. Under RANDOM targeting that is 1 in 8
    #                  automatically; under NEAREST targeting it is also about 1 in 8 absent any skill
    #     skill        you are diluted by POSITIONING yourself well inside the group
    #
    # **Nearest targeting does not remove the dilution. It makes the dilution depend on position rather than
    # be automatic.** I had claimed the opposite more than once, and it does not follow.
    #
    # So the separating measurement is whether POSITION predicts risk WITHIN a density band. If risk falls
    # with density but is flat with position inside a band, the payoff is membership and grouping is a free
    # lunch after all. If position predicts risk within a band, it is skill, which is what the rule was
    # chosen to create.
    #
    # Position here is the mean Chebyshev distance to the nearest predator over the ticks a prey was
    # visible. Prey never seen have no such distance and are excluded rather than imputed.
    if not do_grouping:
        payload["wall_seconds"] = np.float32(time.time() - t0)
        np.savez(sys.argv[1], **payload)
        print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
        return 0

    print("=" * 96, flush=True)
    print("GROUPING: is the payoff MEMBERSHIP or POSITIONING? Owner's decomposition, 2026-08-27.", flush=True)
    print("=" * 96, flush=True)
    print(
        "**The position measure used here is CONTAMINATED and its verdict was withdrawn.** Capture takes",
        flush=True,
    )
    print(
        "several adjacent ticks, those are visible ticks at distance 1, and visible_ticks is small, so the",
        flush=True,
    )
    print(
        "capture window is essentially the whole mean: 89.7 percent of predated prey sit at distance <= 1.05",
        flush=True,
    )
    print(
        "against 1.16 for survivors. corr(distance, predated) is close to measuring whether the prey was",
        flush=True,
    )
    print(
        "caught. Kept so the contamination is reproducible, not because the number means anything.\n",
        flush=True,
    )

    base = run({})
    d, y = base["density"], base["was_predated"]
    dist, seen = base["pred_distance"], base["was_seen"]
    payload["grouping__density"] = d
    payload["grouping__was_predated"] = y
    payload["grouping__pred_distance"] = dist
    payload["grouping__was_seen"] = seen

    lo_q, hi_q = np.quantile(d, 0.25), np.quantile(d, 0.75)
    bottom, top = y[d <= lo_q], y[d >= hi_q]
    print(
        f"  n = {len(d)} prey. density min {d.min():.3f} median {np.median(d):.3f} max {d.max():.3f}",
        flush=True,
    )
    print(
        f"  predation rate, LOWEST density quartile  ({len(bottom):4d} prey): {bottom.mean():.4f}", flush=True
    )
    print(f"  predation rate, HIGHEST density quartile ({len(top):4d} prey): {top.mean():.4f}", flush=True)

    r_overall = float(np.corrcoef(d, y)[0, 1]) if d.std() > 0 else float("nan")
    payload["grouping__corr"] = np.float64(r_overall)
    print(f"  correlation of density with being predated: {r_overall:+.4f}", flush=True)

    print(f"\n  Prey ever seen by a predator: {int(seen.sum())} of {len(seen)}.", flush=True)
    print(
        "  Position is undefined for the rest and they are excluded from the banded test below.", flush=True
    )

    print("\n  WITHIN density terciles, does POSITION predict risk?", flush=True)
    print(
        f"    {'band':<22s} {'n':>5s} {'seen':>5s} {'predated':>9s} {'corr(dist, predated)':>22s}", flush=True
    )
    edges = [d.min() - 1e-9, np.quantile(d, 1 / 3), np.quantile(d, 2 / 3), d.max() + 1e-9]
    within = []
    for i in range(3):
        m = (d > edges[i]) & (d <= edges[i + 1])
        ms = m & seen
        label = f"{edges[i]:.2f} to {edges[i + 1]:.2f}"
        n_band, n_seen = int(m.sum()), int(ms.sum())
        rate = float(y[m].mean()) if n_band else float("nan")
        if n_seen >= 8 and dist[ms].std() > 0 and y[ms].std() > 0:
            rb = float(np.corrcoef(dist[ms], y[ms])[0, 1])
            within.append(rb)
            shown = f"{rb:+.4f}"
        else:
            shown = "too few or degenerate"
        payload[f"grouping__band{i}_rate"] = np.float64(rate)
        print(f"    {label:<22s} {n_band:5d} {n_seen:5d} {rate:9.4f} {shown:>22s}", flush=True)

    print(
        "\n  READING THIS, and both branches are stated so the answer is not chosen after the fact:",
        flush=True,
    )
    print("    density predicts risk ACROSS bands and position does not predict WITHIN them", flush=True)
    print(
        "      -> the payoff is MEMBERSHIP. Grouping is a free lunch and nearest targeting has not",
        flush=True,
    )
    print(
        "         made it a skill, which is a finding about the rule rather than about the config", flush=True
    )
    print("    position predicts risk within bands", flush=True)
    print("      -> the payoff is POSITIONING, which is what the rule was chosen to create", flush=True)
    if within:
        mean_within = float(np.mean(within))
        payload["grouping__within_band_corr"] = np.float64(mean_within)
        print(f"\n    mean within-band corr(distance, predated) = {mean_within:+.4f}", flush=True)
        print(f"    overall corr(density, predated)            = {r_overall:+.4f}", flush=True)
        if abs(mean_within) < 0.1 <= abs(r_overall):
            print("\n    -> MEMBERSHIP. Density predicts risk, position within a band does not.", flush=True)
        elif abs(mean_within) >= 0.1:
            print("\n    -> POSITIONING contributes; the rule is doing at least part of its job.", flush=True)
        else:
            print("\n    -> Neither is clear at this n. Not a verdict.", flush=True)
    else:
        print("\n    -> Not computable: too few seen prey per band. Not a verdict.", flush=True)

    payload["wall_seconds"] = np.float32(time.time() - t0)
    np.savez(sys.argv[1], **payload)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes -> {sys.argv[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
