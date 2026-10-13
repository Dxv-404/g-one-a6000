"""Does hiding's saturated baseline actually COST detection power, or is it merely redundant?

The script's canned verdict said "construction failure, no sample size repairs it". That is the
obvious reading and it may be wrong, which is why this checks it.

**The argument it needs to survive.** `conditional_occlusion = near - baseline`. If the baseline is a
CONSTANT across prey, then subtracting it shifts every value by the same amount, and a z-score against
the null is invariant under a constant shift. In that case the saturation makes the conditional form
REDUNDANT with the raw term but costs no power at all, and hiding's null is a power problem like
fleeing's rather than a construction problem.

The saturation only destroys power if the baseline VARIES per prey in a way correlated with the near
term, because then the subtraction removes real signal.

    baseline SD ~ 0                z(conditional) == z(near). Redundant, not broken. Hiding's null
                                   is power, and the fix is the same conversation as B-13
    baseline SD comparable to
    the near term's SD             the subtraction injects noise and the conditional form is worse
                                   than the raw term. THEN it is a construction failure
"""

import dataclasses
import sys

import jax
import numpy as np

import src  # noqa: F401
from configs.default import DEFAULT
from src.agent import network as net
from src.env import entropy as ent
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius

sys.path.insert(0, "scripts/measure")
from planted_policies import _make  # noqa: E402

N_SEEDS = 16


def per_prey(kind, p, n):
    pol = _make(kind)
    pol.params_ref[0] = p
    near, base = [], []
    for seed in range(n):
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(
            obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
        )
        w = net.init_params(jax.random.key(seed), p)
        _, acc = E.run_episode(
            jax.random.key(1000 + seed), w, static, ent.PRNGSource(jax.random.key(seed)), p, policy=pol
        )
        on = np.asarray(acc.occluded_near, float)
        nt = np.asarray(acc.near_ticks, float)
        of = np.asarray(acc.occluded_far, float)
        ft = np.asarray(acc.far_ticks, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            near.append(np.where(nt > 0, on / np.where(nt > 0, nt, 1), np.nan))
            base.append(np.where(ft > 0, of / np.where(ft > 0, ft, 1), np.nan))
        jax.clear_caches()
    return np.concatenate(near), np.concatenate(base)


def main():
    print(src.assert_runtime_took_effect(), flush=True)
    p = dataclasses.replace(DEFAULT, t_horizon=400)

    fn, fb = per_prey("forager", p, N_SEEDS)
    hn, hb = per_prey("hider", p, N_SEEDS)

    ok_f = np.isfinite(fn) & np.isfinite(fb)
    ok_h = np.isfinite(hn) & np.isfinite(hb)
    fn, fb, hn, hb = fn[ok_f], fb[ok_f], hn[ok_h], hb[ok_h]
    fc, hc = fn - fb, hn - hb

    print("\n" + "=" * 96, flush=True)
    print("PER-PREY SPREAD. The baseline's variability is what decides it.", flush=True)
    print("=" * 96, flush=True)
    print(
        f"  forager  occlusion_near      mean {fn.mean():7.4f}  sd {fn.std():7.4f}  n {len(fn)}", flush=True
    )
    print(f"  forager  occlusion_baseline  mean {fb.mean():7.4f}  sd {fb.std():7.4f}", flush=True)
    print(f"  forager  conditional         mean {fc.mean():7.4f}  sd {fc.std():7.4f}", flush=True)
    print(
        f"  hider    occlusion_near      mean {hn.mean():7.4f}  sd {hn.std():7.4f}  n {len(hn)}", flush=True
    )
    print(f"  hider    conditional         mean {hc.mean():7.4f}  sd {hc.std():7.4f}", flush=True)

    z_near = (hn.mean() - fn.mean()) / fn.std() if fn.std() > 0 else float("nan")
    z_cond = (hc.mean() - fc.mean()) / fc.std() if fc.std() > 0 else float("nan")
    print("\n" + "=" * 96, flush=True)
    print("DETECTION POWER, planted hider against the forager null", flush=True)
    print("=" * 96, flush=True)
    print(f"  z on the RAW term  occlusion_near      {z_near:+7.3f}", flush=True)
    print(f"  z on the CONDITIONAL conditional_occl  {z_cond:+7.3f}", flush=True)
    print(f"  difference                             {z_cond - z_near:+7.3f}", flush=True)
    print(f"\n  baseline sd / near sd = {fb.std() / fn.std():.4f}", flush=True)
    if abs(z_cond - z_near) < 0.05:
        print(
            "\n  The two are the SAME. The saturated baseline is a constant shift, so the conditional",
            flush=True,
        )
        print(
            "  form is REDUNDANT with the raw term and costs no power. Hiding's null is a POWER", flush=True
        )
        print("  problem, the same class as fleeing at B-13, not a construction failure.", flush=True)
    else:
        print(
            "\n  The two DIFFER. The baseline varies enough to change the answer, so the subtraction",
            flush=True,
        )
        print("  is doing real work and the construction reading stands.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
