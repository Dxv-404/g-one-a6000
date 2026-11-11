#!/usr/bin/env python3
"""Several seeds in ONE process: is each seed bit-identical to its one-seed run, and is it faster? D-094.

Separate processes cannot share a GPU at once without MPS; they take turns, and on a free A6000 two processes were
30 percent slower in total than one (run 59). Stacking seeds inside one compiled program is the other way to use a
card one seed does not fill. This measures the two things that decide whether it is worth building into the runner:

  identity     at every batch size S, each batched seed against `loop.make_generation_step` run alone on the same
               seed: fitness every generation and the final population, bit for bit
  throughput   generations per second summed over seeds, against the runner's own one-seed step

Faithful to `sweep.one_seed`'s evolution core: the same world per seed, the same initial population, the same key
sequence, the evaluation jitted as `make_generation_step` jits it and `next_population` run eagerly as the runner
runs it. NOT included: the accumulator pass every SAMPLE_EVERY generations and the in-run S4, so the timing is the
evolution step only, which is most of a generation. Laptop RTX 4060, 2026-11-11 (run 60): identical at S = 4 over 3
generations, and no gain (0.86 to 0.97 of the runner), because one seed already fills that card.

    .venv/bin/python -u scripts/measure/batch_seeds.py runs/session2/batch [--sizes 1,2,4,8,16]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "measure"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import sweep  # noqa: E402

import src  # noqa: E402  runtime control, before any JAX operation
from src.agent import network as net  # noqa: E402
from src.env import entropy as ent  # noqa: E402
from src.env import env as E  # noqa: E402
from src.env import grid, resources  # noqa: E402
from src.env.perception import build_vis_table, required_radius  # noqa: E402
from src.evo import genome, loop  # noqa: E402
from src.evo.fitness import fitness  # noqa: E402

SEED0 = 200  # test seeds, as the concurrency check uses: not Phase A's, the long run's or the calibration's


def world(p, seed):
    """`sweep.one_seed`'s first lines, unchanged."""
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
    static = E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )
    template = net.init_params(jax.random.key(seed), p)
    unflatten = genome.unflattener(template)
    pop = loop.initial_population(jax.random.key(seed + 1), template, p)
    return static, unflatten, pop, jax.random.key(seed + 2)


def reference(p, seed, gens):
    """The runner's one-seed evolution core, as `sweep.one_seed` drives it."""
    static, unflatten, pop, key = world(p, seed)
    step = loop.make_generation_step(unflatten, static, p)
    fits = []
    for _ in range(gens):
        key, kg, _ka = jax.random.split(key, 3)
        pop, fit = step(kg, pop)
        fits.append(np.asarray(fit))
    return np.stack(fits), np.asarray(pop)


def make_batched(p, unflatten):
    def one_episode(vec, k, static):
        k_reset, k_draws = jax.random.split(k)
        state, _ = E.run_episode(k_reset, unflatten(vec), static, ent.PRNGSource(k_draws), p)
        return state.prey_score

    ev = jax.jit(
        jax.vmap(
            jax.vmap(jax.vmap(one_episode, in_axes=(None, 0, None)), in_axes=(0, 0, None)), in_axes=(0, 0, 0)
        )
    )
    n = p.p_pop * p.n_episodes

    def step(keys, pops, statics):
        ks = jax.vmap(jax.random.split)(keys)
        k_eval, k_breed = ks[:, 0], ks[:, 1]
        ekeys = jax.vmap(lambda k: jax.random.split(k, n).reshape(p.p_pop, p.n_episodes))(k_eval)
        fit = fitness(ev(pops, ekeys, statics))
        new = jax.vmap(lambda k, g, f: loop.next_population(k, g, f, p))(k_breed, pops, fit)
        return new, fit

    return step


def batched_setup(p, seeds):
    ws = [world(p, s) for s in seeds]
    statics = jax.tree.map(lambda *xs: jnp.stack(xs), *[w[0] for w in ws])
    return make_batched(p, ws[0][1]), statics, jnp.stack([w[2] for w in ws]), jnp.stack([w[3] for w in ws])


def batched_run(step, statics, pops, keys, gens):
    fits, times = [], []
    for _ in range(gens):
        t = time.perf_counter()
        k3 = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
        keys, kg = k3[:, 0], k3[:, 1]
        pops, fit = step(kg, pops, statics)
        f = np.asarray(fit)  # the runner copies fitness to the host every generation
        times.append(time.perf_counter() - t)
        fits.append(f)
    return np.stack(fits, axis=1), np.asarray(pops), keys, pops, times


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--sizes", default="1,2,4,8,16")
    ap.add_argument("--ident-gens", type=int, default=3)
    ap.add_argument("--time-gens", type=int, default=5)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    observed = src.assert_runtime_took_effect()
    p = sweep.params_for("treatment", "baseline", None)
    sizes = [int(x) for x in a.sizes.split(",")]
    res = {
        "device": jax.devices()[0].device_kind,
        "runtime": observed,
        "world": sweep.frozen_world.WORLD,
        "ident_gens": a.ident_gens,
        "time_gens": a.time_gens,
    }

    def log(m):
        print(m, flush=True)

    log(f"batch_seeds on {res['device']}, sizes {sizes}, identity over {a.ident_gens} generations")

    # The baseline: the runner's own one-seed step.
    static, unflatten, pop, key = world(p, SEED0)
    st = loop.make_generation_step(unflatten, static, p)
    ts = []
    for _ in range(a.time_gens + 1):
        t = time.perf_counter()
        key, kg, _ = jax.random.split(key, 3)
        pop, fit = st(kg, pop)
        np.asarray(fit)
        ts.append(time.perf_counter() - t)
    base = 1.0 / float(np.median(ts[1:]))
    res["runner"] = {"first_call_s": ts[0], "steps_s": ts[1:], "gen_per_s": base}
    log(f"runner one-seed step: {np.median(ts[1:]):.3f} s per generation, {base:.3f} gen/s")

    refs: dict[int, tuple] = {}
    rows = {}
    for S in sizes:
        seeds = list(range(SEED0, SEED0 + S))
        for s in seeds:
            if s not in refs:
                refs[s] = reference(p, s, a.ident_gens)
        step, statics, pops, keys = batched_setup(p, seeds)
        t0 = time.perf_counter()
        bf, bp, keys, pops_dev, its = batched_run(step, statics, pops, keys, a.ident_gens)
        first = time.perf_counter() - t0
        ident = []
        for i, s in enumerate(seeds):
            rf, rp = refs[s]
            ident.append(
                {
                    "seed": s,
                    "fit_identical": bool(np.array_equal(rf, bf[i])),
                    "pop_identical": bool(np.array_equal(rp, bp[i])),
                    "genomes_differing": int((rp != bp[i]).any(axis=1).sum()),
                }
            )
        all_ident = all(r["fit_identical"] and r["pop_identical"] for r in ident)
        _, _, _, _, tts = batched_run(step, statics, pops_dev, keys, a.time_gens)
        sec = float(np.median(tts))
        rows[str(S)] = {
            "seeds": seeds,
            "identical": all_ident,
            "identity": ident,
            "identity_phase_s": first,
            "steps_s": tts,
            "median_s_per_step": sec,
            "gen_per_s_total": S / sec,
            "speedup_vs_runner": (S / sec) / base,
        }
        log(
            f"S = {S:2d}: identical {all_ident}, {sec:.3f} s per step, {S / sec:.3f} gen/s total, "
            f"x{(S / sec) / base:.2f} the runner"
        )
        del step, statics, pops, pops_dev, keys
        jax.clear_caches()
    res["batched"] = rows
    best = max(rows.items(), key=lambda kv: kv[1]["gen_per_s_total"] if kv[1]["identical"] else -1)
    res["best_identical"] = {
        "S": int(best[0]),
        **{k: best[1][k] for k in ("gen_per_s_total", "speedup_vs_runner")},
    }
    log(
        f"SUMMARY: the best bit-identical batch is S = {best[0]}, x{best[1]['speedup_vs_runner']:.2f} the runner's "
        f"one-seed step ({best[1]['gen_per_s_total']:.3f} against {base:.3f} gen/s). Any S not identical is reported "
        "above and is not a candidate."
    )
    (a.out_dir / "batch_seeds.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
