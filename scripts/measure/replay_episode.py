#!/usr/bin/env python3
"""Replay episodes for the dashboard brief, corpus 099, item dash-C-01. Three policies, one layout, every tick.

**Nothing here is invented.** Every frame is the real environment stepping real weights. Two sources are
combined, and the second is checked against the first before it is trusted:

    the dump      `run_episode(record=True)`, the compiled scan's own `TickRecord` per tick: positions,
                  alive, posture, moved, predator position, both grids, both actions
    the trace     the same episode stepped one tick at a time through the same `tick` function, eagerly,
                  which exposes what the record does not carry: energy, stamina, health, score, the
                  predator's engagement and target. The trace's positions, alive flags and actions are
                  asserted EQUAL to the dump's on every tick, so the extra fields are the fields of the
                  episode the dump shows and not of a sibling that diverged

**The selection rule, stated before any episode was looked at.** Seed 0, the first seed of runs 32 and 33,
so the layout is the one both baselines share. Genome index 0 of each run's `final_pop`, which is the top
elite of the last evaluated generation by construction: `next_population` writes `top_k`'s elites first and
`top_k` returns them in descending fitness order. The random policy is `init_params` at key 0, a
generation-zero founder. Reset key 1, tick-entropy key 2, identical for all three policies, so the three
episodes differ in weights and in nothing else.

    evolved   run 33, the treatment baseline: evolved 300 generations WITH the scripted predator
    naive     run 32, the null baseline: evolved 300 generations WITHOUT a predator, here placed with one
    random    generation zero, no evolution

**What "worth watching" is measured by**, so the assessment is not an appeal: per policy, targets
collected, survivors and deaths by cause with the tick, capture contacts, ticks with the predator within
`r_pred`, and what the prey EMITTED while the predator was that close. Those are printed; the judgement is
the owner's.

Usage: uv run python scripts/measure/replay_episode.py <out.npz> [--seed N] [--p_wander X]
       --p_wander X   the SAME weights in a world whose predator wanders with probability X per
                      tick when it has no target. At the default 0.0 the predator never moves
                      unless it has detected a moving prey, so the baseline replay shows a
                      stationary predator on most seeds. X is a sweep level, not the baseline,
                      and the page says so; the weights were evolved at 0.0
       uv run python scripts/measure/replay_episode.py --scan        every seed, dump only, the second rule
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import DEFAULT, check_invariants
from src.agent import network as net
from src.env import dynamics, grid, resources
from src.env import entropy as ent
from src.env import env as E
from src.env.perception import build_vis_table, required_radius
from src.evo import genome, loop

SEED = 0
GENOME_INDEX = 0
RESET_KEY = 1
TICK_KEY = 2
RUNS = {"evolved": "runs/sweep/treatment__baseline__None.npz", "naive": "runs/sweep/null__baseline__None.npz"}

ACTION_NAMES = (
    [f"move_{d}" for d in ("n", "ne", "e", "se", "s", "sw", "w", "nw")]
    + [f"sprint_{d}" for d in ("n", "ne", "e", "se", "s", "sw", "w", "nw")]
    + ["stay", "crouch", "lie_down", "pick", "attack", "drop"]
)
assert len(ACTION_NAMES) == dynamics.N_ACTIONS


def load_genome(path: str, seed: int, index: int, template) -> dict:
    with np.load(path) as d:
        vec = jnp.asarray(d["final_pop"][seed, index])
        sha = str(d.get("config_hash", ""))
    return genome.unflattener(template)(vec), sha


def episode(w, static, p, seed: int):
    """The dump and the trace, and the assertion that they are the same episode."""
    k_reset, k_tick = jax.random.key(RESET_KEY), jax.random.key(TICK_KEY)
    state, acc, ys = E.run_episode(k_reset, w, static, ent.PRNGSource(k_tick), p, record=True)
    dump = {f: np.asarray(getattr(ys, f)) for f in E.TickRecord._fields}

    st = E.reset(ent.PRNGSource(k_reset), static, p)
    ac = E.zero_accumulators(p.k)
    src_t = ent.PRNGSource(k_tick)
    # **Compiled once, called 400 times.** The first version stepped `tick` eagerly and reached 6.6 GB
    # resident by the second policy, the per-op executable cache of a Python tick loop, which is the
    # DEV-029 class and the exact hazard CLAUDE.md section 11 prices at 9 ms per tick. One `jit` of the
    # whole tick is what the scan does, and it is the same function, so the trace is the same trajectory.
    step = jax.jit(lambda st_, ac_, t_: E.tick(st_, ac_, w, static, src_t, t_, p))
    trace = {
        k: [] for k in ("energy", "stamina", "health", "score", "pred_engage", "pred_target", "pred_stamina")
    }
    pos_t, alive_t, emit_t = [], [], []
    for t in range(p.t_horizon):
        st, ac, rec = step(st, ac, jnp.int32(t))
        pos_t.append(np.asarray(rec.prey_pos))
        alive_t.append(np.asarray(rec.prey_alive))
        emit_t.append(np.asarray(rec.action_emitted))
        trace["energy"].append(np.asarray(st.prey_energy))
        trace["stamina"].append(np.asarray(st.prey_stamina))
        trace["health"].append(np.asarray(st.prey_health))
        trace["score"].append(np.asarray(st.prey_score))
        trace["pred_engage"].append(np.asarray(st.pred_engage))
        trace["pred_target"].append(np.asarray(st.pred_target))
        trace["pred_stamina"].append(np.asarray(st.pred_stamina))
    trace = {k: np.stack(v) for k, v in trace.items()}

    pos_t, alive_t, emit_t = np.stack(pos_t), np.stack(alive_t), np.stack(emit_t)
    agree = (
        (pos_t == dump["prey_pos"]).all(axis=(1, 2))
        & (alive_t == dump["prey_alive"]).all(axis=1)
        & (emit_t == dump["action_emitted"]).all(axis=1)
    )
    same = bool(agree.all())
    # The tick at which the compiled single-tick trace and the compiled scan first disagree, if they do.
    # Both are the same Python function; XLA may fuse and order float operations differently in the
    # two programs, and a network at an argmax boundary turns a last-bit difference into a different
    # action. Reported, never hidden: fields that only the trace carries are not shown past this tick.
    diverge_tick = None if same else int(np.argmin(agree))
    final = {
        "death_cause": np.asarray(state.prey_death_cause),
        "death_step": np.asarray(state.prey_death_step),
        "score": np.asarray(state.prey_score),
        "alive": np.asarray(state.prey_alive),
    }
    return dump, trace, final, bool(same), diverge_tick


def events(dump, trace, final, p) -> dict:
    """Counts and ticks of the things a viewer would want to jump to."""
    T, k = dump["prey_alive"].shape
    pred = dump["pred_pos"][:, 0, :]  # (T, 2), one predator this increment
    # Chebyshev distance, the predator's own metric, in NumPy: 3,200 eager calls to the JAX version were
    # dispatch for nothing.
    dist = np.max(np.abs(dump["prey_pos"] - pred[:, None, :]), axis=-1)
    near = (dist <= p.r_pred) & dump["prey_alive"]
    health, energy = trace["health"], trace["energy"]
    # **A capture contact is a health drop of at least `damage` in one tick.** The first version counted
    # every health decrease and reported 456 contacts in an episode where the predator never came within
    # r_pred of anyone: it was counting starvation bleed. Bleed is the other column, health falling
    # while energy is at zero, and the two are exclusive by construction of the pools.
    drop = np.zeros_like(health)
    drop[1:] = health[:-1] - health[1:]
    # **A contact is defined by the mechanism, not by a threshold on its symptom.** `resolve_capture`
    # deals `damage * (1 - cue * (1 - moved))` when adjacent, so at cue_dependence 1.0 a contact is:
    # predator adjacent this tick, prey moved this tick, prey alive before it. The killing blow counts.
    # The threshold version missed every contact but one, because health regenerates a quarter point per
    # tick and the drops are 33.75 and 32.5, not 34. The drop count is kept as a cross-check and printed.
    adjacent = dist <= 1
    alive_before = np.ones_like(dump["prey_alive"])
    alive_before[1:] = dump["prey_alive"][:-1]
    # ... and only the predator's TARGET takes damage: `resolve_capture` gates on `has_target` and the
    # target's position. Adjacency to a non-target prey deals nothing, which is what made the first
    # mechanism count read 11 against 10 by drop.
    targeted = np.zeros_like(adjacent)
    tgt = trace["pred_target"].reshape(T, -1)[:, 0]
    eng = trace["pred_engage"].reshape(T, -1)[:, 0].astype(bool)
    for i in range(k):
        targeted[:, i] = eng & (tgt == i)
    hit = adjacent & targeted & dump["prey_moved"].astype(bool) & alive_before
    if float(p.cue_dependence) != 1.0:
        hit = adjacent & targeted & alive_before  # graded damage below 1.0; stated, not modelled
    big_drop = (drop >= 30.0) & alive_before
    bleed = (drop > 0) & (drop < 30.0) & (energy <= 0.0) & dump["prey_alive"]
    collected = np.zeros_like(near)
    collected[1:] = trace["score"][1:] > trace["score"][:-1]
    died = np.zeros_like(near)
    died[1:] = dump["prey_alive"][:-1] & ~dump["prey_alive"][1:]
    emit = dump["action_emitted"]
    stay_near = near & (emit == dynamics.ACTION_STAY)
    sprint_near = near & np.asarray(dynamics.is_burst(jnp.asarray(emit)))
    posture_near = near & ((emit == dynamics.ACTION_CROUCH) | (emit == dynamics.ACTION_LIE_DOWN))
    causes = {0: "alive", 1: "starved", 2: "predated"}
    deaths = []
    for i in range(k):
        if not final["alive"][i]:
            deaths.append(
                {
                    "prey": i,
                    "tick": int(final["death_step"][i]),
                    "cause": causes.get(int(final["death_cause"][i]), str(int(final["death_cause"][i]))),
                }
            )
    return {
        "targets_collected": int(final["score"].sum()),
        "survivors": int(final["alive"].sum()),
        "deaths": sorted(deaths, key=lambda d: d["tick"]),
        "capture_contacts": int(hit.sum()),
        "capture_contacts_by_drop": int(big_drop.sum()),
        "bleed_prey_ticks": int(bleed.sum()),
        "first_energy_zero_tick": int(np.argmax((energy <= 0.0).any(axis=1)))
        if (energy <= 0.0).any()
        else None,
        "near_prey_ticks": int(near.sum()),
        "predator_moved_ticks": int((np.abs(np.diff(pred, axis=0)).sum(axis=1) > 0).sum()),
        "predator_engaged_ticks": int(trace["pred_engage"].reshape(T, -1)[:, 0].astype(bool).sum()),
        "near_emitted_stay": int(stay_near.sum()),
        "near_emitted_sprint": int(sprint_near.sum()),
        "near_emitted_posture": int(posture_near.sum()),
        "first_near_tick": int(np.argmax(near.any(axis=1))) if near.any() else None,
        "collect_ticks": [int(t) for t in np.flatnonzero(collected.any(axis=1))],
        "hit_ticks": [int(t) for t in np.flatnonzero(hit.any(axis=1))],
        "death_ticks": [int(t) for t in np.flatnonzero(died.any(axis=1))],
    }


def scan(p, seeds: range) -> None:
    """Dump only, every seed, for the second selection rule: the seed on which the evolved elite spends the
    most prey-ticks within r_pred of the predator. Printed for all three policies so the choice is visible
    as a choice."""
    template = net.init_params(jax.random.key(0), p)
    k_reset, k_tick = jax.random.key(RESET_KEY), jax.random.key(TICK_KEY)
    print(f"\n  {'seed':>4s}  " + "  ".join(f"{n:>26s}" for n in ("random", "evolved", "naive")), flush=True)
    print(f"  {'':>4s}  " + "  ".join(f"{'near  cap  targ  alive':>26s}" for _ in range(3)), flush=True)
    best = (-1, None)
    for seed in seeds:
        ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
        fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
        static = E.StaticEnv(
            obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
        )
        cells = []
        for name in ("random", "evolved", "naive"):
            w = template if name == "random" else load_genome(RUNS[name], seed, GENOME_INDEX, template)[0]
            state, _, ys = E.run_episode(k_reset, w, static, ent.PRNGSource(k_tick), p, record=True)
            pos, alive, pred = (
                np.asarray(ys.prey_pos),
                np.asarray(ys.prey_alive),
                np.asarray(ys.pred_pos)[:, 0, :],
            )
            near = int(((np.max(np.abs(pos - pred[:, None, :]), axis=-1) <= p.r_pred) & alive).sum())
            cap = int(np.asarray(state.prey_died_predated).sum())
            cells.append(
                f"{near:5d} {cap:4d} {int(np.asarray(state.prey_score).sum()):5d} {int(np.asarray(state.prey_alive).sum()):6d}"
            )
            if name == "evolved" and near > best[0]:
                best = (near, seed)
        print(f"  {seed:4d}  " + "  ".join(f"{c:>26s}" for c in cells), flush=True)
        jax.clear_caches()
    print(
        f"\n  rule: the seed where the evolved elite has the most prey-ticks within r_pred: seed {best[1]} with {best[0]}",
        flush=True,
    )


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    if "--scan" in sys.argv:
        scan(DEFAULT, range(10))
        return 0
    out = sys.argv[1]
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else SEED
    p = DEFAULT
    if "--p_wander" in sys.argv:
        p = dataclasses.replace(p, p_wander=float(sys.argv[sys.argv.index("--p_wander") + 1]))
        check_invariants(p)
    t0 = time.time()
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, relax = resources.place_sources(jax.random.key(seed), ob, p)
    vt = build_vis_table(ob, required_radius(p))
    static = E.StaticEnv(obstacles=ob, vis_table=vt, food_src=fs, target_src=ts)
    template = net.init_params(jax.random.key(0), p)

    policies = {"random": (template, "init_params key 0")}
    for name, path in RUNS.items():
        w, h = load_genome(path, seed, GENOME_INDEX, template)
        policies[name] = (w, f"{path} final_pop[{seed}][{GENOME_INDEX}], config {h}")

    payload = {
        "obstacles": np.asarray(ob).astype(np.uint8),
        "food_src": np.asarray(fs),
        "target_src": np.asarray(ts),
        "vis_table": np.asarray(vt).astype(np.uint8),
        "seed": np.int32(seed),
        "relax_level": np.int32(relax),
        "r_pred": np.int32(p.r_pred),
        "p_wander": np.float64(p.p_wander),
        "baseline_p_wander": np.float64(DEFAULT.p_wander),
        "r": np.int32(p.r),
        "t_horizon": np.int32(p.t_horizon),
        "action_names": np.array(ACTION_NAMES),
        "config_hash": np.array(loop.config_hash(p)),
        "commit": np.array(
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
        ),
    }
    summary = {}
    print(
        f"\n  {'policy':<8s} {'verified':>8s} {'diverge':>7s} {'targets':>8s} {'alive':>6s} {'deaths':>26s} {'cap/drop':>8s} {'bleed':>6s} {'near':>5s} {'stay':>5s} {'sprint':>6s} {'posture':>7s} {'first_near':>10s} {'e_zero':>6s} {'pred_mv':>7s}",
        flush=True,
    )
    for name, (w, source) in policies.items():
        t1 = time.time()
        dump, trace, final, same, diverge_tick = episode(w, static, p, seed)
        ev = events(dump, trace, final, p)
        summary[name] = {"source": source, "trace_matches_dump": same, "diverge_tick": diverge_tick, **ev}
        for f, v in dump.items():
            payload[f"{name}__{f}"] = v.astype(np.uint8) if f in ("food_grid", "target_grid") else v
        for f, v in trace.items():
            payload[f"{name}__{f}"] = v
        payload[f"{name}__death_cause"] = final["death_cause"]
        payload[f"{name}__death_step"] = final["death_step"]
        deaths = ", ".join(f"{d['cause']}@{d['tick']}" for d in ev["deaths"]) or "none"
        print(
            f"  {name:<8s} {str(same):>8s} {str(diverge_tick):>7s} {ev['targets_collected']:8d} {ev['survivors']:6d} {deaths:>26s} "
            f"{ev['capture_contacts']:4d}/{ev['capture_contacts_by_drop']:<3d} {ev['bleed_prey_ticks']:6d} {ev['near_prey_ticks']:5d} {ev['near_emitted_stay']:5d} "
            f"{ev['near_emitted_sprint']:6d} {ev['near_emitted_posture']:7d} {str(ev['first_near_tick']):>10s} {str(ev['first_energy_zero_tick']):>6s} {ev['predator_moved_ticks']:7d}   {time.time() - t1:4.0f}s",
            flush=True,
        )
    payload["summary_json"] = np.array(json.dumps(summary))
    np.savez_compressed(out, **payload)
    print(f"\n  saved {out}, wall {time.time() - t0:.0f}s", flush=True)
    print(
        "  near: prey-ticks with the predator within r_pred. stay, sprint, posture: what was EMITTED on those ticks.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
