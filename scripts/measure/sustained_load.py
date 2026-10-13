#!/usr/bin/env python3
"""Does per-generation time DEGRADE under sustained load on this machine? The sweep is five days of it.

**Why this is the dominant unknown for the sweep's wall clock.** Every timing figure this project holds is
a cold-start measurement of a few generations. The sweep runs 327,000 of them back to back. This machine has
already shown a **3.5x spread on identical work**: the engineering tier ran 9:26 and then 33:13 on the same
tree with the same result, with the GPU at 83 C and nothing else running. The cause was never established.
Thermal throttling fits the ordering and is recorded as a hypothesis rather than a finding.

**So the honest sweep estimate is not 1.1 s/gen times 327,000.** It is that, times whatever this measures.

The design is deliberately simple: run generations continuously, report per-generation time in windows, and
read the trend. A flat trace means the cold-start figure extrapolates. A rising trace means it does not, and
the ratio is the correction factor.

    flat within noise          the 100 to 126 hour estimate stands
    rises and plateaus         multiply the estimate by the plateau ratio
    rises without plateau      the run needs active cooling or duty cycling, and a five-day
                               job cannot be estimated from any short measurement

**GPU temperature and clocks are read where available**, because a thermal explanation is testable rather
than assumed: throttling shows as temperature rising to a limit and clocks falling with it. If time degrades
while temperature and clocks are flat, the cause is something else and saying so is the point.

Usage: uv run python scripts/measure/sustained_load.py [--minutes N]
"""

from __future__ import annotations

import subprocess
import sys
import time

import jax
import numpy as np

import src  # noqa: F401  runtime control, before any JAX operation
from configs.default import DEFAULT
from src.agent import network as net
from src.env import env as E
from src.env import grid, resources
from src.env.perception import build_vis_table, required_radius
from src.evo import genome, loop

MINUTES = 10.0
WINDOW = 10  # generations per reported window


def gpu_probe() -> tuple[float, float]:
    """(temperature C, graphics clock MHz), or NaN where nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.current.graphics",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return float("nan"), float("nan")
        t, c = out.stdout.strip().splitlines()[0].split(",")
        return float(t), float(c)
    except Exception:
        return float("nan"), float("nan")


def main() -> int:
    print(src.assert_runtime_took_effect(), flush=True)
    global MINUTES
    if "--minutes" in sys.argv:
        MINUTES = float(sys.argv[sys.argv.index("--minutes") + 1])

    p = DEFAULT
    seed = 0
    ob = grid.make_obstacle_grid(jax.random.key(seed), p.h, p.w, p.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, p)
    static = E.StaticEnv(
        obstacles=ob, vis_table=build_vis_table(ob, required_radius(p)), food_src=fs, target_src=ts
    )
    template = net.init_params(jax.random.key(seed), p)
    pop = loop.initial_population(jax.random.key(seed + 1), template, p)
    step = loop.make_generation_step(genome.unflattener(template), static, p)

    t0 = time.time()
    pop, fit = step(jax.random.key(0), pop)
    jax.block_until_ready(fit)
    print(f"compile {time.time() - t0:.1f}s, now running {MINUTES:.0f} minutes continuously", flush=True)
    temp0, clk0 = gpu_probe()
    print(f"start: {temp0:.0f} C, {clk0:.0f} MHz\n", flush=True)

    print(
        f"  {'window':>7s} {'gens':>6s} {'s/gen':>8s} {'vs first':>9s} {'temp C':>8s} {'MHz':>7s}", flush=True
    )

    start = time.time()
    g = 0
    windows = []
    while time.time() - start < MINUTES * 60:
        wt0 = time.time()
        for _ in range(WINDOW):
            pop, fit = step(jax.random.key(1000 + g), pop)
            g += 1
        jax.block_until_ready(fit)
        per = (time.time() - wt0) / WINDOW
        temp, clk = gpu_probe()
        windows.append((per, temp, clk))
        ratio = per / windows[0][0]
        print(f"  {len(windows):7d} {g:6d} {per:8.3f} {ratio:8.2f}x {temp:8.0f} {clk:7.0f}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("VERDICT", flush=True)
    print("=" * 90, flush=True)
    first, last = windows[0][0], windows[-1][0]
    worst = max(w[0] for w in windows)
    print(f"  generations run        {g}", flush=True)
    print(f"  first window           {first:.3f} s/gen", flush=True)
    print(f"  last window            {last:.3f} s/gen   ({last / first:.2f}x)", flush=True)
    print(f"  worst window           {worst:.3f} s/gen   ({worst / first:.2f}x)", flush=True)
    temps = [w[1] for w in windows if np.isfinite(w[1])]
    clks = [w[2] for w in windows if np.isfinite(w[2])]
    if temps:
        print(f"  temperature            {temps[0]:.0f} to {max(temps):.0f} C", flush=True)
    if clks:
        print(f"  graphics clock         {clks[0]:.0f} to {min(clks):.0f} MHz", flush=True)

    print("\n  SWEEP PROJECTION, 327,000 generations:", flush=True)
    for label, rate in (("first window", first), ("last window", last), ("worst window", worst)):
        print(
            f"    at the {label:<13s} {327000 * rate / 3600:7.1f} h = {327000 * rate / 86400:4.1f} days",
            flush=True,
        )

    degr = last / first
    if degr < 1.15:
        print(
            "\n  FLAT within noise. The cold-start figure extrapolates and the estimate stands.", flush=True
        )
    elif temps and max(temps) - temps[0] > 8 and clks and clks[0] - min(clks) > 100:
        print(
            f"\n  DEGRADES {degr:.2f}x, WITH temperature up and clocks down: thermal throttling is",
            flush=True,
        )
        print("  supported. Cooling or duty cycling would recover it.", flush=True)
    else:
        print(
            f"\n  DEGRADES {degr:.2f}x, but NOT with the thermal signature. The cause is something",
            flush=True,
        )
        print("  else and should not be attributed to heat without further evidence.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
