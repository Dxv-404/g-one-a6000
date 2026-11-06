"""Run 51. Is 1,500 generations enough? D-073's estimate re-checked against every run's fitness curve.

    uv run python scripts/measure/budget_check.py

Reads ONLY fitness (fit_mean, the population's mean prey_score per generation), never a behaviour descriptor or
an outcome, so the budget cannot be steered by what the behaviour looks like (D-073's reason for a fixed budget).

Method, fixed before the curves were looked at:

    two saturating models, fitted per run by least squares, the shape parameter on a log grid and the two linear
    parameters solved exactly at each grid point:
        fast    f(g) = a - b exp(-g / tau)        improvement fades geometrically
        slow    f(g) = a - b / (1 + g / c)        improvement fades like 1/g, a much longer tail
    a is the plateau. The fraction of the eventual gain reached by G generations is (f(G) - f(0)) / (a - f(0)).
    A fit whose shape parameter lands on the edge of the grid has no identifiable plateau and is counted as such,
    never reported as a number.

    VALIDATION FIRST. Run 1 (runs/convergence-probe, 900 generations, 3 seeds, an earlier and smaller
    configuration) is fitted on generations 0 to 299 only, and each model's forecast for 300 to 899 is compared
    with what happened. A model that fails that test is reported as failing, and its sweep numbers are not used.

Writes runs/budget-check/summary.json.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
EX = ROOT / "runs" / "a6000-handoff" / "extract"
PROBE = ROOT / "runs" / "convergence-probe"
OUT = ROOT / "runs" / "budget-check" / "summary.json"
HORIZONS = (300, 900, 1500, 2100, 3000)
GRID = np.geomspace(5.0, 1e5, 400)


def fit(y: np.ndarray, model: str) -> dict:
    g = np.arange(y.size, dtype=np.float64)
    best = None
    for s in GRID:
        x = np.exp(-g / s) if model == "fast" else 1.0 / (1.0 + g / s)
        A = np.stack([np.ones_like(x), -x], axis=1)
        coef, *_ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
        sse = float(np.sum((A @ coef - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, s, coef)
    sse, s, (a, b) = best
    edge = s <= GRID[1] or s >= GRID[-2]
    return {"a": float(a), "b": float(b), "shape": float(s), "edge": bool(edge), "sse": sse}


def predict(p: dict, g, model: str):
    g = np.asarray(g, np.float64)
    x = np.exp(-g / p["shape"]) if model == "fast" else 1.0 / (1.0 + g / p["shape"])
    return p["a"] - p["b"] * x


def fraction(p: dict, G: int, model: str) -> float:
    f0 = predict(p, 0, model)
    if p["b"] <= 0 or p["edge"]:
        return float("nan")
    return float((predict(p, G, model) - f0) / (p["a"] - f0))


def main() -> None:
    out = {"models": ["fast", "slow"], "horizons": HORIZONS}

    # ---------------------------------------------------------------- validation on run 1's 900 generations
    print("VALIDATION on run 1 (900 generations, 3 seeds): fit on 0-299, forecast 300-899")
    val = {}
    for f in sorted(PROBE.glob("seed_*.npz")):
        y = np.load(f)["fit_mean"].astype(np.float64)
        row = {}
        for m in ("fast", "slow"):
            p = fit(y[:300], m)
            pred = predict(p, np.arange(300, 900), m)
            actual_gain = float(y[600:900].mean() - y[200:300].mean())
            pred_gain = float(pred[300:600].mean() - predict(p, np.arange(200, 300), m).mean())
            full = fit(y, m)
            row[m] = {
                "edge_300": p["edge"],
                "forecast_gain_late_vs_200_299": pred_gain,
                "actual_gain_late_vs_200_299": actual_gain,
                "ratio_forecast_to_actual": pred_gain / actual_gain if actual_gain else float("nan"),
                "rmse_300_899": float(np.sqrt(np.mean((pred - y[300:900]) ** 2))),
                "full_fit_edge": full["edge"],
                "full_fit_fraction": {G: fraction(full, G, m) for G in HORIZONS},
            }
            print(
                f"  {f.stem} {m:<5s}: forecast gain (600-899 vs 200-299) {pred_gain:+.3f}, actual {actual_gain:+.3f}"
                f", ratio {row[m]['ratio_forecast_to_actual']:.2f}; fitted on all 900: fraction at "
                + ", ".join(f"{G} {row[m]['full_fit_fraction'][G]:.2f}" for G in HORIZONS)
                + (" (no identifiable plateau)" if full["edge"] else "")
            )
        blocks = [float(y[i : i + 300].mean()) for i in (0, 300, 600)]
        row["block_means"] = blocks
        print(f"  {f.stem} block means 0-299, 300-599, 600-899: {', '.join(f'{b:.3f}' for b in blocks)}")
        val[f.stem] = row
    out["validation_run1"] = val

    # ---------------------------------------------------------------- the sweep's 2,120 curves
    print("\nSWEEP (300 generations): per run, the fraction of the eventual gain reached by each horizon")
    files = sorted(EX.glob("*.npz"))
    per = {
        "treatment": {"fast": [], "slow": []},
        "null": {"fast": [], "slow": []},
        "third": {"fast": [], "slow": []},
    }
    edges = {a: {"fast": 0, "slow": 0} for a in per}
    flat = {a: 0 for a in per}
    n = {a: 0 for a in per}
    late = {a: [] for a in per}
    seen = set()
    for f in files:
        d = np.load(f, allow_pickle=True)
        h = str(d["config_hash"])
        if h in seen:  # DEV-035: a configuration counts once
            continue
        seen.add(h)
        arm = str(d["arm"])
        for y in np.asarray(d["fit_mean"], np.float64):
            n[arm] += 1
            total = y[250:300].mean() - y[:10].mean()
            late[arm].append((y[250:300].mean() - y[200:250].mean()) / total if total > 0 else np.nan)
            if total <= 0:
                flat[arm] += 1
            for m in ("fast", "slow"):
                p = fit(y, m)
                if p["edge"] or p["b"] <= 0:
                    edges[arm][m] += 1
                    continue
                per[arm][m].append([fraction(p, G, m) for G in HORIZONS])
    out["sweep"] = {}
    for arm in per:
        lt = np.array(late[arm])
        print(
            f"\n  {arm}: {n[arm]} runs; no gain at all {flat[arm]}; model-free: the last 50 generations added a median "
            f"{np.nanmedian(lt) * 100:.1f}% of the run's total gain"
        )
        out["sweep"][arm] = {
            "runs": n[arm],
            "no_gain": flat[arm],
            "late_share_median": float(np.nanmedian(lt)),
        }
        for m in ("fast", "slow"):
            a = np.array(per[arm][m])
            q = {
                G: [float(x) for x in np.nanpercentile(a[:, i], [25, 50, 75])] for i, G in enumerate(HORIZONS)
            }
            out["sweep"][arm][m] = {"identified": len(a), "no_plateau": edges[arm][m], "quartiles": q}
            print(
                f"    {m:<5s} model: plateau identifiable in {len(a)} of {n[arm]}; median fraction reached "
                + ", ".join(
                    f"at {G}: {q[G][1] * 100:.0f}% ({q[G][0] * 100:.0f}-{q[G][2] * 100:.0f})"
                    for G in HORIZONS
                )
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
