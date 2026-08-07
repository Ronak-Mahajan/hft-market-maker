"""Paired multi-seed evaluation, because one seed of this simulator says nothing.

Why this file exists
--------------------
An earlier version of this project reported a single seed: a $22,322 loss turned
into a $7,336 profit, quoted as a "+132% improvement". That number reproduces —
but it is one draw from a distribution whose 5th and 95th percentiles are -412%
and +462%, and in which the naive benchmark is profitable 44% of the time. A
single path of a 10,000-tick GBM with Poisson fills carries enormous terminal
variance, so a single-seed comparison of two strategies measures the seed, not
the strategies.

Two things fix that:

1. PAIRED COMPARISON with common random numbers. Both strategies face the same
   price path, the same order arrivals and the same uniform fill draws, so the
   per-seed difference has the market's variance differenced out of it. This is
   what makes the comparison sharp enough to resolve at all.
2. Report the DISTRIBUTION of the paired difference across many seeds, with a
   confidence interval and a win rate, instead of a point estimate.

Usage:
    python evaluate.py --seeds 500
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from market_maker import (AvellanedaStoikovMaker, Config, FixedSpreadMaker,
                          metrics, run, simulate_market)


def paired_run(cfg: Config, seed: int) -> dict[str, dict[str, float]]:
    market = simulate_market(cfg, seed)
    return {s.name: metrics(run(s, market))
            for s in (FixedSpreadMaker(cfg), AvellanedaStoikovMaker(cfg))}


def summarise(diffs: np.ndarray, label: str, unit: str = "") -> dict:
    n = len(diffs)
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(n))
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    return {"label": label, "unit": unit, "mean": mean, "se": se,
            "ci95": [lo, hi], "median": float(np.median(diffs)),
            "win_rate": float((diffs > 0).mean()),
            "significant": bool(lo > 0 or hi < 0)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=500)
    p.add_argument("--gamma", type=float, default=None,
                   help="override risk aversion")
    p.add_argument("--json", default="results.json")
    args = p.parse_args()

    cfg = Config() if args.gamma is None else Config(risk_aversion=args.gamma)
    keys = ("final_pnl", "inventory_std", "max_drawdown", "sharpe", "n_fills")
    acc: dict[str, list[float]] = {f"{k}_{w}": [] for k in keys
                                   for w in ("fix", "as")}

    for s in range(args.seeds):
        r = paired_run(cfg, s)
        for k in keys:
            acc[f"{k}_fix"].append(r["fixed spread"][k])
            acc[f"{k}_as"].append(r["Avellaneda-Stoikov"][k])

    A = {k: np.array(v) for k, v in acc.items()}
    out = {"n_seeds": args.seeds, "gamma": cfg.risk_aversion, "results": []}

    print(f"paired comparison over {args.seeds} seeds, common random numbers")
    print(f"{'':<26}{'fixed':>12}{'A-S':>12}{'diff (95% CI)':>26}{'win':>7}")
    for k, better_is_higher in (("final_pnl", True), ("sharpe", True),
                                ("inventory_std", False),
                                ("max_drawdown", True), ("n_fills", True)):
        fix, a_s = A[f"{k}_fix"], A[f"{k}_as"]
        d = (a_s - fix) if better_is_higher else (fix - a_s)
        s = summarise(d, k)
        star = "*" if s["significant"] else " "
        print(f"{k:<26}{fix.mean():>12,.1f}{a_s.mean():>12,.1f}"
              f"{s['mean']:>13,.1f} [{s['ci95'][0]:,.1f}, {s['ci95'][1]:,.1f}]{star}"
              f"{s['win_rate']*100:>6.0f}%")
        out["results"].append({k: s})

    print("\n* = 95% CI excludes zero. 'diff' is oriented so positive = A-S better.")
    print("'win' = fraction of seeds where A-S wins that metric on the same market.")
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
