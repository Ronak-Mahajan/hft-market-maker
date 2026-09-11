"""Paired multi-seed evaluation, because one seed of this simulator says nothing.

Why this file exists
--------------------
An earlier version of this project reported a single seed: a $22,322 loss turned
into a $7,336 profit, quoted as a "+132% improvement". That number reproduces
(simulation.ipynb) -- but it is one draw from a wide distribution (audit.py
replays that old model across seeds). A single path of a 10,000-tick GBM with
Poisson fills carries enormous terminal variance, so a single-seed comparison
of two strategies measures the seed, not the strategies.

Two things fix that:

1. PAIRED COMPARISON with common random numbers. Every strategy faces the same
   price path, the same order arrivals and the same uniform fill draws, so the
   per-seed difference has the market's variance differenced out of it. This is
   what makes the comparison sharp enough to resolve at all.
2. Report the DISTRIBUTION of the paired difference across many seeds, with a
   confidence interval and a win rate, instead of a point estimate.

The decomposition
-----------------
Avellaneda-Stoikov changes two things relative to a naive symmetric quoter: it
skews the quotes around a reservation price, and it sets the spread from the
model (wider than the hand-picked 0.10 at the defaults). A two-arm comparison
cannot say which of the two does the work, so there are four arms, all run on
the same Market draws:

    fixed           symmetric at the benchmark spread          (benchmark)
    spread_matched  symmetric at the model spread, no skew     (spread effect)
    skew_only       model skew around the benchmark spread     (skew effect, narrow)
    as              model skew and model spread                (full model)

and four paired comparisons:

    spread_matched vs fixed   what quoting wider does on its own
    skew_only      vs fixed   what the skew does at the benchmark's own volume
    as             vs fixed   the full model against the benchmark
    as             vs spread_matched   what the skew does at the model spread

Outputs
-------
    results.json          schema v2: per-arm means/SEs and every comparison,
                          keyed by metric
    results_seeds.csv     one row per (seed, arm): the raw per-seed metrics
    results_sweep.json    --gamma-sweep: the same summaries on a gamma grid

Usage:
    python evaluate.py --seeds 500                  # headline artifacts
    python evaluate.py --seeds 500 --gamma 2.0      # results_gamma_2.0.json,
                                                    # never overwrites results.json
    python evaluate.py --seeds 100 --gamma-sweep    # default grid
    python evaluate.py --seeds 100 --gamma-sweep 0.1,0.5,2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, replace

import numpy as np

from market_maker import (ARMS, AvellanedaStoikovMaker, Config,
                          FixedSpreadMaker, SkewOnlyMaker, SpreadMatchedMaker,
                          model_half_spread, metrics, run, simulate_market)

SCHEMA_VERSION = 2
BENCHMARK = FixedSpreadMaker.slug

ARM_BY_SLUG = {cls.slug: cls for cls in ARMS}
ARM_NAMES = {cls.slug: cls.name for cls in ARMS}

# (metric, better_is_higher). Differences are oriented so positive = a better.
METRICS = (("final_pnl", True), ("t_stat", True), ("inventory_std", False),
           ("max_drawdown", True), ("n_fills", True))

# (a, b, question). Each summary is a - b, oriented by METRICS.
COMPARISONS = (
    (SpreadMatchedMaker.slug, BENCHMARK,
     "spread effect: symmetric quotes at the model spread vs the benchmark"),
    (SkewOnlyMaker.slug, BENCHMARK,
     "skew effect at the benchmark spread"),
    (AvellanedaStoikovMaker.slug, BENCHMARK,
     "full model (skew + model spread) vs the benchmark"),
    (AvellanedaStoikovMaker.slug, SpreadMatchedMaker.slug,
     "skew effect at the model spread (isolates the skew)"),
)

# results_seeds.csv columns -> metrics() keys
CSV_COLUMNS = (("inventory_sigma", "inventory_std"),
               ("max_drawdown", "max_drawdown"),
               ("fills", "n_fills"),
               ("final_pnl", "final_pnl"),
               ("t_stat", "t_stat"),
               ("max_abs_inventory", "max_abs_inventory"),
               ("quote_crossings", "n_crossed"))

DEFAULT_GAMMA_GRID = "0.05,0.1,0.2,0.5,1,2,5"


def arm_spreads(cfg: Config) -> dict[str, float]:
    """The spread each arm quotes at tau=1 (the A-S spread decays by
    gamma*sigma^2 over the run)."""
    model = 2.0 * model_half_spread(cfg, tau=1.0)
    return {FixedSpreadMaker.slug: cfg.fixed_spread,
            SpreadMatchedMaker.slug: model,
            SkewOnlyMaker.slug: cfg.fixed_spread,
            AvellanedaStoikovMaker.slug: model}


def summarise(diffs: np.ndarray, label: str, unit: str = "") -> dict:
    """Mean paired difference with a normal-approximation 95% CI, median and
    win rate. `significant` means the CI excludes zero."""
    n = len(diffs)
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    return {"label": label, "unit": unit, "n": int(n), "mean": mean, "se": se,
            "ci95": [lo, hi], "median": float(np.median(diffs)),
            "win_rate": float((diffs > 0).mean()),
            "significant": bool(lo > 0 or hi < 0)}


def compare(arrays: dict[str, dict[str, np.ndarray]], a: str, b: str) -> dict:
    """Paired a - b for every metric, oriented so positive = a better.
    Returns a dict keyed by metric."""
    out = {}
    for k, higher in METRICS:
        d = arrays[a][k] - arrays[b][k]
        if not higher:
            d = -d
        out[k] = summarise(d, k)
    return out


def arm_summary(arrays: dict[str, dict[str, np.ndarray]]) -> dict:
    """Per-arm means and standard errors, keyed by arm then metric."""
    out = {}
    for slug, cols in arrays.items():
        n = len(next(iter(cols.values())))
        out[slug] = {
            "name": ARM_NAMES[slug],
            "mean": {k: float(v.mean()) for k, v in cols.items()},
            "se": {k: (float(v.std(ddof=1) / math.sqrt(n)) if n > 1
                       else float("nan")) for k, v in cols.items()},
        }
    return out


def to_arrays(rows: list[tuple[int, str, dict[str, float]]]
              ) -> dict[str, dict[str, np.ndarray]]:
    """rows of (seed, arm, metrics) -> {arm: {metric: array over seeds}}.
    Rows must be grouped by seed in the same seed order for every arm so that
    the arrays line up for paired differences."""
    acc: dict[str, dict[str, list[float]]] = {}
    for _seed, slug, m in rows:
        cols = acc.setdefault(slug, {})
        for k, v in m.items():
            cols.setdefault(k, []).append(v)
    return {slug: {k: np.array(v) for k, v in cols.items()}
            for slug, cols in acc.items()}


def evaluate(cfg: Config, seeds) -> list[tuple[int, str, dict[str, float]]]:
    """Run all four arms on every seed, paired. Returns per-seed rows."""
    rows = []
    for seed in seeds:
        market = simulate_market(cfg, seed)
        for cls in ARMS:
            rows.append((seed, cls.slug, metrics(run(cls(cfg), market))))
    return rows


def build_report(cfg: Config, rows, n_seeds: int) -> dict:
    arrays = to_arrays(rows)
    comparisons = {}
    for a, b, question in COMPARISONS:
        comparisons[f"{a}_vs_{b}"] = {
            "a": a, "b": b, "question": question,
            "orientation": "positive = a better",
            "metrics": compare(arrays, a, b)}
    return {
        "schema": SCHEMA_VERSION,
        "design": "four paired arms on common random numbers",
        "n_seeds": int(n_seeds),
        "seed_range": [0, int(n_seeds) - 1],
        "gamma": cfg.risk_aversion,
        "config": asdict(cfg),
        "arm_names": ARM_NAMES,
        "arm_spreads": arm_spreads(cfg),
        "arms": arm_summary(arrays),
        "comparisons": comparisons,
        "notes": {
            "t_stat": "mean(step pnl)/std(step pnl)*sqrt(n_steps); the "
                      "whole-horizon Sharpe, not annualised",
            "max_drawdown": "<= 0; a positive difference means a shallower "
                            "drawdown for arm a",
            "quote_crossings": "ticks on which a raw quote crossed the mid and "
                               "was clamped to it (market_maker.MarketMaker.step)",
        },
    }


def write_seeds_csv(path: str, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "arm"] + [c for c, _ in CSV_COLUMNS])
        for seed, slug, m in rows:
            w.writerow([seed, slug] + [repr(float(m[k])) for _, k in CSV_COLUMNS])


def sweep(cfg: Config, seeds, grid: list[float]) -> dict:
    """Gamma sweep. The benchmark does not depend on gamma, so it is run once
    per seed and shared across the grid; the other three arms are run per
    gamma. Everything is paired on the same Market draws."""
    seeds = list(seeds)
    per_gamma: dict[float, list] = {g: [] for g in grid}
    for seed in seeds:
        market = simulate_market(cfg, seed)
        bench = metrics(run(FixedSpreadMaker(cfg), market))
        for g in grid:
            cfg_g = replace(cfg, risk_aversion=g)
            rows = per_gamma[g]
            rows.append((seed, BENCHMARK, bench))
            for cls in ARMS:
                if cls.slug == BENCHMARK:
                    continue
                rows.append((seed, cls.slug, metrics(run(cls(cfg_g), market))))
    points = []
    for g in grid:
        rep = build_report(replace(cfg, risk_aversion=g), per_gamma[g], len(seeds))
        points.append({"gamma": g, "arm_spreads": rep["arm_spreads"],
                       "arms": rep["arms"], "comparisons": rep["comparisons"]})
    return {"schema": SCHEMA_VERSION, "kind": "gamma_sweep",
            "n_seeds": len(seeds), "seed_range": [0, len(seeds) - 1],
            "grid": grid, "config": asdict(cfg), "arm_names": ARM_NAMES,
            "points": points}


def print_report(rep: dict) -> None:
    arms = rep["arms"]
    order = [cls.slug for cls in ARMS]
    print(f"paired comparison over {rep['n_seeds']} seeds, common random "
          f"numbers, gamma = {rep['gamma']}")
    print(f"{'arm':<20}{'spread':>9}" + "".join(f"{k:>16}" for k, _ in METRICS)
          + f"{'crossings':>11}")
    for slug in order:
        a = arms[slug]
        print(f"{ARM_NAMES[slug]:<20}{rep['arm_spreads'][slug]:>9.4f}"
              + "".join(f"{a['mean'][k]:>16,.2f}" for k, _ in METRICS)
              + f"{a['mean']['n_crossed']:>11,.1f}")
    for key, c in rep["comparisons"].items():
        print(f"\n{key}: {c['question']}")
        print(f"{'metric':<16}{'diff':>12}{'95% CI':>26}{'win':>7}")
        for k, s in c["metrics"].items():
            star = "*" if s["significant"] else " "
            print(f"{k:<16}{s['mean']:>12,.2f}   [{s['ci95'][0]:>10,.2f}, "
                  f"{s['ci95'][1]:>10,.2f}]{star}{s['win_rate'] * 100:>6.0f}%")
    print("\n* = 95% CI excludes zero. Differences are a - b, oriented so "
          "positive = a better\n(inventory_std is sign-flipped). 'win' = "
          "fraction of seeds on which a beats b on that metric.")


def print_sweep(sw: dict) -> None:
    print(f"gamma sweep over {sw['n_seeds']} seeds; diff = a - b (positive = a "
          "better), * = CI excludes zero")
    for a, b, _q in COMPARISONS:
        key = f"{a}_vs_{b}"
        print(f"\n{key}")
        print(f"{'gamma':>7}{'model spr':>11}" + "".join(f"{k:>18}" for k, _ in METRICS))
        for pt in sw["points"]:
            c = pt["comparisons"][key]["metrics"]
            line = f"{pt['gamma']:>7g}{pt['arm_spreads']['as']:>11.4f}"
            for k, _ in METRICS:
                s = c[k]
                line += f"{s['mean']:>15,.1f}{'*' if s['significant'] else ' '}  "
            print(line)
    print("\ncrossings per run (mean), skew-only / A-S:")
    for pt in sw["points"]:
        print(f"  gamma {pt['gamma']:>5g}: "
              f"{pt['arms']['skew_only']['mean']['n_crossed']:>8.1f} / "
              f"{pt['arms']['as']['mean']['n_crossed']:>8.1f}")


def parse_grid(text: str) -> list[float]:
    grid = sorted({float(x) for x in text.split(",") if x.strip()})
    if not grid or any(g <= 0 for g in grid):
        raise SystemExit("--gamma-sweep needs a comma-separated list of positive gammas")
    return grid


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=500)
    p.add_argument("--gamma", type=float, default=None,
                   help="override risk aversion; outputs then go to "
                        "results_gamma_<g>.json / results_seeds_gamma_<g>.csv "
                        "so the headline artifacts are never overwritten")
    p.add_argument("--json", default=None,
                   help="summary output (default results.json, or "
                        "results_gamma_<g>.json with --gamma)")
    p.add_argument("--csv", default=None,
                   help="per-seed output (default results_seeds.csv, or "
                        "results_seeds_gamma_<g>.csv with --gamma)")
    p.add_argument("--gamma-sweep", nargs="?", const=DEFAULT_GAMMA_GRID,
                   default=None, metavar="GRID",
                   help="run the four arms on a comma-separated gamma grid "
                        f"(default {DEFAULT_GAMMA_GRID}) and write "
                        "--sweep-json instead of the headline artifacts")
    p.add_argument("--sweep-json", default="results_sweep.json")
    args = p.parse_args()
    if args.seeds < 2:
        raise SystemExit("--seeds must be at least 2 (a CI needs a variance)")

    t0 = time.perf_counter()
    if args.gamma_sweep is not None:
        grid = parse_grid(args.gamma_sweep)
        sw = sweep(Config(), range(args.seeds), grid)
        print_sweep(sw)
        with open(args.sweep_json, "w") as f:
            json.dump(sw, f, indent=1)
        print(f"\nwrote {args.sweep_json}  ({time.perf_counter() - t0:.0f} s)")
        return

    cfg = Config() if args.gamma is None else Config(risk_aversion=args.gamma)
    suffix = "" if args.gamma is None else f"_gamma_{args.gamma:g}"
    json_path = args.json or f"results{suffix}.json"
    csv_path = args.csv or f"results_seeds{suffix}.csv"

    rows = evaluate(cfg, range(args.seeds))
    rep = build_report(cfg, rows, args.seeds)
    print_report(rep)
    with open(json_path, "w") as f:
        json.dump(rep, f, indent=1)
    write_seeds_csv(csv_path, rows)
    print(f"\nwrote {json_path} and {csv_path}  "
          f"({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
