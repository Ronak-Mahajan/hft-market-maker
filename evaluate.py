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
    spread_matched  symmetric at the model spread, no skew
    skew_only       model skew around the benchmark spread
    as              model skew and model spread                (full model)

and five paired comparisons:

    spread_matched vs fixed            quoting wider, skew off
    skew_only      vs fixed            the skew, at the benchmark spread
    as             vs fixed            the full model against the benchmark
    as             vs spread_matched   the skew, at the model spread
    as             vs skew_only        quoting wider, skew on

The arms are a 2x2 design (spread x skew), and the full model's effect splits
into a spread step and a skew step in either order. The two orders differ by
the interaction, so the report also gives the factorial main effects (each
averaged over both orders, summing to as - fixed) and the interaction itself.

Outputs
-------
    results.json          schema v3: per-arm means/SEs, every comparison and
                          the 2x2 factorial effects, keyed by metric
    results_seeds.csv     one row per (seed, arm): the raw per-seed metrics
    results_sweep.json    --gamma-sweep: the same summaries on a gamma grid
    results_uncapped.json, results_seeds_uncapped.csv
                          --max-inventory none: the same run with the
                          position limit lifted

Usage:
    python evaluate.py --seeds 500                  # headline artifacts
    python evaluate.py --seeds 500 --gamma 2.0      # results_gamma_2.json,
                                                    # never overwrites results.json
    python evaluate.py --seeds 500 --max-inventory none   # results_uncapped.json
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

SCHEMA_VERSION = 3
BENCHMARK = FixedSpreadMaker.slug

ARM_BY_SLUG = {cls.slug: cls for cls in ARMS}
ARM_NAMES = {cls.slug: cls.name for cls in ARMS}

# (metric, better_is_higher). Differences are oriented so positive = a better.
METRICS = (("final_pnl", True), ("edge", True), ("inventory_pnl", True),
           ("t_stat", True), ("inventory_std", False), ("max_drawdown", True),
           ("log_max_drawdown", False), ("n_fills", True), ("n_blocked", False),
           ("ticks_at_cap", False))

# Metrics whose paired differences are also reported as percentages, under
# "pct" in each summary:
#   benchmark  100 * difference / |the benchmark arm's mean|: points of the
#              benchmark's level (inventory sigma 17.0% = 8.59 / 50.5)
#   log        100 * (1 - exp(-difference)): the percentage reduction that a
#              mean log ratio corresponds to (a geometric-mean ratio)
PCT_BASIS = {"edge": "benchmark", "inventory_std": "benchmark",
             "max_drawdown": "benchmark", "log_max_drawdown": "log",
             "n_fills": "benchmark"}

# (a, b, question). Each summary is a - b, oriented by METRICS.
COMPARISONS = (
    (SpreadMatchedMaker.slug, BENCHMARK,
     "spread effect: symmetric quotes at the model spread vs the benchmark"),
    (SkewOnlyMaker.slug, BENCHMARK,
     "skew effect at the benchmark spread"),
    (AvellanedaStoikovMaker.slug, BENCHMARK,
     "full model (skew + model spread) vs the benchmark"),
    (AvellanedaStoikovMaker.slug, SpreadMatchedMaker.slug,
     "skew effect at the model spread"),
    (AvellanedaStoikovMaker.slug, SkewOnlyMaker.slug,
     "spread effect with the model skew on"),
)

# The four arms are a 2x2 design, spread (benchmark, model) x skew (off, on):
#     fixed = (benchmark, off)    spread_matched = (model, off)
#     skew_only = (benchmark, on) as = (model, on)
# (as - fixed) splits into a spread step and a skew step in two orders,
# fixed -> spread_matched -> as and fixed -> skew_only -> as, and the two
# orders differ by the interaction. The main effects average the orders, so
# spread_main + skew_main = as - fixed on every seed. (effect, definition)
FACTORIAL = (
    ("spread_main", "0.5 * [(spread_matched - fixed) + (as - skew_only)]"),
    ("skew_main", "0.5 * [(skew_only - fixed) + (as - spread_matched)]"),
    ("interaction", "(as - spread_matched) - (skew_only - fixed): how much "
                    "more the skew is worth at the model spread than at the "
                    "benchmark spread (equivalently, the spread with the skew "
                    "on than off)"),
    ("skew_minus_spread", "skew_main - spread_main"),
)

# results_seeds.csv columns -> metrics() keys
CSV_COLUMNS = (("inventory_sigma", "inventory_std"),
               ("max_drawdown", "max_drawdown"),
               ("fills", "n_fills"),
               ("final_pnl", "final_pnl"),
               ("t_stat", "t_stat"),
               ("max_abs_inventory", "max_abs_inventory"),
               ("quote_crossings", "n_crossed"),
               ("floored_bids", "n_floored"),
               ("edge", "edge"),
               ("inventory_pnl", "inventory_pnl"),
               ("blocked_fills", "n_blocked"),
               ("ticks_at_cap", "ticks_at_cap"))

DEFAULT_GAMMA_GRID = "0.05,0.1,0.2,0.5,1,2,5"
# the metrics print_sweep shows; results_sweep.json carries all of METRICS
SWEEP_PRINTED = ("inventory_std", "max_drawdown", "n_fills", "edge", "final_pnl")

# Every float written to an artifact is rounded to this many significant
# digits. The simulation is deterministic, but the last bits of exp, log and
# long sums differ between platforms and numpy builds: a Windows run differs
# from the Linux CI run by up to 4e-11 relative. Ten digits sit above that
# noise, so another platform writes the committed bytes except where a value
# falls within the noise of a rounding boundary. There it differs by one unit
# in the tenth digit, which scripts/check_artifacts.py accepts.
SIG_DIGITS = 10


def arm_spreads(cfg: Config) -> dict[str, float]:
    """The spread each arm quotes at tau=1 (the A-S spread decays by
    gamma*sigma^2 over the run)."""
    model = 2.0 * model_half_spread(cfg, tau=1.0)
    return {FixedSpreadMaker.slug: cfg.fixed_spread,
            SpreadMatchedMaker.slug: model,
            SkewOnlyMaker.slug: cfg.fixed_spread,
            AvellanedaStoikovMaker.slug: model}


def summarise(diffs: np.ndarray, label: str, unit: str = "",
              pct: tuple[str, float] | None = None) -> dict:
    """Mean paired difference with a normal-approximation 95% CI, median and
    win rate. `significant` means the CI excludes zero.

    pct = (basis, benchmark_mean) adds the mean and CI as percentages (see
    PCT_BASIS)."""
    n = len(diffs)
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    out = {"label": label, "unit": unit, "n": int(n), "mean": mean, "se": se,
           "ci95": [lo, hi], "median": float(np.median(diffs)),
           "win_rate": float((diffs > 0).mean()),
           "significant": bool(lo > 0 or hi < 0)}
    if pct is not None:
        basis, bench = pct
        if basis == "log":
            def f(x): return 100.0 * -math.expm1(-x)
        else:
            def f(x): return 100.0 * x / abs(bench) if bench else math.nan
        out["pct"] = {"basis": basis, "mean": f(mean), "ci95": [f(lo), f(hi)]}
    return out


def oriented(arrays: dict[str, dict[str, np.ndarray]], slug: str,
             k: str, higher: bool) -> np.ndarray:
    """An arm's per-seed metric, negated when lower is better, so that every
    difference of two of these is positive when the first arm is better."""
    v = arrays[slug][k]
    return v if higher else -v


def pct_spec(arrays: dict[str, dict[str, np.ndarray]],
             k: str) -> tuple[str, float] | None:
    basis = PCT_BASIS.get(k)
    if basis is None:
        return None
    return basis, float(arrays[BENCHMARK][k].mean())


def compare(arrays: dict[str, dict[str, np.ndarray]], a: str, b: str) -> dict:
    """Paired a - b for every metric, oriented so positive = a better.
    Returns a dict keyed by metric."""
    out = {}
    for k, higher in METRICS:
        d = oriented(arrays, a, k, higher) - oriented(arrays, b, k, higher)
        out[k] = summarise(d, k, pct=pct_spec(arrays, k))
    return out


def factorial_effects(arrays: dict[str, dict[str, np.ndarray]], k: str,
                      higher: bool) -> dict[str, np.ndarray]:
    """Per-seed factorial contrasts for one metric, oriented so positive =
    better (see FACTORIAL)."""
    f, sm, so, a = (oriented(arrays, slug, k, higher) for slug in (
        FixedSpreadMaker.slug, SpreadMatchedMaker.slug, SkewOnlyMaker.slug,
        AvellanedaStoikovMaker.slug))
    spread_main = 0.5 * ((sm - f) + (a - so))
    skew_main = 0.5 * ((so - f) + (a - sm))
    return {"spread_main": spread_main, "skew_main": skew_main,
            "interaction": (a - sm) - (so - f),
            "skew_minus_spread": skew_main - spread_main}


def factorial(arrays: dict[str, dict[str, np.ndarray]]) -> dict:
    """Main effects, interaction and their difference for every metric, each
    with a paired CI over seeds."""
    per_metric = {k: factorial_effects(arrays, k, higher) for k, higher in METRICS}
    effects = {}
    for name, definition in FACTORIAL:
        effects[name] = {
            "definition": definition,
            "metrics": {k: summarise(per_metric[k][name], k,
                                     pct=pct_spec(arrays, k))
                        for k, _ in METRICS}}
    return {"design": "2x2 on the same draws: spread (benchmark, model) x "
                      "skew (off, on)",
            "cells": {FixedSpreadMaker.slug: ["benchmark", "off"],
                      SpreadMatchedMaker.slug: ["model", "off"],
                      SkewOnlyMaker.slug: ["benchmark", "on"],
                      AvellanedaStoikovMaker.slug: ["model", "on"]},
            "orientation": "positive = better, as in comparisons",
            "effects": effects}


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


def price_paths(cfg: Config, seeds, arrays: dict[str, dict[str, np.ndarray]]
                ) -> dict:
    """Where the simulated mid goes over these seeds, and how much of the
    dollar drawdown the high-price paths carry. The lean and the spreads are
    fixed dollar amounts, so inventory sigma, fills and edge do not depend
    on the price level; dollar P&L and drawdown scale with it. `arrays`
    must come from evaluate() over the same seeds, in order."""
    seeds = list(seeds)
    final, low, high = (np.empty(len(seeds)) for _ in range(3))
    for i, seed in enumerate(seeds):
        p = simulate_market(cfg, seed).prices
        final[i], low[i], high[i] = p[-1], p.min(), p.max()
    bench_dd = -arrays[BENCHMARK]["max_drawdown"]
    gain = (arrays[AvellanedaStoikovMaker.slug]["max_drawdown"]
            - arrays[BENCHMARK]["max_drawdown"])
    top = int(np.argmax(high))
    decile = np.argsort(high)[-max(1, len(seeds) // 10):]
    return {
        "final_mid": {"p5": float(np.percentile(final, 5)),
                      "median": float(np.median(final)),
                      "p95": float(np.percentile(final, 95))},
        "path_min": {"lowest": float(low.min()),
                     "seed": seeds[int(np.argmin(low))],
                     "share_below_1": float((low < 1.0).mean())},
        "path_max": {"highest": float(high.max()), "seed": seeds[top],
                     "share_above_1000": float((high > 1000.0).mean())},
        "dollar_drawdown": {
            "corr_benchmark_drawdown_with_path_max":
                float(np.corrcoef(bench_dd, high)[0, 1]),
            "top_decile_by_path_max_share_of_benchmark_drawdown":
                float(bench_dd[decile].sum() / bench_dd.sum()),
            "highest_path_seed_share_of_as_vs_fixed_sum":
                float(gain[top] / gain.sum()),
        },
    }


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
        "factorial": factorial(arrays),
        "notes": {
            "edge": "spread captured: order_size * sum over fills of the "
                    "fill's distance from that tick's mid",
            "inventory_pnl": "sum over ticks of inventory held into the tick "
                             "times the mid's change; final_pnl = edge + "
                             "inventory_pnl, and E[inventory_pnl] = 0 here "
                             "(market_maker.metrics)",
            "t_stat": "mean(step pnl)/std(step pnl)*sqrt(n_steps); the "
                      "whole-horizon Sharpe, not annualised",
            "max_drawdown": "<= 0, in dollars; a positive difference means a "
                            "shallower drawdown for arm a. Scales with the "
                            "price path, so a few high-price seeds dominate "
                            "the mean",
            "log_max_drawdown": "ln|max_drawdown|; a paired difference is the "
                                "per-seed log of the drawdown ratio b/a, free "
                                "of the price level. pct = 1 - exp(-mean): the "
                                "geometric-mean drawdown reduction",
            "pct": "benchmark basis: 100 * difference / |benchmark mean|; log "
                   "basis: 100 * (1 - exp(-difference))",
            "price_paths": "the simulated mid over these seeds (evaluate.py "
                           "runs only); dollar P&L and drawdown scale with it",
            "quote_crossings": "ticks on which a raw quote crossed the mid and "
                               "was clamped to it (market_maker.MarketMaker.step)",
            "floored_bids": "ticks on which the raw bid was below "
                            "Config.min_price and was raised to it",
            "blocked_fills": "fills that won their draw but were refused "
                             "because the trade would take |q| past "
                             "max_inventory",
            "ticks_at_cap": "ticks that end with the position at the limit, "
                            "where every fill on one side is refused",
        },
    }


def round_sig(x: float, digits: int = SIG_DIGITS) -> float:
    """x rounded to `digits` significant digits (nan and inf pass through)."""
    return float(f"{x:.{digits}g}")


def rounded(obj):
    """A copy of a JSON-ready structure with every float passed through
    round_sig. Integers, strings and booleans are left alone."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round_sig(obj)
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v) for v in obj]
    return obj


def write_json(path: str, obj) -> None:
    """LF line endings and floats rounded to SIG_DIGITS (see SIG_DIGITS)."""
    with open(path, "w", newline="\n") as f:
        json.dump(rounded(obj), f, indent=1)
        f.write("\n")


def write_seeds_csv(path: str, rows) -> None:
    # LF line endings on every platform (the csv default is CRLF, and text
    # mode would translate on Windows); floats rounded to SIG_DIGITS.
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["seed", "arm"] + [c for c, _ in CSV_COLUMNS])
        for seed, slug, m in rows:
            w.writerow([seed, slug]
                       + [repr(round_sig(float(m[k]))) for _, k in CSV_COLUMNS])


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
                       "arms": rep["arms"], "comparisons": rep["comparisons"],
                       "factorial": rep["factorial"]})
    return {"schema": SCHEMA_VERSION, "kind": "gamma_sweep",
            "n_seeds": len(seeds), "seed_range": [0, len(seeds) - 1],
            "grid": grid, "config": asdict(cfg), "arm_names": ARM_NAMES,
            "points": points}


def print_report(rep: dict) -> None:
    arms = rep["arms"]
    order = [cls.slug for cls in ARMS]
    print(f"paired comparison over {rep['n_seeds']} seeds, common random "
          f"numbers, gamma = {rep['gamma']}, max_inventory = "
          f"{rep['config']['max_inventory']}")
    print(f"\n{'per-arm mean':<20}" + "".join(f"{slug:>16}" for slug in order))
    print(f"{'spread':<20}"
          + "".join(f"{rep['arm_spreads'][slug]:>16.4f}" for slug in order))
    for k in arms[order[0]]["mean"]:
        print(f"{k:<20}"
              + "".join(f"{arms[slug]['mean'][k]:>16,.2f}" for slug in order))
    blocks = [(f"{key}: {c['question']}", c["metrics"])
              for key, c in rep["comparisons"].items()]
    blocks += [(f"factorial {name}: {e['definition']}", e["metrics"])
               for name, e in rep["factorial"]["effects"].items()]
    for title, summaries in blocks:
        print(f"\n{title}")
        print(f"{'metric':<20}{'diff':>12}{'95% CI':>26}{'win':>7}{'pct':>9}")
        for k, s in summaries.items():
            star = "*" if s["significant"] else " "
            pct = f"{s['pct']['mean']:>8.1f}%" if "pct" in s else ""
            print(f"{k:<20}{s['mean']:>12,.2f}   [{s['ci95'][0]:>10,.2f}, "
                  f"{s['ci95'][1]:>10,.2f}]{star}{s['win_rate'] * 100:>6.0f}%"
                  f"{pct}")
    print("\n* = 95% CI excludes zero. Differences are a - b, oriented so "
          "positive = a better\n(inventory_std is sign-flipped). 'win' = "
          "fraction of seeds on which a beats b on that metric.")


def print_sweep(sw: dict) -> None:
    print(f"gamma sweep over {sw['n_seeds']} seeds; diff = a - b (positive = a "
          "better), * = CI excludes zero")
    for a, b, _q in COMPARISONS:
        key = f"{a}_vs_{b}"
        print(f"\n{key}")
        print(f"{'gamma':>7}{'model spr':>11}"
              + "".join(f"{k:>18}" for k in SWEEP_PRINTED))
        for pt in sw["points"]:
            c = pt["comparisons"][key]["metrics"]
            line = f"{pt['gamma']:>7g}{pt['arm_spreads']['as']:>11.4f}"
            for k in SWEEP_PRINTED:
                s = c[k]
                line += f"{s['mean']:>15,.1f}{'*' if s['significant'] else ' '}  "
            print(line)
    print("\ncrossings per run (mean), skew-only / A-S:")
    for pt in sw["points"]:
        print(f"  gamma {pt['gamma']:>5g}: "
              f"{pt['arms']['skew_only']['mean']['n_crossed']:>8.1f} / "
              f"{pt['arms']['as']['mean']['n_crossed']:>8.1f}")


def parse_cap(text: str, cfg: Config) -> tuple[int, str]:
    """--max-inventory N|none -> (max_inventory, artifact suffix). 'none'
    sets the cap to n_ticks * order_size, more than any run can hold, so it
    never binds."""
    if text.strip().lower() == "none":
        return cfg.n_ticks * cfg.order_size, "_uncapped"
    try:
        n = int(text)
    except ValueError:
        raise SystemExit("--max-inventory needs a non-negative integer or "
                         "'none'") from None
    return n, f"_cap_{n}"


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
    p.add_argument("--max-inventory", default=None, metavar="N|none",
                   help="override the position limit |q| <= N; 'none' lifts "
                        "it. Outputs then go to results_uncapped.json / "
                        "results_seeds_uncapped.csv (or results_cap_<N>.json "
                        "/ results_seeds_cap_<N>.csv), never the headline "
                        "artifacts")
    p.add_argument("--json", default=None,
                   help="summary output (default results.json, or the "
                        "--gamma / --max-inventory name)")
    p.add_argument("--csv", default=None,
                   help="per-seed output (default results_seeds.csv, or the "
                        "--gamma / --max-inventory name)")
    p.add_argument("--gamma-sweep", nargs="?", const=DEFAULT_GAMMA_GRID,
                   default=None, metavar="GRID",
                   help="run the four arms on a comma-separated gamma grid "
                        f"(default {DEFAULT_GAMMA_GRID}) and write "
                        "--sweep-json instead of the headline artifacts")
    p.add_argument("--sweep-json", default="results_sweep.json")
    args = p.parse_args()
    if args.seeds < 2:
        raise SystemExit("--seeds must be at least 2 (a CI needs a variance)")

    if args.gamma_sweep is not None and (args.gamma is not None
                                         or args.max_inventory is not None):
        raise SystemExit("--gamma and --max-inventory apply to a single run, "
                         "not to --gamma-sweep")

    t0 = time.perf_counter()
    if args.gamma_sweep is not None:
        grid = parse_grid(args.gamma_sweep)
        sw = sweep(Config(), range(args.seeds), grid)
        print_sweep(sw)
        write_json(args.sweep_json, sw)
        print(f"\nwrote {args.sweep_json}  ({time.perf_counter() - t0:.0f} s)")
        return

    overrides, suffix = {}, ""
    if args.gamma is not None:
        overrides["risk_aversion"] = args.gamma
        suffix += f"_gamma_{args.gamma:g}"
    if args.max_inventory is not None:
        overrides["max_inventory"], cap_suffix = parse_cap(args.max_inventory, Config())
        suffix += cap_suffix
    try:
        cfg = Config(**overrides)
    except ValueError as e:
        raise SystemExit(f"invalid setting: {e}") from None
    json_path = args.json or f"results{suffix}.json"
    csv_path = args.csv or f"results_seeds{suffix}.csv"

    rows = evaluate(cfg, range(args.seeds))
    rep = build_report(cfg, rows, args.seeds)
    rep["price_paths"] = price_paths(cfg, range(args.seeds), to_arrays(rows))
    print_report(rep)
    write_json(json_path, rep)
    write_seeds_csv(csv_path, rows)
    print(f"\nwrote {json_path} and {csv_path}  "
          f"({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
