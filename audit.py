"""Multi-seed audit of the OLD notebook model (simulation.ipynb), paired.

What this audits
----------------
The original notebook compared a naive fixed-spread quoter (Strategy A) with an
"inventory-aware" quoter (Strategy B) that skewed the SAME 0.10 spread by
-q * gamma * sigma^2 with gamma = 0.5 and no (T - t) term, on ONE seed, and
reported a $22,322 loss turned into a $7,336 profit (+132%). That single-seed
figure still ships with its code in simulation.ipynb. This script replays the
notebook's strategy logic across many seeds so the single seed can be placed in
its distribution.

This is NOT the model in market_maker.py. The current strategies carry the
(T - t) term and, in the full Avellaneda-Stoikov arm, the model-derived spread;
the closest current analogue of Strategy B is SkewOnlyMaker. For the current
model use evaluate.py.

History
-------
An earlier version of this script (_audit.py, deleted in commit 250eaa9 and
restored here from b678220) produced the "-412% to +462%" and "profitable 44%
of the time" figures quoted in the README. It drew the two strategies' fill
uniforms from DIFFERENT RNG streams (default_rng(s) for A and
default_rng(s + 10**6) for B), so the per-seed difference between the two
strategies mixed strategy effect with fill luck. This version is PAIRED: the
price path, arrivals, directions and per-tick fill uniforms are drawn once per
seed and both strategies consume the same draws, the same design evaluate.py
uses. Because the uniforms are now pre-drawn per tick rather than drawn on
demand, individual seeds (including seed 42) do not reproduce the notebook's
exact numbers; the distribution is what this script is for.

Usage:
    python audit.py --seeds 200            # writes audit_results.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NotebookConfig:
    """The notebook's parameters, verbatim."""
    initial_price: float = 100.0
    drift: float = 0.0001
    volatility: float = 0.02
    dt: float = 1.0
    poisson_rate: float = 0.1
    order_size: int = 10
    n_ticks: int = 10_000
    naive_spread: float = 0.10
    risk_aversion: float = 0.5       # the notebook hard-coded gamma = 0.5 in quote()
    kappa: float = 10.0              # the notebook hard-coded kappa = 10 in tick()
    max_inventory: int = 100


def draw_market(cfg: NotebookConfig, seed: int):
    """Price path, arrivals, directions and fill uniforms from ONE stream."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(cfg.n_ticks - 1)
    log_ret = ((cfg.drift - 0.5 * cfg.volatility ** 2) * cfg.dt
               + cfg.volatility * math.sqrt(cfg.dt) * eps)
    prices = np.empty(cfg.n_ticks)
    prices[0] = cfg.initial_price
    prices[1:] = cfg.initial_price * np.exp(np.cumsum(log_ret))
    arrival_prob = 1.0 - math.exp(-cfg.poisson_rate * cfg.dt)
    arrivals = rng.random(cfg.n_ticks) < arrival_prob
    directions = rng.choice([1, -1], size=cfg.n_ticks)
    fill_draws = rng.random(cfg.n_ticks)
    return (prices.tolist(), arrivals.tolist(), directions.tolist(),
            fill_draws.tolist())


class OldMaker:
    def __init__(self, cfg: NotebookConfig):
        self.cfg = cfg
        self.inventory = 0
        self.cash = 0.0
        self.pnl = np.empty(cfg.n_ticks)
        self.inv_path = np.empty(cfg.n_ticks, dtype=np.int64)

    def quote(self, mid: float) -> tuple[float, float]:
        raise NotImplementedError

    def tick(self, t: int, mid: float, arrived: bool, direction: int,
             draw: float) -> None:
        cfg = self.cfg
        bid, ask = self.quote(mid)
        if arrived:
            if direction == 1:
                p = math.exp(-cfg.kappa * (ask - mid))
                if self.inventory > -cfg.max_inventory and draw < p:
                    self.inventory -= cfg.order_size
                    self.cash += ask * cfg.order_size
            else:
                p = math.exp(-cfg.kappa * (mid - bid))
                if self.inventory < cfg.max_inventory and draw < p:
                    self.inventory += cfg.order_size
                    self.cash -= bid * cfg.order_size
        self.pnl[t] = self.cash + self.inventory * mid
        self.inv_path[t] = self.inventory


class Naive(OldMaker):
    """Strategy A: symmetric quotes at the hand-picked spread."""

    def quote(self, mid: float) -> tuple[float, float]:
        h = self.cfg.naive_spread / 2.0
        return mid - h, mid + h


class InventoryAware(OldMaker):
    """Strategy B: the same spread, skewed by -q*gamma*sigma^2 (no horizon)."""

    def quote(self, mid: float) -> tuple[float, float]:
        cfg = self.cfg
        r = mid - self.inventory * cfg.risk_aversion * cfg.volatility ** 2
        h = cfg.naive_spread / 2.0
        return r - h, r + h


def old_metrics(s: OldMaker) -> dict[str, float]:
    pnl = s.pnl
    ret = np.diff(pnl)
    return {
        "final_pnl": float(pnl[-1]),
        # the notebook's "Sharpe": per-tick mean/std scaled by sqrt(252), a
        # daily-returns convention applied to ticks; kept for comparability only
        "notebook_sharpe": float(ret.mean() / (ret.std() + 1e-6) * math.sqrt(252)),
        "max_drawdown": float(np.min(pnl - np.maximum.accumulate(pnl))),
        "inventory_std": float(s.inv_path.std()),
        "mean_abs_inventory": float(np.abs(s.inv_path).mean()),
    }


def run_seed(cfg: NotebookConfig, seed: int) -> dict[str, dict[str, float]]:
    prices, arrivals, directions, draws = draw_market(cfg, seed)
    out = {}
    for name, cls in (("A", Naive), ("B", InventoryAware)):
        s = cls(cfg)
        tick = s.tick
        for t in range(cfg.n_ticks):
            tick(t, prices[t], arrivals[t], directions[t], draws[t])
        out[name] = old_metrics(s)
    return out


def pct(v: np.ndarray, q: float) -> float:
    return float(np.percentile(v, q))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--json", default="audit_results.json")
    args = p.parse_args()

    cfg = NotebookConfig()
    rows = [run_seed(cfg, s) for s in range(args.seeds)]

    def col(strategy: str, key: str) -> np.ndarray:
        return np.array([r[strategy][key] for r in rows])

    pnl_a, pnl_b = col("A", "final_pnl"), col("B", "final_pnl")
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(pnl_a != 0, (pnl_b - pnl_a) / np.abs(pnl_a) * 100.0, np.nan)
    istd_a, istd_b = col("A", "inventory_std"), col("B", "inventory_std")
    mdd_a, mdd_b = col("A", "max_drawdown"), col("B", "max_drawdown")
    inv_red = (1.0 - istd_b / istd_a) * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        mdd_red = np.where(mdd_a != 0, (1.0 - mdd_b / mdd_a) * 100.0, np.nan)

    n = len(rows)
    d_pnl = pnl_b - pnl_a
    se_pnl = float(d_pnl.std(ddof=1) / math.sqrt(n))

    print(f"{n} paired seeds, the notebook's strategy logic (OLD model, not "
          "market_maker.py)\n")
    print(f"{'metric':<30}{'p5':>12}{'median':>12}{'p95':>12}")
    for lab, v in (("Strategy A final P&L", pnl_a),
                   ("Strategy B final P&L", pnl_b),
                   ("rel. improvement %", rel[~np.isnan(rel)])):
        print(f"{lab:<30}{pct(v, 5):>12,.0f}{np.median(v):>12,.0f}{pct(v, 95):>12,.0f}")
    print()
    print(f"Strategy A profitable in {(pnl_a > 0).mean() * 100:.0f}% of seeds; "
          f"B in {(pnl_b > 0).mean() * 100:.0f}%")
    print(f"B beats A on P&L in {(pnl_b > pnl_a).mean() * 100:.0f}% of seeds; "
          f"paired mean B-A = {d_pnl.mean():,.0f} "
          f"[{d_pnl.mean() - 1.96 * se_pnl:,.0f}, {d_pnl.mean() + 1.96 * se_pnl:,.0f}]")
    print(f"inventory-std reduction: median {np.median(inv_red):+.0f}%  "
          f"(p5 {pct(inv_red, 5):+.0f}%, p95 {pct(inv_red, 95):+.0f}%), "
          f"B lower in {(istd_b < istd_a).mean() * 100:.0f}% of seeds")
    finite = mdd_red[~np.isnan(mdd_red)]
    print(f"max-drawdown reduction : median {np.median(finite):+.0f}%  "
          f"(p5 {pct(finite, 5):+.0f}%, p95 {pct(finite, 95):+.0f}%), "
          f"B shallower in {(mdd_b > mdd_a).mean() * 100:.0f}% of seeds")

    out = {
        "schema": 1,
        "model": "old notebook model (simulation.ipynb): fixed 0.10 spread, "
                 "skew -q*gamma*sigma^2 with gamma=0.5 and no horizon term",
        "paired": True,
        "n_seeds": n,
        "config": cfg.__dict__,
        "per_seed": [{"seed": i, "A": r["A"], "B": r["B"]} for i, r in enumerate(rows)],
        "summary": {
            "pnl_A": {"p5": pct(pnl_a, 5), "median": float(np.median(pnl_a)),
                      "p95": pct(pnl_a, 95), "profitable_share": float((pnl_a > 0).mean())},
            "pnl_B": {"p5": pct(pnl_b, 5), "median": float(np.median(pnl_b)),
                      "p95": pct(pnl_b, 95), "profitable_share": float((pnl_b > 0).mean())},
            "rel_improvement_pct": {"p5": pct(rel[~np.isnan(rel)], 5),
                                    "median": float(np.nanmedian(rel)),
                                    "p95": pct(rel[~np.isnan(rel)], 95)},
            "pnl_B_minus_A": {"mean": float(d_pnl.mean()), "se": se_pnl,
                              "ci95": [float(d_pnl.mean() - 1.96 * se_pnl),
                                       float(d_pnl.mean() + 1.96 * se_pnl)],
                              "win_rate": float((pnl_b > pnl_a).mean())},
            "inventory_std_reduction_pct": {"p5": pct(inv_red, 5),
                                            "median": float(np.median(inv_red)),
                                            "p95": pct(inv_red, 95),
                                            "win_rate": float((istd_b < istd_a).mean())},
            "max_drawdown_reduction_pct": {"p5": pct(finite, 5),
                                           "median": float(np.median(finite)),
                                           "p95": pct(finite, 95),
                                           "win_rate": float((mdd_b > mdd_a).mean())},
        },
    }
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
