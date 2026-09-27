"""Paired multi-seed replay of a horizon-free skew rule against the benchmark.

The rule
--------
Two quoters on the same market, at the market_maker.Config defaults:

    A   symmetric quotes at the hand-picked 0.10 spread (the benchmark)
    B   the same 0.10 spread around a reservation price r = s - q*gamma*sigma^2,
        gamma = 0.5, with no (T - t) term, so the lean is the same at every tick

B is the inventory intuition without the model's horizon. market_maker.py's
SkewOnlyMaker is the same rule with the lean scaled by the time remaining.

Pairing
-------
The price path, arrivals, directions and per-tick fill uniforms are drawn once
per seed, in the same order as market_maker.simulate_market, and both quoters
consume the same draws, so the per-seed difference between them is the rule's
and not the fill luck's. The implementation here is independent of
market_maker.py.

Outputs
-------
    audit_results.json   per-seed metrics for A and B, and the paired summary:
                         the P&L difference with a 95% CI and win rate, each
                         quoter's profitable share, and the per-seed
                         inventory-sigma and drawdown reductions

Usage:
    python audit.py --seeds 200            # writes audit_results.json
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from evaluate import write_json

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class AuditConfig:
    """The market_maker.Config defaults that the rule uses."""
    initial_price: float = 100.0
    drift: float = 0.0001
    volatility: float = 0.02
    dt: float = 1.0
    poisson_rate: float = 0.1
    order_size: int = 10
    n_ticks: int = 10_000
    naive_spread: float = 0.10
    risk_aversion: float = 0.5       # gamma
    kappa: float = 10.0              # fill-intensity decay
    max_inventory: int = 100


def draw_market(cfg: AuditConfig, seed: int):
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


class RuleMaker:
    def __init__(self, cfg: AuditConfig):
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


class Naive(RuleMaker):
    """A: symmetric quotes at the hand-picked spread."""

    def quote(self, mid: float) -> tuple[float, float]:
        h = self.cfg.naive_spread / 2.0
        return mid - h, mid + h


class InventoryAware(RuleMaker):
    """B: the same spread, skewed by -q*gamma*sigma^2 (no horizon)."""

    def quote(self, mid: float) -> tuple[float, float]:
        cfg = self.cfg
        r = mid - self.inventory * cfg.risk_aversion * cfg.volatility ** 2
        h = cfg.naive_spread / 2.0
        return r - h, r + h


def replay(cls: type[RuleMaker], cfg: AuditConfig, market) -> RuleMaker:
    """Run one quoter over a market from draw_market."""
    prices, arrivals, directions, draws = market
    s = cls(cfg)
    tick = s.tick
    for t in range(cfg.n_ticks):
        tick(t, prices[t], arrivals[t], directions[t], draws[t])
    return s


def rule_metrics(s: RuleMaker) -> dict[str, float]:
    pnl = s.pnl
    return {
        "final_pnl": float(pnl[-1]),
        "max_drawdown": float(np.min(pnl - np.maximum.accumulate(pnl))),
        "inventory_std": float(s.inv_path.std()),
        "mean_abs_inventory": float(np.abs(s.inv_path).mean()),
    }


def run_seed(cfg: AuditConfig, seed: int) -> dict[str, dict[str, float]]:
    market = draw_market(cfg, seed)
    return {name: rule_metrics(replay(cls, cfg, market))
            for name, cls in (("A", Naive), ("B", InventoryAware))}


def pct(v: np.ndarray, q: float) -> float:
    return float(np.percentile(v, q))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--json", default="audit_results.json")
    args = p.parse_args()

    cfg = AuditConfig()
    rows = [run_seed(cfg, s) for s in range(args.seeds)]

    def col(strategy: str, key: str) -> np.ndarray:
        return np.array([r[strategy][key] for r in rows])

    pnl_a, pnl_b = col("A", "final_pnl"), col("B", "final_pnl")
    istd_a, istd_b = col("A", "inventory_std"), col("B", "inventory_std")
    mdd_a, mdd_b = col("A", "max_drawdown"), col("B", "max_drawdown")
    inv_red = (1.0 - istd_b / istd_a) * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        mdd_red = np.where(mdd_a != 0, (1.0 - mdd_b / mdd_a) * 100.0, np.nan)

    n = len(rows)
    d_pnl = pnl_b - pnl_a
    se_pnl = float(d_pnl.std(ddof=1) / math.sqrt(n))

    print(f"{n} paired seeds: A = benchmark, B = horizon-free skew rule\n")
    print(f"{'metric':<30}{'p5':>12}{'median':>12}{'p95':>12}")
    for lab, v in (("A final P&L", pnl_a), ("B final P&L", pnl_b)):
        print(f"{lab:<30}{pct(v, 5):>12,.0f}{np.median(v):>12,.0f}{pct(v, 95):>12,.0f}")
    print()
    print(f"A profitable in {(pnl_a > 0).mean() * 100:.1f}% of seeds; "
          f"B in {(pnl_b > 0).mean() * 100:.1f}%")
    print(f"B beats A on P&L in {(pnl_b > pnl_a).mean() * 100:.1f}% of seeds; "
          f"paired mean B-A = {d_pnl.mean():,.0f} "
          f"[{d_pnl.mean() - 1.96 * se_pnl:,.0f}, {d_pnl.mean() + 1.96 * se_pnl:,.0f}]")
    print(f"inventory-std reduction: median {np.median(inv_red):+.1f}%  "
          f"(p5 {pct(inv_red, 5):+.1f}%, p95 {pct(inv_red, 95):+.1f}%), "
          f"B lower in {(istd_b < istd_a).mean() * 100:.1f}% of seeds")
    finite = mdd_red[~np.isnan(mdd_red)]
    print(f"max-drawdown reduction : median {np.median(finite):+.1f}%  "
          f"(p5 {pct(finite, 5):+.1f}%, p95 {pct(finite, 95):+.1f}%), "
          f"B shallower in {(mdd_b > mdd_a).mean() * 100:.1f}% of seeds")

    out = {
        "schema": SCHEMA_VERSION,
        "model": "A: fixed 0.10 spread, symmetric; B: the same spread around "
                 "r = s - q*gamma*sigma^2 with gamma=0.5 and no horizon term",
        "paired": True,
        "n_seeds": n,
        "config": cfg.__dict__,
        "per_seed": [{"seed": i, "A": r["A"], "B": r["B"]} for i, r in enumerate(rows)],
        "summary": {
            "pnl_A": {"p5": pct(pnl_a, 5), "median": float(np.median(pnl_a)),
                      "p95": pct(pnl_a, 95), "profitable_share": float((pnl_a > 0).mean())},
            "pnl_B": {"p5": pct(pnl_b, 5), "median": float(np.median(pnl_b)),
                      "p95": pct(pnl_b, 95), "profitable_share": float((pnl_b > 0).mean())},
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
    write_json(args.json, out)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
