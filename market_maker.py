"""Market-making strategies and an event-driven backtester.

What this simulates
-------------------
A single market maker quoting a two-sided book against Poisson order flow, with
a mid-price following geometric Brownian motion. Fills are stochastic: a quote
posted at distance delta from the mid is hit with intensity

    lambda(delta) = A * exp(-kappa * delta)

which is the Avellaneda & Stoikov (2008) execution model. kappa controls how
fast fill probability decays as you quote further from fair value, so quoting
wide earns more per fill and gets fewer of them.

The question the simulator exists to answer
-------------------------------------------
A market maker who quotes symmetrically around the mid accumulates inventory
whenever order flow is one-sided, and that inventory is directional risk they
were never paid to take. Avellaneda-Stoikov says: skew your quotes around a
RESERVATION PRICE that leans away from your position, and set the spread from
the model rather than by hand. This measures whether that actually pays.

Avellaneda-Stoikov, implemented properly
----------------------------------------
    reservation price   r = s - q * gamma * sigma^2 * (T - t)
    optimal half-spread d = [ gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/kappa) ] / 2
    bid = r - d,   ask = r + d

Both terms carry (T - t): risk aversion to inventory decays to zero at the
terminal time because there is no longer any horizon over which the position
can move against you. An earlier version of this file skewed by -q*gamma*sigma^2
with no time dependence and then quoted a HAND-PICKED fixed spread around it,
which is the inventory intuition without the model — it cannot widen when the
horizon is long, and it has no notion of terminal liquidation.

Reproducibility
---------------
Every strategy is evaluated on IDENTICAL price paths and IDENTICAL fill draws
(common random numbers), so a difference between two strategies is attributable
to the strategies and not to luck. Results are reported over many independent
seeds with confidence intervals, because a single seed of this simulator is
close to meaningless — see evaluate.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Config", "simulate_market", "MarketMaker", "FixedSpreadMaker",
           "AvellanedaStoikovMaker", "run", "metrics"]


@dataclass(frozen=True)
class Config:
    """Simulation parameters. Frozen so a run cannot mutate its own settings."""

    initial_price: float = 100.0
    drift: float = 0.0001            # mu, per tick
    volatility: float = 0.02         # sigma, per tick
    dt: float = 1.0

    poisson_rate: float = 0.1        # market-order arrival intensity
    order_size: int = 10
    kappa: float = 10.0              # fill-intensity decay, lambda = A e^{-kappa d}
    fill_scale: float = 1.0          # A

    n_ticks: int = 10_000
    max_inventory: int = 100

    fixed_spread: float = 0.10       # the naive benchmark's hand-picked spread
    risk_aversion: float = 0.5       # gamma


@dataclass
class Market:
    """One realisation of the market: prices, order arrivals, and the uniform
    draws that decide fills. Shared across strategies so comparisons are paired."""

    prices: np.ndarray
    arrivals: np.ndarray
    directions: np.ndarray
    fill_draws: np.ndarray


def simulate_market(cfg: Config, seed: int) -> Market:
    """Draw one market. The fill draws are generated here, not inside the
    strategy loop, so every strategy faces the same luck."""
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
    return Market(prices, arrivals, directions, fill_draws)


class MarketMaker:
    """Base strategy. Subclasses implement quote()."""

    name = "base"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.inventory = 0
        self.cash = 0.0
        self.pnl = np.empty(cfg.n_ticks)
        self.inv_path = np.empty(cfg.n_ticks, dtype=np.int64)
        self.n_fills = 0

    def quote(self, mid: float, t: int) -> tuple[float, float]:
        raise NotImplementedError

    def step(self, mid: float, t: int, arrived: bool, direction: int,
             draw: float) -> None:
        cfg = self.cfg
        bid, ask = self.quote(mid, t)
        if arrived:
            # lambda(delta) = A exp(-kappa delta), clipped to a probability
            if direction == 1:                       # buyer lifts our offer
                p = min(cfg.fill_scale * math.exp(-cfg.kappa * (ask - mid)), 1.0)
                if draw < p and self.inventory > -cfg.max_inventory:
                    self.inventory -= cfg.order_size
                    self.cash += ask * cfg.order_size
                    self.n_fills += 1
            else:                                    # seller hits our bid
                p = min(cfg.fill_scale * math.exp(-cfg.kappa * (mid - bid)), 1.0)
                if draw < p and self.inventory < cfg.max_inventory:
                    self.inventory += cfg.order_size
                    self.cash -= bid * cfg.order_size
                    self.n_fills += 1
        self.pnl[t] = self.cash + self.inventory * mid
        self.inv_path[t] = self.inventory


class FixedSpreadMaker(MarketMaker):
    """Benchmark: symmetric quotes at a hand-picked spread, no inventory term."""

    name = "fixed spread"

    def quote(self, mid: float, t: int) -> tuple[float, float]:
        h = self.cfg.fixed_spread / 2.0
        return mid - h, mid + h


class AvellanedaStoikovMaker(MarketMaker):
    """Avellaneda & Stoikov (2008), including the terminal horizon.

    Both the inventory skew and the spread scale with the time remaining, so
    the maker becomes progressively less willing to warehouse risk as the
    horizon closes and quotes tighten toward the end.
    """

    name = "Avellaneda-Stoikov"

    def quote(self, mid: float, t: int) -> tuple[float, float]:
        cfg = self.cfg
        tau = 1.0 - t / cfg.n_ticks                  # normalised time remaining
        g, s2 = cfg.risk_aversion, cfg.volatility ** 2
        reservation = mid - self.inventory * g * s2 * tau
        half = 0.5 * (g * s2 * tau
                      + (2.0 / g) * math.log1p(g / cfg.kappa))
        return reservation - half, reservation + half


def run(strategy: MarketMaker, market: Market) -> MarketMaker:
    for t in range(strategy.cfg.n_ticks):
        strategy.step(market.prices[t], t, market.arrivals[t],
                      market.directions[t], market.fill_draws[t])
    return strategy


def metrics(s: MarketMaker) -> dict[str, float]:
    pnl = s.pnl
    step_pnl = np.diff(pnl)
    peak = np.maximum.accumulate(pnl)
    return {
        "final_pnl": float(pnl[-1]),
        "pnl_std": float(step_pnl.std()),
        "sharpe": float(step_pnl.mean() / (step_pnl.std() + 1e-12)
                        * math.sqrt(len(step_pnl))),
        "max_drawdown": float(np.min(pnl - peak)),
        "inventory_std": float(s.inv_path.std()),
        "max_abs_inventory": float(np.abs(s.inv_path).max()),
        "n_fills": float(s.n_fills),
    }
