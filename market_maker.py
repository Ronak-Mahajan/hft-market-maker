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
the model rather than by hand. Those are two separate interventions, so the
experiment has four arms (see evaluate.py):

    fixed spread        symmetric quotes at the hand-picked spread (benchmark)
    spread-matched      symmetric quotes at the model's spread, no skew
    skew-only           model skew around the hand-picked spread
    Avellaneda-Stoikov  model skew and model spread

and the paired differences between them decompose the full model's effect
into a spread part and a skew part.

Avellaneda-Stoikov, implemented properly
----------------------------------------
    reservation price   r = s - q * gamma * sigma^2 * (T - t)
    optimal half-spread d = [ gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/kappa) ] / 2
    bid = r - d,   ask = r + d

Both terms carry (T - t): risk aversion to inventory decays to zero at the
terminal time because there is no longer any horizon over which the position
can move against you. An earlier version of this file skewed by -q*gamma*sigma^2
with no time dependence and then quoted a HAND-PICKED fixed spread around it,
which is the inventory intuition without the model; that design is kept here as
the skew-only arm because it is the cleanest test of the skew on its own.

Units
-----
sigma is the per-tick log-volatility and (T - t) is the normalised fraction of
the horizon remaining, tau = 1 - t/n_ticks in [0, 1]. So gamma*sigma^2*tau is
NOT the paper's formula in tick units (that would carry (n_ticks - t) and be
10,000x larger at the defaults; in the paper's own units at these parameters the
skew would be hundreds of dollars a share). Read gamma as a risk aversion in
per-horizon units. On the yardstick that governs fills, 1/kappa = $0.10, the
default lean of 100 * 0.5 * 0.02^2 = $0.02 at max inventory is a 20% shift in
the fill-decay length, which moves the near-side fill probability from 0.377 to
0.460 and the far side to 0.308. The skew is material; see the units test.

Reproducibility
---------------
Every strategy is evaluated on IDENTICAL price paths and IDENTICAL fill draws
(common random numbers), so a difference between two strategies is attributable
to the strategies and not to luck. Results are reported over many independent
seeds with confidence intervals, because a single seed of this simulator is
close to meaningless -- see evaluate.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Config", "Market", "simulate_market", "MarketMaker",
           "FixedSpreadMaker", "SpreadMatchedMaker", "SkewOnlyMaker",
           "AvellanedaStoikovMaker", "ARMS", "model_half_spread",
           "reservation_skew", "run", "metrics"]


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
    strategy loop, so every strategy faces the same luck.

    The order of RNG consumption (normals, arrivals, directions, fill draws)
    is fixed: changing it would change every seed's market and silently break
    comparability with previously published runs."""
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


def model_half_spread(cfg: Config, tau: float = 1.0) -> float:
    """Avellaneda-Stoikov optimal half-spread
    d = [gamma*sigma^2*tau + (2/gamma) ln(1 + gamma/kappa)] / 2.

    tau is the normalised time remaining; tau=1 is the start of the run. The
    tau term is tiny at the defaults (0.5*0.0004 = 2e-4 on a half-spread of
    ~0.0977), so the model spread is essentially constant over the horizon."""
    g = cfg.risk_aversion
    return 0.5 * (g * cfg.volatility ** 2 * tau
                  + (2.0 / g) * math.log1p(g / cfg.kappa))


def reservation_skew(cfg: Config, inventory: int, tau: float) -> float:
    """q*gamma*sigma^2*tau: how far the reservation price sits BELOW the mid.
    Positive when long (quotes lean down to encourage sells)."""
    return inventory * cfg.risk_aversion * cfg.volatility ** 2 * tau


class MarketMaker:
    """Base strategy. Subclasses implement quote().

    step() applies a quote-crossing guard: if a subclass's raw quote puts the
    bid above the mid or the ask below it (a large gamma or sigma can push the
    reservation-price lean past the half-spread), the crossing side is clamped
    to the mid and the event is counted in n_crossed. Without the guard the
    fill probability on that side clips to 1.0 and the maker gives away edge
    with certainty; with it the maker quotes at fair value on that side, which
    is the most aggressive quote a passive maker can post here. quote() itself
    is left raw so the model formulas can be tested directly."""

    name = "base"
    slug = "base"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.inventory = 0
        self.cash = 0.0
        self.pnl = np.empty(cfg.n_ticks)
        self.inv_path = np.empty(cfg.n_ticks, dtype=np.int64)
        # +1: our bid was hit (we bought); -1: our ask was lifted (we sold)
        self.fills = np.zeros(cfg.n_ticks, dtype=np.int8)
        self.fill_prices = np.zeros(cfg.n_ticks)
        self.n_fills = 0
        self.n_crossed = 0
        self.last_quote: tuple[float, float] = (math.nan, math.nan)

    def quote(self, mid: float, t: int) -> tuple[float, float]:
        raise NotImplementedError

    def step(self, mid: float, t: int, arrived: bool, direction: int,
             draw: float) -> None:
        cfg = self.cfg
        bid, ask = self.quote(mid, t)
        if bid > mid or ask < mid:
            self.n_crossed += 1
            if bid > mid:
                bid = mid
            if ask < mid:
                ask = mid
        self.last_quote = (bid, ask)
        if arrived:
            # lambda(delta) = A exp(-kappa delta), clipped to a probability
            if direction == 1:                       # buyer lifts our offer
                p = min(cfg.fill_scale * math.exp(-cfg.kappa * (ask - mid)), 1.0)
                if draw < p and self.inventory > -cfg.max_inventory:
                    self.inventory -= cfg.order_size
                    self.cash += ask * cfg.order_size
                    self.n_fills += 1
                    self.fills[t] = -1
                    self.fill_prices[t] = ask
            else:                                    # seller hits our bid
                p = min(cfg.fill_scale * math.exp(-cfg.kappa * (mid - bid)), 1.0)
                if draw < p and self.inventory < cfg.max_inventory:
                    self.inventory += cfg.order_size
                    self.cash -= bid * cfg.order_size
                    self.n_fills += 1
                    self.fills[t] = 1
                    self.fill_prices[t] = bid
        self.pnl[t] = self.cash + self.inventory * mid
        self.inv_path[t] = self.inventory


class FixedSpreadMaker(MarketMaker):
    """Benchmark: symmetric quotes at a hand-picked spread, no inventory term.

    `spread` overrides cfg.fixed_spread so the same class can serve as a
    symmetric control at any spread."""

    name = "fixed spread"
    slug = "fixed"

    def __init__(self, cfg: Config, spread: float | None = None):
        super().__init__(cfg)
        self.spread = cfg.fixed_spread if spread is None else float(spread)

    def quote(self, mid: float, t: int) -> tuple[float, float]:
        h = self.spread / 2.0
        return mid - h, mid + h


class SpreadMatchedMaker(FixedSpreadMaker):
    """Spread-matched control: symmetric quotes at the Avellaneda-Stoikov
    model spread (evaluated at tau=1), with no inventory skew.

    Its paired difference against the benchmark is the effect of quoting
    wider on its own; the full model's paired difference against THIS arm is
    the effect of the skew at the model spread. The model spread decays by
    gamma*sigma^2 over the horizon (2e-4 at the defaults), which this control
    ignores."""

    name = "spread-matched"
    slug = "spread_matched"

    def __init__(self, cfg: Config):
        super().__init__(cfg, spread=2.0 * model_half_spread(cfg, tau=1.0))


class SkewOnlyMaker(MarketMaker):
    """Avellaneda-Stoikov reservation-price skew around the benchmark's
    hand-picked spread. Isolates the skew from the model spread at the
    benchmark's own volume. This is the design of the original notebook, with
    the (T - t) decay added so the skew is the same function as the full
    model's."""

    name = "skew-only"
    slug = "skew_only"

    def quote(self, mid: float, t: int) -> tuple[float, float]:
        cfg = self.cfg
        tau = 1.0 - t / cfg.n_ticks
        reservation = mid - reservation_skew(cfg, self.inventory, tau)
        h = cfg.fixed_spread / 2.0
        return reservation - h, reservation + h


class AvellanedaStoikovMaker(MarketMaker):
    """Avellaneda & Stoikov (2008), including the terminal horizon.

    Both the inventory skew and the spread scale with the time remaining, so
    the maker becomes progressively less willing to warehouse risk as the
    horizon closes and quotes tighten toward the end.
    """

    name = "Avellaneda-Stoikov"
    slug = "as"

    def quote(self, mid: float, t: int) -> tuple[float, float]:
        cfg = self.cfg
        tau = 1.0 - t / cfg.n_ticks                  # normalised time remaining
        reservation = mid - reservation_skew(cfg, self.inventory, tau)
        half = model_half_spread(cfg, tau)
        return reservation - half, reservation + half


# The four arms of the decomposition, in reporting order.
ARMS: tuple[type[MarketMaker], ...] = (FixedSpreadMaker, SpreadMatchedMaker,
                                       SkewOnlyMaker, AvellanedaStoikovMaker)


def run(strategy: MarketMaker, market: Market) -> MarketMaker:
    # Iterate over Python scalars rather than numpy scalars: the arithmetic is
    # identical IEEE-754 but several times faster in the per-tick loop.
    prices = market.prices.tolist()
    arrivals = market.arrivals.tolist()
    directions = market.directions.tolist()
    draws = market.fill_draws.tolist()
    step = strategy.step
    for t in range(strategy.cfg.n_ticks):
        step(prices[t], t, arrivals[t], directions[t], draws[t])
    return strategy


def metrics(s: MarketMaker) -> dict[str, float]:
    """Per-run summary.

    t_stat is mean(step P&L) / std(step P&L) * sqrt(n_steps): the per-step
    Sharpe ratio scaled to the whole horizon, which is numerically the
    t-statistic of the mean step P&L against zero. It is NOT annualised and
    should not be read as a Sharpe ratio in the usual sense (earlier versions
    of this file called it 'sharpe')."""
    pnl = s.pnl
    step_pnl = np.diff(pnl)
    peak = np.maximum.accumulate(pnl)
    return {
        "final_pnl": float(pnl[-1]),
        "pnl_std": float(step_pnl.std()),
        "t_stat": float(step_pnl.mean() / (step_pnl.std() + 1e-12)
                        * math.sqrt(len(step_pnl))),
        "max_drawdown": float(np.min(pnl - peak)),
        "inventory_std": float(s.inv_path.std()),
        "max_abs_inventory": float(np.abs(s.inv_path).max()),
        "n_fills": float(s.n_fills),
        "n_crossed": float(s.n_crossed),
    }
