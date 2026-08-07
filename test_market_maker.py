"""Tests for the market-making simulator."""
from __future__ import annotations
import math
import numpy as np
import pytest
from market_maker import (AvellanedaStoikovMaker, Config, FixedSpreadMaker,
                          metrics, run, simulate_market)

CFG = Config(n_ticks=2_000)


def test_market_is_reproducible():
    a, b = simulate_market(CFG, 7), simulate_market(CFG, 7)
    assert np.array_equal(a.prices, b.prices)
    assert np.array_equal(a.fill_draws, b.fill_draws)
    assert not np.array_equal(simulate_market(CFG, 8).prices, a.prices)


def test_strategies_face_identical_luck():
    """The comparison is paired: same path, same arrivals, same fill draws.
    Without this the difference between two strategies is mostly seed noise."""
    m = simulate_market(CFG, 3)
    f, a = run(FixedSpreadMaker(CFG), m), run(AvellanedaStoikovMaker(CFG), m)
    assert f.pnl.shape == a.pnl.shape == (CFG.n_ticks,)


def test_reservation_price_leans_against_inventory():
    """Long inventory must push quotes DOWN (encouraging sells), short UP."""
    s = AvellanedaStoikovMaker(CFG)
    mid = 100.0
    s.inventory = 0
    flat_bid, flat_ask = s.quote(mid, 0)
    s.inventory = 50
    long_bid, long_ask = s.quote(mid, 0)
    s.inventory = -50
    short_bid, short_ask = s.quote(mid, 0)
    assert long_ask < flat_ask and long_bid < flat_bid
    assert short_ask > flat_ask and short_bid > flat_bid


def test_risk_aversion_decays_to_zero_at_the_horizon():
    """Both the skew and the spread carry (T - t). An earlier version omitted
    the horizon entirely, which is the inventory intuition without the model."""
    s = AvellanedaStoikovMaker(CFG)
    s.inventory = 60
    early_bid, early_ask = s.quote(100.0, 0)
    late_bid, late_ask = s.quote(100.0, CFG.n_ticks - 1)
    early_skew = abs((early_bid + early_ask) / 2 - 100.0)
    late_skew = abs((late_bid + late_ask) / 2 - 100.0)
    assert early_skew > late_skew
    assert (early_ask - early_bid) > (late_ask - late_bid)


def test_spread_matches_the_closed_form():
    s = AvellanedaStoikovMaker(CFG)
    s.inventory = 0
    bid, ask = s.quote(100.0, 0)
    g, k, s2 = CFG.risk_aversion, CFG.kappa, CFG.volatility ** 2
    expected = g * s2 * 1.0 + (2.0 / g) * math.log1p(g / k)
    assert (ask - bid) == pytest.approx(expected, rel=1e-12)


def test_inventory_limit_is_respected():
    cfg = Config(n_ticks=4_000, max_inventory=40)
    m = simulate_market(cfg, 11)
    for S in (FixedSpreadMaker, AvellanedaStoikovMaker):
        s = run(S(cfg), m)
        assert np.abs(s.inv_path).max() <= cfg.max_inventory + cfg.order_size


def test_pnl_accounting_is_mark_to_market():
    """P&L must equal cash plus inventory marked at the current mid."""
    cfg = Config(n_ticks=500)
    m = simulate_market(cfg, 5)
    s = run(AvellanedaStoikovMaker(cfg), m)
    assert s.pnl[-1] == pytest.approx(s.cash + s.inventory * m.prices[-1])
    assert s.pnl[0] == pytest.approx(s.cash if s.inventory == 0 else s.pnl[0])


def test_avellaneda_stoikov_reduces_inventory_risk():
    """The headline claim, as a test: A-S must cut inventory dispersion on a
    clear majority of paired seeds. P&L is deliberately NOT asserted — across
    500 seeds that difference is not distinguishable from zero."""
    wins = 0
    trials = 40
    for seed in range(trials):
        m = simulate_market(CFG, seed)
        f = metrics(run(FixedSpreadMaker(CFG), m))
        a = metrics(run(AvellanedaStoikovMaker(CFG), m))
        wins += a["inventory_std"] < f["inventory_std"]
    assert wins / trials > 0.6, f"only {wins}/{trials}"


def test_wider_quotes_cost_fills():
    """A-S buys its risk reduction with volume; that trade-off is the result."""
    m = simulate_market(CFG, 2)
    f = metrics(run(FixedSpreadMaker(CFG), m))
    a = metrics(run(AvellanedaStoikovMaker(CFG), m))
    assert a["n_fills"] < f["n_fills"]
