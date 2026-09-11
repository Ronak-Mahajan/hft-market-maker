"""Tests for the market-making simulator and the paired evaluation."""
from __future__ import annotations

import math

import numpy as np
import pytest

import evaluate
from market_maker import (ARMS, AvellanedaStoikovMaker, Config,
                          FixedSpreadMaker, Market, SkewOnlyMaker,
                          SpreadMatchedMaker, metrics, model_half_spread,
                          run, simulate_market)

CFG = Config(n_ticks=2_000)


def _hand_market(prices, arrivals, directions, draws) -> Market:
    return Market(np.asarray(prices, dtype=float),
                  np.asarray(arrivals, dtype=bool),
                  np.asarray(directions, dtype=np.int64),
                  np.asarray(draws, dtype=float))


# --- market -----------------------------------------------------------------

def test_market_is_reproducible():
    a, b = simulate_market(CFG, 7), simulate_market(CFG, 7)
    assert np.array_equal(a.prices, b.prices)
    assert np.array_equal(a.fill_draws, b.fill_draws)
    assert not np.array_equal(simulate_market(CFG, 8).prices, a.prices)


def test_strategies_face_identical_luck():
    """The comparison is paired: same path, same arrivals, same fill draws.
    Under common random numbers a quote that is always further from the mid
    than the benchmark's can only be filled on a tick where the benchmark's
    would also have been filled (same uniform, smaller probability), so A-S
    fills must be a subset of fixed-spread fills tick by tick, on the same
    side. Without shared draws this would fail on most seeds.

    gamma = 0.1 keeps the A-S near quote (half-spread 0.0995 minus at most
    |q| * 4e-5) outside the benchmark's 0.05 for any inventory reachable in
    2,000 ticks, and the inventory cap is lifted so that it cannot create a
    fill for one maker that the other was blocked from taking."""
    cfg = Config(n_ticks=2_000, risk_aversion=0.1, max_inventory=10 ** 6)
    m = simulate_market(cfg, 3)
    f, a = run(FixedSpreadMaker(cfg), m), run(AvellanedaStoikovMaker(cfg), m)
    assert f.pnl.shape == a.pnl.shape == (cfg.n_ticks,)
    as_filled, fx_filled = a.fills != 0, f.fills != 0
    assert 0 < as_filled.sum() < fx_filled.sum()
    assert np.all(fx_filled[as_filled]), "A-S filled on a tick the benchmark did not"
    assert np.array_equal(a.fills[as_filled], f.fills[as_filled]), \
        "same tick, different side"
    assert np.all(fx_filled <= m.arrivals), "a fill without an arrival"


# --- quoting rules ----------------------------------------------------------

def test_reservation_price_leans_against_inventory():
    """Long inventory must push quotes DOWN (encouraging sells), short UP."""
    for S in (AvellanedaStoikovMaker, SkewOnlyMaker):
        s = S(CFG)
        mid = 100.0
        s.inventory = 0
        flat_bid, flat_ask = s.quote(mid, 0)
        s.inventory = 50
        long_bid, long_ask = s.quote(mid, 0)
        s.inventory = -50
        short_bid, short_ask = s.quote(mid, 0)
        assert long_ask < flat_ask and long_bid < flat_bid
        assert short_ask > flat_ask and short_bid > flat_bid


def test_skew_magnitude_is_gamma_q_sigma2_tau():
    """Units: the lean of the quote midpoint below the mid is exactly
    q * gamma * sigma^2 * tau, with sigma per tick and tau = 1 - t/n_ticks the
    normalised horizon remaining, for both arms that skew. At the defaults
    that is 100 * 0.5 * 0.02^2 = $0.02 at max inventory: one fifth of the
    fill-decay length 1/kappa = $0.10, which is why the skew is material on
    the fill-probability yardstick even though it is small against the
    per-tick price move."""
    cfg = CFG
    for S in (AvellanedaStoikovMaker, SkewOnlyMaker):
        s = S(cfg)
        for q in (100, -100, 30):
            s.inventory = q
            for t, tau in ((0, 1.0), (cfg.n_ticks // 2, 0.5)):
                bid, ask = s.quote(100.0, t)
                lean = 100.0 - (bid + ask) / 2.0
                expected = q * cfg.risk_aversion * cfg.volatility ** 2 * tau
                assert lean == pytest.approx(expected, rel=1e-9, abs=1e-15)
    s = AvellanedaStoikovMaker(cfg)
    s.inventory = 100
    bid, ask = s.quote(100.0, 0)
    assert 100.0 - (bid + ask) / 2.0 == pytest.approx(0.02)


def test_risk_aversion_decays_to_zero_at_the_horizon():
    """Both the skew and the spread carry (T - t). An earlier version omitted
    the horizon entirely, which is the inventory intuition without the model.
    The skew decay is the material half: 0.012 -> ~0 for q = 60; the spread
    decay is gamma*sigma^2 = 2e-4 over the run."""
    s = AvellanedaStoikovMaker(CFG)
    s.inventory = 60
    early_bid, early_ask = s.quote(100.0, 0)
    late_bid, late_ask = s.quote(100.0, CFG.n_ticks - 1)
    early_skew = abs((early_bid + early_ask) / 2 - 100.0)
    late_skew = abs((late_bid + late_ask) / 2 - 100.0)
    assert early_skew == pytest.approx(60 * 0.5 * 0.02 ** 2)
    assert late_skew < early_skew / 1000
    assert (early_ask - early_bid) - (late_ask - late_bid) == pytest.approx(
        CFG.risk_aversion * CFG.volatility ** 2 * (1.0 - 1.0 / CFG.n_ticks) * 1.0,
        rel=1e-9)


def test_spread_matches_the_closed_form():
    s = AvellanedaStoikovMaker(CFG)
    s.inventory = 0
    bid, ask = s.quote(100.0, 0)
    g, k, s2 = CFG.risk_aversion, CFG.kappa, CFG.volatility ** 2
    expected = g * s2 * 1.0 + (2.0 / g) * math.log1p(g / k)
    assert (ask - bid) == pytest.approx(expected, rel=1e-12)
    assert model_half_spread(CFG) == pytest.approx(expected / 2, rel=1e-12)


def test_spread_matched_control_quotes_the_model_spread_without_skew():
    """The control must quote exactly the A-S spread (at tau=1) symmetrically
    around the mid, regardless of inventory."""
    c, a = SpreadMatchedMaker(CFG), AvellanedaStoikovMaker(CFG)
    cb, ca = c.quote(100.0, 0)
    ab, aa = a.quote(100.0, 0)
    assert (ca - cb) == pytest.approx(aa - ab, rel=1e-12)
    assert (ca + cb) / 2 == pytest.approx(100.0)
    assert c.spread == pytest.approx(2 * model_half_spread(CFG), rel=1e-12)
    c.inventory = 80
    assert c.quote(100.0, 0) == (cb, ca)
    assert c.quote(100.0, CFG.n_ticks - 1) == (cb, ca)


def test_skew_only_keeps_the_benchmark_spread():
    s = SkewOnlyMaker(CFG)
    s.inventory = 50
    bid, ask = s.quote(100.0, 0)
    assert ask - bid == pytest.approx(CFG.fixed_spread, rel=1e-12)
    assert (bid + ask) / 2 < 100.0
    s.inventory = 0
    assert s.quote(100.0, 0) == FixedSpreadMaker(CFG).quote(100.0, 0)


def test_quote_crossing_guard_clamps_and_counts():
    """With gamma = 10 the lean at |q| = 100 is 0.40 against a model
    half-spread of ~0.071, so a long maker's raw ask sits below the mid.
    step() must clamp the crossing side to the mid, keep bid <= mid <= ask,
    and count the event; a non-crossing quote must not be counted."""
    cfg = Config(n_ticks=10, risk_aversion=10.0)
    s = AvellanedaStoikovMaker(cfg)
    s.inventory = 100
    raw_bid, raw_ask = s.quote(100.0, 0)
    assert raw_ask < 100.0 and raw_bid < raw_ask     # this regime crosses
    s.step(100.0, 0, False, 1, 0.5)
    bid, ask = s.last_quote
    assert bid <= 100.0 <= ask
    assert ask == pytest.approx(100.0) and bid == pytest.approx(raw_bid)
    assert s.n_crossed == 1
    s.inventory = -100                                # short: bid crosses up
    s.step(100.0, 1, False, 1, 0.5)
    bid, ask = s.last_quote
    assert bid == pytest.approx(100.0) and ask > 100.0
    assert s.n_crossed == 2
    s.inventory = 0
    s.step(100.0, 2, False, 1, 0.5)
    assert s.n_crossed == 2
    # the guard is inert at the defaults
    m = simulate_market(CFG, 4)
    for S in ARMS:
        assert run(S(CFG), m).n_crossed == 0


# --- accounting -------------------------------------------------------------

def test_pnl_accounting_round_trip_captures_the_spread():
    """Flat mid, seller hits our bid at tick 0, buyer lifts our offer at
    tick 1: the round trip on the benchmark must earn exactly
    spread * order_size, and P&L must be marked to the mid at every tick."""
    cfg = Config(n_ticks=3, fixed_spread=0.10, order_size=10)
    m = _hand_market([100.0, 100.0, 101.0], [True, True, True],
                     [-1, 1, -1], [0.0, 0.0, 0.99])
    s = run(FixedSpreadMaker(cfg), m)
    assert s.n_fills == 2 and s.inventory == 0
    assert s.cash == pytest.approx(1.0)                # -999.5 + 1000.5
    assert s.pnl[0] == pytest.approx(-999.5 + 10 * 100.0)   # 0.5, marked at mid
    assert s.pnl[1] == pytest.approx(1.0)
    assert s.pnl[2] == pytest.approx(1.0)              # flat: price move is irrelevant
    assert list(s.fills) == [1, -1, 0]
    assert s.fill_prices[0] == pytest.approx(99.95)
    assert s.fill_prices[1] == pytest.approx(100.05)


def test_pnl_accounting_reconstructs_from_the_fill_record():
    """On a simulated market, cash must equal the signed sum of fill prices
    times size, the inventory path must be the running sum of fills, and the
    P&L at every tick must equal running cash plus inventory marked at that
    tick's mid."""
    cfg = Config(n_ticks=500)
    m = simulate_market(cfg, 5)
    for S in ARMS:
        s = run(S(cfg), m)
        assert s.n_fills > 0 and s.n_fills == int((s.fills != 0).sum())
        cash_flow = -s.fills.astype(float) * s.fill_prices * cfg.order_size
        cash = np.cumsum(cash_flow)
        inv = np.cumsum(s.fills.astype(np.int64)) * cfg.order_size
        assert cash[-1] == pytest.approx(s.cash, abs=1e-6)
        assert np.array_equal(inv, s.inv_path)
        np.testing.assert_allclose(s.pnl, cash + inv * m.prices, rtol=0, atol=1e-6)


def test_inventory_limit_is_respected():
    cfg = Config(n_ticks=4_000, max_inventory=40)
    m = simulate_market(cfg, 11)
    for S in ARMS:
        s = run(S(cfg), m)
        assert np.abs(s.inv_path).max() <= cfg.max_inventory


def test_metrics_label_the_horizon_statistic_as_t_stat():
    """mean/std * sqrt(n) of step P&L is the whole-horizon Sharpe, i.e. the
    t-statistic of the mean step P&L. It is not annualised and must not be
    published as 'sharpe'."""
    cfg = Config(n_ticks=300)
    s = run(FixedSpreadMaker(cfg), simulate_market(cfg, 1))
    m = metrics(s)
    assert "t_stat" in m and "sharpe" not in m
    d = np.diff(s.pnl)
    assert m["t_stat"] == pytest.approx(d.mean() / (d.std() + 1e-12) * math.sqrt(len(d)))
    assert m["max_drawdown"] <= 0
    assert m["n_fills"] == s.n_fills and m["n_crossed"] == 0


# --- the headline properties -------------------------------------------------

def test_avellaneda_stoikov_reduces_inventory_risk():
    """The headline claim, as a test: A-S must cut inventory dispersion on a
    clear majority of paired seeds. P&L is deliberately NOT asserted -- across
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
    """The model spread buys its risk reduction with volume; the skew alone
    does not have to (it moves fills from one side to the other)."""
    m = simulate_market(CFG, 2)
    f = metrics(run(FixedSpreadMaker(CFG), m))
    c = metrics(run(SpreadMatchedMaker(CFG), m))
    a = metrics(run(AvellanedaStoikovMaker(CFG), m))
    assert a["n_fills"] < f["n_fills"]
    assert c["n_fills"] < f["n_fills"]


# --- evaluate.py ------------------------------------------------------------

def test_evaluate_report_schema_v2():
    cfg = Config(n_ticks=300)
    rows = evaluate.evaluate(cfg, range(3))
    assert len(rows) == 3 * len(ARMS)
    rep = evaluate.build_report(cfg, rows, 3)
    assert rep["schema"] == 2 and rep["n_seeds"] == 3
    assert set(rep["arms"]) == {cls.slug for cls in ARMS}
    assert set(rep["comparisons"]) == {f"{a}_vs_{b}" for a, b, _ in evaluate.COMPARISONS}
    c = rep["comparisons"]["as_vs_fixed"]["metrics"]
    assert isinstance(c, dict) and set(c) == {k for k, _ in evaluate.METRICS}
    s = c["inventory_std"]
    assert {"mean", "se", "ci95", "median", "win_rate", "significant", "n"} <= set(s)
    assert s["n"] == 3
    # orientation: positive = a better, so inventory_std is fixed - as
    arrays = evaluate.to_arrays(rows)
    d = arrays["fixed"]["inventory_std"] - arrays["as"]["inventory_std"]
    assert s["mean"] == pytest.approx(d.mean())
    d2 = arrays["as"]["final_pnl"] - arrays["fixed"]["final_pnl"]
    assert c["final_pnl"]["mean"] == pytest.approx(d2.mean())
    assert rep["arms"]["spread_matched"]["mean"]["n_fills"] == pytest.approx(
        arrays["spread_matched"]["n_fills"].mean())
    assert rep["arm_spreads"]["spread_matched"] == pytest.approx(2 * model_half_spread(cfg))


def test_evaluate_seeds_csv_round_trips(tmp_path):
    cfg = Config(n_ticks=200)
    rows = evaluate.evaluate(cfg, range(2))
    path = tmp_path / "seeds.csv"
    evaluate.write_seeds_csv(str(path), rows)
    lines = path.read_text().splitlines()
    assert lines[0].split(",")[:7] == ["seed", "arm", "inventory_sigma",
                                       "max_drawdown", "fills", "final_pnl", "t_stat"]
    assert len(lines) == 1 + 2 * len(ARMS)
    first = lines[1].split(",")
    assert first[0] == "0" and first[1] == "fixed"
    assert float(first[2]) == pytest.approx(rows[0][2]["inventory_std"])


def test_gamma_sweep_shares_the_benchmark_and_varies_the_model():
    cfg = Config(n_ticks=300)
    sw = evaluate.sweep(cfg, range(2), [0.1, 0.5])
    assert sw["kind"] == "gamma_sweep" and [pt["gamma"] for pt in sw["points"]] == [0.1, 0.5]
    b0 = sw["points"][0]["arms"]["fixed"]["mean"]
    b1 = sw["points"][1]["arms"]["fixed"]["mean"]
    assert b0 == b1
    assert sw["points"][0]["arm_spreads"]["as"] != sw["points"][1]["arm_spreads"]["as"]
    assert evaluate.parse_grid("2, 0.5,0.5") == [0.5, 2.0]
