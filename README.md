# Inventory-Aware Market Making: does Avellaneda–Stoikov actually pay?

An event-driven backtester for a single market maker quoting a two-sided book
against Poisson order flow, built to answer one question: **does skewing quotes
against your inventory measurably reduce risk, and what does it cost?**

```bash
pip install -r requirements.txt
python evaluate.py --seeds 500      # the paired comparison
python -m pytest -q                 # 9 tests
```

## The problem

A market maker quoting symmetrically around the mid accumulates inventory
whenever order flow is one-sided, and that inventory is directional risk they
were never paid to take. Avellaneda & Stoikov (2008) prescribe quoting around a
**reservation price** that leans away from your position, with a spread the
model derives rather than one you pick:

```
reservation price   r = s − q·γ·σ²·(T − t)
optimal half-spread d = ½[ γσ²(T − t) + (2/γ)·ln(1 + γ/κ) ]
bid = r − d,   ask = r + d
```

Both terms carry `(T − t)`: aversion to inventory decays to zero at the terminal
time, because there is no longer any horizon over which a position can move
against you. Fills follow the same paper's execution model, `λ(δ) = A·e^{−κδ}`,
so quoting further from fair value earns more per fill and gets fewer of them.

## The result

500 seeds, paired. Both strategies see the **same price path, the same order
arrivals and the same uniform fill draws**, so the market's variance is
differenced out and what remains is attributable to the strategies.

| metric | fixed spread | Avellaneda–Stoikov | difference (95% CI) | A–S wins |
|---|---|---|---|---|
| inventory σ | 50.5 | **42.0** | **8.6 [7.8, 9.4]** | 85% |
| max drawdown | −29,177 | **−25,206** | **3,971 [521, 7,421]** | 74% |
| fills | 553 | 352 | **−201 [−202, −199]** | 0% |
| final P&L | −1,423 | −1,938 | −515 [−3,994, +2,965] | 54% |
| Sharpe | −0.02 | 0.03 | 0.05 [−0.00, 0.10] | 53% |

**Inventory risk falls 17% and maximum drawdown 14%, both significant at 95%.
Profit does not improve — that difference is indistinguishable from zero.** The
mechanism is visible in the last row: A–S quotes wider and takes **36% fewer
fills**. It buys risk reduction with volume, which is what the model claims it
does, and the experiment neither flatters nor contradicts it.

## Why the evaluation is built this way

An earlier version of this project reported a **single seed** — a $22,322 loss
turned into a $7,336 profit, quoted as a "+132% improvement". That number
reproduces exactly. It is also meaningless on its own: across 200 seeds the same
statistic ranges from **−412% to +462%**, and the naive benchmark is profitable
44% of the time. A single path of a 10,000-tick GBM with Poisson fills carries
enormous terminal variance, so a one-seed comparison measures the seed rather
than the strategy.

Two changes make the difference resolvable:

1. **Common random numbers.** Fill draws are generated with the market, not
   inside the strategy loop, so both makers face identical luck. Without this
   the strategy effect is buried in path noise.
2. **Distributions, not point estimates.** Every number above is a mean over 500
   paired seeds with a 95% confidence interval and a win rate.

The earlier version also described features it did not implement — queue-position
dependent fills, a terminal-utility objective, quadratic inventory penalties, and
risk aversion adapting to realised volatility. None of those existed in the code;
the reservation price had no `(T − t)` term and the spread was hand-picked rather
than derived. The model is now implemented as specified, and the claims are
limited to what the code does and the experiment shows.

## Limitations

- Fills are modelled as an intensity in distance from the mid. There is **no
  order book and no queue position**, so nothing here speaks to microstructure
  effects that depend on standing in a real queue.
- The mid is exogenous GBM: quoting does not move the price, and there is no
  adverse selection from informed flow beyond what the fill intensity implies.
- One asset, one maker, no latency, no fees.

## Layout

```
market_maker.py        strategies, market simulation, metrics
evaluate.py            paired multi-seed comparison with confidence intervals
test_market_maker.py   9 tests, including the closed-form spread and the
                       (T − t) decay that the earlier version omitted
simulation.ipynb       original exploratory notebook, kept for the plots
```
