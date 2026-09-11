# Inventory-Aware Market Making: does Avellaneda–Stoikov actually pay, and which half of it?

An event-driven backtester for a single market maker quoting a two-sided book
against Poisson order flow, built to answer one question: **does skewing quotes
against your inventory measurably reduce risk, and what does it cost?**

```bash
pip install -r requirements.txt
python evaluate.py --seeds 500        # four paired arms -> results.json, results_seeds.csv
python evaluate.py --seeds 100 --gamma-sweep   # -> results_sweep.json
python audit.py --seeds 200           # the OLD notebook model, paired -> audit_results.json
python -m pytest -q                   # 18 tests
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

## The decomposition question

Avellaneda–Stoikov changes **two** things relative to a naive symmetric quoter:
it skews the quotes around the reservation price, and it sets the spread from
the model, which at these parameters is about twice the benchmark's hand-picked
0.10. A two-arm comparison (benchmark vs full model) cannot say which of the two
does the work: fewer fills alone would shrink inventory dispersion without any
skew at all. So the experiment has **four arms, all run on the same market
draws**:

| arm | spread | skew | isolates |
|---|---|---|---|
| `fixed` | 0.10, hand-picked | none | the benchmark |
| `spread_matched` | the model spread, at τ = 1 | none | what quoting wider does on its own |
| `skew_only` | 0.10, hand-picked | model skew | what the skew does at the benchmark's own volume |
| `as` | the model spread | model skew | the full model |

and four paired comparisons, each reported for every metric with a 95%
confidence interval and a win rate:

```
spread_matched vs fixed          spread effect
skew_only      vs fixed          skew effect at the benchmark spread
as             vs fixed          full model vs benchmark
as             vs spread_matched skew effect at the model spread
```

`(as − fixed) = (spread_matched − fixed) + (as − spread_matched)` seed by seed,
so the last two rows are an exact decomposition of the first.

**Units.** `σ` is the per-tick log-volatility and `(T − t)` is the normalised
fraction of the horizon remaining, `τ = 1 − t/n_ticks ∈ [0, 1]`, so `γσ²τ` is
not the paper's formula in tick units (that would carry `n_ticks − t`) and `γ`
should be read as a per-horizon risk aversion. The lean is judged on the
yardstick that governs fills: with `κ = 10` the fill-decay length is
`1/κ = $0.10`, and the default lean of `100 · 0.5 · 0.02² = $0.02` at maximum
inventory moves the near-side fill probability from 0.377 to 0.460 and the far
side to 0.308. That is a material asymmetry, which is why the skew is measured
rather than dismissed; `--gamma-sweep` shows how the answer moves with `γ`.

## The result

500 paired seeds (0–499), γ = 0.5, all four arms on the same market draws.
Every number below is read out of `results.json`, which CI regenerates on every
push to this branch (`.github/workflows/ci.yml` runs `evaluate.py --seeds 500`
and commits `results.json` and `results_seeds.csv`).

**Of the 17.0% inventory-σ reduction the full model buys over the benchmark,
12.1 points are the wider spread and 4.9 points are the skew. For maximum
drawdown the split reverses: quoting wider on its own moves drawdown by 1,994
with a 95% CI that straddles zero, while the skew at the same spread moves it
1,977 with a CI that does not. The published 14% drawdown reduction is carried
by the skew, not by the wider spread.**

### Per-arm means (500 seeds)

| arm | spread | inventory σ | max drawdown | fills | final P&L | t-stat |
|---|---|---|---|---|---|---|
| `fixed` | 0.100 | 50.5 | −29,177 | 553.0 | −1,423 | −0.01 |
| `spread_matched` | 0.195 | 44.4 | −27,184 | 344.6 | −1,547 | 0.04 |
| `skew_only` | 0.100 | 46.1 | −26,337 | 566.6 | −1,465 | 0.01 |
| `as` | 0.195 | 42.0 | −25,206 | 352.3 | −1,938 | 0.04 |

### Paired differences, 95% CI, win rate

Positive means the first-named arm is better. The two middle rows of each block
sum, seed by seed, to the `as − fixed` row.

**Inventory σ** (benchmark 50.5)

| comparison | isolates | difference | 95% CI | wins | sig. |
|---|---|---|---|---|---|
| `spread_matched − fixed` | the wider spread | **6.13** (12.1%) | [5.28, 6.98] | 73% | yes |
| `as − spread_matched` | the skew, at the model spread | **2.46** (4.9%) | [2.05, 2.87] | 71% | yes |
| `as − fixed` | the full model | **8.59** (17.0%) | [7.83, 9.35] | 85% | yes |
| `skew_only − fixed` | the skew, at the benchmark spread | **4.40** (8.7%) | [3.98, 4.82] | 85% | yes |

**Maximum drawdown** (benchmark −29,177)

| comparison | difference | 95% CI | wins | sig. |
|---|---|---|---|---|
| `spread_matched − fixed` | 1,994 (6.8%) | [−2,488, +6,475] | 57% | **no** |
| `as − spread_matched` | **1,977** (6.8%) | [665, 3,290] | 79% | yes |
| `as − fixed` | **3,971** (13.6%) | [521, 7,421] | 74% | yes |
| `skew_only − fixed` | **2,840** (9.7%) | [2,226, 3,455] | 83% | yes |

**Fills** (benchmark 553.0)

| comparison | difference | 95% CI | wins | sig. |
|---|---|---|---|---|
| `spread_matched − fixed` | −208.5 (−37.7%) | [−210.1, −206.8] | 0% | yes |
| `as − spread_matched` | +7.8 (+1.4%) | [+7.2, +8.3] | 89% | yes |
| `as − fixed` | −200.7 (−36.3%) | [−202.2, −199.1] | 0% | yes |
| `skew_only − fixed` | +13.6 (+2.5%) | [+12.9, +14.3] | 96% | yes |

**Final P&L and t-stat: nothing is significant in any comparison.**
`as − fixed` P&L is −515 [−3,994, +2,965] at a 54% win rate; the largest
t-stat difference is 0.052 [−0.008, 0.113]. P&L here is inventory noise by
construction (see Limitations), so the honest statement is that the model
reduces risk and does not demonstrably change profit.

### What the decomposition actually says

- **The skew is not just "quote less".** At the benchmark's own spread
  (`skew_only`) it takes 2.5% **more** fills than the benchmark while cutting
  inventory σ 8.7% and drawdown 9.7%. Leaning the quotes makes the near side
  more attractive exactly when inventory wants unwinding, and the extra
  unwinding fills are why the fill count goes up rather than down.
- **The wider spread buys inventory dispersion, not drawdown protection.**
  It removes 37.7% of the fills, which mechanically shrinks inventory σ by a
  significant 12.1%, but its drawdown CI spans zero in both directions: on 43%
  of seeds the benchmark drew down less.
- **Together they are additive, not synergistic.** `(as − fixed)` equals
  `(spread_matched − fixed) + (as − spread_matched)` exactly, seed by seed, by
  construction of the paired design; no interaction term is hiding in the 17%.

### The answer depends on γ

`evaluate.py --seeds 100 --gamma-sweep` → `results_sweep.json`, inventory-σ
differences on a 100-seed paired grid:

| γ | model spread | spread effect | skew effect (at the model spread) | total | quote crossings |
|---|---|---|---|---|---|
| 0.05 | 0.200 | 7.85 | 0.21 (not sig.) | 8.06 | 0 |
| 0.1 | 0.199 | 7.87 | 0.16 (not sig.) | 8.03 | 0 |
| 0.2 | 0.198 | 7.80 | 0.57 | 8.37 | 0 |
| 0.5 | 0.195 | 7.58 | 2.41 | 9.99 | 0 |
| 1.0 | 0.191 | 7.32 | 5.44 | 12.76 | 0 |
| 2.0 | 0.183 | 6.60 | 11.35 | 17.95 | 0 |
| 5.0 | 0.164 | 5.50 | 20.14 | 25.64 | 17.5 |

The spread effect is nearly flat because the model spread barely moves over two
decades of γ. The skew effect is **indistinguishable from zero at γ ≤ 0.1** and
becomes the whole story by γ = 5, where the reservation lean starts crossing the
mid and the guard clamps about 17.5 ticks per seed. So "does the skew pay" has
no γ-free answer; γ = 0.5 is the repo's default and the number quoted above.

**The 17% / 14% / 36% headline is also quoted in the profile README and in the
`quoter.py` docstring of neural-options-lab. Those three numbers are unchanged
by the four-arm rerun — they are the `as − fixed` row — but they should name
the arm, because the decomposition shows 12.1 of the 17 points come from
quoting wider rather than from the inventory skew.**

Per-arm means and every pairwise comparison live in `results.json` (schema v2:
`arms` keyed by arm, `comparisons` keyed by comparison then by metric). The raw
per-seed metrics for every arm are in `results_seeds.csv` (`seed, arm,
inventory_sigma, max_drawdown, fills, final_pnl, t_stat, max_abs_inventory,
quote_crossings`), so any number in the tables above can be recomputed from the
committed artifact.

The seed count was not reduced for CI: the whole job (tests, the 500-seed
four-arm run, the 100-seed γ sweep and the 200-seed audit) finishes well inside
the ten-minute budget, so the headline artifacts are always the full 500 seeds.

## Why the evaluation is built this way

An earlier version of this project reported a **single seed** — a $22,322 loss
turned into a $7,336 profit, quoted as a "+132% improvement". That number
reproduces from `simulation.ipynb`. It is also meaningless on its own: a single
path of a 10,000-tick GBM with Poisson fills carries enormous terminal variance,
so a one-seed comparison measures the seed rather than the strategy. The
multi-seed replay that first showed this (the earlier "−412% to +462%,
benchmark profitable 44% of the time" figures) was produced by a scratch script
that was deleted from the tree and that drew the two strategies' fill luck from
*different* random streams. It is restored as `audit.py`, now **paired**, and
documented as an audit of the old notebook model (fixed 0.10 spread, skew
−q·γ·σ² with no horizon term) rather than of the current strategies. CI runs
it and commits `audit_results.json`; on 200 paired seeds the old model's
per-seed "improvement" still ranges from −229% to +179% (median +4.2%), the
paired P&L difference is +138 [−1,847, +2,123] at a 51% win rate, and the
benchmark is profitable on 49% of seeds against the old skewer's 46.5%. The
old model's risk numbers do hold up under pairing — median inventory-σ
reduction 18.8% (94.5% of seeds) and median drawdown reduction 21.8% (90%) —
which is the same shape as the current result and the reason the notebook's
single-seed "+132%" was never the evidence it was presented as.

Two changes make the difference resolvable:

1. **Common random numbers.** Fill draws are generated with the market, not
   inside the strategy loop, so every maker faces identical luck. A quote that
   is always further from the mid than the benchmark's can only be filled on a
   tick where the benchmark's would have been (same uniform, smaller
   probability), and the test suite asserts exactly that, tick by tick.
2. **Distributions, not point estimates.** Every number is a mean over paired
   seeds with a 95% confidence interval and a win rate, and the per-seed arrays
   are committed.

The earlier version also described features it did not implement — queue-position
dependent fills, a terminal-utility objective, quadratic inventory penalties, and
risk aversion adapting to realised volatility. None of those existed in the code;
the reservation price had no `(T − t)` term and the spread was hand-picked rather
than derived. The model is now implemented as specified, the old design survives
as the `skew_only` arm because it is the cleanest test of the skew, and the
claims are limited to what the code does and the experiment shows.

## Guard rails

- **Quote-crossing guard.** A large `γ` or `σ` pushes the reservation lean past
  the half-spread, so the raw bid sits above the mid (or the ask below it) and
  the fill probability on that side clips to 1. `MarketMaker.step` clamps the
  crossing side to the mid and counts the event; `quote_crossings` is reported
  per arm and per seed. It is zero at the defaults and non-zero at the top of
  the γ grid.
- **`t_stat`, not "Sharpe".** `mean(step P&L) / std(step P&L) · √n` is the
  whole-horizon Sharpe, numerically the t-statistic of the mean step P&L. It is
  not annualised; it was previously labelled `sharpe`.
- **`--gamma` never overwrites the headline.** A `--gamma g` run writes
  `results_gamma_<g>.json` and `results_seeds_gamma_<g>.csv`.
- **Inventory cap.** All arms cap `|q|` at 100 (`max_inventory`), which is itself
  a crude inventory control that makes the benchmark safer than a true
  unconstrained symmetric quoter. The tests lift it where it would interfere.

## Limitations

- Fills are modelled as an intensity in distance from the mid. There is **no
  order book and no queue position**, so nothing here speaks to microstructure
  effects that depend on standing in a real queue.
- The mid is exogenous GBM: quoting does not move the price, and there is no
  adverse selection from informed flow beyond what the fill intensity implies.
- The volatility regime (σ = 0.02 per tick over 10,000 ticks, horizon log-std
  2.0) is far wider than anything tradable and 100× the paper's own example;
  P&L is inventory noise by construction, which is why the P&L rows are never
  significant. Rescaling the regime is deliberately not done here, because it
  would change every published number at once; it is the next study.
- One asset, one maker, no latency, no fees.
- `simulation.ipynb` still implements and plots the old model; its figures are
  not of the current strategies.

## Layout

```
market_maker.py        Config, market draw, the four strategies, run(), metrics()
evaluate.py            paired four-arm evaluation, gamma sweep, per-seed CSV, schema-v2 JSON
audit.py               paired multi-seed replay of the OLD notebook model
test_market_maker.py   18 tests: CRN fill-subset property, P&L accounting from the
                       fill record, closed-form spread, skew units, crossing guard,
                       results schema
results.json           per-arm means and every pairwise comparison (regenerated by CI)
results_seeds.csv      one row per (seed, arm)                     (regenerated by CI)
results_sweep.json     the same summaries on a gamma grid, 100 seeds (regenerated by CI)
audit_results.json     audit.py output                             (regenerated by CI)
simulation.ipynb       original exploratory notebook, old model, kept for provenance
```
