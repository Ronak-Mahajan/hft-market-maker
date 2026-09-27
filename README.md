# Inventory-Aware Market Making: does Avellaneda-Stoikov pay, and which half does the work?

A discrete-time Monte Carlo simulator of a single market maker posting a bid
and an ask against Poisson order flow on a synthetic GBM mid, built to answer
one question: **does skewing quotes against your inventory measurably reduce
risk, and what does it cost?**

```bash
pip install -r requirements.lock      # Python 3.12, the exact versions CI installs
python evaluate.py --seeds 500        # four paired arms -> results.json, results_seeds.csv
python evaluate.py --seeds 500 --max-inventory none   # position limit lifted -> results_uncapped.json
python evaluate.py --seeds 100 --gamma-sweep   # -> results_sweep.json
python audit.py --seeds 200           # horizon-free skew rule, paired -> audit_results.json
python scripts/check_artifacts.py     # regenerated artifacts vs the committed copies
python -m pytest -q                   # 34 tests
```

## The problem

A market maker quoting symmetrically around the mid accumulates inventory
whenever order flow is one-sided, and that inventory is directional risk they
were never paid to take. Avellaneda & Stoikov (2008) prescribe quoting around a
**reservation price** that leans away from your position, with a spread the
model derives rather than one you pick:

```
reservation price   r = s − q*γ*σ²*(T − t)
optimal half-spread d = ½[ γσ²(T − t) + (2/γ)*ln(1 + γ/κ) ]
bid = r − d,   ask = r + d
```

Both terms carry `(T − t)`: aversion to inventory decays to zero at the terminal
time, because there is no longer any horizon over which a position can move
against you. Fills follow the same paper's execution model, `λ(δ) = A*e^{−κδ}`,
so quoting further from fair value earns more per fill and gets fewer of them.

## The decomposition question

Avellaneda-Stoikov changes **two** things relative to a naive symmetric quoter:
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
| `skew_only` | 0.10, hand-picked | model skew | what the skew does at the benchmark spread |
| `as` | the model spread | model skew | the full model |

and five paired comparisons, each reported for every metric with a 95%
confidence interval and a win rate:

```
spread_matched vs fixed            the wider spread, skew off
as             vs spread_matched   the skew, at the model spread
as             vs fixed            the full model
skew_only      vs fixed            the skew, at the benchmark spread
as             vs skew_only        the wider spread, skew on
```

The four arms are a 2x2 design, spread by skew, so `as − fixed` splits into a
spread step and a skew step in either order: rows 1 and 2 sum to row 3 on
every seed, and so do rows 4 and 5. The two orders give different splits
whenever the skew is worth a different amount at the two spreads, so
`results.json` also reports the factorial main effects (each step averaged
over both orders, again summing to `as − fixed`) and the interaction,
`(as − spread_matched) − (skew_only − fixed)`, each with its own paired CI.

**Units.** `σ` is the per-tick log-volatility and `(T − t)` is the normalised
fraction of the horizon remaining, `τ = 1 − t/n_ticks ∈ [0, 1]`, so `γσ²τ` is
not the paper's formula in tick units (that would carry `n_ticks − t`) and `γ`
should be read as a per-horizon risk aversion. The lean is judged on the
yardstick that governs fills: with `κ = 10` the fill-decay length is
`1/κ = $0.10`, and the default lean of `100 * 0.5 * 0.02² = $0.02` at maximum
inventory moves the near-side fill probability from 0.377 to 0.460 and the far
side to 0.308. That is a material asymmetry. `--gamma-sweep` shows how the
answer moves with `γ`.

**Price level.** In the lean, `σ` stands in for the paper's dollar volatility
(`σS` for a GBM mid), so the lean is a fixed dollar amount, at most $0.02,
whatever the price. The price itself ranges widely. Over the 500 seeds the
final mid has a 5th percentile of $1.18, a median of $37.10 and a 95th
percentile of $892; the path rises above $1,000 on 12.6% of seeds (highest
$17,621, seed 279) and falls below $1 on 8% (lowest $0.08, seed 37). Inventory
σ, fills and spread captured are counted in shares or in fixed dollar offsets
from the mid, so they do not scale with the price level. Dollar P&L and
drawdown do (`price_paths` in `results.json`).

## The result

500 paired seeds (0 to 499), γ = 0.5, all four arms on the same market draws.
Every number below is read out of `results.json` unless another artifact is
named. On every push CI reruns `evaluate.py --seeds 500` (with and without the
position limit), the γ sweep and the audit, about a minute of compute, and
fails if the result differs from the committed files
(`scripts/check_artifacts.py`: floats to a relative 1e-9, counts and flags
exactly).

**The full model cuts inventory σ 17.0% against the benchmark, on 36% fewer
fills.** Averaged over both orderings, the wider spread accounts for 10.2
points of that and the skew for 6.8. The two overlap: the skew is worth 8.7%
at the benchmark spread and 4.9% at the model spread, an interaction of −3.8
points whose CI excludes zero.

**Drawdown.** The full model's maximum drawdown is 13.6% shallower in mean
dollars, but dollar drawdown follows the price path (see Price level).
Measured per seed as a ratio, which removes the price level, it is 20.6%
[17.5, 23.5] shallower. Averaged over both orderings, 6.7% of that comes from
the wider spread and 14.9% from the skew, and the skew's share is the larger
with a CI that excludes zero.

**P&L.** Realized P&L does not resolve in any comparison, but its expected
value here is the spread captured, and that does: the full model captures
23.9% more spread per run (+66.0 [64.9, 67.0]), higher on every seed.

### Per-arm means (500 seeds)

| arm | spread | inventory σ | max drawdown | fills | spread captured | final P&L | t-stat |
|---|---|---|---|---|---|---|---|
| `fixed` | 0.100 | 50.5 | −29,177 | 553.0 | 276.5 | −1,423 | −0.01 |
| `spread_matched` | 0.195 | 44.4 | −27,184 | 344.6 | 336.6 | −1,547 | 0.04 |
| `skew_only` | 0.100 | 46.1 | −26,337 | 566.6 | 280.7 | −1,465 | 0.01 |
| `as` | 0.195 | 42.0 | −25,206 | 352.3 | 342.5 | −1,938 | 0.04 |

### Paired differences, 95% CI, win rate

Positive means the first-named arm is better. In each block rows 1 and 2 sum,
seed by seed, to row 3, and so do rows 4 and 5. Outside the ratio table,
percentages are of the benchmark's mean and add the same way. The main
effects average the two orderings; the interaction is row 2 minus row 4.

**Inventory σ** (benchmark 50.5)

| comparison | isolates | difference | 95% CI | wins | sig. |
|---|---|---|---|---|---|
| `spread_matched − fixed` | the wider spread, skew off | **6.13** (12.1%) | [5.28, 6.98] | 73% | yes |
| `as − spread_matched` | the skew, at the model spread | **2.46** (4.9%) | [2.05, 2.87] | 71% | yes |
| `as − fixed` | the full model | **8.59** (17.0%) | [7.83, 9.35] | 85% | yes |
| `skew_only − fixed` | the skew, at the benchmark spread | **4.40** (8.7%) | [3.98, 4.82] | 85% | yes |
| `as − skew_only` | the wider spread, skew on | **4.19** (8.3%) | [3.55, 4.83] | 72% | yes |
| spread main effect | both orderings averaged | **5.16** (10.2%) | [4.45, 5.87] | 73% | yes |
| skew main effect | both orderings averaged | **3.43** (6.8%) | [3.11, 3.76] | 85% | yes |
| interaction | the skew at the model spread minus at the benchmark spread | **−1.94** (−3.8%) | [−2.45, −1.44] | 36% | yes |

**Maximum drawdown, per-seed ratio.** For each seed, the log of the ratio of
the two arms' drawdowns, shown as the geometric-mean reduction. The logs add,
so the reductions compound: (1 − 7.7%)(1 − 13.9%) ≈ 1 − 20.6%.

| comparison | reduction | 95% CI | wins | sig. |
|---|---|---|---|---|
| `spread_matched − fixed` | **7.7%** | [4.2, 11.1] | 57% | yes |
| `as − spread_matched` | **13.9%** | [12.1, 15.7] | 79% | yes |
| `as − fixed` | **20.6%** | [17.5, 23.5] | 74% | yes |
| `skew_only − fixed` | **15.8%** | [14.1, 17.4] | 83% | yes |
| `as − skew_only` | **5.6%** | [2.2, 9.0] | 54% | yes |
| spread main effect | **6.7%** | [3.4, 9.8] | 56% | yes |
| skew main effect | **14.9%** | [13.6, 16.1] | 86% | yes |
| interaction | −2.2% | [−5.0, +0.4] | 44% | no |

The skew main effect exceeds the spread main effect by 0.092 [0.054, 0.129]
in logs.

**Maximum drawdown, dollars** (benchmark −29,177). Dollar drawdown follows
the price path: the benchmark's drawdown correlates 0.93 with the path's
peak, and seed 279, whose mid peaks at $17,621, contributes 36% of the summed
`as − fixed` difference. These means are dominated by the few high-price
seeds, which is why two of the rows below have CIs that span zero.

| comparison | difference | 95% CI | wins | sig. |
|---|---|---|---|---|
| `spread_matched − fixed` | 1,994 (6.8%) | [−2,488, +6,475] | 57% | no |
| `as − spread_matched` | **1,977** (6.8%) | [665, 3,290] | 79% | yes |
| `as − fixed` | **3,971** (13.6%) | [521, 7,421] | 74% | yes |
| `skew_only − fixed` | **2,840** (9.7%) | [2,226, 3,455] | 83% | yes |
| `as − skew_only` | 1,131 (3.9%) | [−2,049, +4,310] | 54% | no |

**Fills** (benchmark 553.0)

| comparison | difference | 95% CI | wins | sig. |
|---|---|---|---|---|
| `spread_matched − fixed` | −208.5 (−37.7%) | [−210.1, −206.8] | 0% | yes |
| `as − spread_matched` | +7.8 (+1.4%) | [+7.2, +8.3] | 89% | yes |
| `as − fixed` | −200.7 (−36.3%) | [−202.2, −199.1] | 0% | yes |
| `skew_only − fixed` | +13.6 (+2.5%) | [+12.9, +14.3] | 96% | yes |
| `as − skew_only` | −214.3 (−38.7%) | [−215.7, −212.8] | 0% | yes |

**Spread captured** (benchmark 276.5): for each fill, its distance from that
tick's mid times the order size, summed over the run.

| comparison | difference | 95% CI | wins | sig. |
|---|---|---|---|---|
| `spread_matched − fixed` | +60.1 (+21.7%) | [+58.9, +61.3] | 100% | yes |
| `as − spread_matched` | +5.9 (+2.1%) | [+5.3, +6.4] | 85% | yes |
| `as − fixed` | **+66.0 (+23.9%)** | [+64.9, +67.0] | 100% | yes |
| `skew_only − fixed` | +4.2 (+1.5%) | [+3.8, +4.6] | 85% | yes |
| `as − skew_only` | +61.8 (+22.3%) | [+60.7, +62.8] | 100% | yes |

**Final P&L and t-stat: no comparison resolves.** `as − fixed` P&L is −515
[−3,994, +2,965] at a 54% win rate; the largest t-stat difference is 0.052
[−0.008, 0.113]. Realized P&L is the spread captured plus the inventory marked
to market (`final_pnl = edge + inventory_pnl` on every run). A fill depends
only on its quote's dollar offset from the mid, so the inventory path does not
depend on the price path (apart from the floored bids on seed 37), and with
symmetric order flow the inventory term has zero mean: expected P&L is the
spread captured. That term is also what hides the spread captured in realized
P&L. Its paired standard deviation is about 40,000 per run, and for
`as − fixed` it comes to −581 [−4,060, +2,899] against +66.0 of spread
captured.

### What the decomposition shows

- **The skew's extra fills come from the position limit.** At the benchmark's
  own spread (`skew_only`) the skew takes 2.5% more fills than the benchmark
  while cutting inventory σ 8.7% and drawdown 15.8% (per-seed ratio). A maker
  at the ±100 limit refuses every fill that would add to its position, and
  `skew_only` ends 495 of 10,000 ticks there against the benchmark's 887. The
  benchmark refuses 25.8 fills per seed that way and `skew_only` 13.3, which
  accounts for 12.5 of the +13.6. With the limit lifted
  (`results_uncapped.json`) the difference is +1.4 [1.0, 1.9].
- **The wider spread does most of the inventory-σ work and about a third of the
  drawdown work.** Its main effect is 10.2 of the 17.0 points of inventory σ,
  and on the drawdown ratio 6.7% against the skew's 14.9% (0.069 of 0.230 in
  logs). It removes 37.7% of the fills and captures 21.7% more spread per run.
  In dollars its drawdown CIs span zero, since the dollar mean follows the few
  high-price paths.
- **The two effects overlap.** At the wider spread there is less inventory to
  lean against, and the skew removes 2.46 points of σ there against 4.40 at
  the benchmark spread: an interaction of −1.94 [−2.45, −1.44], with fills
  −5.8 [−6.6, −5.0]. The drawdown interaction is not resolved (per-seed ratio
  −2.2% [−5.0, +0.4]).

### The answer depends on γ

`evaluate.py --seeds 100 --gamma-sweep` writes `results_sweep.json`: inventory-σ
differences at each γ on seeds 0 to 99, paired. The last two columns are
per-arm crossing counts, the mean ticks per 10,000-tick seed on which a raw
quote crossed the mid and the guard clamped it. They are reported separately
because the two skewing arms cross at very different γ: `skew_only` applies
the same lean around a half-spread about half as wide, so it crosses first.

| γ | model spread | spread effect (skew off) | skew at the model spread | skew at the benchmark spread | total | crossings `as` | crossings `skew_only` |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.200 | 7.85 | 0.21 (not sig.) | 0.61 | 8.06 | 0 | 0 |
| 0.1 | 0.199 | 7.87 | 0.16 (not sig.) | 1.01 | 8.03 | 0 | 0 |
| 0.2 | 0.198 | 7.80 | 0.57 | 1.73 | 8.37 | 0 | 0 |
| 0.5 | 0.195 | 7.58 | 2.41 | 4.26 | 9.99 | 0 | 0 |
| 1.0 | 0.191 | 7.32 | 5.44 | 8.59 | 12.76 | 0 | 0 |
| 2.0 | 0.183 | 6.60 | 11.35 | 14.88 | 17.95 | 0 | 17.4 |
| 5.0 | 0.164 | 5.50 | 20.14 | 23.58 | 25.64 | 17.5 | 438.1 |

The spread effect follows how much wider the model spread is than the
benchmark's. In the rows with no crossings (γ ≤ 1) the model spread moves only
from 0.200 to 0.191 and the spread effect stays between 7.3 and 7.9; at γ = 5
the model spread is 0.164 and the spread effect 5.50, still significant. The
skew effect at the model spread is **indistinguishable from zero at γ ≤ 0.1**
(at the benchmark spread it is 0.61 and 1.01, small but resolved) and grows
with γ to 20.14 of the 25.64 total at γ = 5. From γ = 2 the crossing guard is
part of what is measured: the reservation lean outgrows the half-spread, so
`skew_only` is clamped to the mid on 17.4 ticks per seed at γ = 2 and 438.1 at
γ = 5, and `as` on 17.5 at γ = 5. **No arm crosses the mid anywhere at
γ ≤ 1**, so those rows measure the model alone, and the γ ≥ 2 rows show where
the guard starts to drive the result. How much the skew contributes depends
on γ. The headline tables use the default γ = 0.5 on 500 seeds; the γ = 0.5
row here covers seeds 0 to 99, the first 100 of those, so it reads 7.58 /
2.41 / 9.99 against the 500-seed 6.13 / 2.46 / 8.59.

Per-arm means, every pairwise comparison and the factorial effects live in
`results.json` (schema v3: `arms` keyed by arm, `comparisons` keyed by
comparison then by metric, `factorial` keyed by effect then by metric, each
summary with its percentage under `pct`, and the price range under
`price_paths`). `results_uncapped.json` is the same run with the position
limit lifted. The raw per-seed metrics for every arm are in
`results_seeds.csv` (`seed, arm, inventory_sigma, max_drawdown, fills,
final_pnl, t_stat, max_abs_inventory, quote_crossings, floored_bids, edge,
inventory_pnl, blocked_fills, ticks_at_cap`), so any number in the tables
above can be recomputed from the committed artifacts.

## Why the evaluation is built this way

A single seed cannot separate a strategy from its luck. One path of a
10,000-tick GBM with Poisson fills carries terminal P&L variance that swamps
the difference between two quoters, so a one-seed comparison measures the
seed. Two choices make the difference resolvable:

1. **Common random numbers.** Fill draws are generated with the market, before
   any strategy runs, so every maker faces identical luck. Absent the
   position limit, a quote that is always further from the mid than the
   benchmark's can only be filled on a tick where the benchmark's would have
   been (same uniform, smaller probability); the test suite asserts that tick
   by tick with the limit lifted.
2. **Distributions over seeds.** Every number is a mean over paired seeds with
   a 95% confidence interval and a win rate, and the per-seed arrays are
   committed.

### The horizon-free skew rule

`audit.py` applies the same paired design to a simpler rule: the benchmark's
0.10 spread around a reservation price `s − q*γ*σ²`, γ = 0.5, with no horizon
term, so the lean is the same on every tick. On 200 paired seeds
(`audit_results.json`) its P&L difference against the benchmark is +138
[−1,847, +2,123] at a 51% win rate, and the benchmark is profitable on 49% of
seeds against the rule's 46.5%, so P&L does not resolve here either. Its median
per-seed inventory-σ reduction is 18.8% (lower on 94.5% of seeds) and its
median drawdown reduction 21.8% (shallower on 90%).

The `skew_only` arm is this rule with the lean scaled by the time remaining τ,
which averages one half over the run. On the same 200 seeds its median
inventory-σ reduction is 8.2% (87.5% of seeds) and its median drawdown
reduction 12.6% (81.5%). With τ held at 1, `skew_only` reproduces the rule
tick for tick (`test_market_maker.py` checks it), so the horizon term accounts
for the whole gap.

## Guard rails

- **Quote-crossing guard.** A large `γ` or `σ` pushes the reservation lean past
  the half-spread, so the raw bid sits above the mid (or the ask below it) and
  the maker would buy above fair value, a certain loss on that side, because
  the fill probability there has already clipped to 1. `MarketMaker.step` clamps
  the crossing side to the mid and counts the event. The clamp caps the fill
  **price** at fair value and leaves the fill probability at 1 on that side.
  `quote_crossings` is reported per arm and per seed; it is zero for all four
  arms at the defaults and at every γ ≤ 1 of the sweep, and non-zero above
  that.
- **Price floor.** A raw bid below `min_price` ($0.01) is raised to it. The
  mid is GBM and never reaches zero, but the model half-spread is a fixed
  $0.098, so where the mid falls below that the raw model bid is at or below
  zero. `MarketMaker.step` floors such a bid and counts it in
  `floored_bids`. Across the 500 seeds it binds on one: seed 37, on 145 ticks
  for each of the two model-spread arms.
- **`t_stat`.** `mean(step P&L) / std(step P&L) * √n` is the whole-horizon
  Sharpe, numerically the t-statistic of the mean step P&L. It is not
  annualised and is not comparable to an annualised Sharpe ratio.
- **`--gamma` and `--max-inventory` never overwrite the headline.** A
  `--gamma g` run writes `results_gamma_<g>.json` and
  `results_seeds_gamma_<g>.csv`; `--max-inventory none` writes
  `results_uncapped.json` and `results_seeds_uncapped.csv`, and
  `--max-inventory N` writes `results_cap_<N>.json` and
  `results_seeds_cap_<N>.csv`.
- **Inventory cap.** All arms cap `|q|` at 100 (`max_inventory`), which is itself
  a crude inventory control that makes the benchmark safer than a true
  unconstrained symmetric quoter, and it binds hard. The `max_abs_inventory`
  column of `results_seeds.csv` touches the cap on 100% of seeds for `fixed`,
  99.2% for `skew_only`, 98.0% for `spread_matched` and 92.6% for `as`, and the
  benchmark ends 8.9% of all ticks there (`ticks_at_cap`), so the position
  limit sets part of every arm's measured risk. With the limit lifted
  (`results_uncapped.json`, the same 500 seeds) the full model cuts inventory
  σ 43.2% of the benchmark's 90.0, and the main effects are 16.2 points for
  the spread and 27.0 for the skew: without the limit the skew is the larger
  part. The tests lift it where it would interfere.

## Limitations

- Fills are modelled as an intensity in distance from the mid. There is **no
  order book and no queue position**, so nothing here speaks to microstructure
  effects that depend on standing in a real queue.
- The mid is exogenous GBM: quoting does not move the price, and there is no
  adverse selection from informed flow beyond what the fill intensity implies.
- The volatility regime (σ = 0.02 per tick over 10,000 ticks, horizon log-std
  2.0) is far wider than anything tradable and 100× the paper's own example.
  Realized P&L is dominated by the zero-mean inventory term, which is why the
  realized P&L rows never resolve and the spread-captured rows do. Nothing is
  rescaled to the price path, so the fixed 0.10 spread is under 0.1 basis
  point of the highest mid and more than the whole price at the lowest (see
  Price level above).
- One asset, one maker, no latency, no fees.

## Layout

```
market_maker.py        Config, market draw, the four strategies, run(), metrics()
evaluate.py            paired four-arm evaluation, 2x2 factorial, gamma sweep, per-seed CSV,
                       schema-v3 JSON
audit.py               paired multi-seed replay of the horizon-free skew rule, with
                       skew_only on the same seeds
scripts/
  check_artifacts.py   regenerated artifacts vs the committed copies; CI fails on a difference
test_market_maker.py   30 tests: CRN fill-subset property, P&L accounting from the
                       fill record, the spread-captured and inventory P&L split,
                       closed-form spread, skew units, crossing guard (clamp,
                       count, and price cap), price floor, inventory cap and
                       blocked fills, config validation, factorial identities,
                       results schema, the audited rule as skew_only with τ
                       held at 1
test_artifacts.py      4 tests: 10-digit float rounding in the writers, and the
                       tolerance and exact-match rules of check_artifacts.py
results.json           per-arm means, every pairwise comparison and the factorial
                       effects                                     (checked by CI)
results_seeds.csv      one row per (seed, arm)                     (checked by CI)
results_uncapped.json  the same run with the position limit lifted, and
results_seeds_uncapped.csv its per-seed rows                       (checked by CI)
results_sweep.json     the same summaries on a gamma grid, 100 seeds (checked by CI)
audit_results.json     audit.py output                             (checked by CI)
```
