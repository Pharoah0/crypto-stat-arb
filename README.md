# Statistical Arbitrage in Cryptocurrencies
## Author

Pharoah Evelyn

---

**Goal:** Identify and exploit market inefficiencies in crypto markets using statistical arbitrage techniques — progressing from manual methods to ML-powered strategies.

---

## Overview

This project builds and compares three portfolio strategies on a universe of 14 major cryptocurrencies, sourced from the Binance API at 4-hour bars (2019–present). Each strategy is evaluated on the same 30% out-of-sample window, vol-targeted to 10% annualized.

| Approach | OOS Sharpe | Alpha T-Stat | P-value |
|---|---|---|---|
| Manual | ~2.0–2.28 | 1.736 | 0.084 |
| SKLearn (Ridge + CV) | — | 3.287 | 0.001 |
| PyTorch LSTM | — | 4.671 | <0.001 |

---

## Data

- **Source:** Binance API (`python-binance`)
- **Universe:** ADA, AVAX, BNB, BTC, DASH, DOGE, ETH, HBAR, HYPE, LINK, SOL, SUI, XRP, ZEC
- **Frequency:** 4-hour bars for signal research; resampled to daily for portfolio construction
- **Date pinned** via `DATA_END` constant for reproducibility

---

## Dependencies

```
python-binance
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
cvxpy
torch
```

---

## Usage

1. Add your Binance API credentials
2. Run all cells top-to-bottom — `DATA_END` is pinned; results are fully reproducible
3. Runtime logs are written to `runtime_logs/` on each run
---

## Notebook Sections

### Section 1 — Data Loading
Pulls OHLCV data from Binance, converts prices to log returns, and visualizes cumulative returns per coin.

### Section 2 — Data Investigation
- **Drawdown analysis** — max drawdowns range from –65% to –97% across coins
- **Rolling Beta & Alpha** vs. BTC — XRP, DASH, and ZEC show β < 1 with meaningful α
- **Pairwise correlations** — raw vs. alpha-return heatmaps; alpha returns are far less correlated, creating diversification opportunities
- **Sharpe vs. Information Ratio** — HYPE stands out after removing BTC's market influence

### Section 3 — Signal Generation
Four signal families across two strategy types:

**Time-Series (TS) Momentum**
- *Time Horizon:* Rolling mean signals across 10 daily horizons (1–30d). Short-term momentum is negative; edge improves at longer horizons. 28d peaks.
- *Seasonality:* Weekday vs. weekend returns; hour-of-day analysis (US/EU sessions outperform); monthly seasonality (Jan, Jul, Nov strongest).

**Cross-Sectional (XS) Reversal**
- *Pairs Reversal:* Correlated coin pairs (ρ > 0.7) traded as mean-reversion pairs. Best performer: ETH/ZEC (Sharpe > 5.0).
- *Basket Reversal:* Each coin regressed against the full basket; residuals traded as mean-reversion signals. AVAX leads; HYPE is weakest.

### Section 4 — Weighting & Performance Evaluation
TS and XS signals are each combined under three schemes — Optimal Weights, Equal-Vol Weights, and SR Weights — using a 70/30 train/test split to prevent lookahead bias. SR Weights are selected for the TS leg.

### Section 5 — Backtesting & Portfolio Construction
TS and XS portfolios are merged using optimal weights. Signal transforms (rolling z-score, spike detection) are tested and rejected — raw signal wins. Final manual portfolio achieves a consistent Sharpe of 2.0–2.28.

### Section 6 — Execution & Turnover
- Transaction costs modeled at 20bps (7bps commission + 13bps slippage)
- Gross SR 2.149 → Net SR drag of ~0.226 (~10.5%)
- Convex optimization with Ledoit-Wolf shrinkage + t-cost penalized rebalancing reduces TS turnover by ~6.6%

### Section 7 — SKLearn Parameter Optimization
- **7A** — 5-fold walk-forward CV across 12 horizon candidates; 28d leads at CV Sharpe ~1.44
- **7B** — Pairs reversal grid search over lookback (2–21d) and entry z-score (0.5–2.0)
- **7C** — Basket reversal grid search over lookback (14–60d)
- **7D** — Ridge regression signal combiner replaces manual weighting for TS + XS blending

### Section 8 — PyTorch LSTM Signal Optimization
An LSTM network generates per-coin position signals directly from market features:
- **Features:** Raw returns, 5-day and 20-day z-scores, normalized volatility (regime awareness)
- **Output:** Per-coin signals in [–1, 1] via tanh activation
- Sequence length of 10 periods; full reproducibility seed block (random, numpy, torch, CUDA, cuDNN)
- Model weights saved via `torch.save`

### Section 9 — Final Comparison: Manual vs. SKLearn vs. PyTorch
All three strategies are benchmarked OOS, vol-targeted to 10% annualized. Alpha t-stats are computed against BTC using `statsmodels OLS`:
- **Manual** — most consistent, stable across runs, alpha not statistically significant (p ≈ 0.084)
- **SKLearn** — highest peak cumulative return (~35%), regime-concentrated Oct–Dec 2025, significant alpha (p = 0.001)
- **LSTM** — strongest upward trend and highest alpha t-stat (4.671), highly significant (p < 0.001)

---

## Key Design Decisions

- **70/30 train/test split** with a pinned `DATA_END` date prevents lookahead bias and train/test boundary drift
- **No hyperparameter seed-shopping** — the principled LSTM configuration is reported as-is
- **Alpha significance tested via OLS t-stats** — `✓/✗` flags at |t| > 2.0 in the runtime log
- **Runtime log** auto-generated each run as a timestamped HTML/markdown summary table


