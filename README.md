# Statistical Arbitrage in Cryptocurrencies
<img width="2560" height="1280" alt="image" src="https://github.com/user-attachments/assets/df4d2a66-13cf-41d5-8070-029e0ff6cb92" />

## Author
Pharoah Evelyn

### **Goal:** Identify and exploit market inefficiencies in crypto markets using statistical arbitrage techniques — progressing from manual methods to ML-powered strategies.

## Overview

This project builds and compares three portfolio strategies on a universe of 14 major cryptocurrencies, sourced from the Binance API at 4-hour bars (2019–present). Each strategy is evaluated on the same 30% out-of-sample window, vol-targeted to 10% annualized.

| Approach | OOS Sharpe | Alpha T-Stat | P-value |
|---|---|---|---|
| Manual | ~1.94–2.28 | 1.736 | 0.084 |
| SKLearn (Ridge + CV) | ~3.63-5.36 | 3.287 | 0.001 |
| PyTorch LSTM | ~2.70-3.43 | 4.671 | <0.001 |

## Data

- **Source:** Binance API (`python-binance`)
- **Universe:** ADA, AVAX, BNB, BTC, DASH, DOGE, ETH, HBAR, HYPE, LINK, SOL, SUI, XRP, ZEC
- **Frequency:** 4-hour bars for signal research; resampled to daily for portfolio construction
- **Date pinned** via `DATA_END` constant for reproducibility

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

## Usage

1. Add your Binance API credentials
2. Run all cells top-to-bottom — `DATA_END` is pinned; results are fully reproducible
3. Runtime logs are written to `runtime_logs/` on each run

## Notebook Sections

### Section 1 — Data Loading
Pulls OHLCV data from Binance, converts prices to log returns, and visualizes cumulative returns per coin.
<img width="822" height="428" alt="image" src="https://github.com/user-attachments/assets/5ec909e2-c49f-4e3e-94d4-329acbd09345" />


### Section 2 — Data Investigation
- **Drawdown analysis** — max drawdowns range from –65% to –97% across coins <img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/49efdc82-b180-4f9a-95f3-c7e9edea0fa3" />

- **Rolling Beta & Alpha** vs. BTC — XRP, DASH, and ZEC show β < 1 with meaningful α <img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/112eeae2-ad6b-47d2-865b-e12f4248035c" />

- **Pairwise correlations** — raw vs. alpha-return heatmaps; alpha returns are far less correlated, creating diversification opportunities <img width="542" height="460" alt="image" src="https://github.com/user-attachments/assets/0b1262b5-f60b-4e8d-81a1-e1cb962d50a4" /> <img width="542" height="460" alt="image" src="https://github.com/user-attachments/assets/c7ab628d-ef13-45ec-b263-60a2d4a0840a" />

- **Sharpe vs. Information Ratio** — HYPE stands out after removing BTC's market influence

| coin | sharpe ratio | information ratio |
|---|---|---|
| HYPE | 0.304 | 0.652 |
| SOL | 0.455 | 0.387 |
| BNB | 0.437 | 0.366 |
| DOGE | 0.421 | 0.331 |
| ZEC | 0.294 | 0.303 |
| DASH | 0.281 | 0.297 |
| ETH | 0.347 | 0.196 |
| ADA | 0.325 | 0.168 |
| XRP | 0.256 | 0.148 |
| SUI | 0.348 | 0.140 |
| HBAR | 0.292 | 0.131 |
| LINK | 0.074 | 0.046 |
| USDC | 0.022 | -0.013 |
| AVAX | -0.017 | -0.202 |
| BTC | 0.328 | NaN |

### Section 3 — Signal Generation
Four signal families across two strategy types:

**Time-Series (TS) Momentum**
- *Time Horizon:* Rolling mean signals across 10 daily horizons (1–30d). Short-term momentum is negative; edge improves at longer horizons. 28d peaks. <img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/eabc0fd6-1376-46c0-842c-86f96f0ae655" />
- *Seasonality:* Weekday vs. weekend returns; hour-of-day analysis (US/EU sessions outperform); monthly seasonality (Jan, Jul, Nov strongest).<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/b992150a-6505-479c-9570-5d47a63df836" /> <img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/d0062a12-51c0-464c-9f85-9ed7cb6c0665" />

**Cross-Sectional (XS) Reversal**
- *Pairs Reversal:* Correlated coin pairs (ρ > 0.7) traded as mean-reversion pairs. Best performer: ETH/ZEC (Sharpe > 5.0). <img width="988" height="489" alt="image" src="https://github.com/user-attachments/assets/cd824379-9e11-458d-8e34-3b906ac20d93" />
- *Basket Reversal:* Each coin regressed against the full basket; residuals traded as mean-reversion signals. AVAX leads; HYPE is weakest. <img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/2d97ee69-3be4-430c-8723-ae4f4f5aaa96" />

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
All three strategies are benchmarked OOS, vol-targeted to 10% annualized. Alpha t-stats are computed against BTC using `statsmodels OLS`: <img width="1181" height="584" alt="image" src="https://github.com/user-attachments/assets/9067a945-82f5-4728-944c-59f52f1f9d1a" />
- **Manual** — most consistent, stable across runs, alpha not statistically significant (p ≈ 0.084)
- **SKLearn** — highest peak cumulative return (~35%), regime-concentrated Oct–Dec 2025, significant alpha (p = 0.001)
- **LSTM** — strongest upward trend and highest alpha t-stat (4.671), highly significant (p < 0.001)

## Key Design Decisions

- **70/30 train/test split** with a pinned `DATA_END` date prevents lookahead bias and train/test boundary drift
- **No hyperparameter seed-shopping** — the principled LSTM configuration is reported as-is
- **Alpha significance tested via OLS t-stats** — `✓/✗` flags at |t| > 2.0 in the runtime log
- **Runtime log** auto-generated each run as a timestamped HTML/markdown summary table


