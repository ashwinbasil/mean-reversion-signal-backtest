# RSI Mean Reversion Signal Backtest

Educational quantitative research project. Tests whether RSI oversold signals predict short-term price reversals across US equities, using a modular Python pipeline with walk-forward validation.

**This is not a trading system. Results are not investment advice.**

---

## Research Question

> Do RSI signals predict positive forward returns across different market sectors, and does that signal survive out-of-sample validation?

---

## What It Does

Full pipeline per ticker:

1. Fetches 5 years of OHLCV data via `yfinance`
2. Cleans data (missing values, zero-volume rows, timezone normalisation)
3. Computes indicators: RSI (Wilder smoothing), MACD, moving averages, volatility
4. Generates RSI buy signals at configurable threshold
5. Simulates long-only strategy with realistic transaction costs (0.1% round-trip)
6. Compares strategy against buy-and-hold benchmark
7. Runs RSI threshold parameter sweep (in-sample)
8. Runs walk-forward validation: optimises threshold on 70% train, evaluates on 30% test
9. Outputs per-ticker CSVs and diagnostic plots
10. Aggregates results across all tickers with credibility filter

---

## Project Structure

```
mean-reversion-signal-backtest/
├── main.py                  # Orchestration loop, sector grouping, summary
├── requirements.txt
├── src/
│   ├── analytics.py         # RSI, MACD, moving averages, returns
│   ├── backtester.py        # Signal evaluation, strategy simulation, drawdown
│   ├── data_cleaner.py      # Missing data, zero volume, timezone handling
│   ├── data_fetcher.py      # yfinance download with validation
│   ├── experiments.py       # Parameter sweep, walk-forward validation
│   ├── metrics.py           # Sharpe, Sortino, CAGR, max drawdown
│   ├── validation.py        # Data quality checks
│   └── visualizer.py        # Price/signal plots, equity curves, drawdown charts
├── notebooks/
│   └── eda.ipynb
├── data/
│   └── processed/           # Generated CSVs (gitignored)
└── outputs/                 # Generated plots (gitignored)
```

---

## Quick Start

```bash
git clone https://github.com/ashwinbasil/mean-reversion-signal-backtest.git
cd mean-reversion-signal-backtest
pip install -r requirements.txt
python main.py
```

Outputs saved to `data/processed/` and `outputs/`. Runtime: ~3-5 minutes for default 15 tickers.

---

## Configuration

Edit top of `main.py`:

```python
TICKERS = {
    "tech":     ["AAPL", "MSFT", "GOOGL"],
    "finance":  ["JPM", "BAC", "GS"],
    "energy":   ["XOM", "CVX", "COP"],
    "consumer": ["WMT", "COST", "TGT"],
    "health":   ["JNJ", "PFE", "UNH"],
}

RSI_THRESHOLD  = 30    # default signal threshold
HOLDING_PERIOD = 10    # days to hold after signal
TRAIN_RATIO    = 0.7   # walk-forward split
```

To add tickers: append to any sector list. No other changes needed.

---

## Key Findings

Tested across 15 US equities, 5 sectors, 10 years of data. Walk-forward split: 70% train (2016-2023), 30% test (2023-2026).

**Results filtered to tickers with >= 7 test trades. Below that, Sharpe ratio is statistically meaningless.**

13 of 15 tickers met the credibility threshold. WMT and PFE excluded.

### Walk-Forward Results (Credible Only)

| Ticker | Sector | Train Sharpe | Test Sharpe | Degradation | Test Trades | Win Rate |
|--------|--------|-------------|-------------|-------------|-------------|----------|
| JPM | Finance | 0.93 | 1.31 | +0.38 | 22 | 82% |
| COP | Energy | 0.34 | 1.00 | +0.67 | 26 | 69% |
| COST | Consumer | 0.56 | 1.00 | +0.44 | 18 | 67% |
| AAPL | Tech | 0.46 | 1.02 | +0.56 | 7 | 86% |
| GS | Finance | 0.65 | 0.56 | -0.09 | 13 | 62% |
| GOOGL | Tech | 0.66 | 0.52 | -0.14 | 25 | 60% |
| BAC | Finance | 0.62 | 0.39 | -0.23 | 22 | 64% |
| XOM | Energy | 0.09 | 0.44 | +0.35 | 9 | 78% |
| CVX | Energy | 0.37 | 0.10 | -0.28 | 8 | 50% |
| JNJ | Health | 0.72 | 0.29 | -0.43 | 25 | 48% |
| MSFT | Tech | 0.99 | 0.33 | -0.67 | 29 | 45% |
| UNH | Health | 1.19 | -0.23 | -1.42 | 17 | 47% |
| TGT | Consumer | 0.27 | -0.58 | -0.85 | 7 | 43% |

**Aggregate (credible tickers): mean test Sharpe 0.47, 85% positive, mean degradation -0.13.**

### Sector Pattern

| Sector | Result |
|--------|--------|
| Finance | Strong. JPM 1.31, GS/BAC positive |
| Energy | Mixed. COP 1.00, XOM 0.44, CVX weak |
| Consumer | Mixed. COST 1.00, TGT failed |
| Tech | Mixed. AAPL positive, MSFT/GOOGL weak |
| Health | Weak. JNJ marginal, UNH overfitting confirmed |

**Finding:** No clean sector pattern. Finance and energy showed stronger results in the test period (2023-2026), but sample sizes per ticker remain small. Mean Sharpe degradation of -0.13 across credible tickers suggests limited overfitting — signal holds through time better than expected for a simple RSI rule.

This is a descriptive finding across one 10-year period on 15 stocks. It is not a predictive claim.

### Honest Caveats

- 7-26 test trades per credible ticker. Minimum for statistical reliability is higher.
- 2 tickers excluded for insufficient test signals even at RSI < 45.
- All results gross of taxes. Net returns lower.
- Single 10-year window — no regime diversity testing (bull/bear/flat separately).
- Survivorship bias: all tickers are current S&P 500 constituents.
- Transaction cost model (0.1% round-trip) is approximate.

---

## Methodology

### Signal Generation

RSI computed using Wilder's smoothing (`ewm` with `alpha=1/period`). Standard rolling mean gives incorrect RSI values — corrected in this implementation.

Buy signal fires when `RSI < threshold`. One position at a time. No pyramiding.

### Strategy Simulation

```
Entry:  next day open after signal fires
Exit:   close after N holding days
Costs:  0.05% commission + 0.05% slippage per side (0.1% round-trip)
```

### Walk-Forward Validation

```
Train set (70%): optimise RSI threshold by Sharpe ratio
Test set  (30%): evaluate best threshold, no further optimisation
```

Test results only are reported as findings. Train results shown for comparison (Sharpe degradation = test - train).

### Metrics

- Sharpe ratio (annualised, risk-free rate = 0)
- Sortino ratio (downside deviation)
- CAGR
- Maximum drawdown
- Win rate, trade count

---

## Known Limitations and Next Steps

### Current Limitations

| Gap | Impact |
|-----|--------|
| RSI threshold fires too rarely at < 30 on large caps | Test windows have 1-4 trades for half the tickers |
| Single 5-year test period | No regime diversity (bull/bear/sideways) |
| Only 15 tickers, all S&P 500 | Survivorship bias, correlated universe |
| Fixed holding period, no dynamic exit | Misses early reversals, holds through noise |
| No stop-loss | Losing trades can run beyond signal invalidation |

### If Extended

- Raise threshold range (40-50) for higher signal frequency
- Expand to 50+ tickers across small/mid/large cap
- Separate analysis by market regime (VIX-based)
- Add dynamic exit: exit when RSI normalises to 50
- Add statistical mean reversion tests (ADF, Hurst exponent) before assuming asset mean reverts
- Monte Carlo simulation to test result stability

---

## What I Learned

**On backtesting:**
In-sample results are always positive if you search long enough for parameters. Walk-forward validation showed Sharpe degradation of -1.54 for MSFT — the in-sample result was almost entirely overfitting. Building the infrastructure took days; understanding why results are misleading took longer.

**On signal design:**
RSI < 30 is too strict for large-cap equities. Fires 5-10 times per year at most. Sample sizes of 1-4 test trades make Sharpe ratio meaningless. A signal needs frequency to be testable.

**On transaction costs:**
0.1% round-trip cost is conservative. Real impact depends on order size, liquidity, and broker. Even at 0.1%, strategies with low win rates or small average gains can go negative net of costs.

**On sector behaviour:**
Not all stocks mean revert. Trending stocks (tech, growth) punish RSI mean reversion strategies. Commodity-linked stocks (energy) showed strongest signal persistence out-of-sample.

---

## Tech Stack

| Library | Use |
|---------|-----|
| pandas | Data manipulation, time series |
| numpy | Numerical computation |
| yfinance | Historical OHLCV data |
| matplotlib | Diagnostic plots |
| scipy / statsmodels | Statistical metrics |

Python 3.14

---

## Author

Ashwin — BSc Mathematics, MSc Data Analytics (De Montfort University).
Built to understand quantitative research methodology end-to-end.

GitHub: [ashwinbasil](https://github.com/ashwinbasil)
LinkedIn: [ashwin176](https://linkedin.com/in/ashwin176)