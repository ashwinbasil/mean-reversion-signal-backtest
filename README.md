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

Tested across 15 US equities, 5 sectors, 5 years of data (2021-2026). Walk-forward split: 70% train, 30% test.

**Results filtered to tickers with >= 7 test trades. Below that, Sharpe ratio is statistically meaningless.**

### Walk-Forward Results (Credible Only)

| Ticker | Sector | Train Sharpe | Test Sharpe | Degradation | Test Trades | Win Rate |
|--------|--------|-------------|-------------|-------------|-------------|----------|
| XOM | Energy | 1.15 | 1.73 | +0.58 | 10 | 80% |
| WMT | Consumer | 0.70 | 1.02 | +0.32 | 11 | 64% |
| COST | Consumer | 0.64 | 1.05 | +0.42 | 14 | 64% |
| JPM | Finance | 0.75 | 0.99 | +0.24 | 7 | 57% |
| CVX | Energy | 0.42 | 0.44 | +0.01 | 8 | 50% |
| UNH | Health | 0.94 | 0.11 | -0.83 | 12 | 58% |
| MSFT | Tech | 1.11 | -0.43 | -1.54 | 9 | 56% |

8 of 15 tickers excluded: insufficient test signals at chosen threshold.

### Sector Pattern

| Sector | Signal |
|--------|--------|
| Energy | Positive. XOM test Sharpe 1.73 |
| Consumer | Positive. WMT/COST both above 1.0 |
| Finance | Positive. JPM holds well |
| Health | Mixed. UNH shows overfitting |
| Tech | Negative. MSFT overfitting confirmed |

**Finding:** RSI mean reversion shows sector-dependent behaviour. Energy and consumer stocks exhibit stronger mean-reverting dynamics. Tech stocks trend rather than revert — RSI signals underperform on them.

This is a descriptive finding across one 5-year period on 15 stocks. It is not a predictive claim.

### Honest Caveats

- Test windows yield 1-14 trades per ticker at RSI < 30-40 threshold. Most are statistically weak.
- 8 tickers excluded entirely for insufficient test signals.
- All positive results are gross of taxes; net returns lower.
- Same 5-year period (2021-2026) used for all tickers — no regime diversity.
- Survivorship bias: all tickers are current S&P 500 constituents.
- Transaction cost model (0.1% round-trip) is approximate. Real costs vary by broker and order size.

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

LinkedIn: [ashwin176](https://linkedin.com/in/ashwin176)