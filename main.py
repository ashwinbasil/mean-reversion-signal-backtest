import os
import pandas as pd
from src.data_fetcher import fetch_stock_data
from src.data_cleaner import clean_data
from src.analytics import add_indicators, add_rsi_signal
from src.backtester import benchmark_buy_and_hold, evaluate_signal, simulate_strategy
from src.metrics import evaluate_performance
from src.experiments import rsi_parameter_sweep, horizon_sweep, walk_forward_eval
from src.visualizer import generate_visuals
import sys
sys.stdout.reconfigure(encoding="utf-8")
# ── Configuration ────────────────────────────────────────────
# To scale: add more tickers here. That's the only change needed.
# Organised by sector so results can be compared across sectors.
TICKERS = {
     "tech":     ["AAPL", "MSFT", "GOOGL"],
     "finance":  ["JPM", "BAC", "GS"],
     "energy":   ["XOM", "CVX", "COP"],
     "consumer": ["WMT", "COST", "TGT"],
     "health":   ["JNJ", "PFE", "UNH"],
}

RSI_THRESHOLD  = 30
HOLDING_PERIOD = 10
TRAIN_RATIO    = 0.7   # 70% train, 30% test for walk-forward


def run_ticker(ticker: str) -> dict:
    """
    Full pipeline for a single ticker.
    Returns dict with in-sample perf + walk-forward results for aggregation.
    """
    print(f"\n{'='*50}")
    print(f"  {ticker}")
    print(f"{'='*50}")

    # 1. Fetch and clean
    df = fetch_stock_data(ticker, period="5y")
    df = clean_data(df)
    df.to_csv(f"data/processed/{ticker}_stock.csv", index=False)

    # 2. Indicators and signals
    df = add_indicators(df)
    df = add_rsi_signal(df, threshold=RSI_THRESHOLD)

    # 3. Benchmark
    df = benchmark_buy_and_hold(df)

    # 4. Strategy simulation (with transaction costs)
    df = simulate_strategy(df, signal_col="RSI_SIGNAL",
                           holding_period=HOLDING_PERIOD,
                           commission=0.0005,
                           slippage=0.0005)

    # 5. Save full df (visualizer reads this)
    df.to_csv(f"data/processed/{ticker}_strategy_vs_benchmark.csv", index=False)

    # 6. Signal evaluation
    signal_eval = evaluate_signal(df, "RSI_SIGNAL", horizons=(5, 10, 20))
    signal_eval.to_csv(f"data/processed/{ticker}_rsi_backtest.csv", index=False)
    print(f"\n  Signal Evaluation:")
    print(signal_eval.to_string(index=False))

    # 7. In-sample performance metrics
    perf = evaluate_performance(df)
    perf["ticker"] = ticker
    perf.to_csv(f"data/processed/{ticker}_performance_metrics.csv", index=False)
    print(f"\n  In-Sample Performance:")
    print(perf[["strategy_sharpe", "bh_sharpe",
                "strategy_max_drawdown", "strategy_trade_count",
                "strategy_win_rate"]].to_string(index=False))

    # 8. Parameter sweep (in-sample, for reference only)
    sweep = rsi_parameter_sweep(df, thresholds=(20, 25, 30, 35, 40),
                                holding_period=HOLDING_PERIOD)
    sweep.to_csv(f"data/processed/{ticker}_rsi_parameter_sweep.csv", index=False)

    # 9. Walk-forward validation (the only result that means anything)
    print(f"\n  Walk-Forward Validation (train {int(TRAIN_RATIO*100)}% / "
          f"test {int((1-TRAIN_RATIO)*100)}%):")
    wf = walk_forward_eval(df,
                           train_ratio=TRAIN_RATIO,
                           holding_period=HOLDING_PERIOD,
                           thresholds=(20, 25, 30, 35, 40))

    if not wf.empty:
        wf["ticker"] = ticker
        wf.to_csv(f"data/processed/{ticker}_walk_forward.csv", index=False)
        print(f"\n  Walk-Forward Results:")
        print(wf[["test_sharpe", "train_sharpe", "sharpe_degradation",
                  "test_trade_count", "test_win_rate",
                  "best_threshold"]].to_string(index=False))
    else:
        wf = pd.DataFrame()

    # 10. Visualisations
    generate_visuals(ticker)

    return {
        "ticker":   ticker,
        "in_sample": perf,
        "walk_forward": wf,
    }


def print_final_summary(all_results: list):
    """
    Aggregates in-sample and walk-forward metrics across all tickers.
    This is the section most useful for comparing signal behaviour across assets.
    """
    print(f"\n\n{'='*60}")
    print("  FINAL SUMMARY — ALL TICKERS")
    print(f"{'='*60}")

    is_rows, wf_rows = [], []
    for r in all_results:
        if not r["in_sample"].empty:
            is_rows.append(r["in_sample"])
        if not r["walk_forward"].empty:
            wf_rows.append(r["walk_forward"])

    if is_rows:
        is_summary = pd.concat(is_rows, ignore_index=True)
        is_summary.to_csv("data/processed/_summary_in_sample.csv", index=False)
        print("\n  In-Sample (DO NOT use for conclusions):")
        print(is_summary[["ticker", "strategy_sharpe", "bh_sharpe",
                           "strategy_max_drawdown", "strategy_trade_count",
                           "strategy_win_rate"]].to_string(index=False))

    if wf_rows:
        wf_summary = pd.concat(wf_rows, ignore_index=True)
        wf_summary.to_csv("data/processed/_summary_walk_forward.csv", index=False)
        print("\n  Walk-Forward (the only results that matter):")
        print(wf_summary[["ticker", "train_sharpe", "test_sharpe",
                           "sharpe_degradation", "test_trade_count",
                           "test_win_rate", "best_threshold"]].to_string(index=False))

        # Aggregate signal: does the strategy generalise?
        mean_test_sharpe = wf_summary["test_sharpe"].mean()
        pct_positive     = (wf_summary["test_sharpe"] > 0).mean() * 100
        mean_degradation = wf_summary["sharpe_degradation"].mean()
        print(f"\n  Aggregate signal across {len(wf_summary)} tickers:")
        print(f"    Mean test Sharpe:        {mean_test_sharpe:.3f}")
        print(f"    % tickers positive:      {pct_positive:.0f}%")
        print(f"    Mean Sharpe degradation: {mean_degradation:.3f}")

        if mean_test_sharpe < 0:
            print("  [FAIL] Signal does not generalise out-of-sample on this asset set.")
        elif mean_test_sharpe < 0.3:
            print("  [WEAK] Marginal edge at best. Costs likely kill it.")
        else:
            print("  [OK] Some signal present. Needs validation on more assets + regimes.")


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    all_tickers = [t for sector in TICKERS.values() for t in sector]
    all_results = []

    for ticker in all_tickers:
        result = run_ticker(ticker)
        all_results.append(result)

    print_final_summary(all_results)


if __name__ == "__main__":
    main()