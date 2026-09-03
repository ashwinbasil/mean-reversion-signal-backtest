import pandas as pd
from src.analytics import add_rsi_signal
from src.backtester import evaluate_signal, benchmark_buy_and_hold, simulate_strategy
from src.metrics import evaluate_performance


def rsi_parameter_sweep(df: pd.DataFrame,
                         thresholds: tuple = (20, 25, 30, 35, 40),
                         holding_period: int = 10) -> pd.DataFrame:
    """
    Sweeps RSI thresholds — IN-SAMPLE only.
    Used internally by walk_forward_eval to find best threshold on train set.
    """
    results = []
    for t in thresholds:
        df_test = add_rsi_signal(df.copy(), threshold=t)
        df_test = benchmark_buy_and_hold(df_test)
        df_test = simulate_strategy(df_test, signal_col="RSI_SIGNAL",
                                    holding_period=holding_period)
        perf = evaluate_performance(df_test)
        perf["rsi_threshold"] = t
        perf["num_signals"] = int((df_test["RSI_SIGNAL"] == 1).sum())
        results.append(perf)

    combined = pd.concat(results, ignore_index=True)
    for col in ["strategy_trade_count"]:
        if col in combined.columns:
            combined[col] = combined[col].astype("Int64")
    return combined


def horizon_sweep(df: pd.DataFrame,
                  thresholds: tuple = (25, 30, 35),
                  horizons: tuple = (5, 10, 20)) -> pd.DataFrame:
    """
    Cross-sweeps RSI threshold x holding period.
    Does NOT imply out-of-sample validity.
    """
    results = []
    for t in thresholds:
        df_test = add_rsi_signal(df.copy(), threshold=t)
        signal_results = evaluate_signal(df_test, "RSI_SIGNAL", horizons=horizons)
        signal_results["rsi_threshold"] = t
        results.append(signal_results)
    return pd.concat(results, ignore_index=True)


def walk_forward_eval(df: pd.DataFrame,
                      train_ratio: float = 0.7,
                      holding_period: int = 10,
                      thresholds: tuple = (20, 25, 30, 35, 40)) -> pd.DataFrame:
    """
    Proper train/test validation.

    Step 1: Split data temporally (no shuffling — time series data)
    Step 2: Optimise RSI threshold on TRAIN set only (by Sharpe ratio)
    Step 3: Evaluate that threshold on TEST set only
    Step 4: Return test metrics + what threshold was chosen + period labels

    Why this matters:
    - In-sample sweep always finds a "best" threshold even on random data
    - Only out-of-sample performance tells you if the signal has real edge
    - If test Sharpe drops significantly from train Sharpe → overfitting confirmed
    """
    split_idx = int(len(df) * train_ratio)

    # Hard temporal split — train is older data, test is newer data
    # Never shuffle time series data. Future leaks into past = lookahead bias.
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    print(f"  [walk_forward] Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"  [walk_forward] Train period: {train_df['Date'].iloc[0].date()} -> "
          f"{train_df['Date'].iloc[-1].date()}")
    print(f"  [walk_forward] Test  period: {test_df['Date'].iloc[0].date()} -> "
          f"{test_df['Date'].iloc[-1].date()}")

    # --- Step 1: Find best threshold on train set ---
    train_sweep = rsi_parameter_sweep(train_df,
                                      thresholds=thresholds,
                                      holding_period=holding_period)

    # Pick threshold with highest Sharpe on train
    # Drop rows where Sharpe is NaN (threshold fired 0 signals)
    valid = train_sweep.dropna(subset=["strategy_sharpe"])
    if valid.empty:
        print("  [walk_forward] No valid threshold found on train set. "
              "Too few signals — consider loosening thresholds.")
        return pd.DataFrame()

    best_row = valid.loc[valid["strategy_sharpe"].idxmax()]
    best_threshold = int(best_row["rsi_threshold"])
    train_sharpe   = round(float(best_row["strategy_sharpe"]), 4)
    train_trades   = int(best_row["strategy_trade_count"]) if pd.notna(best_row["strategy_trade_count"]) else 0

    print(f"  [walk_forward] Best threshold on train: RSI < {best_threshold} "
          f"(Sharpe={train_sharpe}, trades={train_trades})")

    # --- Step 2: Evaluate best threshold on test set ONLY ---
    test_df = add_rsi_signal(test_df, threshold=best_threshold)
    test_df = benchmark_buy_and_hold(test_df)
    test_df = simulate_strategy(test_df, signal_col="RSI_SIGNAL",
                                holding_period=holding_period)
    test_perf = evaluate_performance(test_df)

    # --- Step 3: Annotate results ---
    test_perf["best_threshold"]    = best_threshold
    test_perf["train_sharpe"]      = train_sharpe
    test_perf["test_sharpe"]       = round(float(test_perf["strategy_sharpe"].iloc[0]), 4)
    test_perf["sharpe_degradation"] = round(
        float(test_perf["strategy_sharpe"].iloc[0]) - train_sharpe, 4
    )
    test_perf["train_period"] = (f"{train_df['Date'].iloc[0].date()} -> "
                                  f"{train_df['Date'].iloc[-1].date()}")
    test_perf["test_period"]  = (f"{test_df['Date'].iloc[0].date()} -> "
                                  f"{test_df['Date'].iloc[-1].date()}")
    test_trade_count = int((test_df["Strategy_Return"] != 0).sum())
    test_perf["test_trade_count"] = test_trade_count
    test_perf["test_win_rate"]    = round(
        float((test_df.loc[test_df["Strategy_Return"] != 0, "Strategy_Return"] > 0).mean()), 4
    ) if test_trade_count > 0 else float("nan")

    # Key diagnostic: how much did Sharpe drop from train to test?
    degradation = test_perf["sharpe_degradation"].iloc[0]
    if degradation < -0.3:
     print(f"  [walk_forward] WARNING: Large Sharpe degradation ({degradation:.3f}). "
          "Strong sign of in-sample overfitting.")
    elif degradation < 0:
     print(f"  [walk_forward] Sharpe declined on test ({degradation:.3f}). "
          "Expected. Check if test results are still positive.")
    elif degradation < 0.5:
     print(f"  [walk_forward] Sharpe improved on test ({degradation:+.3f}). "
          "Test period was favorable for this strategy.")
    else:
     print(f"  [walk_forward] Large Sharpe improvement on test ({degradation:+.3f}). "
          "Could indicate train period was unfavorable outlier. Investigate.")

    return test_perf