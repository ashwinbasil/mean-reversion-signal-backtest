import pandas as pd
import numpy as np


def train_test_split(df: pd.DataFrame, split_ratio: float = 0.7):
    """
    Split data into chronological train/test sets.
    """
    split_idx = int(len(df) * split_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def apply_transaction_costs(returns: pd.Series, cost_per_trade: float):
    """
    Apply fixed transaction cost per signal.
    Assumes cost is paid when signal == 1.
    """
    costs = returns.copy()
    costs[:] = 0.0
    costs[returns.index] = cost_per_trade
    return returns - costs


def evaluate_strategy(df: pd.DataFrame, cost_per_trade: float = 0.001):
    """
    Evaluate cumulative return and Sharpe ratio.
    """
    df = df.copy()

    # Strategy daily returns only when signal is active
    df["Strategy_Return"] = df["Daily_Return"] * df["RSI_SIGNAL"]

    # Apply transaction costs
    trade_costs = df["RSI_SIGNAL"] * cost_per_trade
    df["Net_Return"] = df["Strategy_Return"] - trade_costs

    cumulative_return = (1 + df["Net_Return"]).prod() - 1

    if df["Net_Return"].std() == 0:
        sharpe = 0.0
    else:
        sharpe = (
            df["Net_Return"].mean() / df["Net_Return"].std()
        ) * np.sqrt(252)

    return {
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe,
        "num_trades": int(df["RSI_SIGNAL"].sum())
    }


def run_validation(ticker: str):
    """
    Full validation pipeline:
    - Train/test split
    - Performance comparison
    - Degradation reporting
    """
    df = pd.read_csv(f"data/processed/{ticker}_stock.csv")

    # Ensure required columns exist
    required_cols = {"Daily_Return", "RSI_SIGNAL"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Required columns missing for validation")

    train, test = train_test_split(df)

    train_metrics = evaluate_strategy(train)
    test_metrics = evaluate_strategy(test)

    degradation = (
        test_metrics["cumulative_return"]
        - train_metrics["cumulative_return"]
    )

    results = {
        "ticker": ticker,
        "train_cumulative_return": train_metrics["cumulative_return"],
        "test_cumulative_return": test_metrics["cumulative_return"],
        "train_sharpe": train_metrics["sharpe_ratio"],
        "test_sharpe": test_metrics["sharpe_ratio"],
        "train_trades": train_metrics["num_trades"],
        "test_trades": test_metrics["num_trades"],
        "performance_degradation": degradation
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(
        f"data/processed/{ticker}_validation_results.csv",
        index=False
    )

    print(f"Validation results saved for {ticker}")

    return results_df
