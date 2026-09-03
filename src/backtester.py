import pandas as pd
import numpy as np


def evaluate_signal(df: pd.DataFrame, signal_col: str, horizons=(5, 10, 20)) -> pd.DataFrame:
    """
    Measures average forward returns after signal fires.
    This is a signal correlation study, not a full backtest.
    Results are in-sample and gross of transaction costs.
    """
    results = []
    for h in horizons:
        forward_return = df["Close"].shift(-h) / df["Close"] - 1
        signal_returns = forward_return[df[signal_col] == 1]
        results.append({
            "horizon_days": h,
            "num_signals": int(signal_returns.count()),
            "mean_return": round(signal_returns.mean(), 4),
            "median_return": round(signal_returns.median(), 4),
            "win_rate": round((signal_returns > 0).mean(), 4),
        })
    return pd.DataFrame(results)


def simulate_strategy(df: pd.DataFrame, signal_col: str = "RSI_SIGNAL",
                      holding_period: int = 10,
                      commission: float = 0.0005,
                      slippage: float = 0.0005) -> pd.DataFrame:
    """
    Simulates a simple long-only strategy based on RSI signals.

    - Enters at next day's open when signal fires
    - Exits after holding_period days
    - Applies commission + slippage on entry and exit
    - One position at a time (no pyramiding)

    This produces a real strategy equity curve distinct from buy-and-hold.
    commission/slippage default: 0.05% each = 0.1% round-trip cost
    """
    df = df.copy().reset_index(drop=True)
    strategy_returns = pd.Series(0.0, index=df.index)

    in_position = False
    entry_idx = None

    for i in range(len(df) - holding_period - 1):
        if not in_position and df.loc[i, signal_col] == 1:
            # Enter at next day's open
            entry_price = df.loc[i + 1, "Open"] if "Open" in df.columns else df.loc[i + 1, "Close"]
            entry_idx = i + 1
            in_position = True

        if in_position and i == entry_idx + holding_period:
            # Exit at close after holding period
            exit_price = df.loc[i, "Close"]
            gross_return = (exit_price / entry_price) - 1
            # Deduct round-trip transaction costs
            net_return = gross_return - 2 * (commission + slippage)
            strategy_returns.iloc[i] = net_return
            in_position = False

    df["Strategy_Return"] = strategy_returns
    df["Strategy_Cumulative"] = (1 + df["Strategy_Return"]).cumprod() - 1
    return df


def benchmark_buy_and_hold(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BH_Return"] = df["Close"].pct_change()
    df["BH_Cumulative_Return"] = (1 + df["BH_Return"]).cumprod() - 1
    return df


def compute_excess_return(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy cumulative return minus buy-and-hold cumulative return."""
    df = df.copy()
    strat_col = "Strategy_Cumulative" if "Strategy_Cumulative" in df.columns else "Cumulative_Return"
    df["Excess_Return"] = df[strat_col] - df["BH_Cumulative_Return"]
    return df


def compute_max_drawdown(series: pd.Series) -> float:
    """
    Maximum drawdown of a cumulative return series.
    Fix: guard against division by zero when cumulative_max is 0.
    """
    cumulative = series + 1  # convert returns to equity curve
    cumulative_max = cumulative.cummax()
    # Replace zeros to avoid division by zero
    cumulative_max = cumulative_max.replace(0, np.nan)
    drawdown = (cumulative - cumulative_max) / cumulative_max
    return float(drawdown.min())