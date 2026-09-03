import pandas as pd
import numpy as np

TRADING_DAYS = 252


def compute_cagr(cumulative_return: float, periods: int) -> float:
    if periods <= 0:
        return np.nan
    return (1 + cumulative_return) ** (TRADING_DAYS / periods) - 1


def compute_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / TRADING_DAYS
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def compute_sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / TRADING_DAYS
    downside = excess[excess < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def compute_max_drawdown(cumulative_returns: pd.Series) -> float:
    cumulative = cumulative_returns + 1
    rolling_max = cumulative.cummax().replace(0, np.nan)
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def evaluate_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares strategy (RSI-signal-driven simulation) vs buy-and-hold.

    Requires columns produced by:
    - analytics.add_indicators()         -> Return, Cumulative_Return
    - backtester.benchmark_buy_and_hold() -> BH_Return, BH_Cumulative_Return
    - backtester.simulate_strategy()     -> Strategy_Return, Strategy_Cumulative
    """
    df = df.dropna(subset=["BH_Return"]).copy()
    periods = len(df)
    metrics = {}

    # --- Strategy metrics ---
    col_exists = "Strategy_Return" in df.columns
    has_trades = col_exists and df["Strategy_Return"].abs().sum() > 0

    if col_exists and has_trades:
        strat_returns = df["Strategy_Return"]
        strat_cum = df["Strategy_Cumulative"]
        trade_mask = strat_returns != 0
        metrics["strategy_cagr"]         = compute_cagr(strat_cum.iloc[-1], periods)
        metrics["strategy_volatility"]   = compute_volatility(strat_returns)
        metrics["strategy_sharpe"]       = compute_sharpe(strat_returns)
        metrics["strategy_sortino"]      = compute_sortino(strat_returns)
        metrics["strategy_max_drawdown"] = compute_max_drawdown(strat_cum)
        metrics["strategy_trade_count"]  = int(trade_mask.sum())
        metrics["strategy_win_rate"]     = float((strat_returns[trade_mask] > 0).mean())
    elif col_exists and not has_trades:
        # Column exists but this threshold produced zero signals — valid, not an error
        metrics["strategy_cagr"]         = 0.0
        metrics["strategy_volatility"]   = 0.0
        metrics["strategy_sharpe"]       = np.nan
        metrics["strategy_sortino"]      = np.nan
        metrics["strategy_max_drawdown"] = 0.0
        metrics["strategy_trade_count"]  = 0
        metrics["strategy_win_rate"]     = np.nan
    else:
        # Column genuinely missing — pipeline not run correctly
        print("  [evaluate_performance] Warning: Strategy_Return column missing. "
              "Run backtester.simulate_strategy() first. Falling back to raw returns.")
        metrics["strategy_cagr"]         = compute_cagr(df["Cumulative_Return"].iloc[-1], periods)
        metrics["strategy_volatility"]   = compute_volatility(df["Return"])
        metrics["strategy_sharpe"]       = compute_sharpe(df["Return"])
        metrics["strategy_sortino"]      = compute_sortino(df["Return"])
        metrics["strategy_max_drawdown"] = compute_max_drawdown(df["Cumulative_Return"])
        metrics["strategy_trade_count"]  = np.nan
        metrics["strategy_win_rate"]     = np.nan

    # --- Buy and hold metrics ---
    metrics["bh_cagr"]         = compute_cagr(df["BH_Cumulative_Return"].iloc[-1], periods)
    metrics["bh_volatility"]   = compute_volatility(df["BH_Return"])
    metrics["bh_sharpe"]       = compute_sharpe(df["BH_Return"])
    metrics["bh_sortino"]      = compute_sortino(df["BH_Return"])
    metrics["bh_max_drawdown"] = compute_max_drawdown(df["BH_Cumulative_Return"])

    return pd.DataFrame([metrics])