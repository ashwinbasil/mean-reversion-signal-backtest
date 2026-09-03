import pandas as pd
import numpy as np

MIN_ROWS = 60  # minimum rows needed to compute all indicators


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["Return"] = df["Close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Return"]).cumprod() - 1
    return df


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df["Volatility20"] = df["Return"].rolling(20).std() * np.sqrt(252)
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI using Wilder's smoothing (EMA with alpha=1/period).
    Previous implementation used simple rolling mean — this is incorrect
    and gives different values than any standard charting tool.
    """
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing: seed with simple mean, then EMA
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)  # avoid division by zero
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    # Renamed from "Signal" to "MACD_Signal" — avoids confusion with RSI_SIGNAL
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < MIN_ROWS:
        print(f"  [add_indicators] Only {len(df)} rows — too few to compute indicators (min {MIN_ROWS}).")
        # Use np.nan, not None — keeps columns as float dtype
        for col in ["Return", "Cumulative_Return", "MA20", "MA50",
                    "Volatility20", "RSI", "MACD", "MACD_Signal"]:
            df[col] = np.nan
        return df

    df = compute_returns(df)
    df = compute_moving_averages(df)
    df = compute_volatility(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    return df


def add_rsi_signal(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    df = df.copy()
    if "RSI" not in df.columns or df["RSI"].isna().all():
        raise ValueError("RSI column missing or all NaN. Run add_indicators() first.")
    df["RSI_SIGNAL"] = (df["RSI"] < threshold).astype(int)
    return df
