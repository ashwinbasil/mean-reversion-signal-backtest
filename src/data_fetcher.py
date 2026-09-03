
import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker.
    Defaults to 5 years — 1y gives too few RSI signals to be meaningful.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'. Check if ticker is valid.")

        df.reset_index(inplace=True)

        # yfinance sometimes returns MultiIndex columns — flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure expected columns exist
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {ticker}: {e}")