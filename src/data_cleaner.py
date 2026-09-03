import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)

    # Strip timezone from Date so comparisons and CSV saves work cleanly
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    # Drop rows with missing OHLC values
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Drop zero-volume rows (data quality issue, often non-trading days)
    zero_vol = (df["Volume"] == 0).sum()
    if zero_vol > 0:
        print(f"  [clean_data] Dropping {zero_vol} zero-volume rows.")
        df = df[df["Volume"] > 0]

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Sort by date, reset index
    df = df.sort_values("Date").reset_index(drop=True)

    dropped = original_len - len(df)
    if dropped > 0:
        print(f"  [clean_data] Dropped {dropped} rows total. {len(df)} rows remaining.")

    return df