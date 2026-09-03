import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # safe: runs at import time


def _apply_date_formatting(ax):
    """Single date formatter — not overridden elsewhere anymore."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def plot_price_with_signals(df: pd.DataFrame, ticker: str):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Date"], df["Close"], label="Close Price", color="steelblue", linewidth=1)

    signal_points = df[df["RSI_SIGNAL"] == 1]
    if len(signal_points) > 0:
        ax.scatter(
            signal_points["Date"],
            signal_points["Close"],
            marker="^",
            color="green",
            zorder=5,
            s=60,
            label=f"RSI Buy Signal (n={len(signal_points)})"
        )
    else:
        print(f"  [visualizer] No signals found for {ticker} — signal scatter will be empty.")

    _apply_date_formatting(ax)
    ax.set_title(f"{ticker} — Close Price with RSI Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{ticker}_price_signals.png", dpi=150)
    plt.close()


def plot_strategy_vs_benchmark(df: pd.DataFrame, ticker: str):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Date"], df["BH_Cumulative_Return"], label="Buy & Hold",
            color="steelblue", linewidth=1.2)

    if "Strategy_Cumulative" in df.columns and df["Strategy_Cumulative"].abs().sum() > 0:
        ax.plot(df["Date"], df["Strategy_Cumulative"], label="RSI Strategy (net of costs)",
                color="darkorange", linewidth=1.2)
    else:
        print(f"  [visualizer] Strategy_Cumulative not found for {ticker}. "
              "Run simulate_strategy() before plotting.")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    _apply_date_formatting(ax)
    ax.set_title(f"{ticker} — Strategy vs Buy & Hold (In-Sample, Gross Returns)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{ticker}_strategy_vs_bh.png", dpi=150)
    plt.close()


def plot_drawdown(df: pd.DataFrame, ticker: str):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    cumulative = df["BH_Cumulative_Return"] + 1
    rolling_max = cumulative.cummax().replace(0, float("nan"))
    drawdown = (cumulative - rolling_max) / rolling_max

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(df["Date"], drawdown, 0, color="crimson", alpha=0.4, label="Drawdown")
    ax.plot(df["Date"], drawdown, color="crimson", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    _apply_date_formatting(ax)
    ax.set_title(f"{ticker} — Buy & Hold Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{ticker}_drawdown.png", dpi=150)
    plt.close()


def generate_visuals(ticker: str):
    """
    Fix: reads the strategy CSV (which has RSI_SIGNAL + Strategy_Cumulative),
    not the raw stock CSV (which only has OHLCV).
    """
    path = f"data/processed/{ticker}_strategy_vs_benchmark.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected {path}. Run the full pipeline in main.py first."
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    plot_price_with_signals(df, ticker)
    plot_strategy_vs_benchmark(df, ticker)
    plot_drawdown(df, ticker)
    print(f"  [visualizer] Plots saved to {OUTPUT_DIR}/ for {ticker}")