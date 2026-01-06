import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf


def fetch_history(ticker: str, period: str = "1y", interval: str = "1d"):
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    return data


def plot_time_series(data, ticker: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data["Close"], label="Close")
    plt.title(f"{ticker} Closing Price ({len(data)} points)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Download ticker data and plot its time series.")
    parser.add_argument("--ticker", default="ABR", help="Ticker symbol to download")
    parser.add_argument("--period", default="1y", help="Lookback period (e.g., 1y, 6mo)")
    parser.add_argument("--interval", default="1d", help="Data interval (e.g., 1d, 1h)")
    parser.add_argument("--output", default="artifacts/ABR_timeseries.png", help="Output image path")
    args = parser.parse_args()

    data = fetch_history(args.ticker, period=args.period, interval=args.interval)
    output_path = Path(args.output)
    plot_time_series(data, args.ticker, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
