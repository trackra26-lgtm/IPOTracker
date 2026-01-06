import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests


EODHD_BASE_URL = "https://eodhd.com/api/eod"


def fetch_history(ticker: str, from_date: str, to_date: str, api_key: str) -> pd.DataFrame:
    if not api_key:
        raise ValueError("Missing EODHD API key. Set EODHD_API_KEY environment variable.")

    symbol = f"{ticker}.EGX"
    params = {
        "api_token": api_key,
        "from": from_date,
        "to": to_date,
        "fmt": "json",
    }
    response = requests.get(f"{EODHD_BASE_URL}/{symbol}", params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not payload:
        raise ValueError(f"No data returned for EGX ticker '{ticker}'.")

    data = pd.DataFrame(payload)
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date").sort_index()
    return data


def plot_time_series(data: pd.DataFrame, ticker: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data["close"], label="Close")
    plt.title(f"EGX {ticker} Closing Price ({len(data)} points)")
    plt.xlabel("Date")
    plt.ylabel("Price (EGP)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def default_dates():
    today = datetime.utcnow().date()
    start = today - timedelta(days=365)
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def main():
    parser = argparse.ArgumentParser(
        description="Download EGX ticker data and plot its time series."
    )
    parser.add_argument("--ticker", default="ABR", help="EGX ticker symbol to download")
    parser.add_argument("--from-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="artifacts/ABR_timeseries.png", help="Output image path")
    parser.add_argument(
        "--api-key",
        default=None,
        help="EODHD API key (or set EODHD_API_KEY env var)",
    )
    args = parser.parse_args()

    from_date, to_date = args.from_date, args.to_date
    if not from_date or not to_date:
        from_date, to_date = default_dates()

    api_key = args.api_key or os.getenv("EODHD_API_KEY")
    data = fetch_history(args.ticker, from_date=from_date, to_date=to_date, api_key=api_key)
    output_path = Path(args.output)
    plot_time_series(data, args.ticker, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
