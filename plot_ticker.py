import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import investpy


def resolve_egx_stock(ticker: str):
    results = investpy.search_quotes(
        text=ticker,
        products=["stocks"],
        countries=["egypt"],
    )
    if not results:
        raise ValueError(f"No EGX stock found matching '{ticker}'.")

    exact = [item for item in results if item.symbol.upper() == ticker.upper()]
    if exact:
        return exact[0]
    exact_name = [item for item in results if item.name.upper() == ticker.upper()]
    if exact_name:
        return exact_name[0]
    return results[0]


def fetch_history(ticker: str, from_date: str, to_date: str):
    quote = resolve_egx_stock(ticker)
    data = quote.retrieve_historical_data(from_date=from_date, to_date=to_date)
    if data.empty:
        raise ValueError(f"No data returned for EGX ticker '{ticker}'.")
    return data


def plot_time_series(data, ticker: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    close_col = "Close"
    if close_col not in data.columns:
        for col in data.columns:
            if col.lower() == "close":
                close_col = col
                break

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[close_col], label="Close")
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
    return start.strftime("%d/%m/%Y"), today.strftime("%d/%m/%Y")


def main():
    parser = argparse.ArgumentParser(
        description="Download EGX ticker data and plot its time series."
    )
    parser.add_argument("--ticker", default="ABR", help="EGX ticker symbol to download")
    parser.add_argument("--from-date", help="Start date (DD/MM/YYYY)")
    parser.add_argument("--to-date", help="End date (DD/MM/YYYY)")
    parser.add_argument("--output", default="artifacts/ABR_timeseries.png", help="Output image path")
    args = parser.parse_args()

    from_date, to_date = args.from_date, args.to_date
    if not from_date or not to_date:
        from_date, to_date = default_dates()

    data = fetch_history(args.ticker, from_date=from_date, to_date=to_date)
    output_path = Path(args.output)
    plot_time_series(data, args.ticker, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
