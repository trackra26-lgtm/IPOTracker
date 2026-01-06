#!/usr/bin/env python3
"""Run FFT denoising + autocorrelation checks for a single EGX ticker."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_TOP_K = 30


@dataclass(frozen=True)
class CycleCandidate:
    period_days: float
    frequency: float
    power: float


def resolve_egxpy_loaders() -> list[Tuple[str, Callable]]:
    try:
        import egxpy  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "egxpy is required to fetch EGX data. Install it and retry."
        ) from exc

    loaders: list[Tuple[str, Callable]] = []
    module_candidates = [
        "get_historical_prices",
        "get_historical_data",
        "get_prices",
        "history",
    ]
    for name in module_candidates:
        if hasattr(egxpy, name):
            loaders.append((f"egxpy.{name}", getattr(egxpy, name)))

    client_candidates = ["EGXClient", "Client", "EGX"]
    client_methods = [
        "get_historical_prices",
        "get_historical_data",
        "historical_prices",
        "history",
    ]
    for class_name in client_candidates:
        if not hasattr(egxpy, class_name):
            continue
        client = getattr(egxpy, class_name)()
        for method in client_methods:
            if hasattr(client, method):
                loaders.append((f"{class_name}.{method}", getattr(client, method)))

    if not loaders:
        raise RuntimeError(
            "No compatible egxpy loader found. Check egxpy's API and update this script."
        )
    return loaders


def normalize_egx_data(data: object) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, pd.Series):
        df = data.to_frame(name="Close")
    else:
        df = pd.DataFrame(data)

    if df.empty:
        return df

    date_col = next(
        (col for col in df.columns if str(col).lower() in {"date", "datetime", "timestamp"}),
        None,
    )
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("egxpy data must include a date column or DatetimeIndex.")

    price_columns = [
        "close",
        "adj_close",
        "adj close",
        "last",
        "price",
    ]
    price_col = next(
        (col for col in df.columns if str(col).lower() in price_columns),
        None,
    )
    if price_col is None:
        raise RuntimeError("Unable to locate close/price column in egxpy data.")

    df = df[[price_col]].rename(columns={price_col: "Close"}).sort_index()
    return df.dropna()


def fetch_egx_history(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    loaders = resolve_egxpy_loaders()
    last_error: Exception | None = None

    for name, loader in loaders:
        try:
            try:
                data = loader(ticker=ticker, start=start, end=end)
            except TypeError:
                data = loader(ticker, start, end)
            df = normalize_egx_data(data)
            if not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001 - want to keep trying loaders
            last_error = exc
            continue

    message = "All egxpy loaders failed."
    if last_error is not None:
        message = f"{message} Last error: {last_error}"
    raise RuntimeError(message)


def compute_log_returns(prices: pd.Series) -> np.ndarray:
    log_prices = np.log(prices.to_numpy(dtype=float))
    return np.diff(log_prices)


def fft_denoise(
    returns: np.ndarray, top_k: int, dt_years: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fft_values = np.fft.fft(returns)
    freqs = np.fft.fftfreq(len(returns), d=dt_years)
    power = np.abs(fft_values) ** 2

    dc_idx = np.argmin(np.abs(freqs))
    power_no_dc = power.copy()
    power_no_dc[dc_idx] = 0.0

    top_k = min(top_k, len(returns))
    top_idx = np.argpartition(power_no_dc, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(power_no_dc[top_idx])[::-1]]

    filtered = np.zeros_like(fft_values)
    filtered[top_idx] = fft_values[top_idx]
    denoised = np.fft.ifft(filtered).real
    return denoised, freqs, power, top_idx


def extract_cycles(
    freqs: np.ndarray,
    power: np.ndarray,
    top_idx: np.ndarray,
    min_period_days: int,
    max_period_days: int,
) -> list[CycleCandidate]:
    cycles: list[CycleCandidate] = []
    for idx in top_idx:
        freq = freqs[idx]
        if freq == 0:
            continue
        period_days = abs((1 / freq) * TRADING_DAYS)
        if min_period_days <= period_days <= max_period_days:
            cycles.append(CycleCandidate(period_days, freq, power[idx]))
    cycles.sort(key=lambda c: c.power, reverse=True)
    return cycles


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    if series.size == 0:
        return np.array([])
    series = series - np.mean(series)
    denom = np.dot(series, series)
    if denom == 0:
        return np.zeros(max_lag + 1)
    corr = np.correlate(series, series, mode="full")
    corr = corr[corr.size // 2 :]
    corr = corr[: max_lag + 1]
    return corr / denom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FFT denoising and autocorrelation on an EGX ticker."
    )
    parser.add_argument("--ticker", required=True, help="EGX ticker symbol")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-period-days", type=int, default=5)
    parser.add_argument("--max-period-days", type=int, default=TRADING_DAYS)
    parser.add_argument("--max-lag-days", type=int, default=TRADING_DAYS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prices_df = fetch_egx_history(args.ticker, args.start, args.end)
    prices = prices_df["Close"].dropna()

    if len(prices) < 60:
        raise RuntimeError("Not enough data points for FFT analysis.")

    returns = compute_log_returns(prices)
    if len(returns) < 50:
        raise RuntimeError("Not enough log returns for FFT analysis.")

    denoised, freqs, power, top_idx = fft_denoise(
        returns, top_k=args.top_k, dt_years=1 / TRADING_DAYS
    )
    cycles = extract_cycles(
        freqs, power, top_idx, args.min_period_days, args.max_period_days
    )

    acf = autocorrelation(denoised, max_lag=args.max_lag_days)
    acf_lags = np.arange(acf.size)
    peak_lags = (
        acf_lags[1:][np.argsort(acf[1:])[::-1][:5]] if acf.size > 1 else []
    )

    print("\n" + "=" * 60)
    print(f"EGX CYCLICALITY CHECK: {args.ticker}")
    print("=" * 60)
    print(f"Data points: {len(prices)}")
    print(f"Log returns: {len(returns)}")
    print(f"Top-k frequencies: {args.top_k}")

    print("\nTop cycle candidates (period days / power):")
    if cycles:
        for idx, cycle in enumerate(cycles[:10], 1):
            months = cycle.period_days / 21
            print(
                f"{idx:>2}. {cycle.period_days:6.1f} days (~{months:4.1f} months)"
                f" | power={cycle.power:.2e}"
            )
    else:
        print("No cycles detected in the requested period range.")

    print("\nTop autocorrelation lags (days / corr):")
    if len(peak_lags) > 0:
        for lag in peak_lags:
            print(f"lag {lag:>3} days | corr={acf[lag]:.3f}")
    else:
        print("Autocorrelation not available.")


if __name__ == "__main__":
    main()
