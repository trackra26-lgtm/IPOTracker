#!/usr/bin/env python3
"""Calculate volatility and max drawdown for a weighted ticker basket."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


@dataclass(frozen=True)
class Metrics:
    ticker: str
    volatility: float
    max_drawdown: float


def annualized_volatility(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    return returns.std() * np.sqrt(trading_days)


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Total weight must be positive.")
    return {ticker: weight / total for ticker, weight in weights.items()}


def load_prices(tickers: Iterable[str], start: str | None, end: str | None) -> pd.DataFrame:
    data = yf.download(list(tickers), start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise RuntimeError("No data returned from Yahoo Finance.")
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"] if "Close" in data.columns.levels[0] else data["Adj Close"]
    else:
        prices = data["Close"] if "Close" in data.columns else data
    prices = prices.dropna(how="all")
    return prices


def compute_metrics(returns: pd.DataFrame) -> List[Metrics]:
    results = []
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        if series.empty:
            continue
        results.append(
            Metrics(
                ticker=ticker,
                volatility=annualized_volatility(series),
                max_drawdown=max_drawdown(series),
            )
        )
    return results


def portfolio_returns(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    aligned = returns[list(weights.keys())].dropna(how="all")
    weight_vector = pd.Series(weights)
    weight_vector = weight_vector.reindex(aligned.columns).fillna(0.0)
    return aligned.mul(weight_vector, axis=1).sum(axis=1)


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_default_weights() -> Dict[str, float]:
    raw_weights = {
        "TLT": 0.20,
        "SHY": 0.20,
        "GLD": 0.30,
        "SPY": 0.20,
        "LLY": 0.10,
    }
    return normalize_weights(raw_weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate volatility and max drawdown.")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = build_default_weights()
    tickers = sorted(weights.keys())

    prices = load_prices(tickers, args.start, args.end)
    returns = prices.pct_change().dropna(how="all")

    metrics = compute_metrics(returns)
    portfolio = portfolio_returns(returns, weights)
    portfolio_metrics = Metrics(
        ticker="PORTFOLIO",
        volatility=annualized_volatility(portfolio),
        max_drawdown=max_drawdown(portfolio),
    )

    rows = metrics + [portfolio_metrics]
    df = pd.DataFrame(
        [
            {
                "Ticker": row.ticker,
                "Volatility": format_percentage(row.volatility),
                "Max Drawdown": format_percentage(row.max_drawdown),
            }
            for row in rows
        ]
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
