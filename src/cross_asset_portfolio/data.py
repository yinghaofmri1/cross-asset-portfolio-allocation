"""Yahoo Finance download and local data loading."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd
import yfinance as yf

from .config import TICKERS


def download_yahoo_data(
    data_dir: Path,
    tickers: Sequence[str] = TICKERS,
    start_date: str = "2007-01-01",
    end_date_exclusive: str | None = None,
) -> dict[str, Path]:
    """Download adjusted daily ETF data and save it in local CSV files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    end_date_exclusive = end_date_exclusive or (
        date.today() + timedelta(days=1)
    ).isoformat()
    ticker_list = list(tickers)

    # auto_adjust=True means OHLC prices are already adjusted.
    prices = yf.download(
        tickers=ticker_list,
        start=start_date,
        end=end_date_exclusive,
        interval="1d",
        auto_adjust=True,
        actions=False,
        group_by="column",
        threads=True,
        progress=True,
    )
    if prices.empty:
        raise RuntimeError("Yahoo Finance returned no data")

    prices.index.name = "Date"
    adjusted_close = prices["Close"].reindex(columns=ticker_list)
    adjusted_close.index.name = "Date"

    paths = {
        "adjusted_close": data_dir / "etf_adjusted_close.csv",
        "ohlcv": data_dir / "etf_daily_ohlcv.csv",
        "coverage": data_dir / "etf_data_coverage.csv",
        "metadata": data_dir / "metadata.json",
    }
    adjusted_close.to_csv(paths["adjusted_close"], date_format="%Y-%m-%d")
    prices.to_csv(paths["ohlcv"], date_format="%Y-%m-%d")

    coverage_rows = []
    for ticker in ticker_list:
        series = adjusted_close[ticker].dropna()
        coverage_rows.append(
            {
                "ticker": ticker,
                "first_date": series.index.min().date().isoformat(),
                "last_date": series.index.max().date().isoformat(),
                "observations": int(series.size),
                "missing_values": int(adjusted_close[ticker].isna().sum()),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(paths["coverage"], index=False)

    metadata = {
        "source": "Yahoo Finance via yfinance",
        "yfinance_version": yf.__version__,
        "tickers": ticker_list,
        "requested_start_date": start_date,
        "requested_end_date_exclusive": end_date_exclusive,
        "interval": "1d",
        "auto_adjust": True,
        "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "notes": [
            "Open, High, Low and Close are adjusted because auto_adjust=True.",
            "Please check Yahoo terms before using the data.",
        ],
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return paths


def load_adjusted_close(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load adjusted close prices and calculate daily returns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file is not found: {path}. "
            "Please run scripts/download_data.py first."
        )

    prices = pd.read_csv(path, index_col="Date", parse_dates=True)
    prices = prices.sort_index().dropna(how="any")
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    if returns.empty:
        raise RuntimeError("No usable return observation is found")
    return prices, returns

