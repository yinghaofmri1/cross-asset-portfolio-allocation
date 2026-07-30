"""Command line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .backtest import run_backtest
from .config import BacktestConfig
from .data import download_yahoo_data, load_adjusted_close
from .models import solver_name
from .reporting import save_backtest_result


def _download_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download cross-asset ETF data from Yahoo Finance.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Local output directory. Default: data",
    )
    parser.add_argument(
        "--start",
        default="2007-01-01",
        help="Download start date. Default: 2007-01-01",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Exclusive end date. Default: tomorrow",
    )
    return parser


def _backtest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run walk-forward MPT and Black-Litterman backtest.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/etf_adjusted_close.csv"),
        help="Adjusted close CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Backtest output directory.",
    )
    parser.add_argument("--lookback-days", type=int, default=756)
    parser.add_argument("--max-weight", type=float, default=0.30)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--frontier-portfolios", type=int, default=5_000)
    return parser


def download_main(argv: Sequence[str] | None = None) -> None:
    """Run the Yahoo Finance downloader."""
    args = _download_parser().parse_args(argv)
    paths = download_yahoo_data(
        data_dir=args.data_dir,
        start_date=args.start,
        end_date_exclusive=args.end,
    )
    for name, path in paths.items():
        print(f"{name:>14}: {path.resolve()}")


def backtest_main(argv: Sequence[str] | None = None) -> None:
    """Run analysis and save all report files."""
    args = _backtest_parser().parse_args(argv)
    config = BacktestConfig(
        lookback_days=args.lookback_days,
        max_weight=args.max_weight,
        transaction_cost_bps=args.transaction_cost_bps,
        risk_free_rate=args.risk_free_rate,
    )
    prices, returns = load_adjusted_close(args.data_path)
    result = run_backtest(
        prices,
        returns,
        config,
        n_frontier_portfolios=args.frontier_portfolios,
    )
    save_backtest_result(result, args.output_dir, config)

    columns = [
        "strategy",
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "annual_turnover",
    ]
    print(result.summary[columns].to_string(index=False))
    print(f"\nSolver: {solver_name()}")
    print(f"Results: {args.output_dir.resolve()}")

