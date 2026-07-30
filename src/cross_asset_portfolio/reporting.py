"""Create CSV, SVG and HTML backtest reports."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .config import STRATEGIES, BacktestConfig
from .models import solver_name


COLORS = [
    "#2563eb",
    "#ef4444",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#64748b",
]


def line_chart_svg(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    percent_axis: bool = False,
) -> None:
    """Write one simple multi-line chart as SVG."""
    width, height = 1200, 700
    left, right, top, bottom = 90, 30, 70, 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if len(clean) > 800:
        positions = np.linspace(0, len(clean) - 1, 800).astype(int)
        clean = clean.iloc[np.unique(positions)]
    y_min = float(np.nanmin(clean.to_numpy()))
    y_max = float(np.nanmax(clean.to_numpy()))
    padding = max((y_max - y_min) * 0.05, 1e-6)
    y_min -= padding
    y_max += padding

    def x_coordinate(index: int) -> float:
        return left + plot_width * index / max(len(clean) - 1, 1)

    def y_coordinate(value: float) -> float:
        return top + plot_height * (y_max - value) / (y_max - y_min)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="35" font-family="Arial" font-size="24" font-weight="700">{html.escape(title)}</text>',
    ]
    for grid_index in range(6):
        value = y_min + (y_max - y_min) * grid_index / 5
        y = y_coordinate(value)
        label = f"{value * 100:.0f}%" if percent_axis else f"{value:.2f}"
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e2e8f0"/>'
        )
        svg.append(
            f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#475569">{label}</text>'
        )

    for series_index, column in enumerate(clean.columns):
        points = []
        for index, value in enumerate(clean[column].to_numpy()):
            if np.isfinite(value):
                points.append(f"{x_coordinate(index):.1f},{y_coordinate(float(value)):.1f}")
        svg.append(
            f'<polyline fill="none" stroke="{COLORS[series_index % len(COLORS)]}" stroke-width="2" points="{" ".join(points)}"/>'
        )

    for tick_index in range(6):
        position = int((len(clean) - 1) * tick_index / 5)
        x = x_coordinate(position)
        label = pd.Timestamp(clean.index[position]).strftime("%Y-%m")
        svg.append(
            f'<text x="{x:.1f}" y="{height-bottom+30}" text-anchor="middle" font-family="Arial" font-size="13" fill="#475569">{label}</text>'
        )

    for series_index, column in enumerate(clean.columns):
        x = left + series_index * 155
        svg.append(
            f'<line x1="{x}" y1="{height-25}" x2="{x+22}" y2="{height-25}" stroke="{COLORS[series_index % len(COLORS)]}" stroke-width="4"/>'
        )
        svg.append(
            f'<text x="{x+28}" y="{height-20}" font-family="Arial" font-size="12">{html.escape(str(column))}</text>'
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def frontier_chart_svg(frontier: pd.DataFrame, path: Path) -> None:
    """Write one simulated frontier scatter chart as SVG."""
    width, height = 1000, 700
    left, right, top, bottom = 90, 50, 70, 80
    x = frontier["annual_volatility"].to_numpy()
    y = frontier["annual_return"].to_numpy()
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    def sx(value: float) -> float:
        return left + (width - left - right) * (value - x_min) / (x_max - x_min)

    def sy(value: float) -> float:
        return top + (height - top - bottom) * (y_max - value) / (y_max - y_min)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="35" font-family="Arial" font-size="24" font-weight="700">Latest 3-Year Feasible Portfolios</text>',
    ]
    sharpe_min = float(frontier["sharpe"].min())
    sharpe_range = float(frontier["sharpe"].max() - sharpe_min + 1e-12)
    for row in frontier.iloc[::3].itertuples():
        normalized = (row.sharpe - sharpe_min) / sharpe_range
        red = int(230 * (1.0 - normalized) + 30 * normalized)
        green = int(90 * (1.0 - normalized) + 160 * normalized)
        blue = int(80 * (1.0 - normalized) + 220 * normalized)
        svg.append(
            f'<circle cx="{sx(row.annual_volatility):.1f}" cy="{sy(row.annual_return):.1f}" r="2.2" fill="rgb({red},{green},{blue})" opacity="0.55"/>'
        )
    svg.extend(
        [
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#334155"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#334155"/>',
            f'<text x="{(left+width-right)/2}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="15">Annualized Volatility</text>',
            f'<text x="24" y="{(top+height-bottom)/2}" transform="rotate(-90 24 {(top+height-bottom)/2})" text-anchor="middle" font-family="Arial" font-size="15">Expected Annual Return</text>',
        ]
    )
    for tick in range(6):
        x_value = x_min + (x_max - x_min) * tick / 5
        y_value = y_min + (y_max - y_min) * tick / 5
        svg.append(
            f'<text x="{sx(x_value):.1f}" y="{height-bottom+25}" text-anchor="middle" font-family="Arial" font-size="12">{x_value:.0%}</text>'
        )
        svg.append(
            f'<text x="{left-10}" y="{sy(y_value)+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y_value:.0%}</text>'
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def format_table(frame: pd.DataFrame, percent_columns: set[str]) -> str:
    """Format a DataFrame before putting it in HTML."""
    formatted = frame.copy()
    for column in formatted.columns:
        if column in percent_columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
        elif pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    return formatted.to_html(index=False, border=0, classes="data-table")


def write_html_report(
    output_dir: Path,
    summary: pd.DataFrame,
    latest_weights: pd.DataFrame,
    regime_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    solver: str,
    lookback_days: int,
    max_weight: float,
    transaction_cost_bps: float,
) -> None:
    """Write the main self-contained HTML report."""
    summary_table = format_table(
        summary,
        {
            "total_return",
            "cagr",
            "annual_return",
            "annual_volatility",
            "max_drawdown",
            "annual_turnover",
            "estimated_total_cost",
        },
    )
    weights_table = format_table(
        latest_weights.reset_index().rename(columns={"index": "strategy"}),
        set(latest_weights.columns),
    )
    regime_pivot = regime_summary.pivot(
        index="strategy", columns="regime", values="sharpe"
    ).reset_index()
    scenario_pivot = scenario_summary.pivot(
        index="strategy", columns="scenario", values="period_return"
    ).reset_index()

    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cross-Asset Portfolio Allocation Report</title>
<style>
body{{font-family:Inter,Arial,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#172033;background:#f8fafc}}
h1,h2{{color:#0f172a}} .card{{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:24px;margin:20px 0;box-shadow:0 4px 18px rgba(15,23,42,.05)}}
.data-table{{border-collapse:collapse;width:100%;font-size:13px}} .data-table th,.data-table td{{padding:9px;border-bottom:1px solid #e2e8f0;text-align:right}}
.data-table th:first-child,.data-table td:first-child{{text-align:left}} img{{width:100%;height:auto}} code{{background:#e2e8f0;padding:2px 5px;border-radius:4px}}
.note{{color:#475569;line-height:1.6}}
</style></head><body>
<h1>Cross-Asset MPT & Black–Litterman</h1>
<p class="note">Walk-forward test from {start_date.date()} to {end_date.date()}. Monthly rebalancing, {lookback_days} trading-day lookback, long-only, {max_weight:.0%} maximum asset weight, and {transaction_cost_bps:.0f} bps transaction cost. Solver: <code>{solver}</code>.</p>
<div class="card"><h2>Strategy comparison</h2>{summary_table}</div>
<div class="card"><h2>Cumulative growth</h2><img src="cumulative_growth.svg"></div>
<div class="card"><h2>Drawdowns</h2><img src="drawdowns.svg"></div>
<div class="card"><h2>Latest allocations</h2>{weights_table}</div>
<div class="card"><h2>Sharpe by price regime</h2>{format_table(regime_pivot, set())}</div>
<div class="card"><h2>Historical scenario returns</h2>{format_table(scenario_pivot, set(scenario_pivot.columns)-{"strategy"})}</div>
<div class="card"><h2>Feasible risk-return set</h2><img src="efficient_frontier.svg"></div>
<div class="card"><h2>Methodology notes</h2>
<p class="note">MPT uses shrinkage estimates and constrained monthly optimization. Black–Litterman starts from an inverse-volatility neutral prior and adds 12–1 month momentum views. The regime-aware version adds price-only Risk-On, Defensive, or Stress tilts. Signals observed at month-end become effective on the next trading day. This initial version assumes a zero risk-free rate; FRED DGS3MO should replace it later.</p>
</div></body></html>"""
    (output_dir / "report.html").write_text(report, encoding="utf-8")


def save_backtest_result(
    result: BacktestResult,
    output_dir: Path,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Save all result tables and return the latest weight table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_date = result.weights["rebalance_date"].max()
    latest_weights = (
        result.weights.loc[result.weights["rebalance_date"] == latest_date]
        .pivot(index="strategy", columns="ticker", values="weight")
        .reindex(STRATEGIES)
    )
    nav = (1.0 + result.strategy_returns).cumprod()
    drawdowns = nav / nav.cummax() - 1.0

    result.summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    result.strategy_returns.to_csv(output_dir / "portfolio_daily_returns.csv")
    nav.to_csv(output_dir / "portfolio_nav.csv")
    result.weights.to_csv(output_dir / "monthly_weights.csv", index=False)
    latest_weights.to_csv(output_dir / "latest_weights.csv")
    result.rebalance_regimes.to_csv(
        output_dir / "rebalance_regimes.csv",
        index=False,
    )
    result.regime_summary.to_csv(
        output_dir / "regime_summary.csv",
        index=False,
    )
    result.scenario_summary.to_csv(
        output_dir / "scenario_summary.csv",
        index=False,
    )
    result.asset_scenario_summary.to_csv(
        output_dir / "asset_scenario_summary.csv",
        index=False,
    )
    result.turnover.to_csv(output_dir / "turnover.csv", index=False)
    result.frontier.to_csv(
        output_dir / "efficient_frontier.csv",
        index=False,
    )

    line_chart_svg(nav, output_dir / "cumulative_growth.svg", "Growth of $1")
    line_chart_svg(
        drawdowns,
        output_dir / "drawdowns.svg",
        "Portfolio Drawdowns",
        percent_axis=True,
    )
    frontier_chart_svg(
        result.frontier,
        output_dir / "efficient_frontier.svg",
    )
    write_html_report(
        output_dir,
        result.summary,
        latest_weights,
        result.regime_summary,
        result.scenario_summary,
        result.strategy_returns.index.min(),
        result.strategy_returns.index.max(),
        solver_name(),
        config.lookback_days,
        config.max_weight,
        config.transaction_cost_bps,
    )

    metadata = {
        "solver": solver_name(),
        "lookback_days": config.lookback_days,
        "maximum_weight": config.max_weight,
        "transaction_cost_bps": config.transaction_cost_bps,
        "risk_free_rate": config.risk_free_rate,
        "backtest_start": (
            result.strategy_returns.index.min().date().isoformat()
        ),
        "backtest_end": (
            result.strategy_returns.index.max().date().isoformat()
        ),
        "strategies": list(STRATEGIES),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return latest_weights
