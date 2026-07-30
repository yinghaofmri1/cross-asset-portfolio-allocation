"""Walk-forward portfolio backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    DEFENSIVE_ASSETS,
    EQUITY_ASSETS,
    STRATEGIES,
    TRADING_DAYS,
    BacktestConfig,
)
from .models import (
    black_litterman_posterior,
    inverse_volatility_weights,
    maximum_sharpe_weights,
    minimum_variance_weights,
    regularize_covariance,
    shrink_expected_returns,
)


SCENARIOS = {
    "Global Financial Crisis": ("2008-09-01", "2009-03-31"),
    "Pandemic Crash": ("2020-02-19", "2020-03-23"),
    "2022 Inflation Shock": ("2022-01-01", "2022-12-31"),
}


@dataclass
class BacktestResult:
    """Main tables returned by one backtest run."""

    strategy_returns: pd.DataFrame
    turnover: pd.DataFrame
    weights: pd.DataFrame
    rebalance_regimes: pd.DataFrame
    summary: pd.DataFrame
    regime_summary: pd.DataFrame
    scenario_summary: pd.DataFrame
    asset_scenario_summary: pd.DataFrame
    frontier: pd.DataFrame


def cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """Calculate robust cross-sectional z-score."""
    centered = values - np.median(values)
    scale = np.std(centered, ddof=1)
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(values)
    return np.clip(centered / scale, -2.5, 2.5)


def momentum_signal(price_history: pd.DataFrame) -> np.ndarray:
    """Use 12-1 month momentum and skip the latest month."""
    if len(price_history) < 253:
        return np.zeros(price_history.shape[1])
    momentum = (
        price_history.iloc[-22].to_numpy()
        / price_history.iloc[-253].to_numpy()
        - 1.0
    )
    return cross_sectional_zscore(momentum)


def classify_regime(return_history: pd.DataFrame) -> str:
    """Classify regime only with data known at rebalance time."""
    spy = return_history["SPY"]
    momentum_126 = float((1.0 + spy.iloc[-126:]).prod() - 1.0)
    volatility_63 = float(
        spy.iloc[-63:].std(ddof=1) * np.sqrt(TRADING_DAYS)
    )
    if momentum_126 < -0.05 or volatility_63 > 0.25:
        return "Stress"
    if momentum_126 > 0.08 and volatility_63 < 0.18:
        return "Risk-On"
    return "Defensive"


def regime_view_tilt(columns: list[str], regime: str) -> np.ndarray:
    """Convert one regime label to annual return views."""
    tilt = np.zeros(len(columns))
    for index, ticker in enumerate(columns):
        if regime == "Stress":
            if ticker in EQUITY_ASSETS:
                tilt[index] = -0.06
            elif ticker in DEFENSIVE_ASSETS:
                tilt[index] = 0.04
            elif ticker == "LQD":
                tilt[index] = 0.01
        elif regime == "Risk-On":
            if ticker in EQUITY_ASSETS:
                tilt[index] = 0.025
            elif ticker in {"TLT", "IEF"}:
                tilt[index] = -0.015
        else:
            if ticker in DEFENSIVE_ASSETS:
                tilt[index] = 0.02
            elif ticker in EQUITY_ASSETS:
                tilt[index] = -0.01
    return tilt


def build_rebalance_targets(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[
    dict[str, dict[pd.Timestamp, np.ndarray]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Build monthly targets without using future return data."""
    columns = list(returns.columns)
    n_assets = len(columns)
    config.validate(n_assets)
    if "SPY" not in columns:
        raise ValueError("SPY is required for benchmark and regime signal")

    month_ends = returns.groupby(returns.index.to_period("M")).tail(1).index
    rebalance_dates = [
        date
        for date in month_ends
        if returns.index.get_loc(date) >= config.lookback_days
        and returns.index.get_loc(date) < len(returns) - 1
    ]
    if not rebalance_dates:
        raise ValueError("Not enough data for the selected lookback window")

    targets = {name: {} for name in STRATEGIES}
    weight_records: list[dict[str, object]] = []
    regime_records: list[dict[str, object]] = []

    for rebalance_date in rebalance_dates:
        location = returns.index.get_loc(rebalance_date)
        effective_date = returns.index[location + 1]

        # The window ends at rebalance date. New weight starts next trading day.
        history = returns.iloc[
            location - config.lookback_days + 1 : location + 1
        ]
        covariance = regularize_covariance(history.to_numpy())
        expected_returns = shrink_expected_returns(history.to_numpy())
        inverse_volatility = inverse_volatility_weights(
            covariance,
            config.max_weight,
        )

        equal_weight = np.full(n_assets, 1.0 / n_assets)
        spy_weight = np.zeros(n_assets)
        spy_weight[columns.index("SPY")] = 1.0
        minimum_variance = minimum_variance_weights(
            covariance,
            config.max_weight,
        )
        maximum_sharpe = maximum_sharpe_weights(
            expected_returns,
            covariance,
            config.max_weight,
            config.risk_free_rate,
        )

        signal = momentum_signal(prices.loc[:rebalance_date])
        equilibrium_returns = 2.5 * covariance @ inverse_volatility
        momentum_views = equilibrium_returns + 0.06 * signal
        bl_returns, bl_covariance = black_litterman_posterior(
            covariance,
            inverse_volatility,
            momentum_views,
        )
        bl_weights = maximum_sharpe_weights(
            bl_returns,
            bl_covariance,
            config.max_weight,
            config.risk_free_rate,
        )

        regime = classify_regime(history)
        regime_views = (
            momentum_views + regime_view_tilt(columns, regime)
        )
        regime_returns, regime_covariance = black_litterman_posterior(
            covariance,
            inverse_volatility,
            regime_views,
        )
        regime_weights = maximum_sharpe_weights(
            regime_returns,
            regime_covariance,
            config.max_weight,
            config.risk_free_rate,
        )

        current_weights = {
            "SPY Buy & Hold": spy_weight,
            "Equal Weight": equal_weight,
            "Inverse Volatility": inverse_volatility,
            "MPT Minimum Variance": minimum_variance,
            "MPT Maximum Sharpe": maximum_sharpe,
            "Black-Litterman Momentum": bl_weights,
            "Regime-Aware Black-Litterman": regime_weights,
        }
        for strategy, weights in current_weights.items():
            targets[strategy][effective_date] = weights
            for ticker, weight in zip(columns, weights, strict=True):
                weight_records.append(
                    {
                        "rebalance_date": rebalance_date,
                        "effective_date": effective_date,
                        "strategy": strategy,
                        "ticker": ticker,
                        "weight": float(weight),
                    }
                )
        regime_records.append(
            {
                "rebalance_date": rebalance_date,
                "effective_date": effective_date,
                "regime": regime,
            }
        )
    return targets, weight_records, regime_records


def simulate_strategies(
    returns: pd.DataFrame,
    targets: dict[str, dict[pd.Timestamp, np.ndarray]],
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply target weights and include simple proportional cost."""
    first_effective = min(min(item) for item in targets.values())
    test_returns = returns.loc[first_effective:]
    result = pd.DataFrame(
        index=test_returns.index,
        columns=STRATEGIES,
        dtype=float,
    )
    turnover_records: list[dict[str, object]] = []

    for strategy in STRATEGIES:
        weights = np.zeros(test_returns.shape[1])
        for date, asset_return_row in test_returns.iterrows():
            transaction_cost = 0.0
            if date in targets[strategy]:
                target = targets[strategy][date]
                turnover = float(np.abs(target - weights).sum())
                transaction_cost = turnover * transaction_cost_bps / 10_000.0
                weights = target.copy()
                turnover_records.append(
                    {
                        "date": date,
                        "strategy": strategy,
                        "turnover": turnover,
                        "transaction_cost": transaction_cost,
                    }
                )

            asset_returns = asset_return_row.to_numpy(dtype=float)
            gross_return = float(weights @ asset_returns)
            result.loc[date, strategy] = (
                (1.0 - transaction_cost) * (1.0 + gross_return) - 1.0
            )

            # Let weights drift with market return until next rebalance.
            denominator = 1.0 + gross_return
            if denominator > 0.0:
                weights = weights * (1.0 + asset_returns) / denominator

    return result, pd.DataFrame(turnover_records)


def performance_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Calculate common annualized portfolio metrics."""
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        raise ValueError("daily_returns cannot be empty")

    years = len(daily_returns) / TRADING_DAYS
    nav = (1.0 + daily_returns).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    cagr = float(nav.iloc[-1] ** (1.0 / years) - 1.0)
    volatility = float(
        daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )
    annual_return = float(daily_returns.mean() * TRADING_DAYS)
    sharpe = (
        (annual_return - risk_free_rate) / volatility
        if volatility > 0.0
        else np.nan
    )
    downside = daily_returns[daily_returns < 0.0]
    downside_volatility = float(
        np.sqrt(np.mean(np.square(downside))) * np.sqrt(TRADING_DAYS)
    )
    sortino = (
        (annual_return - risk_free_rate) / downside_volatility
        if downside_volatility > 0.0
        else np.nan
    )
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0.0 else np.nan
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
    }


def daily_regime_labels(returns: pd.DataFrame) -> pd.Series:
    """Create daily labels and lag them by one trading day."""
    spy = returns["SPY"]
    momentum_126 = (
        (1.0 + spy).rolling(126).apply(np.prod, raw=True) - 1.0
    )
    volatility_63 = (
        spy.rolling(63).std(ddof=1) * np.sqrt(TRADING_DAYS)
    )
    labels = pd.Series("Defensive", index=returns.index, dtype="object")
    labels[(momentum_126 < -0.05) | (volatility_63 > 0.25)] = "Stress"
    labels[(momentum_126 > 0.08) & (volatility_63 < 0.18)] = "Risk-On"
    return labels.shift(1).fillna("Defensive")


def build_summary(
    strategy_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Compare return, risk, drawdown and turnover."""
    years = len(strategy_returns) / TRADING_DAYS
    rows = []
    for strategy in STRATEGIES:
        row = {
            "strategy": strategy,
            **performance_metrics(
                strategy_returns[strategy],
                config.risk_free_rate,
            ),
        }
        mask = turnover["strategy"] == strategy
        row["annual_turnover"] = float(
            turnover.loc[mask, "turnover"].sum() / years
        )
        row["estimated_total_cost"] = float(
            turnover.loc[mask, "transaction_cost"].sum()
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def build_regime_summary(
    strategy_returns: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """Calculate strategy results under each price regime."""
    rows = []
    for regime in ("Risk-On", "Defensive", "Stress"):
        mask = labels.reindex(strategy_returns.index) == regime
        for strategy in STRATEGIES:
            subset = strategy_returns.loc[mask, strategy]
            annual_return = float(subset.mean() * TRADING_DAYS)
            annual_volatility = float(
                subset.std(ddof=1) * np.sqrt(TRADING_DAYS)
            )
            rows.append(
                {
                    "regime": regime,
                    "strategy": strategy,
                    "observations": len(subset),
                    "annual_return": annual_return,
                    "annual_volatility": annual_volatility,
                    "sharpe": (
                        annual_return / annual_volatility
                        if annual_volatility > 0.0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_scenario_summary(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate strategy return in selected market shocks."""
    rows = []
    for scenario, (start, end) in SCENARIOS.items():
        period = strategy_returns.loc[start:end]
        if period.empty:
            continue
        for strategy in STRATEGIES:
            values = period[strategy].dropna()
            nav = (1.0 + values).cumprod()
            drawdown = nav / nav.cummax() - 1.0
            rows.append(
                {
                    "scenario": scenario,
                    "start": start,
                    "end": end,
                    "strategy": strategy,
                    "period_return": float(nav.iloc[-1] - 1.0),
                    "max_drawdown": float(drawdown.min()),
                    "annual_volatility": float(
                        values.std(ddof=1) * np.sqrt(TRADING_DAYS)
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_asset_scenario_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate each ETF result in selected market shocks."""
    rows = []
    for scenario, (start, end) in SCENARIOS.items():
        period = returns.loc[start:end]
        if period.empty:
            continue
        for ticker in returns.columns:
            values = period[ticker].dropna()
            nav = (1.0 + values).cumprod()
            drawdown = nav / nav.cummax() - 1.0
            rows.append(
                {
                    "scenario": scenario,
                    "start": start,
                    "end": end,
                    "ticker": ticker,
                    "period_return": float(nav.iloc[-1] - 1.0),
                    "max_drawdown": float(drawdown.min()),
                    "annual_volatility": float(
                        values.std(ddof=1) * np.sqrt(TRADING_DAYS)
                    ),
                }
            )
    return pd.DataFrame(rows)


def simulate_frontier(
    returns: pd.DataFrame,
    config: BacktestConfig,
    n_portfolios: int = 5_000,
) -> pd.DataFrame:
    """Simulate the latest feasible risk-return set."""
    history = returns.tail(config.lookback_days)
    covariance = regularize_covariance(history.to_numpy())
    expected_returns = shrink_expected_returns(history.to_numpy())
    generator = np.random.default_rng(config.random_seed)
    accepted = []
    accepted_count = 0

    while accepted_count < n_portfolios:
        batch = generator.dirichlet(
            np.ones(history.shape[1]),
            size=8_000,
        )
        valid = batch[np.max(batch, axis=1) <= config.max_weight]
        accepted.append(valid)
        accepted_count += valid.shape[0]

    weights = np.vstack(accepted)[:n_portfolios]
    annual_return = weights @ expected_returns
    annual_variance = np.einsum(
        "ij,jk,ik->i",
        weights,
        covariance,
        weights,
    )
    annual_volatility = np.sqrt(np.maximum(annual_variance, 0.0))
    return pd.DataFrame(
        {
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe": (
                annual_return - config.risk_free_rate
            ) / annual_volatility,
        }
    )


def run_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: BacktestConfig,
    n_frontier_portfolios: int = 5_000,
) -> BacktestResult:
    """Run all strategies and return clean result tables."""
    targets, weight_records, regime_records = build_rebalance_targets(
        prices,
        returns,
        config,
    )
    strategy_returns, turnover = simulate_strategies(
        returns,
        targets,
        config.transaction_cost_bps,
    )
    labels = daily_regime_labels(returns).reindex(strategy_returns.index)
    weights = pd.DataFrame(weight_records)

    return BacktestResult(
        strategy_returns=strategy_returns,
        turnover=turnover,
        weights=weights,
        rebalance_regimes=pd.DataFrame(regime_records),
        summary=build_summary(strategy_returns, turnover, config),
        regime_summary=build_regime_summary(strategy_returns, labels),
        scenario_summary=build_scenario_summary(strategy_returns),
        asset_scenario_summary=build_asset_scenario_summary(returns),
        frontier=simulate_frontier(
            returns,
            config,
            n_frontier_portfolios,
        ),
    )

