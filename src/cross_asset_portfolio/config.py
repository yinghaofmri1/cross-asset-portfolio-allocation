"""Project configuration."""

from __future__ import annotations

from dataclasses import dataclass


TRADING_DAYS = 252

TICKERS = (
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "LQD",
    "GLD",
    "DBC",
    "VNQ",
)

EQUITY_ASSETS = frozenset({"SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ"})
DEFENSIVE_ASSETS = frozenset({"TLT", "IEF", "GLD"})

STRATEGIES = (
    "SPY Buy & Hold",
    "Equal Weight",
    "Inverse Volatility",
    "MPT Minimum Variance",
    "MPT Maximum Sharpe",
    "Black-Litterman Momentum",
    "Regime-Aware Black-Litterman",
)


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters used by the walk-forward backtest."""

    lookback_days: int = 756
    max_weight: float = 0.30
    transaction_cost_bps: float = 10.0
    risk_free_rate: float = 0.0
    random_seed: int = 42

    def validate(self, n_assets: int) -> None:
        if self.lookback_days < 253:
            raise ValueError("lookback_days should be at least 253")
        if self.max_weight <= 0.0 or self.max_weight > 1.0:
            raise ValueError("max_weight should be in (0, 1]")
        if self.max_weight * n_assets < 1.0 - 1e-12:
            raise ValueError("max_weight is too small for this asset number")
        if self.transaction_cost_bps < 0.0:
            raise ValueError("transaction_cost_bps cannot be negative")

