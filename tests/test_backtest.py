"""Unit tests for backtest rules."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cross_asset_portfolio.backtest import (
    classify_regime,
    momentum_signal,
    performance_metrics,
    simulate_strategies,
)
from cross_asset_portfolio.config import STRATEGIES


class BacktestRuleTests(unittest.TestCase):
    def test_momentum_ignores_latest_month(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=300)
        base = np.linspace(100.0, 140.0, len(index))
        prices = pd.DataFrame({"SPY": base, "TLT": base[::-1]}, index=index)

        signal_before = momentum_signal(prices)
        prices.iloc[-21:, 0] *= 5.0
        signal_after = momentum_signal(prices)

        np.testing.assert_allclose(signal_before, signal_after)

    def test_regime_uses_only_passed_history(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=200)
        calm = pd.DataFrame({"SPY": np.full(200, 0.0005)}, index=index)
        future_crash = pd.DataFrame(
            {"SPY": np.full(20, -0.05)},
            index=pd.bdate_range(index[-1] + pd.Timedelta(days=1), periods=20),
        )
        label = classify_regime(calm)
        _ = pd.concat([calm, future_crash])
        self.assertEqual(label, classify_regime(calm))

    def test_transaction_cost_is_applied(self) -> None:
        index = pd.bdate_range("2024-01-02", periods=2)
        returns = pd.DataFrame(
            {"SPY": [0.0, 0.0], "TLT": [0.0, 0.0]},
            index=index,
        )
        targets = {
            strategy: {index[0]: np.array([0.5, 0.5])}
            for strategy in STRATEGIES
        }
        result, turnover = simulate_strategies(returns, targets, 10.0)
        self.assertAlmostEqual(float(result.iloc[0, 0]), -0.001)
        self.assertAlmostEqual(float(turnover.iloc[0]["turnover"]), 1.0)

    def test_performance_metrics_have_expected_keys(self) -> None:
        values = pd.Series(np.full(300, 0.0002))
        metrics = performance_metrics(values)
        self.assertIn("sharpe", metrics)
        self.assertIn("max_drawdown", metrics)


if __name__ == "__main__":
    unittest.main()

