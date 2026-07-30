"""Unit tests for portfolio models."""

from __future__ import annotations

import unittest

import numpy as np

from cross_asset_portfolio.models import (
    black_litterman_posterior,
    maximum_sharpe_weights,
    minimum_variance_weights,
    project_capped_simplex,
    regularize_covariance,
    shrink_expected_returns,
)


class PortfolioModelTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(7)
        self.returns = generator.normal(0.0003, 0.012, size=(800, 6))
        self.covariance = regularize_covariance(self.returns)

    def assert_valid_weights(self, weights: np.ndarray, cap: float) -> None:
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertGreaterEqual(float(weights.min()), -1e-10)
        self.assertLessEqual(float(weights.max()), cap + 1e-8)
        self.assertTrue(np.isfinite(weights).all())

    def test_capped_simplex_projection(self) -> None:
        values = np.array([2.0, -1.0, 0.4, 0.2, 0.1, 0.0])
        weights = project_capped_simplex(values, 0.30)
        self.assert_valid_weights(weights, 0.30)

    def test_minimum_variance_weights(self) -> None:
        weights = minimum_variance_weights(self.covariance, 0.30)
        self.assert_valid_weights(weights, 0.30)

    def test_maximum_sharpe_weights(self) -> None:
        expected = np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03])
        weights = maximum_sharpe_weights(
            expected,
            self.covariance,
            0.30,
        )
        self.assert_valid_weights(weights, 0.30)

    def test_black_litterman_dimensions(self) -> None:
        prior = np.full(6, 1.0 / 6)
        views = np.array([0.08, 0.07, 0.06, 0.04, 0.03, 0.02])
        posterior, posterior_covariance = black_litterman_posterior(
            self.covariance,
            prior,
            views,
        )
        self.assertEqual(posterior.shape, (6,))
        self.assertEqual(posterior_covariance.shape, (6, 6))
        self.assertTrue(np.isfinite(posterior).all())
        self.assertGreater(
            float(np.linalg.eigvalsh(posterior_covariance).min()),
            0.0,
        )

    def test_expected_return_shrinkage(self) -> None:
        raw = self.returns.mean(axis=0) * 252
        shrunk = shrink_expected_returns(self.returns, shrinkage=0.65)
        raw_dispersion = float(np.std(raw))
        shrunk_dispersion = float(np.std(shrunk))
        self.assertLess(shrunk_dispersion, raw_dispersion)


if __name__ == "__main__":
    unittest.main()

