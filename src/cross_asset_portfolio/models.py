"""Portfolio optimization models."""

from __future__ import annotations

import numpy as np

from .config import TRADING_DAYS

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - depends on local environment
    cp = None


def solver_name() -> str:
    """Return the optimizer name used in this environment."""
    return "cvxpy" if cp is not None else "numpy_projected_gradient"


def regularize_covariance(
    returns: np.ndarray,
    annualization: int = TRADING_DAYS,
    diagonal_shrinkage: float = 0.15,
) -> np.ndarray:
    """Estimate annual covariance with diagonal shrinkage."""
    sample = np.cov(returns, rowvar=False, ddof=1) * annualization
    diagonal = np.diag(np.diag(sample))

    # Keep most sample covariance, but reduce unstable correlations.
    covariance = (
        (1.0 - diagonal_shrinkage) * sample
        + diagonal_shrinkage * diagonal
    )
    covariance = 0.5 * (covariance + covariance.T)

    # A very small ridge can avoid numerical issue in matrix inversion.
    ridge = max(float(np.trace(covariance)) / covariance.shape[0], 1e-8) * 1e-6
    return covariance + np.eye(covariance.shape[0]) * ridge


def shrink_expected_returns(
    returns: np.ndarray,
    annualization: int = TRADING_DAYS,
    shrinkage: float = 0.65,
) -> np.ndarray:
    """Shrink asset means toward the average mean."""
    raw = np.mean(returns, axis=0) * annualization
    grand_mean = float(np.mean(raw))

    # Historical means are noisy, so we do not use 100% raw value.
    return (1.0 - shrinkage) * raw + shrinkage * grand_mean


def project_capped_simplex(values: np.ndarray, upper_bound: float) -> np.ndarray:
    """Project values to long-only weights with one upper bound."""
    values = np.asarray(values, dtype=float)
    n_assets = values.size
    if upper_bound * n_assets < 1.0 - 1e-12:
        raise ValueError("upper_bound is too small for the asset number")

    lower_theta = float(np.min(values - upper_bound))
    upper_theta = float(np.max(values))
    for _ in range(40):
        theta = 0.5 * (lower_theta + upper_theta)
        projected = np.clip(values - theta, 0.0, upper_bound)
        if projected.sum() > 1.0:
            lower_theta = theta
        else:
            upper_theta = theta

    weights = np.clip(
        values - 0.5 * (lower_theta + upper_theta),
        0.0,
        upper_bound,
    )
    weights /= weights.sum()
    return weights


def inverse_volatility_weights(
    covariance: np.ndarray,
    upper_bound: float,
) -> np.ndarray:
    """Allocate more weight to assets with lower volatility."""
    volatility = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    raw = 1.0 / volatility
    raw /= raw.sum()
    return project_capped_simplex(raw, upper_bound)


def _numpy_quadratic_utility(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
    upper_bound: float,
) -> np.ndarray:
    """Solve quadratic utility by projected gradient."""
    n_assets = expected_returns.size
    weights = project_capped_simplex(
        np.full(n_assets, 1.0 / n_assets),
        upper_bound,
    )
    largest_eigenvalue = max(
        float(np.linalg.eigvalsh(covariance).max()),
        1e-10,
    )
    step = 0.95 / (risk_aversion * largest_eigenvalue + 1e-10)

    for _ in range(60):
        gradient = risk_aversion * covariance @ weights - expected_returns
        candidate = project_capped_simplex(
            weights - step * gradient,
            upper_bound,
        )
        if np.max(np.abs(candidate - weights)) < 1e-8:
            weights = candidate
            break
        weights = candidate
    return weights


def quadratic_utility_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
    upper_bound: float,
) -> np.ndarray:
    """Maximize return minus one covariance risk penalty."""
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    if cp is None:
        return _numpy_quadratic_utility(
            expected_returns,
            covariance,
            risk_aversion,
            upper_bound,
        )

    weights = cp.Variable(expected_returns.size)
    objective = cp.Maximize(
        expected_returns @ weights
        - 0.5 * risk_aversion * cp.quad_form(weights, covariance)
    )
    constraints = [
        cp.sum(weights) == 1.0,
        weights >= 0.0,
        weights <= upper_bound,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(warm_start=True)
    if weights.value is None:
        return _numpy_quadratic_utility(
            expected_returns,
            covariance,
            risk_aversion,
            upper_bound,
        )
    return project_capped_simplex(
        np.asarray(weights.value).ravel(),
        upper_bound,
    )


def minimum_variance_weights(
    covariance: np.ndarray,
    upper_bound: float,
) -> np.ndarray:
    """Find the long-only minimum variance portfolio."""
    zeros = np.zeros(covariance.shape[0])
    return quadratic_utility_weights(zeros, covariance, 1.0, upper_bound)


def maximum_sharpe_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    upper_bound: float,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """Select the highest Sharpe candidate from utility solutions."""
    excess_returns = np.asarray(expected_returns, dtype=float) - risk_free_rate
    candidates = [
        inverse_volatility_weights(covariance, upper_bound),
        minimum_variance_weights(covariance, upper_bound),
    ]
    for risk_aversion in np.logspace(-1.5, 3.5, 7):
        candidates.append(
            quadratic_utility_weights(
                excess_returns,
                covariance,
                risk_aversion,
                upper_bound,
            )
        )

    def predicted_sharpe(weights: np.ndarray) -> float:
        variance = float(weights @ covariance @ weights)
        if variance <= 0.0:
            return -np.inf
        return float(weights @ excess_returns) / np.sqrt(variance)

    return max(candidates, key=predicted_sharpe)


def black_litterman_posterior(
    covariance: np.ndarray,
    prior_weights: np.ndarray,
    view_returns: np.ndarray,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    view_confidence: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine equilibrium returns and investor views."""
    n_assets = covariance.shape[0]
    pick_matrix = np.eye(n_assets)
    equilibrium_returns = risk_aversion * covariance @ prior_weights

    tau_covariance = tau * covariance
    confidence = float(np.clip(view_confidence, 0.05, 0.95))
    omega_scale = (1.0 - confidence) / confidence
    omega = np.diag(
        np.diag(pick_matrix @ tau_covariance @ pick_matrix.T)
    )
    omega = np.maximum(
        omega * omega_scale,
        np.eye(n_assets) * 1e-10,
    )

    inverse_tau = np.linalg.pinv(tau_covariance)
    inverse_omega = np.linalg.pinv(omega)
    posterior_precision = (
        inverse_tau + pick_matrix.T @ inverse_omega @ pick_matrix
    )
    posterior_mean_covariance = np.linalg.pinv(posterior_precision)
    posterior_returns = posterior_mean_covariance @ (
        inverse_tau @ equilibrium_returns
        + pick_matrix.T @ inverse_omega @ view_returns
    )
    posterior_covariance = covariance + posterior_mean_covariance
    return posterior_returns, posterior_covariance

