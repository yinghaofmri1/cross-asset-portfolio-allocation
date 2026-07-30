# Methodology

## 1. Data and return

The project downloads daily ETF prices with `auto_adjust=True`. Daily simple
return is:

```text
r[t] = P[t] / P[t-1] - 1
```

Rows with missing values are removed so every strategy uses the same asset
universe on the same day.

## 2. Walk-forward design

At every month-end:

1. take the previous 756 trading days;
2. estimate expected return and covariance;
3. calculate target weights;
4. start the new weights from the next trading day;
5. hold and let weights drift until the next rebalance.

This design gives many rolling train/test periods. For example, the first
training window ends on 2010-01-29 and its weight starts on 2010-02-01. No future
return is included when that weight is calculated.

## 3. Covariance shrinkage

The sample covariance can have unstable correlations. The project shrinks it
toward its diagonal:

```text
Sigma_shrunk = 0.85 * Sigma_sample + 0.15 * diag(Sigma_sample)
```

The asset variances are kept, while off-diagonal covariances are reduced by
15%. A very small ridge is also added to the diagonal for numerical stability.

This is a simple shrinkage method. Ledoit-Wolf or nonlinear shrinkage can be
added in a later version.

## 4. Expected return shrinkage

Historical average return has high estimation error. The project uses:

```text
mu_shrunk[i] = 0.35 * mu_sample[i] + 0.65 * average(mu_sample)
```

Each ETF mean is moved toward the cross-asset average. The rank information is
still kept, but extreme estimates become smaller.

## 5. MPT

Minimum variance solves:

```text
minimize    w' Sigma w
subject to  sum(w) = 1
            0 <= w[i] <= 0.30
```

Maximum Sharpe compares several quadratic-utility solutions:

```text
maximize    mu' w - gamma / 2 * w' Sigma w
```

Different risk-aversion values produce different portfolios. The candidate with
the highest predicted Sharpe ratio is selected.

CVXPY is used when it is installed. Otherwise, a NumPy projected-gradient
solver is used.

## 6. Black-Litterman

The neutral prior is inverse-volatility weight. Equilibrium return is:

```text
pi = delta * Sigma * w_prior
```

The momentum view uses 12-1 month momentum. The latest 21 trading days are
excluded:

```text
momentum = P[t-21] / P[t-252] - 1
```

The signal is converted to a cross-sectional z-score and added to equilibrium
return. Black-Litterman combines the prior and view according to view
uncertainty.

The regime-aware version adds another return tilt:

- Risk-On: positive equity view and negative Treasury view;
- Defensive: positive defensive-asset view and negative equity view;
- Stress: stronger negative equity view and positive Treasury/gold view.

The regime rule uses 126-day SPY momentum and 63-day annualized volatility. It
does not use future information.

## 7. Transaction cost

One-way turnover at rebalance date is:

```text
turnover = sum(abs(target_weight - current_weight))
```

Daily transaction cost on that date is:

```text
cost = turnover * 10 / 10000
```

The model lets weights drift between rebalances, so turnover is calculated
against the actual pre-trade portfolio, not the previous target only.

## 8. Performance metrics

The report includes:

- CAGR;
- annualized mean return;
- annualized volatility;
- Sharpe ratio;
- Sortino ratio;
- maximum drawdown;
- Calmar ratio;
- annual turnover;
- estimated total transaction cost.

The current Sharpe ratio uses a 0% annual risk-free rate. A better next version
can use point-in-time FRED DGS3MO data.

## 9. Limitations and next steps

The current project is a transparent research baseline. Important next steps
are:

1. add turnover penalty directly in optimization;
2. calibrate Black-Litterman view confidence by rolling validation;
3. replace the fixed risk-free rate with point-in-time Treasury data;
4. add VIX, inflation and yield-curve regimes with release-date control;
5. compare against risk parity, HRP and robust optimization;
6. run block bootstrap and parameter-sensitivity tests;
7. test on a second data vendor.

## References

- Markowitz, H. (1952), *Portfolio Selection*, Journal of Finance.
- Black, F. and Litterman, R. (1992), *Global Portfolio Optimization*,
  Financial Analysts Journal.
- Ledoit, O. and Wolf, M. (2004), *Honey, I Shrunk the Sample Covariance
  Matrix*, Journal of Portfolio Management.
- DeMiguel, V., Garlappi, L. and Uppal, R. (2009), *Optimal Versus Naive
  Diversification*, Review of Financial Studies.

