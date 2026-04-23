"""
Price regime classifier for ADP shadow-cost conditioning (Note 4 A3.2).

The full-ADP backward-induction DP (A3.3) conditions its expected
next-day value on the *regime* the market is in — different regimes have
different expected arbitrage opportunities and therefore different
shadow costs. This module turns a raw day-ahead price series into a
categorical regime label per day, plus the empirical transition matrix
and per-regime stats the DP needs.

Regime assignment is percentile-based on the **intraday spread** (p95-p5
of within-day prices). Three default regimes:
  * 0 = "quiet"   — bottom tercile of daily spreads (calm market)
  * 1 = "normal"  — mid tercile
  * 2 = "volatile" — top tercile (storm / scarcity / DR events)

Percentile-based is deliberate: no ML dependency, reproducible across
runs, easy to interpret, and aligned with how BESS traders actually
bucket days in practice. More regimes (k=5, k=7) can be passed via the
``n_regimes`` parameter for ADP sensitivity studies.

Reference: plan ``~/.claude/plans/eager-enchanting-glade.md`` §A3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RegimeStats:
    """Empirical statistics for one regime label."""
    label: int
    name: str
    n_days: int
    mean_spread_eur_mwh: float
    mean_price_eur_mwh: float
    daily_spread_bounds: tuple[float, float]  # (min, max) spread in this regime


@dataclass
class RegimeClassification:
    """Full output of :func:`fit_regimes`."""

    n_regimes: int
    regime_labels: pd.Series       # int label per calendar date
    daily_spreads: pd.Series       # p95-p5 spread per day, EUR/MWh
    daily_means: pd.Series         # mean price per day
    transition_matrix: np.ndarray  # (n_regimes, n_regimes), row-stochastic
    stationary: np.ndarray         # (n_regimes,) long-run regime shares
    stats: list[RegimeStats] = field(default_factory=list)

    def regime_for_date(self, date: pd.Timestamp) -> int:
        """Return the regime label assigned to ``date`` (raises KeyError if
        the date is outside the fit window)."""
        return int(self.regime_labels.loc[pd.Timestamp(date).normalize().date()])

    def expected_spread(self, regime: int) -> float:
        return self.stats[regime].mean_spread_eur_mwh

    def transition_probs(self, from_regime: int) -> np.ndarray:
        """Row of the transition matrix — P(next | from)."""
        return self.transition_matrix[from_regime]


_DEFAULT_REGIME_NAMES = {
    3: ("quiet", "normal", "volatile"),
    4: ("quiet", "low-mid", "mid-high", "volatile"),
    5: ("quiet", "low-mid", "mid", "mid-high", "volatile"),
}


def _price_series_to_daily_spreads(
    prices: pd.Series, low_pct: float, high_pct: float
) -> tuple[pd.Series, pd.Series]:
    """Collapse an hourly/15-min price series to (daily_spread, daily_mean)."""
    if prices.index.tz is None:
        local = prices.tz_localize("UTC")
    else:
        local = prices.tz_convert("UTC")
    by_day = local.groupby(local.index.normalize().date)
    daily_spread = by_day.apply(
        lambda s: float(np.percentile(s.to_numpy(dtype=float), high_pct)
                        - np.percentile(s.to_numpy(dtype=float), low_pct))
    )
    daily_mean = by_day.mean()
    return daily_spread, daily_mean


def _assign_percentile_regimes(
    daily_spreads: pd.Series, n_regimes: int
) -> pd.Series:
    """Split daily spreads into ``n_regimes`` equal-quantile buckets."""
    quantiles = np.linspace(0.0, 1.0, n_regimes + 1)
    thresholds = np.quantile(daily_spreads.to_numpy(dtype=float), quantiles[1:-1])
    labels = np.digitize(daily_spreads.to_numpy(dtype=float), thresholds)
    return pd.Series(labels, index=daily_spreads.index, name="regime")


def _estimate_transition_matrix(labels: pd.Series, n_regimes: int) -> np.ndarray:
    """Empirical P(next day regime | current day regime)."""
    arr = labels.to_numpy(dtype=int)
    M = np.zeros((n_regimes, n_regimes))
    for prev, curr in zip(arr[:-1], arr[1:]):
        M[prev, curr] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    # Uniform fallback for regimes that never occurred (row_sum = 0).
    zero_rows = (row_sums.squeeze() == 0)
    row_sums[zero_rows] = 1.0
    M = M / row_sums
    if zero_rows.any():
        M[zero_rows] = 1.0 / n_regimes
    return M


def _stationary_distribution(transition_matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Left eigenvector of M with eigenvalue 1 (long-run regime shares)."""
    n = transition_matrix.shape[0]
    eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    stationary = np.real(eigvecs[:, idx])
    stationary = stationary / stationary.sum()
    stationary = np.clip(stationary, tol, None)
    stationary = stationary / stationary.sum()
    return stationary


def fit_regimes(
    prices: pd.Series,
    n_regimes: int = 3,
    spread_low_pct: float = 5.0,
    spread_high_pct: float = 95.0,
    regime_names: tuple[str, ...] | None = None,
) -> RegimeClassification:
    """
    Fit percentile-based regime labels on a historical DA price series.

    Args:
        prices: pandas Series indexed by timestamp (hourly or 15-min),
            values in EUR/MWh. Multiple years OK; regimes are assigned
            globally on the pooled empirical distribution.
        n_regimes: Number of regime buckets (3 is the production default
            for A3.3 DP; 5 for sensitivity).
        spread_low_pct / spread_high_pct: Percentiles used to compute the
            daily "spread" feature (default 5th-95th → robust to outliers).
        regime_names: Optional tuple of length ``n_regimes`` overriding
            the default names.

    Returns:
        :class:`RegimeClassification` with labels, transition matrix,
        stationary distribution, and per-regime empirical stats.

    Notes:
        * Regimes are *ordinal* by spread: higher label = more volatile day.
        * The transition matrix is MLE on consecutive observed days;
          regimes that never occurred get a uniform row (no regime gets
          zero-probability transitions).
    """
    if n_regimes < 2:
        raise ValueError(f"n_regimes must be >= 2, got {n_regimes}")
    if regime_names is None:
        regime_names = _DEFAULT_REGIME_NAMES.get(
            n_regimes, tuple(f"regime_{i}" for i in range(n_regimes))
        )
    if len(regime_names) != n_regimes:
        raise ValueError(
            f"regime_names length {len(regime_names)} != n_regimes {n_regimes}"
        )

    daily_spread, daily_mean = _price_series_to_daily_spreads(
        prices, low_pct=spread_low_pct, high_pct=spread_high_pct,
    )
    labels = _assign_percentile_regimes(daily_spread, n_regimes)
    transition = _estimate_transition_matrix(labels, n_regimes)
    stationary = _stationary_distribution(transition)

    stats: list[RegimeStats] = []
    for r in range(n_regimes):
        mask = (labels == r)
        regime_spreads = daily_spread[mask]
        regime_means = daily_mean[mask]
        bounds = (
            float(regime_spreads.min()) if len(regime_spreads) else 0.0,
            float(regime_spreads.max()) if len(regime_spreads) else 0.0,
        )
        stats.append(RegimeStats(
            label=r,
            name=regime_names[r],
            n_days=int(mask.sum()),
            mean_spread_eur_mwh=float(regime_spreads.mean()) if len(regime_spreads) else 0.0,
            mean_price_eur_mwh=float(regime_means.mean()) if len(regime_means) else 0.0,
            daily_spread_bounds=bounds,
        ))

    return RegimeClassification(
        n_regimes=n_regimes,
        regime_labels=labels,
        daily_spreads=daily_spread,
        daily_means=daily_mean,
        transition_matrix=transition,
        stationary=stationary,
        stats=stats,
    )
