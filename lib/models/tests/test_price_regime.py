"""
Tests for price regime classifier (Note 4 A3.2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lib.models.price_regime import (
    RegimeClassification,
    _stationary_distribution,
    fit_regimes,
)


def _synthetic_prices(days: int, seed: int = 42) -> pd.Series:
    """Synthetic hourly DA prices with varying daily volatility."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=days * 24, freq="h", tz="UTC")
    # Baseline 80 EUR/MWh + diurnal sinusoid + daily vol shocks
    hours = np.arange(len(idx)) % 24
    baseline = 80 + 20 * np.sin((hours - 3) * np.pi / 12)
    daily_vol = rng.gamma(shape=2.0, scale=15.0, size=days)  # some days calm, some volatile
    noise = rng.normal(0, np.repeat(daily_vol, 24), size=len(idx))
    prices = baseline + noise
    return pd.Series(prices, index=idx, name="price_eur_mwh")


def test_fits_three_regimes_by_default():
    prices = _synthetic_prices(days=60)
    rc = fit_regimes(prices, n_regimes=3)
    assert rc.n_regimes == 3
    assert len(rc.regime_labels) == 60
    assert set(rc.regime_labels.unique()) <= {0, 1, 2}


def test_regimes_ordered_by_spread():
    prices = _synthetic_prices(days=120)
    rc = fit_regimes(prices, n_regimes=3)
    spreads = [s.mean_spread_eur_mwh for s in rc.stats]
    # Regime 0 (quiet) has smallest spread, 2 (volatile) has largest
    assert spreads[0] < spreads[1] < spreads[2]


def test_transition_matrix_rows_sum_to_one():
    prices = _synthetic_prices(days=80)
    rc = fit_regimes(prices, n_regimes=3)
    row_sums = rc.transition_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_stationary_distribution_sums_to_one():
    prices = _synthetic_prices(days=100)
    rc = fit_regimes(prices, n_regimes=3)
    assert rc.stationary.sum() == pytest.approx(1.0, rel=1e-6)
    assert np.all(rc.stationary > 0)


def test_stationary_consistent_with_transition():
    """Stationary π satisfies π = π @ M."""
    prices = _synthetic_prices(days=200, seed=7)
    rc = fit_regimes(prices, n_regimes=3)
    post = rc.stationary @ rc.transition_matrix
    assert np.allclose(post, rc.stationary, atol=1e-6)


def test_transition_matrix_uniform_row_for_unseen_regime():
    """If a regime never occurred (pathological short series), its row
    must still be row-stochastic (uniform fallback)."""
    # Use constant prices → all days fall in the lowest regime; regimes 1, 2
    # never observed. We force this to exercise the uniform-fallback branch.
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    flat = pd.Series(np.full(len(idx), 50.0), index=idx)
    rc = fit_regimes(flat, n_regimes=3)
    # Row sums still = 1
    row_sums = rc.transition_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_regime_labels_indexed_by_date():
    prices = _synthetic_prices(days=30)
    rc = fit_regimes(prices)
    # Labels index is list of date objects, matching grouped rollup
    assert len(rc.regime_labels) == 30


def test_custom_regime_names_round_trip():
    prices = _synthetic_prices(days=60)
    rc = fit_regimes(prices, n_regimes=4,
                     regime_names=("A", "B", "C", "D"))
    assert [s.name for s in rc.stats] == ["A", "B", "C", "D"]


def test_regime_for_date_lookup():
    prices = _synthetic_prices(days=14)
    rc = fit_regimes(prices)
    label = rc.regime_for_date(pd.Timestamp("2023-01-05"))
    assert label in (0, 1, 2)


def test_invalid_n_regimes_raises():
    prices = _synthetic_prices(days=10)
    with pytest.raises(ValueError, match=">= 2"):
        fit_regimes(prices, n_regimes=1)


def test_regime_name_length_mismatch_raises():
    prices = _synthetic_prices(days=10)
    with pytest.raises(ValueError, match="regime_names length"):
        fit_regimes(prices, n_regimes=3, regime_names=("A", "B"))


def test_stationary_of_uniform_matrix_is_uniform():
    n = 3
    uniform = np.full((n, n), 1.0 / n)
    s = _stationary_distribution(uniform)
    assert np.allclose(s, np.full(n, 1.0 / n), atol=1e-9)
