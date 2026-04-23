"""
Tests for the intraday ADP solver (Note 4 A3-extension).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lib.models.adp_solver import default_grids
from lib.models.adp_solver_intraday import (
    IntradayADPSolver,
    fit_hourly_price_profiles,
)
from lib.models.price_regime import RegimeClassification, fit_regimes


def _synthetic_da_series(days: int = 60) -> pd.Series:
    """Hourly prices with evening-peak diurnal pattern + random daily volatility."""
    rng = np.random.default_rng(17)
    idx = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC")
    hours = np.arange(len(idx)) % 24
    # Cosine centred at hour=18 (evening peak), through at hour=6 (morning low).
    baseline = 80 + 40 * np.cos((hours - 18) * 2 * np.pi / 24)
    daily_vol = rng.gamma(2.0, 12.0, size=days)
    noise = rng.normal(0, np.repeat(daily_vol, 24), size=len(idx))
    return pd.Series(baseline + noise, index=idx, name="price_eur_mwh")


def test_fit_hourly_profiles_shape():
    da = _synthetic_da_series(60)
    rc = fit_regimes(da, n_regimes=3)
    profiles = fit_hourly_price_profiles(da, rc)
    assert profiles.shape == (3, 24)
    assert np.all(np.isfinite(profiles))


def test_profiles_show_diurnal_pattern():
    da = _synthetic_da_series(120)
    rc = fit_regimes(da, n_regimes=3)
    profiles = fit_hourly_price_profiles(da, rc)
    # Each regime should have an evening hour with higher mean than a pre-dawn hour
    for r in range(rc.n_regimes):
        assert profiles[r, 18] > profiles[r, 3]


def test_solver_runs_on_simple_inputs():
    da = _synthetic_da_series(60)
    rc = fit_regimes(da, n_regimes=3)
    profiles = fit_hourly_price_profiles(da, rc)
    solver = IntradayADPSolver(
        regime_classification=rc,
        hourly_price_profiles=profiles,
        grids=default_grids(),
        energy_mwh=2.0, power_mw=1.0,
    )
    result = solver.solve()
    n_soh = len(default_grids().soh_grid)
    assert result.shadow_cost.shape == (n_soh, 3, 24)
    assert np.all(np.isfinite(result.shadow_cost))
    assert np.all(result.shadow_cost >= 0)


def test_shadow_cost_varies_by_hour():
    """Principal test: intraday DP must produce hour-varying shadow cost,
    which simplified-state DP cannot."""
    da = _synthetic_da_series(120)
    rc = fit_regimes(da, n_regimes=3)
    profiles = fit_hourly_price_profiles(da, rc)
    solver = IntradayADPSolver(
        regime_classification=rc, hourly_price_profiles=profiles,
        energy_mwh=2.0, power_mw=1.0,
    )
    result = solver.solve()
    # Pick mid-SoH and normal regime
    mid_soh = result.shadow_cost.shape[0] // 2
    hourly = result.shadow_cost[mid_soh, 1]
    # Max/min ratio > 1.5 means meaningful hour-to-hour variation
    assert hourly.max() / max(hourly.min(), 1e-6) > 1.5


def test_shadow_cost_higher_in_volatile_regime():
    """Volatile regime (wider price spreads) → higher opportunity cost of
    wasting SoC on marginal cycles."""
    da = _synthetic_da_series(120)
    rc = fit_regimes(da, n_regimes=3)
    profiles = fit_hourly_price_profiles(da, rc)
    solver = IntradayADPSolver(
        regime_classification=rc, hourly_price_profiles=profiles,
        energy_mwh=2.0, power_mw=1.0,
    )
    result = solver.solve()
    mid_soh = result.shadow_cost.shape[0] // 2
    mean_quiet = result.shadow_cost[mid_soh, 0].mean()
    mean_volatile = result.shadow_cost[mid_soh, 2].mean()
    assert mean_volatile > mean_quiet


def test_profiles_shape_validation():
    da = _synthetic_da_series(30)
    rc = fit_regimes(da, n_regimes=3)
    bad = np.zeros((2, 24))  # wrong regime count
    with pytest.raises(ValueError, match="hourly_price_profiles must be shape"):
        IntradayADPSolver(
            regime_classification=rc, hourly_price_profiles=bad,
            energy_mwh=2.0, power_mw=1.0,
        )
