"""
Tests for real aFRR activation-energy revenue (A1.1).

Uses synthetic (not network) activation + price tables to exercise the
math without relying on netztransparenz/regelleistung availability.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from lib.data import ancillary_prices


def _synthetic_prices(year: int) -> pd.DataFrame:
    """4 x 15-min intervals at constant price per direction."""
    idx = pd.DatetimeIndex(
        [
            f"{year}-01-01 00:00:00+00:00",
            f"{year}-01-01 00:15:00+00:00",
            f"{year}-01-01 00:30:00+00:00",
            f"{year}-01-01 00:45:00+00:00",
        ],
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "pos_avg_eur_mwh": [1000.0, 1000.0, 1000.0, 1000.0],
            "pos_marginal_eur_mwh": [15000.0, 15000.0, 15000.0, 15000.0],
            "neg_avg_eur_mwh": [-500.0, -500.0, -500.0, -500.0],
            "neg_marginal_eur_mwh": [-15000.0, -15000.0, -15000.0, -15000.0],
        },
        index=idx,
    )


def _synthetic_activations(year: int) -> pd.DataFrame:
    """Matching 4 intervals with constant activation volumes."""
    idx = pd.DatetimeIndex(
        [
            f"{year}-01-01 00:00:00+00:00",
            f"{year}-01-01 00:15:00+00:00",
            f"{year}-01-01 00:30:00+00:00",
            f"{year}-01-01 00:45:00+00:00",
        ],
        tz="UTC",
    )
    return pd.DataFrame(
        {"pos_mw": [100.0, 100.0, 100.0, 100.0], "neg_mw": [50.0, 50.0, 50.0, 50.0]},
        index=idx,
    )


def test_revenue_math_positive_direction():
    """POS revenue = activated_mw / pool * price * 0.25h summed over intervals.
    4 intervals × 100 MW × 1000 EUR/MWh × 0.25h / 2000 = 50 EUR per MW = 0.05 kEUR."""
    prices = _synthetic_prices(2024)
    acts = _synthetic_activations(2024)
    with patch.object(ancillary_prices, "fetch_afrr_energy_prices", return_value=prices), \
         patch("lib.data.afrr_activations.fetch_afrr_activations", return_value=acts):
        r = ancillary_prices.compute_afrr_energy_revenue_real(
            2024, contracted_pos_mw=2000.0, contracted_neg_mw=1800.0,
        )
    assert r["pos_keur_per_mw"] == pytest_approx_scalar(0.05)


def test_revenue_math_negative_direction_is_negative():
    """NEG price is negative → operator pays → NEG revenue is negative.
    4 intervals × 50 MW × -500 EUR/MWh × 0.25h / 1800 = -13.889 EUR = -0.01389 kEUR."""
    prices = _synthetic_prices(2024)
    acts = _synthetic_activations(2024)
    with patch.object(ancillary_prices, "fetch_afrr_energy_prices", return_value=prices), \
         patch("lib.data.afrr_activations.fetch_afrr_activations", return_value=acts):
        r = ancillary_prices.compute_afrr_energy_revenue_real(
            2024, contracted_pos_mw=2000.0, contracted_neg_mw=1800.0,
        )
    expected_neg_eur = 4 * 50.0 * -500.0 * 0.25 / 1800.0
    assert r["neg_keur_per_mw"] == pytest_approx_scalar(expected_neg_eur / 1000.0)


def test_net_combines_both_directions():
    prices = _synthetic_prices(2024)
    acts = _synthetic_activations(2024)
    with patch.object(ancillary_prices, "fetch_afrr_energy_prices", return_value=prices), \
         patch("lib.data.afrr_activations.fetch_afrr_activations", return_value=acts):
        r = ancillary_prices.compute_afrr_energy_revenue_real(
            2024, contracted_pos_mw=2000.0, contracted_neg_mw=1800.0,
        )
    assert r["net_keur_per_mw"] == pytest_approx_scalar(
        r["pos_keur_per_mw"] + r["neg_keur_per_mw"]
    )


def test_returns_none_on_missing_prices():
    with patch.object(ancillary_prices, "fetch_afrr_energy_prices", return_value=None):
        r = ancillary_prices.compute_afrr_energy_revenue_real(2024)
    assert r is None


def test_handles_empty_overlap_between_activations_and_prices():
    prices = _synthetic_prices(2024)
    # Activations for a different year
    acts = _synthetic_activations(2099)
    with patch.object(ancillary_prices, "fetch_afrr_energy_prices", return_value=prices), \
         patch("lib.data.afrr_activations.fetch_afrr_activations", return_value=acts):
        r = ancillary_prices.compute_afrr_energy_revenue_real(2024)
    assert r is None


def pytest_approx_scalar(value, rel=1e-6):
    """Helper — pytest.approx but works with numpy scalars, strings, etc."""
    import pytest
    return pytest.approx(value, rel=rel)
