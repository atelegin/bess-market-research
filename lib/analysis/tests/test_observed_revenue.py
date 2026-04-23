"""
Tests for Tier 1/2 ground-truth historical revenue (no benchmark anchors).
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from lib.analysis import observed_revenue


@pytest.fixture
def mock_components():
    """Patch all external fetches with deterministic synthetic returns."""
    fake_da = 80.0  # kEUR/MW/yr from LP dispatch
    fake_fcr = 40.0
    fake_afrr = {"afrr_cap": 100.0, "afrr_energy": -50.0}
    with (
        patch.object(observed_revenue, "_wholesale_da_revenue_keur_per_mw",
                     return_value=fake_da),
        patch.object(observed_revenue, "fetch_fcr_annual_revenue",
                     return_value=fake_fcr),
        patch.object(observed_revenue, "fetch_afrr_annual_revenue",
                     return_value=fake_afrr),
    ):
        yield fake_da, fake_fcr, fake_afrr


def test_aggregates_all_components(mock_components):
    r = observed_revenue.observed_total_revenue(2024, duration_h=2.0)
    assert r is not None
    assert r["year"] == 2024
    assert r["duration_h"] == 2.0
    # DA 80 + ID 80*id_da_ratio + FCR 40 + aFRR_cap 100 + aFRR_en -50
    expected_id = 80.0 * observed_revenue.id_da_ratio(2024)
    assert r["da"] == pytest.approx(80.0, rel=1e-3)
    assert r["id"] == pytest.approx(expected_id, rel=1e-3)
    assert r["fcr"] == pytest.approx(40.0, rel=1e-3)
    assert r["afrr_cap"] == pytest.approx(100.0, rel=1e-3)
    assert r["afrr_energy"] == pytest.approx(-50.0, rel=1e-3)
    assert r["total"] == pytest.approx(80.0 + expected_id + 40.0 + 100.0 - 50.0, rel=1e-3)


def test_1h_battery_halves_ancillary_scaled_components(mock_components):
    r = observed_revenue.observed_total_revenue(2024, duration_h=1.0)
    assert r is not None
    # dur_scale = 0.5, so FCR/aFRR components halve; DA/ID unchanged
    assert r["fcr"] == pytest.approx(20.0, rel=1e-3)
    assert r["afrr_cap"] == pytest.approx(50.0, rel=1e-3)
    assert r["afrr_energy"] == pytest.approx(-25.0, rel=1e-3)
    assert r["da"] == pytest.approx(80.0, rel=1e-3)


def test_rejects_non_historical_years():
    r = observed_revenue.observed_total_revenue(2030, duration_h=2.0)
    assert r is None


def test_returns_none_if_fcr_fetch_fails():
    with (
        patch.object(observed_revenue, "_wholesale_da_revenue_keur_per_mw",
                     return_value=80.0),
        patch.object(observed_revenue, "fetch_fcr_annual_revenue",
                     return_value=None),
        patch.object(observed_revenue, "fetch_afrr_annual_revenue",
                     return_value={"afrr_cap": 100.0, "afrr_energy": -50.0}),
    ):
        r = observed_revenue.observed_total_revenue(2024, duration_h=2.0)
    assert r is None


def test_returns_none_if_da_dispatch_fails():
    with (
        patch.object(observed_revenue, "_wholesale_da_revenue_keur_per_mw",
                     return_value=None),
        patch.object(observed_revenue, "fetch_fcr_annual_revenue",
                     return_value=40.0),
        patch.object(observed_revenue, "fetch_afrr_annual_revenue",
                     return_value={"afrr_cap": 100.0, "afrr_energy": -50.0}),
    ):
        r = observed_revenue.observed_total_revenue(2024, duration_h=2.0)
    assert r is None


def test_afrr_energy_can_be_negative(mock_components):
    """Honest accounting: when NEG activations cost the operator more than
    POS activations earn, aFRR energy component is negative. The function
    should NOT clamp this to zero."""
    r = observed_revenue.observed_total_revenue(2024, duration_h=2.0)
    assert r["afrr_energy"] < 0
