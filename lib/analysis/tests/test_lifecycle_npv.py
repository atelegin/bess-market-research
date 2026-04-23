"""
Tests for lifecycle NPV simulator (Note 4 A4).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from lib.analysis.lifecycle_npv import (
    LifecycleResult,
    compare_policies,
    simulate_lifecycle,
)
from lib.models.adp_shadow_cost import (
    DepreciationProxyPolicy,
    NaivePolicy,
)
from lib.models.dispatch_stacked import PERIODS_PER_DAY, StackedDispatchResult


def _fake_frames() -> dict:
    """Return a frames-dict shape expected by the assembler.
    Content is ignored because we mock assembly outputs."""
    return {
        "da": MagicMock(), "id": MagicMock(),
        "afrr_cap": MagicMock(), "afrr_energy": MagicMock(),
        "activations": MagicMock(),
    }


def _mocked_day_inputs():
    """A StackedDayInputs-shaped object with valid array lengths."""
    from lib.analysis.stacked_day_assembler import StackedDayInputs
    from datetime import date
    zeros = np.zeros(PERIODS_PER_DAY)
    return StackedDayInputs(
        date=date(2024, 1, 1),
        prices_da=np.full(PERIODS_PER_DAY, 50.0),
        prices_id=np.full(PERIODS_PER_DAY, 50.0),
        afrr_cap_pos_price=np.zeros(6),
        afrr_cap_neg_price=np.zeros(6),
        afrr_energy_pos_price=zeros,
        afrr_energy_neg_price=zeros,
        afrr_activation_rate_pos=zeros,
        afrr_activation_rate_neg=zeros,
    )


def _mocked_dispatch_result(revenue=1000.0, fec=1.0):
    return StackedDispatchResult(
        charge_da=np.zeros(PERIODS_PER_DAY),
        discharge_da=np.zeros(PERIODS_PER_DAY),
        charge_id=np.zeros(PERIODS_PER_DAY),
        discharge_id=np.zeros(PERIODS_PER_DAY),
        r_pos=np.zeros(PERIODS_PER_DAY),
        r_neg=np.zeros(PERIODS_PER_DAY),
        a_pos=np.zeros(PERIODS_PER_DAY),
        a_neg=np.zeros(PERIODS_PER_DAY),
        soc=np.zeros(PERIODS_PER_DAY),
        revenue_da=revenue, revenue_id=0.0,
        revenue_afrr_cap_pos=0.0, revenue_afrr_cap_neg=0.0,
        revenue_afrr_energy_pos=0.0, revenue_afrr_energy_neg=0.0,
        revenue_total=revenue,
        full_equivalent_cycles=fec,
        success=True,
    )


def test_simulator_runs_short_horizon_with_mocks():
    """Smoke test: simulator progresses through days with mocked market
    inputs and LP outputs, accumulates revenue and FEC, degrades SoH."""
    with (
        patch("lib.analysis.lifecycle_npv._prefetch_year_frames", return_value=_fake_frames()),
        patch("lib.analysis.lifecycle_npv.assemble_day_inputs", return_value=_mocked_day_inputs()),
        patch("lib.analysis.lifecycle_npv.optimize_day_stacked",
              return_value=_mocked_dispatch_result(revenue=500.0, fec=0.8)),
    ):
        result = simulate_lifecycle(
            policy=NaivePolicy(),
            n_years=2, template_years=(2024,),
            max_days_per_year=30,  # fast
            initial_soh=1.0,
        )
    assert result.n_years == 2
    assert result.days_solved == 60
    assert (result.annual_revenue_eur > 0).all()
    assert result.end_of_year_soh[0] < 1.0
    assert result.end_of_year_soh[1] < result.end_of_year_soh[0]
    assert result.lifetime_npv_eur > 0
    # Discounted < nominal
    assert result.lifetime_npv_eur < result.annual_revenue_eur.sum()


def test_simulator_stops_at_warranty_floor():
    """Catastrophically high fade → SoH hits floor mid-simulation; year
    terminates early and subsequent years get zero revenue."""
    with (
        patch("lib.analysis.lifecycle_npv._prefetch_year_frames", return_value=_fake_frames()),
        patch("lib.analysis.lifecycle_npv.assemble_day_inputs", return_value=_mocked_day_inputs()),
        patch("lib.analysis.lifecycle_npv.optimize_day_stacked",
              return_value=_mocked_dispatch_result(revenue=500.0, fec=2.0)),
    ):
        result = simulate_lifecycle(
            policy=NaivePolicy(),
            n_years=10,
            template_years=(2024,),
            max_days_per_year=30,
            initial_soh=0.81,  # very close to floor
            warranty_floor=0.80,
            fade_per_fec_at_soh_1=1e-2,  # extreme fade
        )
    assert result.years_to_floor is not None
    # Later years have zero revenue
    assert result.annual_revenue_eur[-1] == 0.0
    assert result.end_of_year_soh[-1] == pytest.approx(0.80, abs=1e-6)


def test_compare_policies_all_receive_same_frames():
    """compare_policies prefetches once and passes the same frames to
    every policy — the contract that policies see identical markets."""
    prefetched = {2024: _fake_frames()}
    with (
        patch("lib.analysis.lifecycle_npv._prefetch_year_frames", return_value=_fake_frames()) as mock_prefetch,
        patch("lib.analysis.lifecycle_npv.assemble_day_inputs", return_value=_mocked_day_inputs()),
        patch("lib.analysis.lifecycle_npv.optimize_day_stacked",
              return_value=_mocked_dispatch_result(revenue=500.0, fec=0.8)),
    ):
        results = compare_policies(
            policies={
                "naive": NaivePolicy(),
                "depr": DepreciationProxyPolicy(capex_eur_per_mwh=100_000, lifetime_throughput_ratio=6000),
            },
            n_years=1, template_years=(2024,),
            max_days_per_year=5,
        )
    # Prefetch happened once per template year, reused across both policies
    # (2 calls = 1 in compare_policies prefetch step + 0 per policy since prefetched_frames passed)
    assert mock_prefetch.call_count == 1
    assert set(results) == {"naive", "depr"}
    for r in results.values():
        assert r.days_solved == 5


def test_discount_factor_applied_correctly():
    """NPV should equal Σ annual_rev × (1+r)^-year."""
    with (
        patch("lib.analysis.lifecycle_npv._prefetch_year_frames", return_value=_fake_frames()),
        patch("lib.analysis.lifecycle_npv.assemble_day_inputs", return_value=_mocked_day_inputs()),
        patch("lib.analysis.lifecycle_npv.optimize_day_stacked",
              return_value=_mocked_dispatch_result(revenue=1000.0, fec=0.5)),
    ):
        result = simulate_lifecycle(
            policy=NaivePolicy(),
            n_years=3, template_years=(2024,),
            max_days_per_year=10,
            discount_rate=0.10,
            fade_per_fec_at_soh_1=0,  # no degradation to isolate NPV arithmetic
            calendar_fade_per_day=0,
        )
    expected = sum(
        result.annual_revenue_eur[y] * (1.10 ** -(y + 1))
        for y in range(3)
    )
    assert result.lifetime_npv_eur == pytest.approx(expected, rel=1e-9)
