"""
Tests for Note 3 physics integration in the lifecycle simulator.
"""
from __future__ import annotations

import numpy as np

from lib.analysis.physics_degradation import physics_degradation_per_day
from lib.models.dispatch_stacked import StackedDispatchResult


def _build_day(soc_trace: np.ndarray, fec: float = 1.0) -> StackedDispatchResult:
    zeros = np.zeros(96)
    charge = np.zeros(96); charge[:48] = 1.0 / 4
    discharge = np.zeros(96); discharge[48:] = 1.0 / 4
    return StackedDispatchResult(
        charge_da=charge, discharge_da=discharge, charge_id=zeros, discharge_id=zeros,
        r_pos=zeros, r_neg=zeros, a_pos=zeros, a_neg=zeros,
        soc=soc_trace,
        revenue_da=0.0, revenue_id=0.0,
        revenue_afrr_cap_pos=0.0, revenue_afrr_cap_neg=0.0,
        revenue_afrr_energy_pos=0.0, revenue_afrr_energy_neg=0.0,
        revenue_total=0.0, full_equivalent_cycles=fec, success=True,
    )


def test_per_day_fade_in_realistic_lfp_range():
    """At 1 FEC/day and SoH=1, daily Δ SoH should be ~1e-4 (annual ≈ 3 %)."""
    soc = np.concatenate([np.linspace(0.4, 1.6, 48), np.linspace(1.6, 0.4, 48)])
    delta = physics_degradation_per_day(
        _build_day(soc), energy_mwh=2.0, soh_current=1.0,
    )
    assert 2e-5 < delta < 3e-4  # generous band around ~1e-4


def test_zero_dispatch_day_yields_calendar_only_fade():
    zero_soc = np.full(96, 1.0)
    delta = physics_degradation_per_day(
        _build_day(zero_soc, fec=0.0), energy_mwh=2.0, soh_current=1.0,
    )
    assert delta > 0  # calendar still fades
    assert delta < 1e-4  # less than a cycling day


def test_aged_soh_scarcer_fade():
    """Age acceleration: at SoH=0.82 same duty consumes more life."""
    soc = np.concatenate([np.linspace(0.4, 1.6, 48), np.linspace(1.6, 0.4, 48)])
    d_fresh = physics_degradation_per_day(_build_day(soc), energy_mwh=2.0, soh_current=1.0)
    d_aged = physics_degradation_per_day(_build_day(soc), energy_mwh=2.0, soh_current=0.82)
    assert d_aged > d_fresh


def test_deeper_dod_higher_fade_at_same_soh():
    """Deep-DoD day (0.10→1.90) should fade more than shallow-DoD (0.80→1.20)."""
    soc_deep = np.concatenate([np.linspace(0.1, 1.9, 48), np.linspace(1.9, 0.1, 48)])
    soc_shallow = np.concatenate([np.linspace(0.8, 1.2, 48), np.linspace(1.2, 0.8, 48)])
    d_deep = physics_degradation_per_day(_build_day(soc_deep), energy_mwh=2.0, soh_current=1.0)
    d_shallow = physics_degradation_per_day(_build_day(soc_shallow), energy_mwh=2.0, soh_current=1.0)
    assert d_deep >= d_shallow


def test_delta_monotone_with_fec():
    """More FEC in the day = more fade (holding everything else equal-ish)."""
    soc = np.concatenate([np.linspace(0.4, 1.6, 48), np.linspace(1.6, 0.4, 48)])
    d_one = physics_degradation_per_day(_build_day(soc, fec=1.0), energy_mwh=2.0, soh_current=1.0)
    d_two = physics_degradation_per_day(_build_day(soc, fec=2.0), energy_mwh=2.0, soh_current=1.0)
    assert d_two > d_one
