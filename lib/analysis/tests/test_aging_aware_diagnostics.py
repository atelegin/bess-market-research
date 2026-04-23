"""
Tests for the five owner-facing aging-aware diagnostics (Note 4 C).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from lib.analysis.aging_aware_diagnostics import (
    DayDiagnosticData,
    ancillary_vs_arbitrage_mix,
    compute_all_signals,
    crate_histogram,
    dod_by_spread_decile,
    revenue_per_cycle_by_quartile,
    soc_hours_histogram,
)


def _synthetic_day(
    day: date,
    spread: float = 30.0,
    fec: float = 1.0,
    soc_centre_frac: float = 0.5,
    soc_swing_frac: float = 0.4,
    power_mw: float = 1.0,
    energy_mwh: float = 2.0,
    revenue_eur: float = 400.0,
    revenue_breakdown: dict | None = None,
) -> DayDiagnosticData:
    """Build a minimal DayDiagnosticData for testing."""
    hours = np.arange(96)
    # Prices: spread around 80 EUR with sinusoid of amplitude=spread/2
    prices = 80.0 + (spread / 2.0) * np.sin((hours - 48) * 2 * np.pi / 96)
    # SoC trace: sinusoid around centre, amplitude soc_swing_frac/2
    soc_centre = soc_centre_frac * energy_mwh
    soc_amp = (soc_swing_frac / 2.0) * energy_mwh
    soc = soc_centre + soc_amp * np.sin((hours - 12) * 2 * np.pi / 96)
    # Power: derivative of SoC (rough)
    power = np.gradient(soc) / 0.25
    if revenue_breakdown is None:
        revenue_breakdown = {"da": revenue_eur, "id": 0.0, "afrr_cap_pos": 0.0,
                             "afrr_cap_neg": 0.0, "afrr_energy_pos": 0.0,
                             "afrr_energy_neg": 0.0}
    return DayDiagnosticData(
        date=day,
        power_mw_signed=power,
        soc_mwh=soc,
        prices_da=prices,
        power_mw=power_mw, energy_mwh=energy_mwh,
        daily_revenue_eur=revenue_eur,
        full_equivalent_cycles=fec,
        revenue_breakdown=revenue_breakdown,
    )


def test_empty_inputs_return_empty_frames():
    assert dod_by_spread_decile([]).empty
    assert soc_hours_histogram([]).empty
    assert revenue_per_cycle_by_quartile([]).empty
    assert crate_histogram([]).empty
    mix = ancillary_vs_arbitrage_mix([])
    assert mix["total"] == 0.0


def test_dod_vs_spread_aging_aware_vs_naive():
    """Synthetic test: aging-aware operator has high DoD on high-spread
    days, low on low-spread days; naive operator has constant high DoD.
    DoD-by-spread-decile for aging-aware shows monotone rise."""
    days_aging = []
    days_naive = []
    base_date = date(2024, 1, 1)
    for i in range(30):
        d = base_date + timedelta(days=i)
        spread = 20 + i * 4  # ranges 20 to 136
        # Aging-aware: swing ∝ spread (skip depth on low-spread days)
        aging_swing = 0.1 + 0.7 * (spread / 150)  # 0.1-0.76 fraction
        # Naive: always swings ~0.6
        days_aging.append(_synthetic_day(d, spread=spread, soc_swing_frac=aging_swing))
        days_naive.append(_synthetic_day(d, spread=spread, soc_swing_frac=0.6))

    d_aging = dod_by_spread_decile(days_aging)
    d_naive = dod_by_spread_decile(days_naive)
    # Aging-aware: DoD rises with decile; naive: flat
    aging_range = d_aging["mean_dod"].max() - d_aging["mean_dod"].min()
    naive_range = d_naive["mean_dod"].max() - d_naive["mean_dod"].min()
    assert aging_range > naive_range + 0.05


def test_soc_hours_histogram_concentrates():
    """Aging-aware centres at 50%; naive spends more time at extremes."""
    base_date = date(2024, 1, 1)
    days_centred = [_synthetic_day(base_date + timedelta(days=i),
                                    soc_centre_frac=0.5, soc_swing_frac=0.2) for i in range(30)]
    days_extreme = [_synthetic_day(base_date + timedelta(days=i),
                                    soc_centre_frac=0.5, soc_swing_frac=0.9) for i in range(30)]

    hist_centred = soc_hours_histogram(days_centred, n_bins=10)
    hist_extreme = soc_hours_histogram(days_extreme, n_bins=10)
    # Centred: more mass in middle 4 buckets (bins 3-6); extreme: more in edges
    mid_centred = hist_centred.iloc[3:7]["hours"].sum()
    mid_extreme = hist_extreme.iloc[3:7]["hours"].sum()
    assert mid_centred > mid_extreme


def test_revenue_per_cycle_higher_on_low_spread_for_aging_aware():
    """Aging-aware skips marginal cycles on low-spread days. Only the
    small number of cycles it does run are high-margin (caught the best
    sub-hour windows), so EUR/FEC is HIGH. Naive runs 1 FEC every day
    regardless, dragging low-spread EUR/FEC down."""
    base_date = date(2024, 1, 1)
    days_aging = []
    days_naive = []
    for i in range(40):
        d = base_date + timedelta(days=i)
        spread = 20 + i * 3  # 20 → 137
        if spread < 40:
            # Low-spread day. Aging-aware runs tiny FEC but captures peak moments
            # (revenue per cycle ~ high). Naive forces 1 FEC at the day's average
            # (revenue per cycle ~ mediocre).
            aging_fec, aging_rev = 0.1, 80.0
            naive_fec, naive_rev = 1.0, 150.0
        else:
            # High-spread day: both cycle near full; revenue-per-cycle similar
            aging_fec, aging_rev = 1.2, spread * 12
            naive_fec, naive_rev = 1.2, spread * 12
        days_aging.append(_synthetic_day(d, spread=spread, fec=aging_fec, revenue_eur=aging_rev))
        days_naive.append(_synthetic_day(d, spread=spread, fec=naive_fec, revenue_eur=naive_rev))

    q_aging = revenue_per_cycle_by_quartile(days_aging)
    q_naive = revenue_per_cycle_by_quartile(days_naive)
    # Bottom quartile (spread < 40): aging-aware earns 800 €/FEC, naive 150
    assert q_aging.iloc[0]["eur_per_fec"] > q_naive.iloc[0]["eur_per_fec"]


def test_crate_histogram_aging_aware_lower_tail():
    """Aging-aware reduces high-C-rate hours; naive leaves more mass at
    high-C. Construct directly: aging-aware stays in lowest bin, naive
    covers mid+upper bins."""
    base_date = date(2024, 1, 1)
    # Aging-aware: very low swing → C-rates mostly < 0.2
    days_aging = [_synthetic_day(base_date + timedelta(days=i),
                                  soc_swing_frac=0.1,
                                  energy_mwh=2.0) for i in range(20)]
    # Naive-style: use a saw-tooth power signal directly in higher C-rates.
    days_naive = []
    for i in range(20):
        d = _synthetic_day(base_date + timedelta(days=i), energy_mwh=1.0)
        # Force high-C by setting explicit power schedule: 1 MW charge/discharge
        # for 6h each. At energy_mwh=1.0 this is C-rate 1.0 (top bin).
        power = np.zeros(96)
        power[:24] = 1.0
        power[48:72] = -1.0
        d.power_mw_signed = power
        days_naive.append(d)
    h_aging = crate_histogram(days_aging, n_bins=5)
    h_naive = crate_histogram(days_naive, n_bins=5)
    # Upper half (bins 3+4): naive has mass, aging-aware does not
    upper_aging = h_aging.iloc[3:]["hours"].sum()
    upper_naive = h_naive.iloc[3:]["hours"].sum()
    assert upper_naive > upper_aging


def test_ancillary_mix_flags_as_dominant_asset():
    """When aFRR revenue dominates, as_share > 0.5."""
    base_date = date(2024, 1, 1)
    as_days = [_synthetic_day(
        base_date + timedelta(days=i),
        revenue_eur=500,
        revenue_breakdown={
            "da": 50, "id": 20, "afrr_cap_pos": 200, "afrr_cap_neg": 100,
            "afrr_energy_pos": 130, "afrr_energy_neg": 0,
        },
    ) for i in range(10)]
    mix = ancillary_vs_arbitrage_mix(as_days)
    assert mix["as_share"] > 0.5
    assert mix["total"] == pytest.approx(5000)


def test_ancillary_mix_flags_wholesale_dominant_asset():
    """Arbitrage-dominant asset → as_share < 0.5."""
    base_date = date(2024, 1, 1)
    wh_days = [_synthetic_day(
        base_date + timedelta(days=i),
        revenue_eur=300,
        revenue_breakdown={
            "da": 200, "id": 80, "afrr_cap_pos": 10, "afrr_cap_neg": 10,
            "afrr_energy_pos": 0, "afrr_energy_neg": 0,
        },
    ) for i in range(10)]
    mix = ancillary_vs_arbitrage_mix(wh_days)
    assert mix["as_share"] < 0.5


def test_compute_all_signals_returns_report():
    base_date = date(2024, 1, 1)
    days = [_synthetic_day(base_date + timedelta(days=i),
                            spread=20 + i * 3) for i in range(40)]
    report = compute_all_signals(policy_name="test", days=days)
    assert report.policy_name == "test"
    assert report.n_days == 40
    assert not report.dod_vs_spread.empty
    assert not report.soc_hours.empty
    assert not report.revenue_per_cycle_quartile.empty
    assert not report.crate_hist.empty
    assert "as_share" in report.ancillary_mix
