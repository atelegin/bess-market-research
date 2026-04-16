"""Detailed-model physics + anchor + monotonicity tests."""
from __future__ import annotations

import numpy as np
import pytest

from lib.models.degradation import PRESETS
from lib.models.degradation_detailed import (
    DutyCycle,
    cell_soh_detailed,
    project_capacity_detailed,
)


def _baseline_duty(**overrides) -> DutyCycle:
    kwargs = dict(fec_per_year=730, mean_dod=0.80, mean_soc=0.55, mean_crate=0.5, mean_temp_C=25.0)
    kwargs.update(overrides)
    return DutyCycle.from_mean(**kwargs)


def test_eve_anchor_matches_spec():
    """EVE LF280K: 6000 FEC @ 100% DoD, 0.5C, 25 °C → ~80% SoH (±2pp)."""
    duty = DutyCycle.from_mean(fec_per_year=6000, mean_dod=1.0, mean_soc=0.55, mean_crate=0.5, mean_temp_C=25.0)
    soh = cell_soh_detailed(duty, years=1.0, preset=PRESETS["eve_lf280k"], n_mc=1)
    assert abs(soh - 0.80) <= 0.02, soh


def test_all_presets_anchor_within_tolerance():
    """Each preset reproduces its own ``test_anchor.retention`` to within 1.5pp."""
    for name in ["eve_lf280k", "catl_enerc_plus_306ah", "byd_mc_cube_t", "trina_elementa_280ah", "baseline_fleet"]:
        p = PRESETS[name]
        target = p.test_anchor.retention
        duty = DutyCycle.from_mean(
            fec_per_year=p.test_anchor.cycles,
            mean_dod=p.test_anchor.dod,
            mean_soc=0.55,
            mean_crate=p.test_anchor.c_rate,
            mean_temp_C=p.test_anchor.temp_C,
        )
        soh = cell_soh_detailed(duty, years=1.0, preset=p, n_mc=1)
        assert abs(soh - target) <= 0.015, (name, soh, target)


def test_dod_monotonicity():
    preset = PRESETS["eve_lf280k"]
    prev = 1.0
    for dod in [0.4, 0.6, 0.8, 1.0]:
        soh = cell_soh_detailed(_baseline_duty(mean_dod=dod), years=10.0, preset=preset, n_mc=1)
        assert soh <= prev + 1e-9
        prev = soh


def test_soc_dwell_monotonicity():
    preset = PRESETS["eve_lf280k"]
    soh_low = cell_soh_detailed(_baseline_duty(mean_soc=0.30), years=10.0, preset=preset, n_mc=1)
    soh_mid = cell_soh_detailed(_baseline_duty(mean_soc=0.55), years=10.0, preset=preset, n_mc=1)
    soh_high = cell_soh_detailed(_baseline_duty(mean_soc=0.85), years=10.0, preset=preset, n_mc=1)
    assert soh_low > soh_mid > soh_high


def test_temperature_monotonicity():
    preset = PRESETS["eve_lf280k"]
    prev = 1.0
    for T in [15, 25, 35, 45]:
        soh = cell_soh_detailed(_baseline_duty(mean_temp_C=T), years=10.0, preset=preset, n_mc=1)
        assert soh <= prev + 1e-9
        prev = soh


def test_c_rate_monotonicity():
    preset = PRESETS["eve_lf280k"]
    prev = 1.0
    for c in [0.25, 0.5, 1.0]:
        soh = cell_soh_detailed(_baseline_duty(mean_crate=c), years=10.0, preset=preset, n_mc=1)
        assert soh <= prev + 1e-9
        prev = soh


def test_temp_range_warning():
    preset = PRESETS["eve_lf280k"]
    with pytest.warns(UserWarning):
        cell_soh_detailed(_baseline_duty(mean_temp_C=5.0), years=5.0, preset=preset, n_mc=1)


def test_mc_determinism_with_fixed_seed():
    preset = PRESETS["eve_lf280k"]
    duty = _baseline_duty()
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    a = cell_soh_detailed(duty, 10.0, preset, n_mc=50, rng=rng1)
    b = cell_soh_detailed(duty, 10.0, preset, n_mc=50, rng=rng2)
    np.testing.assert_allclose(a, b)


def test_pack_percentile_below_median():
    preset = PRESETS["eve_lf280k"]
    duty = _baseline_duty()
    p10, p50, p90 = project_capacity_detailed(
        duty, years=10.0, preset=preset, n_mc=500, return_kind="distribution", rng=np.random.default_rng(0)
    )
    assert p10 <= p50 <= p90


def test_duty_from_timeseries_roundtrip():
    rng = np.random.default_rng(0)
    soc = rng.uniform(0.2, 0.9, size=8760)
    duty = DutyCycle.from_timeseries(soc, fec_per_year=500, mean_dod=0.6)
    assert abs(duty.total_hours - 8760.0) < 1e-6


def test_self_heating_off_by_default_noop():
    """Every shipped preset has self_heating_coeff_C_per_C2=0; output must be
    identical with and without explicit passthrough. Guards against a caller
    accidentally enabling it via a stale preset."""
    for name, p in PRESETS.items():
        assert p.self_heating_coeff_C_per_C2 == 0.0, (name, p.self_heating_coeff_C_per_C2)


def test_self_heating_accelerates_hot_high_c_duty():
    """Enable self-heating on an EVE clone and verify cycle loss grows with C²
    at fixed ambient T. Also verify that at C=0.1 the effect is negligible
    (0.01 °C bump at coeff=2 → Arrhenius shift ~0.05%)."""
    from dataclasses import replace
    base = PRESETS["eve_lf280k"]
    hot = replace(base, self_heating_coeff_C_per_C2=2.0)
    duty_2c = DutyCycle.from_mean(fec_per_year=730, mean_dod=0.80, mean_soc=0.55, mean_crate=2.0, mean_temp_C=30.0)
    duty_01c = DutyCycle.from_mean(fec_per_year=730, mean_dod=0.80, mean_soc=0.55, mean_crate=0.1, mean_temp_C=30.0)
    # 2C → T_cell = 38 °C vs ambient 30 °C — material Arrhenius shift.
    soh_base = cell_soh_detailed(duty_2c, years=10.0, preset=base, n_mc=1)
    soh_hot = cell_soh_detailed(duty_2c, years=10.0, preset=hot, n_mc=1)
    assert soh_hot < soh_base - 0.01, (soh_hot, soh_base)
    # 0.1C → T_cell = 30.02 °C — negligible.
    soh_cold_base = cell_soh_detailed(duty_01c, years=10.0, preset=base, n_mc=1)
    soh_cold_hot = cell_soh_detailed(duty_01c, years=10.0, preset=hot, n_mc=1)
    assert abs(soh_cold_hot - soh_cold_base) < 0.001


def test_mc_noise_survives_zero_cycle_duty():
    """Calendar-dominated duty (zero FEC, high-SoC storage) must still produce
    non-zero cell-to-cell spread. The original sigma formula scaled with cycle
    stress only, so a stored cell had a degenerate p10=p50=p90 band — wrong
    for FCR-narrow, warranty/valuation, and second-life scenarios.
    """
    preset = PRESETS["eve_lf280k"]
    duty = DutyCycle.from_mean(
        fec_per_year=0.0, mean_dod=0.0, mean_soc=0.85,
        mean_crate=0.25, mean_temp_C=25.0,
    )
    samples = cell_soh_detailed(duty, years=10.0, preset=preset, n_mc=500, rng=np.random.default_rng(0))
    assert np.std(samples) > 1e-4, f"calendar-dominated duty has degenerate MC spread: std={np.std(samples):.6f}"


def test_from_mean_soc_continuous_no_branch_aliasing():
    """mean_soc=0.75 and mean_soc=0.80 must produce distinguishable calendar
    load. Under the legacy branch allocation both fell into the same third
    branch (0.70 <= mean_soc < 0.85) and aliased to identical SoH — killing
    the resolution that rest-SoC dispatch work (Note 3b) depends on. The
    continuous interpolation introduced alongside the Naumann-cubic k_cal
    evaluator must preserve strict monotonicity between adjacent mean_soc
    values across the 0.30–0.90 interior.
    """
    preset = PRESETS["eve_lf280k"]
    prev = 1.0
    # walk across the interior where continuous behavior matters; the
    # old branch-step code would repeat values within each branch.
    for mean_soc in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        soh = cell_soh_detailed(_baseline_duty(mean_soc=mean_soc), years=10.0, preset=preset, n_mc=1)
        assert soh < prev - 1e-4, f"mean_soc={mean_soc}: SoH {soh:.5f} not strictly below previous {prev:.5f}"
        prev = soh


def test_from_mean_soc_anchor_invariance():
    """Calendar load at anchor mean_soc values (0.30, 0.55, 0.75, 0.90) must
    match the legacy branch-allocation exactly. Locks Note 1/2/Trina anchor
    calibrations against silent drift from future allocation tweaks.
    """
    from lib.models.degradation_detailed import LFPGraphiteWangNaumann
    preset = PRESETS["baseline_fleet"]
    kernel = LFPGraphiteWangNaumann()
    expected_k_cal = {
        0.30: (6000 * 0.60 + 2760 * 1.00 + 0 * 1.60) / 8760,
        0.55: (2000 * 0.60 + 6000 * 1.00 + 760 * 1.60) / 8760,
        0.75: (500 * 0.60 + 3760 * 1.00 + 4500 * 1.60) / 8760,
        0.90: (0 * 0.60 + 2000 * 1.00 + 6760 * 1.60) / 8760,
    }
    for mean_soc, target in expected_k_cal.items():
        duty = _baseline_duty(mean_soc=mean_soc)
        q = kernel.calendar_loss(
            elapsed_years=1.0,
            soc_bucket_hours=duty.soc_bucket_hours,
            temp_C=25.0,
            preset=preset,
        )
        # Back out the k_cal bucket weight by dividing out all other factors.
        # With t=1 yr, T=25 °C (arrhenius=1), q = k_cal * bucket_weight * 1 * 1.
        implied_bucket = q / preset.k_cal
        assert abs(implied_bucket - target) < 1e-9, (mean_soc, implied_bucket, target)
