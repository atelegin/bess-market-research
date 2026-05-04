"""
Tests for the Rainflow sidebar wear function and policy
(:mod:`lib.analysis.rainflow_wear`, :class:`lib.models.adp.shadow_cost.RainflowPolicy`).

Methodology lives parallel to L6 physics_wear_from_duty; tests mirror
the structure of test_physics_degradation but probe the rainflow-
extraction layer specifically.
"""
from __future__ import annotations

import numpy as np
import pytest

from lib.analysis.rainflow_wear import (
    extract_rainflow_cycles,
    rainflow_wear_from_duty,
    evaluate_piecewise,
)
from lib.models.adp.shadow_cost import RainflowPolicy
from lib.models.dispatch.stacked import StackedDispatchResult


# ---------------------------------------------------------------------------
# Synthetic day-result builder mirroring test_physics_degradation.
# ---------------------------------------------------------------------------


def _build_day(soc_trace: np.ndarray, fec: float = 1.0) -> StackedDispatchResult:
    zeros = np.zeros(96)
    charge = np.zeros(96); charge[:48] = 1.0 / 4
    discharge = np.zeros(96); discharge[48:] = 1.0 / 4
    return StackedDispatchResult(
        charge_da=charge, discharge_da=discharge,
        charge_id=zeros, discharge_id=zeros,
        r_pos=zeros, r_neg=zeros, a_pos=zeros, a_neg=zeros,
        soc=soc_trace,
        revenue_da=0.0, revenue_id=0.0,
        revenue_afrr_cap_pos=0.0, revenue_afrr_cap_neg=0.0,
        revenue_afrr_energy_pos=0.0, revenue_afrr_energy_neg=0.0,
        revenue_total=0.0, full_equivalent_cycles=fec, success=True,
    )


@pytest.fixture
def linear_coeffs():
    """Linear f(DoD) = 1e-5 × DoD — slope-ratio = 1, used for sanity tests."""
    return {
        "dod_breakpoints": np.linspace(0.0, 1.0, 11),
        "cycle_fade": np.linspace(0.0, 1.0, 11) * 1e-5,
    }


@pytest.fixture
def convex_coeffs():
    """Quadratic f(DoD) = 1e-5 × DoD² — convex, slope-ratio ≈ 19×."""
    dod = np.linspace(0.0, 1.0, 11)
    return {"dod_breakpoints": dod, "cycle_fade": (dod ** 2) * 1e-5}


# ---------------------------------------------------------------------------
# Rainflow extraction
# ---------------------------------------------------------------------------


def test_rainflow_zero_signal():
    assert extract_rainflow_cycles(np.zeros(10)) == []


def test_rainflow_monotonic_ramp_residue_only():
    cycles = extract_rainflow_cycles(np.linspace(0.0, 1.0, 5))
    assert len(cycles) == 1  # single residue half-cycle covering full range
    assert cycles[0][0] == pytest.approx(1.0)
    assert cycles[0][1] == pytest.approx(0.5)


def test_rainflow_repeated_full_cycles_count():
    """Eight up-down cycles between 0.2 and 0.8 should yield 8 closed cycles
    (plus a residue half-cycle from the open ends)."""
    sig = np.array([0.2, 0.8] * 8 + [0.2])
    cycles = extract_rainflow_cycles(sig)
    full_cycles = [c for c in cycles if c[0] == pytest.approx(0.6, abs=1e-9)]
    assert len(full_cycles) >= 8


def test_rainflow_shallow_oscillation_uniform():
    """Uniform shallow ±0.05 oscillations: every extracted cycle has range = 0.10."""
    sig = np.array([0.5 + (0.05 if i % 2 == 0 else -0.05) for i in range(20)])
    cycles = extract_rainflow_cycles(sig)
    assert all(c[0] == pytest.approx(0.10, abs=1e-9) for c in cycles)


# ---------------------------------------------------------------------------
# Piecewise-linear evaluation
# ---------------------------------------------------------------------------


def test_piecewise_linear_eval_inside_range(linear_coeffs):
    # f(0.5) = 0.5 × 1e-5 = 5e-6
    val = evaluate_piecewise(0.5, linear_coeffs["dod_breakpoints"], linear_coeffs["cycle_fade"])
    assert val == pytest.approx(5e-6, rel=1e-9)


def test_piecewise_linear_eval_clamps_outside_range(linear_coeffs):
    # below 0 clamps to value at 0
    val_below = evaluate_piecewise(-0.5, linear_coeffs["dod_breakpoints"], linear_coeffs["cycle_fade"])
    assert val_below == pytest.approx(0.0)
    # above 1 clamps to value at 1
    val_above = evaluate_piecewise(1.5, linear_coeffs["dod_breakpoints"], linear_coeffs["cycle_fade"])
    assert val_above == pytest.approx(1e-5)


# ---------------------------------------------------------------------------
# Wear function — semantics
# ---------------------------------------------------------------------------


def test_zero_fec_returns_zero_wear(linear_coeffs):
    soc = np.full(96, 1.0)  # 0.5 fraction at energy=2.0
    day = _build_day(soc, fec=0.0)
    wear = rainflow_wear_from_duty(
        day_result=day, energy_mwh=2.0, soh_current=1.0,
        rainflow_coeffs=linear_coeffs,
    )
    assert wear == 0.0


def test_one_full_cycle_a_day_yields_finite_wear(linear_coeffs):
    """1 full cycle/day at SoH=1 should produce non-zero, well-bounded wear."""
    soc = np.concatenate([np.linspace(0.4, 1.6, 48),
                          np.linspace(1.6, 0.4, 48)])
    day = _build_day(soc, fec=1.0)
    wear = rainflow_wear_from_duty(
        day_result=day, energy_mwh=2.0, soh_current=1.0,
        rainflow_coeffs=linear_coeffs,
    )
    assert 0.0 < wear < 500.0  # within max_wear cap


def test_convex_kernel_penalises_deeper_cycles_more(convex_coeffs):
    """Per-MWh wear must be HIGHER on a single deep cycle than on many
    shallow cycles delivering the same throughput, because f(DoD) is convex.

    This is the core methodology test: it captures the channel-(b)
    distinction that scalar wear and Collath's throughput-per-window form
    cannot capture.
    """
    # Throughput-equivalent scenarios:
    #   A: one full cycle 0 -> 2.0 -> 0  (range 1.0 in fraction)
    #   B: ten cycles 0 -> 0.2 -> 0      (each range 0.1 in fraction,
    #       same total throughput)
    soc_deep = np.concatenate([np.linspace(0.0, 2.0, 48), np.linspace(2.0, 0.0, 48)])
    # 10 shallow swings between 0 and 0.4 over 96 steps.
    # 10 cycles → 20 reversal points → split 96 steps into 20 segments
    one_cycle_steps = 96 // 20  # ~4.8, round down to 4 (some leftover)
    shallow_segments = []
    for k in range(20):
        if k % 2 == 0:
            shallow_segments.append(np.linspace(0.0, 0.4, one_cycle_steps,
                                                 endpoint=False))
        else:
            shallow_segments.append(np.linspace(0.4, 0.0, one_cycle_steps,
                                                 endpoint=False))
    soc_shallow = np.concatenate(shallow_segments)
    # Pad to 96
    if len(soc_shallow) < 96:
        soc_shallow = np.concatenate([soc_shallow,
                                       np.full(96 - len(soc_shallow), soc_shallow[-1])])
    else:
        soc_shallow = soc_shallow[:96]

    # FEC at energy=2 MWh: throughput / (2 × E_usable)
    # Deep: total throughput ≈ 4.0 MWh → FEC ≈ 1.0
    # Shallow: 10 × 0.4 × 2 = 8 MWh in fraction terms / (2×2) = 1.0 (matched)
    deep_day = _build_day(soc_deep, fec=1.0)
    shallow_day = _build_day(soc_shallow, fec=1.0)

    deep_wear = rainflow_wear_from_duty(
        day_result=deep_day, energy_mwh=2.0, soh_current=1.0,
        rainflow_coeffs=convex_coeffs,
    )
    shallow_wear = rainflow_wear_from_duty(
        day_result=shallow_day, energy_mwh=2.0, soh_current=1.0,
        rainflow_coeffs=convex_coeffs,
    )
    assert deep_wear > shallow_wear * 1.5, (
        f"convex f(DoD) should penalise deep cycles more than shallow at "
        f"same throughput. deep_wear={deep_wear:.2f}, shallow_wear={shallow_wear:.2f}"
    )


def test_age_accel_increases_wear_at_low_soh(linear_coeffs):
    """At SoH=0.85 (15 % aged), wear should be ~1.375× of fresh-cell wear,
    AFTER accounting for the headroom shrinkage from 0.20 to 0.05."""
    soc = np.concatenate([np.linspace(0.4, 1.6, 48),
                          np.linspace(1.6, 0.4, 48)])
    day = _build_day(soc, fec=1.0)
    wear_fresh = rainflow_wear_from_duty(
        day_result=day, energy_mwh=2.0, soh_current=1.0,
        rainflow_coeffs=linear_coeffs,
    )
    wear_aged = rainflow_wear_from_duty(
        day_result=day, energy_mwh=2.0, soh_current=0.85,
        rainflow_coeffs=linear_coeffs,
    )
    assert wear_aged > wear_fresh
    # Capped, but ratio is a healthy multiple
    assert wear_aged / max(wear_fresh, 1e-9) > 2.0


def test_max_wear_cap_respected(convex_coeffs):
    """Even with diverging headroom near the warranty floor, the wear caps
    out at max_wear_eur_per_mwh."""
    soc = np.concatenate([np.linspace(0.0, 2.0, 48),
                          np.linspace(2.0, 0.0, 48)])
    day = _build_day(soc, fec=1.0)
    wear = rainflow_wear_from_duty(
        day_result=day, energy_mwh=2.0, soh_current=0.81,  # very close to floor
        rainflow_coeffs=convex_coeffs,
        max_wear_eur_per_mwh=200.0,
    )
    assert wear <= 200.0


# ---------------------------------------------------------------------------
# Policy integration
# ---------------------------------------------------------------------------


def test_rainflow_policy_two_pass_signature(linear_coeffs):
    pol = RainflowPolicy(rainflow_coeffs=linear_coeffs)
    assert pol.uses_two_pass is True
    overrides = pol.lp_overrides(soh_current=1.0)
    assert overrides == {"soc_min_frac": 0.20, "soc_max_frac": 0.80}


def test_rainflow_policy_pass1_returns_bootstrap(linear_coeffs):
    pol = RainflowPolicy(rainflow_coeffs=linear_coeffs,
                         bootstrap_wear_eur_per_mwh=42.0)
    wear = pol.wear_cost(soh_current=1.0, periods_per_day=96)
    assert wear.shape == (96,)
    assert np.allclose(wear, 42.0)


def test_rainflow_policy_pass2_uses_rainflow(linear_coeffs):
    pol = RainflowPolicy(rainflow_coeffs=linear_coeffs)
    soc = np.concatenate([np.linspace(0.4, 1.6, 48),
                          np.linspace(1.6, 0.4, 48)])
    day = _build_day(soc, fec=1.0)
    refined = pol.refine_wear_cost(
        first_pass_soc=soc, energy_mwh=2.0, soh_current=1.0,
        day_result=day, periods_per_day=96,
    )
    # Should be a flat scalar vector ≠ bootstrap (rainflow > 0 on a real cycle)
    assert refined.shape == (96,)
    assert np.allclose(refined, refined[0])  # flat
    assert refined[0] > 0.0


def test_rainflow_policy_validates_coeffs():
    with pytest.raises(ValueError, match="dod_breakpoints"):
        RainflowPolicy(rainflow_coeffs={"foo": np.zeros(3), "bar": np.zeros(3)})
