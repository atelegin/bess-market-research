"""
Tests for backward-induction DP solver (Note 4 A3.3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lib.models.adp_solver import (
    ADPGrids,
    ADPResult,
    ADPSolver,
    default_grids,
    degradation_per_day,
    empirical_daily_revenue_curve,
)
from lib.models.price_regime import RegimeClassification


def _trivial_classification(n_regimes: int = 3, persistent: float = 0.7) -> RegimeClassification:
    """Synthesise a RegimeClassification with balanced stationary dist."""
    off = (1 - persistent) / (n_regimes - 1)
    M = np.full((n_regimes, n_regimes), off)
    np.fill_diagonal(M, persistent)
    dates = pd.to_datetime(pd.date_range("2024-01-01", periods=365, freq="D")).date
    labels = pd.Series(np.arange(365) % n_regimes, index=dates)
    return RegimeClassification(
        n_regimes=n_regimes,
        regime_labels=labels,
        daily_spreads=pd.Series(np.zeros(365), index=dates),
        daily_means=pd.Series(np.zeros(365), index=dates),
        transition_matrix=M,
        stationary=np.full(n_regimes, 1.0 / n_regimes),
        stats=[],
    )


def test_degradation_per_day_accelerates_with_intensity():
    d_low = degradation_per_day(intensity=0.5, soh=0.95, fade_per_fec_at_soh_1=2e-4)
    d_high = degradation_per_day(intensity=2.0, soh=0.95, fade_per_fec_at_soh_1=2e-4)
    assert d_high > d_low
    # Roughly linear
    assert d_high == pytest.approx(d_low * 4, rel=0.1)


def test_degradation_per_day_accelerates_at_low_soh():
    d_fresh = degradation_per_day(intensity=1.0, soh=1.00, fade_per_fec_at_soh_1=2e-4)
    d_aged = degradation_per_day(intensity=1.0, soh=0.82, fade_per_fec_at_soh_1=2e-4)
    assert d_aged > d_fresh


def test_empirical_revenue_curve_scales_linearly():
    idx = pd.to_datetime(pd.date_range("2024-01-01", periods=30, freq="D")).date
    rev = pd.Series(np.linspace(300, 700, 30), index=idx)
    # All in regime 0
    labels = pd.Series(np.zeros(30, dtype=int), index=idx)
    curve = empirical_daily_revenue_curve(
        daily_revenue=rev, regime_labels=labels, n_regimes=1,
        action_grid=np.array([0.5, 1.0, 1.5, 2.0]),
        reference_intensity=1.0,
    )
    assert curve.shape == (1, 4)
    # Reference intensity (1.0) → mean revenue
    assert curve[0, 1] == pytest.approx(rev.mean(), rel=1e-6)
    # Double intensity → double revenue (linear approximation)
    assert curve[0, 3] == pytest.approx(2.0 * rev.mean(), rel=1e-6)


def test_solver_converges_with_simple_inputs():
    grids = default_grids()
    rc = _trivial_classification()
    rev_curve = np.array([
        [0, 100, 180, 240, 280, 300, 310, 315, 318],   # quiet regime
        [0, 200, 360, 480, 560, 600, 620, 630, 636],   # normal
        [0, 400, 720, 960, 1120, 1200, 1240, 1260, 1272],  # volatile
    ])
    assert rev_curve.shape == (3, 9)
    solver = ADPSolver(
        regime_classification=rc, revenue_curve=rev_curve, grids=grids,
        fade_per_fec_at_soh_1=2e-4, calendar_fade_per_day=2e-5,
        discount_per_year=0.98,
    )
    result = solver.solve(tol=1e-2, max_iter=2000)
    assert result.converged
    assert result.value.shape == (grids.n_soh, rc.n_regimes)
    assert result.policy.shape == (grids.n_soh, rc.n_regimes)
    assert result.shadow_cost.shape == result.value.shape


def test_value_higher_at_higher_soh():
    """With more life left, expected remaining value is greater."""
    grids = default_grids()
    rc = _trivial_classification()
    rev_curve = np.array([
        [0, 100, 180, 240, 280, 300, 310, 315, 318],
        [0, 200, 360, 480, 560, 600, 620, 630, 636],
        [0, 400, 720, 960, 1120, 1200, 1240, 1260, 1272],
    ])
    solver = ADPSolver(rc, rev_curve, grids)
    result = solver.solve(tol=1e-2)
    # Pick volatile regime (highest revenue); value at high SoH > low SoH
    V_col = result.value[:, 2]
    assert all(a <= b + 1e-6 for a, b in zip(V_col, V_col[1:]))


def test_value_higher_in_volatile_regime():
    """At the same SoH, volatile regime (higher spreads) has higher value."""
    grids = default_grids()
    rc = _trivial_classification()
    rev_curve = np.array([
        [0, 100, 180, 240, 280, 300, 310, 315, 318],
        [0, 200, 360, 480, 560, 600, 620, 630, 636],
        [0, 400, 720, 960, 1120, 1200, 1240, 1260, 1272],
    ])
    solver = ADPSolver(rc, rev_curve, grids)
    result = solver.solve(tol=1e-2)
    # At mid SoH (idx 5), V increases with regime volatility
    mid = result.value.shape[0] // 2
    assert result.value[mid, 0] < result.value[mid, 1] < result.value[mid, 2]


def test_optimal_policy_more_aggressive_in_volatile_regime():
    """In volatile regimes where revenue per cycle is larger, optimal
    intensity is higher."""
    grids = default_grids()
    rc = _trivial_classification()
    rev_curve = np.array([
        [0, 100, 180, 240, 280, 300, 310, 315, 318],
        [0, 200, 360, 480, 560, 600, 620, 630, 636],
        [0, 400, 720, 960, 1120, 1200, 1240, 1260, 1272],
    ])
    solver = ADPSolver(rc, rev_curve, grids)
    result = solver.solve(tol=1e-2)
    # Mid SoH: quiet ≤ normal ≤ volatile intensity
    mid = result.value.shape[0] // 2
    assert result.policy[mid, 0] <= result.policy[mid, 1] <= result.policy[mid, 2]


def test_shadow_cost_non_negative():
    grids = default_grids()
    rc = _trivial_classification()
    rev_curve = np.array([
        [0, 100, 180, 240, 280, 300, 310, 315, 318],
        [0, 200, 360, 480, 560, 600, 620, 630, 636],
        [0, 400, 720, 960, 1120, 1200, 1240, 1260, 1272],
    ])
    solver = ADPSolver(rc, rev_curve, grids)
    result = solver.solve(tol=1e-2)
    assert np.all(result.shadow_cost >= 0)


def test_warranty_penalty_lowers_value_function():
    """Adding a warranty-breach penalty must make V everywhere more
    negative (or at least not more positive)."""
    grids = default_grids()
    rc = _trivial_classification()
    rev_curve = np.array([
        [0, 90, 160, 210, 240, 255, 262, 264, 265],
        [0, 180, 320, 420, 480, 510, 524, 528, 530],
        [0, 360, 640, 840, 960, 1020, 1048, 1056, 1060],
    ])
    no_pen = ADPSolver(rc, rev_curve, grids, warranty_breach_penalty_eur=0)
    with_pen = ADPSolver(rc, rev_curve, grids, warranty_breach_penalty_eur=500_000)
    r_no = no_pen.solve(tol=1e-2)
    r_w = with_pen.solve(tol=1e-2)
    assert np.all(r_w.value <= r_no.value + 1e-6)


def test_shadow_cost_varies_across_soh_grid():
    """Regardless of curve shape, shadow cost must depend on SoH — a
    constant shadow cost signals a DP that's effectively the flat
    depreciation proxy (nothing learned from the DP structure)."""
    grids = default_grids()
    rc = _trivial_classification()
    rev_curve = np.array([
        [0, 90, 160, 210, 240, 255, 262, 264, 265],
        [0, 180, 320, 420, 480, 510, 524, 528, 530],
        [0, 360, 640, 840, 960, 1020, 1048, 1056, 1060],
    ])
    solver = ADPSolver(rc, rev_curve, grids)
    result = solver.solve(tol=1e-2)
    # Non-trivial variation across SoH (> 1% ratio)
    sc = result.shadow_cost[:, 1]
    ratio = sc.max() / max(sc.min(), 1e-6)
    assert ratio > 1.01


def test_grid_validation_rejects_descending_soh():
    with pytest.raises(ValueError, match="strictly ascending"):
        ADPGrids(
            soh_grid=np.array([1.0, 0.95, 0.90]),
            action_grid=np.array([0, 1, 2]),
        )


def test_grid_validation_rejects_soh_below_floor():
    with pytest.raises(ValueError, match="below warranty_floor"):
        ADPGrids(
            soh_grid=np.array([0.75, 0.80, 0.85]),
            action_grid=np.array([0, 1, 2]),
            warranty_floor=0.80,
        )


def test_revenue_curve_shape_validation():
    grids = default_grids()
    rc = _trivial_classification(n_regimes=3)
    # Wrong regime dim
    bad_curve = np.zeros((2, grids.n_actions))
    with pytest.raises(ValueError, match="regime dim"):
        ADPSolver(rc, bad_curve, grids)
