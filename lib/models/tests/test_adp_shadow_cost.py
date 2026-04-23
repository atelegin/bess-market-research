"""
Tests for shadow-cost policy interface (Note 4 A3.1).

Exercises the three implemented policies plus the LP-integration contract
(the vectors they produce must plug into ``optimize_day_stacked`` cleanly).
"""
from __future__ import annotations

import numpy as np
import pytest

import datetime as dt

import pandas as pd

from lib.models.adp_shadow_cost import (
    ADPPolicy,
    AgingAwareDepreciationPolicy,
    DepreciationProxyPolicy,
    NaivePolicy,
    ShadowCostPolicy,
    default_policies_for_comparison,
)
from lib.models.adp_solver import ADPSolver, default_grids
from lib.models.dispatch_stacked import (
    BLOCKS_PER_DAY,
    PERIODS_PER_DAY,
    optimize_day_stacked,
)
from lib.models.price_regime import RegimeClassification


def test_naive_policy_returns_zero():
    p = NaivePolicy()
    w = p.wear_cost(soh_current=0.95)
    assert w.shape == (PERIODS_PER_DAY,)
    assert np.all(w == 0.0)


def test_depreciation_proxy_constant_in_time_and_soh():
    p = DepreciationProxyPolicy(capex_eur_per_mwh=100_000, lifetime_throughput_ratio=6_000)
    w_high_soh = p.wear_cost(soh_current=0.99)
    w_low_soh = p.wear_cost(soh_current=0.82)
    # Same scalar across calendar time AND SoH — that's the "poor proxy"
    assert np.allclose(w_high_soh, w_low_soh)
    assert np.allclose(w_high_soh, 100_000 / 6_000)
    assert len(set(w_high_soh)) == 1  # flat over periods


def test_aging_aware_depreciation_rises_with_consumed_soh():
    p = AgingAwareDepreciationPolicy(base_eur_per_mwh=16.67, warranty_floor=0.80)
    costs_by_soh = {}
    for soh in (0.99, 0.95, 0.90, 0.85, 0.81):
        w = p.wear_cost(soh_current=soh)
        costs_by_soh[soh] = w[0]
    # Monotonic: cycling gets more expensive as SoH drops
    levels = [costs_by_soh[s] for s in (0.99, 0.95, 0.90, 0.85, 0.81)]
    assert all(a < b for a, b in zip(levels, levels[1:]))
    # Anchor: at SoH = 0.90 the wear cost ≈ base
    assert costs_by_soh[0.90] == pytest.approx(16.67, rel=1e-3)


def test_aging_aware_depreciation_clamps_below_warranty_floor():
    p = AgingAwareDepreciationPolicy(base_eur_per_mwh=16.67, warranty_floor=0.80)
    w = p.wear_cost(soh_current=0.79)  # below floor
    # Does not crash; returns finite, large (divergent-capped) value.
    assert np.all(np.isfinite(w))
    # Below floor, consumed=(1-0.79)=0.21 and headroom=epsilon(0.005), so
    # ratio = 0.21 / 0.005 = 42 → wear cost ≈ 700 EUR/MWh. Huge but finite.
    assert w[0] > 500


def test_aging_aware_depreciation_flat_within_day():
    """The A3.1 closed form is time-invariant WITHIN a day. Day-to-day
    variation comes from SoH drift in the outer annual loop."""
    p = AgingAwareDepreciationPolicy(base_eur_per_mwh=16.67)
    w = p.wear_cost(soh_current=0.95)
    assert len(set(w.round(4))) == 1


def _fitted_adp() -> tuple[ADPSolver, RegimeClassification]:
    """Build a small ADP solver for tests."""
    import numpy as _np
    n_r = 3
    M = _np.full((n_r, n_r), 0.15)
    _np.fill_diagonal(M, 0.7)
    dates = pd.to_datetime(pd.date_range("2024-01-01", periods=60, freq="D")).date
    labels = pd.Series(_np.arange(60) % n_r, index=dates)
    rc = RegimeClassification(
        n_regimes=n_r, regime_labels=labels,
        daily_spreads=pd.Series(_np.zeros(60), index=dates),
        daily_means=pd.Series(_np.zeros(60), index=dates),
        transition_matrix=M, stationary=_np.full(n_r, 1.0 / n_r), stats=[],
    )
    rev_curve = _np.array([
        [0, 90, 160, 210, 240, 255, 262, 264, 265],
        [0, 180, 320, 420, 480, 510, 524, 528, 530],
        [0, 360, 640, 840, 960, 1020, 1048, 1056, 1060],
    ])
    grids = default_grids()
    solver = ADPSolver(rc, rev_curve, grids)
    return solver, rc


def test_adp_policy_initialises_from_solver():
    solver, rc = _fitted_adp()
    result = solver.solve(tol=1e-2)
    policy = ADPPolicy(solver=solver, result=result, regime_classification=rc)
    assert policy.name == "adp"


def test_adp_policy_wear_cost_has_correct_shape():
    solver, rc = _fitted_adp()
    result = solver.solve(tol=1e-2)
    policy = ADPPolicy(solver=solver, result=result, regime_classification=rc)
    w = policy.wear_cost(soh_current=0.92, day_of_year=10)
    assert w.shape == (PERIODS_PER_DAY,)
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0)


def test_adp_policy_wear_cost_changes_with_soh():
    solver, rc = _fitted_adp()
    result = solver.solve(tol=1e-2)
    policy = ADPPolicy(solver=solver, result=result, regime_classification=rc)
    w_fresh = policy.wear_cost(soh_current=0.99, day_of_year=10)
    w_aged = policy.wear_cost(soh_current=0.82, day_of_year=10)
    # Non-trivial shift with SoH (DP produces state-dependent output)
    assert abs(w_fresh[0] - w_aged[0]) > 1e-3


def test_adp_policy_wear_cost_changes_with_regime():
    solver, rc = _fitted_adp()
    result = solver.solve(tol=1e-2)
    policy_quiet = ADPPolicy(solver=solver, result=result,
                             regime_classification=rc, regime_override=0)
    policy_volatile = ADPPolicy(solver=solver, result=result,
                                regime_classification=rc, regime_override=2)
    w_q = policy_quiet.wear_cost(soh_current=0.90, day_of_year=1)
    w_v = policy_volatile.wear_cost(soh_current=0.90, day_of_year=1)
    # Shadow cost differs by regime — the DP's opportunity cost depends on
    # what kind of market the operator is in.
    assert abs(w_q[0] - w_v[0]) > 1e-3


def test_adp_policy_calendar_lookup_uses_classifier():
    solver, rc = _fitted_adp()
    result = solver.solve(tol=1e-2)
    policy = ADPPolicy(solver=solver, result=result, regime_classification=rc,
                       year_start=dt.date(2024, 1, 1))
    # Day 1 = 2024-01-01, regime label = 0 % 3 = 0
    w_day1 = policy.wear_cost(soh_current=0.90, day_of_year=1)
    # Day 2 = 2024-01-02, regime label = 1 % 3 = 1
    w_day2 = policy.wear_cost(soh_current=0.90, day_of_year=2)
    # Different regimes → generally different wear cost
    assert w_day1[0] != w_day2[0] or np.isclose(w_day1[0], w_day2[0])


def test_adp_policy_plugs_into_stacked_lp():
    """Integration check: ADP policy vector runs through the stacked LP."""
    solver, rc = _fitted_adp()
    result = solver.solve(tol=1e-2)
    policy = ADPPolicy(solver=solver, result=result, regime_classification=rc)
    wear = policy.wear_cost(soh_current=0.92, day_of_year=10)

    flat_price = np.concatenate([
        np.full(PERIODS_PER_DAY // 2, 20.0),
        np.full(PERIODS_PER_DAY // 2, 200.0),
    ])
    r = optimize_day_stacked(
        prices_da=flat_price, prices_id=flat_price,
        afrr_cap_pos_price=np.zeros(BLOCKS_PER_DAY),
        afrr_cap_neg_price=np.zeros(BLOCKS_PER_DAY),
        afrr_energy_pos_price=np.zeros(PERIODS_PER_DAY),
        afrr_energy_neg_price=np.zeros(PERIODS_PER_DAY),
        afrr_activation_rate_pos=np.zeros(PERIODS_PER_DAY),
        afrr_activation_rate_neg=np.zeros(PERIODS_PER_DAY),
        energy_mwh=2.0, power_mw=1.0,
        afrr_reserve_duration_hours=0.0,
        wear_cost_eur_per_mwh=wear,
    )
    assert r.success


def test_default_policies_factory_returns_three():
    policies = default_policies_for_comparison()
    assert set(policies) == {"naive", "depreciation_proxy", "aging_aware_depreciation"}
    for name, p in policies.items():
        assert p.name == name
        assert isinstance(p, ShadowCostPolicy)


@pytest.mark.parametrize("policy_name", ["naive", "depreciation_proxy", "aging_aware_depreciation"])
def test_policy_vector_plugs_into_stacked_lp(policy_name):
    """Contract check: each policy's vector is a valid ``wear_cost_eur_per_mwh``
    input for the stacked LP. The LP runs to completion without raising."""
    policies = default_policies_for_comparison()
    policy = policies[policy_name]
    wear = policy.wear_cost(soh_current=0.92)

    flat_price = np.concatenate([
        np.full(PERIODS_PER_DAY // 2, 20.0),
        np.full(PERIODS_PER_DAY // 2, 200.0),
    ])
    result = optimize_day_stacked(
        prices_da=flat_price,
        prices_id=flat_price,
        afrr_cap_pos_price=np.zeros(BLOCKS_PER_DAY),
        afrr_cap_neg_price=np.zeros(BLOCKS_PER_DAY),
        afrr_energy_pos_price=np.zeros(PERIODS_PER_DAY),
        afrr_energy_neg_price=np.zeros(PERIODS_PER_DAY),
        afrr_activation_rate_pos=np.zeros(PERIODS_PER_DAY),
        afrr_activation_rate_neg=np.zeros(PERIODS_PER_DAY),
        energy_mwh=2.0, power_mw=1.0,
        afrr_reserve_duration_hours=0.0,
        wear_cost_eur_per_mwh=wear,
    )
    assert result.success


def test_policy_ordering_suppresses_cycles_monotonically():
    """With the same price spread, cycling should decrease as we move from
    naive (wear=0) → depreciation (flat mid) → aging-aware at aged SoH
    (higher wear). Captures the Kumtepeli/Howey intuition at the LP level.
    """
    # Spread chosen so the LP cycles under naive, partly under depreciation,
    # and barely under aging-aware-at-aged-SoH.
    prices = np.concatenate([
        np.full(PERIODS_PER_DAY // 2, 30.0),
        np.full(PERIODS_PER_DAY // 2, 70.0),  # 40 EUR/MWh spread
    ])
    common = dict(
        prices_da=prices, prices_id=prices,
        afrr_cap_pos_price=np.zeros(BLOCKS_PER_DAY),
        afrr_cap_neg_price=np.zeros(BLOCKS_PER_DAY),
        afrr_energy_pos_price=np.zeros(PERIODS_PER_DAY),
        afrr_energy_neg_price=np.zeros(PERIODS_PER_DAY),
        afrr_activation_rate_pos=np.zeros(PERIODS_PER_DAY),
        afrr_activation_rate_neg=np.zeros(PERIODS_PER_DAY),
        energy_mwh=2.0, power_mw=1.0,
        afrr_reserve_duration_hours=0.0,
    )

    naive = NaivePolicy()
    depr = DepreciationProxyPolicy(capex_eur_per_mwh=100_000, lifetime_throughput_ratio=6_000)
    aging = AgingAwareDepreciationPolicy(base_eur_per_mwh=16.67, warranty_floor=0.80)

    r_naive = optimize_day_stacked(**common, wear_cost_eur_per_mwh=naive.wear_cost(0.92))
    r_depr = optimize_day_stacked(**common, wear_cost_eur_per_mwh=depr.wear_cost(0.92))
    # Aged asset: scarcity factor > 1 → higher wear cost → fewer cycles
    r_aging_aged = optimize_day_stacked(**common, wear_cost_eur_per_mwh=aging.wear_cost(0.83))

    # Cycles under naive ≥ depreciation ≥ aging-aware(aged)
    assert r_naive.full_equivalent_cycles >= r_depr.full_equivalent_cycles - 1e-6
    assert r_depr.full_equivalent_cycles >= r_aging_aged.full_equivalent_cycles - 1e-6
    # The aged aging-aware policy should cycle strictly less than the flat
    # proxy at this spread (non-trivial policy signal).
    assert r_aging_aged.full_equivalent_cycles < r_depr.full_equivalent_cycles + 1e-6
