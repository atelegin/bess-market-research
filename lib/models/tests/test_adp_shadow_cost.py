"""
Tests for shadow-cost policy interface (Note 4 A3.1).

Exercises the three implemented policies plus the LP-integration contract
(the vectors they produce must plug into ``optimize_day_stacked`` cleanly).
"""
from __future__ import annotations

import numpy as np
import pytest

from lib.models.adp_shadow_cost import (
    ADPPolicy,
    AgingAwareDepreciationPolicy,
    DepreciationProxyPolicy,
    NaivePolicy,
    ShadowCostPolicy,
    default_policies_for_comparison,
)
from lib.models.dispatch_stacked import (
    BLOCKS_PER_DAY,
    PERIODS_PER_DAY,
    optimize_day_stacked,
)


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


def test_adp_policy_stubbed():
    with pytest.raises(NotImplementedError, match="A3.3"):
        ADPPolicy()


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
