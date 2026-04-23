"""
Tests for stacked-market LP dispatch (Note 4 A2.1).

Exercises the LP on synthetic inputs. Goals:
  1. Smoke: LP solves on well-posed input.
  2. Degenerate: when aFRR prices & activations are zero, reduces to DA+ID
     arbitrage (sanity vs the simpler dispatch).
  3. Conjugate coupling: when aFRR NEG gives "free charge" and DA has a
     later peak, the LP exploits both — reserves NEG, absorbs energy,
     discharges at DA peak.
  4. Reservation blocks power: when r_pos high, DA discharge capped below
     nameplate.
  5. SoC reservation headroom: operator cannot discharge below soc_min +
     reserve buffer.
"""
from __future__ import annotations

import numpy as np
import pytest

from lib.models.dispatch_stacked import (
    PERIODS_PER_DAY,
    BLOCKS_PER_DAY,
    optimize_day_stacked,
)


def _zeros_day():
    return np.zeros(PERIODS_PER_DAY)


def _zeros_block():
    return np.zeros(BLOCKS_PER_DAY)


def test_smoke_lp_solves_with_flat_inputs():
    flat = np.full(PERIODS_PER_DAY, 50.0)
    result = optimize_day_stacked(
        prices_da=flat, prices_id=flat,
        afrr_cap_pos_price=np.full(BLOCKS_PER_DAY, 5.0),
        afrr_cap_neg_price=np.full(BLOCKS_PER_DAY, 5.0),
        afrr_energy_pos_price=np.full(PERIODS_PER_DAY, 100.0),
        afrr_energy_neg_price=np.full(PERIODS_PER_DAY, -100.0),
        afrr_activation_rate_pos=np.full(PERIODS_PER_DAY, 0.03),
        afrr_activation_rate_neg=np.full(PERIODS_PER_DAY, 0.03),
        energy_mwh=2.0, power_mw=1.0,
    )
    assert result.success
    # With flat DA/ID prices, DA and ID arbitrage revenue ≈ 0 (no spread)
    # But aFRR cap pays symmetric reservations regardless of activation economics.
    assert result.revenue_total > 0


def test_degenerate_no_afrr_markets_reduces_to_energy_arbitrage():
    """When aFRR cap prices are zero AND activation is zero, LP reduces to
    DA+ID energy arbitrage. With a bimodal signal the LP charges in the
    low block and discharges in the high. Identical DA and ID prices make
    the LP's DA/ID split degenerate (either works); we check the combined
    schedule instead."""
    prices_da = np.concatenate([
        np.full(PERIODS_PER_DAY // 2, 20.0),   # morning cheap
        np.full(PERIODS_PER_DAY // 2, 200.0),  # afternoon peak
    ])
    result = optimize_day_stacked(
        prices_da=prices_da, prices_id=prices_da,
        afrr_cap_pos_price=_zeros_block(),
        afrr_cap_neg_price=_zeros_block(),
        afrr_energy_pos_price=_zeros_day(),
        afrr_energy_neg_price=_zeros_day(),
        afrr_activation_rate_pos=_zeros_day(),
        afrr_activation_rate_neg=_zeros_day(),
        energy_mwh=2.0, power_mw=1.0,
        afrr_reserve_duration_hours=0.0,
    )
    assert result.success
    morning_charge = (result.charge_da + result.charge_id)[:PERIODS_PER_DAY // 2].sum()
    afternoon_discharge = (result.discharge_da + result.discharge_id)[PERIODS_PER_DAY // 2:].sum()
    assert morning_charge > 0
    assert afternoon_discharge > 0
    # Combined arbitrage revenue is positive (spread beats eta losses)
    assert result.revenue_da + result.revenue_id > 0
    # No aFRR activity whatsoever
    assert np.allclose(result.r_pos, 0)
    assert np.allclose(result.r_neg, 0)


def test_afrr_cap_alone_is_profitable_at_positive_price():
    """With zero activation prices/rates and a positive cap price, the LP
    should reserve aFRR capacity up to the nameplate — pure option value
    since there's no actual energy flow."""
    result = optimize_day_stacked(
        prices_da=_zeros_day(), prices_id=_zeros_day(),
        afrr_cap_pos_price=np.full(BLOCKS_PER_DAY, 10.0),
        afrr_cap_neg_price=np.full(BLOCKS_PER_DAY, 10.0),
        afrr_energy_pos_price=_zeros_day(),
        afrr_energy_neg_price=_zeros_day(),
        afrr_activation_rate_pos=_zeros_day(),
        afrr_activation_rate_neg=_zeros_day(),
        energy_mwh=2.0, power_mw=1.0,
    )
    assert result.success
    # Both directions reserved at nameplate → 24h of 1 MW × 10 EUR = 240 EUR
    # (6 blocks × 4h × 1 MW × 10 EUR/MW/h each direction = 240 each)
    assert result.revenue_afrr_cap_pos == pytest.approx(240.0, rel=1e-3)
    assert result.revenue_afrr_cap_neg == pytest.approx(240.0, rel=1e-3)
    # r should be at nameplate across all blocks
    assert np.allclose(result.r_pos, 1.0)
    assert np.allclose(result.r_neg, 1.0)


def test_reservation_blocks_power():
    """If operator reserves r_pos=1 MW (full nameplate) for aFRR, they
    cannot discharge any DA/ID energy — reservation consumes the power
    budget regardless of activation outcome."""
    prices_da = np.concatenate([
        np.full(PERIODS_PER_DAY // 2, 20.0),
        np.full(PERIODS_PER_DAY // 2, 200.0),
    ])
    # Huge aFRR cap price forces full reservation
    result = optimize_day_stacked(
        prices_da=prices_da, prices_id=prices_da,
        afrr_cap_pos_price=np.full(BLOCKS_PER_DAY, 1000.0),  # dominant incentive
        afrr_cap_neg_price=np.full(BLOCKS_PER_DAY, 1000.0),
        afrr_energy_pos_price=_zeros_day(),
        afrr_energy_neg_price=_zeros_day(),
        afrr_activation_rate_pos=_zeros_day(),
        afrr_activation_rate_neg=_zeros_day(),
        energy_mwh=2.0, power_mw=1.0,
    )
    assert result.success
    assert np.allclose(result.r_pos, 1.0)
    assert np.allclose(result.r_neg, 1.0)
    # DA/ID cannot move energy — power budget exhausted
    assert result.discharge_da.sum() == pytest.approx(0.0, abs=1e-6)
    assert result.charge_da.sum() == pytest.approx(0.0, abs=1e-6)


def test_conjugate_coupling_afrr_neg_subsidises_wholesale():
    """Demonstrates the aFRR↔wholesale coupling: NEG activation price
    moderately negative (operator pays to charge) in morning but DA peak
    very high in afternoon. If the absorbed energy's wholesale resale
    value exceeds the NEG cost plus eta losses, the LP should reserve
    aFRR NEG and route the "free" charge into afternoon peak discharge.

    Activation rate kept small (5%) so absorbed energy fits the 2 MWh
    battery within one morning block."""
    # DA: cheap morning (50), expensive afternoon (400) — wide spread gives
    # NEG-subsidised charge strong value.
    prices_da = np.concatenate([
        np.full(PERIODS_PER_DAY // 2, 50.0),
        np.full(PERIODS_PER_DAY // 2, 400.0),
    ])
    # NEG price -5 EUR/MWh in morning. At α=0.05, per 15-min interval
    # a_neg = 0.05 × 1 × 0.25 = 0.0125 MWh; 48 intervals → 0.6 MWh total.
    # NEG cost = 0.6 × -5 = -3 EUR. Resale at 400 − eta loss = ~370/MWh
    # → revenue ~220 EUR. Massive net benefit.
    afrr_energy_neg = np.concatenate([
        np.full(PERIODS_PER_DAY // 2, -5.0),
        np.zeros(PERIODS_PER_DAY // 2),
    ])
    act_neg = np.concatenate([
        np.full(PERIODS_PER_DAY // 2, 0.05),  # 5% activation
        np.zeros(PERIODS_PER_DAY // 2),
    ])
    baseline = optimize_day_stacked(
        prices_da=prices_da, prices_id=prices_da,
        afrr_cap_pos_price=_zeros_block(),
        afrr_cap_neg_price=_zeros_block(),
        afrr_energy_pos_price=_zeros_day(),
        afrr_energy_neg_price=_zeros_day(),
        afrr_activation_rate_pos=_zeros_day(),
        afrr_activation_rate_neg=_zeros_day(),
        energy_mwh=2.0, power_mw=1.0,
        afrr_reserve_duration_hours=0.0,
    )
    coupled = optimize_day_stacked(
        prices_da=prices_da, prices_id=prices_da,
        afrr_cap_pos_price=_zeros_block(),
        afrr_cap_neg_price=_zeros_block(),
        afrr_energy_pos_price=_zeros_day(),
        afrr_energy_neg_price=afrr_energy_neg,
        afrr_activation_rate_pos=_zeros_day(),
        afrr_activation_rate_neg=act_neg,
        energy_mwh=2.0, power_mw=1.0,
        afrr_reserve_duration_hours=0.0,
    )
    # Coupled LP must strictly outperform DA-only baseline
    assert coupled.revenue_total > baseline.revenue_total + 1.0
    # It actually reserves aFRR NEG for the morning block(s)
    assert coupled.r_neg.sum() > 0
    # aFRR NEG revenue is negative (operator pays TSO)
    assert coupled.revenue_afrr_energy_neg < 0


def test_wear_cost_vector_discourages_throughput():
    """Per-interval wear cost makes LP reduce activity compared to zero
    wear. Sets up the shadow-cost integration for A3."""
    prices_da = np.concatenate([
        np.full(PERIODS_PER_DAY // 2, 20.0),
        np.full(PERIODS_PER_DAY // 2, 200.0),
    ])
    common = dict(
        prices_da=prices_da, prices_id=prices_da,
        afrr_cap_pos_price=_zeros_block(),
        afrr_cap_neg_price=_zeros_block(),
        afrr_energy_pos_price=_zeros_day(),
        afrr_energy_neg_price=_zeros_day(),
        afrr_activation_rate_pos=_zeros_day(),
        afrr_activation_rate_neg=_zeros_day(),
        energy_mwh=2.0, power_mw=1.0,
        afrr_reserve_duration_hours=0.0,
    )
    no_wear = optimize_day_stacked(**common, wear_cost_eur_per_mwh=None)
    # Heavy wear (100 €/MWh applied to both charge and discharge = 200 total
    # per cycle). 20→200 spread delivers ~180 EUR/MWh; net after wear
    # becomes negative → LP should stop cycling.
    heavy_wear = optimize_day_stacked(
        **common, wear_cost_eur_per_mwh=np.full(PERIODS_PER_DAY, 100.0),
    )
    assert no_wear.full_equivalent_cycles > heavy_wear.full_equivalent_cycles + 1e-6
    # Pre-wear revenue must be at least as high without wear
    assert no_wear.revenue_total >= heavy_wear.revenue_total - 1e-6


def test_cycle_cap_binds():
    """Tight cycle cap limits total FEC regardless of revenue opportunity."""
    prices_da = np.concatenate([
        np.full(PERIODS_PER_DAY // 2, 0.0),
        np.full(PERIODS_PER_DAY // 2, 1000.0),
    ])
    result = optimize_day_stacked(
        prices_da=prices_da, prices_id=prices_da,
        afrr_cap_pos_price=_zeros_block(),
        afrr_cap_neg_price=_zeros_block(),
        afrr_energy_pos_price=_zeros_day(),
        afrr_energy_neg_price=_zeros_day(),
        afrr_activation_rate_pos=_zeros_day(),
        afrr_activation_rate_neg=_zeros_day(),
        energy_mwh=2.0, power_mw=1.0,
        max_cycles=0.5,                   # strict cap
        afrr_reserve_duration_hours=0.0,
    )
    assert result.success
    # FEC must not exceed cap (with tiny numerical slack)
    assert result.full_equivalent_cycles <= 0.5 + 1e-6


def test_revenue_split_reconciles_with_total():
    """Sum of revenue streams must equal revenue_total."""
    prices_da = np.sin(np.linspace(0, 4 * np.pi, PERIODS_PER_DAY)) * 50 + 100
    result = optimize_day_stacked(
        prices_da=prices_da, prices_id=prices_da * 1.05,
        afrr_cap_pos_price=np.full(BLOCKS_PER_DAY, 3.0),
        afrr_cap_neg_price=np.full(BLOCKS_PER_DAY, 3.0),
        afrr_energy_pos_price=np.full(PERIODS_PER_DAY, 200.0),
        afrr_energy_neg_price=np.full(PERIODS_PER_DAY, -150.0),
        afrr_activation_rate_pos=np.full(PERIODS_PER_DAY, 0.05),
        afrr_activation_rate_neg=np.full(PERIODS_PER_DAY, 0.04),
        energy_mwh=2.0, power_mw=1.0,
    )
    assert result.success
    sum_streams = (
        result.revenue_da + result.revenue_id
        + result.revenue_afrr_cap_pos + result.revenue_afrr_cap_neg
        + result.revenue_afrr_energy_pos + result.revenue_afrr_energy_neg
    )
    assert sum_streams == pytest.approx(result.revenue_total, rel=1e-6)
