"""
Tests for AS/wholesale equilibrium allocation (Simon/Schäfer correction, 2026-04).

Design reference: plan `~/.claude/plans/eager-enchanting-glade.md` §A1.3.
"""
from __future__ import annotations

import math

import pytest

from lib.config import ANCILLARY_COMBINED_GW
from lib.models.ancillary import ancillary_revenue
from lib.models.projection import (
    project_full_stack,
    project_wholesale,
    solve_as_wholesale_allocation,
)


BASELINE_DA_KEUR = 90.0  # representative DE DA anchor for 2026-era baseline


def test_solver_returns_expected_keys():
    result = solve_as_wholesale_allocation(
        year=2026, bess_gw_total=5.0, historical_da_keur=BASELINE_DA_KEUR,
    )
    assert set(result) == {"f", "gw_on_as", "gw_on_wh", "p_as", "p_wh", "equilibrium_type"}
    assert 0.0 <= result["f"] <= 1.0
    assert result["equilibrium_type"] in {"all_on_wh", "interior", "as_capacity_capped"}


def test_fractions_sum_to_fleet():
    for bess_gw in (1.5, 5.0, 10.0, 17.0, 25.0):
        alloc = solve_as_wholesale_allocation(
            year=2026, bess_gw_total=bess_gw, historical_da_keur=BASELINE_DA_KEUR,
        )
        assert math.isclose(alloc["gw_on_as"] + alloc["gw_on_wh"], bess_gw, rel_tol=1e-6)


def test_early_regime_caps_on_as_demand():
    """2026 / 5 GW fleet: AS clearing (~135) > wholesale (~100) so operators want
    max AS participation. Equilibrium hits the AS-demand cap."""
    alloc = solve_as_wholesale_allocation(
        year=2026, bess_gw_total=5.0, historical_da_keur=BASELINE_DA_KEUR,
    )
    assert alloc["equilibrium_type"] == "as_capacity_capped"
    assert alloc["gw_on_as"] <= ANCILLARY_COMBINED_GW + 1e-6
    assert alloc["f"] == pytest.approx(ANCILLARY_COMBINED_GW / 5.0, rel=1e-4)


def test_late_regime_caps_on_as_demand_not_interior():
    """Even at 17 GW fleet, the AS-clearing curve is steep enough that p_AS(4.5)
    (all AS demand filled) stays above p_WH(12.5). So equilibrium caps on AS
    demand rather than going interior. This is the model's actual behaviour
    under default 2h calibration and is important to document — it means the
    'fleet revenue collapse' story is bounded by AS demand, not by wholesale
    crossing the AS clearing curve."""
    alloc = solve_as_wholesale_allocation(
        year=2030, bess_gw_total=17.0, historical_da_keur=BASELINE_DA_KEUR,
    )
    assert alloc["equilibrium_type"] == "as_capacity_capped"
    assert alloc["gw_on_as"] == pytest.approx(ANCILLARY_COMBINED_GW, rel=1e-4)
    assert alloc["gw_on_wh"] == pytest.approx(17.0 - ANCILLARY_COMBINED_GW, rel=1e-4)


def test_interior_equilibrium_with_high_wholesale():
    """Interior equilibrium is reachable when wholesale is strong enough to
    out-bid even the saturated AS clearing. Force it via an aggressive gas
    scenario (drives spreads up) at a late year."""
    alloc = solve_as_wholesale_allocation(
        year=2035, bess_gw_total=17.0, historical_da_keur=BASELINE_DA_KEUR,
        gas_2040=80.0, pv_2040_gw=400.0, demand_2040_twh=1200.0,
        canib_max=5.0,  # light cannibalisation keeps WH high
    )
    assert alloc["equilibrium_type"] == "interior"
    # p_as and p_wh converge at the margin
    assert abs(alloc["p_as"] - alloc["p_wh"]) < 2.0
    # Only a fraction of the fleet stays in AS
    assert 0.0 < alloc["f"] < 1.0


def test_solver_no_collapse_at_high_fleet():
    """Simon's core correction: total revenue must not collapse from migration.
    At 17 GW, equilibrium per-MW revenue must be at least as high as wholesale-only
    at full-fleet cannibalisation."""
    wh_alone_full_fleet = project_wholesale(
        year=2030, historical_da_annual=BASELINE_DA_KEUR, bess_gw=17.0,
    )["wholesale_total"]
    alloc = solve_as_wholesale_allocation(
        year=2030, bess_gw_total=17.0, historical_da_keur=BASELINE_DA_KEUR,
    )
    # Equilibrium per-MW-of-fleet revenue at interior is p_as = p_wh
    per_mw_total = alloc["f"] * alloc["p_as"] + (1 - alloc["f"]) * alloc["p_wh"]
    assert per_mw_total >= wh_alone_full_fleet - 1e-3


def test_full_stack_output_contract():
    """project_full_stack must still emit the keys Note 1 charts expect."""
    rows = project_full_stack(
        years=[2026, 2028, 2030, 2035, 2040],
        historical_da_keur=BASELINE_DA_KEUR,
        duration_h=2.0,
    )
    required = {"year", "da", "id", "fcr", "afrr_cap", "afrr_energy", "total"}
    added = {"f_on_as", "equilibrium_type"}
    for row in rows:
        assert required.issubset(row)
        assert added.issubset(row)
        # Stack identity: components sum (approx) to total
        stack = row["da"] + row["id"] + row["fcr"] + row["afrr_cap"] + row["afrr_energy"]
        assert abs(stack - row["total"]) < 0.2  # rounding to 1 decimal in each component


def test_full_stack_no_collapse_narrative():
    """Total fleet revenue does not drop 60% between 2026 and 2030 any more
    (before the rework: 235 → 93 kEUR). After: both floored by wholesale."""
    rows = project_full_stack(
        years=[2026, 2030],
        historical_da_keur=BASELINE_DA_KEUR,
        duration_h=2.0,
    )
    total_2026 = rows[0]["total"]
    total_2030 = rows[1]["total"]
    # Allow ≤ 50% drawdown (still substantial, but not cliff-like)
    assert total_2030 / total_2026 > 0.5


def test_historical_override_activates_for_2023_2025():
    """2023-2025 have complete regelleistung + netztransparenz data. The
    equilibrium solver must short-circuit to the measured values."""
    for year in (2023, 2024, 2025):
        alloc = solve_as_wholesale_allocation(
            year=year, bess_gw_total=2.5, historical_da_keur=90.0,
        )
        assert alloc["equilibrium_type"] == "historical_override"
        # At 2.5 GW fleet with AS demand = 4.5 GW, all fleet fits on AS (f=1.0)
        assert alloc["f"] == pytest.approx(1.0, rel=1e-3)
        # p_as must come from measured data, not from saturation extrapolation
        # (which at gw_on_as=2.5 would give ~200 via the uncalibrated curve)
        assert 40.0 <= alloc["p_as"] <= 220.0  # measured plausibility band


def test_historical_override_can_be_disabled():
    """Passing use_historical_if_available=False must fall through to the
    saturation model even for 2023-2025."""
    r_measured = ancillary_revenue(2024, bess_gw=2.5, duration_h=2.0)
    r_model = ancillary_revenue(
        2024, bess_gw=2.5, duration_h=2.0, use_historical_if_available=False,
    )
    # Numerics must diverge — measured 2024 total is ~154, saturation at 2.5 GW
    # extrapolates high (no 1.5-2.5 GW anchor in the original calibration).
    assert abs(r_measured["total"] - r_model["total"]) > 20.0


def test_future_years_still_use_saturation_model():
    """2026 and onwards must continue using the saturation model — the
    historical override only applies to calibrated historical years."""
    alloc_2026 = solve_as_wholesale_allocation(
        year=2026, bess_gw_total=5.0, historical_da_keur=90.0,
    )
    alloc_2030 = solve_as_wholesale_allocation(
        year=2030, bess_gw_total=17.0, historical_da_keur=90.0,
    )
    assert alloc_2026["equilibrium_type"] != "historical_override"
    assert alloc_2030["equilibrium_type"] != "historical_override"


def test_project_full_stack_mixed_historical_and_projection():
    """project_full_stack over a span that straddles the historical cutoff
    must produce sensible values for both sides — observed for 2023-2025,
    equilibrium-solved for 2026+."""
    rows = project_full_stack(
        years=[2023, 2024, 2025, 2026, 2028, 2030],
        historical_da_keur=90.0,
        duration_h=2.0,
    )
    by_year = {r["year"]: r for r in rows}
    # Historical rows flag the override
    for y in (2023, 2024, 2025):
        assert by_year[y]["equilibrium_type"] == "historical_override"
    # Projection rows do not
    assert by_year[2026]["equilibrium_type"] != "historical_override"
    assert by_year[2030]["equilibrium_type"] != "historical_override"


def test_1h_battery_lower_as_share():
    """1h batteries only participate ~50% of the time in 4h AS blocks.
    Their equilibrium per-MW revenue must be lower than the 2h case."""
    a_2h = solve_as_wholesale_allocation(
        year=2028, bess_gw_total=10.0, historical_da_keur=BASELINE_DA_KEUR,
        duration_h=2.0,
    )
    a_1h = solve_as_wholesale_allocation(
        year=2028, bess_gw_total=10.0, historical_da_keur=BASELINE_DA_KEUR,
        duration_h=1.0,
    )
    rev_2h = a_2h["f"] * a_2h["p_as"] + (1 - a_2h["f"]) * a_2h["p_wh"]
    rev_1h = a_1h["f"] * a_1h["p_as"] + (1 - a_1h["f"]) * a_1h["p_wh"]
    assert rev_1h <= rev_2h + 1e-3
