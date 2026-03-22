"""
Forward projection — extends historical dispatch results to 2026-2040.

Uses reduced-form growth factors calibrated to industry consensus anchors.
"""

import numpy as np
from lib.config import (
    DEFAULT_BESS_BUILDOUT,
    DEMAND_2026, DEMAND_2040, MODO_V2,
    TTF_2026, TTF_2040,
    PV_GW_2026, PV_GW_2040, PV_SPREAD_SENSITIVITY,
    GAS_SPREAD_ELASTICITY,
    fleet_degradation_factor,
)
from lib.models.ancillary import ancillary_revenue


def id_da_ratio(year: int) -> float:
    """
    ID/DA wholesale revenue ratio for projection.

    Empirical basis: LP-optimal dispatch on DE 15-min ID-AEP prices (capped ±150 €/MWh
    to exclude unrealistic imbalance spikes) vs hourly DA prices, 2023-2025.
    Raw ID/DA ≈ 1.2x, but real operators trade mostly on DA with ID for adjustment,
    so effective split is DA ≈ 65%, ID ≈ 35% of wholesale → ratio ≈ 0.55.

    Forward trend: RE growth increases intraday volatility → ID share rises slightly.
    """
    if year <= 2026:
        return 0.50
    elif year >= 2040:
        return 0.65
    else:
        return 0.50 + (0.65 - 0.50) * (year - 2026) / (2040 - 2026)


def interpolate_linear(v_start: float, v_end: float, y_start: int, y_end: int, year: int) -> float:
    if year <= y_start:
        return v_start
    if year >= y_end:
        return v_end
    return v_start + (v_end - v_start) * (year - y_start) / (y_end - y_start)


def project_wholesale(
    year: int,
    historical_da_annual: float,       # kEUR/MW/yr — average of recent historical
    bess_gw: float,
    demand_twh: float | None = None,
    demand_2040_twh: float | None = None,
    gas_price: float | None = None,    # €/MWh TTF; None = interpolate from defaults
    gas_2040: float | None = None,     # override TTF_2040
    pv_gw: float | None = None,        # installed PV GW; None = interpolate
    pv_2040_gw: float | None = None,   # override PV_GW_2040
    # Calibrated on DE+UK panel 2023-2025 (validation/calibrate.py):
    beta_d: float = 30.0,              # demand growth sensitivity (accelerating)
    canib_max: float = 15.0,           # max cannibalization (kEUR) at saturation
    canib_half: float = 25.0,          # fleet size (GW) at half-saturation
    canib_steep: float = 0.8,          # logistic steepness
) -> dict[str, float]:
    """
    Project wholesale (DA + ID) revenue for a future year.

    R_wh(t) = baseline
              + beta_D * (D_t/D_base - 1)^1.5     (demand growth)
              - canib_max / (1 + exp(-k*(B-B_half)))  (fleet cannibalization)
              + baseline * gas_elast * (TTF/TTF_base - 1)  (gas price)
              + pv_sens * (PV - PV_base) / 100     (solar duck curve)

    Calibration:
      Stage 1: gas_elast from DE DA spreads 2020-2025 (R²=0.97)
      Stage 2: other params from DE+UK revenue panel 2023-2025
      See validation/calibrate.py for full provenance.
    """
    d_2040 = demand_2040_twh if demand_2040_twh is not None else DEMAND_2040
    if demand_twh is None:
        demand_twh = interpolate_linear(DEMAND_2026, d_2040, 2026, 2040, year)

    # Gas price: default trajectory from TTF forwards
    g_2040 = gas_2040 if gas_2040 is not None else TTF_2040
    if gas_price is None:
        gas_price = interpolate_linear(TTF_2026, g_2040, 2026, 2040, year)

    # Solar PV fleet: default trajectory
    pv_target = pv_2040_gw if pv_2040_gw is not None else PV_GW_2040
    if pv_gw is None:
        pv_gw = interpolate_linear(PV_GW_2026, pv_target, 2026, 2040, year)

    # Demand factor: accelerating (power 1.5) — electrification drives spreads
    d_ratio = demand_twh / DEMAND_2026
    demand_factor = beta_d * max(d_ratio - 1, 0.0) ** 1.5

    # Logistic cannibalization: saturates at canib_max
    storage_factor = canib_max / (1 + np.exp(-canib_steep * (bess_gw - canib_half)))

    # Gas price factor: deviation from baseline TTF drives peak prices
    # At baseline (TTF_2026=35), factor=0. Higher gas → higher spreads.
    gas_factor = historical_da_annual * GAS_SPREAD_ELASTICITY * (gas_price / TTF_2026 - 1)

    # Solar PV factor: more PV deepens duck curve, creating wider spreads
    # Each 100 GW above baseline adds PV_SPREAD_SENSITIVITY kEUR
    pv_factor = PV_SPREAD_SENSITIVITY * (pv_gw - PV_GW_2026) / 100.0

    r_wholesale = historical_da_annual + demand_factor - storage_factor + gas_factor + pv_factor
    r_wholesale = max(r_wholesale, 40.0)

    # Split DA / ID
    ratio = id_da_ratio(year)
    r_da = r_wholesale / (1 + ratio)
    r_id = r_wholesale * ratio / (1 + ratio)

    return {
        "da": r_da,
        "id": r_id,
        "wholesale_total": r_wholesale,
    }


def project_full_stack(
    years: list[int],
    historical_da_keur: float,
    bess_buildout: dict[int, float] | None = None,
    duration_h: float = 2.0,
    gas_2040: float | None = None,
    pv_2040_gw: float | None = None,
    **wholesale_kwargs,
) -> list[dict]:
    """
    Generate full revenue stack for each year.

    Returns list of dicts with keys: year, da, id, fcr, afrr_cap, afrr_energy, total.
    All values in kEUR/MW/yr.
    """
    if bess_buildout is None:
        bess_buildout = DEFAULT_BESS_BUILDOUT

    results = []
    for year in years:
        bess_gw = bess_buildout.get(year, bess_buildout[max(k for k in bess_buildout if k <= year)])

        wh = project_wholesale(
            year=year,
            historical_da_annual=historical_da_keur,
            bess_gw=bess_gw,
            gas_2040=gas_2040,
            pv_2040_gw=pv_2040_gw,
            **wholesale_kwargs,
        )

        anc = ancillary_revenue(year=year, bess_gw=bess_gw, duration_h=duration_h)

        # Degradation: fleet-average across projection-era cohorts only
        # (pre-2026 fleet is already captured in the historical baseline)
        proj_buildout = {y: v for y, v in bess_buildout.items() if y >= min(years)}
        deg = fleet_degradation_factor(year, proj_buildout)

        results.append({
            "year": year,
            "da": round(wh["da"] * deg, 1),
            "id": round(wh["id"] * deg, 1),
            "fcr": round(anc["fcr"] * deg, 1),
            "afrr_cap": round(anc["afrr_cap"] * deg, 1),
            "afrr_energy": round(anc["afrr_energy"] * deg, 1),
            "total": round((wh["wholesale_total"] + anc["total"]) * deg, 1),
        })

    return results


def modo_v2_overlay(years: list[int]) -> list[dict]:
    """Return Modo v2 replica data for overlay comparison."""
    results = []
    for y in years:
        if y in MODO_V2:
            row = MODO_V2[y].copy()
            row["year"] = y
            row["total"] = sum(v for k, v in MODO_V2[y].items())
            results.append(row)
    return results
