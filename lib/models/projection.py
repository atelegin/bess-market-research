"""
Forward projection — extends historical dispatch results to 2026-2040.

Uses reduced-form growth factors calibrated to industry consensus anchors.
"""

import numpy as np
from lib.config import (
    ANCILLARY_COMBINED_GW,
    DEFAULT_BESS_BUILDOUT,
    DEMAND_2026, DEMAND_2040,
    TTF_2026, TTF_2040,
    PV_GW_2026, PV_GW_2040, PV_SPREAD_SENSITIVITY,
    GAS_SPREAD_ELASTICITY,
)
from lib.models.ancillary import (
    HISTORICAL_YEARS_WITH_MEASURED_DATA,
    ancillary_revenue,
)
from lib.models.degradation import PRESETS, fleet_average_capacity


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


def solve_as_wholesale_allocation(
    year: int,
    bess_gw_total: float,
    historical_da_keur: float,
    duration_h: float = 2.0,
    as_demand_gw: float = ANCILLARY_COMBINED_GW,
    tol: float = 1e-4,
    max_iter: int = 60,
    **wholesale_kwargs,
) -> dict[str, float | str]:
    """
    Equilibrium split of BESS fleet between ancillary services and wholesale
    arbitrage (Simon / Schäfer correction, 2026-04).

    Each rational operator chooses a fraction ``f`` of capacity to bid into
    AS. Marginal condition at interior equilibrium:

        p_AS(f · bess_gw)  ==  p_WH((1-f) · bess_gw)

    where ``p_AS(gw_on_as)`` is the AS clearing price with ``gw_on_as`` MW
    competing for the fixed AS demand (~4.5 GW), and ``p_WH(gw_on_wh)`` is
    the wholesale spread revenue with ``gw_on_wh`` MW cannibalising spreads.

    Simplification (Schäfer, 2026-04): wholesale is not perturbed by AS
    volume shifts — AS depth (~4.5 GW) is small vs wholesale throughput.

    Corner cases:
      * ``f = 0``  — AS clearing at tiny supply already below wholesale
        (all fleet arbitrages).
      * ``f = f_cap = min(1, as_demand_gw / bess_gw_total)`` — AS demand
        saturated; remaining fleet on wholesale.
      * interior — bisection on ``f`` in ``[0, f_cap]``.
      * historical override — for years in ``HISTORICAL_YEARS_WITH_MEASURED_DATA``
        the equilibrium concept does not apply (operators already acted).
        Returns observed AS revenue alongside projected wholesale at the
        implied fleet-on-wh split ``gw_on_wh = bess_gw_total × (1 − f_cap)``,
        and flags ``equilibrium_type = "historical_override"``.

    Returns dict with:
      f                  — equilibrium AS fraction of fleet (0..f_cap)
      gw_on_as           — f * bess_gw_total
      gw_on_wh           — (1 - f) * bess_gw_total
      p_as               — kEUR/MW-on-AS/yr at equilibrium
      p_wh               — kEUR/MW-on-WH/yr at equilibrium
      equilibrium_type   — "all_on_wh", "interior", "as_capacity_capped",
                           or "historical_override"
    """
    # Avoid numerical degeneracies at gw = 0 by floating a tiny epsilon.
    _eps = max(1e-3, bess_gw_total * 1e-4)

    # Historical override: short-circuit equilibrium for years where we have
    # measured ancillary-revenue data. We still compute the "would-be" f_cap
    # split (the AS-demand-constrained upper bound) so downstream consumers
    # have a consistent interpretation for gw_on_as / gw_on_wh.
    if year in HISTORICAL_YEARS_WITH_MEASURED_DATA:
        f_hist = min(1.0, as_demand_gw / bess_gw_total) if bess_gw_total > 0 else 1.0
        return {
            "f": f_hist,
            "gw_on_as": f_hist * bess_gw_total,
            "gw_on_wh": (1.0 - f_hist) * bess_gw_total,
            "p_as": ancillary_revenue(
                year=year, bess_gw=max(f_hist * bess_gw_total, _eps),
                duration_h=duration_h,
            )["total"],
            "p_wh": project_wholesale(
                year=year, historical_da_annual=historical_da_keur,
                bess_gw=max((1.0 - f_hist) * bess_gw_total, _eps),
                **wholesale_kwargs,
            )["wholesale_total"],
            "equilibrium_type": "historical_override",
        }

    def p_as_of(gw_on_as: float) -> float:
        return ancillary_revenue(
            year=year,
            bess_gw=max(gw_on_as, _eps),
            duration_h=duration_h,
        )["total"]

    def p_wh_of(gw_on_wh: float) -> float:
        return project_wholesale(
            year=year,
            historical_da_annual=historical_da_keur,
            bess_gw=max(gw_on_wh, _eps),
            **wholesale_kwargs,
        )["wholesale_total"]

    f_cap = min(1.0, as_demand_gw / bess_gw_total) if bess_gw_total > 0 else 1.0

    # Corner: at tiny AS supply, is AS already worth less than wholesale?
    if p_as_of(_eps) <= p_wh_of(bess_gw_total):
        f = 0.0
        return {
            "f": 0.0,
            "gw_on_as": 0.0,
            "gw_on_wh": bess_gw_total,
            "p_as": p_as_of(_eps),
            "p_wh": p_wh_of(bess_gw_total),
            "equilibrium_type": "all_on_wh",
        }

    # Corner: at f_cap, is AS still more attractive than wholesale?
    p_as_at_cap = p_as_of(f_cap * bess_gw_total)
    p_wh_at_cap = p_wh_of((1.0 - f_cap) * bess_gw_total)
    if p_as_at_cap >= p_wh_at_cap:
        return {
            "f": f_cap,
            "gw_on_as": f_cap * bess_gw_total,
            "gw_on_wh": (1.0 - f_cap) * bess_gw_total,
            "p_as": p_as_at_cap,
            "p_wh": p_wh_at_cap,
            "equilibrium_type": "as_capacity_capped",
        }

    # Interior: bisect on f where g(f) = p_AS(f·B) − p_WH((1−f)·B) transitions
    # from positive (at f=0) to negative (at f=f_cap).
    lo, hi = 0.0, f_cap
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        diff = p_as_of(mid * bess_gw_total) - p_wh_of((1.0 - mid) * bess_gw_total)
        if diff > 0:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break
    f = 0.5 * (lo + hi)
    return {
        "f": f,
        "gw_on_as": f * bess_gw_total,
        "gw_on_wh": (1.0 - f) * bess_gw_total,
        "p_as": p_as_of(f * bess_gw_total),
        "p_wh": p_wh_of((1.0 - f) * bess_gw_total),
        "equilibrium_type": "interior",
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
    Generate full revenue stack for each year under the AS/wholesale
    equilibrium allocation (Simon / Schäfer correction, 2026-04).

    Per-MW-of-fleet revenue = f · p_AS + (1 − f) · p_WH where ``f`` is the
    equilibrium AS participation share from ``solve_as_wholesale_allocation``.
    At interior equilibrium this equals ``p_AS = p_WH``.

    Returns list of dicts with keys: year, da, id, fcr, afrr_cap, afrr_energy,
    total, f_on_as, equilibrium_type. All revenue values in kEUR/MW/yr of
    fleet nameplate.
    """
    if bess_buildout is None:
        bess_buildout = DEFAULT_BESS_BUILDOUT

    results = []
    proj_buildout = {y: v for y, v in bess_buildout.items() if y >= min(years)}
    for year in years:
        bess_gw = bess_buildout.get(year, bess_buildout[max(k for k in bess_buildout if k <= year)])

        alloc = solve_as_wholesale_allocation(
            year=year,
            bess_gw_total=bess_gw,
            historical_da_keur=historical_da_keur,
            duration_h=duration_h,
            gas_2040=gas_2040,
            pv_2040_gw=pv_2040_gw,
            **wholesale_kwargs,
        )
        f = alloc["f"]

        # Wholesale component at reduced competition (gw_on_wh MW cannibalise)
        _eps = max(1e-3, bess_gw * 1e-4)
        wh = project_wholesale(
            year=year,
            historical_da_annual=historical_da_keur,
            bess_gw=max(alloc["gw_on_wh"], _eps),
            gas_2040=gas_2040,
            pv_2040_gw=pv_2040_gw,
            **wholesale_kwargs,
        )
        # AS component at reduced supply (gw_on_as MW competing for AS demand)
        anc = ancillary_revenue(
            year=year,
            bess_gw=max(alloc["gw_on_as"], _eps),
            duration_h=duration_h,
        )

        # Scale each side by the fleet fraction to get per-MW-of-fleet revenue.
        r_da = (1.0 - f) * wh["da"]
        r_id = (1.0 - f) * wh["id"]
        r_fcr = f * anc["fcr"]
        r_afrr_cap = f * anc["afrr_cap"]
        r_afrr_energy = f * anc["afrr_energy"]

        # Degradation: fleet-average across projection-era cohorts only
        # (pre-2026 fleet is already captured in the historical baseline)
        deg = fleet_average_capacity(
            year=year,
            buildout=proj_buildout,
            preset=PRESETS["baseline_fleet"],
        )

        results.append({
            "year": year,
            "da": round(r_da * deg, 1),
            "id": round(r_id * deg, 1),
            "fcr": round(r_fcr * deg, 1),
            "afrr_cap": round(r_afrr_cap * deg, 1),
            "afrr_energy": round(r_afrr_energy * deg, 1),
            "total": round((r_da + r_id + r_fcr + r_afrr_cap + r_afrr_energy) * deg, 1),
            "f_on_as": round(f, 3),
            "equilibrium_type": alloc["equilibrium_type"],
        })

    return results
