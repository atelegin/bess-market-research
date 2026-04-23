"""
Ancillary revenue model — FCR + aFRR saturation.

Models revenue collapse as BESS fleet outgrows ancillary market depth.
"""

import logging

import numpy as np
from lib.config import (
    FCR_DEMAND_MW, AFRR_DEPTH_MW, ANCILLARY_COMBINED_GW,
    DEFAULT_BESS_BUILDOUT,
)

logger = logging.getLogger(__name__)

# Historical years with measured FCR + aFRR cap + aFRR energy data from
# regelleistung.net + netztransparenz.de. For these years ``ancillary_revenue``
# bypasses the saturation model and returns the observed values directly.
HISTORICAL_YEARS_WITH_MEASURED_DATA: tuple[int, ...] = (2023, 2024, 2025)


def _measured_historical_components(
    year: int, duration_h: float = 2.0
) -> dict[str, float] | None:
    """
    Return measured ancillary components for a historical year.

    Aggregates regelleistung.net FCR + aFRR capacity auctions and the real
    netztransparenz.de aFRR activation-energy revenue (see
    ``lib/data/ancillary_prices.py``). Returns kEUR/MW-on-AS/yr per component,
    scaled by duration participation factor (1h batteries participate ~50 %
    in 4h AS blocks, 2h+ participate fully).

    Participation factors are baked into the underlying loaders
    (``fetch_fcr_annual_revenue`` at 0.35; ``fetch_afrr_annual_revenue`` at
    0.40) — they are calibrated for 2h BESS, so this function applies the
    duration scaling on top.

    Returns ``None`` if any component fails to fetch; caller should fall
    back to the saturation model.
    """
    # Import locally to avoid a heavy import at module load time and to
    # isolate a potential (not actual today) circular dep risk.
    from lib.data.ancillary_prices import (
        fetch_afrr_annual_revenue,
        fetch_fcr_annual_revenue,
    )

    fcr = fetch_fcr_annual_revenue(year)
    afrr = fetch_afrr_annual_revenue(year, use_real_energy=True)
    if fcr is None or afrr is None:
        logger.warning(
            f"measured_historical_components({year}): data fetch failed — "
            "will fall back to saturation model"
        )
        return None

    dur_scale = min(duration_h / 2.0, 1.0)
    return {
        "fcr": float(fcr) * dur_scale,
        "afrr_cap": float(afrr["afrr_cap"]) * dur_scale,
        "afrr_energy": float(afrr["afrr_energy"]) * dur_scale,
        "total": float(fcr + afrr["afrr_cap"] + afrr["afrr_energy"]) * dur_scale,
    }


def ancillary_revenue(
    year: int,
    bess_gw: float,
    duration_h: float = 2.0,
    # Calibration anchors (2h battery):
    # Derived from regelleistung.net auction data + market depth constraints.
    # At 1.5-3.5 GW (2023-2025), observed ancillary ~143-176 kEUR/MW
    # (FCR 35% participation, aFRR 40% participation on regelleistung prices).
    # At 5 GW (slightly above 4.5 GW combined depth), prices start to compress → 135 kEUR.
    # At 17 GW (3.8x depth), bid competition has largely collapsed prices → 13 kEUR.
    r_anc_2026: float = 135.0,     # kEUR/MW-on-AS at 5 GW competing for AS demand
    r_anc_2030: float = 13.0,      # kEUR/MW-on-AS at 17 GW competing for AS demand
    r_anc_floor: float = 2.0,      # residual floor: minimum participation revenue
    ancillary_depth_gw: float = ANCILLARY_COMBINED_GW,
    use_historical_if_available: bool = True,
) -> dict[str, float]:
    """
    Ancillary revenue per MW-on-AS under supply saturation.

    R_anc(gw_on_as) = floor + amplitude / (1 + (gw_on_as / depth)^alpha)

    **Semantics (2026-04-23 rework for Schäfer dynamic floor):** ``bess_gw``
    denotes the volume of BESS capacity *actively bidding into ancillary
    services*, not the total fleet. In an equilibrium allocation between AS
    and wholesale arbitrage (``projection.solve_as_wholesale_allocation``),
    this is ``f * bess_gw_fleet`` where ``f`` is the AS-participation share.

    Calibrated for 2h battery. The historical anchors (5 GW → 135 kEUR/MW-on-AS;
    17 GW → 13 kEUR/MW-on-AS) reflect the regimes 2026 and 2030 where
    essentially the whole fleet bids AS — so ``gw_on_as ≈ gw_fleet`` and the
    anchors remain valid.

    Duration scaling: FCR/aFRR are auctioned in 4h blocks — a 1h battery can
    only participate ~50 % of the time (must reserve energy), while 2h+
    batteries can participate fully.

    Returns dict with fcr, afrr_cap, afrr_energy, total (all kEUR/MW-on-AS/yr).

    Historical override (2023-2025, 2026-04 rework per ROADMAP note
    ``trader-aging-aware`` A1.2): for years with full-year measured data
    from regelleistung.net + netztransparenz.de, the saturation model is
    bypassed and observed values are returned. Set
    ``use_historical_if_available=False`` to force the saturation model
    (useful for sensitivity analysis or testing).
    """
    if use_historical_if_available and year in HISTORICAL_YEARS_WITH_MEASURED_DATA:
        measured = _measured_historical_components(year, duration_h=duration_h)
        if measured is not None:
            return measured

    # Duration scaling: 1h=50%, 2h+=100% participation in 4h ancillary blocks
    dur_scale = min(duration_h / 2.0, 1.0)
    bess_2026 = DEFAULT_BESS_BUILDOUT.get(2026, 5.0)
    bess_2030 = DEFAULT_BESS_BUILDOUT.get(2030, 17.0)

    s_2026 = bess_2026 / ancillary_depth_gw
    s_2030 = bess_2030 / ancillary_depth_gw

    alpha = _solve_alpha(
        r_anc_2026 - r_anc_floor,
        r_anc_2030 - r_anc_floor,
        s_2026, s_2030,
    )

    amp = (r_anc_2026 - r_anc_floor) * (1 + s_2026 ** alpha)

    s_t = bess_gw / ancillary_depth_gw
    r_total = r_anc_floor + amp / (1 + s_t ** alpha)
    r_total = max(r_total, r_anc_floor)

    # Split into FCR / aFRR_cap / aFRR_energy
    # Each component has its own saturation curve:
    #   FCR:   8 (2026, 5GW) → 2 (2030, 17GW) → 0 (2035+)
    #   aFRRE: 12 (2026) → 3 (2030) → 1 (2035+)
    #   aFRR_cap: remainder of total
    fcr_abs = _component_saturate(bess_gw, val_at_5gw=8.0, val_at_17gw=2.0, floor=0.0)
    afrre_abs = _component_saturate(bess_gw, val_at_5gw=12.0, val_at_17gw=3.0, floor=0.5)
    afrr_cap = max(r_total - fcr_abs - afrre_abs, 0.0)

    return {
        "fcr": fcr_abs * dur_scale,
        "afrr_cap": afrr_cap * dur_scale,
        "afrr_energy": afrre_abs * dur_scale,
        "total": r_total * dur_scale,
    }


def afrr_prequal_fraction(
    max_ramp_pct_per_min: float | None,
    full_activation_seconds: float = 300.0,
) -> float:
    """
    Fraction of rated power that meets the aFRR full-activation SLA under a
    Leistungsgradient limit. German aFRR requires reaching 100 % of reserved
    capacity within the full-activation time (default 5 min). With a ramp
    limit of r %/min, the battery reaches r × (full_activation_seconds / 60) %
    of P in that window; that caps prequalified MW as a fraction of rated.

    BDEW TAB-MS v1 band: default 22 %/min → 1.0 (no binding), floor 6 %/min
    → 0.30 (only 30 % of P can prequalify). ``None`` = unconstrained → 1.0.
    """
    if max_ramp_pct_per_min is None:
        return 1.0
    reachable_pct = max_ramp_pct_per_min * (full_activation_seconds / 60.0)
    return float(min(1.0, max(0.0, reachable_pct / 100.0)))


def _component_saturate(
    bess_gw: float, val_at_5gw: float, val_at_17gw: float, floor: float,
) -> float:
    """
    Individual ancillary component with exponential decay calibrated
    to two known points (5 GW and 17 GW).
    """
    # Solve: val = floor + A * exp(-k * bess)
    # At 5:  val_5  = floor + A * exp(-5k)
    # At 17: val_17 = floor + A * exp(-17k)
    a5 = val_at_5gw - floor
    a17 = val_at_17gw - floor
    if a5 <= 0 or a17 <= 0 or a17 >= a5:
        return max(floor, 0.0)
    k = np.log(a5 / a17) / (17.0 - 5.0)
    A = a5 / np.exp(-k * 5.0)
    return max(floor + A * np.exp(-k * bess_gw), floor)


def _solve_alpha(
    a1: float, a2: float,
    s1: float, s2: float,
    tol: float = 0.001,
) -> float:
    """Bisection to solve: a1*(1+s1^α) = a2*(1+s2^α)."""
    def f(alpha):
        return a1 * (1 + s1 ** alpha) - a2 * (1 + s2 ** alpha)

    lo, hi = 0.5, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2
