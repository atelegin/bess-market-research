"""
Ancillary revenue model — FCR + aFRR saturation.

Models revenue collapse as BESS fleet outgrows ancillary market depth.
"""

import numpy as np
from lib.config import (
    FCR_DEMAND_MW, AFRR_DEPTH_MW, ANCILLARY_COMBINED_GW,
    DEFAULT_BESS_BUILDOUT,
)


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
    r_anc_2026: float = 135.0,     # kEUR/MW total ancillary at 5 GW
    r_anc_2030: float = 13.0,      # kEUR/MW total ancillary at 17 GW
    r_anc_floor: float = 2.0,      # residual floor: minimum participation revenue
    ancillary_depth_gw: float = ANCILLARY_COMBINED_GW,
) -> dict[str, float]:
    """
    Compute ancillary revenue per MW using saturation model.

    R_anc(t) = floor + amplitude / (1 + (bess_gw / depth)^alpha)

    Calibrated for 2h battery. Duration scaling: FCR/aFRR are auctioned
    in 4h blocks — a 1h battery can only participate ~50% of the time
    (must reserve energy), while 2h+ batteries can participate fully.

    Returns dict with fcr, afrr_cap, afrr_energy, total (all kEUR/MW/yr).
    """
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
