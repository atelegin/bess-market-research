"""
Market-saturation trend helpers for Note 4 sensitivity analysis.

Applies Note 1's (``de-bess-outlook``) fleet-equilibrium revenue
projection to an already-computed :class:`LifecycleResult`, without
re-solving the LP. The dispatch decisions are unchanged — only the
annual revenue stream is scaled by Note 1's PER-STREAM factors —
which captures the differential market-decay paths Note 1 projects:

  * **aFRR capacity & FCR collapse** (-85 % from 2026 to 2035): fleet
    growth saturates ancillary demand at ~4.5 GW; everything beyond
    must compete on wholesale.
  * **Wholesale (DA + ID) explosion** (+1000 %): the fleet pushed off
    ancillary onto wholesale lifts spreads, plus PV duck curve and
    demand growth deepen them further.
  * **Total revenue** stays roughly flat over the 10-year horizon
    (~130-150 kEUR/MW/yr) — the streams average out.

Why per-stream matters for Note 4: scalar total scaling assumes a
policy with 50 % aFRR share and a policy with 90 % wholesale share
both decline equally. They don't. Under per-stream scaling, an
aFRR-heavy policy collapses (-50% revenue by 2030) while a
wholesale-heavy policy roughly doubles. This is not a small
correction — it can flip rankings and reverse the "tight envelope
helps aFRR-heavy operators more" finding entirely.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from lib.analysis.lifecycle_npv import LifecycleResult
from lib.config import DEFAULT_BESS_BUILDOUT
from lib.models.projection import project_full_stack


# Mid-case Note 1 per-stream revenue trajectory for a 2 h LFP operating
# 2026-2035 under the default buildout + scenario parameters. Each row is
# a stream; columns are years. Values in kEUR/MW/yr (not ratios — we
# convert to per-year scaling factors in :func:`compute_note1_per_stream_scaling`).
# Generated from ``project_full_stack`` (see ``compute_note1_per_stream_scaling``
# below for regeneration).
NOTE1_MID_CASE_PER_STREAM: dict[str, np.ndarray] = {
    "da":          np.array([ 7.0, 25.6, 40.5, 49.5, 57.3, 61.9, 65.8, 68.8, 70.0, 69.6]),
    "id":          np.array([ 3.5, 13.1, 21.1, 26.3, 31.1, 34.3, 37.1, 39.6, 41.0, 41.5]),
    "afrr_cap":    np.array([120.2, 84.8, 58.8, 44.8, 34.0, 28.6, 24.7, 21.6, 19.8, 18.2]),
    "afrr_energy": np.array([11.5,  8.1,  5.6,  4.3,  3.3,  2.7,  2.4,  2.1,  1.9,  1.7]),
}

# Convenience: total per-year (sum of streams)
NOTE1_MID_CASE_TOTAL_2026_2035: np.ndarray = (
    NOTE1_MID_CASE_PER_STREAM["da"]
    + NOTE1_MID_CASE_PER_STREAM["id"]
    + NOTE1_MID_CASE_PER_STREAM["afrr_cap"]
    + NOTE1_MID_CASE_PER_STREAM["afrr_energy"]
)
NOTE1_MID_CASE_TOTAL_FACTOR_2026_2035: np.ndarray = (
    NOTE1_MID_CASE_TOTAL_2026_2035 / NOTE1_MID_CASE_TOTAL_2026_2035[0]
)
# Backward-compat alias for previous flat-scalar callers
NOTE1_MID_CASE_2026_2035 = NOTE1_MID_CASE_TOTAL_FACTOR_2026_2035


# ── Fleet-saturation yearly scaling schedules ──────────────────────────
# Per-year multipliers on `bid_win_rate` (aFRR cap clearing) and
# `fcr_revenue` representing the BESS-pool growth past ancillary-demand
# capacity. Used in `simulate_lifecycle(yearly_bid_win_rate_scale=...,
# yearly_fcr_revenue_scale=...)` for the multi-year trajectory scenario
# (vs the flat-baseline scenario which uses np.ones(N_YEARS)).
#
# Calendar-year anchors (Y0 = 2024 COD):
#   Y0 (2024) = 1.00 — current state, regelleistung public auction CSVs
#   Y1 (2025) = 1.00 — Clean Horizon Storage Index shows stable revenue
#   Y2 (2026) = 0.85 — fleet ~5 GW vs aFRR demand ~2 GW (saturation onset)
#   Y3 (2027) = 0.70 — continued fleet growth
#   Y4 (2028) = 0.55 — overbuild ratio passes 1.5 (GB DCL precedent timing)
#   Y5 (2029) = 0.40
#   Y6 (2030) = 0.30 — Modo "wholesale 95%" regime
#   Y7 (2031) = 0.25
#   Y8 (2032) = 0.20
#   Y9 (2033) = 0.15 — saturated long-run floor
#
# bid_win_rate (aFRR cap clearing) and FCR revenue follow the same shape
# because both compress with the same fleet-vs-ancillary-demand ratio.
# Caller passes these directly to `simulate_lifecycle` /
# `compare_policies` via `yearly_bid_win_rate_scale` and
# `yearly_fcr_revenue_scale`.
FLEET_SATURATION_BID_WIN_SCALE: np.ndarray = np.array([
    1.00, 1.00, 0.85, 0.70, 0.55, 0.40, 0.30, 0.25, 0.20, 0.15,
])
FLEET_SATURATION_FCR_SCALE: np.ndarray = np.array([
    1.00, 1.00, 0.85, 0.70, 0.55, 0.40, 0.30, 0.25, 0.20, 0.15,
])



def compute_note1_per_stream_scaling(
    start_year: int = 2026,
    n_years: int = 10,
    duration_h: float = 2.0,
    historical_da_keur: float = 105.0,
    canib_max: float = 33.0,
    canib_half: float = 13.0,
    canib_steep: float = 0.17,
) -> dict[str, np.ndarray]:
    """Compute per-stream per-year scaling factors from Note 1.

    For each of (da, id, afrr_cap, afrr_energy), returns an
    ``(n_years,)`` array of factors with year-1 = 1.0. Each factor is
    the ratio of Note 1's projected stream revenue in year y to its
    year-1 value.

    Returns:
        Dict with keys ``da``, ``id``, ``afrr_cap``, ``afrr_energy``.
    """
    years = list(range(start_year, start_year + n_years))
    result = project_full_stack(
        years=years,
        historical_da_keur=historical_da_keur,
        bess_buildout=DEFAULT_BESS_BUILDOUT,
        duration_h=duration_h,
        canib_max=canib_max, canib_half=canib_half, canib_steep=canib_steep,
    )
    out = {}
    for key in ("da", "id", "afrr_cap", "afrr_energy"):
        vals = np.array([r[key] for r in result], dtype=float)
        out[key] = vals / max(vals[0], 1e-9) if vals[0] > 0 else np.ones(n_years)
    return out


def apply_trend_to_result(
    result: LifecycleResult,
    scaling: Optional[np.ndarray | dict[str, np.ndarray]] = None,
) -> LifecycleResult:
    """Return a new :class:`LifecycleResult` with per-year revenue scaled
    using Note 1's PER-STREAM trajectory.

    If the result has per-stream tracking populated, applies one factor
    per stream. Otherwise falls back to scalar-total scaling for
    backward compatibility.

    Args:
        result: Original flat-market result.
        scaling: Either:
          * ``dict[str, np.ndarray]`` with keys
            ``da``/``id``/``afrr_cap``/``afrr_energy`` (preferred;
            invokes the per-stream path)
          * ``np.ndarray`` flat year factors (backward compat)
          * ``None`` — use :data:`NOTE1_MID_CASE_PER_STREAM` factors
            from year-1 baseline
    """
    n = result.n_years
    has_streams = (
        result.annual_revenue_da_eur is not None
        and result.annual_revenue_id_eur is not None
        and result.annual_revenue_afrr_cap_eur is not None
        and result.annual_revenue_afrr_energy_eur is not None
    )

    if scaling is None:
        # Default: per-stream from Note 1 mid-case
        scaling = {
            k: v / max(v[0], 1e-9) for k, v in NOTE1_MID_CASE_PER_STREAM.items()
        }

    if isinstance(scaling, dict) and has_streams:
        # Per-stream path
        factors = {}
        for key in ("da", "id", "afrr_cap", "afrr_energy"):
            f = np.asarray(scaling[key], dtype=float)
            if f.shape[0] < n:
                f = np.concatenate([f, np.full(n - f.shape[0], f[-1])])
            factors[key] = f[:n]
        scaled_da = result.annual_revenue_da_eur * factors["da"]
        scaled_id = result.annual_revenue_id_eur * factors["id"]
        scaled_afrr_cap = result.annual_revenue_afrr_cap_eur * factors["afrr_cap"]
        scaled_afrr_energy = result.annual_revenue_afrr_energy_eur * factors["afrr_energy"]
        scaled_rev = scaled_da + scaled_id + scaled_afrr_cap + scaled_afrr_energy
    else:
        # Fallback: scalar total scaling
        if isinstance(scaling, dict):
            # Compose composite factor weighted by year-1 totals
            scalar_factors = (
                NOTE1_MID_CASE_TOTAL_FACTOR_2026_2035[:n]
            )
        else:
            scalar_factors = np.asarray(scaling, dtype=float)
        if scalar_factors.shape[0] < n:
            scalar_factors = np.concatenate([
                scalar_factors,
                np.full(n - scalar_factors.shape[0], scalar_factors[-1]),
            ])
        scalar_factors = scalar_factors[:n]
        scaled_rev = result.annual_revenue_eur * scalar_factors
        scaled_da = scaled_id = scaled_afrr_cap = scaled_afrr_energy = None

    years = np.arange(1, n + 1)
    discount_factors = (1.0 + result.discount_rate) ** -years
    new_npv = float((scaled_rev * discount_factors).sum())

    update = dict(
        policy_name=f"{result.policy_name}_trend",
        annual_revenue_eur=scaled_rev,
        lifetime_npv_eur=new_npv,
    )
    if scaled_da is not None:
        update.update(
            annual_revenue_da_eur=scaled_da,
            annual_revenue_id_eur=scaled_id,
            annual_revenue_afrr_cap_eur=scaled_afrr_cap,
            annual_revenue_afrr_energy_eur=scaled_afrr_energy,
        )
    return replace(result, **update)
