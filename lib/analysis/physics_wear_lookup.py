"""
Physics-informed wear cost lookups for the progressive stack (Note 4).

Two functions here, same underlying Note 3 Wang+Naumann kernel:

1. :func:`physics_wear_cost_per_mwh` — **1D SoH lookup**. Fixed
   reference duty (DoD 0.6, C-rate 0.5, temp 25 °C). Fast, offline,
   returns an array indexed by SoH grid. Used as an additive layer in
   :class:`ADPPolicyIntraday` for cheap SoH-dependent aging.

2. :func:`physics_wear_from_duty` — **Per-day, duty-dependent**. Called
   in a 2-pass LP: after the first LP solve reveals the day's actual
   SoC trajectory and dispatch, build a :class:`DutyCycle` from the
   observed ``day_result``, invoke the physics kernel, and monetise
   the fade into an EUR/MWh throughput cost. This is the "real"
   physics-informed shadow cost — it varies not just with SoH but
   with *how* the battery was actually cycled that day (deep vs
   shallow, high-C vs low-C). Closes Note 4 methodology simplification
   #2 properly: the physics kernel's DoD + C-rate sensitivities enter
   the optimisation, not just its SoH axis.

Both functions return EUR per MWh of throughput, suitable for the
``wear_cost_eur_per_mwh`` parameter of
:func:`lib.models.dispatch.stacked.optimize_day_stacked`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from lib.analysis.physics_degradation import physics_degradation_per_day
from lib.models.degradation.simple import CellPreset
from lib.models.degradation.detailed import DutyCycle, project_capacity_detailed

if TYPE_CHECKING:
    from lib.models.dispatch.stacked import StackedDispatchResult


def physics_wear_cost_per_mwh(
    soh_grid: np.ndarray,
    preset: CellPreset,
    temperature_c: float = 25.0,
    reference_fec_per_day: float = 1.5,
    reference_dod: float = 0.60,
    reference_crate: float = 0.50,
    reference_soc_mid: float = 0.50,
    capex_eur_per_mwh: float = 100_000.0,
    warranty_floor: float = 0.80,
    age_accel_slope: float = 2.5,
    epsilon: float = 0.005,
    n_mc: int = 80,
    kernel_scale: float = 1.0,
) -> np.ndarray:
    """Per-SoH physics-derived wear cost in EUR/MWh throughput.

    Args:
        soh_grid: SoH grid values (e.g. ``np.linspace(0.80, 1.00, 11)``).
        preset: Note 3 :class:`CellPreset` (default in caller: ``eve_lf280k``).
        temperature_c: Cell-internal temperature, °C (default 25).
        reference_fec_per_day: FEC/day for the reference duty. 1.5 is
            representative of a moderately active 2 h BESS (≈550 FEC/year).
        reference_dod, reference_crate, reference_soc_mid: Representative
            duty-cycle shape. DoD ≈ 0.60, C-rate ≈ 0.50, SoC mid-band
            match typical German arbitrage operation at 2 h duration.
        capex_eur_per_mwh: Installed CAPEX (default €100 k/MWh, matches
            :class:`DepreciationProxyPolicy` anchor).
        warranty_floor: SoH below which the warranty is breached (0.80).
        age_accel_slope: Per-unit SoH-loss acceleration applied to fade.
            Matches the linear model's 2.5 slope — physics kernel is fresh-
            cell calibrated, so aged-cell behaviour is approximated by this
            multiplicative factor. Documented limitation.
        epsilon: Numerical safety for the headroom denominator near the
            warranty floor.
        n_mc: Monte Carlo samples for the physics kernel (default 80,
            ~2 s per call — cheap since only called once per SoH point).

    Returns:
        np.ndarray shape ``(len(soh_grid),)`` — EUR/MWh throughput cost
        for the intraday DP's Bellman equation.
    """
    if kernel_scale != 1.0:
        from dataclasses import replace
        preset = replace(
            preset,
            k_cal=preset.k_cal * float(kernel_scale),
            k_cyc=preset.k_cyc * float(kernel_scale),
        )
    duty = DutyCycle.from_mean(
        fec_per_year=reference_fec_per_day * 365.0,
        mean_dod=reference_dod,
        mean_soc=reference_soc_mid,
        mean_crate=reference_crate,
        mean_temp_C=temperature_c,
    )
    capacity_frac = float(project_capacity_detailed(
        duty=duty, years=1.0, preset=preset, n_mc=n_mc,
    ))
    annual_fade_soh1 = max(1e-6, 1.0 - capacity_frac)

    # Annual throughput budget: FEC × 365 days × 2 (charge + discharge)
    # in MWh per MWh-of-usable-energy. The fade per-MWh is computed
    # per unit-energy basis — the LP then multiplies by absolute MWh of
    # throughput, giving the absolute EUR cost.
    annual_throughput_per_unit = reference_fec_per_day * 365.0 * 2.0
    fade_per_mwh_soh1 = annual_fade_soh1 / annual_throughput_per_unit

    wear = np.zeros(len(soh_grid))
    for i, soh in enumerate(soh_grid):
        soh = float(soh)
        age_accel = 1.0 + age_accel_slope * max(0.0, 1.0 - soh)
        fade_per_mwh = fade_per_mwh_soh1 * age_accel
        headroom = max(soh - warranty_floor, epsilon)
        wear[i] = fade_per_mwh * capex_eur_per_mwh / headroom
    return wear


def physics_wear_from_duty(
    day_result: "StackedDispatchResult",
    energy_mwh: float,
    soh_current: float,
    preset: CellPreset,
    temperature_c: float = 25.0,
    capex_eur_per_mwh: float = 100_000.0,
    warranty_floor: float = 0.80,
    epsilon: float = 0.005,
    max_wear_eur_per_mwh: float = 500.0,
    kernel_scale: float = 1.0,
) -> float:
    """Scalar EUR/MWh throughput cost from the kernel applied to an
    observed day's dispatch.

    Pipeline:

    1. Feed ``day_result`` to :func:`physics_degradation_per_day`, which
       builds a :class:`DutyCycle` from the actual SoC trace + dispatch
       power and calls the Note 3 kernel at ``years=1.0`` (calibration
       horizon), then divides by 365 for today's fade.
    2. Convert absolute fade to per-MWh-throughput: divide by the day's
       observed throughput (charge + discharge, approximately
       ``2 × FEC × energy_mwh``).
    3. Monetise: multiply by ``CAPEX / headroom(SoH)`` — standard
       scarcity-value-of-lost-SoH formula.
    4. Cap at ``max_wear_eur_per_mwh`` (default €500 / MWh) to keep LP
       numerics well-behaved near the warranty floor, where raw physics
       wear can diverge to thousands of EUR/MWh and either stall the
       solver or effectively zero out all cycling.

    Returns a **scalar** — the wear cost is the same per-interval
    estimate for the day. Pass this scalar as a flat vector (``np.full
    (96, wear)``) back into :func:`optimize_day_stacked` for pass 2.

    Zero-FEC days (no cycling observed): returns ``0.0`` — no
    throughput to price. Calendar fade is captured separately in the
    SoH update loop.
    """
    fec_day = float(day_result.full_equivalent_cycles)
    if fec_day < 1e-6 or energy_mwh <= 0:
        return 0.0
    # Day-level fade from observed duty (captures DoD, C-rate, SoC-band)
    fade_today = physics_degradation_per_day(
        day_result=day_result,
        energy_mwh=energy_mwh,
        soh_current=soh_current,
        preset=preset,
        temperature_c=temperature_c,
        kernel_scale=kernel_scale,
    )
    # Observed throughput this day: ~2 × FEC × usable_energy MWh
    throughput_mwh = 2.0 * fec_day * energy_mwh
    if throughput_mwh < 1e-6:
        return 0.0
    fade_per_mwh = fade_today / throughput_mwh
    headroom = max(soh_current - warranty_floor, epsilon)
    wear = fade_per_mwh * capex_eur_per_mwh / headroom
    return float(min(wear, max_wear_eur_per_mwh))
