"""
Per-day physics-based degradation for lifecycle simulation (Note 4 step 2).

Replaces the linear-in-FEC ``degradation_per_day`` from
:mod:`lib.models.adp.solver` with a call into the Note 3 physics kernel
(``lib.models.degradation.detailed.project_capacity_detailed``).

Why
---
The linear formula loses DoD, SoC-band, C-rate, and temperature
sensitivities that Note 3 calibrated against Naumann 2018 + SNL 2020 +
Stanford 2024 anchors. For Note 4 policy comparisons this matters
because aging-aware policies don't just cycle *less* — they cycle at
lower DoD, in mid-SoC bands, and at lower C-rates. The linear
formulation ignores those second-order benefits, understating the gap
between aging-aware and naive.

Design (daily physics call)
---------------------------
For each day, build a :class:`DutyCycle` representing "this day's
duty sustained for a year" (i.e. ``fec_per_year = fec_today × 365`` and
annualised SoC distribution), call :func:`project_capacity_detailed`
with ``years=1.0`` to get annual fade under that duty, then divide by
365 for this single day's contribution. Scale the result by an
age-acceleration factor (``1 + 2.5·(1 − SoH)``) for the scarcity signal
the DP relies on.

Why ``years=1.0 / 365`` is NOT the right call
---------------------------------------------
Note 3's kernel has a square-root calendar channel (Naumann 2018
SEI-growth kinetics) — evaluating at ``years=1/365`` amplifies the
calendar component to ~5 % of full-year value even for a single day
(√(1/365) ≈ 0.052), systematically over-estimating daily fade. The
``annual/365`` approach avoids this by evaluating the kernel at a
horizon it's calibrated for, then attributing 1/365 of the annual
decay to today.

Notes on the age scaling
------------------------
Note 3's kernel is fresh-cell calibrated; extending to aged-cell
behaviour strictly requires re-calibration on aged-cell data (not
available in-project). The simple multiplicative age factor is a
pragmatic approximation — same one the linear model used — so swapping
physics for linear preserves *incremental* realism (DoD/SoC/C-rate)
without introducing new calibration risk on aged cells.

Inputs
------
``day_result`` — :class:`lib.models.dispatch.stacked.StackedDispatchResult`
from the daily LP solve.
``energy_mwh`` — usable energy (MWh) used for normalisation. Typically
``power_mw × duration_h × soh_current`` (SoH-derated).
``soh_current`` — current state of health in [warranty_floor, 1.0].
``preset`` — Note 3 :class:`CellPreset` (default ``eve_lf280k``).
``temperature_c`` — cell-internal temperature (default 25 °C, fixed).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from lib.models.degradation.simple import PRESETS, CellPreset
from lib.models.degradation.detailed import (
    DutyCycle,
    project_capacity_detailed,
)

if TYPE_CHECKING:
    from lib.models.dispatch.stacked import StackedDispatchResult


DEFAULT_PRESET: CellPreset = PRESETS["eve_lf280k"]


def physics_degradation_per_day(
    day_result: "StackedDispatchResult",
    energy_mwh: float,
    soh_current: float,
    preset: Optional[CellPreset] = None,
    temperature_c: float = 25.0,
    age_accel_slope: float = 2.5,
    kernel_scale: float = 1.0,
) -> float:
    """Compute Δ SoH for one day under the Note 3 physics kernel.

    Args:
        day_result: Daily LP output (SoC trace + power + FEC + activations).
        energy_mwh: Usable MWh (for normalising SoC fraction and C-rate).
        soh_current: Current SoH; used for age-scaling.
        preset: Note 3 CellPreset (defaults to EVE LF280K).
        temperature_c: Cell-internal temperature (°C). Default 25 (indoor).
        age_accel_slope: Acceleration factor slope; SoH-reduction consumes
            life faster as SoH drops. Default 2.5 matches the linear model.
        kernel_scale: Joint multiplier on ``k_cal`` and ``k_cyc`` of the
            preset. Default ``1.0`` reproduces the academic Naumann /
            Stanford-anchored kernel (gated by Note 3's tripwire test).
            Set to ``0.66`` to land the EVE LF280K manufacturer endurance
            anchor (6000 cycles to 80 % retention at 25 °C / 0.5 C / 1.0
            DoD) — used by Note 4 paper headline. The scale represents
            the residual physical difference between Stanford-era K2
            18650 academic cells and modern 280 Ah prismatic LFP cells.

    Returns:
        Positive Δ SoH to subtract from current SoH.
    """
    if preset is None:
        preset = DEFAULT_PRESET
    if kernel_scale != 1.0:
        from dataclasses import replace
        preset = replace(
            preset,
            k_cal=preset.k_cal * float(kernel_scale),
            k_cyc=preset.k_cyc * float(kernel_scale),
        )

    fec_day = float(day_result.full_equivalent_cycles)
    # Zero-dispatch day: only calendar fade applies. Call at years=1 and
    # divide by 365 — kernel is calibrated at annual horizons.
    if fec_day < 1e-6:
        duty = DutyCycle.from_mean(
            fec_per_year=0.0,
            mean_dod=0.05,
            mean_soc=0.50,
            mean_crate=0.0,
            mean_temp_C=temperature_c,
        )
        cap = project_capacity_detailed(duty, years=1.0, preset=preset, n_mc=40)
        annual_fade = max(0.0, 1.0 - float(cap))
        base = annual_fade / 365.0
        age_accel = 1.0 + age_accel_slope * max(0.0, 1.0 - soh_current)
        return base * age_accel

    soc_frac = np.asarray(day_result.soc, dtype=float) / max(energy_mwh, 1e-9)
    soc_frac = np.clip(soc_frac, 0.0, 1.0)

    # Rough mean-DoD from SoC span. For multi-cycle days this underestimates
    # average cycle depth, but for single-cycle-per-day it's exact. Note 3
    # kernel's DoD sensitivity is moderate, so the approximation is OK.
    mean_dod = float(soc_frac.max() - soc_frac.min())
    mean_dod = max(min(mean_dod, 1.0), 0.01)

    net_power = (
        np.asarray(day_result.discharge_da, dtype=float)
        + np.asarray(day_result.discharge_id, dtype=float)
        + np.asarray(day_result.a_pos, dtype=float)
        - np.asarray(day_result.charge_da, dtype=float)
        - np.asarray(day_result.charge_id, dtype=float)
        - np.asarray(day_result.a_neg, dtype=float)
    )
    active_mask = np.abs(net_power) > 0.01
    if active_mask.any():
        mean_crate = float(np.abs(net_power[active_mask]).mean() / max(energy_mwh, 1e-9))
    else:
        mean_crate = 0.1
    mean_crate = float(np.clip(mean_crate, 0.05, 2.0))

    duty = DutyCycle.from_timeseries(
        soc=soc_frac,
        fec_per_year=fec_day * 365.0,
        mean_dod=mean_dod,
        mean_crate=mean_crate,
        mean_temp_C=temperature_c,
    )
    # Evaluate kernel at years=1.0 (calibration horizon), then take 1/365 of
    # the resulting annual fade as today's contribution. This avoids the
    # sqrt-calendar amplification of small-horizon calls.
    capacity_frac = float(project_capacity_detailed(
        duty=duty, years=1.0, preset=preset, n_mc=40,
    ))
    annual_fade = max(0.0, 1.0 - capacity_frac)
    base = annual_fade / 365.0
    age_accel = 1.0 + age_accel_slope * max(0.0, 1.0 - soh_current)
    return base * age_accel
