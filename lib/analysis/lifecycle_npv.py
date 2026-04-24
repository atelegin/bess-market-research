"""
Lifecycle NPV simulator for policy comparison (Note 4 A4).

Runs an annual loop over N years for a given shadow-cost policy. Each
day: market data → policy gives wear-cost vector → stacked-market LP
solves → dispatch accumulates → SoH degrades. Year-end: discount annual
revenue to present. Lifetime NPV = discounted sum.

Designed for side-by-side policy comparison: same price series, same
market mechanics, same degradation physics — the *only* difference
between runs is the policy's wear cost vector.

Price data rotation
-------------------
We have observed 2023 and 2025 regelleistung + netztransparenz +
EnergyCharts data (2024 has an unreliable DA fetch). The simulator
rotates through a list of available "template years" for each year in
the simulation horizon; by default ``[2023, 2025]`` alternating.
Physical realism of using old prices for a future year is modest but
sufficient for *relative* policy comparison which is what Note 4
cares about.

Degradation
-----------
Uses the same :func:`lib.models.adp_solver.degradation_per_day` formula
as the DP, to keep policies consistent with the backward-induction
model. Daily SoH decrement = linear-in-FEC + mild age acceleration +
calendar fade. At end-of-life (SoH ≤ warranty_floor) the year terminates
and remaining years contribute zero revenue.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np

from lib.analysis.stacked_day_assembler import (
    DEFAULT_POOL_NEG_MW,
    DEFAULT_POOL_POS_MW,
    assemble_day_inputs,
)
from lib.analysis.stacked_year_runner import _prefetch_year_frames
from lib.models.adp_shadow_cost import ShadowCostPolicy
from lib.models.adp_solver import degradation_per_day
from lib.models.dispatch_stacked import optimize_day_stacked

logger = logging.getLogger(__name__)


@dataclass
class LifecycleResult:
    """Per-policy lifetime simulation output."""

    policy_name: str
    n_years: int
    initial_soh: float
    warranty_floor: float
    discount_rate: float
    annual_revenue_eur: np.ndarray       # shape (n_years,), nominal EUR/MW
    annual_fec: np.ndarray               # shape (n_years,)
    end_of_year_soh: np.ndarray          # shape (n_years,)
    days_solved: int
    days_skipped: int
    lifetime_npv_eur: float              # discounted to present, EUR/MW
    years_to_floor: Optional[int] = None  # first year SoH hit floor; None if never
    # Optional dispatch logs for diagnostics. Populated only when
    # ``collect_diagnostics_year`` passed to simulate_lifecycle.
    diagnostic_days: Optional[list] = None  # list[DayDiagnosticData] — first collected year

    def revenue_by_year_keur(self) -> np.ndarray:
        return self.annual_revenue_eur / 1000.0


def _rotation_year(sim_year_idx: int, template_years: tuple[int, ...]) -> int:
    return template_years[sim_year_idx % len(template_years)]


def simulate_lifecycle(
    policy: ShadowCostPolicy,
    n_years: int = 15,
    template_years: tuple[int, ...] = (2023, 2025),
    initial_soh: float = 1.0,
    warranty_floor: float = 0.80,
    discount_rate: float = 0.07,
    power_mw: float = 1.0,
    duration_h: float = 2.0,
    rte: float = 0.85,
    max_cycles: float = 2.0,
    max_afrr_participation: float = 0.40,
    afrr_reserve_duration_hours: float = 0.25,
    pool_pos_mw: float = DEFAULT_POOL_POS_MW,
    pool_neg_mw: float = DEFAULT_POOL_NEG_MW,
    fade_per_fec_at_soh_1: float = 3.3e-5,
    calendar_fade_per_day: float = 2e-5,
    use_physics_degradation: bool = False,
    physics_preset_name: str = "eve_lf280k",
    physics_temperature_c: float = 25.0,
    max_days_per_year: Optional[int] = None,
    prefetched_frames: Optional[dict[int, dict]] = None,
    collect_diagnostics_year: Optional[int] = None,
) -> LifecycleResult:
    """
    Simulate lifetime revenue under a shadow-cost policy.

    Args:
        policy: A :class:`ShadowCostPolicy` that produces per-day wear
            cost vectors given ``(soh_current, day_of_year)``.
        n_years: Horizon in years (typical 10–15 for LFP).
        template_years: Historical years whose market data is replayed
            in rotation. Default ``(2023, 2025)`` — both have complete
            regelleistung + netztransparenz + EnergyCharts coverage.
        initial_soh: Starting state of health.
        warranty_floor: Below this SoH, year terminates and subsequent
            years contribute zero revenue.
        discount_rate: Annual discount (default 7% — typical merchant-
            BESS hurdle rate).
        power_mw / duration_h / rte: Asset specs (nominal, 2h LFP 1 MW).
        max_cycles: Daily cycling cap in the LP (same for all policies
            — differentiation comes from the wear cost vector).
        max_afrr_participation: aFRR reservation cap (same as Note 1
            calibrated ≈ 0.40).
        fade_per_fec_at_soh_1 / calendar_fade_per_day: Degradation
            parameters (shared with ADP solver).
        max_days_per_year: Cap days per year for fast smoke testing
            (default None = full year).
        prefetched_frames: If provided, reuse these per-year frame
            dicts instead of re-fetching. Keyed by ``template_year``.
            Multi-policy comparisons should pass the same dict to all
            calls for speed + identical inputs.
        collect_diagnostics_year: 0-based year index whose dispatch
            days should be retained on the result for diagnostic
            signal computation. Memory-cheap: one year × 96 floats × 4
            arrays × ~365 days ≈ 5 MB.

    Returns:
        :class:`LifecycleResult` with per-year revenue, SoH trajectory,
        FEC count, and discounted lifetime NPV.
    """
    annual_revenue = np.zeros(n_years)
    end_of_year_soh = np.zeros(n_years)
    annual_fec = np.zeros(n_years)
    days_solved = 0
    days_skipped = 0
    years_to_floor: Optional[int] = None

    current_soh = float(initial_soh)
    diagnostic_days: list = []

    # Optional Note 3 physics kernel
    physics_preset = None
    if use_physics_degradation:
        from lib.models.degradation import PRESETS
        physics_preset = PRESETS[physics_preset_name]
        from lib.analysis.physics_degradation import physics_degradation_per_day

    # Prefetch frames per template year once.
    frames_by_year: dict[int, dict] = dict(prefetched_frames) if prefetched_frames else {}
    for ty in template_years:
        if ty not in frames_by_year:
            frames_by_year[ty] = _prefetch_year_frames(ty)

    for y_idx in range(n_years):
        if current_soh <= warranty_floor + 1e-9:
            # Warranty floor hit; no further operation
            end_of_year_soh[y_idx] = current_soh
            if years_to_floor is None:
                years_to_floor = y_idx
            continue

        template = _rotation_year(y_idx, template_years)
        frames = frames_by_year[template]
        start_date = date(template, 1, 1)
        end_date = date(template + 1, 1, 1)
        current = start_date
        day_idx = 0
        year_revenue = 0.0
        year_fec = 0.0
        while current < end_date:
            if max_days_per_year is not None and day_idx >= max_days_per_year:
                break
            inputs = assemble_day_inputs(
                target_date=current,
                pool_pos_mw=pool_pos_mw, pool_neg_mw=pool_neg_mw,
                da_frame=frames.get("da"), id_frame=frames.get("id"),
                spot_frame=frames.get("spot"),
                afrr_cap_frame=frames.get("afrr_cap"),
                afrr_energy_frame=frames.get("afrr_energy"),
                activations_frame=frames.get("activations"),
            )
            if inputs is None:
                days_skipped += 1
                current += timedelta(days=1)
                day_idx += 1
                continue

            # Policy-derived per-interval wear cost
            day_of_year = (current - start_date).days + 1
            wear = policy.wear_cost(
                soh_current=current_soh,
                day_of_year=day_of_year,
                periods_per_day=96,
                duration_h=duration_h,
            )

            # SoH-derated usable energy (SoC window scales with SoH)
            usable_energy_mwh = power_mw * duration_h * current_soh

            day_out = optimize_day_stacked(
                **inputs.as_kwargs(),
                energy_mwh=usable_energy_mwh, power_mw=power_mw, rte=rte,
                max_cycles=max_cycles,
                afrr_reserve_duration_hours=afrr_reserve_duration_hours,
                max_afrr_participation=max_afrr_participation,
                wear_cost_eur_per_mwh=wear,
            )
            if not day_out.success:
                days_skipped += 1
                current += timedelta(days=1)
                day_idx += 1
                continue

            year_revenue += day_out.revenue_total
            year_fec += day_out.full_equivalent_cycles
            days_solved += 1

            # Collect diagnostic-day record if this is the target year.
            if collect_diagnostics_year is not None and y_idx == collect_diagnostics_year:
                from lib.analysis.aging_aware_diagnostics import (
                    day_diagnostic_from_stacked,
                )
                diagnostic_days.append(
                    day_diagnostic_from_stacked(
                        day_result=day_out, inputs=inputs,
                        target_date=current, power_mw=power_mw,
                        energy_mwh=usable_energy_mwh,
                    )
                )

            # Degrade SoH for today's throughput
            if use_physics_degradation and physics_preset is not None:
                delta = physics_degradation_per_day(
                    day_result=day_out,
                    energy_mwh=usable_energy_mwh,
                    soh_current=current_soh,
                    preset=physics_preset,
                    temperature_c=physics_temperature_c,
                )
            else:
                delta = degradation_per_day(
                    intensity=day_out.full_equivalent_cycles,
                    soh=current_soh,
                    fade_per_fec_at_soh_1=fade_per_fec_at_soh_1,
                    calendar_fade_per_day=calendar_fade_per_day,
                )
            current_soh -= delta
            if current_soh < warranty_floor:
                current_soh = warranty_floor
                if years_to_floor is None:
                    years_to_floor = y_idx
                break

            current += timedelta(days=1)
            day_idx += 1

        annual_revenue[y_idx] = year_revenue
        annual_fec[y_idx] = year_fec
        end_of_year_soh[y_idx] = current_soh

    # Discount annual revenue to present (year 1 discounted by 1/(1+r)^1, etc.)
    years = np.arange(1, n_years + 1)
    discount_factors = (1.0 + discount_rate) ** -years
    lifetime_npv = float((annual_revenue * discount_factors).sum())

    return LifecycleResult(
        policy_name=policy.name,
        n_years=n_years,
        initial_soh=initial_soh,
        warranty_floor=warranty_floor,
        discount_rate=discount_rate,
        annual_revenue_eur=annual_revenue,
        annual_fec=annual_fec,
        end_of_year_soh=end_of_year_soh,
        days_solved=days_solved,
        days_skipped=days_skipped,
        lifetime_npv_eur=lifetime_npv,
        years_to_floor=years_to_floor,
        diagnostic_days=diagnostic_days if diagnostic_days else None,
    )


def compare_policies(
    policies: dict[str, ShadowCostPolicy],
    **simulate_kwargs,
) -> dict[str, LifecycleResult]:
    """
    Run :func:`simulate_lifecycle` for each policy sharing prefetched
    market frames. Returns results keyed by policy name.

    All policies see identical market inputs — the only differentiation
    is their wear cost vector. This is the critical property for
    Note 4's comparison claims.
    """
    # Prefetch once, reuse across all policies.
    template_years = simulate_kwargs.get("template_years", (2023, 2025))
    shared_frames = {ty: _prefetch_year_frames(ty) for ty in template_years}
    results: dict[str, LifecycleResult] = {}
    for name, policy in policies.items():
        logger.info(f"compare_policies: simulating {name}…")
        results[name] = simulate_lifecycle(
            policy=policy,
            prefetched_frames=shared_frames,
            **simulate_kwargs,
        )
    return results
