"""
Battery degradation model — calendar and cycle ageing.

Computes residual capacity from depth-of-discharge, annual cycle count,
and calendar fade over the project lifetime.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DegradationAssumptions:
    reference_cycle_life_fec: float = 7300.0
    calendar_life_years: float = 20.0
    eol_capacity_fraction: float = 0.60
    calendar_capacity_loss_at_reference: float = 0.18
    dod_reference: float = 1.0
    dod_exponent: float = 1.30
    cycle_fade_exponent: float = 1.25

    @property
    def reference_warranty_fec_per_year(self) -> float:
        return self.reference_cycle_life_fec / self.calendar_life_years


DEFAULT_DEGRADATION_ASSUMPTIONS = DegradationAssumptions()


def compute_annual_degradation(
    cycles_per_year: float,
    avg_dod: float,
    base_cycle_degradation: float = 0.003,
    dod_exponent: float = 1.5,
    calendar_aging: float = 0.008,
) -> float:
    dod_factor = (max(avg_dod, 1e-6) / 0.8) ** dod_exponent
    cycle_deg = cycles_per_year * (base_cycle_degradation / 100) * dod_factor
    return calendar_aging + cycle_deg


def equivalent_stress_fec_per_year(
    dispatch_frame: pd.DataFrame,
    assumptions: DegradationAssumptions = DEFAULT_DEGRADATION_ASSUMPTIONS,
) -> float:
    if dispatch_frame.empty:
        return 0.0
    return float(dispatch_frame["full_equivalent_cycles"].fillna(0.0).sum())


def project_capacity_fraction(
    elapsed_years: float,
    annual_stress_fec: float,
    assumptions: DegradationAssumptions = DEFAULT_DEGRADATION_ASSUMPTIONS,
) -> float:
    if elapsed_years <= 0:
        return 1.0
    cycle_loss_budget = max(1.0 - assumptions.eol_capacity_fraction - assumptions.calendar_capacity_loss_at_reference, 0.0)
    calendar_loss = assumptions.calendar_capacity_loss_at_reference * np.sqrt(
        max(elapsed_years / max(assumptions.calendar_life_years, 1e-9), 0.0)
    )
    cycle_ratio = max((annual_stress_fec * elapsed_years) / max(assumptions.reference_cycle_life_fec, 1e-9), 0.0)
    cycle_loss = cycle_loss_budget * (cycle_ratio**assumptions.cycle_fade_exponent)
    return float(max(1.0 - calendar_loss - cycle_loss, 0.0))


def estimate_years_to_eol(
    dispatch_frame: pd.DataFrame,
    assumptions: DegradationAssumptions = DEFAULT_DEGRADATION_ASSUMPTIONS,
    max_years: float = 40.0,
    step_years: float = 1.0 / 12.0,
) -> float:
    annual_stress_fec = equivalent_stress_fec_per_year(dispatch_frame, assumptions=assumptions)
    elapsed_years = 0.0
    while elapsed_years <= max_years:
        capacity_fraction = project_capacity_fraction(
            elapsed_years=elapsed_years,
            annual_stress_fec=annual_stress_fec,
            assumptions=assumptions,
        )
        if capacity_fraction <= assumptions.eol_capacity_fraction:
            return float(elapsed_years)
        elapsed_years += step_years
    return float(max_years)


def lifecycle_value_profile(
    year1_revenue: float,
    dispatch_frame: pd.DataFrame,
    years: int,
    discount_rate: float = 0.0,
    annual_market_decline: float = 0.0,
    assumptions: DegradationAssumptions = DEFAULT_DEGRADATION_ASSUMPTIONS,
) -> pd.DataFrame:
    annual_stress_fec = equivalent_stress_fec_per_year(dispatch_frame, assumptions=assumptions)
    records = []
    cumulative_revenue = 0.0
    cumulative_discounted = 0.0
    for year in range(1, years + 1):
        elapsed_years = year - 1
        capacity_fraction = project_capacity_fraction(
            elapsed_years=elapsed_years,
            annual_stress_fec=annual_stress_fec,
            assumptions=assumptions,
        )
        retired = bool(elapsed_years > 0 and capacity_fraction <= assumptions.eol_capacity_fraction)
        market_factor = (1 + annual_market_decline) ** elapsed_years
        annual_revenue = 0.0 if retired else float(year1_revenue * capacity_fraction * market_factor)
        discounted_revenue = 0.0 if retired else float(annual_revenue / (1 + discount_rate) ** elapsed_years)
        cumulative_revenue += annual_revenue
        cumulative_discounted += discounted_revenue
        records.append(
            {
                "year": year,
                "capacity_fraction_start": capacity_fraction,
                "annual_revenue_eur_per_mw": annual_revenue,
                "discounted_revenue_eur_per_mw": discounted_revenue,
                "cumulative_revenue_eur_per_mw": cumulative_revenue,
                "cumulative_discounted_revenue_eur_per_mw": cumulative_discounted,
                "retired": retired,
            }
        )
    return pd.DataFrame(records)


def summarize_dispatch_degradation(dispatch_frame: pd.DataFrame) -> dict[str, float]:
    cycles_per_year = float(dispatch_frame["cycles"].sum())
    if cycles_per_year > 0:
        avg_dod = float(np.average(dispatch_frame["avg_dod"], weights=dispatch_frame["cycles"]))
    else:
        avg_dod = 0.0
    annual_degradation = compute_annual_degradation(cycles_per_year=cycles_per_year, avg_dod=avg_dod)
    return {
        "cycles_per_year": cycles_per_year,
        "full_equivalent_cycles_per_year": float(dispatch_frame["full_equivalent_cycles"].sum()) if "full_equivalent_cycles" in dispatch_frame else 0.0,
        "stress_fec_per_year": equivalent_stress_fec_per_year(dispatch_frame),
        "avg_dod": avg_dod,
        "annual_degradation": annual_degradation,
        "years_to_eol": estimate_years_to_eol(dispatch_frame),
    }


def capacity_trajectory(annual_degradation: float, years: int) -> np.ndarray:
    return np.array([(1 - annual_degradation) ** year for year in range(years + 1)])


def cumulative_revenue_profile(
    year1_revenue: float,
    annual_degradation: float,
    years: int,
) -> pd.DataFrame:
    capacity = capacity_trajectory(annual_degradation=annual_degradation, years=years)
    annual_revenue = year1_revenue * capacity[:-1]
    return pd.DataFrame(
        {
            "year": np.arange(1, years + 1),
            "annual_revenue_eur_per_mw": annual_revenue,
            "cumulative_revenue_eur_per_mw": np.cumsum(annual_revenue),
            "capacity_fraction": capacity[:-1],
        }
    )
