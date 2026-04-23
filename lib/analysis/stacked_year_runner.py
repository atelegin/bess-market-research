"""
Annual runner for the stacked-market LP (Note 4 A2.3).

Loops daily ``optimize_day_stacked`` over one calendar year using
pre-fetched data frames (no per-day API calls). Aggregates to annual
per-MW revenue by stream plus diagnostics for policy comparison.

Usage
-----
::

    result = run_stacked_year(2024, duration_h=2.0, max_cycles=2.0)
    print(result.total_keur_per_mw)           # total annual revenue kEUR/MW
    print(result.revenue_by_stream_annual())  # stream breakdown
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.analysis.stacked_day_assembler import (
    DEFAULT_POOL_NEG_MW,
    DEFAULT_POOL_POS_MW,
    assemble_day_inputs,
)
from lib.data.afrr_activations import fetch_afrr_activations
from lib.data.ancillary_prices import (
    fetch_afrr_cap_prices_daily,
    fetch_afrr_energy_prices,
)
from lib.data.day_ahead_prices import fetch_day_ahead_prices
from lib.data.intraday_prices import fetch_id_aep
from lib.models.dispatch_stacked import optimize_day_stacked

logger = logging.getLogger(__name__)


@dataclass
class StackedYearResult:
    year: int
    duration_h: float
    max_cycles: float
    power_mw: float
    days_solved: int
    days_skipped: int
    # Annual totals (EUR, per MW of nameplate)
    total_eur: float
    eur_da: float
    eur_id: float
    eur_afrr_cap_pos: float
    eur_afrr_cap_neg: float
    eur_afrr_energy_pos: float
    eur_afrr_energy_neg: float
    # Diagnostics
    annual_fec: float
    mean_r_pos_mw: float
    mean_r_neg_mw: float

    @property
    def total_keur_per_mw(self) -> float:
        return self.total_eur / 1000.0

    def revenue_by_stream_annual(self) -> dict[str, float]:
        return {
            "da": self.eur_da / 1000.0,
            "id": self.eur_id / 1000.0,
            "afrr_cap_pos": self.eur_afrr_cap_pos / 1000.0,
            "afrr_cap_neg": self.eur_afrr_cap_neg / 1000.0,
            "afrr_energy_pos": self.eur_afrr_energy_pos / 1000.0,
            "afrr_energy_neg": self.eur_afrr_energy_neg / 1000.0,
            "total": self.total_eur / 1000.0,
        }


def _prefetch_year_frames(year: int) -> dict[str, Optional[pd.DataFrame]]:
    """Load all year-level sources once to avoid API hits per day."""
    frames: dict[str, Optional[pd.DataFrame]] = {
        "da": None, "id": None, "afrr_cap": None, "afrr_energy": None,
        "activations": None,
    }
    try:
        frames["da"] = fetch_day_ahead_prices(
            start=f"{year}-01-01", end=f"{year}-12-31",
        )
    except Exception as e:
        logger.warning(f"prefetch({year}): DA frame unavailable: {e}")
    try:
        frames["id"] = fetch_id_aep(start=f"{year}-01-01", end=f"{year}-12-31")
    except Exception as e:
        logger.info(f"prefetch({year}): ID AEP unavailable: {e}")
    try:
        frames["afrr_cap"] = fetch_afrr_cap_prices_daily(year)
    except Exception as e:
        logger.warning(f"prefetch({year}): aFRR cap unavailable: {e}")
    try:
        frames["afrr_energy"] = fetch_afrr_energy_prices(year)
    except Exception as e:
        logger.warning(f"prefetch({year}): aFRR energy unavailable: {e}")
    try:
        frames["activations"] = fetch_afrr_activations(
            start=f"{year}-01-01", end=f"{year + 1}-01-01",
        )
    except Exception as e:
        logger.warning(f"prefetch({year}): activations unavailable: {e}")
    return frames


def run_stacked_year(
    year: int,
    duration_h: float = 2.0,
    max_cycles: float = 2.0,
    power_mw: float = 1.0,
    rte: float = 0.85,
    soc_min_frac: float = 0.05,
    soc_max_frac: float = 0.95,
    afrr_reserve_duration_hours: float = 0.25,
    pool_pos_mw: float = DEFAULT_POOL_POS_MW,
    pool_neg_mw: float = DEFAULT_POOL_NEG_MW,
    wear_cost_eur_per_mwh: np.ndarray | None = None,
    max_afrr_participation: float = 1.0,
    max_days: int | None = None,
) -> StackedYearResult:
    """
    Solve the stacked LP for every day of ``year`` and aggregate annual
    revenue per MW of nameplate.

    Set ``max_days`` for smoke-testing on a partial year.
    """
    frames = _prefetch_year_frames(year)

    energy_mwh = power_mw * duration_h
    days_solved = 0
    days_skipped = 0

    totals = {
        "da": 0.0, "id": 0.0,
        "afrr_cap_pos": 0.0, "afrr_cap_neg": 0.0,
        "afrr_energy_pos": 0.0, "afrr_energy_neg": 0.0,
        "total": 0.0,
    }
    total_fec = 0.0
    sum_r_pos = 0.0
    sum_r_neg = 0.0
    sum_intervals = 0

    current = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    while current < end:
        if max_days is not None and days_solved + days_skipped >= max_days:
            break

        inputs = assemble_day_inputs(
            target_date=current,
            pool_pos_mw=pool_pos_mw, pool_neg_mw=pool_neg_mw,
            da_frame=frames["da"], id_frame=frames["id"],
            afrr_cap_frame=frames["afrr_cap"],
            afrr_energy_frame=frames["afrr_energy"],
            activations_frame=frames["activations"],
        )
        if inputs is None:
            days_skipped += 1
            current += timedelta(days=1)
            continue

        day = optimize_day_stacked(
            **inputs.as_kwargs(),
            energy_mwh=energy_mwh, power_mw=power_mw, rte=rte,
            soc_min_frac=soc_min_frac, soc_max_frac=soc_max_frac,
            max_cycles=max_cycles,
            afrr_reserve_duration_hours=afrr_reserve_duration_hours,
            wear_cost_eur_per_mwh=wear_cost_eur_per_mwh,
            max_afrr_participation=max_afrr_participation,
        )
        if not day.success:
            days_skipped += 1
            current += timedelta(days=1)
            continue

        totals["da"] += day.revenue_da
        totals["id"] += day.revenue_id
        totals["afrr_cap_pos"] += day.revenue_afrr_cap_pos
        totals["afrr_cap_neg"] += day.revenue_afrr_cap_neg
        totals["afrr_energy_pos"] += day.revenue_afrr_energy_pos
        totals["afrr_energy_neg"] += day.revenue_afrr_energy_neg
        totals["total"] += day.revenue_total
        total_fec += day.full_equivalent_cycles
        sum_r_pos += day.r_pos.sum()
        sum_r_neg += day.r_neg.sum()
        sum_intervals += len(day.r_pos)
        days_solved += 1
        current += timedelta(days=1)

    if sum_intervals == 0:
        logger.warning(f"run_stacked_year({year}): no days solved")
        mean_r_pos = mean_r_neg = 0.0
    else:
        mean_r_pos = sum_r_pos / sum_intervals
        mean_r_neg = sum_r_neg / sum_intervals

    return StackedYearResult(
        year=year,
        duration_h=duration_h,
        max_cycles=max_cycles,
        power_mw=power_mw,
        days_solved=days_solved,
        days_skipped=days_skipped,
        total_eur=totals["total"],
        eur_da=totals["da"],
        eur_id=totals["id"],
        eur_afrr_cap_pos=totals["afrr_cap_pos"],
        eur_afrr_cap_neg=totals["afrr_cap_neg"],
        eur_afrr_energy_pos=totals["afrr_energy_pos"],
        eur_afrr_energy_neg=totals["afrr_energy_neg"],
        annual_fec=total_fec,
        mean_r_pos_mw=mean_r_pos,
        mean_r_neg_mw=mean_r_neg,
    )
