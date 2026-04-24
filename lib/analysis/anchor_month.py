"""
Anchor-month simulator for Note 4 (C-plan: Feb 2026 test, swap to March
when CH index publishes).

Runs each shadow-cost policy over a single calendar month on real market
data at starting SoH=1.0, producing per-day dispatch logs, monthly
revenue totals, and diagnostic signals — ready for the benchmark-fan
comparison (when public benchmarks for the target month are available).

Unlike the 10-year lifecycle simulator, anchor-month:
  * uses a single fresh-cell SoH (no multi-year age drift)
  * keeps all per-day dispatch logs (small memory: 30 days × 5 policies)
  * targets narrative depth over statistical robustness
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.analysis.aging_aware_diagnostics import (
    DayDiagnosticData,
    compute_all_signals,
    day_diagnostic_from_stacked,
)
from lib.analysis.stacked_day_assembler import (
    DEFAULT_POOL_NEG_MW,
    DEFAULT_POOL_POS_MW,
    assemble_day_inputs,
)
from lib.analysis.stacked_year_runner import _prefetch_year_frames
from lib.models.adp_shadow_cost import ShadowCostPolicy
from lib.models.dispatch_stacked import optimize_day_stacked

logger = logging.getLogger(__name__)


@dataclass
class AnchorMonthPolicyResult:
    """Per-policy output for the anchor month."""
    policy_name: str
    days_solved: int
    monthly_revenue_eur: float
    annualised_revenue_keur_per_mw: float   # × 12 — for fan-chart comparability
    total_fec: float
    mean_r_pos_mw: float
    mean_r_neg_mw: float
    daily_logs: list[DayDiagnosticData] = field(default_factory=list)
    diagnostics: object | None = None   # DiagnosticReport


@dataclass
class AnchorMonthRun:
    year: int
    month: int
    results: dict[str, AnchorMonthPolicyResult]
    market_summary: dict[str, float]     # per-stream totals across all days
    notes: list[str] = field(default_factory=list)


def _month_days(year: int, month: int) -> list[date]:
    start = date(year, month, 1)
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    return [start + timedelta(days=i) for i in range((nxt - start).days)]


def run_anchor_month(
    policies: dict[str, ShadowCostPolicy],
    year: int,
    month: int,
    initial_soh: float = 1.0,
    power_mw: float = 1.0,
    duration_h: float = 2.0,
    rte: float = 0.85,
    max_cycles: float = 2.0,
    max_afrr_participation: float = 0.40,
    afrr_reserve_duration_hours: float = 0.25,
    pool_pos_mw: float = DEFAULT_POOL_POS_MW,
    pool_neg_mw: float = DEFAULT_POOL_NEG_MW,
) -> AnchorMonthRun:
    """Solve stacked LP per day over the target month for each policy.

    All policies see identical market inputs (same pre-fetched frames).
    Differentiation is entirely in the per-day wear cost vector each
    policy emits.

    Returns an :class:`AnchorMonthRun` with per-policy revenue,
    diagnostics, and full daily logs.
    """
    frames = _prefetch_year_frames(year)
    days = _month_days(year, month)
    energy_mwh = power_mw * duration_h * initial_soh

    results: dict[str, AnchorMonthPolicyResult] = {}
    market_totals = {"da": 0.0, "id": 0.0}
    first_policy = True

    for name, policy in policies.items():
        logger.info(f"anchor_month({year}-{month:02d}): running policy={name}")
        day_logs: list[DayDiagnosticData] = []
        rev = 0.0
        fec_total = 0.0
        r_pos_sum = 0.0
        r_neg_sum = 0.0
        intervals = 0

        for d_idx, d in enumerate(days):
            inputs = assemble_day_inputs(
                target_date=d,
                pool_pos_mw=pool_pos_mw, pool_neg_mw=pool_neg_mw,
                da_frame=frames.get("da"), id_frame=frames.get("id"),
                spot_frame=frames.get("spot"),
                afrr_cap_frame=frames.get("afrr_cap"),
                afrr_energy_frame=frames.get("afrr_energy"),
                activations_frame=frames.get("activations"),
            )
            if inputs is None:
                continue
            wear = policy.wear_cost(
                soh_current=initial_soh, day_of_year=d.timetuple().tm_yday,
                periods_per_day=96, duration_h=duration_h,
            )
            overrides = policy.lp_overrides(
                soh_current=initial_soh, day_of_year=d.timetuple().tm_yday,
            )
            lp_kwargs = dict(
                energy_mwh=energy_mwh, power_mw=power_mw, rte=rte,
                max_cycles=max_cycles,
                afrr_reserve_duration_hours=afrr_reserve_duration_hours,
                max_afrr_participation=max_afrr_participation,
                wear_cost_eur_per_mwh=wear,
            )
            lp_kwargs.update(overrides)
            day_out = optimize_day_stacked(**inputs.as_kwargs(), **lp_kwargs)
            if not day_out.success:
                continue
            rev += day_out.revenue_total
            fec_total += day_out.full_equivalent_cycles
            r_pos_sum += day_out.r_pos.sum()
            r_neg_sum += day_out.r_neg.sum()
            intervals += len(day_out.r_pos)
            day_logs.append(day_diagnostic_from_stacked(
                day_result=day_out, inputs=inputs, target_date=d,
                power_mw=power_mw, energy_mwh=energy_mwh,
            ))
            # Record market totals once (same for every policy)
            if first_policy:
                market_totals["da"] += float(
                    (day_out.discharge_da.sum() - day_out.charge_da.sum())
                    * 0.25 * inputs.prices_da.mean()
                )

        first_policy = False
        n = len(day_logs)
        rev_ann = rev / 1000.0 * 12.0 if n else 0.0
        results[name] = AnchorMonthPolicyResult(
            policy_name=name,
            days_solved=n,
            monthly_revenue_eur=rev,
            annualised_revenue_keur_per_mw=rev_ann,
            total_fec=fec_total,
            mean_r_pos_mw=(r_pos_sum / intervals) if intervals else 0.0,
            mean_r_neg_mw=(r_neg_sum / intervals) if intervals else 0.0,
            daily_logs=day_logs,
            diagnostics=compute_all_signals(policy_name=name, days=day_logs) if day_logs else None,
        )

    return AnchorMonthRun(
        year=year, month=month, results=results,
        market_summary=market_totals,
        notes=[
            "Starting SoH = {:.2f} (fresh cell at month start)".format(initial_soh),
            f"Duration {duration_h}h, max_cycles {max_cycles}, max_afrr_participation {max_afrr_participation}",
        ],
    )
