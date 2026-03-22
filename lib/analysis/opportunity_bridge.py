"""
Inter-annual stability and opportunity-bridge analysis.

Summarises year-over-year dispatch revenue, concentration metrics,
and builds bridge charts showing how revenue shifts between periods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from lib.analysis.concentration import compute_concentration_stats, days_to_revenue_share
from lib.data.day_ahead_prices import compute_daily_price_metrics


def summarize_interannual_stability(dispatch_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for year in sorted(dispatch_by_year):
        dispatch = dispatch_by_year[year]
        revenue = dispatch["revenue_eur_per_mw"]
        concentration = compute_concentration_stats(revenue)
        rows.append(
            {
                "year": year,
                "annual_revenue_eur_per_mw": float(revenue.sum()),
                "top_20pct_days_pct_of_revenue": 100 * concentration["top_20_days_pct_of_revenue"],
                "days_to_50pct_revenue": days_to_revenue_share(revenue, 0.50),
                "days_to_80pct_revenue": days_to_revenue_share(revenue, 0.80),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def summarize_within_day_concentration(
    interval_revenue: pd.DataFrame,
    top_windows: Sequence[int] = (2, 4, 6),
    top_day_dates: Iterable[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if interval_revenue.empty:
        return pd.DataFrame(), pd.DataFrame()

    daily_rows = []
    for date, group in interval_revenue.groupby("date"):
        total_revenue = float(group["revenue_eur_per_mw"].sum())
        if total_revenue <= 0:
            continue
        ranked = group["revenue_eur_per_mw"].sort_values(ascending=False)
        row: dict[str, float | pd.Timestamp] = {"date": date, "daily_revenue_eur_per_mw": total_revenue}
        for count in top_windows:
            row[f"top_{count}_windows_share_pct"] = 100 * float(ranked.head(count).sum()) / total_revenue
        daily_rows.append(row)

    if not daily_rows:
        return pd.DataFrame(), pd.DataFrame()
    daily = pd.DataFrame(daily_rows).set_index("date").sort_index()
    if daily.empty:
        return daily, pd.DataFrame()

    summary_rows = []
    segments: list[tuple[str, pd.DataFrame]] = [("all_positive_days", daily)]
    if top_day_dates is not None:
        top_index = pd.Index(pd.to_datetime(list(top_day_dates)))
        segments.append(("top_days", daily[daily.index.isin(top_index)]))

    for label, segment in segments:
        if segment.empty:
            continue
        row: dict[str, float | str] = {"segment": label, "days": int(len(segment))}
        for count in top_windows:
            column = f"top_{count}_windows_share_pct"
            row[f"median_{column}"] = float(segment[column].median())
            row[f"p90_{column}"] = float(segment[column].quantile(0.90))
        summary_rows.append(row)

    return daily, pd.DataFrame(summary_rows).set_index("segment")


def summarize_opportunity_day_signals(
    dispatch_frame: pd.DataFrame,
    price_frame: pd.DataFrame,
    top_n: int = 20,
    spread_thresholds: Sequence[float] = (200.0, 500.0, 1000.0),
    evening_peak_thresholds: Sequence[float] = (200.0, 500.0, 1000.0),
    evening_squeeze_threshold: float = 200.0,
    low_midday_thresholds: Sequence[float] = (0.0, 25.0),
) -> pd.DataFrame:
    if dispatch_frame.empty or price_frame.empty:
        return pd.DataFrame()

    price_daily = compute_daily_price_metrics(price_frame)
    joined = dispatch_frame.join(price_daily, how="left")
    top_days = joined.sort_values("revenue_eur_per_mw", ascending=False).head(top_n)
    if top_days.empty:
        return pd.DataFrame()

    rows = []
    for threshold in spread_thresholds:
        qualifier_top = top_days["spread_eur_mwh"] >= threshold
        qualifier_all = joined["spread_eur_mwh"] >= threshold
        rows.append(
            {
                "signal": f"Spread >= {threshold:.0f} €/MWh",
                "top_days_share_pct": 100 * float(qualifier_top.mean()),
                "all_days_share_pct": 100 * float(qualifier_all.mean()),
            }
        )
    for threshold in evening_peak_thresholds:
        qualifier_top = top_days["evening_peak_price_eur_mwh"] >= threshold
        qualifier_all = joined["evening_peak_price_eur_mwh"] >= threshold
        rows.append(
            {
                "signal": f"Evening peak >= {threshold:.0f} €/MWh",
                "top_days_share_pct": 100 * float(qualifier_top.mean()),
                "all_days_share_pct": 100 * float(qualifier_all.mean()),
            }
        )
    for threshold in low_midday_thresholds:
        qualifier_top = (top_days["midday_min_price_eur_mwh"] <= threshold) & (
            top_days["evening_peak_price_eur_mwh"] >= evening_squeeze_threshold
        )
        qualifier_all = (joined["midday_min_price_eur_mwh"] <= threshold) & (
            joined["evening_peak_price_eur_mwh"] >= evening_squeeze_threshold
        )
        label = "Negative midday" if threshold <= 0 else f"Midday <= {threshold:.0f} €/MWh"
        rows.append(
            {
                "signal": f"{label} + evening spike >= {evening_squeeze_threshold:.0f} €/MWh",
                "top_days_share_pct": 100 * float(qualifier_top.mean()),
                "all_days_share_pct": 100 * float(qualifier_all.mean()),
            }
        )

    return pd.DataFrame(rows).set_index("signal")


def build_daily_value_curve(dispatch_by_cycle_cap: dict[float, pd.DataFrame]) -> pd.DataFrame:
    if not dispatch_by_cycle_cap:
        return pd.DataFrame()

    all_dates = sorted(
        {
            pd.Timestamp(date).tz_localize(None)
            for frame in dispatch_by_cycle_cap.values()
            for date in frame.index
        }
    )
    raw_rows = []
    for date in all_dates:
        raw_rows.append(
            {
                "date": date,
                "cycle_cap": 0.0,
                "full_equivalent_cycles": 0.0,
                "revenue_eur_per_mw": 0.0,
            }
        )
    for cycle_cap, dispatch in sorted(dispatch_by_cycle_cap.items()):
        for date, row in dispatch.iterrows():
            raw_rows.append(
                {
                    "date": pd.Timestamp(date).tz_localize(None),
                    "cycle_cap": float(cycle_cap),
                    "full_equivalent_cycles": max(float(row.get("full_equivalent_cycles", 0.0)), 0.0),
                    "revenue_eur_per_mw": max(float(row.get("revenue_eur_per_mw", 0.0)), 0.0),
                }
            )

    raw = pd.DataFrame(raw_rows)
    normalized_rows = []
    for date, group in raw.groupby("date"):
        points = group.sort_values(["full_equivalent_cycles", "revenue_eur_per_mw", "cycle_cap"])
        cleaned: list[dict[str, float | pd.Timestamp]] = []
        for row in points.itertuples(index=False):
            fec = float(row.full_equivalent_cycles)
            revenue = float(row.revenue_eur_per_mw)
            if cleaned and abs(fec - float(cleaned[-1]["full_equivalent_cycles"])) <= 1e-9:
                if revenue >= float(cleaned[-1]["revenue_eur_per_mw"]):
                    cleaned[-1] = {
                        "date": date,
                        "cycle_cap": float(row.cycle_cap),
                        "full_equivalent_cycles": fec,
                        "revenue_eur_per_mw": revenue,
                    }
                continue
            if cleaned and revenue <= float(cleaned[-1]["revenue_eur_per_mw"]) + 1e-9:
                continue
            cleaned.append(
                {
                    "date": date,
                    "cycle_cap": float(row.cycle_cap),
                    "full_equivalent_cycles": fec,
                    "revenue_eur_per_mw": revenue,
                }
            )
        normalized_rows.extend(cleaned)

    return pd.DataFrame(normalized_rows).sort_values(["date", "full_equivalent_cycles"]).reset_index(drop=True)


def build_throughput_segments(daily_value_curve: pd.DataFrame) -> pd.DataFrame:
    if daily_value_curve.empty:
        return pd.DataFrame()

    rows = []
    for date, group in daily_value_curve.groupby("date"):
        ordered = group.sort_values("full_equivalent_cycles").reset_index(drop=True)
        for position in range(1, len(ordered)):
            prior = ordered.iloc[position - 1]
            current = ordered.iloc[position]
            delta_fec = float(current["full_equivalent_cycles"] - prior["full_equivalent_cycles"])
            delta_revenue = float(current["revenue_eur_per_mw"] - prior["revenue_eur_per_mw"])
            if delta_fec <= 1e-9 or delta_revenue <= 1e-9:
                continue
            rows.append(
                {
                    "date": date,
                    "segment_order": len(rows),
                    "from_cycle_cap": float(prior["cycle_cap"]),
                    "to_cycle_cap": float(current["cycle_cap"]),
                    "from_fec": float(prior["full_equivalent_cycles"]),
                    "to_fec": float(current["full_equivalent_cycles"]),
                    "from_revenue_eur_per_mw": float(prior["revenue_eur_per_mw"]),
                    "to_revenue_eur_per_mw": float(current["revenue_eur_per_mw"]),
                    "delta_fec": delta_fec,
                    "delta_revenue_eur_per_mw": delta_revenue,
                    "marginal_revenue_eur_per_fec": delta_revenue / delta_fec,
                }
            )
    if not rows:
        return pd.DataFrame()

    segments = pd.DataFrame(rows).sort_values(["date", "to_fec", "to_revenue_eur_per_mw"]).reset_index(drop=True)
    segments["segment_order"] = segments.groupby("date").cumcount()
    return segments


@dataclass(frozen=True)
class ThroughputAllocationResult:
    annual_fec_budget: float
    total_captured_revenue_eur_per_mw: float
    total_allocated_fec: float
    daily: pd.DataFrame
    segments: pd.DataFrame


def allocate_annual_throughput_budget(
    daily_value_curve: pd.DataFrame,
    annual_fec_budget: float,
) -> ThroughputAllocationResult:
    if daily_value_curve.empty:
        empty = pd.DataFrame(columns=["allocated_fec", "captured_revenue_eur_per_mw"])
        return ThroughputAllocationResult(
            annual_fec_budget=float(annual_fec_budget),
            total_captured_revenue_eur_per_mw=0.0,
            total_allocated_fec=0.0,
            daily=empty,
            segments=pd.DataFrame(),
        )

    segments = build_throughput_segments(daily_value_curve)
    if segments.empty:
        empty = pd.DataFrame(index=daily_value_curve["date"].drop_duplicates(), columns=["allocated_fec", "captured_revenue_eur_per_mw"]).fillna(0.0)
        return ThroughputAllocationResult(
            annual_fec_budget=float(annual_fec_budget),
            total_captured_revenue_eur_per_mw=0.0,
            total_allocated_fec=0.0,
            daily=empty,
            segments=segments,
        )

    variable_count = len(segments)
    c = -segments["delta_revenue_eur_per_mw"].to_numpy(dtype=float)
    row_idx = []
    col_idx = []
    values = []
    bounds = [(0.0, 1.0)] * variable_count
    b_ub = [float(annual_fec_budget)]

    for idx, delta_fec in enumerate(segments["delta_fec"].to_numpy(dtype=float)):
        row_idx.append(0)
        col_idx.append(idx)
        values.append(delta_fec)

    constraint_row = 1
    for _, group in segments.groupby("date").groups.items():
        group_positions = list(group)
        for prior, current in zip(group_positions, group_positions[1:]):
            row_idx.extend([constraint_row, constraint_row])
            col_idx.extend([current, prior])
            values.extend([1.0, -1.0])
            b_ub.append(0.0)
            constraint_row += 1

    a_ub = sparse.csr_matrix((values, (row_idx, col_idx)), shape=(constraint_row, variable_count))
    result = linprog(
        c=c,
        A_ub=a_ub,
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Annual throughput allocation failed: {result.message}")

    solved_segments = segments.copy()
    solved_segments["take_fraction"] = result.x
    solved_segments["allocated_fec"] = solved_segments["delta_fec"] * solved_segments["take_fraction"]
    solved_segments["captured_revenue_eur_per_mw"] = (
        solved_segments["delta_revenue_eur_per_mw"] * solved_segments["take_fraction"]
    )
    daily = (
        solved_segments.groupby("date")
        .agg(
            allocated_fec=("allocated_fec", "sum"),
            captured_revenue_eur_per_mw=("captured_revenue_eur_per_mw", "sum"),
        )
        .sort_index()
    )
    all_dates = pd.Index(pd.to_datetime(daily_value_curve["date"].drop_duplicates().sort_values()))
    daily = daily.reindex(all_dates, fill_value=0.0)

    return ThroughputAllocationResult(
        annual_fec_budget=float(annual_fec_budget),
        total_captured_revenue_eur_per_mw=float(daily["captured_revenue_eur_per_mw"].sum()),
        total_allocated_fec=float(daily["allocated_fec"].sum()),
        daily=daily,
        segments=solved_segments,
    )


def full_flex_daily_value(daily_value_curve: pd.DataFrame) -> pd.DataFrame:
    if daily_value_curve.empty:
        return pd.DataFrame(columns=["full_equivalent_cycles", "revenue_eur_per_mw"])
    return (
        daily_value_curve.sort_values(["date", "full_equivalent_cycles", "revenue_eur_per_mw"])
        .groupby("date")
        .tail(1)
        .set_index("date")[
            ["full_equivalent_cycles", "revenue_eur_per_mw"]
        ]
        .sort_index()
    )


def summarize_throughput_budget_scenarios(
    daily_value_curve: pd.DataFrame,
    budgets: Sequence[float],
    top_opportunity_day_count: int = 20,
) -> tuple[pd.DataFrame, dict[str, ThroughputAllocationResult]]:
    full_flex = full_flex_daily_value(daily_value_curve)
    if full_flex.empty:
        return pd.DataFrame(), {}

    full_flex_budget = float(full_flex["full_equivalent_cycles"].sum())
    full_flex_revenue = float(full_flex["revenue_eur_per_mw"].sum())
    top_opportunity_dates = full_flex["revenue_eur_per_mw"].sort_values(ascending=False).head(top_opportunity_day_count).index
    top_opportunity_revenue = float(full_flex.loc[top_opportunity_dates, "revenue_eur_per_mw"].sum())

    allocations: dict[str, ThroughputAllocationResult] = {}
    rows = []
    for budget in budgets:
        allocation = allocate_annual_throughput_budget(daily_value_curve, annual_fec_budget=float(budget))
        allocations[f"{float(budget):.0f}"] = allocation
        captured_top_opportunity = float(
            allocation.daily.reindex(top_opportunity_dates, fill_value=0.0)["captured_revenue_eur_per_mw"].sum()
        )
        rows.append(
            {
                "scenario": f"{float(budget):.0f} FEC/year",
                "fec_budget": float(budget),
                "allocated_fec": allocation.total_allocated_fec,
                "captured_revenue_eur_per_mw": allocation.total_captured_revenue_eur_per_mw,
                "share_of_full_flex_revenue_pct": 100 * allocation.total_captured_revenue_eur_per_mw / full_flex_revenue
                if full_flex_revenue
                else 0.0,
                "share_of_top_opportunity_day_revenue_pct": 100 * captured_top_opportunity / top_opportunity_revenue
                if top_opportunity_revenue
                else 0.0,
            }
        )

    allocations["full_flex"] = ThroughputAllocationResult(
        annual_fec_budget=full_flex_budget,
        total_captured_revenue_eur_per_mw=full_flex_revenue,
        total_allocated_fec=full_flex_budget,
        daily=full_flex.rename(columns={"full_equivalent_cycles": "allocated_fec", "revenue_eur_per_mw": "captured_revenue_eur_per_mw"}),
        segments=pd.DataFrame(),
    )
    rows.append(
        {
            "scenario": "Full flex",
            "fec_budget": full_flex_budget,
            "allocated_fec": full_flex_budget,
            "captured_revenue_eur_per_mw": full_flex_revenue,
            "share_of_full_flex_revenue_pct": 100.0,
            "share_of_top_opportunity_day_revenue_pct": 100.0,
        }
    )

    summary = pd.DataFrame(rows).set_index("scenario")
    return summary, allocations


def summarize_value_outside_warranty_pace(
    daily_value_curve: pd.DataFrame,
    reference_warranty_fec_per_year: float,
    top_n_days: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_flex = full_flex_daily_value(daily_value_curve)
    if full_flex.empty:
        return pd.DataFrame(), pd.DataFrame()

    warranty = allocate_annual_throughput_budget(daily_value_curve, annual_fec_budget=reference_warranty_fec_per_year)
    full_revenue = float(full_flex["revenue_eur_per_mw"].sum())
    inside_warranty = float(warranty.daily["captured_revenue_eur_per_mw"].sum())
    extra_revenue = full_revenue - inside_warranty
    full_fec = float(full_flex["full_equivalent_cycles"].sum())

    summary = pd.DataFrame(
        [
            {
                "reference_warranty_fec_per_year": float(reference_warranty_fec_per_year),
                "full_flex_fec_per_year": full_fec,
                "revenue_inside_warranty_eur_per_mw": inside_warranty,
                "share_inside_warranty_pct": 100 * inside_warranty / full_revenue if full_revenue else 0.0,
                "extra_revenue_above_warranty_eur_per_mw": extra_revenue,
                "share_above_warranty_pct": 100 * extra_revenue / full_revenue if full_revenue else 0.0,
            }
        ]
    )

    daily = (
        full_flex.rename(
            columns={
                "full_equivalent_cycles": "full_flex_fec",
                "revenue_eur_per_mw": "full_flex_revenue_eur_per_mw",
            }
        )
        .join(
            warranty.daily.rename(
                columns={
                    "allocated_fec": "warranty_fec",
                    "captured_revenue_eur_per_mw": "warranty_revenue_eur_per_mw",
                }
            ),
            how="left",
        )
        .fillna(0.0)
    )
    daily["incremental_fec_above_warranty"] = daily["full_flex_fec"] - daily["warranty_fec"]
    daily["incremental_revenue_above_warranty_eur_per_mw"] = (
        daily["full_flex_revenue_eur_per_mw"] - daily["warranty_revenue_eur_per_mw"]
    )
    daily = daily.sort_values("incremental_revenue_above_warranty_eur_per_mw", ascending=False).head(top_n_days)
    return summary, daily


def summarize_annual_budget_vs_strict_daily_cap(
    dispatch_by_cycle_cap: dict[float, pd.DataFrame],
    daily_value_curve: pd.DataFrame,
    daily_caps: Sequence[float],
    top_day_counts: Sequence[int] = (10, 20, 50),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not dispatch_by_cycle_cap or daily_value_curve.empty:
        return pd.DataFrame(), pd.DataFrame()

    available_caps = np.asarray(sorted(float(cap) for cap in dispatch_by_cycle_cap), dtype=float)
    if available_caps.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    full_flex = full_flex_daily_value(daily_value_curve)
    if full_flex.empty:
        return pd.DataFrame(), pd.DataFrame()

    full_flex_revenue = float(full_flex["revenue_eur_per_mw"].sum())
    top_dates_by_count = {
        int(count): full_flex["revenue_eur_per_mw"].sort_values(ascending=False).head(int(count)).index
        for count in sorted({int(value) for value in top_day_counts if int(value) > 0})
    }

    summary_rows = []
    uplift_rows = []
    for daily_cap in sorted({float(value) for value in daily_caps if float(value) > 0}):
        nearest_position = int(np.argmin(np.abs(available_caps - daily_cap)))
        matched_cap = float(available_caps[nearest_position])
        if abs(matched_cap - daily_cap) > 1e-9:
            raise ValueError(
                f"Missing dispatch curve for strict daily cap {daily_cap:.3f}. "
                f"Closest available cap is {matched_cap:.3f}."
            )

        strict_dispatch = dispatch_by_cycle_cap[matched_cap].copy()
        strict_daily = strict_dispatch[["full_equivalent_cycles", "revenue_eur_per_mw"]].copy()
        strict_daily.index = pd.Index(pd.to_datetime(strict_daily.index)).rename("date")
        strict_daily = strict_daily.sort_index()

        annual_budget = float(daily_cap * 365.0)
        annual_allocation = allocate_annual_throughput_budget(daily_value_curve, annual_fec_budget=annual_budget)
        annual_daily = annual_allocation.daily.copy()
        annual_daily.index = pd.Index(pd.to_datetime(annual_daily.index)).rename("date")
        annual_daily = annual_daily.sort_index()

        comparison = strict_daily.rename(
            columns={
                "full_equivalent_cycles": "strict_fec",
                "revenue_eur_per_mw": "strict_revenue_eur_per_mw",
            }
        ).join(
            annual_daily.rename(
                columns={
                    "allocated_fec": "annual_budget_fec",
                    "captured_revenue_eur_per_mw": "annual_budget_revenue_eur_per_mw",
                }
            ),
            how="outer",
        ).fillna(0.0)
        comparison["uplift_eur_per_mw"] = (
            comparison["annual_budget_revenue_eur_per_mw"] - comparison["strict_revenue_eur_per_mw"]
        )

        strict_revenue = float(comparison["strict_revenue_eur_per_mw"].sum())
        annual_revenue = float(comparison["annual_budget_revenue_eur_per_mw"].sum())
        uplift = annual_revenue - strict_revenue
        summary_row = {
            "pair": f"{daily_cap:.1f}/day vs {annual_budget:.1f}/year",
            "strict_daily_cap": daily_cap,
            "annual_budget_fec": annual_budget,
            "strict_daily_cap_revenue_eur_per_mw": strict_revenue,
            "strict_realized_fec": float(comparison["strict_fec"].sum()),
            "annual_budget_revenue_eur_per_mw": annual_revenue,
            "annual_budget_allocated_fec": float(comparison["annual_budget_fec"].sum()),
            "uplift_eur_per_mw": uplift,
            "uplift_pct_vs_strict": 100 * uplift / strict_revenue if strict_revenue else 0.0,
            "strict_share_of_full_flex_revenue_pct": 100 * strict_revenue / full_flex_revenue if full_flex_revenue else 0.0,
            "annual_budget_share_of_full_flex_revenue_pct": 100 * annual_revenue / full_flex_revenue
            if full_flex_revenue
            else 0.0,
            "days_above_strict_daily_cap": int((comparison["annual_budget_fec"] > daily_cap + 1e-9).sum()),
            "max_daily_fec_in_annual_budget": float(comparison["annual_budget_fec"].max()),
        }
        for count, top_dates in top_dates_by_count.items():
            uplift_on_top_days = float(comparison.reindex(top_dates, fill_value=0.0)["uplift_eur_per_mw"].sum())
            summary_row[f"uplift_share_top_{count}_opportunity_days_pct"] = (
                100 * uplift_on_top_days / uplift if uplift > 1e-9 else 0.0
            )
        summary_rows.append(summary_row)

        uplift_rows.append(
            {
                "pair": summary_row["pair"],
                "strict_daily_cap": daily_cap,
                "annual_budget_fec": annual_budget,
                "days_above_strict_daily_cap": summary_row["days_above_strict_daily_cap"],
                "max_daily_fec_in_annual_budget": summary_row["max_daily_fec_in_annual_budget"],
                "uplift_share_top_10_opportunity_days_pct": summary_row.get("uplift_share_top_10_opportunity_days_pct", 0.0),
                "uplift_share_top_20_opportunity_days_pct": summary_row.get("uplift_share_top_20_opportunity_days_pct", 0.0),
                "uplift_share_top_50_opportunity_days_pct": summary_row.get("uplift_share_top_50_opportunity_days_pct", 0.0),
            }
        )

    summary = pd.DataFrame(summary_rows).set_index("pair")
    diagnostics = pd.DataFrame(uplift_rows).set_index("pair")
    return summary, diagnostics


def summarize_reallocated_same_throughput_vs_strict_daily_cap(
    dispatch_by_cycle_cap: dict[float, pd.DataFrame],
    daily_value_curve: pd.DataFrame,
    daily_caps: Sequence[float],
    top_day_counts: Sequence[int] = (10, 20, 50),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not dispatch_by_cycle_cap or daily_value_curve.empty:
        return pd.DataFrame(), pd.DataFrame()

    available_caps = np.asarray(sorted(float(cap) for cap in dispatch_by_cycle_cap), dtype=float)
    if available_caps.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    full_flex = full_flex_daily_value(daily_value_curve)
    if full_flex.empty:
        return pd.DataFrame(), pd.DataFrame()

    full_flex_revenue = float(full_flex["revenue_eur_per_mw"].sum())
    top_dates_by_count = {
        int(count): full_flex["revenue_eur_per_mw"].sort_values(ascending=False).head(int(count)).index
        for count in sorted({int(value) for value in top_day_counts if int(value) > 0})
    }

    summary_rows = []
    diagnostics_rows = []
    for daily_cap in sorted({float(value) for value in daily_caps if float(value) > 0}):
        nearest_position = int(np.argmin(np.abs(available_caps - daily_cap)))
        matched_cap = float(available_caps[nearest_position])
        if abs(matched_cap - daily_cap) > 1e-9:
            raise ValueError(
                f"Missing dispatch curve for strict daily cap {daily_cap:.3f}. "
                f"Closest available cap is {matched_cap:.3f}."
            )

        strict_dispatch = dispatch_by_cycle_cap[matched_cap].copy()
        strict_daily = strict_dispatch[["full_equivalent_cycles", "revenue_eur_per_mw"]].copy()
        strict_daily.index = pd.Index(pd.to_datetime(strict_daily.index)).rename("date")
        strict_daily = strict_daily.sort_index()
        strict_realized_fec = float(strict_daily["full_equivalent_cycles"].sum())

        annual_allocation = allocate_annual_throughput_budget(daily_value_curve, annual_fec_budget=strict_realized_fec)
        annual_daily = annual_allocation.daily.copy()
        annual_daily.index = pd.Index(pd.to_datetime(annual_daily.index)).rename("date")
        annual_daily = annual_daily.sort_index()

        comparison = strict_daily.rename(
            columns={
                "full_equivalent_cycles": "strict_fec",
                "revenue_eur_per_mw": "strict_revenue_eur_per_mw",
            }
        ).join(
            annual_daily.rename(
                columns={
                    "allocated_fec": "reallocated_same_fec",
                    "captured_revenue_eur_per_mw": "reallocated_same_fec_revenue_eur_per_mw",
                }
            ),
            how="outer",
        ).fillna(0.0)
        comparison["uplift_eur_per_mw"] = (
            comparison["reallocated_same_fec_revenue_eur_per_mw"] - comparison["strict_revenue_eur_per_mw"]
        )

        strict_revenue = float(comparison["strict_revenue_eur_per_mw"].sum())
        reallocated_revenue = float(comparison["reallocated_same_fec_revenue_eur_per_mw"].sum())
        uplift = reallocated_revenue - strict_revenue
        summary_row = {
            "pair": f"{daily_cap:.1f}/day vs reallocated same FEC",
            "strict_daily_cap": daily_cap,
            "strict_realized_fec": strict_realized_fec,
            "strict_daily_cap_revenue_eur_per_mw": strict_revenue,
            "reallocated_same_fec_revenue_eur_per_mw": reallocated_revenue,
            "reallocated_same_fec_allocated_fec": float(comparison["reallocated_same_fec"].sum()),
            "uplift_eur_per_mw": uplift,
            "uplift_pct_vs_strict": 100 * uplift / strict_revenue if strict_revenue else 0.0,
            "strict_share_of_full_flex_revenue_pct": 100 * strict_revenue / full_flex_revenue if full_flex_revenue else 0.0,
            "reallocated_same_fec_share_of_full_flex_revenue_pct": 100 * reallocated_revenue / full_flex_revenue
            if full_flex_revenue
            else 0.0,
            "days_above_strict_daily_cap": int((comparison["reallocated_same_fec"] > daily_cap + 1e-9).sum()),
            "max_daily_fec_in_reallocated_same_fec": float(comparison["reallocated_same_fec"].max()),
        }
        for count, top_dates in top_dates_by_count.items():
            uplift_on_top_days = float(comparison.reindex(top_dates, fill_value=0.0)["uplift_eur_per_mw"].sum())
            summary_row[f"uplift_share_top_{count}_opportunity_days_pct"] = (
                100 * uplift_on_top_days / uplift if uplift > 1e-9 else 0.0
            )
        summary_rows.append(summary_row)

        diagnostics_rows.append(
            {
                "pair": summary_row["pair"],
                "strict_daily_cap": daily_cap,
                "strict_realized_fec": strict_realized_fec,
                "days_above_strict_daily_cap": summary_row["days_above_strict_daily_cap"],
                "max_daily_fec_in_reallocated_same_fec": summary_row["max_daily_fec_in_reallocated_same_fec"],
                "uplift_share_top_10_opportunity_days_pct": summary_row.get("uplift_share_top_10_opportunity_days_pct", 0.0),
                "uplift_share_top_20_opportunity_days_pct": summary_row.get("uplift_share_top_20_opportunity_days_pct", 0.0),
                "uplift_share_top_50_opportunity_days_pct": summary_row.get("uplift_share_top_50_opportunity_days_pct", 0.0),
            }
        )

    summary = pd.DataFrame(summary_rows).set_index("pair")
    diagnostics = pd.DataFrame(diagnostics_rows).set_index("pair")
    return summary, diagnostics
