"""
Interval-level revenue breakdown — per-slot charge/discharge attribution.

Builds detailed revenue tables at 15-min or hourly resolution,
with optional intraday price overlay for DA+ID stacking analysis.
"""
from __future__ import annotations

from math import sqrt
from typing import Sequence

import pandas as pd

from lib.data.day_ahead_prices import compute_daily_price_metrics
from lib.models.dispatch_detailed import DispatchStrategy, infer_timestep_hours, optimize_day


def _format_interval_fields(frame: pd.DataFrame, interval_minutes: float) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["date"] = enriched.index.tz_localize(None).normalize()
    enriched["hour"] = enriched.index.hour
    enriched["quarter_hour"] = enriched.index.strftime("%H:%M")
    enriched["interval_minutes"] = float(interval_minutes)
    return enriched


def build_interval_revenue_table(
    day_ahead_price_frame: pd.DataFrame,
    strategy: DispatchStrategy,
    energy_mwh: float,
    rte: float,
    intraday_price_frame: pd.DataFrame | None = None,
    power_mw: float = 1.0,
) -> pd.DataFrame:
    if intraday_price_frame is not None and not intraday_price_frame.empty:
        return _build_intraday_overlay_interval_revenue_table(
            day_ahead_price_frame=day_ahead_price_frame,
            intraday_price_frame=intraday_price_frame,
            strategy=strategy,
            energy_mwh=energy_mwh,
            rte=rte,
            power_mw=power_mw,
        )
    return _build_single_market_interval_revenue_table(
        price_frame=day_ahead_price_frame,
        strategy=strategy,
        energy_mwh=energy_mwh,
        rte=rte,
        power_mw=power_mw,
    )


def _build_single_market_interval_revenue_table(
    price_frame: pd.DataFrame,
    strategy: DispatchStrategy,
    energy_mwh: float,
    rte: float,
    power_mw: float,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    eta = sqrt(rte)
    throughput_penalty = strategy.min_spread_eur_mwh / 2
    prices = price_frame.sort_index()
    for _, group in prices.groupby(prices.index.normalize()):
        dt_hours = infer_timestep_hours(group.index)
        price_values = group["price_eur_mwh"].to_numpy(dtype=float)
        result = optimize_day(
            prices=price_values,
            energy_mwh=energy_mwh,
            rte=rte,
            soc_min_frac=strategy.soc_min_frac,
            soc_max_frac=strategy.soc_max_frac,
            max_cycles=strategy.max_cycles,
            power_mw=power_mw,
            dt_hours=dt_hours,
            min_spread_eur_mwh=strategy.min_spread_eur_mwh,
        )
        interval_frame = pd.DataFrame(
            {
                "price_eur_mwh": price_values,
                "charge_mw": result["charge"],
                "discharge_mw": result["discharge"],
            },
            index=group.index,
        )
        interval_frame["revenue_eur_per_mw"] = (
            interval_frame["discharge_mw"] * dt_hours * (interval_frame["price_eur_mwh"] * eta - throughput_penalty)
            - interval_frame["charge_mw"] * dt_hours * (interval_frame["price_eur_mwh"] / eta + throughput_penalty)
        )
        interval_frame["settlement_adjustment_eur_per_mw"] = 0.0
        interval_frame["execution_revenue_eur_per_mw"] = interval_frame["revenue_eur_per_mw"]
        records.append(_format_interval_fields(interval_frame, interval_minutes=dt_hours * 60))

    if not records:
        return pd.DataFrame()
    return pd.concat(records).sort_index()


def _build_intraday_overlay_interval_revenue_table(
    day_ahead_price_frame: pd.DataFrame,
    intraday_price_frame: pd.DataFrame,
    strategy: DispatchStrategy,
    energy_mwh: float,
    rte: float,
    power_mw: float,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    day_ahead_prices = day_ahead_price_frame.sort_index()
    intraday_prices = intraday_price_frame.sort_index()
    common_days = sorted(
        set(day_ahead_prices.index.normalize()).intersection(set(intraday_prices.index.normalize()))
    )
    eta = sqrt(rte)
    throughput_penalty = strategy.min_spread_eur_mwh / 2

    for day in common_days:
        da_group = day_ahead_prices[day_ahead_prices.index.normalize() == day]
        id_group = intraday_prices[intraday_prices.index.normalize() == day]
        if da_group.empty or id_group.empty:
            continue

        da_dt_hours = infer_timestep_hours(da_group.index)
        id_dt_hours = infer_timestep_hours(id_group.index)
        da_prices = da_group["price_eur_mwh"].to_numpy(dtype=float)
        id_prices = id_group["price_eur_mwh"].to_numpy(dtype=float)

        da_result = optimize_day(
            prices=da_prices,
            energy_mwh=energy_mwh,
            rte=rte,
            soc_min_frac=strategy.soc_min_frac,
            soc_max_frac=strategy.soc_max_frac,
            max_cycles=strategy.max_cycles,
            power_mw=power_mw,
            dt_hours=da_dt_hours,
            min_spread_eur_mwh=strategy.min_spread_eur_mwh,
        )
        id_result = optimize_day(
            prices=id_prices,
            energy_mwh=energy_mwh,
            rte=rte,
            soc_min_frac=strategy.soc_min_frac,
            soc_max_frac=strategy.soc_max_frac,
            max_cycles=strategy.max_cycles,
            power_mw=power_mw,
            dt_hours=id_dt_hours,
            min_spread_eur_mwh=strategy.min_spread_eur_mwh,
        )

        da_charge_on_id_grid = pd.Series(da_result["charge"], index=da_group.index).reindex(id_group.index, method="ffill")
        da_discharge_on_id_grid = pd.Series(da_result["discharge"], index=da_group.index).reindex(id_group.index, method="ffill")
        da_price_on_id_grid = pd.Series(da_prices, index=da_group.index).reindex(id_group.index, method="ffill")
        da_grid_flow = da_discharge_on_id_grid * eta - da_charge_on_id_grid / eta

        interval_frame = pd.DataFrame(
            {
                "price_eur_mwh": id_prices,
                "day_ahead_price_eur_mwh": da_price_on_id_grid.to_numpy(dtype=float),
                "charge_mw": id_result["charge"],
                "discharge_mw": id_result["discharge"],
                "scheduled_grid_flow_mw": da_grid_flow.to_numpy(dtype=float),
            },
            index=id_group.index,
        )
        interval_frame["execution_revenue_eur_per_mw"] = (
            interval_frame["discharge_mw"] * id_dt_hours * (interval_frame["price_eur_mwh"] * eta - throughput_penalty)
            - interval_frame["charge_mw"] * id_dt_hours * (interval_frame["price_eur_mwh"] / eta + throughput_penalty)
        )
        interval_frame["settlement_adjustment_eur_per_mw"] = (
            interval_frame["scheduled_grid_flow_mw"]
            * (interval_frame["day_ahead_price_eur_mwh"] - interval_frame["price_eur_mwh"])
            * id_dt_hours
        )
        interval_frame["revenue_eur_per_mw"] = (
            interval_frame["execution_revenue_eur_per_mw"] + interval_frame["settlement_adjustment_eur_per_mw"]
        )
        records.append(_format_interval_fields(interval_frame, interval_minutes=id_dt_hours * 60))

    if not records:
        return pd.DataFrame()
    return pd.concat(records).sort_index()


def summarize_top_hours(interval_revenue: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if interval_revenue.empty:
        return pd.DataFrame()
    hourly = (
        interval_revenue.resample("1h")
        .agg(
            revenue_eur_per_mw=("revenue_eur_per_mw", "sum"),
            avg_price_eur_mwh=("price_eur_mwh", "mean"),
            charge_mw=("charge_mw", "mean"),
            discharge_mw=("discharge_mw", "mean"),
        )
        .sort_values("revenue_eur_per_mw", ascending=False)
        .head(top_n)
    )
    return hourly


def summarize_top_quarter_hours(interval_revenue: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if interval_revenue.empty:
        return pd.DataFrame()
    quarter_hours = interval_revenue[interval_revenue["interval_minutes"] <= 15.0 + 1e-9]
    if quarter_hours.empty:
        return pd.DataFrame()
    return quarter_hours.sort_values("revenue_eur_per_mw", ascending=False).head(top_n)[
        [
            "price_eur_mwh",
            "charge_mw",
            "discharge_mw",
            "execution_revenue_eur_per_mw",
            "settlement_adjustment_eur_per_mw",
            "revenue_eur_per_mw",
        ]
    ]


def summarize_top_spreads(
    dispatch_frame: pd.DataFrame,
    price_frame: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    if dispatch_frame.empty or price_frame.empty:
        return pd.DataFrame()
    price_metrics = compute_daily_price_metrics(price_frame)
    merged = dispatch_frame.join(price_metrics, how="left")
    return merged.sort_values(["spread_eur_mwh", "revenue_eur_per_mw"], ascending=False).head(top_n)[
        [
            "revenue_eur_per_mw",
            "spread_eur_mwh",
            "tb2_spread_eur_mwh",
            "min_price_eur_mwh",
            "max_price_eur_mwh",
            "negative_intervals",
            "midday_min_price_eur_mwh",
            "evening_peak_price_eur_mwh",
        ]
    ]


def summarize_missing_top_days(
    revenue_series: pd.Series,
    top_day_counts: Sequence[int] = (1, 3, 5, 10, 20, 30),
) -> pd.DataFrame:
    ordered = revenue_series.sort_values(ascending=False)
    if ordered.empty:
        return pd.DataFrame(
            columns=[
                "missed_top_days",
                "lost_revenue_eur_per_mw",
                "lost_share_pct",
                "remaining_revenue_eur_per_mw",
                "remaining_share_pct",
            ]
        )
    total_revenue = float(ordered.sum())
    rows = []
    for count in sorted({int(value) for value in top_day_counts if int(value) > 0}):
        clipped = min(count, len(ordered))
        lost = float(ordered.head(clipped).sum())
        remaining = total_revenue - lost
        rows.append(
            {
                "missed_top_days": clipped,
                "lost_revenue_eur_per_mw": lost,
                "lost_share_pct": 100 * lost / total_revenue if total_revenue else 0.0,
                "remaining_revenue_eur_per_mw": remaining,
                "remaining_share_pct": 100 * remaining / total_revenue if total_revenue else 0.0,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["missed_top_days"]).set_index("missed_top_days")
