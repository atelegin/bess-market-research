"""
Revenue driver analysis — daily feature table for regression and exploration.

Joins dispatch results with price metrics and generation data,
adding season labels, weekday flags, and derived spread features.
"""
from __future__ import annotations

import pandas as pd

from lib.data.day_ahead_prices import compute_daily_price_metrics


def build_daily_driver_table(
    dispatch_frame: pd.DataFrame,
    price_frame: pd.DataFrame,
    generation_daily: pd.DataFrame,
) -> pd.DataFrame:
    price_daily = compute_daily_price_metrics(price_frame)
    merged = dispatch_frame.join(price_daily, how="left").join(generation_daily, how="left")
    merged["season"] = merged.index.month.map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }
    )
    return merged.sort_index()


def compute_correlation_summary(driver_table: pd.DataFrame) -> dict[str, float]:
    series_pairs = {
        "revenue_vs_residual_load_range": ("revenue_eur_per_mw", "residual_load_range_mw"),
        "revenue_vs_solar_generation": ("revenue_eur_per_mw", "solar_generation_gwh"),
        "revenue_vs_wind_generation": ("revenue_eur_per_mw", "wind_generation_gwh"),
    }
    summary = {}
    for label, (lhs, rhs) in series_pairs.items():
        subset = driver_table[[lhs, rhs]].dropna()
        summary[label] = float(subset.corr().iloc[0, 1]) if len(subset) > 1 else 0.0
    return summary


def compute_price_shape_profiles(price_frame: pd.DataFrame, top_dates: pd.Index) -> dict[str, pd.DataFrame]:
    hourly = price_frame["price_eur_mwh"].resample("1h").mean().to_frame("price_eur_mwh")
    hourly["date"] = hourly.index.tz_localize(None).normalize()
    hourly["hour"] = hourly.index.hour
    profile_matrix = (
        hourly.pivot_table(index="date", columns="hour", values="price_eur_mwh", aggfunc="mean")
        .reindex(columns=range(24))
        .sort_index()
    )
    top_index = pd.DatetimeIndex(pd.to_datetime(top_dates))
    if top_index.tz is not None:
        top_index = top_index.tz_localize(None)
    top_dates_normalized = top_index.normalize()
    all_profile = profile_matrix.median(axis=0).rename("all_days")
    top_profile = (
        profile_matrix.loc[profile_matrix.index.isin(top_dates_normalized)].median(axis=0).rename("top_days")
    )
    return {
        "median_profiles": pd.concat([all_profile, top_profile], axis=1).reset_index(),
    }


def tail_day_signal_summary(top_days: pd.DataFrame) -> dict[str, float]:
    if top_days.empty:
        return {"share_with_negative_midday": 0.0, "median_midday_floor": 0.0, "median_evening_peak": 0.0}
    qualifying = top_days["midday_min_price_eur_mwh"] < 0
    return {
        "share_with_negative_midday": float(qualifying.mean()),
        "median_midday_floor": float(top_days["midday_min_price_eur_mwh"].median()),
        "median_evening_peak": float(top_days["evening_peak_price_eur_mwh"].median()),
    }


def classify_tail_patterns(driver_table: pd.DataFrame, top_n: int = 20) -> dict[str, object]:
    if driver_table.empty:
        return {"dominant_pattern": "No data", "dominant_share": 0.0, "distribution": {}}
    top_days = driver_table.sort_values("revenue_eur_per_mw", ascending=False).head(top_n)
    solar_p75 = driver_table["solar_generation_gwh"].quantile(0.75)
    wind_p25 = driver_table["wind_generation_gwh"].quantile(0.25)
    peak_p90 = driver_table["evening_peak_price_eur_mwh"].quantile(0.90)
    spread_p85 = driver_table["spread_eur_mwh"].quantile(0.85)
    residual_p75 = driver_table["residual_load_range_mw"].quantile(0.75)

    labels = []
    for _, row in top_days.iterrows():
        if (
            row["midday_min_price_eur_mwh"] < 0
            and row["solar_generation_gwh"] >= solar_p75
            and row["evening_peak_price_eur_mwh"] >= peak_p90
        ):
            labels.append("Solar surplus + evening squeeze")
        elif row["wind_generation_gwh"] <= wind_p25 and row["evening_peak_price_eur_mwh"] >= peak_p90:
            labels.append("Wind drought + evening squeeze")
        elif row["spread_eur_mwh"] >= spread_p85 and row["evening_peak_price_eur_mwh"] >= peak_p90:
            labels.append("Explosive evening repricing")
        elif row["residual_load_range_mw"] >= residual_p75:
            labels.append("System stress day")
        else:
            labels.append("Wide spread day")

    distribution = pd.Series(labels).value_counts()
    dominant_pattern = distribution.index[0]
    dominant_share = float(distribution.iloc[0] / distribution.sum())
    return {
        "dominant_pattern": dominant_pattern,
        "dominant_share": dominant_share,
        "distribution": distribution.to_dict(),
    }
