"""
Day-ahead price signal detection — predefined rules for high-revenue days.

Defines signal predicates (e.g. evening ramp >= 200 €/MWh, negative midday)
and evaluates their precision/recall against actual dispatch revenue.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

from lib.data.day_ahead_prices import compute_daily_price_metrics


@dataclass(frozen=True)
class DayAheadSignalDefinition:
    name: str
    predicate: Callable[[pd.DataFrame], pd.Series]


def day_ahead_signal_groups() -> dict[str, list[str]]:
    return {
        "tight": [
            "DA evening-midday ramp >= 200 €/MWh",
            "DA spread >= 200 €/MWh",
            "DA evening peak >= 200 €/MWh",
            "DA midday <= 25 and evening >= 200",
        ],
        "broad": [
            "DA evening-midday ramp >= 150 €/MWh",
            "DA top 3 hours share >= 25%",
        ],
    }


def build_day_ahead_observable_table(
    day_ahead_price_frame: pd.DataFrame,
    outcome_dispatch: pd.DataFrame,
    top_day_counts: Sequence[int] = (10, 20),
) -> pd.DataFrame:
    price_daily = compute_daily_price_metrics(day_ahead_price_frame)
    hourly = day_ahead_price_frame["price_eur_mwh"].resample("1h").mean().to_frame("price_eur_mwh")
    hourly["date"] = hourly.index.tz_localize(None).normalize()

    hourly_rows = []
    for date, group in hourly.groupby("date"):
        prices = group["price_eur_mwh"].astype(float)
        hours = group.index.hour
        midday = prices[(hours >= 10) & (hours <= 15)]
        evening = prices[(hours >= 17) & (hours <= 21)]
        positive_price_mass = prices.clip(lower=0.0)
        positive_mass_sum = float(positive_price_mass.sum())
        top_3_share = (
            100 * float(positive_price_mass.nlargest(min(3, len(positive_price_mass))).sum()) / positive_mass_sum
            if positive_mass_sum > 0
            else 0.0
        )
        hourly_rows.append(
            {
                "date": date,
                "da_midday_mean_price_eur_mwh": float(midday.mean()) if not midday.empty else float("nan"),
                "da_evening_mean_price_eur_mwh": float(evening.mean()) if not evening.empty else float("nan"),
                "da_evening_minus_midday_mean_eur_mwh": float(evening.mean() - midday.mean())
                if not midday.empty and not evening.empty
                else float("nan"),
                "da_evening_minus_midday_ramp_eur_mwh": float(evening.max() - midday.min())
                if not midday.empty and not evening.empty
                else float("nan"),
                "da_hours_ge_200": int((prices >= 200).sum()),
                "da_hours_ge_500": int((prices >= 500).sum()),
                "da_top_3_hours_share_positive_price_pct": top_3_share,
            }
        )
    hourly_features = pd.DataFrame(hourly_rows).set_index("date").sort_index()

    feature_table = outcome_dispatch[["revenue_eur_per_mw"]].join(price_daily, how="left").join(hourly_features, how="left")
    feature_table["weekday"] = feature_table.index.day_name()
    feature_table["month"] = feature_table.index.month
    feature_table["season"] = feature_table.index.month.map(
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

    ordered = feature_table["revenue_eur_per_mw"].sort_values(ascending=False)
    for count in sorted({int(value) for value in top_day_counts if int(value) > 0}):
        feature_table[f"is_top_{count}_revenue_day"] = False
        if not ordered.empty:
            feature_table.loc[ordered.head(min(count, len(ordered))).index, f"is_top_{count}_revenue_day"] = True

    return feature_table.sort_index()


def concatenate_day_ahead_observable_tables(feature_tables_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for year, table in sorted(feature_tables_by_year.items()):
        frame = table.copy()
        frame["analysis_year"] = int(year)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def default_day_ahead_signal_definitions() -> list[DayAheadSignalDefinition]:
    return [
        DayAheadSignalDefinition(
            name="DA evening-midday ramp >= 200 €/MWh",
            predicate=lambda frame: frame["da_evening_minus_midday_ramp_eur_mwh"] >= 200.0,
        ),
        DayAheadSignalDefinition(
            name="DA spread >= 200 €/MWh",
            predicate=lambda frame: frame["spread_eur_mwh"] >= 200.0,
        ),
        DayAheadSignalDefinition(
            name="DA evening peak >= 200 €/MWh",
            predicate=lambda frame: frame["evening_peak_price_eur_mwh"] >= 200.0,
        ),
        DayAheadSignalDefinition(
            name="DA midday <= 25 and evening >= 200",
            predicate=lambda frame: (frame["midday_min_price_eur_mwh"] <= 25.0)
            & (frame["evening_peak_price_eur_mwh"] >= 200.0),
        ),
        DayAheadSignalDefinition(
            name="DA top 3 hours share >= 25%",
            predicate=lambda frame: frame["da_top_3_hours_share_positive_price_pct"] >= 25.0,
        ),
        DayAheadSignalDefinition(
            name="DA evening-midday ramp >= 150 €/MWh",
            predicate=lambda frame: frame["da_evening_minus_midday_ramp_eur_mwh"] >= 150.0,
        ),
    ]


def summarize_day_ahead_feature_separation(feature_table: pd.DataFrame) -> pd.DataFrame:
    if feature_table.empty:
        return pd.DataFrame()

    features = [
        "spread_eur_mwh",
        "evening_peak_price_eur_mwh",
        "midday_min_price_eur_mwh",
        "da_evening_minus_midday_ramp_eur_mwh",
        "da_top_3_hours_share_positive_price_pct",
        "da_hours_ge_200",
        "da_hours_ge_500",
    ]
    segments = {
        "all_days": feature_table,
        "top_20_revenue_days": feature_table[feature_table["is_top_20_revenue_day"]],
        "top_10_revenue_days": feature_table[feature_table["is_top_10_revenue_day"]],
    }
    rows = []
    for label, segment in segments.items():
        if segment.empty:
            continue
        row: dict[str, float | str] = {"segment": label, "days": int(len(segment))}
        for feature in features:
            row[f"median_{feature}"] = float(segment[feature].median())
        rows.append(row)
    return pd.DataFrame(rows).set_index("segment")


def evaluate_day_ahead_signals(
    feature_table: pd.DataFrame,
    signal_definitions: Sequence[DayAheadSignalDefinition] | None = None,
    top_day_counts: Sequence[int] = (10, 20),
) -> pd.DataFrame:
    if feature_table.empty:
        return pd.DataFrame()

    definitions = list(signal_definitions) if signal_definitions is not None else default_day_ahead_signal_definitions()
    annual_revenue = float(feature_table["revenue_eur_per_mw"].sum())
    avg_revenue_all = float(feature_table["revenue_eur_per_mw"].mean())
    total_days = len(feature_table)

    rows = []
    for definition in definitions:
        mask = definition.predicate(feature_table).fillna(False).astype(bool)
        signal_days = int(mask.sum())
        signal_slice = feature_table.loc[mask]
        non_signal_slice = feature_table.loc[~mask]
        avg_signal_revenue = float(signal_slice["revenue_eur_per_mw"].mean()) if not signal_slice.empty else 0.0
        avg_non_signal_revenue = float(non_signal_slice["revenue_eur_per_mw"].mean()) if not non_signal_slice.empty else 0.0

        row: dict[str, float | int | str] = {
            "signal": definition.name,
            "signal_days": signal_days,
            "signal_day_pct": 100 * signal_days / total_days if total_days else 0.0,
            "avg_revenue_signal_eur_per_mw": avg_signal_revenue,
            "avg_revenue_non_signal_eur_per_mw": avg_non_signal_revenue,
            "avg_revenue_all_eur_per_mw": avg_revenue_all,
            "revenue_on_signal_days_eur_per_mw": float(signal_slice["revenue_eur_per_mw"].sum()),
            "revenue_on_signal_days_pct": 100 * float(signal_slice["revenue_eur_per_mw"].sum()) / annual_revenue
            if annual_revenue
            else 0.0,
            "maintenance_day_value_shift_eur_per_mw": avg_signal_revenue - avg_non_signal_revenue,
        }

        for count in sorted({int(value) for value in top_day_counts if int(value) > 0}):
            target_col = f"is_top_{count}_revenue_day"
            positives = int(feature_table[target_col].sum())
            hits = int(feature_table.loc[mask, target_col].sum())
            precision_pct = 100 * hits / signal_days if signal_days else 0.0
            recall_pct = 100 * hits / positives if positives else 0.0
            base_rate_pct = 100 * positives / total_days if total_days else 0.0
            row[f"precision_top_{count}_pct"] = precision_pct
            row[f"recall_top_{count}_pct"] = recall_pct
            row[f"lift_vs_top_{count}_base_rate_x"] = precision_pct / base_rate_pct if base_rate_pct else 0.0

        rows.append(row)

    result = pd.DataFrame(rows).set_index("signal")
    sort_candidates = [f"precision_top_{count}_pct" for count in sorted({int(value) for value in top_day_counts if int(value) > 0}, reverse=True)]
    for column in sort_candidates:
        if column in result.columns:
            return result.sort_values(column, ascending=False)
    return result


def build_day_ahead_watchlist_table(
    feature_table: pd.DataFrame,
    target_count: int,
    signal_names: Sequence[str] | None = None,
    signal_definitions: Sequence[DayAheadSignalDefinition] | None = None,
) -> pd.DataFrame:
    evaluated = evaluate_day_ahead_signals(
        feature_table=feature_table,
        signal_definitions=signal_definitions,
        top_day_counts=(target_count,),
    )
    if evaluated.empty:
        return pd.DataFrame()
    if signal_names is not None:
        evaluated = evaluated.loc[evaluated.index.intersection(list(signal_names))]
    if evaluated.empty:
        return evaluated
    return evaluated[
        [
            "signal_days",
            "signal_day_pct",
            f"precision_top_{target_count}_pct",
            f"recall_top_{target_count}_pct",
            f"lift_vs_top_{target_count}_base_rate_x",
            "avg_revenue_signal_eur_per_mw",
            "avg_revenue_non_signal_eur_per_mw",
            "maintenance_day_value_shift_eur_per_mw",
        ]
    ].rename(
        columns={
            f"precision_top_{target_count}_pct": "precision_pct",
            f"recall_top_{target_count}_pct": "recall_pct",
            f"lift_vs_top_{target_count}_base_rate_x": "lift_x",
            "avg_revenue_signal_eur_per_mw": "avg_revenue_signal_eur_per_mw",
            "avg_revenue_non_signal_eur_per_mw": "avg_revenue_non_signal_eur_per_mw",
            "maintenance_day_value_shift_eur_per_mw": "maintenance_day_value_shift_eur_per_mw",
        }
    ).sort_values(["precision_pct", "recall_pct"], ascending=False)


def evaluate_day_ahead_signals_by_year(
    pooled_feature_table: pd.DataFrame,
    target_count: int,
    signal_definitions: Sequence[DayAheadSignalDefinition] | None = None,
) -> pd.DataFrame:
    if pooled_feature_table.empty or "analysis_year" not in pooled_feature_table:
        return pd.DataFrame()

    rows = []
    for year, group in pooled_feature_table.groupby("analysis_year"):
        evaluated = evaluate_day_ahead_signals(
            feature_table=group,
            signal_definitions=signal_definitions,
            top_day_counts=(target_count,),
        )
        for signal, row in evaluated.iterrows():
            rows.append(
                {
                    "analysis_year": int(year),
                    "signal": signal,
                    "signal_days": int(row["signal_days"]),
                    "signal_day_pct": float(row["signal_day_pct"]),
                    "precision_pct": float(row[f"precision_top_{target_count}_pct"]),
                    "recall_pct": float(row[f"recall_top_{target_count}_pct"]),
                    "lift_x": float(row[f"lift_vs_top_{target_count}_base_rate_x"]),
                    "avg_revenue_signal_eur_per_mw": float(row["avg_revenue_signal_eur_per_mw"]),
                    "avg_revenue_non_signal_eur_per_mw": float(row["avg_revenue_non_signal_eur_per_mw"]),
                    "maintenance_day_value_shift_eur_per_mw": float(row["maintenance_day_value_shift_eur_per_mw"]),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["signal", "analysis_year"]).sort_index()


def summarize_day_ahead_signal_stability(
    yearly_signal_evaluation: pd.DataFrame,
    signal_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    if yearly_signal_evaluation.empty:
        return pd.DataFrame()

    frame = yearly_signal_evaluation.reset_index()
    if signal_names is not None:
        frame = frame[frame["signal"].isin(list(signal_names))]
    if frame.empty:
        return pd.DataFrame()

    rows = []
    for signal, group in frame.groupby("signal"):
        rows.append(
            {
                "signal": signal,
                "years": int(group["analysis_year"].nunique()),
                "positive_maintenance_shift_years": int((group["maintenance_day_value_shift_eur_per_mw"] > 0).sum()),
                "median_precision_pct": float(group["precision_pct"].median()),
                "min_precision_pct": float(group["precision_pct"].min()),
                "max_precision_pct": float(group["precision_pct"].max()),
                "median_recall_pct": float(group["recall_pct"].median()),
                "min_recall_pct": float(group["recall_pct"].min()),
                "max_recall_pct": float(group["recall_pct"].max()),
                "median_lift_x": float(group["lift_x"].median()),
                "min_lift_x": float(group["lift_x"].min()),
                "max_lift_x": float(group["lift_x"].max()),
                "median_maintenance_shift_eur_per_mw": float(group["maintenance_day_value_shift_eur_per_mw"].median()),
                "min_maintenance_shift_eur_per_mw": float(group["maintenance_day_value_shift_eur_per_mw"].min()),
                "max_maintenance_shift_eur_per_mw": float(group["maintenance_day_value_shift_eur_per_mw"].max()),
            }
        )
    return pd.DataFrame(rows).set_index("signal").sort_values("median_lift_x", ascending=False)
