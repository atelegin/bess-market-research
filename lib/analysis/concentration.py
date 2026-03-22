"""
Revenue concentration analysis — Gini coefficient, Pareto curve, top-N stats.

Quantifies how few days drive the bulk of annual BESS revenue.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def gini_coefficient(values: pd.Series) -> float:
    array = np.sort(values.to_numpy(dtype=float))
    if len(array) == 0 or np.allclose(array.sum(), 0):
        return 0.0
    index = np.arange(1, len(array) + 1)
    return float((2 * np.sum(index * array) / (len(array) * np.sum(array))) - (len(array) + 1) / len(array))


def compute_pareto_curve(revenue_series: pd.Series) -> pd.DataFrame:
    ordered = revenue_series.sort_values(ascending=False).reset_index(drop=True)
    cumulative_revenue = ordered.cumsum()
    total_revenue = cumulative_revenue.iloc[-1] if not cumulative_revenue.empty else 0.0
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(ordered) + 1),
            "daily_revenue_eur_per_mw": ordered,
            "cumulative_days_pct": np.arange(1, len(ordered) + 1) / max(len(ordered), 1),
            "cumulative_revenue_pct": cumulative_revenue / total_revenue if total_revenue else 0.0,
        }
    )


def compute_concentration_stats(revenue_series: pd.Series) -> dict[str, float]:
    ordered = revenue_series.sort_values(ascending=False)
    total = float(ordered.sum())
    stats = {
        "annual_revenue_eur_per_mw": total,
        "gini": gini_coefficient(revenue_series),
        "p50_daily_revenue_eur_per_mw": float(revenue_series.quantile(0.50)),
        "p90_daily_revenue_eur_per_mw": float(revenue_series.quantile(0.90)),
        "p95_daily_revenue_eur_per_mw": float(revenue_series.quantile(0.95)),
        "p99_daily_revenue_eur_per_mw": float(revenue_series.quantile(0.99)),
    }
    for share in (0.10, 0.15, 0.20, 0.50):
        top_n = max(1, int(np.ceil(len(ordered) * share)))
        stats[f"top_{int(share * 100)}_days_pct_of_revenue"] = float(ordered.head(top_n).sum() / total) if total else 0.0
    return stats


def days_to_revenue_share(revenue_series: pd.Series, revenue_share: float) -> int:
    ordered = revenue_series.sort_values(ascending=False)
    if ordered.empty or ordered.sum() <= 0:
        return 0
    cumulative = ordered.cumsum() / ordered.sum()
    return int((cumulative < revenue_share).sum() + 1)
