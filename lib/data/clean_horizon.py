"""
Clean Horizon Battery Storage Index — Germany.

Source: https://www.cleanhorizon.com/battery-index/
Free public data: monthly annualised gross revenue (kEUR/MW/yr) for 1h/2h/4h batteries.
Covers DA, ID, FCR, aFRR combined (no per-market breakdown in free tier).
"""

import pandas as pd
from pathlib import Path

CACHE_PATH = Path(__file__).parent / "cache" / "clean_horizon_de_index.csv"


def load_index(duration_h: float = 2.0) -> pd.DataFrame:
    """
    Load Clean Horizon index and return DataFrame with columns:
    year_month (str), year (int), month (int), revenue_keur (float).
    """
    col_map = {1.0: "1h index", 2.0: "2h index", 4.0: "4h index"}
    col = col_map.get(duration_h, "2h index")

    df = pd.read_csv(CACHE_PATH)
    df["year"] = df["Category"].str[:4].astype(int)
    df["month"] = df["Category"].str[5:7].astype(int)
    df["revenue_keur"] = df[col].astype(float)
    return df[["Category", "year", "month", "revenue_keur"]]


def annual_average(duration_h: float = 2.0) -> dict[int, float]:
    """
    Return {year: avg_keur_mw_yr} — annual average of monthly index values.
    Only includes years with all 12 months of data.
    """
    df = load_index(duration_h)
    grouped = df.groupby("year")
    result = {}
    for year, gdf in grouped:
        if len(gdf) >= 12:
            result[year] = round(gdf["revenue_keur"].mean(), 1)
    return result


def annual_average_all(duration_h: float = 2.0) -> dict[int, float]:
    """
    Return {year: avg_keur_mw_yr} — includes partial years too.
    """
    df = load_index(duration_h)
    return {
        int(year): round(gdf["revenue_keur"].mean(), 1)
        for year, gdf in df.groupby("year")
    }
