"""
Pre-compute all analytics for note 01 — The Cost of Missing the Best Days.

Run once:  python notes/01-de-bess-best-days/precompute.py
Results saved to notes/01-de-bess-best-days/data/precomputed.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from lib.analysis.concentration import compute_concentration_stats
from lib.analysis.day_ahead_signals import (
    build_day_ahead_observable_table,
    build_day_ahead_watchlist_table,
    concatenate_day_ahead_observable_tables,
    summarize_day_ahead_feature_separation,
)
from lib.analysis.drivers import compute_price_shape_profiles
from lib.analysis.opportunity_bridge import (
    build_daily_value_curve,
    summarize_reallocated_same_throughput_vs_strict_daily_cap,
)
from lib.analysis.revenue_breakdown import summarize_missing_top_days
from lib.data.cache import get_or_build_dataframe, make_cache_key
from lib.data.intraday_prices import fetch_id_aep
from lib.data.day_ahead_prices import fetch_day_ahead_prices
from lib.models.degradation import (
    DEFAULT_DEGRADATION_ASSUMPTIONS,
    equivalent_stress_fec_per_year,
    estimate_years_to_eol,
    lifecycle_value_profile,
)
from lib.models.dispatch_detailed import (
    AGGRESSIVE_STRATEGY,
    CONSERVATIVE_STRATEGY,
    DispatchStrategy,
    run_dispatch_for_period,
    run_dispatch_with_intraday_overlay_for_period,
)

YEARS = list(range(2021, 2026))
DEFAULT_RTE = 0.86
BASE_CASE_DURATION_HOURS = 2
BASE_CASE_ANALYSIS_YEAR = 2025
BASE_CASE_DISCOUNT_RATE = 0.08
BASE_CASE_PROJECT_LIFETIME = 15
DATA_DIR = Path(__file__).parent / "data"


def compute_dispatch_with_disk_cache(
    year: int,
    strategy: DispatchStrategy,
    price_frame: pd.DataFrame,
    energy_mwh: float,
    rte: float,
    market_key: str = "day_ahead",
    intraday_price_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cache_key = make_cache_key(
        "dispatch",
        year=year,
        market=market_key,
        strategy=strategy.name,
        energy_mwh=energy_mwh,
        rte=round(rte, 4),
        power_mw=1.0,
        version=4,
    )
    return get_or_build_dataframe(
        cache_key=cache_key,
        builder=lambda: run_dispatch_with_intraday_overlay_for_period(
            day_ahead_price_frame=price_frame,
            intraday_price_frame=intraday_price_frame,
            strategy=strategy,
            energy_mwh=energy_mwh,
            rte=rte,
        )
        if intraday_price_frame is not None and not intraday_price_frame.empty
        else run_dispatch_for_period(price_frame=price_frame, strategy=strategy, energy_mwh=energy_mwh, rte=rte),
        ttl_hours=None,
        force_refresh=False,
        metadata={"year": year, "strategy": strategy.name, "market": market_key},
    )


def build_cycle_intensity_frontier_payload(
    year: int,
    price_frame: pd.DataFrame,
    intraday_price_frame: pd.DataFrame,
    energy_mwh: float,
    rte: float,
    discount_rate: float,
    project_lifetime: int,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    cycle_caps = np.arange(0.25, 4.01, 0.25)
    records = []
    dispatch_by_cycle_cap: dict[float, pd.DataFrame] = {}
    for cycle_cap in cycle_caps:
        strategy = DispatchStrategy(
            name=f"frontier_{year}_{energy_mwh:.1f}h_{str(cycle_cap).replace('.', 'p')}",
            label=f"{cycle_cap:.2f} cycles/day",
            max_cycles=float(cycle_cap),
            soc_min_frac=AGGRESSIVE_STRATEGY.soc_min_frac,
            soc_max_frac=AGGRESSIVE_STRATEGY.soc_max_frac,
            min_spread_eur_mwh=AGGRESSIVE_STRATEGY.min_spread_eur_mwh,
        )
        dispatch = compute_dispatch_with_disk_cache(
            year=year,
            strategy=strategy,
            price_frame=price_frame,
            energy_mwh=energy_mwh,
            rte=rte,
            market_key="da_id_overlay_frontier_v2",
            intraday_price_frame=intraday_price_frame,
        )
        dispatch_by_cycle_cap[float(cycle_cap)] = dispatch
        lifecycle = lifecycle_value_profile(
            year1_revenue=float(dispatch["revenue_eur_per_mw"].sum()),
            dispatch_frame=dispatch,
            years=project_lifetime,
            discount_rate=discount_rate,
            annual_market_decline=0.0,
            assumptions=DEFAULT_DEGRADATION_ASSUMPTIONS,
        )
        records.append(
            {
                "max_cycles_per_day": float(cycle_cap),
                "year1_revenue_eur_per_mw": float(dispatch["revenue_eur_per_mw"].sum()),
                "full_equivalent_cycles_per_year": float(dispatch["full_equivalent_cycles"].sum()),
                "stress_fec_per_year": equivalent_stress_fec_per_year(
                    dispatch,
                    assumptions=DEFAULT_DEGRADATION_ASSUMPTIONS,
                ),
                "years_to_eol": estimate_years_to_eol(
                    dispatch,
                    assumptions=DEFAULT_DEGRADATION_ASSUMPTIONS,
                ),
                "discounted_lifetime_value_eur_per_mw": float(
                    lifecycle["cumulative_discounted_revenue_eur_per_mw"].iloc[-1]
                ),
            }
        )
    return pd.DataFrame(records), dispatch_by_cycle_cap


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    energy_mwh = float(BASE_CASE_DURATION_HOURS)
    rte = DEFAULT_RTE
    analysis_year = BASE_CASE_ANALYSIS_YEAR
    discount_rate = BASE_CASE_DISCOUNT_RATE
    project_lifetime = BASE_CASE_PROJECT_LIFETIME

    # ── Fetch prices ──────────────────────────────────────────
    print("Fetching day-ahead prices...")
    prices = fetch_day_ahead_prices(start=f"{YEARS[0]}-01-01", end=f"{YEARS[-1]}-12-31", force_refresh=False)

    # ── Dispatch across years ─────────────────────────────────
    conservative_dispatch_by_year: dict[int, pd.DataFrame] = {}
    intraday_prices_by_year: dict[int, pd.DataFrame] = {}
    intraday_missing_years: list[int] = []

    for year in YEARS:
        print(f"  Dispatching {year}...")
        year_prices = prices[prices.index.year == year]
        try:
            intraday_prices_by_year[year] = fetch_id_aep(
                start=f"{year}-01-01", end=f"{year}-12-31", force_refresh=False,
            )
        except Exception:
            intraday_prices_by_year[year] = pd.DataFrame(columns=["price_eur_mwh"])
            intraday_missing_years.append(year)

        conservative_dispatch_by_year[year] = compute_dispatch_with_disk_cache(
            year=year,
            strategy=CONSERVATIVE_STRATEGY,
            price_frame=year_prices,
            energy_mwh=energy_mwh,
            rte=rte,
            market_key="da_id_overlay",
            intraday_price_frame=intraday_prices_by_year[year],
        )

    # ── Analytics for analysis year ───────────────────────────
    print("Computing analytics...")
    selected_prices = prices[prices.index.year == analysis_year]
    selected_intraday_prices = intraday_prices_by_year.get(analysis_year, pd.DataFrame(columns=["price_eur_mwh"]))
    conservative_selected = conservative_dispatch_by_year[analysis_year]

    concentration_stats = compute_concentration_stats(conservative_selected["revenue_eur_per_mw"])
    missed_days_curve = summarize_missing_top_days(
        conservative_selected["revenue_eur_per_mw"],
        top_day_counts=tuple(range(1, 51)),
    )
    selected_top_day_dates = conservative_selected["revenue_eur_per_mw"].sort_values(ascending=False).head(20).index
    price_shape_profiles = compute_price_shape_profiles(selected_prices, selected_top_day_dates)

    selected_day_ahead_observables = build_day_ahead_observable_table(
        day_ahead_price_frame=selected_prices,
        outcome_dispatch=conservative_selected,
        top_day_counts=(10, 20),
    )
    feature_comparison = summarize_day_ahead_feature_separation(selected_day_ahead_observables)

    day_ahead_observables_by_year = {
        year: build_day_ahead_observable_table(
            day_ahead_price_frame=prices[prices.index.year == year],
            outcome_dispatch=conservative_dispatch_by_year[year],
            top_day_counts=(10, 20),
        )
        for year in YEARS
    }
    pooled_day_ahead_observables = concatenate_day_ahead_observable_tables(day_ahead_observables_by_year)
    pooled_watchlist_top20 = build_day_ahead_watchlist_table(pooled_day_ahead_observables, target_count=20)
    pooled_base_rate_pct = 100 * float(pooled_day_ahead_observables["is_top_20_revenue_day"].mean())

    # ── Cycle intensity frontier ──────────────────────────────
    print("Building cycle intensity frontier...")
    _, cycle_dispatch_by_cap = build_cycle_intensity_frontier_payload(
        year=analysis_year,
        price_frame=selected_prices,
        intraday_price_frame=selected_intraday_prices,
        energy_mwh=energy_mwh,
        rte=rte,
        discount_rate=discount_rate,
        project_lifetime=project_lifetime,
    )
    daily_value_curve = build_daily_value_curve(cycle_dispatch_by_cap)
    equal_throughput_summary, _ = summarize_reallocated_same_throughput_vs_strict_daily_cap(
        dispatch_by_cycle_cap=cycle_dispatch_by_cap,
        daily_value_curve=daily_value_curve,
        daily_caps=(1.0, 1.5, 2.0),
    )

    # ── Serialize ─────────────────────────────────────────────
    payload = {
        "concentration_stats": concentration_stats,
        "missed_days_curve": missed_days_curve,
        "price_shape_profiles": price_shape_profiles,
        "feature_comparison": feature_comparison,
        "pooled_watchlist_top20": pooled_watchlist_top20,
        "pooled_base_rate_pct": pooled_base_rate_pct,
        "equal_throughput_summary": equal_throughput_summary,
        "intraday_missing_years": intraday_missing_years,
        "analysis_year": analysis_year,
        "duration_hours": BASE_CASE_DURATION_HOURS,
        "conservative_selected": conservative_selected,
        "selected_prices": selected_prices,
    }

    path = DATA_DIR / "precomputed.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    size_mb = path.stat().st_size / 1e6
    print(f"\nSaved to {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
