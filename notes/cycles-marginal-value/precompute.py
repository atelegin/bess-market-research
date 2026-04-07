"""
Pre-compute analytics for Note 2 — Cycles and Marginal Value.

Run once:  python notes/cycles-marginal-value/precompute.py
Results saved to notes/cycles-marginal-value/data/precomputed.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from lib.data.day_ahead_prices import fetch_day_ahead_prices
from lib.data.intraday_prices import fetch_id_aep
from lib.models.dispatch_detailed import (
    AGGRESSIVE_STRATEGY,
    DispatchStrategy,
    run_dispatch_with_intraday_overlay_for_period,
    run_dispatch_for_period,
)
from lib.models.degradation import (
    DEFAULT_DEGRADATION_ASSUMPTIONS,
    equivalent_stress_fec_per_year,
    estimate_years_to_eol,
)
from lib.data.cache import get_or_build_dataframe, make_cache_key
from lib.config import DEFAULT_BESS_BUILDOUT

YEARS = [2021, 2022, 2023, 2024, 2025]
DURATIONS = [1.0, 2.0, 4.0]
RTE = 0.86
CYCLE_CAPS = list(np.arange(0.25, 4.01, 0.25))
DATA_DIR = Path(__file__).parent / "data"


def cached_dispatch(year, strategy, price_frame, energy_mwh, intraday_price_frame=None):
    cache_key = make_cache_key(
        "dispatch", year=year, market="da_id_overlay_frontier_v2",
        strategy=strategy.name, energy_mwh=energy_mwh,
        rte=round(RTE, 4), power_mw=1.0, version=4,
    )
    return get_or_build_dataframe(
        cache_key=cache_key,
        builder=lambda: run_dispatch_with_intraday_overlay_for_period(
            day_ahead_price_frame=price_frame,
            intraday_price_frame=intraday_price_frame,
            strategy=strategy, energy_mwh=energy_mwh, rte=RTE,
        )
        if intraday_price_frame is not None and not intraday_price_frame.empty
        else run_dispatch_for_period(
            price_frame=price_frame, strategy=strategy,
            energy_mwh=energy_mwh, rte=RTE,
        ),
        ttl_hours=None, force_refresh=False,
        metadata={"year": year, "strategy": strategy.name},
    )


def build_frontier(year, duration_h, da_prices, id_prices):
    energy_mwh = duration_h
    year_da = da_prices[da_prices.index.year == year]
    year_id = (
        id_prices[id_prices.index.year == year]
        if not id_prices.empty
        else pd.DataFrame(columns=["price_eur_mwh"])
    )

    records = []
    for cap in CYCLE_CAPS:
        strategy = DispatchStrategy(
            name=f"frontier_{year}_{energy_mwh:.1f}h_{str(cap).replace('.', 'p')}",
            label=f"{cap:.2f} c/d", max_cycles=float(cap),
            soc_min_frac=AGGRESSIVE_STRATEGY.soc_min_frac,
            soc_max_frac=AGGRESSIVE_STRATEGY.soc_max_frac,
            min_spread_eur_mwh=AGGRESSIVE_STRATEGY.min_spread_eur_mwh,
        )
        dispatch = cached_dispatch(year, strategy, year_da, energy_mwh, year_id)
        annual_rev = float(dispatch["revenue_eur_per_mw"].sum())
        annual_fec = float(dispatch["full_equivalent_cycles"].sum())
        records.append({
            "year": year,
            "duration_h": duration_h,
            "max_cycles_per_day": float(cap),
            "annual_revenue_eur_per_mw": annual_rev,
            "annual_fec": annual_fec,
            "years_to_eol": estimate_years_to_eol(dispatch),
        })
    frontier = pd.DataFrame(records)
    max_rev = frontier["annual_revenue_eur_per_mw"].max()
    frontier["pct_of_max"] = (
        100 * frontier["annual_revenue_eur_per_mw"] / max_rev if max_rev > 0 else 0
    )
    frontier["marginal_eur_per_fec"] = (
        frontier["annual_revenue_eur_per_mw"].diff()
        / frontier["annual_fec"].diff()
    )
    return frontier


def build_half_yearly_frontiers(years, da_prices, id_by_year):
    """Split existing cached dispatch into H1/H2 and build annualised frontiers.

    Re-uses the full-year dispatch cache — no re-optimisation needed.
    Revenue and FEC are summed per half, then annualised (×2).
    """
    records = []
    for duration_h in DURATIONS:
        energy_mwh = duration_h
        for year in years:
            year_da = da_prices[da_prices.index.year == year]
            year_id = (
                id_by_year.get(year, pd.DataFrame(columns=["price_eur_mwh"]))
            )
            for cap in CYCLE_CAPS:
                strategy = DispatchStrategy(
                    name=f"frontier_{year}_{energy_mwh:.1f}h_{str(cap).replace('.', 'p')}",
                    label=f"{cap:.2f} c/d", max_cycles=float(cap),
                    soc_min_frac=AGGRESSIVE_STRATEGY.soc_min_frac,
                    soc_max_frac=AGGRESSIVE_STRATEGY.soc_max_frac,
                    min_spread_eur_mwh=AGGRESSIVE_STRATEGY.min_spread_eur_mwh,
                )
                dispatch = cached_dispatch(year, strategy, year_da, energy_mwh, year_id)
                for half, label in [(1, "H1"), (2, "H2")]:
                    if half == 1:
                        mask = dispatch.index.month <= 6
                    else:
                        mask = dispatch.index.month > 6
                    sub = dispatch[mask]
                    if sub.empty:
                        continue
                    days = len(sub)
                    half_rev = float(sub["revenue_eur_per_mw"].sum())
                    half_fec = float(sub["full_equivalent_cycles"].sum())
                    # Annualise: scale to 365 days
                    ann_rev = half_rev * 365 / days
                    ann_fec = half_fec * 365 / days
                    records.append({
                        "year": year,
                        "half": label,
                        "period": f"{year} {label}",
                        "duration_h": duration_h,
                        "max_cycles_per_day": float(cap),
                        "annual_revenue_eur_per_mw": ann_rev,
                        "annual_fec": ann_fec,
                        "days_in_half": days,
                    })
    df = pd.DataFrame(records)
    # Add pct_of_max per (year, half, duration_h)
    for (yr, h, dur), grp in df.groupby(["year", "half", "duration_h"]):
        max_rev = grp["annual_revenue_eur_per_mw"].max()
        df.loc[grp.index, "pct_of_max"] = (
            100 * grp["annual_revenue_eur_per_mw"] / max_rev if max_rev > 0 else 0
        )
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching prices...")
    da_prices = fetch_day_ahead_prices(
        start=f"{YEARS[0]}-01-01", end=f"{YEARS[-1]}-12-31", force_refresh=False,
    )

    id_by_year = {}
    for year in YEARS:
        try:
            id_by_year[year] = fetch_id_aep(
                start=f"{year}-01-01", end=f"{year}-12-31", force_refresh=False,
            )
            print(f"  ID {year}: {len(id_by_year[year])} rows")
        except Exception as e:
            print(f"  ID {year}: FAILED ({e})")
            id_by_year[year] = pd.DataFrame(columns=["price_eur_mwh"])

    # Build all frontiers
    all_frontiers = []
    for duration_h in DURATIONS:
        for year in YEARS:
            print(f"  Frontier: {year}/{duration_h:.0f}h ...", end="", flush=True)
            frontier = build_frontier(
                year, duration_h, da_prices,
                id_by_year.get(year, pd.DataFrame(columns=["price_eur_mwh"])),
            )
            all_frontiers.append(frontier)
            max_rev = frontier["annual_revenue_eur_per_mw"].max()
            print(f" {max_rev/1000:.1f} kEUR")

    frontiers_df = pd.concat(all_frontiers, ignore_index=True)

    # Build half-yearly frontiers (H1/H2) from cached daily dispatch
    print("\nBuilding half-yearly frontiers...")
    half_yearly_df = build_half_yearly_frontiers(YEARS, da_prices, id_by_year)
    for (yr, h, dur), grp in half_yearly_df.groupby(["year", "half", "duration_h"]):
        if dur == 2.0:
            max_rev = grp["annual_revenue_eur_per_mw"].max()
            print(f"  {yr} {h} / {dur:.0f}h: {max_rev/1000:.1f} kEUR (annualised)")

    # Fleet sizes for context
    fleet_gw = {y: DEFAULT_BESS_BUILDOUT.get(y, 0) for y in YEARS}

    # ── Daily value of second cycle ────────────────────────────
    # For each year: revenue at cap=2 minus revenue at cap=1, per day.
    # Shows how much the second daily cycle is worth — the "why" behind
    # the shrinking orange segment.
    print("\nBuilding daily second-cycle value...")
    second_cycle_value = {}
    energy_mwh = 2.0
    for year in YEARS:
        year_da = da_prices[da_prices.index.year == year]
        year_id = id_by_year.get(year, pd.DataFrame(columns=["price_eur_mwh"]))
        revs = {}
        for cap in [1.0, 2.0]:
            strategy = DispatchStrategy(
                name=f"frontier_{year}_{energy_mwh:.1f}h_{str(cap).replace('.', 'p')}",
                label=f"{cap:.2f} c/d", max_cycles=float(cap),
                soc_min_frac=AGGRESSIVE_STRATEGY.soc_min_frac,
                soc_max_frac=AGGRESSIVE_STRATEGY.soc_max_frac,
                min_spread_eur_mwh=AGGRESSIVE_STRATEGY.min_spread_eur_mwh,
            )
            dispatch = cached_dispatch(year, strategy, year_da, energy_mwh, year_id)
            revs[cap] = dispatch["revenue_eur_per_mw"].values
        delta = (revs[2.0] - revs[1.0]).tolist()
        second_cycle_value[year] = delta
        arr = np.array(delta)
        print(f"  {year}: median=€{np.median(arr):.0f}/day, "
              f">€50: {(arr>50).sum()} days, "
              f"≤€0: {(arr<=0).sum()} days")

    # ── Median SoC profiles: "rich" vs "poor" second-cycle days ──
    # Rich: days where 2nd cycle earns >€50. Poor: ≤€10.
    # Shows the structural difference in battery behaviour.
    from lib.models.dispatch_detailed import optimize_day
    print("\nBuilding median SoC profiles (rich vs poor days)...")
    soc_profiles = {}
    energy_mwh_soc = 2.0
    for year in YEARS:
        year_da = da_prices[da_prices.index.year == year]
        second_vals = second_cycle_value[year]
        soc_rich = []
        soc_poor = []
        soc_init = (0.05 + 0.95) / 2 * energy_mwh_soc
        for i, (date, grp) in enumerate(year_da.groupby(year_da.index.date)):
            prices = grp["price_eur_mwh"].values
            if len(prices) != 24 or i >= len(second_vals):
                continue
            result = optimize_day(
                prices=prices, energy_mwh=energy_mwh_soc, rte=RTE,
                soc_min_frac=0.05, soc_max_frac=0.95,
                max_cycles=4.0, power_mw=1.0,
            )
            full_soc = np.concatenate([[soc_init], result["soc"]])
            soc_pct = (full_soc / energy_mwh_soc * 100).tolist()
            val = second_vals[i]
            if val > 50:
                soc_rich.append(soc_pct)
            elif val <= 10:
                soc_poor.append(soc_pct)
        soc_profiles[year] = {
            "rich_median": np.median(soc_rich, axis=0).tolist() if soc_rich else [],
            "poor_median": np.median(soc_poor, axis=0).tolist() if soc_poor else [],
            "rich_count": len(soc_rich),
            "poor_count": len(soc_poor),
        }
        print(f"  {year}: rich={len(soc_rich)} days, poor={len(soc_poor)} days")

    payload = {
        "frontiers": frontiers_df,
        "half_yearly_frontiers": half_yearly_df,
        "years": YEARS,
        "durations": DURATIONS,
        "rte": RTE,
        "fleet_gw": fleet_gw,
        "second_cycle_value": second_cycle_value,
        "soc_profiles": soc_profiles,
    }

    # ── Average hourly price profiles ─────────────────────────
    print("\nBuilding average hourly profiles...")
    for year in YEARS:
        year_da = da_prices[da_prices.index.year == year].copy()
        year_da["hour"] = year_da.index.hour
        profile = year_da.groupby("hour")["price_eur_mwh"].mean()
        payload[f"avg_profile_{year}"] = {
            "hours": profile.index.tolist(),
            "prices": profile.values.tolist(),
        }
        print(f"  {year}: range €{profile.max()-profile.min():.0f} "
              f"(low h{profile.idxmin()} €{profile.min():.0f}, "
              f"high h{profile.idxmax()} €{profile.max():.0f})")

    path = DATA_DIR / "precomputed.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    size_mb = path.stat().st_size / 1e6
    print(f"\nSaved to {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
