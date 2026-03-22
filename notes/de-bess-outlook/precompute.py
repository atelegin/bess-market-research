"""
Pre-compute historical dispatch and ancillary data for note 02 — BESS Revenue Outlook.

Run once:  python notes/02-de-bess-outlook/precompute.py
Results saved to notes/02-de-bess-outlook/data/precomputed.pkl

Pre-computes data for all three duration options (1h, 2h, 4h) so the
Streamlit app can switch durations instantly.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import logging

from lib.data.day_ahead_prices import fetch_day_ahead_prices, prices_to_daily_arrays
from lib.data.ancillary_prices import fetch_fcr_annual_revenue, fetch_afrr_annual_revenue
from lib.data.clean_horizon import annual_average, annual_average_all
from lib.models.dispatch import dispatch_day, annual_revenue
from lib.models.projection import id_da_ratio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HIST_YEARS = [2023, 2024, 2025]
DURATIONS = [1.0, 2.0, 4.0]
RTE = 0.85
MAX_CYCLES = 2
CALIBRATED_BASELINE_2H = 105.0
DATA_DIR = Path(__file__).parent / "data"


def compute_historical(years: list[int], dur: float, rte: float, cycles: int) -> tuple[dict, list]:
    yearly_rev = {}
    failed_years = []
    for y in years:
        try:
            df = fetch_day_ahead_prices(start=f"{y}-01-01", end=f"{y}-12-31")
            daily = prices_to_daily_arrays(df, resolution_minutes=60)
            year_key = str(y)
            if year_key in daily:
                results = [dispatch_day(p, duration_h=dur, rte=rte, max_cycles=cycles)
                           for p in daily[year_key]]
                yearly_rev[y] = annual_revenue(results) / 1000
        except Exception as e:
            logger.warning(f"Dispatch failed for {y}: {e}")
            failed_years.append((y, str(e)))
    return yearly_rev, failed_years


def compute_hist_stack_ch(hist_rev: dict, dur: float) -> list[dict]:
    try:
        ch_annual = annual_average_all(duration_h=dur)
    except Exception as e:
        logger.warning(f"Clean Horizon data unavailable: {e}")
        return []
    bars = []
    for y in sorted(hist_rev.keys()):
        ch_total = ch_annual.get(y)
        if ch_total is None:
            continue
        da_rev = hist_rev.get(y, 0.0)
        id_rev = da_rev * id_da_ratio(y)
        try:
            fcr = fetch_fcr_annual_revenue(y) or 0.0
        except Exception:
            fcr = 0.0
        try:
            afrr = fetch_afrr_annual_revenue(y) or {"afrr_cap": 0.0, "afrr_energy": 0.0}
        except Exception:
            afrr = {"afrr_cap": 0.0, "afrr_energy": 0.0}
        model_total = da_rev + id_rev + fcr + afrr["afrr_cap"] + afrr["afrr_energy"]
        if model_total <= 0:
            model_total = 1.0
        s = ch_total / model_total
        bars.append({
            "year": y,
            "da": round(da_rev * s, 1), "id": round(id_rev * s, 1),
            "fcr": round(fcr * s, 1), "afrr_cap": round(afrr["afrr_cap"] * s, 1),
            "afrr_energy": round(afrr["afrr_energy"] * s, 1),
            "total": round(ch_total, 1),
        })
    return bars


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict = {}

    for dur in DURATIONS:
        dur_key = f"{dur:.0f}h"
        print(f"Computing historical dispatch for {dur_key}...")
        hist_rev, failed = compute_historical(HIST_YEARS, dur, RTE, MAX_CYCLES)
        print(f"  Revenue: {hist_rev}")
        if failed:
            print(f"  Failed: {failed}")

        print(f"  Building historical stack (Clean Horizon)...")
        hist_bars = compute_hist_stack_ch(hist_rev, dur)

        # Compute baseline
        try:
            ch_full_years = annual_average(duration_h=dur)
            ch_full_years_2h = annual_average(duration_h=2.0)
        except Exception:
            ch_full_years, ch_full_years_2h = {}, {}

        if ch_full_years and ch_full_years_2h and dur != 2.0:
            dur_ratio = np.mean(list(ch_full_years.values())) / np.mean(list(ch_full_years_2h.values()))
            baseline_da = CALIBRATED_BASELINE_2H * dur_ratio
        elif dur == 2.0:
            baseline_da = CALIBRATED_BASELINE_2H
        elif hist_rev:
            baseline_da = np.mean(list(hist_rev.values()))
        else:
            baseline_da = CALIBRATED_BASELINE_2H

        # Clean Horizon annual averages for all durations
        try:
            ch_all = annual_average_all(duration_h=dur)
        except Exception:
            ch_all = {}

        payload[dur_key] = {
            "hist_rev": hist_rev,
            "failed_years": failed,
            "hist_bars": hist_bars,
            "baseline_da": float(baseline_da),
            "ch_all": ch_all,
        }
        print(f"  Baseline DA: {baseline_da:.1f} kEUR")

    path = DATA_DIR / "precomputed.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    size_mb = path.stat().st_size / 1e6
    print(f"\nSaved to {path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
