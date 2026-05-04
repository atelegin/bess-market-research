"""Anchor-month precompute for Note 4 trader-portal section, **v4.2 M-stack**.

Builds the publication M1–M5 policy stack (not the legacy L-stack) at
v4.2 paper-grade calibration (FCR phantom + Stage-2 pass-2 + EVE-anchored
kernel_scale=0.66 + bid_win=1.0 + capture=0.10 + r_min=0.20 + bid_hurdle=20)
and runs ONE template year on real DE markets (default 2026), then keeps
only the days for the requested anchor month.

The output pkl matches the lightweight shape app.py expects:
``{'year', 'month', 'results': {policy_name: {'daily_logs': [...]}}}``

Default: ``--year 2026 --month 3`` (March 2026, the most recent fully
published anchor as of late April 2026).

Usage::

    .venv/bin/python notes/trader-aging-aware/precompute/anchor_month.py
    .venv/bin/python notes/trader-aging-aware/precompute/anchor_month.py --month 4
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from lib.analysis.lifecycle_npv import compare_policies
from lib.analysis.physics_wear_lookup import physics_wear_cost_per_mwh
from lib.data.day_ahead_prices import fetch_day_ahead_prices
from lib.models.adp_shadow_cost import NaivePolicy, ProgressivePolicy
from lib.models.adp_solver import default_grids
from lib.models.adp_solver_intraday import (
    IntradayADPSolver, fit_hourly_price_scenarios,
)
from lib.models.degradation import PRESETS
from lib.models.price_regime import fit_regimes

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# v4.2 calibration knobs (mirror precompute_v40_with_fcr_regen.py + v4.2 single-template-year + kernel_scale=0.66).
BID_WIN_RATE = 1.00
DISPATCH_MODE = "two_stage"
RESERVE_H = 1.0
BID_WIN_IN_LP = True
CAPTURE_FACTOR = 0.10
R_MIN = 0.20
KERNEL_SCALE = 0.66
BID_HURDLE = 20.0
FCR_REVENUE = 36.0


def build_m_stack(regime_fit_year: int = 2024) -> dict:
    """Build the publication M1–M5 stack at v4.2 calibration.

    M1_naive    — no shadow cost
    M2_flat     — flat €/MWh wear (Kumtepeli proxy)
    M3_scarcity — flat × SoH-state scarcity
    M4_physics  — ADP base + physics-from-duty (Stage-2 pass-2 refined)
    M5_adp      — pure intraday ADP, no flat / no scarcity / no physics
    """
    print(f"Fitting regimes + intraday ADP on {regime_fit_year} DA…")
    da = fetch_day_ahead_prices(start=f"{regime_fit_year}-01-01",
                                end=f"{regime_fit_year}-12-31")
    rc = fit_regimes(da["price_eur_mwh"], n_regimes=3)
    grids = default_grids()
    profiles = fit_hourly_price_scenarios(
        da["price_eur_mwh"], rc, n_scenarios=20,
    )
    intraday_solver = IntradayADPSolver(
        regime_classification=rc, hourly_price_profiles=profiles,
        grids=grids, energy_mwh=2.0, power_mw=1.0,
    )
    t0 = time.time()
    intraday_result = intraday_solver.solve()
    print(f"Intraday DP solved in {time.time() - t0:.1f}s")

    flat_base = 100_000.0 / 6_000.0  # ≈ 16.67 €/MWh

    return {
        "M1_naive": NaivePolicy(),
        "M2_flat": ProgressivePolicy(
            name="M2_flat",
            flat_base_eur_per_mwh=flat_base,
        ),
        "M3_scarcity": ProgressivePolicy(
            name="M3_scarcity",
            flat_base_eur_per_mwh=flat_base, use_scarcity=True,
        ),
        "M4_physics": ProgressivePolicy(
            name="M4_physics",
            flat_base_eur_per_mwh=0.0, use_scarcity=False,
            adp_result=intraday_result, adp_grids=grids, adp_regime=rc,
            use_physics_from_duty=True,
            physics_preset=PRESETS["eve_lf280k"],
            physics_temperature_c=25.0,
            physics_capex_eur_per_mwh=100_000.0,
            physics_max_wear_eur_per_mwh=500.0,
            physics_kernel_scale=KERNEL_SCALE,
        ),
        "M5_adp": ProgressivePolicy(
            name="M5_adp",
            flat_base_eur_per_mwh=0.0, use_scarcity=False,
            adp_result=intraday_result, adp_grids=grids, adp_regime=rc,
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--month", type=int, default=3,
                   help="Anchor month (1-12). Default 3 = March.")
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    policies = build_m_stack()

    print(f"\nRunning 1y × {len(policies)} policies on template year {args.year}…")
    print(f"  v4.2 calibration: bid_win={BID_WIN_RATE}, capture={CAPTURE_FACTOR}, "
          f"r_min={R_MIN}, kernel_scale={KERNEL_SCALE}, hurdle={BID_HURDLE}, "
          f"fcr={FCR_REVENUE}, mode={DISPATCH_MODE}")
    t0 = time.time()
    results = compare_policies(
        policies=policies,
        n_years=1, template_years=(args.year,),
        initial_soh=1.0, warranty_floor=0.80, discount_rate=0.07,
        power_mw=1.0, duration_h=2.0, max_cycles=2.0,
        bid_win_rate=BID_WIN_RATE,
        bid_win_in_lp=BID_WIN_IN_LP,
        afrr_reserve_duration_hours=RESERVE_H,
        use_physics_degradation=True,
        physics_preset_name="eve_lf280k",
        physics_temperature_c=25.0,
        dispatch_mode=DISPATCH_MODE,
        afrr_energy_capture_factor=CAPTURE_FACTOR,
        r_min_per_block_mw=R_MIN,
        physics_kernel_scale=KERNEL_SCALE,
        afrr_bid_hurdle_eur_per_mw_h=BID_HURDLE,
        fcr_revenue_keur_per_mw_per_year=FCR_REVENUE,
        yearly_bid_win_rate_scale=np.ones(1),
        yearly_fcr_revenue_scale=np.ones(1),
        collect_diagnostics_year=0,
    )
    print(f"Lifecycle solve done in {time.time() - t0:.1f}s\n")

    # Filter dispatch logs to the target anchor month.
    out_results: dict[str, dict] = {}
    print(f"## Anchor {args.year}-{args.month:02d} summary")
    print(f"{'policy':>14}  {'days':>4}  {'rev €':>10}  {'FEC':>6}")
    for name, r in results.items():
        days_logs = [d for d in (r.diagnostic_days or [])
                     if d.date.month == args.month and d.date.year == args.year]
        rev = sum(d.daily_revenue_eur for d in days_logs)
        fec = sum(d.full_equivalent_cycles for d in days_logs)
        print(f"{name:>14}  {len(days_logs):>4}  {rev:>10.0f}  {fec:>6.1f}")
        out_results[name] = {"daily_logs": days_logs}

    payload = {
        "year": args.year,
        "month": args.month,
        "calibration": {
            "bid_win_rate": BID_WIN_RATE,
            "dispatch_mode": DISPATCH_MODE,
            "capture_factor": CAPTURE_FACTOR,
            "r_min_per_block_mw": R_MIN,
            "physics_kernel_scale": KERNEL_SCALE,
            "afrr_bid_hurdle_eur_per_mw_h": BID_HURDLE,
            "fcr_revenue_keur_per_mw_per_year": FCR_REVENUE,
        },
        "results": out_results,
    }
    out = DATA_DIR / f"anchor_{args.year}-{args.month:02d}_v42.pkl"
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nWrote {out} ({out.stat().st_size / (1024 * 1024):.2f} MB) "
          f"total {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
