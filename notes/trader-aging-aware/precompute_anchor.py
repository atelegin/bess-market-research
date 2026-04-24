"""
Anchor-month precompute for Note 4 Section "One month, five policies".

Runs all 5 shadow-cost policies over a single calendar month (default
Feb 2026 — last month with Clean Horizon index published). Uses
starting SoH = 1.0 (fresh cell), so the comparison isolates *policy
behavior* on identical market conditions without multi-year age-drift.

Outputs to ``data/anchor_month.pkl`` alongside the 10y ``precomputed.pkl``
so the Streamlit app can show both narratives: (a) lifetime NPV ordering
on realistic age, (b) single-month dispatch-footprint differences on a
concrete recent month.

Usage::

    .venv/bin/python notes/trader-aging-aware/precompute_anchor.py --month 2  # Feb 2026
    .venv/bin/python notes/trader-aging-aware/precompute_anchor.py --month 3  # March 2026 once CH publishes
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from lib.analysis.anchor_month import run_anchor_month
from lib.data.day_ahead_prices import fetch_day_ahead_prices
from lib.models.adp_shadow_cost import (
    ADPPolicy, ADPPolicyIntraday,
    AgingAwareDepreciationPolicy, DepreciationProxyPolicy, NaivePolicy,
    SoCWindowPolicy,
)
from lib.models.adp_solver import ADPSolver, default_grids
from lib.models.adp_solver_intraday import (
    IntradayADPSolver, fit_hourly_price_profiles,
)
from lib.models.price_regime import fit_regimes

DATA_DIR = Path(__file__).parent / "data"


def build_policies(regime_fit_year: int = 2024) -> dict:
    """Fit regime classifier + ADP solvers on recent historical DA."""
    print(f"Fitting regimes + solvers on {regime_fit_year} DA…")
    da = fetch_day_ahead_prices(start=f"{regime_fit_year}-01-01",
                                end=f"{regime_fit_year}-12-31")
    rc = fit_regimes(da["price_eur_mwh"], n_regimes=3)

    grids = default_grids()
    ref_rev = np.array([
        (rc.stats[r].mean_spread_eur_mwh * 2.0 * 0.60) for r in range(rc.n_regimes)
    ])
    rev_curve = np.outer(ref_rev, grids.action_grid / 1.5)
    adp_simpl_solver = ADPSolver(
        regime_classification=rc, revenue_curve=rev_curve, grids=grids,
        warranty_breach_penalty_eur=200_000,
    )
    adp_simpl_result = adp_simpl_solver.solve(tol=1e-2, max_iter=1500)

    profiles = fit_hourly_price_profiles(da["price_eur_mwh"], rc)
    intraday_solver = IntradayADPSolver(
        regime_classification=rc, hourly_price_profiles=profiles,
        grids=grids, energy_mwh=2.0, power_mw=1.0,
    )
    intraday_result = intraday_solver.solve()

    base = 100_000.0 / 6_000.0
    return {
        "naive": NaivePolicy(),
        "soc_window": SoCWindowPolicy(soc_min_frac=0.20, soc_max_frac=0.80),
        "depreciation_proxy": DepreciationProxyPolicy(
            capex_eur_per_mwh=100_000, lifetime_throughput_ratio=6_000,
        ),
        "aging_aware_depreciation": AgingAwareDepreciationPolicy(
            base_eur_per_mwh=base, warranty_floor=0.80,
        ),
        "adp_simplified": ADPPolicy(
            solver=adp_simpl_solver, result=adp_simpl_result,
            regime_classification=rc,
        ),
        "adp_intraday": ADPPolicyIntraday(
            intraday_result=intraday_result, grids=grids,
            regime_classification=rc,
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--month", type=int, default=2,
                   help="Anchor month (1-12). Default 2 = Feb 2026.")
    p.add_argument("--starting-soh", type=float, default=1.0)
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    policies = build_policies()
    print(f"\nRunning anchor month {args.year}-{args.month:02d} × 5 policies…")
    t1 = time.time()
    run = run_anchor_month(
        policies=policies,
        year=args.year, month=args.month,
        initial_soh=args.starting_soh,
        power_mw=1.0, duration_h=2.0,
        max_cycles=2.0, max_afrr_participation=0.40,
    )
    print(f"Done in {time.time() - t1:.1f}s\n")

    print("## Anchor-month summary")
    print(f"{'policy':>28}  {'month EUR':>10}  {'ann. k€/MW':>11}  {'FEC':>6}  {'r_pos':>6}  {'days':>5}")
    for name, r in run.results.items():
        print(f"{name:>28}  {r.monthly_revenue_eur:>10.0f}  "
              f"{r.annualised_revenue_keur_per_mw:>11.1f}  "
              f"{r.total_fec:>6.1f}  {r.mean_r_pos_mw:>6.2f}  {r.days_solved:>5}")

    out = DATA_DIR / f"anchor_{args.year}-{args.month:02d}.pkl"
    with open(out, "wb") as f:
        pickle.dump(run, f)
    print(f"\nWrote {out} ({out.stat().st_size / (1024*1024):.2f} MB) total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
