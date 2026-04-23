"""
Empirical policy comparison for Note 4 (A4 validation).

Runs the 4 shadow-cost policies (naive / depreciation-proxy /
aging-aware-depreciation / ADP) over an N-year lifecycle on shared
real market data, then reports lifetime NPV, annual revenue profile,
SoH trajectory, and FEC for each.

Decision path from Anton's option C: if policies already separate
meaningfully on NPV with the simplified DP, we don't need to extend
the ADP to full Holtorf-Shin. If they collapse → extend state with
SoC and time-of-day.

Usage::

    .venv/bin/python scripts/note4_policy_comparison.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from lib.analysis.lifecycle_npv import compare_policies
from lib.analysis.stacked_year_runner import _prefetch_year_frames
from lib.data.day_ahead_prices import fetch_day_ahead_prices
from lib.models.adp_shadow_cost import (
    ADPPolicy,
    AgingAwareDepreciationPolicy,
    DepreciationProxyPolicy,
    NaivePolicy,
    default_policies_for_comparison,
)
from lib.models.adp_solver import (
    ADPSolver,
    default_grids,
    empirical_daily_revenue_curve,
)
from lib.models.price_regime import fit_regimes


def build_adp_policy() -> ADPPolicy:
    """Fit regimes on 2024 DA, build revenue curve from regime means,
    solve the DP, and wrap as a policy."""
    da = fetch_day_ahead_prices(start="2024-01-01", end="2024-12-31")
    rc = fit_regimes(da["price_eur_mwh"], n_regimes=3)
    # Revenue curve: use regime mean daily spread as a proxy for per-FEC
    # arbitrage capture. ~2h BESS captures ~60 % of p95-p5 at 1 FEC/day.
    # We anchor the revenue curve to EUR/MW/day at reference intensity 1.5
    # so it matches observed stacked-LP outputs (~$500-700/day in 2024).
    grid = default_grids()
    ref_rev_per_regime = np.array([
        (rc.stats[r].mean_spread_eur_mwh * 2.0 * 0.60) for r in range(rc.n_regimes)
    ])  # 2 MWh × 60% × spread
    curve = np.outer(ref_rev_per_regime, grid.action_grid / 1.5)
    solver = ADPSolver(
        regime_classification=rc, revenue_curve=curve, grids=grid,
        fade_per_fec_at_soh_1=2e-4, calendar_fade_per_day=2e-5,
        discount_per_year=0.98, warranty_breach_penalty_eur=200_000,
    )
    result = solver.solve(tol=1e-2, max_iter=1500)
    return ADPPolicy(
        solver=solver, result=result, regime_classification=rc,
    )


def main(n_years: int = 10, max_days_per_year: int | None = None) -> None:
    print(f"Building 4 policies…")
    base = 100_000.0 / 6_000.0
    policies = {
        "naive": NaivePolicy(),
        "depreciation_proxy": DepreciationProxyPolicy(
            capex_eur_per_mwh=100_000, lifetime_throughput_ratio=6_000,
        ),
        "aging_aware_depreciation": AgingAwareDepreciationPolicy(
            base_eur_per_mwh=base, warranty_floor=0.80,
        ),
        "adp": build_adp_policy(),
    }
    print()
    print(f"Simulating {n_years}y × 4 policies on 2023↔2025 rotating data…")
    print(f"(max_days_per_year={max_days_per_year or 'full'})")
    t0 = time.time()
    results = compare_policies(
        policies=policies,
        n_years=n_years,
        template_years=(2023, 2025),
        initial_soh=1.0, warranty_floor=0.80,
        discount_rate=0.07,
        power_mw=1.0, duration_h=2.0,
        max_cycles=2.0,
        max_afrr_participation=0.40,
        max_days_per_year=max_days_per_year,
    )
    dt = time.time() - t0
    print(f"Completed in {dt:.1f}s\n")

    # Summary table
    print("## Lifetime NPV comparison")
    print()
    print(f"{'policy':>28} | {'NPV (k€/MW)':>12} | {'end SoH':>8} | {'Σ FEC':>7} | {'days':>5}")
    print("-" * 72)
    npv_baseline = results["naive"].lifetime_npv_eur
    for name, r in results.items():
        npv_k = r.lifetime_npv_eur / 1000.0
        end_soh = r.end_of_year_soh[-1]
        total_fec = r.annual_fec.sum()
        pct_vs_naive = (r.lifetime_npv_eur - npv_baseline) / npv_baseline * 100 if npv_baseline else 0
        print(
            f"{name:>28} | {npv_k:>12.1f} | {end_soh:>8.3f} | {total_fec:>7.0f} | {r.days_solved:>5}"
            f"   Δnpv={pct_vs_naive:+.1f}%"
        )
    print()

    # Annual revenue
    print("## Annual revenue (k€/MW, nominal)")
    print()
    header = " | ".join(f"{name:>10}" for name in results)
    print(f"{'year':>5} | {header}")
    print("-" * (8 + len(results) * 13))
    for y in range(n_years):
        row = " | ".join(
            f"{r.annual_revenue_eur[y] / 1000:>10.1f}" for r in results.values()
        )
        print(f"{y + 1:>5} | {row}")
    print()

    # SoH trajectory
    print("## SoH trajectory")
    print()
    header = " | ".join(f"{name:>10}" for name in results)
    print(f"{'year':>5} | {header}")
    print("-" * (8 + len(results) * 13))
    for y in range(n_years):
        row = " | ".join(
            f"{r.end_of_year_soh[y]:>10.4f}" for r in results.values()
        )
        print(f"{y + 1:>5} | {row}")
    print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--max-days", type=int, default=None,
                   help="Cap days per simulated year (smoke test).")
    args = p.parse_args()
    main(n_years=args.years, max_days_per_year=args.max_days)
