"""
Precompute for Note 4 (trader-aging-aware).

Runs all five shadow-cost policies over a 10-year lifecycle on real DE
market data, collects lifetime NPV, SoH trajectories, dispatch logs for
the "diagnostic" year (year 1 = 2023), and the diagnostic signals
computed over those logs.

Outputs are pickled to ``notes/trader-aging-aware/data/precomputed.pkl``
for the Streamlit app.

Runtime: ~5 minutes (intraday ADP solver dominates — ~90 s per fit, plus
5 × 10y × 2 market-year lookups at ~30 s each).

Usage::

    .venv/bin/python notes/trader-aging-aware/precompute.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from lib.analysis.aging_aware_diagnostics import compute_all_signals
from lib.analysis.lifecycle_npv import compare_policies
from lib.analysis.stacked_year_runner import _prefetch_year_frames
from lib.data.day_ahead_prices import fetch_day_ahead_prices
from lib.models.adp_shadow_cost import (
    ADPPolicy,
    ADPPolicyIntraday,
    AgingAwareDepreciationPolicy,
    DepreciationProxyPolicy,
    NaivePolicy,
    SoCWindowPolicy,
)
from lib.models.adp_solver import ADPSolver, default_grids
from lib.models.adp_solver_intraday import (
    IntradayADPSolver,
    fit_hourly_price_profiles,
)
from lib.models.price_regime import fit_regimes

N_YEARS = 10
TEMPLATE_YEARS = (2023, 2025)
DATA_DIR = Path(__file__).parent / "data"


def build_policies() -> dict:
    """Build the 5 policies, fitting ADP solvers on 2024 DA."""
    print("Fitting regimes + solvers on 2024 DA…")
    da_2024 = fetch_day_ahead_prices(start="2024-01-01", end="2024-12-31")
    rc = fit_regimes(da_2024["price_eur_mwh"], n_regimes=3)

    # Simplified ADP
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

    # Intraday ADP
    profiles = fit_hourly_price_profiles(da_2024["price_eur_mwh"], rc)
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    policies = build_policies()

    print(f"\nRunning {N_YEARS}y × 5 policies over 2023↔2025 rotation…")
    t0 = time.time()
    results = compare_policies(
        policies=policies,
        n_years=N_YEARS, template_years=TEMPLATE_YEARS,
        initial_soh=1.0, warranty_floor=0.80,
        discount_rate=0.07,
        power_mw=1.0, duration_h=2.0,
        max_cycles=2.0, max_afrr_participation=0.40,
        use_physics_degradation=True,          # Note 3 physics kernel
        physics_preset_name="eve_lf280k",
        physics_temperature_c=25.0,
        collect_diagnostics_year=0,  # year 1 = 2023 template
    )
    print(f"Lifecycle runs done in {time.time() - t0:.1f}s\n")

    print("Computing diagnostic signals over Y1 dispatch logs…")
    diagnostics = {}
    for name, r in results.items():
        if r.diagnostic_days:
            diagnostics[name] = compute_all_signals(
                policy_name=name, days=r.diagnostic_days,
            )
        else:
            diagnostics[name] = None

    print("\nHeadline NPV:")
    for name, r in results.items():
        print(f"  {name:>28}: {r.lifetime_npv_eur / 1000:>8.1f} k€/MW  "
              f"(EOL Y{r.years_to_floor or N_YEARS}, Σ FEC {r.annual_fec.sum():.0f})")

    payload = {
        "n_years": N_YEARS,
        "template_years": TEMPLATE_YEARS,
        "results": results,
        "diagnostics": diagnostics,
    }

    out_path = DATA_DIR / "precomputed.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {out_path} ({size_mb:.1f} MB) in {time.time() - t_all:.1f}s total.")


if __name__ == "__main__":
    main()
