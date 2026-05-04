"""
Precompute for Note 4 (trader-aging-aware).

Runs all five shadow-cost policies over a 10-year lifecycle on real DE
market data, collects lifetime NPV, SoH trajectories, dispatch logs for
the "diagnostic" year (year 1 = 2023), and the diagnostic signals
computed over those logs.

Outputs are pickled to
``notes/trader-aging-aware/data/precomputed_v42_5methods_10y.pkl``
for the Streamlit app.

Runtime: ~5 minutes (intraday ADP solver dominates — ~90 s per fit, plus
5 × 10y × 2 market-year lookups at ~30 s each).

Usage::

    .venv/bin/python notes/trader-aging-aware/precompute/policies.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from lib.analysis.aging_aware_diagnostics import compute_all_signals
from lib.analysis.lifecycle_npv import compare_policies
from lib.analysis.physics_wear_lookup import physics_wear_cost_per_mwh
from lib.analysis.stacked_year_runner import _prefetch_year_frames
from lib.data.day_ahead_prices import fetch_day_ahead_prices
from lib.models.adp.shadow_cost import (
    NaivePolicy,
    ProgressivePolicy,
)
from lib.models.adp.solver import default_grids
from lib.models.adp.solver_intraday import (
    IntradayADPSolver,
    fit_hourly_price_profiles,
    fit_hourly_price_scenarios,
)
from lib.models.price_regime import fit_regimes

N_YEARS = 10
TEMPLATE_YEARS = (2023, 2025)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_policies() -> dict:
    """Build the progressive 7-layer stack, fitting regime + ADP on 2024 DA."""
    print("Fitting regimes + solvers on 2024 DA…")
    da_2024 = fetch_day_ahead_prices(start="2024-01-01", end="2024-12-31")
    rc = fit_regimes(da_2024["price_eur_mwh"], n_regimes=3)
    grids = default_grids()

    # Intraday ADP — full-stochastic DP (bootstraps 20 historical daily
    # price realisations per regime, averages V over scenarios). Closes
    # methodology simplification #1 (deterministic hourly means). Physics
    # wear added as additive layer at online lookup time (addresses
    # simplification #2).
    profiles = fit_hourly_price_scenarios(
        da_2024["price_eur_mwh"], rc, n_scenarios=20,
    )
    print(f"Stochastic profiles shape: {profiles.shape} "
          f"(3 regimes × 20 scenarios × 24 hours)")
    from lib.models.degradation.simple import PRESETS
    physics_wear = physics_wear_cost_per_mwh(
        soh_grid=grids.soh_grid,
        preset=PRESETS["eve_lf280k"],
        temperature_c=25.0,
        capex_eur_per_mwh=100_000.0,
        warranty_floor=0.80,
    )
    print("Physics-derived wear cost per SoH grid (EUR/MWh):")
    for s, w in zip(grids.soh_grid, physics_wear):
        print(f"  SoH={s:.2f} → {w:7.2f}")
    intraday_solver = IntradayADPSolver(
        regime_classification=rc, hourly_price_profiles=profiles,
        grids=grids, energy_mwh=2.0, power_mw=1.0,
    )
    t_solver = time.time()
    intraday_result = intraday_solver.solve()
    print(f"Stochastic intraday DP solved in {time.time() - t_solver:.1f}s")

    flat_base = 100_000.0 / 6_000.0  # ≈ 16.67 €/MWh
    # Progressive stack: each level adds one layer. Every level L_n
    # includes everything from L_(n-1).
    return {
        # L1 — unconstrained revenue-max
        "L1_naive": NaivePolicy(),
        # L2 — + SoC window [20-80 %] (how commercial warranties implement
        # aging-awareness: a hard envelope, no shadow cost).
        "L2_soc_window": ProgressivePolicy(
            name="L2_soc_window", use_soc_window=True,
        ),
        # L3 — + flat €/MWh throughput cost (Kumtepeli's "poor proxy":
        # CAPEX ÷ lifetime throughput, constant across time and state).
        "L3_flat_wear": ProgressivePolicy(
            name="L3_flat_wear", use_soc_window=True,
            flat_base_eur_per_mwh=flat_base,
        ),
        # L4 — + scarcity scaling (SoH-dependent: cheap at fresh cell,
        # divergent near the warranty floor — a closed-form stepping-stone
        # toward state-dependent opportunity cost).
        "L4_scarcity": ProgressivePolicy(
            name="L4_scarcity", use_soc_window=True,
            flat_base_eur_per_mwh=flat_base, use_scarcity=True,
        ),
        # L5 — + hour-varying |∂V/∂SoC|[h] from intraday ADP (Holtorf-Shin
        # opportunity cost: save SoC for the peak, spend SoC at the trough).
        # NB: the previous iteration also included a multiplicative DoD-
        # rainflow penalty here; empirical runs showed it regressive (-1 pp)
        # because the 20-80 SoC window already suppresses deep-cycle damage.
        # Dropped for the final simplified stack.
        "L5_intraday_adp": ProgressivePolicy(
            name="L5_intraday_adp", use_soc_window=True,
            flat_base_eur_per_mwh=flat_base, use_scarcity=True,
            adp_result=intraday_result, adp_grids=grids, adp_regime=rc,
        ),
        # L6 — "ideal" full physics model: keeps SoC window + intraday ADP,
        # replaces the heuristic flat/scarcity base with a 2-pass
        # physics-from-duty wear computed daily from the observed dispatch
        # (DoD + C-rate + SoC-band sensitivities from Note 3 Wang+Naumann
        # kernel). Closes the biggest methodology simplification: the
        # kernel's full sensitivity set enters the optimisation.
        "L6_physics_full": ProgressivePolicy(
            name="L6_physics_full", use_soc_window=True,
            flat_base_eur_per_mwh=0.0, use_scarcity=False,
            adp_result=intraday_result, adp_grids=grids, adp_regime=rc,
            use_physics_from_duty=True,
            physics_preset=PRESETS["eve_lf280k"],
            physics_temperature_c=25.0,
            physics_capex_eur_per_mwh=100_000.0,
            physics_max_wear_eur_per_mwh=500.0,
            # Match v4.0 simulator's SoH-tracking kernel anchored to EVE
            # LF280K manufacturer 6000-cycle / 80% retention endurance.
            # Without this, the policy's pass-2 wear over-estimates by 1.5×
            # vs the simulator's actual SoH update → over-suppression.
            physics_kernel_scale=0.66,
        ),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    policies = build_policies()

    print(f"\nRunning {N_YEARS}y × 5 policies over 2023↔2025 rotation…")
    t0 = time.time()
    # Main run = LP upper bound: perfect-foresight LP, full bid acceptance,
    # PQ-correct 1 h aFRR reservation. Public indices (CH, Modo, etc.) sit
    # 30-60 % lower because they embed forecast error, market access (bid
    # win rates), risk limits, product deliverability, and recharge
    # management costs the LP doesn't model. We do not "calibrate" the
    # model to indices — the layer comparison is the point, not the level.
    results = compare_policies(
        policies=policies,
        n_years=N_YEARS, template_years=TEMPLATE_YEARS,
        initial_soh=1.0, warranty_floor=0.80,
        discount_rate=0.07,
        power_mw=1.0, duration_h=2.0,
        max_cycles=2.0, max_afrr_participation=0.40,
        afrr_reserve_duration_hours=1.0,
        use_physics_degradation=True,
        physics_preset_name="eve_lf280k",
        physics_temperature_c=25.0,
        collect_diagnostics_year=0,
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

    out_path = DATA_DIR / "precomputed_v42_5methods_10y.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {out_path} ({size_mb:.1f} MB) in {time.time() - t_all:.1f}s total.")


if __name__ == "__main__":
    main()
