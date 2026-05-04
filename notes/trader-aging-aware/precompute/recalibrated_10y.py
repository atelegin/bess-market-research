"""Recalibrated 10-y headline: same M-stack + v4.2 calibration as the
flat-market run, but with the *German BESS Outlook* mid-case per-stream
trajectory baked in via per-year price scaling. The LP sees year-y
prices in year y and adapts dispatch each year (commits less aFRR cap
as it compresses, leans on wholesale as DA/ID grow).

Outputs ``data/precomputed_v42_recalibrated_10y.pkl`` with M-keyed
results in the same shape as ``precomputed_v42_5methods_10y.pkl``.

Usage::

    .venv/bin/python notes/trader-aging-aware/precompute/recalibrated_10y.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from lib.analysis.lifecycle_npv import compare_policies
# market_trend imports kept available but no longer used — K-factor
# calibration replaces the raw revenue-ratio scaling that was overshooting
# Note 1 magnitudes by ~2×.

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# v4.2 calibration knobs (mirror anchor_month.py exactly).
BID_WIN_RATE = 1.00
DISPATCH_MODE = "two_stage"
RESERVE_H = 1.0
BID_WIN_IN_LP = True
CAPTURE_FACTOR = 0.10
R_MIN = 0.20
KERNEL_SCALE = 0.66
BID_HURDLE = 20.0
FCR_REVENUE = 36.0

N_YEARS = 10
TEMPLATE_YEAR = 2025  # template prices replayed each year (then scaled)


def _build_m_stack():
    """Re-import build_m_stack from sibling anchor_month.py by file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_anchor_month", Path(__file__).parent / "anchor_month.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_m_stack()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # K-factor calibration: per-stream scaling against the in-repo
    # `lib/models/projection.py` mid-case (matches what Note 1 v2 will
    # show once it redeploys; deployed Note 1 is currently frozen at
    # commit 59b3215 with older numbers, but this trajectory panel does
    # NOT try to match deployed totals — its purpose is to show how the
    # M1→M5 ranking changes under per-stream compression, not absolute
    # levels). The right panel is normalized to its own M1 = 100 in
    # app.py; absolute trajectory totals are illustrative only.
    #
    # Note 1 mid-case per-stream targets (k€/MW/yr), 2026→2035:
    note1_da = np.array([7.0, 25.6, 40.5, 49.5, 57.3, 61.9, 65.8, 68.8, 70.0, 69.6])
    note1_id = np.array([3.5, 13.1, 21.1, 26.3, 31.1, 34.3, 37.1, 39.6, 41.0, 41.5])
    note1_cap = np.array([120.2, 84.8, 58.8, 44.8, 34.0, 28.6, 24.7, 21.6, 19.8, 18.2])
    note1_en = np.array([11.5, 8.1, 5.6, 4.3, 3.3, 2.7, 2.4, 2.1, 1.9, 1.7])
    note1_wholesale = note1_da + note1_id  # LP lumps DA+ID into wholesale

    # Our LP M1 stream split under flat-2025 (Y1 fresh cell, from
    # precomputed_v42_5methods_10y.pkl):
    M1_FLAT_WHOLESALE = 43.8   # k€/MW/yr (DA only; ID lumps into DA)
    M1_FLAT_CAP = 185.1
    M1_FLAT_ENERGY = 14.7

    yearly_wholesale = note1_wholesale / M1_FLAT_WHOLESALE
    yearly_afrr_cap = note1_cap / M1_FLAT_CAP
    yearly_afrr_en = note1_en / M1_FLAT_ENERGY
    yearly_bid_win = np.ones(N_YEARS)
    # FCR phantom follows Note 1's aFRR cap trajectory (same fleet-
    # saturation driver).
    yearly_fcr = yearly_afrr_cap.copy()

    print("=" * 76)
    print(" v4.2 RECALIBRATED 10-y headline (per-stream Note 1 mid-case)")
    print("=" * 76)
    target_total = note1_wholesale + note1_cap + note1_en
    print(f"Note 1 fleet target (k€/MW/yr) Y1→Y10: "
          f"{target_total[0]:.0f} → {target_total[-1]:.0f}")
    print(f"K_wholesale Y1→Y10: {yearly_wholesale[0]:.2f} → {yearly_wholesale[-1]:.2f}")
    print(f"K_cap       Y1→Y10: {yearly_afrr_cap[0]:.2f} → {yearly_afrr_cap[-1]:.2f}")
    print(f"K_energy    Y1→Y10: {yearly_afrr_en[0]:.2f} → {yearly_afrr_en[-1]:.2f}")
    print(f"K_FCR       Y1→Y10: {yearly_fcr[0]:.2f} → {yearly_fcr[-1]:.2f}")
    print()

    t_all = time.time()
    policies = _build_m_stack()

    # Common kwargs for all 5 simulate_lifecycle calls.
    common_kwargs = dict(
        n_years=N_YEARS, template_years=(TEMPLATE_YEAR,),
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
        yearly_bid_win_rate_scale=yearly_bid_win,
        yearly_fcr_revenue_scale=yearly_fcr,
        yearly_price_scale_wholesale=yearly_wholesale,
        yearly_price_scale_afrr_cap=yearly_afrr_cap,
        yearly_price_scale_afrr_energy=yearly_afrr_en,
        collect_diagnostics_year=None,
    )

    # M1 / M2 / M3: regime-independent shadow costs by design (M1=0,
    # M2=flat €/MWh, M3=flat × SoH-scarcity). No wear-scale applied —
    # their value-add is exactly that they're robust without retraining.
    print(f"\nRunning M1 / M2 / M3 (no wear-scale, n_workers=3)…")
    t0 = time.time()
    results = compare_policies(
        policies={k: v for k, v in policies.items()
                  if k in ("M1_naive", "M2_flat", "M3_scarcity")},
        n_workers=3,
        **common_kwargs,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    # M4 / M5: ADP-based shadow cost calibrated on 2024 regime would
    # otherwise go stale under the trajectory. Scale wear vector by
    # K_wholesale[y] each year — linear approximation to re-solving the
    # intraday DP per year. Captures "save SoC for high-spread hour"
    # gradient correctly when wholesale price level shifts.
    # ADP value function is calibrated on 2024 regimes — under trajectory
    # this goes stale. Tried linear yearly_wear_scale = K_wholesale
    # (multiplying wear by 2.5× in Y10) but it over-restricts the LP
    # because wear scales but aFRR cap revenue collapses (K_cap=0.10 in
    # Y10 makes aFRR uneconomic regardless), leaving the LP idle at the
    # high wear hurdle. Honest answer without re-solving the DP per year:
    # leave M4 / M5 wear at 2025-calibration. Result reads "M5 ADP loses
    # its lead under regime shift because its shadow cost is regime-stale".
    print(f"\nRunning M4 / M5 (n_workers=2)…")
    t0 = time.time()
    results_adp = compare_policies(
        policies={k: v for k, v in policies.items()
                  if k in ("M4_physics", "M5_adp")},
        n_workers=2,
        **common_kwargs,
    )
    print(f"  done in {time.time() - t0:.1f}s")
    results.update(results_adp)
    print(f"\nLifecycle runs done in {time.time() - t_all:.1f}s\n")

    naive_npv = results["M1_naive"].lifetime_npv_eur
    print(f"{'policy':>14}  {'NPV k€':>9}  {'Δ vs M1':>9}  {'EOL':>4}  {'FEC':>6}")
    for name in ["M1_naive", "M2_flat", "M3_scarcity", "M4_physics", "M5_adp"]:
        r = results[name]
        npv = r.lifetime_npv_eur / 1000.0
        delta = (r.lifetime_npv_eur / naive_npv - 1) * 100
        eol = r.years_to_floor + 1 if r.years_to_floor is not None else N_YEARS
        fec = int(r.annual_fec.sum())
        print(f"{name:>14}  {npv:>9.1f}  {delta:>+8.1f}%  {eol:>4d}  {fec:>6,}")

    payload = {
        "n_years": N_YEARS,
        "template_year": TEMPLATE_YEAR,
        "trajectory": "Outlook mid-case, K-factor anchored to Note 1 per-stream targets",
        "calibration": {
            "bid_win_rate": BID_WIN_RATE,
            "dispatch_mode": DISPATCH_MODE,
            "capture_factor": CAPTURE_FACTOR,
            "r_min_per_block_mw": R_MIN,
            "physics_kernel_scale": KERNEL_SCALE,
            "afrr_bid_hurdle_eur_per_mw_h": BID_HURDLE,
            "fcr_revenue_keur_per_mw_per_year": FCR_REVENUE,
        },
        "yearly_factors": {
            "wholesale": yearly_wholesale,
            "afrr_cap": yearly_afrr_cap,
            "afrr_energy": yearly_afrr_en,
            "bid_win": yearly_bid_win,
            "fcr": yearly_fcr,
        },
        "results": results,
    }
    out = DATA_DIR / "precomputed_v42_recalibrated_10y.pkl"
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nWrote {out} ({out.stat().st_size / (1024 * 1024):.2f} MB) "
          f"total {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
