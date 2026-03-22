"""
Calibrate BESS revenue model — hybrid empirical + structural approach.

What we CAN calibrate from data:
  - Gas → spread elasticity: DE DA spreads 2020-2025 (R²=0.97)
  - Revenue level at known fleet sizes: CH (DE 2023-2025), Modo (UK 2023-2025)

What we CANNOT calibrate from 2023-2025 data:
  - Cannibalization at high fleet sizes (DE had only 1.5-3.5 GW)
  - Ancillary-wholesale split (CH reports combined totals)

For the unobservable parameters, we use structural constraints:
  - Ancillary market depth is physically bounded (FCR ~0.6 GW, aFRR ~4 GW)
  - Cannibalization must match UK's observed revenue decline at 3.5→6.8 GW
  - At extreme fleet sizes (40+ GW), revenue cannot go below SRMC floor

This is transparent: we state what's empirical, what's structural, and what's assumed.
"""

import numpy as np
from scipy.optimize import minimize
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# STAGE 1: Gas elasticity (fully empirical)
# ══════════════════════════════════════════════════════════════

# DE annual average daily DA price spread (€/MWh) vs TTF gas price
# Source: Energy-Charts hourly prices, ICE TTF settlement
DE_SPREAD = {
    2020: (28, 9),     # (avg daily spread, TTF €/MWh)
    2021: (55, 47),
    2022: (95, 130),
    2023: (52, 41),
    2024: (48, 34),
    2025: (50, 36),
}


def stage1_gas_elasticity():
    """Fit spread = base * (1 + e * (gas/gas_ref - 1)). Returns (e, R²)."""
    gas_ref = 41.0  # 2023 TTF
    spread_ref = 52.0

    spreads = np.array([s for s, _ in DE_SPREAD.values()])
    gases = np.array([g for _, g in DE_SPREAD.values()])

    result = minimize(
        lambda e: np.sum((spread_ref * (1 + e[0] * (gases / gas_ref - 1)) - spreads) ** 2),
        x0=[0.4], bounds=[(0.1, 1.5)],
    )
    e = result.x[0]
    predicted = spread_ref * (1 + e * (gases / gas_ref - 1))
    ss_res = np.sum((predicted - spreads) ** 2)
    ss_tot = np.sum((spreads - spreads.mean()) ** 2)
    return e, 1 - ss_res / ss_tot


# ══════════════════════════════════════════════════════════════
# STAGE 2: Full model — structural + empirical
# ══════════════════════════════════════════════════════════════

# Revenue observations (kEUR/MW/yr)
# DE: Clean Horizon 2h annual average
# UK: Modo Energy annual (post-DC era), converted to EUR
# ES: Clean Horizon 2h annual average (Spain)
# IT: Clean Horizon 2h annual average (Italy)
PANEL = [
    # (country, year, total_rev, gas, bess_gw, pv_gw, demand_twh)
    ("DE", 2023, 230, 41, 1.5, 82, 517),
    ("DE", 2024, 200, 34, 2.4, 95, 530),
    ("DE", 2025, 237, 36, 3.5, 100, 560),
    ("UK", 2023, 75, 35, 3.5, 16, 300),
    ("UK", 2024, 64, 30, 4.5, 18, 310),
    ("UK", 2025, 78, 32, 6.8, 20, 315),
    # ES: high PV/demand ratio → extreme duck curve spreads, near-zero BESS fleet
    # PV: REE (Red Eléctrica) installed capacity data
    # BESS: AESIA / MITECO reports (grid-scale only)
    # Demand: REE annual demand statistics
    ("ES", 2023, 352, 41, 0.1, 25, 250),
    ("ES", 2024, 303, 34, 0.3, 30, 255),
    ("ES", 2025, 273, 36, 0.8, 35, 260),
    # IT: zonal market (Terna MSD), lower spreads, growing BESS fleet
    # PV: Terna statistical data / GSE
    # BESS: Terna grid-scale capacity reports
    # Demand: Terna annual demand
    # Gas: PSV ~€5 premium over TTF
    ("IT", 2023, 83, 46, 0.5, 30, 310),
    ("IT", 2024, 69, 39, 1.0, 35, 315),
    ("IT", 2025, 88, 41, 2.5, 40, 320),
]

# Physical constraints (not fitted — these are measurable)
FCR_CAPACITY_GW = 0.6      # DK-West/DE LFC block FCR demand ~613 MW (ENTSO-E 2026)
AFRR_DEPTH_GW = 4.0        # aFRR addressable by BESS
TOTAL_ANC_DEPTH_GW = 4.5   # Combined ancillary market depth


def full_model(
    country, gas, bess, pv, demand,
    # Fitted parameters:
    bl_de, bl_uk, bl_es, bl_it, gas_elast,
    pv_sens, beta_d,
    canib_max, canib_half, canib_steep,
):
    """
    Total revenue = wholesale(drivers) + ancillary(fleet saturation).

    Ancillary is modeled structurally from market depth, not fitted.
    """
    # Country-specific refs
    COUNTRY_PARAMS = {
        "DE": {
            "bl": bl_de, "gas_ref": 41, "pv_ref": 82, "dem_ref": 517,
            "bess_ref": 1.5, "anc_at_ref": 135.0, "anc_at_17": 13.0,
        },
        "UK": {
            "bl": bl_uk, "gas_ref": 35, "pv_ref": 16, "dem_ref": 300,
            "bess_ref": 3.5, "anc_at_ref": 22.0, "anc_at_17": 5.0,
        },
        "ES": {
            # Spain: OMIE single zone, minimal BESS fleet, high PV/demand ratio
            # Ancillary: negligible BESS participation in aFRR (REE)
            "bl": bl_es, "gas_ref": 41, "pv_ref": 25, "dem_ref": 250,
            "bess_ref": 0.1, "anc_at_ref": 5.0, "anc_at_17": 1.0,
        },
        "IT": {
            # Italy: zonal MSD market, lower spreads than single-zone markets
            # Gas: PSV premium ~€5 over TTF
            # Ancillary: Terna MSD, some BESS participation
            "bl": bl_it, "gas_ref": 46, "pv_ref": 30, "dem_ref": 310,
            "bess_ref": 0.5, "anc_at_ref": 15.0, "anc_at_17": 3.0,
        },
    }
    p = COUNTRY_PARAMS[country]
    bl = p["bl"]
    gas_ref, pv_ref, dem_ref = p["gas_ref"], p["pv_ref"], p["dem_ref"]
    bess_ref = p["bess_ref"]
    anc_at_ref = p["anc_at_ref"]
    anc_at_17 = p["anc_at_17"]

    # ── Wholesale ──
    gas_f = bl * gas_elast * (gas / gas_ref - 1)
    pv_f = pv_sens * (pv - pv_ref) / 100
    dem_f = beta_d * max(demand / dem_ref - 1, 0) ** 1.5
    can_f = canib_max / (1 + np.exp(-canib_steep * (bess - canib_half)))
    can_ref = canib_max / (1 + np.exp(-canib_steep * (bess_ref - canib_half)))
    wholesale = max(bl + gas_f + pv_f + dem_f - (can_f - can_ref), 20)

    # ── Ancillary (structural, not fitted) ──
    # Exponential decay calibrated to two known points
    a1 = max(anc_at_ref, 1)
    a2 = max(anc_at_17, 0.5)
    k = np.log(a1 / a2) / (17.0 - bess_ref)
    A = a1 / np.exp(-k * bess_ref)
    ancillary = max(A * np.exp(-k * bess), 2.0)

    return wholesale + ancillary


def stage2_fit(gas_elast):
    """Fit remaining parameters on DE+UK+ES+IT panel."""

    def objective(x):
        bl_de, bl_uk, bl_es, bl_it, pv_sens, beta_d, canib_max, canib_half, canib_steep = x
        errors = []
        for country, year, rev, gas, bess, pv, demand in PANEL:
            pred = full_model(country, gas, bess, pv, demand,
                              bl_de, bl_uk, bl_es, bl_it, gas_elast,
                              pv_sens, beta_d,
                              canib_max, canib_half, canib_steep)
            errors.append((pred - rev) ** 2)
        return np.mean(errors)

    bounds = [
        (60, 180),    # bl_de — wholesale baseline at 2023 conditions
        (20, 80),     # bl_uk
        (150, 400),   # bl_es — Spain baseline (high PV spreads)
        (30, 120),    # bl_it — Italy baseline (zonal market, lower spreads)
        (0, 60),      # pv_sens — wider bounds with ES/IT cross-section
        (5, 30),      # beta_d
        (15, 45),     # canib_max
        (10, 25),     # canib_half
        (0.1, 0.8),   # canib_steep
    ]

    from scipy.optimize import differential_evolution
    result = differential_evolution(
        objective, bounds=bounds,
        seed=42, maxiter=5000, tol=1e-12, polish=True,
    )

    names = ["bl_de", "bl_uk", "bl_es", "bl_it", "pv_sens", "beta_d",
             "canib_max", "canib_half", "canib_steep"]
    return dict(zip(names, result.x)), result.fun


def main():
    print("=" * 70)
    print("BESS Revenue Model — Empirical Calibration")
    print("=" * 70)

    # ── Stage 1 ──
    print(f"\n{'━' * 70}")
    print("STAGE 1: Gas → spread elasticity (DE 2020-2025, 6 points)")
    print(f"{'━' * 70}")
    gas_elast, r2 = stage1_gas_elasticity()
    print(f"\n  e_gas = {gas_elast:.3f}   R² = {r2:.3f}")
    print(f"  When gas doubles, DA spread increases {gas_elast*100:.0f}%")

    spread_ref = 52.0
    gas_ref = 41.0
    print(f"\n  {'Year':>6} {'Spread':>8} {'Model':>8} {'Gas':>6} {'Err%':>7}")
    for y, (s, g) in DE_SPREAD.items():
        p = spread_ref * (1 + gas_elast * (g / gas_ref - 1))
        print(f"  {y:>6} {s:>8.0f} {p:>8.1f} {g:>6.0f} {(p-s)/s*100:>+6.1f}%")

    # ── Stage 2 ──
    print(f"\n{'━' * 70}")
    print("STAGE 2: Full model on DE+UK+ES+IT panel (2023-2025, 12 points)")
    print(f"{'━' * 70}")
    params, obj = stage2_fit(gas_elast)
    params["gas_elast"] = gas_elast

    print(f"\n  Fitted parameters:")
    for k, v in params.items():
        print(f"    {k:>15s} = {v:.4f}")

    # ── Show fit ──
    print(f"\n  {'Ctry':>4} {'Year':>5} {'Actual':>8} {'Model':>8} {'Err%':>7}")
    all_errs = []
    for country, year, rev, gas, bess, pv, demand in PANEL:
        pred = full_model(country, gas, bess, pv, demand,
                          params["bl_de"], params["bl_uk"],
                          params["bl_es"], params["bl_it"], gas_elast,
                          params["pv_sens"], params["beta_d"],
                          params["canib_max"], params["canib_half"], params["canib_steep"])
        err = pred - rev
        all_errs.append(err)
        print(f"  {country:>4} {year:>5} {rev:>8.0f} {pred:>8.1f} {err/rev*100:>+6.1f}%")
    rmse = np.sqrt(np.mean(np.array(all_errs) ** 2))
    print(f"\n  Panel RMSE: {rmse:.1f} kEUR")

    # ── Forward projection ──
    print(f"\n{'━' * 70}")
    print("DE forward projection 2026-2040 (default scenario)")
    print(f"{'━' * 70}")

    from config import DEFAULT_BESS_BUILDOUT, DEMAND_2026, DEMAND_2040, fleet_degradation_factor
    from model.projection import interpolate_linear
    from config import TTF_2026, TTF_2040, PV_GW_2026, PV_GW_2040

    print(f"\n  {'Year':>6} {'WS':>7} {'Anc':>7} {'Deg':>6} {'Total':>7}")
    for y in range(2026, 2041):
        gas = interpolate_linear(TTF_2026, TTF_2040, 2026, 2040, y)
        bess = DEFAULT_BESS_BUILDOUT.get(y, 40)
        pv = interpolate_linear(PV_GW_2026, PV_GW_2040, 2026, 2040, y)
        demand = interpolate_linear(DEMAND_2026, DEMAND_2040, 2026, 2040, y)

        total = full_model("DE", gas, bess, pv, demand,
                           params["bl_de"], params["bl_uk"],
                           params["bl_es"], params["bl_it"], gas_elast,
                           params["pv_sens"], params["beta_d"],
                           params["canib_max"], params["canib_half"], params["canib_steep"])

        # Ancillary component
        a1, a2 = 135.0, 13.0
        k = np.log(a1 / a2) / (17.0 - 1.5)
        A = a1 / np.exp(-k * 1.5)
        anc = max(A * np.exp(-k * bess), 2.0)
        ws = total - anc

        proj_buildout = {yr: v for yr, v in DEFAULT_BESS_BUILDOUT.items() if yr >= 2026}
        deg = fleet_degradation_factor(y, proj_buildout)

        print(f"  {y:>6} {ws*deg:>7.1f} {anc*deg:>7.1f} {deg:>6.3f} {total*deg:>7.1f}")

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print("CALIBRATION PROVENANCE")
    print(f"{'═' * 70}")
    print(f"""
┌────────────────────┬────────────────────────────────────────────────────┐
│ Parameter          │ Source                                              │
├────────────────────┼────────────────────────────────────────────────────┤
│ gas_elast = {gas_elast:.3f}  │ Fitted: DE DA spreads 2020-2025 (R²={r2:.3f})          │
│ pv_sens = {params['pv_sens']:.1f}     │ Fitted: DE+UK+ES+IT panel (PV range 16-100 GW)     │
│ beta_d = {params['beta_d']:.1f}      │ Fitted: DE+UK+ES+IT panel 2023-2025                │
│ canib_max = {params['canib_max']:.1f}  │ Fitted: DE+UK+ES+IT panel 2023-2025                │
│ canib_half = {params['canib_half']:.1f} │ Fitted: DE+UK+ES+IT panel 2023-2025                │
│ canib_steep = {params['canib_steep']:.3f}│ Fitted: DE+UK+ES+IT panel 2023-2025                │
│ anc @ 1.5 GW = 135 │ Structural: regelleistung.net auction data            │
│ anc @ 17 GW = 13   │ Structural: ancillary depth 4.5 GW (measured)        │
│ anc floor = 2      │ Assumption: residual participation                    │
└────────────────────┴────────────────────────────────────────────────────┘

Fit quality:
  Stage 1 (gas elasticity): R² = {r2:.3f}, 6 observations
  Stage 2 (panel fit): RMSE = {rmse:.1f} kEUR, 12 observations (DE+UK+ES+IT)

PV sensitivity identification:
  Panel PV range: UK 16-20 GW, ES 25-35 GW, DE 82-100 GW, IT 30-40 GW
  Cross-section variation from 16 to 100 GW provides much stronger
  identification than DE time-series alone (82→100 GW).
  Note: country baselines absorb market-structure differences,
  so pv_sens captures the marginal PV effect on spreads.

Honest limitations:
  - DE BESS fleet was only 1.5-3.5 GW in 2023-2025. Cannibalization at
    10-40 GW is an extrapolation constrained by UK cross-section + market depth.
  - Ancillary split is structural (from regelleistung.net depth), not fitted.
  - ES/IT ancillary assumptions are approximate (different market structures).
  - Country baselines absorb structural market differences (zonal vs nodal,
    ancillary design, interconnection) — they are not directly comparable.

CONFIG VALUES:
  GAS_SPREAD_ELASTICITY = {gas_elast:.3f}
  PV_SPREAD_SENSITIVITY = {params['pv_sens']:.1f}
  beta_d      = {params['beta_d']:.2f}
  canib_max   = {params['canib_max']:.2f}
  canib_half  = {params['canib_half']:.2f}
  canib_steep = {params['canib_steep']:.4f}
""")

    return params


if __name__ == "__main__":
    main()
