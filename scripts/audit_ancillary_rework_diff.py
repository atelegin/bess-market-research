"""
Audit diff: old additive stacking vs new AS/wholesale equilibrium.

Compares `project_full_stack` output under the pre-2026-04 additive model
(wh_total + anc_total per MW of fleet) against the new equilibrium allocation
(Simon/Schäfer correction). Prints markdown table for Anton's sign-off.

Usage:
    .venv/bin/python scripts/audit_ancillary_rework_diff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.analysis.observed_revenue import observed_total_revenue
from lib.config import DEFAULT_BESS_BUILDOUT
from lib.data.clean_horizon import annual_average
from lib.models.ancillary import HISTORICAL_YEARS_WITH_MEASURED_DATA, ancillary_revenue
from lib.models.degradation import PRESETS, fleet_average_capacity
from lib.models.projection import project_full_stack, project_wholesale


def legacy_full_stack(
    years: list[int],
    historical_da_keur: float,
    bess_buildout: dict[int, float],
    duration_h: float = 2.0,
    **wholesale_kwargs,
) -> list[dict]:
    """Replica of the pre-rework additive-stacking projection. Disables the
    historical override so the saturation model runs for all years (that's
    what the legacy pipeline did)."""
    results = []
    proj_buildout = {y: v for y, v in bess_buildout.items() if y >= min(years)}
    for year in years:
        bess_gw = bess_buildout.get(
            year, bess_buildout[max(k for k in bess_buildout if k <= year)]
        )
        wh = project_wholesale(
            year=year, historical_da_annual=historical_da_keur, bess_gw=bess_gw,
            **wholesale_kwargs,
        )
        anc = ancillary_revenue(
            year=year, bess_gw=bess_gw, duration_h=duration_h,
            use_historical_if_available=False,
        )
        deg = fleet_average_capacity(
            year=year, buildout=proj_buildout, preset=PRESETS["baseline_fleet"],
        )
        results.append({
            "year": year,
            "total": round((wh["wholesale_total"] + anc["total"]) * deg, 1),
            "wh": round(wh["wholesale_total"] * deg, 1),
            "anc": round(anc["total"] * deg, 1),
        })
    return results


def main() -> None:
    years = list(range(2023, 2041))
    historical_da = 90.0  # representative 2023–2025 DA anchor

    legacy = legacy_full_stack(years, historical_da, DEFAULT_BESS_BUILDOUT, duration_h=2.0)
    new = project_full_stack(
        years=years,
        historical_da_keur=historical_da,
        bess_buildout=DEFAULT_BESS_BUILDOUT,
        duration_h=2.0,
    )

    legacy_by = {r["year"]: r for r in legacy}
    new_by = {r["year"]: r for r in new}
    ch = annual_average(2.0)  # context only, NOT calibration anchor

    # Internal Tier 1/2 ground truth for historical years
    observed = {}
    for y in HISTORICAL_YEARS_WITH_MEASURED_DATA:
        if y in years:
            obs = observed_total_revenue(y, duration_h=2.0)
            if obs is not None:
                observed[y] = obs["total"]

    print("# Note 1 audit diff — AS/wholesale equilibrium rework")
    print()
    print(f"Baseline DA: {historical_da} kEUR/MW/yr | Duration: 2h | Default buildout")
    print()
    print("Columns:")
    print("  old total     — legacy additive stacking (wh + anc)")
    print("  new total     — new equilibrium allocation (Simon/Schäfer)")
    print("  observed (T2) — Tier 1/2 ground truth for historical years")
    print("                  (regelleistung + netztransparenz + LP dispatch)")
    print("  CH (context)  — Clean Horizon index. Context only — not a")
    print("                  calibration target (see benchmark-reconciliation).")
    print()
    print(
        "| year | bess_gw | old total | new total | Δ new-old | observed (T2) | CH (context) | eq_type | f_on_as |"
    )
    print(
        "|---:|---:|---:|---:|---:|---:|---:|:---|---:|"
    )
    for year in years:
        bess_gw = DEFAULT_BESS_BUILDOUT.get(
            year,
            DEFAULT_BESS_BUILDOUT[max(k for k in DEFAULT_BESS_BUILDOUT if k <= year)],
        )
        old_t = legacy_by[year]["total"]
        new_t = new_by[year]["total"]
        delta = new_t - old_t
        eq = new_by[year]["equilibrium_type"]
        f = new_by[year]["f_on_as"]
        obs_s = f"{observed[year]:.0f}" if year in observed else "—"
        ch_val = ch.get(year)
        ch_s = f"{ch_val:.0f}" if ch_val is not None else "—"
        print(
            f"| {year} | {bess_gw:.1f} | {old_t:.1f} | {new_t:.1f} | "
            f"{delta:+.1f} | {obs_s} | {ch_s} | {eq} | {f:.2f} |"
        )

    print()
    # Two-year narrative summary:
    r_2026_old = legacy_by[2026]["total"]
    r_2030_old = legacy_by[2030]["total"]
    r_2026_new = new_by[2026]["total"]
    r_2030_new = new_by[2030]["total"]
    print(f"2026 → 2030 drawdown (old): {r_2030_old/r_2026_old:.2f}× "
          f"({r_2026_old:.0f} → {r_2030_old:.0f})")
    print(f"2026 → 2030 drawdown (new): {r_2030_new/r_2026_new:.2f}× "
          f"({r_2026_new:.0f} → {r_2030_new:.0f})")
    print()
    print("## Interpretation notes")
    print()
    print("**Historical years (2023-2025)**: project_full_stack under-predicts")
    print("because the equilibrium model splits fleet mutually-exclusively between")
    print("AS and wholesale (f=1 → zero WH contribution). In reality BESS operators")
    print("do both simultaneously on the same MW. Note 1 shows historical from")
    print("hist_bars (separate pipeline), not from project_full_stack, so the")
    print("displayed UI is unaffected. The 'observed (T2)' column is our internal")
    print("ground truth from Tier 1/2 sources — still undercounts CH by ~35% in")
    print("2023 because the aFRR↔wholesale conjugate coupling is not captured")
    print("here (stand-alone LP dispatch misses the resale value of energy")
    print("absorbed via NEG activations). This will be closed by the stacked-")
    print("market LP in Note 4 A2.")
    print()
    print("**Projection years (2026+)**: rework materially improves alignment.")
    print("2026 legacy 219 kEUR → new 144 kEUR vs CH early-2026 prints 110-129.")
    print("Simon's 'ancillary collapse' narrative muted (2026→2030 drawdown 19%")
    print("not 46%). CH included for context only — our projection isn't")
    print("calibrated to it. A proper benchmark-fan validation (LCP, enspired,")
    print("suena, RWTH) belongs in `benchmark-reconciliation`.")


if __name__ == "__main__":
    main()
