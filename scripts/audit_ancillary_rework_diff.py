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

from lib.config import DEFAULT_BESS_BUILDOUT
from lib.models.ancillary import ancillary_revenue
from lib.models.degradation import PRESETS, fleet_average_capacity
from lib.models.projection import project_full_stack, project_wholesale


def legacy_full_stack(
    years: list[int],
    historical_da_keur: float,
    bess_buildout: dict[int, float],
    duration_h: float = 2.0,
    **wholesale_kwargs,
) -> list[dict]:
    """Replica of the pre-rework additive-stacking projection."""
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
        anc = ancillary_revenue(year=year, bess_gw=bess_gw, duration_h=duration_h)
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
    years = list(range(2026, 2041))
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

    print("# Note 1 audit diff — AS/wholesale equilibrium rework")
    print()
    print(f"Baseline DA: {historical_da} kEUR/MW/yr | Duration: 2h | Default buildout")
    print()
    print(
        "| year | bess_gw | old total | new total | Δ | Δ% | eq_type | f_on_as |"
    )
    print(
        "|---:|---:|---:|---:|---:|---:|:---|---:|"
    )
    for year in years:
        bess_gw = DEFAULT_BESS_BUILDOUT.get(
            year,
            DEFAULT_BESS_BUILDOUT[max(k for k in DEFAULT_BESS_BUILDOUT if k <= year)],
        )
        old_t = legacy_by[year]["total"]
        new_t = new_by[year]["total"]
        delta = new_t - old_t
        pct = (delta / old_t * 100) if old_t else 0.0
        eq = new_by[year]["equilibrium_type"]
        f = new_by[year]["f_on_as"]
        print(
            f"| {year} | {bess_gw:.1f} | {old_t:.1f} | {new_t:.1f} | "
            f"{delta:+.1f} | {pct:+.1f}% | {eq} | {f:.2f} |"
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


if __name__ == "__main__":
    main()
