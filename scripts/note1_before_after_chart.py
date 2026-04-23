"""
Before/after stacked bar chart showing Note 1 impact from A1+A2 rework.

Side-by-side comparison for each year:
  * OLD: pre-rework stacking (proxy aFRR energy + additive AS+WH)
  * NEW: post-rework (real aFRR energy + AS/WH equilibrium)

Historical years (2023-2025): total stays CH-anchored; breakdown shifts.
Projection years (2026-2040): total changes via equilibrium; big swing
in 2026-2027 as the ancillary-collapse narrative softens.

Output: scripts/note1_before_after.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from lib.config import DEFAULT_BESS_BUILDOUT
from lib.data.ancillary_prices import (
    fetch_afrr_annual_revenue,
    fetch_fcr_annual_revenue,
)
from lib.data.clean_horizon import annual_average
from lib.data.day_ahead_prices import fetch_day_ahead_prices, prices_to_daily_arrays
from lib.models.ancillary import ancillary_revenue
from lib.models.degradation import PRESETS, fleet_average_capacity
from lib.models.dispatch import annual_revenue, dispatch_day
from lib.models.projection import (
    id_da_ratio,
    project_full_stack,
    project_wholesale,
)

HIST_YEARS = [2023, 2024, 2025]
PROJ_YEARS = [2026, 2027, 2028, 2029, 2030, 2032, 2035, 2040]
ALL_YEARS = HIST_YEARS + PROJ_YEARS

LABELS = ["da", "id", "fcr", "afrr_cap", "afrr_energy"]
COLORS = {
    "da": "#93c5fd",        # light blue
    "id": "#3b82f6",        # blue
    "fcr": "#fbbf24",       # yellow
    "afrr_cap": "#f87171",  # light red
    "afrr_energy": "#dc2626",  # red
}
PRETTY = {
    "da": "Day-Ahead", "id": "Intraday",
    "fcr": "FCR", "afrr_cap": "aFRR capacity", "afrr_energy": "aFRR energy",
}


def _historical_dispatch_da(year: int, duration_h: float) -> float:
    """DA revenue from LP dispatch on observed prices (kEUR/MW/yr)."""
    try:
        prices_df = fetch_day_ahead_prices(start=f"{year}-01-01", end=f"{year}-12-31")
    except Exception:
        return 80.0  # rough fallback if network is flaky
    daily = prices_to_daily_arrays(prices_df, resolution_minutes=60)
    year_key = str(year)
    if year_key not in daily:
        return 80.0
    results = [
        dispatch_day(p, duration_h=duration_h, rte=0.85, max_cycles=2)
        for p in daily[year_key]
    ]
    return annual_revenue(results) / 1000.0


def _historical_bars_for(year: int, duration_h: float, use_real_energy: bool) -> dict:
    """Build hist_bars-style row for a single year.

    Mirrors notes/de-bess-outlook/precompute.py::compute_hist_stack_ch. The
    total is anchored to CH via s = ch_total / model_total and components
    are rescaled. We compute both OLD (proxy aFRR energy) and NEW (real)
    by flipping the single flag.
    """
    ch_annual = annual_average(duration_h)
    ch_total = ch_annual.get(year)
    if ch_total is None:
        return {k: 0.0 for k in LABELS + ["total"]}
    da_rev = _historical_dispatch_da(year, duration_h)
    id_rev = da_rev * id_da_ratio(year)
    try:
        fcr = fetch_fcr_annual_revenue(year) or 0.0
    except Exception:
        fcr = 0.0
    afrr = fetch_afrr_annual_revenue(year, use_real_energy=use_real_energy)
    if afrr is None:
        afrr = {"afrr_cap": 0.0, "afrr_energy": 0.0}
    model_total = da_rev + id_rev + fcr + afrr["afrr_cap"] + afrr["afrr_energy"]
    if model_total <= 0:
        model_total = 1.0
    s = ch_total / model_total
    return {
        "da": da_rev * s,
        "id": id_rev * s,
        "fcr": fcr * s,
        "afrr_cap": afrr["afrr_cap"] * s,
        "afrr_energy": afrr["afrr_energy"] * s,
        "total": ch_total,
    }


def _legacy_projection_row(year: int, duration_h: float, historical_da_keur: float) -> dict:
    """Replica of pre-rework project_full_stack (additive stacking, proxy
    aFRR energy)."""
    bess_gw = DEFAULT_BESS_BUILDOUT.get(
        year, DEFAULT_BESS_BUILDOUT[max(k for k in DEFAULT_BESS_BUILDOUT if k <= year)]
    )
    wh = project_wholesale(year, historical_da_annual=historical_da_keur, bess_gw=bess_gw)
    anc = ancillary_revenue(
        year=year, bess_gw=bess_gw, duration_h=duration_h,
        use_historical_if_available=False,  # force legacy saturation
    )
    proj_buildout = {y: v for y, v in DEFAULT_BESS_BUILDOUT.items() if y >= 2026}
    deg = fleet_average_capacity(
        year=year, buildout=proj_buildout, preset=PRESETS["baseline_fleet"],
    )
    return {
        "da": float(wh["da"]) * deg,
        "id": float(wh["id"]) * deg,
        "fcr": float(anc["fcr"]) * deg,
        "afrr_cap": float(anc["afrr_cap"]) * deg,
        # Legacy aFRR energy value from the saturation model (proxy-era anchors)
        "afrr_energy": float(anc["afrr_energy"]) * deg,
        "total": (wh["wholesale_total"] + anc["total"]) * deg,
    }


def build_old_new_rows(duration_h: float = 2.0) -> tuple[dict, dict]:
    historical_da = 90.0
    old = {}
    new = {}
    for y in HIST_YEARS:
        old[y] = _historical_bars_for(y, duration_h, use_real_energy=False)
        new[y] = _historical_bars_for(y, duration_h, use_real_energy=True)
    proj_new = {
        r["year"]: r for r in project_full_stack(
            years=PROJ_YEARS, historical_da_keur=historical_da, duration_h=duration_h,
        )
    }
    for y in PROJ_YEARS:
        old[y] = _legacy_projection_row(y, duration_h, historical_da)
        new[y] = {k: proj_new[y].get(k, 0.0) for k in LABELS + ["total"]}
    return old, new


def plot_before_after(old: dict, new: dict, out_path: Path) -> None:
    """Side-by-side grouped stacked bars: OLD (left, hatched) | NEW (right)
    for each year."""
    fig, ax = plt.subplots(figsize=(16, 7))
    width = 0.38
    x = np.arange(len(ALL_YEARS))

    def _stack(ax, offsets, rows, hatch=None, alpha=1.0, label_prefix=""):
        bottoms_pos = np.zeros(len(ALL_YEARS))
        bottoms_neg = np.zeros(len(ALL_YEARS))
        for comp in LABELS:
            values = np.array([rows[y].get(comp, 0.0) for y in ALL_YEARS])
            pos = np.where(values > 0, values, 0)
            neg = np.where(values < 0, values, 0)
            ax.bar(offsets, pos, width, bottom=bottoms_pos,
                   color=COLORS[comp], hatch=hatch, alpha=alpha,
                   edgecolor="white", linewidth=0.6,
                   label=f"{label_prefix}{PRETTY[comp]}" if label_prefix == "" else None)
            ax.bar(offsets, neg, width, bottom=bottoms_neg,
                   color=COLORS[comp], hatch=hatch, alpha=alpha,
                   edgecolor="white", linewidth=0.6)
            bottoms_pos += pos
            bottoms_neg += neg

    _stack(ax, x - width / 2, old, hatch="///", alpha=0.75, label_prefix="")
    _stack(ax, x + width / 2, new, hatch=None, alpha=1.0, label_prefix="")

    # Totals text on top of each bar
    for i, y in enumerate(ALL_YEARS):
        t_old = old[y]["total"]
        t_new = new[y]["total"]
        ax.text(x[i] - width / 2, t_old + 5, f"{t_old:.0f}",
                ha="center", va="bottom", fontsize=8, color="#555")
        ax.text(x[i] + width / 2, t_new + 5, f"{t_new:.0f}",
                ha="center", va="bottom", fontsize=8, color="#000", weight="bold")

    ax.axhline(0, color="#666", linewidth=0.8)
    ax.axvline(len(HIST_YEARS) - 0.5, color="#999", linestyle=":", linewidth=1)
    ax.text(len(HIST_YEARS) / 2 - 0.5, ax.get_ylim()[1] * 0.95, "historical (CH-anchored)",
            ha="center", fontsize=9, style="italic", color="#666")
    ax.text((len(HIST_YEARS) + len(ALL_YEARS)) / 2 - 0.5,
            ax.get_ylim()[1] * 0.95, "projection",
            ha="center", fontsize=9, style="italic", color="#666")

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in ALL_YEARS])
    ax.set_ylabel("Revenue (kEUR / MW / yr)")
    ax.set_title(
        "Note 1 — before/after A1+A2 rework\n"
        "Left bar (hatched): OLD  |  Right bar: NEW  (2h BESS)",
        fontsize=11,
    )
    # Dedupe legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
    ax.legend([h for h, _ in uniq], [l for _, l in uniq],
              loc="upper right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")


def main() -> None:
    old, new = build_old_new_rows(duration_h=2.0)
    # Print the table for inspection
    print(f"{'year':>6} | {'old':>6} | {'new':>6} | Δ%")
    print("-" * 36)
    for y in ALL_YEARS:
        t_old, t_new = old[y]["total"], new[y]["total"]
        delta_pct = (t_new - t_old) / t_old * 100 if t_old else 0.0
        print(f"{y:>6} | {t_old:>6.0f} | {t_new:>6.0f} | {delta_pct:+5.1f}%")
    plot_before_after(old, new, Path(__file__).parent / "note1_before_after.png")


if __name__ == "__main__":
    main()
