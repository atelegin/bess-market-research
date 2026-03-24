"""
Generate static publication-quality charts.

Usage:  python notes/de-bess-outlook/generate_charts.py
Output: notes/de-bess-outlook/charts/*.png
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import datetime as dt

from lib.config import DEFAULT_BESS_BUILDOUT
from lib.models.projection import project_full_stack

# ── Config ────────────────────────────────────────────────────
DURATION = 2.0
BESS_2040 = 40
GAS_2040 = 30
PV_2040 = 300
DEMAND_2040 = 1020

CANIB_SCENARIOS = {
    "Low (optimistic)":    {"canib_max": 15.0, "canib_half": 25.0, "canib_steep": 0.80},
    "Mid (merit order)":   {"canib_max": 33.0, "canib_half": 13.0, "canib_steep": 0.17},
    "High (CAISO-scaled)": {"canib_max": 82.0, "canib_half": 11.0, "canib_steep": 0.18},
}

PRECOMPUTED_PATH = Path(__file__).parent / "data" / "precomputed.pkl"
OUTPUT_DIR = Path(__file__).parent / "charts"

COMMITTED_THROUGH = 2027

# ── Style ─────────────────────────────────────────────────────
FONT_FAMILY = "Source Serif 4"
BG_COLOR = "#faf9f6"
TEXT_COLOR = "#14213d"
MUTED = "#5c677d"

STACK_KEYS = ["da", "id", "fcr", "afrr_cap", "afrr_energy"]
LABELS = {"da": "Day-Ahead", "id": "Intraday", "fcr": "FCR",
          "afrr_cap": "aFRR capacity", "afrr_energy": "aFRR energy"}
COLORS = {"da": "#93c5fd", "id": "#3b82f6", "fcr": "#fbbf24",
          "afrr_cap": "#f87171", "afrr_energy": "#dc2626"}

import matplotlib.font_manager as fm

# Register bundled fonts from lib/fonts/
_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "lib" / "fonts"
for _ttf in _FONT_DIR.glob("*.ttf"):
    fm.fontManager.addfont(str(_ttf))

_TITLE_FONT = "Source Serif 4"
_BODY_FONT = "IBM Plex Sans"

# FontProperties objects for direct use (bypasses weight matching issues)
_TITLE_FP = fm.FontProperties(fname=str(_FONT_DIR / "SourceSerif4.ttf"), weight="bold")
_BODY_FP = fm.FontProperties(fname=str(_FONT_DIR / "IBMPlexSans-Regular.ttf"))
_BODY_BOLD_FP = fm.FontProperties(fname=str(_FONT_DIR / "IBMPlexSans-Bold.ttf"))

# Verify registration, fall back if missing
_available = {f.name for f in fm.fontManager.ttflist}
if _TITLE_FONT not in _available:
    _TITLE_FONT = "Georgia"
if _BODY_FONT not in _available:
    _BODY_FONT = "Helvetica Neue"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_BODY_FONT, "Helvetica Neue", "Arial"],
    "font.size": 11,
    "axes.facecolor": BG_COLOR,
    "figure.facecolor": BG_COLOR,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT_COLOR,
    "axes.titlepad": 12,
})


def _scale_buildout(target_2040: float) -> dict[int, float]:
    committed_val = DEFAULT_BESS_BUILDOUT[COMMITTED_THROUGH]
    default_2040 = DEFAULT_BESS_BUILDOUT[2040]
    out = {}
    for y, v in DEFAULT_BESS_BUILDOUT.items():
        if y <= COMMITTED_THROUGH:
            out[y] = v
        else:
            if default_2040 == committed_val:
                out[y] = committed_val
            else:
                frac = (v - committed_val) / (default_2040 - committed_val)
                out[y] = committed_val + frac * (target_2040 - committed_val)
    out[2040] = target_2040
    return out


def _mute(hex_color: str, factor: float = 0.55) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    grey = 200
    return f"#{int(r*factor+grey*(1-factor)):02x}{int(g*factor+grey*(1-factor)):02x}{int(b*factor+grey*(1-factor)):02x}"


# ── Load data ─────────────────────────────────────────────────
with open(PRECOMPUTED_PATH, "rb") as f:
    precomputed = pickle.load(f)

dur_key = f"{DURATION:.0f}h"
dur_data = precomputed[dur_key]
baseline_da = dur_data["baseline_da"]
hist_bars = dur_data["hist_bars"]
ch_all = dur_data["ch_all"]

user_buildout = _scale_buildout(BESS_2040)
proj_years = list(range(2026, 2041))

proj = project_full_stack(
    years=proj_years,
    historical_da_keur=baseline_da,
    bess_buildout=user_buildout,
    duration_h=DURATION,
    demand_2040_twh=float(DEMAND_2040),
    gas_2040=float(GAS_2040),
    pv_2040_gw=float(PV_2040),
    **CANIB_SCENARIOS["Mid (merit order)"],
)
proj_df = pd.DataFrame(proj)
if hist_bars:
    hist_df = pd.DataFrame(hist_bars)
    proj_df = pd.concat([hist_df, proj_df], ignore_index=True)

hist_year_set = set(dur_data["hist_rev"].keys())

# Bull/Bear
_BULL = dict(bess_2040=20, gas_2040=80, demand_2040=1200, pv_2040=215,
             **CANIB_SCENARIOS["Low (optimistic)"])
_BEAR = dict(bess_2040=60, gas_2040=15, demand_2040=700, pv_2040=400,
             **CANIB_SCENARIOS["High (CAISO-scaled)"])

bands = {}
for label, params in [("Bull", _BULL), ("Bear", _BEAR)]:
    bo = _scale_buildout(params["bess_2040"])
    rows = project_full_stack(
        years=proj_years, historical_da_keur=baseline_da,
        bess_buildout=bo, duration_h=DURATION,
        demand_2040_twh=float(params["demand_2040"]),
        gas_2040=float(params["gas_2040"]),
        pv_2040_gw=float(params["pv_2040"]),
        canib_max=params["canib_max"],
        canib_half=params["canib_half"],
        canib_steep=params["canib_steep"],
    )
    bands[label] = pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# CHART 1: Main revenue stack with KPI annotations
# ══════════════════════════════════════════════════════════════
def chart_main_revenue():
    fig, ax = plt.subplots(figsize=(12, 7))

    years = proj_df["year"].values
    bar_width = 0.7

    # Stacked bars
    bottom = np.zeros(len(years))
    for key in STACK_KEYS:
        colors = [_mute(COLORS[key]) if y in hist_year_set else COLORS[key] for y in years]
        ax.bar(years, proj_df[key].values, bar_width, bottom=bottom,
               color=colors, label=LABELS[key], edgecolor="none")
        bottom += proj_df[key].values

    # Bull/Bear band
    ax.fill_between(bands["Bear"]["year"], bands["Bear"]["total"],
                    bands["Bull"]["total"], alpha=0.06, color=TEXT_COLOR, label="_nolegend_")
    ax.plot(bands["Bear"]["year"], bands["Bear"]["total"],
            color="#ef4444", linewidth=1, linestyle=":", label="Bear case")
    ax.plot(bands["Bull"]["year"], bands["Bull"]["total"],
            color="#22c55e", linewidth=1, linestyle=":", label="Bull case")

    # Total line (projection only)
    proj_mask = ~proj_df["year"].isin(hist_year_set)
    ax.plot(proj_df.loc[proj_mask, "year"], proj_df.loc[proj_mask, "total"],
            color=TEXT_COLOR, linewidth=1.5, linestyle=":", marker=".", markersize=3,
            label="_nolegend_")

    # Actual/forecast divider
    if hist_year_set:
        boundary = max(hist_year_set) + 0.5
        ax.axvline(boundary, color=MUTED, linewidth=0.8, linestyle=":")
        ax.text(boundary - 0.15, 1.01, "actual ", transform=ax.get_xaxis_transform(),
                ha="right", va="bottom", fontsize=8, color=MUTED)
        ax.text(boundary + 0.15, 1.01, " forecast", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=8, color=MUTED)

    # KPI annotations at top
    kpi_years = {2026: "2026", 2030: "2030", 2040: "2040"}
    # Find floor
    proj_only = proj_df[proj_df["year"].isin(proj_years)]
    floor_year = int(proj_only.loc[proj_only["total"].idxmin(), "year"])
    kpi_years[floor_year] = f"{floor_year} floor"

    y_top = max(proj_df["total"].max(), bands["Bull"]["total"].max()) * 1.08
    ax.set_ylim(0, y_top)

    # Place KPIs above chart — with enough gap below for actual/forecast label
    kpi_sorted = sorted(kpi_years.items())
    for i, (y, label) in enumerate(kpi_sorted):
        row = proj_df[proj_df["year"] == y].iloc[0]
        x_frac = 0.02 + i * 0.24
        ax.text(x_frac, 1.22, label, transform=ax.transAxes,
                fontsize=12, color=MUTED, fontweight="normal", ha="left",
                fontproperties=_BODY_FP)
        ax.text(x_frac, 1.10, f"€{row['total']:.0f}k/MW", transform=ax.transAxes,
                fontsize=20, color=TEXT_COLOR, ha="left",
                fontproperties=_BODY_BOLD_FP)

    ax.set_ylabel("€k / MW / year", fontsize=11)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.96)
    return fig


# ══════════════════════════════════════════════════════════════
# CHART 2: Cannibalisation scenario fan
# ══════════════════════════════════════════════════════════════
def chart_cannibalisation_fan():
    fig, ax = plt.subplots(figsize=(12, 5))

    scen_colors = {
        "Low (optimistic)": "#22c55e",
        "Mid (merit order)": "#eab308",
        "High (CAISO-scaled)": "#ef4444",
    }
    scen_labels = {
        "Low (optimistic)": "Low (optimistic)",
        "Mid (merit order)": "Mid (merit order)",
        "High (CAISO-scaled)": "High (CAISO-scaled)",
    }

    scen_data = {}
    for sname, sparams in CANIB_SCENARIOS.items():
        rows = project_full_stack(
            years=proj_years, historical_da_keur=baseline_da,
            bess_buildout=user_buildout, duration_h=DURATION,
            demand_2040_twh=float(DEMAND_2040), gas_2040=float(GAS_2040),
            pv_2040_gw=float(PV_2040), **sparams,
        )
        scen_data[sname] = pd.DataFrame(rows)

    # Shaded band
    low_df = scen_data["Low (optimistic)"]
    high_df = scen_data["High (CAISO-scaled)"]
    ax.fill_between(low_df["year"], high_df["total"], low_df["total"],
                    alpha=0.08, color=TEXT_COLOR)

    for sname, sdf in scen_data.items():
        is_mid = sname == "Mid (merit order)"
        ax.plot(sdf["year"], sdf["total"], color=scen_colors[sname],
                linewidth=2.5 if is_mid else 1.2,
                marker="o" if is_mid else None, markersize=4,
                alpha=1.0 if is_mid else 0.5,
                label=scen_labels[sname])
        # End-of-line label
        last = sdf.iloc[-1]
        ax.text(last["year"] + 0.3, last["total"],
                f"€{last['total']:.0f}k", fontsize=9, color=scen_colors[sname],
                va="center")

    # 10 GW marker
    yr_10gw = None
    for y in sorted(user_buildout.keys()):
        if user_buildout[y] >= 10:
            yr_10gw = y
            break
    if yr_10gw:
        ax.axvline(yr_10gw, color=MUTED, linewidth=1, linestyle=":")
        ax.text(yr_10gw, ax.get_ylim()[1] * 0.03, f"~10 GW ({yr_10gw})",
                ha="center", va="bottom", fontsize=9, color=MUTED)

    ax.set_ylabel("€k / MW / year", fontsize=11)
    ax.set_xlabel("")
    ax.set_xticks(proj_years)
    ax.set_xticklabels(proj_years, rotation=45, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.set_title("Revenue range by cannibalisation assumption",
                 fontsize=13, color=TEXT_COLOR, pad=12, fontproperties=_TITLE_FP)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# CHART 3: Fleet buildout
# ══════════════════════════════════════════════════════════════
def chart_fleet_buildout():
    fig, ax = plt.subplots(figsize=(12, 5))

    # Historical quarterly
    hist_quarterly = [
        (dt.date(2022, 1, 1), 0.55), (dt.date(2022, 4, 1), 0.60),
        (dt.date(2022, 7, 1), 0.70), (dt.date(2022, 10, 1), 0.80),
        (dt.date(2023, 1, 1), 0.90), (dt.date(2023, 4, 1), 1.00),
        (dt.date(2023, 7, 1), 1.15), (dt.date(2023, 10, 1), 1.30),
        (dt.date(2024, 1, 1), 1.45), (dt.date(2024, 4, 1), 1.60),
        (dt.date(2024, 7, 1), 1.80), (dt.date(2024, 10, 1), 2.05),
        (dt.date(2025, 1, 1), 2.20), (dt.date(2025, 4, 1), 2.35),
        (dt.date(2025, 7, 1), 2.40), (dt.date(2025, 10, 1), 2.45),
        (dt.date(2026, 1, 1), 2.80),
    ]
    hist_dates = [d for d, _ in hist_quarterly]
    hist_vals = [v for _, v in hist_quarterly]

    ax.plot(hist_dates, hist_vals, color=TEXT_COLOR, linewidth=2,
            marker="o", markersize=4, label="Installed (MaStR)")

    last_hist_date, last_hist_gw = hist_quarterly[-1]

    def quarterly_trajectory(buildout):
        anchors = [(last_hist_date, last_hist_gw)]
        for y in sorted(buildout):
            if y < 2026:
                continue
            anchors.append((dt.date(y, 12, 31), buildout[y]))
        pts = []
        for i in range(len(anchors) - 1):
            d0, v0 = anchors[i]
            d1, v1 = anchors[i + 1]
            span = (d1 - d0).days
            d = d0
            while d <= d1:
                frac = (d - d0).days / span if span > 0 else 0
                pts.append((d, v0 + (v1 - v0) * frac))
                m = d.month + 3
                y = d.year + (m - 1) // 12
                m = (m - 1) % 12 + 1
                d = dt.date(y, m, 1)
        if pts[-1][0] != anchors[-1][0]:
            pts.append(anchors[-1])
        return pts

    bo_colors = {"20 GW": "#22c55e", "40 GW": "#eab308", "60 GW": "#ef4444"}
    for label, target in {"20 GW": 20, "40 GW": 40, "60 GW": 60}.items():
        bo = _scale_buildout(target)
        pts = quarterly_trajectory(bo)
        is_default = target == BESS_2040
        ax.plot([d for d, _ in pts], [v for _, v in pts],
                color=bo_colors[label],
                linewidth=2.5 if is_default else 1.2,
                linestyle="solid" if is_default else "--",
                alpha=1.0 if is_default else 0.4,
                label=label)
        # End label
        ax.text(pts[-1][0] + dt.timedelta(days=90), pts[-1][1],
                f"{target} GW", fontsize=10, color=bo_colors[label],
                va="center", fontweight="bold" if is_default else "normal")

    # Shaded band
    pts_20 = quarterly_trajectory(_scale_buildout(20))
    pts_60 = quarterly_trajectory(_scale_buildout(60))
    common_dates = [d for d, _ in pts_20 if d in {d2 for d2, _ in pts_60}]
    vals_20 = {d: v for d, v in pts_20}
    vals_60 = {d: v for d, v in pts_60}
    if common_dates:
        ax.fill_between(common_dates, [vals_20[d] for d in common_dates],
                        [vals_60[d] for d in common_dates],
                        alpha=0.06, color=TEXT_COLOR)

    # Committed line
    committed_date = dt.date(COMMITTED_THROUGH, 1, 1)
    ax.axvline(committed_date, color=MUTED, linewidth=1, linestyle=":")
    ax.text(committed_date, ax.get_ylim()[1] * 0.03,
            f"  Committed ({COMMITTED_THROUGH})",
            ha="left", va="bottom", fontsize=9, color=MUTED)

    ax.set_ylabel("GW installed", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.8)
    ax.set_title("BESS fleet buildout scenarios",
                 fontsize=13, color=TEXT_COLOR, pad=12, fontproperties=_TITLE_FP)

    fig.tight_layout()
    return fig


# ── Generate all ──────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    charts = [
        ("01_revenue_outlook.png", chart_main_revenue),
        ("02_cannibalisation_fan.png", chart_cannibalisation_fan),
        ("03_fleet_buildout.png", chart_fleet_buildout),
    ]

    for fname, fn in charts:
        print(f"Generating {fname}...")
        fig = fn()
        fig.text(0.98, 0.01, "\u00a9 2026 Anton Telegin \u00b7 BESS Market Notes",
                 ha="right", va="bottom", fontsize=8, color="#999999")
        fig.savefig(OUTPUT_DIR / fname, dpi=200, bbox_inches="tight",
                    facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)
        print(f"  Saved to {OUTPUT_DIR / fname}")

    print("\nDone.")
