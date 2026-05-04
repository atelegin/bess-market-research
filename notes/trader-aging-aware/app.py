"""
Note 4 — Cost of a Cycle, or Is Your Optimiser Aging-Aware?

Single pkl drives headline bar chart + diagnostic dispatch logs +
ablation footnote: `data/precomputed_v42_5methods_10y.pkl`. v4.2
calibration (v4.0 B + FCR phantom + SIDC IDA2 intraday + Stage-2
pass-2 wear refinement + `kernel_scale=0.66`). Single template year
2025 (every simulation year sees full IDA2 spreads — no fallback to
Spotmarktpreis-as-DA-proxy). Y1 = 2025 captured for diagnostic
dispatch logs.

Five publication methodologies keyed M1-M5 directly:
    M1_naive    — no shadow cost
    M2_flat     — flat €/MWh wear (Kumtepeli proxy)
    M3_scarcity — flat × SoH-state scarcity
    M4_physics  — ADP + physics-from-duty (Stage-2 pass-2 refined)
    M5_adp      — pure ADP, the empirical winner at +36.5 % vs M1

Two ablation variants for "Why blending doesn't help" footnote in
the same pkl under separate keys:
    ablation_M5_with_classical_stack — flat + scarcity + ADP additive
    ablation_physics_without_adp     — physics alone, no ADP base
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.shared.theme import (
    apply_theme,
    render_chart_caption,
    render_chart_title,
    render_closing,
    render_footer,
    render_header,
    render_takeaway,
)


DATA_DIR = Path(__file__).parent / "data"
# Headline + diagnostic dispatch logs + ablation footnote — all from a
# single v4.2 pkl: v4.0 B + FCR phantom + SIDC IDA2 wholesale-ID feed
# + Stage-2 pass-2 wear refinement + kernel_scale=0.66. Single template
# year 2025 throughout the 10-year horizon (every simulation year sees
# full IDA2 spreads). Under this calibration the M1 → M5 stack orders
# monotonically (M5 > M4 > M3 > M2 > M1) at +36.5 % NPV uplift at the
# M5 winner. Each Mn shows ONE shadow-cost form; M1-M3 are the
# classical depreciation-proxy family (no shadow → throughput-only →
# SoH-aware), M4-M5 are the state-aware family (physics-from-duty
# observed dispatch → ADP opportunity-cost gradient). Empirical winner
# is M5 (pure ADP); the ablation footnote shows that adding physics
# refinement on top of M5 (= M4) or stacking the M2/M3 classical
# proxy under M5 both hurt by 1-4 pp — best shadow cost is one
# shadow cost, chosen well.
PRECOMPUTED_HEADLINE = DATA_DIR / "precomputed_v42_5methods_10y.pkl"


# Five shadow-cost methodologies. M1-M3 are classical depreciation-proxy
# family (no shadow → throughput-only → SoH-aware multiplier). M4-M5 are
# state-aware family (physics-from-duty → ADP opportunity gradient).
# Each Mn shows ONE methodology, not a stack — the labels intentionally
# avoid "+" framing that would imply progressive layering.
POLICY_LABELS = {
    "M1_naive":     "M1 — No cycle cost",
    "M2_flat":      "M2 — Fixed fee",
    "M3_scarcity":  "M3 — Health-aware",
    "M4_physics":   "M4 — Cycle-shape-aware",
    "M5_adp":       "M5 — Time-and-state-aware",
}
POLICY_SHORT = {
    "M1_naive":     "M1 No-cost",
    "M2_flat":      "M2 Fixed-fee",
    "M3_scarcity":  "M3 Health",
    "M4_physics":   "M4 Shape",
    "M5_adp":       "M5 Time-state",
}
# M1 sits in warm terra-coral as the unconstrained-baseline reference —
# visually orthogonal to the teal stack so it reads as the "odd one out"
# at a glance. M2 → M5 progress through a single-hue tint gradient
# anchored on a deep-teal base.
POLICY_COLORS = {
    "M1_naive":     "#e07a5f",
    "M2_flat":      "#b4d2d8",
    "M3_scarcity":  "#6fa3b0",
    "M4_physics":   "#3d7a8a",
    "M5_adp":       "#0d3f4a",
}
POLICY_ORDER = [
    "M1_naive", "M2_flat", "M3_scarcity", "M4_physics", "M5_adp",
]


def _eol_year_1indexed(years_to_floor, n_years):
    if years_to_floor is None:
        return n_years
    return int(years_to_floor) + 1


@st.cache_data(show_spinner=False)
def _load_pkl_with_mtime(path_str: str, mtime: float):
    with open(path_str, "rb") as f:
        return pickle.load(f)


def _load_pkl(path_str: str):
    # Cache key includes mtime so a regenerated pkl invalidates the cache
    # without needing to restart the Streamlit server.
    return _load_pkl_with_mtime(path_str, Path(path_str).stat().st_mtime)


st.set_page_config(
    page_title="Cost of a Cycle: Is Your Optimiser Aging-Aware?",
    page_icon="🔋",
    layout="wide",
)
apply_theme(show_sidebar=False)


# ── Header ──────────────────────────────────────────────────
render_header(
    title="Cost of a Cycle: Is Your Optimiser Aging-Aware?",
    kicker="GERMAN BESS | COST OF A CYCLE",
    subtitle="Every cycle wears the battery. How the optimiser charges for that wear — or whether it does at all — sets the trade-off between early-life and lifetime revenue.",
)

# ── Intro ───────────────────────────────────────────────────
st.markdown("""
What should the wear fee depend on? A flat €/MWh per cycle, the
battery's age, or the shape of today's dispatch — how deep each
cycle goes, how fast, and which part of the 0–100 % range it sits
in? The optimiser then skips any trade whose spread doesn't cover
the fee.

This note runs five answers in parallel. All see the same prices
and activations; only the cycle cost differs.
""")

# ── Load data ───────────────────────────────────────────────
# Single pkl drives headline + diagnostics + ablation footnote.
data = _load_pkl(str(PRECOMPUTED_HEADLINE))
results = data["results"]
n_years = data["n_years"]
ablation_results = results  # Same pkl; ablation_* keys live alongside M1-M5


# ── Headline references ─────────────────────────────────────
naive = results["M1_naive"]
winner = results["M5_adp"]

eol_naive = _eol_year_1indexed(naive.years_to_floor, n_years)
eol_winner = _eol_year_1indexed(winner.years_to_floor, n_years)
fec_naive = int(naive.annual_fec.sum())
fec_winner = int(winner.annual_fec.sum())


# ── Main chart ──────────────────────────────────────────────
render_chart_title(
    "Five ways to price a cycle, ranked by 10-year discounted "
    "lifetime revenue (sum of revenue over the asset's life, future "
    "euros discounted at 7 %/yr). Going right, each policy looks at "
    "more of the battery's state — from no cycle cost at all to a "
    "full time-and-state price."
)

npv_rows = []
naive_npv = naive.lifetime_npv_eur
for name in POLICY_ORDER:
    if name not in results:
        continue
    r = results[name]
    npv_rows.append({
        "policy": POLICY_LABELS[name],
        "index": r.lifetime_npv_eur / naive_npv * 100,
        "pct_vs_naive": (r.lifetime_npv_eur / naive_npv - 1) * 100,
        "color": POLICY_COLORS[name],
        "eol_year": _eol_year_1indexed(r.years_to_floor, n_years),
        "total_fec": int(r.annual_fec.sum()),
    })
npv_df = pd.DataFrame(npv_rows)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=npv_df["policy"], y=npv_df["index"],
    marker_color=npv_df["color"],
    text=[f"{p:+.1f}%" for p in npv_df["pct_vs_naive"]],
    textposition="outside",
    hovertemplate=(
        "%{x}<br>Index: %{y:.1f} (M1 = 100)<br>"
        "%{customdata[0]:+.1f}% vs M1<br>"
        "EOL: year %{customdata[1]}<br>"
        "Lifetime FEC: %{customdata[2]:,}"
        "<extra></extra>"
    ),
    customdata=np.stack([
        npv_df["pct_vs_naive"], npv_df["eol_year"], npv_df["total_fec"],
    ], axis=-1),
))
fig.add_hline(y=100, line_dash="dot", line_color="#94a3b8", line_width=1,
              annotation_text="M1 baseline", annotation_position="top left")
fig.update_layout(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=440, margin=dict(l=40, r=20, t=40, b=80),
    yaxis=dict(title="Lifetime revenue (M1 = 100)"),
    xaxis=dict(tickfont=dict(size=10)),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
render_chart_caption(
    "Ten-year simulation, 2 h LFP battery, 1 MW / 2 MWh, 2025 German "
    "markets replayed every year. All five policies see the same "
    "prices and activation signals; only the cycle cost differs. "
    "Bars indexed to M1 = 100. The ranking and ~+36 % M1 → M5 spread "
    "are robust to modelling choices. The section further down — "
    "<i>Does the ranking survive when ancillary revenue collapses?</i> — "
    "shows how it changes once wholesale grows and aFRR cap shrinks."
)


# ── The five methodologies, plain language ──────────────────
st.markdown("---")
st.markdown("## The five cycle-cost methodologies")
_m2 = results["M2_flat"]
_m3 = results["M3_scarcity"]
_m4 = results["M4_physics"]
_m5 = results["M5_adp"]
_abl_blend = ablation_results.get("ablation_M5_with_classical_stack")
_abl_solo = ablation_results.get("ablation_physics_without_adp")
st.markdown(f"""
Each policy uses a richer view of the battery's state: nothing →
fixed fee → health → cycle shape → time and state. Full formulas
are in the methodology expander at the bottom.

**M1 — No cycle cost.** Trade every profitable spread, no penalty
for wear. The simplest baseline: most cycles, shortest battery
life. **{fec_naive:,} full-equivalent cycles** (FEC — one full
charge plus one full discharge of the nameplate energy) lifetime;
battery hits the 80 % warranty floor in year {eol_naive}; lifetime
revenue index = 100 (by definition).

**M2 — Fixed fee.** Subtract a flat €/MWh wear charge from every
cycle — CAPEX divided by expected lifetime throughput
(≈ €16.67 / MWh here). Cycling drops to
**{int(_m2.annual_fec.sum()):,} FEC**; revenue
**+{(_m2.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.1f}%**
vs M1 — the single biggest jump in the stack.

**M3 — Health-aware.** Same fee, but scaled up as the cell ages
(1× fresh → 2× at the warranty floor). Adds
**+{(_m3.lifetime_npv_eur / _m2.lifetime_npv_eur - 1) * 100:.1f} pp**
over M2 by suppressing cycling once the battery is already worn.

**M4 — Cycle-shape-aware.** The optimiser solves the day twice:
a first pass dispatches with a rough wear estimate; that schedule
is fed through the
[degradation physics](https://bess-degradation-drivers.streamlit.app/)
to price the *actual* wear of that specific dispatch shape — depth,
C-rate and where in the 0–100 % range the cycle sits; the second
pass re-solves the intraday with the refined per-MWh wear. Adds
**+{(_m4.lifetime_npv_eur / _m3.lifetime_npv_eur - 1) * 100:.1f} pp**
over M3.

**M5 — Time-and-state-aware.** The wear cost now varies hour by
hour and with the *market regime* — *volatile* days (fat spreads
worth chasing) get a low cost so the battery cycles freely, *calm*
days (tight spreads, cycling isn't worth much) get a high cost so
the battery sits. At 18:00 on a volatile day one MWh in the cell is
worth more than at 03:00 on a calm one. Cycles drop to **{int(_m5.annual_fec.sum()):,} FEC**
lifetime — about {(fec_naive / max(int(_m5.annual_fec.sum()), 1)):.0f}× fewer
than M1. Revenue
**+{(_m5.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.1f}%**
vs M1 — peak of the stack.
""", unsafe_allow_html=True)


# ── Annual revenue + SoH trajectories ───────────────────────
st.markdown("---")
st.markdown("### Lifetime revenue comes from cycling less")
render_chart_title(
    "M1 earns most early, then hits the warranty floor — aging-aware "
    "policies extend life"
)

col_rev, col_soh = st.columns(2)

with col_rev:
    fig_rev = go.Figure()
    discount_factors = np.array([
        1.0 / (1.0 + 0.07) ** y for y in range(n_years)
    ])
    for name in POLICY_ORDER:
        r = results[name]
        annual_disc = r.annual_revenue_eur * discount_factors / 1000.0
        fig_rev.add_trace(go.Scatter(
            x=np.arange(1, n_years + 1),
            y=annual_disc,
            mode="lines+markers",
            name=POLICY_SHORT[name],
            line=dict(color=POLICY_COLORS[name], width=2.5),
        ))
    fig_rev.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=360, margin=dict(l=40, r=20, t=20, b=50),
        xaxis=dict(title="Year"),
        yaxis=dict(title="Annual discounted revenue (k€/MW)"),
        legend=dict(orientation="h", y=-0.20, font=dict(size=10)),
    )
    st.plotly_chart(fig_rev, use_container_width=True,
                    config={"displayModeBar": False})

with col_soh:
    fig_soh = go.Figure()
    floor = 0.80
    for name in POLICY_ORDER:
        r = results[name]
        soh_full = np.concatenate([[r.initial_soh], r.end_of_year_soh])
        # Truncate display at warranty floor — past EOL the simulator
        # ceases dispatch and the SoH series flatlines at the floor; in
        # reality the cell would continue degrading if operated, but we
        # don't model post-warranty operation. Hide the flat tail.
        soh_display = soh_full.copy()
        # Find first index where SoH <= floor; keep that point but mask
        # everything strictly past it.
        below = np.where(soh_full <= floor)[0]
        if len(below) > 0:
            cut = below[0]
            soh_display[cut + 1:] = np.nan
        fig_soh.add_trace(go.Scatter(
            x=np.arange(0, n_years + 1),
            y=soh_display,
            mode="lines+markers",
            name=POLICY_SHORT[name],
            line=dict(color=POLICY_COLORS[name], width=2),
            connectgaps=False,
        ))
    fig_soh.add_hline(y=0.80, line_dash="dot", line_color="#dc2626",
                     annotation_text="warranty floor", annotation_position="bottom right")
    fig_soh.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=360, margin=dict(l=40, r=20, t=20, b=50),
        xaxis=dict(title="Year"),
        yaxis=dict(title="SoH (fraction)", range=[0.75, 1.005]),
        legend=dict(orientation="h", y=-0.20, font=dict(size=10)),
    )
    st.plotly_chart(fig_soh, use_container_width=True,
                    config={"displayModeBar": False})

render_chart_caption(
    f"M1 (no cycle cost) earns the most while the battery is fresh, "
    f"then hits the 80 % warranty floor in year {eol_naive} and "
    f"stops earning. M2 and M3 cycle less as the battery ages, "
    f"extending its life by ~2 years. M4 and M5 stay above the "
    f"floor for the full 10 years."
)


# ── Reading aging-awareness off the dashboard (merged) ─────
st.markdown("---")
st.markdown("## How aging-awareness shows up on your optimiser's dashboard")

# 4-tile dashboard: M1 naive vs M5 ADP (the empirical winner) on real
# 2026 monthly dispatch (anchor_2026-{MM}_v42.pkl, from
# precompute_anchor_v42.py — same calibration as the headline 10-year
# run, just one recent month at fresh cell). Pkl shape: dict with
# year/month/calibration/results, results[name] = {'daily_logs': [...]}.
# Radio selector lets the reader compare winter (Jan/Feb — thin spreads,
# ID slice empty, optimiser skips intraday) vs spring (Mar/Apr — fat
# spreads, ID returns) seasonality.
_ANCHOR_MONTHS = {
    "Jan 2026": ("anchor_2026-01_v42.pkl", "January 2026"),
    "Feb 2026": ("anchor_2026-02_v42.pkl", "February 2026"),
    "Mar 2026": ("anchor_2026-03_v42.pkl", "March 2026"),
    "Apr 2026": ("anchor_2026-04_v42.pkl", "April 2026"),
}
# Read selected month from session state BEFORE the radio renders, so
# the intro/core-tension paragraphs above the chart panels can reference
# `month_label` even though the radio itself is rendered further down
# (right above Panel 1, where it sits next to the visuals it controls).
_selected_month_short = st.session_state.get("anchor_month_radio", "Apr 2026")
_anchor_filename, month_label = _ANCHOR_MONTHS[_selected_month_short]
_anchor_path = DATA_DIR / _anchor_filename
_anchor = _load_pkl(str(_anchor_path))

if _anchor and "M1_naive" in _anchor["results"] and "M5_adp" in _anchor["results"]:
    st.markdown("""
Most BESS optimisers ship the asset owner a live monitoring portal
with similar tiles across vendors: cycles per day, state-of-charge
over time, revenue split by market. Each tile carries a clue about
the cycle-cost policy.
""")

    _uplift_pct = (winner.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100

    # Selected-month gross per policy — used inline in the "core
    # tension" paragraph to anchor the monthly-revenue gap with concrete
    # numbers (changes with the radio selector above).
    def _march_gross_keur(name: str) -> float:
        return sum(
            sum(d.revenue_breakdown.values())
            for d in _anchor["results"][name]["daily_logs"]
        ) / 1000.0

    _m1_mar = _march_gross_keur("M1_naive")
    _m4_mar = _march_gross_keur("M4_physics")
    _m5_mar = _march_gross_keur("M5_adp")

    st.markdown(f"""
**The core tension.** Aging-aware policies (M4, M5) park the asset
on aFRR capacity and rarely trade wholesale, shifting revenue from
cycling-paid to availability-paid. That looks weak on monthly
revenue while the battery is fresh: in {month_label} M4 and M5 earned
**{(_m4_mar/_m1_mar - 1)*100:.0f} %** and
**{(_m5_mar/_m1_mar - 1)*100:.0f} %** less than M1
(€{_m4_mar:.1f} k and €{_m5_mar:.1f} k vs €{_m1_mar:.1f} k).
Yet over 10 years M5 wins **+{_uplift_pct:.0f}%** in discounted
revenue by avoiding the warranty-floor cliff that ends M1 in year
{eol_naive}. You can't have both. Either benchmark the optimiser on
lifetime revenue, or accept the early-life shortfall as the price of
battery longevity. The four panels below show how that trade-off
reads on your portal.
""")
    st.markdown("")
    st.markdown(
        f"M1 (no cycle cost) vs M5 (time-and-state-aware) on four "
        f"standard dashboard tiles — *Daily Revenue per Market*, "
        f"*Revenue Share per Market*, *State of Charge Development*, "
        f"*Daily Cycles*. Same calibration as the headline (2 h LFP, "
        f"1 MW / 2 MWh, **2-cycle/day cap**); selected month: "
        f"**{month_label}**. Switch the month below to compare "
        f"winter and spring."
    )

    # Radio selector sits right above the chart panels — close to the
    # visuals it controls. Session-state value is read above so the
    # intro/core-tension paragraphs already reflect the selected month.
    st.radio(
        "Compare across recent months",
        options=list(_ANCHOR_MONTHS.keys()),
        index=list(_ANCHOR_MONTHS.keys()).index(_selected_month_short),
        horizontal=True,
        key="anchor_month_radio",
    )

    m1_year_days = list(_anchor["results"]["M1_naive"]["daily_logs"])
    m5_year_days = list(_anchor["results"]["M5_adp"]["daily_logs"])

    NAIVE_C = "#e07a5f"
    AGING_C = "#0d3f4a"
    NAIVE_FILL = "rgba(224, 122, 95, 0.2)"
    AGING_FILL = "rgba(13, 63, 74, 0.2)"

    # Stream palette — orthogonal to the teal+coral policy palette so
    # the reader doesn't visually confuse a stream with a policy:
    # wholesale streams (DA, ID) sit in warm yellow/amber; ancillary
    # capacity products (aFRR cap, aFRR energy) sit in cool indigo.
    STREAM_COLORS = {
        "DA": "#ca8a04",          # yellow-700 — DA wholesale
        "ID": "#f59e0b",          # amber-500 — ID wholesale (brighter)
        "aFRR cap": "#6366f1",    # indigo-500 — capacity reservation
        "aFRR energy": "#a5b4fc", # indigo-300 — activation energy
    }

    def _daily_stream_breakdown(days):
        rows = []
        for d in days:
            b = d.revenue_breakdown
            rows.append({
                "date": d.date,
                "DA": b.get("da", 0),
                "ID": b.get("id", 0),
                "aFRR cap": b.get("afrr_cap_pos", 0) + b.get("afrr_cap_neg", 0),
                "aFRR energy": b.get("afrr_energy_pos", 0) + b.get("afrr_energy_neg", 0),
            })
        return pd.DataFrame(rows)

    # ─── Panel 1: Daily revenue per market (stacked bar) ──────
    st.markdown("##### Panel 1 — *Daily Revenue per Market*")
    st.caption(
        "Stacked bars of **gross** daily revenue across the month, "
        "split by market (DA + ID + aFRR cap + aFRR energy — directly "
        "comparable to your optimiser portal). The no-cost policy "
        "captures big DA / ID arbitrage spikes on volatile days; the "
        "aging-aware policy lives almost entirely on the steady aFRR "
        "capacity base."
    )
    col_l1, col_l6 = st.columns(2)

    # Shared y-axis range for the daily-revenue stacked bars — biggest
    # signed positive sum and biggest signed negative sum across L1 and
    # L6 — so the visual comparison is honest, not autoscaled per panel.
    def _signed_extremes(df: pd.DataFrame) -> tuple[float, float]:
        cols = list(STREAM_COLORS)
        pos_sum = df[cols].clip(lower=0).sum(axis=1)
        neg_sum = df[cols].clip(upper=0).sum(axis=1)
        return float(pos_sum.max()), float(neg_sum.min())

    _df_l1_pre = _daily_stream_breakdown(m1_year_days)
    _df_l6_pre = _daily_stream_breakdown(m5_year_days)
    _p1, _n1 = _signed_extremes(_df_l1_pre)
    _p2, _n2 = _signed_extremes(_df_l6_pre)
    _rev_y_top = max(_p1, _p2) * 1.05
    _rev_y_bot = min(_n1, _n2) * 1.05

    def _daily_revenue_panel(col, days, label, y_range):
        with col:
            with st.container(border=True):
                df = _daily_stream_breakdown(days)
                fig = go.Figure()
                for stream, color in STREAM_COLORS.items():
                    fig.add_trace(go.Bar(
                        x=df["date"], y=df[stream],
                        name=stream,
                        marker_color=color,
                        marker_line_width=0,
                        hovertemplate=(
                            "%{x|%d %b}<br>"
                            f"{stream}: €%{{y:.0f}}<extra></extra>"
                        ),
                    ))
                fig.update_layout(
                    barmode="relative",
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=260, margin=dict(l=50, r=10, t=10, b=40),
                    yaxis=dict(title="Daily revenue (€)", range=list(y_range)),
                    xaxis=dict(showgrid=False),
                    legend=dict(orientation="h", y=-0.20, font=dict(size=9)),
                    bargap=0.0,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

    _daily_revenue_panel(col_l1, m1_year_days, "M1 No-cost", (_rev_y_bot, _rev_y_top))
    _daily_revenue_panel(col_l6, m5_year_days, "M5 Time-state", (_rev_y_bot, _rev_y_top))

    st.markdown(
        "**What you're looking for.** Tall wholesale spikes (DA and "
        "ID) on volatile days = the optimiser cycles on every spread "
        "the market offers. A near-flat aFRR-cap base every day, with "
        "only the occasional wholesale bar = the cycle cost is doing "
        "the work."
    )

    st.markdown("")

    # ─── Panel 2: Revenue share pie (relative-only) ───────────
    # Reuses _daily_stream_breakdown for share-of-total computation.
    _stream_breakdown = _daily_stream_breakdown

    st.markdown("##### Panel 2 — *Revenue Share per Market*")
    st.caption(
        "Pie of monthly revenue split, shares only. Look for the "
        "*M1 → M5 shift* — the wholesale slice shrinking as the "
        "optimiser parks more capacity in aFRR."
    )
    col_l1, col_l6 = st.columns(2)

    def _share_pie_panel(col, days, label):
        with col:
            with st.container(border=True):
                df = _stream_breakdown(days)
                shares = {s: max(df[s].sum(), 0) for s in STREAM_COLORS}
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=list(shares.keys()),
                    values=list(shares.values()),
                    marker=dict(colors=[STREAM_COLORS[s] for s in shares]),
                    hole=0.45,
                    textinfo="label+percent",
                    textfont=dict(size=11),
                ))
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=250, margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

    _share_pie_panel(col_l1, m1_year_days, "M1 No-cost")
    _share_pie_panel(col_l6, m5_year_days, "M5 Time-state")

    def _afrr_pct(days):
        df = _stream_breakdown(days)
        afrr = df["aFRR cap"].sum() + df["aFRR energy"].sum()
        net = (df["DA"].sum() + df["ID"].sum()
               + df["aFRR cap"].sum() + df["aFRR energy"].sum())
        return afrr / max(net, 1.0) * 100

    l1_afrr = _afrr_pct(m1_year_days)
    l6_afrr = _afrr_pct(m5_year_days)
    l1_ws = 100 - l1_afrr
    l6_ws = 100 - l6_afrr
    st.markdown(
        f"**What you're looking for — which way the split moves.** "
        f"Wholesale (DA + ID) makes money from *cycling*; aFRR cap "
        f"makes money from *being available*. The higher the cycle "
        f"cost, the more the optimiser shifts toward availability. "
        f"Here M1's wholesale slice is ~{l1_ws:.0f}% of gross "
        f"revenue; M5 shrinks it to ~{l6_ws:.0f}% — a "
        f"{-(l1_ws - l6_ws):+.0f} pp shift toward aFRR. Practically: if "
        f"your optimiser's aFRR share rose year-on-year while cycles "
        f"dropped, that's a cycle-cost policy at work."
    )

    st.markdown("")

    # ─── Panel 3: SoC trace ───────────────────────────────────
    st.markdown("##### Panel 3 — *State of Charge Development*")
    st.caption(
        "Quarter-hour state-of-charge trace across the month — the "
        "same view an optimiser sees on its portal. Aging-aware parks "
        "the battery in the middle of the band when nothing worth "
        "trading is on; the no-cost policy bounces between empty and "
        "full."
    )
    col_l1, col_l6 = st.columns(2)

    def _soc_trace_panel(col, days, label, color):
        with col:
            with st.container(border=True):
                ts_all, soc_all = [], []
                for d in days:
                    soc_frac = np.array(d.soc_mwh) / max(d.energy_mwh, 1e-6)
                    day_ts = pd.date_range(
                        start=pd.Timestamp(d.date),
                        periods=len(soc_frac), freq="15min",
                    )
                    ts_all.extend(day_ts)
                    soc_all.extend(soc_frac.tolist())
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts_all, y=soc_all, mode="lines",
                    line=dict(color=color, width=1.0),
                    hovertemplate="%{x|%d %b %H:%M}<br>SoC: %{y:.2f}<extra></extra>",
                    showlegend=False,
                ))
                fig.add_hline(y=0.20, line_dash="dot",
                              line_color="#cbd5e1", line_width=1)
                fig.add_hline(y=0.80, line_dash="dot",
                              line_color="#cbd5e1", line_width=1)
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=240, margin=dict(l=40, r=10, t=10, b=30),
                    yaxis=dict(title="SoC fraction", range=[0, 1]),
                    xaxis=dict(showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

    _soc_trace_panel(col_l1, m1_year_days, "M1 No-cost", NAIVE_C)
    _soc_trace_panel(col_l6, m5_year_days, "M5 Time-state", AGING_C)

    l1_max_avg = float(np.mean([np.array(d.soc_mwh).max() / d.energy_mwh
                                for d in m1_year_days]))
    l1_min_avg = float(np.mean([np.array(d.soc_mwh).min() / d.energy_mwh
                                for d in m1_year_days]))
    l6_max_avg = float(np.mean([np.array(d.soc_mwh).max() / d.energy_mwh
                                for d in m5_year_days]))
    l6_min_avg = float(np.mean([np.array(d.soc_mwh).min() / d.energy_mwh
                                for d in m5_year_days]))
    st.markdown(
        f"**What you're looking for.** A trace that swings between "
        f"empty and full every day = the optimiser chases every spread. "
        f"A trace that hovers around 0.40 – 0.60 = the cycle cost is "
        f"doing the work. Over {month_label}, M1's daily min/max averaged "
        f"[{l1_min_avg:.2f}, {l1_max_avg:.2f}]; M5 averaged "
        f"[{l6_min_avg:.2f}, {l6_max_avg:.2f}] — visibly tighter, "
        f"pinned to the middle."
    )

    st.markdown("")

    # ─── Panel 4: Daily cycles bars ───────────────────────────
    st.markdown("##### Panel 4 — *Daily Cycles*")
    st.caption(
        "Full-equivalent cycles per day — the standard "
        "cycling-intensity tile."
    )
    col_l1, col_l6 = st.columns(2)

    # Shared y-axis for daily FEC bars — biggest single-day FEC across
    # both panels so the M1 vs M5 cycling intensity reads at a glance.
    _fec_y_max = max(
        max(d.full_equivalent_cycles for d in m1_year_days),
        max(d.full_equivalent_cycles for d in m5_year_days),
    ) * 1.10

    def _daily_cycles_panel(col, days, label, color, y_max):
        with col:
            with st.container(border=True):
                fec_per_day = np.array([d.full_equivalent_cycles for d in days])
                gross_per_day = np.array([
                    sum(d.revenue_breakdown.values()) for d in days
                ])
                total_fec = float(fec_per_day.sum())
                avg_per_day = float(fec_per_day.mean())
                total_rev = float(gross_per_day.sum())
                dates = [d.date for d in days]
                kcol1, kcol2, kcol3 = st.columns(3)
                kcol1.metric("Total FEC", f"{total_fec:.0f}")
                kcol2.metric("Avg / day", f"{avg_per_day:.2f}")
                kcol3.metric("Gross revenue", f"€{total_rev/1000:.1f}k")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=dates, y=fec_per_day,
                    marker_color=color, marker_line_width=0,
                    hovertemplate="%{x|%d %b}<br>FEC: %{y:.2f}<extra></extra>",
                ))
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=200, margin=dict(l=40, r=10, t=10, b=30),
                    yaxis=dict(title="FEC / day", range=[0, y_max]),
                    xaxis=dict(showgrid=False),
                    bargap=0.15,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

    _daily_cycles_panel(col_l1, m1_year_days, "M1 No-cost", NAIVE_C, _fec_y_max)
    _daily_cycles_panel(col_l6, m5_year_days, "M5 Time-state", AGING_C, _fec_y_max)

    l1_total_fec = sum(d.full_equivalent_cycles for d in m1_year_days)
    l6_total_fec = sum(d.full_equivalent_cycles for d in m5_year_days)
    st.markdown(
        f"**What you're looking for.** Tall, near-uniform bars pressed "
        f"up against the **2 FEC/day** cap = an optimiser with no cycle "
        f"cost. Sparse short bars with lots of near-zero days = the "
        f"cycle cost is doing the work. The M1 → M5 cycling gap here "
        f"is **{l1_total_fec/l6_total_fec:.0f}×** for the same month "
        f"({l1_total_fec:.0f} vs {l6_total_fec:.0f} FEC)."
    )


# ── German BESS Outlook market trajectory sensitivity ──────
st.markdown("---")
st.markdown("## Does the ranking survive when ancillary revenue collapses?")
st.markdown("""
**Reminder from [*German BESS Outlook*](https://de-bess-outlook.streamlit.app).**
Between 2026 and 2035, aFRR capacity revenue per MW shrinks ~85 %
as the German BESS fleet saturates ancillary demand, while DA + ID
arbitrage grows as midday solar deepens and electrification lifts
evening peaks. The two move in opposite directions; total per-MW
revenue settles into a U-shape with the low point around 2030–32.

The headline holds 2025 prices for all ten years to isolate the
cycle-cost effect. Does the M5 &gt; M4 &gt; M3 &gt; M2 &gt; M1
ranking survive when the market shifts? The right panel re-runs
the dispatch each year under projected prices, with per-source
scaling matching Note 1's mid-case.
""")

# Recalibrated 10-y headline: same M-stack at v4.2 calibration, with
# year-by-year prices uniformly scaled to match deployed Note 1
# published total trajectory (2026 €233/MW → 2030 €112/MW → 2035
# €115/MW; see precompute_v42_recalibrated_10y.py NOTE1_PUBLISHED_TOTALS).
# Single per-year scale applied across all streams + FCR phantom —
# stream-mix evolution is intentionally suppressed for cross-note
# coherence. Per-stream story belongs in benchmark-reconciliation once
# Note 1 v2 publishes.
_recal = _load_pkl(str(DATA_DIR / "precomputed_v42_recalibrated_10y.pkl"))
_recal_results = _recal["results"]

trend_rows = []
naive_flat_npv = results["M1_naive"].lifetime_npv_eur
naive_trend_npv = _recal_results["M1_naive"].lifetime_npv_eur
for name in POLICY_ORDER:
    if name not in results:
        continue
    r_flat = results[name]
    r_trend = _recal_results[name]
    trend_rows.append({
        "policy": POLICY_LABELS[name],
        "short": POLICY_SHORT[name],
        "color": POLICY_COLORS[name],
        # Each panel normalized to its own M1 = 100 — the section's
        # purpose is the relative ranking shift between scenarios, not
        # cross-scenario absolute-level comparison.
        "flat_index": r_flat.lifetime_npv_eur / naive_flat_npv * 100,
        "trend_index": r_trend.lifetime_npv_eur / naive_trend_npv * 100,
    })
trend_df = pd.DataFrame(trend_rows)

render_chart_title(
    "Flat 2025 markets vs <i>German BESS Outlook</i> mid-case "
    "trajectory (2026 → 2035, "
    "dispatch re-run each year under projected prices). Each panel "
    "indexed to its own M1 = 100 — the question is how the M1→M5 "
    "ranking changes when the market shifts, not absolute levels."
)

_y_max = float(max(trend_df["flat_index"].max(),
                   trend_df["trend_index"].max())) * 1.10

col_flat, col_trend = st.columns(2)

def _trend_panel(col, y_col, panel_title):
    with col:
        st.markdown(
            f"<div style='font-size:11px;color:#64748b;text-align:center;"
            f"text-transform:uppercase;letter-spacing:0.5px;'>"
            f"{panel_title}</div>",
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=trend_df["short"], y=trend_df[y_col],
            marker_color=trend_df["color"], marker_line_width=0,
            text=[f"{v:.0f}" for v in trend_df[y_col]],
            textposition="outside",
            hovertemplate="%{x}: %{y:.0f}<extra></extra>",
        ))
        fig.add_hline(y=100, line_dash="dot",
                      line_color="#94a3b8", line_width=1)
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=360, margin=dict(l=40, r=10, t=10, b=50),
            yaxis=dict(title="Index (M1 in this scenario = 100)",
                       range=[0, _y_max]),
            xaxis=dict(tickfont=dict(size=10)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

_trend_panel(col_flat, "flat_index", "10-y NPV — 2025 prices replayed every year")
_trend_panel(col_trend, "trend_index", "10-y NPV — Outlook trajectory 2026 → 2035")

flat_peak = trend_df.loc[trend_df["flat_index"].idxmax()]
trend_peak = trend_df.loc[trend_df["trend_index"].idxmax()]

_m5_flat_uplift = trend_df[trend_df["short"] == POLICY_SHORT["M5_adp"]].iloc[0]["flat_index"] - 100
_trend_m2 = trend_df[trend_df["short"] == POLICY_SHORT["M2_flat"]].iloc[0]["trend_index"]
_trend_m3 = trend_df[trend_df["short"] == POLICY_SHORT["M3_scarcity"]].iloc[0]["trend_index"]
_trend_m4 = trend_df[trend_df["short"] == POLICY_SHORT["M4_physics"]].iloc[0]["trend_index"]
_trend_m5 = trend_df[trend_df["short"] == POLICY_SHORT["M5_adp"]].iloc[0]["trend_index"]
# Each panel indexed to its own M1 = 100, so the trend_index value
# IS the % uplift over trajectory M1 (e.g. 112 = +12 % vs M1).
_trend_m2_uplift_pct = _trend_m2 - 100
_trend_m3_uplift_pct = _trend_m3 - 100
_trend_m4_uplift_pct = _trend_m4 - 100
_trend_m5_uplift_pct = _trend_m5 - 100
# Identify the trajectory winner dynamically (M2 and M3 typically
# essentially tied — slight edge depends on regen rounding).
_trend_uplifts = {
    "M2 (Fixed fee)": _trend_m2_uplift_pct,
    "M3 (Health-aware)": _trend_m3_uplift_pct,
    "M4 (Shape-aware)": _trend_m4_uplift_pct,
    "M5 (Time-and-state)": _trend_m5_uplift_pct,
}
_trend_winner_name = max(_trend_uplifts, key=_trend_uplifts.get)
_trend_winner_pct = _trend_uplifts[_trend_winner_name]

render_chart_caption(
    f"Under flat 2025, M5 wins <b>+{_m5_flat_uplift:.0f}%</b>. Under "
    f"the projected trajectory, the simple regime-independent "
    f"policies (M2, M3) take the lead. Each panel is normalised to "
    f"its own M1 = 100; absolute lifetime NPV differs between "
    f"scenarios but that's not the point of this comparison."
)


render_takeaway(
    f"<b>Pick the policy that matches your market, not the most "
    f"sophisticated one.</b> On flat 2025, M5 wins +{_m5_flat_uplift:.0f} %. "
    f"Under the projected market trajectory, "
    f"<b>{_trend_winner_name} wins +{_trend_winner_pct:.0f} %</b> — "
    f"a regime-independent rule adapts to the shifting market better "
    f"than 2024-calibrated state-aware functions. For a "
    f"2026-commissioned asset operating into the 2030s, simple wins."
)


# ── Methodology expander ────────────────────────────────────
st.markdown("---")
with st.expander("Methodology"):
    st.markdown(f"""
**Asset.** 2 h LFP, 1 MW / 2 MWh nominal, EVE LF280K cell preset
calibrated against the manufacturer 6 000-cycle / 80% retention spec
on the [*What Drives Degradation*](https://bess-degradation-drivers.streamlit.app/)
Wang + Naumann two-channel physics kernel. Warranty floor 0.80 SoH;
round-trip efficiency 0.88. Discount rate 7%.

**Markets.** DE day-ahead from EnergyCharts; intraday from DE-LU
15-minute auctions and continuous trading
([ENTSO-E Transparency Platform](https://www.entsoe.eu/network_codes/cacm/implementation/ida/)).
aFRR capacity and activation energy from regelleistung.net and
netztransparenz.de.

**Dispatch LP.** Two-stage market-aware. Stage 1 commits
per-4-hour-block aFRR capacity at D−1 under a regime-conditional
α-forecast (`bid_win_rate = 1.0`; see *Where this model stops
working* below for the caveat). Stage 2 re-optimises full
DA + ID + activation dispatch under realised α with the Stage 1
commitment locked. For two-pass methodologies (M4 —
physics-from-duty), Stage 2 runs a second LP with wear re-priced
via the [*What Drives Degradation*](https://bess-degradation-drivers.streamlit.app/)
kernel (`physics_wear_from_duty`) on the observed first-pass
dispatch.

**Cycle-cost forms — implementation detail.**
- **M1 (No cycle cost).** Wear cost = 0 €/MWh. Trades every profitable spread.
- **M2 (Fixed fee, *Kumtepeli depreciation proxy*).** Wear =
  `CAPEX_eur_per_mwh / lifetime_throughput` ≈ €16.67 / MWh on this
  asset (€100 k / MWh CAPEX ÷ 6 000 lifetime full cycles × 2 MWh =
  12 000 MWh).
- **M3 (Health-aware, *scarcity scaling on SoH-state*).** Wear =
  M2 × `1 + (1 − SoH) / (1 − floor)`, bounded 1× at fresh cell, 2× at
  the warranty floor.
- **M4 (Cycle-shape-aware, *physics-from-duty*).** Two-pass LP. Pass 1
  uses ADP (approximate dynamic programming — see M5) intraday shadow
  cost as a starting estimate (so Stage-1 commitments are reasonable); the observed dispatch is fed into the
  [*What Drives Degradation*](https://bess-degradation-drivers.streamlit.app/)
  physics kernel (Wang + Naumann two-channel, calibrated to
  EVE LF280K); pass 2 re-solves Stage-2 with the resulting per-MWh
  wear that reflects the actual DoD / C-rate / SoC band of the day's
  duty.
- **M5 (Time-and-state-aware, *intraday ADP, Holtorf-Shin*).** An
  offline backward-induction DP over `(SoC, SoH, regime, hour-of-day)`
  returns a state-value function; its gradient with respect to SoC is
  the shadow cost. Because hour-of-day is in the state, the cost
  varies hour by hour: at 18:00 in a volatile regime one MWh of
  stored energy is worth more than at 03:00 in a calm regime.

**Why one cycle cost beats two.** Two cross-checks confirm the best
cycle cost is one cost, picked well. Stacking M2+M3 under M5 gives
**+{(_abl_blend.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.2f}%**
vs M1 ({(_abl_blend.lifetime_npv_eur / _m5.lifetime_npv_eur - 1) * 100:+.2f} pp vs M5 alone) — the rough flat fee
double-charges what the time-and-state cost already prices. Running
M4's shape kernel without an ADP base gives
**+{(_abl_solo.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.2f}%**
({(_abl_solo.lifetime_npv_eur / _m4.lifetime_npv_eur - 1) * 100:+.2f} pp vs full M4) — without time-and-state
guidance in pass 1, the day-ahead bids are committed naively before
the shape correction in pass 2 can act.

**Lifetime simulation.** 10 years × single template year 2025. Each
year sees identical inputs across all five methodologies; only the
wear-cost vector each methodology emits differs.

**Where this model stops working.**
- The naive policy is a strawman. It assumes zero cycle pricing AND
  zero DoD ceiling. Real commercial operators respect warranty cycle
  caps and DoD limits, which is a crude form of cycle pricing.
- The Y7 naive EOL at SoH = 0.80 is contract-life, not physical life.
  Under contracts that accept operation past 0.80 the naive policy
  earns another 2 – 3 years at derated capacity.
- The BESS is modelled as a single aggregated unit. Real systems have
  hundreds of modules with manufacturing variance, edge-vs-centre
  thermal asymmetry, and uneven usage. The pack-level allocation
  problem sits one abstraction below this analysis.
- Calendar year mapping. 2025 prices are held flat for all 10
  simulation years rather than projected forward. The
  [*German BESS Outlook*](https://de-bess-outlook.streamlit.app)
  trajectory chart layers fleet-saturation revenue decay on top of
  each policy's dispatch and shows how the ranking shifts.
- The `bid_win_rate=1.0` is an LP-upper-bound assumption (see
  Calibration anchors above). Real-world clearing rates are < 100 %
  for individual bidders, depending on bid strategy and fleet
  capacity vs auction demand. Lower clearing would compress aFRR cap
  revenue and shrink the M5 uplift.
""")


# ── Related work ────────────────────────────────────────────
with st.expander("Related work"):
    st.markdown("""
**Direct anchors (the policies implement these).**

- **Kumtepeli, Hesse, Morstyn, Nosratabadi, Aunedi, Howey (2024).**
  *Depreciation Cost is a Poor Proxy for Revenue Lost to Aging in
  Grid Storage Optimization.*
  [arXiv:2403.10617](https://arxiv.org/abs/2403.10617).
  The reframe from "CAPEX ÷ lifetime throughput" (M2) to "forgone
  future revenue" (M5 / M4).
- **Holtorf, Shin (2026).** *Approximate Dynamic Programming for
  Degradation-aware Market Participation of BESS.*
  [arXiv:2603.21089](https://arxiv.org/abs/2603.21089). The state-
  dependent opportunity-cost formulation M5 follows.
- **Collath, Englberger, Jossen, Hesse (2023).** *Increasing the
  lifetime profitability of battery energy storage systems through
  aging-aware operation.*
  [Applied Energy 348, 121531](https://www.sciencedirect.com/science/article/pii/S0306261923008954).
  Reports +29.3% lifetime profit from a piecewise-linear in-objective
  form. This note recovers the directional claim and somewhat larger
  magnitude on DE markets with a publicly-anchored calibration; M5 vs
  M1 on the headline run lands at +36 %.

**Physics foundation.**

- **Naumann et al. (2018) + Wang et al. (2014) + Stanford (2024).**
  Cycle and calendar fade anchors for the [*What Drives Degradation*](https://bess-degradation-drivers.streamlit.app/) physics kernel; EVE
  LF280K manufacturer datasheet for the asset-specific re-anchoring.
- **Collath, Winner, Frank, Durdel, Jossen (2024).** *Suitability of
  late-life lithium-ion cells for battery energy storage systems.*
  [J. Energy Storage 86, 111645](https://www.sciencedirect.com/science/article/pii/S2352152X24010934).
  Mechanism-side justification for the age-acceleration multiplier
  and the asymmetric-down SoC optimum.

**Ancillary-specific anchor.**

- **He, Malkaby-Epstein et al. (2016).** *Optimal bidding strategy of
  battery storage in power markets considering performance-based
  regulation and battery cycle life.*
  [DTU Orbit](https://orbit.dtu.dk/en/publications/optimal-bidding-strategy-of-battery-storage-in-power-markets-cons).
  Eight-year precedent for the signal-5 point that €/MWh is the wrong
  denominator on availability-paid assets.

**Independent confirmation.**

- **Humiston, Cetin, de Queiroz (2026).** *Evaluating Battery
  Degradation Models in Rolling-Horizon BESS Arbitrage Optimization.*
  [Energies 19(4), 1056](https://www.mdpi.com/1996-1073/19/4/1056).
  ERCOT replication of the Kumtepeli "shape matters" claim on
  15-minute real-time data.
""")


# ── Closing ─────────────────────────────────────────────────
render_closing(
    "Fourth note in the German BESS merchant-economics series. Next: "
    "*Which BESS revenue number is real?* — the public German indices "
    "disagree by ~€100 k/MW/yr on the same months; the next note "
    "unpacks where the spread comes from."
)

render_footer()
