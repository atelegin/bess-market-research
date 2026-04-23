"""
Note 4 — Is Your Trader Aging-Aware? The Cost of a Cycle.

Results pre-computed by precompute.py and loaded from data/precomputed.pkl.
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

from lib.ui.theme import (
    apply_theme,
    render_chart_caption,
    render_chart_title,
    render_closing,
    render_footer,
    render_header,
)

PRECOMPUTED_PATH = Path(__file__).parent / "data" / "precomputed.pkl"


POLICY_LABELS = {
    "naive": "Naive (no cycle cost)",
    "depreciation_proxy": "Flat depreciation proxy",
    "aging_aware_depreciation": "Aging-aware formula",
    "adp_simplified": "ADP — simplified state",
    "adp_intraday": "ADP — intraday (Holtorf-Shin)",
}
POLICY_COLORS = {
    "naive": "#94a3b8",                    # slate
    "depreciation_proxy": "#fbbf24",       # yellow (Kumtepeli target)
    "aging_aware_depreciation": "#3b82f6", # blue
    "adp_simplified": "#a78bfa",           # light purple
    "adp_intraday": "#10b981",             # green (winner)
}


@st.cache_data(show_spinner=False)
def load_precomputed() -> dict:
    with open(PRECOMPUTED_PATH, "rb") as f:
        return pickle.load(f)


st.set_page_config(
    page_title="Is Your Trader Aging-Aware?",
    page_icon="🔋",
    layout="wide",
)
apply_theme(show_sidebar=False)


# ── Header ──────────────────────────────────────────────────
render_header(
    title="Is Your Trader Aging-Aware?",
    kicker="GERMAN BESS | COST OF A CYCLE",
    subtitle="Most dispatch optimisers price cycles against a number that's wrong. Here's what that costs over a battery's life, and how to tell if your trader is doing better.",
)

# ── Intro ───────────────────────────────────────────────────
st.markdown("""
Every BESS owner is told their trader runs a "degradation-aware" dispatch.
Almost none of them can verify it. This note puts five dispatch policies —
ranging from no cycle pricing at all to a full Holtorf-Shin opportunity-cost
optimiser — on the same German market data and asks a simple question:
over a ten-year life, what does pricing cycles correctly actually earn?

The gap is bigger than the published benchmarks suggest. A well-priced
optimiser delivers **3.4× the lifetime revenue** of a naive one — not because
it cycles more, but because it cycles **less, at the right hours, and the
battery lasts years longer**.
""")

data = load_precomputed()
results = data["results"]
diagnostics = data["diagnostics"]
n_years = data["n_years"]


# ── KPI row ─────────────────────────────────────────────────
intraday = results["adp_intraday"]
naive = results["naive"]
formula = results["aging_aware_depreciation"]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Best-policy lifetime NPV",
        f"€{intraday.lifetime_npv_eur / 1000:,.0f}k / MW",
        f"+{(intraday.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.0f}% vs naive",
    )
with col2:
    eol_naive = (naive.years_to_floor or n_years) + 1
    eol_intra = (intraday.years_to_floor or n_years) + 1
    st.metric(
        "Battery life (aging-aware vs naive)",
        f"Y{eol_intra}",
        f"{eol_intra - eol_naive:+d} years",
    )
with col3:
    st.metric(
        "Cycle count (intraday ADP vs naive)",
        f"{intraday.annual_fec.sum():.0f} FEC",
        f"{(intraday.annual_fec.sum() - naive.annual_fec.sum()):+.0f} vs naive",
        delta_color="inverse",  # fewer cycles = better here
    )


# ── Main chart: lifetime NPV by policy ──────────────────────
render_chart_title(
    "Over ten years, the policy decides the outcome. "
    "Pricing cycles correctly is worth +239% of revenue."
)

npv_rows = []
for name in ["naive", "depreciation_proxy", "aging_aware_depreciation",
             "adp_simplified", "adp_intraday"]:
    r = results[name]
    npv_rows.append({
        "policy": POLICY_LABELS[name],
        "npv_keur": r.lifetime_npv_eur / 1000,
        "pct_vs_naive": (r.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100,
        "color": POLICY_COLORS[name],
        "eol_year": (r.years_to_floor or n_years) + 1,
        "total_fec": int(r.annual_fec.sum()),
    })
npv_df = pd.DataFrame(npv_rows)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=npv_df["policy"], y=npv_df["npv_keur"],
    marker_color=npv_df["color"],
    text=[f"€{v:,.0f}k" for v in npv_df["npv_keur"]],
    textposition="outside",
    hovertemplate=(
        "%{x}<br>NPV: €%{y:,.0f}k / MW<br>"
        "%{customdata[0]:+.0f}% vs naive<br>"
        "EOL: year %{customdata[1]}<br>"
        "Total FEC: %{customdata[2]}"
        "<extra></extra>"
    ),
    customdata=np.stack([
        npv_df["pct_vs_naive"], npv_df["eol_year"], npv_df["total_fec"],
    ], axis=-1),
))
fig.update_layout(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=400, margin=dict(l=40, r=20, t=20, b=60),
    yaxis=dict(title="Lifetime NPV (k€ / MW)"),
    xaxis=dict(tickfont=dict(size=11)),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
render_chart_caption(
    "Ten-year lifecycle simulation on real 2023/2025 German DA + ID + aFRR "
    "markets. All policies see identical prices, identical activation "
    "signals, identical warranty terms. Differences are entirely in how "
    "each optimiser priced the cost of a cycle."
)


# ── Annual revenue + SoH trajectories ───────────────────────
st.markdown("### The battery lives longer when you charge it less often")

col_rev, col_soh = st.columns(2)

with col_rev:
    render_chart_title("Annual revenue (nominal k€/MW)")
    fig_rev = go.Figure()
    for name in ["naive", "depreciation_proxy", "aging_aware_depreciation", "adp_intraday"]:
        r = results[name]
        fig_rev.add_trace(go.Scatter(
            x=np.arange(1, n_years + 1),
            y=r.annual_revenue_eur / 1000,
            mode="lines+markers",
            name=POLICY_LABELS[name],
            line=dict(color=POLICY_COLORS[name], width=2),
        ))
    fig_rev.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=360, margin=dict(l=40, r=20, t=20, b=50),
        xaxis=dict(title="Year"),
        yaxis=dict(title="k€ / MW"),
        legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
    )
    st.plotly_chart(fig_rev, use_container_width=True, config={"displayModeBar": False})

with col_soh:
    render_chart_title("End-of-year State of Health")
    fig_soh = go.Figure()
    for name in ["naive", "depreciation_proxy", "aging_aware_depreciation", "adp_intraday"]:
        r = results[name]
        fig_soh.add_trace(go.Scatter(
            x=np.arange(1, n_years + 1),
            y=r.end_of_year_soh,
            mode="lines+markers",
            name=POLICY_LABELS[name],
            line=dict(color=POLICY_COLORS[name], width=2),
        ))
    fig_soh.add_hline(y=0.80, line_dash="dot", line_color="#666",
                     annotation_text="warranty floor")
    fig_soh.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=360, margin=dict(l=40, r=20, t=20, b=50),
        xaxis=dict(title="Year"),
        yaxis=dict(title="SoH", range=[0.78, 1.01]),
        legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
    )
    st.plotly_chart(fig_soh, use_container_width=True, config={"displayModeBar": False})


# ── Executive summary ────────────────────────────────────────
st.markdown("""
### Executive summary

**Naive dispatch (no cycle pricing)** maxes out cycling, earns peak annual
revenue in Year 1 (€244k/MW), but degrades the cell to the warranty floor
inside two years. Over ten years the battery earns €361k/MW cumulatively.

**The "flat depreciation proxy"** (CAPEX divided by lifetime throughput —
the Kumtepeli/Howey 2024 "poor proxy") earns +84% over naive. It suppresses
cycling uniformly, extending life to four years. Better, but still uniform
— it doesn't know that cycling at the morning shoulder costs the same as
cycling at the evening peak, which is not how real BESS economics work.

**The intraday opportunity-cost optimiser** (Holtorf-Shin style) earns **+239%**.
It doesn't cycle more — it cycles *less* (591 FEC vs naive's 761). It
cycles at the right hours: full at the 18:00 peak, nothing at the
14:00 solar trough. The battery lasts eight years at a stable ~€220k/yr.
""")


# ── Diagnostic section ──────────────────────────────────────
st.markdown("---")
st.markdown("## How can an owner tell which they have?")
st.markdown("""
Five signals, each computable from one year of an operator's dispatch log.
Each one compares the intraday-ADP ("genuine aging-aware") policy against
the naive baseline — patterns in your own data that look like the naive
panel are the tell.
""")

diag_intraday = diagnostics.get("adp_intraday")
diag_naive = diagnostics.get("naive")

_NAIVE_COLOR = POLICY_COLORS["naive"]
_INTRA_COLOR = POLICY_COLORS["adp_intraday"]


def _grouped_bar_chart(
    title: str, caption: str, attr: str, x_col: str, y_col: str,
    x_label: str, y_label: str,
):
    """Render grouped bar comparison (naive vs intraday-ADP) for one signal."""
    st.markdown(f"**{title}**")
    st.caption(caption)
    fig = go.Figure()
    if diag_naive is not None:
        df_n = getattr(diag_naive, attr)
        fig.add_trace(go.Bar(
            x=df_n[x_col], y=df_n[y_col],
            name="Naive",
            marker_color=_NAIVE_COLOR,
        ))
    if diag_intraday is not None:
        df_i = getattr(diag_intraday, attr)
        fig.add_trace(go.Bar(
            x=df_i[x_col], y=df_i[y_col],
            name="Intraday-ADP",
            marker_color=_INTRA_COLOR,
        ))
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=40, r=20, t=10, b=40),
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label),
        legend=dict(orientation="h", y=-0.20, font=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


_grouped_bar_chart(
    "1. Depth-of-discharge by spread decile",
    "Aging-aware cycles deeper only on high-spread days; naive keeps DoD roughly flat regardless of day quality.",
    attr="dod_vs_spread", x_col="spread_decile", y_col="mean_dod",
    x_label="Daily-spread decile (0=calm → 9=volatile)",
    y_label="Mean DoD (fraction of capacity)",
)
_grouped_bar_chart(
    "2. SoC-hours distribution",
    "Aging-aware parks at 40–60% when the market is quiet; naive sits at whatever SoC the last trade left.",
    attr="soc_hours", x_col="soc_bin_pct", y_col="hours",
    x_label="SoC band (% of usable energy)",
    y_label="Hours in band (pooled across year)",
)
_grouped_bar_chart(
    "3. Revenue per FEC by spread quartile",
    "Aging-aware earns high EUR/FEC on weak-spread days because it skipped the marginal cycles; naive is dragged down.",
    attr="revenue_per_cycle_quartile",
    x_col="spread_quartile", y_col="eur_per_fec",
    x_label="Daily-spread quartile (0=bottom)",
    y_label="Mean EUR per FEC",
)
_grouped_bar_chart(
    "4. C-rate distribution",
    "Aging-aware holds back on marginal cycles (low-C mass); naive piles at P-max whenever the trade clears.",
    attr="crate_hist", x_col="crate_bin", y_col="hours",
    x_label="C-rate band",
    y_label="Hours in band (pooled across year)",
)


# ── Ancillary vs arbitrage mix ──────────────────────────────
st.markdown("---")
st.markdown("### Signal 5: Why €/MWh is the wrong yardstick on aFRR-heavy assets")

mix_intraday = diagnostics["adp_intraday"].ancillary_mix if diagnostics.get("adp_intraday") else None
if mix_intraday:
    st.markdown(
        f"In Year 1 of our simulation the intraday-ADP policy earned "
        f"**{mix_intraday['as_share'] * 100:.0f}% of its revenue from ancillary services** "
        f"(FCR + aFRR capacity + activation energy). That's not a cycling-paid asset. "
        f"Using EUR/MWh throughput to benchmark such an asset is nonsense — "
        f"availability, not cycling, is what pays. Use €/lost-capacity or "
        f"€/MW-reserved instead. (Point flagged by Dr. Sebastian Kawollek in "
        f"response to *What Actually Drives Degradation*, where I used EUR/MWh "
        f"because there the physics didn't depend on revenue stream. Here it does.)"
    )


# ── Methodology expander ────────────────────────────────────
st.markdown("---")
with st.expander("Methodology & where this model stops working"):
    st.markdown("""
**Asset**: 2 h LFP, 1 MW / 2 MWh nominal, EVE LF280K preset (calibrated
in Note 3 — *What Actually Drives Degradation*). Warranty floor 0.80;
initial SoH 1.0.

**Markets**: DE day-ahead + intraday (DA-proxy — no free historical ID
data; documented in roadmap); aFRR capacity + real activation-energy
revenue from regelleistung.net + netztransparenz.de; FCR explicitly
dropped (the BESS fleet has grown past FCR demand — the market is
past its moment for a 2h battery).

**Dispatch LP**: joint DA + ID + aFRR cap + aFRR energy solved as a
stacked-market LP per day (see `lib.models.dispatch_stacked`).
`max_afrr_participation = 0.40` — empirical fit to CH benchmark (see
roadmap `benchmark-reconciliation` for why this is a reduced-form catch-all).

**Lifetime**: 10 years, rotation through 2023 and 2025 DE markets
(2024 DA API unreliable). Each template year sees identical inputs
across all policies; the only differentiation is the wear cost vector
each policy emits.

**Degradation**: simple linear-in-FEC + calendar fade + mild age
acceleration (`fade_per_fec_at_soh_1 = 2e-4`, calendar `2e-5/day`).
Physics-grade replacement with `project_capacity_detailed` from Note 3
is a refinement planned for the warranty follow-up.

**ADP**: backward-induction DP with state `(SoC, SoH, regime, hour-of-day)`.
Simplifications: deterministic hourly price means per regime (no
stochastic transitions within a day), cyclic SoC boundary, SoH-constant
within a day. Full stochastic Holtorf-Shin would add real price
uncertainty — a further refinement.

**Where this model stops working**:
- The participation cap of 0.40 is a reduced-form catch-all. A trader
  with proper forecasting tools and lower risk aversion could push it
  higher; one operating under tighter ORL or battery-safety constraints
  would be lower.
- Real intraday (ID1/ID3 continuous markets) are not modelled as separate
  liquidity — we use the DA price twice. Operators with genuine ID
  trading would see higher absolute numbers across ALL policies, but the
  relative ordering should hold.
- The naive policy assumes *zero* cycle pricing, which is a strawman.
  Real operators at least respect warranty cycle caps. The naive column
  in this note is an extreme anchor, not a claim about any specific firm.
""")


# ── Closing ─────────────────────────────────────────────────
render_closing(
    "This note is the fourth in a series on German BESS merchant economics. "
    "Next up — the warranty: what it's worth as a real put option, and how "
    "much economic value owners leave on the table by treating it as a hard "
    "boundary rather than a priced protection."
)


render_footer()
