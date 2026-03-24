"""
BESS Revenue Model — Streamlit App
Open-data German BESS revenue projection 2026–2040.

Results are pre-computed by precompute.py and loaded from data/precomputed.pkl.
"""

import pickle
import sys
from pathlib import Path
# Add repo root to path so lib/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging

from lib.config import DEFAULT_BESS_BUILDOUT
from lib.models.projection import project_full_stack
from lib.ui.theme import (
    apply_theme,
    render_header,
    render_standfirst,
    render_takeaway,
    render_chart_title,
    render_chart_caption,
    render_annotation,
    render_footer_note,
    render_closing,
    render_footer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PRECOMPUTED_PATH = Path(__file__).parent / "data" / "precomputed.pkl"

@st.cache_data(show_spinner=False)
def _load_precomputed() -> dict:
    with open(PRECOMPUTED_PATH, "rb") as f:
        return pickle.load(f)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="German BESS Revenue Forecast",
    page_icon="⚡",
    layout="wide",
)

apply_theme(show_sidebar=True)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown("#### ⚡ Scenario Parameters")

duration = st.sidebar.selectbox("Battery duration", options=[1.0, 2.0, 4.0], index=1,
                               format_func=lambda x: f"{x:.0f}h")
st.sidebar.markdown("---")
st.sidebar.markdown("**Market scenarios**")
bess_2040 = st.sidebar.slider("BESS fleet 2040 (GW)", 20, 60, 40, 5,
                              help="Grid-scale BESS capacity target")
pv_2040 = st.sidebar.slider("Solar PV 2040 (GW)", 150, 400, 300, 25,
                             help="Installed PV capacity — drives duck curve depth")
gas_2040 = st.sidebar.slider("Gas price 2040 (€/MWh TTF)", 15, 80, 30, 5,
                             help="TTF (Title Transfer Facility) is the European gas benchmark (ICE). "
                                  "Gas sets peaker marginal cost → peak electricity prices → spread for BESS.")
demand_2040 = st.sidebar.slider("Demand 2040 (TWh)", 700, 1200, 1020, 50,
                                help="Electricity demand incl. electrification")
st.sidebar.markdown("---")
st.sidebar.markdown("**Cannibalisation**")
CANIB_SCENARIOS = {
    "Low (optimistic)":    {"canib_max": 15.0, "canib_half": 25.0, "canib_steep": 0.80},
    "Mid (merit order)":   {"canib_max": 33.0, "canib_half": 13.0, "canib_steep": 0.17},
    "High (CAISO-scaled)": {"canib_max": 82.0, "canib_half": 11.0, "canib_steep": 0.18},
}
canib_scenario = st.sidebar.radio(
    "Scenario",
    list(CANIB_SCENARIOS.keys()),
    index=1,
    help=(
        "**Low**: DE+UK 2023-2025 (1.5-6.8 GW) — optimistic extrapolation. "
        "**Mid**: merit order simulation on 2024 DE DA prices. "
        "**High**: CAISO observed (4→15 GW, 2022-2025)."
    ),
)
canib_params = CANIB_SCENARIOS[canib_scenario]

# Scale buildout trajectory toward 2040 target, but preserve near-term
# committed capacity (2024-2027 is largely under construction / permitted).
COMMITTED_THROUGH = 2027  # years up to here are not scaled down

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

user_buildout = _scale_buildout(bess_2040)

# ── Load pre-computed historical data ─────────────────────────
_precomputed = _load_precomputed()
_dur_key = f"{duration:.0f}h"
_dur_data = _precomputed[_dur_key]

hist_rev = _dur_data["hist_rev"]
_failed = _dur_data["failed_years"]

if _failed:
    _fail_msg = ", ".join(f"{y} ({e})" for y, e in _failed)
    st.warning(
        f"Could not fetch DA prices for: {_fail_msg}. "
        "The projection still works using calibrated baselines, but historical bars may be incomplete. "
        "Check your internet connection or try again later.",
        icon="⚠️",
    )

baseline_da = _dur_data["baseline_da"]
hist_bars = _dur_data["hist_bars"]
hist_bars_df = pd.DataFrame(hist_bars) if hist_bars else pd.DataFrame()

# Projection 2026–2040
proj_years = list(range(2026, 2041))
proj = project_full_stack(
    years=proj_years,
    historical_da_keur=baseline_da,
    bess_buildout=user_buildout,
    duration_h=duration,
    demand_2040_twh=float(demand_2040),
    gas_2040=float(gas_2040),
    pv_2040_gw=float(pv_2040),
    **canib_params,
)
proj_df = pd.DataFrame(proj)
if not hist_bars_df.empty:
    proj_df = pd.concat([hist_bars_df, proj_df], ignore_index=True)

# ── Chart setup ──────────────────────────────────────────────
STACK_KEYS = ["da", "id", "fcr", "afrr_cap", "afrr_energy"]
LABELS = {
    "da": "Day-Ahead", "id": "Intraday",
    "fcr": "FCR", "afrr_cap": "aFRR capacity", "afrr_energy": "aFRR energy",
}
COLORS = {
    "da": "#93c5fd", "id": "#3b82f6",
    "fcr": "#fbbf24", "afrr_cap": "#f87171", "afrr_energy": "#dc2626",
}
hist_year_set = set(hist_rev.keys()) if hist_rev else set()

# Compute Bull/Bear confidence bands across ALL parameters
_DIV_YEARS = list(range(2026, 2041))

# Bull: low competition, expensive gas (wide spreads), high demand, low cannibalisation
_BULL = dict(bess_2040=20, gas_2040=80, demand_2040=1200, pv_2040=215,
             **CANIB_SCENARIOS["Low (optimistic)"])
# Bear: crowded market, cheap gas (narrow spreads), low demand, high cannibalisation
_BEAR = dict(bess_2040=60, gas_2040=15, demand_2040=700, pv_2040=400,
             **CANIB_SCENARIOS["High (CAISO-scaled)"])

@st.cache_data(show_spinner=False)
def _compute_bull_bear(_dur, _baseline):
    bands = {}
    for label, params in [("Bull", _BULL), ("Bear", _BEAR)]:
        buildout = _scale_buildout(params["bess_2040"])
        rows = project_full_stack(
            years=_DIV_YEARS,
            historical_da_keur=_baseline,
            bess_buildout=buildout,
            duration_h=_dur,
            demand_2040_twh=float(params["demand_2040"]),
            gas_2040=float(params["gas_2040"]),
            pv_2040_gw=float(params["pv_2040"]),
            canib_max=params["canib_max"],
            canib_half=params["canib_half"],
            canib_steep=params["canib_steep"],
        )
        bands[label] = pd.DataFrame(rows)
    return bands

_bands = _compute_bull_bear(duration, baseline_da)
_band_bull = _bands["Bull"]
_band_bear = _bands["Bear"]

# Also compute per-cannibalisation scenarios (used in fan chart below)
@st.cache_data(show_spinner=False)
def _compute_scenario_annual(_buildout_items, _dur, _baseline, _demand_2040, _gas_2040, _pv_2040):
    buildout = dict(_buildout_items)
    results = {}
    for sname, params in CANIB_SCENARIOS.items():
        rows = project_full_stack(
            years=_DIV_YEARS,
            historical_da_keur=_baseline,
            bess_buildout=buildout,
            duration_h=_dur,
            demand_2040_twh=_demand_2040,
            gas_2040=_gas_2040,
            pv_2040_gw=_pv_2040,
            **params,
        )
        results[sname] = pd.DataFrame(rows)
    return results

scen_annual = _compute_scenario_annual(
    tuple(sorted(user_buildout.items())), duration, baseline_da,
    float(demand_2040), float(gas_2040), float(pv_2040),
)

y_max = max(
    proj_df[STACK_KEYS].sum(axis=1).max(),
    _band_bull["total"].max(),
) * 1.08

def _mute(hex_color, factor=0.55):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    grey = 200
    return f"#{int(r*factor+grey*(1-factor)):02x}{int(g*factor+grey*(1-factor)):02x}{int(b*factor+grey*(1-factor)):02x}"

def make_chart(df, height=520):
    fig = go.Figure()
    df_hist = df[df["year"].isin(hist_year_set)] if hist_year_set else pd.DataFrame()
    df_proj = df[~df["year"].isin(hist_year_set)]

    # Confidence band: Bull–Bear range across all parameters
    fig.add_trace(go.Scatter(
        x=_band_bear["year"], y=_band_bear["total"],
        mode="lines", line=dict(color="#ef4444", width=1, dash="dot"),
        name="Bear case", showlegend=True, legendgroup="band",
    ))
    fig.add_trace(go.Scatter(
        x=_band_bull["year"], y=_band_bull["total"],
        mode="lines", line=dict(color="#22c55e", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(20,33,61,0.05)",
        name="Bull case", showlegend=True, legendgroup="band",
    ))

    for key in STACK_KEYS:
        if not df_hist.empty:
            fig.add_trace(go.Bar(
                x=df_hist["year"], y=df_hist[key],
                name=LABELS[key], marker_color=_mute(COLORS[key]),
                showlegend=False, legendgroup=key,
            ))
    for key in STACK_KEYS:
        fig.add_trace(go.Bar(
            x=df_proj["year"], y=df_proj[key],
            name=LABELS[key], marker_color=COLORS[key],
            showlegend=True, legendgroup=key,
        ))

    fig.add_trace(go.Scatter(
        x=df_proj["year"], y=df_proj["total"],
        name="Total", mode="lines+markers",
        line=dict(color="#14213d", width=1.5, dash="dot"),
        marker=dict(size=3), showlegend=False,
    ))

    if hist_year_set:
        boundary = max(hist_year_set) + 0.5
        fig.add_vline(x=boundary, line_dash="dot", line_color="#5c677d", line_width=1)
        fig.add_annotation(
            text="\u2190 actual | forecast \u2192", x=boundary, y=y_max * 0.95,
            showarrow=False, font=dict(size=10, color="#5c677d"),
        )

    fig.update_layout(
        barmode="stack", template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=50, r=20, t=30, b=40),
        xaxis=dict(tickvals=[2025, 2028, 2031, 2034, 2037, 2040], tickangle=0, tickfont=dict(color="#5c677d")),
        yaxis=dict(title="\u20ack / MW / year", range=[0, y_max], title_font=dict(color="#5c677d"), tickfont=dict(color="#5c677d")),
        legend=dict(orientation="h", y=-0.12, font=dict(size=11, color="#14213d"), traceorder="normal"),
    )
    return fig

# ── Key numbers ──────────────────────────────────────────────
def _get(year):
    row = proj_df[proj_df.year == year]
    return row.iloc[0] if len(row) else None

r26 = _get(2026)
r30 = _get(2030)
r40 = _get(2040)

# Find the floor year dynamically (minimum total in projection range)
proj_only = proj_df[proj_df["year"].isin(proj_years)]
if not proj_only.empty:
    floor_year = int(proj_only.loc[proj_only["total"].idxmin(), "year"])
    r_floor = _get(floor_year)
else:
    floor_year = 2037
    r_floor = _get(2037)

ch_all = _dur_data["ch_all"]
ch_2023 = ch_all.get(2023, 0)
ch_2024 = ch_all.get(2024, 0)
ch_2025 = ch_all.get(2025, 0)

anc_share_2026 = (r26['fcr'] + r26['afrr_cap'] + r26['afrr_energy']) / r26['total'] * 100 if r26 is not None else 0

# Dynamic buildout values for text
bess_2026 = user_buildout.get(2026, 5.0)
bess_2030 = user_buildout.get(2030, 17.0)

# ════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════

render_header(
    title="German BESS Revenue Outlook 2026\u20132040",
    kicker="GERMAN BESS | REVENUE FORECAST",
    subtitle="An open-data revenue projection for grid-scale battery storage in Germany.",
)

st.markdown("""
Germany is building grid-scale battery storage at record pace — but what will
these assets actually earn over a 15-year project life? This interactive model
combines publicly available market data (wholesale prices, ancillary auction
results, fleet deployment statistics) to project BESS revenue across five market
segments from 2026 to 2040. Use the sidebar to stress-test the key assumptions
and explore how the revenue stack shifts under different scenarios.
""")

# ── KPIs ─────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("2026", f"€{r26['total']:.0f}k/MW" if r26 is not None else "—")
with col2:
    st.metric("2030", f"€{r30['total']:.0f}k/MW" if r30 is not None else "—")
with col3:
    st.metric(f"{floor_year} floor", f"€{r_floor['total']:.0f}k/MW" if r_floor is not None else "—")
with col4:
    st.metric("2040", f"€{r40['total']:.0f}k/MW" if r40 is not None else "—")

st.markdown("")

# ── Chart ────────────────────────────────────────────────────
st.plotly_chart(make_chart(proj_df), use_container_width=True, config={"displayModeBar": False})

render_chart_caption(
    f"Stacked revenue by market. Historical bars (2023\u20132025): "
    f"<a href='https://www.cleanhorizon.com/battery-index/'>Clean Horizon Battery Storage Index</a>. "
    f"Forecast (2026\u20132040): open-data model. {duration:.0f}h battery, 85% RTE, 2 cycles/day, "
    f"incl. fleet degradation with augmentation. "
    f"Shaded band = Bull/Bear range across all parameters (BESS buildout, gas, demand, PV, cannibalisation)."
)

# ── Article ──────────────────────────────────────────────────
st.markdown("---")

st.markdown(f"""
## Executive Summary

A {duration:.0f}-hour battery storage system in Germany currently earns around
**€{ch_2025:.0f}k per MW per year** across wholesale arbitrage and ancillary services
([Clean Horizon](https://www.cleanhorizon.com/battery-index/), 2025 average).
The model projects total revenue will fall to **~€{r30['total']:.0f}k by 2030** as
the BESS fleet scales from {bess_2026:.0f} GW to {bess_2030:.0f} GW, before stabilising at
**€{r_floor['total']:.0f}–{r40['total']:.0f}k** through 2040 as electricity demand growth
from electrification offsets storage cannibalisation.

The key investor question — *"will batteries still earn enough?"* — depends on
where you are on the build-out curve and which revenue streams you're counting on.
""")

# ── Section 1 ────────────────────────────────────────────────
st.markdown(f"""
## Where the money comes from today

German BESS revenue has five sources. In 2023–2025, the
[Clean Horizon Battery Storage Index](https://www.cleanhorizon.com/battery-index/)
recorded average annual revenues of **€{ch_2023:.0f}k** (2023), **€{ch_2024:.0f}k** (2024),
and **€{ch_2025:.0f}k** (2025) per MW for a {duration:.0f}h system:

| Revenue stream | What it is | Share in 2026 |
|:---|:---|---:|
| **Day-Ahead arbitrage** | Buy low, sell high on EPEX day-ahead auction | ~{r26['da']/r26['total']*100:.0f}% |
| **Intraday arbitrage** | Buy low, sell high on EPEX intraday continuous market | ~{r26['id']/r26['total']*100:.0f}% |
| **FCR** | Frequency containment (primary reserve) | ~{r26['fcr']/r26['total']*100:.0f}% |
| **aFRR capacity** | Automatic frequency restoration reserve | ~{r26['afrr_cap']/r26['total']*100:.0f}% |
| **aFRR energy** | Energy delivered when aFRR is activated | ~{r26['afrr_energy']/r26['total']*100:.0f}% |

**Today, ancillary services (FCR + aFRR) account for ~{anc_share_2026:.0f}%** of total revenue.
As the fleet scales, this picture will shift dramatically — the next section explains why.
""")

# ── Section 2 ────────────────────────────────────────────────
st.markdown(f"""
## The ancillary cliff: 2026–2030

Ancillary markets have a fixed size. FCR demand for the Denmark West / Germany
LFC block is **613 MW**
([FCR Cooperation, Demand per LFC block in 2026](https://eepublicdownloads.blob.core.windows.net/public-cdn-container/clean-documents/Network%20codes%20documents/NC%20EB/2025/Announcement_Demand_per_LFC_block_in_2026.pdf)). Combined FCR + aFRR capacity
addressable by batteries is approximately 4.5 GW.

With the BESS fleet projected to grow from **{bess_2026:.0f} GW (2026) to {bess_2030:.0f} GW (2030)**,
batteries will rapidly saturate these markets. The model projects ancillary revenue
per MW collapsing from **€{r26['fcr']+r26['afrr_cap']+r26['afrr_energy']:.0f}k (2026)
to €{r30['fcr']+r30['afrr_cap']+r30['afrr_energy']:.0f}k (2030)** — a
{(1 - (r30['fcr']+r30['afrr_cap']+r30['afrr_energy']) / max(r26['fcr']+r26['afrr_cap']+r26['afrr_energy'], 1)) * 100:.0f}% decline.

This is the single biggest risk for early-stage BESS investments that underwrite
revenue projections anchored to today's ancillary prices.
""")

# ── Section 3 ────────────────────────────────────────────────
wh_26 = r26['da'] + r26['id'] if r26 is not None else 0
wh_30 = r30['da'] + r30['id'] if r30 is not None else 0
wh_floor = r_floor['da'] + r_floor['id'] if r_floor is not None else 0
wh_40 = r40['da'] + r40['id'] if r40 is not None else 0

st.markdown(f"""
## Wholesale arbitrage: the long-term revenue base

While ancillary revenue is projected to decline sharply, wholesale arbitrage (DA + ID) {"holds up well" if wh_30 >= wh_26 * 0.85 else "declines more gradually"}
— **€{wh_26:.0f}k (2026)** {"→" if wh_30 != wh_26 else ""} **€{wh_30:.0f}k (2030)**,
then {"recovering to" if wh_40 > wh_floor else "stabilising around"}
**€{wh_floor:.0f}–{wh_40:.0f}k** through 2040.

Four forces are at work:

1. **Cannibalisation** (negative): more batteries compete for the same price spreads.
   The model projects this effect accelerating between 10–25 GW of installed fleet,
   then plateauing — marginal GW beyond ~30 GW have diminishing impact on spreads.

2. **Solar PV growth** (positive): Germany targets ~{pv_2040} GW of installed solar
   by 2040 (from ~100 GW today). More PV deepens the midday price trough ("duck curve"),
   creating wider daily spreads that batteries can arbitrage.

3. **Gas price** (structural): natural gas sets the marginal cost during evening peak hours.
   Higher gas prices → higher evening peaks → wider spreads. At
   [TTF](https://www.theice.com/products/27996665/Dutch-TTF-Gas-Futures)
   (Title Transfer Facility — the European gas benchmark) €{gas_2040}/MWh
   (2040), this is a {'+' if gas_2040 >= 35 else ''}{'tailwind' if gas_2040 >= 35 else 'headwind'}
   vs. today's ~€35/MWh.

4. **Demand growth** (positive): electrification of transport, heating, and industry
   is projected to push demand from ~600 to {demand_2040} TWh. Higher demand →
   higher peak prices → better spreads.

{"By ~" + str(floor_year) + ", demand growth overtakes cannibalisation, producing a modest **recovery to €" + f"{r40['total']:.0f}" + "k by 2040**." if r40['total'] > r_floor['total'] else "Under the current scenario, cannibalisation dominates through 2040."}
""")

# ── Section 4 ────────────────────────────────────────────────
st.markdown(f"""
## The revenue floor: €{r_floor['total']:.0f}–{r40['total']:.0f}k/MW — is it enough for positive project economics?

The model projects a floor of **~€{r_floor['total']:.0f}k/MW/year** around {floor_year}. This is the
"steady state" after ancillary revenues have been competed away and before demand
growth fully compensates.

For context:
- A 2h BESS at current CAPEX of ~€200k/MW needs roughly **€50–60k/MW/year** to
  cover debt service and return on equity over a 15-year life.
- At €{r_floor['total']:.0f}k, there is {"a comfortable margin" if r_floor['total'] >= 80 else "a margin, but it's thin" if r_floor['total'] >= 60 else "a challenging outlook"}. Projects banking on
  ancillary revenue post-2030 face significant downside risk.
- **Duration matters**: use the sidebar to switch between 1h, 2h, and 4h systems
  and see how the revenue stack changes.
""")

# ── Section 5 ────────────────────────────────────────────────
st.markdown(f"""
## Scenario sensitivity

Use the sidebar sliders to explore (sorted by impact on 2035 revenue):

| Scenario | Impact on 2035 revenue |
|:---|:---|
| Cannibalisation: Low → High | ~53% swing (dominant uncertainty) |
| BESS fleet 60 GW (vs {bess_2040:.0f} GW base) | ~15–20% lower wholesale |
| BESS fleet 20 GW | ~15% higher wholesale |
| Gas TTF €60/MWh (vs €30 base) | ~15% higher (wider peak spreads) |
| Gas TTF €15/MWh | ~10% lower (compressed peak prices) |
| Demand 1200 TWh (vs {demand_2040} base) | ~8% higher via stronger recovery |
| Solar PV 400 GW (vs {pv_2040} GW base) | ~6% higher (deeper duck curve) |

**Cannibalisation is the dominant uncertainty**, dwarfing all other scenario parameters.
The three scenarios in the sidebar range from optimistic (fitted on 1.5–6.8 GW DE+UK data)
to CAISO-observed (4→15 GW, revenue −60%). Use the scenario selector to see the full range.

The other big swing factor is **gas price**.
""")

# ── Section 6: Cannibalisation uncertainty ──────────────────

# Compute floor revenue for Low and High scenarios
_floor_scenarios = {}
for _sname, _sparams in {"Low": CANIB_SCENARIOS["Low (optimistic)"],
                          "High": CANIB_SCENARIOS["High (CAISO-scaled)"]}.items():
    _s_proj = project_full_stack(
        proj_years, baseline_da, user_buildout, duration,
        gas_2040=float(gas_2040), pv_2040_gw=float(pv_2040),
        demand_2040_twh=float(demand_2040), **_sparams,
    )
    _s_df = pd.DataFrame(_s_proj)
    _floor_scenarios[_sname] = _s_df["total"].min()

floor_low = _floor_scenarios["Low"]
floor_high = _floor_scenarios["High"]

st.markdown(f"""
## Cannibalisation uncertainty: what we know and what we don't

Cannibalisation is the model's dominant uncertainty — but **it cannot be observed yet.**
At the current fleet size (~3.5 GW in 2025), all three scenarios produce nearly identical
predictions. The scenarios only diverge meaningfully once the fleet exceeds ~10 GW
(expected ~2028), and even then, gas price swings can mask the signal.

**Why this uncertainty is irreducible right now:**
- Cannibalisation is a *structural* parameter — how much BESS fleet compresses DA spreads per GW added.
- It can only be measured when fleet growth is large enough to move prices (~10+ GW).
- Germany has never had 10 GW of BESS. The only market that has is CAISO (California) — hence the High scenario.
- Until the fleet grows past ~10 GW, all three scenarios predict revenue within ±10% of each other.

**What this means for investors:**
- Don't wait for clarity — it won't come before 2028–2029.
- Instead, evaluate whether the investment works **across all scenarios** (see sensitivity analysis above).
- The revenue floor (worst year across projection) ranges from
  ~€{floor_low:.0f}k/MW (Low) to ~€{floor_high:.0f}k/MW (High) — this is the range to underwrite against.
""")

# ── Scenario divergence chart ─────────────────────────────────
SCENARIO_COLORS = {
    "Low (optimistic)": "#22c55e",
    "Mid (merit order)": "#eab308",
    "High (CAISO-scaled)": "#ef4444",
}

# ── Chart: scenario fan over full projection horizon ──────────
def _styled_chart(fig, title, y_label, height=320):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=50, r=15, t=35, b=30),
        title=dict(text=title, font=dict(size=13, color="#14213d", family="Source Serif 4, serif")),
        xaxis=dict(tickfont=dict(size=9, color="#5c677d")),
        yaxis=dict(title=y_label, title_font=dict(size=10, color="#5c677d"), tickfont=dict(size=9, color="#5c677d")),
        legend=dict(
            orientation="h", y=-0.18, font=dict(size=9, color="#14213d"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
    )
    return fig

fig_fan = go.Figure()

# Shaded band between Low and High
low_vals = scen_annual["Low (optimistic)"]
high_vals = scen_annual["High (CAISO-scaled)"]
fig_fan.add_trace(go.Scatter(
    x=low_vals["year"], y=low_vals["total"],
    mode="lines", line=dict(width=0), showlegend=False,
))
fig_fan.add_trace(go.Scatter(
    x=high_vals["year"], y=high_vals["total"],
    mode="lines", line=dict(width=0), showlegend=False,
    fill="tonexty", fillcolor="rgba(20,33,61,0.06)",
))

# Scenario lines
for sname, sdf in scen_annual.items():
    is_active = sname == canib_scenario
    short = sname.split("(")[1].rstrip(")") if "(" in sname else sname
    fig_fan.add_trace(go.Scatter(
        x=sdf["year"], y=sdf["total"],
        name=short, mode="lines+markers",
        line=dict(
            color=SCENARIO_COLORS[sname],
            width=2.5 if is_active else 1.2,
        ),
        marker=dict(size=4 if is_active else 0),
        opacity=1.0 if is_active else 0.4,
    ))

# Mark the ~10 GW point where scenarios diverge
_yr_10gw = None
for y in sorted(user_buildout.keys()):
    if user_buildout[y] >= 10:
        _yr_10gw = y
        break
if _yr_10gw:
    fig_fan.add_vline(x=_yr_10gw, line=dict(color="#5c677d", width=1, dash="dot"))
    fig_fan.add_annotation(
        x=_yr_10gw, y=0.97, xref="x", yref="paper",
        text=f"~10 GW ({_yr_10gw})", showarrow=False,
        font=dict(size=9, color="#5c677d"),
    )

_fan_styled = _styled_chart(fig_fan, "Scenario fan: revenue range by cannibalisation assumption", "kEUR/MW/yr", height=380)
_fan_styled.update_xaxes(tickvals=[2025, 2028, 2031, 2034, 2037, 2040], tickangle=0)
st.plotly_chart(_fan_styled, use_container_width=True, config={"displayModeBar": False})
render_chart_caption(
    "Three cannibalisation scenarios over the full projection horizon. "
    "Shaded area = range of outcomes. Scenarios are indistinguishable until fleet passes ~10 GW. "
    "Switch scenarios in the sidebar to highlight each."
)

st.markdown(f"""
### Implications for investment decisions

Since the cannibalisation scenario **cannot be determined before ~{_yr_10gw or 2028}**,
the practical question is whether the investment works across the range:
""")

_col_lo, _col_hi, _col_txt = st.columns([1, 1, 2])
with _col_lo:
    _lo_label = "Comfortable" if floor_low >= 80 else "Workable" if floor_low >= 60 else "Tight"
    st.metric("Low (optimistic)", f"€{floor_low:.0f}k/MW", _lo_label, delta_color="off")
with _col_hi:
    _hi_label = "Comfortable" if floor_high >= 80 else "Workable" if floor_high >= 60 else "Challenging"
    st.metric("High (CAISO)", f"€{floor_high:.0f}k/MW", _hi_label, delta_color="off")
with _col_txt:
    st.markdown(f"""
**Key levers that don't depend on cannibalisation:**
- **Gas price** (±€20k/MW) — observable now, hedgeable
- **Fleet buildout speed** — observable via MaStR; see [Energy-Charts: Installed Power](https://www.energy-charts.info/charts/installed_power/chart.htm?l=en&c=DE) for a visual summary
- **Capacity market** — policy decision, not modelled (upside)
- **CAPEX decline** (~8%/yr) — improves IRR even if revenue falls
""")

st.markdown("---")

# ── Section 7: Fleet buildout scenarios ─────────────────────
st.markdown(f"""
## Fleet buildout: where 20–60 GW comes from

The BESS fleet slider (currently set to **{bess_2040:.0f} GW by 2040**) is the second-largest
driver of revenue outcomes after cannibalisation. Here is the logic behind the range.

### Near-term (~2025–2027): largely committed

Germany's grid-scale BESS fleet reached **~2.4 GW** by late 2025
([MaStR](https://www.marktstammdatenregister.de)). The visible pipeline — projects
with grid connection agreements and construction timelines — totals **~9.5 GW**,
with ~5.6 GW scheduled to commission through 2026–2027
([Modo Energy](https://modoenergy.com/research/en/de-germany-bess-batteries-buildout-capacity-growth-installation-grid-scale-february-2026)).
If delivered on time, the fleet would reach **~8 GW by Q4 2027**.

This near-term trajectory is not a scenario — it is an observable pipeline. The model
holds capacity through {COMMITTED_THROUGH} fixed regardless of the 2040 slider.

### The signal vs noise problem

Beyond the committed pipeline, the numbers become dramatic — and misleading:

| Metric | Value | What it means |
|:---|---:|:---|
| Installed fleet (late 2025) | ~2.4 GW | What actually exists |
| Pipeline (grid connection agreed) | ~9.5 GW | High confidence, 2–3 year horizon |
| Grid connections approved | ≥78 GW | BNetzA-confirmed, but many speculative |
| Grid connection *requests* | 500+ GW | Includes placeholder applications, ~6× peak demand |

The gap between 78 GW approved and 2.4 GW installed is the key uncertainty.
Most approved connections are speculative or conditional — developers secure grid access
early, then decide whether to build based on evolving economics. The **500+ GW of requests**
([pv magazine](https://www.pv-magazine.com/2025/09/02/germany-battery-storage-grid-connection-requests-exceed-500-gw/))
is a pipeline bubble, not a buildout forecast.

### The NEP anchor: 41–94 GW by 2037

The [Netzentwicklungsplan (NEP) 2037/2045](https://www.netzentwicklungsplan.de/en/nep-aktuell/netzentwicklungsplan-20372045-2025),
Version 2025, approved by BNetzA in April 2025, assumes **41–94 GW** of battery storage
by 2037 across scenarios A–C. This is a dramatic increase from the previous NEP
(2023 version: 24 GW by 2037).

The wide range reflects genuine uncertainty about electrification speed, grid expansion,
and the role storage plays relative to other flexibility options (demand response,
interconnectors, hydrogen).

### How the slider maps to these anchors

| Slider | Interpretation | Consistent with |
|:---|:---|:---|
| **20 GW** | Grid bottlenecks, permitting delays, slower electrification | Below NEP Scenario A — a stress test |
| **40 GW** (default) | Mid-range NEP trajectory, moderate delivery rate | NEP Scenario A/B lower bound |
| **60 GW** | Fast delivery, supportive regulation, CAPEX decline | NEP Scenario B/C range |

All three are plausible. The model default of 40 GW is deliberately conservative
relative to the NEP midpoint, reflecting historical experience that grid planning
targets tend to lead actual deployment.

### Buildout trajectories
""")

# ── Chart: fleet buildout trajectories (GW) ──────────────────
BUILDOUT_SCENARIOS = {"20 GW": 20, "40 GW": 40, "60 GW": 60}
BUILDOUT_COLORS = {"20 GW": "#22c55e", "40 GW": "#eab308", "60 GW": "#ef4444"}

# Historical grid-scale installed capacity (MaStR, >1 MW systems)
# Sources: Modo Energy buildout reports, ess-news.com, BNetzA MaStR
import datetime as _dt
_HIST_QUARTERLY_GW = [
    (_dt.date(2022, 1, 1), 0.55), (_dt.date(2022, 4, 1), 0.60),
    (_dt.date(2022, 7, 1), 0.70), (_dt.date(2022, 10, 1), 0.80),
    (_dt.date(2023, 1, 1), 0.90), (_dt.date(2023, 4, 1), 1.00),
    (_dt.date(2023, 7, 1), 1.15), (_dt.date(2023, 10, 1), 1.30),
    (_dt.date(2024, 1, 1), 1.45), (_dt.date(2024, 4, 1), 1.60),
    (_dt.date(2024, 7, 1), 1.80), (_dt.date(2024, 10, 1), 2.05),
    (_dt.date(2025, 1, 1), 2.20), (_dt.date(2025, 4, 1), 2.35),
    (_dt.date(2025, 7, 1), 2.40), (_dt.date(2025, 10, 1), 2.45),
    (_dt.date(2026, 1, 1), 2.80),
]
_hist_dates = [d for d, _ in _HIST_QUARTERLY_GW]
_hist_vals = [v for _, v in _HIST_QUARTERLY_GW]

fig_bo = go.Figure()

# Historical quarterly markers
fig_bo.add_trace(go.Scatter(
    x=_hist_dates, y=_hist_vals,
    name="Installed grid-scale (MaStR)", mode="lines+markers",
    line=dict(color="#14213d", width=2),
    marker=dict(size=5, color="#14213d", symbol="circle"),
))

# Convert forward years to dates for consistent x-axis
def _year_to_date(y):
    return _dt.date(y, 1, 1)

# Anchor: last historical point — forward lines start here
_last_hist_date, _last_hist_gw = _HIST_QUARTERLY_GW[-1]

# Build smooth forward trajectory: linear interpolation at quarterly resolution
def _quarterly_trajectory(buildout: dict[int, float]) -> list[tuple[_dt.date, float]]:
    """Interpolate annual buildout values to quarterly dates,
    anchored from the last historical observation."""
    # Anchor points: last observation + year-end values mapped to Dec 31
    anchors = [(_last_hist_date, _last_hist_gw)]
    for y in sorted(buildout):
        if y < 2026:
            continue
        anchors.append((_dt.date(y, 12, 31), buildout[y]))

    # Generate quarterly dates and interpolate linearly between anchors
    pts = []
    for i in range(len(anchors) - 1):
        d0, v0 = anchors[i]
        d1, v1 = anchors[i + 1]
        span = (d1 - d0).days
        # Emit quarterly points within this segment
        d = d0
        while d <= d1:
            frac = (d - d0).days / span if span > 0 else 0
            pts.append((d, v0 + (v1 - v0) * frac))
            # Advance ~3 months
            m = d.month + 3
            y = d.year + (m - 1) // 12
            m = (m - 1) % 12 + 1
            d = _dt.date(y, m, 1)
    # Add final anchor if not already included
    if pts[-1][0] != anchors[-1][0]:
        pts.append(anchors[-1])
    return pts

# Shaded band between 20 and 60 GW trajectories
_bo_20 = _scale_buildout(20)
_bo_60 = _scale_buildout(60)
_q_20 = _quarterly_trajectory(_bo_20)
_q_60 = _quarterly_trajectory(_bo_60)
fig_bo.add_trace(go.Scatter(
    x=[d for d, _ in _q_20], y=[v for _, v in _q_20],
    mode="lines", line=dict(width=0), showlegend=False,
))
fig_bo.add_trace(go.Scatter(
    x=[d for d, _ in _q_60], y=[v for _, v in _q_60],
    mode="lines", line=dict(width=0), showlegend=False,
    fill="tonexty", fillcolor="rgba(20,33,61,0.06)",
))

# Scenario lines
for label, target in BUILDOUT_SCENARIOS.items():
    bo = _scale_buildout(target)
    q_pts = _quarterly_trajectory(bo)
    is_active = target == bess_2040
    fig_bo.add_trace(go.Scatter(
        x=[d for d, _ in q_pts], y=[v for _, v in q_pts],
        name=label, mode="lines",
        line=dict(
            color=BUILDOUT_COLORS[label],
            width=2.5 if is_active else 1.2,
            dash="solid" if is_active else "dash",
        ),
        opacity=1.0 if is_active else 0.4,
    ))

# Committed pipeline marker
_committed_date = _year_to_date(COMMITTED_THROUGH)
fig_bo.add_vline(x=_committed_date, line=dict(color="#5c677d", width=1, dash="dot"))
fig_bo.add_annotation(
    x=_committed_date, y=0.97, xref="x", yref="paper",
    text=f"Committed ({COMMITTED_THROUGH})", showarrow=False,
    font=dict(size=9, color="#5c677d"),
)

_bo_styled = _styled_chart(fig_bo, "BESS fleet buildout scenarios (GW installed)", "GW", height=380)
_bo_styled.update_xaxes(
    tickvals=[_dt.date(y, 1, 1) for y in [2025, 2028, 2031, 2034, 2037, 2040]],
    ticktext=["2025", "2028", "2031", "2034", "2037", "2040"],
    tickangle=0,
)
st.plotly_chart(_bo_styled, use_container_width=True, config={"displayModeBar": False})
render_chart_caption(
    "Quarterly grid-scale installed capacity from MaStR (>1 MW systems). "
    f"Forward trajectories for 20 / 40 / 60 GW by 2040. "
    f"Capacity through {COMMITTED_THROUGH} is fixed (pipeline committed). "
    "Slider selection highlighted."
)

st.markdown(f"""
### What to watch

Fleet buildout is the **one uncertainty that resolves in real time**. Unlike cannibalisation
(which requires 10+ GW to measure), buildout progress is observable quarter by quarter via
[MaStR](https://www.marktstammdatenregister.de) and
[Energy-Charts](https://www.energy-charts.info/charts/installed_power/chart.htm?l=en&c=DE).
If you are tracking this market, the commissioning rate in 2026–2028 is the single most
informative leading indicator for long-term revenue.
""")

st.markdown("---")

st.markdown("""
## What this model does NOT include

- **Capacity market** — Germany does not currently have a capacity market for generation/storage
  (unlike UK, France, or Poland). The *Kraftwerksstrategie* (power plant strategy) discussions
  include a possible *Kapazitätsmechanismus*, but as of early 2026 neither the market design,
  eligibility rules, nor price levels have been defined. If introduced, this could add an
  estimated €5–15k/MW/yr — pure upside not reflected in this model.

- **Inertia / grid services** — Germany has no market or tariff for synthetic inertia
  (unlike Ireland's DS3 or Great Britain's stability pathfinder). The concept of
  *Momentanreserve* from inverter-based resources is under discussion in ENTSO-E and
  BNetzA grid code processes, but there is no timeline, no price signal, and no data
  to calibrate against. Until a concrete mechanism exists, this remains speculative upside.

- **Co-location with renewables** — co-locating BESS with solar/wind behind a shared grid
  connection point offers additional revenue streams (grid fee savings of ~5–8 kEUR/MW/yr,
  curtailment capture, *Direktvermarktung* optimisation). However, this is a fundamentally
  different business case (solar+storage vs standalone BESS) with project-specific economics
  that cannot be meaningfully generalised at market level.

- **Merchant vs contracted** — this models a fully merchant battery; PPA/tolling structures reduce volatility.
- **Technology cost declines** — we model revenue, not IRR. CAPEX is falling ~8%/yr.

These omissions are generally **upside risks** for investors — the model presents
a conservative, merchant-only revenue floor.
""")

st.markdown("---")

# ════════════════════════════════════════════════════════════
# Data & Methodology
# ════════════════════════════════════════════════════════════
st.markdown("""
## Data Sources & Methodology
""")

with st.expander("Wholesale model: empirically calibrated structural model"):
    st.markdown("""
The wholesale revenue projection uses a structural model with four drivers:

```
R_wholesale(t) = baseline
    + beta_D * (D_t / D_ref - 1)^1.5               (demand growth)
    - C_max / (1 + exp(-k * (B_t - B_half)))        (cannibalisation)
    + elasticity_gas * baseline * (TTF_t/TTF_ref-1)  (gas price)
    + sensitivity_pv * (PV_t - PV_ref) / 100         (solar duck curve)
```

**Two-stage calibration on observed data:**

*Stage 1 — Gas elasticity:* fitted on 6 annual observations of DE day-ahead price
spreads vs TTF gas prices (2020–2025). Gas varied from €9/MWh (2020) to €130/MWh (2022),
providing a wide identification range. **R² = 0.97**.

*Stage 2 — Remaining parameters:* fitted on a cross-country panel of 12 annual
observations: DE, UK, ES, IT revenue
([Clean Horizon](https://www.cleanhorizon.com/battery-index/) 2023–2025 for DE/ES/IT,
[Modo Energy](https://modoenergy.com/) 2023–2025 for UK post-DC era).
The UK data provides a cannibalization signal (fleet 3.5→6.8 GW). The ES/IT data
dramatically improves PV sensitivity identification: the panel covers PV from
16 GW (UK) to 100 GW (DE), with ES (25–35 GW) and IT (30–40 GW) filling the gap.

**Panel RMSE = 18.2 kEUR (~9% of mean revenue).** Full calibration code:
`validation/calibrate.py`.

| Parameter | Value | Source | Meaning |
|:---|---:|:---|:---|
| `baseline` | ~105 kEUR | Calibrated: CH total − structural ancillary | Wholesale revenue at reference conditions |
| `gas elasticity` | 0.405 | **Fitted**: DE spreads 2020–2025, R²=0.97 | When gas doubles, BESS spread revenue +40% |
| `PV sensitivity` | 33 kEUR/100GW | **Fitted**: DE+UK+ES+IT panel (16–100 GW) | Each 100 GW solar deepens duck curve |
| `beta_D` | 30.0 | **Fitted**: DE+UK+ES+IT panel | Demand growth → wider peak spreads |
| `C_max` | 15–50 kEUR | **Scenario** (see sidebar) | Max cannibalisation at fleet saturation |
| `B_half` | 10–25 GW | **Scenario** (see sidebar) | Fleet size at half-max cannibalisation |
| `k` | 0.15–0.80 | **Scenario** (see sidebar) | Cannibalisation steepness |

**Cannibalisation scenarios** (selectable in sidebar):
- **Low (optimistic):** `C_max`=15, `B_half`=25 GW. Fitted on DE+UK panel (1.5–6.8 GW only).
  Optimistic extrapolation to 40 GW — at 30 GW, only -14% wholesale.
- **Mid (merit order):** `C_max`=33, `B_half`=13 GW. Derived from merit order simulation:
  fleet BESS shifts supply/demand during charge/discharge hours, compressing spreads.
  Simulated on 2024 DE DA prices with slope ≈ 1.5 EUR/MWh/GW. At 30 GW, -34% wholesale.
- **High (CAISO-scaled):** `C_max`=82, `B_half`=11 GW. Fitted on
  [CAISO](https://www.caiso.com/documents/2024-special-report-on-battery-storage-may-29-2025.pdf)
  observed revenue decline: $103/kW (2022, ~4 GW) → $78/kW (2023, ~8 GW) → $53/kW (2024, ~13 GW).
  Scaled to DE by fleet/peak-demand ratio (DE 80 GW peak vs CAISO 52 GW). This is the only market
  with observed cannibalization at 10+ GW fleet sizes.

**Other limitations:**
- PV sensitivity is now better identified via ES/IT cross-section (16–100 GW range),
  but country baselines absorb market-structure differences, so the PV effect
  is identified from within-country time variation + cross-sectional level differences.
- The model captures structural drivers but not seasonal or weather effects.

Wholesale revenue is split DA/ID using a hardcoded ratio (not LP dispatch on ID prices).
The ratio was calibrated by comparing LP dispatch on DA hourly prices (85–99 kEUR/MW) vs
ID 15-min prices (108–114 kEUR/MW, 2023–2025), then adjusted for real-world trading patterns
(operators trade ~65% DA, ~35% ID). The ratio trends from 0.50 (2026) → 0.65 (2040),
reflecting expected RE-driven growth in intraday volatility. This forward trend is an
assumption, not a calibrated parameter.
""")

with st.expander("Ancillary model: exponential saturation per component"):
    st.markdown("""
Each ancillary stream (FCR, aFRR capacity, aFRR energy) follows an exponential
decay calibrated to two anchor points derived from
[regelleistung.net](https://www.regelleistung.net/) historical auction data
and ancillary market depth constraints (FCR ~0.6 GW, aFRR ~4 GW):

```
R_component(B) = floor + A * exp(-k * B)
```

Calibration anchors (for a 2h battery):

| Component | At 5 GW (2026) | At 17 GW (2030) | Floor |
|:---|---:|---:|---:|
| FCR | 8 kEUR | 2 kEUR | 0 |
| aFRR capacity | 115 kEUR | 8 kEUR | 2 kEUR |
| aFRR energy | 12 kEUR | 3 kEUR | 0.5 kEUR |

Total ancillary market depth: ~4.5 GW (FCR ~0.6 GW + aFRR ~4 GW).

**Duration scaling**: FCR and aFRR are auctioned in 4-hour blocks. A 1h battery
can only participate approximately 50% of the time (insufficient energy buffer
to sustain delivery), while 2h+ batteries can participate fully.
""")

with st.expander("Dispatch model: LP for DA revenue estimation (not projection baseline)"):
    st.markdown("""
**Important:** The LP dispatch model is used **only for estimating DA arbitrage
revenue** on historical prices — it does NOT set the projection baseline.

The projection baseline comes from
[Clean Horizon](https://www.cleanhorizon.com/battery-index/) observed revenue
(real market outcomes, not theoretical optimum).

**How the DA/ID split actually works:**

The DA/ID revenue split in the projection uses a **hardcoded ratio** (`id_da_ratio()`),
not LP dispatch on ID prices. The ratio trends from 0.50 (2026) to 0.65 (2040),
reflecting the expectation that RE growth increases intraday volatility and thus
the ID share of wholesale revenue.

This ratio was calibrated by comparing LP-optimal dispatch on:
- DA hourly prices ([Energy-Charts](https://energy-charts.info/)) → 85–99 kEUR/MW/yr (2023–2025)
- ID 15-min ID-AEP prices ([Netztransparenz](https://www.netztransparenz.de/),
  capped at ±150 €/MWh) → 108–114 kEUR/MW/yr (2023–2025)

Under perfect-foresight LP, the ID market appears ~20–30% more valuable than DA
(wider 15-min spreads). However, real operators trade primarily on DA (~65%) with
ID for position adjustment (~35%), because perfect foresight on 15-min prices is
unrealistic — the LP upper bound overstates achievable ID revenue more than DA.
The hardcoded ratio reflects this real-world trading pattern, not LP optimality.

**Limitation:** the forward trend (ID share rising to 65% by 2040) is an assumption,
not a calibrated parameter. It could be wrong in either direction.

**LP formulation (for DA revenue estimation):**

For each day, the battery is a price-taker solving:

```
maximise  Σ_t  p_t · (d_t − c_t) · Δt

subject to:
  0 ≤ SoC_t ≤ E_max           (state of charge within capacity)
  SoC_T = SoC_0               (return to starting SoC)
  Σ d_t · Δt ≤ cycles · E_max (cycle budget per day)
  0 ≤ c_t, d_t ≤ P_rated      (power limits)
```

where `c_t` = charge power, `d_t` = discharge power, `p_t` = price in slot `t`,
`Δt` = slot duration (1h for hourly DA), and efficiency losses are
applied as `η_c = η_d = √RTE` (split equally between charge and discharge).

**Key properties:**
- **Perfect foresight** — the LP sees all prices for the day. This produces an upper
  bound on achievable revenue (~80–90% capture rate in practice). Since the projection
  uses Clean Horizon observed revenue (not LP output), this bias does not affect the forecast.
- **2 cycles/day** — discharge energy ≤ 2 × E_max per day (hardcoded).
- **Solver:** `scipy.optimize.linprog` with HiGHS backend.

**Data inputs:**
- DA prices: [Energy-Charts API](https://energy-charts.info/) hourly prices (DE-LU bidding zone)
- ID-AEP prices: [Netztransparenz](https://www.netztransparenz.de/) 15-min settlement prices
  (cached locally; live fetch requires OAuth2 credentials via `NTP_CLIENT_ID` / `NTP_CLIENT_SECRET`)
""")

with st.expander("Historical revenue: Clean Horizon Battery Storage Index"):
    st.markdown("""
Historical bars use the [Clean Horizon Battery Storage Index](https://www.cleanhorizon.com/battery-index/)
— a publicly available monthly index of annualised gross revenue for 1h, 2h, and 4h
battery systems in Germany. It covers DA, ID, FCR, and aFRR combined.

The market-level breakdown shown in the chart is estimated proportionally from:
- DA arbitrage dispatch on [Energy-Charts](https://energy-charts.info/) hourly prices (Fraunhofer ISE)
- ID-AEP dispatch on [Netztransparenz](https://www.netztransparenz.de/) 15-min prices (TSO consortium)
- FCR/aFRR capacity prices from [regelleistung.net](https://www.regelleistung.net/) auction data

Component proportions are normalised so they sum to the Clean Horizon total.
""")

with st.expander("Battery degradation & augmentation model"):
    st.markdown("""
All projections include a **cohort-based degradation model**:

- Each vintage year (year of commissioning) is tracked as a separate cohort.
- Each cohort loses ~1.8% effective capacity per year (~0.25% per 100 full cycles
  at 2 cycles/day).
- After approximately 11–12 years of operation (~8,500 cycles), cell augmentation
  restores capacity to ~92% of nameplate.
- Fleet-average degradation = MW-weighted mean across all active cohorts.
- Minimum capacity floor: 50% of nameplate (end-of-life threshold).

By 2040, the fleet-average capacity factor is ~89%, meaning revenue per nameplate MW
is ~11% lower than per effective MW. This is applied to both wholesale and ancillary
revenue in the projection.

See `config.py: fleet_degradation_factor()` for implementation.
""")

with st.expander("Full data source table"):
    prov_data = [
        {"#": "1", "Input": "DA hourly prices DE-LU",
         "Source": "[Energy-Charts API](https://api.energy-charts.info/price?bzn=DE-LU) (Fraunhofer ISE)",
         "Confidence": "Direct measurement"},
        {"#": "2", "Input": "ID-AEP 15-min prices",
         "Source": "[Netztransparenz](https://ds.netztransparenz.de/api/v1/) (TSO consortium)",
         "Confidence": "Direct measurement"},
        {"#": "3", "Input": "Historical total revenue (DE/ES/IT)",
         "Source": "[Clean Horizon Battery Storage Index](https://www.cleanhorizon.com/battery-index/)",
         "Confidence": "Direct measurement"},
        {"#": "4", "Input": "FCR auction results",
         "Source": "[regelleistung.net Datacenter](https://www.regelleistung.net/apps/datacenter/tenders/)",
         "Confidence": "Direct measurement"},
        {"#": "5", "Input": "aFRR auction results",
         "Source": "[regelleistung.net Datacenter](https://www.regelleistung.net/apps/datacenter/tenders/)",
         "Confidence": "Direct measurement"},
        {"#": "6", "Input": "BESS installed capacity",
         "Source": "[Marktstammdatenregister](https://www.marktstammdatenregister.de) (BNetzA)",
         "Confidence": "Direct measurement"},
        {"#": "7", "Input": f"BESS buildout (2024–2040): {bess_2026:.0f} to {bess_2040} GW",
         "Source": "[NEP](https://www.netzentwicklungsplan.de/) / [BNetzA](https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/Netzentwicklung/start.html) grid development plan",
         "Confidence": "Scenario assumption"},
        {"#": "8", "Input": "RE generation (2026–2040): 280 to 695 TWh",
         "Source": "[Energy-Charts](https://energy-charts.info/) / [EEG 2023 §4](https://www.gesetze-im-internet.de/eeg_2014/__4.html)",
         "Confidence": "Policy target"},
        {"#": "9", "Input": f"Demand (2026–2040): 600 to {demand_2040} TWh",
         "Source": "[dena Leitstudie](https://www.dena.de/fileadmin/dena/Publikationen/PDFs/2021/Abschlussbericht_dena-Leitstudie_Aufbruch_Klimaneutralitaet.pdf) / [Agora Energiewende](https://www.agora-energiewende.de/data-tools/agorameter)",
         "Confidence": "Scenario assumption"},
        {"#": "10", "Input": f"Gas price TTF (2026–2040): 35 to {gas_2040} €/MWh",
         "Source": "[ICE TTF](https://www.theice.com/products/27996665/Dutch-TTF-Gas-Futures) / [EEX Power Derivatives](https://www.eex.com/en/market-data/power/futures)",
         "Confidence": "Scenario assumption"},
        {"#": "11", "Input": f"Solar PV (2026–2040): 100 to {pv_2040} GW",
         "Source": "[Marktstammdatenregister](https://www.marktstammdatenregister.de) / [EEG 2023 §4](https://www.gesetze-im-internet.de/eeg_2014/__4.html)",
         "Confidence": "Policy target"},
        {"#": "12", "Input": "ES/IT revenue for PV sensitivity calibration",
         "Source": "[Clean Horizon](https://www.cleanhorizon.com/battery-index/) + [REE](https://www.ree.es/) / [Terna](https://www.terna.it/) for PV/BESS fleet",
         "Confidence": "Cross-validation"},
    ]
    header = "| # | Input | Source | Confidence |\n|---|-------|--------|------------|\n"
    rows = "\n".join(
        f"| {r['#']} | {r['Input']} | {r['Source']} | {r['Confidence']} |"
        for r in prov_data
    )
    st.markdown(header + rows, unsafe_allow_html=True)

# ── Series anchor & next-note hook ────────────────────────────
st.markdown("---")
render_closing(
    "This is the first note in a series on BESS merchant economics — "
    "from market opportunity and cycling trade-offs to degradation, "
    "warranty economics, and owner governance."
)
st.markdown(
    '<div style="margin-top: 0.5rem; font-size: 0.95rem; color: #666;">'
    "<b>Next:</b> the revenue pool is there — but how many cycles does it "
    "actually take to capture most of it, and at what point does an extra "
    "cycle stop paying for itself?"
    "</div>",
    unsafe_allow_html=True,
)

render_footer()
