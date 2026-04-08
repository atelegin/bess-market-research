"""
Note 2 — How Many Cycles Does a Battery Actually Need?

Streamlit article on wholesale cycling requirements, marginal value per cycle,
and the market saturation trend.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.optimize import curve_fit

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

PRECOMPUTED_PATH = Path(__file__).parent / "data" / "precomputed.pkl"

# ── Colours ──────────────────────────────────────────────────
YEAR_COLORS = {
    2021: "#94a3b8",
    2022: "#f87171",
    2023: "#fbbf24",
    2024: "#3b82f6",
    2025: "#14213d",
}


def styled_layout(fig, height=400, y_title="", x_title=""):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=55, r=20, t=30, b=50),
        xaxis=dict(
            title=x_title,
            title_font=dict(size=11, color="#5c677d"),
            tickfont=dict(size=10, color="#5c677d"),
        ),
        yaxis=dict(
            title=y_title,
            title_font=dict(size=11, color="#5c677d"),
            tickfont=dict(size=10, color="#5c677d"),
        ),
        legend=dict(
            orientation="h", y=-0.18,
            font=dict(size=10, color="#14213d"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
    )
    return fig


# ── Load data ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    with open(PRECOMPUTED_PATH, "rb") as f:
        return pickle.load(f)


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="How Many Cycles Does a Battery Actually Need?",
    page_icon="🔋",
    layout="wide",
)
apply_theme(show_sidebar=False)

# ── Scenario defaults (Mid cannibalisation) ───────────────────
from lib.config import DEFAULT_BESS_BUILDOUT as BUILDOUT
from lib.models.projection import project_full_stack


CANIB_MID = {"canib_max": 33.0, "canib_half": 13.0, "canib_steep": 0.17}
CANIB_LOW = {"canib_max": 15.0, "canib_half": 25.0, "canib_steep": 0.80}
CANIB_HIGH = {"canib_max": 82.0, "canib_half": 11.0, "canib_steep": 0.18}
GAS_DEFAULT = 30       # €/MWh TTF
PV_DEFAULT = 300       # GW installed PV
DEMAND_DEFAULT = 1020  # TWh

# Bull: low competition, expensive gas, high demand, low cannibalisation
BULL_PARAMS = dict(bess_2040=20, gas_2040=80, demand_2040=1200, pv_2040=215, **CANIB_LOW)
# Bear: crowded market, cheap gas, low demand, high cannibalisation
BEAR_PARAMS = dict(bess_2040=60, gas_2040=15, demand_2040=700, pv_2040=400, **CANIB_HIGH)

COMMITTED_THROUGH = 2027

NOTE1_URL = "https://de-bess-outlook.streamlit.app/"
NOTE1_TITLE = "Revenue Outlook"


def _scale_buildout(target_2040: float) -> dict[int, float]:
    committed_val = BUILDOUT[COMMITTED_THROUGH]
    default_2040 = BUILDOUT[2040]
    out = {}
    for y, v in BUILDOUT.items():
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


def _project_revenue(buildout, gas=GAS_DEFAULT, pv=PV_DEFAULT, demand=DEMAND_DEFAULT,
                     canib=None):
    """Run note 1 projection model and return {year: dict} map."""
    canib_params = canib if canib is not None else CANIB_MID
    stack = project_full_stack(
        years=list(range(2026, 2041)),
        historical_da_keur=105.0,
        bess_buildout=buildout,
        duration_h=2.0,
        demand_2040_twh=float(demand),
        gas_2040=float(gas),
        pv_2040_gw=float(pv),
        **canib_params,
    )
    return {r["year"]: r for r in stack}


# ── Load precomputed data ───────────────────────────────────
data = load_data()
frontiers = data["frontiers"]
YEARS = data["years"]
DURATIONS = data["durations"]
RTE = data["rte"]
fleet_gw = data["fleet_gw"]

# Duration will be selected by user — filtered below after page config

# Q1 2026 annualised values (pre-computed from explore runs, 2h base)
# For 1h and 4h: scaled from 2h using 2025 duration ratios from precomputed data.
Q1_2026_GW = 4.5
# Q1 2026 dispatch results per duration (DA+ID overlay, annualised from 90 days).
_Q1_2026_BY_DUR = {
    1.0: {"rev": {1.0: 51, 2.0: 66, 3.0: 70, "max": 70},
          "fec": {"1c": 329, "max": 944}, "max_cpd": 3.75},
    2.0: {"rev": {1.0: 84, 2.0: 100, 3.0: 102, "max": 102},
          "fec": {"1c": 328, "max": 675}, "max_cpd": 2.75},
    4.0: {"rev": {1.0: 129, 2.0: 140, 3.0: 140, "max": 140},
          "fec": {"1c": 321, "max": 444}, "max_cpd": 2.0},
}

# ── Ancillary cycling (FCR + aFRR) per year ─────────────────
ANCILLARY_CPD = {
    2021: (0.25, 0.50), 2022: (0.25, 0.48),
    2023: (0.25, 0.45), 2024: (0.25, 0.40),
    2025: (0.25, 0.31), 2026: (0.25, 0.25),
}
ANCILLARY_TEAL = "#2a9d8f"
ANCILLARY_FCR_COLOR = "#e9c46a"

# aFRR energy net spread revenue (kEUR/MW/yr, incremental over wholesale)
AFRR_ENERGY_NET_KEUR = {2023: 20, 2024: 10, 2025: -5}

# aFRR activation data
AFRR_YEARLY = {
    2023: {"avg_pos_mw": 68.6, "avg_neg_mw": 80.9, "total_twh": 1.31,
           "fec_per_day": 0.45, "annual_fec": 164, "contracted_pos": 2000, "contracted_neg": 1800},
    2024: {"avg_pos_mw": 62.6, "avg_neg_mw": 62.3, "total_twh": 1.10,
           "fec_per_day": 0.40, "annual_fec": 145, "contracted_pos": 2000, "contracted_neg": 1800},
    2025: {"avg_pos_mw": 45.4, "avg_neg_mw": 59.3, "total_twh": 0.92,
           "fec_per_day": 0.31, "annual_fec": 114, "contracted_pos": 2000, "contracted_neg": 1800},
}


# ── Helper ───────────────────────────────────────────────────
def fec_at_pct(frontier_subset, pct):
    max_rev = frontier_subset["annual_revenue_eur_per_mw"].max()
    target = max_rev * pct / 100
    above = frontier_subset[frontier_subset["annual_revenue_eur_per_mw"] >= target]
    return above.iloc[0]["annual_fec"] if not above.empty else frontier_subset.iloc[-1]["annual_fec"]


# ════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════

render_header(
    title="How Many Cycles Does a Battery Actually Need?",
    kicker="GERMAN BESS | CYCLING INTENSITY",
    subtitle=(
        "The market assumes cycling and revenue go hand in hand. "
        "The data says they don't."
    ),
)

st.markdown(f"""
This is the second note in a series on German BESS merchant economics.
The [first note]({NOTE1_URL}) projected revenue through 2040; this one asks
how hard a battery has to work to capture it.
""")

render_standfirst(
    "Ask a battery trader how many cycles they need and the answer is "
    "<em>as many as the warranty allows</em>. The assumption behind most BESS "
    "investment models is the same: more cycles means more revenue. "
    "This note tests it against five years of German wholesale prices."
    "\n\n"
    "The short answer: a German battery today needs about 1.5 cycles per "
    "day to capture most of the available wholesale revenue — well inside "
    "standard warranty limits. And the cycling rate is falling: as the fleet "
    "grows, the market offers fewer profitable windows. By 2030, total "
    "cycling is projected below 1 cycle per day."
)

st.markdown("---")


# ════════════════════════════════════════════════════════════════
#  HERO CHART: Revenue + cycles (history + projection)
# ════════════════════════════════════════════════════════════════

def _rev_and_cpd_at_pct(yf, pct, days):
    """Interpolate revenue and cycles/day for a given capture percentage."""
    max_rev = yf["annual_revenue_eur_per_mw"].max()
    target_rev = max_rev * pct / 100
    sorted_yf = yf.sort_values("annual_revenue_eur_per_mw")
    below = sorted_yf[sorted_yf["annual_revenue_eur_per_mw"] <= target_rev]
    above = sorted_yf[sorted_yf["annual_revenue_eur_per_mw"] >= target_rev]
    if below.empty:
        row = above.iloc[0]
        return row["annual_revenue_eur_per_mw"] / 1000, row["annual_fec"] / days
    if above.empty:
        row = below.iloc[-1]
        return row["annual_revenue_eur_per_mw"] / 1000, row["annual_fec"] / days
    lo = below.iloc[-1]
    hi = above.iloc[0]
    if lo["annual_revenue_eur_per_mw"] == hi["annual_revenue_eur_per_mw"]:
        return lo["annual_revenue_eur_per_mw"] / 1000, lo["annual_fec"] / days
    frac = ((target_rev - lo["annual_revenue_eur_per_mw"])
            / (hi["annual_revenue_eur_per_mw"] - lo["annual_revenue_eur_per_mw"]))
    fec = lo["annual_fec"] + frac * (hi["annual_fec"] - lo["annual_fec"])
    return target_rev / 1000, fec / days


def _power_law(x, a, b):
    return a * np.power(x, b)


# ── Duration helpers — each chart gets its own pills, synced via session_state ──
_DUR_OPTIONS = [1.0, 2.0, 4.0]

if "_dur" not in st.session_state:
    st.session_state["_dur"] = 2.0

def _dur_sync(source_key):
    """Callback: propagate the changed pills value to all other duration keys."""
    val = st.session_state[source_key]
    if val is None:
        return
    st.session_state["_dur"] = val
    for k in ("dur_hero", "dur_stacked", "dur_lifetime"):
        if k != source_key:
            st.session_state[k] = val

def _dur_pills(label, key):
    """Render a duration pills widget and return (selected_duration, f2h, Q1_2026 vars)."""
    render_chart_title(label)
    dur = st.pills(
        "Battery duration", options=_DUR_OPTIONS,
        default=st.session_state.get(key, st.session_state["_dur"]),
        format_func=lambda d: f"{int(d)}h battery",
        label_visibility="collapsed",
        key=key, on_change=_dur_sync, args=(key,),
    )
    if dur is None:
        dur = 2.0
    filt = frontiers[frontiers["duration_h"] == dur]
    q = _Q1_2026_BY_DUR[dur]
    return dur, filt, q["rev"], q["fec"], q["max_cpd"]


# First chart needs values early for computations — read from session state
selected_duration = st.session_state["_dur"]
f2h = frontiers[frontiers["duration_h"] == selected_duration]
_q26 = _Q1_2026_BY_DUR[selected_duration]
Q1_2026_REV = _q26["rev"]
Q1_2026_FEC = _q26["fec"]
Q1_2026_MAX_CPD = _q26["max_cpd"]

# ── Historical data ─────────────────────────────────────────────
hero_years_with_2026 = list(YEARS) + [2026]

hero_rev_max = []
for year in YEARS:
    yf = f2h[f2h["year"] == year]
    hero_rev_max.append(yf["annual_revenue_eur_per_mw"].max() / 1000)
hero_rev_max.append(Q1_2026_REV["max"])

hero_fcr_cpd = [ANCILLARY_CPD[y][0] for y in hero_years_with_2026]
hero_afrr_cpd = [ANCILLARY_CPD[y][1] for y in hero_years_with_2026]
hero_ancillary_cpd = [a + f for a, f in zip(hero_afrr_cpd, hero_fcr_cpd)]

# Precompute wholesale cpd for every capture % (historical years)
PCT_RANGE = list(range(70, 101))
all_rev = {}
all_cpd = {}
for pct in PCT_RANGE:
    revs, cpds = [], []
    for year in YEARS:
        yf = f2h[f2h["year"] == year]
        days = 366 if year % 4 == 0 else 365
        r, c = _rev_and_cpd_at_pct(yf, pct, days)
        revs.append(r)
        cpds.append(c)
    q26_pts = [
        (Q1_2026_REV[1.0], Q1_2026_FEC["1c"]),
        (Q1_2026_REV[2.0], Q1_2026_FEC["1c"] + 200),
        (Q1_2026_REV[3.0], Q1_2026_FEC["max"]),
    ]
    q26_target = Q1_2026_REV["max"] * pct / 100
    if q26_target <= q26_pts[0][0]:
        frac = q26_target / q26_pts[0][0] if q26_pts[0][0] > 0 else 0
        revs.append(q26_target)
        cpds.append(frac * q26_pts[0][1] / 365)
    elif q26_target >= q26_pts[-1][0]:
        revs.append(float(Q1_2026_REV["max"]))
        cpds.append(Q1_2026_MAX_CPD)
    else:
        for i in range(len(q26_pts) - 1):
            if q26_pts[i][0] <= q26_target <= q26_pts[i + 1][0]:
                frac = ((q26_target - q26_pts[i][0])
                        / (q26_pts[i + 1][0] - q26_pts[i][0]))
                fec = q26_pts[i][1] + frac * (q26_pts[i + 1][1] - q26_pts[i][1])
                revs.append(q26_target)
                cpds.append(fec / 365)
                break
    all_rev[pct] = revs
    all_cpd[pct] = cpds

# ── Half-yearly data for power-law fit ──────────────────────────
hy_frontiers = data.get("half_yearly_frontiers")
hy2h = hy_frontiers[hy_frontiers["duration_h"] == 2.0] if hy_frontiers is not None else None

# Build fleet_gw at H1/H2 midpoints (linear interpolation between year-end values)
_sorted_years = sorted(fleet_gw.keys())
_hy_gw = {}  # (year, "H1"/"H2") -> GW
for yr in YEARS:
    prev_yr = yr - 1
    gw_start = fleet_gw.get(prev_yr, fleet_gw[_sorted_years[0]] * 0.7)
    gw_end = fleet_gw[yr]
    _hy_gw[(yr, "H1")] = gw_start + (gw_end - gw_start) * 0.25  # mid-Q1/Q2
    _hy_gw[(yr, "H2")] = gw_start + (gw_end - gw_start) * 0.75  # mid-Q3/Q4

# Build half-yearly cpd at each capture % (for 2h)
hy_cpd = {}  # pct -> list of cpd values
hy_gw_list = []  # parallel list of GW values
_hy_labels = []  # for reference
if hy2h is not None:
    for yr in YEARS:
        for h in ["H1", "H2"]:
            hy_gw_list.append(_hy_gw[(yr, h)])
            _hy_labels.append(f"{yr} {h}")
    # Add Q1 2026
    hy_gw_list.append(Q1_2026_GW)
    _hy_labels.append("2026 H1")

    for pct in PCT_RANGE:
        cpds_hy = []
        for yr in YEARS:
            for h in ["H1", "H2"]:
                hyf = hy2h[(hy2h["year"] == yr) & (hy2h["half"] == h)]
                if hyf.empty:
                    cpds_hy.append(np.nan)
                    continue
                days = 365  # already annualised in precompute
                _, c = _rev_and_cpd_at_pct(hyf, pct, days)
                cpds_hy.append(c)
        # Append Q1 2026 (same as before)
        q26_pts = [
            (Q1_2026_REV[1.0], Q1_2026_FEC["1c"]),
            (Q1_2026_REV[2.0], Q1_2026_FEC["1c"] + 200),
            (Q1_2026_REV[3.0], Q1_2026_FEC["max"]),
        ]
        q26_target = Q1_2026_REV["max"] * pct / 100
        if q26_target <= q26_pts[0][0]:
            frac = q26_target / q26_pts[0][0] if q26_pts[0][0] > 0 else 0
            cpds_hy.append(frac * q26_pts[0][1] / 365)
        elif q26_target >= q26_pts[-1][0]:
            cpds_hy.append(Q1_2026_MAX_CPD)
        else:
            for i in range(len(q26_pts) - 1):
                if q26_pts[i][0] <= q26_target <= q26_pts[i + 1][0]:
                    frac = ((q26_target - q26_pts[i][0])
                            / (q26_pts[i + 1][0] - q26_pts[i][0]))
                    fec = q26_pts[i][1] + frac * (q26_pts[i + 1][1] - q26_pts[i][1])
                    cpds_hy.append(fec / 365)
                    break
        hy_cpd[pct] = cpds_hy

# ── Power-law fit for projection (half-yearly points) ──────────
_fit_gw = np.array(hy_gw_list) if hy_gw_list else np.array([fleet_gw[y] for y in YEARS] + [Q1_2026_GW])
_fit_cpd90 = np.array(hy_cpd.get(90, all_cpd[90]))

# Remove NaN
_valid = ~np.isnan(_fit_cpd90)
_fit_gw_clean = _fit_gw[_valid]
_fit_cpd90_clean = _fit_cpd90[_valid]

_popt90, _pcov90 = curve_fit(_power_law, _fit_gw_clean, _fit_cpd90_clean, p0=[2.0, -0.3])

# Confidence band: compute prediction interval from residuals
_residuals = _fit_cpd90_clean - _power_law(_fit_gw_clean, *_popt90)
_rmse = float(np.sqrt(np.mean(_residuals**2)))
_n_fit_points = len(_fit_gw_clean)

# aFRR power-law (declines with fleet)
_afrr_gw = np.array([3.0, 3.5, 4.0])
_afrr_cpd_fit = np.array([0.45, 0.40, 0.31])
_afrr_popt, _ = curve_fit(_power_law, _afrr_gw, _afrr_cpd_fit, p0=[1.0, -0.5])
FCR_CPD_CONST = 0.25

# ── Fleet buildout (fixed at 40 GW base case) ─────────────────
BESS_2040_DEFAULT = 40
user_buildout = _scale_buildout(BESS_2040_DEFAULT)
proj_rev_by_year = _project_revenue(user_buildout)

proj_years = list(range(2027, 2041))
proj_gw = [user_buildout.get(y, user_buildout[max(k for k in user_buildout if k <= y)]) for y in proj_years]

# ── Bull/Bear buildouts & revenue projections ──────────────────
bull_buildout = _scale_buildout(BULL_PARAMS["bess_2040"])
bear_buildout = _scale_buildout(BEAR_PARAMS["bess_2040"])
bull_rev_by_year = _project_revenue(
    bull_buildout, gas=BULL_PARAMS["gas_2040"],
    pv=BULL_PARAMS["pv_2040"], demand=BULL_PARAMS["demand_2040"],
    canib={k: BULL_PARAMS[k] for k in ("canib_max", "canib_half", "canib_steep")},
)
bear_rev_by_year = _project_revenue(
    bear_buildout, gas=BEAR_PARAMS["gas_2040"],
    pv=BEAR_PARAMS["pv_2040"], demand=BEAR_PARAMS["demand_2040"],
    canib={k: BEAR_PARAMS[k] for k in ("canib_max", "canib_half", "canib_steep")},
)
bull_proj_gw = [bull_buildout.get(y, bull_buildout[max(k for k in bull_buildout if k <= y)]) for y in proj_years]
bear_proj_gw = [bear_buildout.get(y, bear_buildout[max(k for k in bear_buildout if k <= y)]) for y in proj_years]

# Project wholesale c/d for each capture % (using half-yearly fit)
all_cpd_proj = {}        # base-case
all_cpd_proj_bull = {}   # bull scenario (low fleet)
all_cpd_proj_bear = {}   # bear scenario (high fleet)
for pct in PCT_RANGE:
    _fit_vals = np.array(hy_cpd.get(pct, all_cpd[pct]))
    _v = ~np.isnan(_fit_vals)
    _gw_v = _fit_gw[_v]
    _cpd_v = _fit_vals[_v]
    popt_pct, _ = curve_fit(_power_law, _gw_v, _cpd_v, p0=[2.0, -0.3])
    all_cpd_proj[pct] = [_power_law(gw, *popt_pct) for gw in proj_gw]
    all_cpd_proj_bull[pct] = [_power_law(gw, *popt_pct) for gw in bull_proj_gw]
    all_cpd_proj_bear[pct] = [_power_law(gw, *popt_pct) for gw in bear_proj_gw]

# Project ancillary cycling (base / bull / bear)
afrr_proj = [_power_law(gw, *_afrr_popt) for gw in proj_gw]
ancillary_proj = [a + FCR_CPD_CONST for a in afrr_proj]
afrr_proj_bull = [_power_law(gw, *_afrr_popt) for gw in bull_proj_gw]
ancillary_proj_bull = [a + FCR_CPD_CONST for a in afrr_proj_bull]
afrr_proj_bear = [_power_law(gw, *_afrr_popt) for gw in bear_proj_gw]
ancillary_proj_bear = [a + FCR_CPD_CONST for a in afrr_proj_bear]

# Projected wholesale revenue from note 1 model (DA+ID)
proj_rev_ws = [(proj_rev_by_year[y]["da"] + proj_rev_by_year[y]["id"]) if y in proj_rev_by_year else None
               for y in proj_years]
bull_rev_ws = [(bull_rev_by_year[y]["da"] + bull_rev_by_year[y]["id"]) if y in bull_rev_by_year else None
               for y in proj_years]
bear_rev_ws = [(bear_rev_by_year[y]["da"] + bear_rev_by_year[y]["id"]) if y in bear_rev_by_year else None
               for y in proj_years]

# ── Build the combined chart ────────────────────────────────────
default_pct = 90

fig_hero = go.Figure()

# X-axis labels
hist_labels = [str(y) for y in YEARS] + ["2026*"]
proj_labels = [str(y) for y in proj_years]
all_labels = hist_labels + proj_labels

# Bars: wholesale revenue (perfect foresight upper bound)
hist_rev = all_rev[default_pct]
proj_rev_scaled = [v * default_pct / 100 if v is not None else None for v in proj_rev_ws]
bar_colors = ["rgba(42, 157, 143, 0.35)"] * len(hist_rev) + ["rgba(42, 157, 143, 0.18)"] * len(proj_years)
all_rev_bars = hist_rev + proj_rev_scaled
_label_years_proj = {2030, 2035, 2040}
bar_text = [f"€{v:.0f}k" if v is not None else "" for v in hist_rev]
bar_text += [f"€{v:.0f}k" if (v is not None and proj_years[i] in _label_years_proj) else ""
             for i, v in enumerate(proj_rev_scaled)]
fig_hero.add_trace(go.Bar(
    x=all_labels, y=all_rev_bars,
    name="Wholesale revenue",
    marker_color=bar_colors,
    text=bar_text,
    textposition="outside",
    textfont=dict(size=9, color="#2a9d8f"),
    yaxis="y",
))

# Total c/d: historical (solid) + projected (dashed)
hist_total = [w + anc for w, anc in zip(all_cpd[default_pct], hero_ancillary_cpd)]
proj_ws = all_cpd_proj[default_pct]
proj_total = [w + anc for w, anc in zip(proj_ws, ancillary_proj)]

fig_hero.add_trace(go.Scatter(
    x=hist_labels, y=hist_total,
    name="Total c/d (historical)",
    mode="lines+markers+text",
    line=dict(color="#14213d", width=3),
    marker=dict(size=8, color="#14213d"),
    text=[f"{v:.1f}" for v in hist_total],
    textposition="top center",
    textfont=dict(size=10, color="#14213d"),
    yaxis="y2",
))

fig_hero.add_trace(go.Scatter(
    x=["2026*"] + proj_labels, y=[hist_total[-1]] + proj_total,
    name="Total c/d (projected)",
    mode="lines+text",
    line=dict(color="#14213d", width=2.5, dash="dash"),
    text=[""] + [f"{v:.1f}" if y in (2030, 2035, 2040) else ""
          for v, y in zip(proj_total, proj_years)],
    textposition="top center",
    textfont=dict(size=10, color="#14213d"),
    yaxis="y2",
))

# Bull/Bear scenario band around projected total c/d
bull_ws = all_cpd_proj_bull[default_pct]
bear_ws = all_cpd_proj_bear[default_pct]
proj_upper = [w + anc for w, anc in zip(bull_ws, ancillary_proj_bull)]
proj_lower = [max(0, w + anc) for w, anc in zip(bear_ws, ancillary_proj_bear)]
fig_hero.add_trace(go.Scatter(
    x=["2026*"] + proj_labels + proj_labels[::-1] + ["2026*"],
    y=[hist_total[-1]] + proj_upper + proj_lower[::-1] + [hist_total[-1]],
    fill="toself",
    fillcolor="rgba(20, 33, 61, 0.10)",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip",
    yaxis="y2",
))

# 1 c/d reference line
fig_hero.add_hline(
    y=1.0, line=dict(color="#e76f51", width=1, dash="dot"),
    annotation_text="1 cycle/day",
    annotation_font=dict(size=9, color="#e76f51"),
    annotation_position="bottom right",
    yref="y2",
)

# Build Plotly frames for capture slider
frames = []
for pct in PCT_RANGE:
    hist_rev_p = all_rev[pct]
    proj_rev_p = [v * pct / 100 if v is not None else None for v in proj_rev_ws]
    all_rev_p = hist_rev_p + proj_rev_p
    bar_colors_p = ["rgba(42, 157, 143, 0.35)"] * len(hist_rev_p) + ["rgba(42, 157, 143, 0.18)"] * len(proj_years)
    bar_text_p = [f"€{v:.0f}k" if v is not None else "" for v in hist_rev_p]
    bar_text_p += [f"€{v:.0f}k" if (v is not None and proj_years[i] in _label_years_proj) else ""
                   for i, v in enumerate(proj_rev_p)]
    h_total = [w + anc for w, anc in zip(all_cpd[pct], hero_ancillary_cpd)]
    p_ws = all_cpd_proj[pct]
    p_total = [w + anc for w, anc in zip(p_ws, ancillary_proj)]
    p_bull_ws = all_cpd_proj_bull[pct]
    p_bear_ws = all_cpd_proj_bear[pct]
    p_upper = [w + anc for w, anc in zip(p_bull_ws, ancillary_proj_bull)]
    p_lower = [max(0, w + anc) for w, anc in zip(p_bear_ws, ancillary_proj_bear)]

    frames.append(go.Frame(
        data=[
            go.Bar(
                x=all_labels, y=all_rev_p,
                marker_color=bar_colors_p,
                text=bar_text_p,
                textposition="outside",
                textfont=dict(size=9, color="#2a9d8f"),
                yaxis="y",
            ),
            go.Scatter(
                x=hist_labels, y=h_total,
                mode="lines+markers+text",
                line=dict(color="#14213d", width=3),
                marker=dict(size=8, color="#14213d"),
                text=[f"{v:.1f}" for v in h_total],
                textposition="top center",
                textfont=dict(size=10, color="#14213d"),
                yaxis="y2",
            ),
            go.Scatter(
                x=["2026*"] + proj_labels, y=[h_total[-1]] + p_total,
                mode="lines+text",
                line=dict(color="#14213d", width=2.5, dash="dash"),
                text=[""] + [f"{v:.1f}" if y in (2030, 2035, 2040) else ""
                      for v, y in zip(p_total, proj_years)],
                textposition="top center",
                textfont=dict(size=10, color="#14213d"),
                yaxis="y2",
            ),
            go.Scatter(
                x=["2026*"] + proj_labels + proj_labels[::-1] + ["2026*"],
                y=[h_total[-1]] + p_upper + p_lower[::-1] + [h_total[-1]],
                fill="toself",
                fillcolor="rgba(20, 33, 61, 0.10)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                yaxis="y2",
            ),
        ],
        name=str(pct),
    ))
fig_hero.frames = frames

# Capture % slider (Plotly, client-side)
sliders = [dict(
    active=PCT_RANGE.index(default_pct),
    currentvalue=dict(
        prefix="Capture target: ",
        suffix="%",
        font=dict(size=13, color="#14213d"),
    ),
    pad=dict(t=40),
    steps=[dict(
        args=[[str(pct)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
        label=str(pct),
        method="animate",
    ) for pct in PCT_RANGE],
)]

styled_layout(fig_hero, height=480, y_title="")
fig_hero.update_layout(
    showlegend=False,
    margin=dict(l=50, r=50, t=25, b=90),
    xaxis=dict(
        title="", type="category",
        tickangle=-45,
        tickfont=dict(size=9),
    ),
    yaxis=dict(
        title="Wholesale revenue (kEUR/MW)",
        title_font=dict(size=10, color="#2a9d8f"),
        tickfont=dict(size=9, color="#2a9d8f"),
        tickprefix="\u20ac", ticksuffix="k",
        range=[0, max(hero_rev_max) * 1.8],
        showgrid=False,
    ),
    yaxis2=dict(
        title="Cycles / day",
        title_font=dict(size=10, color="#14213d"),
        tickfont=dict(size=9, color="#14213d"),
        overlaying="y", side="right",
        range=[0, 3.2],
        gridcolor="rgba(148,163,184,0.12)",
    ),
    sliders=sliders,
)

st.markdown("""
Use the **capture slider** below to explore the trade-off: at 100%, the battery
chases every last spread the model can find; at 90%, it skips the smallest
windows — ~30% fewer cycles for 10% less revenue.
""")

_dur_pills("Total cycling is falling — and the market, not the warranty, sets the limit", "dur_hero")
st.plotly_chart(fig_hero, use_container_width=True, config={"displayModeBar": False})

render_chart_caption(
    "Bars (left axis) = wholesale revenue (perfect foresight upper bound). "
    "Solid line (right axis) = total cycles/day (wholesale + aFRR + FCR). "
    f"Shaded band = bull / bear scenario range "
    f"({BULL_PARAMS['bess_2040']}–{BEAR_PARAMS['bess_2040']} GW fleet, "
    f"gas €{BULL_PARAMS['gas_2040']}–{BEAR_PARAMS['gas_2040']}/MWh, "
    f"PV {BULL_PARAMS['pv_2040']}–{BEAR_PARAMS['pv_2040']} GW — all 2040 targets). "
    "Drag the slider from 90% toward 100%: revenue gains ~10%, "
    "but cycles jump ~30%. 2026* = Q1 annualised from 90 days (Jan–Mar) — "
    "captures winter spreads but misses summer solar surplus; likely overstates full-year revenue."
)

st.markdown("")  # spacer

st.markdown(f"""
Notice how narrow the shaded band is compared to the revenue projections in [{NOTE1_TITLE}]({NOTE1_URL}).
The same bull / bear scenarios ({BULL_PARAMS['bess_2040']} – {BEAR_PARAMS['bess_2040']} GW
fleet by 2040, gas €{BEAR_PARAMS['gas_2040']}–{BULL_PARAMS['gas_2040']}/MWh,
PV {BULL_PARAMS['pv_2040']}–{BEAR_PARAMS['pv_2040']} GW) produce wide revenue
bands but tight cycling bands — because gas prices widen spreads but don't
create new ones, while fleet growth eliminates windows only gradually.
""")

st.markdown("---")


# ════════════════════════════════════════════════════════════════
#  CONCEPT ILLUSTRATION: more cycles ≠ more revenue
# ════════════════════════════════════════════════════════════════

st.markdown("""
Why? Because revenue comes from price spreads — how deep they are, not how
many times the battery cycles. Two real days illustrate this:
""")

_hours_24 = list(range(24))
# Day A: 28 Aug 2022 — energy crisis, one massive spread, 0.93 FEC, €1038
_prices_a = [585.0, 491.4, 467.7, 430.0, 426.0, 409.9, 427.3, 441.9,
             467.4, 437.6, 275.1, 256.9, 116.0, 75.8, 13.3, 62.2,
             105.9, 336.2, 579.2, 676.4, 700.8, 646.1, 606.7, 590.0]
_chg_a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.94,
          0, 0, 0, 0, 0, 0, 0, 0]
_dis_a = [0.83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0.02, 1.0, 0, 0, 0]
# Day B: 29 May 2024 — solar duck curve, three clear windows, 2.09 FEC, €110
_prices_b = [72.2, 57.5, 54.3, 50.6, 49.8, 45.9, 72.2, 86.4,
             91.5, 69.1, 67.8, 62.1, 59.1, 62.2, 62.2, 69.5,
             72.7, 89.2, 94.7, 111.4, 140.5, 143.3, 120.4, 95.5]
_chg_b = [0, 0, 0, 0, 0.94, 1.0, 0, 0, 0, 0, 0, 0.94, 1.0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0.75]
_dis_b = [0.83, 0, 0, 0, 0, 0, 0, 0.67, 1.0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0.67, 1.0, 0, 0]

_fig_concept = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "<b>28 Aug 2022</b>  ·  0.9 FEC  ·  <b>€1 038</b>",
        "<b>29 May 2024</b>  ·  2.1 FEC  ·  <b>€110</b>",
    ],
    horizontal_spacing=0.08,
)
_CHG_COLOR = "rgba(34,197,94,0.20)"
_DIS_COLOR = "rgba(249,115,22,0.20)"
for col, prices, chg, dis, line_color in [
    (1, _prices_a, _chg_a, _dis_a, "#3b82f6"),
    (2, _prices_b, _chg_b, _dis_b, "#f87171"),
]:
    _fig_concept.add_trace(go.Scatter(
        x=_hours_24, y=prices, mode="lines",
        line=dict(color=line_color, width=2.5),
        showlegend=False,
    ), row=1, col=col)
    _xref = "x" if col == 1 else "x2"
    _yref = "y" if col == 1 else "y2"
    for h in range(24):
        if chg[h] > 0.01:
            _fig_concept.add_vrect(
                x0=h - 0.5, x1=h + 0.5,
                fillcolor=_CHG_COLOR, line_width=0,
                xref=_xref, yref=_yref,
            )
        if dis[h] > 0.01:
            _fig_concept.add_vrect(
                x0=h - 0.5, x1=h + 0.5,
                fillcolor=_DIS_COLOR, line_width=0,
                xref=_xref, yref=_yref,
            )
_fig_concept.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(size=10, color=_CHG_COLOR, symbol="square"),
    name="Buy",
), row=1, col=1)
_fig_concept.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(size=10, color=_DIS_COLOR, symbol="square"),
    name="Sell",
), row=1, col=1)

_fig_concept.update_layout(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=270,
    margin=dict(l=40, r=20, t=40, b=35),
    legend=dict(
        orientation="h", y=-0.15,
        font=dict(size=10, color="#14213d"),
        bgcolor="rgba(0,0,0,0)",
    ),
)
_y_max = max(max(_prices_a), max(_prices_b)) * 1.05
for ax in ["xaxis", "xaxis2"]:
    _fig_concept.update_layout(**{ax: dict(
        title="", tickfont=dict(size=9, color="#5c677d"), dtick=6,
    )})
_fig_concept.update_layout(
    yaxis=dict(title="€/MWh", title_font=dict(size=10, color="#5c677d"),
               tickfont=dict(size=9, color="#5c677d"), range=[0, _y_max]),
    yaxis2=dict(title="", tickfont=dict(size=9, color="#5c677d"),
                range=[0, _y_max]),
)
st.plotly_chart(_fig_concept, use_container_width=True, config={"displayModeBar": False})
render_chart_caption(
    "Real DA prices (line) and optimal dispatch (2h battery, perfect foresight). "
    "Green shading = charging hours, orange = discharging. "
    "FEC = full equivalent cycle. "
    "Left: one deep spread — the battery charges once and earns €1 038/MW. "
    "Right: a solar duck-curve day with three clear buy/sell windows — "
    "more than twice the cycling for 10× less revenue."
)

st.markdown("---")


# ════════════════════════════════════════════════════════════════
#  SECTION 2: Stacked bar — revenue per cycle
# ════════════════════════════════════════════════════════════════

st.markdown("## The first cycle captures most of the revenue")

st.markdown("""
The stacked bars below break annual wholesale revenue into first, second,
and third cycle contributions. The second cycle's value fluctuates with
market conditions — high in the volatile years of 2021–2022, compressed
as the fleet grew through 2023–2025. The 2025 dip stands out: a mild winter
and low gas prices flattened spreads, pushing both revenue and cycling
below the trend. But in every year the first cycle alone captures the majority.
""")

_s2_dur, _s2_f, _s2_rev, _s2_fec, _s2_cpd = _dur_pills(
    "The first cycle captures 70–90% of wholesale revenue in every year", "dur_stacked")

fig0 = go.Figure()

years_list = YEARS
chart0_labels = [str(y) for y in years_list] + ["2026*"]
rev_1st, rev_2nd, rev_3rd = [], [], []
for year in years_list:
    yf = _s2_f[_s2_f["year"] == year]
    r1 = yf[yf["max_cycles_per_day"] == 1.0].iloc[0]["annual_revenue_eur_per_mw"] / 1000
    r2 = yf[yf["max_cycles_per_day"] == 2.0].iloc[0]["annual_revenue_eur_per_mw"] / 1000
    r3 = yf[yf["max_cycles_per_day"] == 3.0].iloc[0]["annual_revenue_eur_per_mw"] / 1000
    rev_1st.append(r1)
    rev_2nd.append(r2 - r1)
    rev_3rd.append(r3 - r2)

rev_1st.append(_s2_rev[1.0])
rev_2nd.append(_s2_rev[2.0] - _s2_rev[1.0])
rev_3rd.append(_s2_rev[3.0] - _s2_rev[2.0])

fig0.add_trace(go.Bar(
    x=chart0_labels, y=rev_1st,
    name="First cycle",
    marker_color="#2a9d8f",
    text=[f"\u20ac{v:.0f}k" for v in rev_1st],
    textposition="inside",
    textfont=dict(size=11, color="white"),
))
fig0.add_trace(go.Bar(
    x=chart0_labels, y=rev_2nd,
    name="Second cycle",
    marker_color="rgba(42, 157, 143, 0.5)",
    text=[f"\u20ac{v:.0f}k" for v in rev_2nd],
    textposition="inside",
    textfont=dict(size=11, color="white"),
))
fig0.add_trace(go.Bar(
    x=chart0_labels, y=rev_3rd,
    name="Third cycle",
    marker_color="rgba(42, 157, 143, 0.25)",
    text=[f"\u20ac{v:.0f}k" if v >= 1 else "" for v in rev_3rd],
    textposition="inside",
    textfont=dict(size=11, color="#5c677d"),
))

styled_layout(fig0, height=380, y_title="kEUR / MW / yr")
fig0.update_layout(barmode="stack", xaxis=dict(title=""), legend=dict(traceorder="normal"))

st.plotly_chart(fig0, use_container_width=True, config={"displayModeBar": False})

render_chart_caption(
    "Perfect foresight, 100% capture — the maximum each cycle can earn. "
    f"The second cycle peaked at €{max(rev_2nd):.0f}k in 2022 (energy crisis spreads), "
    f"and fell to €{rev_2nd[4]:.0f}k in 2025 as the fleet tripled. "
    f"The third cycle never exceeds €8k in any year. "
    f"2026* = Q1 annualised ({Q1_2026_GW} GW fleet)."
)

st.markdown("---")


# ════════════════════════════════════════════════════════════════
#  SECTION 3: Lifetime revenue frontier — cycling vs degradation
# ════════════════════════════════════════════════════════════════

st.markdown("## More cycling means shorter battery life — where does lifetime revenue peak?")

st.markdown(f"""
The charts above show annual revenue. But a battery owner earns over the full
asset life — and every cycle wears the cells. More aggressive cycling captures
more revenue per year, but the battery reaches end-of-life sooner.

The chart below shows **total lifetime wholesale revenue** at each cycling rate,
using projected revenue from [{NOTE1_TITLE}]({NOTE1_URL}) and the same degradation model.
Ancillary capacity revenue (FCR, aFRR) is fixed — it scales with fleet size,
not with the wholesale cycling rate shown here.
""")

# ── Lifetime revenue using projected revenue stream ──
# Degradation model consistent with Note 1:
# - Linear fade: calendar component + cycling component (proportional to FEC)
# - Augmentation at 8,500 FEC restores to 92%
# - EOL at 50% residual capacity
# At 2 c/d (730 FEC/yr) this gives ~1.8%/yr fade — matching Note 1.
_CAL_FADE = 0.005        # calendar-only fade per year
_CYC_FADE = 0.013        # cycling fade per year at 730 FEC/yr reference
_CYC_REF = 730.0         # reference FEC/yr (2 c/d)
_AUG_FEC = 8500.0        # augmentation threshold (total FEC)
_AUG_RESTORE = 0.92      # post-augmentation capacity
_EOL_FLOOR = 0.60        # end-of-life threshold

def _cohort_cap(age_years, annual_fec):
    """Effective capacity fraction for a single cohort."""
    fade = _CAL_FADE + _CYC_FADE * (annual_fec / _CYC_REF)
    cap = 1.0 - fade * age_years
    if annual_fec * age_years >= _AUG_FEC:
        aug_yr = _AUG_FEC / annual_fec
        cap = _AUG_RESTORE - fade * (age_years - aug_yr)
    return max(cap, _EOL_FLOOR)

# Build capture profile per year: for each cycling rate, what % of max revenue
# is captured? Use historical data for past years, power-law projection for future.
# This ensures chart 3 is consistent with chart 1's projection.

# Historical capture profile (2023-2024 average as shape reference)
_frontier_ref_years = [2023, 2024]
_frontier_ref = f2h[f2h["year"].isin(_frontier_ref_years)].groupby(
    "max_cycles_per_day", as_index=False
).agg(
    annual_revenue_eur_per_mw=("annual_revenue_eur_per_mw", "mean"),
    annual_fec=("annual_fec", "mean"),
    pct_of_max=("pct_of_max", "mean"),
).sort_values("max_cycles_per_day")

# Power-law fit: given GW fleet, what cpd corresponds to each capture %?
# We already have popt per pct from the projection above.
# Build a lookup: for a given year's GW, the max cpd at 100% capture.
_popt_by_pct = {}
for pct in PCT_RANGE:
    _fit_vals = np.array(hy_cpd.get(pct, all_cpd[pct]))
    _v = ~np.isnan(_fit_vals)
    _gw_v = _fit_gw[_v]
    _cpd_v = _fit_vals[_v]
    popt_pct, _ = curve_fit(_power_law, _gw_v, _cpd_v, p0=[2.0, -0.3])
    _popt_by_pct[pct] = popt_pct

def _projected_capture_pct(cpd_target, year_gw):
    """Given a target c/d and fleet GW, what capture % does the market allow?

    Uses the power-law fits: for each capture %, we know what cpd the market
    offers at that fleet size. Find the highest capture % whose cpd <= target.
    """
    best_pct = PCT_RANGE[0]
    for pct in PCT_RANGE:
        cpd_at_pct = _power_law(year_gw, *_popt_by_pct[pct])
        if cpd_at_pct <= cpd_target:
            best_pct = pct
        else:
            break
    return best_pct

# Build unified revenue-by-year dict (historical + projected)
import pickle as _pkl
_note1_path = Path(__file__).parent.parent / "de-bess-outlook" / "data" / "precomputed.pkl"
_hist_rev_by_year = {}
if _note1_path.exists():
    with open(_note1_path, "rb") as _f:
        _note1 = _pkl.load(_f)
    for _hb in _note1.get("2h", {}).get("hist_bars", []):
        _hist_rev_by_year[_hb["year"]] = _hb

# For years without calibrated ancillary (2021, 2022): wholesale only from frontier
for _yr in YEARS:
    if _yr not in _hist_rev_by_year:
        _yf = f2h[f2h["year"] == _yr]
        _ws_max = _yf["annual_revenue_eur_per_mw"].max() / 1000 if not _yf.empty else 0
        _hist_rev_by_year[_yr] = {
            "da": _ws_max * 0.67, "id": _ws_max * 0.33,
            "fcr": 0, "afrr_cap": 0, "afrr_energy": 0,
        }

_all_rev_by_year = {}
_all_rev_by_year.update(_hist_rev_by_year)
for _yr, _data in proj_rev_by_year.items():
    _all_rev_by_year[_yr] = _data

# Fleet GW lookup for each calendar year (for projected capture)
_gw_by_year = dict(fleet_gw)
_gw_by_year[2026] = Q1_2026_GW
for i, y in enumerate(proj_years):
    _gw_by_year[y] = proj_gw[i]
_MAX_PROJ_GW = _gw_by_year[2040]  # for years beyond projection horizon

ASSET_LIFE_CAP = 25

def _lifetime_revenue(cod_year, target_cpd, annual_fec, cap_pct=100, frontier_ref=None):
    """Sum projected revenue × capacity_fraction over asset life.

    For each year, the effective capture % is the minimum of:
    - what this cycling rate achieves on the 2023-2024 historical profile
    - what the projected market can support at that year's fleet size
    - the operator's capture target (cap_pct)
    """
    _fref = frontier_ref if frontier_ref is not None else _frontier_ref
    total = 0.0
    # Historical capture at this cycling rate (from 2023-2024 shape)
    hist_pct = float(_fref[_fref["annual_fec"] <= annual_fec + 1]
                     ["pct_of_max"].max()) if annual_fec > 0 else 0
    for yr_offset in range(ASSET_LIFE_CAP):
        cal_year = cod_year + yr_offset
        cap_frac = _cohort_cap(float(yr_offset), annual_fec)
        if cap_frac <= _EOL_FLOOR:
            break
        rev_data = _all_rev_by_year.get(cal_year, _all_rev_by_year.get(2040, {}))
        ws_rev = rev_data.get("da", 0) + rev_data.get("id", 0)
        anc_rev = (rev_data.get("fcr", 0) + rev_data.get("afrr_cap", 0)
                   + rev_data.get("afrr_energy", 0))
        # For projected years, cap the capture % by what the market offers
        year_gw = _gw_by_year.get(cal_year, _MAX_PROJ_GW)
        if cal_year > max(YEARS):
            eff_pct = min(hist_pct, _projected_capture_pct(target_cpd, year_gw), cap_pct)
        else:
            eff_pct = min(hist_pct, cap_pct)
        total += (ws_rev * (eff_pct / 100) + anc_rev) * cap_frac
    return total

COD_YEARS = [
    (2026, "#14213d", 2.5),
]

_s3_dur, _s3_f, _, _, _ = _dur_pills(
    "Lifetime revenue peaks near warranty limits — pushing harder doesn't pay off", "dur_lifetime")

# Rebuild frontier ref for the lifetime chart's selected duration
_frontier_ref_lt = _s3_f[_s3_f["year"].isin(_frontier_ref_years)].groupby(
    "max_cycles_per_day", as_index=False
).agg(
    annual_revenue_eur_per_mw=("annual_revenue_eur_per_mw", "mean"),
    annual_fec=("annual_fec", "mean"),
    pct_of_max=("pct_of_max", "mean"),
).sort_values("max_cycles_per_day")

# Extend frontier beyond market saturation: add synthetic high-cpd points
# where revenue stays at max but FEC keeps growing (pure degradation cost).
_max_fec_lt = _frontier_ref_lt["annual_fec"].max()
_max_cpd_lt = _max_fec_lt / 365
_TARGET_MAX_CPD = 3.5  # extend all durations to this for comparison
_ext_rows = []
if _max_cpd_lt < _TARGET_MAX_CPD:
    _max_rev_lt = _frontier_ref_lt["annual_revenue_eur_per_mw"].max()
    for extra_cpd in np.arange(_max_cpd_lt + 0.25, _TARGET_MAX_CPD + 0.01, 0.25):
        _ext_rows.append({
            "max_cycles_per_day": extra_cpd,
            "annual_fec": extra_cpd * 365,
            "annual_revenue_eur_per_mw": _max_rev_lt,
            "pct_of_max": 100.0,
        })
if _ext_rows:
    _frontier_ref_lt = pd.concat(
        [_frontier_ref_lt, pd.DataFrame(_ext_rows)], ignore_index=True
    ).sort_values("annual_fec")

fig_lt = go.Figure()

_peak_cpd, _peak_rev = 0, 0  # from first (primary) COD
for cod_year, color, width in COD_YEARS:
    cpd_vals, lt_rev_k = [], []
    for _, row in _frontier_ref_lt.iterrows():
        cpd = row["annual_fec"] / 365
        lt = _lifetime_revenue(cod_year, cpd, row["annual_fec"], cap_pct=100,
                               frontier_ref=_frontier_ref_lt)
        cpd_vals.append(cpd)
        lt_rev_k.append(lt / 1000)

    peak_idx = int(np.argmax(lt_rev_k))
    if cod_year == COD_YEARS[0][0]:
        _peak_cpd = cpd_vals[peak_idx]
        _peak_rev = lt_rev_k[peak_idx]

    fig_lt.add_trace(go.Scatter(
        x=cpd_vals, y=lt_rev_k,
        mode="lines",
        name=f"COD {cod_year}",
        line=dict(color=color, width=width),
        hovertemplate=f"COD {cod_year}<br>" + "%{x:.1f} c/d<br>€%{y:.1f}M<extra></extra>",
    ))
    fig_lt.add_trace(go.Scatter(
        x=[cpd_vals[peak_idx]], y=[lt_rev_k[peak_idx]],
        mode="markers+text",
        marker=dict(size=8, color=color, symbol="diamond"),
        text=[f"{cpd_vals[peak_idx]:.1f}"],
        textposition="top center",
        textfont=dict(size=9, color=color),
        showlegend=False,
    ))

fig_lt.add_hline(
    y=_peak_rev, line=dict(color="#5c677d", width=1, dash="dot"),
    annotation_text=f"€{_peak_rev:.1f}M",
    annotation_position="top left",
    annotation_font=dict(size=10, color="#5c677d"),
)

styled_layout(fig_lt, height=400, y_title="Lifetime revenue (M€/MW)")
fig_lt.update_layout(
    showlegend=True,
    legend=dict(orientation="h", y=-0.15),
    margin=dict(l=55, r=25, t=25, b=70),
    xaxis=dict(title="Cycles per day", range=[0, _TARGET_MAX_CPD]),
    yaxis=dict(
        tickprefix="€", ticksuffix="M",
        gridcolor="rgba(148,163,184,0.12)",
    ),
)

st.plotly_chart(fig_lt, use_container_width=True, config={"displayModeBar": False})

render_chart_caption(
    f"{int(_s3_dur)}h battery, COD {COD_YEARS[0][0]}. "
    f"At each cycling rate, the model asks: how much revenue does this capture "
    f"in each year of the battery's life, given projected fleet growth? "
    f"Higher cycling earns more per year but wears the battery faster. "
    f'Revenue: <a href="{NOTE1_URL}">{NOTE1_TITLE}</a> projections (to 2040, held flat beyond). '
    f"Degradation: calendar + cycling fade (faster cycling = shorter life), "
    f"augmentation at ~{_AUG_FEC:.0f} FEC, "
    f"EOL at {_EOL_FLOOR:.0%}. Ancillary revenue (FCR + aFRR) included, independent of cycling rate."
)

st.markdown("")  # spacer after caption

st.markdown(f"""
For a COD {COD_YEARS[0][0]} battery, lifetime revenue peaks at
**~{_peak_cpd:.1f} cycles/day** (€{_peak_rev:.1f}M) — right around typical
warranty limits of 1.5–2 cycles/day. Cycling beyond the warranty envelope
does not pay for the extra degradation it causes.
The curve is flat near the peak: cycling slightly less costs very little
but extends the battery by years. At projected cycling rates (falling
toward 1 c/d by 2030), warranty limits are unlikely to bind — the
constraint is the market, not the battery.
""")

with st.expander("Degradation model assumptions"):
    _fade_at_2 = _CAL_FADE + _CYC_FADE * (730 / _CYC_REF)
    _fade_at_1 = _CAL_FADE + _CYC_FADE * (365 / _CYC_REF)
    st.markdown(f"""
**Same model as [{NOTE1_TITLE}]({NOTE1_URL})**, extended to variable cycling rates:
- **Linear fade** = calendar component ({_CAL_FADE:.1%}/yr) + cycling component
  (proportional to FEC/year). At 2 c/d → {_fade_at_2:.1%}/yr. At 1 c/d → {_fade_at_1:.1%}/yr.
- **Augmentation** at {_AUG_FEC:.0f} total FEC restores capacity to {_AUG_RESTORE:.0%}.
  At 2 c/d this is year ~{_AUG_FEC/730:.0f}. At 1 c/d → year ~{_AUG_FEC/365:.0f}.
- **EOL** at {_EOL_FLOOR:.0%} residual capacity.
- **Revenue stream:** [{NOTE1_TITLE}]({NOTE1_URL}) projections for 2026–2040, held flat beyond 2040.
- **No discounting.** Time value of money would favour more aggressive cycling
  (€1 earned today > €1 in year 15) and push the peak to the right.
""")

st.markdown("---")


# ════════════════════════════════════════════════════════════════
#  ANCILLARY CYCLING — compact paragraph + expander
# ════════════════════════════════════════════════════════════════

st.markdown("## What about ancillary cycling?")

st.markdown(f"""
The total cycling line in the first chart above already includes ancillary
activations. Here is what drives those extra cycles.

**FCR** adds a stable **~0.25 cycles/day** — driven by grid frequency physics,
not market conditions. **aFRR** adds **~0.4 cycles/day** in 2024, but is declining
as more batteries split the activation signal (down from 0.45 in 2023 to 0.31
in 2025). Combined with wholesale (~1.1 cycles/day at 90% capture), total cycling
is **~1.7 cycles/day** in 2024 — and shrinking as the fleet grows.

The incremental *revenue* from aFRR energy activations is also vanishing.
Net revenue here means the spread between what BESS earns on activated energy
and what it would have earned trading the same MWh on the wholesale market
(opportunity cost). That spread was ~€20k/MW in 2023, ~€10k in 2024, and
**negative** in 2025 (−€5k) as activation bid prices converged toward wholesale
levels. Ancillary cycling is increasingly a pure cost, not a revenue source.
""")

with st.expander("Ancillary cycling details"):
    st.markdown(f"""
| Source | 2023 | 2024 | 2025 |
|:---|:---|:---|:---|
| **FCR** (grid frequency) | 0.25 | 0.25 | 0.25 |
| **aFRR** (activated reserves) | 0.45 | 0.40 | 0.31 |
| **Total ancillary** | 0.70 | 0.65 | 0.56 |
| aFRR energy net revenue | €20k/MW | €10k/MW | −€5k/MW |
| aFRR activation ratio | {AFRR_YEARLY[2023]['avg_pos_mw']/AFRR_YEARLY[2023]['contracted_pos']*100:.1f}% | {AFRR_YEARLY[2024]['avg_pos_mw']/AFRR_YEARLY[2024]['contracted_pos']*100:.1f}% | {AFRR_YEARLY[2025]['avg_pos_mw']/AFRR_YEARLY[2025]['contracted_pos']*100:.1f}% |

**FCR:** computed from 1-second grid frequency measurements
([TransnetBW via power-grid-frequency.org](https://osf.io/m43tg/), 2015–2020),
standard droop curve (±10 mHz deadband, full activation at ±200 mHz).
Result ~0.25 FEC/day — validated against
[M5BAT study](https://doi.org/10.1016/j.est.2020.101982)
(0.28 EFC/day over 4 years of real FCR operation in Germany).

**aFRR:** actual activated reserve volumes (*AktivierteSRL*) from
[netztransparenz.de](https://ds.netztransparenz.de) (15-minute resolution,
quality-assured data). Activation ratio is the average positive activated
volume as a share of total contracted positive capacity.

**A 4h battery needs roughly half the cycling rate of a 1h system** for the
same revenue share — the saturation curve shifts left with longer duration.
This means a 4h system's unused cycle budget can serve as degradation buffer
or absorb ancillary activations more comfortably.
    """)

st.markdown("---")


# ════════════════════════════════════════════════════════════════
#  PROJECTION SUMMARY
# ════════════════════════════════════════════════════════════════

_gw_2030 = user_buildout.get(2030, BESS_2040_DEFAULT)
_gw_2035 = user_buildout.get(2035, BESS_2040_DEFAULT)
_cpd_2030_ws = all_cpd_proj[90][proj_years.index(2030)]
_cpd_2035_ws = all_cpd_proj[90][proj_years.index(2035)]
_afrr_2030 = _power_law(float(_gw_2030), *_afrr_popt)
_afrr_2035 = _power_law(float(_gw_2035), *_afrr_popt)
_total_2030 = _cpd_2030_ws + _afrr_2030 + FCR_CPD_CONST
_total_2035 = _cpd_2035_ws + _afrr_2035 + FCR_CPD_CONST
_rev_2030 = proj_rev_by_year.get(2030, {})
_rev_2035 = proj_rev_by_year.get(2035, {})

st.markdown("## The cycle budget is shrinking from both sides")

st.markdown("""
The market offers fewer profitable cycling windows (supply side), and the first
cycle captures an increasing share of what remains (demand side).

**The cycle budget is a consequence of trading strategy, not a fixed market
requirement.**
""")



# ── Method notes ─────────────────────────────────────────────────────────

st.markdown("---")

st.markdown("## What this model does NOT include")

st.markdown(f"""
- **Time value of money.** No discounting — €1 earned in year 15 counts the same as
  €1 today. This understates the value of aggressive early cycling.
- **Real-world forecast error.** Dispatch uses perfect foresight — an upper bound
  that no real operator achieves. In practice, capture rates of 70–90% are typical,
  which reduces both revenue *and* required cycling.
- **Co-optimisation across markets.** Wholesale, FCR, and aFRR are modelled
  independently. In practice, ancillary commitments constrain wholesale dispatch
  and vice versa.
- **Cell-level degradation physics.** The model uses a simplified linear fade.
  Real degradation depends on depth of discharge, temperature, and C-rate —
  the subject of the next note.
- **Capacity market or redispatch revenue.** Germany's capacity mechanism is
  still under development; redispatch participation varies by project.
""")

st.markdown("---")

st.markdown("## Data Sources & Methodology")

with st.expander("Dispatch model"):
    st.markdown("""
Perfect-foresight linear programme (SciPy HiGHS) — sets the **upper bound**
on arbitrage revenue at each cycling rate.

| Parameter | Value |
|:---|:---|
| Duration | 2h (base case) |
| SoC window | 5–95% |
| Round-trip efficiency | 86% |
| Min spread threshold | €5/MWh |
| Cycle cap range | 0.25–4.0 cycles/day |
| Resolution | Hourly (DA) + 15-min (intraday) |
""")

with st.expander("Price data"):
    st.markdown("""
| Market | Source | Resolution |
|:---|:---|:---|
| Day-ahead | [Energy-Charts](https://energy-charts.info/) (Fraunhofer ISE), DE-LU bidding zone | Hourly |
| Intraday | [Netztransparenz](https://ds.netztransparenz.de) ID-AEP index | 15-min |
| FCR capacity | [regelleistung.net](https://www.regelleistung.net) tender results | 4h products |
| aFRR capacity | [regelleistung.net](https://www.regelleistung.net) tender results | 4h products |

Intraday prices capped at ±150 €/MWh to limit outlier sensitivity.
**2026** values are Q1 annualised from 90 days (Jan–Mar) — captures winter spreads but misses summer solar surplus; likely overstates full-year revenue.
""")

with st.expander("Ancillary cycling data"):
    st.markdown("""
| Source | Data | Reference |
|:---|:---|:---|
| **FCR** | 1-second grid frequency (Continental Europe) | [TransnetBW via power-grid-frequency.org](https://osf.io/m43tg/), 2015–2020 |
| **FCR validation** | M5BAT real-world FCR operation (0.28 EFC/day, 4 years) | [Figgener et al. 2020](https://doi.org/10.1016/j.est.2020.101982) |
| **aFRR activations** | Activated reserves (*AktivierteSRL*), quality-assured | [Netztransparenz.de](https://ds.netztransparenz.de), 15-min resolution |

FCR cycling is computed from a standard droop curve (±10 mHz deadband,
full activation at ±200 mHz). Result: ~0.25 FEC/day, stable across years.
""")

with st.expander("FEC definition"):
    st.markdown("""
**Full equivalent cycle (FEC)** = cumulative discharged energy ÷ nameplate
energy capacity.

This is the standard definition used in manufacturer warranties
(CATL, BYD, LG, Samsung SDI) and bankability assessments (DNV).
Partial cycles are summed pro-rata. Only the discharge side counts —
charging losses are excluded from the cycle count.
""")

# ── Summary & series anchor ──────────────────────────────────
st.markdown("---")

st.markdown("""
**In short:** German BESS needs far fewer cycles than most models assume.
The first daily cycle captures the bulk of wholesale revenue; the second adds
little; the third is negligible. As the fleet grows, the market offers fewer
profitable windows — total cycling is falling toward 1 c/d by 2030.
The binding constraint is the market, not the warranty.
""")

render_closing(
    "Second in a series on BESS merchant economics — "
    "from market opportunity and cycling trade-offs to degradation, "
    "warranty economics, and owner governance."
)
st.markdown(
    '<div style="margin-top: 0.5rem; font-size: 0.95rem; color: #666;">'
    "<b>Next:</b> cycle count is only part of the story — depth of discharge "
    "(how deeply each cycle drains the cell), resting state of charge "
    "(the level the battery sits at between cycles), and C-rate "
    "(charge/discharge speed relative to capacity) matter more for cell health "
    "than throughput alone. What actually drives degradation, and how should "
    "operators think about it?"
    "</div>",
    unsafe_allow_html=True,
)

render_footer()
