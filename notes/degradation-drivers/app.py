"""
Market Note 3: What Actually Drives Degradation

Cycle count tells almost nothing about cell health. This note ranks the levers
an owner actually controls, using a production-grade LFP/graphite model.

Results are pre-computed by precompute.py and loaded from data/precomputed.pkl.
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

from plotly.subplots import make_subplots

from lib.models.degradation import PRESETS
from lib.models.degradation_detailed import DutyCycle, project_capacity_detailed
from lib.ui.theme import (
    apply_theme,
    render_chart_caption,
    render_chart_title,
    render_closing,
    render_footer,
    render_header,
    render_takeaway,
)

PRECOMPUTED_PATH = Path(__file__).parent / "data" / "precomputed.pkl"


@st.cache_data(show_spinner=False)
def load_precomputed() -> dict:
    with open(PRECOMPUTED_PATH, "rb") as f:
        return pickle.load(f)


# ── Page ─────────────────────────────────────────────────────
st.set_page_config(page_title="What Actually Drives Degradation", layout="wide")
apply_theme(show_sidebar=False)

render_header(
    title="What actually drives degradation",
    kicker="GERMAN BESS | DEGRADATION",
    subtitle="Battery health depends on more than cycle count. Here's the rest of the picture.",
)

# ── Intro ────────────────────────────────────────────────────
st.markdown(
    """
Most investor-side BESS models start from the same shortcut: pick a number of cycles per year, multiply by a fade rate, call it degradation. That is how lifetime revenue gets projected, how warranty calls get argued, how augmentation gets sized. The number of cycles is *the* input.

It is the wrong input to anchor on. Two packs that log an identical 730 cycles in a year can land years apart at end of life — one at year 10, the other at year 14 — purely because of what happened *between* those cycles. Where it rested. How warm the room was. How deep each cycle went. Cycle count catches none of it.

So I rebuilt the model to price these factors explicitly. This note moves each driver — depth of discharge, rest state, how fast it cycles, how many cycles it runs, and how warm it sits — across its full operating range and ranks them by how many years of life each one buys or burns.

€/MWh throughput is the per-MWh "wear bill" a pack runs up over its lifetime. The ranking uses that basis.

**The short version: temperature and C-rate dominate everything else.** Rest SoC and cycles per day each move the bill by half as much. Depth of discharge barely moves it at all, because the years of life it burns are cancelled almost exactly by the extra MWh per cycle it delivers. Which cell you buy sets the ceiling; how the pack is run decides where you land inside it.
"""
)

data = load_precomputed()

# ── The levers ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
### The five levers

Five parameters set the fade rate of a stationary LFP pack. The trader controls four of them through dispatch; site design fixes the fifth (temperature).

- **Depth of discharge (DoD)** — how far each cycle swings.
- **Rest SoC** — the state of charge the pack sits at during the ~20 hours a day it isn't actively moving energy.
- **C-rate** — how fast power moves in and out of the battery.
- **Cycling rate** — full-equivalent cycles per day.
- **Temperature** — cell-internal average over the year.

Below, each lever moves across its full range while the other four stay at baseline (2 c/d, 80 % DoD, 55 % rest SoC, 0.5C, 25 °C). Toggle the y-axis between **€/MWh throughput** and **years until 70 % SoH** — the ranking shifts between the two views. The dashed horizontal line in each panel marks the baseline value.
"""
)

# ── Chart 1 — driver response curves (years-to-EOL) ─────────
st.markdown("---")
render_chart_title("How each driver moves the number")
st.markdown(
    "<div style='color:#6b6b6b;font-size:0.95em;margin-bottom:0.5em'>"
    "Each panel moves one driver across its full operating range, with the other four held at the baseline point (dashed line). "
    "Toggle the axis between <b>€/MWh throughput</b> and <b>years until 70 % SoH</b>. "
    "Read it as: <i>\"if average cell temperature rises from 25 °C to 35 °C, the €/MWh bill jumps by ~€38 — and ~5 years of life disappear with it.\"</i>"
    "</div>",
    unsafe_allow_html=True,
)

_curves = data["response_curves"]
_drivers_by_key = {d["key"]: d for d in _curves["drivers"]}


# x-raw, y=years. `x_transform` converts raw → display units.
_curve_points = {
    k: [{"x": p["x"], "y": p["years"]} for p in d["years_to_70_mid"]]
    for k, d in _drivers_by_key.items()
}

_baseline_x = {
    "temp": 25.0, "crate": 0.50, "dod": 0.80, "soc": 0.55, "fec": 730.0,
}
_driver_meta = {
    "temp":  {"label": "Average cell temperature", "color": "#d35f5f",
              "x_transform": lambda x: x,         "x_unit": "°C",
              "x_fmt": lambda v: f"{v:.0f} °C"},
    "dod":   {"label": "Average DoD",              "color": "#7a5cff",
              "x_transform": lambda x: x * 100.0, "x_unit": "%",
              "x_fmt": lambda v: f"{v:.0f}%"},
    "crate": {"label": "Average C-rate",           "color": "#3b7dd8",
              "x_transform": lambda x: x,         "x_unit": "C",
              "x_fmt": lambda v: f"{v:.2f} C"},
    "soc":   {"label": "Average rest SoC",         "color": "#1aa179",
              "x_transform": lambda x: x * 100.0, "x_unit": "%",
              "x_fmt": lambda v: f"{v:.0f}%"},
    "fec":   {"label": "Cycles per day",           "color": "#e0a83b",
              "x_transform": lambda x: x / 365.0, "x_unit": " c/d",
              "x_fmt": lambda v: f"{v:.1f} c/d"},
}

# ── Inline chart: years OR €/MWh small multiples, 1×5 (absolute, switchable) ─────
_order = ["temp", "dod", "crate", "soc", "fec"]
_GAIN, _LOSS = "#1a7a3e", "#b42020"
_AXIS_GREY = "#555"

_CAPEX_EUR_PER_KWH = 180.0


def _cost_eur_per_mwh(driver_key: str, x_raw: float, years: float) -> float:
    if years <= 0:
        return float("inf")
    fec = x_raw if driver_key == "fec" else 730.0
    dod = x_raw if driver_key == "dod" else 0.80
    return _CAPEX_EUR_PER_KWH * 1000.0 / (years * fec * dod)


def _cost_eur_per_cycle(driver_key: str, x_raw: float, years: float, capacity_mwh: float) -> float:
    if years <= 0:
        return float("inf")
    fec = x_raw if driver_key == "fec" else 730.0
    return _CAPEX_EUR_PER_KWH * capacity_mwh * 1000.0 / (years * fec)


_series = {}
_all_costs_delta: list[float] = []
_all_years_delta: list[float] = []
# Each sweep is an independent MC estimate of the same baseline point;
# average them so every panel shares one canonical baseline, not per-sweep noise.
_baseline_y_per_driver: list[float] = []
for k in _order:
    pts = [p for p in _curve_points[k] if p["y"] is not None]
    xs_raw = [p["x"] for p in pts]
    ys_abs = [p["y"] for p in pts]
    b_raw = _baseline_x[k]
    _baseline_y_per_driver.append(float(np.interp(b_raw, xs_raw, ys_abs)))
_canonical_b_y = float(np.mean(_baseline_y_per_driver))
_canonical_b_cost = _cost_eur_per_mwh("temp", 25.0, _canonical_b_y)

for k in _order:
    pts = [p for p in _curve_points[k] if p["y"] is not None]
    xs_raw = [p["x"] for p in pts]
    ys_abs = [p["y"] for p in pts]
    b_raw = _baseline_x[k]
    costs = [_cost_eur_per_mwh(k, x, y) for x, y in zip(xs_raw, ys_abs)]
    costs_delta = [c - _canonical_b_cost for c in costs]
    years_delta = [y - _canonical_b_y for y in ys_abs]
    _all_costs_delta.extend(costs_delta)
    _all_years_delta.extend(years_delta)
    _series[k] = {
        "xs_disp": [_driver_meta[k]["x_transform"](x) for x in xs_raw],
        "costs_delta": costs_delta,
        "years_delta": years_delta,
        "b_disp": _driver_meta[k]["x_transform"](b_raw),
    }


def _pad_range(vals: list[float], pad: float = 0.10) -> list[float]:
    lo, hi = min(vals), max(vals)
    span = hi - lo
    return [lo - pad * span, hi + pad * span]


def _fmt_delta_eur(v: float) -> str:
    sign = "+" if v >= 0 else "−"
    return f"{sign}€{abs(v):.0f}"


def _fmt_delta_yr(v: float) -> str:
    sign = "+" if v >= 0 else "−"
    return f"{sign}{abs(v):.1f} yr"


_view_labels = {
    "€/MWh throughput": {
        "series_key": "costs_delta",
        "y_label": "Δ €/MWh vs baseline",
        "y_range": _pad_range(_all_costs_delta),
        "fmt": _fmt_delta_eur,
        "higher_is_better": False,
        "hover_unit": " €/MWh",
        "abs_anchor": _canonical_b_cost,
        "abs_fmt": lambda v: f"€{v:.0f}/MWh",
    },
    "Years to end of life": {
        "series_key": "years_delta",
        "y_label": "Δ years vs baseline",
        "y_range": _pad_range(_all_years_delta),
        "fmt": _fmt_delta_yr,
        "higher_is_better": True,
        "hover_unit": " yr",
        "abs_anchor": _canonical_b_y,
        "abs_fmt": lambda v: f"{v:.1f} yr",
    },
}

_view_choice = st.radio(
    "Y axis",
    list(_view_labels.keys()) + ["€ per cycle"],
    horizontal=True,
    label_visibility="collapsed",
    key="lever_chart_view",
)

if _view_choice == "€ per cycle":
    _cap_col, _spacer = st.columns([1, 4])
    with _cap_col:
        _capacity_mwh = st.number_input(
            "Pack capacity (MWh)",
            min_value=1.0, max_value=1000.0, value=100.0, step=10.0,
            key="lever_chart_capacity",
            help="Plug in your pack size to see € per cycle for your own asset.",
        )
    _b_cost_cycle = _cost_eur_per_cycle("temp", 25.0, _canonical_b_y, _capacity_mwh)
    _all_cost_cycle_delta: list[float] = []
    for k in _order:
        pts = [p for p in _curve_points[k] if p["y"] is not None]
        xs_raw = [p["x"] for p in pts]
        ys_abs = [p["y"] for p in pts]
        cc = [_cost_eur_per_cycle(k, x, y, _capacity_mwh) for x, y in zip(xs_raw, ys_abs)]
        cc_delta = [c - _b_cost_cycle for c in cc]
        _series[k]["cost_cycle_delta"] = cc_delta
        _all_cost_cycle_delta.extend(cc_delta)

    def _fmt_delta_cycle(v: float) -> str:
        sign = "+" if v >= 0 else "−"
        a = abs(v)
        return f"{sign}€{a/1000:.1f}k" if a >= 1000 else f"{sign}€{a:.0f}"

    _view = {
        "series_key": "cost_cycle_delta",
        "y_label": f"Δ € per cycle vs baseline ({_capacity_mwh:.0f} MWh)",
        "y_range": _pad_range(_all_cost_cycle_delta),
        "fmt": _fmt_delta_cycle,
        "higher_is_better": False,
        "hover_unit": " €/cycle",
        "abs_anchor": _b_cost_cycle,
        "abs_fmt": lambda v: f"€{v/1000:.1f}k/cycle" if v >= 1000 else f"€{v:.0f}/cycle",
    }
else:
    _view = _view_labels[_view_choice]

fig1 = make_subplots(rows=1, cols=5, horizontal_spacing=0.045)

for i, k in enumerate(_order):
    meta = _driver_meta[k]
    s = _series[k]
    row, col = 1, i + 1
    color = meta["color"]
    xs_disp = s["xs_disp"]
    b_disp = s["b_disp"]
    ys = s[_view["series_key"]]
    b_val = 0.0
    axis_suffix = "" if i == 0 else str(i + 1)

    # Horizontal baseline-value reference line
    fig1.add_shape(
        type="line",
        xref=f"x{axis_suffix} domain", yref=f"y{axis_suffix}",
        x0=0, x1=1, y0=b_val, y1=b_val,
        line=dict(color="#888", width=1.1, dash="dash"),
    )
    # Main curve
    fig1.add_trace(
        go.Scatter(
            x=xs_disp, y=ys, mode="lines",
            line=dict(color=color, width=2.8, shape="spline"),
            hovertemplate=f"%{{x:.2f}}{meta['x_unit']}<br>%{{y:.1f}}{_view['hover_unit']}<extra></extra>",
            showlegend=False,
        ),
        row=row, col=col,
    )
    # Endpoint markers
    lo_y, hi_y = ys[0], ys[-1]
    fig1.add_trace(
        go.Scatter(
            x=[xs_disp[0], xs_disp[-1]], y=[lo_y, hi_y],
            mode="markers", marker=dict(size=9, color=color),
            hoverinfo="skip", showlegend=False, cliponaxis=False,
        ),
        row=row, col=col,
    )
    # Endpoint labels — absolute values, coloured vs baseline
    for xd, yv, side in [(xs_disp[0], lo_y, "right"), (xs_disp[-1], hi_y, "left")]:
        is_gain = (yv >= b_val) if _view["higher_is_better"] else (yv <= b_val)
        fig1.add_annotation(
            x=xd, y=yv, text=f"<b>{_view['fmt'](yv)}</b>", showarrow=False,
            xshift=8 if side == "right" else -8,
            yshift=12 if yv >= b_val else -12,
            xanchor="left" if side == "right" else "right",
            font=dict(size=12, color=_GAIN if is_gain else _LOSS),
            row=row, col=col,
        )
    # Vertical baseline guide
    fig1.add_shape(
        type="line",
        xref=f"x{axis_suffix}", yref=f"y{axis_suffix} domain",
        x0=b_disp, x1=b_disp, y0=0, y1=1,
        line=dict(color="#888", width=1.1, dash="dash"),
    )
    # Baseline value label at the baseline point — shows absolute anchor
    # (€31 / 9.8 yr) so the reader has a unit-aware number behind the 0 %.
    fig1.add_annotation(
        x=b_disp, y=b_val, text=_view["abs_fmt"](_view["abs_anchor"]),
        showarrow=False, yshift=-14,
        font=dict(size=11, color="#555"),
        bgcolor="rgba(246,243,236,0.85)",
        borderpad=2,
        row=row, col=col,
    )
    # Per-panel coloured driver title
    fig1.add_annotation(
        xref=f"x{axis_suffix} domain", yref=f"y{axis_suffix} domain",
        x=0.5, y=1.22, text=f"<b>{meta['label']}</b>",
        showarrow=False, xanchor="center",
        font=dict(size=14, color=color),
    )
    # X-axis: ticks on TOP — min, baseline (coloured), max
    tick_vals = [xs_disp[0], b_disp, xs_disp[-1]]
    tick_texts = [
        meta["x_fmt"](xs_disp[0]),
        f"<span style='color:{color}'><b>{meta['x_fmt'](b_disp)}</b></span>",
        meta["x_fmt"](xs_disp[-1]),
    ]
    fig1.update_xaxes(
        row=row, col=col,
        side="top",
        tickvals=tick_vals,
        ticktext=tick_texts,
        showline=True, linecolor=_AXIS_GREY, linewidth=1,
        ticks="outside", ticklen=5, tickcolor=_AXIS_GREY,
        tickfont=dict(size=12, color="#222"),
        showgrid=False, zeroline=False,
        range=[xs_disp[0], xs_disp[-1]],
    )
    # Y-axis: only leftmost column
    if col == 1:
        fig1.update_yaxes(
            row=row, col=col, range=_view["y_range"],
            showline=True, linecolor=_AXIS_GREY, linewidth=1,
            ticks="outside", ticklen=5, tickcolor=_AXIS_GREY,
            tickfont=dict(size=11, color="#222"),
            showgrid=False, zeroline=False,
            title_text=_view["y_label"],
            title_font=dict(size=11, color=_AXIS_GREY),
        )
    else:
        fig1.update_yaxes(
            row=row, col=col, range=_view["y_range"],
            showline=False, showticklabels=False,
            showgrid=False, zeroline=False, ticks="",
        )

fig1.update_layout(
    height=340, margin=dict(l=60, r=15, t=75, b=20),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
)
st.markdown(
    """
    <style>
    .driver-wide-marker { display: none; }
    .stElementContainer:has(.driver-wide-marker) + .stElementContainer {
        width: min(1400px, 100vw - 40px) !important;
        max-width: min(1400px, 100vw - 40px) !important;
        margin-left: calc((100% - min(1400px, 100vw - 40px)) / 2) !important;
    }
    .stElementContainer:has(.driver-wide-marker) + .stElementContainer [data-testid="stPlotlyChart"] {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Mobile: force horizontal scroll on the 5-panel chart so panels stay
       readable instead of crushing to 60px each. */
    @media (max-width: 768px) {
        .stElementContainer:has(.driver-wide-marker) + .stElementContainer {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
        }
        .stElementContainer:has(.driver-wide-marker) + .stElementContainer [data-testid="stPlotlyChart"] {
            min-width: 1000px !important;
            width: 1000px !important;
        }

        /* Stack the preset button row, slider row, and metric row vertically
           on narrow viewports — default Streamlit columns don't wrap. */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    </style>
    <div class="driver-wide-marker"></div>
    """,
    unsafe_allow_html=True,
)
st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
render_chart_caption(
    "Axes show deltas from baseline. Baseline anchor ≈ €31/MWh and 9.8 years "
    "to 70 % SoH. DoD barely moves the €/MWh bill but cuts years-to-EOL by "
    "~4 yr; cycles/day behaves similarly. Rest SoC moves both. Temperature "
    "here is <b>cell-internal</b>, so the C-rate panel already includes "
    "self-heating; at fixed ambient the two aren't fully independent."
)
render_takeaway(
    "Temperature and C-rate dominate the €/MWh bill. DoD barely moves it at all."
)

st.markdown(
    """
Three things jump out.

**Rest SoC is the invisible lever.** The pack sits idle most of the day; where it rests decides how fast it ages. Zero extra cycles, zero extra MWh — every year of life rest SoC costs lands straight on the €/MWh bill. An arbitrage pack parked at 85% waiting for the morning peak carries a higher per-MWh cost than an FCR pack resting near 50%. No cycle counter shows it; the chart does.

**Cycles per day — not what the counter suggests.** Running the pack harder (2 → 2.5 c/d) pushes the €/MWh bill slightly *below* baseline; running it lighter (2 → 1 c/d) pushes it *up* by ~€10. Calendar aging runs on wall-clock time: a pack that cycles rarely still ages, spreading CAPEX across fewer MWh. The industry's headline metric misranks the lever — implying gentle is cheaper when it isn't.

**Temperature and C-rate dominate.** Both curves bend sharply: +10 °C or a doubling of C-rate each move the bill by more than rest SoC and cycles/day combined. These are the two levers that decide whether a pack makes its warranty.

Duration doesn't appear as its own panel but hides inside C-rate. A 1h battery discharging a full cycle runs at 1C; a 4h battery at 0.25C. For the same daily dispatch, the short-duration pack sits higher on the C-rate axis, and the C-rate panel is where that shows up. The baseline assumes a 2h pack at 0.5C (typical arbitrage). A 1h pack chasing the same revenue would run at ~1C — landing on the steep part of the curve.
"""
)

# ── The DoD invariance ──────────────────────────────────────
st.markdown("---")
st.markdown(
    """
### Depth of discharge is a zero-cost lever

Deeper cycles age the pack faster, but each cycle delivers more energy.
The two effects cancel. The cell carries a fixed **lifetime-throughput
budget** (~5.7 MWh/kWh for this preset); DoD decides whether you spend
it in many shallow cycles or few deep ones.

That's why the €/MWh curve is nearly flat: cycling at 50% or 95% DoD
costs about the same per MWh. Many dispatch optimisers treat DoD as
**the** cost-of-cycling parameter: deeper = faster fade = more
expensive. Within the LFP operating window, that framing is wrong.

This is the Ah-throughput picture of LFP — physics and citations in the
methodology below. The pattern is LFP-shaped: NMC behaves differently,
and even LFP bends above ~95 % DoD.

**Practical implication:** within the LFP window, DoD decides **how fast
you spend the budget**, not **how much each MWh costs**. Choosing 95%
over 80% compresses the same lifetime MWh into fewer years, with more
energy per cycle to trade. Whether that compression is worth doing
depends on price spreads — the question
[*Cycles & Marginal Value*](https://de-bess-cycles.streamlit.app)
answers — not on cost-per-MWh.
"""
)

# ── Interactive — "Build your own schedule" ─────────────────
st.markdown("---")
st.markdown("## Build your own schedule")
st.caption(
    "Pick a preset to reproduce a finding from the note, then swing the sliders to stress-test it."
)

_INTERACTIVE_PRESETS = {
    "Typical arbitrage": {
        "vals": dict(dod=0.80, fec=730, mean_soc=0.55, c_rate=0.50, temp=25),
        "desc": "Two cycles a day, moderate depth, pack parks near the middle. "
                "Close to what a well-behaved German arbitrage operator looks like.",
    },
    "Aggressive arbitrage": {
        "vals": dict(dod=0.95, fec=730, mean_soc=0.85, c_rate=0.50, temp=25),
        "desc": "Day-ahead peak-to-peak: 95% swings, pack parked at 85% "
                "overnight. The depth + rest-SoC combination burns years "
                "off life.",
    },
    "FCR-heavy": {
        "vals": dict(dod=0.55, fec=1095, mean_soc=0.50, c_rate=0.25, temp=25),
        "desc": "50% more cycles per year than arbitrage — but shallow, slow, and "
                "resting at midpoint. Many cycles, little fade: the counter "
                "overstates the wear.",
    },
    "Texas summer": {
        "vals": dict(dod=0.90, fec=730, mean_soc=0.65, c_rate=0.50, temp=30),
        "desc": "ERCOT-style duty on an HVAC-cooled site: cell-internal 30 °C "
                "vs 25 °C baseline. The extra 5 °C costs years, not months — "
                "the climate lever on top of an aggressive schedule.",
    },
    "Gentle": {
        "vals": dict(dod=0.60, fec=730, mean_soc=0.50, c_rate=0.30, temp=25),
        "desc": "What a warranty-preserving trader looks like: shallow cycles, "
                "mid-point rest SoC, slow C-rate, cool cell.",
    },
}
_DEFAULT_PRESET = "Typical arbitrage"
for _k, _v in _INTERACTIVE_PRESETS[_DEFAULT_PRESET]["vals"].items():
    st.session_state.setdefault(f"ddr_{_k}", _v)
st.session_state.setdefault("ddr_preset", _DEFAULT_PRESET)

def _apply_preset(name: str) -> None:
    for k, v in _INTERACTIVE_PRESETS[name]["vals"].items():
        st.session_state[f"ddr_{k}"] = v
    st.session_state["ddr_preset"] = name

_active_preset = st.session_state.get("ddr_preset", _DEFAULT_PRESET)

preset_cols = st.columns(len(_INTERACTIVE_PRESETS))
for pc, name in zip(preset_cols, _INTERACTIVE_PRESETS.keys()):
    pc.button(
        name, on_click=_apply_preset, args=(name,),
        use_container_width=True,
        type="primary" if name == _active_preset else "secondary",
    )
st.markdown(
    f"<div style='display:flex;gap:0.5em;align-items:baseline;"
    f"margin:0.3em 0.2em 0.8em 0.2em;max-width:1000px;font-size:0.9em'>"
    f"<span style='color:#1aa179;font-weight:700;flex-shrink:0'>"
    f"{_active_preset} →</span>"
    f"<span style='color:#444'>{_INTERACTIVE_PRESETS[_active_preset]['desc']}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

_cols = st.columns(5)
with _cols[0]:
    dod = st.slider("DoD", 0.50, 1.00, step=0.05, key="ddr_dod")
with _cols[1]:
    fec = st.slider("Cycles/yr", 300, 1500, step=10, key="ddr_fec",
                    help="Full-equivalent cycles per year. 730 ≈ 2/day.")
with _cols[2]:
    mean_soc = st.slider("Rest SoC", 0.20, 0.90, step=0.05, key="ddr_mean_soc",
                         help="Average SoC between trades.")
with _cols[3]:
    c_rate = st.slider("C-rate", 0.10, 1.00, step=0.05, key="ddr_c_rate")
with _cols[4]:
    temp = st.slider("Cell T (°C)", 10, 40, step=1, key="ddr_temp")

duty = DutyCycle.from_mean(
    fec_per_year=float(fec),
    mean_dod=float(dod),
    mean_soc=float(mean_soc),
    mean_crate=float(c_rate),
    mean_temp_C=float(temp),
)
preset = PRESETS["baseline_fleet"]

years = np.arange(0.0, 20.01, 0.5)
rng = np.random.default_rng(0)
rows = []
for yr in years:
    if yr <= 0:
        rows.append((0.0, 1.0, 1.0, 1.0))
        continue
    p10, p50, p90 = project_capacity_detailed(
        duty=duty, years=yr, preset=preset, n_mc=300, return_kind="distribution", rng=rng
    )
    rows.append((float(yr), p10, p50, p90))
soh_df = pd.DataFrame(rows, columns=["year", "p10", "p50", "p90"])

eol = preset.eol_capacity_fraction
# Median (p50) to match the driver response curves above. The pack p10
# convention used in revenue models runs ~10 % shorter; flagged in the
# caption below the chart.
below = soh_df[soh_df["p50"] <= eol]
years_to_eol = float(below["year"].iloc[0]) if len(below) else float("nan")

# Cost-of-cycle: replacement pack (≈180 €/kWh) amortised flat over lifetime
# throughput in MWh. Schimpe 2018 LFP benchmark sits near 13 €/MWh.
_PACK_REPLACEMENT_EUR_PER_KWH = 180.0
if not np.isnan(years_to_eol) and years_to_eol > 0 and dod > 0:
    lifetime_mwh_per_kwh = years_to_eol * fec * dod / 1000.0
    eur_per_mwh_cycle = _PACK_REPLACEMENT_EUR_PER_KWH / max(lifetime_mwh_per_kwh, 1e-6)
else:
    eur_per_mwh_cycle = float("nan")

col_a, col_b = st.columns([2, 1])
with col_a:
    f = go.Figure()
    # Truncate model traces at EOL — below 70 % SoH the two-channel kernel
    # stops being reliable (LAM knee not modelled). The dashed red segment
    # added further down covers that region illustratively.
    _valid = soh_df[soh_df["p50"] >= eol - 1e-6].copy()
    if len(_valid) > 0 and len(_valid) < len(soh_df):
        # Anchor the last valid point exactly at EOL so the main trace ends
        # cleanly at 70 % rather than hanging a little above or below.
        _anchor_x = float(np.interp(eol, soh_df["p50"][::-1], soh_df["year"][::-1]))
        _anchor = pd.DataFrame([[_anchor_x, eol, eol, eol]], columns=_valid.columns)
        _valid = pd.concat([_valid, _anchor], ignore_index=True)
    f.add_trace(go.Scatter(x=_valid["year"], y=_valid["p50"], name="median (model)", line=dict(color="#0b5fff")))
    f.add_trace(
        go.Scatter(
            x=np.concatenate([_valid["year"], _valid["year"][::-1]]),
            y=np.concatenate([_valid["p90"], _valid["p10"][::-1]]),
            fill="toself",
            fillcolor="rgba(11,95,255,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="p10–p90",
        )
    )
    f.add_hline(y=eol, line_dash="dot", line_color="#888", annotation_text=f"EOL {int(eol*100)}%")

    # Illustrative knee: past 70 % SoH the physics model stops being reliable
    # (LAM kicks in, fade accelerates). Show a muted dashed extension so the
    # reader sees what typically happens — not a model output.
    if not np.isnan(years_to_eol) and years_to_eol < 19.0:
        knee_end_year = min(years_to_eol + 1.8, 20.0)
        f.add_trace(go.Scatter(
            x=[years_to_eol, knee_end_year],
            y=[eol, 0.50],
            mode="lines",
            line=dict(color="#c15a5a", width=2, dash="dash"),
            name="illustrative knee",
            hovertemplate="Past EOL the curve typically accelerates (LAM).<extra></extra>",
        ))
        f.add_annotation(
            x=knee_end_year, y=0.50,
            text="<i>illustrative knee<br>(LAM acceleration)</i>",
            showarrow=False,
            xanchor="right", yanchor="bottom",
            font=dict(size=10, color="#c15a5a"),
        )

    f.update_layout(
        xaxis_title="years",
        yaxis_title="SoH",
        yaxis=dict(range=[0.4, 1.0]),
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
with col_b:
    st.metric(
        "€ per MWh throughput",
        f"{eur_per_mwh_cycle:.1f}" if not np.isnan(eur_per_mwh_cycle) else "—",
    )
    st.metric("Years to EOL (median cell)",  f"{years_to_eol:.1f}" if not np.isnan(years_to_eol) else "> 20")
    if not (preset.temp_range_C[0] <= temp <= preset.temp_range_C[1]):
        st.warning(f"Temperature outside preset range {preset.temp_range_C}.")
    if c_rate > 2.0:
        st.warning("C-rate above calibration range (>2C).")

st.caption(
    "€/MWh throughput = replacement cost (~180 €/kWh) divided across lifetime discharge MWh "
    "(single-direction — charging not double-counted). For a 100 MWh pack at baseline, one "
    "80% DoD cycle discharges 80 MWh — wear bill ~€2.4k. "
    "Schimpe 2018 benchmarked ~13 €/MWh at 2018-era CAPEX (~€80/kWh); at today's €180/kWh the "
    "same arithmetic gives ~€30/MWh. Values well above that signal a pack working itself to "
    "death faster. The <span style='color:#c15a5a'>dashed red segment past 70 % SoH</span> is "
    "illustrative, not a model output — LAM kicks in, fade accelerates (see methodology)."
)

# ── Methodology ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## Data Sources & Methodology")

with st.expander("Which cell these curves are built on"):
    st.markdown(
        """
All response curves, the DoD table, and the interactive above use the
`baseline_fleet` preset in `lib.models.degradation`. It is **not a real
cell** — it is a synthetic fleet-average LFP/graphite surrogate
describing a typical stationary pack, not any vendor's datasheet. The
kernel form (Wang 2011 cycle × Naumann 2018 calendar, two-channel
Arrhenius) is pinned to real LFP data; pre-factors are internally
calibrated.

Four real-cell presets are also calibrated in `lib.models.degradation`
— EVE LF280K, CATL EnerC+ 306 Ah, BYD MC Cube-T, Trina Elementa 280 Ah
— and sit behind revenue/warranty notes elsewhere in the series. Treat
absolute years-to-EOL numbers as illustrative; the lever ranking —
temperature and C-rate dominating, DoD nearly flat — reproduces across
all four real-cell presets.
"""
    )

with st.expander("Where this model stops working"):
    st.markdown(
        """
- **Chemistry.** NMC has a different calendar-aging shape (stronger
  SoC dependence, lower Ea) and needs a separate kernel — not in scope
  here.
- **Temperature — calibration window.** 15–60 °C calibrated (Wang 2011
  + Naumann 2018 coverage). At the 60 °C edge the Naumann Arrhenius
  power-law breaks down — independently confirmed on Stanford Lam 2024
  K2 cells — so the calendar channel should not be trusted beyond
  ~50 °C. Cold end sparsely sampled; use with care below 15 °C.
- **Temperature — cell-internal vs ambient.** The Arrhenius terms
  consume cell-internal temperature, not ambient.
  `project_capacity_detailed` takes cell-internal T directly (the
  default, used throughout this note);
  `project_capacity_detailed_from_ambient` applies
  `T_cell = T_amb + k · C²` under active cycling (typical LFP prismatic
  `k = 2 °C/C²` → +8 °C at 2C, +0.5 °C at 0.5C). Calendar integration
  stays on ambient because active cycling is <20 % of wall-clock time.
  A full RC pack thermal model with cooling dynamics and module-level
  gradients is out of scope here.
- **C-rate.** 0C–2C. Below 0.5C the cycle term extrapolates linearly;
  above 2C the estimate is a conservative overestimate. Grid-scale LFP
  stationary operation rarely exits this window.
- **SoC representation.** Bucketed hours per year — transition
  dynamics, partial cycles, and calendar/cycle coupling inside a cycle
  are not modelled. For stationary LFP the bucketing approximation
  holds; it will need revisiting when NMC enters.
- **Knee point.** LFP cells fade gently for most of their life — a slow, near-linear slope driven by lithium-inventory loss (LLI): lithium gets pinned into the solid-electrolyte interphase and lost to cycling. Past ~70% SoH, a second mechanism kicks in: active material begins to detach (LAM), and fade accelerates sharply — the "knee". The model stops at 70% SoH — upstream of the knee. Second-life modelling would need a knee-aware kernel.
- **Pack effects.** Cell-to-cell spread is parametric (Severson 2019
  CoV 8%, split across cycle and calendar channels, combined in
  quadrature). Thermal gradients and string-level imbalance aren't
  modelled from pack topology; field data anchors the span — pack fade
  runs ~10–20% faster than best-cell fade (Schimpe 2018; Reniers 2019).
  The operator lever is monitoring discipline, not BMS spec.
  Use `return_kind="min"` on the Monte
  Carlo projection for warranty/insurance floors.
- **Field validation beyond calibration cells.** SNL Preger 2020
  (30 A123 18650 cells, cycling) and Stanford Lam 2024 (80 K2 Energy
  LFP18650 cells, calendar) are wired in as out-of-sample regression
  tripwires — they flag if a future recalibration makes either dataset
  fit worse, but they are not pass criteria. No public dataset exists
  for modern large-format prismatic LFP cells, so prismatic-specific
  bias is inferred from small-format anchors.
"""
    )

with st.expander("Why this kernel — the model-choice problem"):
    st.markdown(
        """
[Humiston, Cetin, de Queiroz (Energies 2026)](https://www.mdpi.com/1996-1073/19/4/1056) ran the same BESS project
through three degradation formulations — linear-calendar (LC),
energy-throughput (ET), and cycle-based rainflow (CB) — on 2024 ERCOT
data. LC and ET both returned modest annual capacity loss (~2 %) and
left the project profitable; CB rainflow predicted substantially
heavier degradation and flipped the valuation deeply negative. Not
dispatch strategy. Not chemistry. The **model the analyst chose**
decided whether the asset looked like a good deal. The same sensitivity
shows up at our parameter scale: swinging the cycle pre-factor `k_cyc`
by ±20 % moves lifetime NPV by a magnitude comparable to picking one
chemistry over another.

A ranking model must be tighter than the choices it ranks. Three
things rule out the common shortcuts:

- **Rainflow-only.** Captures cycle depth but attributes all fade to
  cycling without a calendar channel. Under arbitrage-like duty this
  over-penalises wear — the formulation that flipped Humiston 2026's
  project to deeply negative.
- **Flat throughput / €-per-cycle models.** Cheap but blind to rest SoC
  (no calendar channel) and price every cycle the same whether fresh
  or tired. Conservative on headline NPV, but cannot rank the levers
  this note is about.
- **Single-parameter Arrhenius.** One activation energy hides that
  cycle and calendar channels respond to temperature differently
  (Ea ≈ 0.30 vs 0.55 eV). Collapsing them misprices climate.

The two-channel Wang 2011 + Naumann 2018 kernel used here keeps cycle
and calendar separable, gives SoC its own cubic dependence, and lets
each channel carry its own Arrhenius. It is calibrated end-to-end
against the same Sony/Murata LFP cell Naumann tested (3.2 % median
residual, in-distribution) and cross-checked on three further datasets,
in two independence tiers. **Calibration-referential** (same Naumann
lineage, not independent): SimSES — TU Munich's open-source reference —
gives 0.048 median |ΔSoH| over 20 years. This is a parity check, not
independent confirmation. **Out-of-sample tripwires** (independent
cells, different manufacturers): Sandia SNL Preger 2020 for cycling
(30 A123 18650 cells, T × DoD × C-rate grid); Stanford Lam 2024 for
calendar (80 K2 18650 cells, 24–85 °C × 50/100 % SoC, up to 8 years).
Both disclose bias at their physics edges rather than blocking release.
"""
    )

with st.expander("Why DoD is a zero-cost lever — the Ah-throughput picture"):
    st.markdown(
        """
This is the **Ah-throughput** (or **throughput-limited-aging**)
picture of LFP — an interpretive framing, not a direct empirical law. It dates back to
[Drouilhet & Johnson 1997](https://www.nrel.gov/docs/legosti/fy97/21978.pdf)
for lead-acid and was generalised to Li-ion across the 2010s.
[Preger et al. 2020 (Sandia)](https://www.batteryarchive.org/snl_study.html)
sweep on 30 A123 18650 LFP cells showed that among the chemistries in
the study LFP was the least DoD-sensitive, with the mid-band 40 – 60 %
window outperforming the 0 – 100 % window and 20 – 80 % landing at
comparable wear — directionally consistent with a near-Ah-throughput
picture, but **not** a tight 1/DoD scaling: narrow-DoD traces in the
same dataset are noisy enough that we drop them from our own validation
loader. [Schmalstieg et al. 2014](https://doi.org/10.1016/j.jpowsour.2014.02.012)
went further for NMC and formulated the cycle channel as
√(ΔAh-throughput) rather than as a count of cycles at all.

**What the kernel actually does, and why it still lands flat.** This
model does *not* give you the invariance through a single sub-linear
DoD exponent. The cycle channel is Wang 2011 linear-in-FEC (`z_cyc = 1`)
multiplied by an empirical super-linear DoD term from
[Xu et al. 2018](https://doi.org/10.1109/TSG.2016.2578950) —
effectively `Q_cyc ∝ FEC · DoD^1.5` at constant temperature and
C-rate. **Taken alone this would make deeper cycles more than proportionally
damaging — the opposite of the flat €/MWh curve.**
The flat €/MWh curve emerges from interplay with the calendar channel:
calendar fade is DoD-independent, so a pack that cycles deeper reaches
EOL sooner — before the calendar term accumulates as much loss. Years-
to-EOL shrinks roughly as 1/DoD, and `years × FEC × DoD` ≈ constant
falls out of two-channel balance, not a single α < 1 on cycling.

**Limits.** The flat-cost pattern holds mid-DoD on LFP at moderate
temperature with baseline calendar/cycle split. Above ~95 % DoD the
stress function bends super-linearly; past the knee (below 70 % SoH) a
different mechanism takes over. For cycle-dominated duty (hot cell,
high FEC) the super-linear DoD term shows through — deeper cycles cost
more per MWh. The pattern is LFP- and window-specific, not universal.
"""
    )

with st.expander("Kernel equations"):
    st.markdown(
        r"""
Two-channel Qloss. Cycle and calendar channels are independent; cell-to-cell
noise is drawn separately on each and combined in quadrature.

All Arrhenius terms are evaluated as **ratios referenced to 25 °C**, not as
absolute $e^{-E_a/RT}$ factors — the pre-factors $k_{\text{cyc}}$, $k_{\text{cal}}$
are the losses at 25 °C, and temperature only bends them via the ratio.

$$
\text{Arr}(T, E_a) \;\equiv\; \exp\!\left(-\frac{E_a}{k_B}\left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right)\right),
\quad T_{\text{ref}} = 298.15\ \text{K}
$$

$$
Q_{\text{loss,cyc}} = k_{\text{cyc}} \cdot \text{Arr}(T_{\text{cell}}, E_{a,\text{cyc}})
  \cdot (\text{FEC} \cdot \text{DoD})^{z_{\text{cyc}}}
  \cdot C_{\text{rate}}^{c_{\text{exp}} \cdot z_{\text{cyc}}}
  \cdot f_{\text{DoD}}(\text{DoD})
$$

$$
Q_{\text{loss,cal}} = k_{\text{cal}} \cdot \text{Arr}(T, E_{a,\text{cal}})
  \cdot t^{\beta_{\text{cal}}}
  \cdot \sum_{s} \frac{h_s}{\sum_b h_b} \cdot k_{\text{cal}}(\text{SoC}_s)
$$

$$
\sigma \;=\; \sqrt{\,(\text{CoV}_{\text{cyc}} \cdot |Q_{\text{loss,cyc}}|)^2
               + (\text{CoV}_{\text{cal}} \cdot |Q_{\text{loss,cal}}|)^2\,},
\quad \varepsilon_{\text{cell}} \sim \mathcal{N}(0, \sigma)
$$

$$
\text{SoH}(t) = 1 - Q_{\text{loss,cyc}} - Q_{\text{loss,cal}} - \varepsilon_{\text{cell}}
$$

where $T_{\text{cell}} = T_{\text{amb}} + c_{\text{sh}} \cdot C_{\text{rate}}^2$
(self-heating coefficient $c_{\text{sh}} = 0$ in the default cell-internal
entry point; the ambient entry point `project_capacity_detailed_from_ambient`
sets $c_{\text{sh}} = 2$ °C/C² for typical LFP prismatic), $k_{\text{cal}}(\text{SoC}) = a + b\,u + c\,u^3$
with $u = \max(\text{SoC} - 0.5,\, 0)$ (Naumann continuous form, flat below
the thermodynamic midpoint), $f_{\text{DoD}}(\text{DoD}) = (\text{DoD}/0.80)^{0.5}$
(empirical super-linear multiplier), and bucket hours $h_s$ are normalised by
the total duty-hour budget $\sum_b h_b$ (== 8760 for a full-year duty).

Cycle term: Wang, Liu, Hicks-Garner et al. 2011 LFP Arrhenius + power-law form.
FEC exponent $z_{\text{cyc}} = 1$ (near-linear) from Naumann 2020 /
Sarasketa-Zabala 2014 LFP stationary-duty fits rather than Wang's
$z \approx 0.55$, which produces year-1 front-loading inconsistent with LFP
field data.

Calendar term: Naumann et al. 2018 LFP calendar model. SoC enters as a
continuous cubic above the thermodynamic midpoint,

$$
k_{\text{cal}}(\text{SoC}) = a + b \cdot u + c \cdot u^3, \quad u = \max(\text{SoC} - 0.5, 0)
$$

flat below SoC=0.5 (no calendar acceleration below the midpoint), rising
monotonically above it. The duty's SoC histogram — whatever resolution the
caller supplies — is evaluated point-by-point through this function.

Cell-to-cell CoV $\sigma_{\text{CoV}} = 0.08$ from Severson 2019. Applied
channel-independent so a pack dominated by calendar aging and a pack dominated
by cycling see stochastic spread scaled to the actual source of fade, not a
fixed fraction of the cycle term.
"""
    )

with st.expander("Literature table"):
    st.markdown(
        """
| Paper | Role in this note |
|---|---|
| Wang, Liu, Hicks-Garner et al. 2011 — *[Cycle-life model for graphite-LiFePO4 cells](https://doi.org/10.1016/j.jpowsour.2010.11.134)* | Cycle-life power law (LFP) — cycle channel |
| Naumann et al. 2018 — *[Analysis and modeling of calendar aging of a commercial LiFePO4/graphite cell](https://doi.org/10.1016/j.est.2018.01.019)* | Calendar + SoC dependence (LFP) — calendar channel |
| Naumann et al. 2020 — *[Analysis and modeling of cycle aging of a commercial LiFePO4/graphite cell](https://doi.org/10.1016/j.jpowsour.2019.227666)* | LFP cycle aging refinement (used in SimSES parity) |
| Naumann et al. — *[Data for: Analysis and modeling of calendar aging of a commercial LiFePO4/graphite cell](https://data.mendeley.com/datasets/kxh42bfgtj/1)* (Mendeley, CC BY 4.0) | Raw calendar anchors |
| Severson et al. 2019 — *[Data-driven prediction of battery cycle life before capacity degradation](https://doi.org/10.1038/s41560-019-0356-8)* (Nature Energy) | Cell-to-cell CoV source |
| TU Munich — *[SimSES: open-source techno-economic simulation of stationary energy storage](https://gitlab.lrz.de/open-ees-ses/simses)* | Parity anchor for the Naumann kernel |
| Xu et al. 2018 — *[Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment](https://doi.org/10.1109/TSG.2016.2578950)* (IEEE TSG) | DoD super-linear exponent |
| Schimpe et al. 2018 — *[Comprehensive Modeling of Temperature-Dependent Degradation Mechanisms in Lithium Iron Phosphate Batteries](https://doi.org/10.1149/2.1181714jes)* (J. Electrochem. Soc.) | LFP flat-cost benchmark (~12.8 €/MWh) |
| Preger et al. 2020 — *[Degradation of Commercial Lithium-Ion Cells as a Function of Chemistry and Cycling Conditions](https://www.batteryarchive.org/snl_study.html)* (Sandia / batteryarchive.org) | SNL 18650 LFP cycling dataset |
| Lam et al. 2024 — *[Stanford Long-Term Calendar Aging Dataset](https://osf.io/ju325/)* (Joule / OSF) | K2 18650 LFP calendar dataset |
| Humiston, Cetin, de Queiroz 2026 — *[Evaluating Battery Degradation Models in Rolling-Horizon BESS Arbitrage Optimization](https://www.mdpi.com/1996-1073/19/4/1056)* (Energies) | Model-choice sensitivity in BESS valuation |
"""
    )

# ── Closing ─────────────────────────────────────────────────
st.markdown("---")
render_closing(
    "Part of an ongoing series on BESS merchant economics. Next: what a trader who "
    "prices this wear into every bid does differently — and how much revenue a "
    "warranty-respecting schedule leaves on the table."
)
render_footer()
