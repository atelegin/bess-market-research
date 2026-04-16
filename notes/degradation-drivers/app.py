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
    subtitle="Cycle count is a bad proxy for cell health. Here is what moves the needle.",
)

# ── Intro ────────────────────────────────────────────────────
st.markdown(
    """
Notes 1 and 2 used a single number for degradation — 1.8%/year, flat. It is
enough for headline fleet revenue, and nothing more. The moment you ask a
real design question — should I size 4 hours or 2, should I sit at 50% SoC
or swing to 80%, does hotter climate kill the asset — that single number
goes silent.

The industry still reports cell health in cycles. Cycle count is a
thermometer left in the freezer: it registers something, but it misses
most of what is happening. Two identical packs can finish the same year at
730 equivalent full cycles and have SoH 12 percentage points apart,
because one spent its evenings parked at 90% SoC and the other drifted at
55%. Calendar aging ran differently. Neither pack's cycle counter noticed.

This note takes the fleet-average fade apart. I rebuilt the degradation
model on two calibrated physics channels — Wang 2011 for cycle, Naumann
2018 for calendar with explicit SoC dependence — cross-checked against
SimSES (TU Munich's open-source reference). Then I swept the levers an
owner actually controls and converted each one into lifetime NPV. The
ranking that falls out is the subject of the first chart.

The short version: **DoD, rest SoC, and C-rate each move lifetime NPV
by tens of percent — more than any credible cross-cell envelope.** Cell
brand mostly sets the ceiling. How you dispatch decides where you land
under it.
"""
)

data = load_precomputed()

# ── The levers an owner actually controls ──────────────────
st.markdown("---")
st.markdown(
    """
### The levers an owner actually controls

Before the chart, a frame. The parameters that move lifetime NPV
through degradation fall into three buckets, and the distinction
matters because it tells you who owns the decision.

**Dispatch** — DoD, rest SoC, C-rate, cycles/day. Re-tunable every
trading day. The operator owns these end-to-end, even under a
fixed LTSA. This is where most of the chart lives.

**Design** — duration (hours of storage), chemistry, HVAC
specification. Locked at COD; the owner chooses once and lives
with it. Design levers do not appear as bars below for three
reasons: duration materialises through C-rate (already on the
chart), chemistry gets a separate treatment further down (public
datasheets do not support a head-to-head ranking), and HVAC is
manufacturer-set and runs to its own control loop — not an
operator decision after delivery.

**Given** — climate temperature, calendar age, cell-to-cell
variation. Not chosen at all, at least not on the time horizon of
a running asset. They still move NPV and still deserve a line
item: the question is whether they move it by more or less than
the dispatch choices the operator makes daily.

The chart ranks what we can swing. Two colours: **dispatch** in
blue (operator-owned), **given** in grey (the floor the physics
hands you).
"""
)

# ── Chart 1 — lever ranking ─────────────────────────────────
st.markdown("---")
render_chart_title("Each lever priced against baseline")
st.markdown(
    "<div style='color:#6b6b6b;font-size:0.9em;margin-bottom:0.5em'>"
    "Each parameter is a <b>20-year average</b>, held constant across asset life. "
    "Y-axis shows baseline value; bars show where NPV lands if that one parameter "
    "shifts. Colour of the row label marks the bucket: "
    "<span style='color:#0b5fff'><b>dispatch</b></span> (operator-owned) vs "
    "<span style='color:#6b6b6b'><b>given</b></span> (physics / climate)."
    "</div>",
    unsafe_allow_html=True,
)
lever = data["lever_sweep"].copy().sort_values("span").reset_index(drop=True)
BASELINE_VALUES = {
    "Temperature":  "25 °C",
    "C-rate":       "0.5C",
    "Rest SoC":     "55%",
    "Cycles / day": "2 c/d",
    "DoD":          "80%",
    "Imbalance":    "5 pp spread",
}
GROUP_COLORS = {"dispatch": "#0b5fff", "given": "#6b6b6b"}
lever["y_label"] = lever.apply(
    lambda r: f"<b style='color:{GROUP_COLORS.get(r['group'], '#111')}'>{r['lever']}</b>",
    axis=1,
)
_npv_base = float(data["baseline_npv_keur"])
lever["low_pct"] = lever["low_delta"] / _npv_base * 100.0
lever["high_pct"] = lever["high_delta"] / _npv_base * 100.0

GREEN = "#2ca02c"
RED = "#c15a5a"


def _fmt_end(value, pct):
    sign = "+" if pct >= 0 else "−"
    return f"  {value} · {sign}{abs(pct):.0f}%  "


# Convention: GAIN (green) always on the LEFT (negative x), LOSS (red) always on the RIGHT
# (positive x), regardless of whether improvement comes from lowering or raising the parameter.
# One-sided levers (e.g. Imbalance: baseline is best case, both perturbations hurt) show only
# the loss bar.
gain_x, gain_text, loss_x, loss_text = [], [], [], []
for _, r in lever.iterrows():
    low_pct, low_lbl = r["low_pct"], r["low_label"].split("(")[-1].rstrip(")")
    high_pct, high_lbl = r["high_pct"], r["high_label"].split("(")[-1].rstrip(")")
    if low_pct >= 0 and high_pct >= 0:
        gain_pct, gain_lbl = (low_pct, low_lbl) if low_pct > high_pct else (high_pct, high_lbl)
        loss_pct, loss_lbl = 0.0, ""
    elif low_pct < 0 and high_pct < 0:
        gain_pct, gain_lbl = 0.0, ""
        loss_pct, loss_lbl = (low_pct, low_lbl) if low_pct < high_pct else (high_pct, high_lbl)
    elif low_pct >= 0:
        gain_pct, gain_lbl = low_pct, low_lbl
        loss_pct, loss_lbl = high_pct, high_lbl
    else:
        gain_pct, gain_lbl = high_pct, high_lbl
        loss_pct, loss_lbl = low_pct, low_lbl
    gain_x.append(-abs(gain_pct))
    loss_x.append(abs(loss_pct))
    gain_text.append(_fmt_end(gain_lbl, gain_pct) if gain_lbl else "")
    loss_text.append(_fmt_end(loss_lbl, loss_pct) if loss_lbl else "")

fig_a = go.Figure()
fig_a.add_trace(
    go.Bar(
        x=gain_x,
        y=lever["y_label"],
        orientation="h",
        marker_color=GREEN,
        text=gain_text,
        textposition="outside",
        textfont=dict(size=13, color="#1a5a1a"),
        cliponaxis=False,
        hoverinfo="skip",
    )
)
fig_a.add_trace(
    go.Bar(
        x=loss_x,
        y=lever["y_label"],
        orientation="h",
        marker_color=RED,
        text=loss_text,
        textposition="outside",
        textfont=dict(size=13, color="#7a2828"),
        cliponaxis=False,
        hoverinfo="skip",
    )
)
fig_a.add_vline(x=0, line_color="#111", line_width=2)
fig_a.add_annotation(
    x=0, y=1.06, xref="x", yref="paper",
    text="<b>baseline</b>",
    showarrow=False, font=dict(size=12, color="#111"),
    yanchor="bottom",
)
for _, r in lever.iterrows():
    fig_a.add_annotation(
        x=0, y=r["y_label"],
        text=f"<b>{BASELINE_VALUES[r['lever']]}</b>",
        showarrow=False,
        font=dict(size=12, color="#111"),
        bgcolor="#fbf7ef",
        bordercolor="#111",
        borderwidth=1,
        borderpad=4,
        xanchor="center", yanchor="middle",
    )
fig_a.add_annotation(
    x=-1, y=1.06, xref="paper", yref="paper",
    text="<b>← better NPV</b>",
    showarrow=False, font=dict(size=12, color=GREEN),
    xanchor="left", yanchor="bottom",
)
fig_a.add_annotation(
    x=1, y=1.06, xref="paper", yref="paper",
    text="<b>worse NPV →</b>",
    showarrow=False, font=dict(size=12, color=RED),
    xanchor="right", yanchor="bottom",
)
_lim = max(max(abs(v) for v in gain_x), max(abs(v) for v in loss_x)) * 1.35
fig_a.update_layout(
    barmode="overlay",
    xaxis_title="",
    yaxis_title="",
    yaxis=dict(tickfont=dict(size=15)),
    xaxis=dict(range=[-_lim, _lim], zeroline=False, showticklabels=False),
    height=480,
    margin=dict(l=10, r=10, t=70, b=10),
    showlegend=False,
    plot_bgcolor="rgba(0,0,0,0)",
    bargap=0.45,
)
st.plotly_chart(fig_a, use_container_width=True, config={"displayModeBar": False})
render_chart_caption(
    f"NPV in % of baseline lifetime NPV (≈{_npv_base:.0f} kEUR/MW, year-1 revenue "
    f"120 kEUR/MW from Note 2, 20 yr, 8% discount). Kernel uses averaged inputs; "
    f"real sites with wide thermal or DoD swings fade worse than shown. "
    f"**Imbalance** is a weakest-cell proxy: string-level SoC spread shifts the "
    f"worst cell to deeper effective DoD and higher dwell than the string mean. "
    f"Span anchored to field pack-vs-cell gap (10–20%, Schimpe 2018 / Reniers 2019). "
    f"LTSA fixes the BMS; operator owns monitoring discipline and intervention cadence."
)
render_takeaway(
    "Dispatch levers (DoD, Rest SoC, C-rate, cycles/day) individually swing lifetime NPV by more than the given-side floor of climate temperature and cell imbalance combined. The levers the operator re-tunes every day dominate the ones physics hands them."
)

st.markdown(
    """
Two things in that chart are worth sitting with.

**DoD is not symmetric.** Going from 80% to 95% DoD costs more than going
from 80% to 60% saves. Cycle stress scales super-linearly with depth
(Xu 2018, α≈1.5 in our kernel), so the top 15% of depth hurts out of
proportion to the extra throughput it buys. Dispatch optimisers that
chase every EUR inside the 80–100% range are spending warranty headroom
they will not get back.

**Cycle count and DoD point opposite ways.** Cutting throughput is easy
wins — dropping from 2 c/d to 1 c/d (FCR-style) buys more NPV than
pushing DoD from 80% down to 60% does. Adding cycles hurts less than
adding depth: going from 2 to 2.5 c/d costs half of what going from 80%
to 95% DoD does. The old instinct "more cycles = more wear" is half
right and half misleading — depth beats count, every time.

**Rest SoC matters even without cycling.** *Rest SoC* is the average
state of charge the battery sits at between trades — where the pack
"lives" when it is not actively cycling. A pack that finishes each
afternoon topped up at 85% waiting for the morning peak has a high
rest SoC; one that idles near 50% (FCR-style) has a low one.

Raising rest SoC from 55% to 75% costs meaningful NPV with *zero*
extra cycles — that is pure calendar aging, the Naumann SoC³ term
doing its work (stress on lithium plating scales with the cube of how
far the cell sits above 50%). FCR-heavy operators who sit near 50% by
protocol get this for free. Arbitrage operators who end the day at 85%
are paying for it, and no cycle counter will flag it.

**Chemistry is a ceiling, not a strategy.** The cell datasheet sets the
physics headroom; the schedule decides how much of it survives to year
20. I am deliberately not ranking EVE / CATL / BYD / Trina head-to-head
here — see the note on cross-chemistry comparison below for why public
datasheets do not support that ranking. The claim this chart supports is
the narrower one: every operator lever above moves more NPV than a
credible cross-cell envelope would.
"""
)

# ── SoH panels (four scenarios) ────────────────────────────
st.markdown("---")
render_chart_title("Same cell, four duty cycles — trajectories diverge by year 5")
cols = st.columns(2)
scenario_labels = {
    "baseline": "Baseline (2 c/d, 80% DoD, 55% SoC)",
    "deep_dod": "Deep DoD (95%)",
    "high_soc": "High rest SoC (85% mean)",
    "c_rate_discipline": "C-rate discipline (0.25C)",
}
for i, (key, df) in enumerate(data["soh_panels"].items()):
    f = go.Figure()
    f.add_trace(go.Scatter(x=df["year"], y=df["p50"], name="median cell", line=dict(color="#0b5fff")))
    f.add_trace(
        go.Scatter(
            x=np.concatenate([df["year"], df["year"][::-1]]),
            y=np.concatenate([df["p90"], df["p10"][::-1]]),
            fill="toself",
            fillcolor="rgba(11,95,255,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="p10–p90 (cell variation)",
        )
    )
    f.add_hline(y=0.70, line_dash="dot", line_color="#888", annotation_text="EOL 70%")
    f.update_layout(
        title=scenario_labels.get(key, key.replace("_", " ").title()),
        xaxis_title="years",
        yaxis_title="SoH (fraction)",
        yaxis=dict(range=[0.6, 1.0]),
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=(i == 0),
    )
    cols[i % 2].plotly_chart(f, use_container_width=True, config={"displayModeBar": False})

render_chart_caption(
    "Same `baseline_fleet` preset, 500 Monte Carlo cells per scenario. "
    "Bands are cell-to-cell spread (CoV 8%, Severson 2019)."
)

st.markdown(
    """
Four trajectories, one cell. The baseline owner and the deep-DoD owner
diverge by year 5 and keep diverging — the deep-DoD asset hits EOL 70%
around year 14 where the baseline sits at 75% and still has runway.
High rest SoC does the same thing more quietly: no extra FEC, just
calendar curvature. C-rate discipline at 0.25C buys a year or two on
its own; stacked with DoD moderation it stretches further.
"""
)

# ── Chemistry — why no head-to-head chart ───────────────────
st.markdown("---")
st.markdown(
    """
### A note on cross-chemistry comparison

The natural next question is *which cell ages best under identical
duty*. I am not answering it here. The public anchors for EVE, CATL,
BYD, and Trina come from different documents, at different retention
targets (CATL quotes FEC-to-70%, the others to 80%), from different
labs under different implicit margins — spec sheets for some,
marketing pages for others. Lining up four trajectories on the same
axes would be a chart of how each vendor writes numbers, not a chart
of how the cells age.

A credible cross-chemistry ranking needs a head-to-head bench on
matched protocol. That exists in principle (third-party labs like DNV
and UL run them) but not in the open. Until one surfaces, the
lever-ranking chart above stands on its own message: **cell selection
sits inside a narrower envelope than any single dispatch choice**.
That holds whether BYD's 12 000-FEC marketing claim or EVE's 6 000-FEC
spec point is closer to pack reality — both envelopes are smaller than
the spread from DoD 60% to 95%.
"""
)

# ── EUR/FEC ruler ───────────────────────────────────────────
st.markdown("---")
render_chart_title("The EUR/FEC ruler — what one more cycle costs you")
st.markdown(
    """
Cost-of-cycle is the number every dispatch optimiser needs and almost
nobody computes right. The naive version (EoL replacement cost / total
lifetime FEC) hands you a flat €/MWh figure around 10–15 for LFP — the
Schimpe 2018 reference for ~12.8 €/MWh lives in this range. That flat
number is what a *throughput* degradation model produces, and it is the
first entry in the "models that flip profitability" row of the
Humiston 2026 result.

The degradation-aware version is state-dependent. The same cycle is
cheap when the pack is fresh and sitting at 50% SoC at 20 °C; it is
expensive when the pack is tired, dwelling at 85%, and trading into
warm weather. Our kernel computes the marginal NPV cost of one extra
cycle as a function of (current SoH, SoC, temperature) — that is the
Holtorf & Shin 2026 opportunity-cost framing, and it is the subject of
Note 3b. The point for this note: **flat €/MWh cost-of-cycle is wrong
in the direction that matters.** It under-prices cycling in the back
half of asset life, when every remaining cycle is expensive, and
over-prices it early, when cycles are nearly free.
"""
)
render_takeaway(
    "Flat €/MWh cost-of-cycle mis-prices degradation in both directions — under the early asset, "
    "over the tired asset. Dispatch economics need state-dependent cost."
)

# ── Climate case ────────────────────────────────────────────
st.markdown("---")
render_chart_title("Climate case — how much does temperature move the asset?")
preset = PRESETS["baseline_fleet"]
climate_fig = go.Figure()
climate_trajectories = {}
for T, color in [(15, "#2ca02c"), (25, "#0b5fff"), (30, "#ff8c00"), (35, "#c15a5a")]:
    duty = DutyCycle.from_mean(fec_per_year=730, mean_dod=0.80, mean_soc=0.55, mean_crate=0.5, mean_temp_C=float(T))
    xs = np.arange(0.0, 20.01, 1.0)
    rng = np.random.default_rng(2)
    ys = []
    for yr in xs:
        if yr <= 0:
            ys.append(1.0)
            continue
        _, p50, _ = project_capacity_detailed(
            duty=duty, years=float(yr), preset=preset, n_mc=200, return_kind="distribution", rng=rng
        )
        ys.append(p50)
    climate_trajectories[T] = (xs, ys)
    climate_fig.add_trace(
        go.Scatter(x=xs, y=ys, name=f"{T} °C mean", line=dict(color=color, width=2))
    )
climate_fig.add_hline(y=0.70, line_dash="dot", line_color="#888", annotation_text="EOL 70%")
climate_fig.update_layout(
    xaxis_title="years",
    yaxis_title="SoH (median)",
    yaxis=dict(range=[0.55, 1.0]),
    height=380,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(climate_fig, use_container_width=True, config={"displayModeBar": False})
render_chart_caption(
    "Baseline duty (2 c/d, 80% DoD, 55% SoC) at four ambient means. "
    "Calendar channel uses Naumann-form Arrhenius Ea ≈ 0.55 eV (≈53 kJ/mol, "
    "within Naumann 2018's reported 0.55–0.73 eV range across SoC); cycle channel "
    "Ea ≈ 0.30 eV (≈29 kJ/mol). X-axis treats `mean_temp_C` as the cell-internal "
    "temperature — there is no self-heating term in the kernel, so at high C-rate "
    "ambient should be adjusted upward (≈5–10 °C for 2C prismatic LFP) before "
    "being passed in."
)

st.markdown(
    """
The Modo 2025 fleet-level finding — ERCOT losing ~7% per 365 cycles vs
GB ~4% — shows up here. A well-conditioned German site at 25 °C mean
and a hot-climate site at 35 °C mean are not operating the same asset.
The gap compounds: the hot site reaches 70% SoH years earlier, and
every year before that carries more capacity fade per cycle. Cooling
design is an economic lever, not a hygiene item. Liquid cooling
(PowerTitan-class systems) buys back a lot of the delta in hot
climates; it is not free on colder sites.
"""
)

# ── Interactive — "Build your duty cycle" ───────────────────
st.markdown("---")
st.markdown("## Build your own duty cycle")
st.caption(
    "Single-cell math, real time. Results are illustrative — pack effects and "
    "cell-to-cell variation shown as a p10–p90 band."
)

col1, col2, col3 = st.columns(3)
with col1:
    dod = st.slider("DoD per cycle", 0.50, 1.00, 0.80, 0.05)
    fec = st.slider("FEC / year", 300, 900, 730, 10)
with col2:
    soc_preset = st.selectbox(
        "Rest SoC preset",
        ["balanced (rest 55%)", "high rest SoC / arbitrage (85%)", "low rest SoC / FCR (35%)", "custom"],
        help="Rest SoC is where the pack sits between trades — the average "
             "state of charge outside active cycling. High rest SoC accelerates "
             "calendar aging even without extra cycles.",
    )
    if soc_preset.startswith("custom"):
        mean_soc = st.slider("rest SoC", 0.20, 0.90, 0.55, 0.05)
    elif soc_preset.startswith("balanced"):
        mean_soc = 0.55
    elif soc_preset.startswith("high"):
        mean_soc = 0.85
    else:
        mean_soc = 0.35
    c_rate = st.slider("C-rate", 0.1, 1.0, 0.5, 0.05)
with col3:
    temp = st.slider("Mean temperature (°C)", 10, 40, 25, 1)
    st.caption(
        "Cell: `baseline_fleet` synthetic LFP — see cross-chemistry note above "
        "for why vendor-specific presets are not exposed here."
    )

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
below = soh_df[soh_df["p10"] <= eol]
years_to_eol = float(below["year"].iloc[0]) if len(below) else float("nan")

col_a, col_b = st.columns([2, 1])
with col_a:
    f = go.Figure()
    f.add_trace(go.Scatter(x=soh_df["year"], y=soh_df["p50"], name="median", line=dict(color="#0b5fff")))
    f.add_trace(
        go.Scatter(
            x=np.concatenate([soh_df["year"], soh_df["year"][::-1]]),
            y=np.concatenate([soh_df["p90"], soh_df["p10"][::-1]]),
            fill="toself",
            fillcolor="rgba(11,95,255,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="p10–p90",
        )
    )
    f.add_hline(y=eol, line_dash="dot", line_color="#888", annotation_text=f"EOL {int(eol*100)}%")
    f.update_layout(
        xaxis_title="years",
        yaxis_title="SoH",
        yaxis=dict(range=[0.5, 1.0]),
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
with col_b:
    st.metric("Years to EOL (pack)", f"{years_to_eol:.1f}" if not np.isnan(years_to_eol) else "> 20")
    st.metric("SoH at year 10 (pack)", f"{float(soh_df.iloc[min(len(soh_df)-1, 20)]['p10']):.2f}")
    if not (preset.temp_range_C[0] <= temp <= preset.temp_range_C[1]):
        st.warning(f"Temperature outside preset range {preset.temp_range_C}.")
    if c_rate > 2.0:
        st.warning("C-rate above calibration range (>2C).")

st.markdown(
    """
Things worth playing with: set FEC at 730 and move DoD between 60% and
95% to reproduce the asymmetric-DoD finding. Hold everything else
constant and walk mean SoC from 35% to 85% to watch pure calendar aging
pull the curve down without any extra cycles. Push temperature to 35 °C
to see the Modo 2025 ERCOT-vs-GB spread emerge from first principles.
"""
)

# ── Model boundaries ────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
## Where this model stops working

- **Chemistry.** LFP/graphite only in this release. `ChemistryFamily`
  keeps NMC/LTO/NCA slots for future calibration. NMC has a different
  calendar-aging shape (stronger SoC dependence, lower Ea) and needs a
  separate kernel.
- **Temperature.** 15–60 °C calibrated (Wang 2011 + Naumann 2018
  coverage). The 60 °C edge has a 34% residual in the Naumann back-test
  — flagged, not suppressed; cold end sparsely sampled, use with care
  below 15 °C.
- **C-rate.** 0C–2C. Below 0.5C cycle term extrapolates linearly; above
  2C the estimate is a conservative overestimate. Grid-scale LFP
  stationary operation rarely exits this window.
- **SoC representation.** Bucketed hours per year — transition
  dynamics, partial cycles, and calendar/cycle coupling inside a cycle
  are not modelled. For stationary LFP the bucketing approximation
  holds; it will need revisiting when NMC enters.
- **Knee point.** LLI → LLI+LAM transition (Han 2019) not modelled.
  Most LFP cells knee below 70% SoH — our 70% EOL sits on the safe
  side of that transition, so the omission is not load-bearing for
  stated-life numbers.
- **Pack effects.** Cell-to-cell spread is parametric (Severson 2019
  CoV 8%). Thermal gradients across modules and string-level imbalance
  are captured in the lever chart as a weakest-cell proxy — effective
  DoD and SoC dwell shifted by the observed spread — but not from a
  first-principles pack model. Field data anchors the span: pack fade
  runs ~10–20% faster than best-cell fade (Schimpe 2018; Reniers 2019).
  The operator lever is monitoring discipline and intervention cadence
  with OEM, not BMS spec (set at LTSA signing).
- **Augmentation.** Out of scope. Augmentation is a commercial decision
  about when to inject fresh capacity — not a degradation mechanism —
  and mixing the two gives a SoH curve with stair-steps that hide what
  the cells are actually doing. Gets its own note (timing, sizing,
  trigger-SoH, auction vs LTSA).
- **Field validation beyond calibration cells.** SNL Preger 2020
  (30 A123 18650 cells, cycling) and Stanford Lam 2024 (80 K2 Energy
  LFP18650 cells, calendar) are now wired in as out-of-sample tripwires —
  see the back-test expander. What remains missing is a public dataset
  on a 280 Ah prismatic LFP cell; none exists today, so prismatic-specific
  bias is inferred from small-format anchors.
"""
)

# ── Where this goes next ────────────────────────────────────
st.markdown("---")
st.markdown(
    """
## Where this goes next

- **Note 3b — Degradation-aware vs naive trading.** Plugs the
  state-dependent cost-of-cycle from the EUR/FEC ruler into a
  rolling-horizon optimiser. Targets the Humiston 2026 / Holtorf 2026
  gap — concretely, what fraction of revenue does a naive throughput
  optimiser leave on the table against a kernel-aware one, on 2025–26
  DE DA+ID spreads.
- **Note 4 — Warranty, LTSA, and the augmentation decision.** Pulls
  OEM warranty envelopes (CATL, Sungrow, Huawei on system level; BYD,
  EVE, CATL on cell) into the same EUR/MW frame. Target question:
  where does warranty-preserving operation give up NPV, and how
  expensive is the preservation.
- **NMC kernel.** Schmalstieg 2014 calibration, reusing the
  `ChemistryFamily` slot. Unlocks hybrid PV-BESS and EV-second-life
  cases.
- **Pack-level thermal.** First-principles pack model — module-to-module
  gradients, string topology, thermal coupling between weak and strong
  cells — replacing the current weakest-cell proxy. The model itself is
  tractable (thermal RC network coupled to the existing cell kernel);
  the binding constraint is calibration data. Public grid-scale
  per-string telemetry is effectively absent — RWTH Aachen ISEA's
  M5BAT demonstrator (module-level, periodic measurements) is the only
  realistic interim anchor. Stronger anchors wait for operational BESS
  telemetry 12–18 months post-COD (Strübbel / fleet pulls).
"""
)

# ── Methodology ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## Data Sources & Methodology")

with st.expander("Why this kernel — the model-choice problem"):
    st.markdown(
        """
Humiston, Cetin, de Queiroz (Energies 2026) ran the same BESS project
through two degradation formulations — simple throughput vs rainflow —
on 2024 ERCOT data and watched it flip from profitable to deeply
negative. Not the dispatch strategy. Not the chemistry. The **model the
analyst chose** decided whether the asset looked like a good deal. The
same sensitivity shows up at our parameter scale: swinging the cycle
pre-factor `k_cyc` by ±20% moves lifetime NPV by a magnitude comparable
to picking one chemistry over another.

A model asked to rank operator choices has to be tighter than the
choices it is ranking. Three things rule out the common shortcuts:

- **Flat throughput / €-per-cycle models.** Cannot see the rest-SoC
  lever at all (no calendar channel), and mis-price cycling in both
  directions over asset life — see the EUR/FEC ruler section above.
  This is the formulation that flipped profitability in Humiston 2026.
- **Rainflow-only.** Captures cycle depth, but without an explicit
  calendar channel it misses the Naumann SoC³ effect that drives the
  arbitrage-vs-FCR gap — and that gap is one of the headline findings
  here.
- **Single-parameter Arrhenius.** One activation energy for "aging" hides
  that cycle and calendar channels respond to temperature differently
  (Ea ≈ 0.30 eV vs 0.18 eV in this kernel). Collapsing them misprices
  climate.

The two-channel Wang 2011 + Naumann 2018 kernel used here keeps cycle
and calendar separable, gives SoC its own cubic dependence, and lets
each channel carry its own Arrhenius. It is calibrated end-to-end
against the same Sony/Murata LFP cell Naumann tested (3.2% median
residual, in-distribution) and cross-checked on four independent
sources: SimSES (TU Munich's open-source reference, 0.049 median
|ΔSoH| over 20 years), Sandia SNL Preger 2020 for the cycle channel
(30 A123 18650 cells across a T × DoD × C-rate grid), and Stanford
Lam 2024 for the calendar channel (80 K2 Energy LFP18650 cells across
24/45/60/85 °C × 50/100 % SoC, up to 8 years). Numbers and scope
caveats are in the back-test expander below.
"""
    )

with st.expander("Calibration & back-test — the numbers"):
    st.markdown(
        """
Four back-tests with different standards of evidence. Two
calibration-referential (Naumann, SimSES) that the kernel is fit
against; two out-of-sample tripwires on cells from different
manufacturers (SNL, Stanford) that flag regressions without being fit
targets.

| Back-test | Metric | Result | Target |
|---|---|---|---|
| Naumann 2018 (17 test points, in-distribution T ≤ 40 °C, 476 measurements) | median \\|error\\| | **3.2%** | <10% ✓ |
| Naumann 2018 (edge, T = 60 °C, 102 measurements) | median \\|error\\| | 34% | — (edge, flagged) |
| SimSES LFP parity (5 duty scenarios × 6 year points) | median \\|ΔSoH\\| | **0.049** | <0.10 ✓ |
| SimSES LFP parity | max \\|ΔSoH\\| | 0.19 (hot climate, 20yr) | — |
| SNL Preger 2020 cycling, Trina preset (30 anchors, A123 18650) | MAE | **12.3 pp** | — (out-of-sample tripwire) |
| Stanford Lam 2024 calendar, Trina preset (112 anchors, K2 LFP, T ≤ 45 °C) | MAE | **4.6 pp** | <5 pp ✓ |
| Stanford Lam 2024 calendar, Trina preset (60/85 °C anchors) | MAE | 17 pp | — (edge, flagged) |

**What the calibration-referential back-tests are.** Naumann 2018 is a
17-point static calendar test of the same Sony/Murata LFP cell the kernel
was calibrated against — tests the calendar channel and its SoC dependence
directly, end-to-end. SimSES is TU Munich's open-source simulator that
ships the same Naumann parameters; the parity test takes five stationary
duty scenarios (baseline, deep-DoD, high-SoC dwell, FCR-narrow, hot-climate)
and compares SoH trajectories year-by-year out to 20 years.

**What the out-of-sample back-tests are — and their bias disclosure.**

*SNL cycling (Preger 2020, batteryarchive.org).* 30 A123 18650 LFP cells
on a T × DoD × C-rate cycling grid. Different cell format from our
prismatic presets; bias is expected and material. Per-preset bias runs
−4.7 to −19.7 pp (model under-predicts retention, i.e. over-penalises
wear), worst at the 35 °C × 2C corner where the Trina preset shows
−25.6 pp residual. The 12.3 pp Trina MAE in the table above is this
bias, not a calibrated residual. Used as a regression tripwire: any
future recalibration of `Ea_cyc_eV` / `c_rate_exponent` must improve
this shape on SNL before shipping. **Not evidence that the kernel is
accurate on A123 18650 at 35 °C × 2C**; it evidently is not.

*Stanford calendar (Lam 2024 Joule, OSF ju325).* 80 K2 Energy LFP18650
cells stored up to 8 years on a 4T × 2SoC grid. Different manufacturer
from Naumann (K2 vs Sony/Murata), genuinely independent calendar anchor.
Within Naumann's own temperature envelope (T ≤ 45 °C), bias splits by
preset calibration quality: Trina and EVE (multi-anchor / datasheet
anchor) fit to bias ≤3.4 pp, MAE ≤4.6 pp. Single-anchor presets (BYD,
CATL, baseline_fleet) underpredict more (bias −6.8 to −10.1 pp) — a
calibration-quality artefact already flagged in preset notes, not a
model-form failure. Beyond 50 °C the Naumann Arrhenius power-law breaks
down (Li-inventory exhaustion saturates) with bias blowing to −16 to
−30 pp — flagged as edge, retained for drift detection only.

**What they still do not cover.** A public cycling dataset on a 280 Ah
prismatic LFP cell does not exist today — all public LFP cycling data
is on small-format 18650. SNL validates the T × C-rate *shape*; absolute
bias on prismatic chemistry remains inferred. Severson 2019 and Attia
2020 are not used here — both are fast-charging (3.6C–8C) studies on
A123 cells, outside our stationary-BESS regime and redundant with SNL's
better grid coverage.

Calibration scripts: `notes/degradation-drivers/calibration/`.
Out-of-sample validation modules: `lib/validation/snl_lfp.py`,
`lib/validation/stanford_calendar.py`.
"""
    )

with st.expander("Kernel equations"):
    st.markdown(
        r"""
Two-channel Qloss, cell-to-cell noise added parametrically:

$$
Q_{\text{loss,cyc}} = B_{\text{cyc}} \cdot \exp\!\left(-\frac{E_{a,\text{cyc}}}{R T}\right) \cdot \left[\text{FEC} \cdot \text{DoD} \cdot C_{\text{nom}}\right]^{z_{\text{cyc}}} \cdot f_{\text{DoD}}(\text{DoD})
$$

$$
Q_{\text{loss,cal}} = \sum_{b \in \text{SoC buckets}} \frac{h_b}{8760} \cdot k_{\text{cal}}(\text{SoC}_b) \cdot t^{\beta_{\text{cal}}} \cdot \exp\!\left(-\frac{E_{a,\text{cal}}}{R T}\right)
$$

$$
\varepsilon_{\text{cell}} \sim \mathcal{N}(0,\, k_{\text{cyc}} \cdot \text{CoV})
$$

$$
\text{SoH}(t) = 1 - Q_{\text{loss,cyc}} - Q_{\text{loss,cal}} - \varepsilon_{\text{cell}}
$$

Cycle term: Wang, Liu, Kloess 2011 LFP power law, Ea from LFP/graphite
literature. Calendar term: Naumann et al. 2018 LFP calendar model with
SoC-cubic dependence, $k_{\text{cal}}(\text{SoC}) = C \cdot (\text{SoC}-0.5)^3 + D$.
SoC enters through bucketed hours/year — the integration that captures
"high-SoC dwell hurts" without needing a full timeseries. CoV 0.08
from Severson 2019 cell-to-cell spread measurements.
"""
    )

with st.expander("Preset provenance"):
    p = PRESETS["baseline_fleet"]
    st.markdown(
        f"""
All charts and the interactive in this note run on a single synthetic
preset, `baseline_fleet` — tuned to reproduce the Note 1 legacy
~1.8%/yr fleet fade within ±0.2pp through year 20. Calibration anchor:
**{int(p.test_anchor.cycles)} FEC @ DoD {p.test_anchor.dod:.2f}, C-rate
{p.test_anchor.c_rate:.2f}, {p.test_anchor.temp_C:.0f} °C → retention
{p.test_anchor.retention:.2f}**.

Vendor-specific cell presets (EVE, CATL, BYD, Trina) exist in the code
but are not exposed here — see the "A note on cross-chemistry
comparison" section above for the methodology reasons. A credible
head-to-head ranking needs a matched-protocol bench; published
datasheets do not provide one.
"""
    )

with st.expander("Literature table"):
    st.markdown(
        """
| Paper | Role |
|---|---|
| Wang, Liu, Kloess 2011 | Cycle-life power law (LFP) — cycle channel |
| Naumann, Schimpe 2018 | Calendar + SoC dependence (LFP) — calendar channel |
| Naumann et al. 2020 | LFP cycle aging refinement (used in SimSES parity) |
| Severson 2019 | Cell-to-cell CoV source (not used as back-test — see calibration expander) |
| SimSES (TU Munich) | Open-source parity anchor (Sony US26650 CLFP row) |
| Xu 2018 | DoD super-linear exponent |
| Schimpe 2018 | ~12.8 €/MWh LFP flat-cost benchmark |
| Kim 2022 | Experimental: DoD > C-rate for LFP |
| Modo 2025 | Fleet-level field: ERCOT 7% vs GB 4% per 365 cycles |
| Humiston, Cetin, de Queiroz 2026 | Model choice dominates BESS valuation |
| Holtorf, Shin 2026 | Physics-based cost-of-cycle in rolling-horizon dispatch |
| Esquivel, Harris 2026 (SAGE) | Synthetic aging sim — future benchmarking |
| Han 2019 | LFP knee-point (not modelled; boundary note) |
| Schmalstieg 2014 | NMC calendar kernel (reserved for future) |
"""
    )

with st.expander("Reproducibility"):
    st.markdown(
        """
Everything in this note runs from a clean checkout:

```bash
python notes/degradation-drivers/precompute.py
streamlit run notes/degradation-drivers/app.py
python notes/degradation-drivers/generate_charts.py

# Calibration (re-runs the numbers in the back-test table)
python notes/degradation-drivers/calibration/fit_naumann_calendar.py
python notes/degradation-drivers/calibration/parity_simses.py
```

Model code: `lib/models/degradation.py` (simple closed-form),
`lib/models/degradation_detailed.py` (Wang + Naumann, Monte Carlo).
Parity between the two enforced by
`tests/test_simple_detailed_parity.py` — the simple model is never
allowed to drift from the detailed baseline by more than the tolerance
declared there.
"""
    )

# ── Closing ─────────────────────────────────────────────────
st.markdown("---")
render_closing(
    "Third note in a series on BESS merchant economics. Notes 3b and 4 extend "
    "this model into dispatch and warranty."
)
st.markdown(
    '<div style="margin-top: 0.5rem; font-size: 0.95rem; color: #666;">'
    "<b>Next:</b> Note 3b — Degradation-aware vs naive trading."
    "</div>",
    unsafe_allow_html=True,
)
render_footer()
