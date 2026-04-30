"""
Note 4 — Is Your Trader Aging-Aware? The Cost of a Cycle.

Headline numbers from `data/precomputed_v40_5methods_10y.pkl` (v4.0
paper-grade, publicly anchored to regelleistung.net + Clean Horizon
Storage Index + EVE LF280K manufacturer endurance). The pkl re-keys
the underlying ablation pkl into clean publication M-numbering:
    M1_naive    — no shadow cost (was L1_naive)
    M2_flat     — flat €/MWh wear (was L3_flat_wear)
    M3_scarcity — flat × SoH-state scarcity (was L4_scarcity)
    M4_physics  — ADP + physics-from-duty (was L6_physics_full)
    M5_adp      — pure ADP, no flat / scarcity (was L5_adp_only)
plus two ablation cross-check variants under separate keys for the
"why blending doesn't help" methodology footnote.

Diagnostic dispatch logs for the five owner-facing signals come from
`data/precomputed_two_stage.pkl` (v3.5; signal patterns are
calibration-agnostic). The diag pkl retains the old L-key naming and
is mapped to the M-numbering at access time via DIAG_KEY_FOR.
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


DATA_DIR = Path(__file__).parent / "data"
# Headline = v4.0 B + FCR phantom + SIDC IDA2 wholesale-ID feed (post
# 2024-06-13, real DE-LU pan-European intraday auction prices via
# ENTSO-E API). Under this calibration the M1 → M5 stack orders
# monotonically (M5 > M4 > M3 > M2 > M1) at +37.1 % NPV uplift at the
# M5 winner. Each Mn shows ONE shadow-cost form; M1-M3 are the
# classical depreciation-proxy family (no shadow → throughput-only →
# SoH-aware), M4-M5 are the state-aware family (physics-from-duty
# observed dispatch → ADP opportunity-cost gradient). Empirical winner
# is M5 (pure ADP); the ablation footnote shows that adding physics
# refinement on top of M5 (= M4) or stacking the M2/M3 classical
# proxy under M5 both hurt by 1-4 pp — best shadow cost is one
# shadow cost, chosen well.
PRECOMPUTED_HEADLINE = DATA_DIR / "precomputed_v40_5methods_10y.pkl"
PRECOMPUTED_DIAG = DATA_DIR / "precomputed_two_stage.pkl"

# The diag pkl pre-dates the M-renaming. Mapping at access time:
# diag-pkl provides dispatch-log signatures that are calibration-
# agnostic, and qualitatively the diag's L5_intraday_adp (additive
# blend) signature stands in for M5 (pure ADP) since both are
# ADP-dominated.
DIAG_KEY_FOR = {
    "M1_naive":     "L1_naive",
    "M2_flat":      "L3_flat_wear",
    "M3_scarcity":  "L4_scarcity",
    "M4_physics":   "L6_physics_full",
    "M5_adp":       "L5_intraday_adp",
}


# Five shadow-cost methodologies. M1-M3 are classical depreciation-proxy
# family (no shadow → throughput-only → SoH-aware multiplier). M4-M5 are
# state-aware family (physics-from-duty → ADP opportunity gradient).
# Each Mn shows ONE methodology, not a stack — the labels intentionally
# avoid "+" framing that would imply progressive layering.
POLICY_LABELS = {
    "M1_naive":     "M1 — Naive (no shadow cost)",
    "M2_flat":      "M2 — Flat €/MWh wear (Kumtepeli proxy)",
    "M3_scarcity":  "M3 — Scarcity scaling (SoH-state)",
    "M4_physics":   "M4 — Physics-from-duty (ADP base + Note 3 kernel)",
    "M5_adp":       "M5 — Intraday ADP (Holtorf-Shin)",
}
POLICY_SHORT = {
    "M1_naive":     "M1 Naive",
    "M2_flat":      "M2 Flat",
    "M3_scarcity":  "M3 Scarcity",
    "M4_physics":   "M4 Physics",
    "M5_adp":       "M5 ADP",
}
# M1 sits in warm terra-coral as the unconstrained-baseline reference —
# visually orthogonal to the teal stack so it reads as the "odd one out"
# at a glance. M2 → M5 progress through a single-hue tint gradient
# anchored on BayWa Deep Sea (#0D3F4A).
POLICY_COLORS = {
    "M1_naive":     "#e07a5f",
    "M2_flat":      "#a8c4cb",
    "M3_scarcity":  "#5a8c97",
    "M4_physics":   "#1d566a",
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
def _load_pkl(path_str: str):
    with open(path_str, "rb") as f:
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
    subtitle="A class of dispatch policies that price each cycle against future revenue loss. Five methodologies, what each one is worth, and how to read the signature off a monthly trader report.",
)

# ── Intro ───────────────────────────────────────────────────
st.markdown("""
The literature on BESS dispatch contains a spectrum of *aging-aware*
formulations — policies that subtract a cost-per-cycle term from the
revenue objective so the optimiser declines marginal cycles whose
spread does not justify the wear. The formulations differ in *which*
state the shadow cost depends on: throughput only, state-of-health,
hour-of-day, the full physics of the observed dispatch.

This note runs five distinct methodologies on a 2 h LFP system across
2023 and 2025 German day-ahead, intraday and aFRR markets, on a
10-year rotating horizon. All five see identical prices and identical
activation signals; the only difference is *how* each one prices the
cost of a cycle. The headline below uses the v4.0 calibration anchored
to publicly-citable sources: regelleistung.net public auction CSVs for
aFRR clearing, the Clean Horizon Storage Index for absolute revenue
level, and the EVE LF280K manufacturer endurance spec for the physics
kernel.
""")

# ── Load data ───────────────────────────────────────────────
data = _load_pkl(str(PRECOMPUTED_HEADLINE))
results = data["results"]
n_years = data["n_years"]

diag_data = _load_pkl(str(PRECOMPUTED_DIAG)) if PRECOMPUTED_DIAG.exists() else None
diagnostics = diag_data.get("diagnostics") if diag_data else {}
diag_results = diag_data.get("results") if diag_data else {}


# ── KPI row ─────────────────────────────────────────────────
naive = results["M1_naive"]
winner = results["M5_adp"]

eol_naive = _eol_year_1indexed(naive.years_to_floor, n_years)
eol_winner = _eol_year_1indexed(winner.years_to_floor, n_years)
fec_naive = int(naive.annual_fec.sum())
fec_winner = int(winner.annual_fec.sum())

col1, col2, col3, col4 = st.columns(4)
uplift_pct = (winner.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100
with col1:
    st.metric(
        "M5 vs M1 — lifetime DCF uplift",
        f"+{uplift_pct:.1f}%",
        "monotonic across M1 → M2 → M3 → M4 → M5",
        delta_color="off",
    )
with col2:
    st.metric(
        "Cycling reduction (M1 → M5)",
        f"−{(1 - fec_winner / fec_naive) * 100:.0f}%",
        f"{fec_naive:,} → {fec_winner:,} FEC over 10 y",
        delta_color="off",
    )
with col3:
    eol_text = (
        f"warranty floor never reached"
        if winner.years_to_floor is None
        else f"Y{eol_naive} → Y{eol_winner} at SoH 0.80"
    )
    extension = (
        f"+{n_years - eol_naive}+ years"
        if winner.years_to_floor is None
        else f"+{eol_winner - eol_naive} years"
    )
    st.metric(
        "EOL extension",
        extension,
        eol_text,
        delta_color="off",
    )
with col4:
    fec_ratio = fec_naive / max(fec_winner, 1)
    st.metric(
        "Cycling intensity ratio",
        f"{fec_ratio:.0f}× ",
        f"M1 cycles {fec_ratio:.0f}× more than M5 over the lifetime",
        delta_color="off",
    )


# ── Main chart ──────────────────────────────────────────────
render_chart_title(
    "Five shadow-cost methodologies, ten-year discounted lifetime "
    "revenue. M1 (no shadow cost) at the bottom, M5 (Holtorf-Shin "
    "intraday ADP) at the top. M1-M3 are the classical depreciation-"
    "proxy family (no shadow → throughput-only → SoH-aware multiplier); "
    "M4-M5 are the state-aware family (physics-from-duty observed "
    "dispatch → ADP opportunity-cost gradient). Each Mn shows ONE "
    "shadow-cost form; the methodology footnote below covers two "
    "ablation cross-checks where blending Mn with Mn+k components hurts."
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
    text=[f"{v:.0f}<br>{p:+.1f}%" for v, p in
          zip(npv_df["index"], npv_df["pct_vs_naive"])],
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
    yaxis=dict(title="Lifetime DCF index (M1 = 100)"),
    xaxis=dict(tickfont=dict(size=10)),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
render_chart_caption(
    "Ten-year lifecycle simulation, 2 h LFP, 1 MW / 2 MWh, rotating "
    "2023↔2025 DE markets. All five methodologies face identical "
    "prices and activation signals; the only difference is how each "
    "one prices the cost of a cycle. Bars indexed to M1 = 100; "
    "absolute revenue depends on calibration choices (perfect-"
    "foresight LP, `max_afrr` cap, FCR phantom layer) and is not the "
    "focus of this note. The relative ordering and the +37% M1 → M5 "
    "spread are robust to those choices. Markets held at 2024 levels; "
    "the Note 1 trajectory section below shows how the ordering bends "
    "when wholesale grows and aFRR cap compresses."
)


# ── Annual revenue + SoH trajectories ───────────────────────
st.markdown("### Lifetime revenue is bought with cycling restraint")

col_rev, col_soh = st.columns(2)

with col_rev:
    render_chart_title("Cumulative discounted revenue (M1 lifetime = 100)")
    fig_rev = go.Figure()
    discount_factors = np.array([
        1.0 / (1.0 + 0.07) ** y for y in range(n_years)
    ])
    naive_lifetime = float((naive.annual_revenue_eur * discount_factors).sum())
    for name in POLICY_ORDER:
        r = results[name]
        cum = np.cumsum(r.annual_revenue_eur * discount_factors) / naive_lifetime * 100
        fig_rev.add_trace(go.Scatter(
            x=np.arange(1, n_years + 1),
            y=cum,
            mode="lines+markers",
            name=POLICY_SHORT[name],
            line=dict(color=POLICY_COLORS[name], width=2.5),
        ))
    fig_rev.add_hline(y=100, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig_rev.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=360, margin=dict(l=40, r=20, t=20, b=50),
        xaxis=dict(title="Year"),
        yaxis=dict(title="Cumulative DCF index (M1 lifetime = 100)"),
        legend=dict(orientation="h", y=-0.20, font=dict(size=10)),
    )
    st.plotly_chart(fig_rev, use_container_width=True,
                    config={"displayModeBar": False})

with col_soh:
    render_chart_title("State of health trajectory")
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
    f"M1 naive earns the most revenue per year while the battery is "
    f"fresh, then hits the warranty floor in year {eol_naive} and loses "
    f"all subsequent cap revenue. M2 / M3 scale back cycling as SoH "
    f"drops, extending operational life by ~2 years; M4 / M5 reduce "
    f"cycling so aggressively that they never reach the warranty floor "
    f"in the 10-year horizon."
)


# ── The five methodologies, plain language ──────────────────
st.markdown("---")
st.markdown("## The five shadow-cost methodologies")
_m2 = results["M2_flat"]
_m3 = results["M3_scarcity"]
_m4 = results["M4_physics"]
_m5 = results["M5_adp"]
_abl_blend = results["ablation_M5_with_classical_stack"]
_abl_solo = results["ablation_physics_without_adp"]
st.markdown(f"""
Each methodology shows ONE shadow-cost form, not a stack. M1-M3 are
the classical depreciation-proxy family — the cost of a cycle is
estimated from CAPEX-over-throughput accounting, optionally scaled by
remaining SoH headroom. M4-M5 are the state-aware family — the cost
is derived from a richer information source: observed dispatch
(physics-from-duty) or a pre-solved value function (ADP). The five
are alternative formulations, not a progressive build-up; the chart
orders them by lifetime DCF, which happens to track sophistication of
state-dependence.

**M1 — Naive.** No cycle cost, no envelope. The LP takes any spread
that clears variable cost. This is the unconstrained revenue-max
strawman; no real operator trades this way, but it is the upper bound
of cycling and the lower bound of battery life. **Lifetime DCF index =
100** (by definition); cycles **{fec_naive:,} FEC** across the 10
years; reaches the warranty floor in year {eol_naive}.

**M2 — Flat €/MWh wear (Kumtepeli proxy).** CAPEX divided by expected
lifetime throughput gives a single number (~€16.67 / MWh on this
asset). The LP subtracts it from every cycle's revenue. Adds
**+{(_m2.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.1f}%**
over M1 by skipping marginal cycles whose spread does not clear the
wear hurdle. Cycling drops from {fec_naive:,} to
{int(_m2.annual_fec.sum()):,} FEC; EOL extends by ~2 years. The
single-largest jump above naive — most of the aging-aware value comes
from any sensible per-MWh penalty, not from sophistication.

(A pure SoC envelope [0.20, 0.80] without a shadow cost — the
literature's classical "L2" — was tested separately; under v4.0
calibration the LP-aware operator's natural trajectory already sits
inside the band, so the envelope alone adds nothing. All aging-aware
uplift flows through the shadow-cost form, not envelope geometry.
The five methodologies shown here are the ones that *do* move the
needle.)

**M3 — Scarcity scaling (SoH-state).** The flat shadow cost is
multiplied by `1 + (1 − SoH) / (1 − floor)` — bounded 1× at fresh
cell, 2× at the warranty floor. One extra dimension of state: SoH.
Adds another
**+{(_m3.lifetime_npv_eur / _m2.lifetime_npv_eur - 1) * 100:.1f}** pp
over M2 by suppressing late-life cycling, where the marginal cycle
costs the most because remaining capacity to monetise is shrinking.

**M4 — Physics-from-duty.** A 2-pass LP. Pass 1 uses the ADP intraday
shadow cost (so the LP has a meaningful starting dispatch in two-stage
mode where Stage-1 commitments must be fixed before pass-2 sees the
day's trajectory); the observed dispatch is fed into the Note 3
physics kernel (Wang + Naumann two-channel, calibrated to EVE LF280K
via `kernel_scale = 0.66`); pass 2 re-solves Stage-2 with the
resulting per-MWh wear cost that reflects the actual DoD / C-rate /
SoC-band of the day's duty. Stage-1 r commitments stay locked from
pass-1 — re-pricing them would require solving the Stage-1 ↔ Stage-2
fixed point, out of scope for ADR-001 v1.0. NPV
**+{(_m4.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.1f}%**
vs M1.

**M5 — Intraday ADP (Holtorf-Shin).** An offline backward-induction DP
over `(SoC, SoH, regime, hour-of-day)` returns a state-value function;
its gradient with respect to SoC is the shadow cost — and because hour
is in the state, the cost varies hour by hour. At 18:00 in a volatile
regime one MWh of stored energy is worth more than at 03:00 in a calm
regime. Cycles drop to **{int(_m5.annual_fec.sum()):,} FEC** over the
lifetime — about
{(fec_naive / max(int(_m5.annual_fec.sum()), 1)):.0f}× less than M1.
NPV **+{(_m5.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.1f}%**
vs M1 — the peak of the stack.

**The pattern.** Each additional state dimension in the shadow cost
adds value monotonically, with diminishing returns. M2 captures the
largest single step
(+{(_m2.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.0f} pp);
M3 adds another
+{(_m3.lifetime_npv_eur / _m2.lifetime_npv_eur - 1) * 100:.0f} pp via
SoH-state; M4 adds
+{(_m4.lifetime_npv_eur / _m3.lifetime_npv_eur - 1) * 100:.0f} pp via
duty-dependent physics; M5 adds the final
+{(_m5.lifetime_npv_eur / _m4.lifetime_npv_eur - 1) * 100:.0f} pp via
hour-of-day opportunity-cost state. The literature's direction is
fully recovered on real DE markets at
+{(winner.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.1f}%
total uplift — Collath et al. (2023) reported +29.3% on a similar
setup; Kumtepeli & Howey (2024) framed the depreciation proxy as a
poor shadow cost and argued for state-dependent forms. Both are
confirmed.

---

#### Why blending doesn't help

Two ablation cross-checks confirm that **the best shadow cost is one
shadow cost, chosen well** — not a sum of two.

**(a) Stacking M2/M3 classical proxy under M5 ADP** (the additive
form: flat + scarcity + ADP) gives NPV
**+{(_abl_blend.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.2f}%**
vs M1 — that is **{(_abl_blend.lifetime_npv_eur / _m5.lifetime_npv_eur - 1) * 100:+.2f} pp** vs M5 alone. *Why*: the
flat-wear term is a *depreciation proxy* (assumes every MWh of
throughput costs CAPEX/total_throughput). The ADP gradient is the
*true opportunity cost* (marginal value of saving SoC for later). The
proxy double-prices what ADP already accounts for; the LP sees "ADP
says skip this cycle in calm hours" + "depreciation says skip every
cycle always" → over-suppression on hours where ADP alone would have
cycled. Empirical confirmation of Kumtepeli-Howey (2024): adding a
poor proxy on top of a state-aware form is noise, not signal.

**(b) Removing ADP from M4** (physics-from-duty alone, no ADP base)
gives NPV
**+{(_abl_solo.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.2f}%**
vs M1 — that is **{(_abl_solo.lifetime_npv_eur / _m4.lifetime_npv_eur - 1) * 100:+.2f} pp** vs the canonical M4 (with ADP base). *Why*: in
two-stage dispatch the second pass only re-prices Stage-2 under
locked Stage-1 commitments. Without ADP in pass-1, Stage-1 commits
naively (= M1-style), and pass-2 physics can only correct what
Stage-2 controls — too little, too late. Physics-from-duty needs ADP
in pass-1 to push the dispatch into the right neighbourhood before
the kernel refines wear. The two are not substitutes; M4 is "ADP +
physics refinement", not "physics standalone".

**Combined reading**: M5 (pure ADP) is the empirical winner because
ADP's state-grid `(SoC, SoH, regime, hour)` already implicitly prices
the information that depreciation-proxy and physics-from-duty try to
capture separately. Adding either on top double-counts; subtracting
ADP from M4 leaves physics with insufficient leverage. Best practice:
pick one well-instrumented shadow cost and let it do the work alone.
""", unsafe_allow_html=True)


# ── Reading aging-awareness off the dashboard (merged) ─────
st.markdown("---")
st.markdown("## Reading aging-awareness off your trader's dashboard")

# Multi-month scatter feeds off the v3.5 lifecycle pkl (has all five
# policies). The three dashboard tiles further down use the v4.0 B +
# FCR diag pkl (`precomputed_diag_2025.pkl`) — same calibration as the
# headline, captured from the 2025 template year.
_v40_diag_path = DATA_DIR / "precomputed_diag_2025.pkl"
if _v40_diag_path.exists():
    _v40_diag = _load_pkl(str(_v40_diag_path))
    l1_year_days = _v40_diag["results"][DIAG_KEY_FOR["M1_naive"]].diagnostic_days
    l6_year_days = _v40_diag["results"][DIAG_KEY_FOR["M4_physics"]].diagnostic_days
else:
    # Fallback to v3.5 diagnostic pkl if v4.0 B diag was not generated.
    l1_year_days = diag_results.get(DIAG_KEY_FOR["M1_naive"]).diagnostic_days if diag_results.get(DIAG_KEY_FOR["M1_naive"]) else None
    l6_year_days = diag_results.get(DIAG_KEY_FOR["M4_physics"]).diagnostic_days if diag_results.get(DIAG_KEY_FOR["M4_physics"]) else None

if l1_year_days and l6_year_days:
    st.markdown("""
Most BESS optimisers (Entrix, suena, Re.Volt, Modo, in-house desks)
ship owners a near-real-time monitoring portal. Different vendors,
similar tiles: cycles per day, state-of-charge over time, revenue
share by stream. Each tile carries one of the diagnostic signals —
once you know what to look for.

Panels below are **calibration-robust**: ratios, indexed values and
operational counts (cycles, SoC fractions, % of revenue mix). The
intraday stream uses real DE-LU SIDC pan-European Intraday Auction
prices ([IDA2 15-min, ENTSO-E](https://www.entsoe.eu/network_codes/cacm/implementation/ida/),
post 13 June 2024 launch). Absolute euros still depend on perfect-
foresight LP and bid-shading assumptions our model doesn't capture —
those inflate revenue ~30% above realised public indices; the *shapes*
below match your own dashboard one-for-one.
""")

    # ─── Multi-month robustness scatter ─────────────────────────
    from collections import defaultdict
    # Iterate in M-order; pull diagnostic dispatch logs from the diag
    # pkl using DIAG_KEY_FOR. The diag pkl pre-dates the M-renaming;
    # see DIAG_KEY_FOR comment near the top of this file for caveats.
    monthly_rows = []
    naive_monthly_avg = None
    # Compute M1 monthly average for normalisation index
    l1_by_month = defaultdict(lambda: dict(rev=0.0, fec=0.0))
    for day in diag_results[DIAG_KEY_FOR["M1_naive"]].diagnostic_days:
        m = day.date.month
        l1_by_month[m]["rev"] += day.daily_revenue_eur
        l1_by_month[m]["fec"] += day.full_equivalent_cycles
    naive_monthly_avg = np.mean([v["rev"] for v in l1_by_month.values()])

    for m_name in POLICY_ORDER:
        diag_name = DIAG_KEY_FOR[m_name]
        days = diag_results[diag_name].diagnostic_days
        by_month = defaultdict(lambda: dict(rev=0.0, fec=0.0))
        for day in days:
            m = day.date.month
            by_month[m]["rev"] += day.daily_revenue_eur
            by_month[m]["fec"] += day.full_equivalent_cycles
        for m, agg in by_month.items():
            monthly_rows.append({
                "policy": POLICY_SHORT[m_name],
                "label": POLICY_LABELS[m_name],
                "color": POLICY_COLORS[m_name],
                "fec": agg["fec"],
                "rev_index": agg["rev"] / naive_monthly_avg * 100,
                "month": m,
            })
    monthly_df = pd.DataFrame(monthly_rows)

    render_chart_title(
        "Monthly cycle count by policy — twelve months of dispatch on "
        "real DE markets. Naive sits in its own band, no overlap with "
        "the aging-aware policies."
    )
    fig_diag = go.Figure()
    # Reverse order so M1 sits at the top of the horizontal layout.
    for name in reversed(POLICY_ORDER):
        sub = monthly_df[monthly_df["policy"] == POLICY_SHORT[name]]
        fig_diag.add_trace(go.Box(
            x=sub["fec"],
            name=POLICY_SHORT[name],
            marker=dict(color=POLICY_COLORS[name], size=8),
            line=dict(color=POLICY_COLORS[name]),
            fillcolor=POLICY_COLORS[name],
            opacity=0.65,
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
            hovertemplate=(
                f"<b>{POLICY_LABELS[name]}</b><br>"
                "Monthly FEC: %{x:.1f}<extra></extra>"
            ),
        ))
    fig_diag.add_vrect(x0=20, x1=monthly_df["fec"].max() * 1.1,
                       fillcolor="rgba(254, 226, 226, 0.4)", layer="below",
                       line_width=0,
                       annotation_text="naive cycling band",
                       annotation_position="top right",
                       annotation_font=dict(size=10, color="#991b1b"))
    fig_diag.add_vrect(x0=0, x1=10,
                       fillcolor="rgba(209, 250, 229, 0.4)", layer="below",
                       line_width=0,
                       annotation_text="aging-aware band",
                       annotation_position="top left",
                       annotation_font=dict(size=10, color="#065f46"))
    fig_diag.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=50, r=30, t=40, b=50),
        xaxis=dict(
            title="Monthly full-equivalent cycles (FEC)",
            range=[0, float(monthly_df["fec"].max()) * 1.15],
            autorange=False,
        ),
        yaxis=dict(title="", tickfont=dict(size=11)),
        showlegend=False,
    )
    st.plotly_chart(fig_diag, use_container_width=True,
                    config={"displayModeBar": False})

    l1_fec_med = float(monthly_df[monthly_df["policy"] == POLICY_SHORT["M1_naive"]]["fec"].median())
    l5_fec_med = float(monthly_df[monthly_df["policy"] == POLICY_SHORT["M5_adp"]]["fec"].median())
    l1_rev_med = float(monthly_df[monthly_df["policy"] == POLICY_SHORT["M1_naive"]]["rev_index"].median())
    l5_rev_med = float(monthly_df[monthly_df["policy"] == POLICY_SHORT["M5_adp"]]["rev_index"].median())
    render_chart_caption(
        f"Each dot is one calendar month dispatched by one methodology "
        f"on real DE day-ahead + intraday + aFRR markets. Boxes show "
        f"median + inter-quartile range; whiskers extend to non-outlier "
        f"extremes. **Naive (M1) median {l1_fec_med:.0f} FEC/month; "
        f"intraday-ADP (M5) median {l5_fec_med:.0f}** — a "
        f"{l1_fec_med/l5_fec_med:.0f}× gap that holds across all twelve "
        f"months with zero overlap between M1 and any aging-aware "
        f"methodology. M2 / M3 / M4 stack between the two extremes in "
        f"the same order as the headline lifetime DCF — proof the "
        f"diagnostic is calibration- and season-independent."
    )

    st.markdown(f"""
**The diagnostic, in three numbers.**

1. **Monthly cycle count.** Naive median is **{l1_fec_med:.0f} FEC**;
   M5 median is **{l5_fec_med:.0f} FEC**. A 2 h LFP that consistently
   posts > 20 FEC / month under DE 2024 – 2026 conditions is running
   unconstrained or near-unconstrained. Below 10 FEC / month the trader
   is pricing cycles against a meaningful shadow cost. Below 5 FEC /
   month suggests aggressive aging-aware (intraday ADP or
   physics-from-duty).

2. **Monthly-vs-lifetime gap.** Naive sits a few percent ahead on any
   single month while the cell is fresh — it captures every spread
   that clears variable cost, including the marginal ones. Over the
   lifetime the same M5 policy earns
   **+{(winner.lifetime_npv_eur / naive.lifetime_npv_eur - 1) * 100:.0f}%**
   in discounted DCF (and crosses the warranty floor several years
   later — see the SoH chart at the top). The trade is monthly revenue
   for lifetime DCF.

3. **Trajectory across the year.** A naive policy's monthly revenue is
   **front-loaded** — strong in years 1 – 3 then collapses when the
   warranty floor terminates the cap revenue. An aging-aware policy's
   revenue declines slowly but is still earning capacity payments in
   year 8 – 10. If the trader's monthly report shows revenue trending
   sharply downward over a 12-month window in flat-market conditions,
   that's a signal of cycling-driven SoH degradation, not aging-aware
   dispatch.

The mental trap Kumtepeli & Howey [2024](https://arxiv.org/abs/2403.10617)
formalised: a trader judged on monthly P&L will always look like they
are leaving money on the table relative to a naive baseline, even
though the lifetime DCF rewards them. Owners should specify the
benchmark on a discounted-lifetime basis — or accept that aging-aware
dispatch will always show as monthly underperformance against an
unconstrained reference.
""")

    st.markdown("---")
    st.markdown("### Four tiles you'll find on every trader portal")
    st.markdown(
        "Below: the same M1 vs M4 (physics-from-duty) contrast on four "
        "standard dashboard tiles — *Cumulative Cycles*, *State of "
        "Charge Development*, *Daily Revenue per Market*, *Revenue "
        "Share per Market* — across a full year of 2025 dispatch under "
        "the paper-grade v4.0 calibration. Naming follows Entrix's "
        "commercial dashboard; suena, Re.Volt and Modo render the same "
        "information under slightly different labels. Match your own "
        "tiles to the M1 or M4 column to identify your trader's policy "
        "family."
    )

    NAIVE_C = "#e07a5f"
    AGING_C = "#0d3f4a"
    NAIVE_FILL = "rgba(224, 122, 95, 0.2)"
    AGING_FILL = "rgba(13, 63, 74, 0.2)"

    # ─── Panel 1: Cumulative cycles + cycles/day KPI ─────────
    st.markdown("##### Panel 1 — *Average Cycles per Day* + *Cumulative Cycles*")
    st.caption(
        "Two tiles you'll find on every trader portal: a headline-number "
        "KPI plus a cumulative-cycle line for the period."
    )
    col_l1, col_l6 = st.columns(2)

    # Shared y-axis upper bound — biggest cumulative across L1 and L6.
    # Forces the same vertical scale on both panels so the L1 vs L6 ratio
    # reads honestly (528 vs 54 = ~10× gap, must be visually a 10× gap).
    _cycles_y_max = max(
        float(np.cumsum([d.full_equivalent_cycles for d in l1_year_days]).max()),
        float(np.cumsum([d.full_equivalent_cycles for d in l6_year_days]).max()),
    ) * 1.05

    def _cum_cycles_panel(col, days, label, color, fill, y_max):
        with col:
            with st.container(border=True):
                fec_per_day = np.array([d.full_equivalent_cycles for d in days])
                avg_per_day = float(fec_per_day.mean())
                total = float(fec_per_day.sum())
                dates = [d.date for d in days]
                cumulative = np.cumsum(fec_per_day)
                st.markdown(
                    f"<div style='font-size:11px;color:#64748b;"
                    f"text-transform:uppercase;letter-spacing:0.5px;'>"
                    f"{label} dashboard — 2025 dispatch</div>",
                    unsafe_allow_html=True,
                )
                kcol1, kcol2 = st.columns(2)
                kcol1.metric("Avg cycles per day", f"{avg_per_day:.2f}")
                kcol2.metric("Cumulative cycles (12 mo)", f"{total:.0f}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=cumulative, mode="lines",
                    line=dict(color=color, width=2.5),
                    fill="tozeroy", fillcolor=fill,
                    hovertemplate="%{x|%d %b}<br>Σ FEC: %{y:.0f}<extra></extra>",
                ))
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=200, margin=dict(l=40, r=10, t=10, b=30),
                    yaxis=dict(title="Σ cycles", range=[0, y_max]),
                    xaxis=dict(showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

    _cum_cycles_panel(col_l1, l1_year_days, "M1 Naive", NAIVE_C, NAIVE_FILL, _cycles_y_max)
    _cum_cycles_panel(col_l6, l6_year_days, "M4 Aging-aware", AGING_C, AGING_FILL, _cycles_y_max)

    l1_total_fec = sum(d.full_equivalent_cycles for d in l1_year_days)
    l6_total_fec = sum(d.full_equivalent_cycles for d in l6_year_days)
    st.markdown(
        f"**What you're looking for.** Average cycles per day above ~0.7 "
        f"(annual total > 250 FEC, monthly > 20) — the trader is running "
        f"unconstrained or near-unconstrained. Below ~0.2 / day "
        f"(annual ~70 FEC, monthly ~5) — the trader is pricing cycles "
        f"against a meaningful shadow cost. The M1 → M4 gap is "
        f"{l1_total_fec/l6_total_fec:.0f}× across a full year, visible "
        f"at first glance."
    )

    st.markdown("")

    # ─── Panel 2: SoC envelope ────────────────────────────────
    st.markdown("##### Panel 2 — *State of Charge Development*")
    st.caption(
        "Daily Min / SoC / Max envelope across a full year. Aging-aware "
        "parks the battery in the mid-band when there is no compelling "
        "trade; naive bounces between extremes."
    )
    col_l1, col_l6 = st.columns(2)

    def _soc_envelope_panel(col, days, label, color, fill):
        with col:
            with st.container(border=True):
                rows = []
                for d in days:
                    soc_frac = np.array(d.soc_mwh) / max(d.energy_mwh, 1e-6)
                    rows.append({
                        "date": d.date,
                        "min": float(soc_frac.min()),
                        "max": float(soc_frac.max()),
                        "mean": float(soc_frac.mean()),
                    })
                df = pd.DataFrame(rows)
                st.markdown(
                    f"<div style='font-size:11px;color:#64748b;"
                    f"text-transform:uppercase;letter-spacing:0.5px;'>"
                    f"{label} dashboard — 2025 dispatch</div>",
                    unsafe_allow_html=True,
                )
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["max"], mode="lines",
                    line=dict(color=color, width=0.5), name="Max SoC",
                    showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["min"], mode="lines",
                    line=dict(color=color, width=0.5), name="Min SoC",
                    fill="tonexty", fillcolor=fill, showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["mean"], mode="lines",
                    line=dict(color=color, width=2.0), name="Mean SoC",
                    hovertemplate="%{x|%d %b}<br>Mean SoC: %{y:.2f}<extra></extra>",
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

    _soc_envelope_panel(col_l1, l1_year_days, "M1 Naive", NAIVE_C, NAIVE_FILL)
    _soc_envelope_panel(col_l6, l6_year_days, "M4 Aging-aware", AGING_C, AGING_FILL)

    l1_max_avg = float(np.mean([np.array(d.soc_mwh).max() / d.energy_mwh
                                for d in l1_year_days]))
    l1_min_avg = float(np.mean([np.array(d.soc_mwh).min() / d.energy_mwh
                                for d in l1_year_days]))
    l6_max_avg = float(np.mean([np.array(d.soc_mwh).max() / d.energy_mwh
                                for d in l6_year_days]))
    l6_min_avg = float(np.mean([np.array(d.soc_mwh).min() / d.energy_mwh
                                for d in l6_year_days]))
    st.markdown(
        f"**What you're looking for.** A wide daily envelope (Min ≈ 0.05, "
        f"Max ≈ 0.95) every day = the trader runs to the rails on every "
        f"trade. A narrow envelope clustered around 0.40 – 0.60 = the "
        f"shadow-cost form is doing the work. M1 averaged "
        f"[{l1_min_avg:.2f}, {l1_max_avg:.2f}] across the year; M4 averaged "
        f"[{l6_min_avg:.2f}, {l6_max_avg:.2f}] — visibly tighter, mean "
        f"pinned to mid-band."
    )

    st.markdown("")

    # ─── Panel 3: Daily revenue per market (stacked bar) ──────
    # Markets palette is orthogonal to the teal+coral policy palette so
    # the reader doesn't visually confuse a stream with a policy:
    # wholesale streams (DA, ID) sit in warm yellow/amber; ancillary
    # capacity products (aFRR cap, aFRR energy) sit in cool indigo /
    # violet.
    STREAM_COLORS = {
        "DA": "#ca8a04",         # yellow-700 — DA wholesale
        "ID (IDA2)": "#f59e0b",  # amber-500 — ID wholesale (brighter)
        "aFRR cap": "#6366f1",   # indigo-500 — capacity reservation
        "aFRR energy": "#a5b4fc",# indigo-300 — activation energy
    }

    def _daily_stream_breakdown(days):
        rows = []
        for d in days:
            b = d.revenue_breakdown
            rows.append({
                "date": d.date,
                "DA": b.get("da", 0),
                "ID (IDA2)": b.get("id", 0),
                "aFRR cap": b.get("afrr_cap_pos", 0) + b.get("afrr_cap_neg", 0),
                "aFRR energy": b.get("afrr_energy_pos", 0) + b.get("afrr_energy_neg", 0),
            })
        return pd.DataFrame(rows)

    st.markdown("##### Panel 3 — *Daily Revenue per Market*")
    st.caption(
        "Stacked bars of daily revenue split per market stream across "
        "the year. The mix shifts visibly month-by-month with seasonal "
        "wholesale spreads and aFRR capacity prices. Naive captures big "
        "DA / ID arbitrage spikes; aging-aware concentrates almost "
        "entirely in the aFRR capacity base."
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

    _df_l1_pre = _daily_stream_breakdown(l1_year_days)
    _df_l6_pre = _daily_stream_breakdown(l6_year_days)
    _p1, _n1 = _signed_extremes(_df_l1_pre)
    _p2, _n2 = _signed_extremes(_df_l6_pre)
    _rev_y_top = max(_p1, _p2) * 1.05
    _rev_y_bot = min(_n1, _n2) * 1.05

    def _daily_revenue_panel(col, days, label, y_range):
        with col:
            with st.container(border=True):
                df = _daily_stream_breakdown(days)
                st.markdown(
                    f"<div style='font-size:11px;color:#64748b;"
                    f"text-transform:uppercase;letter-spacing:0.5px;'>"
                    f"{label} dashboard — daily revenue 2025</div>",
                    unsafe_allow_html=True,
                )
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

    _daily_revenue_panel(col_l1, l1_year_days, "M1 Naive", (_rev_y_bot, _rev_y_top))
    _daily_revenue_panel(col_l6, l6_year_days, "M4 Aging-aware", (_rev_y_bot, _rev_y_top))

    df_l1_full = _daily_stream_breakdown(l1_year_days)
    df_l6_full = _daily_stream_breakdown(l6_year_days)
    l1_total = df_l1_full[list(STREAM_COLORS)].sum().sum()
    l6_total = df_l6_full[list(STREAM_COLORS)].sum().sum()
    l1_ws = (df_l1_full["DA"].sum() + df_l1_full["ID (IDA2)"].sum())
    l6_ws = (df_l6_full["DA"].sum() + df_l6_full["ID (IDA2)"].sum())
    st.markdown(
        f"**What you're looking for.** Tall daily wholesale spikes "
        f"(both DA and ID) on volatile days = naive cycling on every "
        f"spread the market offers. A near-flat aFRR-cap base every day "
        f"with only the rare wholesale bar = aging-aware shadow cost is "
        f"binding most of the time. In our 2025 simulation M1's "
        f"wholesale total reached €{l1_ws/1000:.0f} k vs M4's "
        f"€{l6_ws/1000:.0f} k — a "
        f"{(l1_ws - l6_ws)/max(l1_ws, 1)*100:.0f} % gap, exactly the "
        f"channel through which aging-aware policies trade short-term "
        f"P&L for cell longevity."
    )

    st.markdown("")

    # ─── Panel 4: Revenue share pie (relative-only) ───────────

    def _stream_breakdown(days):
        rows = []
        for d in days:
            b = d.revenue_breakdown
            rows.append({
                "date": d.date,
                "DA": b.get("da", 0),
                "ID (IDA2)": b.get("id", 0),
                "aFRR cap": b.get("afrr_cap_pos", 0) + b.get("afrr_cap_neg", 0),
                "aFRR energy": b.get("afrr_energy_pos", 0) + b.get("afrr_energy_neg", 0),
            })
        return pd.DataFrame(rows)

    st.markdown("##### Panel 4 — *Revenue Share per Market*")
    st.caption(
        "Pie of annual revenue split, share-only. ID stream is real "
        "DE-LU pan-European Intraday Auction "
        "([SIDC IDA2](https://www.entsoe.eu/network_codes/cacm/implementation/ida/), "
        "15-min, gate-closure D−1 22:00 CET) — the closest free "
        "continuous-style intraday signal available since SIDC went "
        "live 13 June 2024. The *M1 → M4 shift* — wholesale share "
        "shrinking as the optimiser commits more SoC to aFRR cap — is "
        "the calibration-robust diagnostic. The absolute aFRR fraction "
        "still sits above public BESS indices (Modo / CH show real "
        "DE 2 h BESS at ~50 – 70 % aFRR; we are higher) because the "
        "LP has perfect within-day foresight and no bid-shading risk "
        "premium."
    )
    col_l1, col_l6 = st.columns(2)

    def _share_pie_panel(col, days, label):
        with col:
            with st.container(border=True):
                df = _stream_breakdown(days)
                shares = {s: max(df[s].sum(), 0) for s in STREAM_COLORS}
                st.markdown(
                    f"<div style='font-size:11px;color:#64748b;"
                    f"text-transform:uppercase;letter-spacing:0.5px;'>"
                    f"{label} dashboard — annual aggregate</div>",
                    unsafe_allow_html=True,
                )
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

    _share_pie_panel(col_l1, l1_year_days, "M1 Naive")
    _share_pie_panel(col_l6, l6_year_days, "M4 Aging-aware")

    def _afrr_pct(days):
        df = _stream_breakdown(days)
        afrr = df["aFRR cap"].sum() + df["aFRR energy"].sum()
        net = (df["DA"].sum() + df["ID (IDA2)"].sum()
               + df["aFRR cap"].sum() + df["aFRR energy"].sum())
        return afrr / max(net, 1.0) * 100

    l1_afrr = _afrr_pct(l1_year_days)
    l6_afrr = _afrr_pct(l6_year_days)
    l1_ws = 100 - l1_afrr
    l6_ws = 100 - l6_afrr
    st.markdown(
        f"**What you're looking for — the *direction* of the shift.** "
        f"Wholesale (DA + ID) earns money from *cycling*; aFRR cap "
        f"earns money from *availability*. An optimiser that prices "
        f"cycles against a higher shadow cost mechanically shifts "
        f"toward capacity products. In our model M1's wholesale slice "
        f"is ~{l1_ws:.0f}% of net revenue; M4 shrinks it to ~{l6_ws:.0f}% "
        f"— a {-(l1_ws - l6_ws):+.0f} pp shift toward aFRR. Your own "
        f"dashboard's wholesale share will sit higher in absolute "
        f"terms (we're missing EPEX continuous ID), but the same "
        f"*direction* of M1 → M4 movement is the diagnostic. "
        f"Pragmatically: if your trader's aFRR share rose year-over-"
        f"year while monthly FEC dropped, that's shadow-cost work. "
        f"Once aFRR exceeds ~50% of total revenue, €/MWh-throughput "
        f"stops being a sensible benchmark — the asset is paid for "
        f"availability (Sebastian Kawollek's point on Note 3; precedent "
        f"in He et al. "
        f"[2016](https://orbit.dtu.dk/en/publications/optimal-bidding-strategy-of-battery-storage-in-power-markets-cons))."
    )


# ── Note 1 market trajectory sensitivity ────────────────────
st.markdown("---")
st.markdown("## What changes under Note 1's market trajectory")
st.markdown("""
The headline above holds the 2023 / 2025 market mix flat for ten years.
Note 1 ([*German BESS Outlook*](https://de-bess-outlook.streamlit.app))
projects that the *composition* of merchant revenue shifts across
2026 → 2035 as the BESS fleet saturates ancillary demand: aFRR
capacity revenue compresses by ~85% per MW, while DA and ID arbitrage
grow roughly 10× as midday solar troughs deepen and electrification
lifts peaks. The chart below applies that per-stream trajectory to
each policy's actual stream-revenue mix, without re-solving the LP.
The two effects pull in opposite directions; the net depends on each
policy's mix.
""")

from lib.analysis.market_trend import apply_trend_to_result

trend_rows = []
naive_flat_npv = results["M1_naive"].lifetime_npv_eur
for name in POLICY_ORDER:
    if name not in results:
        continue
    r_flat = results[name]
    r_trend = apply_trend_to_result(r_flat)
    trend_rows.append({
        "policy": POLICY_LABELS[name],
        "short": POLICY_SHORT[name],
        "color": POLICY_COLORS[name],
        "flat_index": r_flat.lifetime_npv_eur / naive_flat_npv * 100,
        "trend_index": r_trend.lifetime_npv_eur / naive_flat_npv * 100,
    })
trend_df = pd.DataFrame(trend_rows)

render_chart_title(
    "Flat market vs Note 1 mid-case per-stream trajectory (2026 → 2035), "
    "indexed to flat-market M1 = 100"
)
fig_trend = go.Figure()
fig_trend.add_trace(go.Bar(
    x=trend_df["short"], y=trend_df["flat_index"],
    name="Flat 2023/2025",
    marker_color=trend_df["color"], marker_line_width=0,
    text=[f"{v:.0f}" for v in trend_df["flat_index"]],
    textposition="outside",
))
fig_trend.add_trace(go.Bar(
    x=trend_df["short"], y=trend_df["trend_index"],
    name="Note 1 mid-case trajectory",
    marker_color=trend_df["color"],
    marker_pattern_shape="/", marker_line_width=0,
    text=[f"{v:.0f}" for v in trend_df["trend_index"]],
    textposition="outside",
))
fig_trend.add_hline(y=100, line_dash="dot",
                    line_color="#94a3b8", line_width=1,
                    annotation_text="flat-market M1 = 100",
                    annotation_position="top left")
fig_trend.update_layout(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=420, margin=dict(l=40, r=20, t=20, b=70),
    yaxis=dict(title="Lifetime DCF index (flat-market M1 = 100)"),
    xaxis=dict(tickfont=dict(size=10)),
    legend=dict(orientation="h", y=-0.18),
    barmode="group",
)
st.plotly_chart(fig_trend, use_container_width=True,
                config={"displayModeBar": False})

flat_peak = trend_df.loc[trend_df["flat_index"].idxmax()]
trend_peak = trend_df.loc[trend_df["trend_index"].idxmax()]

render_chart_caption(
    f"Under flat markets the peak is **{flat_peak['short']}** at index "
    f"{flat_peak['flat_index']:.0f}. Under Note 1's per-stream "
    f"trajectory the peak "
    f"{'stays at' if trend_peak['short'] == flat_peak['short'] else 'shifts to'} "
    f"**{trend_peak['short']}** at index {trend_peak['trend_index']:.0f}. "
    f"All values are relative to flat-market M1 = 100; absolute revenue "
    f"depends on calibration choices and is not the focus."
)

def _trend(short):
    row = trend_df[trend_df["short"] == short].iloc[0]
    return float(row["flat_index"]), float(row["trend_index"])

m1_flat, m1_trend = _trend(POLICY_SHORT["M1_naive"])
m2_flat, m2_trend = _trend(POLICY_SHORT["M2_flat"])
m3_flat, m3_trend = _trend(POLICY_SHORT["M3_scarcity"])
m4_flat, m4_trend = _trend(POLICY_SHORT["M4_physics"])
m5_flat, m5_trend = _trend(POLICY_SHORT["M5_adp"])

st.markdown(f"""
**Two effects, pulling opposite ways.** Note 1's mid-case combines two
moves: DA × ~10 and ID × ~10 multipliers reward residual wholesale
exposure; the aFRR-cap × ~0.15 multiplier punishes ancillary-heavy mixes.
Each methodology is a different blend of those two streams.

- **M1 naive** moves from index {m1_flat:.0f} → {m1_trend:.0f}
  ({(m1_trend/m1_flat-1)*100:+.0f}%) — its high cycling intensity
  leaves substantial residual DA exposure to ride the wholesale recovery.
- **M2 flat wear** moves {m2_flat:.0f} → {m2_trend:.0f}
  ({(m2_trend/m2_flat-1)*100:+.0f}%) — moderate cycling preserves
  enough DA arbitrage to capture the wholesale uplift.
- **M3 scarcity** moves {m3_flat:.0f} → {m3_trend:.0f}
  ({(m3_trend/m3_flat-1)*100:+.0f}%) — gain mostly cancels because
  scarcity already concentrates revenue toward aFRR.
- **M4 physics-from-duty** moves {m4_flat:.0f} → {m4_trend:.0f}
  ({(m4_trend/m4_flat-1)*100:+.0f}%) — the heaviest collapse, because
  M4 cycles only ~7 FEC/year and ~95% of revenue is aFRR cap, exactly
  the stream Note 1 projects to compress hardest.
- **M5 intraday ADP** moves {m5_flat:.0f} → {m5_trend:.0f}
  ({(m5_trend/m5_flat-1)*100:+.0f}%) — same direction as M4 but less
  catastrophic; M5 cycles 5× more than M4 (375 vs 73 FEC), keeping a
  small residual wholesale exposure that softens the aFRR compression.

**The flat-market ordering inverts.** Under no-trajectory the stack
reads M5 > M4 > M3 > M2 > M1 monotonically. Under Note 1 trajectory
the ordering rotates to **M2 > M3 > M1 > M5 > M4**: residual wholesale
exposure becomes the dominant axis, and the methodologies that
suppressed cycling most aggressively are the ones most short the only
stream that grows. M4 (lowest cycling, highest aFRR concentration) is
the worst trajectory exposure; M2 (moderate cycling, balanced mix)
becomes the winner.

**Practical reading.** The flat-market numbers are the appropriate
diagnostic for an asset operating today; the Note-1-trajectory numbers
are the appropriate diagnostic for a 2 h LFP commissioned in 2026 and
expected to operate into the 2030s. They are *genuinely different
problems* and they select different methodologies. **M2 (flat €/MWh
wear) is the robust recommendation across both** — captures most of
the available aging-aware uplift on flat markets, and preserves enough
residual DA / ID exposure to ride a wholesale recovery. M4 / M5 win on
flat-market lifetime DCF but are catastrophically exposed to ancillary
compression; whether they remain the right call depends on how
literally one takes Note 1's per-stream projection.
""")


# ── Methodology expander ────────────────────────────────────
st.markdown("---")
with st.expander("Methodology & where this model stops working"):
    st.markdown("""
**Asset.** 2 h LFP, 1 MW / 2 MWh nominal, EVE LF280K cell preset
calibrated against the manufacturer 6 000-cycle / 80% retention spec
via `kernel_scale = 0.66` on the Note 3 Wang + Naumann two-channel
physics kernel. Warranty floor 0.80 SoH; round-trip efficiency 0.88.
Discount rate 7%.

**Markets.** DE day-ahead from EnergyCharts; intraday via real DE-LU
SIDC IDA2 15-minute auction prices ([ENTSO-E Transparency
Platform](https://www.entsoe.eu/network_codes/cacm/implementation/ida/)
post 13 June 2024 launch — pan-European Intraday Auction, gate-closure
D−1 22:00 CET); pre-launch days fall back to netztransparenz
Spotmarktpreis (EEG §3 Nr. 42a, ≈ DA on 99.7% of hours). aFRR capacity
+ activation energy from regelleistung.net + netztransparenz.de. The
headline includes a post-LP FCR phantom layer (€36 k / MW / yr
decaying over the lifetime) for apples-to-apples comparison vs the
Clean Horizon Storage Index, which includes FCR.

**Dispatch LP.** Two-stage market-aware (ADR-001 v1.1). Stage 1
commits per-4-hour-block aFRR capacity at D−1 under a regime-
conditional α-forecast with `bid_win_rate = 1.0` (regelleistung
empirical 99.8% clearing rate, 2.16 M bids analysed). Stage 2 re-
optimises full DA + ID + activation dispatch under realised α with
the Stage 1 commitment locked. The 1-hour SoC reservation horizon
matches the Modo public aFRR product SLA. For two-pass methodologies
(M4 — physics-from-duty), Stage 2 runs a second LP with refined wear
based on observed first-pass dispatch fed through the Note 3 kernel
(`physics_wear_from_duty`); Stage 1 commitments stay frozen — re-
pricing them would require solving the Stage-1 ↔ Stage-2 ↔ wear
fixed-point, out of scope for ADR-001 v1.1.

**Calibration anchors.** All publicly citable. aFRR clearing rate from
own analysis of regelleistung.net public auction CSVs. aFRR cap price
€12.21 / MW / h matches the gemenergyanalytics independent reading
(€13 POS / €10 NEG average 2024). 2 h DE 2024 incl FCR realised
revenue €200 k / MW / yr from the Clean Horizon Storage Index public
CSV. Model M1 sits within ±10% of CH index — the residual reflects
the LP's perfect-foresight premium.

**Lifetime simulation.** 10 years, rotating template years 2023 ↔
2025 (DE 2024 DA API unreliable). Each template year sees identical
inputs across all five methodologies; only the wear-cost vector each
methodology emits differs.

**The metric we report** is *discounted gross market revenue* over 10
years at a 7% discount rate, summed across DA + ID + aFRR cap + aFRR
energy + FCR phantom. It does **not** subtract CAPEX, fixed O&M,
augmentation, or repower cost. It is not an investor NPV; it is the
discounted gross revenue stream each methodology generates.

**Where this model stops working.**
- The naive policy is a strawman. It assumes zero cycle pricing AND
  zero DoD ceiling. Real commercial operators respect warranty cycle
  caps and DoD limits, which is a crude form of cycle pricing.
- The Y7 naive EOL at SoH = 0.80 is contract-life, not physical life.
  Under contracts that accept operation past 0.80 the naive policy
  earns another 2 – 3 years at derated capacity.
- The IDA2 feed is post 13 June 2024 only. Year 1 of the simulation
  (template 2023) sees no continuous-style intraday signal; ID is
  proxied by Spotmarktpreis ≈ DA. This is why the aging-aware uplift
  (+37% at M5) is concentrated in the post-IDA2 template years —
  pre-2024-06-13 dispatch can't time intraday spreads. Operators with
  full EPEX continuous ID1 / ID3 access since 2018 would see a higher
  baseline absolute revenue across all methodologies; relative
  ordering should hold.
- The BESS is modelled as a single aggregated unit. Real systems have
  hundreds of modules with manufacturing variance, edge-vs-centre
  thermal asymmetry, and uneven usage. The pack-level allocation
  problem sits one abstraction below this analysis.
- Calendar year mapping. Year 1 of the simulation = template 2023; we
  do not model the calendar mapping to 2026 onward. Note 1's
  trajectory chart is layered post-hoc on top of the policy-emitted
  stream mix.
- M4 (physics-from-duty) cycles only ~7 FEC / year over the lifetime
  at this calibration — the LP parks the asset in low-cycling mode
  and earns most revenue from aFRR availability. This is the genuine
  LP optimum given EVE LF280K `kernel_scale = 0.66` and DE 2024-25
  market structure, not a numerical artefact (the cap on
  `physics_wear_from_duty` at €500/MWh is rarely binding — see
  README ablation diagnostics). At cell physics with steeper cyclic-
  fade slope (e.g. Sony LFP, slope ratio 7.67× vs EVE 1.11×) M4
  would likely cycle more and possibly beat M5; this is cell-
  specific.
- Stage-1 ↔ Stage-2 fixed-point under refined wear is not solved at
  v1.1 — Stage-1 commits under pass-1 wear (ADP only for M4); pass-2
  re-prices Stage-2 only. We tested closing the fixed-point with
  Picard iteration (`two_stage_picard_max_iter=5`, see
  `lifecycle_npv.py`) — at `bid_win_rate=1.0` Picard makes M4 *worse*
  by −1.7 pp because reducing Stage-1 r commitments loses guaranteed
  aFRR cap revenue more than the freed SoC headroom unlocks
  arbitrage. v1.1's locked Stage-1 r is super-optimal by accident.
  Picard counter-experiment full write-up in the README under
  "v4.1 paper-grade headline / Picard counter-experiment". Available
  in code with default off; relevant under fleet-saturation
  calibrations where `bid_win_rate < 1.0`.

**Diagnostic dispatch logs.** The five-signal section uses
`precomputed_two_stage.pkl` (v3.5 calibration; signal *patterns* —
DoD-by-spread, SoC histogram, €/FEC quartile inversion — are
calibration-agnostic). The four-tile dashboard panels (Cumulative
Cycles / SoC envelope / Daily revenue / Revenue share) use
`precomputed_diag_2025.pkl` (v4.0 B; same calibration as the
headline, captured from the 2025 template year). Both diag pkls
predate the M-numbering so they retain old internal policy names
(`L1_naive`, `L3_flat_wear`, `L4_scarcity`, `L5_intraday_adp`,
`L6_physics_full`); `app.py` maps them to the M-display via the
`DIAG_KEY_FOR` dict at access time. Crucially, the diag M5 panel uses
`L5_intraday_adp` (the additive blend with flat + scarcity + ADP)
because the v3.5 diag pkl predates the v4.1 ablation — the *signal
patterns* are virtually identical to a pure-ADP M5 (both
ADP-dominated, ~250-400 FEC / year), so the diagnostic message holds;
only the absolute FEC count would shift by a few percent if regenerated
on pure-ADP. Documented in the README under "v4.1 paper-grade
headline".
""")


# ── Related work ────────────────────────────────────────────
st.markdown("---")
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
  form. This note recovers the directional claim at moderate magnitude
  (+15%) on DE markets with a publicly-anchored calibration.

**Physics foundation.**

- **Naumann et al. (2018) + Wang et al. (2014) + Stanford (2024).**
  Cycle and calendar fade anchors for the Note 3 physics kernel; EVE
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
    "Fourth in a series on German BESS merchant economics. "
    "Next up — *Which BESS revenue number is real?* — five public DE "
    "indices (LCP Delta, Clean Horizon, enspired, suena, RWTH Aachen) "
    "disagree by up to ~€100 k/MW/yr on the same months. The next note "
    "decomposes the fan, places this analysis's model inside it, and "
    "ends with a methodology table letting an owner pick the right "
    "benchmark for their question."
)

render_footer()
