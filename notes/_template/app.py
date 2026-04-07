"""
Market Note: [TITLE]
[One-line description]

Results are pre-computed by precompute.py and loaded from data/precomputed.pkl.
"""

import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

# from lib.data.day_ahead_prices import fetch_day_ahead_prices
# from lib.models.dispatch_detailed import optimize_day

PRECOMPUTED_PATH = Path(__file__).parent / "data" / "precomputed.pkl"


# ── Data loading ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_precomputed() -> dict:
    with open(PRECOMPUTED_PATH, "rb") as f:
        return pickle.load(f)


# ── Page config ─────────────────────────────────────────────
st.set_page_config(page_title="[TITLE]", layout="wide")
apply_theme(show_sidebar=False)  # set True if sidebar is needed


# ── Header ──────────────────────────────────────────────────
render_header(
    title="[TITLE]",
    kicker="GERMAN BESS | [TOPIC]",
    subtitle="[One-line tagline]",
)

# ── Intro (2-4 sentences) ──────────────────────────────────
# What this article does, for whom, what the reader gets.
# Never jump straight into data or executive summary.
st.markdown("""
[Intro paragraph: context, motivation, what the reader will learn.
Use the sidebar to...]
""")


# ── Load pre-computed results ────────────────────────────────
data = load_precomputed()


# ── KPIs ────────────────────────────────────────────────────
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("Label", "€Xk/MW")


# ── Main chart ──────────────────────────────────────────────
# st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
# render_chart_caption("What the chart shows. Source links. Assumptions.")


# ── Executive Summary ────────────────────────────────────────
# st.markdown("---")
# st.markdown("""
# ## Executive Summary
#
# [2-3 paragraphs. The whole story compressed. Use hedged language
# for projections: "the model projects", "is expected to".]
# """)


# ── Sections ─────────────────────────────────────────────────
# Each section heading must be self-explanatory (no "Is it enough?"
# without "enough for what?").
#
# Language reminders:
# - Projections need hedging ("is projected to", not "will")
# - Explain abbreviations at first use: CAISO (California grid operator)
# - German terms get English in parens: *Kraftwerksstrategie* (power plant strategy)
# - Use → for causal chains: Higher demand → higher prices → wider spreads
# - Relative claims over absolutes: "loses a quarter" not "earns €230k"
# - No loose ends: if a claim sets up a topic, reference where it's addressed
#
# st.markdown("""
# ## [Self-explanatory heading]
#
# [Section content]
# """)
#
# render_chart_title("Chart headline")
# st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
# render_chart_caption("What the chart shows.")
#
# render_takeaway("Key insight from this section.")


# ── What this model does NOT include ─────────────────────────
# st.markdown("---")
# st.markdown("""
# ## What this model does NOT include
#
# - **[Omission]** — [why it's excluded, what it would mean if included]
# """)


# ── Data Sources & Methodology ───────────────────────────────
# Use expanders to keep detail out of the main flow.
# Each expander covers one topic: model, data, metrics, etc.
# Include source links and parameter tables.
#
# st.markdown("---")
# st.markdown("## Data Sources & Methodology")
#
# with st.expander("Model / dispatch"):
#     st.markdown("""
# [Model description, key assumptions]
#
# | Parameter | Value |
# |:---|:---|
# | ... | ... |
# """)
#
# with st.expander("Price / market data"):
#     st.markdown("""
# | Market | Source | Resolution |
# |:---|:---|:---|
# | Day-ahead | [Energy-Charts](https://energy-charts.info/) | Hourly |
# | Intraday | [Netztransparenz](https://ds.netztransparenz.de) | 15-min |
# | FCR capacity | [regelleistung.net](https://www.regelleistung.net) | 4h products |
# | aFRR capacity | [regelleistung.net](https://www.regelleistung.net) | 4h products |
# """)
#
# with st.expander("Key metrics / definitions"):
#     st.markdown("""
# **Full equivalent cycle (FEC)** = cumulative discharged energy ÷ nameplate
# energy capacity. Standard definition per manufacturer warranties and DNV.
# """)


# ── Closing ──────────────────────────────────────────────────
# st.markdown("---")
# render_closing(
#     "[Series framing — e.g. 'This is the Nth note in a series on BESS merchant economics...']"
# )
# st.markdown(
#     '<div style="margin-top: 0.5rem; font-size: 0.95rem; color: #666;">'
#     "<b>Next:</b> [teaser for next note]"
#     "</div>",
#     unsafe_allow_html=True,
# )

render_footer()
