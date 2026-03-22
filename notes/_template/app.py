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
    subtitle="[One-line description]",
)

render_standfirst("[Opening paragraph — context and motivation.]")


# ── Load pre-computed results ────────────────────────────────
data = load_precomputed()


# ── Analysis ────────────────────────────────────────────────

# TODO: analysis code here — use data["key"] to access precomputed results

# render_chart_title("Chart headline")
# st.plotly_chart(fig, use_container_width=True)
# render_chart_caption("What the chart shows.")

# render_takeaway("Key insight from this section.")

# render_annotation("Why this matters", "Explanation text.")


# ── Footer ──────────────────────────────────────────────────
render_footer()
