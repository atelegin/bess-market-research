"""
Scatter and bar charts — tail-day pattern distribution.

Shows what kind of market conditions (spike, duck curve, etc.)
dominate the highest-revenue days.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_tail_pattern_figure(pattern_distribution: dict[str, int]) -> go.Figure:
    pattern_frame = pd.DataFrame(
        {
            "pattern": list(pattern_distribution.keys()),
            "count": list(pattern_distribution.values()),
        }
    )
    fig = go.Figure(
        go.Bar(
            x=pattern_frame["count"],
            y=pattern_frame["pattern"],
            orientation="h",
            marker={"color": ["#264653", "#e76f51", "#2a9d8f", "#94a3b8"][: len(pattern_frame)]},
            text=pattern_frame["count"],
            textposition="outside",
            hovertemplate="%{y}<br>%{x} of top 20 days<extra></extra>",
        )
    )
    fig.update_layout(
        title="What Kind of Days Dominate the Tail",
        xaxis_title="Count of top-20 days",
        yaxis_title="",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=320,
    )
    return fig


def build_price_shape_figure(profile_payload: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    median_profiles = profile_payload["median_profiles"]
    fig.add_trace(
        go.Scatter(
            x=median_profiles["hour"],
            y=median_profiles["all_days"],
            mode="lines",
            name="Median of all days",
            line={"color": "#264653", "width": 3, "dash": "dash"},
            hovertemplate="All days<br>Hour %{x}<br>Price %{y:.1f} €/MWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=median_profiles["hour"],
            y=median_profiles["top_days"],
            mode="lines",
            name="Median of top-20 revenue days",
            line={"color": "#e76f51", "width": 4},
            hovertemplate="Top-20 revenue days<br>Hour %{x}<br>Price %{y:.1f} €/MWh<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Hour of day",
        yaxis_title="Day-ahead price (€/MWh)",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 24, "b": 10},
        height=380,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    return fig
