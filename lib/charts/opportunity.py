"""
Opportunity charts — missed-days impact curve.

Visualises how much annual revenue is lost by missing the top-N days.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_missed_days_figure(missed_days: pd.DataFrame, highlight_day: int = 20) -> go.Figure:
    x_values = missed_days.index.to_numpy(dtype=int)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=missed_days["lost_share_pct"],
            mode="lines+markers",
            name="Revenue lost",
            line={"color": "#264653", "width": 4},
            marker={"size": 7, "color": "#264653"},
            hovertemplate="Miss top %{x} days<br>Lost share %{y:.1f}%<extra></extra>",
        )
    )
    if highlight_day in missed_days.index:
        highlight_row = missed_days.loc[highlight_day]
        fig.add_trace(
            go.Scatter(
                x=[highlight_day],
                y=[highlight_row["lost_share_pct"]],
                mode="markers",
                marker={"size": 14, "color": "#e76f51", "line": {"color": "#ffffff", "width": 2}},
                hovertemplate=(
                    f"Miss top {highlight_day} days"
                    "<br>Lost share %{y:.1f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=highlight_day,
            y=float(highlight_row["lost_share_pct"]),
            text=f"<b>{highlight_day} days missed</b><br>{highlight_row['lost_share_pct']:.1f}% revenue loss",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            xshift=18,
            yshift=16,
            bgcolor="rgba(255, 251, 245, 0.96)",
            bordercolor="rgba(231, 111, 81, 0.35)",
            borderwidth=1,
            borderpad=6,
            font={"size": 11},
        )
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 24, "b": 10},
        height=360,
    )
    fig.update_xaxes(title_text="Number of top revenue days missed")
    fig.update_yaxes(title_text="Annual revenue lost (%)")
    return fig


def build_early_warning_screen_figure(screen_frame: pd.DataFrame) -> go.Figure:
    frame = screen_frame.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["stage"],
            y=frame["lift_x"],
            marker={"color": frame["color"].tolist()},
            width=0.52,
            text=[f"{value:.2f}x" for value in frame["lift_x"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Lift %{y:.2f}x<extra></extra>",
            showlegend=False,
        )
    )
    for _, row in frame.iterrows():
        fig.add_annotation(
            x=row["stage"],
            y=-0.16,
            xref="x",
            yref="paper",
            text=row["rule_label_html"],
            showarrow=False,
            xanchor="center",
            yanchor="top",
            align="center",
            font={"size": 10, "color": "#14213d"},
        )
        fig.add_annotation(
            x=row["stage"],
            y=-0.28,
            xref="x",
            yref="paper",
            text=(
                row["supporting_label_html"]
                if "supporting_label_html" in row and pd.notna(row["supporting_label_html"])
                else (
                    f"<span style='font-size:10px;color:#5c677d'>"
                    f"{row['precision_pct']:.2f}% precision | {row['recall_pct']:.0f}% recall"
                    "</span>"
                )
            ),
            showarrow=False,
            xanchor="center",
            yanchor="top",
            align="center",
        )
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 24, "b": 130},
        height=420,
        bargap=0.36,
        showlegend=False,
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(
        title_text="Lift vs base rate",
        range=[0, max(4.3, float(frame["lift_x"].max()) + 0.55)],
        gridcolor="rgba(20, 33, 61, 0.08)",
        zeroline=False,
    )
    return fig


def build_same_cycles_reallocation_figure(summary_frame: pd.DataFrame) -> go.Figure:
    frame = summary_frame.reset_index().copy()
    frame["pair_label"] = frame["strict_daily_cap"].map(lambda value: f"{value:.1f}/day")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["pair_label"],
            y=frame["uplift_eur_per_mw"],
            name="Revenue uplift",
            marker={"color": "#e76f51"},
            text=[
                f"+{uplift / 1000:.1f}k €/MW<br>(+{uplift_pct:.1f}%)"
                for uplift, uplift_pct in zip(frame["uplift_eur_per_mw"], frame["uplift_pct_vs_strict"])
            ],
            textposition="outside",
            cliponaxis=False,
            customdata=frame[["strict_realized_fec"]].to_numpy(),
            hovertemplate=(
                "%{x}<br>Revenue uplift %{y:,.0f} €/MW"
                "<br>Same realised throughput %{customdata[0]:.1f} FEC"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 24, "b": 10},
        height=360,
        bargap=0.32,
        showlegend=False,
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Revenue uplift from reallocation (€/MW)")
    return fig
