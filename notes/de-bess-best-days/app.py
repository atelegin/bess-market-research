from __future__ import annotations

import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from lib.charts.opportunity import (
    build_early_warning_screen_figure,
    build_missed_days_figure,
    build_same_cycles_reallocation_figure,
)
from lib.charts.scatter import build_price_shape_figure
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

REPORT_TITLE = "The Cost of Missing the Best Days"


def apply_app_style() -> None:
    apply_theme(show_sidebar=False)


def render_report_header() -> None:
    render_header(
        title="The Cost of Missing the Best Days",
        kicker="GERMAN BESS | MERCHANT REVENUES | AVAILABILITY | FLEXIBILITY",
        subtitle="German BESS merchant revenues are concentrated in a limited set of days. Many of those days are already partly visible in the day-ahead curve, making availability and cycling flexibility most valuable when they are timed.",
    )


@st.cache_data(show_spinner=False)
def load_precomputed() -> dict:
    with open(PRECOMPUTED_PATH, "rb") as f:
        return pickle.load(f)


def render_intro() -> None:
    render_standfirst(
        "German BESS merchant revenues are not earned evenly through the year. A limited set of high-opportunity days drives a disproportionate share of annual value, and missing them is expensive. That matters because many of the best days are already partly visible in the day-ahead curve, making availability, maintenance timing, and cycling flexibility market-timed decisions rather than purely operational ones."
    )
    render_standfirst(
        "This note uses 2025 as the main case study for a 2h battery, with pooled 2021\u20132025 data used to test whether simple day-ahead signals generalise beyond a single year."
    )


def main() -> None:
    st.set_page_config(page_title=REPORT_TITLE, layout="wide", initial_sidebar_state="collapsed")
    apply_app_style()
    render_report_header()
    render_intro()
    render_footer_note(
        "<strong>Base case:</strong> 2h battery, 2025 deep dive"
        "<br><strong>Validation:</strong> pooled 2021–2025"
        "<br><strong>Method note:</strong> Merchant revenue is modelled as a combined day-ahead plus intraday series, using Energy-Charts day-ahead prices and the official Netztransparenz ID-AEP index for the intraday layer. Dispatch is modelled sequentially across day-ahead and intraday with a fixed round-trip efficiency of 0.86."
    )

    # ── Load pre-computed results ─────────────────────────────
    data = load_precomputed()
    analysis_year = data["analysis_year"]
    duration_hours = data["duration_hours"]
    concentration_stats = data["concentration_stats"]
    missed_days_curve = data["missed_days_curve"]
    price_shape_profiles = data["price_shape_profiles"]
    feature_comparison = data["feature_comparison"]
    pooled_watchlist_top20 = data["pooled_watchlist_top20"]
    pooled_base_rate_pct = data["pooled_base_rate_pct"]
    equal_throughput_summary = data["equal_throughput_summary"]
    intraday_missing_years = data["intraday_missing_years"]
    conservative_selected = data["conservative_selected"]
    selected_prices = data["selected_prices"]

    top_20pct_share = concentration_stats["top_20_days_pct_of_revenue"]
    missed_day_twenty = missed_days_curve.loc[20]

    d1_watchlist = pooled_watchlist_top20.loc["DA spread >= 200 €/MWh"]
    d2_watchlist = {
        "lift_x": 2.24,
        "precision_pct": 12.27,
        "recall_pct": 27.0,
    }
    watchlist_screen_summary = pd.DataFrame(
        [
            {
                "stage": "D-2 early warning",
                "lift_x": d2_watchlist["lift_x"],
                "precision_pct": d2_watchlist["precision_pct"],
                "recall_pct": d2_watchlist["recall_pct"],
                "rule_label_html": "weekday + recent 3d mean spread<br>≥ 175 €/MWh",
                "supporting_label_html": "<span style='font-size:10px;color:#5c677d'>~1 in 8 flagged days is a top day | catches 27% of top days</span>",
                "color": "#264653",
            },
            {
                "stage": "D-1 day-ahead screen",
                "lift_x": float(d1_watchlist["lift_x"]),
                "precision_pct": float(d1_watchlist["precision_pct"]),
                "recall_pct": float(d1_watchlist["recall_pct"]),
                "rule_label_html": "day-ahead spread<br>≥ 200 €/MWh",
                "supporting_label_html": "<span style='font-size:10px;color:#5c677d'>~1 in 5 flagged days is a top day | catches 50% of top days</span>",
                "color": "#e76f51",
            },
        ]
    )

    st.subheader("Revenue is concentrated where it matters most")
    st.markdown(
        f"Merchant BESS value is not earned evenly through the year. In {analysis_year}, the top 20% of days generated {top_20pct_share * 100:.1f}% of annual merchant revenue for a {duration_hours}h battery. That concentration matters because missing only a limited number of the best days can do disproportionate damage to annual returns. In {analysis_year}, missing the top 20 revenue days would have reduced annual revenue by {missed_day_twenty['lost_share_pct']:.1f}%."
    )
    render_chart_title("Missing a small number of the best days can do disproportionate damage to annual revenue")
    missed_days_figure = build_missed_days_figure(missed_days_curve, highlight_day=20)
    st.plotly_chart(missed_days_figure, width="stretch")
    render_footer_note(
        "Revenue loss measured as annual merchant revenue foregone when the highest-revenue days are assumed unavailable."
    )
    render_chart_caption(
        "In 2025, a relatively small number of days accounted for a disproportionate share of annual merchant value. Missing the top 20 revenue days would have reduced annual revenue by 20.4%."
    )
    render_annotation(
        "Why this matters",
        "For a merchant BESS owner, the main commercial risk is not a weak average day. It is being unavailable when a small number of highly valuable days arrives.",
    )

    st.subheader("The best days are partly visible ahead of delivery")
    st.markdown(
        f"The highest-value days were not identical, but they often shared a recognisable commercial shape. Relative to a normal day, they were more likely to show weaker prices around midday and stronger prices into the evening, creating a wider charge-discharge window for a {duration_hours}h battery."
    )
    st.markdown(
        f"In {analysis_year}, top-20 revenue days had a median day-ahead trough of {feature_comparison.loc['top_20_revenue_days', 'median_midday_min_price_eur_mwh']:.0f} €/MWh versus {feature_comparison.loc['all_days', 'median_midday_min_price_eur_mwh']:.0f} €/MWh on an average day, and a median evening-minus-midday ramp of {feature_comparison.loc['top_20_revenue_days', 'median_da_evening_minus_midday_ramp_eur_mwh']:.0f} €/MWh versus {feature_comparison.loc['all_days', 'median_da_evening_minus_midday_ramp_eur_mwh']:.0f} €/MWh."
    )
    st.markdown(
        "Some of these days were classic solar-surplus days, with cheap midday charging and stronger evening discharge. Others were driven more by broader market stress and repricing. The market drivers varied, but the commercial shape was similar: a wider and more valuable charging-to-discharging window."
    )
    render_chart_title("High-value days tended to show a deeper midday trough and a wider evening-minus-midday ramp")
    price_shape_figure = build_price_shape_figure(price_shape_profiles)
    median_profiles = price_shape_profiles["median_profiles"].set_index("hour")[["all_days", "top_days"]].astype(float)
    midday_profiles = median_profiles.loc[10:15]
    evening_profiles = median_profiles.loc[17:21]

    all_trough_hour = float(midday_profiles["all_days"].idxmin())
    all_trough_line_value = float(midday_profiles["all_days"].min())
    top_trough_hour = float(midday_profiles["top_days"].idxmin())
    top_trough_line_value = float(midday_profiles["top_days"].min())
    all_evening_hour = float(evening_profiles["all_days"].idxmax())
    all_evening_line_value = float(evening_profiles["all_days"].max())
    top_evening_hour = float(evening_profiles["top_days"].idxmax())
    top_evening_line_value = float(evening_profiles["top_days"].max())

    all_trough_metric = float(feature_comparison.loc["all_days", "median_midday_min_price_eur_mwh"])
    top_trough_metric = float(feature_comparison.loc["top_20_revenue_days", "median_midday_min_price_eur_mwh"])
    all_evening_metric = float(feature_comparison.loc["all_days", "median_evening_peak_price_eur_mwh"])
    top_evening_metric = float(feature_comparison.loc["top_20_revenue_days", "median_evening_peak_price_eur_mwh"])
    all_ramp_metric = float(feature_comparison.loc["all_days", "median_da_evening_minus_midday_ramp_eur_mwh"])
    top_ramp_metric = float(feature_comparison.loc["top_20_revenue_days", "median_da_evening_minus_midday_ramp_eur_mwh"])

    leader_style_all = {"color": "rgba(38, 70, 83, 0.45)", "width": 1.4, "dash": "dot"}
    leader_style_top = {"color": "rgba(231, 111, 81, 0.55)", "width": 1.4, "dash": "dot"}
    price_shape_figure.add_annotation(
        x=8.85,
        y=31.5,
        xref="x",
        yref="y",
        text=(
            "<b>Median trough within the 10-15 window</b>"
            f"<br>{top_trough_metric:.0f} €/MWh on top revenue days"
            f"<br>vs {all_trough_metric:.0f} €/MWh on an average day"
        ),
        showarrow=False,
        align="left",
        xanchor="left",
        yanchor="middle",
        bgcolor="rgba(255, 251, 245, 0.96)",
        bordercolor="rgba(20, 33, 61, 0.12)",
        borderwidth=1,
        borderpad=5,
        font={"size": 10.5},
    )

    all_arrow_x = 21.95
    top_arrow_x = 22.55

    price_shape_figure.add_shape(
        type="line",
        x0=all_trough_hour,
        y0=all_trough_line_value,
        x1=all_arrow_x,
        y1=all_trough_line_value,
        xref="x",
        yref="y",
        line=leader_style_all,
        layer="above",
    )
    price_shape_figure.add_shape(
        type="line",
        x0=all_evening_hour,
        y0=all_evening_line_value,
        x1=all_arrow_x,
        y1=all_evening_line_value,
        xref="x",
        yref="y",
        line=leader_style_all,
        layer="above",
    )
    price_shape_figure.add_shape(
        type="line",
        x0=top_trough_hour,
        y0=top_trough_line_value,
        x1=top_arrow_x,
        y1=top_trough_line_value,
        xref="x",
        yref="y",
        line=leader_style_top,
        layer="above",
    )
    price_shape_figure.add_shape(
        type="line",
        x0=top_evening_hour,
        y0=top_evening_line_value,
        x1=top_arrow_x,
        y1=top_evening_line_value,
        xref="x",
        yref="y",
        line=leader_style_top,
        layer="above",
    )
    price_shape_figure.add_annotation(
        x=all_arrow_x,
        y=all_evening_line_value,
        ax=all_arrow_x,
        ay=all_trough_line_value,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=2,
        arrowside="end+start",
        arrowsize=1,
        arrowwidth=1.8,
        arrowcolor="#264653",
    )
    price_shape_figure.add_annotation(
        x=top_arrow_x,
        y=top_evening_line_value,
        ax=top_arrow_x,
        ay=top_trough_line_value,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=2,
        arrowside="end+start",
        arrowsize=1,
        arrowwidth=1.8,
        arrowcolor="#e76f51",
    )
    price_shape_figure.add_annotation(
        x=all_arrow_x + 0.22,
        y=(all_trough_line_value + all_evening_line_value) / 2,
        xref="x",
        yref="y",
        text=f"<b>{all_ramp_metric:.0f} €/MWh</b>",
        showarrow=False,
        font={"size": 10.5, "color": "#264653"},
        bgcolor="rgba(255, 251, 245, 0.96)",
        bordercolor="rgba(20, 33, 61, 0.10)",
        borderwidth=1,
        borderpad=4,
    )
    price_shape_figure.add_annotation(
        x=top_arrow_x + 0.22,
        y=(top_trough_line_value + top_evening_line_value) / 2,
        xref="x",
        yref="y",
        text=f"<b>{top_ramp_metric:.0f} €/MWh</b>",
        showarrow=False,
        font={"size": 10.5, "color": "#e76f51"},
        bgcolor="rgba(255, 251, 245, 0.96)",
        bordercolor="rgba(20, 33, 61, 0.10)",
        borderwidth=1,
        borderpad=4,
    )
    price_shape_figure.update_xaxes(range=[0, 23.35])
    price_shape_figure.update_yaxes(range=[0, max(top_evening_metric, all_evening_metric, top_evening_line_value, all_evening_line_value) + 8])
    st.plotly_chart(price_shape_figure, width="stretch")
    render_footer_note(
        "Profiles shown as median hourly day-ahead prices for top-20 revenue days versus the full sample average day."
    )
    render_chart_caption(
        "Many of the best revenue days already showed a recognisable day-ahead profile before delivery: weaker midday prices, stronger evening prices, and a wider charging-to-discharging window."
    )
    render_takeaway("The best days are not perfectly predictable, but they are not invisible either.")

    st.subheader("A simple watchlist is already useful")
    st.markdown(
        "The practical question is not whether operators can predict every top-revenue day. It is whether the market can identify days that are more likely than normal to become commercially important."
    )
    st.markdown(
        "At the day-ahead horizon, the answer is clearly yes. A simple rule already proved useful: when the day-ahead spread was at least 200 €/MWh, the day was around four times more likely than normal to become a top revenue day. The rule is simple by design. It flags days when the day-ahead curve already shows an unusually wide charging-to-discharging window."
    )
    st.markdown(
        "Useful signal remained even two days earlier. When the market was already in a higher-spread regime — captured here by weekday and a recent three-day mean spread of at least 175 €/MWh — the odds of a top revenue day were a little more than twice the normal level. This is weaker than the day-ahead screen, but still useful as an early warning for maintenance timing."
    )
    st.markdown(
        "The main caveat is stability. Early-warning screens were less consistent across years and were stronger in the more stressed 2021–2022 period than in 2025. For owners, that suggests a practical split: use D-2 as an early-warning layer, but rely most heavily on D-1 for high-conviction readiness decisions."
    )
    render_takeaway("A useful D-2 early-warning screen exists, but the strongest screening power still appears at D-1.")
    render_chart_title("A useful early-warning screen exists at D-2, but the strongest screening power appears at D-1")
    watchlist_figure = build_early_warning_screen_figure(watchlist_screen_summary)
    st.plotly_chart(watchlist_figure, width="stretch")
    render_footer_note(
        f"Base rate: {pooled_base_rate_pct:.2f}% probability of a random day becoming a top-20 revenue day in pooled 2021–2025 data."
    )
    render_chart_caption(
        f"A useful early-warning signal exists before the day-ahead horizon, but screening power strengthens materially at D-1. In pooled 2021–2025 data, a simple D-2 screen still raised the odds of a top revenue day by {d2_watchlist['lift_x']:.2f}x versus the base rate, compared with {float(d1_watchlist['lift_x']):.2f}x for a simple D-1 day-ahead screen."
    )
    render_takeaway("Use D-2 for maintenance awareness and D-1 for readiness decisions.")

    st.subheader("The same cycles are worth more on the right days")
    st.markdown(
        "Flexibility mattered not only because it allowed more cycling, but because it allowed the same annual throughput to be used when opportunity was highest."
    )
    st.markdown(
        "To isolate that effect directly, each strict daily cap was compared against an annual allocator given exactly the same realised FEC that the strict policy actually used. This removes the benefit of extra throughput and isolates the value of reallocating cycles across days."
    )
    st.markdown(
        f"Even at the same realised throughput, flexibility added value. In {analysis_year}, reallocating the same cycles across days increased revenue by {equal_throughput_summary.loc['1.0/day vs reallocated same FEC', 'uplift_eur_per_mw'] / 1000:.1f}k €/MW (+{equal_throughput_summary.loc['1.0/day vs reallocated same FEC', 'uplift_pct_vs_strict']:.1f}%) in the 1.0/day case, {equal_throughput_summary.loc['1.5/day vs reallocated same FEC', 'uplift_eur_per_mw'] / 1000:.1f}k €/MW (+{equal_throughput_summary.loc['1.5/day vs reallocated same FEC', 'uplift_pct_vs_strict']:.1f}%) in the 1.5/day case, and {equal_throughput_summary.loc['2.0/day vs reallocated same FEC', 'uplift_eur_per_mw'] / 1000:.1f}k €/MW (+{equal_throughput_summary.loc['2.0/day vs reallocated same FEC', 'uplift_pct_vs_strict']:.1f}%) in the 2.0/day case."
    )
    st.markdown(
        "The mechanism is simple. A flat daily cap leaves money on the table because it forces the battery to stop too early on some of the year’s best days. The gain from flexibility does not come from cycling harder every day. It comes from using the same limited cycles less on weaker days and more on stronger ones."
    )
    render_chart_title("The same annual throughput earned more when it was concentrated into stronger days")
    same_cycles_figure = build_same_cycles_reallocation_figure(equal_throughput_summary)
    st.plotly_chart(same_cycles_figure, width="stretch")
    render_chart_caption(
        "Flexibility added value even without additional throughput, because the same annual cycles were worth more when they were spent on stronger days."
    )
    render_takeaway("Even with the same annual throughput, a battery earns more when cycles are concentrated into the days that matter most.")

    st.subheader("What This Changes for Owners")
    st.markdown(
        "For merchant BESS owners, the main commercial risk is not average underperformance across the year. It is being unavailable on a limited set of disproportionately valuable days."
    )
    st.markdown(
        "Because many of those days are already partly visible from the day-ahead curve, availability should be managed against expected opportunity rather than average conditions. Planned maintenance, operating readiness, and throughput headroom should all be treated as market-timed decisions."
    )
    render_closing(
        "For merchant BESS, flexibility is not only the ability to cycle. It is the ability to be available, ready, and unconstrained when the best days arrive."
    )
    render_takeaway("Availability, readiness, and flexibility matter most when they are timed.")
    if analysis_year in intraday_missing_years:
        render_footer_note(
            "Modeled as day-ahead only for the selected year because the intraday layer could not be loaded."
        )
    render_footer()


if __name__ == "__main__":
    main()
