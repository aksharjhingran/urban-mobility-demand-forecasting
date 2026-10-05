"""
╔══════════════════════════════════════════════════════════════════╗
║  URBAN MOBILITY DEMAND FORECASTING 2024 — STREAMLIT DASHBOARD    ║
║  Cross-platform (Windows / Mac / Linux)                          ║
╚══════════════════════════════════════════════════════════════════╝

USAGE
─────
  1. Make sure these files exist next to this script:
        data/processed_data.csv
        data/forecast_output.csv
        data/top_zones_forecast.csv
     (If missing, run main_pipeline.py first)

  2. Install dependencies:
        pip install streamlit pandas plotly numpy

  3. Launch the dashboard:
        streamlit run app.py

  The app opens automatically at http://localhost:8501
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"

PROCESSED_CSV = DATA_DIR / "processed_data.csv"
FORECAST_CSV  = DATA_DIR / "forecast_output.csv"
TOP_ZONES_CSV = DATA_DIR / "top_zones_forecast.csv"
MODEL_FILE    = SCRIPT_DIR / "models" / "random_forest.joblib"

# Color palette (dark theme)
ACCENT   = "#7c6fcd"
ACCENT2  = "#56cfb2"
ACCENT3  = "#f7c59f"
ACCENT4  = "#e84393"
ACCENT5  = "#60a5fa"
PALETTE  = [ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5, "#fb923c"]

DEMAND_COLORS = {
    "LOW"    : "#56cfb2",
    "MEDIUM" : "#f7c59f",
    "HIGH"   : "#e84393",
}

PLOTLY_TEMPLATE = "plotly_dark"
PLOT_BG  = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG (must be first Streamlit call)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Urban Mobility Forecast",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM CSS — dark theme with clean KPI cards
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
  .stApp { background: #0f1117; }
  section[data-testid="stSidebar"] { background: #1a1d2e; }
  h1, h2, h3, h4 { color: #e6e6f0 !important; letter-spacing: -0.01em; }
  .main-title {
      font-size: 36px; font-weight: 700;
      background: linear-gradient(90deg,#7c6fcd 0%, #56cfb2 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      margin-bottom: 4px;
  }
  .subtitle { color: #a0a0b0; font-size: 15px; margin-bottom: 24px; }
  .kpi-card {
      background: linear-gradient(145deg,#1a1d2e 0%,#22263d 100%);
      border: 1px solid #2d3250; border-radius: 14px;
      padding: 18px 20px; height: 100%;
      transition: transform .2s, border-color .2s;
  }
  .kpi-card:hover { transform: translateY(-2px); border-color: #7c6fcd; }
  .kpi-label { color: #a0a0b0; font-size: 12px; text-transform: uppercase;
      letter-spacing: 1.2px; margin-bottom: 6px; font-weight: 600; }
  .kpi-value { color: #ffffff; font-size: 30px; font-weight: 700; line-height: 1.1; }
  .kpi-delta { color: #56cfb2; font-size: 12px; margin-top: 4px; }
  .section-header {
      color: #e6e6f0; font-size: 18px; font-weight: 600;
      margin: 28px 0 12px 0; padding-left: 10px;
      border-left: 3px solid #7c6fcd;
  }
  div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING (cached)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(show_spinner=False)
def load_data():
    """Load all CSV data files with error handling."""
    missing = [p.name for p in (PROCESSED_CSV, FORECAST_CSV, TOP_ZONES_CSV)
               if not p.exists()]
    if missing:
        return None, None, None, missing

    df_main     = pd.read_csv(PROCESSED_CSV, parse_dates=["datetime"])
    df_forecast = pd.read_csv(FORECAST_CSV)
    df_top      = pd.read_csv(TOP_ZONES_CSV)

    # Add readable day name for display
    day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    df_main["day_name"] = df_main["day_of_week"].map(day_map)
    df_forecast["day_name"] = df_forecast["day_of_week"].map(day_map)
    return df_main, df_forecast, df_top, []


@st.cache_data(show_spinner=False)
def filter_data(df: pd.DataFrame, zones, days, traffic_levels):
    """Apply sidebar filters — cached for performance."""
    out = df.copy()
    if zones:          out = out[out["zone_id"].isin(zones)]
    if days:           out = out[out["day_name"].isin(days)]
    if traffic_levels: out = out[out["traffic_level"].isin(traffic_levels)]
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def style_fig(fig, height=380, show_legend=True):
    """Apply consistent dark styling to every Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="Inter, system-ui, sans-serif", color="#e6e6f0", size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        height=height,
        showlegend=show_legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2d3250"),
        xaxis=dict(gridcolor="#2d3250", linecolor="#2d3250"),
        yaxis=dict(gridcolor="#2d3250", linecolor="#2d3250"),
    )
    return fig


def kpi_card(col, label, value, delta=None):
    """Render a KPI card in the given column."""
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ''
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>',
                unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 1 — OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def page_overview(df, df_forecast, df_top):
    st.markdown('<div class="main-title">Overview Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">High-level KPIs and demand distribution across the urban network</div>',
                unsafe_allow_html=True)

    # KPI row
    total_pickups       = int(df["pickup_count"].sum())
    avg_demand          = df["demand_intensity"].mean()
    peak_pct            = df["peak_hour_flag"].mean() * 100
    high_demand_zones   = df[df["demand_label"] == "HIGH"]["zone_id"].nunique()
    total_records       = len(df)
    unique_zones        = df["zone_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Total Pickups",        f"{total_pickups:,}",
             delta=f"{unique_zones} zones")
    kpi_card(c2, "Avg Demand Intensity", f"{avg_demand:.1f}",
             delta=f"{total_records:,} records")
    kpi_card(c3, "Peak Hour Records",    f"{peak_pct:.1f}%",
             delta="7–9 AM & 5–7 PM")
    kpi_card(c4, "High Demand Zones",    f"{high_demand_zones}",
             delta="of 20 total")

    st.markdown("")

    # Two charts side by side
    left, right = st.columns([2, 1])

    # ── Demand over time ─────────────────────────────────────────
    with left:
        section("📈 Demand Trend Over Time")
        daily = (df.set_index("datetime")
                   .resample("D")["pickup_count"]
                   .sum().reset_index())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["datetime"], y=daily["pickup_count"],
            mode="lines", line=dict(color=ACCENT, width=2),
            fill="tozeroy",
            fillcolor="rgba(124,111,205,0.15)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Pickups: %{y:,}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="", yaxis_title="Daily Pickups")
        st.plotly_chart(style_fig(fig, height=380, show_legend=False),
                        use_container_width=True)

    # ── Demand label distribution (donut) ─────────────────────────
    with right:
        section("🎯 Demand Distribution")
        counts = df["demand_label"].value_counts().reindex(["LOW","MEDIUM","HIGH"])
        fig = go.Figure(data=[go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.6,
            marker=dict(colors=[DEMAND_COLORS[k] for k in counts.index],
                        line=dict(color="#0f1117", width=3)),
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>%{value:,} records<br>%{percent}<extra></extra>",
        )])
        fig.add_annotation(text=f"<b>{len(df):,}</b><br><span style='font-size:11px;color:#a0a0b0'>records</span>",
                            x=0.5, y=0.5, font_size=22, showarrow=False,
                            font_color="#ffffff")
        st.plotly_chart(style_fig(fig, height=380, show_legend=False),
                        use_container_width=True)

    # ── Zone demand snapshot ──────────────────────────────────────
    section("🏙️ Top 10 Zones by Average Demand")
    zone_avg = (df.groupby("zone_id")["demand_intensity"].mean()
                  .sort_values(ascending=False).head(10).reset_index())
    fig = go.Figure(go.Bar(
        x=zone_avg["demand_intensity"], y=zone_avg["zone_id"],
        orientation="h",
        marker=dict(color=zone_avg["demand_intensity"],
                    colorscale=[[0,"#56cfb2"],[0.5,"#7c6fcd"],[1,"#e84393"]],
                    line=dict(color="#0f1117", width=1)),
        text=zone_avg["demand_intensity"].round(1),
        textposition="outside", textfont=dict(color="#e6e6f0"),
        hovertemplate="<b>%{y}</b><br>Avg Demand: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Avg Demand Intensity", yaxis_title="",
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(style_fig(fig, height=420, show_legend=False),
                    use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 2 — DEMAND ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def page_analysis(df, df_forecast, df_top):
    st.markdown('<div class="main-title">Demand Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Deep-dive into temporal and contextual demand patterns</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # ── Hourly bar ────────────────────────────────────────────────
    with c1:
        section("⏱️ Demand by Hour of Day")
        hourly = df.groupby("hour_of_day")["pickup_count"].mean().reset_index()
        colors = [ACCENT4 if (7 <= h <= 9) or (17 <= h <= 19) else ACCENT
                   for h in hourly["hour_of_day"]]
        fig = go.Figure(go.Bar(
            x=hourly["hour_of_day"], y=hourly["pickup_count"],
            marker_color=colors,
            hovertemplate="<b>Hour %{x}</b><br>Avg Pickups: %{y:.1f}<extra></extra>",
        ))
        fig.add_vrect(x0=6.5, x1=9.5, fillcolor=ACCENT4, opacity=0.08,
                       line_width=0, annotation_text="Morning Rush",
                       annotation_position="top left",
                       annotation_font_color="#a0a0b0")
        fig.add_vrect(x0=16.5, x1=19.5, fillcolor=ACCENT4, opacity=0.08,
                       line_width=0, annotation_text="Evening Rush",
                       annotation_position="top left",
                       annotation_font_color="#a0a0b0")
        fig.update_layout(xaxis_title="Hour", yaxis_title="Avg Pickups")
        st.plotly_chart(style_fig(fig, height=380, show_legend=False),
                        use_container_width=True)

    # ── Day of week ───────────────────────────────────────────────
    with c2:
        section("📅 Demand by Day of Week")
        day_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        daily = df.groupby("day_name")["pickup_count"].mean().reindex(day_order).reset_index()
        day_colors = [ACCENT4 if d in ["Sat","Sun"] else ACCENT
                       for d in daily["day_name"]]
        fig = go.Figure(go.Bar(
            x=daily["day_name"], y=daily["pickup_count"],
            marker_color=day_colors,
            text=daily["pickup_count"].round(1), textposition="outside",
            textfont=dict(color="#e6e6f0"),
            hovertemplate="<b>%{x}</b><br>Avg Pickups: %{y:.1f}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="", yaxis_title="Avg Pickups")
        st.plotly_chart(style_fig(fig, height=380, show_legend=False),
                        use_container_width=True)

    c3, c4 = st.columns(2)

    # ── Traffic vs demand ─────────────────────────────────────────
    with c3:
        section("🚦 Demand by Traffic Level")
        traffic = df.groupby("traffic_level")["pickup_count"].mean()
        traffic = traffic.reindex(["low","medium","high"]).reset_index()
        fig = go.Figure(go.Bar(
            x=traffic["traffic_level"].str.title(),
            y=traffic["pickup_count"],
            marker_color=[ACCENT2, ACCENT3, ACCENT4],
            text=traffic["pickup_count"].round(1), textposition="outside",
            textfont=dict(color="#e6e6f0"),
            hovertemplate="<b>%{x} Traffic</b><br>Avg Pickups: %{y:.1f}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="", yaxis_title="Avg Pickups")
        st.plotly_chart(style_fig(fig, height=380, show_legend=False),
                        use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────
    with c4:
        section("🔗 Feature Correlation Heatmap")
        corr_cols = ["pickup_count","dropoff_count","temperature","rain",
                     "peak_hour_flag","weekend_flag","demand_intensity",
                     "traffic_encoded","event_flag"]
        corr = df[corr_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdYlGn", zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate="%{text}",
            textfont={"size":9,"color":"#0f1117"},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.2f}<extra></extra>",
            colorbar=dict(thickness=10, len=0.7),
        ))
        fig.update_layout(xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                           yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(style_fig(fig, height=380, show_legend=False),
                        use_container_width=True)

    # ── Heat calendar: hour × day ─────────────────────────────────
    section("🌡️ Demand Heatmap — Hour × Day")
    day_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    pivot = (df.pivot_table(values="pickup_count", index="day_name",
                              columns="hour_of_day", aggfunc="mean")
                .reindex(day_order))
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0,"#1a1d2e"],[0.3,"#7c6fcd"],[0.7,"#e84393"],[1,"#f7c59f"]],
        hovertemplate="<b>%{y}, Hour %{x}</b><br>Avg Pickups: %{z:.1f}<extra></extra>",
        colorbar=dict(thickness=10, len=0.7),
    ))
    fig.update_layout(xaxis_title="Hour of Day", yaxis_title="")
    st.plotly_chart(style_fig(fig, height=360, show_legend=False),
                    use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 3 — FORECAST INSIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def page_forecast(df, df_forecast, df_top):
    st.markdown('<div class="main-title">Scenario Forecast Insights</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">24-hour demand scenario forecast with confidence scores across all zones</div>',
                unsafe_allow_html=True)

    # KPIs
    total_forecasts = len(df_forecast)
    high_count      = (df_forecast["predicted_demand"] == "HIGH").sum()
    avg_conf        = df_forecast["demand_proba"].mean() * 100
    zones_covered   = df_forecast["zone_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Scenario Records", f"{total_forecasts:,}",
             delta=f"{zones_covered} zones × 24h")
    kpi_card(c2, "High-Demand Hours", f"{high_count:,}",
             delta=f"{high_count/total_forecasts*100:.1f}% of scenario records")
    kpi_card(c3, "Avg Confidence",    f"{avg_conf:.1f}%",
             delta="Model certainty")
    kpi_card(c4, "Top Zone",
             df_top.iloc[0]["zone_id"] if len(df_top) > 0 else "—",
             delta=f"{int(df_top.iloc[0]['high_demand_hours'])} hrs" if len(df_top) > 0 else "")

    st.markdown("")

    left, right = st.columns([1, 1])

    # ── Top zones table ───────────────────────────────────────────
    with left:
        section("🏆 Top High-Demand Zones (Ranked)")
        if len(df_top) == 0:
            st.info("No high-demand zones predicted in current scenario forecast.")
        else:
            top_display = df_top.copy()
            top_display.insert(0, "Rank", range(1, len(top_display) + 1))
            top_display["avg_confidence"] = (top_display["avg_confidence"] * 100).round(1)
            top_display.columns = ["Rank","Zone","High-Demand Hours","Confidence %"]
            st.dataframe(top_display, use_container_width=True, hide_index=True,
                          height=380)

    # ── Bar chart: zones by high-demand hours ─────────────────────
    with right:
        section("📊 Zones Ranked by High-Demand Hours")
        if len(df_top) > 0:
            top10 = df_top.head(10)
            fig = go.Figure(go.Bar(
                x=top10["high_demand_hours"], y=top10["zone_id"],
                orientation="h",
                marker=dict(color=top10["avg_confidence"],
                            colorscale=[[0,"#56cfb2"],[0.5,"#7c6fcd"],[1,"#e84393"]],
                            colorbar=dict(title="Confidence", thickness=8, len=0.7)),
                text=top10["high_demand_hours"], textposition="outside",
                textfont=dict(color="#e6e6f0"),
                hovertemplate="<b>%{y}</b><br>Hours: %{x}<br>Confidence: %{marker.color:.2%}<extra></extra>",
            ))
            fig.update_layout(xaxis_title="High-Demand Hours", yaxis_title="",
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, height=380, show_legend=False),
                            use_container_width=True)
        else:
            st.info("No data to display.")

    # ── 24-hour scenario forecast for sample zones ────────────────
    section("🕐 24-Hour Demand Prediction Trajectory")
    demand_map = {"LOW":1,"MEDIUM":2,"HIGH":3}
    df_fc = df_forecast.copy()
    df_fc["demand_num"] = df_fc["predicted_demand"].map(demand_map)

    sample_zones = (df_top["zone_id"].head(5).tolist()
                     if len(df_top) > 0
                     else sorted(df_fc["zone_id"].unique())[:5])

    fig = go.Figure()
    for i, zone in enumerate(sample_zones):
        zd = df_fc[df_fc["zone_id"] == zone].sort_values("hour_of_day")
        fig.add_trace(go.Scatter(
            x=zd["hour_of_day"], y=zd["demand_num"],
            mode="lines+markers", name=zone,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
            marker=dict(size=8),
            hovertemplate=f"<b>{zone}</b><br>Hour %{{x}}<br>Demand: %{{text}}<extra></extra>",
            text=zd["predicted_demand"],
        ))
    fig.update_layout(
        xaxis_title="Hour of Day", yaxis_title="Predicted Demand Level",
        yaxis=dict(tickmode="array", tickvals=[1,2,3],
                    ticktext=["LOW","MEDIUM","HIGH"]),
    )
    st.plotly_chart(style_fig(fig, height=400),
                    use_container_width=True)

    # ── Confidence distribution ───────────────────────────────────
    section("📉 Forecast Confidence Distribution")
    fig = go.Figure()
    for label, color in DEMAND_COLORS.items():
        data = df_forecast[df_forecast["predicted_demand"] == label]["demand_proba"]
        if len(data) > 0:
            fig.add_trace(go.Histogram(
                x=data, name=label, marker_color=color, opacity=0.75,
                xbins=dict(start=0, end=1, size=0.05),
                hovertemplate="<b>"+label+"</b><br>Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>",
            ))
    fig.update_layout(barmode="overlay",
                       xaxis_title="Prediction Confidence (probability)",
                       yaxis_title="Frequency")
    st.plotly_chart(style_fig(fig, height=340), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 4 — ZONE EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def page_zone_explorer(df, df_forecast, df_top):
    st.markdown('<div class="main-title">Zone Explorer</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Drill into a single zone and see its full demand story</div>',
                unsafe_allow_html=True)

    zones_list = sorted(df["zone_id"].unique().tolist())
    selected = st.selectbox("🔍 Select a Zone", zones_list, index=0,
                             key="zone_explorer_select")

    zd      = df[df["zone_id"] == selected].copy()
    zfc     = df_forecast[df_forecast["zone_id"] == selected].copy()

    if zd.empty:
        st.warning(f"No data for {selected}.")
        return

    # Zone KPIs
    total_pickups = int(zd["pickup_count"].sum())
    avg_demand    = zd["demand_intensity"].mean()
    peak_share    = zd["peak_hour_flag"].mean() * 100
    high_share    = (zd["demand_label"] == "HIGH").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, f"{selected} Pickups", f"{total_pickups:,}",
             delta=f"{len(zd):,} records")
    kpi_card(c2, "Avg Demand",          f"{avg_demand:.1f}")
    kpi_card(c3, "Peak-Hour Share",     f"{peak_share:.1f}%")
    kpi_card(c4, "High Demand Share",   f"{high_share:.1f}%")

    st.markdown("")

    # ── Historical demand trend for this zone ─────────────────────
    section(f"📈 Historical Demand Trend — {selected}")
    daily = (zd.set_index("datetime")
                .resample("D")["pickup_count"]
                .sum().reset_index())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["datetime"], y=daily["pickup_count"],
        mode="lines", line=dict(color=ACCENT2, width=2),
        fill="tozeroy", fillcolor="rgba(86,207,178,0.15)",
        hovertemplate="<b>%{x|%b %d}</b><br>Pickups: %{y:,}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="", yaxis_title="Daily Pickups")
    st.plotly_chart(style_fig(fig, height=320, show_legend=False),
                    use_container_width=True)

    # ── Hourly pattern & Forecast side by side ────────────────────
    left, right = st.columns(2)

    with left:
        section(f"⏰ Hourly Pattern — {selected}")
        hourly = zd.groupby("hour_of_day")["pickup_count"].mean().reset_index()
        colors = [ACCENT4 if (7 <= h <= 9) or (17 <= h <= 19) else ACCENT
                   for h in hourly["hour_of_day"]]
        fig = go.Figure(go.Bar(
            x=hourly["hour_of_day"], y=hourly["pickup_count"],
            marker_color=colors,
            hovertemplate="<b>Hour %{x}</b><br>Avg Pickups: %{y:.1f}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Hour", yaxis_title="Avg Pickups")
        st.plotly_chart(style_fig(fig, height=340, show_legend=False),
                        use_container_width=True)

    with right:
        section(f"🔮 24-Hour Scenario — {selected}")
        if zfc.empty:
            st.info(f"No scenario forecast data available for {selected}.")
        else:
            demand_map = {"LOW":1,"MEDIUM":2,"HIGH":3}
            zfc_sorted = zfc.sort_values("hour_of_day").copy()
            zfc_sorted["demand_num"] = zfc_sorted["predicted_demand"].map(demand_map)

            fig = go.Figure()
            # Confidence area
            fig.add_trace(go.Bar(
                x=zfc_sorted["hour_of_day"],
                y=zfc_sorted["demand_proba"] * 3,  # scale for secondary visual
                marker_color="rgba(124,111,205,0.2)",
                name="Confidence", yaxis="y2",
                hovertemplate="<b>Hour %{x}</b><br>Confidence: %{customdata:.1%}<extra></extra>",
                customdata=zfc_sorted["demand_proba"],
            ))
            # Predicted demand line
            fig.add_trace(go.Scatter(
                x=zfc_sorted["hour_of_day"], y=zfc_sorted["demand_num"],
                mode="lines+markers", name="Demand",
                line=dict(color=ACCENT4, width=3),
                marker=dict(size=10,
                            color=[DEMAND_COLORS[d] for d in zfc_sorted["predicted_demand"]],
                            line=dict(color="#0f1117", width=2)),
                hovertemplate="<b>Hour %{x}</b><br>%{text}<extra></extra>",
                text=zfc_sorted["predicted_demand"],
            ))
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis=dict(title="Predicted Demand",
                            tickmode="array", tickvals=[1,2,3],
                            ticktext=["LOW","MEDIUM","HIGH"], range=[0.5,3.5]),
                yaxis2=dict(overlaying="y", side="right", showgrid=False,
                             showticklabels=False, range=[0, 3.5]),
            )
            st.plotly_chart(style_fig(fig, height=340, show_legend=False),
                            use_container_width=True)

    # ── Weekday vs Weekend comparison ─────────────────────────────
    section(f"📊 Weekday vs Weekend — {selected}")
    wk_data = zd.groupby("weekend_flag")["pickup_count"].agg(["mean","sum","count"]).reset_index()
    wk_data["label"] = wk_data["weekend_flag"].map({0:"Weekday",1:"Weekend"})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=wk_data["label"], y=wk_data["mean"],
        marker_color=[ACCENT, ACCENT4],
        text=wk_data["mean"].round(1), textposition="outside",
        textfont=dict(color="#e6e6f0"),
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f}<br>Total: %{customdata:,}<extra></extra>",
        customdata=wk_data["sum"].astype(int),
    ))
    fig.update_layout(xaxis_title="", yaxis_title="Avg Pickups")
    st.plotly_chart(style_fig(fig, height=320, show_legend=False),
                    use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 5 — MODEL PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(show_spinner="Training Random Forest for evaluation...")
def train_and_evaluate(df: pd.DataFrame):
    """Evaluate the saved RF model on the processed data."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix)

    FEATURES = [
        "hour_of_day", "day_of_week", "month", "quarter",
        "temperature", "rain", "event_flag",
        "peak_hour_flag", "weekend_flag",
        "zone_encoded", "traffic_encoded",
    ]
    X = df[FEATURES].copy()
    y = df["demand_label"].copy()

    # Drop rows with any NaN in features/target
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Saved model not found at {MODEL_FILE}. Run python main_pipeline.py first."
        )
    model = joblib.load(MODEL_FILE)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall"   : recall_score(y_test, y_pred, average="weighted"),
        "f1"       : f1_score(y_test, y_pred, average="weighted"),
        "train_n"  : len(X_train),
        "test_n"   : len(X_test),
    }
    cm = confusion_matrix(y_test, y_pred, labels=["LOW","MEDIUM","HIGH"])
    importances = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    return metrics, cm, importances


def page_model(df, df_forecast, df_top):
    st.markdown('<div class="main-title">Model Performance</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Random Forest classifier — evaluation metrics, confusion matrix, and feature importance</div>',
                unsafe_allow_html=True)

    metrics, cm, importances = train_and_evaluate(df)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Accuracy",  f"{metrics['accuracy']*100:.1f}%",
             delta=f"{metrics['test_n']:,} test records")
    kpi_card(c2, "Precision", f"{metrics['precision']:.3f}",
             delta="Weighted avg")
    kpi_card(c3, "Recall",    f"{metrics['recall']:.3f}",
             delta="Weighted avg")
    kpi_card(c4, "F1 Score",  f"{metrics['f1']:.3f}",
             delta=f"{metrics['train_n']:,} training rows")

    st.markdown("")

    left, right = st.columns(2)

    # ── Confusion Matrix ─────────────────────────────────────────
    with left:
        section("🎯 Confusion Matrix")
        labels = ["LOW", "MEDIUM", "HIGH"]
        # Percentage annotations
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        text = [[f"<b>{cm[i][j]}</b><br>{cm_pct[i][j]:.1f}%"
                 for j in range(3)] for i in range(3)]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0,"#1a1d2e"],[0.5,"#7c6fcd"],[1,"#e84393"]],
            text=text, texttemplate="%{text}",
            textfont={"size":14, "color":"#ffffff"},
            hovertemplate="<b>True: %{y}</b><br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            colorbar=dict(thickness=10, len=0.7),
        ))
        fig.update_layout(
            xaxis=dict(title="Predicted Label", side="bottom"),
            yaxis=dict(title="True Label", autorange="reversed"),
        )
        st.plotly_chart(style_fig(fig, height=420, show_legend=False),
                        use_container_width=True)

    # ── Feature Importance ───────────────────────────────────────
    with right:
        section("🏅 Feature Importance")
        fig = go.Figure(go.Bar(
            x=importances["importance"], y=importances["feature"],
            orientation="h",
            marker=dict(color=importances["importance"],
                        colorscale=[[0,"#56cfb2"],[0.5,"#7c6fcd"],[1,"#e84393"]],
                        line=dict(color="#0f1117", width=1)),
            text=importances["importance"].round(3),
            textposition="outside", textfont=dict(color="#e6e6f0"),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Importance Score", yaxis_title="")
        st.plotly_chart(style_fig(fig, height=420, show_legend=False),
                        use_container_width=True)

    # ── Per-class breakdown ──────────────────────────────────────
    section("📋 Per-Class Performance")
    per_class = []
    labels = ["LOW", "MEDIUM", "HIGH"]
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support   = cm[i, :].sum()
        per_class.append({
            "Class": label,
            "Precision": f"{precision:.3f}",
            "Recall": f"{recall:.3f}",
            "F1-Score": f"{f1:.3f}",
            "Support": int(support),
        })
    st.dataframe(pd.DataFrame(per_class), use_container_width=True,
                  hide_index=True)

    # Model summary box
    st.markdown(f"""
    <div style="background:linear-gradient(145deg,#1a1d2e 0%,#22263d 100%);
                border:1px solid #2d3250;border-radius:12px;padding:16px 20px;margin-top:16px">
      <div style="color:#a0a0b0;font-size:11px;text-transform:uppercase;
                  letter-spacing:1.2px;margin-bottom:8px;font-weight:600">
        Model Configuration
      </div>
      <div style="color:#e6e6f0;font-size:14px;line-height:1.8">
        <b>Algorithm:</b> Random Forest Classifier<br>
        <b>Trees:</b> 100 &nbsp;|&nbsp; <b>Max Depth:</b> 10 &nbsp;|&nbsp;
        <b>Min Samples Split:</b> 2<br>
        <b>Class Weight:</b> Balanced &nbsp;|&nbsp; <b>Stratified Split:</b> 80/20
      </div>
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    # Load data
    df_main, df_forecast, df_top, missing = load_data()

    if missing:
        st.error("🚫 Missing required data files")
        st.markdown(f"""
        The following files were not found in `{DATA_DIR}`:

        {''.join(f'- `{m}`\n' for m in missing)}

        **Fix:** Run `python main_pipeline.py` first to generate the data files,
        then restart this app.
        """)
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🚇 Urban Mobility")
        st.caption("Demand Forecasting 2024")
        st.markdown("---")

        page = st.radio("**Navigate**", [
            "📊 Overview",
            "🔍 Demand Analysis",
            "🔮 Scenario Forecast",
            "🏙️ Zone Explorer",
            "🤖 Model Performance",
        ], label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### 🎛️ Filters")
        st.caption("Applied to Overview & Analysis pages")

        zones_all   = sorted(df_main["zone_id"].unique().tolist())
        days_all    = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        traffic_all = sorted(df_main["traffic_level"].dropna().unique().tolist())

        sel_zones = st.multiselect("🏙️ Zones", zones_all,
                                     default=[], placeholder="All zones")
        sel_days  = st.multiselect("📅 Day of Week", days_all,
                                     default=[], placeholder="All days")
        sel_traffic = st.multiselect("🚦 Traffic Level", traffic_all,
                                       default=[], placeholder="All levels")

        st.markdown("---")
        st.markdown("##### 📈 Quick Stats")
        st.caption(f"**{len(df_main):,}** records loaded")
        st.caption(f"**{df_main['zone_id'].nunique()}** zones")
        st.caption(f"**{len(df_forecast):,}** scenario records")

        st.markdown("---")
        st.markdown("##### 📥 Download Data")
        st.caption("Filtered data reflects active filters")

    # Apply filters
    df_filtered = filter_data(df_main, sel_zones, sel_days, sel_traffic)

    # Guard against empty filter result
    if df_filtered.empty:
        st.warning("⚠️ No records match the selected filters. Please adjust your selections.")
        st.stop()

    # ── Download buttons (rendered into sidebar) ─────────────────
    with st.sidebar:
        # Cached CSV conversion so re-renders don't re-encode
        @st.cache_data
        def _to_csv_bytes(df):
            return df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📄 Filtered Data",
            data=_to_csv_bytes(df_filtered),
            file_name="filtered_mobility_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "🔮 Scenario Output",
            data=_to_csv_bytes(df_forecast),
            file_name="forecast_output.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "🏆 Top Zones",
            data=_to_csv_bytes(df_top),
            file_name="top_zones_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("---")
        st.caption("💡 Tip: Leave filters empty to include all data")

    # Route to page
    if page == "📊 Overview":
        page_overview(df_filtered, df_forecast, df_top)
    elif page == "🔍 Demand Analysis":
        page_analysis(df_filtered, df_forecast, df_top)
    elif page == "🔮 Scenario Forecast":
        page_forecast(df_main, df_forecast, df_top)  # forecast page uses full data
    elif page == "🏙️ Zone Explorer":
        page_zone_explorer(df_main, df_forecast, df_top)
    elif page == "🤖 Model Performance":
        page_model(df_main, df_forecast, df_top)     # model trains on full data

    # Footer
    st.markdown("---")
    st.caption("🚇 Urban Mobility Demand Forecasting · Built with Streamlit + Plotly · Random Forest")


if __name__ == "__main__":
    main()
