import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
FRED_API_KEY = "e335584b268ae95e5d8725bf8c5f853d"
START_DATE   = "1990-01-01"
CYCLE_COLORS = ["#ff7b72", "#d2a8ff", "#79c0ff"]  # red / purple / blue

DARK_LAYOUT = dict(
    plot_bgcolor  = "#161b22",
    paper_bgcolor = "#0d1117",
    font = dict(color="#c9d1d9", family="Inter, Arial, sans-serif", size=13),
)
AXIS_STYLE = dict(gridcolor="#21262d", zerolinecolor="#30363d", showgrid=True)

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fed Surprise Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #e6edf3; }
  .block-container { padding-top: 1.5rem; }
  .stTabs [data-baseweb="tab-list"] {
    background-color: #161b22; border-radius: 8px; padding: 4px; gap: 8px;
  }
  .stTabs [data-baseweb="tab"]       { color: #8b949e; font-size: 1rem; }
  .stTabs [aria-selected="true"]     {
    background-color: #1f6feb; color: #ffffff; border-radius: 6px;
  }
  [data-testid="stDataFrame"] { background: #161b22; }
</style>
""", unsafe_allow_html=True)

# ── FRED client (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def get_fred():
    return Fred(api_key=FRED_API_KEY)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch(series_id, start=START_DATE):
    """Fetch a FRED series, resample to monthly, forward-fill gaps."""
    try:
        raw = get_fred().get_series(series_id, observation_start=start)
        return raw.resample("MS").mean().ffill().dropna()
    except Exception as exc:
        st.warning(f"Could not load **{series_id}**: {exc}")
        return pd.Series(dtype=float, name=series_id)

# ── Rate-cut cycle detection ──────────────────────────────────────────────────
def find_cut_cycles(fedfunds, n=3):
    """Return the n most recent rate-cut cycle start dates."""
    delta = fedfunds.diff().dropna()
    cut_months = delta[delta < -0.05].index
    if len(cut_months) == 0:
        return []
    cycles, prev = [], None
    for dt in cut_months:
        if prev is None or (dt - prev).days > 180:
            cycles.append(dt)
        prev = dt
    return cycles[-n:]

# ── Chart helpers ─────────────────────────────────────────────────────────────
def add_cut_overlays(fig, cut_dates, window_months=12,
                     row=None, col=None, annotate=True):
    """Shade post-cut windows and draw dashed cycle-start lines."""
    rc = dict(row=row, col=col) if row is not None else {}
    for i, dt in enumerate(cut_dates):
        color   = CYCLE_COLORS[i % len(CYCLE_COLORS)]
        end_dt  = dt + pd.DateOffset(months=window_months)
        x0_str  = dt.strftime("%Y-%m-%d")
        x1_str  = end_dt.strftime("%Y-%m-%d")

        fig.add_vrect(
            x0=x0_str, x1=x1_str,
            fillcolor=color, opacity=0.08,
            layer="below", line_width=0,
            **rc,
        )
        vl_kwargs = dict(
            x=x0_str,
            line=dict(color=color, width=1.5, dash="dash"),
        )
        if annotate:
            vl_kwargs.update(
                annotation_text=f"Cut {i+1} ({dt.strftime('%b %Y')})",
                annotation_font_color=color,
                annotation_font_size=10,
                annotation_position="top right",
            )
        fig.add_vline(**vl_kwargs, **rc)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Fed Surprise Dashboard — Rate Cut Analysis")
st.caption(
    f"50 bps surprise cut  ·  {datetime.now().strftime('%B %d, %Y  %H:%M')}  ·  "
    "Source: FRED, Federal Reserve Bank of St. Louis  ·  Internal use only"
)
st.divider()

# ── Load shared FEDFUNDS data ─────────────────────────────────────────────────
with st.spinner("Connecting to FRED…"):
    fedfunds  = fetch("FEDFUNDS")
    cut_dates = find_cut_cycles(fedfunds)

if not cut_dates:
    st.error("Could not detect rate-cut cycles in FEDFUNDS data.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Lending Impact",
    "⚠️   Credit Risk",
    "💳  Payment Volumes",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Lending Impact
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Should we reprice our loan products?")
    st.markdown(
        "How quickly do mortgage and prime rates follow Fed rate cuts? "
        "**Shaded windows** = 12 months after each of the 3 most recent cycle starts."
    )

    with st.spinner("Loading lending rates…"):
        mortgage = fetch("MORTGAGE30US")
        dprime   = fetch("DPRIME")

    fig1 = go.Figure()
    for label, series, color in [
        ("Fed Funds Rate",                          fedfunds, "#58a6ff"),
        ("30-Yr Fixed Mortgage (MORTGAGE30US)",     mortgage, "#3fb950"),
        ("Prime Rate / Business Loans (DPRIME)",    dprime,   "#d2a8ff"),
    ]:
        if not series.empty:
            fig1.add_trace(go.Scatter(
                x=series.index, y=series.values,
                name=label, line=dict(color=color, width=2),
                hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>" + label + "</extra>",
            ))

    add_cut_overlays(fig1, cut_dates, window_months=12)
    fig1.update_layout(
        **DARK_LAYOUT, height=500, hovermode="x unified",
        xaxis=dict(title="", **AXIS_STYLE),
        yaxis=dict(title="Rate (%)", **AXIS_STYLE),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Pass-through lag table
    lag_rows = []
    for i, dt in enumerate(cut_dates):
        cycle_label = f"Cycle {i+1}  ({dt.strftime('%b %Y')})"
        for sname, series in [
            ("Mortgage (MORTGAGE30US)", mortgage),
            ("Prime (DPRIME)",          dprime),
        ]:
            if series.empty:
                continue
            base = series.asof(dt)
            post = series[series.index >= dt].head(24)
            if pd.isna(base):
                lag = "n/a"
            else:
                idx = next(
                    (j for j, v in enumerate(post.values)
                     if not pd.isna(v) and (v - base) <= -0.25),
                    None,
                )
                lag = f"{idx} mo" if idx is not None else ">24 mo"
            lag_rows.append({"Cycle": cycle_label, "Series": sname, "Lag": lag})

    if lag_rows:
        st.markdown("##### Rate pass-through lag — months until first ≥25 bps decline")
        lag_df = pd.DataFrame(lag_rows).pivot(
            index="Cycle", columns="Series", values="Lag"
        )
        st.dataframe(lag_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Credit Risk
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Are our credit models still valid?")
    st.markdown(
        "Delinquency rate trajectory across **6 quarters (18 months)** following "
        "each rate-cut cycle start. Fed funds overlaid for context."
    )

    with st.spinner("Loading delinquency data…"):
        cc_delinq  = fetch("DRCCLACBS")   # credit card
        con_delinq = fetch("DRCLACBS")    # consumer loans
        mtg_delinq = fetch("DRSFRMACBS")  # single-family residential mortgage

    fig2 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Delinquency Rates (%)", "Fed Funds Rate (%)"),
        row_heights=[0.65, 0.35],
        vertical_spacing=0.1,
    )

    for label, series, color in [
        ("Credit Card (DRCCLACBS)",    cc_delinq,  "#ff7b72"),
        ("Consumer Loans (DRCLACBS)",  con_delinq, "#ffa657"),
        ("SF Mortgage (DRSFRMACBS)",   mtg_delinq, "#3fb950"),
    ]:
        if not series.empty:
            fig2.add_trace(go.Scatter(
                x=series.index, y=series.values,
                name=label, line=dict(color=color, width=2),
                hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>" + label + "</extra>",
            ), row=1, col=1)

    if not fedfunds.empty:
        fig2.add_trace(go.Scatter(
            x=fedfunds.index, y=fedfunds.values,
            name="Fed Funds Rate", line=dict(color="#58a6ff", width=2),
            hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>Fed Funds</extra>",
        ), row=2, col=1)

    # Shade both panels; annotate only on top panel
    add_cut_overlays(fig2, cut_dates, window_months=18, row=1, col=1, annotate=True)
    add_cut_overlays(fig2, cut_dates, window_months=18, row=2, col=1, annotate=False)

    fig2.update_layout(
        **DARK_LAYOUT, height=600, hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.08),
    )
    for r in [1, 2]:
        fig2.update_xaxes(**AXIS_STYLE, row=r, col=1)
        fig2.update_yaxes(**AXIS_STYLE, row=r, col=1)
    fig2.update_yaxes(title_text="Delinquency Rate (%)", row=1, col=1)
    fig2.update_yaxes(title_text="Fed Funds (%)",        row=2, col=1)

    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Payment Volumes
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Should we expect payment volume to shift?")
    st.markdown(
        "Consumer spending trends in the **12 months following** each rate-cut cycle. "
        "RSXFS on left axis; PCE on right."
    )

    with st.spinner("Loading spending data…"):
        rsxfs = fetch("RSXFS")   # retail sales excl. food services
        pce   = fetch("PCE")     # personal consumption expenditures

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

    if not rsxfs.empty:
        fig3.add_trace(go.Scatter(
            x=rsxfs.index, y=rsxfs.values,
            name="Retail Sales ex. Food (RSXFS, $M)",
            line=dict(color="#58a6ff", width=2),
            hovertemplate="%{x|%b %Y}: $%{y:,.0f}M<extra>RSXFS</extra>",
        ), secondary_y=False)

    if not pce.empty:
        fig3.add_trace(go.Scatter(
            x=pce.index, y=pce.values,
            name="Personal Consumption (PCE, $B)",
            line=dict(color="#d2a8ff", width=2),
            hovertemplate="%{x|%b %Y}: $%{y:,.1f}B<extra>PCE</extra>",
        ), secondary_y=True)

    add_cut_overlays(fig3, cut_dates, window_months=12)

    fig3.update_layout(
        **DARK_LAYOUT, height=500, hovermode="x unified",
        xaxis=dict(**AXIS_STYLE),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18),
    )
    fig3.update_yaxes(title_text="RSXFS ($ Millions)", **AXIS_STYLE, secondary_y=False)
    fig3.update_yaxes(title_text="PCE ($ Billions)",   **AXIS_STYLE, secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)

    # YoY growth comparison table
    growth_rows = []
    for i, dt in enumerate(cut_dates):
        cycle_label = f"Cycle {i+1}  ({dt.strftime('%b %Y')})"
        for sname, series in [("RSXFS", rsxfs), ("PCE", pce)]:
            if series.empty:
                continue
            post = series[
                (series.index > dt) &
                (series.index <= dt + pd.DateOffset(months=12))
            ]
            pre = series[
                (series.index > dt - pd.DateOffset(months=12)) &
                (series.index <= dt)
            ]
            if not post.empty and not pre.empty and pre.mean() != 0:
                pct = (post.mean() - pre.mean()) / pre.mean() * 100
                growth_rows.append({
                    "Cycle":  cycle_label,
                    "Series": sname,
                    "Avg Change vs Prior 12mo": f"{pct:+.1f}%",
                })

    if growth_rows:
        st.markdown("##### Average spending — 12 months post-cut vs. 12 months pre-cut")
        gdf = pd.DataFrame(growth_rows).pivot(
            index="Cycle", columns="Series", values="Avg Change vs Prior 12mo"
        )
        st.dataframe(gdf, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with FRED API · Streamlit · Plotly  ·  FinTechCo internal use only")
