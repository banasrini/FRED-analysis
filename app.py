import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
FRED_API_KEY      = os.environ.get("FRED_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
START_DATE        = "1990-01-01"
CYCLE_COLORS      = ["#ff7b72", "#d2a8ff", "#79c0ff"]  # red / purple / blue

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
        return pd.Series(dtype=float, name=series_id, index=pd.DatetimeIndex([]))

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
        fig.add_vline(
            x=x0_str,
            line=dict(color=color, width=1.5, dash="dash"),
            **rc,
        )
        if annotate:
            xref = "x" if row is None else f"x{'' if row == 1 else row}"
            fig.add_annotation(
                x=x0_str,
                y=1.0,
                xref=xref,
                yref="paper",
                text=f"Cut {i+1} ({dt.strftime('%b %Y')})",
                font=dict(color=color, size=10),
                showarrow=False,
                xanchor="left",
                yanchor="top",
            )

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊  Lending Impact",
    "⚠️   Credit Risk",
    "💳  Payment Volumes",
    "📈  Yield Curve",
    "🔥  Credit Spreads",
    "🛡️  Portfolio Hedges",
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Yield Curve
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("How does the yield curve shift around Fed rate cuts?")
    st.markdown(
        "Treasury yields across maturities and the **2s10s spread** — "
        "a key indicator of curve shape (inversion = negative). "
        "Shaded windows = 12 months after each cycle start."
    )

    with st.spinner("Loading Treasury yield data…"):
        dgs2  = fetch("DGS2")   # 2-Year Treasury
        dgs10 = fetch("DGS10")  # 10-Year Treasury
        dgs30 = fetch("DGS30")  # 30-Year Treasury

    # 2s10s spread
    spread_2s10s = (dgs10 - dgs2).dropna()

    fig4 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Treasury Yields (%)", "2s10s Spread (bps)"),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
    )

    for label, series, color in [
        ("2-Year (DGS2)",   dgs2,  "#58a6ff"),
        ("10-Year (DGS10)", dgs10, "#3fb950"),
        ("30-Year (DGS30)", dgs30, "#d2a8ff"),
    ]:
        if not series.empty:
            fig4.add_trace(go.Scatter(
                x=series.index, y=series.values,
                name=label, line=dict(color=color, width=2),
                hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>" + label + "</extra>",
            ), row=1, col=1)

    if not spread_2s10s.empty:
        fig4.add_trace(go.Scatter(
            x=spread_2s10s.index, y=spread_2s10s.values,
            name="2s10s Spread", line=dict(color="#ffa657", width=2),
            fill="tozeroy", fillcolor="rgba(255,166,87,0.08)",
            hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>2s10s</extra>",
        ), row=2, col=1)
        # Zero line to highlight inversion
        fig4.add_hline(y=0, line=dict(color="#ff7b72", width=1, dash="dot"), row=2, col=1)

    add_cut_overlays(fig4, cut_dates, window_months=12, row=1, col=1, annotate=True)
    add_cut_overlays(fig4, cut_dates, window_months=12, row=2, col=1, annotate=False)

    fig4.update_layout(
        **DARK_LAYOUT, height=620, hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.08),
    )
    for r in [1, 2]:
        fig4.update_xaxes(**AXIS_STYLE, row=r, col=1)
        fig4.update_yaxes(**AXIS_STYLE, row=r, col=1)
    fig4.update_yaxes(title_text="Yield (%)",       row=1, col=1)
    fig4.update_yaxes(title_text="Spread (%)",      row=2, col=1)
    st.plotly_chart(fig4, use_container_width=True)

    # Curve snapshot table: yield levels at cycle start vs. 12mo later
    snap_rows = []
    for i, dt in enumerate(cut_dates):
        cycle_label = f"Cycle {i+1}  ({dt.strftime('%b %Y')})"
        end_dt = dt + pd.DateOffset(months=12)
        for sname, series in [("2Y", dgs2), ("10Y", dgs10), ("30Y", dgs30), ("2s10s", spread_2s10s)]:
            if series.empty:
                continue
            at_cut  = series.asof(dt)
            at_12mo = series.asof(end_dt)
            if not pd.isna(at_cut) and not pd.isna(at_12mo):
                snap_rows.append({
                    "Cycle": cycle_label,
                    "Series": sname,
                    "At Cut Start": f"{at_cut:.2f}%",
                    "12mo Later": f"{at_12mo:.2f}%",
                    "Change": f"{at_12mo - at_cut:+.2f}%",
                })

    if snap_rows:
        st.markdown("##### Yield levels at cycle start vs. 12 months later")
        snap_df = pd.DataFrame(snap_rows)
        st.dataframe(snap_df.set_index(["Cycle", "Series"]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Credit Spreads & Risk Appetite
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("How does credit risk appetite respond to Fed cuts?")
    st.markdown(
        "High-yield and investment-grade OAS spreads signal risk appetite. "
        "**Spread compression** post-cut = risk-on. VIX overlaid as a fear gauge."
    )

    with st.spinner("Loading credit spread data…"):
        hy_spread = fetch("BAMLH0A0HYM2")   # HY OAS spread
        ig_spread = fetch("BAMLC0A0CM")     # IG OAS spread
        vix       = fetch("VIXCLS")         # VIX

    fig5 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("OAS Credit Spreads (bps)", "VIX — Equity Volatility"),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
    )

    for label, series, color in [
        ("HY Spread — BAMLH0A0HYM2", hy_spread, "#ff7b72"),
        ("IG Spread — BAMLC0A0CM",   ig_spread, "#58a6ff"),
    ]:
        if not series.empty:
            fig5.add_trace(go.Scatter(
                x=series.index, y=series.values,
                name=label, line=dict(color=color, width=2),
                hovertemplate="%{x|%b %Y}: %{y:.0f}bps<extra>" + label + "</extra>",
            ), row=1, col=1)

    if not vix.empty:
        fig5.add_trace(go.Scatter(
            x=vix.index, y=vix.values,
            name="VIX", line=dict(color="#ffa657", width=2),
            fill="tozeroy", fillcolor="rgba(255,166,87,0.06)",
            hovertemplate="%{x|%b %Y}: %{y:.1f}<extra>VIX</extra>",
        ), row=2, col=1)

    add_cut_overlays(fig5, cut_dates, window_months=12, row=1, col=1, annotate=True)
    add_cut_overlays(fig5, cut_dates, window_months=12, row=2, col=1, annotate=False)

    fig5.update_layout(
        **DARK_LAYOUT, height=620, hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.08),
    )
    for r in [1, 2]:
        fig5.update_xaxes(**AXIS_STYLE, row=r, col=1)
        fig5.update_yaxes(**AXIS_STYLE, row=r, col=1)
    fig5.update_yaxes(title_text="Spread (bps)", row=1, col=1)
    fig5.update_yaxes(title_text="VIX",          row=2, col=1)
    st.plotly_chart(fig5, use_container_width=True)

    # Spread change table: level at cut start vs. 6mo and 12mo later
    spread_rows = []
    for i, dt in enumerate(cut_dates):
        cycle_label = f"Cycle {i+1}  ({dt.strftime('%b %Y')})"
        for sname, series in [("HY Spread", hy_spread), ("IG Spread", ig_spread), ("VIX", vix)]:
            if series.empty:
                continue
            at_cut  = series.asof(dt)
            at_6mo  = series.asof(dt + pd.DateOffset(months=6))
            at_12mo = series.asof(dt + pd.DateOffset(months=12))
            if pd.isna(at_cut):
                continue
            spread_rows.append({
                "Cycle":    cycle_label,
                "Series":   sname,
                "At Cut":   f"{at_cut:.0f}",
                "6mo":      f"{at_6mo:.0f}" if not pd.isna(at_6mo) else "n/a",
                "12mo":     f"{at_12mo:.0f}" if not pd.isna(at_12mo) else "n/a",
                "Δ 12mo":   f"{at_12mo - at_cut:+.0f}" if not pd.isna(at_12mo) else "n/a",
            })

    if spread_rows:
        st.markdown("##### Spread / VIX levels at cut start, 6mo, and 12mo later")
        st.dataframe(
            pd.DataFrame(spread_rows).set_index(["Cycle", "Series"]),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Portfolio Hedges
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("How do portfolio hedges perform around Fed cuts?")
    st.markdown(
        "Gold and the broad USD index — traditional portfolio hedges — "
        "indexed to **100 at each cycle start** so cycles are directly comparable."
    )

    with st.spinner("Loading hedge asset data…"):
        gold   = fetch("GOLDPMGBD228NLBM")  # Gold Fixing Price 3:00 P.M. (London) USD/troy oz
        dollar = fetch("DTWEXBGS")          # Trade-weighted USD index (broad)

    fig6 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Gold — Indexed to 100 at Cut Start", "USD Index — Indexed to 100 at Cut Start"),
        horizontal_spacing=0.08,
    )

    window_months = 18
    for col_idx, (asset_name, series) in enumerate(
        [("Gold (GOLDPMGBD228NLBM)", gold), ("USD Index (DTWEXBGS)", dollar)], start=1
    ):
        if series.empty:
            continue
        for i, dt in enumerate(cut_dates):
            color = CYCLE_COLORS[i % len(CYCLE_COLORS)]
            end_dt = dt + pd.DateOffset(months=window_months)
            window = series[(series.index >= dt) & (series.index <= end_dt)]
            base   = series.asof(dt)
            if window.empty or pd.isna(base) or base == 0:
                continue
            indexed = (window / base * 100)
            months_offset = ((window.index - dt) / pd.Timedelta(days=30.44)).round(1)
            fig6.add_trace(go.Scatter(
                x=months_offset, y=indexed.values,
                name=f"Cycle {i+1} ({dt.strftime('%b %Y')})",
                line=dict(color=color, width=2),
                showlegend=(col_idx == 1),
                hovertemplate="Month %{x}: %{y:.1f}<extra>Cycle " + str(i+1) + "</extra>",
            ), row=1, col=col_idx)

        # Baseline at 100
        fig6.add_hline(y=100, line=dict(color="#8b949e", width=1, dash="dot"), row=1, col=col_idx)
        fig6.update_xaxes(title_text="Months after cut", **AXIS_STYLE, row=1, col=col_idx)
        fig6.update_yaxes(title_text="Indexed (100 = cut date)", **AXIS_STYLE, row=1, col=col_idx)

    fig6.update_layout(
        **DARK_LAYOUT, height=480, hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
    )
    st.plotly_chart(fig6, use_container_width=True)

    # Peak/trough table for each asset per cycle
    hedge_rows = []
    for i, dt in enumerate(cut_dates):
        cycle_label = f"Cycle {i+1}  ({dt.strftime('%b %Y')})"
        end_dt = dt + pd.DateOffset(months=window_months)
        for sname, series in [("Gold", gold), ("USD Index", dollar)]:
            if series.empty:
                continue
            window = series[(series.index >= dt) & (series.index <= end_dt)]
            base   = series.asof(dt)
            if window.empty or pd.isna(base) or base == 0:
                continue
            indexed = window / base * 100
            peak   = indexed.max()
            trough = indexed.min()
            final  = indexed.iloc[-1]
            hedge_rows.append({
                "Cycle":       cycle_label,
                "Asset":       sname,
                "Peak":        f"{peak:.1f}",
                "Trough":      f"{trough:.1f}",
                f"At {window_months}mo": f"{final:.1f}",
            })

    if hedge_rows:
        st.markdown(f"##### Indexed performance over {window_months} months post-cut (base = 100)")
        st.dataframe(
            pd.DataFrame(hedge_rows).set_index(["Cycle", "Asset"]),
            use_container_width=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with FRED API · Streamlit · Plotly  ·  FinTechCo internal use only")
