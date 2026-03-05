# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Single-file Streamlit dashboard (`app.py`) that analyzes the macroeconomic impact of Fed rate cuts using the FRED API. All logic lives in `app.py` — there are no modules, tests, or config files.

## Commands

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run the app
python3 -m streamlit run app.py
```

## Architecture

The entire app is `app.py`, structured in this order:

1. **Constants** — `FRED_API_KEY`, `START_DATE`, `CYCLE_COLORS`, `DARK_LAYOUT`, `AXIS_STYLE`
2. **Data layer** — `get_fred()` (cached resource) and `fetch(series_id)` (cached 1hr, resamples to monthly `MS` frequency, forward-fills)
3. **Cycle detection** — `find_cut_cycles(fedfunds, n=3)`: scans FEDFUNDS for month-over-month drops >5bps, groups consecutive cuts (gaps ≤180 days = same cycle), returns the `n` most recent cycle start dates
4. **Chart helper** — `add_cut_overlays(fig, cut_dates, ...)`: adds shaded vrects and dashed vlines for each cycle; supports both single-axis figures and subplot figures via `row`/`col` kwargs
5. **Tabs** — Three `st.tabs` rendered sequentially; each fetches its own series and builds a Plotly figure, then displays a summary table below the chart

### Data flow
```
FRED API → fetch() → monthly Series → find_cut_cycles() → cut_dates
                                                              ↓
                    tab figures ← add_cut_overlays(fig, cut_dates)
```

### Tab summary tables
- **Tab 1 (Lending Impact)**: pivot table of lag months until first ≥25bps decline in mortgage/prime rates after each cycle start
- **Tab 2 (Credit Risk)**: dual-panel subplot (delinquency rates top, FEDFUNDS bottom); 18-month shading window
- **Tab 3 (Payment Volumes)**: dual-axis chart (RSXFS left, PCE right); YoY avg spending change table (12mo post vs. 12mo pre each cut)

## Key Design Decisions
- All series resampled to **monthly frequency** (`MS`) with `.ffill()` to handle sparse quarterly data (delinquency series)
- Rate-cut cycles: drop >5bps month-over-month starts a cycle; gaps >180 days separate cycles; 3 most recent highlighted
- `CYCLE_COLORS` maps cycle index → color consistently across all tabs
- `add_cut_overlays` uses `**rc` dict unpacking so the same function works for both plain figures and subplots
- `@st.cache_resource` for the Fred client; `@st.cache_data(ttl=3600)` for fetched series
