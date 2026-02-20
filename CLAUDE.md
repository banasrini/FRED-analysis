# FRED Analysis — Project Guide

## Overview
Interactive Streamlit dashboard analyzing the macroeconomic impact of Fed rate cuts using FRED (Federal Reserve Economic Data) API.

## Running the App
```bash
python3 -m streamlit run app.py
```

## Installing Dependencies
```bash
pip3 install -r requirements.txt
```

## FRED API
- API key is set directly in `app.py` (`FRED_API_KEY`)
- Data is cached for 1 hour via `@st.cache_data(ttl=3600)`

## Dashboard Tabs

| Tab | Title | FRED Series |
|-----|-------|-------------|
| 1 | Lending Impact | FEDFUNDS, MORTGAGE30US, DPRIME |
| 2 | Credit Risk | DRCCLACBS, DRCLACBS, DRSFRMACBS |
| 3 | Payment Volumes | RSXFS, PCE |

## Key Design Decisions
- All series resampled to **monthly frequency** (`MS`) with forward-fill for sparse quarterly data
- Rate-cut cycles auto-detected from FEDFUNDS: a drop >5bps in a month starts a cycle; gaps >180 days separate cycles
- 3 most recent cycles highlighted across all charts
- Plotly dark theme for client-presentation readiness
