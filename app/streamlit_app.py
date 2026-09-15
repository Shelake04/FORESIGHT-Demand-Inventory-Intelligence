"""Streamlit operations view over precomputed FORESIGHT outputs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px
import generate_data
from run_pipeline import main as build_outputs

st.set_page_config(page_title="FORESIGHT", layout="wide")
st.title("FORESIGHT")
st.caption("Demand & Inventory Intelligence | NorthBay Living")

@st.cache_data
def load_outputs():
    return (pd.read_csv(ROOT / "data/processed/inventory_decisions.csv"),
            pd.read_csv(ROOT / "data/processed/forecasts.csv", parse_dates=["forecast_week"]),
            pd.read_csv(ROOT / "data/processed/weekly_demand_features.csv", parse_dates=["week_start"]))

processed_files = [
    ROOT / "data/processed/inventory_decisions.csv",
    ROOT / "data/processed/forecasts.csv",
    ROOT / "data/processed/weekly_demand_features.csv",
]

if not all(path.exists() for path in processed_files):
    with st.spinner("Preparing FORESIGHT data..."):
        generate_data.main()
        build_outputs()

decisions, forecasts, weekly = load_outputs()

categories = st.sidebar.multiselect("Category", sorted(decisions.category.unique()), default=sorted(decisions.category.unique()))
actions = st.sidebar.multiselect("Action", sorted(decisions.recommended_action.unique()), default=sorted(decisions.recommended_action.unique()))
filtered = decisions[decisions.category.isin(categories) & decisions.recommended_action.isin(actions)]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("SKUs monitored", len(filtered))
c2.metric("High stockout risk", int((filtered.stockout_risk == "HIGH").sum()))
c3.metric("High overstock risk", int((filtered.overstock_risk == "HIGH").sum()))
c4.metric("Estimated impact", f"Rs {filtered.estimated_rupee_impact.sum():,.0f}")
c5.metric("Forecast horizon", "6 weeks")
st.subheader("Recommended actions")
st.dataframe(filtered, use_container_width=True, hide_index=True)
st.subheader("Forecast trend")
trend = forecasts.groupby("forecast_week", as_index=False).forecast_units.sum()
st.plotly_chart(px.line(trend, x="forecast_week", y="forecast_units", markers=True, labels={"forecast_units": "Forecast units", "forecast_week": "Week"}), use_container_width=True)
sku = st.selectbox("SKU detail", sorted(filtered.sku_id.unique()))
history = weekly[weekly.sku_id == sku][["week_start", "demand_units"]].rename(columns={"week_start": "week", "demand_units": "units"})
future = forecasts[forecasts.sku_id == sku][["forecast_week", "forecast_units"]].rename(columns={"forecast_week": "week", "forecast_units": "units"})
chart = pd.concat([history.assign(series="Observed"), future.assign(series="Forecast")])
st.plotly_chart(px.line(chart, x="week", y="units", color="series", markers=True), use_container_width=True)