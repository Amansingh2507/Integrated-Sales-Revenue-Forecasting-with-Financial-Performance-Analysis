"""
Page 4: Financial Analysis & Scenario Stress-Testing
Interactive P&L statement, dynamic assumption sliders, multi-scenario comparisons,
and 2D sensitivity stress-testing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from dashboard.data_loader import (
    load_forecast_outputs, 
    load_scenario_summary, 
    load_sensitivity_matrix
)

st.set_page_config(page_title="Financial Modeling | Retail FP&A", page_icon="💵", layout="wide")

st.title("💵 Revenue & Financial Performance Modeling")
st.markdown("Dynamic pro-forma income statements, configurable financial assumptions, 3-point scenario planning, and 2D sensitivity matrices.")

df_forecasts = load_forecast_outputs()
df_scenarios = load_scenario_summary()
df_sensitivity = load_sensitivity_matrix()

# Interactive Financial Assumption Controls
st.sidebar.header("Financial Assumptions (FP&A)")
cogs_slider = st.sidebar.slider("Assumed COGS Ratio (% of Revenue):", min_value=40.0, max_value=80.0, value=60.0, step=1.0) / 100.0
var_opex_slider = st.sidebar.slider("Variable OPEX (% of Revenue):", min_value=5.0, max_value=25.0, value=10.0, step=0.5) / 100.0
fixed_opex_slider = st.sidebar.slider("Fixed Store Daily Overhead ($/day):", min_value=20.0, max_value=200.0, value=80.0, step=5.0)

# Compute Dynamic 28-Day P&L
champion_units = df_forecasts['Random_Forest']
avg_price = df_forecasts['Selling_Price']
gross_revenue = (champion_units * avg_price).sum()
total_cogs = gross_revenue * cogs_slider
gross_profit = gross_revenue - total_cogs
gross_margin = (gross_profit / gross_revenue) * 100

total_var_opex = gross_revenue * var_opex_slider
total_fixed_opex = fixed_opex_slider * 28
total_opex = total_var_opex + total_fixed_opex
operating_profit = gross_profit - total_opex
operating_margin = (operating_profit / gross_revenue) * 100

# Executive Financial Scorecard
f1, f2, f3, f4 = st.columns(4)
f1.metric("Projected Revenue (28D)", f"${gross_revenue:,.2f}", f"{champion_units.sum():,.0f} Units")
f2.metric("Projected COGS", f"${total_cogs:,.2f}", f"{cogs_slider*100:.1f}% of Rev")
f3.metric("Projected Gross Profit", f"${gross_profit:,.2f}", f"{gross_margin:.1f}% Margin")
f4.metric("Operating Profit (EBIT)", f"${operating_profit:,.2f}", f"{operating_margin:.1f}% Margin")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Pro-Forma Income Statement", "🎯 Scenario Comparisons", "🔬 2D Sensitivity Matrix"])

with tab1:
    st.subheader("📋 28-Day Pro-Forma Income Statement (P&L)")
    pl_breakdown = pd.DataFrame([
        {"Financial Line Item": "Gross Revenue", "Amount ($ USD)": gross_revenue, "% of Revenue": 100.0},
        {"Financial Line Item": "Cost of Goods Sold (COGS)", "Amount ($ USD)": -total_cogs, "% of Revenue": -cogs_slider*100},
        {"Financial Line Item": "Gross Profit", "Amount ($ USD)": gross_profit, "% of Revenue": gross_margin},
        {"Financial Line Item": "Fixed Store Overhead", "Amount ($ USD)": -total_fixed_opex, "% of Revenue": -(total_fixed_opex/gross_revenue)*100},
        {"Financial Line Item": "Variable Logistics & SG&A", "Amount ($ USD)": -total_var_opex, "% of Revenue": -var_opex_slider*100},
        {"Financial Line Item": "Total Operating Expenses", "Amount ($ USD)": -total_opex, "% of Revenue": -(total_opex/gross_revenue)*100},
        {"Financial Line Item": "Operating Profit (EBIT)", "Amount ($ USD)": operating_profit, "% of Revenue": operating_margin}
    ])
    
    st.dataframe(
        pl_breakdown.style.format({
            "Amount ($ USD)": "${:,.2f}",
            "% of Revenue": "{:.2f}%"
        }),
        use_container_width=True
    )

with tab2:
    st.subheader("🎯 Multi-Scenario Performance Comparison")
    st.markdown("Evaluating the portfolio across Base, Optimistic (+15% Vol, +2% Px, -1% COGS), and Pessimistic (-15% Vol, -3% Px, +2% COGS) conditions.")
    st.dataframe(df_scenarios, use_container_width=True)
    
    fig_scen = px.bar(
        df_scenarios, x='Scenario', y=['Gross Revenue ($)', 'Operating Profit ($)'],
        barmode='group', labels={'value': 'USD ($)', 'variable': 'Financial Metric'},
        color_discrete_sequence=['#1f77b4', '#2ca02c']
    )
    fig_scen.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=350)
    st.plotly_chart(fig_scen, use_container_width=True)

with tab3:
    st.subheader("🔬 Two-Way Sensitivity Stress-Testing")
    st.markdown("Joint impact of Selling Price Adjustments ($\pm 10\%$) against Cost of Goods Sold Fluctuations ($55\%$ to $65\%$) on Operating Profit ($ USD).")
    
    fig_heat = px.imshow(
        df_sensitivity,
        labels=dict(x="Price Adjustment (%)", y="COGS Ratio (% of Revenue)", color="Operating Profit ($)"),
        color_continuous_scale="RdYlGn", text_auto=True
    )
    fig_heat.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=380)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Key Insight: A 5% price markdown requires a +12.5% unit surge just to break even on operating dollar profit.")
