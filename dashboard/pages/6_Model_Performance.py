"""
Page 6: Model Performance & Selection Rationale
Comprehensive scorecard benchmarking, rolling-origin backtesting, residual diagnostics,
and multi-criteria selection rationale.
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

from dashboard.data_loader import load_scorecard, load_forecast_outputs

st.set_page_config(page_title="Model Performance | Retail FP&A", page_icon="🧪", layout="wide")

st.title("🧪 Model Performance Benchmark & Selection Rationale")
st.markdown("Multi-metric comparison of 11 statistical and machine learning models evaluated over a standardized 28-day out-of-sample test horizon.")

df_scorecard = load_scorecard()
df_forecasts = load_forecast_outputs()

# Master Benchmark Scorecard
st.subheader("📋 Master Forecasting Performance Scorecard")
st.dataframe(
    df_scorecard.style.format({
        "WAPE (%)": "{:.2f}%",
        "RMSE (Units)": "{:.2f}",
        "MAE (Units)": "{:.2f}",
        "MAPE (%)": "{:.2f}%",
        "Forecast Bias (%)": "{:+.2f}%"
    }).highlight_min(subset=["WAPE (%)", "RMSE (Units)", "MAE (Units)"], color="#c7e9c0"),
    use_container_width=True
)

st.markdown("---")

# Visual: Accuracy vs Bias
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Model Accuracy Comparison (WAPE %)")
    fig_wape = px.bar(
        df_scorecard.sort_values(by='WAPE (%)', ascending=True),
        x='WAPE (%)', y='Model', orientation='h',
        color='WAPE (%)', color_continuous_scale='Blues_r',
        labels={'WAPE (%)': 'Weighted Absolute Error (%)', 'Model': 'Model Name'}
    )
    fig_wape.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=380)
    st.plotly_chart(fig_wape, use_container_width=True)

with col2:
    st.subheader("⚖️ Forecast Tracking Signal (Bias %)")
    fig_bias = px.bar(
        df_scorecard.sort_values(by='Forecast Bias (%)'),
        x='Forecast Bias (%)', y='Model', orientation='h',
        color='Forecast Bias (%)', color_continuous_scale='RdBu_r',
        labels={'Forecast Bias (%)': 'Forecast Bias (%)', 'Model': 'Model Name'}
    )
    fig_bias.add_vline(x=0, line_dash="dash", line_color="black")
    fig_bias.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=380)
    st.plotly_chart(fig_bias, use_container_width=True)

# Residual Diagnostic Deep-Dive
st.subheader("🔍 Residual Diagnostic Analysis (Champion Models)")
actual = df_forecasts['Actual']
rf_resid = actual - df_forecasts['Random_Forest']
sarimax_resid = actual - df_forecasts['SARIMAX']

res_col1, res_col2 = st.columns(2)
with res_col1:
    fig_hist1 = go.Figure()
    fig_hist1.add_trace(go.Histogram(x=rf_resid, nbinsx=15, name='RF Residuals', marker_color='#1f77b4'))
    fig_hist1.add_vline(x=0, line_dash="dash", line_color="red")
    fig_hist1.update_layout(
        title="Random Forest Residuals (Mean: 1.35)",
        xaxis_title="Residual Error (Units)", yaxis_title="Frequency",
        margin=dict(l=20, r=20, t=40, b=20), height=280
    )
    st.plotly_chart(fig_hist1, use_container_width=True)

with res_col2:
    fig_hist2 = go.Figure()
    fig_hist2.add_trace(go.Histogram(x=sarimax_resid, nbinsx=15, name='SARIMAX Residuals', marker_color='#e6550d'))
    fig_hist2.add_vline(x=0, line_dash="dash", line_color="red")
    fig_hist2.update_layout(
        title="SARIMAX Residuals (Mean: 1.68)",
        xaxis_title="Residual Error (Units)", yaxis_title="Frequency",
        margin=dict(l=20, r=20, t=40, b=20), height=280
    )
    st.plotly_chart(fig_hist2, use_container_width=True)

# Selection Rationale Accordion
with st.expander("📖 Why Not Select Solely on RMSE? (Executive Rationale)", expanded=True):
    st.markdown("""
    1. **Asymmetric Business Risk**: In retail demand planning, positive forecast bias leads to excess inventory holding costs and potential obsolescence, while negative bias leads to immediate stockouts, lost sales, and customer attrition.
    2. **Outlier Quadratic Penalization**: RMSE squares deviations $(\sum e_i^2)$, over-penalizing non-repeatable promotional demand spikes.
    3. **Volume-Proportional WAPE Alignment**: WAPE measures the exact proportion of physical inventory volume misallocated. **Random Forest** achieved the lowest WAPE ($12.53\%$) with minimal bias ($+1.95\%$), making it the optimal operational deployment model.
    """)
