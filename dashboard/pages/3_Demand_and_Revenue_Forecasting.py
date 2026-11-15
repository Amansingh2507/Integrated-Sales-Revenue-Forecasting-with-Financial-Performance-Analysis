"""
Page 3: Demand and Revenue Forecasting
Interactive model selection, forecast horizon controls, actual vs predicted charts,
prediction intervals, and downloadable CSV forecasts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from dashboard.data_loader import load_forecast_outputs, load_scorecard, load_db_data

st.set_page_config(page_title="Forecasting Suite | Retail FP&A", page_icon="🔮", layout="wide")

st.title("🔮 Demand & Revenue Forecasting Suite")
st.markdown("Interactive 28-day out-of-sample forecast comparison across statistical and machine learning models with 95% empirical prediction bounds.")

df_forecasts = load_forecast_outputs()
df_scorecard = load_scorecard()
df_products = load_db_data()

# Sidebar Controls
st.sidebar.header("Forecast Settings")
store_sel = st.sidebar.selectbox("Store:", ["CA_1 (California)"])
dept_sel = st.sidebar.selectbox("Department:", ["HOBBIES_1 (30 SKUs)"])

available_models = [
    'Random_Forest', 'SARIMAX', 'XGBoost', 'SARIMA', 
    'Holt_Winters', 'ARIMA', 'SES', 'Moving_Avg_7D', 'Seasonal_Naive', 'Naive'
]
model_sel = st.sidebar.selectbox("Select Forecasting Model:", available_models, index=0)

horizon_days = st.sidebar.slider("Forecast Horizon (Days):", min_value=7, max_value=28, value=28, step=7)

# Filter horizon
active_forecast = df_forecasts.head(horizon_days).copy()
dates = active_forecast['date']
actual = active_forecast['Actual']
pred = active_forecast[model_sel]

# Compute metrics for the selected model over the chosen horizon
mae = np.mean(np.abs(actual - pred))
rmse = np.sqrt(np.mean((actual - pred)**2))
wape = (np.sum(np.abs(actual - pred)) / np.sum(actual)) * 100
bias = (np.sum(pred - actual) / np.sum(actual)) * 100

# Metric display
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Selected Model", model_sel)
k2.metric("WAPE", f"{wape:.2f}%", "Weighted Error")
k3.metric("RMSE", f"{rmse:.2f} units", "Root Mean Sq Err")
k4.metric("MAE", f"{mae:.2f} units", "Mean Absolute Err")
k5.metric("Forecast Bias", f"{bias:+.2f}%", "Tracking Signal")

st.markdown("---")

# Visual Section: Actual vs Predicted with 95% Confidence Intervals
st.subheader(f"📈 28-Day Forecast Trajectory: Actual vs {model_sel}")

# Compute empirical 95% prediction interval (1.96 * RMSE)
lower_bound = np.maximum(0, pred - 1.96 * rmse)
upper_bound = pred + 1.96 * rmse

fig = go.Figure()

# Actual
fig.add_trace(go.Scatter(
    x=dates, y=actual,
    mode='lines+markers', name='Actual Ground Truth Units',
    line=dict(color='black', width=2.5),
    marker=dict(size=6, color='black')
))

# Predicted
fig.add_trace(go.Scatter(
    x=dates, y=pred,
    mode='lines+markers', name=f'{model_sel} Forecast',
    line=dict(color='#1f77b4', width=2.5, dash='solid'),
    marker=dict(size=5, color='#1f77b4')
))

# Upper / Lower Bound Shading
fig.add_trace(go.Scatter(
    x=dates, y=upper_bound,
    mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
))
fig.add_trace(go.Scatter(
    x=dates, y=lower_bound,
    mode='lines', line=dict(width=0), fill='tonexty',
    fillcolor='rgba(31, 119, 180, 0.18)', name='95% Prediction Interval', hoverinfo='skip'
))

fig.update_layout(
    xaxis_title="Date", yaxis_title="Total Store Units Sold",
    hovermode="x unified", legend=dict(orientation="h", y=1.1),
    margin=dict(l=20, r=20, t=30, b=20), height=420
)
st.plotly_chart(fig, use_container_width=True)

# Revenue Forecast Calculation
st.subheader("💰 Translated Dollar Revenue Forecast")
avg_selling_price = active_forecast['Selling_Price']
rev_actual = active_forecast['Actual'] * avg_selling_price
rev_forecast = pred * avg_selling_price

col_r1, col_r2 = st.columns([2, 1])
with col_r1:
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(x=dates, y=rev_actual, name='Actual Revenue ($)', marker_color='#9ecae1'))
    fig_rev.add_trace(go.Bar(x=dates, y=rev_forecast, name=f'Forecast Revenue ({model_sel})', marker_color='#1f78b4'))
    fig_rev.update_layout(
        barmode='group', xaxis_title="Date", yaxis_title="Daily Revenue ($ USD)",
        legend=dict(orientation="h", y=1.1), margin=dict(l=20, r=20, t=30, b=20), height=340
    )
    st.plotly_chart(fig_rev, use_container_width=True)

with col_r2:
    st.markdown("#### Forecast Financial Summary")
    st.write(f"**Total Forecast Units:** {pred.sum():,.0f} units")
    st.write(f"**Total Forecast Revenue:** ${rev_forecast.sum():,.2f}")
    st.write(f"**Actual Realized Revenue:** ${rev_actual.sum():,.2f}")
    variance = rev_forecast.sum() - rev_actual.sum()
    var_pct = (variance / rev_actual.sum()) * 100
    st.write(f"**Dollar Variance:** ${variance:+,.2f} ({var_pct:+.2f}%)")
    
    # Download button
    export_df = active_forecast[['date', 'Actual', model_sel, 'Selling_Price']].copy()
    export_df['Revenue_Forecast'] = np.round(export_df[model_sel] * export_df['Selling_Price'], 2)
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv_bytes,
        file_name=f"m5_{model_sel.lower()}_forecast_horizon_{horizon_days}d.csv",
        mime="text/csv",
        use_container_width=True
    )
