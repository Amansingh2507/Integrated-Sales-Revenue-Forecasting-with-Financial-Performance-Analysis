"""
Integrated Sales & Revenue Forecasting with Financial Performance Analysis
Main Dashboard Entry Point
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from dashboard.data_loader import (
    load_daily_data, 
    load_db_data, 
    load_forecast_outputs, 
    load_scorecard
)

# Page configuration
st.set_page_config(
    page_title="Enterprise Retail Forecasting & FP&A Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 5px solid #1f77b4;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #1f77b4; }
    .metric-label { font-size: 14px; color: #6c757d; text-transform: uppercase; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #f1f3f5;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/combo-chart--v1.png", width=80)
st.sidebar.title("Retail FP&A Suite")
st.sidebar.markdown("**Store Scope:** `CA_1` (California)")
st.sidebar.markdown("**Merchandise:** `HOBBIES_1` (30 SKUs)")
st.sidebar.markdown("**Forecast Horizon:** 28 Days")
st.sidebar.markdown("---")

# Download all notebooks button
zip_file = root_dir / 'notebooks' / 'all_notebooks.zip'
if zip_file.exists():
    with open(zip_file, "rb") as f:
        st.sidebar.download_button(
            label="📦 Download All 10 Colab Notebooks (.ZIP)",
            data=f,
            file_name="m5_forecasting_financial_notebooks.zip",
            mime="application/zip",
            use_container_width=True
        )

st.sidebar.markdown("---")
st.sidebar.caption("System Version: 2.0.0 Production | Walmart M5 Engine")

# Main Page Header
st.title("📈 Integrated Sales & Revenue Forecasting with Financial Performance Analysis")
st.markdown("""
**An End-to-End Time-Series Forecasting, Financial Modelling and Business Intelligence System Using Python, SQL, Machine Learning, Google Colab and Streamlit.**
""")

# Top-Level KPI Summary Cards
try:
    df_daily = load_daily_data()
    df_products = load_db_data()
    df_forecasts = load_forecast_outputs()
    df_scorecard = load_scorecard()

    total_units = df_daily['total_units'].sum()
    total_rev = df_daily['total_revenue'].sum()
    avg_price = total_rev / total_units
    active_days = len(df_daily)
    champion_wape = df_scorecard.iloc[0]['WAPE (%)']
    champion_model = df_scorecard.iloc[0]['Model']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Historical Revenue</div>
            <div class="metric-value">${total_rev:,.2f}</div>
            <small style="color: green;">↑ 1,913 Historical Days</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Units Sold</div>
            <div class="metric-value">{total_units:,.0f}</div>
            <small style="color: #495057;">30 Pilot SKUs</small>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Realized Price</div>
            <div class="metric-value">${avg_price:.2f}</div>
            <small style="color: #495057;">Per Unit Sold</small>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Champion Accuracy</div>
            <div class="metric-value">{champion_wape}% WAPE</div>
            <small style="color: blue;">Model: {champion_model}</small>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading summary metrics: {e}")

st.markdown("<br>", unsafe_allow_html=True)

# Overview Tabs
tab1, tab2, tab3 = st.tabs(["🏛️ Platform Architecture", "📊 Module Navigation", "📖 Executive Methodology"])

with tab1:
    st.markdown("### End-to-End Enterprise System Architecture")
    st.markdown("""
    The platform bridges operational retail demand forecasting with corporate Financial Planning & Analysis (FP&A) across 6 core pillars:
    """)
    st.image(
        "https://mermaid.ink/svg/pako:eNpVj8EKwjAMhl8l5LxF9uDB6yAP4kU8dGld1w22ZUpRhO-upcPgIQn58v-TXAomU1DBu-a764H1oQf2QpWf45x2bI_mQj7Qo8sK1lI5tqQ2aIdbO1y7g90f7HCjK6u9b-y3n03r58v9_f1jP7o_9kL_2Q", 
        caption="Operational Demand to Financial Model Architecture", 
        use_container_width=True
    )
    st.markdown("""
    1. **Data Warehouse Layer (SQLite / Star Schema)**: Ingestion of Walmart M5 transactions into indexed relational tables (`fact_sales`, `fact_prices`, `dim_calendar`, `dim_product`, `fact_financials`).
    2. **Econometric Decomposition Layer**: STL decomposition, Augmented Dickey-Fuller stationarity tests, and ACF/PACF autocorrelation diagnostics.
    3. **Forecasting Engine (Statistical & ML)**: 11 models including Naive, Holt-Winters, SARIMA, SARIMAX, Random Forest, and XGBoost.
    4. **Financial Pro-Forma Engine**: Configurable COGS, Gross Profit, and OPEX calculating pro-forma EBIT and Net Margins.
    5. **Risk & Scenario Simulator**: 3-point scenario planning (Base, Optimistic, Pessimistic) with 2D sensitivity stress-testing.
    6. **Interactive Presentation Layer**: Real-time Streamlit multi-page interface deployed to Streamlit Cloud.
    """)

with tab2:
    st.markdown("### Dashboard Module Directory")
    c1, c2 = st.columns(2)
    with c1:
        st.info("""
        **1. [Executive Overview](pages/1_Executive_Overview.py)**
        - High-level KPIs, revenue trends, margin trajectories, and portfolio highlights.
        
        **2. [Sales Analytics](pages/2_Sales_Analytics.py)**
        - Daily and monthly velocity, day-of-week seasonality, holiday uplifts, and SNAP assistance impacts.
        
        **3. [Demand & Revenue Forecasting](pages/3_Demand_and_Revenue_Forecasting.py)**
        - Interactive SKU selector, model chooser (Naive, Holt-Winters, SARIMAX, RF, XGBoost), 28-day predictions with confidence bands.
        """)
    with c2:
        st.info("""
        **4. [Financial Analysis](pages/4_Financial_Analysis.py)**
        - Interactive P&L statements, dynamic COGS/OPEX sliders, 3-point scenario comparison, and 2D sensitivity heatmap.
        
        **5. [Business Insights](pages/5_Business_Insights.py)**
        - Pareto 80/20 inventory analysis, ABC classification, stockout risk mitigation, and commercial recommendations.
        
        **6. [Model Performance](pages/6_Model_Performance.py)**
        - Multi-metric benchmark scorecard (WAPE, RMSE, MAE, Bias), rolling-origin backtesting, and residual diagnostics.
        """)

with tab3:
    st.markdown("### Executive Methodology & Governance")
    st.markdown("""
    - **No Lookahead Leakage**: All machine learning and statistical models are evaluated strictly using chronological train/validation/test splits and rolling-origin backtesting.
    - **Industry Standard Metric (WAPE)**: Retail inventory managers plan working capital on volume percentage error. WAPE ($12.53\%$) penalizes volume errors without quadratic outlier distortion.
    - **Explicit Assumptions**: All financial statements and cost assumptions (COGS 60%, variable OPEX 10%, fixed overhead) are transparently disclosed and user-configurable.
    """)
