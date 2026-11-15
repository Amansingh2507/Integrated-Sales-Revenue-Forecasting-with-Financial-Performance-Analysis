# Executive Presentation: Integrated Sales & Revenue Forecasting with Financial Performance Analysis
### An End-to-End Time-Series Forecasting, Financial Modelling and Business Intelligence System

---

## Slide 1: Title Slide & Executive Context
- **Project Title**: Integrated Sales & Revenue Forecasting with Financial Performance Analysis
- **Subtitle**: Bridging Operational Machine Learning Demand Planning with Corporate Financial Performance Analysis (FP&A)
- **Author**: Senior Data Scientist & Financial Analytics Lead
- **Tech Stack**: Python, SQL (SQLite), Pandas, Scikit-Learn, Statsmodels, Plotly, Streamlit, Google Colab
- **Dataset**: Walmart Historical Retail Sales (M5 Forecasting)

---

## Slide 2: The Core Business Problem
- **The Planning Disconnect**: Operational managers forecast units demanded per SKU; finance directors model revenue, cost of goods sold, and operating profit in isolated spreadsheets.
- **The Risk**: Disconnected demand forecasts cause:
  1. Stockouts during peak weekend and holiday surges (lost revenue and customer churn).
  2. Trapped working capital and carrying costs on slow-moving inventory.
  3. Margin surprises when price discounting is not modeled against supplier COGS inflation.
- **The Solution**: An integrated end-to-end forecasting and pro-forma financial analytics platform delivering 28-day demand visibility, pro-forma income statements, and 2D scenario stress-testing.

---

## Slide 3: Scope & Dataset Architecture
- **Pilot Scope**:
  - Geographic Store: `CA_1` (California Store 1).
  - Merchandise Category: `HOBBIES` (Department `HOBBIES_1`).
  - Products: 30 representative SKUs tracked across 1,913 historical days (5.4 years: Jan 2011 to Apr 2016).
- **Relational Star Schema Warehouse**:
  - `fact_sales` (57,390 transaction records).
  - `dim_calendar` (1,969 days with dates, fiscal weeks, events, SNAP welfare indicators).
  - `dim_product`, `dim_store`, `fact_prices`, `fact_financials`.

---

## Slide 4: Exploratory Data Analysis & Business Intelligence
- **Daily Volume Baseline**: Predictable store baseline of ~70 units/day ($780/day revenue).
- **The Weekend Surge**: Friday through Sunday generates **52% of total weekly revenue** (+35% revenue premium over Tuesday/Wednesday).
- **Promotional Holiday Uplifts**: Super Bowl, Thanksgiving, and Easter drive a **+38% to +45% average sales surge**.
- **California SNAP Assistance**: Government welfare disbursement during the first 10 days of each month drives a measurable **+12.4% sales volume expansion**.

---

## Slide 5: Portfolio Concentration (Pareto 80/20 Rule)
- **The 80/20 Distribution**: The top 12 SKUs (40% of the catalogue) generate **80% of total revenue**.
- **ABC Inventory Classification**:
  - **Class A (Core Revenue)**: 13 SKUs generating **70% of revenue ($1,051,596)**.
  - **Class B (Secondary Volume)**: 7 SKUs generating **18% of revenue ($277,496)**.
  - **Class C (Long Tail)**: 10 SKUs generating **11% of revenue ($173,220)**.
- **Strategic Rule**: Inventory safety stocks must strictly prioritize Class A items to safeguard gross profit.

---

## Slide 6: Econometric Time-Series Decomposition
- **Augmented Dickey-Fuller (ADF) Test**: Confirmed statistical stationarity ($p = 5.08 \times 10^{-18} < 0.05$) for raw daily units ($d=0$).
- **Dominant Seasonality**: Over **52.5% of total time-series variance** is explained by weekly 7-day cyclical oscillations ($s=7$).
- **Autocorrelation (ACF/PACF)**: Heavy, recurring autocorrelation at lags 7, 14, 21, and 28 confirmed that all viable forecasting models must feature explicit weekly seasonal components.

---

## Slide 7: Forecasting Suite: Statistical vs Machine Learning
- **Evaluation Standard**: 28-day out-of-sample forward horizon (Days 1886 to 1913).
- **Models Benchmarked (11 Total)**:
  - *Heuristic Baselines*: Naive, Seasonal Naive (7D), Moving Average.
  - *Exponential Smoothing*: Simple Exponential Smoothing, Holt's Linear, Holt-Winters Additive.
  - *Econometric State-Space*: ARIMA $(1,0,1)$, SARIMA $(1,0,1)\times(1,1,1)_7$, SARIMAX (with calendar & SNAP exogenous features).
  - *Machine Learning*: Random Forest Regressor, Gradient Boosted Trees (XGBoost).

---

## Slide 8: Champion Model Selection & Residual Diagnostics
- **Master Scorecard Top Performers**:
  1. **Random Forest Regressor**: **12.53% WAPE**, 9.97 RMSE, **+1.95% Bias** (Champion Operational Model).
  2. **SARIMAX**: **12.73% WAPE**, 10.28 RMSE, **+2.42% Bias** (Top Statistical Benchmark).
  3. **XGBoost**: **12.96% WAPE**, 10.23 RMSE, **+2.56% Bias**.
  4. *Non-seasonal models* (Naive, SES) failed with **WAPE > 16.5% to 93.6%**.
- **Why WAPE Over RMSE?**: WAPE measures volume-proportional inventory capital misallocation without squaring one-off holiday outliers.
- **Residual White Noise**: Both champion models exhibited zero-mean Gaussian residuals with zero autocorrelation across 14 lags.

---

## Slide 9: Pro-Forma Financial P&L Modeling
- **28-Day Pro-Forma Baseline Income Statement**:
  - Projected Volume: **1,948 units** across the 28-day horizon.
  - Gross Revenue: **$21,124.46** ($10.84 average realized price).
  - Estimated COGS (60%): **$12,674.68** $\rightarrow$ **Gross Profit: $8,449.78 (40.0% Margin)**.
  - Store OPEX (Fixed $80/day + 10% Variable): **$4,352.45**.
  - **Operating Profit (EBIT): $4,097.33 (19.4% Operating Margin)**.

---

## Slide 10: Scenario Stress-Testing & 2D Sensitivity
- **3-Point Scenario Planning**:
  - **Optimistic (+15% Vol, +2% Px, -1% COGS)**: Revenue $24,769 | Operating Profit $5,560 (22.5% Margin) $\rightarrow$ **+35.7% EBIT expansion**.
  - **Base Case (Champion Forecast)**: Revenue $21,124 | Operating Profit $4,097 (19.4% Margin).
  - **Pessimistic (-15% Vol, -3% Px, +2% COGS)**: Revenue $17,422 | Operating Profit $2,547 (14.6% Margin) $\rightarrow$ **-37.8% EBIT compression**.
- **2D Sensitivity Matrix**: Reveals that a 5% promotional markdown requires a **+12.5% volume surge** just to break even on operating dollar profit.

---

## Slide 11: Production Streamlit Dashboard & Engineering
- **Architecture**: Modular multi-page web application (`dashboard/app.py` with 6 interactive pages).
- **Zero-Latency Response**: Pre-calculated analytical databases and `@st.cache_data` ensure instant sub-second page transitions.
- **Interactive Controls**: Dynamic assumption sliders for COGS, variable logistics, and fixed store overhead; SKU selectors; horizon adjusters; CSV download triggers.
- **Quality Assurance**: Automated test suite (17 unit and integration tests) verifying data pipelines, forecasting bounds, and financial identities.

---

## Slide 12: Executive Summary & Strategic Roadmap
- **Immediate Next Steps**:
  1. Align store delivery schedules with Thursday/Friday replenishment windows.
  2. Establish safety stock buffer on the 13 Class A items at $1.65 \times \sigma_{\text{resid}}$ (95% service level).
  3. Deploy the Streamlit platform to Streamlit Community Cloud for ongoing cross-functional executive visibility.
- **Future Enhancements**: Multi-store hierarchy scaling, neural forecasting (DeepAR / N-BEATS), and real-time vendor EDI integration.
