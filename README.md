# Integrated Sales & Revenue Forecasting with Financial Performance Analysis
### An End-to-End Time-Series Forecasting, Financial Modelling and Business Intelligence System Using Python, SQL, Machine Learning, Google Colab and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://share.streamlit.io/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-orange.svg)](https://colab.research.google.com)
[![Tests Passing](https://img.shields.io/badge/Tests-17%20Passed-brightgreen.svg)](tests/)

---

## 1. Executive Summary & Business Problem

In enterprise retail and consumer goods organizations, operational inventory forecasting and financial planning frequently operate in silos:
- **Supply chain teams** forecast unit demand per SKU to plan inventory procurement and warehouse logistics.
- **Finance directors (FP&A)** model revenue, cost of goods sold, gross margin, and operating cash flow in separate static spreadsheets.

When operational demand forecasts fail to directly inform financial statements:
1. **Stockouts & Lost Revenue**: Inaccurate weekend and holiday forecasts cause immediate stockouts on high-turnover products during peak traffic windows.
2. **Trapped Working Capital**: Over-ordering slow-moving catalogue items locks up cash and creates margin-diluting clearance markdowns.
3. **Margin Compression**: Price promotions designed to stimulate volume are planned without stress-testing supplier cost inflation.

### The Solution:
This platform bridges operational machine learning demand planning with corporate financial modeling. Utilizing historical retail transactions from Walmart (M5 Forecasting), the platform delivers:
- Granular 28-day SKU and store demand forecasts across **11 benchmarked statistical and ML models**.
- Automated translation into **Pro-Forma Income Statements** (Revenue, COGS, Gross Margin, OPEX, EBIT).
- **3-Point Scenario Planning** (Base, Optimistic, Pessimistic) and **2D Price Elasticity / COGS Sensitivity Matrices**.
- An enterprise-grade, multi-page **Streamlit Web Application** backed by an SQLite dimensional data warehouse.

---

## 2. Platform Architecture

```
                              [ Walmart M5 Dataset ]
                       (Sales, Calendar, Weekly Sell Prices)
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │       Phase 1: Project Setup & Data Quality         │
             │   - Ingestion, memory downcasting, 10-point audit   │
             │   - Scoped pilot: CA_1 / HOBBIES_1 (30 SKUs)        │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │         Phase 2: Data Cleaning & Engineering        │
             │   - Wide-to-long melt (57,390 panel observations)   │
             │   - Relational calendar & weekly price joins        │
             │   - Revenue = Units * Price; causal lag & rolling   │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │       Phase 3: SQL Relational Data Warehouse        │
             │   - Star schema: fact_sales, fact_prices, dims      │
             │   - CTEs, window functions, revenue analytics       │
             └──────────────────────────┬──────────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
             ┌───────────────────────┐     ┌───────────────────────┐
             │ Phase 4: Exploratory  │     │  Phase 5: Time-Series │
             │     Data Analysis     │     │      Decomposition    │
             │ (Trends, Seasonality, │     │ (STL, ACF/PACF, ADF,  │
             │  Pareto 80/20, SNAP)  │     │  Stationarity Checks) │
             └───────────┬───────────┘     └───────────┬───────────┘
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │     Phase 6: Statistical & ML Forecasting Suite     │
             │ - Statistical: Naive, S.Naive, Holt-Winters, SARIMA │
             │ - Machine Learning: Random Forest, XGBoost          │
             │ - Rolling-origin backtesting & metrics (WAPE, RMSE) │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │     Phase 7 & 8: Financial & Scenario Modelling     │
             │  - Pro-Forma P&L: Revenue, COGS, OPEX, EBIT         │
             │  - Scenarios: Base, Optimistic (+15%), Pessimistic  │
             │  - 2D Sensitivity matrix (Price vs COGS on EBIT)    │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │     Phase 10: Interactive Streamlit Web Platform    │
             │  - Multi-page corporate application (6 modules)     │
             │  - Dynamic assumption sliders & CSV forecast export │
             │  - Ready for Streamlit Community Cloud deployment   │
             └─────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Purpose | Technology | Implementation Details |
|---|---|---|
| **Development** | Google Colab | 10 executable `.ipynb` notebooks with drive integration |
| **Language** | Python 3.10+ | Core language for pipelines and modeling |
| **Data Manipulation** | Pandas, NumPy | Vectorized transformations, memory downcasting |
| **Relational Warehouse**| SQLite & SQL | Star schema, CTEs, window functions (`LAG`, frame windows) |
| **Visualisation** | Plotly, Matplotlib, Seaborn | Interactive Plotly web charts, publication figures |
| **Econometrics** | Statsmodels | STL decomposition, ADF unit-root tests, SARIMAX |
| **Machine Learning** | Scikit-Learn, XGBoost | Random Forest Regressor, Gradient Boosted Trees |
| **Dashboard** | Streamlit | Multi-page enterprise interface with `@st.cache_data` |
| **Testing** | Pytest & Unittest | 17 automated tests for data, models, and finance |
| **Hosting** | Streamlit Cloud | Continuous deployment via GitHub repository |

---

## 4. Google Colab Deliverables (10 Notebooks)

All 10 notebooks are located in `notebooks/` and bundled into **[notebooks/all_notebooks.zip](notebooks/all_notebooks.zip)**:

| # | Notebook Filename | Core Objectives & Deliverables |
|---|---|---|
| **01** | `01_Project_Setup_and_Data_Understanding.ipynb` | Environment setup, Colab Drive mounting, memory downcasting, data quality audit. |
| **02** | `02_Data_Cleaning_and_Engineering.ipynb` | Wide-to-long melt, calendar/price joins, daily revenue calculation, causal lag features. |
| **03** | `03_Exploratory_Data_Analysis.ipynb` | 10 meaningful charts, sales trends, Pareto 80/20 curves, holiday & SNAP uplifts. |
| **04** | `04_Time_Series_Decomposition.ipynb` | STL decomposition, ADF stationarity tests ($p < 10^{-17}$), ACF/PACF diagnostics. |
| **05** | `05_Statistical_Forecasting.ipynb` | Naive, Seasonal Naive, SES, Holt's Linear, Holt-Winters, ARIMA, SARIMA, SARIMAX. |
| **06** | `06_Machine_Learning_Forecasting.ipynb` | Random Forest Regressor, XGBoost, feature importance analysis, chronological validation. |
| **07** | `07_Model_Evaluation_and_Selection.ipynb` | Master benchmark scorecard (11 models), rolling-origin backtesting, residual diagnostics. |
| **08** | `08_Revenue_and_Financial_Analysis.ipynb` | Pro-forma P&L statements, COGS/OPEX modeling, 3-point scenarios, 2D sensitivity matrix. |
| **09** | `09_SQL_and_Business_Analytics.ipynb` | SQLite database queries, CTEs, window functions, ABC inventory classification. |
| **10** | `10_Final_Project_Integration_and_Export.ipynb` | Production artifact exports, data contracts, packaging `all_notebooks.zip`. |

---

## 5. Model Evaluation & Champion Scorecard

Models were evaluated across a standardized **28-day out-of-sample forward horizon** (Days 1886 to 1913):

| Rank | Model Name | Model Family | WAPE (%) | RMSE (Units) | MAE (Units) | Forecast Bias (%) | Selection Status |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| **1** | **Random Forest** | Machine Learning | **12.53%** | **9.97** | **8.54** | **+1.95%** | **🏆 Primary Champion** |
| **2** | **SARIMAX** | Econometric State-Space | **12.73%** | **10.28** | **8.68** | **+2.42%** | **Top Statistical Benchmark** |
| **3** | **XGBoost** | Gradient Boosted Trees | **12.96%** | **10.23** | **8.84** | **+2.56%** | Competitive ML Alternative |
| **4** | **SARIMA** | Seasonal ARIMA | **13.05%** | **10.39** | **8.90** | **+2.85%** | Strong Statistical Model |
| **5** | **Holt-Winters** | Exponential Smoothing | **13.07%** | **10.39** | **8.91** | **+2.86%** | Strong Baseline |
| **6** | **ARIMA** | Standard ARIMA | **16.50%** | **13.49** | **11.25** | **+2.70%** | Non-Seasonal (Suboptimal) |
| **7** | **SES** | Exponential Smoothing | **16.50%** | **13.51** | **11.25** | **+2.86%** | Non-Seasonal (Suboptimal) |
| **8** | **Holt's Linear**| Exponential Smoothing | **16.50%** | **13.51** | **11.25** | **+2.92%** | Non-Seasonal (Suboptimal) |
| **9** | **Moving Avg 7D**| Heuristic Benchmark | **16.67%** | **14.07** | **11.36** | **+6.44%** | Heuristic Baseline |
| **10**| **Seasonal Naive**| Heuristic Benchmark | **30.54%** | **25.83** | **20.82** | **+6.44%** | Naive Seasonal Baseline |
| **11**| **Naive** | Heuristic Benchmark | **93.61%** | **65.21** | **63.82** | **+93.61%** | Unacceptable Error |

### Why WAPE Over RMSE?
- **Outlier Distortion**: RMSE squares deviations ($\sum e_i^2$), over-penalizing non-repeatable promotional demand spikes.
- **Business Alignment**: Retail supply chain directors plan working capital on total volume percentage error. **WAPE (12.53%)** provides a scale-independent metric reflecting physical inventory allocation error.
- **Minimal Tracking Signal**: Champion Random Forest achieves **$+1.95\%$ bias**, balancing customer fill-rate protection against holding cost accumulation.

---

## 6. Pro-Forma Financial P&L & Sensitivity Analysis

### 28-Day Pro-Forma Income Statement (Base Case)
- **Forecast Volume**: $1,948$ units
- **Gross Revenue**: **$\$21,124.46$** ($\$10.84$ average shelf price)
- **COGS (60.0%)**: **$\$12,674.68$**
- **Gross Profit**: **$\$8,449.78$** (**$40.0\%$ Gross Margin**)
- **Fixed Overhead ($80/day) + Variable SG&A (10%)**: **$\$4,352.45$**
- **Operating Profit (EBIT)**: **$\$4,097.33$** (**$19.4\%$ Operating Margin**)

### 2D Sensitivity Matrix: 28-Day EBIT ($ USD)
| COGS Ratio \ Price Adj | Price: -10% | Price: -5% | Price: Base (0%) | Price: +5% | Price: +10% |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **COGS: 55%** | $\$4,409$ | $\$4,778$ | $\$5,147$ | $\$5,517$ | $\$5,886$ |
| **COGS: 58%** | $\$3,839$ | $\$4,177$ | $\$4,514$ | $\$4,852$ | $\$5,190$ |
| **COGS: 60% (Base)** | $\$3,459$ | $\$3,776$ | **$\$4,092$** | $\$4,409$ | $\$4,725$ |
| **COGS: 62%** | $\$3,079$ | $\$3,374$ | $\$3,670$ | $\$3,965$ | $\$4,261$ |
| **COGS: 65%** | $\$2,509$ | $\$2,773$ | $\$3,037$ | $\$3,301$ | $\$3,564$ |

*Strategic Finding*: A $5\%$ promotional markdown reduces 28-day operating profit by $\$316$ ($-7.7\%$). To break even on dollar gross profit, store volume must expand by **$+12.5\%$**.

---

## 7. Streamlit Web Application

The interactive web application comprises **6 specialized modules**:
1. **Executive Overview**: Executive KPI cards, revenue trajectories, top revenue drivers, and monthly margin breakdowns.
2. **Sales Analytics**: Granular daily velocity, day-of-week seasonality, holiday uplifts, and California SNAP disbursement impacts.
3. **Demand & Revenue Forecasting**: Interactive SKU selector, model chooser (11 candidates), 7-28 day horizon slider, empirical 95% confidence intervals, and downloadable CSV forecast exports.
4. **Financial Analysis**: Dynamic pro-forma income statement, configurable COGS and OPEX sliders, 3-point scenario planner, and 2D sensitivity heatmap.
5. **Business Insights**: Pareto 80/20 curve, ABC inventory classification, stockout risk mitigation, and commercial supply chain recommendations.
6. **Model Performance**: Master scorecard benchmarking, rolling-origin backtesting matrix, residual distribution plots, and executive selection rationale.

### Running the App Locally:
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## 8. Automated Testing & Quality Assurance

The system includes a 17-test automated verification suite covering data pipelines, ML predictions, and financial identities:
```bash
python tests/run_all_tests.py
# Or using pytest:
pytest tests/
```

Test coverage includes:
- `test_data_pipeline.py`: Validates file presence, 30-product schema, zero calendar date gaps, and positive revenues.
- `test_forecasting_pipeline.py`: Asserts exact 28-day forecast horizon, non-negative prediction bounds, and champion WAPE $< 15\%$.
- `test_financial_engine.py`: Verifies mathematical P&L identities ($\text{Rev} = P \times Q$, $\text{GP} = \text{Rev} - \text{COGS}$, $\text{EBIT} = \text{GP} - \text{OPEX}$) and scenario orderings.
- `test_database_integrity.py`: Confirms 57,390 transaction records in SQLite and zero orphan foreign keys.

---

## 9. Streamlit Community Cloud Deployment

Follow these quick steps to host the platform publicly on [Streamlit Community Cloud](https://share.streamlit.io):
1. Push this repository to GitHub: `https://github.com/<YOUR_USERNAME>/business-financial-forecasting`.
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in.
3. Select your repository, branch (`main`), and set **Main file path** to `dashboard/app.py`.
4. Click **Deploy!**. The app builds and goes live within 90 seconds. Detailed instructions are documented in [reports/DEPLOYMENT_GUIDE.md](reports/DEPLOYMENT_GUIDE.md).

---

## 10. Repository Structure

```text
business-financial-forecasting/
├── notebooks/                       # 10 Executable Google Colab Notebooks
│   ├── 01_Project_Setup_and_Data_Understanding.ipynb
│   ├── 02_Data_Cleaning_and_Engineering.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Time_Series_Decomposition.ipynb
│   ├── 05_Statistical_Forecasting.ipynb
│   ├── 06_Machine_Learning_Forecasting.ipynb
│   ├── 07_Model_Evaluation_and_Selection.ipynb
│   ├── 08_Revenue_and_Financial_Analysis.ipynb
│   ├── 09_SQL_and_Business_Analytics.ipynb
│   ├── 10_Final_Project_Integration_and_Export.ipynb
│   └── all_notebooks.zip            # Downloadable archive of all 10 notebooks
├── data/
│   ├── raw/                         # Raw M5 CSV files and scoped pilot datasets
│   └── processed/                   # Processed panels, daily aggregates, SQLite DB
│       ├── retail_analytics.db      # SQLite Enterprise Data Warehouse
│       └── daily_store_revenue.csv
├── dashboard/                       # Streamlit Web Application
│   ├── app.py                       # Main application entry point
│   ├── data_loader.py               # Cached data loading utility
│   └── pages/                       # Multi-page dashboard modules
│       ├── 1_Executive_Overview.py
│       ├── 2_Sales_Analytics.py
│       ├── 3_Demand_and_Revenue_Forecasting.py
│       ├── 4_Financial_Analysis.py
│       ├── 5_Business_Insights.py
│       └── 6_Model_Performance.py
├── sql/                             # DDL schemas and analytical SQL queries
│   ├── schema.sql                   # Star-schema DDL with indexes
│   └── analytical_queries.sql       # CTEs, window functions, Pareto queries
├── models/                          # Production forecast and scorecard artifacts
│   ├── final_forecast_outputs.csv
│   ├── master_model_scorecard.csv
│   └── final_financial_projections.csv
├── tests/                           # Automated test suite
│   ├── test_data_pipeline.py
│   ├── test_forecasting_pipeline.py
│   ├── test_financial_engine.py
│   ├── test_database_integrity.py
│   └── run_all_tests.py             # Master test runner
├── reports/                         # Documentation, presentations, and charts
│   ├── PROJECT_PRESENTATION.md      # 12-slide executive presentation
│   ├── INTERVIEW_PREPARATION.md     # Comprehensive Q&A interview guide
│   ├── RESUME_PROJECT_BULLETS.md    # Tailored resume bullet points
│   ├── DEPLOYMENT_GUIDE.md          # Streamlit Cloud deployment manual
│   └── *.png / *.csv                # Analytical figures and scorecards
├── requirements.txt                 # Pinned dependencies
├── .gitignore                       # Git ignore specifications
├── LICENSE                          # MIT License
└── README.md                        # Master documentation
```

---

## 11. Limitations & Future Roadmap

- **Multi-Store Hierarchy Scaling**: Expand the current pilot scope (`CA_1`) across all 10 Walmart retail stores in California, Texas, and Wisconsin using hierarchical reconciliation (MinT / Bottom-Up).
- **Deep Learning Architectures**: Benchmark deep temporal point-process models (DeepAR, Temporal Fusion Transformers, N-BEATS) against Random Forest and SARIMAX.
- **Automated ERP / EDI Ingestion**: Integrate real-time SAP/Oracle ERP connectors for daily automated replenishment purchase orders.

---

## 12. License & Author

- **Author**: Senior Data Scientist & Financial Analytics Lead
- **License**: [MIT License](LICENSE)
