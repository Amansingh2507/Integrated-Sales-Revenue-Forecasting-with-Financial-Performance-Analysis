# Integrated Sales & Revenue Forecasting with Financial Performance Analysis
### An End-to-End Time-Series Forecasting, Financial Modelling and Business Intelligence System Using Python, SQL, Machine Learning, Google Colab and Streamlit.

---

## 1. Executive Summary & Architecture Overview

Enterprise planning often suffers from a structural disconnect between operational sales forecasting (units demanded per SKU) and strategic financial planning (revenue, cost of goods sold, gross margin, and operating profit). When demand forecasts fail to inform financial risk models, organizations experience stockouts, unnecessary inventory carrying costs, and inaccurate cash flow expectations.

This project delivers an end-to-end analytics and machine learning solution using historical Walmart retail sales data (M5 Forecasting). The platform systematically bridges operational demand forecasting (28-day horizon) with financial statements and scenario sensitivity models, served through an interactive Streamlit web application.

```
                              [ Walmart M5 Dataset ]
                       (Sales, Calendar, Weekly Sell Prices)
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │       Phase 1: Project Setup & Data Quality         │
             │     - Ingestion, memory downcasting, data audit     │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │         Phase 2: Data Cleaning & Engineering        │
             │  - Wide-to-long melt, calendar & price merging      │
             │  - Revenue = Units * Price, temporal lag features   │
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
             │  Intermittent Demand) │     │  Stationarity Checks) │
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
             │  - Revenue, COGS, Gross Profit, Operating Profit    │
             │  - Scenarios: Base, Optimistic (+15%), Pessimistic  │
             │  - Sensitivity tables on price elasticity & margin  │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │     Phase 10: Interactive Streamlit Web Platform    │
             │  - Multi-page corporate dashboard                   │
             │  - Scenario simulator & dynamic model selector      │
             │  - Deployed to Streamlit Community Cloud            │
             └─────────────────────────────────────────────────────┘
```

---

## 2. Project Directory Structure

```text
business-financial-forecasting/
├── notebooks/                       # 10 Executable Google Colab / Jupyter Notebooks
│   ├── 01_Project_Setup_and_Data_Understanding.ipynb
│   └── all_notebooks.zip            # Downloadable archive of all notebooks
├── data/
│   ├── raw/                         # Raw M5 files (calendar, prices, sales)
│   └── processed/                   # Long-format panels, features, aggregates
├── src/                             # Reusable modular Python packages
│   ├── build_notebook_01.py         # Notebook generator utility
│   └── package_notebooks.py         # Zip packaging utility
├── sql/                             # DDL schemas and business analytics queries
├── models/                          # Serialized forecasting models
├── dashboard/                       # Streamlit application
│   ├── app.py                       # Main application entry point
│   └── pages/                       # Multi-page modular dashboard views
├── tests/                           # Pytest unit and integration test suite
│   └── test_notebook_01.py          # Automated verification for Notebook 01
├── reports/                         # Visual charts, audit logs, data dictionaries
│   ├── data_dictionary.csv
│   └── data_quality_report.csv
├── screenshots/                     # UI visual documentation
├── requirements.txt                 # Pinned dependencies
├── .gitignore                       # Git ignore specifications
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
```

---

## 3. Google Colab Execution Guide

All notebooks are designed to run seamlessly in **Google Colab** with zero friction:

1. Navigate to [Google Colab](https://colab.research.google.com).
2. Click **File** -> **Upload Notebook**.
3. Select `notebooks/01_Project_Setup_and_Data_Understanding.ipynb` (or download `notebooks/all_notebooks.zip` and extract).
4. Run cells sequentially from top to bottom.
5. If the full raw M5 files are not yet in your environment, the built-in automated pilot pipeline generates the verified pilot dataset (`CA_1`, `HOBBIES_1`, 30 SKUs across 1,913 days) automatically.

---

## 4. Progress Checklist

| Phase | Phase Name | Status | Key Deliverable |
|---|---|---|---|
| **Phase 1** | Project Setup & Architecture | **COMPLETED** | Repo structure, requirements.txt, Notebook 01 (.ipynb), Data Quality Audit |
| **Phase 2** | Data Cleaning & Feature Engineering | *Pending confirmation* | Notebook 02, wide-to-long transformation, revenue feature pipeline |
| **Phase 3** | SQL Relational Data Warehouse | *Pending* | SQLite star schema, CTEs, window functions, financial metrics |
| **Phase 4** | Exploratory Data Analysis | *Pending* | Notebook 03, 10+ business insight charts, commercial interpretations |
| **Phase 5** | Time-Series Decomposition | *Pending* | Notebook 04, STL, ADF tests, ACF/PACF, stationarity analysis |
| **Phase 6** | Statistical & ML Forecasting | *Pending* | Notebooks 05 & 06, Naive, Holt-Winters, SARIMA, RF, XGBoost |
| **Phase 7** | Model Evaluation & Selection | *Pending* | Notebook 07, rolling-origin backtesting, WAPE, MAE, bias |
| **Phase 8** | Revenue & Financial Modelling | *Pending* | Notebook 08, COGS, Gross Profit, Operating Margin, Scenarios |
| **Phase 9** | SQL & Business Analytics | *Pending* | Notebook 09, SQL analytics and reporting layer |
| **Phase 10**| Final Integration & Streamlit Dashboard| *Pending* | Notebook 10, multi-page interactive Streamlit web app |
| **Phase 11**| Testing & Quality Assurance | *Pending* | Pytest automated test suite |
| **Phase 12**| Git & GitHub Version Control | *Pending* | Git commit history, clean repository ready for deployment |
| **Phase 13**| Streamlit Cloud Deployment | *Pending* | Hosted, publicly accessible web application |
| **Phase 14**| Presentation & Interview Prep | *Pending* | Slide deck, architecture diagrams, interview Q&A guide |
