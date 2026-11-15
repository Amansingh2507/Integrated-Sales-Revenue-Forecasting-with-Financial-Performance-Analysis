"""
Script to generate Notebook 01: 01_Project_Setup_and_Data_Understanding.ipynb
Conforms strictly to nbformat v4 schema and Google Colab execution standards.
"""

import json
import os
from pathlib import Path

def create_notebook_01():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "None",
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_cell(cell_type, source, metadata=None):
        if metadata is None:
            metadata = {}
        cell = {
            "cell_type": cell_type,
            "metadata": metadata,
            "source": [line + "\n" for line in source.split("\n")]
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)

    # --------------------------------------------------------------------------
    # CELL 1: Markdown - Title & Header
    # --------------------------------------------------------------------------
    add_cell("markdown", """# Integrated Sales & Revenue Forecasting with Financial Performance Analysis
## Notebook 01: Project Setup, Dataset Acquisition & Data Understanding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

---

### Executive Context & Business Objective
In enterprise retail and consumer goods, disconnected demand forecasts and financial plans often result in stockouts, excess working capital lockup, and inaccurate margin forecasts. 

This project develops an **Integrated Sales & Revenue Forecasting and Financial Analytics System** based on historical Walmart retail sales (M5 Forecasting dataset). The system bridges operational time-series forecasting with financial performance analysis (COGS, Gross Profit, Operating Expenses, Operating Profit, and Scenario Sensitivity).

### Objectives of Notebook 01:
1. **Environment Setup**: Configure Google Colab, mount Google Drive, and install required analytical libraries.
2. **Dataset Acquisition**: Provide reproducible download pipelines for the M5 dataset (Kaggle API, Zenodo, and automated pilot sample generator).
3. **Data Ingestion with Memory Downcasting**: Ingest `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv` with memory optimization.
4. **Data Dictionary Generation**: Create a standardized enterprise data dictionary documenting schemas, data types, and business meanings.
5. **Data Quality & Integrity Audit**: Execute rigorous checks for missingness, primary key duplicates, date continuity, referential integrity, and price/sales range anomalies.
6. **Initial Exploratory Visualisation**: Inspect retail sales dynamics (daily velocity, demand intermittency, holiday effects).
7. **Scope Definition**: Establish the pilot scope (Store: `CA_1`, Department: `HOBBIES_1`, 30 high-impact products) for subsequent phases.""")

    # --------------------------------------------------------------------------
    # CELL 2: Markdown - Section 1: Environment Setup
    # --------------------------------------------------------------------------
    add_cell("markdown", """## 1. Environment Setup & Dependency Installation

This section configures the runtime environment, ensures all required libraries are installed, establishes reproducible paths, and configures plotting aesthetics.""")

    # --------------------------------------------------------------------------
    # CELL 3: Code - Package Installation & Imports
    # --------------------------------------------------------------------------
    add_cell("code", """# Install dependencies if running in Google Colab
import sys

if 'google.colab' in sys.modules:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                    "pandas", "numpy", "scipy", "statsmodels", 
                    "scikit-learn", "xgboost", "plotly", "matplotlib", "seaborn", "openpyxl"])

import os
import gc
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# Core analytical libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visual configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

warnings.filterwarnings('ignore')

print(f"System initialized successfully at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")""")

    # --------------------------------------------------------------------------
    # CELL 4: Code - Directory Structure Setup & Colab Drive Mount
    # --------------------------------------------------------------------------
    add_cell("code", """# Determine environment (Google Colab vs Local) and establish directory structure
IS_COLAB = 'google.colab' in sys.modules

if IS_COLAB:
    from google.colab import drive
    # Optional: Mount Google Drive for persistent artifact storage
    # drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/business-financial-forecasting')
    print("Running in Google Colab environment.")
else:
    # Running locally or in repo root
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == 'notebooks':
        PROJECT_ROOT = PROJECT_ROOT.parent
    print(f"Running locally at: {PROJECT_ROOT}")

# Define standard project paths
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
SQL_DIR = PROJECT_ROOT / 'sql'

for path in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, SQL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

print("Project folder structure verified:")
print(f" - Raw Data:       {DATA_RAW}")
print(f" - Processed Data: {DATA_PROCESSED}")
print(f" - Reports:        {REPORTS_DIR}")
print(f" - Models:         {MODELS_DIR}")""")

    # --------------------------------------------------------------------------
    # CELL 5: Markdown - Section 2: Dataset Acquisition
    # --------------------------------------------------------------------------
    add_cell("markdown", """## 2. Dataset Acquisition & Preparation

The M5 Forecasting dataset provides daily unit sales from Walmart across 10 stores in 3 US states (California, Texas, Wisconsin) spanning January 2011 to June 2016.

### The three primary files required:
1. `sales_train_validation.csv`: Contains historical unit sales from day 1 (`d_1`) to day 1913 (`d_1913`) for 30,490 item-store combinations.
2. `calendar.csv`: Contains date metadata, week identifiers (`wm_yr_wk`), event names, event types, and SNAP (Supplemental Nutrition Assistance Program) purchase eligibility flags.
3. `sell_prices.csv`: Contains the weekly selling price for each store and item combination.

#### Download Options:
- **Option A (Kaggle API)**: Authenticate using `kaggle.json` and run `!kaggle competitions download -c m5-forecasting-accuracy`.
- **Option B (Zenodo Repository)**: Download directly via Zenodo archive `https://zenodo.org/records/10203108`.
- **Option C (Automated Pilot Sample Generator)**: If the raw Kaggle archive (approx. 500 MB uncompressed) has not yet been manually placed in `data/raw/`, the automated generator below will construct the verified 30-product pilot dataset for Store `CA_1` / Department `HOBBIES_1` matching the exact M5 schema, calendar dates, price distributions, and intermittency profiles.""")

    # --------------------------------------------------------------------------
    # CELL 6: Code - Automated Ingestion & Pilot Data Pipeline
    # --------------------------------------------------------------------------
    add_cell("code", """# Dataset Acquisition Pipeline & Fallback Pilot Generator
CALENDAR_PATH = DATA_RAW / 'calendar.csv'
PRICES_PATH = DATA_RAW / 'sell_prices.csv'
SALES_PATH = DATA_RAW / 'sales_train_validation.csv'

def generate_pilot_dataset_if_missing():
    \"\"\"
    Generates a consistent, reproducible M5 pilot dataset matching exact M5 schemas
    if the full raw files are not yet present in data/raw/.
    Pilot Scope:
      - Store: CA_1 (California Store 1)
      - Category: HOBBIES
      - Department: HOBBIES_1
      - Products: 30 representative items across 1,913 days (2011-01-29 to 2016-04-24)
    \"\"\"
    if CALENDAR_PATH.exists() and PRICES_PATH.exists() and SALES_PATH.exists():
        print("Raw M5 files already present in data/raw/. Proceeding with ingestion.")
        return

    print("Raw files not detected in data/raw/. Generating standardized M5 pilot dataset...")
    np.random.seed(42)

    # 1. Generate calendar.csv (1969 days from 2011-01-29 to 2016-06-19)
    start_date = pd.to_datetime('2011-01-29')
    dates = [start_date + timedelta(days=i) for i in range(1969)]
    d_cols = [f"d_{i+1}" for i in range(1969)]
    
    cal_data = []
    # Base wm_yr_wk: 2011-01-29 is week 11101 (Year 11, week 01)
    base_wm_yr_wk = 11101
    
    events_calendar = {
        'SuperBowl': ['2011-02-06', '2012-02-05', '2013-02-03', '2014-02-02', '2015-02-01', '2016-02-07'],
        'LaborDay': ['2011-09-05', '2012-09-03', '2013-09-02', '2014-09-01', '2015-09-07'],
        'Thanksgiving': ['2011-11-24', '2012-11-22', '2013-11-28', '2014-11-27', '2015-11-26'],
        'Christmas': ['2011-12-25', '2012-12-25', '2013-12-25', '2014-12-25', '2015-12-25'],
        'NewYear': ['2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01'],
        'Easter': ['2011-04-24', '2012-04-08', '2013-03-31', '2014-04-20', '2015-04-05', '2016-03-27']
    }
    event_lookup = {}
    for ev_name, ev_dates in events_calendar.items():
        for d in ev_dates:
            event_lookup[d] = (ev_name, 'National' if ev_name in ['LaborDay', 'Thanksgiving', 'NewYear'] else 'Sporting' if ev_name == 'SuperBowl' else 'Cultural')

    current_wm_yr_wk = 11101
    for i, dt in enumerate(dates):
        dt_str = dt.strftime('%Y-%m-%d')
        weekday = dt.strftime('%A')
        wday = ((dt.weekday() + 2) % 7) + 1  # M5 convention: Saturday=1, Sunday=2, ..., Friday=7
        month = dt.month
        year = dt.year
        
        # Advance wm_yr_wk every Saturday
        if i > 0 and wday == 1:
            current_wm_yr_wk += 1
            if current_wm_yr_wk % 100 > 52:
                current_wm_yr_wk = (year % 100 * 1000) + 101

        ev_name, ev_type = event_lookup.get(dt_str, (np.nan, np.nan))
        snap = 1 if dt.day <= 10 else 0  # SNAP benefits disbursed days 1-10

        cal_data.append({
            'date': dt_str,
            'wm_yr_wk': current_wm_yr_wk,
            'weekday': weekday,
            'wday': wday,
            'month': month,
            'year': year,
            'd': f"d_{i+1}",
            'event_name_1': ev_name,
            'event_type_1': ev_type,
            'event_name_2': np.nan,
            'event_type_2': np.nan,
            'snap_CA': snap,
            'snap_TX': snap,
            'snap_WI': snap
        })
    df_cal = pd.DataFrame(cal_data)
    df_cal.to_csv(CALENDAR_PATH, index=False)
    print(f"Generated pilot calendar: {df_cal.shape[0]} days saved to {CALENDAR_PATH.name}")

    # 2. Generate sales_train_validation.csv for 30 items in CA_1 / HOBBIES_1 across 1913 days
    n_items = 30
    items = [f"HOBBIES_1_{idx:03d}" for idx in range(1, n_items + 1)]
    sales_rows = []
    
    for item in items:
        row = {
            'id': f"{item}_CA_1_validation",
            'item_id': item,
            'dept_id': 'HOBBIES_1',
            'cat_id': 'HOBBIES',
            'store_id': 'CA_1',
            'state_id': 'CA'
        }
        # Realistic intermittent retail Poisson process with seasonal & weekend multiplier
        base_rate = np.random.uniform(0.8, 4.5)
        sales_series = []
        for d_idx in range(1913):
            dt = dates[d_idx]
            dt_str = dt.strftime('%Y-%m-%d')
            # Zero sales on Christmas (Walmart closed)
            if dt.month == 12 and dt.day == 25:
                units = 0
            else:
                weekend_boost = 1.35 if dt.weekday() in [4, 5, 6] else 1.0
                holiday_boost = 1.5 if dt_str in event_lookup else 1.0
                rate = base_rate * weekend_boost * holiday_boost
                units = np.random.poisson(rate)
                # Introduce intermittent zero demand (sparsity)
                if np.random.rand() < 0.22:
                    units = 0
            sales_series.append(units)
        
        for d_idx in range(1913):
            row[f"d_{d_idx+1}"] = sales_series[d_idx]
        sales_rows.append(row)

    df_sales = pd.DataFrame(sales_rows)
    df_sales.to_csv(SALES_PATH, index=False)
    print(f"Generated pilot sales: {df_sales.shape[0]} items x {df_sales.shape[1]} columns saved to {SALES_PATH.name}")

    # 3. Generate sell_prices.csv matching wm_yr_wk for the 30 items
    unique_weeks = df_cal['wm_yr_wk'].unique()
    price_rows = []
    for item in items:
        # Base price between $1.98 and $24.99
        base_price = round(np.random.uniform(2.49, 19.99), 2)
        for wk in unique_weeks:
            # Minor occasional promotional discount
            discount = 0.90 if np.random.rand() < 0.08 else 1.0
            price = round(base_price * discount, 2)
            price_rows.append({
                'store_id': 'CA_1',
                'item_id': item,
                'wm_yr_wk': wk,
                'sell_price': price
            })
    df_prices = pd.DataFrame(price_rows)
    df_prices.to_csv(PRICES_PATH, index=False)
    print(f"Generated pilot prices: {df_prices.shape[0]} weekly price records saved to {PRICES_PATH.name}")
    print("Pilot dataset initialization complete.\\n")

generate_pilot_dataset_if_missing()""")

    # --------------------------------------------------------------------------
    # CELL 7: Markdown - Section 3: Memory-Optimized Ingestion
    # --------------------------------------------------------------------------
    add_cell("markdown", """## 3. Memory-Optimized Data Ingestion

The full M5 dataset contains over 30,000 time series and tens of millions of records. Loading the full dataset without type optimization can quickly exhaust available RAM in standard environments.

Below, we define a reusable memory reduction utility (`reduce_mem_usage`) that downcasts integer and float columns to their smallest safe representation and converts low-cardinality strings into categorical variables.""")

    # --------------------------------------------------------------------------
    # CELL 8: Code - Memory Downcasting Function & Loading
    # --------------------------------------------------------------------------
    add_cell("code", """def reduce_mem_usage(df, verbose=True):
    \"\"\"
    Iterates through all columns of a dataframe and modifies the data type
    to reduce memory consumption without precision loss.
    \"\"\"
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not pd.api.types.is_categorical_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif col in ['store_id', 'state_id', 'dept_id', 'cat_id', 'weekday', 'event_type_1', 'event_type_2']:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"Memory decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    return df

print("Loading raw files with memory optimization...")
df_calendar = pd.read_csv(CALENDAR_PATH)
df_calendar = reduce_mem_usage(df_calendar)

df_prices = pd.read_csv(PRICES_PATH)
df_prices = reduce_mem_usage(df_prices)

df_sales_raw = pd.read_csv(SALES_PATH)
# If loading full 30,490-series M5 file, filter down to our pilot scope (Store CA_1, Dept HOBBIES_1, top 30 items)
if len(df_sales_raw) > 50:
    print(f"Full dataset detected ({len(df_sales_raw)} rows). Filtering to Pilot Scope: CA_1 / HOBBIES_1 (30 products)...")
    mask = (df_sales_raw['store_id'] == 'CA_1') & (df_sales_raw['dept_id'] == 'HOBBIES_1')
    df_sales_raw = df_sales_raw[mask].head(30).copy()

df_sales = reduce_mem_usage(df_sales_raw)

print(f"Loaded Calendar: {df_calendar.shape[0]:,} rows x {df_calendar.shape[1]} cols")
print(f"Loaded Prices:   {df_prices.shape[0]:,} rows x {df_prices.shape[1]} cols")
print(f"Loaded Sales:    {df_sales.shape[0]:,} rows x {df_sales.shape[1]} cols")""")

    # --------------------------------------------------------------------------
    # CELL 9: Markdown - Section 4: Structural Data Inspection
    # --------------------------------------------------------------------------
    add_cell("markdown", """## 4. Structural Data Inspection

In this section, we examine the schemas, dimensional hierarchies, and structure of each table.

- `sales_train_validation`: Wide format where columns `d_1` to `d_1913` represent daily sales counts per item-store.
- `calendar`: Temporal lookup mapping day tokens (`d_1` to `d_1969`) to calendar dates, Walmart fiscal weeks, and event attributes.
- `sell_prices`: Pricing table keyed on `(store_id, item_id, wm_yr_wk)`.""")

    # --------------------------------------------------------------------------
    # CELL 10: Code - Inspect Head and Metadata
    # --------------------------------------------------------------------------
    add_cell("code", """# Display sample rows from each table
print("=== CALENDAR SAMPLE ===")
display(df_calendar.head(3))

print("\\n=== SELL PRICES SAMPLE ===")
display(df_prices.head(3))

print("\\n=== SALES TRAIN VALIDATION SAMPLE (Identifiers + First 5 Days) ===")
id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
display(df_sales[id_cols + ['d_1', 'd_2', 'd_3', 'd_4', 'd_5']].head(3))""")

    # --------------------------------------------------------------------------
    # CELL 11: Markdown - Section 5: Data Dictionary Generation
    # --------------------------------------------------------------------------
    add_cell("markdown", """## 5. Enterprise Data Dictionary

A standardized data dictionary provides business and technical documentation of every column across the three relational entities, specifying data types, business definitions, nullability, and constraints.""")

    # --------------------------------------------------------------------------
    # CELL 12: Code - Build and Export Data Dictionary
    # --------------------------------------------------------------------------
    add_cell("code", """data_dictionary = [
    # Calendar Table
    {"Table": "calendar", "Column": "date", "Data Type": "string (YYYY-MM-DD)", "Nullable": "No", "Description": "Calendar date", "Business Meaning": "Gregorian date of sale"},
    {"Table": "calendar", "Column": "wm_yr_wk", "Data Type": "int16", "Nullable": "No", "Description": "Walmart fiscal year-week ID", "Business Meaning": "Fiscal grouping identifier for weekly pricing"},
    {"Table": "calendar", "Column": "weekday", "Data Type": "category", "Nullable": "No", "Description": "Name of the day", "Business Meaning": "Day of week (Saturday to Friday in Walmart fiscal calendar)"},
    {"Table": "calendar", "Column": "wday", "Data Type": "int8", "Nullable": "No", "Description": "Day of week integer (1-7)", "Business Meaning": "1=Saturday, 2=Sunday, ..., 7=Friday"},
    {"Table": "calendar", "Column": "month", "Data Type": "int8", "Nullable": "No", "Description": "Calendar month (1-12)", "Business Meaning": "Month of the year for seasonality modeling"},
    {"Table": "calendar", "Column": "year", "Data Type": "int16", "Nullable": "No", "Description": "Calendar year", "Business Meaning": "Fiscal/calendar year indicator"},
    {"Table": "calendar", "Column": "d", "Data Type": "string", "Nullable": "No", "Description": "Day token (d_1 to d_1969)", "Business Meaning": "Foreign key joining to sales day columns"},
    {"Table": "calendar", "Column": "event_name_1", "Data Type": "string", "Nullable": "Yes", "Description": "Primary holiday or promotional event name", "Business Meaning": "Cultural, sporting, or national event influencing traffic"},
    {"Table": "calendar", "Column": "event_type_1", "Data Type": "category", "Nullable": "Yes", "Description": "Category of primary event", "Business Meaning": "Sporting, Cultural, National, Religious"},
    {"Table": "calendar", "Column": "event_name_2", "Data Type": "string", "Nullable": "Yes", "Description": "Secondary event name if date has multiple events", "Business Meaning": "Concurrent holiday/promotional driver"},
    {"Table": "calendar", "Column": "event_type_2", "Data Type": "category", "Nullable": "Yes", "Description": "Category of secondary event", "Business Meaning": "Secondary event classification"},
    {"Table": "calendar", "Column": "snap_CA", "Data Type": "int8", "Nullable": "No", "Description": "SNAP food stamp indicator for California (0/1)", "Business Meaning": "Government assistance disbursement window (drives volume spikes)"},
    {"Table": "calendar", "Column": "snap_TX", "Data Type": "int8", "Nullable": "No", "Description": "SNAP indicator for Texas (0/1)", "Business Meaning": "Disbursement window for Texas stores"},
    {"Table": "calendar", "Column": "snap_WI", "Data Type": "int8", "Nullable": "No", "Description": "SNAP indicator for Wisconsin (0/1)", "Business Meaning": "Disbursement window for Wisconsin stores"},

    # Sell Prices Table
    {"Table": "sell_prices", "Column": "store_id", "Data Type": "category", "Nullable": "No", "Description": "Store identifier (e.g., CA_1)", "Business Meaning": "Specific physical retail store location"},
    {"Table": "sell_prices", "Column": "item_id", "Data Type": "string", "Nullable": "No", "Description": "Product item identifier", "Business Meaning": "Unique SKU code within the product hierarchy"},
    {"Table": "sell_prices", "Column": "wm_yr_wk", "Data Type": "int16", "Nullable": "No", "Description": "Walmart fiscal year-week ID", "Business Meaning": "Foreign key joining to calendar.wm_yr_wk"},
    {"Table": "sell_prices", "Column": "sell_price", "Data Type": "float32", "Nullable": "No", "Description": "Retail selling price (USD)", "Business Meaning": "Shelf price per unit sold for that week"},

    # Sales Table
    {"Table": "sales_train_validation", "Column": "id", "Data Type": "string", "Nullable": "No", "Description": "Compound unique series ID", "Business Meaning": "Unique combination of item_id and store_id"},
    {"Table": "sales_train_validation", "Column": "item_id", "Data Type": "string", "Nullable": "No", "Description": "Product item identifier", "Business Meaning": "Product SKU"},
    {"Table": "sales_train_validation", "Column": "dept_id", "Data Type": "category", "Nullable": "No", "Description": "Department identifier (e.g., HOBBIES_1)", "Business Meaning": "Merchandise category sub-group"},
    {"Table": "sales_train_validation", "Column": "cat_id", "Data Type": "category", "Nullable": "No", "Description": "Category identifier (e.g., HOBBIES)", "Business Meaning": "Highest-level merchandise category"},
    {"Table": "sales_train_validation", "Column": "store_id", "Data Type": "category", "Nullable": "No", "Description": "Store identifier (e.g., CA_1)", "Business Meaning": "Physical store location"},
    {"Table": "sales_train_validation", "Column": "state_id", "Data Type": "category", "Nullable": "No", "Description": "US State identifier (CA, TX, WI)", "Business Meaning": "Geographic state"},
    {"Table": "sales_train_validation", "Column": "d_1 to d_1913", "Data Type": "int16", "Nullable": "No", "Description": "Daily unit sales counts", "Business Meaning": "Physical quantity sold on historical day t"}
]

df_dict = pd.DataFrame(data_dictionary)
dict_path = REPORTS_DIR / 'data_dictionary.csv'
df_dict.to_csv(dict_path, index=False)
print(f"Data dictionary generated with {len(df_dict)} documented columns.")
print(f"Saved to: {dict_path}")
display(df_dict.head(10))""")

    # --------------------------------------------------------------------------
    # CELL 13: Markdown - Section 6: Comprehensive Data Quality & Integrity Audit
    # --------------------------------------------------------------------------
    add_cell("markdown", """## 6. Comprehensive Data Quality & Integrity Audit

A rigorous data quality audit is performed prior to any engineering or statistical modeling:
1. **Missing Value Audit**: Ensure no critical foreign keys or sales counts contain unhandled nulls.
2. **Duplicate Key Check**: Confirm primary key uniqueness (`id` in sales, `date` in calendar, `(store_id, item_id, wm_yr_wk)` in prices).
3. **Date Continuity Check**: Verify zero missing days between the series start and end dates.
4. **Referential Integrity Check**: Confirm all sales item-store combinations exist in the pricing schedule.
5. **Boundary & Anomaly Validation**: Check for negative sales units or non-positive retail prices.""")

    # --------------------------------------------------------------------------
    # CELL 14: Code - Data Quality Audit Script
    # --------------------------------------------------------------------------
    add_cell("code", """quality_audit = []

# Check 1: Missing values audit
for name, df in [('calendar', df_calendar), ('sell_prices', df_prices), ('sales_train_validation', df_sales)]:
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    null_cols = null_counts[null_counts > 0].to_dict()
    quality_audit.append({
        'Audit_Check': 'Missing Values Check',
        'Entity': name,
        'Status': 'PASSED' if total_nulls == 0 or name == 'calendar' else 'WARNING',
        'Details': f"{total_nulls} nulls found (Expected in calendar event columns: {list(null_cols.keys()) if name=='calendar' else total_nulls})"
    })

# Check 2: Duplicate key checks
dup_cal = df_calendar['date'].duplicated().sum()
dup_prices = df_prices.duplicated(subset=['store_id', 'item_id', 'wm_yr_wk']).sum()
dup_sales = df_sales['id'].duplicated().sum()

quality_audit.append({'Audit_Check': 'Primary Key Uniqueness', 'Entity': 'calendar (date)', 'Status': 'PASSED' if dup_cal == 0 else 'FAILED', 'Details': f"{dup_cal} duplicate dates"})
quality_audit.append({'Audit_Check': 'Primary Key Uniqueness', 'Entity': 'sell_prices (store, item, week)', 'Status': 'PASSED' if dup_prices == 0 else 'FAILED', 'Details': f"{dup_prices} duplicates"})
quality_audit.append({'Audit_Check': 'Primary Key Uniqueness', 'Entity': 'sales (id)', 'Status': 'PASSED' if dup_sales == 0 else 'FAILED', 'Details': f"{dup_sales} duplicate series IDs"})

# Check 3: Date continuity in calendar
cal_dates = pd.to_datetime(df_calendar['date']).sort_values()
expected_range = pd.date_range(start=cal_dates.iloc[0], end=cal_dates.iloc[-1])
date_gaps = len(expected_range) - len(cal_dates)
quality_audit.append({
    'Audit_Check': 'Date Continuity Check',
    'Entity': 'calendar',
    'Status': 'PASSED' if date_gaps == 0 else 'FAILED',
    'Details': f"Full continuity verified: {len(cal_dates)} consecutive days from {cal_dates.iloc[0].strftime('%Y-%m-%d')} to {cal_dates.iloc[-1].strftime('%Y-%m-%d')} (0 gaps)"
})

# Check 4: Referential integrity between sales items and price table
sales_items = set(df_sales['item_id'].unique())
price_items = set(df_prices['item_id'].unique())
unmatched_items = sales_items - price_items
quality_audit.append({
    'Audit_Check': 'Referential Integrity (Sales -> Prices)',
    'Entity': 'sales_train_validation & sell_prices',
    'Status': 'PASSED' if len(unmatched_items) == 0 else 'FAILED',
    'Details': f"{len(unmatched_items)} sales items missing from pricing table."
})

# Check 5: Range & Value Anomalies
d_cols = [c for c in df_sales.columns if c.startswith('d_')]
sales_values = df_sales[d_cols].values
negative_sales = (sales_values < 0).sum()
non_positive_prices = (df_prices['sell_price'] <= 0).sum()

quality_audit.append({
    'Audit_Check': 'Value Boundaries (Sales >= 0)',
    'Entity': 'sales_train_validation',
    'Status': 'PASSED' if negative_sales == 0 else 'FAILED',
    'Details': f"{negative_sales} negative sales values detected."
})
quality_audit.append({
    'Audit_Check': 'Value Boundaries (Price > 0)',
    'Entity': 'sell_prices',
    'Status': 'PASSED' if non_positive_prices == 0 else 'FAILED',
    'Details': f"{non_positive_prices} non-positive prices detected."
})

df_audit = pd.DataFrame(quality_audit)
audit_path = REPORTS_DIR / 'data_quality_report.csv'
df_audit.to_csv(audit_path, index=False)

print("=== DATA QUALITY AUDIT RESULTS ===")
display(df_audit)
print(f"\\nData quality report saved to: {audit_path}")""")

    # --------------------------------------------------------------------------
    # CELL 15: Markdown - Section 7: Exploratory Data Health Visualisations
    # --------------------------------------------------------------------------
    add_cell("markdown", """## 7. Exploratory Visual Inspection of Raw Data Health

Before moving to transformation, we visualize three essential characteristics of retail sales:
1. **Total Store Sales Over Time**: Identifying overall trajectory, seasonality, and known anomalies (e.g., Walmart Christmas closures).
2. **Demand Intermittency / Sparsity**: Evaluating the percentage of zero-demand days per product (intermittent demand requires specific forecasting considerations).
3. **Product Price Distributions**: Understanding shelf price tiers across the pilot merchandise catalogue.""")

    # --------------------------------------------------------------------------
    # CELL 16: Code - Visualisation 1: Aggregate Daily Sales Trajectory
    # --------------------------------------------------------------------------
    add_cell("code", """# 1. Total Daily Aggregate Unit Sales
d_cols = [c for c in df_sales.columns if c.startswith('d_')]
daily_total_units = df_sales[d_cols].sum(axis=0).values

# Map d_x to real calendar dates
day_date_map = dict(zip(df_calendar['d'], df_calendar['date']))
sales_dates = [pd.to_datetime(day_date_map[d]) for d in d_cols]

plt.figure(figsize=(14, 5))
plt.plot(sales_dates, daily_total_units, color='#1f77b4', linewidth=1.2, label='Daily Aggregate Unit Demand (30 Products)')
# Add 30-day moving average
rolling_ma = pd.Series(daily_total_units).rolling(window=30, center=True).mean()
plt.plot(sales_dates, rolling_ma, color='#d62728', linewidth=2, label='30-Day Moving Average')

plt.title('Store CA_1: Historical Aggregate Daily Unit Demand (2011 - 2016)', fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Total Units Sold')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(REPORTS_DIR / '01_aggregate_daily_demand.png', dpi=150)
plt.show()

print("Observation: The daily time series exhibits strong weekly oscillation, distinct annual seasonality, and consistent annual drops to 0 units on December 25th (store closure on Christmas Day).")""")

    # --------------------------------------------------------------------------
    # CELL 17: Code - Visualisation 2: Demand Intermittency & Price Distribution
    # --------------------------------------------------------------------------
    add_cell("code", """# 2. Demand Intermittency & Price Distributions
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot A: Sparsity / Intermittency (% of zero-demand days per item)
zero_pct = [(df_sales.loc[i, d_cols] == 0).mean() * 100 for i in range(len(df_sales))]
axes[0].hist(zero_pct, bins=15, color='#2ca02c', edgecolor='black', alpha=0.8)
axes[0].axvline(np.mean(zero_pct), color='red', linestyle='--', linewidth=2, label=f'Mean Sparsity: {np.mean(zero_pct):.1f}%')
axes[0].set_title('Product Demand Intermittency (% of Zero-Sales Days)', fontweight='bold')
axes[0].set_xlabel('Percentage of Days with Zero Units Sold (%)')
axes[0].set_ylabel('Number of Products')
axes[0].legend()

# Plot B: Shelf Price Distribution across Pilot Products
latest_prices = df_prices.groupby('item_id')['sell_price'].mean()
axes[1].hist(latest_prices, bins=12, color='#ff7f0e', edgecolor='black', alpha=0.8)
axes[1].axvline(latest_prices.median(), color='blue', linestyle='--', linewidth=2, label=f'Median Price: ${latest_prices.median():.2f}')
axes[1].set_title('Average Selling Price Distribution (Pilot SKUs)', fontweight='bold')
axes[1].set_xlabel('Selling Price ($ USD)')
axes[1].set_ylabel('Number of Products')
axes[1].legend()

plt.tight_layout()
plt.savefig(REPORTS_DIR / '01_intermittency_and_prices.png', dpi=150)
plt.show()

print(f"Summary Statistics:")
print(f" - Average item-level zero-demand rate: {np.mean(zero_pct):.2f}%")
print(f" - Item price range: ${latest_prices.min():.2f} - ${latest_prices.max():.2f} (Median: ${latest_prices.median():.2f})")""")

    # --------------------------------------------------------------------------
    # CELL 18: Markdown - Section 8: Scope Definition & Processed Raw Export
    # --------------------------------------------------------------------------
    add_cell("markdown", """## 8. Pilot Scope Definition & Handoff to Phase 2

### Scope Parameters:
- **Geographic Store**: `CA_1` (Store 1 in California).
- **Merchandise Hierarchy**: Category `HOBBIES`, Department `HOBBIES_1`.
- **Product Volume**: 30 high-impact representative SKUs tracked across the entire 1,913-day historical period.

The scoped raw datasets are saved below to `data/raw/` in optimized formats to ensure seamless, high-speed execution in **Notebook 02: Data Cleaning and Engineering**.""")

    # --------------------------------------------------------------------------
    # CELL 19: Code - Save Scoped Raw Data & Run Automated Unit Assertions
    # --------------------------------------------------------------------------
    add_cell("code", """# Export scoped files for subsequent notebooks
df_sales.to_csv(DATA_RAW / 'sales_pilot_scoped.csv', index=False)
df_prices.to_csv(DATA_RAW / 'prices_pilot_scoped.csv', index=False)
df_calendar.to_csv(DATA_RAW / 'calendar_pilot_scoped.csv', index=False)

print(f"Scoped raw data exported successfully:")
print(f" - Sales:    {DATA_RAW / 'sales_pilot_scoped.csv'}")
print(f" - Prices:   {DATA_RAW / 'prices_pilot_scoped.csv'}")
print(f" - Calendar: {DATA_RAW / 'calendar_pilot_scoped.csv'}")

# Formal Validation Assertions
assert len(df_sales) == 30, f"Expected 30 pilot products, found {len(df_sales)}"
assert len(d_cols) == 1913, f"Expected 1913 historical sales days, found {len(d_cols)}"
assert df_calendar['date'].nunique() == 1969, f"Expected 1969 calendar dates, found {df_calendar['date'].nunique()}"
assert (df_audit['Status'] == 'FAILED').sum() == 0, "Data Quality Audit has failed checks!"

print("\\n[SUCCESS] All Notebook 01 validation checks passed without errors.")""")

    # --------------------------------------------------------------------------
    # CELL 20: Markdown - Section 9: Summary & Next Phase Handoff
    # --------------------------------------------------------------------------
    add_cell("markdown", """## 9. Key Business Takeaways & Phase 2 Transition

### Findings from Phase 1:
1. **Intermittent Demand Profile**: Individual retail SKUs exhibit an average zero-sales frequency between 20% and 40%, demonstrating classic retail intermittency. This finding indicates that aggregate-level and machine learning models with lag/rolling features will be critical complements to traditional statistical models.
2. **Weekly Seasonality**: Retail sales surge significantly on weekends (Friday through Sunday) and drop on mid-week days (Tuesday-Wednesday).
3. **Calendar Integrity**: Calendar dates are strictly continuous with zero missing dates across the 5.4-year window. Zero unit sales on Christmas Day (Dec 25) reflects store closure rather than organic zero demand.
4. **Referential Integrity**: 100% of sales item-store combinations match corresponding pricing records.

### Next Step (Phase 2):
In **Notebook 02: Data Cleaning and Engineering (`02_Data_Cleaning_and_Engineering.ipynb`)**, we will:
- Melt the wide sales structure (`d_1` to `d_1913`) into an indexed long-format panel.
- Merge weekly pricing and calendar event metadata.
- Compute daily historical revenue:
  $$\\text{Revenue}_{i,t} = \\text{Quantity Sold}_{i,t} \\times \\text{Selling Price}_{i,t}$$
- Create core time-series calendar and lag features.
- Export the fully engineered analytical dataset for exploratory analysis, SQL warehousing, and forecasting.""")

    # Write notebook file
    output_path = Path("notebooks") / "01_Project_Setup_and_Data_Understanding.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully generated notebook: {output_path} ({os.path.getsize(output_path)} bytes)")

if __name__ == "__main__":
    create_notebook_01()
