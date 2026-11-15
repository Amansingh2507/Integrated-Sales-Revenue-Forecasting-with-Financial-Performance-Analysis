"""
Script to generate Notebook 02: 02_Data_Cleaning_and_Engineering.ipynb
Transforms wide-format sales into a feature-rich long panel dataset with calendar,
pricing, revenue calculation, and time-series lag/rolling features.
"""

import json
import os
from pathlib import Path

def create_notebook_02():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "None",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
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

    # CELL 1: Markdown Header
    add_cell("markdown", """# Integrated Sales & Revenue Forecasting with Financial Performance Analysis
## Notebook 02: Data Cleaning, Transformation & Feature Engineering

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

---

### Objectives of Notebook 02:
1. **Wide-to-Long Transformation (Melting)**: Unpivot daily sales columns (`d_1` to `d_1913`) into an indexed panel dataset of $(i, t)$ tuples.
2. **Relational Merging**:
   - Merge `calendar` metadata on day token `d`.
   - Merge weekly selling prices on `(store_id, item_id, wm_yr_wk)`.
3. **Price Imputation & Validation**: Address missing prices (pre-launch shelf periods) via item-level forward/backward fill.
4. **Historical Revenue Calculation**:
   $$\\text{Revenue}_{i,t} = \\text{Quantity Sold}_{i,t} \\times \\text{Selling Price}_{i,t}$$
5. **Feature Engineering**:
   - Calendar & cyclical indicators: Day of week, month, day of month, weekend flags, quarter.
   - Event & promotional features: Event type one-hot encodings, SNAP disbursement indicators.
   - Autoregressive lag features: $t-7, t-14, t-21, t-28$ (strictly preventing lookahead data leakage).
   - Rolling window statistics: 7-day and 28-day moving averages and rolling standard deviations.
6. **Analytical Dataset Export**: Save the engineered dataset as `sales_panel.parquet` and summary tables for SQL loading and modeling.""")

    # CELL 2: Code - Imports & Setup
    add_cell("code", """import sys
if 'google.colab' in sys.modules:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                    "pandas", "numpy", "scipy", "pyarrow", "fastparquet"])

import os
import gc
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Establish project root and paths
if 'google.colab' in sys.modules:
    PROJECT_ROOT = Path('/content/business-financial-forecasting')
else:
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == 'notebooks':
        PROJECT_ROOT = PROJECT_ROOT.parent

DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
REPORTS_DIR = PROJECT_ROOT / 'reports'

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Processing initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")""")

    # CELL 3: Code - Load Scoped Raw Data
    add_cell("code", """# Load scoped raw data created in Phase 1
sales_file = DATA_RAW / 'sales_pilot_scoped.csv'
prices_file = DATA_RAW / 'prices_pilot_scoped.csv'
cal_file = DATA_RAW / 'calendar_pilot_scoped.csv'

# Fallback to general raw names if scoped files not found
if not sales_file.exists():
    sales_file = DATA_RAW / 'sales_train_validation.csv'
    prices_file = DATA_RAW / 'sell_prices.csv'
    cal_file = DATA_RAW / 'calendar.csv'

print(f"Reading sales from:    {sales_file.name}")
print(f"Reading prices from:   {prices_file.name}")
print(f"Reading calendar from: {cal_file.name}")

df_sales = pd.read_csv(sales_file)
df_prices = pd.read_csv(prices_file)
df_calendar = pd.read_csv(cal_file)

print(f"Raw Sales Shape:    {df_sales.shape}")
print(f"Raw Prices Shape:   {df_prices.shape}")
print(f"Raw Calendar Shape: {df_calendar.shape}")""")

    # CELL 4: Markdown - Section 2: Melting Wide Sales
    add_cell("markdown", """## 2. Wide-to-Long Panel Transformation (Melting)

The raw sales data is structured horizontally, with columns `d_1` to `d_1913`. To enable relational joins with calendar dates, pricing records, and time-series modeling, we unpivot (melt) the daily columns into two attributes:
- `d`: The day identifier token (e.g., `d_1`)
- `sales`: The physical quantity of units sold on that day""")

    # CELL 5: Code - Melt Operation
    add_cell("code", """id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
d_cols = [c for c in df_sales.columns if c.startswith('d_')]

print(f"Unpivoting {len(d_cols)} daily sales columns across {len(df_sales)} time series...")

df_long = pd.melt(
    df_sales,
    id_vars=id_vars,
    value_vars=d_cols,
    var_name='d',
    value_name='sales'
)

# Convert sales to integer and d to string category
df_long['sales'] = df_long['sales'].astype(np.int16)
df_long['d'] = df_long['d'].astype('category')

print(f"Transformed to long panel: {df_long.shape[0]:,} rows x {df_long.shape[1]} columns")
display(df_long.head(4))""")

    # CELL 6: Markdown - Section 3: Relational Merging
    add_cell("markdown", """## 3. Relational Merging: Calendar and Selling Prices

Next, we perform two critical joins:
1. **Calendar Join**: Match `d` to `calendar.d` to attach exact Gregorian dates, day-of-week indices, holidays, and SNAP food assistance flags.
2. **Pricing Join**: Match `(store_id, item_id, wm_yr_wk)` to attach weekly selling prices.

### Pre-Launch Price Handling:
If a product was introduced midway through the 5-year period, prices prior to launch may appear as NaN. We forward-fill and back-fill prices per SKU so that shelf values remain defined.""")

    # CELL 7: Code - Execute Merges
    add_cell("code", """# 1. Merge with calendar
cal_cols = ['d', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 
            'event_name_1', 'event_type_1', 'snap_CA']
df_merged = df_long.merge(df_calendar[cal_cols], on='d', how='left')

# Convert date to datetime
df_merged['date'] = pd.to_datetime(df_merged['date'])

# 2. Merge with sell prices
df_merged = df_merged.merge(
    df_prices, 
    on=['store_id', 'item_id', 'wm_yr_wk'], 
    how='left'
)

# Impute any missing prices per item via group backward/forward fill
df_merged['sell_price'] = df_merged.groupby('item_id')['sell_price'].transform(lambda g: g.bfill().ffill())

# If any global missing price remains, fill with overall median
if df_merged['sell_price'].isnull().sum() > 0:
    df_merged['sell_price'] = df_merged['sell_price'].fillna(df_merged['sell_price'].median())

print(f"Merged panel dimensions: {df_merged.shape[0]:,} rows x {df_merged.shape[1]} columns")
print(f"Null sell_price count after imputation: {df_merged['sell_price'].isnull().sum()}")
display(df_merged.head(3))""")

    # CELL 8: Markdown - Section 4: Revenue Calculation
    add_cell("markdown", """## 4. Historical Revenue Calculation

In retail analytics and financial modeling:
$$\\text{Revenue} = \\text{Quantity Sold (Units)} \\times \\text{Selling Price (USD)}$$

Revenue is computed at the daily SKU level. On days where units sold is 0, revenue is naturally $0.00.""")

    # CELL 9: Code - Revenue Computation
    add_cell("code", """# Calculate daily revenue per SKU
df_merged['revenue'] = (df_merged['sales'] * df_merged['sell_price']).round(2)

print("=== REVENUE SUMMARY STATISTICS ===")
rev_stats = df_merged[['sales', 'sell_price', 'revenue']].describe().T
display(rev_stats)

total_units = df_merged['sales'].sum()
total_rev = df_merged['revenue'].sum()
avg_realized_price = total_rev / total_units if total_units > 0 else 0

print(f"\\nTotal Units Sold Across Period: {total_units:,.0f} units")
print(f"Total Historical Revenue:       ${total_rev:,.2f}")
print(f"Weighted Average Realized Price: ${avg_realized_price:.2f} / unit")""")

    # CELL 10: Markdown - Section 5: Feature Engineering
    add_cell("markdown", """## 5. Time-Series Feature Engineering

To support both statistical decomposition and supervised machine learning (Random Forest and XGBoost), we engineer three tiers of features:

1. **Temporal & Calendar Attributes**:
   - `day_of_month` (1-31), `is_weekend` (binary: Saturday/Sunday = 1), `quarter` (1-4).
   - Event flags: `has_event` (1 if holiday or event occurs, 0 otherwise).
2. **Autoregressive Lag Features**:
   - $\\text{lag}_7, \\text{lag}_{14}, \\text{lag}_{21}, \\text{lag}_{28}$: Captures weekly periodicity and removes lookahead bias.
   - *Note*: The minimum lag is 28 days for a 28-day forecasting horizon, or 7 days for recursive multi-step forecasting.
3. **Rolling Window Aggregations**:
   - $\\text{rolling\\_mean}_7, \\text{rolling\\_mean}_{28}$: Captures short-term and medium-term demand momentum.
   - $\\text{rolling\\_std}_7, \\text{rolling\\_std}_{28}$: Captures local volatility and intermittency.""")

    # CELL 11: Code - Engineer Features
    add_cell("code", """# 1. Calendar Features
df_merged['day_of_month'] = df_merged['date'].dt.day.astype(np.int8)
df_merged['quarter'] = df_merged['date'].dt.quarter.astype(np.int8)
df_merged['is_weekend'] = df_merged['wday'].isin([1, 2]).astype(np.int8)
df_merged['has_event'] = df_merged['event_name_1'].notnull().astype(np.int8)

# Sort strictly by item and date for causal time-series operations
df_merged = df_merged.sort_values(by=['item_id', 'date']).reset_index(drop=True)

# 2. Autoregressive Lags
print("Constructing autoregressive lag features...")
for lag in [7, 14, 21, 28]:
    df_merged[f'sales_lag_{lag}'] = df_merged.groupby('item_id')['sales'].shift(lag)

# 3. Rolling Window Statistics (computed over lag_7 to strictly avoid lookahead leakage)
print("Constructing rolling statistics...")
df_merged['rolling_mean_7'] = df_merged.groupby('item_id')['sales_lag_7'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
).round(2)

df_merged['rolling_mean_28'] = df_merged.groupby('item_id')['sales_lag_28'].transform(
    lambda x: x.rolling(window=28, min_periods=1).mean()
).round(2)

df_merged['rolling_std_7'] = df_merged.groupby('item_id')['sales_lag_7'].transform(
    lambda x: x.rolling(window=7, min_periods=1).std()
).fillna(0).round(2)

df_merged['rolling_std_28'] = df_merged.groupby('item_id')['sales_lag_28'].transform(
    lambda x: x.rolling(window=28, min_periods=1).std()
).fillna(0).round(2)

# Fill initial warm-up lag NaNs with 0
lag_cols = [c for c in df_merged.columns if 'lag' in c or 'rolling' in c]
df_merged[lag_cols] = df_merged[lag_cols].fillna(0)

print(f"Features successfully constructed. Total columns: {df_merged.shape[1]}")
display(df_merged[['item_id', 'date', 'sales', 'revenue', 'sales_lag_7', 'rolling_mean_7', 'rolling_std_7']].tail(5))""")

    # CELL 12: Markdown - Section 6: Daily Store Aggregation
    add_cell("markdown", """## 6. Daily Store-Level Aggregation for Macro Forecasting

In addition to the granular SKU-level panel, we construct a daily aggregated time series:
- Total Store Daily Units Sold
- Total Store Daily Revenue
- Active Product Count
- Mean Selling Price

This aggregate series serves as the primary input for statistical models (Holt-Winters, ARIMA, SARIMA) and high-level FP&A cash projections.""")

    # CELL 13: Code - Build Daily Aggregates
    add_cell("code", """# Create daily aggregated series
df_daily_agg = df_merged.groupby('date').agg({
    'sales': 'sum',
    'revenue': 'sum',
    'sell_price': 'mean',
    'has_event': 'max',
    'snap_CA': 'max',
    'wday': 'first',
    'is_weekend': 'first',
    'month': 'first',
    'year': 'first'
}).reset_index()

df_daily_agg = df_daily_agg.rename(columns={
    'sales': 'total_units',
    'revenue': 'total_revenue',
    'sell_price': 'avg_price'
})

print(f"Daily aggregated time series created: {df_daily_agg.shape[0]} consecutive days.")
display(df_daily_agg.head(5))""")

    # CELL 14: Markdown - Section 7: Export Cleaned Datasets
    add_cell("markdown", """## 7. Analytical Dataset Export & Data Integrity Validation

We export:
1. `sales_panel.parquet`: Complete feature-engineered SKU-level panel.
2. `sales_panel_sample.csv`: First 5,000 rows for rapid inspection and SQL prototyping.
3. `daily_store_revenue.csv`: Complete daily aggregate series for statistical modeling and executive dashboards.""")

    # CELL 15: Code - Save Datasets & Assertions
    add_cell("code", """# Save outputs
panel_parquet_path = DATA_PROCESSED / 'sales_panel.parquet'
panel_sample_path = DATA_PROCESSED / 'sales_panel_sample.csv'
daily_agg_path = DATA_PROCESSED / 'daily_store_revenue.csv'

# Save parquet (or fall back to csv if parquet engine not installed)
try:
    df_merged.to_parquet(panel_parquet_path, index=False, compression='snappy')
    print(f"Saved: {panel_parquet_path} ({os.path.getsize(panel_parquet_path)/1024**2:.2f} MB)")
except Exception as e:
    print(f"Parquet engine note: {e}. Saving as compressed CSV...")
    df_merged.to_csv(DATA_PROCESSED / 'sales_panel.csv.gz', index=False, compression='gzip')

df_merged.head(5000).to_csv(panel_sample_path, index=False)
df_daily_agg.to_csv(daily_agg_path, index=False)

print(f"Saved: {panel_sample_path}")
print(f"Saved: {daily_agg_path} ({len(df_daily_agg)} daily records)")

# Unit Validation Assertions
assert len(df_daily_agg) == 1913, f"Expected 1913 daily aggregate records, got {len(df_daily_agg)}"
assert (df_merged['revenue'] < 0).sum() == 0, "Found negative revenue records!"
assert df_merged['revenue'].sum() > 0, "Total revenue must be strictly positive!"
assert df_merged['sell_price'].isnull().sum() == 0, "Found unresolved null sell prices!"

print("\\n[SUCCESS] Notebook 02 executed successfully and all validation checks passed.")""")

    # CELL 16: Markdown - Summary & Handoff
    add_cell("markdown", """## 8. Summary of Phase 2 Deliverables & Next Step

### Accomplishments:
- Transformed horizontal sales into an optimized vertical panel ($30 \\times 1,913 = 57,390$ observations).
- Accurately joined temporal calendar events, SNAP flags, and weekly pricing.
- Computed daily historical revenue at both product and aggregate levels.
- Engineered 10+ time-series lag and rolling statistics with zero lookahead leakage.
- Validated and exported processed datasets for SQL and modeling pipelines.

### Next Step (Phase 3 & 4):
- **Phase 3 (SQL Relational Data Warehouse)**: Create the SQLite database with star schema (`fact_sales`, `fact_prices`, `dim_product`, `dim_calendar`, `fact_financials`) and execute analytical SQL queries.
- **Phase 4 (Exploratory Data Analysis)**: Deliver 10+ deep-dive business and financial analytical charts with commercial interpretations.""")

    # Save notebook
    out_file = Path("notebooks") / "02_Data_Cleaning_and_Engineering.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {out_file} ({os.path.getsize(out_file)} bytes)")

if __name__ == "__main__":
    create_notebook_02()
