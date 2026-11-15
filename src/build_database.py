"""
Builds and populates the SQLite Enterprise Data Warehouse (retail_analytics.db)
and executes analytical verification queries.
"""

import sqlite3
import pandas as pd
from pathlib import Path

def build_database():
    root = Path(__file__).resolve().parent.parent
    db_path = root / 'data' / 'processed' / 'retail_analytics.db'
    schema_path = root / 'sql' / 'schema.sql'
    raw_dir = root / 'data' / 'raw'
    proc_dir = root / 'data' / 'processed'

    print(f"Building SQLite database at: {db_path}")

    # Remove existing database if present
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Execute schema DDL
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    cursor.executescript(schema_sql)
    print("Schema DDL executed successfully.")

    # 2. Populate dim_calendar
    df_cal = pd.read_csv(raw_dir / 'calendar_pilot_scoped.csv')
    cal_cols = ['date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'd', 'event_name_1', 'event_type_1', 'snap_CA']
    df_cal[cal_cols].to_sql('dim_calendar', conn, if_exists='append', index=False)
    print(f"Loaded dim_calendar: {len(df_cal)} records.")

    # 3. Populate dim_product and dim_store
    df_sales_raw = pd.read_csv(raw_dir / 'sales_pilot_scoped.csv')
    df_product = df_sales_raw[['item_id', 'dept_id', 'cat_id']].drop_duplicates()
    df_product.to_sql('dim_product', conn, if_exists='append', index=False)
    print(f"Loaded dim_product: {len(df_product)} records.")

    df_store = df_sales_raw[['store_id', 'state_id']].drop_duplicates()
    df_store.to_sql('dim_store', conn, if_exists='append', index=False)
    print(f"Loaded dim_store: {len(df_store)} records.")

    # 4. Populate fact_prices
    df_prices = pd.read_csv(raw_dir / 'prices_pilot_scoped.csv')
    df_prices[['store_id', 'item_id', 'wm_yr_wk', 'sell_price']].to_sql('fact_prices', conn, if_exists='append', index=False)
    print(f"Loaded fact_prices: {len(df_prices)} records.")

    # 5. Populate fact_sales
    panel_file = proc_dir / 'sales_panel.csv.gz'
    if not panel_file.exists():
        panel_file = proc_dir / 'sales_panel_sample.csv'
    
    print(f"Reading panel data from: {panel_file.name}")
    df_panel = pd.read_csv(panel_file)
    sales_insert = df_panel[['date', 'item_id', 'store_id', 'sales', 'sell_price', 'revenue']].rename(
        columns={'sales': 'units_sold'}
    )
    sales_insert.to_sql('fact_sales', conn, if_exists='append', index=False)
    print(f"Loaded fact_sales: {len(sales_insert)} records.")

    # 6. Populate fact_financials (Base assumptions: COGS=60%, OPEX=20% of revenue)
    df_daily = pd.read_csv(proc_dir / 'daily_store_revenue.csv')
    financials = pd.DataFrame()
    financials['date'] = df_daily['date']
    financials['revenue'] = df_daily['total_revenue']
    financials['cogs'] = (financials['revenue'] * 0.60).round(2)
    financials['gross_profit'] = (financials['revenue'] - financials['cogs']).round(2)
    financials['gross_margin_pct'] = 40.0
    financials['opex'] = (financials['revenue'] * 0.20).round(2)
    financials['operating_profit'] = (financials['gross_profit'] - financials['opex']).round(2)
    financials['operating_margin_pct'] = 20.0
    financials.to_sql('fact_financials', conn, if_exists='append', index=False)
    print(f"Loaded fact_financials: {len(financials)} records.")

    conn.commit()

    # 7. Verification Query
    test_query = """
    SELECT 
        COUNT(*) AS total_sales_records,
        ROUND(SUM(units_sold), 0) AS total_units,
        ROUND(SUM(revenue), 2) AS total_revenue
    FROM fact_sales;
    """
    res = pd.read_sql_query(test_query, conn)
    print("\n--- Database Verification ---")
    print(res)

    conn.close()
    print(f"\n[SUCCESS] SQLite database successfully built and verified: {db_path}")

if __name__ == "__main__":
    build_database()
