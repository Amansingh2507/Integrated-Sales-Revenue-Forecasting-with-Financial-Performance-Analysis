"""
Data loading and caching utility for the Streamlit Web Application.
Ensures zero-latency page transitions with @st.cache_data.
"""

import os
import sqlite3
from pathlib import Path
import pandas as pd

try:
    import streamlit as st
    cache_decorator = st.cache_data
except ImportError:
    def cache_decorator(ttl=None):
        def decorator(fn):
            return fn
        return decorator

@cache_decorator(ttl=3600)
def get_project_root():
    current = Path(__file__).resolve().parent
    if current.name == 'dashboard':
        return current.parent
    return current

@cache_decorator(ttl=3600)
def load_daily_data():
    root = get_project_root()
    path = root / 'data' / 'processed' / 'daily_store_revenue.csv'
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

@cache_decorator(ttl=3600)
def load_db_data():
    root = get_project_root()
    db_path = root / 'data' / 'processed' / 'retail_analytics.db'
    conn = sqlite3.connect(db_path)
    
    # Products summary
    df_products = pd.read_sql_query("""
        SELECT 
            p.item_id,
            p.dept_id,
            p.cat_id,
            SUM(s.units_sold) AS total_units,
            ROUND(SUM(s.revenue), 2) AS total_revenue,
            ROUND(AVG(s.sell_price), 2) AS avg_price
        FROM fact_sales s
        JOIN dim_product p ON s.item_id = p.item_id
        GROUP BY p.item_id, p.dept_id, p.cat_id
        ORDER BY total_revenue DESC;
    """, conn)
    
    conn.close()
    return df_products

@cache_decorator(ttl=3600)
def load_forecast_outputs():
    root = get_project_root()
    path = root / 'models' / 'final_forecast_outputs.csv'
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

@cache_decorator(ttl=3600)
def load_scorecard():
    root = get_project_root()
    path = root / 'models' / 'master_model_scorecard.csv'
    return pd.read_csv(path)

@cache_decorator(ttl=3600)
def load_financial_projections():
    root = get_project_root()
    path = root / 'models' / 'final_financial_projections.csv'
    df = pd.read_csv(path)
    return df

@cache_decorator(ttl=3600)
def load_scenario_summary():
    root = get_project_root()
    path = root / 'reports' / '08_scenario_summary.csv'
    return pd.read_csv(path)

@cache_decorator(ttl=3600)
def load_sensitivity_matrix():
    root = get_project_root()
    path = root / 'reports' / '08_sensitivity_matrix.csv'
    return pd.read_csv(path, index_col=0)
