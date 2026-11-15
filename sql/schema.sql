-- ==============================================================================
-- Enterprise Retail Analytics & Financial Performance Data Warehouse Schema
-- Relational Star Schema (SQLite Dialect)
-- ==============================================================================

-- 1. Dimension: Calendar / Temporal Attributes
CREATE TABLE IF NOT EXISTS dim_calendar (
    date TEXT PRIMARY KEY,
    wm_yr_wk INTEGER NOT NULL,
    weekday TEXT NOT NULL,
    wday INTEGER NOT NULL,
    month INTEGER NOT NULL,
    year INTEGER NOT NULL,
    d TEXT NOT NULL,
    event_name_1 TEXT,
    event_type_1 TEXT,
    snap_CA INTEGER NOT NULL
);

-- 2. Dimension: Product Merchandise Hierarchy
CREATE TABLE IF NOT EXISTS dim_product (
    item_id TEXT PRIMARY KEY,
    dept_id TEXT NOT NULL,
    cat_id TEXT NOT NULL
);

-- 3. Dimension: Physical Retail Store
CREATE TABLE IF NOT EXISTS dim_store (
    store_id TEXT PRIMARY KEY,
    state_id TEXT NOT NULL
);

-- 4. Fact: Weekly Selling Prices
CREATE TABLE IF NOT EXISTS fact_prices (
    store_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    wm_yr_wk INTEGER NOT NULL,
    sell_price REAL NOT NULL,
    PRIMARY KEY (store_id, item_id, wm_yr_wk),
    FOREIGN KEY (store_id) REFERENCES dim_store(store_id),
    FOREIGN KEY (item_id) REFERENCES dim_product(item_id)
);

-- 5. Fact: Daily Product Sales & Revenue Transactions
CREATE TABLE IF NOT EXISTS fact_sales (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    item_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    units_sold INTEGER NOT NULL,
    sell_price REAL NOT NULL,
    revenue REAL NOT NULL,
    FOREIGN KEY (date) REFERENCES dim_calendar(date),
    FOREIGN KEY (item_id) REFERENCES dim_product(item_id),
    FOREIGN KEY (store_id) REFERENCES dim_store(store_id)
);

-- 6. Fact: Store-Level Daily Financial Performance
CREATE TABLE IF NOT EXISTS fact_financials (
    date TEXT PRIMARY KEY,
    revenue REAL NOT NULL,
    cogs REAL NOT NULL,
    gross_profit REAL NOT NULL,
    gross_margin_pct REAL NOT NULL,
    opex REAL NOT NULL,
    operating_profit REAL NOT NULL,
    operating_margin_pct REAL NOT NULL,
    FOREIGN KEY (date) REFERENCES dim_calendar(date)
);

-- Performance Optimization Indexes
CREATE INDEX IF NOT EXISTS idx_fact_sales_date ON fact_sales(date);
CREATE INDEX IF NOT EXISTS idx_fact_sales_item ON fact_sales(item_id);
CREATE INDEX IF NOT EXISTS idx_fact_sales_store ON fact_sales(store_id);
CREATE INDEX IF NOT EXISTS idx_dim_calendar_wm_yr_wk ON dim_calendar(wm_yr_wk);
