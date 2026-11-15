-- ==============================================================================
-- Enterprise Business Intelligence & Financial Analytics SQL Queries
-- Demonstrating CTEs, Window Functions, Rolling Averages, and Pareto Analytics
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- QUERY 1: Monthly Revenue Trajectory & Month-over-Month (MoM) Growth Analysis
-- Uses CTE and LAG() Window Function
-- ------------------------------------------------------------------------------
WITH monthly_revenue AS (
    SELECT 
        c.year,
        c.month,
        PRINTF('%04d-%02d', c.year, c.month) AS year_month,
        SUM(s.units_sold) AS total_units,
        ROUND(SUM(s.revenue), 2) AS total_revenue
    FROM fact_sales s
    JOIN dim_calendar c ON s.date = c.date
    GROUP BY c.year, c.month
)
SELECT 
    year_month,
    total_units,
    total_revenue,
    LAG(total_revenue, 1) OVER (ORDER BY year_month) AS prev_month_revenue,
    ROUND(
        (total_revenue - LAG(total_revenue, 1) OVER (ORDER BY year_month)) / 
        LAG(total_revenue, 1) OVER (ORDER BY year_month) * 100.0, 
        2
    ) AS mom_growth_pct
FROM monthly_revenue
ORDER BY year_month;

-- ------------------------------------------------------------------------------
-- QUERY 2: 7-Day and 30-Day Rolling Moving Averages for Revenue Smoothing
-- Uses Frame Specifications: ROWS BETWEEN N PRECEDING AND CURRENT ROW
-- ------------------------------------------------------------------------------
WITH daily_totals AS (
    SELECT 
        date,
        SUM(units_sold) AS daily_units,
        ROUND(SUM(revenue), 2) AS daily_revenue
    FROM fact_sales
    GROUP BY date
)
SELECT 
    date,
    daily_units,
    daily_revenue,
    ROUND(AVG(daily_revenue) OVER (
        ORDER BY date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_avg_revenue_7d,
    ROUND(AVG(daily_revenue) OVER (
        ORDER BY date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_avg_revenue_30d
FROM daily_totals
ORDER BY date;

-- ------------------------------------------------------------------------------
-- QUERY 3: Product Pareto Analysis (80/20 Rule) & ABC Inventory Classification
-- Uses Cumulative Window Aggregations to identify revenue concentration
-- ------------------------------------------------------------------------------
WITH product_totals AS (
    SELECT 
        p.item_id,
        p.dept_id,
        SUM(s.units_sold) AS total_units,
        ROUND(SUM(s.revenue), 2) AS total_revenue
    FROM fact_sales s
    JOIN dim_product p ON s.item_id = p.item_id
    GROUP BY p.item_id, p.dept_id
),
ranked_products AS (
    SELECT 
        item_id,
        dept_id,
        total_units,
        total_revenue,
        SUM(total_revenue) OVER () AS portfolio_total_revenue,
        SUM(total_revenue) OVER (ORDER BY total_revenue DESC) AS cumulative_revenue,
        ROUND(
            SUM(total_revenue) OVER (ORDER BY total_revenue DESC) * 100.0 / 
            SUM(total_revenue) OVER (), 
            2
        ) AS cumulative_revenue_pct
    FROM product_totals
)
SELECT 
    item_id,
    dept_id,
    total_units,
    total_revenue,
    cumulative_revenue_pct,
    CASE 
        WHEN cumulative_revenue_pct <= 70.0 THEN 'Class A (High Value / Core)'
        WHEN cumulative_revenue_pct <= 90.0 THEN 'Class B (Medium Value)'
        ELSE 'Class C (Long Tail / Low Volume)'
    END AS abc_category
FROM ranked_products
ORDER BY total_revenue DESC;

-- ------------------------------------------------------------------------------
-- QUERY 4: Event & Holiday Uplift Analysis
-- Evaluates the commercial impact of national, cultural, and sporting events
-- ------------------------------------------------------------------------------
SELECT 
    CASE 
        WHEN c.event_name_1 IS NOT NULL THEN c.event_name_1 
        ELSE 'Non-Event Days' 
    END AS event_classification,
    c.event_type_1,
    COUNT(DISTINCT daily_sales.date) AS day_count,
    ROUND(AVG(daily_sales.units), 1) AS avg_daily_units,
    ROUND(AVG(daily_sales.rev), 2) AS avg_daily_revenue
FROM (
    SELECT date, SUM(units_sold) AS units, SUM(revenue) AS rev
    FROM fact_sales
    GROUP BY date
) daily_sales
JOIN dim_calendar c ON daily_sales.date = c.date
GROUP BY event_classification, c.event_type_1
ORDER BY avg_daily_revenue DESC;

-- ------------------------------------------------------------------------------
-- QUERY 5: SNAP Assistance Disbursement Impact on Sales Velocity
-- Analyzes California food assistance benefit timing (Days 1-10 vs remaining days)
-- ------------------------------------------------------------------------------
SELECT 
    CASE WHEN c.snap_CA = 1 THEN 'SNAP Window (Days 1-10)' ELSE 'Non-SNAP Period' END AS snap_period,
    COUNT(DISTINCT s.date) AS total_days,
    ROUND(SUM(s.units_sold) * 1.0 / COUNT(DISTINCT s.date), 2) AS units_per_day,
    ROUND(SUM(s.revenue) / COUNT(DISTINCT s.date), 2) AS revenue_per_day
FROM fact_sales s
JOIN dim_calendar c ON s.date = c.date
GROUP BY c.snap_CA;
