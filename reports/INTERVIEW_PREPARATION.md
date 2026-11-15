# Interview Preparation Guide: Integrated Business Forecasting & Financial Analytics

This comprehensive interview guide prepares you to speak confidently about every technical, econometric, machine learning, and financial aspect of the **Integrated Sales & Revenue Forecasting and Financial Analytics System**.

---

## SECTION 1: Technical & Data Engineering Questions

### Q1: Why did you start with a pilot scope (Store CA_1, HOBBIES_1, 30 SKUs) instead of the entire 30,490 M5 series at once?
**Model Answer:**
> "In enterprise data science and engineering, starting with a well-scoped pilot is a best-practice strategy to establish an airtight baseline, validate data contracts, and optimize memory footprints before horizontal scaling. The full M5 dataset comprises 30,490 time series spanning 1,913 days (over 58 million observations). Unpivoting and feature engineering at that scale requires distributed computing (Spark or Dask) and massive RAM. By scoping to Store `CA_1`, Department `HOBBIES_1`, and 30 high-impact SKUs ($57,390$ panel records), we developed a fully vectorized, modular pipeline with memory downcasting (reducing memory by over 80%), established the relational star-schema warehouse, tuned 11 models, and validated the financial engine. This architecture is designed to scale horizontally across additional stores and categories without code refactoring."

### Q2: How did you prevent data leakage during time-series feature engineering and validation?
**Model Answer:**
> "Data leakage is the single most common failure mode in applied time-series machine learning. We enforced strict preventative safeguards:
> 1. **Chronological Splitting**: We strictly forbade random train-test splitting or standard k-fold cross-validation. The validation set was partitioned chronologically (the final 28 calendar days as the test horizon, preceding days as training).
> 2. **Causal Lag Features**: All autoregressive lag features ($\text{lag}_7, \text{lag}_{14}, \text{lag}_{21}, \text{lag}_{28}$) were generated prior to window statistics.
> 3. **Shifted Rolling Windows**: Rolling statistics (such as 7-day and 28-day moving averages and standard deviations) were calculated strictly over shifted lag series ($\text{lag}_7$), ensuring that no observation from within the forecast horizon $t$ could inform feature values at time $t$."

### Q3: Explain your SQLite relational data warehouse design. Why use a star schema?
**Model Answer:**
> "We structured the data warehouse as a normalized dimensional Star Schema in SQLite:
> - **Fact Tables**: `fact_sales` (57,390 granular daily transactional records containing `units_sold`, `sell_price`, and calculated `revenue`) and `fact_financials` (daily aggregated revenue, COGS, gross profit, and OPEX).
> - **Dimension Tables**: `dim_calendar` (temporal attributes, Walmart fiscal week tokens, holidays, SNAP assistance flags), `dim_product` (merchandise hierarchy: `item_id`, `dept_id`, `cat_id`), and `dim_store` (`store_id`, `state_id`).
> A star schema minimizes data redundancy, enforces referential integrity through foreign keys, and provides optimal OLAP query performance for complex analytical aggregations, window functions (`LAG()`, `SUM() OVER ()`), and rolling averages."

---

## SECTION 2: Time-Series Econometrics & Statistical Modeling

### Q4: How did you test for stationarity, and what did the Augmented Dickey-Fuller (ADF) test tell you?
**Model Answer:**
> "We applied the Augmented Dickey-Fuller (ADF) test to evaluate the null hypothesis ($H_0$) that the daily sales series contains a unit root (is non-stationary). The test returned an ADF test statistic of $-10.23$ with a $p$-value of $5.08 \times 10^{-18}$, which is orders of magnitude below the $0.05$ critical threshold. Therefore, we rejected $H_0$ and concluded that the series is mean-stationary ($d=0$). However, our ACF diagnostics revealed strong seasonal non-stationarity at 7-day intervals ($s=7$). Consequently, for state-space ARIMA modeling, we applied a seasonal difference ($D=1, s=7$) rather than an integer difference ($d=1$), preserving trend stability while neutralizing weekly cyclicality."

### Q5: How did you interpret the ACF and PACF plots to parameterize SARIMA?
**Model Answer:**
> "The Autocorrelation Function (ACF) exhibited prominent, recurring spikes at exact multiples of 7 ($k = 7, 14, 21, 28$), proving a dominant weekly retail cycle ($s=7$). The Partial Autocorrelation Function (PACF) showed significant cutoffs at lags 1 and 2, along with a seasonal spike at lag 7. This specified candidate autoregressive orders $p=1$ or $2$, seasonal order $P=1$, seasonal differencing $D=1$, and moving average order $q=1$. We finalized the SARIMA specification as $(1, 0, 1) \times (1, 1, 1)_7$, and further augmented it as SARIMAX by adding binary exogenous regressors (`is_weekend`, `has_event`, `snap_CA`)."

### Q6: Why did Seasonal Naive and Holt-Winters outperform standard ARIMA and SES?
**Model Answer:**
> "In daily retail sales, over **52% of total series variance** is driven by day-of-week shopping rhythms (Friday through Sunday volume surges). Simple Exponential Smoothing (SES) and standard non-seasonal ARIMA $(1, 0, 1)$ only model trend and short-term autocorrelation, effectively predicting a flat horizontal line near the series mean (~70 units) across the entire 28-day horizon. They miss the weekend peaks and weekday troughs entirely, resulting in high WAPE (>16.5% to 93%). In contrast, Seasonal Naive explicitly repeats the 7-day pattern, and Holt-Winters fits exponential level, trend, and 7-day additive seasonal indices ($\gamma$), capturing the cyclical peaks and achieving **13.07% WAPE**."

---

## SECTION 3: Machine Learning & Model Evaluation

### Q7: Why did you choose WAPE over RMSE as your primary model selection metric?
**Model Answer:**
> "In commercial demand planning, selecting models solely by RMSE is fundamentally flawed:
> 1. **Quadratic Outlier Distortion**: RMSE squares individual errors ($\sum e_i^2$). In retail, an unexpected promotional surge or holiday spike can produce a single large residual that artificially inflates RMSE, causing the algorithm to reject a model that is otherwise highly accurate on baseline days.
> 2. **Scale and Volume Alignment**: Retail supply chain directors allocate working capital in proportion to total physical units. **WAPE (Weighted Absolute Percentage Error)**:
>    $$\text{WAPE} = \frac{\sum_{t=1}^T |Y_t - \hat{Y}_t|}{\sum_{t=1}^T Y_t} \times 100\%$$
>    normalizes the total absolute error by aggregate demand. It avoids division-by-zero on intermittent zero-demand days (which breaks standard MAPE) and provides an intuitive, executive-friendly measure of inventory allocation error."

### Q8: What is Forecast Bias, and why is tracking signal critical for inventory planning?
**Model Answer:**
> "Forecast Bias measures whether a model systematically over-forecasts or under-forecasts:
> $$\text{Bias} = \frac{\sum (\hat{Y}_t - Y_t)}{\sum Y_t} \times 100\%$$
> A model with a low RMSE could have an asymmetric negative bias (e.g., $-8\%$). In retail, chronic under-forecasting causes immediate stockouts, empty shelves, lost revenue, and customer defection to competitors. Conversely, a positive bias causes inventory bloat, working capital lockup, and margin markdowns to clear obsolete stock. Our champion **Random Forest** model achieved a near-zero tracking signal of **$+1.95\%$**, providing an ideal commercial balance between high availability and lean inventory."

### Q9: How did Random Forest capture retail dynamics compared to SARIMAX?
**Model Answer:**
> "Random Forest achieved **12.53% WAPE** and 9.97 RMSE, slightly outperforming SARIMAX (12.73% WAPE). Random Forest's advantage stems from its ability to model non-linear interactions without rigid parametric assumptions:
> - It learned non-linear decision splits between `is_weekend` and `rolling_mean_7` (e.g., weekend volume surges are larger when the 7-day rolling momentum is already elevated).
> - It utilized feature importance (MDI/Gini) where `wday` and `is_weekend` dominated the trees.
> - Furthermore, Random Forest predictions are bounded by construction within the target range of historical values, preventing negative sales forecasts."

---

## SECTION 4: Financial Analytics, FP&A & Scenario Planning

### Q10: Walk me through your pro-forma Income Statement engine. How do you transition from units to operating profit?
**Model Answer:**
> "The transition from physical demand to financial statements follows a structured pro-forma P&L pipeline:
> 1. **Gross Revenue**: Forecasted units multiplied by shelf selling price ($\text{Revenue} = \text{Units} \times \text{Price}$). Across the 28-day horizon, the champion forecast projected $1,948$ units at $\$10.84$ average price, generating **$\$21,124.46$ in Gross Revenue**.
> 2. **COGS**: Estimated under a transparent baseline direct cost ratio of $60\%$ ($\$12,674.68$).
> 3. **Gross Profit**: $\text{Revenue} - \text{COGS} = \$8,449.78$, delivering a **$40.0\%$ Gross Margin**.
> 4. **Operating Expenses (OPEX)**: Modeled as fixed store daily overhead ($\$80.00/\text{day} \times 28 = \$2,240.00$) plus variable logistics, store labor, and payment processing fees ($10\%$ of revenue = $\$2,112.45$), totaling **$\$4,352.45$ in OPEX**.
> 5. **Operating Profit (EBIT)**: $\text{Gross Profit} - \text{OPEX} = \mathbf{\$4,097.33}$, yielding a **$19.4\%$ Operating Margin**."

### Q11: Explain your 2D Sensitivity Matrix. What strategic insight did it reveal to executive leadership?
**Model Answer:**
> "We constructed a two-way sensitivity matrix evaluating Operating Profit across price adjustments ($\pm 10\%$) and COGS inflation ($55\%$ to $65\%$). The matrix revealed a critical commercial insight regarding operating leverage and price elasticity:
> - Because fixed overhead is constant and COGS represents $60\%$ of sales, a **$5\%$ price reduction** cuts 28-day operating profit from $\$4,097$ to $\$3,776$ ($-7.8\%$).
> - To break even on dollar operating profit after a $5\%$ discount, store sales volume must increase by **$+12.5\%$**.
> - This warned marketing leadership against broad blanket discounts on Class A merchandise, directing promotional markdowns strictly toward highly price-elastic items where volume response can offset margin compression."

### Q12: Why is Revenue not equivalent to Operating Cash Flow in retail FP&A?
**Model Answer:**
> "Revenue recognition occurs at the cash register when the customer takes possession of goods. However, cash flow is dictated by the **Cash Conversion Cycle (CCC)**:
> - **Inflows**: Retail cash and debit purchases settle within $T+1$ days; credit card receivables take 2-3 business days.
> - **Outflows & Working Capital Lag**: Retailers purchase inventory under supplier terms such as Net 30 or Net 60. When sales expand in November/December, revenue surges immediately, while cash disbursements for that inventory are deferred for 30 to 60 days, creating a temporary liquidity surplus. Conversely, during post-holiday volume drawdowns, supplier invoices come due while revenue drops, resulting in negative net operating cash flow. Modeling both unit demand and payment schedules prevents liquidity shortfalls."

---

## SECTION 5: Business Intelligence & Inventory Strategy

### Q13: What is the commercial significance of your Product Pareto Analysis?
**Model Answer:**
> "Our SQL Pareto analysis revealed classic retail concentration: the top **13 SKUs (Class A)** generate **$70\%$ of total revenue ($1,051,596)**. The next 7 SKUs (Class B) contribute $18\%$, and the remaining 10 SKUs (Class C) represent the long tail contributing only $11\%$.
> From an inventory and supply chain standpoint:
> 1. Stockout prevention must be concentrated on Class A items, setting safety stock at $1.65 \times \sigma_{\text{resid}}$ to maintain a $95\%$ service level.
> 2. For Class C items, we recommend moving from scheduled weekly replenishment to a periodic pull reorder point (ROP) model, freeing up estimated trapped working capital of over $\$8,500$."

### Q14: How does your Streamlit application handle performance and model retraining?
**Model Answer:**
> "In enterprise production, interactive web dashboards must never execute expensive training routines on page load. We implemented:
> 1. **Decoupled Architecture**: All statistical models, machine learning algorithms, backtesting runs, and SQL tables are pre-computed in modular training pipelines and serialized to SQLite and optimized CSV/parquet contracts.
> 2. **Session Caching**: The Streamlit application leverages `@st.cache_data` decorators with TTL management to load analytical dataframes in memory upon startup.
> 3. **Sub-Second Latency**: Filtering date horizons, toggling candidate models, or adjusting financial P&L sliders executes vector math in milliseconds, delivering a seamless, corporate-grade user experience."
