# Streamlit Community Cloud Deployment Guide

This guide walks you through deploying the **Enterprise Retail Forecasting & Financial Analytics Web Platform** to Streamlit Community Cloud (`https://share.streamlit.io/`).

---

## 1. Prerequisites
- A free **GitHub account** ([github.com](https://github.com)).
- A free **Streamlit Community Cloud account** ([share.streamlit.io](https://share.streamlit.io)).

---

## 2. Step-by-Step Deployment Instructions

### Step 1: Initialize and Push to Your GitHub Repository
1. Open terminal in the project directory:
   ```bash
   cd "C:\Users\lenovo\OneDrive\Desktop\Forecast Project"
   ```
2. Verify git status:
   ```bash
   git status
   ```
3. Create a new public repository on GitHub named:
   `business-financial-forecasting`
4. Link your local repository to GitHub and push the master branch:
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/business-financial-forecasting.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Verify Repository Configuration on GitHub
Confirm that your GitHub repository contains:
- `requirements.txt` in the root directory.
- `dashboard/app.py` (main entry point) and `dashboard/pages/` (the 6 modular dashboard views).
- `data/processed/retail_analytics.db` and `data/processed/daily_store_revenue.csv`.
- `models/final_forecast_outputs.csv` and `models/master_model_scorecard.csv`.
- `notebooks/all_notebooks.zip` (for one-click download).

### Step 3: Connect to Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub credentials.
2. Click **"New app"** (or **"Create app"**).
3. In the deployment configuration dialog, specify:
   - **Repository**: `<YOUR_GITHUB_USERNAME>/business-financial-forecasting`
   - **Branch**: `main` (or `master`)
   - **Main file path**: `dashboard/app.py`
4. Click **"Advanced settings"** (optional):
   - Python version: Select **3.10** or **3.11** (standard for Streamlit Cloud).
5. Click **"Deploy!"**.

### Step 4: Monitor Deployment Logs
1. Click the **"Manage app"** button in the lower-right corner to open the real-time build logs.
2. Streamlit Cloud will:
   - Clone your repository.
   - Install packages specified in `requirements.txt`.
   - Launch `dashboard/app.py`.
3. Startup completes within 60-90 seconds.

### Step 5: Test and Verify Every Page in Production
Verify the following pages in the live browser:
- **Main Home Page**: High-level KPI cards render properly; ZIP download button delivers `m5_forecasting_financial_notebooks.zip`.
- **Page 1 (Executive Overview)**: Plotly revenue trend and top 5 SKUs render smoothly.
- **Page 2 (Sales Analytics)**: Day-of-week seasonality and holiday uplifts load without lag.
- **Page 3 (Demand & Revenue Forecasting)**: Toggle models (Random Forest, SARIMAX, Holt-Winters); verify 95% prediction interval shading and CSV forecast download.
- **Page 4 (Financial Analysis)**: Adjust COGS slider (e.g., from 60% to 55%) and confirm real-time P&L table recalculation.
- **Page 5 (Business Insights)**: Pareto 80/20 curve and ABC summary display accurately.
- **Page 6 (Model Performance)**: Benchmark table highlights the top champion model.

---

## 3. Alternative Hosting Options (If Needed)
If you require private enterprise hosting or containerized deployment:
- **Docker**: Build a lightweight container:
  ```dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8501
  CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```
- **Hugging Face Spaces**: Select Streamlit SDK and connect your GitHub repository.
- **Railway / Render**: Deploy as a Python web service using start command:
  `streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0`
