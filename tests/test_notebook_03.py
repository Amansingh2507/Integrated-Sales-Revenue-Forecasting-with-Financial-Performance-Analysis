"""
Test runner for Notebook 03: executes all code cells and verifies all 10 charts.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

def test_notebook_03():
    nb_path = Path("notebooks") / "03_Exploratory_Data_Analysis.ipynb"
    assert nb_path.exists(), f"Notebook {nb_path} missing!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [cell["source"] for cell in nb_data["cells"] if cell["cell_type"] == "code"]
    print(f"Executing {len(code_cells)} code cells from Notebook 03...")

    exec_globals = {
        '__name__': '__main__',
        'display': lambda x: print(x)
    }

    for idx, cell_lines in enumerate(code_cells, start=1):
        cell_code = "".join(cell_lines)
        cleaned_lines = []
        for line in cell_code.splitlines():
            if line.strip().startswith('!') or line.strip().startswith('%'):
                cleaned_lines.append("# " + line)
            else:
                cleaned_lines.append(line)
        executable_code = "\n".join(cleaned_lines)
        print(f"Executing cell {idx}...")
        exec(executable_code, exec_globals)

    reports_dir = Path("reports")
    expected_charts = [
        'eda_01_daily_weekly_sales_trend.png',
        'eda_02_monthly_revenue_trajectory.png',
        'eda_03_day_of_week_seasonality.png',
        'eda_04_monthly_seasonality_heatmap.png',
        'eda_05_product_pareto_curve.png',
        'eda_06_top10_revenue_generating_products.png',
        'eda_07_holiday_and_event_uplift.png',
        'eda_08_snap_disbursement_impact.png',
        'eda_09_price_vs_demand_scatter.png',
        'eda_10_quarterly_financial_trends.png'
    ]

    for chart in expected_charts:
        assert (reports_dir / chart).exists(), f"Missing chart: {chart}"
    print(f"[SUCCESS] All 10 EDA charts generated and verified in {reports_dir}!")

if __name__ == "__main__":
    test_notebook_03()
