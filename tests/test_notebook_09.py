"""
Test runner for Notebook 09: executes SQL queries and verifies report exports.
"""

import json
from pathlib import Path

def test_notebook_09():
    nb_path = Path("notebooks") / "09_SQL_and_Business_Analytics.ipynb"
    assert nb_path.exists(), f"Notebook {nb_path} missing!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [cell["source"] for cell in nb_data["cells"] if cell["cell_type"] == "code"]
    print(f"Executing {len(code_cells)} code cells from Notebook 09...")

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
    assert (reports_dir / "sql_01_monthly_mom_growth.csv").exists(), "Missing sql_01"
    assert (reports_dir / "sql_02_rolling_revenue.csv").exists(), "Missing sql_02"
    assert (reports_dir / "sql_03_product_abc_classification.csv").exists(), "Missing sql_03"
    assert (reports_dir / "sql_04_holiday_uplift.csv").exists(), "Missing sql_04"
    assert (reports_dir / "sql_05_snap_impact.csv").exists(), "Missing sql_05"
    print("[SUCCESS] Notebook 09 executed and all SQL reports verified!")

if __name__ == "__main__":
    test_notebook_09()
