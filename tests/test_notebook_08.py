"""
Test runner for Notebook 08: executes financial model, scenarios, and sensitivity matrix.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

def test_notebook_08():
    nb_path = Path("notebooks") / "08_Revenue_and_Financial_Analysis.ipynb"
    assert nb_path.exists(), f"Notebook {nb_path} missing!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [cell["source"] for cell in nb_data["cells"] if cell["cell_type"] == "code"]
    print(f"Executing {len(code_cells)} code cells from Notebook 08...")

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
    models_dir = Path("models")
    assert (reports_dir / "08_scenario_summary.csv").exists(), "Missing scenario summary"
    assert (reports_dir / "08_sensitivity_matrix.csv").exists(), "Missing sensitivity matrix"
    assert (reports_dir / "08_scenario_comparison.png").exists(), "Missing scenario chart"
    assert (reports_dir / "08_sensitivity_matrix.png").exists(), "Missing sensitivity chart"
    assert (models_dir / "base_case_financials.csv").exists(), "Missing base case financials"
    print("[SUCCESS] Notebook 08 executed and all financial models verified!")

if __name__ == "__main__":
    test_notebook_08()
