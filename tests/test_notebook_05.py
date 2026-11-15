"""
Test runner for Notebook 05: executes all statistical models and verifies forecasts.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

def test_notebook_05():
    nb_path = Path("notebooks") / "05_Statistical_Forecasting.ipynb"
    assert nb_path.exists(), f"Notebook {nb_path} missing!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [cell["source"] for cell in nb_data["cells"] if cell["cell_type"] == "code"]
    print(f"Executing {len(code_cells)} code cells from Notebook 05...")

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

    models_dir = Path("models")
    reports_dir = Path("reports")
    assert (models_dir / "statistical_forecasts.csv").exists(), "Missing statistical_forecasts.csv"
    assert (reports_dir / "05_statistical_metrics.csv").exists(), "Missing 05_statistical_metrics.csv"
    assert (reports_dir / "05_statistical_forecasts_comparison.png").exists(), "Missing chart"
    print("[SUCCESS] All statistical models executed and verified!")

if __name__ == "__main__":
    test_notebook_05()
