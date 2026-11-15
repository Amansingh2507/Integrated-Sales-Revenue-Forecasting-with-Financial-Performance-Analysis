"""
Test runner for Notebook 07: executes backtesting, scorecard, and residual diagnostics.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

def test_notebook_07():
    nb_path = Path("notebooks") / "07_Model_Evaluation_and_Selection.ipynb"
    assert nb_path.exists(), f"Notebook {nb_path} missing!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [cell["source"] for cell in nb_data["cells"] if cell["cell_type"] == "code"]
    print(f"Executing {len(code_cells)} code cells from Notebook 07...")

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
    assert (reports_dir / "07_master_model_scorecard.csv").exists(), "Missing master scorecard"
    assert (reports_dir / "07_rolling_backtest_matrix.csv").exists(), "Missing backtest matrix"
    assert (reports_dir / "07_residual_diagnostics.png").exists(), "Missing residual diagnostics chart"
    assert (reports_dir / "07_champion_forecast_intervals.png").exists(), "Missing intervals chart"
    print("[SUCCESS] Notebook 07 executed and all evaluation artifacts verified!")

if __name__ == "__main__":
    test_notebook_07()
