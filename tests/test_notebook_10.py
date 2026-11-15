"""
Test runner for Notebook 10: executes final integration, artifact exports, and zip packaging.
"""

import json
from pathlib import Path

def test_notebook_10():
    nb_path = Path("notebooks") / "10_Final_Project_Integration_and_Export.ipynb"
    assert nb_path.exists(), f"Notebook {nb_path} missing!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [cell["source"] for cell in nb_data["cells"] if cell["cell_type"] == "code"]
    print(f"Executing {len(code_cells)} code cells from Notebook 10...")

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
    notebooks_dir = Path("notebooks")
    assert (models_dir / "final_forecast_outputs.csv").exists(), "Missing final_forecast_outputs.csv"
    assert (models_dir / "master_model_scorecard.csv").exists(), "Missing master_model_scorecard.csv"
    assert (models_dir / "final_financial_projections.csv").exists(), "Missing final_financial_projections.csv"
    assert (notebooks_dir / "all_notebooks.zip").exists(), "Missing all_notebooks.zip"
    print("[SUCCESS] Notebook 10 executed and all final production artifacts verified!")

if __name__ == "__main__":
    test_notebook_10()
