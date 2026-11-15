"""
Test runner for Notebook 04: executes all code cells and verifies decomposition outputs.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

def test_notebook_04():
    nb_path = Path("notebooks") / "04_Time_Series_Decomposition.ipynb"
    assert nb_path.exists(), f"Notebook {nb_path} missing!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [cell["source"] for cell in nb_data["cells"] if cell["cell_type"] == "code"]
    print(f"Executing {len(code_cells)} code cells from Notebook 04...")

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
    expected_files = [
        '04_rolling_statistics.png',
        '04_adf_test_results.csv',
        '04_stl_decomposition.png',
        '04_acf_pacf_analysis.png'
    ]

    for f in expected_files:
        assert (reports_dir / f).exists(), f"Missing file: {f}"
    print(f"[SUCCESS] All decomposition artifacts verified in {reports_dir}!")

if __name__ == "__main__":
    test_notebook_04()
