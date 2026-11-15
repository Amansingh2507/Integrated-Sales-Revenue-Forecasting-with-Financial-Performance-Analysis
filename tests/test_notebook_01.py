"""
Unit test to verify that 01_Project_Setup_and_Data_Understanding.ipynb
executes cleanly from top to bottom and produces all required artifacts.
"""

import json
import sys
from pathlib import Path

def test_notebook_01_execution():
    nb_path = Path("notebooks") / "01_Project_Setup_and_Data_Understanding.ipynb"
    assert nb_path.exists(), f"Notebook file {nb_path} does not exist!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [cell["source"] for cell in nb_data["cells"] if cell["cell_type"] == "code"]
    print(f"Total code cells to execute: {len(code_cells)}")

    # Execute in a shared global namespace
    exec_globals = {
        '__name__': '__main__',
        'display': lambda x: print(x)
    }

    # Set matplotlib backend to Agg to avoid display popups
    import matplotlib
    matplotlib.use('Agg')

    for idx, cell_lines in enumerate(code_cells, start=1):
        cell_code = "".join(cell_lines)
        # Filter out ipython magic lines like !pip or %
        cleaned_lines = []
        for line in cell_code.splitlines():
            if line.strip().startswith('!') or line.strip().startswith('%'):
                cleaned_lines.append("# " + line)
            else:
                cleaned_lines.append(line)
        executable_code = "\n".join(cleaned_lines)
        print(f"Executing cell {idx}...")
        try:
            exec(executable_code, exec_globals)
        except Exception as e:
            print(f"Error in cell {idx}: {e}")
            raise

    print("\n--- Testing verification of generated artifacts ---")
    data_raw = Path("data/raw")
    reports_dir = Path("reports")
    
    assert (data_raw / "calendar.csv").exists(), "calendar.csv was not created!"
    assert (data_raw / "sell_prices.csv").exists(), "sell_prices.csv was not created!"
    assert (data_raw / "sales_train_validation.csv").exists(), "sales_train_validation.csv was not created!"
    assert (data_raw / "sales_pilot_scoped.csv").exists(), "sales_pilot_scoped.csv was not created!"
    assert (reports_dir / "data_dictionary.csv").exists(), "data_dictionary.csv was not created!"
    assert (reports_dir / "data_quality_report.csv").exists(), "data_quality_report.csv was not created!"
    assert (reports_dir / "01_aggregate_daily_demand.png").exists(), "Chart 1 was not created!"
    assert (reports_dir / "01_intermittency_and_prices.png").exists(), "Chart 2 was not created!"

    print("[SUCCESS] All notebook code cells executed successfully and all artifacts verified!")

if __name__ == "__main__":
    test_notebook_01_execution()
