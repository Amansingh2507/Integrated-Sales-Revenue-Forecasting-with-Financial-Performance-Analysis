"""
Unit test to verify that 02_Data_Cleaning_and_Engineering.ipynb
executes cleanly from top to bottom and produces all processed datasets.
"""

import json
from pathlib import Path

def test_notebook_02_execution():
    nb_path = Path("notebooks") / "02_Data_Cleaning_and_Engineering.ipynb"
    assert nb_path.exists(), f"Notebook file {nb_path} does not exist!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [cell["source"] for cell in nb_data["cells"] if cell["cell_type"] == "code"]
    print(f"Total code cells to execute: {len(code_cells)}")

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
        try:
            exec(executable_code, exec_globals)
        except Exception as e:
            print(f"Error in cell {idx}: {e}")
            raise

    print("\n--- Testing verification of generated artifacts ---")
    data_processed = Path("data/processed")
    assert (data_processed / "daily_store_revenue.csv").exists(), "daily_store_revenue.csv was not created!"
    assert (data_processed / "sales_panel_sample.csv").exists(), "sales_panel_sample.csv was not created!"

    print("[SUCCESS] Notebook 02 code executed and artifacts verified!")

if __name__ == "__main__":
    test_notebook_02_execution()
