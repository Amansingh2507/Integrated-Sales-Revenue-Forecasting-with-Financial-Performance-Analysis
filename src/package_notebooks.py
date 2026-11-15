"""
Utility to package all current notebooks into a single downloadable ZIP archive.
"""

import os
import zipfile
from pathlib import Path

def package_notebooks():
    root = Path(__file__).resolve().parent.parent
    notebooks_dir = root / 'notebooks'
    zip_path = notebooks_dir / 'all_notebooks.zip'

    notebook_files = sorted(list(notebooks_dir.glob("*.ipynb")))
    if not notebook_files:
        print("No notebooks found to package.")
        return

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for nb in notebook_files:
            zipf.write(nb, arcname=nb.name)
            print(f"Added {nb.name} to archive.")

    print(f"\n[SUCCESS] Packaged {len(notebook_files)} notebook(s) into: {zip_path} ({os.path.getsize(zip_path)} bytes)")

if __name__ == "__main__":
    package_notebooks()
