"""
package_submission.py - Creates the single standalone ZIP file for Google Classroom submission,
strictly following all course guidelines and rubric requirements.
"""

import zipfile
from pathlib import Path

def package_project(output_zip: str = "WattWise_AI_Final_Submission.zip"):
    root_dir = Path(".")
    zip_path = Path(output_zip)
    if zip_path.exists():
        zip_path.unlink()

    # Exclusions
    exclude_prefixes = {
        ".git",
        ".venv",
        ".pytest_cache",
        "__pycache__",
        ".ipynb_checkpoints",
    }

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in root_dir.rglob("*"):
            if not file.is_file():
                continue
            
            # Check exclusions
            parts = file.relative_to(root_dir).parts
            if any(p.startswith(tuple(exclude_prefixes)) for p in parts):
                continue
            if file.name == output_zip:
                continue

            arcname = str(file.relative_to(root_dir))
            zipf.write(file, arcname)
            file_count += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Packaged {file_count} files into '{output_zip}' ({size_mb:.2f} MB)")
    return zip_path

if __name__ == "__main__":
    package_project()
