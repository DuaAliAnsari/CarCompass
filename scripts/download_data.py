"""Download the full datasets (they are NOT committed to git).

1. pip install kaggle   and put your API token in ~/.kaggle/kaggle.json
2. Fill in the dataset slugs below (the part of the Kaggle URL after kaggle.com/datasets/).
3. python scripts/download_data.py
"""
import subprocess
from pathlib import Path

DATASETS = {
    "pakwheels_used_cars": "OWNER/SLUG",   # Dataset A (muhamedumarjamil), fill in
    "pakwheels_listings": "OWNER/SLUG",    # Dataset B (asimzahid), fill in
}

for name, slug in DATASETS.items():
    if slug == "OWNER/SLUG":
        print(f"Skipping {name}: set its slug in scripts/download_data.py")
        continue
    target = Path("data/raw") / name
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(target),
                    "--unzip"], check=True)
