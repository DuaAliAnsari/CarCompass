"""Download the full dataset (it is NOT committed to git).

1. pip install kaggle   and put your API token in ~/.kaggle/kaggle.json
   (Kaggle -> Settings -> API -> Create New Token)
2. python scripts/download_data.py

Dataset A is saved to data/raw/pakistan_used_cars.csv, the path make prepare-real expects.
Dataset B is only needed for the optional merged-data experiment, so it is skipped
until you fill in its slug.
"""
import shutil
import subprocess
from pathlib import Path

DATASETS = {
    # Dataset A: https://www.kaggle.com/datasets/muhamedumarjamil/pakistan-used-cars-dataset
    "pakistan_used_cars": "muhamedumarjamil/pakistan-used-cars-dataset",
    "pakwheels_listings": "OWNER/SLUG",    # Dataset B (asimzahid), optional, fill in
}

for name, slug in DATASETS.items():
    if slug == "OWNER/SLUG":
        print(f"Skipping {name}: set its slug in scripts/download_data.py")
        continue
    target = Path("data/raw") / name
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(target),
                    "--unzip"], check=True)
    if name == "pakistan_used_cars":
        csvs = sorted(target.glob("*.csv"))
        if not csvs:
            raise SystemExit(f"No CSV found in {target}")
        shutil.copy(csvs[0], Path("data/raw/pakistan_used_cars.csv"))
        print(f"Copied {csvs[0].name} -> data/raw/pakistan_used_cars.csv")
