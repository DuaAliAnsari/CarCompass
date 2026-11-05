"""Download the full datasets (they are NOT committed to git).

1. pip install kaggle   and put your API token in ~/.kaggle/kaggle.json
   (Kaggle -> Settings -> API -> Create New Token)
2. python scripts/download_data.py

Dataset A is saved to data/raw/pakistan_used_cars.csv (used by make prepare-real) and
Dataset B to data/raw/usedCars.json; make train-multi needs both.
"""
import shutil
import subprocess
from pathlib import Path

# name: (Kaggle slug, file pattern inside the download, where the pipeline expects it)
DATASETS = {
    # Dataset A: https://www.kaggle.com/datasets/muhamedumarjamil/pakistan-used-cars-dataset
    "pakistan_used_cars": ("muhamedumarjamil/pakistan-used-cars-dataset", "*.csv",
                           "data/raw/pakistan_used_cars.csv"),
    # Dataset B: https://www.kaggle.com/datasets/asimzahid/pakistans-largest-pakwheels-automobiles-listings
    "pakwheels_listings": ("asimzahid/pakistans-largest-pakwheels-automobiles-listings",
                           "*.json", "data/raw/usedCars.json"),
}

for name, (slug, pattern, dest) in DATASETS.items():
    target = Path("data/raw") / name
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(target),
                    "--unzip"], check=True)
    files = sorted(target.rglob(pattern))
    if not files:
        raise SystemExit(f"No {pattern} file found in {target}")
    shutil.copy(files[0], dest)
    print(f"Copied {files[0].name} -> {dest}")
