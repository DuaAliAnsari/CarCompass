"""Copy the models the API serves into deploy/models/, the folder the Docker image uses.

models/ holds every trained family and is git-ignored. deploy/models/ holds only what
models/manifest.json says to serve (the default model and the switched-on
specialists, about 25 MB) and IS committed, so every Docker build, in CI or on the
hosting platform, contains a working model. Run after training:

    python scripts/export_models.py

It loads the exported models and prices one test car before finishing, so a broken
export is caught here and not in production.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_router import MANIFEST_FILE, ModelRouter  # noqa: E402
from src.model_utils import predict_price  # noqa: E402

TEST_CAR = {"make": "Toyota", "model": "Corolla", "year": 2016, "mileage": 75000}


def export(src: Path, dest: Path) -> dict:
    manifest = json.loads((src / MANIFEST_FILE).read_text())
    files = [MANIFEST_FILE]
    for name in manifest["served"]:
        entry = manifest["families"][name]
        files += [entry["file"], entry["meta"]]

    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.iterdir():
        if old.is_file() and old.name not in files and old.name != ".gitkeep":
            old.unlink()  # a model that is no longer served
    for name in files:
        shutil.copy(src / name, dest / name)

    router = ModelRouter.load(dest)
    for year in (2005, 2016):
        name, art = router.choose(year)
        price = predict_price(art, {**TEST_CAR, "year": year})["estimated_price_pkr"]
        print(f"check: {year} Toyota Corolla -> {name}: PKR {price:,}")
    size = sum((dest / f).stat().st_size for f in files) / 1e6
    print(f"Exported {', '.join(manifest['served'])} ({size:.1f} MB) to {dest}")
    return manifest


def main():
    p = argparse.ArgumentParser(description="Export the served models for Docker.")
    p.add_argument("--src", default="models")
    p.add_argument("--dest", default="deploy/models")
    args = p.parse_args()
    export(Path(args.src), Path(args.dest))


if __name__ == "__main__":
    main()
