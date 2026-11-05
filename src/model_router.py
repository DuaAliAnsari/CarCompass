"""Pick which price model answers a request, using models/manifest.json.

src/train_families.py writes the manifest: the default model plus the era
specialists that beat it on validation rows. The router only follows it:
a request goes to an enabled specialist whose years cover the car, otherwise to the
default. If a specialist's files are missing it is skipped, so the default always
answers. Without a manifest the router falls back to the single model in
models/price_model.joblib (the older one-model setup).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib

from src.model_utils import Artifacts, load_artifacts

MANIFEST_FILE = "manifest.json"


def load_family(model_dir, entry: dict) -> Artifacts:
    """Load one family's model and metadata. Raises FileNotFoundError if missing."""
    model_dir = Path(model_dir)
    model_path, meta_path = model_dir / entry["file"], model_dir / entry["meta"]
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing {model_path.name} or {meta_path.name}.")
    return Artifacts(pipeline=joblib.load(model_path),
                     meta=json.loads(meta_path.read_text()))


def covers(rule: dict, year: int) -> bool:
    return rule.get("year_min", year) <= year <= rule.get("year_max", year)


@dataclass
class ModelRouter:
    default_name: str
    models: dict  # family name -> Artifacts
    specialists: dict = field(default_factory=dict)  # family name -> year rule
    manifest: dict | None = None

    @classmethod
    def load(cls, model_dir) -> "ModelRouter":
        """Raises FileNotFoundError when neither a manifest nor a model exists."""
        model_dir = Path(model_dir)
        path = model_dir / MANIFEST_FILE
        if not path.exists():
            art = load_artifacts(model_dir)
            return cls(default_name=art.meta.get("family", "price_model"),
                       models={art.meta.get("family", "price_model"): art})

        manifest = json.loads(path.read_text())
        default = manifest["default"]
        models = {default: load_family(model_dir, manifest["families"][default])}
        specialists = {}
        for name, spec in manifest["routing"]["specialists"].items():
            if not spec.get("enabled"):
                continue
            try:
                models[name] = load_family(model_dir, manifest["families"][name])
            except (FileNotFoundError, KeyError):
                continue  # the default model answers these years instead
            specialists[name] = {k: spec[k] for k in ("year_min", "year_max")
                                 if k in spec}
        return cls(default_name=default, models=models, specialists=specialists,
                   manifest=manifest)

    def choose(self, year: int) -> tuple[str, Artifacts]:
        for name, rule in self.specialists.items():
            if covers(rule, year):
                return name, self.models[name]
        return self.default_name, self.models[self.default_name]

    def describe(self) -> dict:
        """What GET /models shows: loaded models, their versions and test scores."""
        out = {"default": self.default_name, "models": {}}
        for name, art in self.models.items():
            meta = art.meta
            out["models"][name] = {
                "version": meta.get("version"),
                "algorithm": meta.get("model"),
                "run_name": meta.get("run_name"),
                "years": self.specialists.get(name) or (
                    "all other years" if self.specialists else "all years"),
                "test_mae_pkr": round(meta.get("metrics", {}).get("mae_pkr", 0)),
            }
        if self.manifest:
            out["manifest_version"] = self.manifest.get("version")
        return out
