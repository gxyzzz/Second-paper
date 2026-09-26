from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "src/configs/second_paper"
ALIASES = {"electronics": "elec"}


def canonical_dataset(dataset: str) -> str:
    dataset = str(dataset).strip().lower()
    return ALIASES.get(dataset, dataset)


def load_dataset_config(dataset: str) -> dict:
    dataset = canonical_dataset(dataset)
    path = CONFIG_DIR / f"{dataset}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Second-paper dataset config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("dataset") != dataset:
        raise ValueError(f"dataset config mismatch: requested={dataset}, file={cfg.get('dataset')}")
    cfg["_config_path"] = str(path.resolve())
    resolved = {}
    for name, value in cfg["paths"].items():
        p = Path(value)
        if not p.is_absolute():
            p = ROOT / p
        resolved[name] = p.resolve()
    cfg["resolved_paths"] = resolved
    return cfg
