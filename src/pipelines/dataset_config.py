from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG_DIR = ROOT / "src/configs/dataset"
METHOD_CONFIG_PATH = ROOT / "src/configs/model/CoLiftRecDiffusion.yaml"
ALIASES = {"electronics": "elec"}
DISPLAY_NAMES = {"baby": "Baby", "sports": "Sports", "elec": "Electronics"}


def canonical_dataset(dataset: str) -> str:
    dataset = str(dataset).strip().lower()
    return ALIASES.get(dataset, dataset)


def load_publication_method_config() -> dict:
    cfg = yaml.safe_load(METHOD_CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("model") != "CoLiftRecDiffusion":
        raise ValueError(f"unexpected publication method config: {METHOD_CONFIG_PATH}")
    return cfg


def _dataset_paths(dataset: str, source_cfg: dict) -> dict[str, str]:
    base = ROOT / "data" / dataset
    required = {
        "interaction": source_cfg.get("inter_file_name"),
        "text_feature": source_cfg.get("text_feature_file"),
        "visual_feature": source_cfg.get("vision_feature_file"),
        "metadata": source_cfg.get("metadata_file"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"dataset config missing publication paths {missing}: {dataset}")
    return {k: str((base / v).resolve()) for k, v in required.items()}


def _runtime_diffusion(method_diff: dict) -> dict:
    """Return a runtime-compatible view without persisting any search space."""
    d = deepcopy(method_diff)
    beta = float(d["beta"])
    d["beta_candidates"] = [beta]
    d["t_edit_candidates"] = [int(d["t_edit"])]
    d["guidance_candidates"] = [float(d["guidance"])]
    d["rho_text_candidates"] = [float(d["rho_text"])]
    d["rho_visual_candidates"] = [float(d["rho_visual"])]
    if d["training_protocol"] == "m32_train_monitor":
        # Historical trainer derives the actual seed as seed_base + int(beta*1000).
        d["train_seed_base"] = int(d["training_seed"]) - int(beta * 1000)
    return d


def load_dataset_config(dataset: str) -> dict:
    dataset = canonical_dataset(dataset)
    dataset_path = DATASET_CONFIG_DIR / f"{dataset}.yaml"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset config not found: {dataset_path}")

    source_cfg = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    method_cfg = load_publication_method_config()
    if dataset not in method_cfg["datasets"]:
        raise KeyError(f"publication method config has no dataset={dataset}")

    common = method_cfg["common"]
    dataset_method = method_cfg["datasets"][dataset]
    colift = deepcopy(dataset_method["coliftrec"])
    attr_common = deepcopy(common["attribute"])
    colift["attribute"] = {**attr_common, **colift["attribute"]}
    for name in ("text", "attribute", "visual"):
        colift[name]["enabled"] = True
    colift["top_l"] = int(common["top_l"])

    cfg = {
        "dataset": dataset,
        "display_name": DISPLAY_NAMES.get(dataset, dataset),
        "paths": _dataset_paths(dataset, source_cfg),
        "coliftrec": colift,
        "diffusion": _runtime_diffusion(dataset_method["diffusion"]),
        "backbone": deepcopy(method_cfg["backbone"]),
        "_config_path": str(METHOD_CONFIG_PATH.resolve()),
        "_method_config_path": str(METHOD_CONFIG_PATH.resolve()),
        "_dataset_config_path": str(dataset_path.resolve()),
    }
    cfg["resolved_paths"] = {k: Path(v).resolve() for k, v in cfg["paths"].items()}
    return cfg
