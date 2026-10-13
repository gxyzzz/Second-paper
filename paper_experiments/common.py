from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.msca import MSCA
from modules.attribute import attribute_z
from modules.coliftrec import CoLiftConfig, fit_backgrounds, score_coliftrec
from modules.ranking import metrics_at, rank_by_score, topk_from_embeddings
from pipelines.preference_pipeline import DomainEvaluator, _make_config, _make_data

ROOT = Path(__file__).resolve().parents[1]
SEED = 1000
ALL = ("R10", "N10", "R20", "N20", "R50", "N50")
PRIMARY = ("R10", "N10", "R20", "N20")

CANONICAL = {
    "baby": {
        "b0": ROOT / "diffusion_experiments/round26_preference_directed_diffusion/outputs/seed1000/B0/best.pt",
        "p1": ROOT / "diffusion_experiments/round32_round26_parameter_sweep/outputs/seed1000/M25/best.pt",
    },
    "sports": {
        "b0": ROOT / "diffusion_experiments/round33_seed1000_cross_domain/outputs/sports/B0/best.pt",
        "p1": ROOT / "diffusion_experiments/round33_seed1000_cross_domain/outputs/sports/M25/best.pt",
    },
    "elec": {
        "b0": ROOT / "diffusion_experiments/round33_seed1000_cross_domain/outputs/elec/B0/best.pt",
        "p1": ROOT / "diffusion_experiments/round33_seed1000_cross_domain/outputs/elec/M25/best.pt",
    },
}

EXPECTED_SHA = {
    "baby": {
        "b0": "3d099347a4bbb0ce9df143f68bdb96d2c513fb99852b2e1817f84bf08577c944",
        "p1": "b72786ce4873d659868da950f5d618330cd4fc1486a638eb2cfd8267c18c7820",
    },
    "sports": {
        "b0": "79cdc2fa5c14846cd33718c21cf41f74e1fdd01fc1d9e668b5731407665d1f32",
        "p1": "2b35652573e91f3526d85193baed4987a39414234453bd40b1cf4f9873402fd1",
    },
    "elec": {
        "b0": "695c37330e2b337cf46fbd43a78510c524d0c0bbc84c21d45537d2d5ec0b40b6",
        "p1": "b0ff6e52f718b7dd80db63357f8b5861420b483e87067b20515d26448572a232",
    },
}


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_info(dataset: str, stage: str) -> dict:
    path = CANONICAL[dataset][stage]
    if not path.exists():
        raise FileNotFoundError(path)
    sha = sha256_file(path)
    if sha != EXPECTED_SHA[dataset][stage]:
        raise RuntimeError(f"checkpoint SHA mismatch: {dataset}/{stage}: {sha}")
    pack = torch.load(path, map_location="cpu", weights_only=False)
    if int(pack.get("seed", SEED)) != SEED:
        raise RuntimeError(f"seed mismatch in {path}")
    return {"path": str(path), "sha256": sha, "epoch": int(pack["epoch"]), "pack": pack}


def load_model(dataset: str, stage: str, gpu_id: int = 1):
    info = checkpoint_info(dataset, stage)
    config = _make_config(dataset, gpu_id, ROOT / "paper_experiments" / "_runtime" / dataset)
    _, _, _, train_data, _ = _make_data(config)
    model = MSCA(config, train_data).to(config["device"])
    model.load_state_dict(info["pack"]["model_state_dict"], strict=True)
    model.eval()
    return model, config, info


@torch.no_grad()
def candidate_components(dataset: str, stage: str, split: str = "test", gpu_id: int = 1):
    model, config, info = load_model(dataset, stage, gpu_id)
    evaluator = DomainEvaluator(dataset, model, split=split)
    user_emb, item_emb = model.forward(test=True)
    items, scores = topk_from_embeddings(user_emb, item_emb, evaluator.users, evaluator.histories, 100, 1024)
    pseudo_items, _ = topk_from_embeddings(
        user_emb, item_emb, evaluator.pseudo_users, evaluator.pseudo, 100, 1024
    )
    ztp = evaluator._semantic(evaluator.text, evaluator.text_pseudo, pseudo_items, 128)
    zte = evaluator._semantic(evaluator.text, evaluator.text_full, items, 128)
    zvp = evaluator._semantic(evaluator.visual, evaluator.visual_pseudo, pseudo_items, 64)
    zve = evaluator._semantic(evaluator.visual, evaluator.visual_full, items, 64)
    zap, _ = attribute_z(
        evaluator.mats, evaluator.pseudo_profiles, evaluator.pseudo_users, pseudo_items,
        256, evaluator.attribute_weights,
    )
    zae, _ = attribute_z(
        evaluator.mats, evaluator.full_profiles, evaluator.users, items,
        256, evaluator.attribute_weights,
    )
    backgrounds = fit_backgrounds(pseudo_items, ztp, zap, zvp, evaluator.n_items)
    result = {
        "users": evaluator.users.copy(),
        "sets": evaluator.sets,
        "items": items,
        "scores": scores,
        "pseudo_items": pseudo_items,
        "z_text": zte,
        "z_attribute": zae,
        "z_visual": zve,
        "backgrounds": backgrounds,
        "params": evaluator.params,
        "enabled": evaluator.enabled,
        "checkpoint": {k: v for k, v in info.items() if k != "pack"},
        "split": split,
        "diffusion_inference_calls": 0,
    }
    del model, evaluator, user_emb, item_emb
    torch.cuda.empty_cache(); gc.collect()
    return result


def full_colift_rank(components: dict):
    scores, _ = score_coliftrec(
        components["scores"], components["items"],
        components["z_text"], components["z_attribute"], components["z_visual"],
        components["backgrounds"], components["params"], components["enabled"],
    )
    ranked = rank_by_score(components["items"], scores)
    return ranked, scores


def metrics_for_rank(ranked: np.ndarray, users: np.ndarray, sets: dict) -> dict:
    return metrics_at(ranked, users, sets)


def relative_u4(new: dict, base: dict) -> float:
    return float(np.mean([(new[k] - base[k]) / max(base[k], 1e-12) for k in PRIMARY]))


def dump_json(path: Path | str, payload: dict) -> None:
    def default(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=default) + "\n", encoding="utf-8")
