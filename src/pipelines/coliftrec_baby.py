from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import CoLiftConfig, fit_backgrounds, score_coliftrec
from modules.ranking import (
    metrics_at,
    rank_by_score,
    row_zscore,
    semantic_z_for_candidates,
    sha256_file,
)
from pipelines.msca_assets import build_train_histories_and_validation

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ("R10", "N10", "R20", "N20")
ALL = ("R10", "N10", "R20", "N20", "R50", "N50")


def _metrics(items, scores, users, eval_sets):
    return metrics_at(rank_by_score(items, scores), users, eval_sets)


def run(assets_dir: Path, out_dir: Path, smoke_users: int | None = None,
        config_path: Path | None = None, text_path: Path | None = None,
        image_path: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_path or (ROOT / "src/configs/second_paper.yaml")
    full_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ccfg = full_config["coliftrec"]
    p = CoLiftConfig(
        lambda_text=float(ccfg["text"]["lambda"]),
        lambda_attribute=float(ccfg["attribute"]["lambda"]),
        lambda_visual=float(ccfg["visual"]["lambda"]),
        alpha_text=float(ccfg["text"]["alpha"]),
        alpha_attribute=float(ccfg["attribute"]["alpha"]),
        alpha_visual=float(ccfg["visual"]["alpha"]),
    )
    enabled = {
        "text": bool(ccfg["text"]["enabled"]),
        "attribute": bool(ccfg["attribute"]["enabled"]),
        "visual": bool(ccfg["visual"]["enabled"]),
    }
    asset_audit = json.loads((assets_dir / "audit.json").read_text(encoding="utf-8"))
    if asset_audit.get("TEST_ACCESSED") is not False:
        raise RuntimeError("MSCA asset set is not Validation-only")

    valid_asset = np.load(assets_dir / "validation_top100.npz")
    pseudo_asset = np.load(assets_dir / "train_pseudo_top100.npz")
    valid_users = valid_asset["users"].astype(np.int64)
    valid_items = valid_asset["items"].astype(np.int32)
    valid_scores = valid_asset["scores"].astype(np.float32)
    pseudo_users = pseudo_asset["users"].astype(np.int64)
    pseudo_items = pseudo_asset["items"].astype(np.int32)

    n_users = int(asset_audit["n_users"])
    n_items = int(asset_audit["n_items"])
    histories, pseudo_histories, expected_pseudo_users, expected_valid_users, eval_sets = (
        build_train_histories_and_validation(
            ROOT / "data/baby/baby.inter", n_users
        )
    )
    if not np.array_equal(valid_users, expected_valid_users):
        raise RuntimeError("Validation user order mismatch between assets and canonical split")
    if not np.array_equal(pseudo_users, expected_pseudo_users):
        raise RuntimeError("Pseudo-train user order mismatch between assets and canonical split")

    mode = "formal"
    if smoke_users is not None:
        mode = "smoke"
        valid_users = valid_users[:smoke_users]
        valid_items = valid_items[:smoke_users]
        valid_scores = valid_scores[:smoke_users]
        eval_sets = {int(u): eval_sets[int(u)] for u in valid_users}
        pseudo_limit = min(len(pseudo_users), max(smoke_users * 2, smoke_users))
        pseudo_users = pseudo_users[:pseudo_limit]
        pseudo_items = pseudo_items[:pseudo_limit]

    text_path = text_path or (ROOT / "data/baby/text_feat.npy")
    image_path = image_path or (ROOT / "data/baby/image_feat.npy")
    metadata_path = ROOT / "data/baby/metadata_text_cache.jsonl"

    z_text_train, text_train_audit = semantic_z_for_candidates(
        text_path, pseudo_histories, pseudo_users, pseudo_items, batch_users=256
    )
    z_text_valid, text_valid_audit = semantic_z_for_candidates(
        text_path, histories, valid_users, valid_items, batch_users=256
    )
    z_visual_train, visual_train_audit = semantic_z_for_candidates(
        image_path, pseudo_histories, pseudo_users, pseudo_items, batch_users=128
    )
    z_visual_valid, visual_valid_audit = semantic_z_for_candidates(
        image_path, histories, valid_users, valid_items, batch_users=128
    )

    item_matrices, attribute_audit = build_item_matrices(metadata_path, n_items)
    full_profiles = build_profiles(item_matrices, histories, n_items)
    pseudo_profiles = build_profiles(item_matrices, pseudo_histories, n_items)
    z_attribute_train, _ = attribute_z(
        item_matrices, pseudo_profiles, pseudo_users, pseudo_items, batch=256
    )
    z_attribute_valid, _ = attribute_z(
        item_matrices, full_profiles, valid_users, valid_items, batch=256
    )

    backgrounds = fit_backgrounds(
        pseudo_items,
        z_text_train,
        z_attribute_train,
        z_visual_train,
        n_items,
    )
    z_msca = row_zscore(valid_scores)
    lift_text = row_zscore(
        z_text_valid - p.lambda_text * backgrounds["text"]["shrunk_mean"][valid_items]
    )
    lift_attribute = row_zscore(
        z_attribute_valid
        - p.lambda_attribute * backgrounds["attribute"]["shrunk_mean"][valid_items]
    )
    lift_visual = row_zscore(
        z_visual_valid
        - p.lambda_visual * backgrounds["visual"]["shrunk_mean"][valid_items]
    )

    score_tv, _ = score_coliftrec(
        valid_scores,
        valid_items,
        z_text_valid,
        z_attribute_valid,
        z_visual_valid,
        backgrounds,
        p,
        enabled={"text": enabled["text"], "attribute": False, "visual": enabled["visual"]},
    )
    score_raw_attribute = score_tv + (
        p.alpha_attribute * z_attribute_valid if enabled["attribute"] else 0.0
    )
    score_full, _ = score_coliftrec(
        valid_scores,
        valid_items,
        z_text_valid,
        z_attribute_valid,
        z_visual_valid,
        backgrounds,
        p,
        enabled=enabled,
    )

    metrics = {
        "MSCA_CANONICAL": metrics_at(valid_items, valid_users, eval_sets),
        "MSCA_COLIFTREC_TV": _metrics(valid_items, score_tv, valid_users, eval_sets),
        "MSCA_TV_PLUS_RAW_ATTRIBUTE": _metrics(
            valid_items, score_raw_attribute, valid_users, eval_sets
        ),
        "MSCA_FULL_COLIFTREC_TAV": _metrics(
            valid_items, score_full, valid_users, eval_sets
        ),
    }
    baseline = metrics["MSCA_CANONICAL"]
    full = metrics["MSCA_FULL_COLIFTREC_TAV"]
    tv = metrics["MSCA_COLIFTREC_TV"]
    raw_a = metrics["MSCA_TV_PLUS_RAW_ATTRIBUTE"]

    deltas_vs_msca = {
        name: {k: float(value[k] - baseline[k]) for k in ALL}
        for name, value in metrics.items()
    }
    full_primary_positive = sum(full[k] > baseline[k] for k in PRIMARY)
    attribute_primary_positive = sum(full[k] > tv[k] for k in PRIMARY)
    background_beats_raw = sum(full[k] > raw_a[k] for k in PRIMARY)

    historical = {
        "MSCA_CANONICAL": {
            "R10": 0.06727559, "N10": 0.03675184,
            "R20": 0.10258556, "N20": 0.04572824,
            "R50": 0.16836029, "N50": 0.05889247,
        },
        "MSCA_FULL_COLIFTREC_TAV": {
            "R10": 0.07210594, "N10": 0.03889475,
            "R20": 0.10639759, "N20": 0.04762860,
            "R50": 0.17185389, "N50": 0.06071393,
        },
    }

    summary = {
        "phase": "PHASE2_COLIFTREC_MIGRATION",
        "dataset": "baby",
        "mode": mode,
        "source_checkpoint_sha256": asset_audit["checkpoint_sha256"],
        "source_checkpoint_epoch": asset_audit["checkpoint_epoch"],
        "config_path": str(config_path.resolve()),
        "config": p.to_dict(),
        "branches_enabled": enabled,
        "metrics": metrics,
        "deltas_vs_msca": deltas_vs_msca,
        "full_vs_msca_primary_positive_count": full_primary_positive,
        "full_vs_tv_primary_positive_count": attribute_primary_positive,
        "full_vs_raw_attribute_primary_positive_count": background_beats_raw,
        "historical_reference": historical,
        "FULL_PORTABILITY": "PASS" if full_primary_positive == 4 else "FAIL",
        "ATTRIBUTE_INCREMENT": "PASS" if attribute_primary_positive == 4 else "FAIL",
        "TEST_ACCESSED": False,
        "TEST_USED_FOR_SELECTION": False,
    }

    audit = {
        "inputs": {
            "interaction": {
                "path": str((ROOT / "data/baby/baby.inter").resolve()),
                "sha256": sha256_file(ROOT / "data/baby/baby.inter"),
            },
            "text": text_valid_audit,
            "visual": visual_valid_audit,
            "metadata": attribute_audit,
        },
        "train_semantic_audit": {
            "text": text_train_audit,
            "visual": visual_train_audit,
        },
        "users": {
            "validation": int(len(valid_users)),
            "pseudo_train": int(len(pseudo_users)),
        },
        "candidate_space": {
            "top_l": int(valid_items.shape[1]),
            "validation_shape": list(valid_items.shape),
            "pseudo_shape": list(pseudo_items.shape),
        },
        "background_coverage": {
            name: {
                "items_seen": int(np.sum(block["count"] > 0)),
                "coverage": float(np.mean(block["count"] > 0)),
                "global_mean": float(block["global_mean"]),
            }
            for name, block in backgrounds.items()
        },
        "definitions": {
            "generic_background": "one-global-pseudo-observation item shrinkage over TRAIN pseudo Top100",
            "lift": "row-z(z_ui - lambda_m * mu_i)",
            "score": "row-z(MSCA candidate score) + sum(alpha_m * lift_m)",
            "attribute": "row-z(0.45*z_title + 0.20*z_brand + 0.35*z_description)",
        },
        "TEST_ACCESSED": False,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        out_dir / "backgrounds.npz",
        mu_text=backgrounds["text"]["shrunk_mean"],
        mu_attribute=backgrounds["attribute"]["shrunk_mean"],
        mu_visual=backgrounds["visual"]["shrunk_mean"],
        count_text=backgrounds["text"]["count"],
        count_attribute=backgrounds["attribute"]["count"],
        count_visual=backgrounds["visual"]["count"],
    )
    np.savez_compressed(
        out_dir / "validation_scores.npz",
        users=valid_users,
        items=valid_items,
        msca=valid_scores,
        colift_tv=score_tv.astype(np.float32),
        raw_attribute=score_raw_attribute.astype(np.float32),
        full_coliftrec=score_full.astype(np.float32),
    )
    print(json.dumps(summary, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke-users", type=int)
    parser.add_argument("--text-feature")
    parser.add_argument("--image-feature")
    parser.add_argument(
        "--config",
        default=str(ROOT / "src/configs/second_paper.yaml"),
    )
    args = parser.parse_args()
    result = run(
        Path(args.assets), Path(args.out), args.smoke_users, Path(args.config),
        Path(args.text_feature) if args.text_feature else None,
        Path(args.image_feature) if args.image_feature else None,
    )
    if args.smoke_users is None:
        if result["FULL_PORTABILITY"] != "PASS":
            raise SystemExit(2)
        if result["ATTRIBUTE_INCREMENT"] != "PASS":
            raise SystemExit(3)


if __name__ == "__main__":
    main()
