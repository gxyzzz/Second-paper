from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import CoLiftConfig, fit_backgrounds, score_coliftrec
from modules.ranking import metrics_at, rank_by_score, sha256_file, semantic_z_for_candidates
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation

PRIMARY = ("R10", "N10", "R20", "N20")
ALL = ("R10", "N10", "R20", "N20", "R50", "N50")


def _params(ccfg):
    return CoLiftConfig(
        lambda_text=float(ccfg["text"]["lambda"]),
        lambda_attribute=float(ccfg["attribute"]["lambda"]),
        lambda_visual=float(ccfg["visual"]["lambda"]),
        alpha_text=float(ccfg["text"]["alpha"]),
        alpha_attribute=float(ccfg["attribute"]["alpha"]),
        alpha_visual=float(ccfg["visual"]["alpha"]),
    )


def _metrics(items, scores, users, eval_sets):
    return metrics_at(rank_by_score(items, scores), users, eval_sets)


def run(dataset: str, assets_dir: Path, out_dir: Path, smoke_users: int | None = None,
        text_path: Path | None = None, visual_path: Path | None = None) -> dict:
    cfg = load_dataset_config(dataset)
    dataset = cfg["dataset"]
    paths = cfg["resolved_paths"]
    ccfg = cfg["coliftrec"]
    p = _params(ccfg)
    enabled = {m: bool(ccfg[m]["enabled"]) for m in ("text", "attribute", "visual")}
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_audit = json.loads((assets_dir / "audit.json").read_text(encoding="utf-8"))
    if asset_audit.get("TEST_ACCESSED") is not False:
        raise RuntimeError("MSCA asset set is not Validation-only")
    if asset_audit.get("dataset") != dataset:
        raise RuntimeError(f"asset dataset mismatch: {asset_audit.get('dataset')} != {dataset}")

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
        build_train_histories_and_validation(paths["interaction"], n_users)
    )
    if not np.array_equal(valid_users, expected_valid_users):
        raise RuntimeError("Validation user order mismatch")
    if not np.array_equal(pseudo_users, expected_pseudo_users):
        raise RuntimeError("Pseudo-train user order mismatch")

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

    text_path = Path(text_path) if text_path else paths["text_feature"]
    visual_path = Path(visual_path) if visual_path else paths["visual_feature"]
    metadata_path = paths["metadata"]

    z_text_train, text_train_audit = semantic_z_for_candidates(
        text_path, pseudo_histories, pseudo_users, pseudo_items, batch_users=256
    )
    z_text_valid, text_valid_audit = semantic_z_for_candidates(
        text_path, histories, valid_users, valid_items, batch_users=256
    )
    z_visual_train, visual_train_audit = semantic_z_for_candidates(
        visual_path, pseudo_histories, pseudo_users, pseudo_items, batch_users=128
    )
    z_visual_valid, visual_valid_audit = semantic_z_for_candidates(
        visual_path, histories, valid_users, valid_items, batch_users=128
    )

    acfg = ccfg["attribute"]
    item_matrices, attribute_audit = build_item_matrices(
        metadata_path, n_items,
        min_df=int(acfg.get("tfidf_min_df", 2)),
        max_df=float(acfg.get("tfidf_max_df", 0.8)),
        description_len=int(acfg.get("description_len", 128)),
    )
    full_profiles = build_profiles(item_matrices, histories, n_items)
    pseudo_profiles = build_profiles(item_matrices, pseudo_histories, n_items)
    z_attr_train, _ = attribute_z(item_matrices, pseudo_profiles, pseudo_users, pseudo_items, batch=256)
    z_attr_valid, _ = attribute_z(item_matrices, full_profiles, valid_users, valid_items, batch=256)

    backgrounds = fit_backgrounds(
        pseudo_items, z_text_train, z_attr_train, z_visual_train, n_items
    )
    score_tv, _ = score_coliftrec(
        valid_scores, valid_items, z_text_valid, z_attr_valid, z_visual_valid,
        backgrounds, p,
        enabled={"text": enabled["text"], "attribute": False, "visual": enabled["visual"]},
    )
    score_raw_attribute = score_tv + (
        p.alpha_attribute * z_attr_valid if enabled["attribute"] else 0.0
    )
    score_full, _ = score_coliftrec(
        valid_scores, valid_items, z_text_valid, z_attr_valid, z_visual_valid,
        backgrounds, p, enabled=enabled,
    )
    p_zero = CoLiftConfig(
        lambda_text=p.lambda_text,
        lambda_attribute=p.lambda_attribute,
        lambda_visual=p.lambda_visual,
        alpha_text=0.0,
        alpha_attribute=0.0,
        alpha_visual=0.0,
    )
    score_identity, _ = score_coliftrec(
        valid_scores, valid_items, z_text_valid, z_attr_valid, z_visual_valid,
        backgrounds, p_zero, enabled=enabled,
    )
    identity_rank = rank_by_score(valid_items, score_identity)
    identity_metrics = metrics_at(identity_rank, valid_users, eval_sets)
    baseline_rank = rank_by_score(valid_items, valid_scores)
    identity_gate = {
        "ranking_ids_exact": bool(np.array_equal(identity_rank, baseline_rank)),
        "metrics_exact": identity_metrics == metrics_at(valid_items, valid_users, eval_sets),
        "score_coordinate": "row-z(MSCA candidate scores)",
    }
    identity_gate["PASS"] = bool(
        identity_gate["ranking_ids_exact"] and identity_gate["metrics_exact"]
    )
    if not identity_gate["PASS"]:
        raise RuntimeError(f"alpha=0 CoLiftRec identity failed: {identity_gate}")

    metrics = {
        "MSCA_CANONICAL": metrics_at(valid_items, valid_users, eval_sets),
        "MSCA_COLIFTREC_TV": _metrics(valid_items, score_tv, valid_users, eval_sets),
        "MSCA_TV_PLUS_RAW_ATTRIBUTE": _metrics(valid_items, score_raw_attribute, valid_users, eval_sets),
        "MSCA_FULL_COLIFTREC_TAV": _metrics(valid_items, score_full, valid_users, eval_sets),
    }
    baseline, full = metrics["MSCA_CANONICAL"], metrics["MSCA_FULL_COLIFTREC_TAV"]
    tv, raw_a = metrics["MSCA_COLIFTREC_TV"], metrics["MSCA_TV_PLUS_RAW_ATTRIBUTE"]
    deltas = {
        name: {k: float(value[k] - baseline[k]) for k in ALL}
        for name, value in metrics.items()
    }
    summary = {
        "phase": "COLIFTREC_VALIDATION",
        "dataset": dataset,
        "display_name": cfg.get("display_name", dataset),
        "mode": mode,
        "source_checkpoint_sha256": asset_audit["checkpoint_sha256"],
        "source_checkpoint_epoch": int(asset_audit["checkpoint_epoch"]),
        "dataset_config": cfg["_config_path"],
        "config": p.to_dict(),
        "branches_enabled": enabled,
        "metrics": metrics,
        "deltas_vs_msca": deltas,
        "full_vs_msca_primary_positive_count": int(sum(full[k] > baseline[k] for k in PRIMARY)),
        "full_vs_tv_primary_positive_count": int(sum(full[k] > tv[k] for k in PRIMARY)),
        "full_vs_raw_attribute_primary_positive_count": int(sum(full[k] > raw_a[k] for k in PRIMARY)),
        "alpha_zero_identity": identity_gate,
        "TEST_ACCESSED": False,
        "TEST_USED_FOR_SELECTION": False,
    }
    audit = {
        "dataset": dataset,
        "inputs": {
            "interaction": {"path": str(paths["interaction"]), "sha256": sha256_file(paths["interaction"])},
            "text": text_valid_audit,
            "visual": visual_valid_audit,
            "metadata": attribute_audit,
        },
        "train_semantic_audit": {"text": text_train_audit, "visual": visual_train_audit},
        "users": {"validation": int(len(valid_users)), "pseudo_train": int(len(pseudo_users))},
        "n_users": n_users,
        "n_items": n_items,
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
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-users", type=int)
    ap.add_argument("--text-feature")
    ap.add_argument("--visual-feature")
    args = ap.parse_args()
    run(
        args.dataset,
        Path(args.assets),
        Path(args.out),
        args.smoke_users,
        Path(args.text_feature) if args.text_feature else None,
        Path(args.visual_feature) if args.visual_feature else None,
    )


if __name__ == "__main__":
    main()
