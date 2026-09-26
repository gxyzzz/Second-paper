from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import CoLiftConfig, fit_backgrounds, score_coliftrec
from modules.ranking import metrics_at, rank_by_score, semantic_z_for_candidates, sha256_file, topk_from_embeddings
from pipelines.msca_assets import build_train_histories_and_validation

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ("R10", "N10", "R20", "N20")
ALL = ("R10", "N10", "R20", "N20", "R50", "N50")


def build_test_eval(inter_path: Path, n_users: int):
    df = pd.read_csv(inter_path, sep="\t", usecols=["userID", "itemID", "timestamp", "x_label"])
    df["_row"] = np.arange(len(df), dtype=np.int64)
    train = df[df.x_label == 0].copy().sort_values(["userID", "timestamp", "_row"], kind="stable")
    histories = [[] for _ in range(n_users)]
    for user, group in train.groupby("userID", sort=False):
        histories[int(user)] = group.itemID.astype(np.int64).tolist()
    train_users = {u for u, h in enumerate(histories) if h}
    test = df[(df.x_label == 2) & df.userID.isin(train_users)].copy()
    test_users = test.userID.drop_duplicates().astype(np.int64).to_numpy()
    test_sets = {int(u): set(g.itemID.astype(int).tolist()) for u, g in test.groupby("userID", sort=False)}
    return histories, test_users, test_sets


def run(assets_dir: Path, out_dir: Path, config_path: Path) -> dict:
    marker = out_dir / "TEST_RUN_COMPLETED"
    if marker.exists():
        raise RuntimeError("Frozen CoLiftRec Test already executed; refusing a second Test run.")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ccfg = cfg["coliftrec"]
    p = CoLiftConfig(lambda_text=float(ccfg["text"]["lambda"]), lambda_attribute=float(ccfg["attribute"]["lambda"]), lambda_visual=float(ccfg["visual"]["lambda"]), alpha_text=float(ccfg["text"]["alpha"]), alpha_attribute=float(ccfg["attribute"]["alpha"]), alpha_visual=float(ccfg["visual"]["alpha"]))
    enabled = {m: bool(ccfg[m]["enabled"]) for m in ("text", "attribute", "visual")}

    asset_audit = json.loads((assets_dir / "audit.json").read_text(encoding="utf-8"))
    if asset_audit.get("TEST_ACCESSED") is not False:
        raise RuntimeError("Source MSCA assets are not Validation-only")

    inter_path = ROOT / "data/baby/baby.inter"
    histories, pseudo_histories, expected_pseudo_users, _, _ = build_train_histories_and_validation(inter_path, int(asset_audit["n_users"]))
    histories_test, test_users, test_sets = build_test_eval(inter_path, int(asset_audit["n_users"]))
    if histories != histories_test:
        raise RuntimeError("TRAIN history construction mismatch")

    pseudo_asset = np.load(assets_dir / "train_pseudo_top100.npz")
    pseudo_users = pseudo_asset["users"].astype(np.int64)
    pseudo_items = pseudo_asset["items"].astype(np.int32)
    if not np.array_equal(pseudo_users, expected_pseudo_users):
        raise RuntimeError("Pseudo-train user order mismatch")

    emb = np.load(assets_dir / "embeddings.npz")
    final_user = torch.as_tensor(emb["final_user"], device="cuda")
    final_item = torch.as_tensor(emb["final_item"], device="cuda")
    test_items, test_scores = topk_from_embeddings(final_user, final_item, test_users, histories, top_l=int(ccfg.get("top_l", 100)), batch_users=1024)

    text_path = ROOT / "data/baby/text_feat.npy"
    visual_path = ROOT / "data/baby/image_feat.npy"
    metadata_path = ROOT / "data/baby/metadata_text_cache.jsonl"

    z_text_pseudo, text_train_audit = semantic_z_for_candidates(text_path, pseudo_histories, pseudo_users, pseudo_items, batch_users=256)
    z_text_test, text_test_audit = semantic_z_for_candidates(text_path, histories, test_users, test_items, batch_users=256)
    z_visual_pseudo, visual_train_audit = semantic_z_for_candidates(visual_path, pseudo_histories, pseudo_users, pseudo_items, batch_users=128)
    z_visual_test, visual_test_audit = semantic_z_for_candidates(visual_path, histories, test_users, test_items, batch_users=128)
    item_matrices, attribute_audit = build_item_matrices(metadata_path, int(asset_audit["n_items"]), min_df=int(ccfg["attribute"]["tfidf_min_df"]), max_df=float(ccfg["attribute"]["tfidf_max_df"]), description_len=int(ccfg["attribute"]["description_len"]))
    full_profiles = build_profiles(item_matrices, histories, int(asset_audit["n_items"]))
    pseudo_profiles = build_profiles(item_matrices, pseudo_histories, int(asset_audit["n_items"]))
    z_attr_pseudo, _ = attribute_z(item_matrices, pseudo_profiles, pseudo_users, pseudo_items, batch=256)
    z_attr_test, _ = attribute_z(item_matrices, full_profiles, test_users, test_items, batch=256)

    backgrounds = fit_backgrounds(pseudo_items, z_text_pseudo, z_attr_pseudo, z_visual_pseudo, int(asset_audit["n_items"]))
    full_scores, _ = score_coliftrec(test_scores, test_items, z_text_test, z_attr_test, z_visual_test, backgrounds, p, enabled=enabled)

    baseline = metrics_at(test_items, test_users, test_sets)
    full = metrics_at(rank_by_score(test_items, full_scores), test_users, test_sets)
    deltas = {k: float(full[k] - baseline[k]) for k in ALL}
    summary = {
        "phase": "PHASE2_COLIFTREC_FROZEN_TEST",
        "dataset": "baby",
        "source_checkpoint_sha256": asset_audit["checkpoint_sha256"],
        "source_checkpoint_epoch": int(asset_audit["checkpoint_epoch"]),
        "parameters": p.to_dict(),
        "parameters_frozen_from_validation": True,
        "MSCA_CANONICAL": baseline,
        "MSCA_FULL_COLIFTREC_TAV": full,
        "deltas_full_vs_msca": deltas,
        "primary_positive_count": int(sum(deltas[k] > 0 for k in PRIMARY)),
        "overall_positive_count": int(sum(deltas[k] > 0 for k in ALL)),
        "BABY_COLIFTREC_TEST_RUN_COUNT": 1,
        "TEST_USED_FOR_SELECTION": False,
        "NO_POST_TEST_TUNING": True,
    }
    audit = {
        "interaction_sha256": sha256_file(inter_path),
        "test_users": int(len(test_users)),
        "pseudo_train_users": int(len(pseudo_users)),
        "candidate_shape": list(test_items.shape),
        "history_definition": "x_label==0 TRAIN only",
        "target_definition": "x_label==2 TEST only",
        "candidate_definition": "current-run frozen MSCA Top100 with TRAIN positives masked",
        "background_definition": "TRAIN-only pseudo histories with last TRAIN interaction removed",
        "text_train": text_train_audit,
        "text_test": text_test_audit,
        "visual_train": visual_train_audit,
        "visual_test": visual_test_audit,
        "attribute": attribute_audit,
        "TEST_ACCESSED": True,
        "TEST_USED_FOR_SELECTION": False,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "test_scores.npz", users=test_users, items=test_items, msca=test_scores, full_coliftrec=full_scores)
    marker.write_text("BABY_COLIFTREC_TEST_RUN_COUNT=1\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=str(ROOT / "src/configs/second_paper.yaml"))
    args = parser.parse_args()
    run(Path(args.assets), Path(args.out), Path(args.config))


if __name__ == "__main__":
    main()
