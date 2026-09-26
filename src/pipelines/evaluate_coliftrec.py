from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import fit_backgrounds, score_coliftrec
from modules.ranking import metrics_at, rank_by_score, semantic_z_for_candidates, topk_from_embeddings
from pipelines.coliftrec import _params, ALL, PRIMARY
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation


def build_test_eval(inter_path: Path, n_users: int):
    df = pd.read_csv(inter_path, sep="\t", usecols=["userID", "itemID", "timestamp", "x_label"])
    df["_row"] = np.arange(len(df), dtype=np.int64)
    train = df[df.x_label == 0].copy().sort_values(["userID", "timestamp", "_row"], kind="stable")
    histories = [[] for _ in range(n_users)]
    for user, group in train.groupby("userID", sort=False):
        histories[int(user)] = group.itemID.astype(np.int64).tolist()
    train_users = {u for u, h in enumerate(histories) if h}
    test = df[(df.x_label == 2) & df.userID.isin(train_users)].copy()
    users = test.userID.drop_duplicates().astype(np.int64).to_numpy()
    sets = {int(u): set(g.itemID.astype(int).tolist()) for u, g in test.groupby("userID", sort=False)}
    return histories, users, sets


def run(dataset: str, assets_dir: Path, out_dir: Path) -> dict:
    cfg = load_dataset_config(dataset)
    dataset = cfg["dataset"]
    paths = cfg["resolved_paths"]
    marker = out_dir / "TEST_RUN_COMPLETED"
    if marker.exists():
        raise RuntimeError(f"{dataset} CoLiftRec Test already executed in {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ccfg = cfg["coliftrec"]
    p = _params(ccfg)
    enabled = {m: bool(ccfg[m]["enabled"]) for m in ("text", "attribute", "visual")}
    audit = json.loads((assets_dir / "audit.json").read_text())
    if audit.get("TEST_ACCESSED") is not False or audit.get("dataset") != dataset:
        raise RuntimeError("invalid Validation-only MSCA asset set")

    n_users, n_items = int(audit["n_users"]), int(audit["n_items"])
    histories, pseudo_histories, expected_pseudo_users, _, _ = build_train_histories_and_validation(
        paths["interaction"], n_users
    )
    histories_test, test_users, test_sets = build_test_eval(paths["interaction"], n_users)
    if histories != histories_test:
        raise RuntimeError("TRAIN history mismatch")

    pseudo_asset = np.load(assets_dir / "train_pseudo_top100.npz")
    pseudo_users = pseudo_asset["users"].astype(np.int64)
    pseudo_items = pseudo_asset["items"].astype(np.int32)
    if not np.array_equal(pseudo_users, expected_pseudo_users):
        raise RuntimeError("pseudo user order mismatch")

    emb = np.load(assets_dir / "embeddings.npz")
    final_user = torch.as_tensor(emb["final_user"], device="cuda")
    final_item = torch.as_tensor(emb["final_item"], device="cuda")
    test_items, test_scores = topk_from_embeddings(
        final_user, final_item, test_users, histories,
        top_l=int(ccfg.get("top_l", 100)), batch_users=1024,
    )

    ztp, _ = semantic_z_for_candidates(paths["text_feature"], pseudo_histories, pseudo_users, pseudo_items, 256)
    ztt, _ = semantic_z_for_candidates(paths["text_feature"], histories, test_users, test_items, 256)
    zvp, _ = semantic_z_for_candidates(paths["visual_feature"], pseudo_histories, pseudo_users, pseudo_items, 128)
    zvt, _ = semantic_z_for_candidates(paths["visual_feature"], histories, test_users, test_items, 128)
    acfg = ccfg["attribute"]
    mats, _ = build_item_matrices(
        paths["metadata"], n_items,
        min_df=int(acfg.get("tfidf_min_df", 2)),
        max_df=float(acfg.get("tfidf_max_df", 0.8)),
        description_len=int(acfg.get("description_len", 128)),
    )
    full_profiles = build_profiles(mats, histories, n_items)
    pseudo_profiles = build_profiles(mats, pseudo_histories, n_items)
    zap, _ = attribute_z(mats, pseudo_profiles, pseudo_users, pseudo_items, 256)
    zat, _ = attribute_z(mats, full_profiles, test_users, test_items, 256)
    bg = fit_backgrounds(pseudo_items, ztp, zap, zvp, n_items)
    full_scores, _ = score_coliftrec(test_scores, test_items, ztt, zat, zvt, bg, p, enabled)

    baseline = metrics_at(test_items, test_users, test_sets)
    full = metrics_at(rank_by_score(test_items, full_scores), test_users, test_sets)
    deltas = {k: float(full[k] - baseline[k]) for k in ALL}
    summary = {
        "phase": "COLIFTREC_FROZEN_TEST",
        "dataset": dataset,
        "MSCA_CANONICAL": baseline,
        "MSCA_FULL_COLIFTREC_TAV": full,
        "deltas_full_vs_msca": deltas,
        "primary_positive_count": int(sum(deltas[k] > 0 for k in PRIMARY)),
        "overall_positive_count": int(sum(deltas[k] > 0 for k in ALL)),
        "TEST_RUN_COUNT": 1,
        "TEST_USED_FOR_SELECTION": False,
        "NO_POST_TEST_TUNING": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    marker.write_text("TEST_RUN_COUNT=1\n")
    np.savez_compressed(
        out_dir / "test_scores.npz",
        users=test_users, items=test_items, msca=test_scores, full_coliftrec=full_scores,
    )
    print(json.dumps(summary, sort_keys=True))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.dataset, Path(a.assets), Path(a.out))


if __name__ == "__main__":
    main()
