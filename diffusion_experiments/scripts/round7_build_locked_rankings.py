from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from diffusion_experiments.modules.round7_common import (
    build_boundary_mask,
    cfg_round7,
    load_frozen_backbone,
    rerank_slots,
    sha,
)
from diffusion_experiments.scripts.round7_cache_validation import generate_all, load_model
from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import fit_backgrounds, score_coliftrec
from modules.ranking import candidate_dot_scores, semantic_z_for_candidates, topk_from_embeddings
from pipelines.coliftrec import _params
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-seed", type=int, required=True)
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--generator-101", required=True)
    ap.add_argument("--generator-102", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = cfg_round7()
    if float(a.eta) not in [float(x) for x in cfg["fusion"]["eta_candidates"]]:
        raise RuntimeError("eta is not one of registered candidates")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refuse overwrite nonempty output {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    frozen = load_frozen_backbone(a.backbone_seed)
    msca_assets = frozen["msca_dir"]
    round7_assets = Path(a.assets)
    asset_audit = json.loads((round7_assets / "audit.json").read_text())
    if int(asset_audit["backbone_seed"]) != int(a.backbone_seed):
        raise RuntimeError("Round7 asset/backbone mismatch")
    emb = frozen["embeddings"]
    n_users = int(frozen["msca_audit"]["n_users"])
    n_items = int(frozen["msca_audit"]["n_items"])
    dcfg = load_dataset_config("baby")
    paths = dcfg["resolved_paths"]
    ccfg = dcfg["coliftrec"]
    p = _params(ccfg)
    enabled = {m: bool(ccfg[m]["enabled"]) for m in ("text", "attribute", "visual")}

    histories, pseudo_histories, pseudo_users_expected, _, _ = build_train_histories_and_validation(
        paths["interaction"], n_users
    )
    users = np.arange(n_users, dtype=np.int64)
    final_user = torch.as_tensor(emb["final_user"], device="cuda:0")
    final_item = torch.as_tensor(emb["final_item"], device="cuda:0")
    m0_items, m0_scores = topk_from_embeddings(
        final_user, final_item, users, histories, top_l=100, batch_users=1024
    )

    pseudo_asset = np.load(msca_assets / "train_pseudo_top100.npz")
    pseudo_users = pseudo_asset["users"].astype(np.int64)
    pseudo_items = pseudo_asset["items"].astype(np.int32)
    if not np.array_equal(pseudo_users, pseudo_users_expected):
        raise RuntimeError("pseudo user order mismatch")
    ztp, _ = semantic_z_for_candidates(paths["text_feature"], pseudo_histories, pseudo_users, pseudo_items, 256)
    zta, _ = semantic_z_for_candidates(paths["text_feature"], histories, users, m0_items, 256)
    zvp, _ = semantic_z_for_candidates(paths["visual_feature"], pseudo_histories, pseudo_users, pseudo_items, 128)
    zva, _ = semantic_z_for_candidates(paths["visual_feature"], histories, users, m0_items, 128)
    acfg = ccfg["attribute"]
    mats, _ = build_item_matrices(
        paths["metadata"], n_items,
        min_df=int(acfg.get("tfidf_min_df", 2)),
        max_df=float(acfg.get("tfidf_max_df", 0.8)),
        description_len=int(acfg.get("description_len", 128)),
        weights=acfg.get("weights"),
    )
    full_profiles = build_profiles(mats, histories, n_items)
    pseudo_profiles = build_profiles(mats, pseudo_histories, n_items)
    zap, _ = attribute_z(mats, pseudo_profiles, pseudo_users, pseudo_items, 256, weights=acfg.get("weights"))
    zaa, _ = attribute_z(mats, full_profiles, users, m0_items, 256, weights=acfg.get("weights"))
    bg = fit_backgrounds(pseudo_items, ztp, zap, zvp, n_items)
    full_scores, _ = score_coliftrec(m0_scores, m0_items, zta, zaa, zva, bg, p, enabled)
    m1_order = np.argsort(-full_scores, axis=1, kind="stable")
    m1_items = np.take_along_axis(m0_items, m1_order, axis=1).astype(np.int32)
    m1_s0 = np.take_along_axis(full_scores, m1_order, axis=1).astype(np.float32)

    baseline = np.load(round7_assets / "baseline_all_users.npz")
    if not np.array_equal(users, baseline["users"].astype(np.int64)):
        raise RuntimeError("baseline user order mismatch")
    if not np.array_equal(m0_items, baseline["m0_items"].astype(np.int32)):
        raise RuntimeError("independent M0 recomputation mismatch")
    if not np.array_equal(m1_items, baseline["m1_items"].astype(np.int32)):
        raise RuntimeError("independent M1 recomputation mismatch")
    m1_score_diff = float(np.max(np.abs(m1_s0 - baseline["m1_s0"].astype(np.float32))))
    if m1_score_diff > 1e-6:
        raise RuntimeError(f"independent M1 score recomputation mismatch {m1_score_diff}")

    dep = np.load(round7_assets / "deployment.npz")
    collab_user = dep["collab_user"].astype(np.float32)
    collab_item = dep["collab_item"].astype(np.float32)
    cf_candidate = candidate_dot_scores(collab_user, collab_item, users, m0_items)
    q = np.maximum(cf_candidate.std(axis=1), float(cfg["inference"]["q_floor"])).astype(np.float32)
    fc = cfg["fusion"]
    A = build_boundary_mask(
        m1_items, m1_s0, histories, n_items,
        cutoffs=tuple(fc["cutoffs"]),
        protected_rank_le=int(fc["protected_rank_le"]),
        max_score_distance=float(fc["max_score_distance"]),
        per_cutoff_quota=int(fc["per_cutoff_quota"]),
        max_items=int(fc["max_items"]),
    )
    if not np.array_equal(A, baseline["A_mask"].astype(bool)):
        raise RuntimeError("independent A recomputation mismatch")
    q_diff = float(np.max(np.abs(q - baseline["q"].astype(np.float32))))
    if q_diff > 1e-6:
        raise RuntimeError(f"independent q recomputation mismatch {q_diff}")

    m2 = {}
    cell_meta = {}
    for dseed, gpath in [(202610101, Path(a.generator_101)), (202610102, Path(a.generator_102))]:
        model, ck = load_model(gpath, torch.device("cuda:0"))
        if int(ck["backbone_seed"]) != int(a.backbone_seed) or int(ck["diffusion_seed"]) != int(dseed):
            raise RuntimeError("generator identity mismatch")
        g_raw = generate_all(model, dep, a.backbone_seed, dseed, cfg)
        anchor_raw = dep["anchor_raw"].astype(np.float32)
        delta = g_raw - anchor_raw
        finite = np.isfinite(g_raw).all(axis=1)
        delta[~finite] = 0.0
        d = np.einsum("bd,bld->bl", delta, collab_item[m1_items], optimize=True).astype(np.float32)
        r = np.clip(d / q[:, None], -float(cfg["inference"]["clip_r"]), float(cfg["inference"]["clip_r"])).astype(np.float32)
        r[~finite] = 0.0
        rank = rerank_slots(m1_items, m1_s0, A, r, float(a.eta))
        if not np.array_equal(rank[:, :int(fc["protected_rank_le"])], m1_items[:, :int(fc["protected_rank_le"])]):
            raise RuntimeError("protected prefix changed")
        if not np.array_equal(rank[~A], m1_items[~A]):
            raise RuntimeError("outside-A slot changed")
        m2[dseed] = rank.astype(np.int32)
        cell_meta[str(dseed)] = {
            "generator": str(gpath.resolve()),
            "generator_sha256": sha(gpath),
            "finite_users": int(finite.sum()),
            "fallback_users": int((~finite).sum()),
            "changed_users": int(np.sum(np.any(rank != m1_items, axis=1))),
            "changed_slots": int(np.sum(rank != m1_items)),
        }

    for dseed in [202610101, 202610102]:
        pth = out / f"b{a.backbone_seed}_d{dseed}_rankings.npz"
        np.savez_compressed(
            pth,
            users=users.astype(np.int32),
            m0=m0_items.astype(np.int32),
            m1=m1_items.astype(np.int32),
            m2=m2[dseed].astype(np.int32),
        )
        cell_meta[str(dseed)]["ranking_file"] = str(pth.resolve().relative_to(ROOT))
        cell_meta[str(dseed)]["ranking_sha256"] = sha(pth)

    audit = {
        "status": "COMPLETE_UNLABELED_LOCK_RANKINGS",
        "protocol_version": cfg["protocol_version"],
        "dataset": "baby",
        "backbone_seed": int(a.backbone_seed),
        "selected_eta": float(a.eta),
        "users": int(len(users)),
        "source_checkpoint_sha256": asset_audit["source_checkpoint_sha256"],
        "source_round7_assets_sha256": sha(round7_assets / "audit.json"),
        "independent_recompute": {
            "M0_exact_vs_round7_asset": True,
            "M1_exact_vs_round7_asset": True,
            "M1_score_max_abs_diff_vs_round7_asset": m1_score_diff,
            "A_exact_vs_round7_asset": True,
            "q_max_abs_diff_vs_round7_asset": q_diff,
        },
        "cells": cell_meta,
        "target_item_ids_accessed": False,
        "test_metrics_computed": False,
        "elapsed_seconds": time.time() - t0,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": audit["status"], "seed": a.backbone_seed, "eta": a.eta, "cells": cell_meta}, sort_keys=True))


if __name__ == "__main__":
    main()
