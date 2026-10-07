from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from diffusion_experiments.modules.round7_common import (
    ALL_METRICS,
    build_boundary_mask,
    build_history_arrays,
    cfg_round7,
    fit_cf_statistics,
    fit_user_statistics,
    git_sha,
    label_sets,
    load_frozen_backbone,
    load_interactions,
    m1_sorted,
    sha,
    train_frame,
    unique_histories,
)
from modules.ranking import candidate_dot_scores, metrics_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refuse overwrite nonempty output {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = cfg_round7()
    frozen = load_frozen_backbone(a.seed)
    emb = frozen["embeddings"]
    scores = frozen["scores"]
    n_users = int(frozen["msca_audit"]["n_users"])
    n_items = int(frozen["msca_audit"]["n_items"])
    if emb["collab_user"].shape != (n_users, 64) or emb["collab_item"].shape != (n_items, 64):
        raise RuntimeError("frozen collaborative embedding dimensions are not 64D")

    df = load_interactions()
    train = train_frame(df)
    if len(train) != 118551:
        raise RuntimeError(f"full TRAIN cardinality mismatch {len(train)}")
    histories = unique_histories(train, n_users)
    if any(len(h) == 0 for h in histories):
        raise RuntimeError("Round7 expects every Baby user to have TRAIN history")
    train_users = np.sort(train.userID.unique()).astype(np.int64)
    if len(train_users) != n_users:
        raise RuntimeError("TRAIN user coverage mismatch")

    item_z, item_mean, item_std, item_std_safe, item_const, observed_items = fit_cf_statistics(
        emb["collab_item"], train, float(cfg["cf"]["std_floor"])
    )
    user_z, user_mean, user_std, user_std_safe = fit_user_statistics(
        emb["collab_user"], train_users, float(cfg["cf"]["std_floor"])
    )
    hist = build_history_arrays(train, histories, item_z, emb["collab_item"], user_z)
    if hist["cond"].shape != (len(train), 129) or hist["cond_full"].shape != (n_users, 129):
        raise RuntimeError("Round7 condition dimension mismatch")
    expected_event_len = hist["full_len"][hist["users"]] - 1
    if not np.array_equal(hist["event_len"], expected_event_len):
        bad = int(np.flatnonzero(hist["event_len"] != expected_event_len)[0])
        raise RuntimeError(f"positive-removal history length mismatch at event {bad}")
    if np.any(hist["event_len"] < 0):
        raise RuntimeError("negative event history length")
    z_pos = item_z[hist["pos"]].astype(np.float32)

    valid_users, valid_sets = label_sets(df, 1)
    users, m0_items, m0_scores, m1_items, m1_s0 = m1_sorted(scores)
    if not np.array_equal(users, valid_users):
        raise RuntimeError("existing all-user ranking cache order does not match Validation users")
    m0_metrics = metrics_at(m0_items, users, valid_sets)
    m1_metrics = metrics_at(m1_items, users, valid_sets)
    expected_m0 = frozen["colift_summary"]["metrics"]["MSCA_CANONICAL"]
    expected_m1 = frozen["colift_summary"]["metrics"]["MSCA_FULL_COLIFTREC_TAV"]
    max_m0 = max(abs(float(m0_metrics[k]) - float(expected_m0[k])) for k in ALL_METRICS)
    max_m1 = max(abs(float(m1_metrics[k]) - float(expected_m1[k])) for k in ALL_METRICS)
    if max_m0 > 1e-12 or max_m1 > 1e-12:
        raise RuntimeError(f"BASE_IDENTITY_MISMATCH metrics m0={max_m0} m1={max_m1}")

    cf_candidate = candidate_dot_scores(emb["collab_user"], emb["collab_item"], users, m0_items)
    q = np.maximum(cf_candidate.std(axis=1), float(cfg["inference"]["q_floor"])).astype(np.float32)
    fc = cfg["fusion"]
    A = build_boundary_mask(
        m1_items,
        m1_s0,
        histories,
        n_items,
        cutoffs=tuple(fc["cutoffs"]),
        protected_rank_le=int(fc["protected_rank_le"]),
        max_score_distance=float(fc["max_score_distance"]),
        per_cutoff_quota=int(fc["per_cutoff_quota"]),
        max_items=int(fc["max_items"]),
    )
    if np.any(A[:, : int(fc["protected_rank_le"])]):
        raise RuntimeError("protected ranks leaked into A")
    hist_sets = [set(map(int, h)) for h in histories]
    leaked = 0
    for u in range(n_users):
        leaked += sum(int(it) in hist_sets[u] for it in m1_items[u, A[u]])
    if leaked:
        raise RuntimeError(f"TRAIN-history item leaked into A: {leaked}")

    np.savez_compressed(
        out / "events.npz",
        users=hist["users"].astype(np.int32),
        pos=hist["pos"].astype(np.int32),
        z_pos=z_pos,
        cond=hist["cond"],
        anchor_std=hist["event_h_std"],
        event_len=hist["event_len"].astype(np.int32),
        has_anchor=hist["has_anchor"],
    )
    np.savez_compressed(
        out / "deployment.npz",
        users=np.arange(n_users, dtype=np.int32),
        cond=hist["cond_full"],
        anchor_std=hist["full_h_std"],
        anchor_raw=hist["full_h_raw"],
        history_len=hist["full_len"].astype(np.int32),
        item_mean=item_mean,
        item_std=item_std,
        item_std_safe=item_std_safe,
        item_constant_mask=item_const,
        user_mean=user_mean,
        user_std=user_std,
        user_std_safe=user_std_safe,
        observed_items=observed_items.astype(np.int32),
        collab_user=emb["collab_user"].astype(np.float32),
        collab_item=emb["collab_item"].astype(np.float32),
    )
    np.savez_compressed(
        out / "baseline_all_users.npz",
        users=users.astype(np.int32),
        m0_items=m0_items.astype(np.int32),
        m0_scores=m0_scores.astype(np.float32),
        m1_items=m1_items.astype(np.int32),
        m1_s0=m1_s0.astype(np.float32),
        A_mask=A,
        q=q,
        cf_candidate_std=q,
    )

    files = ["events.npz", "deployment.npz", "baseline_all_users.npz"]
    guide = ROOT / "diffusion_experiments/ADVISOR_EXPERIMENT_GUIDE.md"
    audit = {
        "status": "COMPLETE_ROUND7_ASSETS",
        "protocol_version": cfg["protocol_version"],
        "dataset": "baby",
        "backbone_seed": int(a.seed),
        "source_checkpoint": str(frozen["checkpoint"].resolve()),
        "source_checkpoint_sha256": sha(frozen["checkpoint"]),
        "source_checkpoint_epoch": int(frozen["msca_audit"]["checkpoint_epoch"]),
        "source_msca_assets": str(frozen["msca_dir"].resolve()),
        "source_colift_assets": str(frozen["colift_dir"].resolve()),
        "interaction_sha256": frozen["colift_audit"]["inputs"]["interaction"]["sha256"],
        "feature_hashes": {
            "text": frozen["colift_audit"]["inputs"]["text"]["feature_sha256"],
            "visual": frozen["colift_audit"]["inputs"]["visual"]["feature_sha256"],
            "metadata": frozen["colift_audit"]["inputs"]["metadata"]["metadata_sha256"],
        },
        "train_edges": int(len(train)),
        "train_users": int(len(train_users)),
        "train_observed_items": int(len(observed_items)),
        "n_users": n_users,
        "n_items": n_items,
        "condition_dim": int(hist["cond"].shape[1]),
        "events_with_anchor": int(hist["has_anchor"].sum()),
        "events_without_anchor": int((~hist["has_anchor"]).sum()),
        "constant_cf_dims": int(item_const.sum()),
        "item_inverse_standardization_max_abs_diff": float(
            np.max(np.abs(item_mean + item_std_safe * item_z - emb["collab_item"]))
        ),
        "log_len_stats": {
            "mean": hist["log_len_mean"],
            "std": hist["log_len_std"],
            "std_safe": hist["log_len_std_safe"],
        },
        "validation_users": int(len(users)),
        "validation_m0_metrics": m0_metrics,
        "validation_m1_metrics": m1_metrics,
        "validation_identity_max_abs_diff": {"M0": max_m0, "M1": max_m1},
        "A": {
            "mean_size": float(A.sum(axis=1).mean()),
            "min_size": int(A.sum(axis=1).min()),
            "max_size": int(A.sum(axis=1).max()),
            "empty_users": int(np.sum(A.sum(axis=1) == 0)),
            "protected_rank_le": int(fc["protected_rank_le"]),
            "train_history_leaks": int(leaked),
        },
        "q": {"min": float(q.min()), "mean": float(q.mean()), "max": float(q.max())},
        "guide_sha256": sha(guide),
        "git_sha": git_sha(),
        "artifacts": {name: sha(out / name) for name in files},
        "access": {"TRAIN": True, "VALIDATION_LABELS": True, "TEST_LABELS_USED": False},
        "elapsed_seconds": time.time() - t0,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({
        "status": audit["status"],
        "seed": int(a.seed),
        "M0_R20": m0_metrics["R20"],
        "M1_R20": m1_metrics["R20"],
        "A_mean": audit["A"]["mean_size"],
        "events_with_anchor": audit["events_with_anchor"],
        "inverse_max": audit["item_inverse_standardization_max_abs_diff"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
