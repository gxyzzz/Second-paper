from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modules.ranking import sha256_file
from pipelines.dataset_config import load_dataset_config


def build_current_run_assets(dataset: str, msca_assets: Path, out_dir: Path) -> dict:
    cfg = load_dataset_config(dataset)
    dataset = cfg["dataset"]
    paths = cfg["resolved_paths"]
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = json.loads((msca_assets / "audit.json").read_text())
    if audit.get("TEST_ACCESSED") is not False:
        raise RuntimeError("MSCA source assets must be Validation-only")
    if audit.get("dataset") != dataset:
        raise RuntimeError(f"asset dataset mismatch: {audit.get('dataset')} != {dataset}")

    emb_path = msca_assets / "embeddings.npz"
    emb = np.load(emb_path)
    collab = np.asarray(emb["collab_item"], dtype=np.float32)
    final = np.asarray(emb["final_item"], dtype=np.float32)
    if collab.shape != final.shape:
        raise ValueError(f"condition endpoint mismatch: {collab.shape} vs {final.shape}")
    n_items, embedding_dim = collab.shape
    if n_items != int(audit["n_items"]):
        raise ValueError(f"n_items mismatch: embeddings={n_items}, audit={audit['n_items']}")

    df = pd.read_csv(paths["interaction"], sep="\t", usecols=["itemID", "x_label"])
    train_ids = np.sort(df.loc[df.x_label == 0, "itemID"].unique().astype(np.int64))
    text = np.load(paths["text_feature"], mmap_mode="r", allow_pickle=False)
    visual = np.load(paths["visual_feature"], mmap_mode="r", allow_pickle=False)
    if text.shape[0] != n_items or visual.shape[0] != n_items:
        raise ValueError(f"feature item count mismatch: text={text.shape}, visual={visual.shape}, n_items={n_items}")
    if text.shape[1] != 384 or visual.shape[1] != 4096:
        raise ValueError(f"native diffusion dimensions changed: text={text.shape}, visual={visual.shape}")

    np.save(out_dir / "train_item_ids.npy", train_ids, allow_pickle=False)
    np.savez_compressed(
        out_dir / "condition_endpoints.npz",
        collab_item=collab,
        final_item=final,
    )
    result = {
        "phase": "DIFFUSION_ASSET_BUILDER",
        "dataset": dataset,
        "source_checkpoint_sha256": audit["checkpoint_sha256"],
        "source_checkpoint_epoch": int(audit["checkpoint_epoch"]),
        "embeddings_sha256": sha256_file(emb_path),
        "interaction_sha256": sha256_file(paths["interaction"]),
        "text_sha256": sha256_file(paths["text_feature"]),
        "visual_sha256": sha256_file(paths["visual_feature"]),
        "condition_formula": "Norm[c_collab + beta*(c_final-c_collab)]",
        "condition_endpoint_shape": [int(n_items), int(embedding_dim)],
        "train_item_count": int(len(train_ids)),
        "item_selection": "unique itemID appearing in x_label==0 TRAIN only",
        "text_shape": list(text.shape),
        "visual_shape": list(visual.shape),
        "joint_state_dim": int(text.shape[1] + visual.shape[1]),
        "TEST_ACCESSED": False,
        "TRAIN_ONLY_ITEM_AUDIT": "PASS",
    }
    (out_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--msca-assets", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = build_current_run_assets(a.dataset, Path(a.msca_assets), Path(a.out))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
