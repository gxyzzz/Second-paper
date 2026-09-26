from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modules.ranking import metrics_at, sha256_file, topk_from_embeddings
from models.msca import MSCA
from utils.dataloader import TrainDataLoader
from utils.dataset import RecDataset

ROOT = Path(__file__).resolve().parents[2]


def build_train_histories_and_validation(inter_path: Path, n_users: int):
    df = pd.read_csv(
        inter_path,
        sep="\t",
        usecols=["userID", "itemID", "timestamp", "x_label"],
    )
    df["_row"] = np.arange(len(df), dtype=np.int64)
    train = df[df.x_label == 0].copy().sort_values(
        ["userID", "timestamp", "_row"], kind="stable"
    )
    histories = [[] for _ in range(n_users)]
    for user, group in train.groupby("userID", sort=False):
        histories[int(user)] = group.itemID.astype(np.int64).tolist()

    pseudo = [list(h[:-1]) if len(h) >= 2 else list(h) for h in histories]
    pseudo_users = np.asarray(
        [u for u, h in enumerate(histories) if len(h) >= 2], dtype=np.int64
    )
    train_users = {u for u, h in enumerate(histories) if h}
    valid = df[(df.x_label == 1) & df.userID.isin(train_users)].copy()
    valid_users = valid.userID.drop_duplicates().astype(np.int64).to_numpy()
    valid_sets = {
        int(u): set(g.itemID.astype(int).tolist())
        for u, g in valid.groupby("userID", sort=False)
    }
    return histories, pseudo, pseudo_users, valid_users, valid_sets


def load_msca_checkpoint(checkpoint_path: Path, gpu_id: int = 0):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    config["gpu_id"] = int(gpu_id)
    config["use_gpu"] = True
    config["device"] = torch.device(f"cuda:{gpu_id}")
    config["data_path"] = str(ROOT / "data") + "/"

    dataset = RecDataset(config)
    train_dataset, _, _ = dataset.split()
    train_dataset.inter_num = len(train_dataset.df)
    train_data = TrainDataLoader(
        config,
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=False,
    )
    train_data.pretrain_setup()

    model = MSCA(config, train_data).to(config["device"])
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"strict checkpoint load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model, checkpoint, dataset, train_dataset


@torch.no_grad()
def extract_embeddings(model: MSCA) -> dict[str, np.ndarray]:
    final_u, final_i, collab, _, _, _ = model.forward(test=False)
    collab_u, collab_i = torch.split(
        collab, [model.n_users, model.n_items], dim=0
    )
    return {
        "final_user": final_u.detach().cpu().numpy().astype(np.float32),
        "final_item": final_i.detach().cpu().numpy().astype(np.float32),
        "collab_user": collab_u.detach().cpu().numpy().astype(np.float32),
        "collab_item": collab_i.detach().cpu().numpy().astype(np.float32),
    }


def export_validation_assets(checkpoint_path: Path, out_dir: Path, gpu_id: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint, dataset, _ = load_msca_checkpoint(checkpoint_path, gpu_id=gpu_id)
    embeddings = extract_embeddings(model)

    histories, pseudo, pseudo_users, valid_users, valid_sets = (
        build_train_histories_and_validation(
            ROOT / "data" / checkpoint["config"]["dataset"]
            / checkpoint["config"]["inter_file_name"],
            dataset.user_num,
        )
    )

    final_u_t = torch.as_tensor(embeddings["final_user"], device=model.device)
    final_i_t = torch.as_tensor(embeddings["final_item"], device=model.device)
    valid_items, valid_scores = topk_from_embeddings(
        final_u_t, final_i_t, valid_users, histories, top_l=100, batch_users=1024
    )
    pseudo_items, pseudo_scores = topk_from_embeddings(
        final_u_t, final_i_t, pseudo_users, pseudo, top_l=100, batch_users=1024
    )
    valid_metrics = metrics_at(valid_items, valid_users, valid_sets)

    np.savez_compressed(
        out_dir / "embeddings.npz",
        final_user=embeddings["final_user"],
        final_item=embeddings["final_item"],
        collab_user=embeddings["collab_user"],
        collab_item=embeddings["collab_item"],
    )
    np.savez_compressed(
        out_dir / "validation_top100.npz",
        users=valid_users,
        items=valid_items,
        scores=valid_scores,
    )
    np.savez_compressed(
        out_dir / "train_pseudo_top100.npz",
        users=pseudo_users,
        items=pseudo_items,
        scores=pseudo_scores,
    )

    audit = {
        "dataset": checkpoint["config"]["dataset"],
        "seed": int(checkpoint["config"]["seed"]),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "n_users": int(model.n_users),
        "n_items": int(model.n_items),
        "embedding_dim": int(model.embedding_dim),
        "embedding_shapes": {k: list(v.shape) for k, v in embeddings.items()},
        "valid_users": int(len(valid_users)),
        "pseudo_users": int(len(pseudo_users)),
        "top_l": 100,
        "validation_metrics": valid_metrics,
        "TEST_ACCESSED": False,
        "source": "current-run Second-paper frozen MSCA checkpoint",
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    audit = export_validation_assets(
        Path(args.checkpoint), Path(args.out), gpu_id=args.gpu_id
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
