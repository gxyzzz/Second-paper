from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "diffusion_experiments/configs/round1_baby.yaml"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return sha256_file(path)


def _edge_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["_row", "userID", "itemID", "timestamp"]
    return df[cols].sort_values("_row", kind="stable").reset_index(drop=True)


def build_protocol(config_path: Path, out_dir: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg["access"]["confirm_open"] or cfg["access"]["test_open"]:
        raise RuntimeError("Round1 protocol requires CONFIRM and Test closed")
    if cfg["source_dataset"] != "baby":
        raise RuntimeError("Round1 is Baby-only")

    inter = ROOT / "data/baby/baby.inter"
    df = pd.read_csv(
        inter,
        sep="\t",
        usecols=["userID", "itemID", "timestamp", "x_label"],
    )
    df["_row"] = np.arange(len(df), dtype=np.int64)
    train = df[df.x_label == 0].copy().sort_values(
        ["userID", "timestamp", "_row"], kind="stable"
    )
    valid = df[df.x_label == 1].copy()

    split_seed = int(cfg["split_seed"])
    rng = np.random.default_rng(split_seed)
    min_fit = int(cfg["min_fit_history_after_holds"])
    needed = (
        min_fit
        + int(cfg["probe_per_eligible_user"])
        + int(cfg["monitor_per_eligible_user"])
    )

    probe_rows: list[int] = []
    monitor_rows: list[int] = []
    eligible_users: list[int] = []
    excluded = {"train_history_lt_required": 0}
    for user, group in train.groupby("userID", sort=True):
        row_indices = group.index.to_numpy(dtype=np.int64)
        if len(row_indices) < needed:
            excluded["train_history_lt_required"] += 1
            continue
        chosen = rng.choice(row_indices, size=2, replace=False)
        probe_rows.append(int(chosen[0]))
        monitor_rows.append(int(chosen[1]))
        eligible_users.append(int(user))

    probe_set, monitor_set = set(probe_rows), set(monitor_rows)
    if probe_set & monitor_set:
        raise RuntimeError("probe/monitor row overlap")
    held = probe_set | monitor_set
    fit = train.loc[~train.index.isin(held)].copy()
    probe = train.loc[probe_rows].copy().sort_values("userID", kind="stable")
    monitor = train.loc[monitor_rows].copy().sort_values("userID", kind="stable")

    fit_counts = fit.groupby("userID").size()
    if any(int(fit_counts.get(u, 0)) < min_fit for u in eligible_users):
        raise RuntimeError("eligible user lost minimum FIT history")

    valid_users = np.array(
        sorted(set(valid.userID.astype(int).tolist())), dtype=np.int64
    )
    dev_rng = np.random.default_rng(int(cfg["dev_confirm_seed"]))
    perm = dev_rng.permutation(valid_users)
    n_dev = int(np.floor(float(cfg["validation_dev_fraction"]) * len(perm)))
    dev_users = np.sort(perm[:n_dev])
    confirm_users = np.sort(perm[n_dev:])
    if np.intersect1d(dev_users, confirm_users).size:
        raise RuntimeError("DEV/CONFIRM overlap")

    eligible = np.array(sorted(eligible_users), dtype=np.int64)
    rr_rng = np.random.default_rng(int(cfg["reranker_split_seed"]))
    rr_perm = rr_rng.permutation(eligible)
    n_rr_train = int(
        np.floor(float(cfg["reranker_train_fraction"]) * len(rr_perm))
    )
    reranker_train_users = np.sort(rr_perm[:n_rr_train])
    internal_users = np.sort(rr_perm[n_rr_train:])

    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        "fit_edges": _write_csv(_edge_frame(fit), out_dir / "fit_edges.csv"),
        "monitor_edges": _write_csv(
            _edge_frame(monitor), out_dir / "monitor_edges.csv"
        ),
        "probe_edges": _write_csv(_edge_frame(probe), out_dir / "probe_edges.csv"),
    }
    pd.DataFrame({"userID": dev_users}).to_csv(
        out_dir / "dev_users.csv", index=False
    )
    pd.DataFrame({"userID": confirm_users}).to_csv(
        out_dir / "confirm_users.csv", index=False
    )
    pd.DataFrame({"userID": reranker_train_users}).to_csv(
        out_dir / "reranker_train_users.csv", index=False
    )
    pd.DataFrame({"userID": internal_users}).to_csv(
        out_dir / "internal_users.csv", index=False
    )
    for name in (
        "dev_users",
        "confirm_users",
        "reranker_train_users",
        "internal_users",
    ):
        hashes[name] = sha256_file(out_dir / f"{name}.csv")

    fit_pairs = set(zip(fit.userID.astype(int), fit.itemID.astype(int)))
    target_pairs = set(zip(probe.userID.astype(int), probe.itemID.astype(int)))
    target_pair_overlap = len(fit_pairs & target_pairs)
    if target_pair_overlap:
        raise RuntimeError(
            f"probe user-item pair still appears in FIT: {target_pair_overlap}"
        )

    payload = {
        "protocol_version": cfg["protocol_version"],
        "dataset": "baby",
        "source_interaction": str(inter.resolve()),
        "source_interaction_sha256": sha256_file(inter),
        "split_seed": split_seed,
        "dev_confirm_seed": int(cfg["dev_confirm_seed"]),
        "reranker_split_seed": int(cfg["reranker_split_seed"]),
        "fit_backbone_seed": int(cfg["fit_backbone_seed"]),
        "counts": {
            "original_train_edges": int(len(train)),
            "fit_edges": int(len(fit)),
            "monitor_edges": int(len(monitor)),
            "probe_edges": int(len(probe)),
            "train_users": int(train.userID.nunique()),
            "eligible_probe_users": int(len(eligible_users)),
            "ineligible_users": int(train.userID.nunique() - len(eligible_users)),
            "validation_users_total": int(len(valid_users)),
            "dev_users": int(len(dev_users)),
            "confirm_users_frozen_closed": int(len(confirm_users)),
            "reranker_train_users": int(len(reranker_train_users)),
            "internal_users": int(len(internal_users)),
        },
        "excluded": excluded,
        "disjoint_checks": {
            "probe_monitor_row_overlap": 0,
            "probe_fit_user_item_pair_overlap": int(target_pair_overlap),
            "eligible_min_fit_history": min_fit,
            "PASS": True,
        },
        "hashes": hashes,
        "access": {
            "CONFIRM_ACCESSED": False,
            "TEST_ACCESSED": False,
            "TEST_USED_FOR_SELECTION": False,
        },
        "notes": {
            "fit_backbone_prototype": True,
            "ineligible_users": (
                "retained in FIT but do not contribute supervised probe queries"
            ),
            "confirm": (
                "user IDs frozen only; CONFIRM labels are not evaluated in Round1"
            ),
        },
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build_protocol(Path(args.config), Path(args.out))


if __name__ == "__main__":
    main()
