from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.trainer import Trainer
from utils.configurator import Config
from utils.dataloader import EvalDataLoader, TrainDataLoader
from utils.dataset import RecDataset
from utils.utils import get_model, init_seed


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _load_edges(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["userID", "itemID"])
    return frame.astype({"userID": "int64", "itemID": "int64"})


def build_config(out_dir: Path, epochs: int, stopping_step: int) -> Config:
    cfg = Config(
        "MSCA",
        "baby",
        {
            "seed": 999,
            "gpu_id": os.environ.get("ROUND1_GPU_UUID", 0),
            "use_gpu": True,
            "epochs": epochs,
            "stopping_step": stopping_step,
            "eval_step": 1,
            "train_batch_size": 2048,
            "eval_batch_size": 2048,
            "checkpoint_dir": str(out_dir / "checkpoints"),
        },
    )
    # Frozen Baby MSCA hyperparameters: one exact configuration, no search.
    cfg["data_path"] = str(ROOT / "data") + "/"
    cfg["seed"] = 999
    cfg["n_layers"] = 2
    cfg["fusion_coeff"] = 0.4
    cfg["cl_weight"] = 0.005
    cfg["reg_weight"] = 3e-7
    cfg["hyper_parameters"] = []
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["smoke", "formal"], required=True)
    args = ap.parse_args()

    protocol_dir = Path(args.protocol_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    if protocol["access"]["CONFIRM_ACCESSED"] or protocol["access"]["TEST_ACCESSED"]:
        raise RuntimeError("Round1 access invariant violated")
    if sha256_file(protocol_dir / "fit_edges.csv") != protocol["hashes"]["fit_edges"]:
        raise RuntimeError("FIT edge hash mismatch")
    if sha256_file(protocol_dir / "monitor_edges.csv") != protocol["hashes"]["monitor_edges"]:
        raise RuntimeError("monitor edge hash mismatch")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    gpu_name = torch.cuda.get_device_name(0)
    if "5090" not in gpu_name:
        raise RuntimeError(f"Round1 requires RTX 5090, got {gpu_name}")
    gpu_uuid = None
    try:
        gpu_uuid = torch.cuda.get_device_properties(0).uuid
    except Exception:
        pass

    epochs, stopping = (2, 2) if args.mode == "smoke" else (1000, 20)
    cfg = build_config(out_dir, epochs, stopping)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("round1_fit_backbone")

    base = RecDataset(cfg)
    fit_df = _load_edges(protocol_dir / "fit_edges.csv")
    monitor_df = _load_edges(protocol_dir / "monitor_edges.csv")
    fit_ds = base.copy(fit_df)
    monitor_ds = base.copy(monitor_df)
    fit_ds.inter_num = len(fit_ds.df)
    monitor_ds.inter_num = len(monitor_ds.df)

    train_data = TrainDataLoader(
        cfg, fit_ds, batch_size=cfg["train_batch_size"], shuffle=True
    )
    monitor_data = EvalDataLoader(
        cfg,
        monitor_ds,
        additional_dataset=fit_ds,
        batch_size=cfg["eval_batch_size"],
    )
    init_seed(999)
    train_data.pretrain_setup()
    model = get_model("MSCA")(cfg, train_data).to(cfg["device"])
    trainer = Trainer(cfg, model)

    logger.info(
        "ROUND1_FIT_BACKBONE_START mode=%s gpu=%s fit_edges=%d monitor_edges=%d",
        args.mode,
        gpu_name,
        len(fit_df),
        len(monitor_df),
    )
    start = time.time()
    best_score, best_result = trainer.fit(
        train_data, valid_data=monitor_data, saved=True, verbose=True
    )
    elapsed = time.time() - start

    ckpt = Path(trainer.saved_model_file).resolve()
    if not ckpt.is_file():
        raise RuntimeError(f"checkpoint not created: {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    evidence = {
        "status": "COMPLETE",
        "mode": args.mode,
        "fit_backbone_prototype": True,
        "dataset": "baby",
        "seed": 999,
        "gpu_name": gpu_name,
        "gpu_uuid": str(gpu_uuid) if gpu_uuid is not None else None,
        "protocol_json": str((protocol_dir / "protocol.json").resolve()),
        "protocol_sha256": sha256_file(protocol_dir / "protocol.json"),
        "fit_edges_sha256": protocol["hashes"]["fit_edges"],
        "monitor_edges_sha256": protocol["hashes"]["monitor_edges"],
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "best_epoch": int(state["epoch"]),
        "best_valid_score": float(best_score),
        "best_valid_result": {k: float(v) for k, v in best_result.items()},
        "elapsed_seconds": elapsed,
        "config": {
            "n_layers": 2,
            "fusion_coeff": 0.4,
            "cl_weight": 0.005,
            "reg_weight": 3e-7,
            "epochs": epochs,
            "stopping_step": stopping,
            "train_batch_size": 2048,
            "eval_batch_size": 2048,
        },
        "access": {"CONFIRM_ACCESSED": False, "TEST_ACCESSED": False},
    }
    (out_dir / "training.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
