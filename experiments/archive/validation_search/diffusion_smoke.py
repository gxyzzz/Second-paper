from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from modules.diffusion import NativeTVX0Denoiser, cosine_alpha_bar, purify_indices, training_loss
from modules.semantic_purifier import blend_native, condition_beta, joint_native_state
from pipelines.dataset_config import load_dataset_config
from pipelines.diffusion_assets import build_current_run_assets


def run(dataset: str, msca_assets: Path, out_dir: Path, beta: float = 0.5, smoke_items: int = 8):
    cfg = load_dataset_config(dataset)
    dataset = cfg["dataset"]
    paths = cfg["resolved_paths"]
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = out_dir / "assets"
    asset_audit = build_current_run_assets(dataset, msca_assets, asset_dir)

    ep = np.load(asset_dir / "condition_endpoints.npz")
    collab = ep["collab_item"].astype(np.float32)
    final = ep["final_item"].astype(np.float32)
    cond = condition_beta(collab, final, beta)
    train_ids = np.load(asset_dir / "train_item_ids.npy")[:smoke_items]
    text = np.load(paths["text_feature"], mmap_mode="r", allow_pickle=False)
    visual = np.load(paths["visual_feature"], mmap_mode="r", allow_pickle=False)
    text_dim, visual_dim = int(text.shape[1]), int(visual.shape[1])
    if (text_dim, visual_dim) != (384, 4096):
        raise RuntimeError((text.shape, visual.shape))
    dim = text_dim + visual_dim

    x_np = joint_native_state(text[train_ids], visual[train_ids])
    device = "cuda"
    torch.manual_seed(999)
    model = NativeTVX0Denoiser(dim, cond_dim=cond.shape[1], hidden=1024, time_dim=64).to(device)
    x = torch.from_numpy(x_np).to(device)
    c = torch.from_numpy(cond[train_ids]).to(device)
    t = torch.arange(1, len(train_ids) + 1, device=device, dtype=torch.long).clamp(max=49)
    noise = torch.randn_like(x)
    ab = cosine_alpha_bar(50).to(device)
    loss, detail = training_loss(model, x, c, t, noise, ab, lambda_ctr=0.1, p_uncond=0.15)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite smoke loss")
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("non-finite gradient")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    opt.step()
    opt.zero_grad(set_to_none=True)

    ck = out_dir / "smoke_checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "dim": dim}, ck)
    model2 = NativeTVX0Denoiser(dim, cond_dim=cond.shape[1], hidden=1024, time_dim=64).to(device)
    state = torch.load(ck, map_location=device, weights_only=False)
    model2.load_state_dict(state["state_dict"], strict=True)
    model2.eval()
    pt, pv = purify_indices(
        model2, text, visual, cond, train_ids,
        t_edit=1, guidance=1.0, seeds=(20261001,), batch=smoke_items, device=device,
    )
    raw_t = np.asarray(text[train_ids])
    raw_v = np.asarray(visual[train_ids])
    idt, idv = blend_native(raw_t, raw_v, np.zeros_like(raw_t), np.zeros_like(raw_v), 0.0, 0.0)
    identity = bool(np.array_equal(idt, raw_t) and np.array_equal(idv, raw_v))

    summary = {
        "phase": "DIFFUSION_GENERIC_SMOKE",
        "dataset": dataset,
        "beta": float(beta),
        "n_items": int(asset_audit["condition_endpoint_shape"][0]),
        "embedding_dim": int(asset_audit["condition_endpoint_shape"][1]),
        "train_item_count": int(asset_audit["train_item_count"]),
        "text_dim": text_dim,
        "visual_dim": visual_dim,
        "joint_dim": dim,
        "condition_shape": list(cond.shape),
        "small_batch_forward": "PASS",
        "small_batch_backward": "PASS",
        "checkpoint_save_load": "PASS",
        "purified_text_shape": list(pt.shape),
        "purified_visual_shape": list(pv.shape),
        "finite_check": "PASS" if np.isfinite(pt).all() and np.isfinite(pv).all() else "FAIL",
        "rho0_feature_identity": "PASS" if identity else "FAIL",
        "TRAIN_ONLY_ITEM_AUDIT": asset_audit["TRAIN_ONLY_ITEM_AUDIT"],
        "TEST_ACCESSED": False,
        "DIFFUSION_FORMAL_TRAINING": "NOT_STARTED",
        "loss": float(loss.detach().cpu()),
        "loss_detail": detail,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--msca-assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--smoke-items", type=int, default=8)
    a = ap.parse_args()
    run(a.dataset, Path(a.msca_assets), Path(a.out), a.beta, a.smoke_items)


if __name__ == "__main__":
    main()
