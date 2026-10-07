from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from diffusion_experiments.models.round7_anchored_preference import (
    AnchoredPreferenceDDPM,
    anchor_start,
    cosine_alpha_bar,
    ddim_reverse_differentiable,
    inverse_standardize,
)
from diffusion_experiments.modules.round7_common import (
    cfg_round7,
    git_sha,
    load_interactions,
    sha,
    state_hash,
    train_frame,
    unique_histories,
)


def grad_stats(loss_a, loss_b, params):
    ga = torch.autograd.grad(loss_a, params, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss_b, params, retain_graph=True, allow_unused=True)
    aa = []
    bb = []
    for a, b, p in zip(ga, gb, params):
        aa.append(torch.zeros_like(p).reshape(-1) if a is None else a.reshape(-1))
        bb.append(torch.zeros_like(p).reshape(-1) if b is None else b.reshape(-1))
    va = torch.cat(aa)
    vb = torch.cat(bb)
    na = torch.linalg.vector_norm(va)
    nb = torch.linalg.vector_norm(vb)
    cos = torch.dot(va, vb) / (na * nb).clamp_min(1e-12)
    return float(na.detach()), float(nb.detach()), float(cos.detach())


def sample_uniform_negatives(users, histories_sets, observed_items, rng):
    out = np.empty(len(users), dtype=np.int64)
    for i, u in enumerate(users):
        h = histories_sets[int(u)]
        while True:
            cand = int(observed_items[int(rng.integers(0, len(observed_items)))])
            if cand not in h:
                out[i] = cand
                break
    return out


def save_resume(path, model, opt, step, event_rng, neg_rng, torch_gen, history, grad_history, bucket):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": int(step),
            "event_rng_state": event_rng.bit_generator.state,
            "neg_rng_state": neg_rng.bit_generator.state,
            "torch_generator_state": torch_gen.get_state(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "python_rng_state": random.getstate(),
            "history": history,
            "grad_history": grad_history,
            "bucket": bucket,
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-seed", type=int, required=True)
    ap.add_argument("--diffusion-seed", type=int, required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["smoke", "formal"], default="formal")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = cfg_round7()
    if int(a.diffusion_seed) not in [int(x) for x in cfg["diffusion_seeds"]]:
        raise RuntimeError("unregistered Round7 diffusion seed")
    if str(a.backbone_seed) not in {str(k) for k in cfg["backbones"].keys()}:
        raise RuntimeError("unregistered Round7 backbone seed")
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Round7 requires RTX5090 on cuda:0")

    out = Path(a.out)
    latest = out / "resume_latest.pt"
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise RuntimeError(f"refuse overwrite nonempty output {out}")
    out.mkdir(parents=True, exist_ok=True)
    assets = Path(a.assets)
    audit = json.loads((assets / "audit.json").read_text())
    if int(audit["backbone_seed"]) != int(a.backbone_seed):
        raise RuntimeError("asset/backbone mismatch")
    ev = np.load(assets / "events.npz")
    dep = np.load(assets / "deployment.npz")
    users_all = ev["users"].astype(np.int64)
    pos_all = ev["pos"].astype(np.int64)
    z_pos_all = ev["z_pos"].astype(np.float32)
    cond_all = ev["cond"].astype(np.float32)
    anchor_all = ev["anchor_std"].astype(np.float32)
    has_anchor_all = ev["has_anchor"].astype(bool)
    item_raw_np = dep["collab_item"].astype(np.float32)
    item_mean_np = dep["item_mean"].astype(np.float32)
    item_std_safe_np = dep["item_std_safe"].astype(np.float32)
    observed_items = dep["observed_items"].astype(np.int64)
    if cond_all.shape[1] != 129 or z_pos_all.shape[1] != 64:
        raise RuntimeError("Round7 asset dimensions invalid")

    train = train_frame(load_interactions())
    histories = unique_histories(train, int(audit["n_users"]))
    histories_sets = [set(map(int, h)) for h in histories]
    device = torch.device("cuda:0")
    dc = cfg["diffusion"]
    formal_updates = int(dc["total_updates"])
    formal_warmup = int(dc["warmup_updates"])
    updates = 80 if a.mode == "smoke" else formal_updates
    warmup = 20 if a.mode == "smoke" else formal_warmup
    diag_every = 20 if a.mode == "smoke" else int(dc["grad_diag_every"])
    checkpoint_every = 40 if a.mode == "smoke" else int(dc["checkpoint_every"])

    torch.manual_seed(int(a.diffusion_seed))
    torch.cuda.manual_seed_all(int(a.diffusion_seed))
    np.random.seed(int(a.diffusion_seed) & 0xFFFFFFFF)
    random.seed(int(a.diffusion_seed))
    model = AnchoredPreferenceDDPM(
        x_dim=64,
        cond_dim=129,
        hidden_dim=int(dc["hidden_dim"]),
        time_dim=int(dc["time_dim"]),
        dropout=float(dc["dropout"]),
        hidden_layers=int(dc["hidden_layers"]),
    ).to(device)
    initial_hash = state_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=float(dc["lr"]), weight_decay=float(dc["weight_decay"]))
    alpha = cosine_alpha_bar(int(dc["steps"]), float(dc["cosine_s"]), device=device)
    params = [p for p in model.parameters() if p.requires_grad]

    event_rng = np.random.default_rng(int(a.diffusion_seed) + 1000003 * int(a.backbone_seed))
    neg_rng = np.random.default_rng(int(a.diffusion_seed) + 2000003 * int(a.backbone_seed))
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(int(a.diffusion_seed) + 3000017 * int(a.backbone_seed))
    history = []
    grad_history = []
    bucket = {
        "count": np.zeros(5, np.int64),
        "signal2": np.zeros(5, np.float64),
        "noise2": np.zeros(5, np.float64),
        "den": np.zeros(5, np.float64),
    }
    start_step = 1
    if a.resume:
        if not latest.exists():
            raise RuntimeError("--resume requested but resume_latest.pt missing")
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        start_step = int(ck["step"]) + 1
        event_rng.bit_generator.state = ck["event_rng_state"]
        neg_rng.bit_generator.state = ck["neg_rng_state"]
        torch_gen.set_state(ck["torch_generator_state"])
        torch.set_rng_state(ck["cpu_rng_state"])
        torch.cuda.set_rng_state_all(ck["cuda_rng_state"])
        random.setstate(ck["python_rng_state"])
        history = ck["history"]
        grad_history = ck["grad_history"]
        bucket = ck["bucket"]

    item_raw = torch.as_tensor(item_raw_np, device=device)
    item_mean = torch.as_tensor(item_mean_np, device=device)
    item_std_safe = torch.as_tensor(item_std_safe_np, device=device)
    bs = int(dc["batch_size"])
    pref_weight = float(dc["pref_weight"])
    path = [int(x) for x in dc["ddim_path"]]
    t_edit = int(dc["t_edit"])
    sqrt_dim = math.sqrt(64.0)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    nonfinite = 0
    illegal_negative = 0
    pref_batches = 0

    for step in range(start_step, updates + 1):
        ix = event_rng.integers(0, len(users_all), size=bs)
        u_np = users_all[ix]
        p_np = pos_all[ix]
        n_np = sample_uniform_negatives(u_np, histories_sets, observed_items, neg_rng)
        illegal_negative += sum(int(n) in histories_sets[int(u)] for u, n in zip(u_np, n_np))
        u = torch.as_tensor(u_np, dtype=torch.long, device=device)
        p = torch.as_tensor(p_np, dtype=torch.long, device=device)
        n = torch.as_tensor(n_np, dtype=torch.long, device=device)
        z0 = torch.as_tensor(z_pos_all[ix], device=device)
        cond = torch.as_tensor(cond_all[ix], device=device)
        anchor = torch.as_tensor(anchor_all[ix], device=device)
        has_anchor = torch.as_tensor(has_anchor_all[ix], dtype=torch.bool, device=device)

        t = torch.randint(1, int(dc["steps"]) + 1, (bs,), generator=torch_gen, device=device)
        eps = torch.randn((bs, 64), generator=torch_gen, device=device)
        at = alpha[t].unsqueeze(1)
        signal = at.sqrt() * z0
        noise = (1.0 - at).clamp_min(0).sqrt() * eps
        x_t = signal + noise
        model.train()
        pred = model(x_t, t, cond)
        den_per = ((pred - z0) ** 2).mean(dim=1)
        den_loss = den_per.mean()

        pref_loss = torch.zeros((), device=device)
        if step > warmup and bool(has_anchor.any()):
            pref_batches += 1
            valid = has_anchor
            eps_a = torch.randn((int(valid.sum()), 64), generator=torch_gen, device=device)
            x20 = anchor_start(anchor[valid], eps_a, alpha, t_edit)
            was_training = model.training
            model.eval()
            g_std = ddim_reverse_differentiable(model, x20, cond[valid], alpha, path)
            model.train(was_training)
            g_raw = inverse_standardize(g_std, item_mean, item_std_safe)
            diff = item_raw[n[valid]] - item_raw[p[valid]]
            pref_loss = F.softplus((g_raw * diff).sum(dim=1) / sqrt_dim).mean()

        total = den_loss + (pref_weight * pref_loss if step > warmup else 0.0)
        if not torch.isfinite(total):
            nonfinite += 1
            raise RuntimeError(f"nonfinite loss at step {step}")

        if step > warmup and step % diag_every == 0:
            dn, pn, gc = grad_stats(den_loss, pref_loss, params)
            grad_history.append({
                "step": int(step),
                "den_grad_norm": dn,
                "pref_grad_norm": pn,
                "direction_cosine": gc,
                "pref_weight": pref_weight,
                "weighted_pref_grad_norm": pref_weight * pn,
            })
            if not np.isfinite(pn) or pn <= 0:
                raise RuntimeError(f"preference gradient invalid at step {step}: {pn}")

        opt.zero_grad(set_to_none=True)
        total.backward()
        raw_grad = float(torch.nn.utils.clip_grad_norm_(params, float(dc["grad_clip"])).detach())
        opt.step()

        with torch.no_grad():
            bi = torch.minimum((t - 1) // 10, torch.tensor(4, device=device)).detach().cpu().numpy()
            den_np = den_per.detach().cpu().numpy()
            sig_np = (signal ** 2).mean(dim=1).detach().cpu().numpy()
            noi_np = (noise ** 2).mean(dim=1).detach().cpu().numpy()
            for b in range(5):
                m = bi == b
                bucket["count"][b] += int(m.sum())
                bucket["den"][b] += float(den_np[m].sum())
                bucket["signal2"][b] += float(sig_np[m].sum())
                bucket["noise2"][b] += float(noi_np[m].sum())

        log_every = 10 if a.mode == "smoke" else 100
        if step == 1 or step % log_every == 0 or step == updates:
            history.append({
                "step": int(step),
                "den_loss": float(den_loss.detach()),
                "pref_loss": float(pref_loss.detach()),
                "total_loss": float(total.detach()),
                "raw_grad_norm_before_clip": raw_grad,
                "phase": "warmup" if step <= warmup else "joint",
            })
        if step % checkpoint_every == 0 and step < updates:
            save_resume(latest, model, opt, step, event_rng, neg_rng, torch_gen, history, grad_history, bucket)

    model.eval()
    diag_rng = np.random.default_rng(int(a.diffusion_seed) + 7000003 * int(a.backbone_seed))
    diag_ix = diag_rng.choice(len(users_all), size=min(2048, len(users_all)), replace=False)
    dz = torch.as_tensor(z_pos_all[diag_ix], device=device)
    dcnd = torch.as_tensor(cond_all[diag_ix], device=device)
    diagnostic = {}
    diag_gen = torch.Generator(device=device)
    diag_gen.manual_seed(int(a.diffusion_seed) + 9000011 * int(a.backbone_seed))
    with torch.no_grad():
        for ti in [1, 10, 25, 40, 50]:
            tt = torch.full((len(diag_ix),), ti, dtype=torch.long, device=device)
            ee = torch.randn((len(diag_ix), 64), generator=diag_gen, device=device)
            at = alpha[tt].unsqueeze(1)
            xt = at.sqrt() * dz + (1.0 - at).sqrt() * ee
            pp = model(xt, tt, dcnd)
            diagnostic[str(ti)] = {
                "MSE": float(((pp - dz) ** 2).mean()),
                "target_cos_mean": float(F.cosine_similarity(pp, dz, dim=1).mean()),
                "prediction_RMS": float(torch.sqrt((pp ** 2).mean())),
            }

    noise_buckets = []
    for b in range(5):
        c = max(int(bucket["count"][b]), 1)
        noise_buckets.append({
            "t_range": [b * 10 + 1, (b + 1) * 10],
            "count": int(bucket["count"][b]),
            "den_mse": float(bucket["den"][b] / c),
            "signal_rms": float(math.sqrt(bucket["signal2"][b] / c)),
            "noise_rms": float(math.sqrt(bucket["noise2"][b] / c)),
            "empirical_snr": float(bucket["signal2"][b] / max(bucket["noise2"][b], 1e-12)),
        })

    final_hash = state_hash(model)
    final_path = out / "generator.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "backbone_seed": int(a.backbone_seed),
            "diffusion_seed": int(a.diffusion_seed),
            "config": dc,
            "initial_hash": initial_hash,
            "final_hash": final_hash,
            "assets_audit_sha256": sha(assets / "audit.json"),
            "events_sha256": sha(assets / "events.npz"),
            "deployment_sha256": sha(assets / "deployment.npz"),
        },
        final_path,
    )
    if latest.exists():
        latest.unlink()
    result = {
        "status": "COMPLETE",
        "protocol_version": cfg["protocol_version"],
        "mode": a.mode,
        "backbone_seed": int(a.backbone_seed),
        "diffusion_seed": int(a.diffusion_seed),
        "updates": int(updates),
        "warmup_updates": int(warmup),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "initial_hash": initial_hash,
        "final_hash": final_hash,
        "generator_sha256": sha(final_path),
        "source_backbone_checkpoint_sha256": audit["source_checkpoint_sha256"],
        "assets_audit_sha256": sha(assets / "audit.json"),
        "history": history,
        "gradient_diagnostics": grad_history,
        "noise_buckets": noise_buckets,
        "final_train_denoising_diagnostic": diagnostic,
        "pref_batches": int(pref_batches),
        "illegal_negative_count": int(illegal_negative),
        "nonfinite_count": int(nonfinite),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_allocated_gib": float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_uuid": os.popen("nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0").read().strip(),
        "elapsed_seconds": time.time() - t0,
        "git_sha": git_sha(),
        "access": {"TRAIN": True, "VALIDATION_LABELS": False, "TEST_LABELS_USED": False},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETE",
        "mode": a.mode,
        "backbone_seed": int(a.backbone_seed),
        "diffusion_seed": int(a.diffusion_seed),
        "last": history[-1],
        "pref_grad_last": grad_history[-1] if grad_history else None,
        "peak_gib": result["peak_cuda_allocated_gib"],
        "sha": result["generator_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
