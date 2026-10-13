from __future__ import annotations

"""Preference-Directed Diffusion Boundary Refinement (PDBR).

Diffusion is a training-only auxiliary module. Formal Validation/Test ranking
never calls this network.
"""

import hashlib
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import init_seed

DIM = 64
DIFFUSION_STEPS = 25
BETA_START = 1e-4
BETA_END = 0.02
EPS = 1e-8
BANDS = ((60, 100), (30, 60), (10, 30))
BAND_NAMES = ("Easy", "Medium", "Hard")


def stats(values):
    x = np.asarray(values, np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p10": float(np.quantile(x, .1)), "p50": float(np.quantile(x, .5)),
            "p90": float(np.quantile(x, .9)), "p99": float(np.quantile(x, .99)),
            "min": float(x.min()), "max": float(x.max())}


def state_hash(state_dict):
    h = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(np.asarray(value.shape, dtype=np.int64).tobytes()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def time_embedding(t, dim=DIM):
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32)
                     * (-math.log(10000.) / (half - 1)))
    phase = t.float()[:, None] * freq[None, :]
    return torch.cat([phase.sin(), phase.cos()], 1)


class PreferenceDiffusion(nn.Module):
    def __init__(self, dim=DIM, hidden=256, dropout=.1):
        super().__init__(); self.dim = int(dim)
        self.net = nn.Sequential(nn.Linear(dim * 5, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight); nn.init.zeros_(module.bias)

    def forward(self, xt, t, user, text, visual):
        return self.net(torch.cat([xt, time_embedding(t, self.dim), user, text, visual], 1))


class LinearDDPM:
    def __init__(self, device):
        self.device = torch.device(device)
        self.betas = torch.linspace(BETA_START, BETA_END, DIFFUSION_STEPS, device=self.device)
        self.alphas = 1 - self.betas; self.ac = torch.cumprod(self.alphas, 0)
        self.acprev = F.pad(self.ac[:-1], (1, 0), value=1.)
        self.sqrt_ac = torch.sqrt(self.ac); self.sqrt_om = torch.sqrt(1 - self.ac)

    @staticmethod
    def ext(values, t, x):
        return values[t].reshape(len(t), 1).to(x.device)

    def q_sample(self, x0, t, eps):
        return self.ext(self.sqrt_ac, t, x0) * x0 + self.ext(self.sqrt_om, t, x0) * eps

    def x0_from_eps(self, xt, t, eps):
        return (xt - self.ext(self.sqrt_om, t, xt) * eps) / torch.clamp(self.ext(self.sqrt_ac, t, xt), min=1e-8)


def forward_bundle(model):
    fu, fi, collab, struct, image, text = model.forward(test=False)
    _, image_item = torch.split(image, [model.n_users, model.n_items], 0)
    _, text_item = torch.split(text, [model.n_users, model.n_items], 0)
    return {"fu": fu, "fi": fi, "collab": collab, "struct": struct, "image": image, "text": text,
            "image_item": image_item, "text_item": text_item}


def msca_loss_from_bundle(model, interaction, bundle):
    users, pos, neg = interaction[0], interaction[1], interaction[2]
    bpr = model.cal_bpr_loss(bundle["fu"][users], bundle["fi"][pos], bundle["fi"][neg]); reg = model.cal_reg_loss()
    cu, ci = torch.split(bundle["collab"], [model.n_users, model.n_items], 0)
    su, si = torch.split(bundle["struct"], [model.n_users, model.n_items], 0)
    vu, vi = torch.split(bundle["image"], [model.n_users, model.n_items], 0)
    tu, ti = torch.split(bundle["text"], [model.n_users, model.n_items], 0)
    mu = model.cal_cl_loss(cu[users], vu[users], model.tau) + model.cal_cl_loss(cu[users], tu[users], model.tau)
    mi = model.cal_cl_loss(ci[pos], vi[pos], model.tau) + model.cal_cl_loss(ci[pos], ti[pos], model.tau)
    cl = model.cal_cl_loss(cu[users], su[users], model.tau) + model.cal_cl_loss(ci[pos], si[pos], model.tau)
    return bpr + model.cl_weight * (cl + mu + mi) + model.reg_weight * reg


def _scale(x):
    mu = x.mean(0); rms = torch.sqrt(((x - mu) ** 2).mean())
    if float(rms) < 1e-6: raise RuntimeError("latent RMS < 1e-6")
    return mu.detach(), rms.detach()


@torch.no_grad()
def latent_stats(model):
    model.eval(); bundle = forward_bundle(model)
    mu_i, r_i = _scale(bundle["fi"]); mu_u, r_u = _scale(bundle["fu"])
    mu_t, r_t = _scale(bundle["text_item"]); mu_v, r_v = _scale(bundle["image_item"])
    return {"muI": mu_i, "rI": r_i, "muU": mu_u, "rU": r_u,
            "muT": mu_t, "rT": r_t, "muV": mu_v, "rV": r_v}


def z(x, mu, rms): return (x - mu) / rms
def inv_item(x, latent): return latent["muI"] + latent["rI"] * x

def curriculum_count(epoch): return 0 if epoch < 10 else 1 if epoch < 20 else 2 if epoch < 30 else 3


def aux_bpr(user, positive, negatives):
    if not negatives: return torch.zeros((), device=user.device)
    neg = torch.stack(negatives, 1); ps = (user * positive).sum(1, keepdim=True)
    ns = torch.einsum("bd,bkd->bk", user, neg)
    return -F.logsigmoid(ps - ns).mean()


def valid_mask(train_data, n_items, device):
    mask = torch.zeros(n_items, device=device, dtype=torch.bool)
    mask[torch.as_tensor(np.asarray(train_data.all_items, dtype=np.int64), device=device)] = True
    return mask


def rank_offsets(seed, epoch, batch_idx, batch_size, active):
    rng = np.random.default_rng(20281000 + int(seed) * 100000 + int(epoch) * 1000 + int(batch_idx))
    return [rng.integers(lo, hi, size=int(batch_size), dtype=np.int64) for lo, hi in BANDS[:active]]


@torch.no_grad()
def mine_current_ids(bundle, interaction, histories, mask, seed, epoch, batch_idx, active):
    if active <= 0:
        return [], [], ""
    users, pos = interaction[0], interaction[1]
    score = bundle["fu"][users].detach() @ bundle["fi"].detach().T
    score[:, ~mask] = -torch.inf
    for row, (user, positive) in enumerate(zip(users.detach().cpu().tolist(), pos.detach().cpu().tolist())):
        seen = histories[int(user)]
        if seen:
            score[row, torch.as_tensor(seen, device=score.device, dtype=torch.long)] = -torch.inf
        score[row, int(positive)] = -torch.inf
    values, top = torch.topk(score, 100, dim=1)
    if not torch.isfinite(values[:, -1]).all():
        raise RuntimeError("fewer than 100 legal TRAIN items")
    offsets = rank_offsets(seed, epoch, batch_idx, len(users), active)
    rows = torch.arange(len(users), device=score.device)
    ids = [top[rows, torch.as_tensor(off, device=score.device, dtype=torch.long)] for off in offsets]
    digest = hashlib.sha256()
    for off in offsets:
        digest.update(np.asarray(off, dtype=np.int64).tobytes())
    return ids, offsets, digest.hexdigest()


def make_diffusion(seed, device, lr=1e-3):
    rng = capture_rng()
    init_seed(20263400 + int(seed))
    net = PreferenceDiffusion().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))
    schedule = LinearDDPM(device)
    restore_rng(rng)
    return net, optimizer, schedule


def standardized_conditions(bundle, users, item_ids, latent):
    x0 = z(bundle["fi"][item_ids].detach(), latent["muI"], latent["rI"])
    hu = z(bundle["fu"][users].detach(), latent["muU"], latent["rU"])
    tc = z(bundle["text_item"][item_ids].detach(), latent["muT"], latent["rT"])
    vc = z(bundle["image_item"][item_ids].detach(), latent["muV"], latent["rV"])
    return x0, hu, tc, vc


def local_xt(schedule, x0, generator, local_t=(1, 2, 3, 4, 5)):
    values = tuple(int(x) for x in local_t)
    if values != tuple(range(min(values), max(values) + 1)):
        raise ValueError("local_t must be contiguous")
    t = torch.randint(min(values), max(values) + 1, (len(x0),), device=x0.device, generator=generator)
    eps = torch.randn(x0.shape, device=x0.device, generator=generator)
    return schedule.q_sample(x0, t, eps), t, eps


def branches(net, schedule, xt, t, true_user, shuffled_user, text, visual):
    zero = torch.zeros_like(true_user)
    batch = len(true_user)
    pred = net(
        torch.cat([xt, xt, xt], 0),
        torch.cat([t, t, t], 0),
        torch.cat([zero, true_user, shuffled_user], 0),
        torch.cat([text, text, text], 0),
        torch.cat([visual, visual, visual], 0),
    )
    eps_base, eps_true, eps_shuf = torch.split(pred, batch, 0)
    return (
        eps_base, eps_true, eps_shuf,
        schedule.x0_from_eps(xt, t, eps_base),
        schedule.x0_from_eps(xt, t, eps_true),
        schedule.x0_from_eps(xt, t, eps_shuf),
    )


def tangent(anchor, vector):
    h = anchor.detach()
    return vector - ((vector * h).sum(1, keepdim=True) / ((h * h).sum(1, keepdim=True) + EPS)) * h


def unit(vector):
    norm = vector.norm(dim=1, keepdim=True)
    return vector / (norm + EPS), norm.squeeze(1)


def preference_direction(positive, anchor):
    direction, norm = unit(tangent(anchor.detach(), positive.detach() - anchor.detach()))
    return direction.detach(), norm.detach()


def _concat_active(bundle, interaction, latent, target_ids):
    users = interaction[0]
    batch = len(users)
    active = len(target_ids)
    users_all = torch.cat([users for _ in range(active)], 0)
    ids_all = torch.cat(target_ids, 0)
    x0, hu, tc, vc = standardized_conditions(bundle, users_all, ids_all, latent)
    shuffled = torch.cat([hu[j * batch:(j + 1) * batch].roll(1, 0) for j in range(active)], 0)
    anchor = bundle["fi"][ids_all].detach()
    positive = torch.cat([bundle["fi"][interaction[1]].detach() for _ in range(active)], 0)
    return x0, hu, shuffled, tc, vc, anchor, positive, batch


def warmup_step(net, optimizer, schedule, bundle, interaction, latent, seed, local_t=(1, 2, 3, 4, 5)):
    rng = capture_rng()
    generator = torch.Generator(device=bundle["fu"].device)
    generator.manual_seed(int(seed))
    x0, hu, tc, vc = standardized_conditions(bundle, interaction[0], interaction[1], latent)
    shuffled = hu.roll(1, 0)
    xt, t, eps = local_xt_fn(schedule, x0, generator, local_t)
    net.train(); optimizer.zero_grad(set_to_none=True)
    eps_base, eps_true, eps_shuf, *_ = branches(net, schedule, xt, t, hu, shuffled, tc, vc)
    loss_rec = ((eps_base - eps).pow(2).mean() + (eps_true - eps).pow(2).mean() + (eps_shuf - eps).pow(2).mean()) / 3.0
    if not torch.isfinite(loss_rec):
        raise RuntimeError("nonfinite preference-diffusion warmup reconstruction")
    loss_rec.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step(); restore_rng(rng)
    return {"total": float(loss_rec.detach()), "rec": float(loss_rec.detach()), "dir": 0.0,
            "t_min": int(t.min()), "t_max": int(t.max())}


def local_xt_fn(schedule, x0, generator, values):
    return local_xt(schedule, x0, generator, local_t=values)


def direction_inner_step(net, optimizer, schedule, bundle, interaction, latent, target_ids, seed,
                         local_t=(1, 2, 3, 4, 5)):
    if not target_ids:
        raise RuntimeError("active boundary ids required")
    rng = capture_rng(); generator = torch.Generator(device=bundle["fu"].device); generator.manual_seed(int(seed))
    x0, hu, shuffled, tc, vc, anchor, positive, _ = _concat_active(bundle, interaction, latent, target_ids)
    xt, t, eps = local_xt_fn(schedule, x0, generator, local_t)
    pref, pref_norm = preference_direction(positive, anchor)
    net.train(); optimizer.zero_grad(set_to_none=True)
    eps_base, eps_true, eps_shuf, x_base, x_true, _ = branches(net, schedule, xt, t, hu, shuffled, tc, vc)
    h_base = inv_item(x_base, latent); h_true = inv_item(x_true, latent)
    direction, residual_norm = unit(tangent(anchor, h_true - h_base))
    valid = (pref_norm > EPS) & (residual_norm > EPS)
    cosine = (direction * pref).sum(1).clamp(-1, 1)
    loss_vector = 1.0 - cosine
    loss_dir = loss_vector[valid].mean() if bool(valid.any()) else loss_vector.mean() * 0.0
    loss_rec = ((eps_base - eps).pow(2).mean() + (eps_true - eps).pow(2).mean() + (eps_shuf - eps).pow(2).mean()) / 3.0
    total = loss_rec + loss_dir
    if not torch.isfinite(total):
        raise RuntimeError("nonfinite preference-direction loss")
    total.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step(); restore_rng(rng)
    return {"total": float(total.detach()), "rec": float(loss_rec.detach()), "dir": float(loss_dir.detach()),
            "t_min": int(t.min()), "t_max": int(t.max()),
            "valid_direction_fraction": float(valid.float().mean().detach())}


@torch.no_grad()
def predict_directions(net, schedule, bundle, interaction, latent, target_ids, seed,
                       local_t=(1, 2, 3, 4, 5)):
    if not target_ids:
        return [], [], []
    generator = torch.Generator(device=bundle["fu"].device); generator.manual_seed(int(seed))
    x0, hu, shuffled, tc, vc, anchor, positive, batch = _concat_active(bundle, interaction, latent, target_ids)
    xt, t, eps = local_xt_fn(schedule, x0, generator, local_t)
    net.eval()
    eps_base, eps_true, eps_shuf, x_base, x_true, x_shuf = branches(net, schedule, xt, t, hu, shuffled, tc, vc)
    h_base = inv_item(x_base, latent); h_true = inv_item(x_true, latent); h_shuf = inv_item(x_shuf, latent)
    d_true, n_true = unit(tangent(anchor, h_true - h_base))
    d_shuf, n_shuf = unit(tangent(anchor, h_shuf - h_base))
    d_pref, n_pref = preference_direction(positive, anchor)
    a_true = (d_true * d_pref).sum(1).clamp(-1, 1); a_shuf = (d_shuf * d_pref).sum(1).clamp(-1, 1)
    delta_a = a_true - a_shuf
    rec = ((eps_base - eps).pow(2).mean(1) + (eps_true - eps).pow(2).mean(1) + (eps_shuf - eps).pow(2).mean(1)) / 3.0
    directions = [d_true[j * batch:(j + 1) * batch].detach() for j in range(len(target_ids))]
    diagnostics = []
    for j in range(len(target_ids)):
        sl = slice(j * batch, (j + 1) * batch)
        diagnostics.append({"band": BAND_NAMES[j], "A_true": a_true[sl].cpu().numpy(),
                            "A_shuf": a_shuf[sl].cpu().numpy(), "DeltaA": delta_a[sl].cpu().numpy(),
                            "raw_residual_norm": n_true[sl].cpu().numpy(),
                            "shuffled_raw_residual_norm": n_shuf[sl].cpu().numpy(),
                            "preference_tangent_norm": n_pref[sl].cpu().numpy(),
                            "near_zero_residual": (n_true[sl] < EPS).cpu().numpy(),
                            "base_true_shuf_reconstruction_mse": rec[sl].cpu().numpy(),
                            "t": t[sl].cpu().numpy()})
    pref_dirs = [d_pref[j * batch:(j + 1) * batch].detach() for j in range(len(target_ids))]
    return directions, diagnostics, pref_dirs


@torch.no_grad()
def calibrate(user, positive, anchor, direction, half_margin=.25, theta_max_deg=5.0):
    d = direction.detach(); h = anchor.detach(); u = user.detach(); p = positive.detach()
    positive_score = (u * p).sum(1); negative_score = (u * h).sum(1)
    margin = positive_score - negative_score
    requested_shift = float(half_margin) * torch.clamp(margin, min=0)
    directional_gain = (u * d).sum(1)
    eligible = (margin > 0) & (directional_gain > 0)
    required = torch.zeros_like(requested_shift)
    required[eligible] = requested_shift[eligible] / (directional_gain[eligible] + EPS)
    lambda_max = math.tan(math.radians(float(theta_max_deg))) * h.norm(dim=1)
    applied = torch.where(eligible, torch.minimum(required, lambda_max), torch.zeros_like(required))
    delta = (applied.unsqueeze(1) * d).detach(); refined = h + delta
    refined_score = (u * refined).sum(1); target = negative_score + requested_shift
    actual_shift = refined_score - negative_score
    cosine = F.cosine_similarity(h, refined, dim=1).clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cosine))
    cap_hit = eligible & (required > lambda_max + 1e-10)
    reachable = eligible & (required <= lambda_max + 1e-10)
    diagnostic = {
        "positive_margin": (margin > 0).cpu().numpy(),
        "already_hard_no_refine": (margin <= 0).cpu().numpy(),
        "direction_positive": (directional_gain > 0).cpu().numpy(),
        "direction_miss": ((margin > 0) & (directional_gain <= 0)).cpu().numpy(),
        "applied_refinement": (applied > 0).cpu().numpy(),
        "requested_score_shift": requested_shift.cpu().numpy(),
        "actual_score_shift": actual_shift.cpu().numpy(),
        "target_score": target.cpu().numpy(),
        "refined_score": refined_score.cpu().numpy(),
        "positive_score": positive_score.cpu().numpy(),
        "real_negative_score": negative_score.cpu().numpy(),
        "target_hit_abs_error": (refined_score - target).abs().cpu().numpy(),
        "target_reachable": reachable.cpu().numpy(),
        "cap_hit": cap_hit.cpu().numpy(),
        "angle_deg": angle.cpu().numpy(),
        "lambda_req": required.cpu().numpy(),
        "lambda_applied": applied.cpu().numpy(),
        "lambda_max": lambda_max.cpu().numpy(),
        "real_negative_margin": margin.cpu().numpy(),
        "refined_negative_margin": (positive_score - refined_score).cpu().numpy(),
        "G_hard": actual_shift.cpu().numpy(),
        "target_never_exceeds_positive": ((margin <= 0) | (target <= positive_score + 1e-7)).cpu().numpy(),
    }
    return delta, diagnostic


def _gt_zero(values):
    x = np.asarray(values, np.float64); out = stats(x); out["p_gt_zero"] = float((x > 0).mean()); return out


def summarize(direction_batches, calibration_batches, diffusion_losses=None, theta_max_deg=5.0):
    direction_rows = [row for batch in direction_batches for row in batch]
    calibration_rows = [row for batch in calibration_batches for row in batch]
    if not calibration_rows:
        return {}
    catd = lambda key: np.concatenate([row[key] for row in direction_rows])
    catc = lambda key: np.concatenate([row[key] for row in calibration_rows])
    out = {
        "A_true": stats(catd("A_true")), "A_shuf": stats(catd("A_shuf")), "DeltaA": _gt_zero(catd("DeltaA")),
        "raw_residual_norm": stats(catd("raw_residual_norm")),
        "shuffled_raw_residual_norm": stats(catd("shuffled_raw_residual_norm")),
        "near_zero_residual_fraction": float(catd("near_zero_residual").mean()),
        "base_true_shuf_reconstruction_mse": stats(catd("base_true_shuf_reconstruction_mse")),
        "t": stats(catd("t")),
        "positive_margin_fraction": float(catc("positive_margin").mean()),
        "already_hard_no_refine_fraction": float(catc("already_hard_no_refine").mean()),
        "direction_positive_fraction": float(catc("direction_positive").mean()),
        "direction_miss_fraction": float(catc("direction_miss").mean()),
        "applied_refinement_fraction": float(catc("applied_refinement").mean()),
        "requested_score_shift": stats(catc("requested_score_shift")),
        "actual_score_shift": stats(catc("actual_score_shift")),
        "target_score": stats(catc("target_score")),
        "target_hit_abs_error": stats(catc("target_hit_abs_error")),
        "target_reachable_fraction": float(catc("target_reachable").mean()),
        "cap_hit_fraction": float(catc("cap_hit").mean()),
        "angle_deg": stats(catc("angle_deg")),
        "lambda_req": stats(catc("lambda_req")), "lambda_applied": stats(catc("lambda_applied")),
        "real_negative_margin": stats(catc("real_negative_margin")),
        "refined_negative_margin": stats(catc("refined_negative_margin")), "G_hard": _gt_zero(catc("G_hard")),
        "target_never_exceeds_positive": bool(catc("target_never_exceeds_positive").all()),
    }
    out["angle_bound_pass"] = bool(out["angle_deg"]["max"] <= float(theta_max_deg) + .01)
    if diffusion_losses:
        out["L_total"] = stats([row["total"] for row in diffusion_losses])
        out["L_rec"] = stats([row["rec"] for row in diffusion_losses])
        out["L_dir"] = stats([row["dir"] for row in diffusion_losses])
    return out
