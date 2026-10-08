from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 64
T_LATENT = 20
RHO = 0.20
LAMBDA_REC = 1.0
LAMBDA_DIFF = 0.25
LAMBDA_USER = 0.5
USER_MARGIN = 0.05
T_INFER = 3
INFER_SEEDS = (20261301, 20261302)


def sinusoidal_time(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
    ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    out = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    return F.pad(out, (0, dim - out.shape[1])) if out.shape[1] < dim else out


def cosine_alpha_bar(steps: int = T_LATENT, s: float = .008, device='cpu') -> torch.Tensor:
    x = torch.arange(steps + 1, dtype=torch.float64, device=device)
    f = torch.cos(((x / steps + s) / (1 + s)) * math.pi / 2).square()
    return (f / f[0]).clamp(min=1e-8, max=1.0).float()


class ConditionEncoder(nn.Module):
    def __init__(self, in_dim=128, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))
    def forward(self, x):
        return self.net(x)


class LatentX0Denoiser(nn.Module):
    def __init__(self, latent_dim=64, cond_dim=64, hidden=256, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, 64), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + 64, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )
    def forward(self, ht, t, cond):
        te = self.time_mlp(sinusoidal_time(t, self.time_dim))
        return self.net(torch.cat([ht, cond, te], dim=-1))


class IUDLPModules(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = ConditionEncoder(128, 128, 64)
        self.text = LatentX0Denoiser(64, 64, 256, 64)
        self.visual = LatentX0Denoiser(64, 64, 256, 64)


def freeze_recommender(model):
    for p in model.parameters():
        p.requires_grad = False


def parameter_audit(model, modules=None):
    rows = []
    for n, p in model.named_parameters():
        rows.append({'name': n, 'shape': list(p.shape), 'numel': p.numel(), 'trainable': bool(p.requires_grad), 'group': 'recommender'})
    if modules is not None:
        for n, p in modules.named_parameters():
            rows.append({'name': 'iudlp.' + n, 'shape': list(p.shape), 'numel': p.numel(), 'trainable': bool(p.requires_grad), 'group': 'diffusion'})
    return rows


def forward_components(model):
    image_pre = model.item_id_embedding.weight * model.gate_v(model.image_trs(model.image_embedding.weight))
    text_pre = model.item_id_embedding.weight * model.gate_t(model.text_trs(model.text_embedding.weight))
    ego = torch.cat([model.user_embedding.weight, model.item_id_embedding.weight], dim=0)
    xs = []
    for _ in range(model.n_layers):
        ego = torch.sparse.mm(model.norm_adj, ego)
        xs.append(ego)
    collab = torch.stack(xs, dim=1).mean(dim=1)
    struct = model.semantic_encode(model.struct_original_adj, model.item_id_embedding.weight)
    image = model.semantic_encode(model.image_original_adj, image_pre)
    text = model.semantic_encode(model.text_original_adj, text_pre)
    views = [image, text, struct]
    weights = model.softmax(torch.cat([model.query_common(v) for v in views], dim=-1))
    redundant = sum(w.unsqueeze(1) * v for w, v in zip(weights.unbind(dim=1), views))
    multi = sum(views) - redundant
    final = collab + model.fusion_coeff * multi
    fu, fi = torch.split(final, [model.n_users, model.n_items], dim=0)
    cu, ci = torch.split(collab, [model.n_users, model.n_items], dim=0)
    su, si = torch.split(struct, [model.n_users, model.n_items], dim=0)
    iu, ii = torch.split(image, [model.n_users, model.n_items], dim=0)
    tu, ti = torch.split(text, [model.n_users, model.n_items], dim=0)
    return {'final_user': fu, 'final_item': fi, 'collab_user': cu, 'collab_item': ci, 'struct_user': su, 'struct_item': si, 'image_user': iu, 'image_item': ii, 'text_user': tu, 'text_item': ti}


def fuse_item(model, collab, image, text, struct):
    views = [image, text, struct]
    weights = model.softmax(torch.cat([model.query_common(v) for v in views], dim=-1))
    redundant = sum(w.unsqueeze(1) * v for w, v in zip(weights.unbind(dim=1), views))
    multi = sum(views) - redundant
    return collab + model.fusion_coeff * multi


def condition(modules, user_cf, item_cf):
    return modules.condition(torch.cat([user_cf, item_cf], dim=-1))


def noisy_state(h, t, alpha_bar, noise):
    x = F.normalize(h, p=2, dim=1)
    a = alpha_bar[t].unsqueeze(1)
    ht = torch.sqrt(a) * x + torch.sqrt((1.0 - a).clamp_min(0)) * noise
    return x, ht


def tangent_augment(h, residual, rho=RHO):
    if float(rho) == 0.0:
        return h
    n = h.norm(dim=1, keepdim=True).clamp_min(1e-8)
    x = F.normalize(h, p=2, dim=1)
    tang = residual - (x * residual).sum(dim=1, keepdim=True) * x
    d = F.normalize(x + float(rho) * tang, p=2, dim=1)
    return d * n


def differential_prediction(denoiser, modules, h, user_cf, item_cf, variant, t, alpha_bar, noise):
    x, ht = noisy_state(h, t, alpha_bar, noise)
    zeros_u = torch.zeros_like(user_cf)
    zeros_i = torch.zeros_like(item_cf)
    if variant == 'D0':
        c_main = condition(modules, zeros_u, item_cf)
        c_ref = condition(modules, zeros_u, zeros_i)
    elif variant in ('D1', 'D2'):
        c_main = condition(modules, user_cf, item_cf)
        c_ref = condition(modules, zeros_u, item_cf)
    else:
        raise ValueError(variant)
    p_main = denoiser(ht, t, c_main)
    p_ref = denoiser(ht, t, c_ref)
    residual = p_main - p_ref
    diff = F.mse_loss(p_main, x.detach())
    return residual, diff, p_main, p_ref, x, ht


def shuffled_user_ids(users, n_users):
    if n_users <= 2:
        raise RuntimeError('not enough users for shuffled-user supervision')
    out = ((users - 1 + 7919) % (n_users - 1)) + 1
    same = out.eq(users)
    if same.any():
        out[same] = (out[same] % (n_users - 1)) + 1
    return out


def training_loss(model, modules, interaction, c, variant, alpha_bar):
    if variant not in ('D0', 'D1', 'D2'):
        raise ValueError(variant)
    users, pos, neg = interaction[0], interaction[1], interaction[2]
    b = len(users)
    ids = torch.cat([pos, neg], dim=0)
    uids = torch.cat([users, users], dim=0)
    user_cf = c['collab_user'][uids]
    item_cf = c['collab_item'][ids]
    t_user = torch.randint(1, T_LATENT + 1, (b,), device=users.device)
    t = torch.cat([t_user, t_user], dim=0)
    noise_t = torch.randn((2 * b, LATENT_DIM), device=users.device)
    noise_v = torch.randn((2 * b, LATENT_DIM), device=users.device)
    rt, ldt, _, _, _, hnt = differential_prediction(modules.text, modules, c['text_item'][ids], user_cf, item_cf, variant, t, alpha_bar, noise_t)
    rv, ldv, _, _, _, hnv = differential_prediction(modules.visual, modules, c['image_item'][ids], user_cf, item_cf, variant, t, alpha_bar, noise_v)
    at = tangent_augment(c['text_item'][ids], rt, RHO)
    av = tangent_augment(c['image_item'][ids], rv, RHO)
    af = fuse_item(model, item_cf, av, at, c['struct_item'][ids])
    p_aug, n_aug = af[:b], af[b:]
    true_u = c['final_user'][users]
    m_true = (true_u * p_aug).sum(1) - (true_u * n_aug).sum(1)
    lrec = F.softplus(-m_true).mean()
    luser = lrec.new_zeros(())
    m_shuf = None
    if variant == 'D2':
        wrong = shuffled_user_ids(users, model.n_users)
        wrong_ids = torch.cat([wrong, wrong], dim=0)
        wrong_cf = c['collab_user'][wrong_ids]
        zeros_u = torch.zeros_like(wrong_cf)
        c_wrong = condition(modules, wrong_cf, item_cf)
        c_zero = condition(modules, zeros_u, item_cf)
        pt_w = modules.text(hnt, t, c_wrong)
        pt_0 = modules.text(hnt, t, c_zero)
        pv_w = modules.visual(hnv, t, c_wrong)
        pv_0 = modules.visual(hnv, t, c_zero)
        at_w = tangent_augment(c['text_item'][ids], pt_w - pt_0, RHO)
        av_w = tangent_augment(c['image_item'][ids], pv_w - pv_0, RHO)
        af_w = fuse_item(model, item_cf, av_w, at_w, c['struct_item'][ids])
        p_w, n_w = af_w[:b], af_w[b:]
        m_shuf = (true_u * p_w).sum(1) - (true_u * n_w).sum(1)
        luser = F.softplus(m_shuf - m_true + USER_MARGIN).mean()
    ldiff = 0.5 * (ldt + ldv)
    total = LAMBDA_REC * lrec + LAMBDA_DIFF * ldiff
    if variant == 'D2':
        total = total + LAMBDA_USER * luser
    return total, {'L_rec': lrec, 'L_diff_text': ldt, 'L_diff_visual': ldv, 'L_diff': ldiff, 'L_user': luser, 'margin_true_mean': m_true.detach().mean(), 'margin_shuf_mean': (m_true.new_zeros(()) if m_shuf is None else m_shuf.detach().mean())}


def build_noise_tables(n_items, device):
    out = {}
    for seed in INFER_SEEDS:
        for modality, offset in [('text', 0), ('visual', 100003)]:
            g = torch.Generator(device=device)
            g.manual_seed(int(seed + offset))
            out[(seed, modality)] = torch.randn((n_items, LATENT_DIM), generator=g, device=device)
    return out


@torch.no_grad()
def inference_residuals(model, modules, c, user_ids, item_ids, variant, alpha_bar, noise_tables):
    if variant not in ('D0', 'D1', 'D2'):
        raise ValueError(variant)
    user_cf = c['collab_user'][user_ids]
    item_cf = c['collab_item'][item_ids]
    zeros_u = torch.zeros_like(user_cf)
    zeros_i = torch.zeros_like(item_cf)
    t = torch.full((len(item_ids),), T_INFER, device=item_ids.device, dtype=torch.long)
    res_t, res_v = [], []
    for seed in INFER_SEEDS:
        cur = []
        for modality, den, h in [('text', modules.text, c['text_item'][item_ids]), ('visual', modules.visual, c['image_item'][item_ids])]:
            noise = noise_tables[(seed, modality)][item_ids]
            _, ht = noisy_state(h, t, alpha_bar, noise)
            if variant == 'D0':
                cm = condition(modules, zeros_u, item_cf)
                cr = condition(modules, zeros_u, zeros_i)
            else:
                cm = condition(modules, user_cf, item_cf)
                cr = condition(modules, zeros_u, item_cf)
            cur.append(den(ht, t, cm) - den(ht, t, cr))
        res_t.append(cur[0])
        res_v.append(cur[1])
    return torch.stack(res_t, 0).mean(0), torch.stack(res_v, 0).mean(0)


@torch.no_grad()
def pair_scores(model, modules, c, user_ids, item_ids, variant, alpha_bar, noise_tables, rho=RHO, condition_user_ids=None):
    cond_users = user_ids if condition_user_ids is None else condition_user_ids
    rt, rv = inference_residuals(model, modules, c, cond_users, item_ids, variant, alpha_bar, noise_tables)
    at = tangent_augment(c['text_item'][item_ids], rt, rho)
    av = tangent_augment(c['image_item'][item_ids], rv, rho)
    base = c['final_item'][item_ids]
    u = c['final_user'][user_ids]
    base_score = (u * base).sum(1)
    if float(rho) == 0.0:
        aug_score = base_score
    else:
        aug = fuse_item(model, c['collab_item'][item_ids], av, at, c['struct_item'][item_ids])
        aug_score = (u * aug).sum(1)
    return base_score, aug_score, rt, rv


def grad_l2(params):
    vals = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
    return float(torch.sqrt(torch.stack(vals).sum()).item()) if vals else 0.0
