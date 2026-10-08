from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from diffusion_experiments.round13_iudlp import latent_diffusion as r13
from diffusion_experiments.round14_aihu import anchored_latent_diffusion as r14

LATENT_DIM = r13.LATENT_DIM
T_LATENT = r13.T_LATENT
RHO = r13.RHO
T_INFER = r13.T_INFER
INFER_SEEDS = r13.INFER_SEEDS
LAMBDA_REC = 1.0
LAMBDA_USER = 0.5
USER_MARGIN = 0.05
LAMBDA_BASE = 0.25
LAMBDA_BOUND = 0.05
THETA_MAX_DEG = 5.0
TANGENT_NORM_CAP = math.tan(math.radians(THETA_MAX_DEG)) / RHO

BAURPModules = r13.IUDLPModules
freeze_recommender = r13.freeze_recommender
parameter_audit = r13.parameter_audit
forward_components = r13.forward_components
fuse_item = r13.fuse_item
condition = r13.condition
noisy_state = r13.noisy_state
shuffled_user_ids = r13.shuffled_user_ids
cosine_alpha_bar = r13.cosine_alpha_bar
build_noise_tables = r13.build_noise_tables
grad_l2 = r13.grad_l2


def _predictions(denoiser, modules, h, user_cf, item_cf, t, alpha_bar, noise, wrong_cf=None):
    x, ht = noisy_state(h, t, alpha_bar, noise)
    z = torch.zeros_like(user_cf)
    ct = condition(modules, user_cf, item_cf)
    c0 = condition(modules, z, item_cf)
    pt = denoiser(ht, t, ct)
    p0 = denoiser(ht, t, c0)
    ps = None
    if wrong_cf is not None:
        cs = condition(modules, wrong_cf, item_cf)
        ps = denoiser(ht, t, cs)
    return x, ht, pt, p0, ps


def tangent_raw(h, residual):
    x = F.normalize(h, p=2, dim=1)
    tang = residual - (x * residual).sum(dim=1, keepdim=True) * x
    return x, tang


def _angle_deg(a, b):
    aa = F.normalize(a, p=2, dim=1)
    bb = F.normalize(b, p=2, dim=1)
    c = (aa * bb).sum(dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(c))


def tangent_augment(h, residual, bounded: bool, rho=RHO, return_diag=False):
    if float(rho) == 0.0:
        z = torch.zeros(len(h), device=h.device, dtype=h.dtype)
        return (h, {'raw_tangent_norm': z, 'bounded_tangent_norm': z, 'angle_deg': z, 'cap_saturated': z.bool()}) if return_diag else h
    n = h.norm(dim=1, keepdim=True).clamp_min(1e-8)
    x, tang = tangent_raw(h, residual)
    raw_norm = tang.norm(dim=1)
    if bounded:
        scale = (TANGENT_NORM_CAP / raw_norm.clamp_min(1e-12)).clamp(max=1.0)
        tang_used = tang * scale.unsqueeze(1)
    else:
        tang_used = tang
    d = F.normalize(x + float(rho) * tang_used, p=2, dim=1)
    out = d * n
    if not return_diag:
        return out
    diag = {
        'raw_tangent_norm': raw_norm,
        'bounded_tangent_norm': tang_used.norm(dim=1),
        'angle_deg': _angle_deg(h, out),
        'cap_saturated': raw_norm.gt(TANGENT_NORM_CAP),
    }
    return out, diag


def excess_penalty(h, residual):
    _, tang = tangent_raw(h, residual)
    q = tang.norm(dim=1) / TANGENT_NORM_CAP
    return F.relu(q - 1.0).square().mean()


def base_terms(model, modules, c, user_ids, item_ids, t, alpha_bar, noise_t=None, noise_v=None):
    del model, user_ids
    n = len(item_ids)
    if noise_t is None:
        noise_t = torch.randn((n, LATENT_DIM), device=item_ids.device)
    if noise_v is None:
        noise_v = torch.randn((n, LATENT_DIM), device=item_ids.device)
    item_cf = c['collab_item'][item_ids]
    zero = torch.zeros_like(item_cf)
    out = {}
    losses = []
    for name, den, h, noise in (
        ('text', modules.text, c['text_item'][item_ids], noise_t),
        ('visual', modules.visual, c['image_item'][item_ids], noise_v),
    ):
        x, ht = noisy_state(h, t, alpha_bar, noise)
        p0 = den(ht, t, condition(modules, zero, item_cf))
        loss = F.mse_loss(p0, x.detach())
        out[f'L_base_{name}'] = loss
        losses.append(loss)
    out['L_base'] = 0.5 * (losses[0] + losses[1])
    out['t_min'] = t.min().detach().float()
    out['t_max'] = t.max().detach().float()
    return out['L_base'], out


def preference_terms(model, modules, c, users, pos, neg, t, alpha_bar, bounded: bool,
                     noise_t=None, noise_v=None, return_outputs=False):
    b = len(users)
    ids = torch.cat([pos, neg])
    scorer_users = torch.cat([users, users])
    if t.ndim == 1 and len(t) == b:
        t = torch.cat([t, t])
    if noise_t is None:
        z = torch.randn((b, LATENT_DIM), device=users.device)
        noise_t = torch.cat([z, z])
    if noise_v is None:
        z = torch.randn((b, LATENT_DIM), device=users.device)
        noise_v = torch.cat([z, z])
    ucf = c['collab_user'][scorer_users]
    icf = c['collab_item'][ids]
    wrong_users = shuffled_user_ids(scorer_users, model.n_users)
    wrong_cf = c['collab_user'][wrong_users]
    tx = _predictions(modules.text, modules, c['text_item'][ids], ucf, icf, t, alpha_bar, noise_t, wrong_cf)
    vx = _predictions(modules.visual, modules, c['image_item'][ids], ucf, icf, t, alpha_bar, noise_v, wrong_cf)
    xt, _, pt, p0t, pst = tx
    xv, _, pv, p0v, psv = vx
    rt = pt - p0t.detach()
    rst = pst - p0t.detach()
    rv = pv - p0v.detach()
    rsv = psv - p0v.detach()
    at, dt = tangent_augment(c['text_item'][ids], rt, bounded, return_diag=True)
    ast, dst = tangent_augment(c['text_item'][ids], rst, bounded, return_diag=True)
    av, dv = tangent_augment(c['image_item'][ids], rv, bounded, return_diag=True)
    asv, dsv = tangent_augment(c['image_item'][ids], rsv, bounded, return_diag=True)
    true_items = fuse_item(model, icf, av, at, c['struct_item'][ids])
    shuf_items = fuse_item(model, icf, asv, ast, c['struct_item'][ids])
    u = c['final_user'][users]
    mtrue = (u * true_items[:b]).sum(1) - (u * true_items[b:]).sum(1)
    mshuf = (u * shuf_items[:b]).sum(1) - (u * shuf_items[b:]).sum(1)
    lrec = F.softplus(-mtrue).mean()
    luser = F.softplus(mshuf - mtrue + USER_MARGIN).mean()
    lbound = 0.25 * (
        excess_penalty(c['text_item'][ids], rt) + excess_penalty(c['text_item'][ids], rst)
        + excess_penalty(c['image_item'][ids], rv) + excess_penalty(c['image_item'][ids], rsv)
    ) if bounded else lrec.new_zeros(())
    out = {
        'L_rec': lrec, 'L_user': luser, 'L_bound': lbound,
        'm_true': mtrue.mean(), 'm_shuf': mshuf.mean(),
        'm_true_minus_shuf': (mtrue - mshuf).mean(),
        'fraction_true_gt_shuf': (mtrue > mshuf).float().mean(),
        'raw_residual_text_true': rt.norm(dim=1).mean(),
        'raw_residual_text_shuf': rst.norm(dim=1).mean(),
        'raw_residual_visual_true': rv.norm(dim=1).mean(),
        'raw_residual_visual_shuf': rsv.norm(dim=1).mean(),
        'bounded_tangent_text': 0.5 * (dt['bounded_tangent_norm'].mean() + dst['bounded_tangent_norm'].mean()),
        'bounded_tangent_visual': 0.5 * (dv['bounded_tangent_norm'].mean() + dsv['bounded_tangent_norm'].mean()),
        'angle_text_mean': 0.5 * (dt['angle_deg'].mean() + dst['angle_deg'].mean()),
        'angle_text_p95': torch.quantile(torch.cat([dt['angle_deg'], dst['angle_deg']]), .95),
        'angle_text_max': torch.maximum(dt['angle_deg'].max(), dst['angle_deg'].max()),
        'angle_visual_mean': 0.5 * (dv['angle_deg'].mean() + dsv['angle_deg'].mean()),
        'angle_visual_p95': torch.quantile(torch.cat([dv['angle_deg'], dsv['angle_deg']]), .95),
        'angle_visual_max': torch.maximum(dv['angle_deg'].max(), dsv['angle_deg'].max()),
        'cap_fraction_text': 0.5 * (dt['cap_saturated'].float().mean() + dst['cap_saturated'].float().mean()),
        'cap_fraction_visual': 0.5 * (dv['cap_saturated'].float().mean() + dsv['cap_saturated'].float().mean()),
        't_min': t.min().detach().float(), 't_max': t.max().detach().float(),
    }
    if return_outputs:
        out.update({'mtrue_vec': mtrue, 'mshuf_vec': mshuf, 'rt': rt, 'rst': rst, 'rv': rv, 'rsv': rsv,
                    'pt': pt, 'pst': pst, 'p0t': p0t, 'pv': pv, 'psv': psv, 'p0v': p0v,
                    'xt': xt, 'xv': xv})
    return out


def sample_hard_negatives(users, hard_lookup):
    cols = torch.randint(0, hard_lookup.shape[1], (len(users),), device=users.device)
    neg = hard_lookup[users, cols]
    if (neg < 0).any():
        raise RuntimeError('missing hard candidate for TRAIN user')
    return neg, cols + 6


def training_loss(model, modules, interaction, c, variant, alpha_bar, hard_lookup):
    if variant not in ('A1', 'A2'):
        raise ValueError(variant)
    users, pos, orig_neg = interaction[0], interaction[1], interaction[2]
    b = len(users)
    pref_neg, sampled_rank = sample_hard_negatives(users, hard_lookup)
    ids = torch.cat([pos, orig_neg])
    uids = torch.cat([users, users])
    td = torch.randint(1, T_LATENT + 1, (2 * b,), device=users.device)
    _, base = base_terms(model, modules, c, uids, ids, td, alpha_bar)
    tp = torch.full((b,), T_INFER, device=users.device, dtype=torch.long)
    pref = preference_terms(model, modules, c, users, pos, pref_neg, tp, alpha_bar, bounded=(variant == 'A2'))
    total = (LAMBDA_REC * pref['L_rec'] + LAMBDA_USER * pref['L_user'] + LAMBDA_BASE * base['L_base']
             + (LAMBDA_BOUND * pref['L_bound'] if variant == 'A2' else 0.0))
    out = dict(pref)
    out.update({'L_base': base['L_base'], 'L_base_text': base['L_base_text'], 'L_base_visual': base['L_base_visual'],
                'total_loss': total, 'reconstruction_t_min': base['t_min'], 'reconstruction_t_max': base['t_max'],
                'preference_t_min': pref['t_min'], 'preference_t_max': pref['t_max'],
                'sampled_hard_rank_mean': sampled_rank.float().mean(),
                'hard_rank_6_10_frac': ((sampled_rank >= 6) & (sampled_rank <= 10)).float().mean(),
                'hard_rank_11_15_frac': ((sampled_rank >= 11) & (sampled_rank <= 15)).float().mean(),
                'hard_rank_16_20_frac': ((sampled_rank >= 16) & (sampled_rank <= 20)).float().mean(),
                'hard_rank_21_25_frac': ((sampled_rank >= 21) & (sampled_rank <= 25)).float().mean(),
                'hard_rank_26_30_frac': ((sampled_rank >= 26) & (sampled_rank <= 30)).float().mean()})
    return total, out


def training_loss_a0(model, modules, interaction, c, alpha_bar, hard_lookup):
    users, pos, orig_neg = interaction[0], interaction[1], interaction[2]
    b = len(users)
    pref_neg, sampled_rank = sample_hard_negatives(users, hard_lookup)
    ids = torch.cat([pos, orig_neg]); uids = torch.cat([users, users])
    td = torch.randint(1, T_LATENT + 1, (2 * b,), device=users.device)
    _, anch = r14.anchor_terms(model, modules, c, uids, ids, td, alpha_bar)
    tp = torch.full((b,), T_INFER, device=users.device, dtype=torch.long)
    pref = r14.preference_terms(model, modules, c, users, pos, pref_neg, tp, alpha_bar, return_outputs=True)
    total = pref['L_rec'] + 0.5 * pref['L_user'] + 0.25 * anch['L_anchor_total']
    out = {
        'L_rec': pref['L_rec'], 'L_user': pref['L_user'], 'L_base': anch['L_anchor_total'],
        'L_base_text': 0.5 * (anch['L_anchor_text_true'] + anch['L_anchor_text_zero']),
        'L_base_visual': 0.5 * (anch['L_anchor_visual_true'] + anch['L_anchor_visual_zero']),
        'L_bound': total.new_zeros(()), 'total_loss': total,
        'm_true': pref['m_true'], 'm_shuf': pref['m_shuf'], 'm_true_minus_shuf': pref['m_true_minus_shuf'],
        'fraction_true_gt_shuf': (pref['mtrue_vec'] > pref['mshuf_vec']).float().mean(),
        'raw_residual_text_true': pref['rt'].norm(dim=1).mean(), 'raw_residual_text_shuf': pref['rst'].norm(dim=1).mean(),
        'raw_residual_visual_true': pref['rv'].norm(dim=1).mean(), 'raw_residual_visual_shuf': pref['rsv'].norm(dim=1).mean(),
        'bounded_tangent_text': 0.5 * (tangent_raw(c['text_item'][torch.cat([pos, pref_neg])], pref['rt'])[1].norm(dim=1).mean() + tangent_raw(c['text_item'][torch.cat([pos, pref_neg])], pref['rst'])[1].norm(dim=1).mean()),
        'bounded_tangent_visual': 0.5 * (tangent_raw(c['image_item'][torch.cat([pos, pref_neg])], pref['rv'])[1].norm(dim=1).mean() + tangent_raw(c['image_item'][torch.cat([pos, pref_neg])], pref['rsv'])[1].norm(dim=1).mean()),
        'angle_text_mean': 0.5 * (tangent_augment(c['text_item'][torch.cat([pos, pref_neg])], pref['rt'], False, return_diag=True)[1]['angle_deg'].mean() + tangent_augment(c['text_item'][torch.cat([pos, pref_neg])], pref['rst'], False, return_diag=True)[1]['angle_deg'].mean()),
        'angle_text_p95': torch.quantile(torch.cat([tangent_augment(c['text_item'][torch.cat([pos, pref_neg])], pref['rt'], False, return_diag=True)[1]['angle_deg'], tangent_augment(c['text_item'][torch.cat([pos, pref_neg])], pref['rst'], False, return_diag=True)[1]['angle_deg']]), .95),
        'angle_text_max': torch.maximum(tangent_augment(c['text_item'][torch.cat([pos, pref_neg])], pref['rt'], False, return_diag=True)[1]['angle_deg'].max(), tangent_augment(c['text_item'][torch.cat([pos, pref_neg])], pref['rst'], False, return_diag=True)[1]['angle_deg'].max()),
        'angle_visual_mean': 0.5 * (tangent_augment(c['image_item'][torch.cat([pos, pref_neg])], pref['rv'], False, return_diag=True)[1]['angle_deg'].mean() + tangent_augment(c['image_item'][torch.cat([pos, pref_neg])], pref['rsv'], False, return_diag=True)[1]['angle_deg'].mean()),
        'angle_visual_p95': torch.quantile(torch.cat([tangent_augment(c['image_item'][torch.cat([pos, pref_neg])], pref['rv'], False, return_diag=True)[1]['angle_deg'], tangent_augment(c['image_item'][torch.cat([pos, pref_neg])], pref['rsv'], False, return_diag=True)[1]['angle_deg']]), .95),
        'angle_visual_max': torch.maximum(tangent_augment(c['image_item'][torch.cat([pos, pref_neg])], pref['rv'], False, return_diag=True)[1]['angle_deg'].max(), tangent_augment(c['image_item'][torch.cat([pos, pref_neg])], pref['rsv'], False, return_diag=True)[1]['angle_deg'].max()),
        'cap_fraction_text': total.new_tensor(0.0), 'cap_fraction_visual': total.new_tensor(0.0),
        'reconstruction_t_min': anch['t_min'], 'reconstruction_t_max': anch['t_max'],
        'preference_t_min': pref['t_min'], 'preference_t_max': pref['t_max'],
        'sampled_hard_rank_mean': sampled_rank.float().mean(),
        'hard_rank_6_10_frac': ((sampled_rank >= 6) & (sampled_rank <= 10)).float().mean(),
        'hard_rank_11_15_frac': ((sampled_rank >= 11) & (sampled_rank <= 15)).float().mean(),
        'hard_rank_16_20_frac': ((sampled_rank >= 16) & (sampled_rank <= 20)).float().mean(),
        'hard_rank_21_25_frac': ((sampled_rank >= 21) & (sampled_rank <= 25)).float().mean(),
        'hard_rank_26_30_frac': ((sampled_rank >= 26) & (sampled_rank <= 30)).float().mean()}
    return total, out

@torch.no_grad()
def inference_residuals(model, modules, c, condition_user_ids, item_ids, alpha_bar, noise_tables):
    user_cf = c['collab_user'][condition_user_ids]
    item_cf = c['collab_item'][item_ids]
    zero = torch.zeros_like(user_cf)
    t = torch.full((len(item_ids),), T_INFER, device=item_ids.device, dtype=torch.long)
    res_t, res_v = [], []
    for seed in INFER_SEEDS:
        for modality, den, h, acc in (
            ('text', modules.text, c['text_item'][item_ids], res_t),
            ('visual', modules.visual, c['image_item'][item_ids], res_v),
        ):
            noise = noise_tables[(seed, modality)][item_ids]
            _, ht = noisy_state(h, t, alpha_bar, noise)
            ptrue = den(ht, t, condition(modules, user_cf, item_cf))
            p0 = den(ht, t, condition(modules, zero, item_cf))
            acc.append(ptrue - p0)
    return torch.stack(res_t, 0).mean(0), torch.stack(res_v, 0).mean(0)


@torch.no_grad()
def pair_scores(model, modules, c, scorer_user_ids, item_ids, alpha_bar, noise_tables,
                bounded: bool, condition_mode='true', rho=RHO, return_diag=False):
    base = c['final_item'][item_ids]
    u = c['final_user'][scorer_user_ids]
    base_score = (u * base).sum(1)
    if condition_mode == 'zero' or float(rho) == 0.0:
        z = torch.zeros((len(item_ids), LATENT_DIM), device=item_ids.device)
        if return_diag:
            dz = torch.zeros(len(item_ids), device=item_ids.device)
            diag = {
                'text': {'angle_deg': dz, 'cap_saturated': dz.bool(), 'raw_tangent_norm': dz, 'bounded_tangent_norm': dz},
                'visual': {'angle_deg': dz, 'cap_saturated': dz.bool(), 'raw_tangent_norm': dz, 'bounded_tangent_norm': dz},
            }
            return base_score, base_score, z, z, diag
        return base_score, base_score, z, z
    if condition_mode == 'true':
        cond_users = scorer_user_ids
    elif condition_mode == 'shuffled':
        cond_users = shuffled_user_ids(scorer_user_ids, model.n_users)
    else:
        raise ValueError(condition_mode)
    rt, rv = inference_residuals(model, modules, c, cond_users, item_ids, alpha_bar, noise_tables)
    at, dt = tangent_augment(c['text_item'][item_ids], rt, bounded, rho=rho, return_diag=True)
    av, dv = tangent_augment(c['image_item'][item_ids], rv, bounded, rho=rho, return_diag=True)
    aug = fuse_item(model, c['collab_item'][item_ids], av, at, c['struct_item'][item_ids])
    aug_score = (u * aug).sum(1)
    if return_diag:
        return base_score, aug_score, rt, rv, {'text': dt, 'visual': dv}
    return base_score, aug_score, rt, rv
