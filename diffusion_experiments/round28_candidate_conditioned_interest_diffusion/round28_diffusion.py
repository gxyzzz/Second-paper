from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F

from diffusion_experiments.round28_candidate_conditioned_interest_diffusion import round28_core as c
from diffusion_experiments.round28_candidate_conditioned_interest_diffusion.round28_interest_field import summarize_field

def local_xt(sched, x0, generator):
    t = torch.randint(1, 6, (len(x0),), device=x0.device, generator=generator)
    eps = torch.randn(x0.shape, device=x0.device, generator=generator)
    return sched.q_sample(x0, t, eps), t, eps

def tangent(anchor, residual):
    h = anchor.detach()
    return residual - ((residual * h).sum(1, keepdim=True) / ((h * h).sum(1, keepdim=True) + c.EPS)) * h

def unit(v):
    n = v.norm(dim=1, keepdim=True)
    return v / (n + c.EPS), n.squeeze(1)

def _concat_active(b, interaction, s, target_ids):
    users = interaction[0]
    B = len(users)
    active = len(target_ids)
    users_all = torch.cat([users for _ in range(active)], 0)
    ids_all = torch.cat(target_ids, 0)
    x0 = c.z(b['fi'][ids_all].detach(), s['muI'], s['rI'])
    hu = c.z(b['fu'][users_all].detach(), s['muU'], s['rU'])
    tc = c.z(b['text_item'][ids_all].detach(), s['muT'], s['rT'])
    vc = c.z(b['image_item'][ids_all].detach(), s['muV'], s['rV'])
    shuf = torch.cat([hu[j * B:(j + 1) * B].roll(1, 0) for j in range(active)], 0)
    anchor = b['fi'][ids_all].detach()
    return users_all, ids_all, x0, hu, shuf, tc, vc, anchor, B

def _branches(net, sched, xt, t, hu, shuf, tc, vc):
    zero = torch.zeros_like(hu)
    B = len(hu)
    pred = net(
        torch.cat([xt, xt, xt], 0),
        torch.cat([t, t, t], 0),
        torch.cat([zero, hu, shuf], 0),
        torch.cat([tc, tc, tc], 0),
        torch.cat([vc, vc, vc], 0),
    )
    eb, et, es = torch.split(pred, B, 0)
    zb = sched.x0_from_eps(xt, t, eb)
    zt = sched.x0_from_eps(xt, t, et)
    zs = sched.x0_from_eps(xt, t, es)
    return eb, et, es, zb, zt, zs

def _bound_penalty(anchor, residual):
    tr = tangent(anchor, residual)
    cap = math.tan(math.radians(c.THETA_MAX_DEG)) * anchor.detach().norm(dim=1)
    ratio = tr.norm(dim=1) / (cap + c.EPS)
    return F.relu(ratio - 1.0).square().mean()

def base_warmup_step(net, opt, sched, b, interaction, s, seed):
    rng = c.capture_rng()
    g = torch.Generator(device=b['fu'].device)
    g.manual_seed(int(seed))
    pos = interaction[1]
    x0 = c.z(b['fi'][pos].detach(), s['muI'], s['rI'])
    tc = c.z(b['text_item'][pos].detach(), s['muT'], s['rT'])
    vc = c.z(b['image_item'][pos].detach(), s['muV'], s['rV'])
    zero = torch.zeros((len(pos), b['fu'].shape[1]), device=b['fu'].device, dtype=b['fu'].dtype)
    xt, t, eps = local_xt(sched, x0, g)
    net.train()
    opt.zero_grad(set_to_none=True)
    eb = net(xt, t, zero, tc, vc)
    lbase = (eb - eps).pow(2).mean()
    if not torch.isfinite(lbase):
        raise RuntimeError('nonfinite Round28 base warmup loss')
    lbase.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    c.restore_rng(rng)
    return {'total': float(lbase.detach()), 'base': float(lbase.detach()), 'dir': 0.0, 'sep': 0.0, 'bound': 0.0, 't_min': int(t.min()), 't_max': int(t.max())}

def field_inner_step(net, opt, sched, b, interaction, s, target_ids, packs, seed):
    rng = c.capture_rng()
    g = torch.Generator(device=b['fu'].device)
    g.manual_seed(int(seed))
    _, _, x0, hu, shuf, tc, vc, anchor, _ = _concat_active(b, interaction, s, target_ids)
    td = torch.cat([p['direction'] for p in packs], 0).detach()
    cf = torch.cat([p['confidence'] for p in packs], 0).detach()
    xt, t, eps = local_xt(sched, x0, g)
    net.train()
    opt.zero_grad(set_to_none=True)
    eb, et, es, zb, zt, zs = _branches(net, sched, xt, t, hu, shuf, tc, vc)
    hb = c.inv_item(zb, s)
    ht = c.inv_item(zt, s)
    hs = c.inv_item(zs, s)
    rt_raw = ht - hb.detach()
    rs_raw = hs - hb.detach()
    rt = tangent(anchor, rt_raw)
    rs = tangent(anchor, rs_raw)
    dt, nt = unit(rt)
    ds, ns = unit(rs)
    at = (dt * td).sum(1).clamp(-1, 1)
    ash = (ds * td).sum(1).clamp(-1, 1)
    valid = (cf > 0) & (nt > c.EPS)
    dir_vec = cf * (1.0 - at)
    sep_vec = cf * F.relu(c.SEP_MARGIN - at + ash)
    ldir = dir_vec[valid].mean() if bool(valid.any()) else dir_vec.mean() * 0.0
    lsep = sep_vec[valid].mean() if bool(valid.any()) else sep_vec.mean() * 0.0
    lbase = (eb - eps).pow(2).mean()
    lbound = 0.5 * (_bound_penalty(anchor, rt_raw) + _bound_penalty(anchor, rs_raw))
    total = lbase + c.L_DIR * ldir + c.L_SEP * lsep + c.L_BOUND * lbound
    if not torch.isfinite(total):
        raise RuntimeError('nonfinite Round28 diffusion loss')
    total.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    c.restore_rng(rng)
    return {
        'total': float(total.detach()), 'base': float(lbase.detach()), 'dir': float(ldir.detach()), 'sep': float(lsep.detach()), 'bound': float(lbound.detach()),
        'A_true': float(at[valid].mean().detach()) if bool(valid.any()) else 0.0,
        'A_shuf': float(ash[valid].mean().detach()) if bool(valid.any()) else 0.0,
        'valid_field_fraction': float(valid.float().mean().detach()),
        'true_residual_norm': float(nt.mean().detach()), 'shuf_residual_norm': float(ns.mean().detach()),
        't_min': int(t.min()), 't_max': int(t.max()),
    }

def gradient_audit_terms(net, sched, b, interaction, s, target_ids, packs, seed=20261011):
    g = torch.Generator(device=b['fu'].device)
    g.manual_seed(int(seed))
    _, _, x0, hu, shuf, tc, vc, anchor, _ = _concat_active(b, interaction, s, target_ids)
    td = torch.cat([p['direction'] for p in packs], 0).detach()
    cf = torch.cat([p['confidence'] for p in packs], 0).detach()
    xt, t, eps = local_xt(sched, x0, g)
    net.train()
    eb, et, es, zb, zt, zs = _branches(net, sched, xt, t, hu, shuf, tc, vc)
    hb = c.inv_item(zb, s); ht = c.inv_item(zt, s); hs = c.inv_item(zs, s)
    rt_raw = ht - hb.detach(); rs_raw = hs - hb.detach()
    rt = tangent(anchor, rt_raw); rs = tangent(anchor, rs_raw)
    dt, nt = unit(rt); ds, ns = unit(rs)
    at = (dt * td).sum(1).clamp(-1, 1); ash = (ds * td).sum(1).clamp(-1, 1)
    valid = (cf > 0) & (nt > c.EPS)
    dir_vec = cf * (1.0 - at); sep_vec = cf * F.relu(c.SEP_MARGIN - at + ash)
    ldir = dir_vec[valid].mean() if bool(valid.any()) else dir_vec.mean() * 0.0
    lsep = sep_vec[valid].mean() if bool(valid.any()) else sep_vec.mean() * 0.0
    lbase = (eb - eps).pow(2).mean()
    lbound = 0.5 * (_bound_penalty(anchor, rt_raw) + _bound_penalty(anchor, rs_raw))
    def asum(gx): return 0.0 if gx is None else float(gx.detach().abs().sum())
    gb = torch.autograd.grad(lbase, eb, retain_graph=True, allow_unused=True)[0]
    gd_true = torch.autograd.grad(ldir, et, retain_graph=True, allow_unused=True)[0]
    gs_true = torch.autograd.grad(lsep, et, retain_graph=True, allow_unused=True)[0]
    gs_shuf = torch.autograd.grad(lsep, es, retain_graph=True, allow_unused=True)[0]
    gu_base = torch.autograd.grad(ldir + lsep, eb, retain_graph=True, allow_unused=True)[0]
    params = tuple(net.parameters())
    gp_base = torch.autograd.grad(lbase, params, retain_graph=True, allow_unused=True)
    gp_user = torch.autograd.grad(c.L_DIR * ldir + c.L_SEP * lsep + c.L_BOUND * lbound, params, retain_graph=False, allow_unused=True)
    return {
        'L_base': float(lbase.detach()), 'L_dir': float(ldir.detach()), 'L_sep': float(lsep.detach()), 'L_bound': float(lbound.detach()),
        'base_output_grad_abs_sum_from_L_base': asum(gb), 'true_output_grad_abs_sum_from_L_dir': asum(gd_true),
        'true_output_grad_abs_sum_from_L_sep': asum(gs_true), 'shuffled_output_grad_abs_sum_from_L_sep': asum(gs_shuf),
        'base_output_grad_abs_sum_from_user_losses': asum(gu_base),
        'base_parameter_grad_abs_sum': float(sum(asum(x) for x in gp_base)),
        'user_parameter_grad_abs_sum': float(sum(asum(x) for x in gp_user)),
        'valid_field_fraction': float(valid.float().mean().detach()),
    }

@torch.no_grad()
def predict_ensemble_directions(net, sched, b, interaction, s, target_ids, packs):
    if not target_ids:
        return [], []
    _, _, x0, hu, shuf, tc, vc, anchor, B = _concat_active(b, interaction, s, target_ids)
    td = torch.cat([p['direction'] for p in packs], 0).detach()
    true_dirs, shuf_dirs, recs, raw_true, raw_shuf = [], [], [], [], []
    for noise_seed in c.ENSEMBLE_NOISE_SEEDS:
        g = torch.Generator(device=x0.device); g.manual_seed(int(noise_seed))
        xt, t, eps = local_xt(sched, x0, g)
        net.eval()
        eb, et, es, zb, zt, zs = _branches(net, sched, xt, t, hu, shuf, tc, vc)
        hb = c.inv_item(zb, s); ht = c.inv_item(zt, s); hs = c.inv_item(zs, s)
        rt = tangent(anchor, ht - hb.detach()); rs = tangent(anchor, hs - hb.detach())
        dt, nt = unit(rt); ds, ns = unit(rs)
        true_dirs.append(dt); shuf_dirs.append(ds); raw_true.append(nt); raw_shuf.append(ns); recs.append((eb - eps).pow(2).mean(1))
    dt, _ = unit(torch.stack(true_dirs, 0).sum(0)); ds, _ = unit(torch.stack(shuf_dirs, 0).sum(0))
    at = (dt * td).sum(1).clamp(-1, 1); ash = (ds * td).sum(1).clamp(-1, 1); da = at - ash
    nt = torch.stack(raw_true, 0).mean(0); ns = torch.stack(raw_shuf, 0).mean(0); rec = torch.stack(recs, 0).mean(0)
    dirs = [dt[j * B:(j + 1) * B].detach() for j in range(len(target_ids))]
    diags = []
    for j in range(len(target_ids)):
        sl = slice(j * B, (j + 1) * B)
        diags.append({'band': j, 'A_true': at[sl].cpu().numpy(), 'A_shuf': ash[sl].cpu().numpy(), 'DeltaA': da[sl].cpu().numpy(),
                      'raw_residual_norm': nt[sl].cpu().numpy(), 'shuffled_raw_residual_norm': ns[sl].cpu().numpy(),
                      'near_zero_residual': (nt[sl] < c.EPS).cpu().numpy(), 'base_reconstruction_mse': rec[sl].cpu().numpy(),
                      'ensemble_noise_seeds': list(c.ENSEMBLE_NOISE_SEEDS)})
    return dirs, diags

@torch.no_grad()
def calibrate(user, pos, anchor, direction, confidence):
    d = direction.detach(); h = anchor.detach(); u = user.detach(); p = pos.detach(); conf = confidence.detach().clamp(0, 1)
    sp = (u * p).sum(1); sn = (u * h).sum(1); margin = sp - sn
    shift = conf * c.HALF_MARGIN * torch.clamp(margin, min=0)
    align = (u * d).sum(1)
    eligible = (conf > 0) & (margin > 0) & (align > 0)
    req = torch.zeros_like(shift); req[eligible] = shift[eligible] / (align[eligible] + c.EPS)
    lmax = math.tan(math.radians(c.THETA_MAX_DEG)) * h.norm(dim=1)
    lam = torch.where(eligible, torch.minimum(req, lmax), torch.zeros_like(req))
    delta = (lam[:, None] * d).detach(); refined = h + delta
    sr = (u * refined).sum(1); actual = sr - sn; target = sn + shift
    angle = torch.rad2deg(torch.acos(F.cosine_similarity(h, refined, dim=1).clamp(-1, 1)))
    cap = eligible & (req > lmax + 1e-10); reach = eligible & (req <= lmax + 1e-10)
    return delta, {
        'positive_margin': (margin > 0).cpu().numpy(), 'confidence': conf.cpu().numpy(), 'direction_positive': (align > 0).cpu().numpy(),
        'direction_miss': ((conf > 0) & (margin > 0) & (align <= 0)).cpu().numpy(), 'applied_refinement': (lam > 0).cpu().numpy(),
        'requested_score_shift': shift.cpu().numpy(), 'actual_score_shift': actual.cpu().numpy(), 'target_hit_abs_error': (sr - target).abs().cpu().numpy(),
        'target_reachable': reach.cpu().numpy(), 'cap_hit': cap.cpu().numpy(), 'angle_deg': angle.cpu().numpy(),
        'real_negative_margin': margin.cpu().numpy(), 'refined_negative_margin': (sp - sr).cpu().numpy(),
        'target_never_exceeds_positive': ((margin <= 0) | (target <= sp + 1e-7)).cpu().numpy(),
    }

def summarize_mechanism(packs_batches, direction_batches, calibration_batches, diff_losses=None):
    out = {'field': summarize_field(packs_batches)}
    dr = [r for batch in direction_batches for r in batch]; cr = [r for batch in calibration_batches for r in batch]
    if dr:
        catd = lambda k: np.concatenate([np.asarray(r[k]) for r in dr]); da = catd('DeltaA')
        out.update({'A_true': c.stats(catd('A_true')), 'A_shuf': c.stats(catd('A_shuf')), 'DeltaA': dict(c.stats(da), p_gt_zero=float((da > 0).mean())),
                    'raw_residual_norm': c.stats(catd('raw_residual_norm')), 'shuffled_raw_residual_norm': c.stats(catd('shuffled_raw_residual_norm')),
                    'near_zero_residual_fraction': float(catd('near_zero_residual').mean()), 'base_reconstruction_mse': c.stats(catd('base_reconstruction_mse'))})
    if cr:
        catc = lambda k: np.concatenate([np.asarray(r[k]) for r in cr])
        out.update({'applied_refinement_fraction': float(catc('applied_refinement').mean()), 'requested_score_shift': c.stats(catc('requested_score_shift')),
                    'actual_score_shift': c.stats(catc('actual_score_shift')), 'target_hit_abs_error': c.stats(catc('target_hit_abs_error')),
                    'target_reachable_fraction': float(catc('target_reachable').mean()), 'cap_hit_fraction': float(catc('cap_hit').mean()),
                    'angle_deg': c.stats(catc('angle_deg')), 'real_negative_margin': c.stats(catc('real_negative_margin')),
                    'refined_negative_margin': c.stats(catc('refined_negative_margin')), 'target_never_exceeds_positive': bool(catc('target_never_exceeds_positive').all())})
        out['angle_bound_pass'] = bool(out['angle_deg']['max'] <= 5.01)
    if diff_losses:
        for key in ('total', 'base', 'dir', 'sep', 'bound', 'A_true', 'A_shuf', 'true_residual_norm', 'shuf_residual_norm'):
            out['L_' + key if key in ('total', 'base', 'dir', 'sep', 'bound') else key + '_train'] = c.stats([x[key] for x in diff_losses])
    return out
