from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from diffusion_experiments.round29_positive_preference_purification import round29_core as c
from diffusion_experiments.round28_candidate_conditioned_interest_diffusion import round28_diffusion as r28d

base_warmup_step=r28d.base_warmup_step
field_inner_step=r28d.field_inner_step
gradient_audit_terms=r28d.gradient_audit_terms
predict_ensemble_directions=r28d.predict_ensemble_directions

@torch.no_grad()
def purify_positive(user,positive,direction,confidence):
    h=positive.detach(); d=direction.detach(); conf=confidence.detach().clamp(0,1)
    theta=conf*math.radians(c.THETA_MAX_DEG)
    lam=torch.tan(theta)*h.norm(dim=1)
    delta=lam[:,None]*d
    raw=h+delta
    teacher=raw*(h.norm(dim=1,keepdim=True)/(raw.norm(dim=1,keepdim=True)+c.EPS))
    teacher=teacher.detach()
    angle=torch.rad2deg(torch.acos(F.cosine_similarity(h,teacher,dim=1).clamp(-1,1)))
    n0=h.norm(dim=1); n1=teacher.norm(dim=1); score0=(user.detach()*h).sum(1); score1=(user.detach()*teacher).sum(1)
    return teacher, {'confidence':conf.cpu().numpy(),'angle_deg':angle.cpu().numpy(),'original_positive_norm':n0.cpu().numpy(),
      'purified_positive_norm':n1.cpu().numpy(),'norm_relative_error':((n1-n0).abs()/(n0+c.EPS)).cpu().numpy(),
      'u_dot_delta':(user.detach()*(teacher-h)).sum(1).cpu().numpy(),'original_positive_score':score0.cpu().numpy(),
      'purified_positive_score':score1.cpu().numpy(),'score_delta':(score1-score0).cpu().numpy()}

def purification_loss(user,normal_negative,teacher,confidence):
    conf=confidence.detach()
    per=F.softplus(-((user*teacher.detach()).sum(1)-(user*normal_negative).sum(1)))
    den=conf.sum()
    return (conf*per).sum()/(den+c.EPS) if float(den.detach())>0 else per.sum()*0.0

def summarize_field(rows):
    if not rows:return {}
    cat=lambda k:np.concatenate([np.asarray(r[k]).reshape(-1) for r in rows])
    return {'history_size':c.stats(cat('history_size')),'local_support_count':c.stats(cat('local_support_count')),
      'pref_support_count':c.stats(cat('pref_support_count')),'coherent_support_count':c.stats(cat('coherent_support_count')),
      'rho':c.stats(cat('rho')),'confidence':c.stats(cat('confidence')),'invalid_fraction':float(cat('invalid').mean()),
      'weights_sum':c.stats(cat('weights_sum')),'q':c.stats(cat('q')),'locality':c.stats(cat('locality'))}

def summarize(rows_field,rows_dir,rows_geom,diff_losses=None):
    out={'field':summarize_field(rows_field)}
    if rows_dir:
        cat=lambda k:np.concatenate([np.asarray(r[k]).reshape(-1) for r in rows_dir]); da=cat('DeltaA')
        out.update({'A_true':c.stats(cat('A_true')),'A_shuf':c.stats(cat('A_shuf')),
          'DeltaA':dict(c.stats(da),p_gt_zero=float((da>0).mean())),'raw_residual_norm':c.stats(cat('raw_residual_norm')),
          'near_zero_residual_fraction':float(cat('near_zero_residual').mean())})
    if rows_geom:
        catg=lambda k:np.concatenate([np.asarray(r[k]).reshape(-1) for r in rows_geom])
        for k in ('confidence','angle_deg','original_positive_norm','purified_positive_norm','norm_relative_error','u_dot_delta','original_positive_score','purified_positive_score','score_delta','g_field','g_diff'):
            vals=catg(k); out[k]=c.stats(vals)
            if k in ('g_field','g_diff'): out[k]['fraction_gt_zero']=float((vals>0).mean())
        out['angle_bound_pass']=bool(out['angle_deg']['max']<=5.01); out['norm_preservation_pass']=bool(out['norm_relative_error']['max']<=1e-5)
    if diff_losses:
        for k in ('total','base','dir','sep','bound','A_true','A_shuf','true_residual_norm','shuf_residual_norm'):
            out['train_'+k]=c.stats([x[k] for x in diff_losses])
    return out
