from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F

from diffusion_experiments.round26_preference_directed_diffusion import round26_core as c
from diffusion_experiments.round22_corrected_online_gdnsm import round22_core as r22


def standardized_conditions(b,users,item_ids,s):
    x0=c.z(b['fi'][item_ids].detach(),s['muI'],s['rI'])
    hu=c.z(b['fu'][users].detach(),s['muU'],s['rU'])
    tc=c.z(b['text_item'][item_ids].detach(),s['muT'],s['rT'])
    vc=c.z(b['image_item'][item_ids].detach(),s['muV'],s['rV'])
    return x0,hu,tc,vc


def local_xt(sched,x0,generator):
    t=torch.randint(1,6,(len(x0),),device=x0.device,generator=generator)
    eps=torch.randn(x0.shape,device=x0.device,generator=generator)
    return sched.q_sample(x0,t,eps),t,eps


def branches(net,sched,xt,t,hu,shuf,tc,vc):
    zero=torch.zeros_like(hu); B=len(hu)
    pred=net(torch.cat([xt,xt,xt],0),torch.cat([t,t,t],0),torch.cat([zero,hu,shuf],0),torch.cat([tc,tc,tc],0),torch.cat([vc,vc,vc],0))
    eb,et,es=torch.split(pred,B,0)
    zb=sched.x0_from_eps(xt,t,eb); zt=sched.x0_from_eps(xt,t,et); zs=sched.x0_from_eps(xt,t,es)
    return eb,et,es,zb,zt,zs


def tangent(anchor,v):
    h=anchor.detach()
    return v-((v*h).sum(1,keepdim=True)/((h*h).sum(1,keepdim=True)+c.EPS))*h


def unit(v):
    n=v.norm(dim=1,keepdim=True)
    return v/(n+c.EPS),n.squeeze(1)


def preference_direction(pos,anchor):
    q=tangent(anchor.detach(),pos.detach()-anchor.detach())
    d,n=unit(q)
    return d.detach(),n.detach()


def _concat_active(b,interaction,s,target_ids):
    users=interaction[0]; B=len(users); active=len(target_ids)
    users_all=torch.cat([users for _ in range(active)],0); ids_all=torch.cat(target_ids,0)
    x0,hu,tc,vc=standardized_conditions(b,users_all,ids_all,s)
    shuf=torch.cat([hu[j*B:(j+1)*B].roll(1,0) for j in range(active)],0)
    anchor=b['fi'][ids_all].detach(); pos=torch.cat([b['fi'][interaction[1]].detach() for _ in range(active)],0)
    hu_raw=b['fu'][users_all].detach()
    return users_all,ids_all,x0,hu,shuf,tc,vc,hu_raw,anchor,pos,B


def warmup_step(net,opt,sched,b,interaction,s,seed):
    rng=r22.capture_rng(); g=torch.Generator(device=b['fu'].device); g.manual_seed(int(seed))
    x0,hu,tc,vc=standardized_conditions(b,interaction[0],interaction[1],s); shuf=hu.roll(1,0)
    xt,t,eps=local_xt(sched,x0,g); net.train(); opt.zero_grad(set_to_none=True)
    eb,et,es,*_=branches(net,sched,xt,t,hu,shuf,tc,vc)
    lrec=((eb-eps).pow(2).mean()+(et-eps).pow(2).mean()+(es-eps).pow(2).mean())/3.
    if not torch.isfinite(lrec): raise RuntimeError('nonfinite Round26 warmup reconstruction')
    lrec.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); r22.restore_rng(rng)
    return {'total':float(lrec.detach()),'rec':float(lrec.detach()),'dir':0.0,'t_min':int(t.min()),'t_max':int(t.max())}


def direction_inner_step(net,opt,sched,b,interaction,s,target_ids,seed):
    if not target_ids: raise RuntimeError('active ids required')
    rng=r22.capture_rng(); g=torch.Generator(device=b['fu'].device); g.manual_seed(int(seed))
    _,_,x0,hu,shuf,tc,vc,_,anchor,pos,_=_concat_active(b,interaction,s,target_ids)
    xt,t,eps=local_xt(sched,x0,g); dpref,pref_norm=preference_direction(pos,anchor)
    net.train(); opt.zero_grad(set_to_none=True)
    eb,et,es,zb,zt,zs=branches(net,sched,xt,t,hu,shuf,tc,vc)
    hb=c.inv_item(zb,s); ht=c.inv_item(zt,s)
    r=tangent(anchor,ht-hb); d,n=unit(r)
    valid=(pref_norm>c.EPS)&(n>c.EPS)
    cos=(d*dpref).sum(1).clamp(-1,1); ldir_vec=1-cos
    ldir=ldir_vec[valid].mean() if bool(valid.any()) else ldir_vec.mean()*0
    lrec=((eb-eps).pow(2).mean()+(et-eps).pow(2).mean()+(es-eps).pow(2).mean())/3.
    total=lrec+ldir
    if not torch.isfinite(total): raise RuntimeError('nonfinite Round26 direction loss')
    total.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); r22.restore_rng(rng)
    return {'total':float(total.detach()),'rec':float(lrec.detach()),'dir':float(ldir.detach()),'t_min':int(t.min()),'t_max':int(t.max()),'valid_direction_fraction':float(valid.float().mean().detach())}


@torch.no_grad()
def predict_directions(net,sched,b,interaction,s,target_ids,seed):
    if not target_ids: return [],[],[]
    g=torch.Generator(device=b['fu'].device); g.manual_seed(int(seed))
    _,_,x0,hu,shuf,tc,vc,_,anchor,pos,B=_concat_active(b,interaction,s,target_ids)
    xt,t,eps=local_xt(sched,x0,g); net.eval()
    eb,et,es,zb,zt,zs=branches(net,sched,xt,t,hu,shuf,tc,vc)
    hb=c.inv_item(zb,s); ht=c.inv_item(zt,s); hs=c.inv_item(zs,s)
    rt=tangent(anchor,ht-hb); rs=tangent(anchor,hs-hb); dt,nt=unit(rt); ds,ns=unit(rs); dp,npref=preference_direction(pos,anchor)
    at=(dt*dp).sum(1).clamp(-1,1); ash=(ds*dp).sum(1).clamp(-1,1); da=at-ash
    rec=((eb-eps).pow(2).mean(1)+(et-eps).pow(2).mean(1)+(es-eps).pow(2).mean(1))/3.
    dirs=[dt[j*B:(j+1)*B].detach() for j in range(len(target_ids))]
    diags=[]
    for j in range(len(target_ids)):
        sl=slice(j*B,(j+1)*B)
        diags.append({'band':c.BAND_NAMES[j],'A_true':at[sl].cpu().numpy(),'A_shuf':ash[sl].cpu().numpy(),'DeltaA':da[sl].cpu().numpy(),'raw_residual_norm':nt[sl].cpu().numpy(),'shuffled_raw_residual_norm':ns[sl].cpu().numpy(),'preference_tangent_norm':npref[sl].cpu().numpy(),'near_zero_residual':(nt[sl]<c.EPS).cpu().numpy(),'base_true_shuf_reconstruction_mse':rec[sl].cpu().numpy(),'t':t[sl].cpu().numpy()})
    return dirs,diags,[dp[j*B:(j+1)*B].detach() for j in range(len(target_ids))]


@torch.no_grad()
def calibrate(user,pos,anchor,direction):
    d=direction.detach(); h=anchor.detach(); u=user.detach(); p=pos.detach()
    sp=(u*p).sum(1); sn=(u*h).sum(1); margin=sp-sn
    shift=c.HALF_MARGIN*torch.clamp(margin,min=0); a=(u*d).sum(1)
    eligible=(margin>0)&(a>0)
    req=torch.zeros_like(shift); req[eligible]=shift[eligible]/(a[eligible]+c.EPS)
    lmax=math.tan(math.radians(c.THETA_MAX_DEG))*h.norm(dim=1)
    lam=torch.where(eligible,torch.minimum(req,lmax),torch.zeros_like(req))
    delta=(lam.unsqueeze(1)*d).detach(); refined=h+delta
    sr=(u*refined).sum(1); target=sn+shift; actual=sr-sn
    cos=F.cosine_similarity(h,refined,dim=1).clamp(-1,1); angle=torch.rad2deg(torch.acos(cos))
    caphit=eligible&(req>lmax+1e-10); reachable=eligible&(req<=lmax+1e-10)
    diag={'positive_margin':(margin>0).cpu().numpy(),'already_hard_no_refine':(margin<=0).cpu().numpy(),'direction_positive':(a>0).cpu().numpy(),'direction_miss':((margin>0)&(a<=0)).cpu().numpy(),'applied_refinement':(lam>0).cpu().numpy(),'requested_score_shift':shift.cpu().numpy(),'actual_score_shift':actual.cpu().numpy(),'target_score':target.cpu().numpy(),'refined_score':sr.cpu().numpy(),'positive_score':sp.cpu().numpy(),'real_negative_score':sn.cpu().numpy(),'target_hit_abs_error':(sr-target).abs().cpu().numpy(),'target_reachable':reachable.cpu().numpy(),'cap_hit':caphit.cpu().numpy(),'angle_deg':angle.cpu().numpy(),'lambda_req':req.cpu().numpy(),'lambda_applied':lam.cpu().numpy(),'lambda_max':lmax.cpu().numpy(),'real_negative_margin':margin.cpu().numpy(),'refined_negative_margin':(sp-sr).cpu().numpy(),'G_hard':actual.cpu().numpy(),'target_never_exceeds_positive':((margin<=0)|(target<=sp+1e-7)).cpu().numpy()}
    return delta,diag


def _prob(a):
    x=np.asarray(a); o=c.stats(x.astype(np.float64)); o['fraction_true']=float(x.mean()); return o

def _gtzero(a):
    x=np.asarray(a,np.float64); o=c.stats(x); o['p_gt_zero']=float((x>0).mean()); return o


def summarize(direction_batches,calibration_batches,diff_losses=None):
    dr=[d for batch in direction_batches for d in batch]; cr=[d for batch in calibration_batches for d in batch]
    if not cr: return {}
    catd=lambda k:np.concatenate([r[k] for r in dr]); catc=lambda k:np.concatenate([r[k] for r in cr])
    out={'A_true':c.stats(catd('A_true')),'A_shuf':c.stats(catd('A_shuf')),'DeltaA':_gtzero(catd('DeltaA')),'raw_residual_norm':c.stats(catd('raw_residual_norm')),'shuffled_raw_residual_norm':c.stats(catd('shuffled_raw_residual_norm')),'near_zero_residual_fraction':float(catd('near_zero_residual').mean()),'base_true_shuf_reconstruction_mse':c.stats(catd('base_true_shuf_reconstruction_mse')),'t':c.stats(catd('t')),
         'positive_margin_fraction':float(catc('positive_margin').mean()),'already_hard_no_refine_fraction':float(catc('already_hard_no_refine').mean()),'direction_positive_fraction':float(catc('direction_positive').mean()),'direction_miss_fraction':float(catc('direction_miss').mean()),'applied_refinement_fraction':float(catc('applied_refinement').mean()),'requested_score_shift':c.stats(catc('requested_score_shift')),'actual_score_shift':c.stats(catc('actual_score_shift')),'target_score':c.stats(catc('target_score')),'target_hit_abs_error':c.stats(catc('target_hit_abs_error')),'target_reachable_fraction':float(catc('target_reachable').mean()),'cap_hit_fraction':float(catc('cap_hit').mean()),'angle_deg':c.stats(catc('angle_deg')),'lambda_req':c.stats(catc('lambda_req')),'lambda_applied':c.stats(catc('lambda_applied')),'real_negative_margin':c.stats(catc('real_negative_margin')),'refined_negative_margin':c.stats(catc('refined_negative_margin')),'G_hard':_gtzero(catc('G_hard')),'target_never_exceeds_positive':bool(catc('target_never_exceeds_positive').all())}
    out['angle_bound_pass']=bool(out['angle_deg']['max']<=5.01)
    if diff_losses:
        out['L_total']=c.stats([x['total'] for x in diff_losses]); out['L_rec']=c.stats([x['rec'] for x in diff_losses]); out['L_dir']=c.stats([x['dir'] for x in diff_losses])
    return out
