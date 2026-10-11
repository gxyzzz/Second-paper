from __future__ import annotations
import math, numpy as np, torch
import torch.nn.functional as F
from diffusion_experiments.round30_candidate_side_diffusion_purification import round30_core as c

def local_xt(sched,x0,g):
    t=torch.randint(1,6,(len(x0),),device=x0.device,generator=g); eps=torch.randn(x0.shape,device=x0.device,generator=g); return sched.q_sample(x0,t,eps),t,eps
def tangent(anchor,residual):
    h=anchor.detach(); return residual-((residual*h).sum(1,keepdim=True)/((h*h).sum(1,keepdim=True)+c.EPS))*h
def unit(v):
    n=v.norm(dim=1,keepdim=True); return v/(n+c.EPS),n.squeeze(1)
def _inputs(b,users,items,s):
    x0=c.z(b['fi'][items].detach(),s['muI'],s['rI']); hu=c.z(b['fu'][users].detach(),s['muU'],s['rU']); tc=c.z(b['text_item'][items].detach(),s['muT'],s['rT']); vc=c.z(b['image_item'][items].detach(),s['muV'],s['rV']); return x0,hu,tc,vc,b['fi'][items].detach()
def _branches(net,sched,xt,t,hu,tc,vc):
    shuf=hu.roll(1,0); zero=torch.zeros_like(hu); B=len(hu); pred=net(torch.cat([xt,xt,xt]),torch.cat([t,t,t]),torch.cat([zero,hu,shuf]),torch.cat([tc,tc,tc]),torch.cat([vc,vc,vc])); eb,et,es=torch.split(pred,B,0); return eb,et,es,sched.x0_from_eps(xt,t,eb),sched.x0_from_eps(xt,t,et),sched.x0_from_eps(xt,t,es)
def _bound(anchor,res):
    tr=tangent(anchor,res); cap=math.tan(math.radians(c.THETA_MAX_DEG))*anchor.norm(dim=1); ratio=tr.norm(dim=1)/(cap+c.EPS); return F.relu(ratio-1).square().mean()

def base_step(net,opt,sched,b,users,items,s,seed):
    rng=c.capture_rng(); g=torch.Generator(device=b['fu'].device); g.manual_seed(int(seed)); x0,hu,tc,vc,anchor=_inputs(b,users,items,s); xt,t,eps=local_xt(sched,x0,g); net.train(); opt.zero_grad(set_to_none=True); zero=torch.zeros_like(hu); eb=net(xt,t,zero,tc,vc); loss=(eb-eps).pow(2).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); c.restore_rng(rng); return {'total':float(loss.detach()),'base':float(loss.detach()),'dir':0.,'sep':0.,'proj':0.,'bound':0.}

def field_step(net,opt,sched,b,users,items,s,field,seed):
    rng=c.capture_rng(); g=torch.Generator(device=b['fu'].device); g.manual_seed(int(seed)); x0,hu,tc,vc,anchor=_inputs(b,users,items,s); xt,t,eps=local_xt(sched,x0,g); net.train(); opt.zero_grad(set_to_none=True); eb,et,es,zb,zt,zs=_branches(net,sched,xt,t,hu,tc,vc); hb=c.inv_item(zb,s); ht=c.inv_item(zt,s); hs=c.inv_item(zs,s); rt_raw=ht-hb.detach(); rs_raw=hs-hb.detach(); rt=tangent(anchor,rt_raw); rs=tangent(anchor,rs_raw); dt,nt=unit(rt); ds,ns=unit(rs); td=field['direction'].detach(); cf=field['confidence'].detach(); at=(dt*td).sum(1).clamp(-1,1); ash=(ds*td).sum(1).clamp(-1,1); valid=(cf>0)&(nt>c.EPS); ldir=(cf*(1-at))[valid].mean() if bool(valid.any()) else (cf*(1-at)).mean()*0.; lsep=(cf*F.relu(c.SEP_MARGIN-at+ash))[valid].mean() if bool(valid.any()) else (cf*at).mean()*0.
    ur=F.normalize(b['fu'][users].detach(),dim=1); gf=(ur*td).sum(1); gd=(ur*dt).sum(1); pv=cf*F.smooth_l1_loss(gd,gf,reduction='none'); lproj=pv[valid].mean() if bool(valid.any()) else pv.mean()*0.; lbase=(eb-eps).pow(2).mean(); lbound=.5*(_bound(anchor,rt_raw)+_bound(anchor,rs_raw)); total=lbase+c.L_DIR*ldir+c.L_SEP*lsep+c.L_PROJ*lproj+c.L_BOUND*lbound
    if not torch.isfinite(total): raise RuntimeError('nonfinite Round30 diffusion loss')
    total.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); c.restore_rng(rng)
    return {'total':float(total.detach()),'base':float(lbase.detach()),'dir':float(ldir.detach()),'sep':float(lsep.detach()),'proj':float(lproj.detach()),'bound':float(lbound.detach()),'A_true':float(at[valid].mean().detach()) if bool(valid.any()) else 0.,'A_shuf':float(ash[valid].mean().detach()) if bool(valid.any()) else 0.,'P_DeltaA_gt0':float(((at-ash)[valid]>0).float().mean()) if bool(valid.any()) else 0.,'g_field_mean':float(gf[valid].mean()) if bool(valid.any()) else 0.,'g_diff_mean':float(gd[valid].mean()) if bool(valid.any()) else 0.,'proj_abs_mean':float((gd-gf).abs()[valid].mean()) if bool(valid.any()) else 0.,'sign_agreement':float((torch.sign(gd[valid])==torch.sign(gf[valid])).float().mean()) if bool(valid.any()) else 0.,'valid_fraction':float(valid.float().mean())}

@torch.no_grad()
def predict_ensemble(net,sched,b,users,items,s,field):
    x0,hu,tc,vc,anchor=_inputs(b,users,items,s); true=[]; shuf=[]; raw=[]; rec=[]
    for nsd in c.ENSEMBLE_NOISE_SEEDS:
        g=torch.Generator(device=x0.device); g.manual_seed(int(nsd)); xt,t,eps=local_xt(sched,x0,g); net.eval(); eb,et,es,zb,zt,zs=_branches(net,sched,xt,t,hu,tc,vc); hb=c.inv_item(zb,s); ht=c.inv_item(zt,s); hs=c.inv_item(zs,s); dt,nt=unit(tangent(anchor,ht-hb.detach())); ds,nsh=unit(tangent(anchor,hs-hb.detach())); true.append(dt); shuf.append(ds); raw.append(nt); rec.append((eb-eps).pow(2).mean(1))
    dt,_=unit(torch.stack(true).sum(0)); ds,_=unit(torch.stack(shuf).sum(0)); td=field['direction']; at=(dt*td).sum(1).clamp(-1,1); ash=(ds*td).sum(1).clamp(-1,1); ur=F.normalize(b['fu'][users].detach(),dim=1); gf=(ur*td).sum(1); gd=(ur*dt).sum(1)
    diag={'A_true':at.cpu().numpy(),'A_shuf':ash.cpu().numpy(),'DeltaA':(at-ash).cpu().numpy(),'raw_residual_norm':torch.stack(raw).mean(0).cpu().numpy(),'base_reconstruction_mse':torch.stack(rec).mean(0).cpu().numpy(),'g_field':gf.cpu().numpy(),'g_diff':gd.cpu().numpy(),'projection_abs_error':(gd-gf).abs().cpu().numpy(),'sign_agreement':(torch.sign(gd)==torch.sign(gf)).cpu().numpy()}
    return dt.detach(),diag

@torch.no_grad()
def purify_candidates(user,anchor,direction,confidence):
    h=anchor.detach(); d=direction.detach(); conf=confidence.detach().clamp(0,1); theta=conf*math.radians(c.THETA_MAX_DEG); lam=torch.tan(theta)*h.norm(dim=1); raw=h+lam[:,None]*d; out=raw*(h.norm(dim=1,keepdim=True)/(raw.norm(dim=1,keepdim=True)+c.EPS)); n0=h.norm(dim=1); n1=out.norm(dim=1); angle=torch.rad2deg(torch.acos(F.cosine_similarity(h,out,dim=1).clamp(-1,1))); s0=(user.detach()*h).sum(1); s1=(user.detach()*out).sum(1)
    return out.detach(),{'confidence':conf.cpu().numpy(),'angle_deg':angle.cpu().numpy(),'norm_relative_error':((n1-n0).abs()/(n0+c.EPS)).cpu().numpy(),'score_delta':(s1-s0).cpu().numpy()}

def shortcut_stats(diag):
    gf=np.asarray(diag['g_field']); gd=np.asarray(diag['g_diff']); da=np.asarray(diag['DeltaA']); ae=np.abs(gd-gf); sa=np.asarray(diag['sign_agreement']).astype(bool)
    return {'g_field':c.stats(gf),'g_diff':c.stats(gd),'median_abs_gdiff_minus_gfield':float(np.median(ae)),'sign_agreement':float(sa.mean()),'A_true':c.stats(diag['A_true']),'A_shuf':c.stats(diag['A_shuf']),'DeltaA':dict(c.stats(da),p_gt_zero=float((da>0).mean())),'raw_residual_norm':c.stats(diag['raw_residual_norm'])}
