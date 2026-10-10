from __future__ import annotations
import gc,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from diffusion_experiments.round24_stable_boundary_matched import round24_core as r24
from diffusion_experiments.round22_corrected_online_gdnsm import round22_core as r22

ROOT=r24.ROOT
RDIR=ROOT/'diffusion_experiments/round25_catalog_anchored_diffusion'
EVID=RDIR/'evidence'; LOGS=RDIR/'logs'; OUT=RDIR/'outputs'; ASSETS=RDIR/'assets'
for p in (EVID,LOGS,OUT,ASSETS): p.mkdir(parents=True,exist_ok=True)
PROTOCOL='ROUND25_CATALOG_ANCHORED_DIFFUSION_V1'
SEEDS=(999,1000)
VARIANTS=('B0','C1FULL','D2')
WARMUP_EPOCHS=10; FORMAL_EPOCHS=60; LAMBDA_HN=.20
USER_GUIDANCE=2.0; THETA_MAX_DEG=5.0; EPS=1e-8; LOCAL_TS=(1,2,3,4,5)
DIFF_STEPS_WARMUP=1; DIFF_STEPS_ACTIVE=2
ALL=r24.ALL; PRIMARY=r24.PRIMARY; BANDS=r24.BANDS; BAND_NAMES=r24.BAND_NAMES

instantiate_shared=r24.instantiate_shared
TrainEvents=r24.TrainEvents
forward_bundle=r24.forward_bundle
msca_loss_from_bundle=r24.msca_loss_from_bundle
latent_stats=r24.latent_stats
valid_mask=r24.valid_mask
mine_current_ids=r24.mine_current_ids
rank_offsets=r24.rank_offsets
curriculum_count=r24.curriculum_count
aux_bpr=r24.aux_bpr
state_hash=r24.state_hash
stats=r24.stats
delta_pack=r24.delta_pack
CachedEvaluator=r24.CachedEvaluator
sha256_file=r24.sha256_file
make_diff=r24.make_diff
z=r22.z
inv_item=r24.inv_item


def json_write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:float(x) if isinstance(x,np.generic) else str(x))+'\n')


def protocol_hash():
    h=hashlib.sha256()
    for name in ('round25_core.py','round25_train.py','run_round25.py'):
        p=RDIR/name
        if p.exists(): h.update(name.encode()); h.update(p.read_bytes())
    h.update(json.dumps({'lambda':LAMBDA_HN,'guidance':USER_GUIDANCE,'theta':THETA_MAX_DEG,'local_t':LOCAL_TS,'diff_active':DIFF_STEPS_ACTIVE,'loss':[1,.5,.5],'epochs':FORMAL_EPOCHS},sort_keys=True).encode())
    return h.hexdigest()


def standardized_conditions(b,users,item_ids,s):
    x0=z(b['fi'][item_ids].detach(),s['muI'],s['rI'])
    hu=z(b['fu'][users].detach(),s['muU'],s['rU'])
    tc=z(b['text_item'][item_ids].detach(),s['muT'],s['rT'])
    vc=z(b['image_item'][item_ids].detach(),s['muV'],s['rV'])
    return x0,hu,tc,vc


def local_xt(sched,x0,generator):
    t=torch.randint(1,6,(len(x0),),device=x0.device,generator=generator)
    eps=torch.randn(x0.shape,device=x0.device,generator=generator)
    xt=sched.q_sample(x0,t,eps)
    return xt,t,eps


def branches(net,sched,xt,t,hu,shuf,tc,vc):
    zero=torch.zeros_like(hu); B=len(hu)
    pred=net(torch.cat([xt,xt,xt],0),torch.cat([t,t,t],0),torch.cat([zero,hu,shuf],0),torch.cat([tc,tc,tc],0),torch.cat([vc,vc,vc],0))
    eb,et,es=torch.split(pred,B,0)
    return eb,et,es,sched.x0_from_eps(xt,t,eb),sched.x0_from_eps(xt,t,et),sched.x0_from_eps(xt,t,es)


def project_cap(anchor,residual):
    h=anchor.detach()
    tangent=residual-((residual*h).sum(1,keepdim=True)/((h*h).sum(1,keepdim=True)+EPS))*h
    rn=tangent.norm(dim=1,keepdim=True); hn=h.norm(dim=1,keepdim=True)
    cap=math.tan(math.radians(THETA_MAX_DEG))*hn
    scale=torch.clamp(cap/(rn+EPS),max=1.0)
    delta=tangent*scale; final=h+delta
    angle=torch.rad2deg(torch.acos(F.cosine_similarity(h,final,dim=1).clamp(-1,1)))
    hit=(rn>cap+1e-10).float()
    return delta,angle,hit,tangent


def score(user,item): return (user*item).sum(1)


def base_denoise_step(net,opt,sched,b,interaction,s,seed):
    rng=r22.capture_rng(); g=torch.Generator(device=b['fu'].device); g.manual_seed(int(seed))
    x0,hu,tc,vc=standardized_conditions(b,interaction[0],interaction[1],s)
    xt,t,eps=local_xt(sched,x0,g); zero=torch.zeros_like(hu)
    net.eval(); opt.zero_grad(set_to_none=True); pred=net(xt,t,zero,tc,vc); loss=F.mse_loss(pred,eps)
    if not torch.isfinite(loss): raise RuntimeError('nonfinite base diffusion warmup loss')
    loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); r22.restore_rng(rng)
    return {'total':float(loss.detach()),'rec':float(loss.detach()),'hard':0.0,'user':0.0,'t_min':int(t.min()),'t_max':int(t.max())}


def _concat_active(b,interaction,s,target_ids):
    users=interaction[0]; B=len(users)
    us=[]; ids=[]; sh=[]
    for j in range(len(target_ids)):
        us.append(users); ids.append(target_ids[j]);
    users_all=torch.cat(us,0); ids_all=torch.cat(ids,0)
    x0,hu,tc,vc=standardized_conditions(b,users_all,ids_all,s)
    shuf=torch.cat([hu[j*B:(j+1)*B].roll(1,0) for j in range(len(target_ids))],0)
    hu_raw=b['fu'][users_all].detach(); anchor=b['fi'][ids_all].detach()
    return users_all,ids_all,x0,hu,shuf,tc,vc,hu_raw,anchor,B


def diffusion_inner_step(net,opt,sched,b,interaction,s,target_ids,seed):
    if not target_ids: raise RuntimeError('active target ids required')
    rng=r22.capture_rng(); g=torch.Generator(device=b['fu'].device); g.manual_seed(int(seed))
    users_all,ids_all,x0,hu,shuf,tc,vc,hu_raw,anchor,B=_concat_active(b,interaction,s,target_ids)
    xt,t,eps=local_xt(sched,x0,g)
    net.eval(); opt.zero_grad(set_to_none=True)
    eb,et,es,zb,zt,zs=branches(net,sched,xt,t,hu,shuf,tc,vc)
    hb=inv_item(zb,s); ht=inv_item(zt,s); hs=inv_item(zs,s)
    dt,ang_t,hit_t,tan_t=project_cap(anchor,USER_GUIDANCE*(ht-hb))
    ds,ang_s,hit_s,tan_s=project_cap(anchor,USER_GUIDANCE*(hs-hb))
    sreal=score(hu_raw,anchor); strue=score(hu_raw,anchor+dt); sshuf=score(hu_raw,anchor+ds)
    lrec=F.mse_loss(eb,eps); lhard=F.softplus(sreal-strue).mean(); luser=F.softplus(sshuf-strue).mean(); total=lrec+.5*lhard+.5*luser
    if not torch.isfinite(total): raise RuntimeError('nonfinite catalog-anchored diffusion loss')
    total.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); r22.restore_rng(rng)
    return {'total':float(total.detach()),'rec':float(lrec.detach()),'hard':float(lhard.detach()),'user':float(luser.detach()),'t_min':int(t.min()),'t_max':int(t.max()),'angle_max':float(torch.max(torch.cat([ang_t,ang_s])).detach()),'cap_hit_fraction':float(torch.cat([hit_t,hit_s]).mean().detach())}


@torch.no_grad()
def generate_refinement(net,sched,b,interaction,s,target_ids,seed):
    if not target_ids: return [],[]
    g=torch.Generator(device=b['fu'].device); g.manual_seed(int(seed))
    users_all,ids_all,x0,hu,shuf,tc,vc,hu_raw,anchor,B=_concat_active(b,interaction,s,target_ids)
    xt,t,eps=local_xt(sched,x0,g); net.eval()
    eb,et,es,zb,zt,zs=branches(net,sched,xt,t,hu,shuf,tc,vc)
    hb=inv_item(zb,s); ht=inv_item(zt,s); hs=inv_item(zs,s)
    raw_true=USER_GUIDANCE*(ht-hb); raw_shuf=USER_GUIDANCE*(hs-hb)
    dt,ang_t,hit_t,tan_t=project_cap(anchor,raw_true); ds,ang_s,hit_s,tan_s=project_cap(anchor,raw_shuf)
    sreal=score(hu_raw,anchor); strue=score(hu_raw,anchor+dt); sshuf=score(hu_raw,anchor+ds)
    pos_rep=torch.cat([b['fi'][interaction[1]].detach() for _ in target_ids],0); spos=score(hu_raw,pos_rep)
    gh=strue-sreal; gu=strue-sshuf
    eps_mse=((eb-eps)**2).mean(1)
    cos=F.cosine_similarity(dt,ds,dim=1); l2=(dt-ds).norm(dim=1)
    item_norm=anchor.norm(dim=1); dt_norm=dt.norm(dim=1); ds_norm=ds.norm(dim=1)
    all_diag={'score_real':sreal.cpu().numpy(),'score_true':strue.cpu().numpy(),'score_shuf':sshuf.cpu().numpy(),'G_hard':gh.cpu().numpy(),'G_user':gu.cpu().numpy(),'real_negative_margin':(spos-sreal).cpu().numpy(),'refined_negative_margin':(spos-strue).cpu().numpy(),'true_residual_norm':dt_norm.cpu().numpy(),'shuffled_residual_norm':ds_norm.cpu().numpy(),'true_raw_residual_norm':raw_true.norm(dim=1).cpu().numpy(),'shuffled_raw_residual_norm':raw_shuf.norm(dim=1).cpu().numpy(),'residual_norm_item_ratio':(dt_norm/(item_norm+EPS)).cpu().numpy(),'true_vs_shuffled_residual_cosine':cos.cpu().numpy(),'true_vs_shuffled_residual_l2':l2.cpu().numpy(),'angle_deg':ang_t.cpu().numpy(),'shuffled_angle_deg':ang_s.cpu().numpy(),'cap_hit':hit_t.cpu().numpy(),'shuffled_cap_hit':hit_s.cpu().numpy(),'base_epsilon_mse':eps_mse.cpu().numpy(),'t':t.cpu().numpy()}
    deltas=[dt[j*B:(j+1)*B].detach() for j in range(len(target_ids))]
    diags=[]
    for j in range(len(target_ids)):
        d={k:v[j*B:(j+1)*B] for k,v in all_diag.items()}; d['band']=BAND_NAMES[j]; diags.append(d)
    return deltas,diags


def _prob_stats(x):
    a=np.concatenate(x) if isinstance(x,list) else np.asarray(x)
    o=stats(a); o['p_gt_zero']=float((a>0).mean()); return o


def summarize_mechanism(batch_diags,diff_losses=None):
    rows=[d for batch in batch_diags for d in batch]
    if not rows: return {}
    cat=lambda k:np.concatenate([r[k] for r in rows])
    out={
      'score_real':stats(cat('score_real')),'score_true':stats(cat('score_true')),'score_shuf':stats(cat('score_shuf')),
      'G_hard':_prob_stats(cat('G_hard')),'G_user':_prob_stats(cat('G_user')),
      'real_negative_margin':stats(cat('real_negative_margin')),'refined_negative_margin':stats(cat('refined_negative_margin')),
      'true_residual_norm':stats(cat('true_residual_norm')),'shuffled_residual_norm':stats(cat('shuffled_residual_norm')),
      'true_raw_residual_norm':stats(cat('true_raw_residual_norm')),'shuffled_raw_residual_norm':stats(cat('shuffled_raw_residual_norm')),
      'residual_norm_item_ratio':stats(cat('residual_norm_item_ratio')),
      'true_vs_shuffled_residual_cosine':stats(cat('true_vs_shuffled_residual_cosine')),
      'true_vs_shuffled_residual_l2':stats(cat('true_vs_shuffled_residual_l2')),
      'angle_deg':stats(cat('angle_deg')),'shuffled_angle_deg':stats(cat('shuffled_angle_deg')),
      'cap_hit_fraction':float(cat('cap_hit').mean()),'shuffled_cap_hit_fraction':float(cat('shuffled_cap_hit').mean()),
      'base_epsilon_mse':stats(cat('base_epsilon_mse')),'t':stats(cat('t')),
      'angle_bound_pass':bool(max(float(cat('angle_deg').max()),float(cat('shuffled_angle_deg').max()))<=5.01)
    }
    if diff_losses:
        for k in ('total','rec','hard','user'): out['L_'+k]=stats(np.asarray([x[k] for x in diff_losses],np.float64))
    return out


def save_best(model,net,seed,variant,epoch,score):
    p=OUT/f'seed{seed}'/variant/'best.pt'; p.parent.mkdir(parents=True,exist_ok=True)
    obj={'protocol':PROTOCOL,'seed':seed,'variant':variant,'epoch':int(epoch),'full_colift_R20':float(score),'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'diffusion_state_dict':None if net is None else {k:v.detach().cpu() for k,v in net.state_dict().items()},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    torch.save(obj,p); return p


def fairness(seed):
    rows={v:json.loads((EVID/f'ROUND25_{v}_SEED{seed}.json').read_text()) for v in VARIANTS}
    ref=rows['B0']
    out={'protocol':PROTOCOL,'seed':seed,'initial_state_exact':len({rows[v]['initial_state_hash'] for v in VARIANTS})==1,'normal_plan_mismatch_epochs':[],'aux_pair_mismatch_epochs':[],'rank_offset_mismatch_epochs':[],'lr_trajectory_exact':True,'normal_optimizer_steps':{v:rows[v]['normal_optimizer_steps_total'] for v in VARIANTS},'warmup_epochs':{v:rows[v]['warmup_epochs'] for v in VARIANTS},'lambda_HN':{v:rows[v]['lambda_HN'] for v in VARIANTS},'TEST_ACCESSED':False}
    for ep in map(str,range(FORMAL_EPOCHS)):
        if len({rows[v]['normal_plan_hashes'][ep] for v in VARIANTS})!=1: out['normal_plan_mismatch_epochs'].append(int(ep))
        if len({rows[v]['aux_pair_hashes'][ep] for v in VARIANTS})!=1: out['aux_pair_mismatch_epochs'].append(int(ep))
        if len({rows[v]['rank_offset_hashes'][ep] for v in VARIANTS})!=1: out['rank_offset_mismatch_epochs'].append(int(ep))
    out['lr_trajectory_exact']=all(rows[v]['lr_trajectory']==ref['lr_trajectory'] for v in VARIANTS)
    out['PASS']=bool(out['initial_state_exact'] and not out['normal_plan_mismatch_epochs'] and not out['aux_pair_mismatch_epochs'] and not out['rank_offset_mismatch_epochs'] and out['lr_trajectory_exact'] and len(set(out['normal_optimizer_steps'].values()))==1)
    json_write(EVID/f'ROUND25_FAIRNESS_SEED{seed}.json',out); return out


def validation_summary():
    rows={str(s):{v:json.loads((EVID/f'ROUND25_{v}_SEED{s}.json').read_text()) for v in VARIANTS} for s in SEEDS}
    out={'protocol':PROTOCOL,'seeds':{},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    for s in SEEDS:
        r=rows[str(s)]; z={'best_epochs':{v:r[v]['best_epoch'] for v in VARIANTS},'best_metrics':{v:r[v]['best_evaluation']['colift'] for v in VARIANTS}}
        z['C1FULL_vs_B0']=delta_pack(z['best_metrics']['C1FULL'],z['best_metrics']['B0']); z['D2_vs_B0']=delta_pack(z['best_metrics']['D2'],z['best_metrics']['B0']); z['D2_vs_C1FULL']=delta_pack(z['best_metrics']['D2'],z['best_metrics']['C1FULL']); out['seeds'][str(s)]=z
    for key in ('C1FULL_vs_B0','D2_vs_B0','D2_vs_C1FULL'): out['mean_U_'+key]=float(np.mean([out['seeds'][str(s)][key]['U'] for s in SEEDS]))
    out['minimum_diffusion_success']=bool(all(out['seeds'][str(s)]['D2_vs_C1FULL']['U']>0 for s in SEEDS) and out['mean_U_D2_vs_C1FULL']>0)
    out['strong_success']=bool(all(out['seeds'][str(s)]['D2_vs_C1FULL']['primary_positive_count']>=3 and out['seeds'][str(s)]['D2_vs_B0']['U']>0 for s in SEEDS) and out['mean_U_D2_vs_C1FULL']>=.0025)
    json_write(EVID/'ROUND25_VALIDATION_SUMMARY.json',out); return out
