from __future__ import annotations
import gc,hashlib,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from diffusion_experiments.round22_corrected_online_gdnsm import round22_core as b22
from diffusion_experiments.round22_corrected_online_gdnsm.round22_diffusion import generate_trajectory,T

ROOT=b22.ROOT
RDIR=ROOT/'diffusion_experiments/round24_stable_boundary_matched'
EVID=RDIR/'evidence';LOGS=RDIR/'logs';OUT=RDIR/'outputs';ASSETS=RDIR/'assets'
for p in (EVID,LOGS,OUT,ASSETS): p.mkdir(parents=True,exist_ok=True)
PROTOCOL='ROUND24_STABLE_BOUNDARY_MATCHED_V1'
SEEDS=(999,1000)
VARIANTS=('B0','C1FULL','C1DETACH','D1BHM')
MODES=('V','T','TV')
BANDS=((60,100),(30,60),(10,30))
BAND_NAMES=('Easy','Medium','Hard')
ALL=b22.ALL;PRIMARY=b22.PRIMARY
WARMUP_EPOCHS=10;FORMAL_EPOCHS=60;LAMBDA_HN=.20;EPS_NORM=1e-8
CONDITIONAL_TS=tuple(range(1,24));PURE_NOISE_T=24

instantiate_shared=b22.instantiate_shared
TrainEvents=b22.TrainEvents
forward_bundle=b22.forward_bundle
msca_loss_from_bundle=b22.msca_loss_from_bundle
latent_stats=b22.latent_stats
batch_diff_inputs=b22.batch_diff_inputs
isolated_diff_step=b22.isolated_diff_step
condition_audit=b22.condition_audit
make_diff=b22.make_diff
aux_bpr=b22.aux_bpr
state_hash=b22.state_hash
stats=b22.stats
utility=b22.utility
delta_pack=b22.delta_pack
curriculum_count=b22.curriculum_count
inv_item=b22.inv_item
CachedEvaluator=b22.CachedEvaluator
sha256_file=b22.sha256_file
load_warmup=b22.load_warmup


def json_write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:float(x) if isinstance(x,np.generic) else str(x))+'\n')

def valid_mask(td,n_items,device):
    m=torch.zeros(n_items,device=device,dtype=torch.bool)
    ids=torch.as_tensor(np.asarray(td.all_items,dtype=np.int64),device=device)
    m[ids]=True
    return m

def rank_offsets(seed,epoch,batch_idx,batch_size,active):
    rng=np.random.default_rng(20281000+int(seed)*100000+int(epoch)*1000+int(batch_idx))
    return [rng.integers(lo,hi,size=int(batch_size),dtype=np.int64) for lo,hi in BANDS[:active]]

def offset_digest(offsets):
    h=hashlib.sha256()
    for x in offsets:h.update(np.asarray(x,dtype=np.int64).tobytes())
    return h.hexdigest()

@torch.no_grad()
def mine_current_ids(b,interaction,events,mask,seed,epoch,batch_idx,active):
    if active<=0:return [],[],''
    users=interaction[0];pos=interaction[1];hu=b['fu'][users]
    score=hu.detach()@b['fi'].detach().T;score[:,~mask]=-torch.inf
    for r,(u,p) in enumerate(zip(users.detach().cpu().tolist(),pos.detach().cpu().tolist())):
        seen=events.histories[int(u)]
        if seen:score[r,torch.as_tensor(seen,device=score.device,dtype=torch.long)]=-torch.inf
        score[r,int(p)]=-torch.inf
    vals,top=torch.topk(score,100,dim=1)
    if not torch.isfinite(vals[:,-1]).all():raise RuntimeError('fewer than 100 legal TRAIN items')
    offsets=rank_offsets(seed,epoch,batch_idx,len(users),active)
    rows=torch.arange(len(users),device=score.device)
    ids=[top[rows,torch.as_tensor(off,device=score.device,dtype=torch.long)] for off in offsets]
    return ids,offsets,offset_digest(offsets)

def real_negs(b,ids,detach=False):return [b['fi'][ix].detach() if detach else b['fi'][ix] for ix in ids]

def _chunk(x,mode,B):
    j=MODES.index(mode);return x[j*B:(j+1)*B]

def gradient_hardness(m):return torch.sigmoid(-m)
def bpr_loss_from_margin(m):return F.softplus(-m)

@torch.no_grad()
def bhm_match(model,net,sched,b,interaction,s,target_ids,seed,epoch,batch_idx):
    active=len(target_ids);users=interaction[0];pos=interaction[1];B=len(users)
    hu_raw=b['fu'][users].detach();hp=b['fi'][pos].detach();_,hu,tc,vc=batch_diff_inputs(b,interaction,s)
    g=torch.Generator(device=model.device);g.manual_seed(20282000+int(seed)*100000+int(epoch)*1000+int(batch_idx))
    tr=generate_trajectory(net,sched,hu,tc,vc,g)
    if set(tr)!=set(range(25)) or not all(torch.isfinite(v).all() for v in tr.values()):raise RuntimeError('trajectory numerical failure')
    ps=(hu_raw*hp).sum(1);negs=[];diag=[]
    for j in range(active):
        mode=MODES[j];real=b['fi'][target_ids[j]].detach();tn=real.norm(dim=1);tm=ps-(hu_raw*real).sum(1);tg=gradient_hardness(tm);tl=bpr_loss_from_margin(tm)
        cand=[];marg=[];gh=[];loss=[]
        for t in CONDITIONAL_TS:
            raw=inv_item(_chunk(tr[t],mode,B),s)
            raw=raw*(tn/torch.clamp(raw.norm(dim=1),min=EPS_NORM)).unsqueeze(1)
            m=ps-(hu_raw*raw).sum(1)
            cand.append(raw);marg.append(m);gh.append(gradient_hardness(m));loss.append(bpr_loss_from_margin(m))
        cand=torch.stack(cand,0);marg=torch.stack(marg,0);gh=torch.stack(gh,0);loss=torch.stack(loss,0)
        herr=(gh-tg.unsqueeze(0)).abs();sel=herr.argmin(0);rows=torch.arange(B,device=model.device)
        x=cand[sel,rows].detach();sm=marg[sel,rows];sg=gh[sel,rows];sl=loss[sel,rows];sn=x.norm(dim=1)
        if float((sn-tn).abs().max())>2e-5:raise RuntimeError('norm match failure')
        # t=24 pure-noise diagnostic only, never eligible for loss.
        raw24=inv_item(_chunk(tr[PURE_NOISE_T],mode,B),s)
        raw24=raw24*(tn/torch.clamp(raw24.norm(dim=1),min=EPS_NORM)).unsqueeze(1)
        m24=ps-(hu_raw*raw24).sum(1);g24=gradient_hardness(m24)
        besterr=(sg-tg).abs();noiseerr=(g24-tg).abs();gmin=gh.min(0).values;gmax=gh.max(0).values
        covered=(tg>=gmin)&(tg<=gmax);too_easy=tg>gmax;too_hard=tg<gmin
        negs.append(x)
        diag.append({'mode':mode,'target_margin':tm.cpu().numpy(),'synthetic_margin':sm.cpu().numpy(),'absolute_margin_error':(sm-tm).abs().cpu().numpy(),'target_gradient_hardness':tg.cpu().numpy(),'synthetic_gradient_hardness':sg.cpu().numpy(),'gradient_hardness_error':(sg-tg).abs().cpu().numpy(),'target_bpr_loss':tl.cpu().numpy(),'synthetic_bpr_loss':sl.cpu().numpy(),'abs_bpr_loss_error':(sl-tl).abs().cpu().numpy(),'hardness_ratio':(sg/torch.clamp(tg,min=1e-8)).cpu().numpy(),'coverage':covered.cpu().numpy(),'too_easy':too_easy.cpu().numpy(),'too_hard':too_hard.cpu().numpy(),'selected_t':(sel+1).cpu().numpy(),'pure_noise_preferred':(noiseerr<besterr).cpu().numpy(),'target_norm':tn.cpu().numpy(),'synthetic_norm':sn.cpu().numpy(),'old_relative_denominator':(tm.abs()+1e-6).cpu().numpy()})
    return negs,diag

def summarize_bhm(batch_diags):
    out={}
    for j,mode in enumerate(MODES):
        rows=[d[j] for d in batch_diags if len(d)>j]
        if not rows:continue
        cat=lambda k:np.concatenate([r[k] for r in rows])
        ts=cat('selected_t').astype(np.int64);hist=np.bincount(ts,minlength=24)[1:24]
        cov=cat('coverage').astype(bool);easy=cat('too_easy').astype(bool);hard=cat('too_hard').astype(bool);ratio=cat('hardness_ratio')
        out[mode]={
            'target_margin':stats(cat('target_margin')),'synthetic_margin':stats(cat('synthetic_margin')),'absolute_margin_error':stats(cat('absolute_margin_error')),
            'target_gradient_hardness':stats(cat('target_gradient_hardness')),'synthetic_gradient_hardness':stats(cat('synthetic_gradient_hardness')),'gradient_hardness_error':stats(cat('gradient_hardness_error')),
            'target_bpr_loss':stats(cat('target_bpr_loss')),'synthetic_bpr_loss':stats(cat('synthetic_bpr_loss')),'abs_bpr_loss_error':stats(cat('abs_bpr_loss_error')),
            'hardness_ratio':stats(ratio),'coverage_fraction':float(cov.mean()),'too_easy_fraction':float(easy.mean()),'too_hard_fraction':float(hard.mean()),
            'selected_t':dict(stats(ts),histogram={str(i):int(hist[i-1]) for i in range(1,24)}),'pure_noise_preferred_fraction':float(cat('pure_noise_preferred').mean()),
            'target_norm':stats(cat('target_norm')),'synthetic_norm':stats(cat('synthetic_norm')),'norm_max_abs_error':float(np.max(np.abs(cat('target_norm')-cat('synthetic_norm')))),
            'old_relative_denominator':stats(cat('old_relative_denominator')),'new_metric_bounded':True
        }
    return out

def summarize_real(margin_lists):
    out={}
    for j,v in enumerate(margin_lists):
        if not v:continue
        m=np.concatenate(v);g=1/(1+np.exp(m));l=np.logaddexp(0,-m)
        out[BAND_NAMES[j]]={'margin':stats(m),'gradient_hardness':stats(g),'bpr_loss':stats(l),'p_negative_ge_positive':float((m<=0).mean())}
    return out

def protocol_hash():
    h=hashlib.sha256()
    for name in ('round24_core.py','round24_formal.py','round24_formal_train.py','run_round24.py','run_round24_formal.py','run_round24_test.py'):
        p=RDIR/name
        if p.exists():h.update(name.encode());h.update(p.read_bytes())
    h.update(json.dumps({'warmup':10,'epochs':60,'lambda_HN':.20,'T':25,'bands':BANDS,'candidate_t':[1,23],'pure_noise_t':24,'eps_norm':EPS_NORM},sort_keys=True).encode())
    return h.hexdigest()
