from __future__ import annotations
import gc,hashlib,json
from pathlib import Path
import numpy as np
import torch

from diffusion_experiments.round22_corrected_online_gdnsm import round22_core as b22
from diffusion_experiments.round22_corrected_online_gdnsm.round22_diffusion import generate_trajectory,T

ROOT=b22.ROOT
RDIR=ROOT/'diffusion_experiments/round23_hardness_matched_diffusion'
EVID=RDIR/'evidence';LOGS=RDIR/'logs';OUT=RDIR/'outputs';ASSETS=RDIR/'assets'
for p in (EVID,LOGS,OUT,ASSETS): p.mkdir(parents=True,exist_ok=True)
PROTOCOL='ROUND23_HARDNESS_MATCHED_ONLINE_DIFFUSION_V1'
SEEDS=(999,1000)
VARIANTS=('B0','C1FULL','C1DETACH','D1HM')
MODES=('V','T','TV')
BANDS=((60,100),(30,60),(10,30))
ALL=b22.ALL;PRIMARY=b22.PRIMARY
WARMUP_EPOCHS=10;FORMAL_EPOCHS=60;LAMBDA_HN=.20;EPS_NORM=1e-8

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


def json_write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:float(x) if isinstance(x,np.generic) else str(x))+'\n')

def valid_mask(td,n_items,device):
    m=torch.zeros(n_items,device=device,dtype=torch.bool)
    ids=torch.as_tensor(np.asarray(td.all_items,dtype=np.int64),device=device)
    m[ids]=True
    return m

@torch.no_grad()
def mine_real_ids(b,interaction,events,mask,seed,epoch,batch_idx,active=3):
    users=interaction[0];pos=interaction[1];hu=b['fu'][users]
    score=hu.detach()@b['fi'].detach().T;score[:,~mask]=-torch.inf
    for r,(u,p) in enumerate(zip(users.detach().cpu().tolist(),pos.detach().cpu().tolist())):
        seen=events.histories[int(u)]
        if seen: score[r,torch.as_tensor(seen,device=score.device,dtype=torch.long)]=-torch.inf
        score[r,int(p)]=-torch.inf
    vals,top=torch.topk(score,100,dim=1)
    if not torch.isfinite(vals[:,-1]).all(): raise RuntimeError('fewer than 100 legal TRAIN items')
    rng=np.random.default_rng(20271000+int(seed)*100000+int(epoch)*1000+int(batch_idx))
    rows=torch.arange(len(users),device=score.device);ids=[]
    for lo,hi in BANDS[:active]:
        off=torch.as_tensor(rng.integers(lo,hi,size=len(users)),device=score.device,dtype=torch.long)
        ids.append(top[rows,off])
    return ids

def plan_path(seed): return ASSETS/f'ROUND23_REAL_NEG_PLAN_SEED{seed}.pt'

def plan_digest(plan):
    h=hashlib.sha256()
    for ep in sorted(plan):
        for bi,bands in enumerate(plan[ep]):
            h.update(np.asarray([ep,bi],dtype=np.int32).tobytes())
            for x in bands: h.update(np.asarray(x,dtype=np.int32).tobytes())
    return h.hexdigest()

def save_real_plan(seed,plan):
    obj={'protocol':PROTOCOL,'seed':seed,'source':'B0_ONLINE_BOUNDARY_MINER','bands':['Easy61-100','Medium31-60','Hard11-30'],'plan':plan,'digest':plan_digest(plan),'TEST_ACCESSED':False}
    torch.save(obj,plan_path(seed));return obj

def load_real_plan(seed):
    p=plan_path(seed)
    if not p.exists(): raise RuntimeError(f'missing shared real-negative plan seed{seed}; run B0 first')
    return torch.load(p,map_location='cpu',weights_only=False)

def plan_ids(plan,epoch,batch_idx,device,active):
    bands=plan['plan'][int(epoch)][int(batch_idx)]
    return [torch.as_tensor(np.asarray(x,dtype=np.int64),device=device) for x in bands[:active]]

def real_negs(b,ids,detach=False): return [b['fi'][ix].detach() if detach else b['fi'][ix] for ix in ids]

def _chunk(x,mode,B):
    j=MODES.index(mode);return x[j*B:(j+1)*B]

@torch.no_grad()
def hardness_match(model,net,sched,b,interaction,s,target_ids,seed,epoch,batch_idx):
    active=len(target_ids);users=interaction[0];pos=interaction[1];B=len(users)
    hu_raw=b['fu'][users].detach();hp=b['fi'][pos].detach();_,hu,tc,vc=batch_diff_inputs(b,interaction,s)
    g=torch.Generator(device=model.device);g.manual_seed(20272000+int(seed)*100000+int(epoch)*1000+int(batch_idx))
    tr=generate_trajectory(net,sched,hu,tc,vc,g)
    if set(tr)!=set(range(25)) or not all(torch.isfinite(v).all() for v in tr.values()): raise RuntimeError('trajectory numerical failure')
    ps=(hu_raw*hp).sum(1);negs=[];diag=[]
    for j in range(active):
        mode=MODES[j];real=b['fi'][target_ids[j]].detach();tn=real.norm(dim=1);tm=ps-(hu_raw*real).sum(1)
        cand=[];marg=[]
        for t in range(1,25):
            raw=inv_item(_chunk(tr[t],mode,B),s)
            raw=raw*(tn/torch.clamp(raw.norm(dim=1),min=EPS_NORM)).unsqueeze(1)
            cand.append(raw);marg.append(ps-(hu_raw*raw).sum(1))
        cand=torch.stack(cand,0);marg=torch.stack(marg,0);err=(marg-tm.unsqueeze(0)).abs();sel=err.argmin(0);rows=torch.arange(B,device=model.device)
        x=cand[sel,rows].detach();sm=marg[sel,rows];ae=(sm-tm).abs();re=ae/(tm.abs()+1e-6);sn=x.norm(dim=1)
        if float((sn-tn).abs().max())>2e-5: raise RuntimeError('norm match failure')
        negs.append(x);diag.append({'mode':mode,'target_margin':tm.cpu().numpy(),'synthetic_margin':sm.cpu().numpy(),'absolute_error':ae.cpu().numpy(),'relative_error':re.cpu().numpy(),'selected_t':(sel+1).cpu().numpy(),'target_norm':tn.cpu().numpy(),'synthetic_norm':sn.cpu().numpy()})
    return negs,diag

def summarize_match(batch_diags):
    out={}
    for j,mode in enumerate(MODES):
        rows=[d[j] for d in batch_diags if len(d)>j]
        if not rows: continue
        cat=lambda k:np.concatenate([r[k] for r in rows])
        ts=cat('selected_t').astype(np.int64);hist=np.bincount(ts,minlength=25)[1:25];re=cat('relative_error')
        out[mode]={'target_margin':stats(cat('target_margin')),'synthetic_margin':stats(cat('synthetic_margin')),'absolute_error':stats(cat('absolute_error')),'relative_error':stats(re),'p_rel_error_le_025':float((re<=.25).mean()),'p_rel_error_le_050':float((re<=.50).mean()),'target_norm':stats(cat('target_norm')),'synthetic_norm':stats(cat('synthetic_norm')),'norm_max_abs_error':float(np.max(np.abs(cat('target_norm')-cat('synthetic_norm')))),'selected_t':dict(stats(ts),histogram={str(i):int(hist[i-1]) for i in range(1,25)})}
    return out

def protocol_hash():
    h=hashlib.sha256()
    for name in ('round23_core.py','round23_formal.py','round23_formal_train.py','run_round23_formal.py','run_round23_test.py'):
        p=RDIR/name
        if p.exists(): h.update(name.encode());h.update(p.read_bytes())
    h.update(json.dumps({'warmup':10,'epochs':60,'lambda_HN':.20,'T':25,'bands':BANDS,'eps_norm':EPS_NORM},sort_keys=True).encode())
    return h.hexdigest()
