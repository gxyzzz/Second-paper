from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
from diffusion_experiments.round26_preference_directed_diffusion import round26_core as r26

ROOT=r26.ROOT
RDIR=ROOT/'diffusion_experiments/round27_local_preference_field_diffusion'
EVID=RDIR/'evidence'; LOGS=RDIR/'logs'; OUT=RDIR/'outputs'; ASSETS=RDIR/'assets'
for p in (EVID,LOGS,OUT,ASSETS): p.mkdir(parents=True,exist_ok=True)
PROTOCOL='ROUND27_LOCAL_PREFERENCE_FIELD_DIFFUSION_V1'
SEEDS=(999,1000); VARIANTS=('B0','C1FULL','T1','D4')
WARMUP_EPOCHS=10; FORMAL_EPOCHS=60; LAMBDA_HN=.20
THETA_MAX_DEG=5.0; HALF_MARGIN=.5; EPS=1e-8; K_HISTORY=5; MIN_HISTORY=3
LOCAL_TS=(1,2,3,4,5); DIFF_STEPS_WARMUP=1; DIFF_STEPS_ACTIVE=2
ALL=r26.ALL; PRIMARY=r26.PRIMARY; BANDS=r26.BANDS; BAND_NAMES=r26.BAND_NAMES
instantiate_shared=r26.instantiate_shared; TrainEvents=r26.TrainEvents; forward_bundle=r26.forward_bundle
msca_loss_from_bundle=r26.msca_loss_from_bundle; latent_stats=r26.latent_stats; valid_mask=r26.valid_mask
mine_current_ids=r26.mine_current_ids; rank_offsets=r26.rank_offsets; curriculum_count=r26.curriculum_count
aux_bpr=r26.aux_bpr; state_hash=r26.state_hash; stats=r26.stats; delta_pack=r26.delta_pack
CachedEvaluator=r26.CachedEvaluator; sha256_file=r26.sha256_file; make_diff=r26.make_diff; z=r26.z; inv_item=r26.inv_item

def json_write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:float(x) if isinstance(x,np.generic) else str(x))+'\n')

def protocol_hash():
    h=hashlib.sha256()
    for name in ('round27_core.py','round27_lpf.py','round27_audits.py','round27_train.py','run_round27.py'):
        p=RDIR/name
        if p.exists(): h.update(name.encode()); h.update(p.read_bytes())
    h.update(json.dumps({'K_HISTORY':K_HISTORY,'MIN_HISTORY':MIN_HISTORY,'lambda_HN':LAMBDA_HN,'theta':THETA_MAX_DEG,'half_margin':HALF_MARGIN,'local_t':LOCAL_TS,'diff_active':DIFF_STEPS_ACTIVE,'field_loss':['L_rec','L_field'],'epochs':FORMAL_EPOCHS},sort_keys=True).encode())
    return h.hexdigest()

def save_best(model,net,seed,variant,epoch,score):
    p=OUT/f'seed{seed}'/variant/'best.pt'; p.parent.mkdir(parents=True,exist_ok=True)
    import torch
    torch.save({'protocol':PROTOCOL,'seed':seed,'variant':variant,'epoch':int(epoch),'full_colift_R20':float(score),'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'diffusion_state_dict':None if net is None else {k:v.detach().cpu() for k,v in net.state_dict().items()},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False},p)
    return p

def fairness(seed):
    rows={v:json.loads((EVID/f'ROUND27_{v}_SEED{seed}.json').read_text()) for v in VARIANTS}; ref=rows['B0']
    out={'protocol':PROTOCOL,'seed':seed,'initial_state_exact':len({rows[v]['initial_state_hash'] for v in VARIANTS})==1,'normal_plan_mismatch_epochs':[],'aux_pair_mismatch_epochs':[],'rank_offset_mismatch_epochs':[],'lr_trajectory_exact':True,'normal_optimizer_steps':{v:rows[v]['normal_optimizer_steps_total'] for v in VARIANTS},'warmup_epochs':{v:rows[v]['warmup_epochs'] for v in VARIANTS},'lambda_HN':{v:rows[v]['lambda_HN'] for v in VARIANTS},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    for ep in map(str,range(FORMAL_EPOCHS)):
        if len({rows[v]['normal_plan_hashes'][ep] for v in VARIANTS})!=1: out['normal_plan_mismatch_epochs'].append(int(ep))
        if len({rows[v]['aux_pair_hashes'][ep] for v in VARIANTS})!=1: out['aux_pair_mismatch_epochs'].append(int(ep))
        if len({rows[v]['rank_offset_hashes'][ep] for v in VARIANTS})!=1: out['rank_offset_mismatch_epochs'].append(int(ep))
    out['lr_trajectory_exact']=all(rows[v]['lr_trajectory']==ref['lr_trajectory'] for v in VARIANTS)
    out['PASS']=bool(out['initial_state_exact'] and not out['normal_plan_mismatch_epochs'] and not out['aux_pair_mismatch_epochs'] and not out['rank_offset_mismatch_epochs'] and out['lr_trajectory_exact'] and len(set(out['normal_optimizer_steps'].values()))==1)
    json_write(EVID/f'ROUND27_FAIRNESS_SEED{seed}.json',out); return out

def validation_summary():
    rows={str(s):{v:json.loads((EVID/f'ROUND27_{v}_SEED{s}.json').read_text()) for v in VARIANTS} for s in SEEDS}
    out={'protocol':PROTOCOL,'seeds':{},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    for s in SEEDS:
        r=rows[str(s)]; q={'best_epochs':{v:r[v]['best_epoch'] for v in VARIANTS},'best_metrics':{v:r[v]['best_evaluation']['colift'] for v in VARIANTS}}
        q['C1FULL_vs_B0']=delta_pack(q['best_metrics']['C1FULL'],q['best_metrics']['B0'])
        q['T1_vs_C1FULL']=delta_pack(q['best_metrics']['T1'],q['best_metrics']['C1FULL'])
        q['D4_vs_C1FULL']=delta_pack(q['best_metrics']['D4'],q['best_metrics']['C1FULL'])
        q['D4_vs_T1']=delta_pack(q['best_metrics']['D4'],q['best_metrics']['T1'])
        q['D4_vs_B0']=delta_pack(q['best_metrics']['D4'],q['best_metrics']['B0']); out['seeds'][str(s)]=q
    for key in ('C1FULL_vs_B0','T1_vs_C1FULL','D4_vs_C1FULL','D4_vs_T1','D4_vs_B0'):
        out['mean_U_'+key]=float(np.mean([out['seeds'][str(s)][key]['U'] for s in SEEDS]))
    out['minimum_diffusion_success']=bool(all(out['seeds'][str(s)]['D4_vs_C1FULL']['U']>0 and out['seeds'][str(s)]['D4_vs_B0']['U']>0 for s in SEEDS) and out['mean_U_D4_vs_C1FULL']>0)
    out['strong_success']=bool(out['minimum_diffusion_success'] and all(out['seeds'][str(s)]['D4_vs_C1FULL']['primary_positive_count']>=3 for s in SEEDS) and out['mean_U_D4_vs_C1FULL']>=.0025)
    out['ROUND27_BABY_METHOD']='FROZEN' if out['minimum_diffusion_success'] else 'NOT_FROZEN_FOR_CROSS_DOMAIN'
    json_write(EVID/'ROUND27_VALIDATION_SUMMARY.json',out); return out

@__import__('torch').no_grad()
def mine_current_ids_stable(b,interaction,events,mask,seed,epoch,batch_idx,active):
    """Round27 numerical-stable mining: float64 scores + stable item-id tie break."""
    import torch
    if active<=0:return [],[],''
    users=interaction[0]; pos=interaction[1]
    # Mining coordinate only; training/ranking scores remain unchanged.
    score=b['fu'][users].detach().double() @ b['fi'].detach().double().T
    score[:,~mask]=-torch.inf
    for r,(u,p) in enumerate(zip(users.detach().cpu().tolist(),pos.detach().cpu().tolist())):
        seen=events.histories[int(u)]
        if seen: score[r,torch.as_tensor(seen,device=score.device,dtype=torch.long)]=-torch.inf
        score[r,int(p)]=-torch.inf
    # Columns are ascending item IDs. stable=True therefore resolves exact score ties by item ID.
    # The sub-ulp perturbation only breaks numerically indistinguishable near ties and is far below normal score geometry.
    ids=torch.arange(score.shape[1],device=score.device,dtype=torch.float64)
    key=score-ids.unsqueeze(0)*1e-15
    top=torch.argsort(key,dim=1,descending=True,stable=True)[:,:100]
    vals=torch.gather(score,1,top)
    if not torch.isfinite(vals[:,-1]).all():raise RuntimeError('fewer than 100 legal TRAIN items')
    offsets=rank_offsets(seed,epoch,batch_idx,len(users),active)
    rows=torch.arange(len(users),device=score.device)
    chosen=[top[rows,torch.as_tensor(off,device=score.device,dtype=torch.long)] for off in offsets]
    h=hashlib.sha256()
    for x in offsets:h.update(np.asarray(x,dtype=np.int64).tobytes())
    return chosen,offsets,h.hexdigest()

# Round27 baseline reproducibility discipline. This changes no loss/method parameter;
# it removes CUDA execution nondeterminism before current-boundary mining.
_base_instantiate_shared = instantiate_shared
def instantiate_shared(seed):
    import os, torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    return _base_instantiate_shared(seed)
