from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch

from diffusion_experiments.round25_catalog_anchored_diffusion import round25_core as r25
from diffusion_experiments.round24_stable_boundary_matched import round24_core as r24
from diffusion_experiments.round22_corrected_online_gdnsm import round22_core as r22

ROOT=r25.ROOT
RDIR=ROOT/'diffusion_experiments/round26_preference_directed_diffusion'
EVID=RDIR/'evidence'; LOGS=RDIR/'logs'; OUT=RDIR/'outputs'; ASSETS=RDIR/'assets'
for p in (EVID,LOGS,OUT,ASSETS): p.mkdir(parents=True,exist_ok=True)
PROTOCOL='ROUND26_PREFERENCE_DIRECTED_DIFFUSION_V1'
SEEDS=(999,1000)
VARIANTS=('B0','C1FULL','O1','D3')
WARMUP_EPOCHS=10; FORMAL_EPOCHS=60; LAMBDA_HN=.20
THETA_MAX_DEG=5.0; HALF_MARGIN=.5; EPS=1e-8; LOCAL_TS=(1,2,3,4,5)
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
    for name in ('round26_core.py','round26_diffusion.py','round26_train.py','round26_audits.py','run_round26.py'):
        p=RDIR/name
        if p.exists(): h.update(name.encode()); h.update(p.read_bytes())
    h.update(json.dumps({'lambda':LAMBDA_HN,'theta':THETA_MAX_DEG,'half_margin':HALF_MARGIN,'local_t':LOCAL_TS,'diff_active':DIFF_STEPS_ACTIVE,'diff_loss':['L_rec','L_dir'],'epochs':FORMAL_EPOCHS},sort_keys=True).encode())
    return h.hexdigest()


def save_best(model,net,seed,variant,epoch,score):
    p=OUT/f'seed{seed}'/variant/'best.pt'; p.parent.mkdir(parents=True,exist_ok=True)
    obj={'protocol':PROTOCOL,'seed':seed,'variant':variant,'epoch':int(epoch),'full_colift_R20':float(score),'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'diffusion_state_dict':None if net is None else {k:v.detach().cpu() for k,v in net.state_dict().items()},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    torch.save(obj,p); return p


def fairness(seed):
    rows={v:json.loads((EVID/f'ROUND26_{v}_SEED{seed}.json').read_text()) for v in VARIANTS}
    ref=rows['B0']
    out={'protocol':PROTOCOL,'seed':seed,'initial_state_exact':len({rows[v]['initial_state_hash'] for v in VARIANTS})==1,'normal_plan_mismatch_epochs':[],'aux_pair_mismatch_epochs':[],'rank_offset_mismatch_epochs':[],'lr_trajectory_exact':True,'normal_optimizer_steps':{v:rows[v]['normal_optimizer_steps_total'] for v in VARIANTS},'warmup_epochs':{v:rows[v]['warmup_epochs'] for v in VARIANTS},'lambda_HN':{v:rows[v]['lambda_HN'] for v in VARIANTS},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    for ep in map(str,range(FORMAL_EPOCHS)):
        if len({rows[v]['normal_plan_hashes'][ep] for v in VARIANTS})!=1: out['normal_plan_mismatch_epochs'].append(int(ep))
        if len({rows[v]['aux_pair_hashes'][ep] for v in VARIANTS})!=1: out['aux_pair_mismatch_epochs'].append(int(ep))
        if len({rows[v]['rank_offset_hashes'][ep] for v in VARIANTS})!=1: out['rank_offset_mismatch_epochs'].append(int(ep))
    out['lr_trajectory_exact']=all(rows[v]['lr_trajectory']==ref['lr_trajectory'] for v in VARIANTS)
    out['PASS']=bool(out['initial_state_exact'] and not out['normal_plan_mismatch_epochs'] and not out['aux_pair_mismatch_epochs'] and not out['rank_offset_mismatch_epochs'] and out['lr_trajectory_exact'] and len(set(out['normal_optimizer_steps'].values()))==1)
    json_write(EVID/f'ROUND26_FAIRNESS_SEED{seed}.json',out); return out


def validation_summary():
    rows={str(s):{v:json.loads((EVID/f'ROUND26_{v}_SEED{s}.json').read_text()) for v in VARIANTS} for s in SEEDS}
    out={'protocol':PROTOCOL,'seeds':{},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    for s in SEEDS:
        r=rows[str(s)]; z={'best_epochs':{v:r[v]['best_epoch'] for v in VARIANTS},'best_metrics':{v:r[v]['best_evaluation']['colift'] for v in VARIANTS}}
        z['C1FULL_vs_B0']=delta_pack(z['best_metrics']['C1FULL'],z['best_metrics']['B0'])
        z['O1_vs_C1FULL']=delta_pack(z['best_metrics']['O1'],z['best_metrics']['C1FULL'])
        z['D3_vs_C1FULL']=delta_pack(z['best_metrics']['D3'],z['best_metrics']['C1FULL'])
        z['D3_vs_O1']=delta_pack(z['best_metrics']['D3'],z['best_metrics']['O1'])
        z['D3_vs_B0']=delta_pack(z['best_metrics']['D3'],z['best_metrics']['B0'])
        den=z['O1_vs_C1FULL']['U']; z['Recovery']=None if den<=0 else float(z['D3_vs_C1FULL']['U']/den)
        out['seeds'][str(s)]=z
    for key in ('C1FULL_vs_B0','O1_vs_C1FULL','D3_vs_C1FULL','D3_vs_O1','D3_vs_B0'):
        out['mean_U_'+key]=float(np.mean([out['seeds'][str(s)][key]['U'] for s in SEEDS]))
    out['minimum_diffusion_success']=bool(all(out['seeds'][str(s)]['D3_vs_C1FULL']['U']>0 for s in SEEDS) and out['mean_U_D3_vs_C1FULL']>0)
    out['strong_success']=bool(out['minimum_diffusion_success'] and all(out['seeds'][str(s)]['D3_vs_C1FULL']['primary_positive_count']>=3 and out['seeds'][str(s)]['D3_vs_B0']['U']>0 for s in SEEDS) and out['mean_U_D3_vs_C1FULL']>=.0025)
    out['oracle_interface_valid']=bool(all(out['seeds'][str(s)]['O1_vs_C1FULL']['U']>0 for s in SEEDS))
    out['ROUND26_BABY_METHOD']='FROZEN' if out['minimum_diffusion_success'] else 'NOT_FROZEN_FOR_CROSS_DOMAIN'
    json_write(EVID/'ROUND26_VALIDATION_SUMMARY.json',out); return out
