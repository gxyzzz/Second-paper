from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from diffusion_experiments.round28_candidate_conditioned_interest_diffusion import round28_core as r28
ROOT=r28.ROOT
RDIR=ROOT/'diffusion_experiments/round29_positive_preference_purification'
EVID=RDIR/'evidence'; LOGS=RDIR/'logs'; OUT=RDIR/'outputs'; ASSETS=RDIR/'assets'
for p in (EVID,LOGS,OUT,ASSETS): p.mkdir(parents=True,exist_ok=True)
PROTOCOL='ROUND29_POSITIVE_SIDE_USER_CONDITIONED_DIFFUSION_PREFERENCE_PURIFICATION_V1'
SEEDS=(999,1000); VARIANTS=('D6',)
WARMUP_EPOCHS=10; FORMAL_EPOCHS=60; SMOKE_EPOCHS=15
LAMBDA_PUR=.10; THETA_MAX_DEG=5.0; EPS=1e-8
K_HISTORY=8; MIN_HISTORY=3; MIN_COHERENT=2
ENSEMBLE_NOISE_SEEDS=(20261011,20261012,20261013)
LOCAL_TS=(1,2,3,4,5); DIFF_STEPS_WARMUP=1; DIFF_STEPS_ACTIVE=2
SEP_MARGIN=.20; L_DIR=.50; L_SEP=.25; L_BOUND=.05
ALL=r28.ALL; PRIMARY=r28.PRIMARY
instantiate_shared=r28.instantiate_shared
TrainEvents=r28.TrainEvents
forward_bundle=r28.forward_bundle
msca_loss_from_bundle=r28.msca_loss_from_bundle
latent_stats=r28.latent_stats
state_hash=r28.state_hash
stats=r28.stats
CachedEvaluator=r28.CachedEvaluator
sha256_file=r28.sha256_file
make_diff=r28.make_diff
z=r28.z; inv_item=r28.inv_item
capture_rng=r28.capture_rng; restore_rng=r28.restore_rng
R28_EVID=ROOT/'diffusion_experiments/round28_candidate_conditioned_interest_diffusion/evidence'
def json_write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:float(x) if isinstance(x,np.generic) else str(x))+'\n')
def protocol_hash():
    h=hashlib.sha256()
    for name in ('round29_core.py','round29_colift_reference.py','round29_positive_field.py','round29_diffusion.py','round29_train.py','round29_audits.py','run_round29.py'):
        p=RDIR/name
        if p.exists(): h.update(name.encode()); h.update(p.read_bytes())
    h.update(json.dumps({'K_HISTORY':K_HISTORY,'MIN_HISTORY':MIN_HISTORY,'MIN_COHERENT':MIN_COHERENT,'theta':THETA_MAX_DEG,'lambda_pur':LAMBDA_PUR,'warmup':WARMUP_EPOCHS,'epochs':FORMAL_EPOCHS,'local_t':LOCAL_TS,'diff_active':DIFF_STEPS_ACTIVE,'ensemble_noise_seeds':ENSEMBLE_NOISE_SEEDS,'loss':['L_base','0.50*L_dir','0.25*L_sep','0.05*L_bound'],'sep_margin':SEP_MARGIN,'boundary_negative':False,'synthetic_negative':False,'diffusion_negative':False},sort_keys=True).encode())
    return h.hexdigest()
def load_round28_b0(seed):
    return json.loads((R28_EVID/f'ROUND28_B0_SEED{seed}.json').read_text())
def baseline_record(seed):
    x=load_round28_b0(seed)
    return {'baseline_source':'ROUND28_B0','baseline_protocol':x['protocol'],'baseline_protocol_hash':x.get('protocol_hash'),'baseline_checkpoint':x['best_checkpoint'],'baseline_checkpoint_sha256':x['checkpoint_sha256'],'baseline_best_epoch':x['best_epoch'],'baseline_best_evaluation':x['best_evaluation'],'baseline_trajectory_source':str(R28_EVID/f'ROUND28_B0_SEED{seed}.json')}
def save_best(model,net,seed,epoch,score):
    p=OUT/f'seed{seed}/D6/best.pt'; p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'protocol':PROTOCOL,'protocol_hash':protocol_hash(),'seed':int(seed),'variant':'D6','epoch':int(epoch),'full_colift_R20':float(score),'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'diffusion_state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False},p)
    return p
def comparison(metrics,baseline):
    rel={k:float((metrics[k]-baseline[k])/baseline[k]) for k in ALL}; absd={k:float(metrics[k]-baseline[k]) for k in ALL}
    return {'U4':float(np.mean([rel[k] for k in PRIMARY])),'absolute_delta':absd,'relative_delta':rel,'primary_positive_count':int(sum(metrics[k]>baseline[k] for k in PRIMARY)),'overall_positive_count':int(sum(metrics[k]>baseline[k] for k in ALL))}
def validation_summary():
    out={'protocol':PROTOCOL,'protocol_hash':protocol_hash(),'baseline_source':'ROUND28_B0','seeds':{},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}; us=[]
    for seed in SEEDS:
        d=json.loads((EVID/f'ROUND29_D6_SEED{seed}.json').read_text()); b=load_round28_b0(seed); bm=b['best_evaluation']['colift']; dm=d['best_evaluation']['colift']; cmp=comparison(dm,bm); us.append(cmp['U4'])
        out['seeds'][str(seed)]={'baseline_source':'ROUND28_B0','baseline_protocol':b['protocol'],'baseline_checkpoint_sha256':b['checkpoint_sha256'],'best_epochs':{'B0':b['best_epoch'],'D6':d['best_epoch']},'best_metrics':{'B0':bm,'D6':dm},'D6_vs_B0':cmp}
    out['mean_U4_D6_vs_B0']=float(np.mean(us)); all_pos=all(x>0 for x in us)
    strong=all(x>=.005 for x in us) and out['mean_U4_D6_vs_B0']>=.01 and all(out['seeds'][str(s)]['D6_vs_B0']['primary_positive_count']>=3 for s in SEEDS)
    minimum=all_pos and out['mean_U4_D6_vs_B0']>=.01
    positive_below=all_pos and not minimum and all(out['seeds'][str(s)]['D6_vs_B0']['primary_positive_count']>=2 for s in SEEDS)
    out['minimum_success']=bool(minimum); out['strong_pass']=bool(strong); out['positive_but_below_target']=bool(positive_below)
    out['ROUND29_STATUS']='STRONG_PASS' if strong else ('PASS' if minimum else ('POSITIVE_BUT_BELOW_TARGET' if positive_below else 'FAIL')); out['BABY_TEST_ALLOWED_AUTOMATICALLY']=False
    json_write(EVID/'ROUND29_VALIDATION_SUMMARY.json',out); return out
