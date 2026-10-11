from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch

from diffusion_experiments.round28_candidate_conditioned_interest_diffusion import round28_core as r28

ROOT = r28.ROOT
RDIR = ROOT / 'diffusion_experiments/round30_candidate_side_diffusion_purification'
EVID = RDIR / 'evidence'; LOGS = RDIR / 'logs'; OUT = RDIR / 'outputs'; ASSETS = RDIR / 'assets'
for p in (EVID, LOGS, OUT, ASSETS): p.mkdir(parents=True, exist_ok=True)
PROTOCOL = 'ROUND30_FROZEN_BACKBONE_CANDIDATE_SIDE_DIFFUSION_PURIFICATION_V1'
SEEDS=(999,1000); VARIANT='D7'; STD_FLOOR=1e-4
K_HISTORY=8; MIN_HISTORY=3; MIN_COHERENT=2; THETA_MAX_DEG=5.0
WARMUP_EPOCHS=10; ACTIVE_EPOCHS=30; FORMAL_EPOCHS=40; SMOKE_EPOCHS=15
DIFF_STEPS_WARMUP=1; DIFF_STEPS_ACTIVE=2
ENSEMBLE_NOISE_SEEDS=(20261011,20261012,20261013); LOCAL_TS=(1,2,3,4,5)
SEP_MARGIN=.20; L_DIR=.50; L_SEP=.25; L_PROJ=.25; L_BOUND=.05; EPS=1e-8
TOP_PURIFY=30; ANCHOR_BANDS=((1,10),(11,30),(31,60),(61,100))
PRIMARY=r28.PRIMARY; ALL=r28.ALL
instantiate_shared=r28.instantiate_shared; TrainEvents=r28.TrainEvents; forward_bundle=r28.forward_bundle
latent_stats=r28.latent_stats; state_hash=r28.state_hash; stats=r28.stats; make_diff=r28.make_diff
z=r28.z; inv_item=r28.inv_item; capture_rng=r28.capture_rng; restore_rng=r28.restore_rng; sha256_file=r28.sha256_file
R28_EVID=ROOT/'diffusion_experiments/round28_candidate_conditioned_interest_diffusion/evidence'

def json_write(path,obj): Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:float(x) if isinstance(x,np.generic) else str(x))+'\n')
def load_round28_b0(seed): return json.loads((R28_EVID/f'ROUND28_B0_SEED{seed}.json').read_text())
def baseline_record(seed):
    x=load_round28_b0(seed); return {'baseline_source':'ROUND28_B0','baseline_protocol':x['protocol'],'baseline_protocol_hash':x.get('protocol_hash'),'baseline_checkpoint':x['best_checkpoint'],'baseline_checkpoint_sha256':x['checkpoint_sha256'],'baseline_best_epoch':x['best_epoch'],'baseline_best_evaluation':x['best_evaluation'],'baseline_trajectory_source':str(R28_EVID/f'ROUND28_B0_SEED{seed}.json')}
def load_frozen_b0_model(seed):
    model,config,td,vd,pack=instantiate_shared(seed); b0=load_round28_b0(seed); cp=torch.load(b0['best_checkpoint'],map_location='cpu',weights_only=False); model.load_state_dict(cp['model_state_dict'],strict=True)
    for p in model.parameters(): p.requires_grad=False
    model.eval(); h=state_hash(model.state_dict()); return model,config,td,vd,pack,b0,h

def protocol_hash():
    h=hashlib.sha256()
    for name in ('round30_core.py','round30_colift_reference.py','round30_candidate_field.py','round30_diffusion.py','round30_train_diffusion.py','round30_inference.py','round30_audits.py','run_round30.py'):
        p=RDIR/name
        if p.exists(): h.update(name.encode()); h.update(p.read_bytes())
    spec={'STD_FLOOR':STD_FLOOR,'K_HISTORY':K_HISTORY,'MIN_HISTORY':MIN_HISTORY,'MIN_COHERENT':MIN_COHERENT,'theta':THETA_MAX_DEG,'warmup':WARMUP_EPOCHS,'active':ACTIVE_EPOCHS,'epochs':FORMAL_EPOCHS,'local_t':LOCAL_TS,'diff_steps_active':DIFF_STEPS_ACTIVE,'ensemble':ENSEMBLE_NOISE_SEEDS,'top_purify':TOP_PURIFY,'anchor_bands':ANCHOR_BANDS,'loss':['L_base','0.50*L_dir','0.25*L_sep','0.25*L_proj','0.05*L_bound'],'sep_margin':SEP_MARGIN,'ranking_loss':False,'recommender_optimizer':False}
    h.update(json.dumps(spec,sort_keys=True).encode()); return h.hexdigest()
def comparison(metrics,baseline):
    rel={k:float((metrics[k]-baseline[k])/baseline[k]) for k in ALL}; absd={k:float(metrics[k]-baseline[k]) for k in ALL}
    return {'U4':float(np.mean([rel[k] for k in PRIMARY])),'absolute_delta':absd,'relative_delta':rel,'primary_positive_count':int(sum(metrics[k]>baseline[k] for k in PRIMARY)),'overall_positive_count':int(sum(metrics[k]>baseline[k] for k in ALL))}
def save_diffusion(net,seed,epoch,model_hash_before,model_hash_after,trajectory):
    p=OUT/f'seed{seed}'/VARIANT/f'epoch{epoch}.pt'; p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'protocol':PROTOCOL,'protocol_hash':protocol_hash(),'seed':int(seed),'variant':VARIANT,'epoch':int(epoch),'diffusion_state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'model_state_hash_before':model_hash_before,'model_state_hash_after':model_hash_after,'BACKBONE_PARAMETER_CHANGE':0 if model_hash_before==model_hash_after else 1,'trajectory':trajectory,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False},p); return p
def validation_summary():
    out={'protocol':PROTOCOL,'protocol_hash':protocol_hash(),'baseline_source':'ROUND28_B0','seeds':{},'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}; us=[]
    for seed in SEEDS:
        d=json.loads((EVID/f'ROUND30_SEED{seed}_VALIDATION.json').read_text()); us.append(d['D7_vs_B0']['U4']); out['seeds'][str(seed)]=d
    mean=float(np.mean(us)); all_pos=all(x>0 for x in us); minimum=bool(all_pos and mean>=.01); strong=bool(all(x>=.005 for x in us) and mean>=.01 and all(out['seeds'][str(s)]['D7_vs_B0']['primary_positive_count']>=3 for s in SEEDS)); positive_below=bool(all_pos and mean<.01)
    out.update({'mean_U4_D7_vs_B0':mean,'minimum_success':minimum,'strong_pass':strong,'positive_but_below_target':positive_below,'ROUND30_STATUS':'STRONG_PASS' if strong else ('PASS' if minimum else ('POSITIVE_BUT_BELOW_TARGET' if positive_below else 'FAIL')),'BABY_TEST_ALLOWED_AUTOMATICALLY':False}); json_write(EVID/'ROUND30_VALIDATION_SUMMARY.json',out); return out
