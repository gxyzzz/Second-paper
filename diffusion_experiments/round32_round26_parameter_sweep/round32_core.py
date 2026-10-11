from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from diffusion_experiments.round26_preference_directed_diffusion import round26_core as r26

ROOT=r26.ROOT
RDIR=ROOT/'diffusion_experiments/round32_round26_parameter_sweep'
EVID=RDIR/'evidence'; LOGS=RDIR/'logs'; OUT=RDIR/'outputs'; ASSETS=RDIR/'assets'; CONFIGS=RDIR/'configs'
for p in (EVID,LOGS,OUT,ASSETS,CONFIGS): p.mkdir(parents=True,exist_ok=True)
PROTOCOL='ROUND32_ROUND26_PARAMETER_SWEEP_V1'
SEEDS=(999,1000)
VARIANT_CONFIGS={
 'R26_BASE': {'lambda_HN':.20,'half_margin':.50,'theta_max_deg':5.0,'reuse':True,'group':'REFERENCE'},
 'L10': {'lambda_HN':.10,'half_margin':.50,'theta_max_deg':5.0,'group':'A'},
 'L15': {'lambda_HN':.15,'half_margin':.50,'theta_max_deg':5.0,'group':'A'},
 'L25': {'lambda_HN':.25,'half_margin':.50,'theta_max_deg':5.0,'group':'A'},
 'M25': {'lambda_HN':.20,'half_margin':.25,'theta_max_deg':5.0,'group':'B'},
 'M35': {'lambda_HN':.20,'half_margin':.35,'theta_max_deg':5.0,'group':'B'},
 'M65': {'lambda_HN':.20,'half_margin':.65,'theta_max_deg':5.0,'group':'B'},
 'T3': {'lambda_HN':.20,'half_margin':.50,'theta_max_deg':3.0,'group':'C'},
 'T4': {'lambda_HN':.20,'half_margin':.50,'theta_max_deg':4.0,'group':'C'},
 'T6': {'lambda_HN':.20,'half_margin':.50,'theta_max_deg':6.0,'group':'C'},
 'C1': {'lambda_HN':.15,'half_margin':.35,'theta_max_deg':4.0,'group':'D'},
 'C2': {'lambda_HN':.10,'half_margin':.25,'theta_max_deg':3.0,'group':'D'},
 'C3': {'lambda_HN':.20,'half_margin':.35,'theta_max_deg':4.0,'group':'D'},
}
TRAIN_VARIANTS=tuple(k for k,v in VARIANT_CONFIGS.items() if not v.get('reuse'))
WARMUP_EPOCHS=10; FORMAL_EPOCHS=60; EPS=r26.EPS; LOCAL_TS=r26.LOCAL_TS; DIFF_STEPS_WARMUP=1; DIFF_STEPS_ACTIVE=2
LAMBDA_HN=.20; HALF_MARGIN=.50; THETA_MAX_DEG=5.0
ALL=r26.ALL; PRIMARY=r26.PRIMARY; BANDS=r26.BANDS; BAND_NAMES=r26.BAND_NAMES
instantiate_shared=r26.instantiate_shared; TrainEvents=r26.TrainEvents; forward_bundle=r26.forward_bundle
msca_loss_from_bundle=r26.msca_loss_from_bundle; latent_stats=r26.latent_stats; valid_mask=r26.valid_mask
mine_current_ids=r26.mine_current_ids; rank_offsets=r26.rank_offsets; curriculum_count=r26.curriculum_count
aux_bpr=r26.aux_bpr; state_hash=r26.state_hash; stats=r26.stats; delta_pack=r26.delta_pack
CachedEvaluator=r26.CachedEvaluator; sha256_file=r26.sha256_file; make_diff=r26.make_diff; z=r26.z; inv_item=r26.inv_item

def apply_variant(variant):
    global LAMBDA_HN,HALF_MARGIN,THETA_MAX_DEG
    if variant not in VARIANT_CONFIGS: raise ValueError(variant)
    q=VARIANT_CONFIGS[variant]; LAMBDA_HN=float(q['lambda_HN']); HALF_MARGIN=float(q['half_margin']); THETA_MAX_DEG=float(q['theta_max_deg'])
    return dict(q)

def json_write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:float(x) if isinstance(x,np.generic) else str(x))+'\n')

def protocol_hash():
    h=hashlib.sha256()
    for name in ('round32_core.py','round32_diffusion.py','round32_train.py','round32_eval.py','round32_audits.py','run_round32.py'):
        p=RDIR/name
        if p.exists(): h.update(name.encode()); h.update(p.read_bytes())
    h.update(json.dumps(VARIANT_CONFIGS,sort_keys=True).encode())
    return h.hexdigest()

def save_best(model,net,seed,variant,epoch,score):
    p=OUT/f'seed{seed}'/variant/'best.pt'; p.parent.mkdir(parents=True,exist_ok=True)
    cfg=VARIANT_CONFIGS[variant]
    obj={'protocol':PROTOCOL,'seed':seed,'variant':variant,'config':cfg,'epoch':int(epoch),'full_colift_R20':float(score),
         'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
         'diffusion_state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},
         'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    torch.save(obj,p); return p
