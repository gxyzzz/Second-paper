from __future__ import annotations
import argparse, gc, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from modules.diffusion import NativeTVX0Denoiser, purify_indices
from pipelines.dataset_config import load_dataset_config

SEED_MAP={0.0:20261001,0.25:20261251,0.5:20261501,0.75:20261751,1.0:20261101}
SEED_PROTOCOL='HISTORICAL_M31B_EXACT'
PURIFICATION_SEEDS=[20261001,20261002,20261003,20261004]
T_VALUES=[2,3,5,7,10]
G_VALUES=[1.0,1.25,1.5,1.75,2.0]

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def btag(x): return ('%.1f'%float(x)).replace('.','p')
def gtag(x): return ('%.1f'%float(x)).replace('.','p')

def training_meta(root,beta):
 if beta==1.0:
  p=Path('runs/diffusion_repro/baby_m31_formal/evidence/beta_1p0_training.json')
 elif beta==0.0:
  p=root/'beta_0p0/evidence/beta_0p0_training.json'
 elif beta==0.25:
  p=root/'beta_0p25/evidence/beta_0p2_training.json'
 elif beta==0.5:
  p=root/'beta_0p5/evidence/beta_0p5_training.json'
 elif beta==0.75:
  p=root/'beta_0p75/evidence/beta_0p8_training.json'
 else:
  raise ValueError(beta)
 return p,json.loads(p.read_text())

def validate_provenance(root,beta,ck,cond):
 mp,meta=training_meta(root,beta)
 required={'training_seed':SEED_MAP[beta],'epochs':80,'final_epoch':80,'train_item_count':7050,
           'training_protocol':'m31_fixed_all_items','train_scope':'all_catalog_items',
           'checkpoint_selection':'final_epoch','VALIDATION_TARGET_USED':False,
           'TEST_TARGET_USED':False,'TEST_ACCESSED':False}
 for k,v in required.items():
  if meta.get(k)!=v: raise RuntimeError(('provenance mismatch',beta,k,meta.get(k),v))
 if sha256(ck)!=meta['checkpoint_sha256']: raise RuntimeError(('checkpoint sha mismatch',beta))
 if sha256(cond)!=meta['condition_sha256']: raise RuntimeError(('condition sha mismatch',beta))
 return mp,meta

def run(beta,root,smoke=False):
 root=Path(root); beta=float(beta); cfg=load_dataset_config('baby'); paths=cfg['resolved_paths']
 ck=root/'checkpoints'/f'baby_beta_{btag(beta)}.pt'
 condp=root/'assets'/f'condition_beta_{btag(beta)}.npy'
 mp,meta=validate_provenance(root,beta,ck,condp)
 raw_t=np.load(paths['text_feature'],mmap_mode='r',allow_pickle=False)
 raw_v=np.load(paths['visual_feature'],mmap_mode='r',allow_pickle=False)
 cond=np.load(condp,mmap_mode='r',allow_pickle=False)
 z=torch.load(ck,map_location='cpu',weights_only=False)
 model=NativeTVX0Denoiser(int(z['D']),cond_dim=64,hidden=int(z['hidden']),time_dim=64).cuda()
 model.load_state_dict(z['state_dict'],strict=True); model.eval()
 ids=np.arange(len(raw_t),dtype=np.int64)
 ts=T_VALUES[:1] if smoke else T_VALUES
 gs=G_VALUES[:1] if smoke else G_VALUES
 outroot=root/('purified_smoke' if smoke else 'purified')/f'beta_{btag(beta)}'
 outroot.mkdir(parents=True,exist_ok=True)
 rows=[]
 for t in ts:
  for g in gs:
   stem=outroot/f't{t}_g{gtag(g)}'
   tp=Path(str(stem)+'_text.npy'); vp=Path(str(stem)+'_visual.npy'); ep=Path(str(stem)+'.json')
   if tp.exists() and vp.exists() and ep.exists():
    old=json.loads(ep.read_text())
    if old.get('checkpoint_sha256')==sha256(ck) and old.get('training_seed')==SEED_MAP[beta] and old.get('condition_sha256')==sha256(condp):
     rows.append(old); print('HIST_PURIFY_REUSE',beta,t,g,flush=True); continue
   t0=time.time()
   ot,ov=purify_indices(model,raw_t,raw_v,cond,ids,t_edit=t,guidance=g,seeds=tuple(PURIFICATION_SEEDS),batch=256,device='cuda')
   np.save(tp,ot.astype(np.float32)); np.save(vp,ov.astype(np.float32))
   ev={'dataset':'baby','beta':beta,'training_seed':SEED_MAP[beta],'seed_protocol':SEED_PROTOCOL,
       't_edit':t,'guidance':g,'purification_seeds':PURIFICATION_SEEDS,
       'checkpoint_path':str(ck),'checkpoint_sha256':sha256(ck),
       'condition_path':str(condp),'condition_sha256':sha256(condp),
       'training_evidence':str(mp),'text_path':str(tp),'text_sha256':sha256(tp),
       'visual_path':str(vp),'visual_sha256':sha256(vp),'runtime_sec':time.time()-t0,
       'VALIDATION_ONLY':True,'TEST_ACCESSED':False}
   ep.write_text(json.dumps(ev,indent=2)+'\n'); rows.append(ev)
   print('HIST_PURIFY_DONE',beta,t,g,ev['runtime_sec'],flush=True)
   del ot,ov; torch.cuda.empty_cache(); gc.collect()
 summary={'phase':'BABY_HISTORICAL_SEED_PURIFICATION','beta':beta,'training_seed':SEED_MAP[beta],
          'seed_protocol':SEED_PROTOCOL,'asset_count':len(rows),'assets':rows,
          'VALIDATION_ONLY':True,'TEST_ACCESSED':False}
 sp=root/'evidence'/f'historical_beta_{btag(beta)}_purification.json'
 if not smoke: sp.write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({'beta':beta,'training_seed':SEED_MAP[beta],'asset_count':len(rows),'smoke':smoke,'PASS':True},sort_keys=True))

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--beta',type=float,required=True); ap.add_argument('--root',default='runs/diffusion_rescue/baby_historical_seed_replay'); ap.add_argument('--smoke',action='store_true')
 a=ap.parse_args(); run(a.beta,a.root,a.smoke)
