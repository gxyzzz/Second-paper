from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from modules.ranking import sha256_file
from diffusion_experiments.modules.round7_common import (
 PRIMARY,ALL_METRICS,load_interactions,train_frame,unique_histories,label_sets,
 fit_cf_statistics,fit_user_statistics,build_history_arrays,m1_sorted,build_boundary_mask,
 rerank_slots,relative_u,per_user_metric_arrays,bootstrap_u,transition_counts,state_hash)

def cfg_round8(): return yaml.safe_load((ROOT/'diffusion_experiments/configs/round8_baby.yaml').read_text())
def sha(path): return sha256_file(Path(path))
def git_sha():
 import subprocess; return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def backbone_cfg(seed):
 cfg=cfg_round8(); block=cfg['backbones'].get(int(seed),cfg['backbones'].get(str(int(seed))))
 if block is None: raise KeyError(f'unregistered backbone seed {seed}')
 return block

def load_frozen_backbone(seed):
 b=backbone_cfg(seed); checkpoint=ROOT/b['checkpoint']; msca_dir=ROOT/b['msca_assets']; colift_dir=ROOT/b['colift_assets']
 if sha(checkpoint)!=b['checkpoint_sha256']: raise RuntimeError(f'checkpoint hash mismatch for seed {seed}')
 ma=json.loads((msca_dir/'audit.json').read_text()); cs=json.loads((colift_dir/'summary.json').read_text()); ca=json.loads((colift_dir/'audit.json').read_text())
 if ma['checkpoint_sha256']!=b['checkpoint_sha256'] or cs['source_checkpoint_sha256']!=b['checkpoint_sha256']: raise RuntimeError('frozen baseline identity mismatch')
 if abs(float(ma['validation_metrics']['R20'])-float(b['expected_validation_msca_R20']))>1e-12: raise RuntimeError('MSCA Validation R20 mismatch')
 if abs(float(cs['metrics']['MSCA_FULL_COLIFTREC_TAV']['R20'])-float(b['expected_validation_colift_R20']))>1e-12: raise RuntimeError('CoLift Validation R20 mismatch')
 emb=np.load(msca_dir/'embeddings.npz'); scores=np.load(colift_dir/'validation_scores.npz')
 return {'config':b,'checkpoint':checkpoint,'msca_dir':msca_dir,'colift_dir':colift_dir,'msca_audit':ma,'colift_summary':cs,'colift_audit':ca,'embeddings':{k:emb[k].astype(np.float32) for k in emb.files},'scores':{k:scores[k] for k in scores.files}}

def radial_project_np(v,radius):
 x=np.asarray(v,np.float32); n=np.linalg.norm(x,axis=-1,keepdims=True); r=np.asarray(radius,np.float32)
 while r.ndim<n.ndim: r=np.expand_dims(r,-1)
 return (x*np.minimum(1.0,r/np.maximum(n,1e-12))).astype(np.float32)

def build_event_raw_histories(train,histories,item_raw):
 users=train.userID.to_numpy(np.int64); pos=train.itemID.to_numpy(np.int64); out=np.zeros((len(train),item_raw.shape[1]),np.float32); lens=np.zeros(len(train),np.int32); cache={}
 for idx,(u,p) in enumerate(zip(users,pos)):
  key=(int(u),int(p)); val=cache.get(key)
  if val is None:
   rem=[x for x in histories[int(u)] if int(x)!=int(p)]
   val=(item_raw[np.asarray(rem,np.int64)].mean(0).astype(np.float32),len(rem)) if rem else (np.zeros(item_raw.shape[1],np.float32),0); cache[key]=val
  out[idx],lens[idx]=val
 return out,lens

def reference_center_radius(m1_items,A,item_raw,chunk_users=256):
 n=len(m1_items); dim=item_raw.shape[1]; center=np.zeros((n,dim),np.float32); empty=np.zeros(n,bool)
 for u in range(n):
  cols=np.flatnonzero(A[u])
  if len(cols): center[u]=item_raw[m1_items[u,cols]].mean(0)
  else: empty[u]=True
 item_norm2=np.sum(item_raw.astype(np.float64)**2,axis=1); it=item_raw.astype(np.float64); radius=np.zeros(n,np.float32)
 for st in range(0,n,chunk_users):
  en=min(st+chunk_users,n); c=center[st:en].astype(np.float64)
  d2=item_norm2[None,:]+np.sum(c*c,axis=1)[:,None]-2.0*(c@it.T)
  radius[st:en]=np.sqrt(np.maximum(d2.max(axis=1),0.0)).astype(np.float32)
 return center,radius,empty

def bounded_scores_np(delta,item_vectors,center,q,radius):
 delta=np.asarray(delta,np.float32); center=np.asarray(center,np.float32); q=np.asarray(q,np.float32); radius=np.asarray(radius,np.float32)
 dn=np.linalg.norm(delta,axis=1); scale=np.maximum(q,radius*dn).astype(np.float32)
 if item_vectors.ndim==2:
  num=np.einsum('bd,bd->b',delta,item_vectors-center,optimize=True); r=num/scale
 elif item_vectors.ndim==3:
  num=np.einsum('bd,bld->bl',delta,item_vectors-center[:,None,:],optimize=True); r=num/scale[:,None]
 else: raise ValueError('item_vectors must be [B,D] or [B,L,D]')
 if not np.isfinite(r).all(): raise FloatingPointError('nonfinite bounded score')
 return r.astype(np.float32),scale

def keyed_gaussian(backbone_seed,diffusion_seed,user,base_noise_id,dim=64):
 key=f'round8|baby|{int(backbone_seed)}|{int(diffusion_seed)}|{int(user)}|{int(base_noise_id)}'.encode(); digest=hashlib.sha256(key).digest(); seed=int.from_bytes(digest[:8],'little',signed=False)
 return np.random.default_rng(seed).standard_normal(int(dim)).astype(np.float32)
def antithetic_noise(backbone_seed,diffusion_seed,users,dim=64):
 users=np.asarray(users,np.int64); out=np.empty((len(users),4,int(dim)),np.float32)
 for row,u in enumerate(users):
  e1=keyed_gaussian(backbone_seed,diffusion_seed,int(u),0,dim); e2=keyed_gaussian(backbone_seed,diffusion_seed,int(u),1,dim); out[row]=np.stack([e1,-e1,e2,-e2])
 return out
