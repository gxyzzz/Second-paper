from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round8_bounded_residual import BoundedResidualDDPM,cosine_alpha_bar,residual_start,ddim_reverse_inference,decode_residual
from diffusion_experiments.modules.round8_common import *
from modules.ranking import metrics_at

def load_model(path,device):
 ck=torch.load(path,map_location='cpu',weights_only=False); dc=ck['config']; m=BoundedResidualDDPM(hidden_dim=int(dc['hidden_dim']),time_dim=int(dc['time_dim']),dropout=float(dc['dropout']),hidden_layers=int(dc['hidden_layers']),radius=float(dc['radius'])).to(device); m.load_state_dict(ck['model'],strict=True); m.eval(); return m,ck

def generate_delta(model,dep,bseed,dseed,cfg):
 device=torch.device('cuda:0'); users=dep['users'].astype(np.int64); cond=dep['cond'].astype(np.float32); budget=dep['budget'].astype(np.float32); valid=dep['valid'].astype(bool); alpha=cosine_alpha_bar(int(cfg['diffusion']['steps']),float(cfg['diffusion']['cosine_s']),device=device); path=[int(x) for x in cfg['diffusion']['ddim_path']]; tedit=int(cfg['diffusion']['t_edit']); radius=float(cfg['diffusion']['radius']); out=np.zeros((len(users),64),np.float32); batch=512
 for st in range(0,len(users),batch):
  en=min(st+batch,len(users)); ub=users[st:en]; cb=torch.as_tensor(cond[st:en],device=device); bb=torch.as_tensor(budget[st:en],device=device); noises=antithetic_noise(bseed,dseed,ub,64); acc=torch.zeros((len(ub),64),device=device)
  for k in range(4):
   eps=torch.as_tensor(noises[:,k],device=device); Y=ddim_reverse_inference(model,residual_start(eps,alpha,tedit),cb,alpha,path); acc+=decode_residual(Y,bb,radius)
  d=(acc/4).cpu().numpy().astype(np.float32); d[~valid[st:en]]=0; out[st:en]=d
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--backbone-seed',type=int,required=True); ap.add_argument('--diffusion-seed',type=int,required=True); ap.add_argument('--assets',required=True); ap.add_argument('--generator',required=True); ap.add_argument('--out',required=True); ap.add_argument('--unlabeled',action='store_true'); a=ap.parse_args(); out=Path(a.out)
 if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty output')
 out.mkdir(parents=True,exist_ok=True); t0=time.time(); cfg=cfg_round8(); assets=Path(a.assets); generator=Path(a.generator); audit=json.loads((assets/'audit.json').read_text()); train_result=json.loads((generator.parent/'result.json').read_text()); device=torch.device('cuda:0')
 if int(audit['backbone_seed'])!=a.backbone_seed or int(train_result['backbone_seed'])!=a.backbone_seed or int(train_result['diffusion_seed'])!=a.diffusion_seed: raise RuntimeError('identity mismatch')
 model,ck=load_model(generator,device); dep=np.load(assets/'deployment.npz'); base=np.load(assets/'baseline_all_users.npz'); users=base['users'].astype(np.int64); m0=base['m0_items'].astype(np.int32); m1=base['m1_items'].astype(np.int32); s0=base['m1_s0'].astype(np.float32); A=base['A_mask'].astype(bool); q=base['q'].astype(np.float32); center=base['m'].astype(np.float32); R=base['R'].astype(np.float32); item=dep['collab_item'].astype(np.float32); h=dep['history_raw'].astype(np.float32); budget=dep['budget'].astype(np.float32); valid=dep['valid'].astype(bool)
 delta=generate_delta(model,dep,a.backbone_seed,a.diffusion_seed,cfg); dn=np.linalg.norm(delta,axis=1); hn=np.linalg.norm(h,axis=1); ratio=np.divide(dn,budget,out=np.zeros_like(dn),where=budget>0); gratio=np.divide(np.linalg.norm(h+delta,axis=1),hn,out=np.ones_like(hn),where=hn>0)
 if ratio.max()>1.0001 or gratio.max()>1.5001: raise RuntimeError('NUMERICAL_INVALID deployment bound')
 r,scale=bounded_scores_np(delta,item[m1],center,q,R); r[~valid]=0; maxabs=float(np.max(np.abs(r)))
 if maxabs>1.00001: raise RuntimeError(f'NUMERICAL_INVALID score bound {maxabs}')
 Ame=np.array([float(r[u,A[u]].mean()) for u in range(len(users))]); Astd=np.array([float(r[u,A[u]].std()) for u in range(len(users))]); arms=np.array([float(np.sqrt(np.mean(r[u,A[u]]**2))) for u in range(len(users))])
 if np.max(np.abs(Ame))>2e-5: raise RuntimeError(f'A-centering mismatch {np.max(np.abs(Ame))}')
 identity=rerank_slots(m1,s0,A,r,0.0)
 if not np.array_equal(identity,m1): raise RuntimeError('eta0 identity failure')
 fc=cfg['fusion']; rankings={}; eta_results={}
 for eta in [float(x) for x in fc['eta_candidates']]:
  rank=rerank_slots(m1,s0,A,r,eta); rankings[eta]=rank
  if not np.array_equal(rank[:,:int(fc['protected_rank_le'])],m1[:,:int(fc['protected_rank_le'])]) or not np.array_equal(rank[~A],m1[~A]): raise RuntimeError('slot protection failure')
  eta_results[str(eta)]={'changed_users':int(np.sum(np.any(rank!=m1,axis=1))),'changed_slots':int(np.sum(rank!=m1))}
 raw_num=np.einsum('bd,bld->bl',delta,item[m1]-center[:,None,:],optimize=True); rank_mismatch=0
 for u in range(len(users)):
  cols=np.flatnonzero(A[u]); o1=np.argsort(-raw_num[u,cols],kind='stable'); o2=np.argsort(-r[u,cols],kind='stable'); rank_mismatch+=int(not np.array_equal(o1,o2))
 diag={'delta_norm_mean':float(dn.mean()),'delta_norm_median':float(np.median(dn)),'delta_over_b_mean':float(ratio.mean()),'delta_over_b_max':float(ratio.max()),'g_over_h_mean':float(gratio.mean()),'g_over_h_max':float(gratio.max()),'r_min':float(r.min()),'r_max':float(r.max()),'r_abs_max':maxabs,'A_r_mean_abs_max':float(np.max(np.abs(Ame))),'A_r_rms_mean':float(arms.mean()),'A_r_std_mean':float(Astd.mean()),'near_constant_A_fraction':float(np.mean(Astd<1e-5)),'score_rank_mismatch_users':int(rank_mismatch),'scale_q_branch_fraction':float(np.mean(q>=R*dn)),'finite_users':int(np.isfinite(delta).all(1).sum())}
 m0m=m1m=None
 if not a.unlabeled:
  df=load_interactions(); vu,vsets=label_sets(df,1)
  if not np.array_equal(vu,users): raise RuntimeError('Validation user mismatch')
  m0m=metrics_at(m0,users,vsets); m1m=metrics_at(m1,users,vsets)
  if abs(m0m['R20']-audit['validation_m0_metrics']['R20'])>1e-12 or abs(m1m['R20']-audit['validation_m1_metrics']['R20'])>1e-12: raise RuntimeError('baseline Validation mismatch')
  for eta,rank in rankings.items():
   met=metrics_at(rank,users,vsets); eta_results[str(eta)].update({'metrics':met,'U_vs_M1':relative_u(met,m1m),'K10':transition_counts(m1,rank,users,vsets,10),'K20':transition_counts(m1,rank,users,vsets,20)})
 np.savez_compressed(out/'cache.npz',users=users,delta=delta,r=r,scale=scale,valid=valid)
 save={'users':users,'m0':m0,'m1':m1}
 for eta,rank in rankings.items(): save[f"m2_eta_{str(eta).replace('.','p')}"]=rank
 np.savez_compressed(out/'validation_rankings.npz',**save)
 result={'status':'COMPLETE_UNLABELED_PILOT' if a.unlabeled else 'COMPLETE_VALIDATION_CACHE','protocol_version':cfg['protocol_version'],'backbone_seed':a.backbone_seed,'diffusion_seed':a.diffusion_seed,'generator_sha256':sha(generator),'assets_audit_sha256':sha(assets/'audit.json'),'M0_metrics':m0m,'M1_metrics':m1m,'eta_results':eta_results,'diagnostics':diag,'cache_sha256':sha(out/'cache.npz'),'rankings_sha256':sha(out/'validation_rankings.npz'),'git_sha':git_sha(),'access':{'TRAIN':True,'VALIDATION_LABELS_USED':not a.unlabeled,'TEST_LABELS_USED':False},'elapsed_seconds':time.time()-t0}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':result['status'],'cell':f'{a.backbone_seed}x{a.diffusion_seed}','eta':eta_results,'diag':diag},sort_keys=True))
if __name__=='__main__': main()
