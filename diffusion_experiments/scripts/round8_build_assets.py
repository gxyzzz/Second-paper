from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.modules.round8_common import *
from modules.ranking import candidate_dot_scores,metrics_at

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out)
 if out.exists() and any(out.iterdir()): raise RuntimeError(f'refuse overwrite nonempty {out}')
 out.mkdir(parents=True,exist_ok=True); t0=time.time(); cfg=cfg_round8(); frozen=load_frozen_backbone(a.seed); emb=frozen['embeddings']; scores=frozen['scores']
 n_users=int(frozen['msca_audit']['n_users']); n_items=int(frozen['msca_audit']['n_items']); item_raw=emb['collab_item'].astype(np.float32); user_raw=emb['collab_user'].astype(np.float32)
 df=load_interactions(); train=train_frame(df)
 if len(train)!=118551: raise RuntimeError(f'full TRAIN mismatch {len(train)}')
 histories=unique_histories(train,n_users); train_users=np.sort(train.userID.unique()).astype(np.int64)
 item_z,item_mean,item_std,item_std_safe,item_const,observed=fit_cf_statistics(item_raw,train,float(cfg['cf']['std_floor'])); user_z,user_mean,user_std,user_std_safe=fit_user_statistics(user_raw,train_users,float(cfg['cf']['std_floor']))
 hist=build_history_arrays(train,histories,item_z,item_raw,user_z); event_h_raw,event_len_raw=build_event_raw_histories(train,histories,item_raw)
 if not np.array_equal(event_len_raw,hist['event_len']): raise RuntimeError('raw/std positive-removal history mismatch')
 users_e=hist['users'].astype(np.int64); pos=hist['pos'].astype(np.int64); hnorm=np.linalg.norm(event_h_raw,axis=1).astype(np.float32); budget=(0.5*hnorm).astype(np.float32); valid=(budget>1e-6)&hist['has_anchor']
 delta_star=np.zeros_like(event_h_raw); y_target=np.zeros_like(event_h_raw); raw_gap=item_raw[pos]-event_h_raw
 if valid.any():
  delta_star[valid]=radial_project_np(raw_gap[valid],budget[valid]); y_target[valid]=8.0*delta_star[valid]/budget[valid,None]
 y_norm=np.linalg.norm(y_target,axis=1); delta_ratio=np.zeros(len(train),np.float32); delta_ratio[valid]=np.linalg.norm(delta_star[valid],axis=1)/budget[valid]
 if y_norm.max()>8.00001 or delta_ratio.max()>1.00001: raise RuntimeError('residual target bound violation')
 full_h_raw=hist['full_h_raw'].astype(np.float32); full_hnorm=np.linalg.norm(full_h_raw,axis=1).astype(np.float32); full_budget=(0.5*full_hnorm).astype(np.float32); full_valid=full_budget>1e-6
 valid_users,valid_sets=label_sets(df,1); users,m0_items,m0_scores,m1_items,m1_s0=m1_sorted(scores)
 if not np.array_equal(users,valid_users): raise RuntimeError('Validation user order mismatch')
 m0m=metrics_at(m0_items,users,valid_sets); m1m=metrics_at(m1_items,users,valid_sets); expected0=frozen['colift_summary']['metrics']['MSCA_CANONICAL']; expected1=frozen['colift_summary']['metrics']['MSCA_FULL_COLIFTREC_TAV']
 d0=max(abs(float(m0m[k])-float(expected0[k])) for k in ALL_METRICS); d1=max(abs(float(m1m[k])-float(expected1[k])) for k in ALL_METRICS)
 if d0>1e-12 or d1>1e-12: raise RuntimeError(f'BASE_IDENTITY_MISMATCH {d0} {d1}')
 cf_candidate=candidate_dot_scores(user_raw,item_raw,users,m0_items); q=np.maximum(cf_candidate.std(1),float(cfg['inference']['q_floor'])).astype(np.float32); fc=cfg['fusion']; A=build_boundary_mask(m1_items,m1_s0,histories,n_items,cutoffs=tuple(fc['cutoffs']),protected_rank_le=int(fc['protected_rank_le']),max_score_distance=float(fc['max_score_distance']),per_cutoff_quota=int(fc['per_cutoff_quota']),max_items=int(fc['max_items']))
 center,R,emptyA=reference_center_radius(m1_items,A,item_raw)
 if emptyA.any(): raise RuntimeError(f'empty A users {int(emptyA.sum())}')
 # exact directory-radius check on deterministic subset
 check_users=np.linspace(0,n_users-1,33,dtype=np.int64); max_rad_err=0.0
 for u in check_users:
  direct=float(np.max(np.linalg.norm(item_raw-center[u],axis=1))); max_rad_err=max(max_rad_err,abs(direct-float(R[u])))
 if max_rad_err>2e-5: raise RuntimeError(f'catalog radius mismatch {max_rad_err}')
 np.savez_compressed(out/'events.npz',users=users_e.astype(np.int32),pos=pos.astype(np.int32),cond=hist['cond'].astype(np.float32),history_raw=event_h_raw,budget=budget,y_target=y_target,valid=valid,event_len=hist['event_len'].astype(np.int32),m=center[users_e],R=R[users_e],q=q[users_e])
 np.savez_compressed(out/'deployment.npz',users=np.arange(n_users,dtype=np.int32),cond=hist['cond_full'].astype(np.float32),history_raw=full_h_raw,budget=full_budget,valid=full_valid,history_len=hist['full_len'].astype(np.int32),collab_user=user_raw,collab_item=item_raw,item_z=item_z,item_mean=item_mean,item_std_safe=item_std_safe,user_mean=user_mean,user_std_safe=user_std_safe,observed_items=observed.astype(np.int32))
 np.savez_compressed(out/'baseline_all_users.npz',users=users.astype(np.int32),m0_items=m0_items.astype(np.int32),m0_scores=m0_scores.astype(np.float32),m1_items=m1_items.astype(np.int32),m1_s0=m1_s0.astype(np.float32),A_mask=A,q=q,m=center,R=R)
 files=['events.npz','deployment.npz','baseline_all_users.npz']; audit={'status':'COMPLETE_ROUND8_ASSETS','protocol_version':cfg['protocol_version'],'dataset':'baby','backbone_seed':int(a.seed),'source_checkpoint':str(frozen['checkpoint'].resolve()),'source_checkpoint_sha256':sha(frozen['checkpoint']),'source_checkpoint_epoch':int(frozen['msca_audit']['checkpoint_epoch']),'train_edges':int(len(train)),'n_users':n_users,'n_items':n_items,'condition_dim':129,'valid_events':int(valid.sum()),'invalid_events':int((~valid).sum()),'valid_deployment_users':int(full_valid.sum()),'invalid_deployment_users':int((~full_valid).sum()),'target_y_norm_max':float(y_norm.max()),'target_delta_over_b_max':float(delta_ratio.max()),'history_norm_mean':float(full_hnorm.mean()),'budget_mean':float(full_budget.mean()),'validation_m0_metrics':m0m,'validation_m1_metrics':m1m,'validation_identity_max_abs_diff':{'M0':d0,'M1':d1},'A':{'mean_size':float(A.sum(1).mean()),'min_size':int(A.sum(1).min()),'max_size':int(A.sum(1).max()),'empty_users':int(emptyA.sum())},'q':{'min':float(q.min()),'mean':float(q.mean()),'max':float(q.max())},'R':{'min':float(R.min()),'mean':float(R.mean()),'max':float(R.max()),'subset_direct_max_abs_diff':max_rad_err},'guide_sha256':sha(ROOT/'diffusion_experiments/ADVISOR_EXPERIMENT_GUIDE.md'),'git_sha':git_sha(),'artifacts':{f:sha(out/f) for f in files},'access':{'TRAIN':True,'VALIDATION_LABELS':True,'TEST_LABELS_USED':False},'elapsed_seconds':time.time()-t0}
 (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps({'status':audit['status'],'seed':a.seed,'valid_events':audit['valid_events'],'ymax':audit['target_y_norm_max'],'Rmean':audit['R']['mean'],'M1_R20':m1m['R20']},sort_keys=True))
if __name__=='__main__': main()
