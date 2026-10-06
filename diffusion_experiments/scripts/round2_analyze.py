from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
PRIMARY=('R10','N10','R20','N20'); ALL=('R10','N10','R20','N20','R50','N50')
RUNS=[('C',202610061),('C',202610062),('D0',202610061),('D0',202610062),('D1',202610061),('D1',202610062)]

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def dev_sets(users):
 keep=set(map(int,users)); out={int(u):set() for u in users}
 for c in pd.read_csv(ROOT/'data/baby/baby.inter',sep='\t',usecols=['userID','itemID','x_label'],chunksize=50000):
  z=c[(c.x_label==1)&c.userID.isin(keep)]
  for u,g in z.groupby('userID'): out[int(u)].update(g.itemID.astype(int).tolist())
 return out

def contrib(ranked,users,sets):
 k=50; hit=np.zeros((len(users),k),bool); plen=np.empty(len(users),np.int64)
 for r,u in enumerate(users):
  pos=sets[int(u)]; plen[r]=len(pos); hit[r]=np.fromiter((int(i) in pos for i in ranked[r,:k]),bool,count=k)
 cum=np.cumsum(hit,1); recall=cum/plen[:,None]; ranks=np.arange(1,k+1)[None,:]; dcg=np.cumsum(hit/np.log2(ranks+1.),1)
 idcg=np.empty_like(dcg,dtype=np.float64)
 disc=1/np.log2(np.arange(1,k+1)+1.)
 for r,p in enumerate(plen): idcg[r]=np.cumsum(np.r_[np.ones(min(int(p),k)),np.zeros(max(0,k-int(p)))][:k]*disc); idcg[r,int(p):]=idcg[r,max(int(p)-1,0)] if p<k else idcg[r,int(p):]
 ndcg=dcg/idcg
 arr=np.stack([recall[:,9],ndcg[:,9],recall[:,19],ndcg[:,19],recall[:,49],ndcg[:,49]],1)
 return arr,hit

def U_from_means(base,new): return float(np.mean((new[:4]-base[:4])/np.maximum(base[:4],1e-12)))
def summarize_arr(arr): return {k:float(arr[:,i].mean()) for i,k in enumerate(ALL)}
def boot(base,new,seed=202610069,B=1000):
 rng=np.random.default_rng(seed); n=len(base); vals=np.empty(B)
 for b in range(B):
  ix=rng.integers(0,n,n); vals[b]=U_from_means(base[ix].mean(0),new[ix].mean(0))
 return {'U':U_from_means(base.mean(0),new.mean(0)),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'positive_replicate_fraction':float((vals>0).mean()),'resamples':B,'seed':seed,'note':'conditional paired user bootstrap; positive fraction is not a p-value/probability'}
def rankpos(items,targets):
 out=np.zeros(len(items),np.int16)
 for r,(row,t) in enumerate(zip(items,targets)):
  p=np.flatnonzero(row==int(t)); out[r]=int(p[0])+1 if len(p) else 0
 return out

def rankchange(before,after):
 valid=(before>0)&(after>0); d=after.astype(int)-before.astype(int)
 return {'n':int(len(before)),'valid_both':int(valid.sum()),'improved':int((d[valid]<0).sum()),'worsened':int((d[valid]>0).sum()),'unchanged':int((d[valid]==0).sum()),'mean_rank_delta_after_minus_before':float(d[valid].mean()) if valid.any() else None,'top10_corrected':int(((before>10)&(after>0)&(after<=10)).sum()),'top10_broken':int(((before>0)&(before<=10)&((after>10)|(after==0))).sum()),'top20_corrected':int(((before>20)&(after>0)&(after<=20)).sum()),'top20_broken':int(((before>0)&(before<=20)&((after>20)|(after==0))).sum())}
def hit_change(basehit,newhit,k):
 a=basehit[:,:k].sum(1); b=newhit[:,:k].sum(1)
 return {'zero_to_hit':int(((a==0)&(b>0)).sum()),'hit_to_zero':int(((a>0)&(b==0)).sum()),'hit_count_increase':int((b>a).sum()),'hit_count_decrease':int((b<a).sum()),'unchanged':int((b==a).sum()),'net_hit_count':int((b-a).sum())}
def extract_paths(ev,eta):
 k=str(float(eta));
 return {'five_step_four_sample':ev['five_step_four_sample_etas'][k],'one_step_four_sample':ev['one_step_four_sample_etas'][k],'five_step_first_sample':ev['five_step_first_sample_etas'][k],'one_step_first_sample':ev['one_step_first_sample_etas'][k],'five_step_per_sample':{s:x[k] for s,x in ev['five_step_per_sample_etas'].items()},'one_step_per_sample':{s:x[k] for s,x in ev['one_step_per_sample_etas'].items()}}

def main():
 cfg=yaml.safe_load(open(ROOT/'diffusion_experiments/configs/round2_baby.yaml')); R=ROOT/'diffusion_experiments/runs/round2'; A=ROOT/'diffusion_experiments/runs/round1/assets_formal'; out=R/'analysis'; out.mkdir(parents=True,exist_ok=True)
 sup=json.load(open(R/'supervision_v1/manifest.json')); replay=json.load(open(R/'replay_check.json')); pre=json.load(open(R/'preformal_checks.json'))
 first=np.load(R/'C_seed202610061/predictions.npz'); dev_users=first['dev_users'].astype(np.int64); int_users=first['internal_users'].astype(np.int64); int_targets=first['internal_targets'].astype(np.int32)
 dsets=dev_sets(dev_users); isets={int(u):{int(t)} for u,t in zip(int_users,int_targets)}
 pt=np.load(A/'probe_targets.npz'); pa=np.load(A/'probe_top100.npz'); irows=np.flatnonzero(pt['internal'].astype(bool)); assert np.array_equal(pa['users'][irows],int_users); irank=pt['target_rank'][irows]; wh=(irank>=6)&(irank<=30); assert int(wh.sum())==189
 wh_users=int_users[wh]; wh_targets=int_targets[wh]; whsets={int(u):{int(t)} for u,t in zip(wh_users,wh_targets)}
 rows=[]; details={}; carr={}; predmap={}
 for v,seed in RUNS:
  name=f'{v}_seed{seed}'; rd=R/name; r=json.load(open(rd/'result.json')); p=np.load(rd/'predictions.npz')
  assert r['status']=='COMPLETE' and r['git_sha']=='80639d8de9cff892851b97dd9c33a0f4fdf59e96' and not r['tracked_dirty']; assert not r['access']['CONFIRM_ACCESSED'] and not r['access']['TEST_ACCESSED']
  assert np.array_equal(p['dev_users'],dev_users) and np.array_equal(p['internal_users'],int_users) and np.array_equal(p['internal_targets'],int_targets)
  db,dbh=contrib(p['dev_base'],dev_users,dsets); dn,dnh=contrib(p['dev_new'],dev_users,dsets); ib,ibh=contrib(p['internal_base'],int_users,isets); inn,inh=contrib(p['internal_new'],int_users,isets)
  wb,wbh=contrib(p['internal_base'][wh],wh_users,whsets); wn,wnh=contrib(p['internal_new'][wh],wh_users,whsets)
  eta=r['best']['eta']; hist=json.load(open(rd/'history.json')); brec=next(x for x in hist if x['epoch']==r['best']['epoch'])
  detail={'variant':v,'seed':seed,'best_epoch':r['best']['epoch'],'eta':eta,'dev':{'base':summarize_arr(db),'new':summarize_arr(dn),'U':U_from_means(db.mean(0),dn.mean(0)),'bootstrap':boot(db,dn),'hit_change_K10':hit_change(dbh,dnh,10),'hit_change_K20':hit_change(dbh,dnh,20),'paths':extract_paths(r['final_dev'],eta)},'internal_all':{'base':summarize_arr(ib),'new':summarize_arr(inn),'U':U_from_means(ib.mean(0),inn.mean(0)),'bootstrap':boot(ib,inn),'hit_change_K10':hit_change(ibh,inh,10),'hit_change_K20':hit_change(ibh,inh,20),'paths':extract_paths(r['final_internal'],eta),'rank_change':rankchange(rankpos(p['internal_base'],int_targets),rankpos(p['internal_new'],int_targets))},'internal_window189':{'base':summarize_arr(wb),'new':summarize_arr(wn),'U':U_from_means(wb.mean(0),wn.mean(0)),'bootstrap':boot(wb,wn),'rank_change':rankchange(rankpos(p['internal_base'][wh],wh_targets),rankpos(p['internal_new'][wh],wh_targets))},'train_probe_rank':r['train_rank_diagnostic'],'masked_timestep_mse_at_best':brec.get('masked_timestep_mse'),'weighted_grad_norm_first_batch_at_best_epoch':brec.get('weighted_grad_norm_shared_first_batch'),'parameter_count':r['parameter_count'],'optimizer_steps':r['optimizer_steps'],'train_forward_calls':r['train_forward_calls'],'elapsed_train_seconds':r['elapsed_train_seconds'],'peak_cuda_memory_bytes':r['peak_cuda_memory_bytes'],'final_dev_inference_seconds_all_paths':r['final_dev_inference_seconds_all_paths'],'epochs_completed':r['epochs_completed']}
  details[name]=detail; carr[(v,seed,'dev')]=dn; carr[(v,seed,'internal')]=inn; predmap[(v,seed)]=p
  np.savez_compressed(out/f'per_user_{name}.npz',dev_users=dev_users,dev_base_metrics=db,dev_new_metrics=dn,dev_delta_metrics=dn-db,dev_base_hits10=dbh[:,:10].sum(1),dev_new_hits10=dnh[:,:10].sum(1),dev_base_hits20=dbh[:,:20].sum(1),dev_new_hits20=dnh[:,:20].sum(1),internal_users=int_users,internal_base_metrics=ib,internal_new_metrics=inn,internal_delta_metrics=inn-ib,internal_base_hits10=ibh[:,:10].sum(1),internal_new_hits10=inh[:,:10].sum(1),internal_base_hits20=ibh[:,:20].sum(1),internal_new_hits20=inh[:,:20].sum(1))
  rows.append({'variant':v,'seed':seed,'best_epoch':r['best']['epoch'],'eta':eta,'DEV_U':detail['dev']['U'],'INTERNAL_U':detail['internal_all']['U'],'WINDOW189_U':detail['internal_window189']['U'],**{f'DEV_{k}':detail['dev']['new'][k] for k in ALL},**{f'INTERNAL_{k}':detail['internal_all']['new'][k] for k in ALL},'parameter_count':r['parameter_count'],'optimizer_steps':r['optimizer_steps'],'train_forward_calls':r['train_forward_calls'],'train_seconds':r['elapsed_train_seconds'],'peak_cuda_memory_bytes':r['peak_cuda_memory_bytes'],'inference_seconds_all_paths':r['final_dev_inference_seconds_all_paths']})
 # matched-seed paired comparisons
 comparisons={}
 for seed in [202610061,202610062]:
  comparisons[str(seed)]={}
  for ds in ['dev','internal']:
   comparisons[str(seed)][ds]={'D1_vs_C':boot(carr[('C',seed,ds)],carr[('D1',seed,ds)]),'D1_vs_D0':boot(carr[('D0',seed,ds)],carr[('D1',seed,ds)])}
 aggregate={}
 for v in ['C','D0','D1']:
  vv=[details[f'{v}_seed{s}'] for s in [202610061,202610062]]
  aggregate[v]={'mean_DEV_U':float(np.mean([x['dev']['U'] for x in vv])),'mean_INTERNAL_U':float(np.mean([x['internal_all']['U'] for x in vv])),'mean_WINDOW189_U':float(np.mean([x['internal_window189']['U'] for x in vv])),'DEV_U_by_seed':[x['dev']['U'] for x in vv],'INTERNAL_U_by_seed':[x['internal_all']['U'] for x in vv],'mean_train_seconds':float(np.mean([x['elapsed_train_seconds'] for x in vv])),'mean_peak_cuda_memory_bytes':float(np.mean([x['peak_cuda_memory_bytes'] for x in vv])),'mean_train_forward_calls':float(np.mean([x['train_forward_calls'] for x in vv])),'parameter_count':vv[0]['parameter_count']}
 d1=aggregate['D1']; d0=aggregate['D0']
 gate={'at_least_one_D1_internal_positive':bool(max(d1['INTERNAL_U_by_seed'])>0),'D1_mean_internal_positive':bool(d1['mean_INTERNAL_U']>0),'D1_mean_internal_ge_D0':bool(d1['mean_INTERNAL_U']>=d0['mean_INTERNAL_U']),'D1_mean_dev_gt_D0':bool(d1['mean_DEV_U']>d0['mean_DEV_U']),'D1_dev_protected_both':all(details[f'D1_seed{s}']['dev']['paths']['five_step_four_sample']['protected'] for s in [202610061,202610062]),'no_D1_internal_regression_beyond_0p5pct':all(x>=-0.005 for x in d1['INTERNAL_U_by_seed']),'engineering_checks_pass':sup['status']=='COMPLETE' and replay['status']=='PASS' and pre['status']=='PASS'}
 gate['PASS']=all(gate.values()); gate['next_stage']='D2_CJ_ALLOWED' if gate['PASS'] else 'STOP_AFTER_MAIN_MATRIX'
 round1_ref={'source':'ADVISOR_EXPERIMENT_GUIDE.md Round1 reference table','DeterministicResidual':{'mean_DEV_U':0.004741,'INTERNAL_U_by_seed':[0.003042,0.004442]},'BoundaryResidualDiffusion':{'mean_DEV_U':0.002422,'INTERNAL_U_by_seed':[-0.001863,-0.010936]}}
 checks={'status':'PASS','supervision':sup,'replay':replay,'preformal':pre,'formal_runs':{},'gate':gate,'access':{'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False},'classification':'VALID_EXPERIMENT_NO_DIFFUSION_ADVANTAGE'}
 for v,s in RUNS:
  rd=R/f'{v}_seed{s}'; rr=json.load(open(rd/'result.json')); checks['formal_runs'][f'{v}_seed{s}']={'status':open(rd/'status').read().strip(),'exitcode':int(open(rd/'exitcode').read().strip()),'git_sha':rr['git_sha'],'tracked_dirty':rr['tracked_dirty'],'config_sha256':rr['config_sha256'],'supervision_sha256':rr['supervision_sha256'],'result_sha256':sha(rd/'result.json'),'predictions_sha256':sha(rd/'predictions.npz'),'best_epoch':rr['best']['epoch'],'eta':rr['best']['eta']}
  if checks['formal_runs'][f'{v}_seed{s}']['status']!='COMPLETE' or checks['formal_runs'][f'{v}_seed{s}']['exitcode']!=0: checks['status']='FAIL'
 perhash={p.name:sha(p) for p in sorted(out.glob('per_user_*.npz'))}
 analysis={'status':'COMPLETE','round1_reference':round1_ref,'aggregate':aggregate,'matched_seed_comparisons':comparisons,'gate':gate,'details':details,'window189_users':189,'per_user_artifact_hashes':perhash,'interpretation':'D1 reduces INTERNAL negative magnitude versus D0 but remains negative on both seeds and does not beat D0 on mean DEV; D2/CJ gate fails.'}
 (out/'summary.json').write_text(json.dumps(analysis,indent=2)+'\n')
 E=ROOT/'diffusion_experiments/evidence'; E.mkdir(parents=True,exist_ok=True)
 (E/'round2_protocol.json').write_text(json.dumps({'round2_supervision':sup,'round1_replay':replay,'access':checks['access']},indent=2)+'\n')
 (E/'round2_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 with open(E/'round2_results.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),lineterminator='\n'); w.writeheader(); w.writerows(rows)
 print(json.dumps({'aggregate':aggregate,'gate':gate,'matched_seed_comparisons':comparisons},indent=2))
if __name__=='__main__': main()
