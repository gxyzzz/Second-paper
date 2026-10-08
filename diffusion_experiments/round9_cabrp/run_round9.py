from __future__ import annotations
import argparse, gc, hashlib, json, math, shutil, sys, time
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT))
from modules.diffusion import NativeTVX0Denoiser, cosine_alpha_bar, ddim_edit_batch, l2_rows_np, q_sample
from modules.semantic_purifier import blend_native
from modules.ranking import metrics_at, rank_by_score
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation
from pipelines.publication_eval import semantic_lift

PRIMARY=('R10','N10','R20','N20'); ALL=('R10','N10','R20','N20','R50','N50')
SEEDS=(20261001,20261002,20261003,20261004); CAPS=(5.0,10.0,15.0)
EXPECTED={
999:{'msca':'aa389ef8576e491ffb56eaec9d165444487bef9c1e542f280e991e2e063cc1cb','diff':'289a421e34b7aef1c06d7f36be202354de130d71998a80540f878d2778e352ab','cond':'a7f5be2c5a2fd8351bd041a6d805df017b1cecc20353c7065c1215186260c1f4'},
1000:{'msca':'39267b02da4aee97613805a5d2fd54971e3777a3dc57b02afb17688fe7053948','diff':'b963630c7899f9bd16abb87401fea6c1053407f22eacc5931eb64d9422bb138c','cond':'1e93f37e7002454e77d6e7c6283402b9643777e5f85c8bf16fa374d6950eba4c'},
1001:{'msca':'6789e7f3e2a9878cbbcad9e49f1388e474f27313053fe1a1fd5eaf642edf8e07','diff':'2998cc8bf3033fa382042bcfa181643075e4f0b2804a115513fcee9cfa641a97','cond':'bbc8daf85a37ac8dde94e3514c2073ec6911a79746e88d9ba46a608262d10ed1'},
1002:{'msca':'450f01580d9c044dca9bfee0befceaf2fc338bd06a92ff546dd9bbc07c62bc30','diff':'4148bb16f6e29a3b43d40420afcb9f7e6bbd2888dda87f5c1834a67ac7282d39','cond':'e26fb3d961ccde39e07fb2b1a239186de8373aa50352e142dadd6b587ef451ba'}}
HIST=ROOT/'docs/evidence/archive/robustness/baby_multiseed/baby_backbone_robustness_summary.json'

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def paths(seed:int):
 if seed==999:
  return dict(msca=ROOT/'runs/assets/msca_baby_seed999', colift=ROOT/'runs/generic_refactor/baby_formal',
   diff=ROOT/'runs/diffusion_rescue/baby_historical_seed_replay/beta_0p5', purified=ROOT/'runs/backbone_robustness/baby_seed999/purified', hist=ROOT/'runs/backbone_robustness/baby_seed999/evidence/fixed_diffusion_validation.json')
 b=ROOT/f'runs/backbone_robustness/baby_seed{seed}'
 return dict(msca=b/'msca/assets', colift=b/'coliftrec', diff=b/'diffusion', purified=b/'purified', hist=b/'evidence/fixed_diffusion_validation.json')

def load_model(seed,device='cuda'):
 p=paths(seed); ck=p['diff']/'checkpoints/baby_beta_0p5.pt'; cp=p['diff']/'assets/condition_beta_0p5.npy'
 if sha(ck)!=EXPECTED[seed]['diff'] or sha(cp)!=EXPECTED[seed]['cond']: raise RuntimeError('diffusion provenance mismatch')
 st=torch.load(ck,map_location='cpu',weights_only=False); m=NativeTVX0Denoiser(int(st['D']),cond_dim=64,hidden=int(st['hidden']),bottleneck=int(st.get('bottleneck',512)),time_dim=int(st.get('time_dim',64))).to(device); m.load_state_dict(st['state_dict'],strict=True); m.eval(); return m,ck,cp

def stats(x):
 x=np.asarray(x,float); return {k:float(v) for k,v in [('mean',x.mean()),('median',np.median(x)),('p75',np.quantile(x,.75)),('p90',np.quantile(x,.90)),('p95',np.quantile(x,.95)),('p99',np.quantile(x,.99)),('max',x.max())]}
def cstats(x):
 x=np.asarray(x,float); return {'mean':float(x.mean()),'median':float(np.median(x)),'p10':float(np.quantile(x,.1)),'p25':float(np.quantile(x,.25)),'p50':float(np.quantile(x,.5)),'p75':float(np.quantile(x,.75)),'p90':float(np.quantile(x,.9))}
def angle_deg(a,b): return np.degrees(np.arccos(np.clip(np.sum(l2_rows_np(a)*l2_rows_np(b),axis=1),-1,1)))

def slerp_bounded(raw,target,confidence,cap_deg,eps=1e-6):
 a=l2_rows_np(raw); b=l2_rows_np(target); c=np.clip(np.asarray(confidence,np.float32).reshape(-1),0,1); dot=np.clip(np.sum(a*b,axis=1),-1,1); theta=np.arccos(dot); cap=np.deg2rad(float(cap_deg)); move=c*np.minimum(theta,cap); frac=np.divide(move,theta,out=np.zeros_like(move),where=theta>eps); out=np.empty_like(a); small=theta<=eps; anti=np.abs(np.sin(theta))<1e-6; normal=~small & ~anti
 out[small]=a[small]
 if np.any(normal):
  th=theta[normal]; fr=frac[normal]; out[normal]=(np.sin((1-fr)*th)/np.sin(th))[:,None]*a[normal]+(np.sin(fr*th)/np.sin(th))[:,None]*b[normal]
 if np.any(anti & ~small):
  ii=np.flatnonzero(anti & ~small)
  for jj in ii:
   v=a[jj]; k=int(np.argmin(np.abs(v))); basis=np.zeros_like(v); basis[k]=1.0
   ort=basis-v*np.dot(v,basis); ort/=max(np.linalg.norm(ort),1e-12)
   out[jj]=math.cos(float(move[jj]))*v+math.sin(float(move[jj]))*ort
 out=l2_rows_np(out)
 if not np.isfinite(out).all(): raise RuntimeError('nonfinite SLERP')
 return out.astype(np.float32), np.degrees(theta), angle_deg(a,out)

def percentile_conf(margins,train_ids):
 m=np.asarray(margins,float); pos=np.sort(m[np.asarray(train_ids,int)][m[np.asarray(train_ids,int)]>0]); out=np.zeros(len(m),np.float32)
 if len(pos)==0:return out,pos
 idx=np.searchsorted(pos,m,side='right'); good=m>0; out[good]=idx[good]/len(pos); return out,pos

def load_raw():
 cfg=load_dataset_config('baby'); p=cfg['resolved_paths']; return cfg,np.load(p['text_feature'],mmap_mode='r'),np.load(p['visual_feature'],mmap_mode='r')

def ensemble_and_conf(seed,outdir,device='cuda',batch=128):
 cfg,rt,rv=load_raw(); p=paths(seed); model,ck,cp=load_model(seed,device); cond=np.load(cp,mmap_mode='r'); train_ids=np.load(p['diff']/'assets/train_item_ids.npy').astype(np.int64); n=len(rt); dt=rt.shape[1]
 sum_t=np.zeros((n,dt),np.float32); sum_v=np.zeros((n,rv.shape[1]),np.float32); ab=cosine_alpha_bar(50,.008).to(device)
 for ps in SEEDS:
  gen=torch.Generator(device=device); gen.manual_seed(int(ps))
  for st in range(0,n,batch):
   en=min(n,st+batch); x=np.concatenate([l2_rows_np(rt[st:en]),l2_rows_np(rv[st:en])],1); x0=torch.from_numpy(x).to(device); c=torch.from_numpy(np.asarray(cond[st:en],np.float32).copy()).to(device); y=ddim_edit_batch(model,x0,c,3,2.0,gen,ab).cpu().numpy(); sum_t[st:en]+=l2_rows_np(y[:,:dt]); sum_v[st:en]+=l2_rows_np(y[:,dt:])
 mean_t=sum_t/4.
 mean_v=sum_v/4.; R_t=np.linalg.norm(mean_t,axis=1).astype(np.float32); R_v=np.linalg.norm(mean_v,axis=1).astype(np.float32); target_t=l2_rows_np(mean_t); target_v=l2_rows_np(mean_v)
 # Fixed mechanism probe: TRUE / SHUFFLED / NULL use exactly the same noisy x_t.
 e={k:np.empty(n,np.float32) for k in ('true_t','true_v','shuf_t','shuf_v','null_t','null_v')}
 rev=np.arange(n-1,-1,-1,dtype=np.int64); g=torch.Generator(device=device); g.manual_seed(20261008)
 for st in range(0,n,batch):
  en=min(n,st+batch); x=np.concatenate([l2_rows_np(rt[st:en]),l2_rows_np(rv[st:en])],1).astype(np.float32)
  x0=torch.from_numpy(x).to(device); noise=torch.randn(x0.shape,generator=g,device=device); tt=torch.full((en-st,),3,device=device,dtype=torch.long); xt=q_sample(x0,tt,noise,ab)
  ct=torch.from_numpy(np.asarray(cond[st:en],np.float32).copy()).to(device); cs=torch.from_numpy(np.asarray(cond[rev[st:en]],np.float32).copy()).to(device); z=torch.zeros_like(ct)
  with torch.no_grad(): pt=model(xt,tt,ct); ps=model(xt,tt,cs); pn=model(xt,tt,z)
  for name,pred in [('true',pt),('shuf',ps),('null',pn)]:
   e[name+'_t'][st:en]=((pred[:,:dt]-x0[:,:dt])**2).mean(1).cpu().numpy()
   e[name+'_v'][st:en]=((pred[:,dt:]-x0[:,dt:])**2).mean(1).cpu().numpy()
 ref_t=np.minimum(e['shuf_t'],e['null_t']); ref_v=np.minimum(e['shuf_v'],e['null_v'])
 mt=(ref_t-e['true_t'])/(ref_t+1e-12); mv=(ref_v-e['true_v'])/(ref_v+1e-12)
 cc_t,pos_t=percentile_conf(mt,train_ids); cc_v,pos_v=percentile_conf(mv,train_ids)
 cf_t=np.sqrt(np.clip(R_t*cc_t,0,1)).astype(np.float32); cf_v=np.sqrt(np.clip(R_v*cc_v,0,1)).astype(np.float32)
 if not (np.all((cf_t>=0)&(cf_t<=1)) and np.all((cf_v>=0)&(cf_v<=1))): raise RuntimeError('confidence out of bounds')
 outdir.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(outdir/'round9_assets.npz',target_text=target_t,target_visual=target_v,c_seed_text=R_t,c_seed_visual=R_v,c_cond_text=cc_t,c_cond_visual=cc_v,c_final_text=cf_t,c_final_visual=cf_v,margin_text=mt,margin_visual=mv,train_ids=train_ids)
 diag={'seed':seed,'confidence_mode':'modality_specific','diffusion_checkpoint':str(ck.relative_to(ROOT)),'diffusion_sha256':sha(ck),'condition':str(Path(cp).relative_to(ROOT)),'condition_sha256':sha(cp),'train_item_count':int(len(train_ids)),'raw_target_angle_deg':{'text':stats(angle_deg(rt,target_t)),'visual':stats(angle_deg(rv,target_v))},'confidence':{}}
 for mod,R,cc,cf,ang,pos in [('text',R_t,cc_t,cf_t,angle_deg(rt,target_t),pos_t),('visual',R_v,cc_v,cf_v,angle_deg(rv,target_v),pos_v)]:
  order=np.argsort(cf); qs=np.array_split(order,4); qmove=[float(np.mean(cf[ii]*np.minimum(ang[ii],10.0))) for ii in qs]
  diag['confidence'][mod]={'c_seed':cstats(R),'c_cond':{**cstats(cc),'fraction_zero':float(np.mean(cc==0))},'c_final':{**cstats(cf),'fraction_lt_0p1':float(np.mean(cf<.1)),'fraction_lt_0p25':float(np.mean(cf<.25)),'fraction_gt_0p75':float(np.mean(cf>.75)),'fraction_gt_0p9':float(np.mean(cf>.9))},'corr_final_raw_target_angle':float(np.corrcoef(cf,ang)[0,1]),'positive_train_margin_count':int(len(pos)),'adaptive_Q1_Q4_mean_movement_at_10deg':qmove}
 del model; torch.cuda.empty_cache(); gc.collect(); return diag

def metric_delta(m,b): return {k:float(m[k]-b[k]) for k in ALL}
def relative(m,b): return {k:float((m[k]-b[k])/b[k]) for k in ALL}
def util(m,b): return float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY]))

def historical(seed): return json.loads(paths(seed)['hist'].read_text())

def validation_context(seed):
 cfg,rt,rv=load_raw(); p=paths(seed); audit=json.loads((p['msca']/'audit.json').read_text())
 if audit['checkpoint_sha256']!=EXPECTED[seed]['msca'] or audit.get('TEST_ACCESSED') is not False: raise RuntimeError('MSCA provenance/Test audit mismatch')
 val=np.load(p['msca']/'validation_top100.npz'); pseudo=np.load(p['msca']/'train_pseudo_top100.npz'); scores=np.load(p['colift']/'validation_scores.npz')
 users=scores['users'].astype(np.int64); items=scores['items'].astype(np.int32); base=scores['full_coliftrec'].astype(np.float32)
 if not np.array_equal(users,val['users']) or not np.array_equal(items,val['items']): raise RuntimeError('candidate identity mismatch')
 histories,pseudo_hist,pseudo_users,expected_users,eval_sets=build_train_histories_and_validation(cfg['resolved_paths']['interaction'],int(audit['n_users']))
 if not np.array_equal(users,expected_users) or not np.array_equal(pseudo_users,pseudo['users']): raise RuntimeError('Validation identity mismatch')
 raw_lt=semantic_lift(cfg['resolved_paths']['text_feature'],pseudo_hist,pseudo_users,pseudo['items'].astype(np.int32),histories,users,items,int(audit['n_items']),1.0,batch=256)
 raw_lv=semantic_lift(cfg['resolved_paths']['visual_feature'],pseudo_hist,pseudo_users,pseudo['items'].astype(np.int32),histories,users,items,int(audit['n_items']),0.25,batch=128)
 base_rank=rank_by_score(items,base); base_metrics=metrics_at(base_rank,users,eval_sets)
 return dict(cfg=cfg,rt=rt,rv=rv,p=p,audit=audit,pseudo=pseudo,users=users,items=items,base=base,histories=histories,pseudo_hist=pseudo_hist,pseudo_users=pseudo_users,eval_sets=eval_sets,raw_lt=raw_lt,raw_lv=raw_lv,base_rank=base_rank,base_metrics=base_metrics)
def evaluate_features(ctx,text,visual,tmp,name):
 tmp=Path(tmp); tmp.mkdir(parents=True,exist_ok=True); tp=tmp/f'{name}_text.npy'; vp=tmp/f'{name}_visual.npy'
 np.save(tp,np.asarray(text,np.float32)); np.save(vp,np.asarray(visual,np.float32))
 lt=semantic_lift(tp,ctx['pseudo_hist'],ctx['pseudo_users'],ctx['pseudo']['items'].astype(np.int32),ctx['histories'],ctx['users'],ctx['items'],int(ctx['audit']['n_items']),1.0,batch=256)
 lv=semantic_lift(vp,ctx['pseudo_hist'],ctx['pseudo_users'],ctx['pseudo']['items'].astype(np.int32),ctx['histories'],ctx['users'],ctx['items'],int(ctx['audit']['n_items']),0.25,batch=128)
 score=np.asarray(ctx['base'],np.float32).copy(); score+=0.25*(lt-ctx['raw_lt'])+0.025*(lv-ctx['raw_lv'])
 rank=rank_by_score(ctx['items'],score); met=metrics_at(rank,ctx['users'],ctx['eval_sets']); tp.unlink(missing_ok=True); vp.unlink(missing_ok=True)
 return {'metrics':met,'absolute_delta':metric_delta(met,ctx['base_metrics']),'relative_delta':relative(met,ctx['base_metrics']),'U':util(met,ctx['base_metrics']),'primary_positive_count':int(sum(met[k]>ctx['base_metrics'][k] for k in PRIMARY)),'overall_positive_count':int(sum(met[k]>ctx['base_metrics'][k] for k in ALL)),'ranking_exact_vs_colift':bool(np.array_equal(rank,ctx['base_rank'])),'score_max_abs_delta_vs_colift':float(np.max(np.abs(score-ctx['base'])))}, rank

def parity_a0(seed,ctx,outdir):
 h=historical(seed); p=paths(seed); fixed_t=np.load(p['purified']/'fixed_text.npy',mmap_mode='r'); fixed_v=np.load(p['purified']/'fixed_visual.npy',mmap_mode='r')
 if sha(p['purified']/'fixed_text.npy')!=h['DIFFUSION']['purified_text_sha256'] or sha(p['purified']/'fixed_visual.npy')!=h['DIFFUSION']['purified_visual_sha256']: raise RuntimeError('A0 purified provenance mismatch')
 a0t,a0v=blend_native(ctx['rt'],ctx['rv'],fixed_t,fixed_v,.25,1.0); res,rank=evaluate_features(ctx,a0t,a0v,outdir/'tmp','A0')
 hist=h['DIFFUSION']['metrics']; maxdiff=max(abs(res['metrics'][k]-hist[k]) for k in ALL); base_hist=h['COLIFTREC']['metrics']; basediff=max(abs(ctx['base_metrics'][k]-base_hist[k]) for k in ALL)
 audit={'backbone_seed':seed,'backbone_checkpoint':ctx['audit']['checkpoint'],'backbone_sha256':ctx['audit']['checkpoint_sha256'],'diffusion_checkpoint':str((p['diff']/'checkpoints/baby_beta_0p5.pt').relative_to(ROOT)),'diffusion_sha256':EXPECTED[seed]['diff'],'condition_sha256':EXPECTED[seed]['cond'],'beta':.5,'t_edit':3,'guidance':2.0,'rho_text':.25,'rho_visual':1.0,'purification_seeds':list(SEEDS),'metrics':res['metrics'],'historical_metrics':hist,'U_diff_vs_colift':res['U'],'historical_U':float(h['DIFFUSION']['U']),'metric_max_abs_diff':float(maxdiff),'colift_metric_max_abs_diff':float(basediff),'match':bool(maxdiff<=1e-12 and basediff<=1e-12),'TEST_ACCESSED':False}
 if not audit['match']: raise RuntimeError(f'A0 PARITY FAIL seed={seed} metricdiff={maxdiff} basediff={basediff}')
 return res,audit

def movement_record(raw_t,raw_v,target_t,target_v,ft,fv,cap,variant):
 rt=angle_deg(raw_t,target_t); rv=angle_deg(raw_v,target_v); at=angle_deg(raw_t,ft); av=angle_deg(raw_v,fv)
 rec={'variant':variant,'theta_cap_deg':float(cap),'text':{'raw_to_target_deg':stats(rt),'raw_to_final_deg':stats(at)},'visual':{'raw_to_target_deg':stats(rv),'raw_to_final_deg':stats(av)}}
 if variant in ('A1','A2') and (at.max()>cap+0.02 or av.max()>cap+0.02): raise RuntimeError(f'angular cap violation {variant} {cap}: {at.max()} {av.max()}')
 return rec

def slerp_selftests():
 rng=np.random.default_rng(9); raw=l2_rows_np(rng.normal(size=(8,16)).astype(np.float32)); target=l2_rows_np(rng.normal(size=(8,16)).astype(np.float32)); z=np.zeros(8,np.float32)
 out,_,mov=slerp_bounded(raw,target,z,10); c0=float(np.max(np.abs(out-raw)))
 out2,_,mov2=slerp_bounded(raw,raw,np.ones(8,np.float32),10); same=float(np.max(np.abs(out2-raw)))
 anti=np.zeros((1,16),np.float32); anti[0,0]=1; out3,_,mov3=slerp_bounded(anti,-anti,np.ones(1,np.float32),10)
 if c0>2e-6 or same>2e-6 or not np.isfinite(out3).all() or float(mov3.max())>10.02: raise RuntimeError('SLERP selftest fail')
 return {'confidence_zero_max_abs_diff':c0,'raw_equals_target_max_abs_diff':same,'antipodal_finite':True,'antipodal_applied_angle_deg':float(mov3[0])}

def run_seed(seed:int,outdir:Path,mode:str):
 t0=time.time(); outdir.mkdir(parents=True,exist_ok=True); ctx=validation_context(seed); selftest=slerp_selftests(); a0,audit=parity_a0(seed,ctx,outdir)
 # Only after exact A0 parity do we generate Round9 targets/confidence.
 diag=ensemble_and_conf(seed,outdir); z=np.load(outdir/'round9_assets.npz'); tt=z['target_text']; tv=z['target_visual']; cf_t=z['c_final_text']; cf_v=z['c_final_visual']
 variants={'A0':a0}; movements=[]
 if mode=='smoke': caps=[10.0]
 else: caps=list(CAPS)
 # theta=0 identity is a smoke-only non-trial.
 if mode=='smoke':
  i1t=np.asarray(ctx['rt'],np.float32).copy(); i1v=np.asarray(ctx['rv'],np.float32).copy(); ident,_=evaluate_features(ctx,i1t,i1v,outdir/'tmp','A1_theta0')
  if not ident['ranking_exact_vs_colift'] or max(abs(ident['metrics'][k]-ctx['base_metrics'][k]) for k in ALL)>1e-12: raise RuntimeError('A1 theta=0 identity FAIL')
  selftest['A1_theta0_identity']=ident
 for cap in caps:
  a1t,_,_=slerp_bounded(ctx['rt'],tt,np.ones(len(tt),np.float32),cap); a1v,_,_=slerp_bounded(ctx['rv'],tv,np.ones(len(tv),np.float32),cap)
  r1,_=evaluate_features(ctx,a1t,a1v,outdir/'tmp',f'A1_{int(cap)}'); variants[f'A1_{int(cap)}']=r1; movements.append(movement_record(ctx['rt'],ctx['rv'],tt,tv,a1t,a1v,cap,'A1'))
  a2t,_,_=slerp_bounded(ctx['rt'],tt,cf_t,cap); a2v,_,_=slerp_bounded(ctx['rv'],tv,cf_v,cap)
  r2,_=evaluate_features(ctx,a2t,a2v,outdir/'tmp',f'A2_{int(cap)}'); variants[f'A2_{int(cap)}']=r2; movements.append(movement_record(ctx['rt'],ctx['rv'],tt,tv,a2t,a2v,cap,'A2'))
 # Actual adaptive movement quartiles at 10 degrees must rise with confidence in smoke.
 if 10.0 in caps:
  a2t,_,mt=slerp_bounded(ctx['rt'],tt,cf_t,10); a2v,_,mv=slerp_bounded(ctx['rv'],tv,cf_v,10); qdiag={}
  for mod,c,m in [('text',cf_t,mt),('visual',cf_v,mv)]:
   qs=np.array_split(np.argsort(c),4); vals=[float(np.mean(m[ii])) for ii in qs]; qdiag[mod]=vals
   if not all(vals[i] <= vals[i+1]+1e-5 for i in range(3)): raise RuntimeError(f'adaptive quartile trend FAIL {mod} {vals}')
  diag['actual_adaptive_quartile_movement_10deg']=qdiag
 result={'status':'PASS','protocol':'ROUND9_CABRP_V1','mode':mode,'seed':seed,'source_commit':'ee82c3118dcfdd09cda8a430e0f61b88f84e4ab7','coliftrec_metrics':ctx['base_metrics'],'A0_audit':audit,'variants':variants,'movement':movements,'confidence_diagnostics':diag,'selftests':selftest,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False,'elapsed_seconds':time.time()-t0}
 (outdir/'result.json').write_text(json.dumps(result,indent=2)+'\n'); shutil.rmtree(outdir/'tmp',ignore_errors=True); print(json.dumps({'status':'PASS','mode':mode,'seed':seed,'A0_U':a0['U'],'variants':{k:round(v['U'],8) for k,v in variants.items()},'elapsed':result['elapsed_seconds']},sort_keys=True)); return result

def aggregate(rootdir:Path,evid:Path):
 rows={}; a0={}; movement={}; conf={}
 for s in (999,1000,1001,1002):
  p=rootdir/f'seed{s}'/'result.json'
  if not p.exists(): raise RuntimeError(f'missing formal result {p}')
  r=json.loads(p.read_text())
  if r['status']!='PASS' or r.get('TEST_ACCESSED') is not False: raise RuntimeError(f'invalid formal result seed{s}')
  rows[s]=r; a0[str(s)]=r['A0_audit']; movement[str(s)]=r['movement']; conf[str(s)]=r['confidence_diagnostics']
 evid.mkdir(parents=True,exist_ok=True)
 (evid/'A0_REPRODUCTION_AUDIT.json').write_text(json.dumps({'status':'PASS','seeds':a0,'TEST_ACCESSED':False},indent=2)+'\n')
 (evid/'REPRESENTATION_MOVEMENT.json').write_text(json.dumps(movement,indent=2)+'\n')
 (evid/'CONFIDENCE_DIAGNOSTICS.json').write_text(json.dumps(conf,indent=2)+'\n')
 tables={v:{} for v in ('A1','A2')}
 for variant in ('A1','A2'):
  for cap in CAPS:
   key=f'{variant}_{int(cap)}'; us={str(s):float(rows[s]['variants'][key]['U']) for s in rows}; pcs={str(s):int(rows[s]['variants'][key]['primary_positive_count']) for s in rows}
   vals=list(us.values()); tables[variant][str(int(cap))]={'seed_U':us,'mean_U':float(np.mean(vals)),'worst_seed_U':float(min(vals)),'positive_seeds':int(sum(x>0 for x in vals)),'primary_ge3_seeds':int(sum(pcs[str(s)]>=3 for s in rows)),'gate_S':bool(all(x>0 for x in vals) and min(vals)>=0),'gate_P':bool(sum(pcs[str(s)]>=3 for s in rows)>=3),'gate_M_mechanism_positive':bool(np.mean(vals)>0),'gate_M_useful':bool(np.mean(vals)>=.005),'gate_M_target':bool(np.mean(vals)>=.01)}
 # Global A2 theta selection: stability first, then mean U, close tie -> smaller cap.
 stable=[c for c in CAPS if tables['A2'][str(int(c))]['positive_seeds']==4]
 pool=stable if stable else list(CAPS)
 best=max(pool,key=lambda c:(tables['A2'][str(int(c))]['mean_U'],-c)); selected=float(best); sel=tables['A2'][str(int(best))]
 a0_us={str(s):float(rows[s]['variants']['A0']['U']) for s in rows}; a0_vals=list(a0_us.values()); a0_positive=sum(x>0 for x in a0_vals); a0_worst=min(a0_vals)
 if sel['positive_seeds']==4:
  if sel['mean_U']>=.01: verdict='TARGET_PASS'
  elif sel['mean_U']>=.005: verdict='STRONG_STABILITY_PASS'
  else: verdict='STABILITY_PASS_SMALL_GAIN'
 else:
  partial=(sel['positive_seeds']>a0_positive and sel['worst_seed_U']>a0_worst)
  verdict='PARTIAL_STABILITY_IMPROVEMENT' if partial else 'NO_STABILITY_GAIN'
 selected_detail={str(s):rows[s]['variants'][f'A2_{int(selected)}'] for s in rows}
 out={'status':'VALIDATION_DECISION','protocol':'ROUND9_CABRP_V1','A0':{'seed_U':a0_us,'mean_U':float(np.mean(a0_vals)),'worst_seed_U':float(a0_worst),'positive_seeds':int(a0_positive)},'A1':tables['A1'],'A2':tables['A2'],'selected_A2_theta_cap_deg':selected,'selected_A2':sel,'selected_A2_detailed':selected_detail,'verdict':verdict,'TEST_ACCESSED':False,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
 (evid/'ROUND9_VALIDATION_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
 return out,rows

def pct(x): return f'{100*float(x):+.4f}%'
def choose_table(tab):
 stable=[c for c in CAPS if tab[str(int(c))]['positive_seeds']==4]; pool=stable if stable else CAPS
 return float(max(pool,key=lambda c:(tab[str(int(c))]['mean_U'],-c)))

def write_report(out,rows,evid):
 a2cap=float(out['selected_A2_theta_cap_deg']); a1cap=choose_table(out['A1']); a1sel=out['A1'][str(int(a1cap))]; a2sel=out['selected_A2']; a0=out['A0']
 q1=(a1sel['positive_seeds']>a0['positive_seeds'] and a1sel['worst_seed_U']>a0['worst_seed_U'])
 a2_vs_a1_mean=a2sel['mean_U']-out['A1'][str(int(a2cap))]['mean_U']; q2=a2_vs_a1_mean>0
 L=['# Round9 Validation Report — Confidence-Adaptive Bounded Residual Purification','',f"Final status: **VALIDATION_DECISION**",f"Verdict: **ROUND9 = {out['verdict']}**",'',f"Selected global A2 theta cap: **{a2cap:.0f}°**",'', '> Test, Sports and Electronics remained CLOSED throughout this round.','']
 L += ['## Table 1 — Original parity','', '| Backbone | A0 U | Historical U | Match |','|---:|---:|---:|:---:|']
 for s in (999,1000,1001,1002):
  au=rows[s]['A0_audit']; L.append(f"| {s} | {pct(au['U_diff_vs_colift'])} | {pct(au['historical_U'])} | {'PASS' if au['match'] else 'FAIL'} |")
 for variant,title in [('A1','## Table 2 — A1 Uniform Bounded'),('A2','## Table 3 — A2 Confidence-Adaptive')]:
  L += ['',title,'','| theta | seed999 | seed1000 | seed1001 | seed1002 | mean U | positive seeds |','|---:|---:|---:|---:|---:|---:|---:|']
  for c in CAPS:
   x=out[variant][str(int(c))]; u=x['seed_U']; L.append(f"| {int(c)}° | {pct(u['999'])} | {pct(u['1000'])} | {pct(u['1001'])} | {pct(u['1002'])} | {pct(x['mean_U'])} | {x['positive_seeds']}/4 |")
 L += ['','## Table 4 — Selected A2 detailed metrics','', '| Backbone | Metric | Absolute | Δ vs CoLift | Relative Δ |','|---:|---|---:|---:|---:|']
 for s in (999,1000,1001,1002):
  x=out['selected_A2_detailed'][str(s)]
  for k in ALL: L.append(f"| {s} | {k} | {x['metrics'][k]:.8f} | {x['absolute_delta'][k]:+.8f} | {pct(x['relative_delta'][k])} |")
  L.append(f"| {s} | **U** | — | — | **{pct(x['U'])}** |")
 L += ['','## Table 5 — Mechanism diagnostics','', '| Backbone | Modality | raw→target mean | raw→final mean (selected A2) | c_seed mean | c_cond mean | c_final mean |','|---:|---|---:|---:|---:|---:|---:|']
 for s in (999,1000,1001,1002):
  d=rows[s]['confidence_diagnostics']; move=[m for m in rows[s]['movement'] if m['variant']=='A2' and abs(m['theta_cap_deg']-a2cap)<1e-8][0]
  for mod in ('text','visual'):
   c=d['confidence'][mod]; L.append(f"| {s} | {mod} | {d['raw_target_angle_deg'][mod]['mean']:.4f}° | {move[mod]['raw_to_final_deg']['mean']:.4f}° | {c['c_seed']['mean']:.4f} | {c['c_cond']['mean']:.4f} | {c['c_final']['mean']:.4f} |")
 L += ['','## Validation gates','',f"- Gate S (4/4 U>0): **{a2sel['gate_S']}**; worst seed U = **{pct(a2sel['worst_seed_U'])}**.",f"- Gate P (≥3/4 backbones have ≥3/4 primary metrics positive): **{a2sel['gate_P']}** ({a2sel['primary_ge3_seeds']}/4).",f"- Mean utility: **{pct(a2sel['mean_U'])}**; mechanism-positive={a2sel['gate_M_mechanism_positive']}, useful≥0.5%={a2sel['gate_M_useful']}, target≥1%={a2sel['gate_M_target']}.",'']
 L += ['## Scientific questions','',f"**Q1 — Does bounding alone improve stability?** A0 has {a0['positive_seeds']}/4 positive backbones; the globally selected A1 ({a1cap:.0f}°) has {a1sel['positive_seeds']}/4, with worst-seed U {pct(a1sel['worst_seed_U'])}. Strict stability-improvement criterion: **{q1}**.",f"**Q2 — Does confidence add evidence beyond bounding?** At the selected A2 cap ({a2cap:.0f}°), A2 mean U minus A1 mean U is **{pct(a2_vs_a1_mean)}**. Directionally better: **{q2}**. Confidence is deterministic and never uses ranking labels.",f"**Q3 — Can purification recover stable increment without ranking diffusion?** Selected A2 has **{a2sel['positive_seeds']}/4** positive backbones and mean U **{pct(a2sel['mean_U'])}**. Final verdict follows the preregistered stability-first rule: **{out['verdict']}**.",'']
 L += ['## Mechanism sanity','']
 for s in (999,1000,1001,1002):
  d=rows[s]['confidence_diagnostics']; L.append(f"- seed{s}: A2@10° confidence-quartile movement Text={d['actual_adaptive_quartile_movement_10deg']['text']}; Visual={d['actual_adaptive_quartile_movement_10deg']['visual']}.")
 L += ['','## Provenance', '', '- Source Round8 commit: `ee82c3118dcfdd09cda8a430e0f61b88f84e4ab7`.', '- Frozen CoLiftRec: λ(T/A/V)=1.0/0.75/0.25; α(T/A/V)=0.25/0.15/0.025.', '- Diffusion: frozen x0 generator, beta=0.5, t_edit=3, guidance=2.0, K=4 seeds 20261001–20261004.', '- No Test labels or Test loader were used.', '']
 (evid/'ROUND9_REPORT.md').write_text('\n'.join(L))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','formal','aggregate'],required=True); ap.add_argument('--seed',type=int); ap.add_argument('--out-root',default=str(ROOT/'diffusion_experiments/round9_cabrp/outputs')); ap.add_argument('--evidence',default=str(ROOT/'diffusion_experiments/round9_cabrp/evidence')); a=ap.parse_args(); root=Path(a.out_root); evid=Path(a.evidence)
 if a.mode in ('smoke','formal'):
  if a.mode=='smoke':
   seed=999; out=root/'smoke_seed999'
  else:
   if a.seed not in (999,1000,1001,1002): raise RuntimeError('formal seed must be one frozen Baby backbone')
   seed=a.seed; out=root/f'seed{seed}'
  if out.exists(): shutil.rmtree(out)
  run_seed(seed,out,a.mode)
 else:
  out,rows=aggregate(root,evid); write_report(out,rows,evid); print(json.dumps({'status':'VALIDATION_DECISION','selected_theta':out['selected_A2_theta_cap_deg'],'verdict':out['verdict'],'A2':out['selected_A2']},sort_keys=True))
if __name__=='__main__': main()
